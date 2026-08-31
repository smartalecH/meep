/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_initialization.hpp"
#include "backend/nvidia/nvidia_materials.hpp"
#include "backend/nvidia/runtime.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
static void require_invalid(Callable callable, const char *message) {
  bool rejected = false;
  try { callable(); }
  catch (const std::invalid_argument &) { rejected = true; }
  catch (const std::overflow_error &) { rejected = true; }
  require(rejected, message);
}

template <typename T>
static void require_exact(const std::vector<T> &observed, const std::vector<T> &expected,
                          const char *message) {
  require(observed.size() == expected.size(), message);
  for (size_t i = 0; i < observed.size(); ++i)
    if (observed[i] != expected[i]) throw std::runtime_error(message);
}

static size_t append_aligned(std::vector<unsigned char> &bytes, const void *source,
                             size_t count, size_t alignment) {
  const size_t padding = (alignment - bytes.size() % alignment) % alignment;
  bytes.insert(bytes.end(), padding, 0);
  const size_t offset = bytes.size();
  if (count) {
    const unsigned char *begin = static_cast<const unsigned char *>(source);
    bytes.insert(bytes.end(), begin, begin + count);
  }
  return offset;
}

template <typename T>
static size_t append_record(std::vector<unsigned char> &bytes, const T &record) {
  return append_aligned(bytes, &record, sizeof(record), alignof(T));
}

template <typename T>
static void check_geometry_kernels(int device, scalar_precision precision) {
  const size_t points = 257;
  std::vector<unsigned char> compact;
  const double parameters[4] = {0.0, 0.0, 0.0, 0.25};
  const size_t parameter_offset =
      append_aligned(compact, parameters, sizeof(parameters), alignof(double));
  geometry_object_record object = {};
  object.kind = 4;
  object.material = 1;
  object.parameter_offset = parameter_offset;
  object.parameter_count = 4;
  object.low[0] = object.low[1] = object.low[2] = -0.25;
  object.high[0] = object.high[1] = object.high[2] = 0.25;
  const size_t object_offset = append_record(compact, object);
  geometry_image_record image = {};
  image.object = 0;
  image.ordinal = 0;
  image.precedence = 1;
  for (int axis = 0; axis < 3; ++axis) {
    image.low[axis] = -0.25;
    image.high[axis] = 0.25;
  }
  const size_t image_offset = append_record(compact, image);
  geometry_value_record values[2] = {};
  for (int material = 0; material < 2; ++material) {
    values[material].kind = geometry_value_kind::constant;
    values[material].tensor_1[0] = values[material].tensor_1[1] =
        values[material].tensor_1[2] = material ? 4.0 : 1.0;
  }
  const size_t value_offset =
      append_aligned(compact, values, sizeof(values), alignof(geometry_value_record));
  geometry_analytic_record analytic = {};
  analytic.point = 64;
  analytic.front_material = 1;
  analytic.behind_material = 0;
  analytic.normal[0] = 1.0;
  analytic.fill = 0.5;
  const size_t analytic_offset = append_record(compact, analytic);
  geometry_patch_record patches[2] = {{65, 0.625}, {192, 0.12345678901234567}};
  const size_t patch_offset =
      append_aligned(compact, patches, sizeof(patches), alignof(geometry_patch_record));

  stream execution;
  device_buffer device_compact(compact.size(), device);
  std::vector<T> host(points + 2, T(-17.0));
  device_buffer destination(host.size() * sizeof(T), device);
  copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
  copy_host_to_device_async(destination, 0, host.data(), host.size() * sizeof(T), execution);
  geometry_launch_common common = {};
  common.destination = static_cast<unsigned char *>(destination.opaque_handle()) + sizeof(T);
  common.compact_inputs = static_cast<const unsigned char *>(device_compact.opaque_handle());
  common.compact_input_bytes = compact.size();
  common.object_offset = object_offset;
  common.object_count = 1;
  common.image_offset = image_offset;
  common.image_count = 1;
  common.value_offset = value_offset;
  common.material_count = 2;
  common.default_material = 0;
  common.elements = points;
  common.point_count = points;
  common.dimensions = 2;
  common.component = 0;
  common.tensor_row = 0;
  common.tensor_column = 0;
  common.property = 0;
  common.axis_direction[0] = 0;
  common.axis_direction[1] = 1;
  common.axis_direction[2] = 2;
  common.loop_begin[0] = -256;
  common.loop_extent[0] = points;
  common.loop_extent[1] = common.loop_extent[2] = 1;
  common.strides[0] = 1;
  common.cell_size[0] = common.cell_size[1] = common.cell_size[2] = 1.0;
  common.metric[0] = common.metric[4] = common.metric[8] = 1.0;
  common.inva = 1.0 / 256.0;
  common.dt = 0.1;
  common.precision = precision;
  geometry_bulk_launch bulk = {common, 0, points};
  geometry_analytic_launch analytic_launch = {common, analytic_offset, 1};
  geometry_patch_launch patch_launch = {common, patch_offset, 2};
  validate_geometry_bulk_launch(bulk, compact.data(), compact.size());
  validate_geometry_analytic_launch(analytic_launch, compact.data(), compact.size());
  validate_geometry_patch_launch(patch_launch, compact.data(), compact.size());
  launch_material_geometry_bulk(bulk, execution);
  launch_material_geometry_analytic(analytic_launch, execution);
  launch_material_geometry_patch(patch_launch, execution);
  copy_device_to_host_async(host.data(), destination, 0, host.size() * sizeof(T), execution);
  execution.synchronize();
  require(host.front() == T(-17.0) && host.back() == T(-17.0),
          "material geometry kernels overwrote guard elements");
  for (size_t point = 0; point < points; ++point) {
    double expected = point >= 64 && point <= 192 ? 0.25 : 1.0;
    if (point == 64) expected = 0.625;
    if (point == 65) expected = patches[0].value;
    if (point == 192) expected = patches[1].value;
    require(host[point + 1] == T(expected), "material geometry kernel value differs");
  }
  require(host[65] == host[66],
          "analytic Kottke result differs from its independently supplied patch oracle");

  /* The shared point/address helper must be invariant under split launches,
     including both sides of the 256-thread block boundary and a nonzero
     destination base. */
  const size_t leading = 3, trailing = 2;
  std::vector<T> split_host(leading + points + trailing, T(-29.0));
  device_buffer split_destination(split_host.size() * sizeof(T), device);
  copy_host_to_device_async(split_destination, 0, split_host.data(),
                            split_host.size() * sizeof(T), execution);
  geometry_launch_common split_common = common;
  split_common.destination =
      static_cast<unsigned char *>(split_destination.opaque_handle()) + leading * sizeof(T);
  const geometry_bulk_launch zero = {split_common, 0, 0};
  launch_material_geometry_bulk(zero, execution);
  const geometry_bulk_launch split[] = {{split_common, 0, 1},
                                        {split_common, 1, 255},
                                        {split_common, 256, 1}};
  for (const geometry_bulk_launch &part : split) {
    validate_geometry_bulk_launch(part, compact.data(), compact.size());
    launch_material_geometry_bulk(part, execution);
  }
  copy_device_to_host_async(split_host.data(), split_destination, 0,
                            split_host.size() * sizeof(T), execution);
  execution.synchronize();
  for (size_t i = 0; i < leading; ++i)
    require(split_host[i] == T(-29.0), "split geometry launch overwrote a leading guard");
  for (size_t point = 0; point < points; ++point) {
    const double expected = point >= 64 && point <= 192 ? 0.25 : 1.0;
    require(split_host[leading + point] == T(expected),
            "split geometry bulk launch differs from whole launch");
  }
  for (size_t i = leading + points; i < split_host.size(); ++i)
    require(split_host[i] == T(-29.0), "split geometry launch overwrote a trailing guard");

  std::fill(split_host.begin(), split_host.end(), T(-29.0));
  copy_host_to_device_async(split_destination, 0, split_host.data(),
                            split_host.size() * sizeof(T), execution);
  geometry_bulk_launch block_boundary = {split_common, 0, 256};
  validate_geometry_bulk_launch(block_boundary, compact.data(), compact.size());
  launch_material_geometry_bulk(block_boundary, execution);
  copy_device_to_host_async(split_host.data(), split_destination, 0,
                            split_host.size() * sizeof(T), execution);
  execution.synchronize();
  require(split_host[leading + 256] == T(-29.0),
          "256-element geometry launch wrote its excluded tail");

  /* Pin perfect-metal signed zero and CPU-compatible diagonal-zero inverse
     semantics in the geometry kernel itself. */
  std::vector<unsigned char> semantic_compact = compact;
  geometry_value_record *semantic_values = reinterpret_cast<geometry_value_record *>(
      semantic_compact.data() + value_offset);
  semantic_values[1].flags = 1u;
  semantic_values[1].value_1 = -0.0;
  copy_host_to_device_async(device_compact, 0, semantic_compact.data(),
                            semantic_compact.size(), execution);
  std::fill(host.begin(), host.end(), T(-17.0));
  copy_host_to_device_async(destination, 0, host.data(), host.size() * sizeof(T), execution);
  launch_material_geometry_bulk(bulk, execution);
  copy_device_to_host_async(host.data(), destination, 0, host.size() * sizeof(T), execution);
  execution.synchronize();
  require(host[65] == T(0) && std::signbit(host[65]),
          "geometry perfect-metal value lost negative zero");

  semantic_compact = compact;
  semantic_values = reinterpret_cast<geometry_value_record *>(
      semantic_compact.data() + value_offset);
  semantic_values[1].tensor_1[0] = 0.0;
  validate_geometry_bulk_launch(bulk, semantic_compact.data(), semantic_compact.size());
  copy_host_to_device_async(device_compact, 0, semantic_compact.data(),
                            semantic_compact.size(), execution);
  std::fill(host.begin(), host.end(), T(-17.0));
  copy_host_to_device_async(destination, 0, host.data(), host.size() * sizeof(T), execution);
  launch_material_geometry_bulk(bulk, execution);
  copy_device_to_host_async(host.data(), destination, 0, host.size() * sizeof(T), execution);
  execution.synchronize();
  require(std::isinf(double(host[65])) && host[65] > 0,
          "geometry diagonal zero did not preserve positive infinity");
  copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);

  testing::fail_next(testing::failure_point::material_geometry_patch_launch);
  bool rejected = false;
  try { launch_material_geometry_patch(patch_launch, execution); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "material geometry patch failure injection was ignored");
  geometry_patch_launch malformed = patch_launch;
  malformed.patch_offset = compact.size();
  require_invalid([&]() { validate_geometry_patch_launch(malformed, compact.data(), compact.size()); },
                  "out-of-range material geometry patch was accepted");

  std::vector<unsigned char> malformed_compact = compact;
  geometry_value_record *malformed_value = reinterpret_cast<geometry_value_record *>(
      malformed_compact.data() + value_offset + sizeof(geometry_value_record));
  malformed_value->kind = geometry_value_kind::grid_tensor;
  malformed_value->overlap_kind = 4;
  malformed_value->dimensions[0] = malformed_value->dimensions[1] =
      malformed_value->dimensions[2] = 1;
  malformed_value->sample_offset = parameter_offset;
  malformed_value->sample_count = 1;
  require_invalid(
      [&]() { validate_geometry_bulk_launch(bulk, malformed_compact.data(),
                                             malformed_compact.size()); },
      "invalid material geometry overlap mode was accepted");
  malformed_value->overlap_kind = 0;
  malformed_value->dimensions[0] = 0;
  require_invalid(
      [&]() { validate_geometry_bulk_launch(bulk, malformed_compact.data(),
                                             malformed_compact.size()); },
      "zero material geometry grid dimension was accepted");
  malformed_value->dimensions[0] = uint32_t(INT_MAX) + 1u;
  require_invalid(
      [&]() { validate_geometry_bulk_launch(bulk, malformed_compact.data(),
                                             malformed_compact.size()); },
      "oversized material geometry grid dimension was accepted");
  malformed_compact = compact;
  malformed_value = reinterpret_cast<geometry_value_record *>(
      malformed_compact.data() + value_offset + sizeof(geometry_value_record));
  malformed_value->tensor_1[0] = std::numeric_limits<double>::quiet_NaN();
  require_invalid(
      [&]() { validate_geometry_bulk_launch(bulk, malformed_compact.data(),
                                             malformed_compact.size()); },
      "non-finite material geometry tensor was accepted");
  malformed_compact = compact;
  malformed_value = reinterpret_cast<geometry_value_record *>(
      malformed_compact.data() + value_offset + sizeof(geometry_value_record));
  malformed_value->tensor_1[0] = malformed_value->tensor_1[1] =
      malformed_value->tensor_1[2] = 1.0;
  malformed_value->tensor_1[3] = 1.0;
  require_invalid(
      [&]() { validate_geometry_bulk_launch(bulk, malformed_compact.data(),
                                             malformed_compact.size()); },
      "singular non-diagonal material geometry tensor was accepted");

  geometry_analytic_launch malformed_analytic = analytic_launch;
  malformed_analytic.count = 2;
  require_invalid(
      [&]() { validate_geometry_analytic_launch(malformed_analytic, compact.data(),
                                                 compact.size()); },
      "out-of-range material geometry analytic span was accepted");
  geometry_bulk_launch malformed_bulk = bulk;
  malformed_bulk.first_point = points;
  malformed_bulk.count = 1;
  require_invalid(
      [&]() { validate_geometry_bulk_launch(malformed_bulk, compact.data(), compact.size()); },
      "out-of-range material geometry bulk span was accepted");
  malformed_bulk = bulk;
  malformed_bulk.common.object_offset += 1;
  require_invalid(
      [&]() { validate_geometry_bulk_launch(malformed_bulk, compact.data(), compact.size()); },
      "misaligned material geometry object table was accepted");
  malformed_bulk = bulk;
  malformed_bulk.common.destination =
      const_cast<unsigned char *>(malformed_bulk.common.compact_inputs);
  require_invalid(
      [&]() { validate_geometry_bulk_launch(malformed_bulk, compact.data(), compact.size()); },
      "material geometry destination/compact alias was accepted");
  malformed_bulk = bulk;
  malformed_bulk.common.loop_extent[0] = std::numeric_limits<size_t>::max();
  require_invalid(
      [&]() { validate_geometry_bulk_launch(malformed_bulk, compact.data(), compact.size()); },
      "overflowing material geometry loop was accepted");
  malformed_analytic = analytic_launch;
  malformed_compact = compact;
  reinterpret_cast<geometry_analytic_record *>(
      malformed_compact.data() + analytic_offset)->normal[0] = 0.0;
  require_invalid(
      [&]() { validate_geometry_analytic_launch(malformed_analytic, malformed_compact.data(),
                                                 malformed_compact.size()); },
      "zero material geometry analytic normal was accepted");
}

template <typename T>
static void check_geometry_precedence(int device, scalar_precision precision) {
  std::vector<unsigned char> compact;
  const double local_parameters[4] = {0.0, 0.0, 0.0, 0.2};
  const double periodic_parameters[4] = {-1.0, 0.0, 0.0, 0.2};
  geometry_object_record objects[2] = {};
  objects[0].kind = objects[1].kind = 4;
  objects[0].material = 1;
  objects[1].material = 2;
  objects[0].parameter_offset =
      append_aligned(compact, local_parameters, sizeof(local_parameters), alignof(double));
  objects[1].parameter_offset =
      append_aligned(compact, periodic_parameters, sizeof(periodic_parameters), alignof(double));
  objects[0].parameter_count = objects[1].parameter_count = 4;
  for (int axis = 0; axis < 3; ++axis) {
    objects[0].low[axis] = objects[1].low[axis] = -0.2;
    objects[0].high[axis] = objects[1].high[axis] = 0.2;
  }
  objects[1].low[0] = -1.2;
  objects[1].high[0] = -0.8;
  const size_t object_offset =
      append_aligned(compact, objects, sizeof(objects), alignof(geometry_object_record));
  geometry_image_record images[2] = {};
  images[0].object = 1;
  images[0].ordinal = 0;
  images[0].precedence = 2;
  images[0].image[0] = 1;
  images[0].shift[0] = 1.0;
  images[1].object = 0;
  images[1].ordinal = 1;
  images[1].precedence = 1;
  for (int image = 0; image < 2; ++image)
    for (int axis = 0; axis < 3; ++axis) {
      images[image].low[axis] = objects[images[image].object].low[axis] +
                                images[image].shift[axis];
      images[image].high[axis] = objects[images[image].object].high[axis] +
                                 images[image].shift[axis];
    }
  const size_t image_offset =
      append_aligned(compact, images, sizeof(images), alignof(geometry_image_record));
  geometry_value_record values[3] = {};
  for (int material = 0; material < 3; ++material) {
    values[material].kind = geometry_value_kind::constant;
    values[material].flags = 1;
    values[material].value_1 = material == 0 ? 1.0 : material == 1 ? 0.25 : 0.125;
  }
  const size_t value_offset =
      append_aligned(compact, values, sizeof(values), alignof(geometry_value_record));
  stream execution;
  device_buffer device_compact(compact.size(), device);
  device_buffer destination(3 * sizeof(T), device);
  std::vector<T> output(3, T(-37.0));
  copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
  copy_host_to_device_async(destination, 0, output.data(), output.size() * sizeof(T), execution);
  geometry_launch_common common = {};
  common.destination = static_cast<unsigned char *>(destination.opaque_handle()) + sizeof(T);
  common.compact_inputs = static_cast<const unsigned char *>(device_compact.opaque_handle());
  common.compact_input_bytes = compact.size();
  common.object_offset = object_offset;
  common.object_count = 2;
  common.image_offset = image_offset;
  common.image_count = 2;
  common.value_offset = value_offset;
  common.material_count = 3;
  common.default_material = 0;
  common.elements = common.point_count = 1;
  common.dimensions = 2;
  common.component = common.tensor_row = common.tensor_column = common.property = 0;
  common.axis_direction[0] = 0;
  common.axis_direction[1] = 1;
  common.axis_direction[2] = 2;
  common.loop_extent[0] = common.loop_extent[1] = common.loop_extent[2] = 1;
  common.cell_size[0] = common.cell_size[1] = common.cell_size[2] = 1.0;
  common.metric[0] = common.metric[4] = common.metric[8] = 1.0;
  common.inva = common.dt = 1.0;
  common.precision = precision;
  geometry_bulk_launch launch = {common, 0, 1};
  validate_geometry_bulk_launch(launch, compact.data(), compact.size());
  launch_material_geometry_bulk(launch, execution);
  copy_device_to_host_async(output.data(), destination, 0, output.size() * sizeof(T), execution);
  execution.synchronize();
  require(output[0] == T(-37.0) && output[1] == T(0.125) && output[2] == T(-37.0),
          "periodic geometry image did not win in canonical precedence order");

  std::vector<unsigned char> malformed = compact;
  geometry_image_record *bad_images = reinterpret_cast<geometry_image_record *>(
      malformed.data() + image_offset);
  bad_images[1].precedence = 3;
  require_invalid(
      [&]() { validate_geometry_bulk_launch(launch, malformed.data(), malformed.size()); },
      "out-of-order material geometry precedence was accepted");
}

template <typename T>
static void check_geometry_fixed_shapes(int device, scalar_precision precision) {
  struct ShapePayload {
    const char *name;
    geometry_object_record object;
    std::vector<double> parameters;
    std::vector<double> vertices;
    std::vector<double> indices;
    std::vector<double> auxiliary;
    std::vector<geometry_triangle_record> triangles;
    std::vector<geometry_bvh_record> bvh;
    std::vector<uint32_t> faces;
    int varying_direction;
    int loop_begin;
    double inva;
    bool first_inside;
    bool second_inside;
  };

  const auto run = [&](ShapePayload shape) {
    std::vector<unsigned char> compact;
    shape.object.parameter_offset = append_aligned(
        compact, shape.parameters.data(), shape.parameters.size() * sizeof(double),
        alignof(double));
    shape.object.parameter_count = shape.parameters.size();
    shape.object.vertex_offset = append_aligned(
        compact, shape.vertices.data(), shape.vertices.size() * sizeof(double),
        alignof(double));
    shape.object.vertex_count = shape.vertices.size() / 3;
    shape.object.index_offset = append_aligned(
        compact, shape.indices.data(), shape.indices.size() * sizeof(double), alignof(double));
    shape.object.index_count = shape.indices.size();
    shape.object.auxiliary_offset = append_aligned(
        compact, shape.auxiliary.data(), shape.auxiliary.size() * sizeof(double),
        alignof(double));
    shape.object.auxiliary_count = shape.auxiliary.size();
    shape.object.triangle_offset = append_aligned(
        compact, shape.triangles.data(),
        shape.triangles.size() * sizeof(geometry_triangle_record),
        alignof(geometry_triangle_record));
    shape.object.triangle_count = shape.triangles.size();
    shape.object.face_id_offset = append_aligned(
        compact, shape.faces.data(), shape.faces.size() * sizeof(uint32_t), alignof(uint32_t));
    shape.object.face_id_count = shape.faces.size();
    shape.object.bvh_offset = append_aligned(
        compact, shape.bvh.data(), shape.bvh.size() * sizeof(geometry_bvh_record),
        alignof(geometry_bvh_record));
    shape.object.bvh_count = shape.bvh.size();
    const size_t object_offset = append_record(compact, shape.object);
    geometry_image_record image = {};
    image.object = 0;
    image.ordinal = 0;
    image.precedence = 1;
    for (int axis = 0; axis < 3; ++axis) {
      image.low[axis] = shape.object.low[axis];
      image.high[axis] = shape.object.high[axis];
    }
    const size_t image_offset = append_record(compact, image);
    geometry_value_record values[2] = {};
    for (int material = 0; material < 2; ++material) {
      values[material].kind = geometry_value_kind::constant;
      values[material].flags = 1u;
      values[material].value_1 = material ? 0.25 : 1.0;
    }
    const size_t value_offset =
        append_aligned(compact, values, sizeof(values), alignof(geometry_value_record));

    stream execution;
    device_buffer device_compact(compact.size(), device);
    std::vector<T> output(4, T(-91.0));
    device_buffer destination(output.size() * sizeof(T), device);
    copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
    copy_host_to_device_async(destination, 0, output.data(), output.size() * sizeof(T), execution);
    geometry_launch_common common = {};
    common.destination = static_cast<unsigned char *>(destination.opaque_handle()) + sizeof(T);
    common.compact_inputs = static_cast<const unsigned char *>(device_compact.opaque_handle());
    common.compact_input_bytes = compact.size();
    common.object_offset = object_offset;
    common.object_count = 1;
    common.image_offset = image_offset;
    common.image_count = 1;
    common.value_offset = value_offset;
    common.material_count = 2;
    common.default_material = 0;
    common.elements = common.point_count = 2;
    common.dimensions = 2;
    common.component = common.tensor_row = common.tensor_column = common.property = 0;
    common.axis_direction[0] = shape.varying_direction;
    common.axis_direction[1] = shape.varying_direction == 0 ? 1 : 0;
    common.axis_direction[2] = 2;
    common.loop_begin[0] = shape.loop_begin;
    common.loop_extent[0] = 2;
    common.loop_extent[1] = common.loop_extent[2] = 1;
    common.strides[0] = 1;
    common.cell_size[0] = common.cell_size[1] = common.cell_size[2] = 4.0;
    common.metric[0] = common.metric[4] = common.metric[8] = 1.0;
    common.inva = shape.inva;
    common.dt = 0.1;
    common.precision = precision;
    geometry_bulk_launch launch = {common, 0, 2};
    validate_geometry_bulk_launch(launch, compact.data(), compact.size());
    launch_material_geometry_bulk(launch, execution);
    copy_device_to_host_async(output.data(), destination, 0, output.size() * sizeof(T), execution);
    execution.synchronize();
    require(output.front() == T(-91.0) && output.back() == T(-91.0),
            "fixed-shape geometry KAT overwrote an address guard");
    require(output[1] == T(shape.first_inside ? 0.25 : 1.0) &&
                output[2] == T(shape.second_inside ? 0.25 : 1.0),
            shape.name);
  };

  const auto bounds = [](geometry_object_record &object, double radius) {
    for (int axis = 0; axis < 3; ++axis) {
      object.low[axis] = -radius;
      object.high[axis] = radius;
    }
    object.material = 1;
  };

  ShapePayload sphere = {};
  sphere.name = "sphere containment KAT differs";
  sphere.object.kind = 4;
  bounds(sphere.object, 0.5);
  sphere.parameters = {0, 0, 0, 0.5};
  sphere.varying_direction = 0; sphere.loop_begin = 0; sphere.inva = 2.0;
  sphere.first_inside = true; sphere.second_inside = false;
  run(sphere);

  ShapePayload block = sphere;
  block.name = "block containment KAT differs";
  block.object.kind = 3; block.object.subtype = 0;
  bounds(block.object, 0.5);
  block.parameters.assign(25, 0.0);
  block.parameters[3] = block.parameters[7] = block.parameters[11] = 1.0;
  block.parameters[12] = block.parameters[13] = block.parameters[14] = 1.0;
  block.parameters[15] = block.parameters[19] = block.parameters[23] = 1.0;
  run(block);

  ShapePayload ellipsoid = block;
  ellipsoid.name = "ellipsoid containment KAT differs";
  ellipsoid.object.subtype = 1;
  ellipsoid.parameters.resize(28, 0.0);
  ellipsoid.parameters[24] = 1.0;
  ellipsoid.parameters[25] = ellipsoid.parameters[26] = ellipsoid.parameters[27] = 2.0;
  run(ellipsoid);

  ShapePayload transformed = block;
  transformed.name = "transformed block containment KAT differs";
  transformed.parameters[12] = 0.5;
  transformed.parameters[13] = 2.0;
  transformed.parameters[15] = transformed.parameters[19] = 0.0;
  transformed.parameters[16] = transformed.parameters[18] = 1.0;
  transformed.object.low[0] = -1.0; transformed.object.high[0] = 1.0;
  transformed.object.low[1] = -0.25; transformed.object.high[1] = 0.25;
  transformed.loop_begin = 2; transformed.inva = 0.75;
  run(transformed);

  ShapePayload cylinder = sphere;
  cylinder.name = "cylinder containment KAT differs";
  cylinder.object.kind = 5; cylinder.object.subtype = 0;
  cylinder.parameters = {0, 0, 0, 0, 0, 1, 0.5, 1.0, 0};
  run(cylinder);

  ShapePayload cone = cylinder;
  cone.name = "cone containment KAT differs";
  cone.object.subtype = 2;
  cone.parameters.push_back(0.25);
  run(cone);

  ShapePayload wedge = cylinder;
  wedge.name = "positive wedge containment KAT differs";
  wedge.object.subtype = 1;
  wedge.parameters = {0, 0, 0, 0, 0, 1, 0.5, 1.0, 1, 1.5707963267948966,
                      1, 0, 0, 1, 0, 0, 0, 1, 0};
  wedge.varying_direction = 1; wedge.loop_begin = -1; wedge.inva = 0.5;
  wedge.first_inside = false; wedge.second_inside = true;
  run(wedge);
  wedge.name = "negative wedge containment KAT differs";
  wedge.parameters[9] = -1.5707963267948966;
  wedge.first_inside = true; wedge.second_inside = false;
  run(wedge);

  ShapePayload prism = sphere;
  prism.name = "prism containment KAT differs";
  prism.object.kind = 2; prism.object.subtype = 0; prism.object.closed = 1;
  prism.object.fixed_vertex_count = 4;
  prism.parameters.assign(29, 0.0);
  prism.parameters[3] = 1.0;
  prism.parameters[6] = 1.0;
  prism.parameters[11] = prism.parameters[15] = prism.parameters[19] = 1.0;
  prism.parameters[20] = prism.parameters[24] = prism.parameters[28] = 1.0;
  prism.vertices = {-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0};
  prism.auxiliary = prism.vertices;
  prism.auxiliary.insert(prism.auxiliary.end(), 12, 0.0);
  const double prism_top[] = {-0.5, -0.5, 1, 0.5, -0.5, 1,
                               0.5, 0.5, 1, -0.5, 0.5, 1};
  prism.auxiliary.insert(prism.auxiliary.end(), prism_top, prism_top + 12);
  run(prism);

  ShapePayload mesh = sphere;
  mesh.name = "mesh containment KAT differs";
  mesh.object.kind = 1; mesh.object.subtype = 1; mesh.object.closed = 1;
  mesh.object.mesh_lengthscale = 2.0;
  bounds(mesh.object, 1.0);
  mesh.parameters = {0, 0, 0, 1};
  mesh.vertices = {1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1};
  mesh.indices = {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2};
  const uint32_t triangle_vertices[4][3] = {{0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {1, 3, 2}};
  for (int triangle = 0; triangle < 4; ++triangle) {
    geometry_triangle_record record = {};
    for (int vertex = 0; vertex < 3; ++vertex)
      record.vertex[vertex] = triangle_vertices[triangle][vertex];
    for (int axis = 0; axis < 3; ++axis) {
      record.low[axis] = -1.0;
      record.high[axis] = 1.0;
    }
    mesh.triangles.push_back(record);
  }
  geometry_bvh_record node = {};
  for (int axis = 0; axis < 3; ++axis) {
    node.low[axis] = -1.0;
    node.high[axis] = 1.0;
  }
  node.left = node.right = UINT32_MAX;
  node.first_face = 0; node.face_count = 4; node.escape = 1; node.leaf = 1;
  mesh.bvh.push_back(node);
  mesh.faces = {0, 1, 2, 3};
  run(mesh);
}

static double inverse_tensor_member_for_testing(const double tensor[6], int row, int column) {
  const double determinant =
      tensor[0] * tensor[1] * tensor[2] - tensor[4] * tensor[1] * tensor[4] +
      2.0 * tensor[3] * tensor[5] * tensor[4] - tensor[3] * tensor[3] * tensor[2] -
      tensor[5] * tensor[5] * tensor[0];
  if (row == 0 && column == 0)
    return (tensor[1] * tensor[2] - tensor[5] * tensor[5]) / determinant;
  if (row == 0 && column == 1)
    return (tensor[5] * tensor[4] - tensor[3] * tensor[2]) / determinant;
  throw std::logic_error("unsupported geometry tensor oracle member");
}

template <typename T>
static void check_geometry_file_grid_values(int device, scalar_precision precision) {
  const auto run = [&](geometry_value_kind kind, int row, int column) {
    std::vector<unsigned char> compact;
    const double parameters[4] = {0, 0, 0, 1.0};
    const size_t parameter_offset =
        append_aligned(compact, parameters, sizeof(parameters), alignof(double));
    geometry_object_record object = {};
    object.kind = 4;
    object.material = 1;
    object.parameter_offset = parameter_offset;
    object.parameter_count = 4;
    for (int axis = 0; axis < 3; ++axis) {
      object.low[axis] = -1.0;
      object.high[axis] = 1.0;
    }
    const size_t object_offset = append_record(compact, object);
    geometry_image_record image = {};
    image.object = 0; image.ordinal = 0; image.precedence = 1;
    for (int axis = 0; axis < 3; ++axis) {
      image.low[axis] = -1.0;
      image.high[axis] = 1.0;
    }
    const size_t image_offset = append_record(compact, image);
    const double samples[3] = {kind == geometry_value_kind::file_epsilon ? 2.0 : 0.0,
                               kind == geometry_value_kind::file_epsilon ? 3.0 : 0.5,
                               kind == geometry_value_kind::file_epsilon ? 4.0 : 1.0};
    const size_t sample_offset =
        append_aligned(compact, samples, sizeof(samples), alignof(double));
    geometry_value_record values[2] = {};
    values[0].kind = geometry_value_kind::constant;
    values[0].flags = 1u;
    values[0].value_1 = 1.0;
    values[1].kind = kind;
    values[1].overlap_kind = kind == geometry_value_kind::file_epsilon ? 0 : 3;
    values[1].dimensions[0] = 3;
    values[1].dimensions[1] = values[1].dimensions[2] = 1;
    values[1].sample_offset = sample_offset;
    values[1].sample_count = 3;
    const double first[6] = {2.0, 3.0, 4.0, 0.25, 0.1, -0.2};
    const double second[6] = {5.0, 6.0, 7.0, -0.3, 0.2, 0.15};
    for (int i = 0; i < 6; ++i) {
      values[1].tensor_1[i] = first[i];
      values[1].tensor_2[i] = second[i];
    }
    const size_t value_offset =
        append_aligned(compact, values, sizeof(values), alignof(geometry_value_record));

    stream execution;
    device_buffer device_compact(compact.size(), device);
    std::vector<T> output(5, T(-73.0));
    device_buffer destination(output.size() * sizeof(T), device);
    copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
    copy_host_to_device_async(destination, 0, output.data(), output.size() * sizeof(T), execution);
    geometry_launch_common common = {};
    common.destination = static_cast<unsigned char *>(destination.opaque_handle()) + sizeof(T);
    common.compact_inputs = static_cast<const unsigned char *>(device_compact.opaque_handle());
    common.compact_input_bytes = compact.size();
    common.object_offset = object_offset; common.object_count = 1;
    common.image_offset = image_offset; common.image_count = 1;
    common.value_offset = value_offset; common.material_count = 2;
    common.default_material = 0; common.elements = common.point_count = 3;
    common.dimensions = 2; common.component = 0;
    common.tensor_row = row; common.tensor_column = column; common.property = 0;
    common.axis_direction[0] = 0; common.axis_direction[1] = 1; common.axis_direction[2] = 2;
    common.loop_begin[0] = -2;
    common.loop_extent[0] = 3;
    common.loop_extent[1] = common.loop_extent[2] = 1;
    common.strides[0] = 1;
    common.cell_size[0] = common.cell_size[1] = common.cell_size[2] = 2.0;
    common.metric[0] = common.metric[4] = common.metric[8] = 1.0;
    common.inva = 1.0; common.dt = 0.1; common.precision = precision;
    geometry_bulk_launch launch = {common, 0, 3};
    validate_geometry_bulk_launch(launch, compact.data(), compact.size());
    launch_material_geometry_bulk(launch, execution);
    copy_device_to_host_async(output.data(), destination, 0, output.size() * sizeof(T), execution);
    execution.synchronize();
    require(output.front() == T(-73.0) && output.back() == T(-73.0),
            "geometry FILE/Grid value KAT overwrote an address guard");
    for (int point = 0; point < 3; ++point) {
      double expected = 0.0;
      if (kind == geometry_value_kind::file_epsilon)
        expected = row == column ? 1.0 / samples[point] : 0.0;
      else {
        const double weight = samples[point];
        double tensor[6];
        for (int i = 0; i < 6; ++i)
          tensor[i] = first[i] * (1.0 - weight) + second[i] * weight;
        expected = inverse_tensor_member_for_testing(tensor, row, column);
      }
      const double tolerance = sizeof(T) == sizeof(float) ? 2e-7 : 2e-15;
      require(std::fabs(double(output[size_t(point) + 1]) - expected) <= tolerance,
              "geometry FILE/Grid endpoint or fill value differs");
    }
  };
  run(geometry_value_kind::file_epsilon, 0, 0);
  run(geometry_value_kind::file_epsilon, 0, 1);
  run(geometry_value_kind::grid_tensor, 0, 0);
  run(geometry_value_kind::grid_tensor, 0, 1);
}

template <typename T>
static void check_length(int device, scalar_precision precision, size_t elements,
                         bool logical_single = false) {
  stream execution;
  const T sentinel = T(-19.25);
  const double fill_value = 0.12345678901234567;
  std::vector<T> destination(elements + 2, sentinel);
  device_buffer device_destination(destination.size() * sizeof(T), device);
  copy_host_to_device_async(device_destination, 0, destination.data(),
                            destination.size() * sizeof(T), execution);
  material_fill_launch fill = {};
  fill.destination = static_cast<unsigned char *>(device_destination.opaque_handle()) + sizeof(T);
  fill.elements = elements;
  fill.value = fill_value;
  fill.phase = 0;
  fill.precision = precision;
  launch_material_fill(fill, execution);
  copy_device_to_host_async(destination.data(), device_destination, 0,
                            destination.size() * sizeof(T), execution);
  execution.synchronize();
  require(destination.front() == sentinel && destination.back() == sentinel,
          "material fill overwrote a guard element");
  for (size_t i = 0; i < elements; ++i)
    require(destination[i + 1] == T(fill_value), "material fill value differs");
  if (elements) {
    fill.value = -0.0;
    launch_material_fill(fill, execution);
    copy_device_to_host_async(destination.data(), device_destination, 0,
                              destination.size() * sizeof(T), execution);
    execution.synchronize();
    for (size_t i = 0; i < elements; ++i)
      require(destination[i + 1] == T(0) && std::signbit(destination[i + 1]),
              "material fill did not preserve negative zero");
  }

  std::vector<double> profile(elements);
  for (size_t i = 0; i < elements; ++i) {
    profile[i] = 0.03125 + 0.00013 * i;
  }
  const size_t profile_offset = 16;
  std::vector<unsigned char> compact(profile_offset + elements * sizeof(double), 0xa5);
  if (elements)
    std::memcpy(compact.data() + profile_offset, profile.data(), elements * sizeof(double));
  device_buffer device_profile(compact.size(), device);
  device_buffer device_sigma((elements + 2) * sizeof(T), device);
  device_buffer device_kappa((elements + 2) * sizeof(T), device);
  device_buffer device_inverse((elements + 2) * sizeof(T), device);
  std::vector<T> guards(elements + 2, sentinel);
  copy_host_to_device_async(device_profile, 0, compact.data(), compact.size(), execution);
  copy_host_to_device_async(device_sigma, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_kappa, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_inverse, 0, guards.data(), guards.size() * sizeof(T), execution);
  material_pml_launch pml = {};
  pml.sigma_destination = static_cast<unsigned char *>(device_sigma.opaque_handle()) + sizeof(T);
  pml.kappa_destination = static_cast<unsigned char *>(device_kappa.opaque_handle()) + sizeof(T);
  pml.sigma_inv_destination =
      static_cast<unsigned char *>(device_inverse.opaque_handle()) + sizeof(T);
  pml.compact_inputs = static_cast<const unsigned char *>(device_profile.opaque_handle());
  pml.compact_input_bytes = compact.size();
  pml.profile_offset = profile_offset;
  pml.elements = elements;
  pml.little_corner = 0;
  pml.resolution = 8.0;
  pml.dt = 0.03;
  pml.thickness = 10.0;
  pml.boundary_location = 0.0;
  pml.r_asymptotic = 1e-8;
  pml.mean_stretch = 1.5;
  pml.profile_integral = 0.25;
  pml.profile_integral_u = 0.2;
  pml.thickness_cells = int(pml.thickness * (2 * pml.resolution) + 0.5);
  pml.profile_active = true;
  pml.logical_single = logical_single;
  pml.precision = precision;
  launch_material_pml(pml, execution);
  std::vector<T> observed_sigma(elements + 2), observed_kappa(elements + 2),
      observed_inverse(elements + 2);
  copy_device_to_host_async(observed_sigma.data(), device_sigma, 0,
                            observed_sigma.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_kappa.data(), device_kappa, 0,
                            observed_kappa.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_inverse.data(), device_inverse, 0,
                            observed_inverse.size() * sizeof(T), execution);
  execution.synchronize();
  require(observed_sigma.front() == sentinel && observed_sigma.back() == sentinel &&
              observed_kappa.front() == sentinel && observed_kappa.back() == sentinel &&
              observed_inverse.front() == sentinel && observed_inverse.back() == sentinel,
          "material PML initialization overwrote a guard element");
  for (size_t i = 0; i < elements; ++i) {
    const double here = double(i) * 0.5 / pml.resolution;
    const double x = 0.5 / pml.resolution *
                     (int(pml.thickness * (2 * pml.resolution) + 0.5) -
                      int(std::fabs(pml.boundary_location - here) * (2 * pml.resolution) + 0.5));
    double sigma = 0.0, kappa = 1.0, inverse = 1.0;
    if (x > 0) {
      sigma = 0.5 * pml.dt *
              ((-std::log(pml.r_asymptotic)) /
               (4 * pml.thickness * pml.profile_integral)) *
              profile[i];
      kappa = 1 + ((pml.mean_stretch - 1) / pml.profile_integral_u) * profile[i] *
                      (x / pml.thickness);
      if (logical_single) {
        sigma = float(sigma);
        kappa = float(kappa);
        inverse = float(1 / (float(kappa) + float(sigma)));
      }
      else
        inverse = 1 / (kappa + sigma);
    }
    require(observed_sigma[i + 1] == T(sigma), "material PML sigma differs");
    require(observed_kappa[i + 1] == T(kappa), "material PML kappa differs");
    require(observed_inverse[i + 1] == T(inverse), "material PML sigma inverse differs");
  }
}

template <typename T>
static void check_failures(int device, scalar_precision precision) {
  stream execution;
  std::vector<T> initial(5, T(-7));
  device_buffer destination(initial.size() * sizeof(T), device);
  copy_host_to_device_async(destination, 0, initial.data(), initial.size() * sizeof(T), execution);
  execution.synchronize();
  material_fill_launch fill = {destination.opaque_handle(), initial.size(), 3.0, 0, precision};
  testing::fail_next(testing::failure_point::material_pointwise_launch);
  bool rejected = false;
  try { launch_material_fill(fill, execution); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "injected material pointwise launch failure was ignored");
  std::vector<T> observed(initial.size());
  copy_device_to_host_async(observed.data(), destination, 0, observed.size() * sizeof(T), execution);
  execution.synchronize();
  require_exact(observed, initial, "failed material pointwise launch changed output");

  material_fill_launch malformed = fill;
  malformed.destination = NULL;
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "null material fill destination was accepted");
  malformed = fill;
  malformed.destination = static_cast<unsigned char *>(destination.opaque_handle()) + 1;
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "misaligned material fill destination was accepted");
  malformed = fill;
  malformed.precision = static_cast<scalar_precision>(99);
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "invalid material fill precision was accepted");
  malformed = fill;
  malformed.elements = std::numeric_limits<size_t>::max();
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "overflowing material fill range was accepted");
  material_fill_launch empty = {NULL, 0, 1.0, 0, precision};
  launch_material_fill(empty, execution);

  const size_t profile_offset = 16;
  const double profile[2] = {0.25, 0.5};
  std::vector<unsigned char> compact_bytes(profile_offset + sizeof(profile), 0);
  std::memcpy(compact_bytes.data() + profile_offset, profile, sizeof(profile));
  device_buffer compact(compact_bytes.size(), device), kappa(initial.size() * sizeof(T), device),
      inverse(initial.size() * sizeof(T), device);
  copy_host_to_device_async(compact, 0, compact_bytes.data(), compact_bytes.size(), execution);
  copy_host_to_device_async(destination, 0, initial.data(), initial.size() * sizeof(T), execution);
  copy_host_to_device_async(kappa, 0, initial.data(), initial.size() * sizeof(T), execution);
  copy_host_to_device_async(inverse, 0, initial.data(), initial.size() * sizeof(T), execution);
  execution.synchronize();
  material_pml_launch pml = {};
  pml.sigma_destination = destination.opaque_handle();
  pml.kappa_destination = kappa.opaque_handle();
  pml.sigma_inv_destination = inverse.opaque_handle();
  pml.compact_inputs = static_cast<const unsigned char *>(compact.opaque_handle());
  pml.compact_input_bytes = compact.size();
  pml.profile_offset = profile_offset;
  pml.elements = 2;
  pml.resolution = 8;
  pml.dt = 0.05;
  pml.thickness = 0.25;
  pml.boundary_location = 0;
  pml.r_asymptotic = 1e-8;
  pml.mean_stretch = 1.2;
  pml.profile_integral = 1.0 / 3.0;
  pml.profile_integral_u = 0.25;
  pml.thickness_cells = 4;
  pml.profile_active = true;
  pml.precision = precision;
  const auto reject_pml = [&](const material_pml_launch &candidate, const char *message) {
    require_invalid([&]() { launch_material_pml(candidate, execution); }, message);
  };
  material_pml_launch bad_pml = pml;
  bad_pml.profile_offset = 1;
  reject_pml(bad_pml, "misaligned material PML compact offset was accepted");
  bad_pml = pml;
  bad_pml.profile_offset = compact.size();
  reject_pml(bad_pml, "out-of-range material PML compact input was accepted");
  bad_pml = pml;
  bad_pml.compact_inputs += 1;
  reject_pml(bad_pml, "misaligned material PML compact base was accepted");
  bad_pml = pml;
  bad_pml.kappa_destination = bad_pml.sigma_destination;
  reject_pml(bad_pml, "aliased material PML destinations were accepted");
  bad_pml = pml;
  bad_pml.kappa_destination = static_cast<unsigned char *>(bad_pml.sigma_destination) + sizeof(T);
  reject_pml(bad_pml, "overlapping material PML destinations were accepted");
  bad_pml = pml;
  bad_pml.precision = static_cast<scalar_precision>(99);
  reject_pml(bad_pml, "invalid material PML precision was accepted");
  bad_pml = pml;
  bad_pml.resolution = std::numeric_limits<double>::infinity();
  reject_pml(bad_pml, "non-finite material PML resolution was accepted");
  bad_pml = pml;
  bad_pml.dt = 0;
  reject_pml(bad_pml, "zero material PML timestep was accepted");
  bad_pml = pml;
  bad_pml.r_asymptotic = 1;
  reject_pml(bad_pml, "invalid material PML asymptotic reflection was accepted");
  bad_pml = pml;
  bad_pml.profile_integral = 0;
  reject_pml(bad_pml, "zero material PML profile integral was accepted");
  bad_pml = pml;
  bad_pml.thickness_cells = 0;
  reject_pml(bad_pml, "zero material PML thickness cells were accepted");
  bad_pml = pml;
  --bad_pml.thickness_cells;
  reject_pml(bad_pml, "inconsistent material PML thickness cells were accepted");
  bad_pml = pml;
  bad_pml.little_corner = std::numeric_limits<int>::max();
  bad_pml.elements = 2;
  reject_pml(bad_pml, "overflowing material PML logical index was accepted");
  bad_pml = pml;
  bad_pml.boundary_location = double(std::numeric_limits<int>::max());
  reject_pml(bad_pml, "overflowing material PML distance cast was accepted");

  std::vector<T> unchanged(initial.size()), unchanged_kappa(initial.size()),
      unchanged_inverse(initial.size());
  copy_device_to_host_async(unchanged.data(), destination, 0,
                            unchanged.size() * sizeof(T), execution);
  copy_device_to_host_async(unchanged_kappa.data(), kappa, 0,
                            unchanged_kappa.size() * sizeof(T), execution);
  copy_device_to_host_async(unchanged_inverse.data(), inverse, 0,
                            unchanged_inverse.size() * sizeof(T), execution);
  execution.synchronize();
  require_exact(unchanged, initial, "rejected material descriptor changed sigma output");
  require_exact(unchanged_kappa, initial, "rejected material descriptor changed kappa output");
  require_exact(unchanged_inverse, initial, "rejected material descriptor changed inverse output");
}

template <typename T>
static void check_absorber(int device, scalar_precision precision,
                           bool logical_single = false) {
  stream execution;
  const size_t elements = 9;
  const T sentinel = T(-31.5);
  std::vector<double> low = {0.0, 0.4, 0.9};
  std::vector<double> high = {0.0, 0.7, 1.3};
  const size_t header_offset = 16;
  const size_t header_bytes = 2 * sizeof(material_absorber_header);
  const size_t low_offset = header_offset + header_bytes;
  const size_t high_offset = low_offset + low.size() * sizeof(double);
  std::vector<unsigned char> compact(high_offset + high.size() * sizeof(double), 0);
  material_absorber_header headers[2] = {};
  headers[0].version = headers[1].version = 1;
  headers[0].direction = headers[1].direction = 2;
  headers[0].side = 1;
  headers[1].side = 0;
  headers[0].sample_offset = low_offset;
  headers[1].sample_offset = high_offset;
  headers[0].sample_count = low.size();
  headers[1].sample_count = high.size();
  headers[0].thickness = headers[1].thickness = 1.0;
  headers[0].sample_spacing = headers[1].sample_spacing = 0.5;
  std::memcpy(compact.data() + header_offset, headers, sizeof(headers));
  std::memcpy(compact.data() + low_offset, low.data(), low.size() * sizeof(double));
  std::memcpy(compact.data() + high_offset, high.data(), high.size() * sizeof(double));
  validate_material_absorber_headers(compact.data(), compact.size(), header_offset, 2);
  const auto reject_headers = [&](const std::vector<unsigned char> &candidate,
                                  size_t offset, size_t count, const char *message) {
    require_invalid(
        [&]() {
          validate_material_absorber_headers(candidate.data(), candidate.size(), offset, count);
        },
        message);
  };
  std::vector<unsigned char> malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].version = 2;
  reject_headers(malformed, header_offset, 2, "stale absorber header version was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].reserved = 1;
  reject_headers(malformed, header_offset, 2, "nonzero absorber reserved field was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].direction = 5;
  reject_headers(malformed, header_offset, 2, "invalid absorber direction was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].side = 2;
  reject_headers(malformed, header_offset, 2, "invalid absorber side was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_count = 1;
  reject_headers(malformed, header_offset, 2, "short absorber sample count was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_count =
      std::numeric_limits<uint64_t>::max();
  reject_headers(malformed, header_offset, 2, "overflowing absorber sample count was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_offset =
      header_offset;
  reject_headers(malformed, header_offset, 2, "absorber samples overlapping headers were accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_offset =
      low_offset + 1;
  reject_headers(malformed, header_offset, 2, "misaligned absorber sample offset was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].thickness = 0;
  reject_headers(malformed, header_offset, 2, "zero absorber thickness was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_spacing =
      0.25;
  reject_headers(malformed, header_offset, 2, "inconsistent absorber spacing was accepted");
  malformed = compact;
  reinterpret_cast<double *>(malformed.data() + low_offset)[1] =
      std::numeric_limits<double>::quiet_NaN();
  reject_headers(malformed, header_offset, 2, "non-finite absorber sample was accepted");
  reject_headers(compact, header_offset + 1, 2, "misaligned absorber header block was accepted");
  reject_headers(compact, compact.size(), 1, "out-of-range absorber header block was accepted");
  require_invalid(
      [&]() {
        validate_material_absorber_headers(compact.data() + 1, compact.size() - 1,
                                            header_offset, 2);
      },
      "misaligned absorber compact base was accepted");

  device_buffer device_compact(compact.size(), device);
  copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
  std::vector<T> guards(elements + 2, sentinel), observed_cnd(elements + 2),
      observed_inv(elements + 2);
  device_buffer device_cnd(guards.size() * sizeof(T), device);
  device_buffer device_inv(guards.size() * sizeof(T), device);
  copy_host_to_device_async(device_cnd, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_inv, 0, guards.data(), guards.size() * sizeof(T), execution);

  material_conductivity_launch launch = {};
  launch.conductivity_destination =
      static_cast<unsigned char *>(device_cnd.opaque_handle()) + sizeof(T);
  launch.condinv_destination =
      static_cast<unsigned char *>(device_inv.opaque_handle()) + sizeof(T);
  launch.compact_inputs = static_cast<const unsigned char *>(device_compact.opaque_handle());
  launch.compact_input_bytes = compact.size();
  launch.absorber_header_offset = header_offset;
  launch.absorber_count = 2;
  launch.elements = elements;
  launch.loop_count = elements;
  launch.component = 14;
  launch.dimensions = 0;
  launch.axis_direction[0] = 0;
  launch.axis_direction[1] = 1;
  launch.axis_direction[2] = 2;
  launch.loop_begin[0] = launch.loop_begin[1] = 0;
  launch.loop_begin[2] = -8;
  launch.little_corner[0] = launch.little_corner[1] = 0;
  launch.little_corner[2] = -8;
  launch.loop_extent[0] = launch.loop_extent[1] = 1;
  launch.loop_extent[2] = elements;
  launch.strides[0] = launch.strides[1] = 0;
  launch.strides[2] = 1;
  launch.cell_size[2] = 4.0;
  launch.inva = 0.5;
  launch.base_conductivity = 0.25;
  launch.dt = 0.03;
  launch.logical_single = logical_single;
  launch.precision = precision;
  launch_material_conductivity(launch, execution);
  copy_device_to_host_async(observed_cnd.data(), device_cnd, 0,
                            observed_cnd.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_inv.data(), device_inv, 0,
                            observed_inv.size() * sizeof(T), execution);
  execution.synchronize();
  require(observed_cnd.front() == sentinel && observed_cnd.back() == sentinel &&
              observed_inv.front() == sentinel && observed_inv.back() == sentinel,
          "material conductivity initialization overwrote a guard element");
  for (size_t i = 0; i < elements; ++i) {
    const double x = -2.0 + 0.5 * i;
    double expected = 0.25;
    if (x <= -1.0) {
      const double u = 2 * (-1.0 - x);
      const int sample = int(u);
      expected += sample >= 2 ? low[2]
                              : low[sample] * (1 - (u - sample)) + low[sample + 1] * (u - sample);
    }
    if (x >= 1.0) {
      const double u = 2 * (x - 1.0);
      const int sample = int(u);
      expected += sample >= 2
                      ? high[2]
                      : high[sample] * (1 - (u - sample)) + high[sample + 1] * (u - sample);
    }
    double expected_inverse;
    if (logical_single) {
      expected = float(expected);
      expected_inverse = float(1 / (1 + double(float(expected)) * launch.dt * 0.5));
    }
    else
      expected_inverse = 1 / (1 + expected * launch.dt * 0.5);
    require(observed_cnd[i + 1] == T(expected), "material absorber conductivity differs");
    require(observed_inv[i + 1] == T(expected_inverse),
            "material absorber conductivity inverse differs");
  }

  copy_host_to_device_async(device_cnd, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_inv, 0, guards.data(), guards.size() * sizeof(T), execution);
  execution.synchronize();
  const auto reject_launch = [&](const material_conductivity_launch &candidate,
                                 const char *message) {
    require_invalid([&]() { launch_material_conductivity(candidate, execution); }, message);
  };
  material_conductivity_launch bad = launch;
  bad.compact_inputs += 1;
  reject_launch(bad, "misaligned conductivity compact base was accepted");
  bad = launch;
  bad.absorber_header_offset += 1;
  reject_launch(bad, "misaligned conductivity header offset was accepted");
  bad = launch;
  bad.absorber_count = std::numeric_limits<size_t>::max();
  reject_launch(bad, "overflowing conductivity header count was accepted");
  bad = launch;
  bad.condinv_destination = bad.conductivity_destination;
  reject_launch(bad, "aliased conductivity destinations were accepted");
  bad = launch;
  bad.condinv_destination = static_cast<unsigned char *>(bad.conductivity_destination) + sizeof(T);
  reject_launch(bad, "overlapping conductivity destinations were accepted");
  bad = launch;
  bad.precision = static_cast<scalar_precision>(99);
  reject_launch(bad, "invalid conductivity precision was accepted");
  bad = launch;
  bad.dimensions = 4;
  reject_launch(bad, "invalid conductivity dimensions were accepted");
  bad = launch;
  bad.axis_direction[1] = 5;
  reject_launch(bad, "invalid conductivity axis direction was accepted");
  bad = launch;
  bad.loop_extent[1] = 0;
  reject_launch(bad, "zero conductivity loop extent was accepted");
  bad = launch;
  bad.loop_extent[0] = std::numeric_limits<size_t>::max();
  bad.loop_extent[1] = 2;
  reject_launch(bad, "overflowing conductivity loop product was accepted");
  bad = launch;
  bad.loop_count -= 1;
  reject_launch(bad, "mismatched conductivity loop count was accepted");
  bad = launch;
  bad.strides[2] = -1;
  reject_launch(bad, "negative conductivity stride was accepted");
  bad = launch;
  bad.elements -= 1;
  reject_launch(bad, "out-of-range conductivity destination index was accepted");
  bad = launch;
  bad.loop_begin[0] = std::numeric_limits<int>::max();
  bad.little_corner[0] = std::numeric_limits<int>::min();
  reject_launch(bad, "overflowing conductivity stagger subtraction was accepted");
  bad = launch;
  bad.loop_base_offset[0] = 1;
  reject_launch(bad, "inconsistent conductivity base offset was accepted");
  bad = launch;
  bad.dt = std::numeric_limits<double>::quiet_NaN();
  reject_launch(bad, "non-finite conductivity timestep was accepted");
  bad = launch;
  bad.cell_size[2] = std::numeric_limits<double>::infinity();
  reject_launch(bad, "non-finite conductivity cell size was accepted");
  bad = launch;
  bad.inva = std::numeric_limits<double>::max();
  bad.loop_begin[2] = std::numeric_limits<int>::max();
  reject_launch(bad, "overflowing conductivity coordinate was accepted");
  copy_device_to_host_async(observed_cnd.data(), device_cnd, 0,
                            observed_cnd.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_inv.data(), device_inv, 0,
                            observed_inv.size() * sizeof(T), execution);
  execution.synchronize();
  require_exact(observed_cnd, guards,
                "rejected conductivity descriptor changed conductivity output");
  require_exact(observed_inv, guards,
                "rejected conductivity descriptor changed inverse output");
}

static size_t aligned_offset(size_t offset, size_t alignment) {
  return offset + (alignment - offset % alignment) % alignment;
}

static int mirror_index(int i, int n) {
  return i >= n ? 2 * n - 1 - i : (i < 0 ? -1 - i : i);
}

static double scalar_interpolate(double r, const std::vector<double> &samples) {
  r = r < 0 ? -r : (r > 1 ? 1 - r : r);
  const int n = int(samples.size());
  const int first = mirror_index(int(r * n), n);
  double d = r * n - first - 0.5;
  const int second = mirror_index(d >= 0 ? first + 1 : first - 1, n);
  d = std::fabs(d);
  return samples[first] * (1 - d) + samples[second] * d;
}

static double table_interpolate_3d(const double normalized[3], const uint32_t dimensions[3],
                                   const std::vector<double> &samples) {
  int first[3], second[3];
  double fraction[3];
  for (int axis = 0; axis < 3; ++axis) {
    double r = normalized[axis];
    r = r < 0 ? -r : (r > 1 ? 1 - r : r);
    first[axis] = mirror_index(int(r * dimensions[axis]), int(dimensions[axis]));
    double delta = r * dimensions[axis] - first[axis] - 0.5;
    second[axis] = mirror_index(delta >= 0 ? first[axis] + 1 : first[axis] - 1,
                                int(dimensions[axis]));
    fraction[axis] = std::fabs(delta);
  }
  const auto at = [&](int x, int y, int z) {
    return samples[(size_t(x) * dimensions[1] + size_t(y)) * dimensions[2] + size_t(z)];
  };
  const auto lerp = [](double a, double b, double t) { return a * (1 - t) + b * t; };
  return lerp(lerp(lerp(at(first[0], first[1], first[2]),
                         at(second[0], first[1], first[2]), fraction[0]),
                   lerp(at(first[0], second[1], first[2]),
                        at(second[0], second[1], first[2]), fraction[0]),
                   fraction[1]),
              lerp(lerp(at(first[0], first[1], second[2]),
                         at(second[0], first[1], second[2]), fraction[0]),
                   lerp(at(first[0], second[1], second[2]),
                        at(second[0], second[1], second[2]), fraction[0]),
                   fraction[1]),
              fraction[2]);
}

template <typename T>
static void check_table_length(int device, scalar_precision precision, size_t elements,
                               material_table_kind kind) {
  stream execution;
  const T sentinel = T(-63.5);
  const size_t leading_guards = 3, trailing_guards = 5;
  const std::vector<double> samples = {0.2, 0.8};
  material_table_header header = {};
  header.version = 1;
  header.material_id = 41;
  header.kind = kind;
  header.overlap_kind = 1;
  header.dimensions[0] = header.dimensions[1] = 1;
  header.dimensions[2] = 2;
  size_t offset = sizeof(header);
  material_medium_header medium1 = {}, medium2 = {};
  if (kind == material_table_kind::material_grid) {
    offset = aligned_offset(offset, alignof(material_medium_header));
    header.medium_1_offset = offset;
    offset += sizeof(material_medium_header);
    offset = aligned_offset(offset, alignof(material_medium_header));
    header.medium_2_offset = offset;
    offset += sizeof(material_medium_header);
    medium1.version = medium2.version = 1;
    for (int axis = 0; axis < 3; ++axis) {
      medium1.epsilon_diagonal[axis] = 2.0 + axis;
      medium2.epsilon_diagonal[axis] = 5.0 + axis;
    }
  }
  offset = aligned_offset(offset, alignof(double));
  header.sample_offset = offset;
  header.sample_count = samples.size();
  std::vector<unsigned char> compact(offset + samples.size() * sizeof(double), 0);
  std::memcpy(compact.data(), &header, sizeof(header));
  if (kind == material_table_kind::material_grid) {
    std::memcpy(compact.data() + header.medium_1_offset, &medium1, sizeof(medium1));
    std::memcpy(compact.data() + header.medium_2_offset, &medium2, sizeof(medium2));
  }
  std::memcpy(compact.data() + header.sample_offset, samples.data(),
              samples.size() * sizeof(double));

  std::vector<T> guarded(leading_guards + elements + trailing_guards, sentinel), observed;
  device_buffer device_compact(compact.size(), device);
  device_buffer destination(guarded.size() * sizeof(T), device);
  copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
  copy_host_to_device_async(destination, 0, guarded.data(), guarded.size() * sizeof(T), execution);
  material_table_launch launch = {};
  launch.destination = static_cast<unsigned char *>(destination.opaque_handle()) +
                       leading_guards * sizeof(T);
  launch.compact_inputs = static_cast<const unsigned char *>(device_compact.opaque_handle());
  launch.compact_input_bytes = compact.size();
  launch.elements = elements;
  launch.loop_count = elements;
  launch.table_kind = kind;
  launch.operation = kind == material_table_kind::file_scalar_epsilon
                         ? material_table_operation::file_chi1inv
                         : material_table_operation::grid_chi1inv;
  launch.source_material_id = header.material_id;
  launch.destination_component = launch.query_component = 4;
  launch.tensor_row = launch.tensor_column = 2;
  launch.susceptibility_field_type = 8;
  launch.dimensions = 0;
  launch.axis_direction[0] = 0;
  launch.axis_direction[1] = 1;
  launch.axis_direction[2] = 2;
  launch.loop_extent[0] = launch.loop_extent[1] = 1;
  launch.loop_extent[2] = elements ? elements : 1;
  launch.strides[2] = 1;
  launch.loop_end[2] = elements ? int(2 * (elements - 1)) : 0;
  launch.cell_size[2] = std::max(2.0, 2.0 * double(elements));
  launch.inva = 1.0;
  launch.dt = 0.0625;
  launch.precision = precision;
  if (elements) validate_material_table_launch(launch, compact.data(), compact.size());
  launch_material_table(launch, execution);
  observed.resize(guarded.size());
  copy_device_to_host_async(observed.data(), destination, 0,
                            observed.size() * sizeof(T), execution);
  execution.synchronize();
  for (size_t i = 0; i < leading_guards; ++i)
    require(observed[i] == sentinel, "material table length launch overwrote a leading guard");
  for (size_t i = leading_guards + elements; i < observed.size(); ++i)
    require(observed[i] == sentinel, "material table length launch overwrote a trailing guard");
  for (size_t i = 0; i < elements; ++i) {
    const double normalized = 0.5 + double(i) / launch.cell_size[2];
    const double weight = scalar_interpolate(normalized, samples);
    const double epsilon = kind == material_table_kind::file_scalar_epsilon
                               ? weight
                               : medium1.epsilon_diagonal[2] +
                                     weight * (medium2.epsilon_diagonal[2] -
                                               medium1.epsilon_diagonal[2]);
    require(observed[leading_guards + i] == T(1.0 / epsilon),
            "material table length/tail interpolation differs");
  }
}

template <typename T>
static void check_table_materials(int device, scalar_precision precision,
                                  bool logical_single = false) {
  stream execution;
  const size_t elements = 5;
  const T sentinel = T(-41.25);
  std::vector<T> guards(elements + 2, sentinel), observed(elements + 2), inverse(elements + 2);
  device_buffer destination(guards.size() * sizeof(T), device);
  device_buffer secondary(guards.size() * sizeof(T), device);

  const auto make_launch = [&](device_buffer &compact, size_t compact_bytes,
                               material_table_kind kind,
                               material_table_operation operation) {
    material_table_launch launch = {};
    launch.destination = static_cast<unsigned char *>(destination.opaque_handle()) + sizeof(T);
    launch.secondary_destination = operation == material_table_operation::grid_conductivity
                                       ? static_cast<unsigned char *>(secondary.opaque_handle()) +
                                             sizeof(T)
                                       : NULL;
    launch.compact_inputs = static_cast<const unsigned char *>(compact.opaque_handle());
    launch.compact_input_bytes = compact_bytes;
    launch.table_header_offset = 0;
    launch.elements = elements;
    launch.loop_count = elements;
    launch.table_kind = kind;
    launch.operation = operation;
    launch.source_material_id = 7;
    launch.destination_component = operation == material_table_operation::grid_conductivity
                                       ? 14
                                       : 4;
    launch.query_component = launch.destination_component;
    launch.tensor_row = launch.tensor_column = 2;
    launch.susceptibility_field_type = 8;
    launch.dimensions = 0;
    launch.axis_direction[0] = 0;
    launch.axis_direction[1] = 1;
    launch.axis_direction[2] = 2;
    launch.loop_begin[0] = launch.loop_begin[1] = 0;
    launch.loop_begin[2] = -4;
    launch.loop_end[0] = launch.loop_end[1] = 0;
    launch.loop_end[2] = 4;
    launch.little_corner[0] = launch.little_corner[1] = 0;
    launch.little_corner[2] = -4;
    launch.loop_extent[0] = launch.loop_extent[1] = 1;
    launch.loop_extent[2] = elements;
    launch.strides[0] = launch.strides[1] = 0;
    launch.strides[2] = 1;
    launch.cell_size[2] = 4;
    launch.inva = 1;
    launch.dt = 0.0625;
    launch.logical_single = logical_single;
    launch.precision = precision;
    return launch;
  };

  {
    const std::vector<double> samples = {2.0, 4.0};
    material_table_header header = {};
    header.version = 1;
    header.material_id = 7;
    header.kind = material_table_kind::file_scalar_epsilon;
    header.overlap_kind = 3;
    header.dimensions[0] = header.dimensions[1] = 1;
    header.dimensions[2] = 2;
    header.sample_offset = aligned_offset(sizeof(header), alignof(double));
    header.sample_count = samples.size();
    std::vector<unsigned char> compact(size_t(header.sample_offset) +
                                       samples.size() * sizeof(double), 0);
    std::memcpy(compact.data(), &header, sizeof(header));
    std::memcpy(compact.data() + header.sample_offset, samples.data(),
                samples.size() * sizeof(double));
    const size_t header_offset = 0;
    validate_material_table_headers(compact.data(), compact.size(), &header_offset, 1);
    std::vector<unsigned char> malformed = compact;
    reinterpret_cast<material_table_header *>(malformed.data())->dimensions[2] = 0;
    require_invalid(
        [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
        "zero material table dimension was accepted");
    malformed = compact;
    reinterpret_cast<material_table_header *>(malformed.data())->sample_count = 1;
    require_invalid(
        [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
        "short material table sample span was accepted");

    device_buffer device_compact(compact.size(), device);
    copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
    copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T), execution);
    material_table_launch launch = make_launch(
        device_compact, compact.size(), material_table_kind::file_scalar_epsilon,
        material_table_operation::file_chi1inv);
    validate_material_table_launch(launch, compact.data(), compact.size());
    material_table_launch malformed_launch = launch;
    ++malformed_launch.source_material_id;
    require_invalid(
        [&]() { validate_material_table_launch(malformed_launch, compact.data(), compact.size()); },
        "material table source/header ID mismatch was accepted");
    malformed_launch = launch;
    malformed_launch.table_kind = material_table_kind::material_grid;
    require_invalid(
        [&]() { validate_material_table_launch(malformed_launch, compact.data(), compact.size()); },
        "material table source/header kind mismatch was accepted");
    malformed_launch = launch;
    malformed_launch.operation = static_cast<material_table_operation>(99);
    require_invalid([&]() { launch_material_table(malformed_launch, execution); },
                    "unknown material table operation was accepted");
    malformed_launch = launch;
    malformed_launch.axis_direction[0] = 2;
    require_invalid(
        [&]() { validate_material_table_launch(malformed_launch, compact.data(), compact.size()); },
        "same-size material table axis mutation was accepted");
    malformed_launch = launch;
    malformed_launch.loop_begin[2] += 1000;
    malformed_launch.loop_end[2] += 1000;
    malformed_launch.little_corner[2] += 1000;
    require_invalid(
        [&]() { validate_material_table_launch(malformed_launch, compact.data(), compact.size()); },
        "far-out material table coordinates were accepted");
    copy_device_to_host_async(observed.data(), destination, 0,
                              observed.size() * sizeof(T), execution);
    execution.synchronize();
    require_exact(observed, guards,
                  "rejected material table descriptor changed guarded output");
    launch_material_table(launch, execution);
    copy_device_to_host_async(observed.data(), destination, 0,
                              observed.size() * sizeof(T), execution);
    execution.synchronize();
    require(observed.front() == sentinel && observed.back() == sentinel,
            "FILE table launch overwrote a guard");
    for (size_t i = 0; i < elements; ++i) {
      const double z = -2.0 + i;
      const double epsilon = scalar_interpolate(0.5 + z / 4.0, samples);
      require(observed[i + 1] == T(1.0 / epsilon), "FILE table interpolation differs");
    }
    copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T), execution);
    execution.synchronize();
    testing::fail_next(testing::failure_point::material_file_launch);
    bool rejected = false;
    try { launch_material_table(launch, execution); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "FILE table failure injection was ignored");

    material_table_header scalar_header = header;
    scalar_header.dimensions[2] = 1;
    scalar_header.sample_count = 1;
    std::vector<unsigned char> scalar_compact(
        size_t(scalar_header.sample_offset) + sizeof(double), 0);
    std::memcpy(scalar_compact.data(), &scalar_header, sizeof(scalar_header));
    device_buffer scalar_device_compact(scalar_compact.size(), device);
    material_table_launch scalar_launch = launch;
    scalar_launch.compact_inputs =
        static_cast<const unsigned char *>(scalar_device_compact.opaque_handle());
    scalar_launch.compact_input_bytes = scalar_compact.size();
    for (double epsilon : {0.0, -2.0}) {
      std::memcpy(scalar_compact.data() + scalar_header.sample_offset, &epsilon,
                  sizeof(epsilon));
      validate_material_table_launch(scalar_launch, scalar_compact.data(),
                                     scalar_compact.size());
      copy_host_to_device_async(scalar_device_compact, 0, scalar_compact.data(),
                                scalar_compact.size(), execution);
      copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T),
                                execution);
      launch_material_table(scalar_launch, execution);
      copy_device_to_host_async(observed.data(), destination, 0,
                                observed.size() * sizeof(T), execution);
      execution.synchronize();
      for (size_t i = 0; i < elements; ++i)
        if (epsilon == 0.0)
          require(std::isinf(double(observed[i + 1])) && observed[i + 1] > 0,
                  "zero FILE epsilon did not preserve positive infinity");
        else
          require(observed[i + 1] == T(-0.5),
                  "negative FILE epsilon inverse differs");
    }
  }

  {
    struct DimensionCase {
      int dimensions;
      size_t loop_extent[3];
      uint32_t table_extent[3];
      int axis_direction[3];
    };
    const DimensionCase cases[] = {
        {0, {1, 1, 4}, {2, 3, 4}, {0, 1, 2}},
        {1, {1, 2, 3}, {2, 3, 1}, {2, 0, 1}},
        {2, {2, 3, 4}, {2, 3, 4}, {0, 1, 2}},
        {3, {1, 2, 4}, {2, 3, 4}, {4, 3, 2}}};
    for (const DimensionCase &dimension_case : cases) {
      const size_t sample_count = size_t(dimension_case.table_extent[0]) *
                                  dimension_case.table_extent[1] *
                                  dimension_case.table_extent[2];
      std::vector<double> samples(sample_count);
      for (uint32_t x = 0; x < dimension_case.table_extent[0]; ++x)
        for (uint32_t y = 0; y < dimension_case.table_extent[1]; ++y)
          for (uint32_t z = 0; z < dimension_case.table_extent[2]; ++z)
            samples[(size_t(x) * dimension_case.table_extent[1] + y) *
                        dimension_case.table_extent[2] + z] =
                2.0 + 100.0 * x + 10.0 * y + z;
      material_table_header header = {};
      header.version = 1;
      header.material_id = 19;
      header.kind = material_table_kind::file_scalar_epsilon;
      header.overlap_kind = 3;
      for (int axis = 0; axis < 3; ++axis) header.dimensions[axis] =
          dimension_case.table_extent[axis];
      header.sample_offset = aligned_offset(sizeof(header), alignof(double));
      header.sample_count = samples.size();
      std::vector<unsigned char> compact(size_t(header.sample_offset) +
                                         samples.size() * sizeof(double), 0);
      std::memcpy(compact.data(), &header, sizeof(header));
      std::memcpy(compact.data() + header.sample_offset, samples.data(),
                  samples.size() * sizeof(double));
      const size_t point_count = dimension_case.loop_extent[0] *
                                 dimension_case.loop_extent[1] *
                                 dimension_case.loop_extent[2];
      std::vector<T> guarded(point_count + 2, sentinel), values(point_count + 2);
      device_buffer device_compact(compact.size(), device);
      device_buffer device_values(guarded.size() * sizeof(T), device);
      copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
      copy_host_to_device_async(device_values, 0, guarded.data(), guarded.size() * sizeof(T),
                                execution);
      material_table_launch launch = {};
      launch.destination = static_cast<unsigned char *>(device_values.opaque_handle()) + sizeof(T);
      launch.compact_inputs =
          static_cast<const unsigned char *>(device_compact.opaque_handle());
      launch.compact_input_bytes = compact.size();
      launch.table_header_offset = 0;
      launch.elements = point_count;
      launch.loop_count = point_count;
      launch.table_kind = material_table_kind::file_scalar_epsilon;
      launch.operation = material_table_operation::file_chi1inv;
      launch.source_material_id = 19;
      launch.destination_component = launch.query_component = 4;
      launch.tensor_row = launch.tensor_column = 2;
      launch.susceptibility_field_type = 8;
      launch.dimensions = dimension_case.dimensions;
      launch.inva = 1.0;
      launch.dt = 0.0625;
      launch.logical_single = logical_single;
      launch.precision = precision;
      launch.strides[2] = 1;
      launch.strides[1] = dimension_case.loop_extent[2];
      launch.strides[0] = dimension_case.loop_extent[1] * launch.strides[1];
      for (int axis = 0; axis < 3; ++axis) {
        launch.axis_direction[axis] = dimension_case.axis_direction[axis];
        launch.loop_extent[axis] = dimension_case.loop_extent[axis];
        launch.loop_begin[axis] = 1 - int(dimension_case.loop_extent[axis]);
        launch.loop_end[axis] = int(dimension_case.loop_extent[axis]) - 1;
        launch.little_corner[axis] = launch.loop_begin[axis];
        launch.cell_size[axis] = dimension_case.table_extent[axis];
      }
      validate_material_table_launch(launch, compact.data(), compact.size());
      launch_material_table(launch, execution);
      copy_device_to_host_async(values.data(), device_values, 0, values.size() * sizeof(T),
                                execution);
      execution.synchronize();
      require(values.front() == sentinel && values.back() == sentinel,
              "multidimensional FILE table overwrote a guard");
      for (size_t point = 0; point < point_count; ++point) {
        size_t remaining = point;
        const size_t loop_index[3] = {
            remaining / (dimension_case.loop_extent[1] * dimension_case.loop_extent[2]),
            (remaining / dimension_case.loop_extent[2]) % dimension_case.loop_extent[1],
            remaining % dimension_case.loop_extent[2]};
        double coordinate[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; ++axis) {
          const double physical = 0.5 *
              (launch.loop_begin[axis] + 2.0 * loop_index[axis]);
          const int direction = launch.axis_direction[axis];
          if (direction == 0 || direction == 3) coordinate[0] = physical;
          else if (direction == 1) coordinate[1] = physical;
          else if (direction == 2) coordinate[2] = physical;
        }
        if (launch.dimensions == 0) coordinate[0] = coordinate[1] = 0;
        else if (launch.dimensions == 1) coordinate[2] = 0;
        else if (launch.dimensions == 3) coordinate[1] = 0;
        double normalized[3];
        for (int axis = 0; axis < 3; ++axis)
          normalized[axis] = 0.5 + coordinate[axis] / launch.cell_size[axis];
        const double expected = 1.0 /
            table_interpolate_3d(normalized, dimension_case.table_extent, samples);
        require(values[point + 1] == T(expected),
                "multidimensional FILE row-major mapping differs");
      }
    }
  }

  for (uint32_t mode = 0; mode < 4; ++mode) {
    const double raw = mode == 1 ? -0.25 : mode == 2 ? 0.4 : 1.25;
    const std::vector<double> samples = mode == 2 ? std::vector<double>{0.15, 0.65}
                                                   : std::vector<double>{raw};
    material_table_header header = {};
    header.version = 1;
    header.material_id = 7;
    header.kind = material_table_kind::material_grid;
    header.overlap_kind = mode;
    header.dimensions[0] = mode == 2 ? 2 : 1;
    header.dimensions[1] = header.dimensions[2] = 1;
    size_t offset = aligned_offset(sizeof(header), alignof(material_medium_header));
    header.medium_1_offset = offset;
    offset += sizeof(material_medium_header);
    uint64_t susceptibility_offset = 0;
    uint64_t second_susceptibility_offset = 0;
    if (mode == 2) {
      offset = aligned_offset(offset, alignof(material_susceptibility_record));
      susceptibility_offset = offset;
      offset += sizeof(material_susceptibility_record);
    }
    offset = aligned_offset(offset, alignof(material_medium_header));
    header.medium_2_offset = offset;
    offset += sizeof(material_medium_header);
    if (mode == 2) {
      offset = aligned_offset(offset, alignof(material_susceptibility_record));
      second_susceptibility_offset = offset;
      offset += sizeof(material_susceptibility_record);
    }
    offset = aligned_offset(offset, alignof(double));
    header.sample_offset = offset;
    header.sample_count = samples.size();
    if (mode == 2) {
      header.beta = 2.0;
      header.eta = 0.4;
    }
    if (mode == 3) header.projection_offset = -0.25;
    std::vector<unsigned char> compact(offset + samples.size() * sizeof(double), 0);
    material_medium_header medium1 = {}, medium2 = {};
    medium1.version = medium2.version = 1;
    medium1.epsilon_diagonal[0] = medium1.epsilon_diagonal[1] =
        medium1.epsilon_diagonal[2] = 2;
    medium2.epsilon_diagonal[0] = medium2.epsilon_diagonal[1] =
        medium2.epsilon_diagonal[2] = 5;
    if (mode <= 1) {
      medium1.epsilon_diagonal[2] = -2;
      medium2.epsilon_diagonal[2] = 0;
    }
    medium1.conductivity[2] = 0.25;
    medium2.conductivity[2] = 0.75;
    material_susceptibility_record susceptibility = {};
    if (mode == 2) {
      medium1.electric_susceptibility_count = 1;
      medium1.electric_susceptibility_offset = susceptibility_offset;
      medium2.electric_susceptibility_count = 1;
      medium2.electric_susceptibility_offset = second_susceptibility_offset;
      susceptibility.version = 1;
      susceptibility.identity = 17;
      susceptibility.field_type = 0;
      susceptibility.material_ordinal = 0;
      susceptibility.sigma_diagonal[2] = 0.625;
      susceptibility.sigma_offdiagonal[0] = 0.375;
    }
    header.damping = -0.125;
    std::memcpy(compact.data(), &header, sizeof(header));
    std::memcpy(compact.data() + header.medium_1_offset, &medium1, sizeof(medium1));
    std::memcpy(compact.data() + header.medium_2_offset, &medium2, sizeof(medium2));
    if (mode == 2)
      std::memcpy(compact.data() + susceptibility_offset, &susceptibility,
                  sizeof(susceptibility));
    if (mode == 2) {
      material_susceptibility_record duplicate = susceptibility;
      duplicate.sigma_diagonal[2] = 9.0;
      duplicate.sigma_offdiagonal[0] = 8.0;
      std::memcpy(compact.data() + second_susceptibility_offset, &duplicate,
                  sizeof(duplicate));
    }
    std::memcpy(compact.data() + header.sample_offset, samples.data(),
                samples.size() * sizeof(double));
    const size_t header_offset = 0;
    validate_material_table_headers(compact.data(), compact.size(), &header_offset, 1);
    std::vector<unsigned char> malformed = compact;
    reinterpret_cast<material_table_header *>(malformed.data())->sample_offset = 0;
    require_invalid(
        [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
        "material table samples overlapping the header were accepted");
    malformed = compact;
    reinterpret_cast<material_table_header *>(malformed.data())->medium_2_offset =
        header.medium_1_offset;
    require_invalid(
        [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
        "aliased MaterialGrid medium headers were accepted");
    if (mode == 2) {
      malformed = compact;
      reinterpret_cast<material_medium_header *>(
          malformed.data() + header.medium_1_offset)->electric_susceptibility_offset = 0;
      require_invalid(
          [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
          "MaterialGrid susceptibility overlapping its table header was accepted");
      malformed = compact;
      reinterpret_cast<material_medium_header *>(
          malformed.data() + header.medium_1_offset)->electric_susceptibility_offset =
          header.medium_1_offset;
      require_invalid(
          [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
          "MaterialGrid susceptibility overlapping its medium header was accepted");
      malformed = compact;
      reinterpret_cast<material_medium_header *>(
          malformed.data() + header.medium_1_offset)->electric_susceptibility_offset =
          header.sample_offset;
      require_invalid(
          [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
          "MaterialGrid susceptibility overlapping samples was accepted");
    }
    malformed = compact;
    reinterpret_cast<material_table_header *>(malformed.data())->medium_2_offset =
        header.sample_offset;
    require_invalid(
        [&]() { validate_material_table_headers(malformed.data(), malformed.size(), &header_offset, 1); },
        "MaterialGrid medium overlapping samples was accepted");
    device_buffer device_compact(compact.size(), device);
    copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
    copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T), execution);
    copy_host_to_device_async(secondary, 0, guards.data(), guards.size() * sizeof(T), execution);
    material_table_launch launch = make_launch(
        device_compact, compact.size(), material_table_kind::material_grid,
        material_table_operation::grid_conductivity);
    if (mode == 2) {
      launch.dimensions = 2;
      launch.loop_begin[0] = launch.little_corner[0] = -4;
      launch.loop_end[0] = 4;
      launch.loop_extent[0] = elements;
      launch.strides[0] = 1;
      launch.cell_size[0] = 4;
      launch.loop_begin[2] = launch.loop_end[2] = launch.little_corner[2] = 0;
      launch.loop_extent[2] = 1;
      launch.strides[2] = 0;
    }
    const auto grid_weight = [&](size_t point, int x_shift) {
      double normalized[3] = {0.5, 0.5, 0.5};
      if (mode == 2) {
        const double x =
            0.5 * (launch.loop_begin[0] + 2.0 * point + x_shift) * launch.inva;
        normalized[0] = 0.5 + x / launch.cell_size[0];
      }
      double u = table_interpolate_3d(normalized, header.dimensions, samples);
      if (header.overlap_kind == 0) u = std::min(1.0, u);
      u += header.projection_offset;
      if (header.beta != 0)
        u = u == header.eta
                ? 0.5
                : (std::tanh(header.beta * header.eta) +
                   std::tanh(header.beta * (u - header.eta))) /
                      (std::tanh(header.beta * header.eta) +
                       std::tanh(header.beta * (1 - header.eta)));
      return u;
    };
    const auto table_equal = [](T observed_value, T expected_value) {
      const double scale = std::max(1.0, std::fabs(double(expected_value)));
      return observed_value == expected_value ||
             std::fabs(double(observed_value) - double(expected_value)) <=
                 8.0 * std::numeric_limits<T>::epsilon() * scale;
    };
    validate_material_table_launch(launch, compact.data(), compact.size());
    if (mode == 2) {
      material_table_launch valid_sigma = launch;
      valid_sigma.operation = material_table_operation::grid_sigma;
      valid_sigma.secondary_destination = NULL;
      valid_sigma.destination_component = valid_sigma.query_component = 4;
      valid_sigma.source_medium = 1;
      valid_sigma.source_susceptibility = 0;
      valid_sigma.susceptibility_identity = 17;
      valid_sigma.susceptibility_field_type = 0;
      validate_material_table_launch(valid_sigma, compact.data(), compact.size());
      launch_material_table(valid_sigma, execution);
      copy_device_to_host_async(observed.data(), destination, 0,
                                observed.size() * sizeof(T), execution);
      execution.synchronize();
      for (size_t i = 0; i < elements; ++i) {
        const T expected_sigma = T(0.625 * (1.0 - grid_weight(i, 0)));
        require(table_equal(observed[i + 1], expected_sigma),
                "MaterialGrid diagonal sigma launch differs");
      }
      const std::vector<T> diagonal_sigma = observed;
      copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T),
                                execution);
      valid_sigma.destination_component = valid_sigma.query_component = 0;
      valid_sigma.tensor_row = 0;
      valid_sigma.tensor_column = 1;
      valid_sigma.evaluation_shift[0] = -1;
      validate_material_table_launch(valid_sigma, compact.data(), compact.size());
      launch_material_table(valid_sigma, execution);
      copy_device_to_host_async(observed.data(), destination, 0,
                                observed.size() * sizeof(T), execution);
      execution.synchronize();
      bool shift_observable = false;
      for (size_t i = 0; i < elements; ++i) {
        const T shifted = T(0.375 * (1.0 - grid_weight(i, -1)));
        const T unshifted = T(0.375 * (1.0 - grid_weight(i, 0)));
        require(table_equal(observed[i + 1], shifted),
                "MaterialGrid offdiagonal Yee-shifted sigma launch differs");
        shift_observable = shift_observable || shifted != unshifted;
        require(table_equal(diagonal_sigma[i + 1],
                            T(0.625 * (1.0 - grid_weight(i, 0)))),
                "MaterialGrid diagonal sigma oracle changed after shifted launch");
      }
      require(shift_observable,
              "MaterialGrid offdiagonal Yee shift is numerically vacuous");
      material_table_launch wrong_identity = valid_sigma;
      ++wrong_identity.susceptibility_identity;
      require_invalid(
          [&]() { validate_material_table_launch(wrong_identity, compact.data(), compact.size()); },
          "MaterialGrid sigma identity substitution was accepted");
      material_table_launch wrong_source = valid_sigma;
      wrong_source.source_medium = 2;
      wrong_source.source_susceptibility = 0;
      require_invalid(
          [&]() { validate_material_table_launch(wrong_source, compact.data(), compact.size()); },
          "MaterialGrid non-first equivalent sigma source was accepted");
      copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T),
                                execution);
    }
    material_table_launch bad_sigma = launch;
    bad_sigma.operation = material_table_operation::grid_sigma;
    bad_sigma.secondary_destination = NULL;
    bad_sigma.destination_component = bad_sigma.query_component = 4;
    bad_sigma.source_medium = 1;
    bad_sigma.susceptibility_field_type = 0;
    bad_sigma.source_susceptibility = std::numeric_limits<uint64_t>::max();
    require_invalid(
        [&]() { validate_material_table_launch(bad_sigma, compact.data(), compact.size()); },
        "overflowing MaterialGrid susceptibility ordinal was accepted");
    launch_material_table(launch, execution);
    copy_device_to_host_async(observed.data(), destination, 0,
                              observed.size() * sizeof(T), execution);
    copy_device_to_host_async(inverse.data(), secondary, 0,
                              inverse.size() * sizeof(T), execution);
    execution.synchronize();
    for (size_t i = 0; i < elements; ++i) {
      const double u = grid_weight(i, 0);
      double expected = 0.25 + u * 0.5 + u * (1 - u) * header.damping;
      double expected_inverse;
      if (logical_single) {
        expected = float(expected);
        expected_inverse = float(1 / (1 + double(float(expected)) * launch.dt * 0.5));
      }
      else expected_inverse = 1 / (1 + expected * launch.dt * 0.5);
      require(table_equal(observed[i + 1], T(expected)),
              "MaterialGrid reducer/conductivity differs");
      require(table_equal(inverse[i + 1], T(expected_inverse)),
              "MaterialGrid condinv differs");
    }
    if (mode <= 1) {
      copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T),
                                execution);
      material_table_launch chi_launch = make_launch(
          device_compact, compact.size(), material_table_kind::material_grid,
          material_table_operation::grid_chi1inv);
      validate_material_table_launch(chi_launch, compact.data(), compact.size());
      launch_material_table(chi_launch, execution);
      copy_device_to_host_async(observed.data(), destination, 0,
                                observed.size() * sizeof(T), execution);
      execution.synchronize();
      const double epsilon = -2.0 + grid_weight(0, 0) * 2.0;
      for (size_t i = 0; i < elements; ++i)
        if (epsilon == 0.0)
          require(std::isinf(double(observed[i + 1])) && observed[i + 1] > 0,
                  "zero MaterialGrid epsilon did not preserve positive infinity");
        else
          require(observed[i + 1] == T(1.0 / epsilon),
                  "negative MaterialGrid epsilon inverse differs");
    }
    if (mode == 3) {
      copy_host_to_device_async(destination, 0, guards.data(), guards.size() * sizeof(T), execution);
      execution.synchronize();
      testing::fail_next(testing::failure_point::material_grid_launch);
      bool rejected = false;
      try { launch_material_table(launch, execution); }
      catch (const std::runtime_error &) { rejected = true; }
      require(rejected, "MaterialGrid table failure injection was ignored");
    }
  }
}

int main() {
  try {
    require(material_table_mirror_index_for_testing(INT_MAX, INT_MAX) == INT_MAX - 1,
            "wide material mirror index overflowed");
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA devices found");
    const size_t lengths[] = {0, 1, 255, 256, 257};
    const bool table_tail_only = std::getenv("MEEP_NVIDIA_TABLE_TAIL_ONLY") != NULL;
    const bool geometry_profile_only =
        std::getenv("MEEP_NVIDIA_GEOMETRY_PROFILE_ONLY") != NULL;
    for (size_t di = 0; di < devices.size(); ++di) {
      device_scope scope(devices[di].id);
      if (geometry_profile_only) {
        check_geometry_kernels<float>(devices[di].id, scalar_precision::f32);
        check_geometry_kernels<double>(devices[di].id, scalar_precision::f64);
        std::cout << "device " << devices[di].id
                  << ": NVIDIA geometry bulk/analytic/patch profile PASS\n";
        continue;
      }
      if (table_tail_only) {
        check_table_length<float>(devices[di].id, scalar_precision::f32, 257,
                                  material_table_kind::file_scalar_epsilon);
        check_table_length<float>(devices[di].id, scalar_precision::f32, 257,
                                  material_table_kind::material_grid);
        check_table_length<double>(devices[di].id, scalar_precision::f64, 257,
                                   material_table_kind::file_scalar_epsilon);
        check_table_length<double>(devices[di].id, scalar_precision::f64, 257,
                                   material_table_kind::material_grid);
        std::cout << "device " << devices[di].id
                  << ": NVIDIA material table 257-element tails PASS\n";
        continue;
      }
      for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
        check_length<float>(devices[di].id, scalar_precision::f32, lengths[i]);
        check_length<double>(devices[di].id, scalar_precision::f64, lengths[i]);
        check_table_length<float>(devices[di].id, scalar_precision::f32, lengths[i],
                                  material_table_kind::file_scalar_epsilon);
        check_table_length<float>(devices[di].id, scalar_precision::f32, lengths[i],
                                  material_table_kind::material_grid);
        check_table_length<double>(devices[di].id, scalar_precision::f64, lengths[i],
                                   material_table_kind::file_scalar_epsilon);
        check_table_length<double>(devices[di].id, scalar_precision::f64, lengths[i],
                                   material_table_kind::material_grid);
      }
      check_length<float>(devices[di].id, scalar_precision::f32, 17, true);
      check_length<double>(devices[di].id, scalar_precision::f64, 17, true);
      check_failures<float>(devices[di].id, scalar_precision::f32);
      check_failures<double>(devices[di].id, scalar_precision::f64);
      check_absorber<float>(devices[di].id, scalar_precision::f32);
      check_absorber<double>(devices[di].id, scalar_precision::f64);
      check_absorber<float>(devices[di].id, scalar_precision::f32, true);
      check_absorber<double>(devices[di].id, scalar_precision::f64, true);
      check_table_materials<float>(devices[di].id, scalar_precision::f32);
      check_table_materials<double>(devices[di].id, scalar_precision::f64);
      check_table_materials<float>(devices[di].id, scalar_precision::f32, true);
      check_table_materials<double>(devices[di].id, scalar_precision::f64, true);
      check_geometry_kernels<float>(devices[di].id, scalar_precision::f32);
      check_geometry_kernels<double>(devices[di].id, scalar_precision::f64);
      check_geometry_precedence<float>(devices[di].id, scalar_precision::f32);
      check_geometry_precedence<double>(devices[di].id, scalar_precision::f64);
      check_geometry_fixed_shapes<float>(devices[di].id, scalar_precision::f32);
      check_geometry_fixed_shapes<double>(devices[di].id, scalar_precision::f64);
      check_geometry_file_grid_values<float>(devices[di].id, scalar_precision::f32);
      check_geometry_file_grid_values<double>(devices[di].id, scalar_precision::f64);
      std::cout << "device " << devices[di].id << " (" << devices[di].name
                << "): NVIDIA material initialization kernels PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_initialization_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "nvidia_initialization_smoke: PASS\n";
  return 0;
}
