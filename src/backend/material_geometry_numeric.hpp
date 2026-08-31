/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_MATERIAL_GEOMETRY_NUMERIC_HPP
#define MEEP_BACKEND_MATERIAL_GEOMETRY_NUMERIC_HPP

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define MEEP_GEOMETRY_HD __host__ __device__
#else
#define MEEP_GEOMETRY_HD
#endif

namespace meep {
namespace material_geometry_numeric {

struct vector {
  double x, y, z;
};

MEEP_GEOMETRY_HD inline vector add(vector a, vector b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
MEEP_GEOMETRY_HD inline vector subtract(vector a, vector b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
MEEP_GEOMETRY_HD inline vector scale(double s, vector a) {
  return {s * a.x, s * a.y, s * a.z};
}
MEEP_GEOMETRY_HD inline double dot(vector a, vector b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
MEEP_GEOMETRY_HD inline vector cross(vector a, vector b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}
MEEP_GEOMETRY_HD inline vector matrix_vector(const double *m, vector v) {
  return {m[0] * v.x + m[3] * v.y + m[6] * v.z,
          m[1] * v.x + m[4] * v.y + m[7] * v.z,
          m[2] * v.x + m[5] * v.y + m[8] * v.z};
}

MEEP_GEOMETRY_HD inline int mirror_index(int i, int n) {
  const int64_t wide = i, extent = n;
  return int(i >= n ? 2 * extent - 1 - wide : (i < 0 ? -1 - wide : wide));
}

MEEP_GEOMETRY_HD inline double interpolate(const double *samples, const uint32_t dimensions[3],
                                           double rx, double ry, double rz) {
  rx = rx < 0.0 ? -rx : (rx > 1.0 ? 1.0 - rx : rx);
  ry = ry < 0.0 ? -ry : (ry > 1.0 ? 1.0 - ry : ry);
  rz = rz < 0.0 ? -rz : (rz > 1.0 ? 1.0 - rz : rz);
  const int nx = int(dimensions[0]), ny = int(dimensions[1]), nz = int(dimensions[2]);
  const int x1 = mirror_index(int(rx * nx), nx), y1 = mirror_index(int(ry * ny), ny),
            z1 = mirror_index(int(rz * nz), nz);
  double dx = rx * nx - x1 - 0.5, dy = ry * ny - y1 - 0.5,
         dz = rz * nz - z1 - 0.5;
  const int x2 = mirror_index(dx >= 0.0 ? x1 + 1 : x1 - 1, nx);
  const int y2 = mirror_index(dy >= 0.0 ? y1 + 1 : y1 - 1, ny);
  const int z2 = mirror_index(dz >= 0.0 ? z1 + 1 : z1 - 1, nz);
  dx = fabs(dx); dy = fabs(dy); dz = fabs(dz);
#define MEEP_GEOMETRY_SAMPLE(x, y, z) \
  samples[((size_t(x) * size_t(ny) + size_t(y)) * size_t(nz) + size_t(z))]
  const double result =
      (((MEEP_GEOMETRY_SAMPLE(x1, y1, z1) * (1.0 - dx) +
         MEEP_GEOMETRY_SAMPLE(x2, y1, z1) * dx) * (1.0 - dy) +
        (MEEP_GEOMETRY_SAMPLE(x1, y2, z1) * (1.0 - dx) +
         MEEP_GEOMETRY_SAMPLE(x2, y2, z1) * dx) * dy) * (1.0 - dz) +
       ((MEEP_GEOMETRY_SAMPLE(x1, y1, z2) * (1.0 - dx) +
         MEEP_GEOMETRY_SAMPLE(x2, y1, z2) * dx) * (1.0 - dy) +
        (MEEP_GEOMETRY_SAMPLE(x1, y2, z2) * (1.0 - dx) +
         MEEP_GEOMETRY_SAMPLE(x2, y2, z2) * dx) * dy) * dz);
#undef MEEP_GEOMETRY_SAMPLE
  return result;
}

MEEP_GEOMETRY_HD inline vector object_coordinates(int kind, const double *parameters,
                                                   vector point) {
  const vector center = {parameters[0], parameters[1], parameters[2]};
  const vector relative = subtract(point, center);
  if (kind == 4) {
    const double radius = parameters[3];
    return add({0.5, 0.5, 0.5}, scale(0.5 / radius, relative));
  }
  if (kind == 3) {
    vector projected = matrix_vector(parameters + 15, relative);
    if (parameters[12] != 0.0) projected.x /= parameters[12];
    if (parameters[13] != 0.0) projected.y /= parameters[13];
    if (parameters[14] != 0.0) projected.z /= parameters[14];
    return add({0.5, 0.5, 0.5}, projected);
  }
  return {0.0, 0.0, 0.0};
}

MEEP_GEOMETRY_HD inline double norm(vector a) { return sqrt(dot(a, a)); }

enum segment_intersection { non_intersecting = 0, intersecting = 1, in_segment = 2, on_ray = 3 };

MEEP_GEOMETRY_HD inline int intersect_line_with_segment(vector q0, vector q1, vector q2,
                                                         vector u, double *s) {
  const double m00 = u.x, m01 = q1.x - q2.x, m10 = u.y, m11 = q1.y - q2.y;
  const double rhsx = q1.x - q0.x, rhsy = q1.y - q0.y;
  const double determinant = m00 * m11 - m01 * m10;
  const double length2 = m01 * m01 + m11 * m11;
  if (fabs(determinant) < 1e-10 * length2) {
    const vector d1 = subtract(q0, q1), d2 = subtract(q0, q2);
    if (norm(d1) <= 1e-12 * norm(q0) || norm(d2) <= 1e-12 * norm(q0)) return in_segment;
    const double n1 = norm(d1), n2 = norm(d2), product = dot(d1, d2);
    if (fabs(product) < (1.0 - 1e-5) * n1 * n2) return non_intersecting;
    if (product < 0.0) {
      if (s) *s = 0.0;
      return in_segment;
    }
    if (u.x * d1.x + u.y * d1.y < 0.0) {
      if (s) *s = fmin(n1, n2) / sqrt(u.x * u.x + u.y * u.y);
      return on_ray;
    }
    return non_intersecting;
  }
  const float t = float((m00 * rhsy - m10 * rhsx) / determinant);
  if (s) *s = (m11 * rhsx - m01 * rhsy) / determinant;
  return t < -1e-5f || t >= 1.0f - 1e-5f ? non_intersecting : intersecting;
}

MEEP_GEOMETRY_HD inline int intersect_ray_with_segment(vector q0, vector q1, vector q2,
                                                        vector u) {
  double s = 0.0;
  const int status = intersect_line_with_segment(q0, q1, q2, u, &s);
  return status == intersecting && s < 0.0 ? non_intersecting : status;
}

MEEP_GEOMETRY_HD inline vector prism_node(const double *bottom, const double *delta, size_t i,
                                           double z) {
  return {bottom[3 * i] + z * delta[3 * i], bottom[3 * i + 1] + z * delta[3 * i + 1],
          bottom[3 * i + 2] + z * delta[3 * i + 2]};
}

MEEP_GEOMETRY_HD inline bool point_in_polygon(vector point, const double *bottom,
                                               const double *delta, size_t vertex_count,
                                               double z, bool include_boundaries) {
  const vector x_axis = {1.0, 0.0, 0.0};
  int start_position = -1;
  vector start = {0.0, 0.0, 0.0};
  for (size_t i = 0; i < vertex_count; ++i) {
    const vector a = prism_node(bottom, delta, i, z);
    const vector b = prism_node(bottom, delta, (i + 1) % vertex_count, z);
    const vector edge = subtract(b, a);
    const double edge_norm = norm(edge);
    const vector unit = edge_norm == 0.0 ? vector{0.0, 0.0, 0.0} : scale(1.0 / edge_norm, edge);
    if (intersect_ray_with_segment(point, a, b, unit) == in_segment)
      return include_boundaries;
    if (fabs(a.y - point.y) > 1e-5) start_position = int(i), start = a;
  }
  if (start_position < 0) return false;
  int checked = 0, current = start_position, crossed = 0;
  while (checked < int(vertex_count)) {
    const int saved_index = (current + 1) % int(vertex_count);
    const double saved_x = prism_node(bottom, delta, size_t(saved_index), z).x;
    do {
      current = (current + 1) % int(vertex_count);
      ++checked;
    } while (checked < int(vertex_count) &&
             fabs(prism_node(bottom, delta, size_t(current), z).y - point.y) < 1e-5);
    const vector end = prism_node(bottom, delta, size_t(current), z);
    if ((start.y - point.y) * (end.y - point.y) < 0.0) {
      if (saved_index == current) {
        if (intersect_ray_with_segment(point, start, end, x_axis) == intersecting) ++crossed;
      }
      else if (saved_x > point.x + 1e-5 &&
               intersect_line_with_segment(point, start, end, x_axis, NULL) == intersecting)
        ++crossed;
    }
    start = end;
  }
  return crossed % 2 != 0;
}

MEEP_GEOMETRY_HD inline bool ray_triangle(vector origin, vector direction,
                                           vector v0, vector v1, vector v2,
                                           double determinant_epsilon, double &t) {
  const vector edge1 = subtract(v1, v0), edge2 = subtract(v2, v0);
  const vector p = cross(direction, edge2);
  const double determinant = dot(edge1, p);
  if (fabs(determinant) < determinant_epsilon) return false;
  const double inverse = 1.0 / determinant;
  const vector s = subtract(origin, v0);
  const double u = inverse * dot(s, p);
  if (u < -1e-10 || u > 1.0 + 1e-10) return false;
  const vector q = cross(s, edge1);
  const double v = inverse * dot(direction, q);
  if (v < -1e-10 || u + v > 1.0 + 1e-10) return false;
  t = inverse * dot(edge2, q);
  return true;
}

MEEP_GEOMETRY_HD inline int mesh_crossings(vector point, vector direction,
                                            const double *vertices, size_t vertex_count,
                                            const double *indices, size_t index_count,
                                            double lengthscale, bool &degenerate) {
  (void)vertex_count;
  const double determinant_epsilon = 1e-12 * lengthscale * lengthscale;
  const double forward_epsilon = 1e-12 * lengthscale;
  const double duplicate_epsilon = 1e-10 * lengthscale;
  int forward_count = 0;
  for (size_t triangle = 0; triangle < index_count / 3; ++triangle) {
    const size_t i0 = size_t(indices[3 * triangle]);
    const size_t i1 = size_t(indices[3 * triangle + 1]);
    const size_t i2 = size_t(indices[3 * triangle + 2]);
    const vector v0 = {vertices[3 * i0], vertices[3 * i0 + 1], vertices[3 * i0 + 2]};
    const vector v1 = {vertices[3 * i1], vertices[3 * i1 + 1], vertices[3 * i1 + 2]};
    const vector v2 = {vertices[3 * i2], vertices[3 * i2 + 1], vertices[3 * i2 + 2]};
    double t = 0.0;
    if (!ray_triangle(point, direction, v0, v1, v2, determinant_epsilon, t) ||
        !(t > forward_epsilon))
      continue;
    ++forward_count;
  }
  int kept = 0;
  bool have_scanned = false, have_kept = false;
  double scanned = 0.0, last_kept = 0.0;
  for (size_t iteration = 0; iteration < index_count / 3; ++iteration) {
    bool found = false;
    double next = 0.0;
    for (size_t triangle = 0; triangle < index_count / 3; ++triangle) {
      const size_t i0 = size_t(indices[3 * triangle]);
      const size_t i1 = size_t(indices[3 * triangle + 1]);
      const size_t i2 = size_t(indices[3 * triangle + 2]);
      const vector v0 = {vertices[3 * i0], vertices[3 * i0 + 1], vertices[3 * i0 + 2]};
      const vector v1 = {vertices[3 * i1], vertices[3 * i1 + 1], vertices[3 * i1 + 2]};
      const vector v2 = {vertices[3 * i2], vertices[3 * i2 + 1], vertices[3 * i2 + 2]};
      double t = 0.0;
      if (!ray_triangle(point, direction, v0, v1, v2, determinant_epsilon, t) ||
          !(t > forward_epsilon) || (have_scanned && !(t > scanned)))
        continue;
      if (!found || t < next) next = t, found = true;
    }
    if (!found) break;
    if (!have_kept || next - last_kept > duplicate_epsilon)
      last_kept = next, have_kept = true, ++kept;
    scanned = next;
    have_scanned = true;
  }
  degenerate = kept != forward_count;
  return kept;
}

/* Shape tags follow libctl's generated enum: self, mesh, prism, block,
   sphere, cylinder, compound. Fixed records never contain compound nodes. */
MEEP_GEOMETRY_HD inline bool contains(int kind, int subtype, const double *parameters,
                                      const double *vertices, size_t vertex_count,
                                      const double *indices, size_t index_count,
                                      const double *auxiliary, double mesh_lengthscale,
                                      const double metric[9], vector point,
                                      bool prism_include_boundaries) {
  const vector center = {parameters[0], parameters[1], parameters[2]};
  const vector r = subtract(point, center);
  if (kind == 4) { // sphere
    const vector rm = matrix_vector(metric, r);
    const double radius = parameters[3];
    return radius > 0.0 && dot(r, rm) <= radius * radius;
  }
  if (kind == 5) { // cylinder/cone/wedge
    const vector axis = {parameters[3], parameters[4], parameters[5]};
    const vector rm = matrix_vector(metric, r);
    const double projection = dot(axis, rm), height = parameters[7];
    if (fabs(projection) > 0.5 * height) return false;
    double radius = parameters[6];
    if (subtype == 2)
      radius += (projection / height + 0.5) * (parameters[9] - radius);
    else if (subtype == 1) {
      const vector e1 = {parameters[13], parameters[14], parameters[15]};
      const vector e2 = {parameters[16], parameters[17], parameters[18]};
      double angle = atan2(dot(rm, e2), dot(rm, e1));
      const double wedge = parameters[9];
      if (wedge > 0.0) {
        if (angle < 0.0) angle += 2.0 * 3.14159265358979323846;
        if (angle > wedge) return false;
      }
      else {
        if (angle > 0.0) angle -= 2.0 * 3.14159265358979323846;
        if (angle < wedge) return false;
      }
    }
    return radius != 0.0 && dot(r, rm) - projection * projection <= radius * radius;
  }
  if (kind == 3) { // block/ellipsoid
    const vector projected = matrix_vector(parameters + 15, r);
    if (subtype == 0)
      return fabs(projected.x) <= 0.5 * parameters[12] &&
             fabs(projected.y) <= 0.5 * parameters[13] &&
             fabs(projected.z) <= 0.5 * parameters[14];
    const double x = projected.x * parameters[25];
    const double y = projected.y * parameters[26];
    const double z = projected.z * parameters[27];
    return x * x + y * y + z * z <= 1.0;
  }
  if (kind == 2) { // prism
    const vector centroid = {parameters[8], parameters[9], parameters[10]};
    const vector projected = matrix_vector(parameters + 11, subtract(point, centroid));
    if (projected.z < 0.0 || projected.z > parameters[3] || vertex_count < 3) return false;
    /* auxiliary contains fixed bottom vertices, scaled top deltas, then top vertices. */
    const double *bottom = auxiliary;
    const double *delta = auxiliary + 3 * vertex_count;
    return point_in_polygon(projected, bottom, delta, vertex_count, projected.z,
                            prism_include_boundaries);
  }
  if (kind == 1) { // mesh
    if (subtype == 0 || vertex_count < 4 || index_count < 12) return false;
    const vector directions[3] = {
        {0.57735026918962576, 0.57735026918962576, 0.57735026918962576},
        {0.80178372573727319, 0.53452248382484879, 0.26726124191242440},
        {0.12309149097933272, 0.49236596391733088, 0.86164043685532904}};
    bool degenerate = false;
    int crossings = mesh_crossings(point, directions[0], vertices, vertex_count, indices,
                                   index_count, mesh_lengthscale, degenerate);
    if (!degenerate) return crossings % 2 == 1;
    int votes = crossings % 2;
    for (int i = 1; i < 3; ++i) {
      bool ignored = false;
      crossings = mesh_crossings(point, directions[i], vertices, vertex_count, indices,
                                 index_count, mesh_lengthscale, ignored);
      votes += crossings % 2;
    }
    return votes >= 2;
  }
  return false;
}

} // namespace material_geometry_numeric
} // namespace meep

#undef MEEP_GEOMETRY_HD

#endif
