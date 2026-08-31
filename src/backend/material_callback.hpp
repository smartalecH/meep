/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

/* Backend-private ownership seam for the deliberately narrow PR5.4 callback
   route.  Nothing in this file is installed or exposed through SWIG. */
#ifndef MEEP_BACKEND_MATERIAL_CALLBACK_HPP
#define MEEP_BACKEND_MATERIAL_CALLBACK_HPP

#include <functional>
#include <memory>
#include <stdint.h>

#include "material_data.hpp"

namespace meep {

enum OwnedMaterialCallbackCapability : uint64_t {
  owned_material_callback_pure_replay_stable = UINT64_C(1) << 0,
  owned_material_callback_pointwise_no_averaging = UINT64_C(1) << 1,
  owned_material_callback_output_chi1 = UINT64_C(1) << 2
};

static const uint64_t owned_material_callback_tiled_capabilities =
    owned_material_callback_pure_replay_stable |
    owned_material_callback_pointwise_no_averaging |
    owned_material_callback_output_chi1;

struct OwnedMaterialCallback {
  uint64_t id;
  uint64_t signature;
  uint64_t capabilities;
  std::function<void(vector3, meep_geom::medium_struct &)> function;

  OwnedMaterialCallback(
      uint64_t id_, uint64_t signature_, uint64_t capabilities_,
      const std::function<void(vector3, meep_geom::medium_struct &)> &function_)
      : id(id_), signature(signature_), capabilities(capabilities_), function(function_) {}
};

} // namespace meep

namespace meep_geom {

material_type make_owned_user_material_for_backend(
    const std::shared_ptr<const meep::OwnedMaterialCallback> &owner,
    bool do_averaging = false);
bool owned_material_callback(
    const material_data *material,
    std::shared_ptr<const meep::OwnedMaterialCallback> *owner = NULL);
void copy_owned_material_callback(const material_data *source, material_data *destination);
void release_owned_material_callback(const material_data *material);
void evaluate_material_user(material_data &material, vector3 point);

} // namespace meep_geom

#endif
