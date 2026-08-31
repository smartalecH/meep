/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include "material_data.hpp"

#include <algorithm>
#include <map>
#include <mutex>

#include "backend/material_callback.hpp"
#include "meep/mympi.hpp"

namespace meep_geom {

namespace {

std::mutex owned_callback_mutex;
std::map<const material_data *, std::shared_ptr<const meep::OwnedMaterialCallback> >
    owned_callbacks;

} // namespace

bool transition::operator==(const transition &other) const {
  return (from_level == other.from_level && to_level == other.to_level &&
          transition_rate == other.transition_rate && frequency == other.frequency &&
          vector3_equal(sigma_diag, other.sigma_diag) && gamma == other.gamma &&
          pumping_rate == other.pumping_rate);
}

bool transition::operator!=(const transition &other) const { return !(*this == other); }

medium_struct::medium_struct(double epsilon)
    : epsilon_diag{epsilon, epsilon, epsilon}, epsilon_offdiag{}, mu_diag{1, 1, 1}, mu_offdiag{},
      E_susceptibilities(), H_susceptibilities(), E_chi2_diag{}, E_chi3_diag{}, H_chi2_diag{},
      H_chi3_diag{}, D_conductivity_diag{}, B_conductivity_diag{} {}

void medium_struct::check_offdiag_im_zero_or_abort() const {
  if (epsilon_offdiag.x.im != 0 || epsilon_offdiag.y.im != 0 || epsilon_offdiag.z.im != 0 ||
      mu_offdiag.x.im != 0 || mu_offdiag.y.im != 0 || mu_offdiag.z.im != 0) {
    meep::abort("Found non-zero imaginary part of epsilon or mu offdiag.\n");
  }
}

material_data::material_data()
    : which_subclass(MEDIUM), medium(), user_func(NULL), user_data(NULL), do_averaging(false),
      epsilon_data(NULL), epsilon_dims{}, grid_size{}, weights(NULL), medium_1(), medium_2(),
      material_grid_kinds{U_DEFAULT} {}

void material_data::copy_from(const material_data &from) {
  which_subclass = from.which_subclass;
  medium = from.medium;

  user_func = from.user_func;
  // NOTE: the user_data field here opaque/void - so this is the best we can do.
  user_data = from.user_data;
  do_averaging = from.do_averaging;
  copy_owned_material_callback(&from, this);

  std::copy(std::begin(from.epsilon_dims), std::end(from.epsilon_dims), std::begin(epsilon_dims));
  if (from.epsilon_data) {
    size_t N = from.epsilon_dims[0] * from.epsilon_dims[1] * from.epsilon_dims[2];
    epsilon_data = new double[N];
    std::copy_n(from.epsilon_data, N, epsilon_data);
  }

  grid_size = from.grid_size;
  if (from.weights) {
    size_t N = from.grid_size.x * from.grid_size.y * from.grid_size.z;
    weights = new double[N];
    std::copy_n(from.weights, N, weights);
  }

  medium_1 = from.medium_1;
  medium_2 = from.medium_2;
  beta = from.beta;
  eta = from.eta;
  damping = from.damping;
  material_grid_kinds = from.material_grid_kinds;
}

bool owned_material_callback(
    const material_data *material,
    std::shared_ptr<const meep::OwnedMaterialCallback> *owner) {
  std::lock_guard<std::mutex> lock(owned_callback_mutex);
  const auto found = owned_callbacks.find(material);
  if (found == owned_callbacks.end()) return false;
  if (owner) *owner = found->second;
  return true;
}

void copy_owned_material_callback(const material_data *source, material_data *destination) {
  std::lock_guard<std::mutex> lock(owned_callback_mutex);
  owned_callbacks.erase(destination);
  const auto found = owned_callbacks.find(source);
  if (found != owned_callbacks.end()) owned_callbacks[destination] = found->second;
}

void release_owned_material_callback(const material_data *material) {
  std::lock_guard<std::mutex> lock(owned_callback_mutex);
  owned_callbacks.erase(material);
}

material_type make_owned_user_material_for_backend(
    const std::shared_ptr<const meep::OwnedMaterialCallback> &owner, bool do_averaging) {
  if (!owner || !owner->id || !owner->signature || !owner->capabilities || !owner->function)
    meep::abort(
        "owned user material callback requires stable identity, signature, and capabilities");
  material_data *material = new material_data();
  material->which_subclass = material_data::MATERIAL_USER;
  material->do_averaging = do_averaging;
  {
    std::lock_guard<std::mutex> lock(owned_callback_mutex);
    owned_callbacks[material] = owner;
  }
  return material;
}

void evaluate_material_user(material_data &material, vector3 point) {
  std::shared_ptr<const meep::OwnedMaterialCallback> owner;
  if (owned_material_callback(&material, &owner)) {
    owner->function(point, material.medium);
    return;
  }
  if (!material.user_func) meep::abort("user material has no callback");
  material.user_func(point, material.user_data, &material.medium);
}

material_type_list::material_type_list() : items(NULL), num_items(0) {}

} // namespace meep_geom
