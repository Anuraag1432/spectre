// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"

#include <algorithm>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace VariableFixing {

template <size_t Dim>
RadiallyFallingFloor<Dim>::RadiallyFallingFloor(
    const double minimum_radius_at_which_to_apply_floor,
    const double rest_mass_density_scale, const double rest_mass_density_power,
    const double pressure_scale, const double pressure_power)
    : minimum_radius_at_which_to_apply_floor_(
          minimum_radius_at_which_to_apply_floor),
      rest_mass_density_scale_(rest_mass_density_scale),
      rest_mass_density_power_(rest_mass_density_power),
      pressure_scale_(pressure_scale),
      pressure_power_(pressure_power) {}

template <size_t Dim>
void RadiallyFallingFloor<Dim>::pup(PUP::er& p) {  // NOLINT
  p | minimum_radius_at_which_to_apply_floor_;
  p | rest_mass_density_scale_;
  p | rest_mass_density_power_;
  p | pressure_scale_;
  p | pressure_power_;
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void RadiallyFallingFloor<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> density,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    [[maybe_unused]] const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const {

  const size_t num_points = density->get().size();
  const size_t num_coords = coords.get(0).size();

  // Safety checks to ignore empty grids
  if (num_points == 0 || num_coords == 0) {
    return;
  }

  const auto dg_radii = magnitude(coords);

  // If sizes mismatch, we are on the Subcell grid. 
  // We compute the average radius of the element using the DG coordinates.
  const bool use_avg_radius = (num_points != num_coords);
  double avg_radius = 0.0;
  
  if (use_avg_radius) {
    for (size_t j = 0; j < num_coords; j++) {
      avg_radius += dg_radii.get()[j];
    }
    avg_radius /= num_coords;
  }

  for (size_t i = 0; i < num_points; i++) {
    // Apply exact radius for DG, or the element-averaged radius for Subcell
    const double radius = use_avg_radius ? avg_radius : dg_radii.get()[i];

    if (UNLIKELY(radius < minimum_radius_at_which_to_apply_floor_)) {
      continue;
    }

    if constexpr (ThermodynamicDim == 1) {
      density->get()[i] = std::max(
          density->get()[i],
          rest_mass_density_scale_ * pow(radius, rest_mass_density_power_));

      pressure->get()[i] = get(equation_of_state.pressure_from_density(
          Scalar<double>{density->get()[i]}));

      specific_internal_energy->get()[i] =
          get(equation_of_state.specific_internal_energy_from_density(
              Scalar<double>{density->get()[i]}));

      if (temperature->get().size() == num_points) {
        temperature->get()[i] = get(equation_of_state.temperature_from_density(
            Scalar<double>{density->get()[i]}));
      }
    } else if constexpr (ThermodynamicDim == 2) {
      density->get()[i] = std::max(
          density->get()[i],
          rest_mass_density_scale_ * pow(radius, rest_mass_density_power_));

      pressure->get()[i] = std::max(
          pressure->get()[i], pressure_scale_ * pow(radius, pressure_power_));

      specific_internal_energy->get()[i] = get(
          equation_of_state.specific_internal_energy_from_density_and_pressure(
              Scalar<double>{density->get()[i]},
              Scalar<double>{pressure->get()[i]}));

      if (temperature->get().size() == num_points) {
        temperature->get()[i] =
            get(equation_of_state.temperature_from_density_and_energy(
                Scalar<double>{density->get()[i]},
                Scalar<double>{specific_internal_energy->get()[i]}));
      }

    } else if constexpr (ThermodynamicDim == 3) {
      density->get()[i] = std::max(
          density->get()[i],
          rest_mass_density_scale_ * pow(radius, rest_mass_density_power_));

      temperature->get()[i] =
          std::max(density->get()[i] * temperature->get()[i],
                   pressure_scale_ * pow(radius, pressure_power_)) /
          density->get()[i];

      specific_internal_energy->get()[i] =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      Scalar<double>{density->get()[i]},
                      Scalar<double>{temperature->get()[i]},
                      Scalar<double>{electron_fraction->get()[i]}));
      pressure->get()[i] =
          get(equation_of_state.pressure_from_density_and_energy(
              Scalar<double>{density->get()[i]},
              Scalar<double>{specific_internal_energy->get()[i]},
              Scalar<double>{electron_fraction->get()[i]}));
    } else {
      ERROR("RadiallyFallingFloor: Must specify 1D, 2D, or 3D EoS");
    }
  }
}

template <size_t LocalDim>
bool operator==(const RadiallyFallingFloor<LocalDim>& lhs,
                const RadiallyFallingFloor<LocalDim>& rhs) {
  return lhs.minimum_radius_at_which_to_apply_floor_ ==
             rhs.minimum_radius_at_which_to_apply_floor_ and
         lhs.rest_mass_density_scale_ == rhs.rest_mass_density_scale_ and
         lhs.rest_mass_density_power_ == rhs.rest_mass_density_power_ and
         lhs.pressure_scale_ == rhs.pressure_scale_ and
         lhs.pressure_power_ == rhs.pressure_power_;
}

template <size_t Dim>
bool operator!=(const RadiallyFallingFloor<Dim>& lhs,
                const RadiallyFallingFloor<Dim>& rhs) {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template class RadiallyFallingFloor<GET_DIM(data)>;                       \
  template bool operator==(const RadiallyFallingFloor<GET_DIM(data)>& lhs,  \
                           const RadiallyFallingFloor<GET_DIM(data)>& rhs); \
  template bool operator!=(const RadiallyFallingFloor<GET_DIM(data)>& lhs,  \
                           const RadiallyFallingFloor<GET_DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                           \
  template void RadiallyFallingFloor<GET_DIM(data)>::operator()(         \
      gsl::not_null<Scalar<DataVector>*> density,                        \
      gsl::not_null<Scalar<DataVector>*> pressure,                       \
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,       \
      gsl::not_null<Scalar<DataVector>*> temperature,                    \
      gsl::not_null<Scalar<DataVector>*> electron_fraction,              \
      const tnsr::I<DataVector, GET_DIM(data), Frame::Inertial>& coords, \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&   \
          equation_of_state) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2, 3))
#undef INSTANTIATION
#undef THERMO_DIM
#undef GET_DIM

}  // namespace VariableFixing
