// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/KerrAtmosphere.hpp"

#include <array>
#include <pup.h>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace RelativisticEuler::Solutions {

KerrAtmosphere::KerrAtmosphere(CkMigrateMessage* msg) : InitialData(msg) {}

KerrAtmosphere::KerrAtmosphere(const double bh_mass, const double bh_dimless_spin,
                           const double background_density, const double polytropic_constant,
                           const double polytropic_exponent)
    : bh_mass_(bh_mass),
    bh_spin_a_(bh_mass * bh_dimless_spin),
    background_density_(background_density),
    polytropic_constant_(polytropic_constant),
    polytropic_exponent_(polytropic_exponent),
    equation_of_state_{polytropic_constant_, polytropic_exponent_},
    background_spacetime_{bh_mass_, std::array<double, 3>{0.0, 0.0, bh_dimless_spin}, std::array<double, 3>{0.0, 0.0, 0.0}},
    background_pressure_(background_density_ * background_temperature_),
    background_specific_internal_energy_(background_temperature_ / (polytropic_exponent_ - 1.)) {}

std::unique_ptr<evolution::initial_data::InitialData> KerrAtmosphere::get_clone() const {
  return std::make_unique<KerrAtmosphere>(*this);
}

void KerrAtmosphere::pup(PUP::er& p) {
  InitialData::pup(p);
  p | bh_mass_;
  p | bh_spin_a_;
  p | background_density_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | background_spacetime_;
  p | background_temperature_;
  p | background_pressure_;
  p | background_specific_internal_energy_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, background_density_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.1)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 1.0 + background_specific_internal_energy_ + background_pressure_ / background_density_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, background_pressure_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, background_specific_internal_energy_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, background_temperature_)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)}; // Zero Eulerian Velocity!
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 1.0)}; // Lorentz factor for v=0 is 1.0
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>> KerrAtmosphere::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

PUP::able::PUP_ID KerrAtmosphere::my_PUP_ID = 0;

bool operator==(const KerrAtmosphere& lhs, const KerrAtmosphere& rhs) {
  return lhs.bh_mass_ == rhs.bh_mass_ and lhs.bh_spin_a_ == rhs.bh_spin_a_ and
         lhs.background_density_ == rhs.background_density_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const KerrAtmosphere& lhs, const KerrAtmosphere& rhs) { return not(lhs == rhs); }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                                    \
  template tuples::TaggedTuple<hydro::Tags::RestMassDensity<DTYPE(data)>>                       \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::RestMassDensity<DTYPE(data)>> /*meta*/) const; \
  template tuples::TaggedTuple<hydro::Tags::ElectronFraction<DTYPE(data)>>                      \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::ElectronFraction<DTYPE(data)>> /*meta*/) const;\
  template tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DTYPE(data)>>                      \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::SpecificEnthalpy<DTYPE(data)>> /*meta*/) const;\
  template tuples::TaggedTuple<hydro::Tags::Pressure<DTYPE(data)>>                              \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::Pressure<DTYPE(data)>> /*meta*/) const;        \
  template tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>>                \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/) const;\
  template tuples::TaggedTuple<hydro::Tags::Temperature<DTYPE(data)>>                           \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::Temperature<DTYPE(data)>> /*meta*/) const;     \
  template tuples::TaggedTuple<hydro::Tags::MagneticField<DTYPE(data), 3>>                      \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3>> /*meta*/) const;\
  template tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DTYPE(data)>>               \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::DivergenceCleaningField<DTYPE(data)>> /*meta*/) const;\
  template tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DTYPE(data), 3>>                    \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> /*meta*/) const;\
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>>                         \
  KerrAtmosphere::variables(const tnsr::I<DTYPE(data), 3>& x,                                   \
                            tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace RelativisticEuler::Solutions