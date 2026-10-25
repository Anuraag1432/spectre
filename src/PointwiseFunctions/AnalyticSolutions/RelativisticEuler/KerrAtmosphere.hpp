// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Temperature.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace RelativisticEuler::Solutions {

class KerrAtmosphere
    : public virtual evolution::initial_data::InitialData,
    public MarkAsAnalyticSolution,
    // public ::AnalyticSolution<3>,
    public hydro::TemperatureInitialization<KerrAtmosphere> {
 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  struct BhMass {
    using type = double;
    static constexpr Options::String help = {"The mass of the black hole."};
    static type lower_bound() { return 0.0; }
  };
  struct BhDimlessSpin {
    using type = double;
    static constexpr Options::String help = {"The dimensionless black hole spin."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };
  struct BackgroundDensity {
    using type = double;
    static constexpr Options::String help = {"The constant density of the atmosphere."};
    static type lower_bound() { return 0.0; }
  };
  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {"The polytropic constant of the fluid."};
    static type lower_bound() { return 0.; }
  };
  struct PolytropicExponent {
    using type = double;
    static constexpr Options::String help = {"The polytropic exponent of the fluid."};
    static type lower_bound() { return 1.; }
  };

  using options =
      tmpl::list<BhMass, BhDimlessSpin, BackgroundDensity, PolytropicConstant, PolytropicExponent>;
  static constexpr Options::String help = {
      "Constant density fluid on a Kerr background."};

  KerrAtmosphere() = default;
  KerrAtmosphere(const KerrAtmosphere& /*rhs*/) = default;
  KerrAtmosphere& operator=(const KerrAtmosphere& /*rhs*/) = default;
  KerrAtmosphere(KerrAtmosphere&& /*rhs*/) = default;
  KerrAtmosphere& operator=(KerrAtmosphere&& /*rhs*/) = default;
  ~KerrAtmosphere() override = default;

  KerrAtmosphere(double bh_mass, double bh_dimless_spin, double background_density,
                 double polytropic_constant, double polytropic_exponent);

  auto get_clone() const -> std::unique_ptr<evolution::initial_data::InitialData> override;

  explicit KerrAtmosphere(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(KerrAtmosphere);

  template <typename DataType>
  using tags =
      tmpl::append<hydro::grmhd_tags<DataType>,
                   typename gr::Solutions::SphericalKerrSchild::tags<DataType>>;

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double /*t*/,
                                         tmpl::list<Tags...> /*meta*/) const {
    return {std::move(get<Tags>(variables(x, tmpl::list<Tags>{})))...};
  }

  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     const double /*t*/,
                                     tmpl::list<Tag> /*meta*/) const {
    return variables(x, tmpl::list<Tag>{});
  }

  void pup(PUP::er& p) override;

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

 protected:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType, typename Tag,
            Requires<not tmpl::list_contains_v<
                tmpl::push_back<hydro::grmhd_tags<DataType>,
                                hydro::Tags::SpecificEnthalpy<DataType>,
                                hydro::Tags::SpatialVelocity<DataType, 3>,
                                hydro::Tags::LorentzFactor<DataType>>,
                Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x, tmpl::list<Tag> /*meta*/) const {
    return {get<Tag>(background_spacetime_.variables(x, 0.0, tmpl::list<Tag>{}))};
  }

  friend bool operator==(const KerrAtmosphere& lhs, const KerrAtmosphere& rhs);

  double bh_mass_ = std::numeric_limits<double>::signaling_NaN();
  double bh_spin_a_ = std::numeric_limits<double>::signaling_NaN();
  double background_density_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::SphericalKerrSchild background_spacetime_{};

  double background_temperature_ = 2.e-5;
  double background_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double background_specific_internal_energy_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const KerrAtmosphere& lhs, const KerrAtmosphere& rhs);
}  // namespace RelativisticEuler::Solutions