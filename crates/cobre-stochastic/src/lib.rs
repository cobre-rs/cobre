//! # cobre-stochastic
//!
//! Stochastic process models for power systems: PAR(p) processes, correlated
//! sampling, scenario trees, deterministic forward sampling via `SipHash` seed
//! derivation, multiple noise generation methods (SAA, LHS, QMC-Sobol,
//! QMC-Halton), and a unified forward-pass sampling abstraction
//! (`ForwardSampler`) supporting in-sample and out-of-sample noise generation.
//!
//! Designed to be solver-agnostic for scenario-based iterative optimization.

// Allow unwrap/expect/panic in tests (explicit panics communicate test failures).
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

mod error;

pub mod context;
pub mod correlation;
pub mod noise;
pub mod normal;
pub mod par;
pub mod provenance;
pub mod sampling;
pub mod tree;

pub use context::{build_stochastic_context, ClassSchemes, OpeningTreeInputs, StochasticContext};
pub use correlation::{DecomposedCorrelation, GroupFactor, SpectralFactor};
pub use error::StochasticError;
pub use noise::quantile::norm_quantile;
pub use noise::rng::rng_from_seed;
pub use noise::seed::{derive_forward_seed, derive_opening_seed, derive_stage_seed};
pub use normal::precompute::{BlockFactorPair, EntityFactorEntry, PrecomputedNormal};
#[allow(deprecated)]
pub use par::{
    estimate_ar_coefficients, estimate_seasonal_stats, evaluate_par, evaluate_par_batch,
    evaluate_par_inflow, evaluate_par_inflows, solve_par_noise, solve_par_noise_batch,
    solve_par_noises, validate_par_parameters, ArCoefficientEstimate, ParValidationReport,
    ParWarning, PrecomputedPar, SeasonalStats,
};
pub use provenance::{ComponentProvenance, StochasticProvenance};
pub use sampling::insample::sample_forward;
pub use sampling::{
    build_forward_sampler, discover_historical_windows, pad_library_to_uniform,
    standardize_external_inflow, standardize_external_load, standardize_external_ncs,
    standardize_historical_windows, validate_external_library, validate_historical_library,
    ClassSampleRequest, ClassSampler, ExternalScenarioLibrary, ForwardNoise, ForwardSampler,
    ForwardSamplerConfig, HistoricalScenarioLibrary, SampleRequest,
};
pub use tree::{generate_opening_tree, ClassDimensions, OpeningTree, OpeningTreeView};

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::StochasticError;

    #[test]
    fn stochastic_error_is_send_sync_static() {
        fn assert_send_sync_static<E: std::error::Error + Send + Sync + 'static>() {}
        assert_send_sync_static::<StochasticError>();
    }

    #[test]
    fn all_public_modules_accessible() {
        use crate::context as _;
        use crate::correlation::resolve as _;
        use crate::correlation::spectral as _;
        use crate::noise::quantile as _;
        use crate::noise::rng as _;
        use crate::noise::seed as _;
        use crate::normal::precompute as _;
        use crate::par::evaluate as _;
        use crate::par::precompute as _;
        use crate::par::validation as _;
        use crate::sampling::insample as _;
        use crate::tree::generate as _;
        use crate::tree::opening_tree as _;
    }
}
