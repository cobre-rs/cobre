//! # cobre-stochastic
//!
//! Stochastic process models for the [Cobre](https://github.com/cobre-rs/cobre)
//! power systems ecosystem.
//!
//! This crate provides the probabilistic building blocks used in scenario-based
//! stochastic optimization of power systems:
//!
//! - **PAR(p)**: Periodic Autoregressive models for inflow time series,
//!   following the methodology used in the Brazilian power sector.
//! - **Correlated sampling**: Cholesky-based spatial correlation applied to
//!   independent noise draws for multi-variate scenario generation.
//! - **Deterministic noise**: Communication-free noise generation via
//!   SipHash-1-3 seed derivation and `Pcg64` random number generation.
//! - **Scenario trees**: Opening tree construction and scenario sampling
//!   for iterative optimization algorithms.
//! - **Stochastic context**: [`StochasticContext`] bundles precomputed PAR
//!   parameters, correlated factors, and the opening tree into a single
//!   ready-to-use value for iterative optimization algorithms.
//! - **Forward sampling**: [`sample_forward`] draws scenario realisations for
//!   a given iteration using deterministic, communication-free seeds derived
//!   via [`derive_forward_seed`] and [`derive_opening_seed`].
//!
//! Designed to be solver-agnostic — the stochastic models can feed into
//! any scenario-based iterative optimization algorithm.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.

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

pub use context::{StochasticContext, build_stochastic_context};
pub use correlation::{CholeskyFactor, DecomposedCorrelation, GroupFactor};
pub use error::StochasticError;
pub use noise::rng::rng_from_seed;
pub use noise::seed::{derive_forward_seed, derive_opening_seed};
pub use normal::precompute::{BlockFactorPair, EntityFactorEntry, PrecomputedNormal};
#[allow(deprecated)]
pub use par::{
    ArCoefficientEstimate, LevinsonDurbinResult, ParValidationReport, ParWarning, PrecomputedPar,
    SeasonalStats, estimate_ar_coefficients, estimate_seasonal_stats, evaluate_par,
    evaluate_par_batch, evaluate_par_inflow, evaluate_par_inflows, levinson_durbin,
    solve_par_noise, solve_par_noise_batch, solve_par_noises, validate_par_parameters,
};
pub use provenance::{ComponentProvenance, StochasticProvenance};
pub use sampling::insample::sample_forward;
pub use tree::{OpeningTree, OpeningTreeView, generate_opening_tree};

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
        use crate::correlation::cholesky as _;
        use crate::correlation::resolve as _;
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
