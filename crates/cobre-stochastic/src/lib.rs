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

pub mod correlation;
pub mod noise;
pub mod par;
pub mod sampling;
pub mod tree;

pub use error::StochasticError;
pub use par::{validate_par_parameters, ParValidationReport, ParWarning, PrecomputedParLp};

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::StochasticError;

    /// Compile-time assertion: `StochasticError` must be `Send + Sync + 'static`.
    #[test]
    fn stochastic_error_is_send_sync_static() {
        fn assert_send_sync_static<E: std::error::Error + Send + Sync + 'static>() {}
        assert_send_sync_static::<StochasticError>();
    }

    #[test]
    fn par_module_is_accessible() {
        use crate::par::precompute as _;
        use crate::par::validation as _;
    }

    #[test]
    fn correlation_module_is_accessible() {
        use crate::correlation::cholesky as _;
        use crate::correlation::resolve as _;
    }

    #[test]
    fn noise_module_is_accessible() {
        use crate::noise::rng as _;
        use crate::noise::seed as _;
    }

    #[test]
    fn tree_module_is_accessible() {
        use crate::tree::generate as _;
        use crate::tree::opening_tree as _;
    }

    #[test]
    fn sampling_module_is_accessible() {
        use crate::sampling::insample as _;
    }
}
