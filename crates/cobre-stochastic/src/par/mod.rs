//! Periodic Autoregressive (PAR) model construction and validation.
//!
//! This module provides the building blocks for fitting PAR(p) models
//! to periodic time series such as seasonal inflow records. The fitted
//! model parameters are later used to generate correlated synthetic
//! noise sequences during scenario construction.
//!
//! ## Submodules
//!
//! - [`precompute`] — derives LP-ready PAR coefficient matrices from
//!   historical inflow data
//! - [`validation`] — checks that PAR parameters satisfy the stationarity
//!   and invertibility conditions required for sound scenario generation
//! - [`evaluate`] — evaluates the PAR(p) inflow equation in the forward
//!   direction given a lag state and noise realisation

pub mod evaluate;
pub mod precompute;
pub mod validation;

pub use evaluate::{
    compute_truncation_noise, compute_truncation_noises, evaluate_par_inflow, evaluate_par_inflows,
};
pub use precompute::PrecomputedParLp;
pub use validation::{validate_par_parameters, ParValidationReport, ParWarning};
