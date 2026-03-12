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
//! - [`evaluate`] — evaluates the PAR(p) inflow equation (forward) and
//!   solves for the noise that produces a target inflow (inverse)

pub mod evaluate;
pub mod precompute;
pub mod validation;

pub use evaluate::{evaluate_par_inflow, evaluate_par_inflows, solve_par_noise, solve_par_noises};
pub use precompute::PrecomputedParLp;
pub use validation::{ParValidationReport, ParWarning, validate_par_parameters};
