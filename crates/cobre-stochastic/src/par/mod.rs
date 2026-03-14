//! Periodic Autoregressive (PAR) model construction and validation.
//!
//! This module provides the building blocks for fitting PAR(p) models
//! to periodic time series such as seasonal records. The fitted
//! model parameters are later used to generate correlated synthetic
//! noise sequences during scenario construction.
//!
//! ## Submodules
//!
//! - [`precompute`] — derives LP-ready PAR coefficient matrices from
//!   historical series data
//! - [`validation`] — checks that PAR parameters satisfy the stationarity
//!   and invertibility conditions required for sound scenario generation
//! - [`evaluate`] — evaluates the PAR(p) model equation (forward) and
//!   solves for the noise that produces a target value (inverse)
//! - [`fitting`] — Levinson-Durbin recursion for solving Yule-Walker
//!   equations; provides AR coefficients and prediction error variances
//!   for each intermediate order

pub mod evaluate;
pub mod fitting;
pub mod precompute;
pub mod validation;

#[allow(deprecated)]
pub use evaluate::{
    evaluate_par, evaluate_par_batch, evaluate_par_inflow, evaluate_par_inflows, solve_par_noise,
    solve_par_noise_batch, solve_par_noises,
};
pub use fitting::{
    estimate_ar_coefficients, estimate_correlation, estimate_seasonal_stats, find_season_for_date,
    levinson_durbin, select_order_aic, AicSelectionResult, ArCoefficientEstimate,
    LevinsonDurbinResult, SeasonalStats,
};
pub use precompute::PrecomputedParLp;
pub use validation::{validate_par_parameters, ParValidationReport, ParWarning};
