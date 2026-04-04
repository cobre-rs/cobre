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
//! - [`fitting`] — periodic Yule-Walker matrix method for AR coefficient
//!   estimation and PACF-based order selection; provides periodic
//!   autocorrelation, matrix construction, and linear system solver
//! - [`contribution`] — recursive contribution composition for detecting
//!   explosive lag effects in periodic autoregressive models

pub mod contribution;
pub mod evaluate;
pub mod fitting;
pub mod precompute;
pub mod validation;

pub use contribution::{
    check_negative_contributions, compute_contributions, find_max_valid_order, has_negative_phi1,
};
#[allow(deprecated)]
pub use evaluate::{
    evaluate_par, evaluate_par_batch, evaluate_par_inflow, evaluate_par_inflows, solve_par_noise,
    solve_par_noise_batch, solve_par_noises,
};
pub use fitting::{
    build_periodic_yw_matrix, estimate_ar_coefficients, estimate_correlation,
    estimate_periodic_ar_coefficients, estimate_seasonal_stats, find_season_for_date,
    levinson_durbin, periodic_autocorrelation, periodic_pacf, select_order_aic, select_order_pacf,
    solve_linear_system, AicSelectionResult, ArCoefficientEstimate, LevinsonDurbinResult,
    PacfSelectionResult, PeriodicYwResult, SeasonalStats,
};
pub use precompute::PrecomputedPar;
pub use validation::{validate_par_parameters, ParValidationReport, ParWarning};
