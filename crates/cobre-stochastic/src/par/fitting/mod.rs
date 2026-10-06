//! Periodic Yule-Walker estimation and seasonal statistics for PAR model fitting.
//!
//! For a periodic AR(p) process the Yule-Walker system is non-Toeplitz because
//! lags cross season boundaries; [`build_periodic_yw_matrix`] assembles the
//! per-season matrix and [`estimate_periodic_ar_coefficients`] solves it via LU
//! factorisation with partial pivoting.

mod annual;
mod ar_coefficients;
mod correlation;
mod cycle_positions;
mod estimation;
mod order_selection;
mod partitioned_covariance;
mod periodic_ar;
mod seasonal_stats;
mod yw_matrices;

pub use annual::{AnnualSeasonalStats, estimate_annual_seasonal_stats};
pub use ar_coefficients::{ArCoefficientEstimate, estimate_ar_coefficients_with_season_map};
pub use correlation::estimate_correlation_with_season_map;
pub use estimation::{
    ArEstimationConfig, ContributionReduction, ContributionValidationResult, EstimationReport,
    HydroEstimationEntry, ReductionReason, StdRatioDivergence, build_estimation_report,
    estimate_ar_coefficients_with_selection,
};
pub use order_selection::{
    AicSelectionResult, PacfSelectionResult, periodic_pacf, select_order_aic, select_order_pacf,
    select_order_pacf_annual,
};
pub use partitioned_covariance::conditional_facp_partitioned;
pub use periodic_ar::{
    PeriodicYwAnnualResult, PeriodicYwResult, estimate_periodic_ar_annual_coefficients,
    estimate_periodic_ar_coefficients,
};
pub use seasonal_stats::{
    HistoryClass, SeasonalStats, classify_history, estimate_seasonal_stats_with_season_map,
    find_season_for_date,
};
pub use yw_matrices::{
    build_extended_periodic_yw_matrix, build_periodic_yw_matrix, build_periodic_yw_matrix_into,
    cross_correlation_a_z_neg1, cross_correlation_z_a, periodic_autocorrelation,
    solve_linear_system,
};

// `#[cfg(test)]` re-imports reachable by the child `mod tests` via `super::`;
// they add no public path.
#[cfg(test)]
use cobre_core::scenario::CorrelationGroup;
#[cfg(test)]
use correlation::compute_pearson_correlation_matrix;
#[cfg(test)]
use partitioned_covariance::assemble_partitioned_covariance;
#[cfg(test)]
use yw_matrices::BUILD_PERIODIC_YW_MATRIX_CALL_COUNT;

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::doc_markdown
)]
mod tests;
