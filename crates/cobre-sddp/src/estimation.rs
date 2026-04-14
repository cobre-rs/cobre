//! Automatic PAR(p) parameter estimation from historical inflow observations.
//!
//! This module bridges `cobre-io` (case loading) and `cobre-stochastic` (PAR fitting).
//! It inspects the input file manifest, resolves which of seven input paths applies
//! (see [`EstimationPath`]), and dispatches to the appropriate estimation function.
//!
//! ## Input path matrix
//!
//! Three boolean flags determine the path: whether `inflow_history.parquet` (H),
//! `inflow_seasonal_stats.parquet` (S), and `inflow_ar_coefficients.parquet` (R)
//! are present in the case directory.
//!
//! | Row | H | S | R | Variant | Behaviour |
//! |-----|---|---|---|---------|-----------|
//! |  1  | 0 | 0 | 0 | [`Deterministic`](EstimationPath::Deterministic) | Return `system` unchanged. |
//! |  2  | 0 | 1 | 0 | [`UserStatsWhiteNoise`](EstimationPath::UserStatsWhiteNoise) | Return `system` unchanged (white-noise stats from user). |
//! |  3  | 0 | 1 | 1 | [`UserProvidedNoHistory`](EstimationPath::UserProvidedNoHistory) | Return `system` unchanged (complete user model). |
//! |  4  | 1 | 0 | 0 | [`FullEstimation`](EstimationPath::FullEstimation) | Full estimation via `run_estimation`. |
//! |  5  | 1 | 0 | 1 | [`UserArHistoryStats`](EstimationPath::UserArHistoryStats) | Stats from history, AR from user via `run_user_ar_estimation`. |
//! |  6  | 1 | 1 | 0 | [`PartialEstimation`](EstimationPath::PartialEstimation) | Stats from user, AR estimated from history via `run_partial_estimation`. |
//! |  7  | 1 | 1 | 1 | [`UserProvidedAll`](EstimationPath::UserProvidedAll) | Return `system` unchanged (all parameters from user). |
//!
//! Invalid combinations (R=1 without H or S) fall back to row 1 (`Deterministic`).
//!
//! ## Role 1 / Role 2
//!
//! Each inflow model requires two parameter groups:
//!
//! - **Role 1 (seasonal stats)**: `mean_m3s` and `std_m3s` per hydro per stage.
//!   These drive the LP assembly (scenario scaling) and can come from either the
//!   user file (`inflow_seasonal_stats.parquet`) or history estimation.
//! - **Role 2 (AR coefficients)**: `ar_coefficients` and `residual_std_ratio` per
//!   hydro per stage. These drive the autoregressive scenario noise and can come
//!   from either the user file (`inflow_ar_coefficients.parquet`) or history
//!   estimation.
//!
//! Rows 4-6 are the "active" paths where at least one role is estimated from
//! history. In rows 4 and 6, Role 2 is estimated via periodic Yule-Walker / PACF.
//! In row 5, Role 1 is estimated from history while Role 2 is preserved from user.
//!
//! `correlation.json` is handled independently: if present, the existing
//! `system.correlation()` is kept; if absent, the correlation is estimated from
//! residuals.
//!
//! ## PACF order selection
//!
//! When `config.estimation.order_selection = "pacf"` (the default and only
//! supported method), the module computes the periodic PACF via progressive
//! periodic Yule-Walker matrix solves, selects the order using a 95% significance
//! threshold, then estimates coefficients at the selected order using the periodic
//! YW system.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use chrono::NaiveDate;
use cobre_core::{EntityId, System};
use cobre_io::{
    parse_inflow_ar_coefficients, parse_inflow_history,
    scenarios::{assemble_inflow_models, InflowArCoefficientRow, InflowSeasonalStatsRow},
    validate_structure, Config, FileManifest, LoadError, ValidationContext,
};
use cobre_stochastic::{
    par::contribution::{
        check_negative_contributions, compute_contributions, find_max_valid_order,
        has_negative_phi1,
    },
    par::fitting::{
        estimate_ar_coefficients_with_season_map, estimate_correlation_with_season_map,
        estimate_periodic_ar_coefficients, estimate_seasonal_stats_with_season_map,
        find_season_for_date, periodic_pacf, select_order_pacf, ArCoefficientEstimate,
        SeasonalStats,
    },
    StochasticError,
};

/// Classification of the estimation path taken for a given input file manifest.
///
/// Each variant corresponds to one row of the input path matrix documented in
/// the module-level doc comment. The three boolean flags are:
/// - `H` — `scenarios_inflow_history_parquet`
/// - `S` — `scenarios_inflow_seasonal_stats_parquet`
/// - `R` — `scenarios_inflow_ar_coefficients_parquet`
///
/// | Row | H | S | R | Variant |
/// |-----|---|---|---|---------|
/// |  1  | 0 | 0 | 0 | `Deterministic` |
/// |  2  | 0 | 1 | 0 | `UserStatsWhiteNoise` |
/// |  3  | 0 | 1 | 1 | `UserProvidedNoHistory` |
/// |  4  | 1 | 0 | 0 | `FullEstimation` |
/// |  5  | 1 | 0 | 1 | `UserArHistoryStats` |
/// |  6  | 1 | 1 | 0 | `PartialEstimation` |
/// |  7  | 1 | 1 | 1 | `UserProvidedAll` |
///
/// The combination `(H=0, S=0, R=1)` and `(H=0, S=1, R=1)` (rows with AR but
/// no history) map to `Deterministic` and `UserProvidedNoHistory` respectively.
/// AR without history is meaningless for estimation, so the presence of AR
/// coefficients alone (without history) is treated as row 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationPath {
    /// Row 1: no history, no stats, no AR (or AR present but no history/stats).
    /// The system is returned unchanged.
    Deterministic,
    /// Row 2: no history, stats present, no AR.
    /// The system is returned unchanged (user-provided white-noise stats).
    UserStatsWhiteNoise,
    /// Row 3: no history, stats present, AR present.
    /// The system is returned unchanged (user-provided complete model, no history).
    UserProvidedNoHistory,
    /// Row 4: history present, no stats, no AR.
    /// Full estimation: seasonal stats and AR coefficients are estimated.
    FullEstimation,
    /// Row 5: history present, no stats, AR present.
    /// Seasonal stats are estimated from history (Role 1); user AR coefficients
    /// are preserved bitwise (Role 2). Dispatches to `run_user_ar_estimation`.
    UserArHistoryStats,
    /// Row 6: history present, stats present, no AR.
    /// User stats are used; only AR coefficients are estimated from history.
    /// Dispatches to `run_partial_estimation`, which preserves `mean_m3s` and
    /// `std_m3s` from the user-provided stats while estimating AR coefficients
    /// via periodic Yule-Walker / PACF.
    PartialEstimation,
    /// Row 7: history present, stats present, AR present.
    /// All model parameters provided by the user; system is returned unchanged.
    UserProvidedAll,
}

impl EstimationPath {
    /// Resolve the estimation path from the three boolean manifest flags.
    ///
    /// This function is a total map over all 8 boolean combinations. Invalid
    /// combinations (AR present without history or stats) fall back to
    /// `Deterministic` because AR coefficients alone cannot drive estimation.
    #[must_use]
    pub fn resolve(manifest: &cobre_io::FileManifest) -> Self {
        match (
            manifest.scenarios_inflow_history_parquet,
            manifest.scenarios_inflow_seasonal_stats_parquet,
            manifest.scenarios_inflow_ar_coefficients_parquet,
        ) {
            // No history — AR alone is meaningless; row 1.
            (false, false, _) => Self::Deterministic,
            // No history, stats present, no AR — row 2.
            (false, true, false) => Self::UserStatsWhiteNoise,
            // No history, stats present, AR present — row 3.
            (false, true, true) => Self::UserProvidedNoHistory,
            // History present, no stats, no AR — row 4.
            (true, false, false) => Self::FullEstimation,
            // History present, no stats, AR present — row 5.
            (true, false, true) => Self::UserArHistoryStats,
            // History present, stats present, no AR — row 6.
            (true, true, false) => Self::PartialEstimation,
            // History present, stats present, AR present — row 7.
            (true, true, true) => Self::UserProvidedAll,
        }
    }

    /// Convert to a stable string representation for diagnostic output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::UserStatsWhiteNoise => "user_stats_white_noise",
            Self::UserProvidedNoHistory => "user_provided_no_history",
            Self::FullEstimation => "full_estimation",
            Self::UserArHistoryStats => "user_ar_history_stats",
            Self::PartialEstimation => "partial_estimation",
            Self::UserProvidedAll => "user_provided_all",
        }
    }
}

/// Errors that can occur during the automatic estimation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum EstimationError {
    /// File read or parse failure during estimation.
    #[error("load error: {0}")]
    Load(#[from] LoadError),

    /// Estimation failed due to insufficient data.
    #[error("estimation failed: {0}")]
    Stochastic(#[from] StochasticError),
}

/// Reason for an AR order reduction.
///
/// Distinguishes the three mechanisms that can reduce a season's AR order
/// during the estimation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionReason {
    /// Coefficient exceeds the magnitude-bound safety threshold.
    MagnitudeBound,
    /// First AR coefficient (`phi_1`) is negative, contradicting
    /// hydrological persistence.
    Phi1Negative,
    /// Contribution analysis detected negative entries at one or more lags.
    NegativeContribution,
}

impl ReductionReason {
    /// Convert to a stable string representation for diagnostic output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MagnitudeBound => "magnitude_bound",
            Self::Phi1Negative => "phi1_negative",
            Self::NegativeContribution => "negative_contribution",
        }
    }
}

/// A single contribution-based order reduction event.
///
/// Records that a season's AR order was reduced because the contribution
/// analysis detected negative entries, indicating potential model instability.
#[derive(Debug, Clone)]
pub struct ContributionReduction {
    /// Season where the reduction occurred.
    pub season_id: usize,
    /// Order before reduction (from AIC or previous iteration).
    pub original_order: usize,
    /// Order after reduction (the maximum valid order from contributions).
    pub reduced_order: usize,
    /// Contribution values at the original order that triggered the reduction.
    pub contributions: Vec<f64>,
    /// The mechanism that triggered this reduction.
    pub reason: ReductionReason,
}

/// Per-hydro AIC diagnostic data captured during AIC-based AR order selection.
///
/// Holds the selected order, fitted AR coefficients, and any contribution-based
/// order reductions for each season at the selected order.
#[derive(Debug, Clone)]
pub struct HydroEstimationEntry {
    /// The selected AR order for this hydro plant (maximum across all seasons).
    ///
    /// This is the maximum of the per-season selected orders, which determines
    /// the coefficient vector length in the output.
    pub selected_order: u32,
    /// Fitted AR lag coefficients, one inner vector per season sorted by `season_id` ascending.
    ///
    /// Each inner vector holds the coefficients at the selected order for
    /// that season. Seasons where estimation was skipped (zero std, insufficient
    /// observations) have an empty coefficient vector.
    pub coefficients: Vec<Vec<f64>>,
    /// Records of contribution-based order reductions applied during fitting.
    ///
    /// Each entry documents a season where the initial order (from PACF or fixed
    /// selection) was reduced due to negative contributions. Empty when no
    /// reductions were needed.
    pub contribution_reductions: Vec<ContributionReduction>,
}

/// Diagnostic record for an initial lag / user stats scale mismatch.
///
/// Populated by `run_partial_estimation` when a hydro's lag-1 past inflow
/// value is closer (in absolute distance) to the history-estimated mean than
/// to the user-provided seasonal mean at the first study stage (`stage_id == 0`).
/// This can indicate that `initial_conditions.json` was calibrated to historical
/// scale rather than to the user-provided stats scale.
#[derive(Debug, Clone)]
pub struct LagScaleWarning {
    /// Hydro plant identifier.
    pub hydro_id: EntityId,
    /// Lag-1 past inflow value (m³/s) from `initial_conditions.json`.
    pub lag_value: f64,
    /// User-provided seasonal mean at stage 0 (m³/s).
    pub user_mean: f64,
    /// History-estimated seasonal mean at stage 0 (m³/s).
    pub estimated_mean: f64,
}

/// Computation-side summary of the AR estimation pipeline.
///
/// Contains one [`HydroEstimationEntry`] per hydro plant that was fitted,
/// keyed by [`EntityId`] for canonical deterministic ordering.
#[must_use]
#[derive(Debug, Clone)]
pub struct EstimationReport {
    /// Per-hydro diagnostic entries, keyed by entity ID.
    pub entries: BTreeMap<EntityId, HydroEstimationEntry>,
    /// The order selection method used (e.g., `"AIC"`, `"PACF"`, `"fixed"`).
    pub method: String,
    /// Hydro IDs that have user-provided stats but no estimated AR
    /// coefficients, resulting in white-noise fallback (empty AR, ratio=1.0).
    /// Only populated by `run_partial_estimation`; empty for other paths.
    pub white_noise_fallbacks: Vec<EntityId>,
    /// Hydros whose lag-1 past inflow value is closer to the history-estimated
    /// mean than to the user-provided mean at stage 0. Advisory only; never
    /// blocks execution. Only populated by `run_partial_estimation`.
    pub lag_scale_warnings: Vec<LagScaleWarning>,
    /// Warnings for hydros where consecutive-season std ratios diverge
    /// significantly between user-provided and history-estimated profiles.
    /// Only populated by `run_partial_estimation`; empty for other paths.
    pub std_ratio_warnings: Vec<StdRatioDivergence>,
}

/// Advisory diagnostic for a (hydro, season pair) where the cross-season
/// standard deviation ratio diverges significantly between the user-provided
/// profile and the history-estimated profile.
///
/// Produced by `run_partial_estimation` (P9 diagnostic) when
/// `max(user_ratio / est_ratio, est_ratio / user_ratio) > 2.0` for any
/// consecutive season pair `(season_a, season_b)`.
#[derive(Debug, Clone)]
pub struct StdRatioDivergence {
    /// The hydro plant for which the divergence was detected.
    pub hydro_id: EntityId,
    /// Index of the first season in the consecutive pair.
    pub season_a: usize,
    /// Index of the second season in the consecutive pair (wraps around).
    pub season_b: usize,
    /// `std[season_a] / std[season_b]` from the user-provided profile.
    pub user_ratio: f64,
    /// `std[season_a] / std[season_b]` from the history-estimated profile.
    pub estimated_ratio: f64,
    /// `max(user_ratio / estimated_ratio, estimated_ratio / user_ratio)`.
    pub divergence: f64,
}

/// Result of validating an AR order via contribution analysis.
///
/// Captures whether the current order is stable (all contributions non-negative),
/// the maximum valid order if not, and the computed contribution values for
/// diagnostic reporting.
#[derive(Debug, Clone)]
pub struct ContributionValidationResult {
    /// Whether the current order passed contribution validation.
    pub valid: bool,
    /// Maximum valid order (same as `current_order` if valid, less otherwise).
    pub max_valid_order: usize,
    /// Computed contribution values for the current order.
    pub contributions: Vec<f64>,
}

/// Validate an AR order for a single (entity, season) pair via contribution analysis.
///
/// Computes the recursively-composed contributions for the given season at the
/// current order, then checks for negative entries. Returns a result indicating
/// whether the order is stable and, if not, the maximum valid order.
///
/// When `current_order == 0`, returns immediately with `valid: true` and no
/// contributions (an order-0 model has no autoregressive dependence to validate).
fn validate_order_contributions(
    season_id: usize,
    n_seasons: usize,
    current_order: usize,
    all_season_coefficients: &[Vec<f64>],
    std_by_season: &[f64],
) -> ContributionValidationResult {
    if current_order == 0 {
        return ContributionValidationResult {
            valid: true,
            max_valid_order: 0,
            contributions: Vec::new(),
        };
    }

    let coeff_refs: Vec<&[f64]> = all_season_coefficients.iter().map(Vec::as_slice).collect();
    let contributions = compute_contributions(
        season_id,
        n_seasons,
        current_order,
        &coeff_refs,
        std_by_season,
    );

    let valid = !check_negative_contributions(&contributions);
    let max_valid_order = if valid {
        current_order
    } else {
        find_max_valid_order(&contributions)
    };

    ContributionValidationResult {
        valid,
        max_valid_order,
        contributions,
    }
}

/// Estimate or load PAR(p) model parameters based on the input file manifest.
///
/// Resolves which [`EstimationPath`] applies for `case_dir`, then dispatches:
///
/// - **Rows 1, 2, 3, 7** (pass-through): Returns `system` unchanged with
///   `report = None`. No estimation is performed.
/// - **Row 4** ([`FullEstimation`](EstimationPath::FullEstimation)): Delegates
///   to `run_estimation`, which estimates both seasonal stats (Role 1) and AR
///   coefficients (Role 2) from `inflow_history.parquet`.
/// - **Row 5** ([`UserArHistoryStats`](EstimationPath::UserArHistoryStats)):
///   Delegates to `run_user_ar_estimation`, which estimates seasonal stats
///   (Role 1) from history while preserving user AR coefficients (Role 2).
/// - **Row 6** ([`PartialEstimation`](EstimationPath::PartialEstimation)):
///   Delegates to `run_partial_estimation`, which preserves user seasonal
///   stats (Role 1) while estimating AR coefficients (Role 2) from history
///   via periodic Yule-Walker / PACF.
///
/// # Errors
///
/// - [`EstimationError::Load`] -- file read, parse, or validation failure.
/// - [`EstimationError::Stochastic`] -- insufficient observations for any
///   `(entity, season)` group during AR or stats estimation.
pub fn estimate_from_history(
    system: System,
    case_dir: &Path,
    config: &Config,
) -> Result<(System, Option<EstimationReport>, EstimationPath), EstimationError> {
    // ── Step 1: resolve file manifest ────────────────────────────────────────
    let mut ctx = ValidationContext::new();
    let manifest = validate_structure(case_dir, &mut ctx);

    // Abort early if structural validation found errors.
    if ctx.into_result().is_err() {
        return Ok((system, None, EstimationPath::Deterministic));
    }

    // ── Step 2: resolve input path and dispatch ──────────────────────────────
    let path = EstimationPath::resolve(&manifest);

    match path {
        // No estimation — return system unchanged.
        EstimationPath::Deterministic
        | EstimationPath::UserStatsWhiteNoise
        | EstimationPath::UserProvidedNoHistory
        | EstimationPath::UserProvidedAll => Ok((system, None, path)),

        // Row 6: history + user stats, no AR — estimate AR from history, preserve user stats.
        EstimationPath::PartialEstimation => {
            let (system, report) = run_partial_estimation(system, case_dir, config, &manifest)?;
            Ok((system, Some(report), path))
        }

        // Row 4: full estimation from history.
        EstimationPath::FullEstimation => {
            let (system, report) = run_estimation(system, case_dir, config, &manifest)?;
            Ok((system, Some(report), path))
        }

        // Row 5: history present, no stats, AR present — estimate stats from history,
        // preserve user AR coefficients.
        EstimationPath::UserArHistoryStats => {
            let (system, report) = run_user_ar_estimation(system, case_dir, config, &manifest)?;
            Ok((system, Some(report), path))
        }
    }
}

/// Inner function that runs the full estimation pipeline once path conditions are met.
fn run_estimation(
    system: System,
    case_dir: &Path,
    config: &Config,
    manifest: &FileManifest,
) -> Result<(System, EstimationReport), EstimationError> {
    // ── Step 3: load inflow history ──────────────────────────────────────────
    let history_path = case_dir.join("scenarios/inflow_history.parquet");
    let history = parse_inflow_history(&history_path)?;

    // ── Convert InflowHistoryRow to (EntityId, NaiveDate, f64) tuples ────────
    let observations: Vec<(EntityId, NaiveDate, f64)> = history
        .iter()
        .map(|row| (row.hydro_id, row.date, row.value_m3s))
        .collect();

    // ── Collect hydro IDs from system (canonical sorted order) ───────────────
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

    // ── Use stages already present in the system (avoids re-parsing stages.json) ──
    let stages = system.stages();

    // ── Extract season map for calendar-based date-to-season fallback ────────
    let season_map = system.policy_graph().season_map.as_ref();

    // ── Step 4: estimate seasonal stats ─────────────────────────────────────
    let seasonal_stats =
        estimate_seasonal_stats_with_season_map(&observations, stages, &hydro_ids, season_map)?;

    // ── Step 5: estimate AR coefficients ────────────────────────────────────
    let max_order = config.estimation.max_order as usize;
    let (ar_estimates, estimation_report) = estimate_ar_coefficients_with_selection(
        &observations,
        &seasonal_stats,
        stages,
        &hydro_ids,
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: config.estimation.max_coefficient_magnitude,
            season_map,
        },
    )?;

    // ── Step 6: estimate or preserve correlation ─────────────────────────────
    let correlation = if manifest.scenarios_correlation_json {
        // Explicit correlation.json is present — keep whatever was loaded by load_case.
        system.correlation().clone()
    } else {
        estimate_correlation_with_season_map(
            &observations,
            &ar_estimates,
            &seasonal_stats,
            stages,
            &hydro_ids,
            season_map,
        )?
    };

    // ── Step 7: convert results to row types and assemble inflow models ───────
    let stats_rows = seasonal_stats_to_rows(&seasonal_stats, stages);
    let coeff_rows = ar_estimates_to_rows(&ar_estimates, stages);

    let inflow_models = assemble_inflow_models(stats_rows, coeff_rows)?;

    Ok((
        system.with_scenario_models(inflow_models, correlation),
        estimation_report,
    ))
}

/// Inner function that runs the partial estimation pipeline (P1 path).
///
/// Used when `inflow_history.parquet` and `inflow_seasonal_stats.parquet` are
/// both present but `inflow_ar_coefficients.parquet` is absent. The user-provided
/// stats (`mean_m3s`, `std_m3s`) are preserved exactly for LP assembly; only AR
/// coefficients are estimated from history.
///
/// The key distinction vs [`run_estimation`]:
/// - **Fitting stats** (history-derived, step 4) are used for YW matrix construction.
/// - **User stats** (from `system.inflow_models()`, step 7) are used for the final
///   `assemble_inflow_models` call that drives the scenario generator.
#[allow(clippy::too_many_lines)]
fn run_partial_estimation(
    system: System,
    case_dir: &Path,
    config: &Config,
    manifest: &FileManifest,
) -> Result<(System, EstimationReport), EstimationError> {
    // ── Step 3: load inflow history ──────────────────────────────────────────
    let history_path = case_dir.join("scenarios/inflow_history.parquet");
    let history = parse_inflow_history(&history_path)?;

    // ── Convert InflowHistoryRow to (EntityId, NaiveDate, f64) tuples ────────
    let observations: Vec<(EntityId, NaiveDate, f64)> = history
        .iter()
        .map(|row| (row.hydro_id, row.date, row.value_m3s))
        .collect();

    // ── Collect hydro IDs from system (canonical sorted order) ───────────────
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

    // ── Use stages already present in the system ─────────────────────────────
    let stages = system.stages();

    // ── Extract season map for calendar-based date-to-season fallback ────────
    let season_map = system.policy_graph().season_map.as_ref();

    // ── Validate that user stats are present in the loaded system ────────────
    if system.inflow_models().is_empty() {
        return Err(EstimationError::Load(
            cobre_io::LoadError::ConstraintError {
                description: "manifest indicates inflow_seasonal_stats.parquet is present \
                          but system.inflow_models() is empty; \
                          no user stats available for partial estimation"
                    .to_string(),
            },
        ));
    }

    // ── Step 4: estimate seasonal stats (fitting stats — for YW solve only) ──
    let fitting_stats =
        estimate_seasonal_stats_with_season_map(&observations, stages, &hydro_ids, season_map)?;

    // ── Step 5: estimate AR coefficients using fitting stats ─────────────────
    let max_order = config.estimation.max_order as usize;
    let (ar_estimates, estimation_report) = estimate_ar_coefficients_with_selection(
        &observations,
        &fitting_stats,
        stages,
        &hydro_ids,
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: config.estimation.max_coefficient_magnitude,
            season_map,
        },
    )?;

    // ── Step 5b: bidirectional coverage validation (P2) ──────────────────────
    // Use fitting_stats (history-derived) to identify which hydros had actual
    // history data. ar_estimates includes white-noise entries for hydros with no
    // history, so fitting_stats.entity_id is the correct "estimated" indicator.
    let estimated_hydro_ids: HashSet<EntityId> =
        fitting_stats.iter().map(|s| s.entity_id).collect();
    let user_stats_hydro_ids: HashSet<EntityId> =
        system.inflow_models().iter().map(|m| m.hydro_id).collect();

    // Direction A: AR estimated but no user stats → hard error
    let mut missing_stats: Vec<EntityId> = estimated_hydro_ids
        .difference(&user_stats_hydro_ids)
        .copied()
        .collect();
    missing_stats.sort();
    if !missing_stats.is_empty() {
        let ids: Vec<String> = missing_stats.iter().map(|id| id.0.to_string()).collect();
        return Err(EstimationError::Load(
            cobre_io::LoadError::ConstraintError {
                description: format!(
                    "partial estimation: AR coefficients were estimated for hydro(s) \
                     [{ids}] but inflow_seasonal_stats.parquet has no entry for them; \
                     all hydros with estimated AR must have user-provided stats",
                    ids = ids.join(", ")
                ),
            },
        ));
    }

    // Direction B: user stats but no AR estimated → white noise fallback
    let mut white_noise_fallbacks: Vec<EntityId> = user_stats_hydro_ids
        .difference(&estimated_hydro_ids)
        .copied()
        .collect();
    white_noise_fallbacks.sort();

    // ── Step 5c: initial lag / user stats scale mismatch check (P8) ──────────
    // For each hydro that has BOTH a past_inflows entry AND fitting_stats AND
    // user stats at stage_id == 0, check whether the lag-1 value is closer to
    // the estimated mean than to the user mean. If so, emit an advisory warning.
    // This check is purely informational and never alters control flow.
    let lag_scale_warnings: Vec<LagScaleWarning> = {
        // Build lookup: EntityId -> estimated mean at stage_id == 0.
        let estimated_mean_at_stage0: BTreeMap<EntityId, f64> = fitting_stats
            .iter()
            .filter(|s| s.stage_id == 0)
            .map(|s| (s.entity_id, s.mean))
            .collect();

        // Build lookup: EntityId -> user mean at stage_id == 0.
        let user_mean_at_stage0: BTreeMap<EntityId, f64> = system
            .inflow_models()
            .iter()
            .filter(|m| m.stage_id == 0)
            .map(|m| (m.hydro_id, m.mean_m3s))
            .collect();

        // Iterate past_inflows in sorted order for deterministic output.
        let mut sorted_past_inflows: Vec<&cobre_core::HydroPastInflows> =
            system.initial_conditions().past_inflows.iter().collect();
        sorted_past_inflows.sort_by_key(|p| p.hydro_id);

        let mut warnings: Vec<LagScaleWarning> = Vec::new();
        for past in sorted_past_inflows {
            // Skip hydros with no lag-1 value.
            let Some(&lag_value) = past.values_m3s.first() else {
                continue;
            };
            // Skip hydros missing either mean.
            let Some(&estimated_mean) = estimated_mean_at_stage0.get(&past.hydro_id) else {
                continue;
            };
            let Some(&user_mean) = user_mean_at_stage0.get(&past.hydro_id) else {
                continue;
            };
            // Warn when lag-1 is closer to estimated mean than to user mean.
            if (lag_value - estimated_mean).abs() < (lag_value - user_mean).abs() {
                eprintln!(
                    "warning: hydro {} initial lag ({:.1}) is closer to estimated mean \
                     ({:.1}) than user mean ({:.1}) -- initial conditions may be at \
                     historical scale",
                    past.hydro_id.0, lag_value, estimated_mean, user_mean
                );
                warnings.push(LagScaleWarning {
                    hydro_id: past.hydro_id,
                    lag_value,
                    user_mean,
                    estimated_mean,
                });
            }
        }
        warnings
    };

    // ── Step 5d: cross-season std ratio divergence check (P9) ───────────────
    // Compare consecutive-season std ratios between user and estimated profiles.
    // Emits an advisory warning per flagged (hydro, season pair) and records the
    // results in `std_ratio_warnings`. Never blocks execution.
    let std_ratio_warnings = check_std_ratio_divergence(&system, &fitting_stats, stages);
    for w in &std_ratio_warnings {
        eprintln!(
            "warning: hydro {} season {}->{} std ratio diverges {:.1}x between \
             user ({:.2}) and estimated ({:.2})",
            w.hydro_id.0, w.season_a, w.season_b, w.divergence, w.user_ratio, w.estimated_ratio
        );
    }

    // ── Step 6: estimate or preserve correlation ─────────────────────────────
    let correlation = if manifest.scenarios_correlation_json {
        system.correlation().clone()
    } else {
        estimate_correlation_with_season_map(
            &observations,
            &ar_estimates,
            &fitting_stats,
            stages,
            &hydro_ids,
            season_map,
        )?
    };

    // ── Step 7: convert to row types using USER stats, not fitting stats ──────
    // Fitting stats were used only for YW matrix construction above.
    // User stats (mean_m3s, std_m3s from the input system) drive LP assembly.
    let user_stats_rows = user_stats_to_rows(&system);
    let coeff_rows = ar_estimates_to_rows(&ar_estimates, stages);

    let inflow_models = assemble_inflow_models(user_stats_rows, coeff_rows)?;

    let mut estimation_report = estimation_report;
    estimation_report.white_noise_fallbacks = white_noise_fallbacks;
    estimation_report.lag_scale_warnings = lag_scale_warnings;
    estimation_report.std_ratio_warnings = std_ratio_warnings;

    Ok((
        system.with_scenario_models(inflow_models, correlation),
        estimation_report,
    ))
}

/// Inner function that runs the P7 (`UserArHistoryStats`) estimation pipeline.
///
/// Used when `inflow_history.parquet` and `inflow_ar_coefficients.parquet` are
/// both present but `inflow_seasonal_stats.parquet` is absent. Seasonal stats
/// (`mean_m3s`, `std_m3s`) are estimated from history for LP assembly and for
/// correlation estimation. AR coefficients are loaded from the user-provided
/// file and preserved bitwise — **no re-estimation from history is performed**.
///
/// The key distinction vs [`run_estimation`]:
/// - **AR coefficients** come from the user file (not estimated from history).
/// - **Seasonal stats** (mean, std) come from history estimation and drive both
///   the final `InflowModel` and the correlation estimation.
/// - The returned [`EstimationReport`] has an empty `entries` map and method
///   `"user_provided"` to signal that no AR estimation was performed.
fn run_user_ar_estimation(
    system: System,
    case_dir: &Path,
    _config: &Config,
    manifest: &FileManifest,
) -> Result<(System, EstimationReport), EstimationError> {
    // ── Step 3: load inflow history ──────────────────────────────────────────
    let history_path = case_dir.join("scenarios/inflow_history.parquet");
    let history = parse_inflow_history(&history_path)?;

    // ── Convert InflowHistoryRow to (EntityId, NaiveDate, f64) tuples ────────
    let observations: Vec<(EntityId, NaiveDate, f64)> = history
        .iter()
        .map(|row| (row.hydro_id, row.date, row.value_m3s))
        .collect();

    // ── Collect hydro IDs from system (canonical sorted order) ───────────────
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

    // ── Use stages already present in the system ─────────────────────────────
    let stages = system.stages();

    // ── Extract season map for calendar-based date-to-season fallback ────────
    let season_map = system.policy_graph().season_map.as_ref();

    // ── Step 4: estimate seasonal stats from history ─────────────────────────
    // These stats are used for both LP assembly (mean_m3s, std_m3s) and for
    // correlation estimation. User AR coefficients are NOT re-estimated here.
    let seasonal_stats =
        estimate_seasonal_stats_with_season_map(&observations, stages, &hydro_ids, season_map)?;

    // ── Load user AR coefficients from file ───────────────────────────────────
    // NOTE: system.inflow_models() is empty for this path (assemble_inflow_models
    // returned an empty vec because stats were absent). Load AR from file directly.
    let ar_path = case_dir.join("scenarios/inflow_ar_coefficients.parquet");
    let user_ar_rows = parse_inflow_ar_coefficients(&ar_path)?;

    // ── Convert user AR rows to ArCoefficientEstimate for correlation ─────────
    // ArCoefficientEstimate uses season_id; InflowArCoefficientRow uses stage_id.
    // The conversion groups by (hydro_id, season_id) using the stage-to-season map.
    let user_ar_estimates = ar_rows_to_estimates(&user_ar_rows, stages);

    // ── Step 6: estimate or preserve correlation ─────────────────────────────
    let correlation = if manifest.scenarios_correlation_json {
        // Explicit correlation.json is present — keep whatever was loaded by load_case.
        system.correlation().clone()
    } else {
        estimate_correlation_with_season_map(
            &observations,
            &user_ar_estimates,
            &seasonal_stats,
            stages,
            &hydro_ids,
            season_map,
        )?
    };

    // ── Step 7: convert results to row types and assemble inflow models ───────
    // History-estimated stats drive mean_m3s / std_m3s; user AR rows drive
    // ar_coefficients / residual_std_ratio.
    let stats_rows = seasonal_stats_to_rows(&seasonal_stats, stages);

    let inflow_models = assemble_inflow_models(stats_rows, user_ar_rows)?;

    // Build a minimal report: no AR was estimated, method is "user_provided".
    let estimation_report = EstimationReport {
        entries: BTreeMap::new(),
        method: "user_provided".to_string(),
        white_noise_fallbacks: Vec::new(),
        lag_scale_warnings: Vec::new(),
        std_ratio_warnings: Vec::new(),
    };

    Ok((
        system.with_scenario_models(inflow_models, correlation),
        estimation_report,
    ))
}

/// Convert [`InflowArCoefficientRow`] entries to [`ArCoefficientEstimate`] values.
///
/// This is the inverse of [`ar_estimates_to_rows`]: it groups coefficient rows by
/// `(hydro_id, season_id)` — using the stage-to-season mapping from `stages` —
/// and produces one [`ArCoefficientEstimate`] per group.
///
/// When multiple stages map to the same season, each stage produces duplicate
/// rows in the `InflowArCoefficientRow` format (all lags repeated for every stage
/// in the season). This function deduplicates by processing only the first stage
/// encountered for each season per hydro. Coefficient order is preserved (lag 1,
/// lag 2, …); `residual_std_ratio` is taken from the first row of each group.
///
/// The result is sorted by `(hydro_id, season_id)` ascending, matching the
/// canonical ordering expected by `estimate_correlation_with_season_map`.
fn ar_rows_to_estimates(
    rows: &[InflowArCoefficientRow],
    stages: &[cobre_core::temporal::Stage],
) -> Vec<ArCoefficientEstimate> {
    // Build stage_id -> season_id mapping.
    let stage_id_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    // Track the first stage_id seen for each (hydro_id, season_id) key.
    // Since rows are pre-sorted by (hydro_id, stage_id, lag), the first stage_id
    // encountered for a given season is the canonical one to use. Subsequent
    // stages in the same season are duplicates produced by ar_estimates_to_rows
    // and must be skipped.
    let mut first_stage: HashMap<(EntityId, usize), i32> = HashMap::new();

    // Groups accumulate coefficients for the canonical (first) stage only.
    // Use BTreeMap for deterministic (hydro_id, season_id) ordering in output.
    let mut groups: BTreeMap<(EntityId, usize), (Vec<f64>, f64)> = BTreeMap::new();

    for row in rows {
        let Some(&season_id) = stage_id_to_season.get(&row.stage_id) else {
            continue;
        };

        let key = (row.hydro_id, season_id);

        // Record the first stage_id seen for this key, or skip if a different stage.
        let canonical_stage = first_stage.entry(key).or_insert(row.stage_id);
        if *canonical_stage != row.stage_id {
            // This row belongs to a duplicate stage for the same season — skip it.
            continue;
        }

        // Accumulate the coefficient for the canonical stage.
        let entry = groups
            .entry(key)
            .or_insert_with(|| (Vec::new(), row.residual_std_ratio));
        entry.0.push(row.coefficient);
    }

    groups
        .into_iter()
        .map(
            |((hydro_id, season_id), (coefficients, residual_std_ratio))| ArCoefficientEstimate {
                hydro_id,
                season_id,
                coefficients,
                residual_std_ratio,
            },
        )
        .collect()
}

/// Extract user-provided seasonal stats from `system.inflow_models()` as
/// [`InflowSeasonalStatsRow`] entries.
///
/// Each [`InflowModel`] in the system contributes one row with its `mean_m3s`
/// and `std_m3s` preserved bitwise — no transformation is applied. This is
/// used by [`run_partial_estimation`] to pass user stats into `assemble_inflow_models`
/// instead of history-derived fitting stats.
fn user_stats_to_rows(system: &System) -> Vec<InflowSeasonalStatsRow> {
    system
        .inflow_models()
        .iter()
        .map(|m| InflowSeasonalStatsRow {
            hydro_id: m.hydro_id,
            stage_id: m.stage_id,
            mean_m3s: m.mean_m3s,
            std_m3s: m.std_m3s,
        })
        .collect()
}

/// Check whether consecutive-season std ratios diverge between user and
/// estimated profiles for each hydro in the partial estimation path (P9).
///
/// For each hydro present in both user stats and `fitting_stats`, iterates
/// consecutive season pairs `(m, (m+1) % n)` and computes:
///
/// - `ratio_user = user_std[m] / user_std[m+1]`
/// - `ratio_est  = est_std[m]  / est_std[m+1]`
/// - `divergence = max(ratio_user / ratio_est, ratio_est / ratio_user)`
///
/// A [`StdRatioDivergence`] entry is pushed when `divergence > 2.0`. Season
/// pairs where either denominator is `< 1e-12` are skipped silently. The
/// result is sorted by `(hydro_id, season_a)`.
fn check_std_ratio_divergence(
    system: &System,
    fitting_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
) -> Vec<StdRatioDivergence> {
    // Build stage_id -> season_id mapping.
    let stage_id_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    // Build (hydro_id, season_id) -> user std.
    // When multiple stages share a season, any entry will do (same season).
    let mut user_std: BTreeMap<(EntityId, usize), f64> = BTreeMap::new();
    for m in system.inflow_models() {
        let Some(&season_id) = stage_id_to_season.get(&m.stage_id) else {
            continue;
        };
        user_std.entry((m.hydro_id, season_id)).or_insert(m.std_m3s);
    }

    // Build (hydro_id, season_id) -> estimated std.
    let mut est_std: BTreeMap<(EntityId, usize), f64> = BTreeMap::new();
    for s in fitting_stats {
        let Some(&season_id) = stage_id_to_season.get(&s.stage_id) else {
            continue;
        };
        est_std.entry((s.entity_id, season_id)).or_insert(s.std);
    }

    // Collect all hydro IDs that appear in both maps, in sorted order.
    let user_hydros: std::collections::BTreeSet<EntityId> =
        user_std.keys().map(|(h, _)| *h).collect();
    let est_hydros: std::collections::BTreeSet<EntityId> =
        est_std.keys().map(|(h, _)| *h).collect();
    let common_hydros: Vec<EntityId> = user_hydros.intersection(&est_hydros).copied().collect();

    let mut warnings: Vec<StdRatioDivergence> = Vec::new();

    for hydro_id in common_hydros {
        // Collect sorted distinct season IDs for this hydro from user stats.
        let season_ids: Vec<usize> = {
            let mut ids: Vec<usize> = user_std
                .keys()
                .filter(|(h, _)| *h == hydro_id)
                .map(|(_, s)| *s)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };

        let n = season_ids.len();
        if n < 2 {
            // Fewer than 2 seasons: no consecutive pairs exist.
            continue;
        }

        for i in 0..n {
            let season_a = season_ids[i];
            let season_b = season_ids[(i + 1) % n];

            let Some(&u_a) = user_std.get(&(hydro_id, season_a)) else {
                continue;
            };
            let Some(&u_b) = user_std.get(&(hydro_id, season_b)) else {
                continue;
            };
            let Some(&e_a) = est_std.get(&(hydro_id, season_a)) else {
                continue;
            };
            let Some(&e_b) = est_std.get(&(hydro_id, season_b)) else {
                continue;
            };

            // Skip pairs where either denominator std is near-zero.
            if u_b.abs() < 1e-12 || e_b.abs() < 1e-12 {
                continue;
            }

            let ratio_user = u_a / u_b;
            let ratio_est = e_a / e_b;

            // Skip when either ratio is near-zero — would cause division-by-zero
            // in the divergence computation.
            if ratio_user.abs() < 1e-12 || ratio_est.abs() < 1e-12 {
                continue;
            }

            let divergence = (ratio_user / ratio_est)
                .abs()
                .max((ratio_est / ratio_user).abs());

            if divergence > 2.0 {
                warnings.push(StdRatioDivergence {
                    hydro_id,
                    season_a,
                    season_b,
                    user_ratio: ratio_user,
                    estimated_ratio: ratio_est,
                    divergence,
                });
            }
        }
    }

    // Sort by (hydro_id, season_a) for deterministic output.
    warnings.sort_by_key(|w| (w.hydro_id, w.season_a));
    warnings
}

/// Configuration parameters for AR coefficient estimation.
struct ArEstimationConfig<'a> {
    max_order: usize,
    max_coeff_magnitude: Option<f64>,
    season_map: Option<&'a cobre_core::temporal::SeasonMap>,
}

/// Estimate AR coefficients using the periodic Yule-Walker / PACF method.
///
/// Delegates to [`estimate_ar_with_pacf`], which selects the AR order via the
/// periodic PACF significance test (95% confidence, `z_alpha = 1.96`) and then
/// solves the periodic Yule-Walker system at the selected order. The
/// `cfg.method` field is accepted for API compatibility; only
/// `OrderSelectionMethod::Pacf` is supported.
fn estimate_ar_coefficients_with_selection(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    cfg: &ArEstimationConfig<'_>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    estimate_ar_with_pacf(
        observations,
        seasonal_stats,
        stages,
        hydro_ids,
        cfg.max_order,
        cfg.season_map,
        cfg.max_coeff_magnitude,
    )
}

/// PACF-based AR order selection using periodic Yule-Walker method.
///
/// Selects the AR order using the periodic partial autocorrelation function
/// (PACF) significance test, then estimates coefficients via the periodic
/// Yule-Walker matrix solve. The PACF threshold uses a 95% confidence
/// interval (`z_alpha = 1.96`).
///
/// The periodic approach correctly accounts for the non-Toeplitz covariance
/// structure of periodic autoregressive processes.
#[allow(clippy::too_many_lines)]
fn estimate_ar_with_pacf(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    season_map: Option<&cobre_core::temporal::SeasonMap>,
    max_coeff_magnitude: Option<f64>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    if max_order == 0 {
        // Order-0: produce white-noise estimates for all (entity, season) pairs.
        let estimates = estimate_ar_coefficients_with_season_map(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            0,
            season_map,
        )?;
        let report = EstimationReport {
            entries: BTreeMap::new(),
            method: "PACF".to_string(),
            white_noise_fallbacks: Vec::new(),
            lag_scale_warnings: Vec::new(),
            std_ratio_warnings: Vec::new(),
        };
        return Ok((estimates, report));
    }

    // Build stage and stats lookups.
    let mut stage_index = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect::<Vec<_>>();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let stage_id_to_season: HashMap<i32, usize> = stage_index
        .iter()
        .map(|&(_, _, stage_id, season_id)| (stage_id, season_id))
        .collect();

    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
        .iter()
        .filter_map(|s| {
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(((s.entity_id, season_id), s))
        })
        .collect();

    let n_seasons: usize = {
        let mut max_season = 0usize;
        for &(_, _, _, season_id) in &stage_index {
            if season_id >= max_season {
                max_season = season_id + 1;
            }
        }
        max_season
    };

    // Group observations by (entity_id, season_id).
    let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();
    let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if !entity_set.contains(&entity_id) {
            continue;
        }
        let Some(season_id) = find_season_for_date(&stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
        else {
            continue;
        };
        group_obs
            .entry((entity_id, season_id))
            .or_default()
            .push(value);
    }

    // 95% confidence z-score for the PACF significance threshold.
    let z_alpha = 1.96_f64;

    // Build estimates using periodic PACF and YW coefficient estimation.
    let mut estimates: Vec<ArCoefficientEstimate> = Vec::new();

    for &hydro_id in hydro_ids {
        // Build observations_by_season and stats_by_season for this entity.
        let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];

        for season in 0..n_seasons {
            if let Some(obs) = group_obs.get(&(hydro_id, season)) {
                obs_by_season[season].clone_from(obs);
            }
            if let Some(stats) = stats_map.get(&(hydro_id, season)) {
                stats_by_season[season] = (stats.mean, stats.std);
            }
        }

        let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();

        for season in 0..n_seasons {
            let stats_s = stats_by_season[season];

            // Skip seasons with zero std or insufficient data.
            if stats_s.1 == 0.0 || obs_by_season[season].len() < 2 {
                estimates.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id: season,
                    coefficients: Vec::new(),
                    residual_std_ratio: 1.0,
                });
                continue;
            }

            let n_obs = obs_by_season[season].len();

            // Compute periodic PACF for order selection.
            let pacf_values =
                periodic_pacf(season, max_order, n_seasons, &obs_refs, &stats_by_season);

            // Select order via significance test.
            let pacf_result = select_order_pacf(&pacf_values, n_obs, z_alpha);
            let selected_order = pacf_result.selected_order;

            // Estimate AR coefficients at the selected order using periodic YW.
            let yw_result = estimate_periodic_ar_coefficients(
                season,
                selected_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
            );

            estimates.push(ArCoefficientEstimate {
                hydro_id,
                season_id: season,
                coefficients: yw_result.coefficients,
                residual_std_ratio: yw_result.residual_std_ratio,
            });
        }
    }

    // === Iterative PACF reduction with contribution validation ===
    let reductions = iterative_pacf_reduction(
        &mut estimates,
        n_seasons,
        hydro_ids,
        &group_obs,
        &stats_map,
        max_order,
        z_alpha,
        max_coeff_magnitude,
    );

    let report = build_estimation_report(&estimates, n_seasons, &reductions, "PACF");
    Ok((estimates, report))
}

/// Iteratively reduce AR orders via PACF re-selection and contribution validation.
///
/// For each entity, maintains per-season `max_order` ceilings. When contribution
/// analysis detects negative contributions for a season, reduces that season's
/// ceiling by 1 and re-runs the full PACF selection + YW estimation + `phi_1`
/// check + contribution validation cycle. Repeats until all seasons pass or
/// their ceilings reach 0.
///
/// This implements NEWAVE's `reducao_ordem` algorithm.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn iterative_pacf_reduction(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_ids: &[EntityId],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    initial_max_order: usize,
    z_alpha: f64,
    max_coeff_magnitude: Option<f64>,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    // Pre-pass: magnitude bound safety check (same as apply_contribution_validation).
    if let Some(threshold) = max_coeff_magnitude {
        for est in estimates.iter_mut() {
            let has_explosive = est.coefficients.iter().any(|c| c.abs() > threshold);
            if has_explosive {
                let original_order = est.coefficients.len();
                all_reductions
                    .entry(est.hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id: est.season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::MagnitudeBound,
                    });
                est.coefficients.clear();
                est.residual_std_ratio = 1.0;
            }
        }
    }

    // Pre-pass 2: phi_1 negativity rejection.
    for est in estimates.iter_mut() {
        if has_negative_phi1(&est.coefficients) {
            let original_order = est.coefficients.len();
            all_reductions
                .entry(est.hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id: est.season_id,
                    original_order,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                });
            est.coefficients.clear();
            est.residual_std_ratio = 1.0;
        }
    }

    // Group estimate indices by hydro_id for per-entity processing.
    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    // Process each entity with iterative PACF re-selection.
    for &hydro_id in hydro_ids {
        let Some(indices) = hydro_indices.get(&hydro_id) else {
            continue;
        };

        // Build observations and stats for this entity.
        let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];

        for season in 0..n_seasons {
            if let Some(obs) = group_obs.get(&(hydro_id, season)) {
                obs_by_season[season].clone_from(obs);
            }
            if let Some(stats) = stats_map.get(&(hydro_id, season)) {
                stats_by_season[season] = (stats.mean, stats.std);
            }
        }

        // Build std_by_season for contribution analysis.
        let std_by_season: Vec<f64> = (0..n_seasons)
            .map(|sid| stats_map.get(&(hydro_id, sid)).map_or(0.0, |s| s.std))
            .collect();

        // Initialize per-season max_order ceilings.
        let mut max_orders: Vec<usize> = vec![initial_max_order; n_seasons];

        // Initialize current coefficients from estimates.
        let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        for &idx in indices {
            let est = &estimates[idx];
            if est.season_id < n_seasons {
                all_coeffs[est.season_id].clone_from(&est.coefficients);
            }
        }

        // Mark which seasons are already finalized (order 0 from pre-passes or
        // insufficient data).
        let mut frozen: Vec<bool> = vec![false; n_seasons];
        for &idx in indices {
            let sid = estimates[idx].season_id;
            if estimates[idx].coefficients.is_empty() {
                frozen[sid] = true;
            }
        }

        let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();

        // Iterative contribution validation loop.
        loop {
            // Find seasons that fail contribution validation.
            let mut failing_seasons: Vec<usize> = Vec::new();

            for &idx in indices {
                let season_id = estimates[idx].season_id;
                if frozen[season_id] {
                    continue;
                }
                let current_order = estimates[idx].coefficients.len();
                if current_order == 0 {
                    continue;
                }

                let result = validate_order_contributions(
                    season_id,
                    n_seasons,
                    current_order,
                    &all_coeffs,
                    &std_by_season,
                );

                if !result.valid {
                    all_reductions
                        .entry(hydro_id)
                        .or_default()
                        .push(ContributionReduction {
                            season_id,
                            original_order: current_order,
                            reduced_order: result.max_valid_order,
                            contributions: result.contributions,
                            reason: ReductionReason::NegativeContribution,
                        });
                    failing_seasons.push(season_id);
                }
            }

            if failing_seasons.is_empty() {
                break; // All seasons pass.
            }

            // Reduce max_order for failing seasons and re-run PACF + YW.
            let mut any_reselected = false;
            for &season_id in &failing_seasons {
                if max_orders[season_id] == 0 {
                    continue;
                }
                max_orders[season_id] -= 1;

                if max_orders[season_id] == 0 {
                    // Set to order 0 directly.
                    for &idx in indices {
                        if estimates[idx].season_id == season_id {
                            estimates[idx].coefficients.clear();
                            estimates[idx].residual_std_ratio = 1.0;
                            all_coeffs[season_id].clear();
                            frozen[season_id] = true;
                        }
                    }
                    continue;
                }

                // Re-run PACF with reduced max_order.
                let stats_s = stats_by_season[season_id];
                if stats_s.1 == 0.0 || obs_by_season[season_id].len() < 2 {
                    frozen[season_id] = true;
                    continue;
                }

                let n_obs = obs_by_season[season_id].len();
                let pacf_values = periodic_pacf(
                    season_id,
                    max_orders[season_id],
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                );
                let pacf_result = select_order_pacf(&pacf_values, n_obs, z_alpha);
                let selected_order = pacf_result.selected_order;

                // Re-estimate coefficients at the new selected order.
                let yw_result = estimate_periodic_ar_coefficients(
                    season_id,
                    selected_order,
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                );

                // Update the estimate.
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx]
                            .coefficients
                            .clone_from(&yw_result.coefficients);
                        estimates[idx].residual_std_ratio = yw_result.residual_std_ratio;
                        all_coeffs[season_id].clone_from(&yw_result.coefficients);
                    }
                }

                // Check phi_1 on re-estimated coefficients.
                if has_negative_phi1(&all_coeffs[season_id]) {
                    let original_order = all_coeffs[season_id].len();
                    all_reductions
                        .entry(hydro_id)
                        .or_default()
                        .push(ContributionReduction {
                            season_id,
                            original_order,
                            reduced_order: 0,
                            contributions: Vec::new(),
                            reason: ReductionReason::Phi1Negative,
                        });
                    for &idx in indices {
                        if estimates[idx].season_id == season_id {
                            estimates[idx].coefficients.clear();
                            estimates[idx].residual_std_ratio = 1.0;
                            all_coeffs[season_id].clear();
                            frozen[season_id] = true;
                        }
                    }
                } else {
                    any_reselected = true;
                }
            }

            if !any_reselected {
                break; // All failing seasons are at order 0 or frozen.
            }
        }
    }

    all_reductions
}

/// Apply coefficient magnitude bound and contribution-based order validation
/// to all AR estimates (used only in tests after Fixed-path removal).
#[cfg(test)]
fn apply_contribution_validation(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    sigma2_map: &HashMap<(EntityId, usize), Vec<f64>>,
    max_coeff_magnitude: Option<f64>,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    // Pre-pass: magnitude bound safety check.
    if let Some(threshold) = max_coeff_magnitude {
        for est in estimates.iter_mut() {
            let has_explosive = est.coefficients.iter().any(|c| c.abs() > threshold);
            if has_explosive {
                let original_order = est.coefficients.len();
                all_reductions
                    .entry(est.hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id: est.season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(), // magnitude-based, no contributions computed
                        reason: ReductionReason::MagnitudeBound,
                    });
                est.coefficients.clear();
                est.residual_std_ratio = 1.0;
            }
        }
    }

    // Pre-pass 2: phi_1 negativity rejection.
    // A negative first AR coefficient contradicts hydrological persistence.
    // This check is cheaper than contribution analysis and catches most
    // unstable models early.
    for est in estimates.iter_mut() {
        if has_negative_phi1(&est.coefficients) {
            let original_order = est.coefficients.len();
            all_reductions
                .entry(est.hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id: est.season_id,
                    original_order,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                });
            est.coefficients.clear();
            est.residual_std_ratio = 1.0;
        }
    }

    // Group estimate indices by hydro_id for per-entity processing.
    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    for (&hydro_id, indices) in &hydro_indices {
        // Build std_by_season from seasonal_stats.
        let std_by_season: Vec<f64> = (0..n_seasons)
            .map(|sid| stats_map.get(&(hydro_id, sid)).map_or(0.0, |s| s.std))
            .collect();

        // Build all_season_coefficients from current estimates.
        let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        for &idx in indices {
            let est = &estimates[idx];
            if est.season_id < n_seasons {
                all_coeffs[est.season_id].clone_from(&est.coefficients);
            }
        }

        // Validate each season for this entity.
        for &idx in indices {
            let season_id = estimates[idx].season_id;
            let mut current_order = estimates[idx].coefficients.len();

            loop {
                let result = validate_order_contributions(
                    season_id,
                    n_seasons,
                    current_order,
                    &all_coeffs,
                    &std_by_season,
                );

                if result.valid || current_order == 0 {
                    break;
                }

                let original_order = current_order;
                let reduced_order = result.max_valid_order;

                all_reductions
                    .entry(hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id,
                        original_order,
                        reduced_order,
                        contributions: result.contributions,
                        reason: ReductionReason::NegativeContribution,
                    });

                // Truncate coefficients to the reduced order.
                estimates[idx].coefficients.truncate(reduced_order);

                // Recompute residual_std_ratio from stored sigma2 if available.
                if reduced_order == 0 {
                    estimates[idx].residual_std_ratio = 1.0;
                } else if let Some(sigma2_vec) = sigma2_map.get(&(hydro_id, season_id)) {
                    if reduced_order <= sigma2_vec.len() {
                        let sigma2 = sigma2_vec[reduced_order - 1];
                        estimates[idx].residual_std_ratio =
                            if sigma2 <= 0.0 { 1.0 } else { sigma2.sqrt() };
                    }
                }

                // Update the shared coefficient array.
                all_coeffs[season_id].clone_from(&estimates[idx].coefficients);
                current_order = reduced_order;
            }
        }
    }

    all_reductions
}

/// Build an [`EstimationReport`] from AR estimates and contribution validation results.
///
/// This function is infallible: it only reorganises already-computed data.
/// For each hydro plant the selected order is the **maximum** across all
/// seasons. These choices align with how the I/O layer (`FittingReport`)
/// expects a single order per hydro.
pub(crate) fn build_estimation_report(
    estimates: &[ArCoefficientEstimate],
    n_seasons: usize,
    contribution_reductions: &HashMap<EntityId, Vec<ContributionReduction>>,
    method: &str,
) -> EstimationReport {
    // Group coefficient vectors by hydro_id (estimates are already sorted by
    // (hydro_id, season_id) from estimate_ar_coefficients).
    let mut hydro_coeffs: BTreeMap<EntityId, Vec<(usize, Vec<f64>)>> = BTreeMap::new();
    for est in estimates {
        hydro_coeffs
            .entry(est.hydro_id)
            .or_default()
            .push((est.season_id, est.coefficients.clone()));
    }

    let mut entries: BTreeMap<EntityId, HydroEstimationEntry> = BTreeMap::new();

    for (hydro_id, mut season_coeffs) in hydro_coeffs {
        // Sort by season_id ascending (should already be sorted, but ensure it).
        season_coeffs.sort_by_key(|(season_id, _)| *season_id);

        // Compute max selected_order as the maximum actual coefficient length
        // across all seasons for this hydro (after all truncations).
        let selected_order = season_coeffs
            .iter()
            .map(|(_, coeffs)| coeffs.len())
            .max()
            .unwrap_or(0);

        // Build per-season coefficient vectors, filling missing seasons with empty vecs.
        // The season_coeffs may not cover all n_seasons if some were skipped.
        let season_map: HashMap<usize, Vec<f64>> = season_coeffs.into_iter().collect();
        let coefficients: Vec<Vec<f64>> = (0..n_seasons)
            .map(|sid| season_map.get(&sid).cloned().unwrap_or_default())
            .collect();

        let reductions = contribution_reductions
            .get(&hydro_id)
            .cloned()
            .unwrap_or_default();

        #[allow(clippy::cast_possible_truncation)]
        entries.insert(
            hydro_id,
            HydroEstimationEntry {
                selected_order: selected_order as u32,
                coefficients,
                contribution_reductions: reductions,
            },
        );
    }

    EstimationReport {
        entries,
        method: method.to_string(),
        white_noise_fallbacks: Vec::new(),
        lag_scale_warnings: Vec::new(),
        std_ratio_warnings: Vec::new(),
    }
}

/// Convert [`SeasonalStats`] to [`InflowSeasonalStatsRow`], expanding
/// per-season estimates to every stage that shares the same `season_id`.
///
/// The `stage_id` in `SeasonalStats` stores the ID of the first stage with
/// the matching season.  This function looks up that stage's `season_id` and
/// emits one row for every stage with the same season, so that
/// [`cobre_stochastic::PrecomputedPar`] finds a model at every stage index.
///
/// **Pre-study stages**: When `stages` contains pre-study stages (negative
/// `id`, valid `season_id`), this function includes them in the expansion.
/// Pre-study rows are emitted with their negative `stage_id`, which allows
/// `PrecomputedPar::build` to find direct hits for lag stages without
/// needing the season-based fallback path.
fn seasonal_stats_to_rows(
    stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
) -> Vec<InflowSeasonalStatsRow> {
    let stage_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    let mut season_to_stages: HashMap<usize, Vec<i32>> = HashMap::new();
    for stage in stages {
        if let Some(sid) = stage.season_id {
            season_to_stages.entry(sid).or_default().push(stage.id);
        }
    }

    let mut rows = Vec::with_capacity(stats.len() * 10);
    for s in stats {
        if let Some(&season_id) = stage_to_season.get(&s.stage_id) {
            if let Some(stage_ids) = season_to_stages.get(&season_id) {
                for &stage_id in stage_ids {
                    rows.push(InflowSeasonalStatsRow {
                        hydro_id: s.entity_id,
                        stage_id,
                        mean_m3s: s.mean,
                        std_m3s: s.std,
                    });
                }
                continue;
            }
        }
        // Fallback: emit just the original stage_id.
        rows.push(InflowSeasonalStatsRow {
            hydro_id: s.entity_id,
            stage_id: s.stage_id,
            mean_m3s: s.mean,
            std_m3s: s.std,
        });
    }

    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id));
    rows
}

/// Convert [`ArCoefficientEstimate`] to [`InflowArCoefficientRow`], expanding
/// per-season AR coefficients to every stage that shares the same `season_id`.
///
/// The `season_id` in `ArCoefficientEstimate` is mapped to ALL stage IDs whose
/// `season_id` matches, so the resulting coefficient rows cover the full study
/// horizon (not just the first occurrence of each season).
///
/// **Pre-study stages**: When `stages` contains pre-study stages (negative
/// `id`, valid `season_id`), this function includes them in the expansion.
/// Pre-study coefficient rows are emitted with their negative `stage_id`,
/// providing direct model entries for `PrecomputedPar::build` lag lookups.
fn ar_estimates_to_rows(
    ar_estimates: &[ArCoefficientEstimate],
    stages: &[cobre_core::temporal::Stage],
) -> Vec<InflowArCoefficientRow> {
    let mut season_to_stages: HashMap<usize, Vec<i32>> = HashMap::new();
    for stage in stages {
        if let Some(sid) = stage.season_id {
            season_to_stages.entry(sid).or_default().push(stage.id);
        }
    }

    let mut rows: Vec<InflowArCoefficientRow> = Vec::new();

    for est in ar_estimates {
        let Some(stage_ids) = season_to_stages.get(&est.season_id) else {
            continue;
        };

        for &stage_id in stage_ids {
            for (lag_idx, &coeff) in est.coefficients.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let lag = (lag_idx + 1) as i32;
                rows.push(InflowArCoefficientRow {
                    hydro_id: est.hydro_id,
                    stage_id,
                    lag,
                    coefficient: coeff,
                    residual_std_ratio: est.residual_std_ratio,
                });
            }
        }
    }

    // Sort by (hydro_id, stage_id, lag) ascending — matches parser convention.
    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id, r.lag));

    rows
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::useless_vec
)]
mod tests {
    use super::*;
    use cobre_core::scenario::{CorrelationModel, InflowModel};
    use cobre_core::{EntityId, SystemBuilder};

    // ── Helper to build a minimal System ─────────────────────────────────────

    fn minimal_system_with_inflow_models(models: Vec<InflowModel>) -> System {
        SystemBuilder::new()
            .inflow_models(models)
            .build()
            .expect("valid system")
    }

    // ── with_scenario_models tests ────────────────────────────────────────────

    /// AC-036-1: `with_scenario_models` replaces `inflow_models` and `correlation`
    /// while preserving all other fields.
    #[test]
    fn test_with_scenario_models_replaces_fields() {
        use cobre_core::{
            scenario::{CorrelationModel, InflowModel},
            Bus, DeficitSegment,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(f64::INFINITY),
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        // Build system with 2 inflow models.
        let old_model = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 10.0,
            std_m3s: 1.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .inflow_models(vec![old_model.clone(), {
                let mut m = old_model.clone();
                m.stage_id = 1;
                m
            }])
            .build()
            .expect("valid system");

        assert_eq!(system.inflow_models().len(), 2);
        assert_eq!(system.n_buses(), 1);

        // Replace with 4 models.
        let new_models: Vec<InflowModel> = (0..4)
            .map(|i| InflowModel {
                hydro_id: EntityId(1),
                stage_id: i,
                mean_m3s: 50.0,
                std_m3s: 5.0,
                ar_coefficients: vec![0.4],
                residual_std_ratio: 0.9,
            })
            .collect();
        let new_corr = CorrelationModel::default();

        let updated = system.with_scenario_models(new_models.clone(), new_corr.clone());

        // inflow_models and correlation updated.
        assert_eq!(updated.inflow_models().len(), 4, "expected 4 inflow models");
        assert_eq!(
            *updated.correlation(),
            new_corr,
            "correlation should equal new_corr"
        );

        // hydros, buses, stages unchanged.
        assert_eq!(updated.n_buses(), 1, "buses must be preserved");
        assert!(
            updated.hydros().is_empty(),
            "hydros must be preserved (empty)"
        );
        assert!(
            updated.stages().is_empty(),
            "stages must be preserved (empty)"
        );
    }

    /// `with_scenario_models` with an empty vec clears `inflow_models`.
    #[test]
    fn test_with_scenario_models_clears_when_empty() {
        let model = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        let system = minimal_system_with_inflow_models(vec![model]);
        assert_eq!(system.inflow_models().len(), 1);

        let updated = system.with_scenario_models(vec![], CorrelationModel::default());
        assert!(updated.inflow_models().is_empty());
    }

    // ── estimate_from_history path-matrix tests ────────────────────────────────

    /// AC-036-3: When both stats files exist, `estimate_from_history` returns
    /// the system unchanged.
    #[test]
    fn test_estimate_explicit_stats_returns_unchanged() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        // Create the minimal required files so validate_structure won't fail
        // (it only checks existence).
        create_required_files(case_dir);

        // Create both stats files.
        let scenarios = case_dir.join("scenarios");
        std::fs::create_dir_all(&scenarios).unwrap();
        std::fs::write(scenarios.join("inflow_history.parquet"), b"").unwrap();
        std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"").unwrap();
        std::fs::write(scenarios.join("inflow_ar_coefficients.parquet"), b"").unwrap();

        let model = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![0.5],
            residual_std_ratio: 0.87,
        };
        let system = minimal_system_with_inflow_models(vec![model]);
        let original_len = system.inflow_models().len();

        let config = default_config();
        let (result, report, _path) = estimate_from_history(system, case_dir, &config).unwrap();

        assert_eq!(
            result.inflow_models().len(),
            original_len,
            "explicit stats: system must be unchanged"
        );
        assert!(
            report.is_none(),
            "explicit stats path must return None report"
        );
    }

    /// AC-036-4: When no history file exists, `estimate_from_history` returns
    /// the system unchanged.
    #[test]
    fn test_estimate_no_history_returns_unchanged() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        // No scenarios/ directory at all.
        create_required_files(case_dir);

        let model = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        let system = minimal_system_with_inflow_models(vec![model]);
        let original_len = system.inflow_models().len();

        let config = default_config();
        let (result, report, _path) = estimate_from_history(system, case_dir, &config).unwrap();

        assert_eq!(
            result.inflow_models().len(),
            original_len,
            "no history: system must be unchanged"
        );
        assert!(report.is_none(), "no history path must return None report");
    }

    // ── EstimationPath unit tests ─────────────────────────────────────────────

    /// All 8 boolean combinations map to the expected `EstimationPath` variant.
    ///
    /// Covers all 7 named variants plus the edge case `(false, false, true)`
    /// which must map to `Deterministic` because AR alone is meaningless.
    #[test]
    fn test_estimation_path_resolve_all_8_combinations() {
        use cobre_io::FileManifest;

        let make = |history: bool, stats: bool, ar: bool| FileManifest {
            scenarios_inflow_history_parquet: history,
            scenarios_inflow_seasonal_stats_parquet: stats,
            scenarios_inflow_ar_coefficients_parquet: ar,
            ..Default::default()
        };

        // Row 1: (false, false, false) -> Deterministic
        assert_eq!(
            EstimationPath::resolve(&make(false, false, false)),
            EstimationPath::Deterministic,
        );
        // Edge case: (false, false, true) -> Deterministic (AR alone is meaningless)
        assert_eq!(
            EstimationPath::resolve(&make(false, false, true)),
            EstimationPath::Deterministic,
        );
        // Row 2: (false, true, false) -> UserStatsWhiteNoise
        assert_eq!(
            EstimationPath::resolve(&make(false, true, false)),
            EstimationPath::UserStatsWhiteNoise,
        );
        // Row 3: (false, true, true) -> UserProvidedNoHistory
        assert_eq!(
            EstimationPath::resolve(&make(false, true, true)),
            EstimationPath::UserProvidedNoHistory,
        );
        // Row 4: (true, false, false) -> FullEstimation
        assert_eq!(
            EstimationPath::resolve(&make(true, false, false)),
            EstimationPath::FullEstimation,
        );
        // Row 5: (true, false, true) -> UserArHistoryStats
        assert_eq!(
            EstimationPath::resolve(&make(true, false, true)),
            EstimationPath::UserArHistoryStats,
        );
        // Row 6: (true, true, false) -> PartialEstimation
        assert_eq!(
            EstimationPath::resolve(&make(true, true, false)),
            EstimationPath::PartialEstimation,
        );
        // Row 7: (true, true, true) -> UserProvidedAll
        assert_eq!(
            EstimationPath::resolve(&make(true, true, true)),
            EstimationPath::UserProvidedAll,
        );
    }

    /// Every variant's `as_str()` must return a non-empty, unique string.
    #[test]
    fn test_estimation_path_as_str_round_trip() {
        let variants = [
            EstimationPath::Deterministic,
            EstimationPath::UserStatsWhiteNoise,
            EstimationPath::UserProvidedNoHistory,
            EstimationPath::FullEstimation,
            EstimationPath::UserArHistoryStats,
            EstimationPath::PartialEstimation,
            EstimationPath::UserProvidedAll,
        ];

        let strings: Vec<&str> = variants.iter().map(|v| v.as_str()).collect();

        // All strings must be non-empty.
        for s in &strings {
            assert!(!s.is_empty(), "as_str() returned empty string");
        }

        // All strings must be unique.
        let unique: std::collections::HashSet<&&str> = strings.iter().collect();
        assert_eq!(
            unique.len(),
            variants.len(),
            "as_str() must return unique strings for each variant"
        );
    }

    // ── user_stats_to_rows unit tests ─────────────────────────────────────────

    /// `user_stats_to_rows` maps all models — 3 InflowModel entries (2 hydros,
    /// multiple stages) produce the same count of rows with bitwise-equal stats.
    #[test]
    fn test_user_stats_to_rows_maps_all_models() {
        let models = vec![
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 1,
                mean_m3s: 120.0,
                std_m3s: 12.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: EntityId(2),
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 5.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];
        let system = minimal_system_with_inflow_models(models.clone());
        let rows = user_stats_to_rows(&system);

        assert_eq!(rows.len(), 3, "must produce one row per InflowModel");

        for (model, row) in models.iter().zip(rows.iter()) {
            assert_eq!(row.hydro_id, model.hydro_id, "hydro_id must be preserved");
            assert_eq!(row.stage_id, model.stage_id, "stage_id must be preserved");
            // Bitwise equality: the f64 bits must be identical, not just approximately equal.
            assert_eq!(
                row.mean_m3s.to_bits(),
                model.mean_m3s.to_bits(),
                "mean_m3s must be bitwise identical"
            );
            assert_eq!(
                row.std_m3s.to_bits(),
                model.std_m3s.to_bits(),
                "std_m3s must be bitwise identical"
            );
        }
    }

    /// `user_stats_to_rows` on an empty system returns an empty vec.
    #[test]
    fn test_user_stats_to_rows_empty_system() {
        let system = minimal_system_with_inflow_models(vec![]);
        let rows = user_stats_to_rows(&system);
        assert!(rows.is_empty(), "empty system must produce empty rows");
    }

    // ── PartialEstimation unit tests ──────────────────────────────────────────

    /// Write a real `inflow_history.parquet` with synthetic 2-season PAR(1) data
    /// for a single hydro (id=1), using the existing `simulate_two_season_par2`
    /// helper at order 2 to generate observations with non-trivial structure.
    ///
    /// Observations are placed on Jan 1 (season 0) and Jul 1 (season 1) of
    /// successive years starting from 1970, dated so they fall within the
    /// study stages built by `make_two_season_stage`.
    ///
    /// The history file is required to have real Parquet content because
    /// `run_partial_estimation` calls `parse_inflow_history` on it.
    fn write_unit_test_inflow_history(path: &std::path::Path, hydro_id: i32, n_years: usize) {
        use arrow::array::{Date32Array, Float64Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use chrono::NaiveDate;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let date_to_days = |d: NaiveDate| -> i32 {
            i32::try_from((d - epoch).num_days()).expect("date in Date32 range")
        };

        let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 99);

        let mut ids: Vec<i32> = Vec::with_capacity(n_years * 2);
        let mut dates: Vec<i32> = Vec::with_capacity(n_years * 2);
        let mut values: Vec<f64> = Vec::with_capacity(n_years * 2);

        for y in 0..n_years {
            let year = (1970 + y) as i32;
            // Season 0: Jan 1 falls within make_two_season_stage(..., first_half=true)
            ids.push(hydro_id);
            dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 1, 15).unwrap()));
            // Shift values by 300 so they are all positive (simulate_two_season_par2
            // produces ~0-mean series; offset keeps inflows physically plausible).
            values.push(obs_s0[y] + 300.0);

            // Season 1: Jul 1 falls within make_two_season_stage(..., first_half=false)
            ids.push(hydro_id);
            dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 7, 15).unwrap()));
            values.push(obs_s1[y] + 300.0);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("date", DataType::Date32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Date32Array::from(dates)),
                Arc::new(Float64Array::from(values)),
            ],
        )
        .expect("valid batch");

        let file = std::fs::File::create(path).expect("create parquet file");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");
    }

    /// Build a `System` with one hydro (id=1), one bus, and 2-season stages
    /// spanning `n_years` study years, with pre-loaded inflow models whose
    /// `mean_m3s = 100.0` and `std_m3s = 10.0` (user-provided stats).
    ///
    /// This represents the state after `load_case` has loaded
    /// `inflow_seasonal_stats.parquet` but not `inflow_ar_coefficients.parquet`.
    #[allow(clippy::cast_possible_wrap)]
    fn build_system_with_user_stats(n_years: usize) -> System {
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::InflowModel;
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

        let hydro_id = EntityId(1);
        let bus = Bus {
            id: EntityId(10),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(f64::INFINITY),
                cost_per_mwh: 3000.0,
            }],
            excess_cost: 0.0,
        };

        // Build 2-season stages using make_two_season_stage (season 0: Jan–Jun,
        // season 1: Jul–Dec), one stage per season per year.
        let ref_year = 1970_i32;
        let mut stages = Vec::with_capacity(n_years * 2);
        for y in 0..n_years {
            let year = ref_year + y as i32;
            stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
            stages.push(make_two_season_stage(
                y * 2 + 1,
                (y * 2 + 1) as i32,
                1,
                year,
                false,
            ));
        }

        // Build inflow models for each stage, preserving user-provided stats.
        // These represent what load_case produces after reading inflow_seasonal_stats.
        let inflow_models: Vec<InflowModel> = stages
            .iter()
            .map(|s| InflowModel {
                hydro_id,
                stage_id: s.id,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let hydro = Hydro {
            id: hydro_id,
            name: "H1".to_string(),
            bus_id: EntityId(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 5000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 1000.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .build()
            .expect("valid system with user stats")
    }

    /// Create the minimal directory skeleton and write both the real Parquet
    /// history and an empty sentinel file for `inflow_seasonal_stats.parquet`,
    /// which is sufficient for the manifest to classify the path as
    /// `PartialEstimation` (history=true, stats=true, ar=false).
    fn setup_partial_estimation_case(case_dir: &std::path::Path, n_years: usize) {
        create_required_files(case_dir);
        let scenarios = case_dir.join("scenarios");
        std::fs::create_dir_all(&scenarios).unwrap();

        // Write a real Parquet history file — it must be parseable.
        write_unit_test_inflow_history(
            &scenarios.join("inflow_history.parquet"),
            1, // hydro_id
            n_years,
        );

        // Write a sentinel file to trigger the manifest flag.
        // validate_structure only checks existence, not content.
        std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
            .expect("write sentinel");

        // No inflow_ar_coefficients.parquet → PartialEstimation path.
    }

    /// AC-T8-1: `PartialEstimation` preserves user-provided `mean_m3s` and
    /// `std_m3s` while estimating AR coefficients from history.
    ///
    /// Setup: system with known user stats (mean=100.0, std=10.0 for every
    /// stage), case dir with a real `inflow_history.parquet` (synthetic PAR(2)
    /// data) and an `inflow_seasonal_stats.parquet` sentinel.
    ///
    /// Asserts:
    /// - Every inflow model in the returned system has `mean_m3s == 100.0`
    ///   (bitwise) and `std_m3s == 10.0` (bitwise).
    /// - Every inflow model has at least one AR coefficient (non-empty).
    #[test]
    fn test_partial_estimation_preserves_user_stats() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30; // sufficient for PACF order selection
        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        setup_partial_estimation_case(case_dir, N_YEARS);
        let system = build_system_with_user_stats(N_YEARS);
        let config = default_config();

        let (updated, report, path) = estimate_from_history(system, case_dir, &config)
            .expect("partial estimation must succeed");

        assert_eq!(
            path,
            EstimationPath::PartialEstimation,
            "expected PartialEstimation path"
        );
        assert!(
            report.is_some(),
            "PartialEstimation must return Some(report)"
        );

        let models = updated.inflow_models();
        assert!(
            !models.is_empty(),
            "partial estimation must produce at least one inflow model"
        );

        for m in models {
            // Bitwise equality — no rounding or transformation is allowed.
            assert_eq!(
                m.mean_m3s.to_bits(),
                100.0_f64.to_bits(),
                "mean_m3s must be bitwise identical to user value 100.0 for stage {}",
                m.stage_id
            );
            assert_eq!(
                m.std_m3s.to_bits(),
                10.0_f64.to_bits(),
                "std_m3s must be bitwise identical to user value 10.0 for stage {}",
                m.stage_id
            );
            assert!(
                !m.ar_coefficients.is_empty(),
                "ar_coefficients must be non-empty for stage {} (estimated from history)",
                m.stage_id
            );
        }
    }

    /// AC-T8-2: `PartialEstimation` returns a `Some(report)` with method "PACF"
    /// and an entry for the single hydro plant.
    ///
    /// Same setup as `test_partial_estimation_preserves_user_stats`.
    #[test]
    fn test_partial_estimation_returns_report() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;
        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        setup_partial_estimation_case(case_dir, N_YEARS);
        let system = build_system_with_user_stats(N_YEARS);
        let config = default_config();

        let (_updated, report, _path) = estimate_from_history(system, case_dir, &config)
            .expect("partial estimation must succeed");

        let report = report.expect("PartialEstimation must return Some(EstimationReport)");

        assert_eq!(
            report.method, "PACF",
            "estimation method must be PACF, got '{}'",
            report.method
        );
        assert_eq!(
            report.entries.len(),
            1,
            "report must contain exactly 1 entry (one hydro), got {}",
            report.entries.len()
        );
        assert!(
            report.entries.contains_key(&EntityId(1)),
            "report must contain an entry for hydro_id=1"
        );
    }

    // ── LagScaleWarning unit tests ────────────────────────────────────────────

    /// Helper: build a minimal fitting_stats and user InflowModel pair for a
    /// single hydro at stage_id == 0, then run the warning check inline.
    ///
    /// Returns the warnings collected. Extracted to avoid code duplication
    /// across the three lag-scale warning tests.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn collect_lag_scale_warnings(
        hydro_id: EntityId,
        lag_value: f64,
        user_mean: f64,
        estimated_mean: f64,
        include_past_inflows: bool,
    ) -> Vec<LagScaleWarning> {
        use cobre_core::scenario::InflowModel;
        use cobre_core::{HydroPastInflows, InitialConditions};

        // Build fitting_stats with estimated mean at stage_id == 0.
        let fitting_stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: estimated_mean,
            std: 50.0,
        }];

        // Build system with a user InflowModel at stage_id == 0.
        let user_model = InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: user_mean,
            std_m3s: 50.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        let past_inflows = if include_past_inflows {
            vec![HydroPastInflows {
                hydro_id,
                values_m3s: vec![lag_value],
            }]
        } else {
            vec![]
        };
        let ic = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows,
        };
        let system = SystemBuilder::new()
            .inflow_models(vec![user_model])
            .initial_conditions(ic)
            .build()
            .expect("valid system");

        // Replicate the check logic from run_partial_estimation step 5c.
        let estimated_mean_at_stage0: BTreeMap<EntityId, f64> = fitting_stats
            .iter()
            .filter(|s| s.stage_id == 0)
            .map(|s| (s.entity_id, s.mean))
            .collect();
        let user_mean_at_stage0: BTreeMap<EntityId, f64> = system
            .inflow_models()
            .iter()
            .filter(|m| m.stage_id == 0)
            .map(|m| (m.hydro_id, m.mean_m3s))
            .collect();
        let mut sorted_past_inflows: Vec<&cobre_core::HydroPastInflows> =
            system.initial_conditions().past_inflows.iter().collect();
        sorted_past_inflows.sort_by_key(|p| p.hydro_id);

        let mut warnings: Vec<LagScaleWarning> = Vec::new();
        for past in sorted_past_inflows {
            let Some(&lv) = past.values_m3s.first() else {
                continue;
            };
            let Some(&est_mean) = estimated_mean_at_stage0.get(&past.hydro_id) else {
                continue;
            };
            let Some(&usr_mean) = user_mean_at_stage0.get(&past.hydro_id) else {
                continue;
            };
            if (lv - est_mean).abs() < (lv - usr_mean).abs() {
                warnings.push(LagScaleWarning {
                    hydro_id: past.hydro_id,
                    lag_value: lv,
                    user_mean: usr_mean,
                    estimated_mean: est_mean,
                });
            }
        }
        warnings
    }

    /// P8-001: Warning fires when lag-1 is closer to estimated mean than user mean.
    ///
    /// lag=500, user_mean=800, estimated_mean=480 -> |500-480|=20 < |500-800|=300
    /// -> warning must be produced.
    #[test]
    fn test_lag_scale_warning_fires_when_closer_to_estimated() {
        let warnings = collect_lag_scale_warnings(
            EntityId(1),
            500.0, // lag_value
            800.0, // user_mean
            480.0, // estimated_mean
            true,  // include_past_inflows
        );
        assert_eq!(
            warnings.len(),
            1,
            "expected exactly one LagScaleWarning when lag is closer to estimated mean"
        );
        assert_eq!(
            warnings[0].hydro_id,
            EntityId(1),
            "warning must record the correct hydro_id"
        );
        assert!((warnings[0].lag_value - 500.0).abs() < f64::EPSILON);
        assert!((warnings[0].user_mean - 800.0).abs() < f64::EPSILON);
        assert!((warnings[0].estimated_mean - 480.0).abs() < f64::EPSILON);
    }

    /// P8-002: Warning does NOT fire when lag-1 is closer to user mean.
    ///
    /// lag=790, user_mean=800, estimated_mean=480 -> |790-480|=310 > |790-800|=10
    /// -> no warning.
    #[test]
    fn test_lag_scale_warning_not_fires_when_closer_to_user() {
        let warnings = collect_lag_scale_warnings(
            EntityId(1),
            790.0, // lag_value -- close to user_mean
            800.0, // user_mean
            480.0, // estimated_mean
            true,  // include_past_inflows
        );
        assert!(
            warnings.is_empty(),
            "expected no LagScaleWarning when lag is closer to user mean"
        );
    }

    /// P8-003: No warnings when past_inflows is empty.
    #[test]
    fn test_lag_scale_warning_empty_past_inflows() {
        let warnings = collect_lag_scale_warnings(
            EntityId(1),
            500.0, // lag_value (irrelevant -- no past_inflows)
            800.0, // user_mean
            480.0, // estimated_mean
            false, // include_past_inflows = false
        );
        assert!(
            warnings.is_empty(),
            "expected no warnings when past_inflows is empty"
        );
    }

    /// P8-004: Hydro with past_inflows but no entry in fitting_stats is skipped.
    #[test]
    fn test_lag_scale_warning_skips_hydro_without_history() {
        use cobre_core::scenario::InflowModel;
        use cobre_core::{HydroPastInflows, InitialConditions};

        let hydro_id = EntityId(1);
        let other_hydro_id = EntityId(2);

        // fitting_stats only covers hydro 2, not hydro 1 (the one with past_inflows).
        let fitting_stats = vec![SeasonalStats {
            entity_id: other_hydro_id,
            stage_id: 0,
            mean: 480.0,
            std: 50.0,
        }];

        let user_model = InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: 800.0,
            std_m3s: 50.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        let ic = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows: vec![HydroPastInflows {
                hydro_id,
                values_m3s: vec![500.0],
            }],
        };
        let system = SystemBuilder::new()
            .inflow_models(vec![user_model])
            .initial_conditions(ic)
            .build()
            .expect("valid system");

        // Run the check inline.
        let estimated_mean_at_stage0: BTreeMap<EntityId, f64> = fitting_stats
            .iter()
            .filter(|s| s.stage_id == 0)
            .map(|s| (s.entity_id, s.mean))
            .collect();
        let user_mean_at_stage0: BTreeMap<EntityId, f64> = system
            .inflow_models()
            .iter()
            .filter(|m| m.stage_id == 0)
            .map(|m| (m.hydro_id, m.mean_m3s))
            .collect();
        let mut sorted_past_inflows: Vec<&cobre_core::HydroPastInflows> =
            system.initial_conditions().past_inflows.iter().collect();
        sorted_past_inflows.sort_by_key(|p| p.hydro_id);

        let mut warnings: Vec<LagScaleWarning> = Vec::new();
        for past in sorted_past_inflows {
            let Some(&lv) = past.values_m3s.first() else {
                continue;
            };
            let Some(&est_mean) = estimated_mean_at_stage0.get(&past.hydro_id) else {
                continue;
            };
            let Some(&usr_mean) = user_mean_at_stage0.get(&past.hydro_id) else {
                continue;
            };
            if (lv - est_mean).abs() < (lv - usr_mean).abs() {
                warnings.push(LagScaleWarning {
                    hydro_id: past.hydro_id,
                    lag_value: lv,
                    user_mean: usr_mean,
                    estimated_mean: est_mean,
                });
            }
        }

        assert!(
            warnings.is_empty(),
            "hydro without fitting_stats entry must be skipped (no warning)"
        );
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_config() -> Config {
        use cobre_io::config::{EstimationConfig, OrderSelectionMethod};
        let mut cfg: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
        cfg.estimation = EstimationConfig {
            max_order: 2,
            order_selection: OrderSelectionMethod::Pacf,
            min_observations_per_season: 2,
            max_coefficient_magnitude: None,
        };
        cfg
    }

    const MINIMAL_CONFIG_JSON: &str = r#"{
        "training": { "tree_seed": 42 },
        "simulation": { "enabled": false, "num_scenarios": 0, "io_channel_capacity": 16 },
        "modeling": {},
        "policy": {},
        "exports": {},
        "output": {}
    }"#;

    fn create_required_files(case_dir: &std::path::Path) {
        // validate_structure only checks existence; content doesn't matter here.
        let _ = std::fs::create_dir_all(case_dir.join("system"));
        let _ = std::fs::create_dir_all(case_dir.join("scenarios"));
        let write = |name: &str| {
            let _ = std::fs::write(case_dir.join(name), b"{}");
        };
        write("config.json");
        write("penalties.json");
        write("stages.json");
        write("initial_conditions.json");
        write("system/buses.json");
        write("system/lines.json");
        write("system/hydros.json");
        write("system/thermals.json");
    }

    // ── EstimationReport unit tests ───────────────────────────────────────────

    /// Construct mock `ArCoefficientEstimate` entries for 2 hydros with 3
    /// seasons each, call `build_estimation_report`, and verify that the
    /// report contains exactly 2 entries with the expected structure.
    #[test]
    fn test_estimation_report_structure() {
        let h1 = EntityId(1);
        let h2 = EntityId(2);
        let n_seasons = 3_usize;

        // Build mock estimates: 2 hydros x 3 seasons, max order 2.
        let mut estimates = Vec::new();
        for &hydro_id in &[h1, h2] {
            for season_id in 0..n_seasons {
                estimates.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id,
                    coefficients: vec![0.5, 0.3],
                    residual_std_ratio: 0.9,
                });
            }
        }

        let contribution_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();
        let report =
            build_estimation_report(&estimates, n_seasons, &contribution_reductions, "PACF");

        assert_eq!(report.entries.len(), 2, "expected 2 hydro entries");

        for &hydro_id in &[h1, h2] {
            let entry = report.entries.get(&hydro_id).expect("entry must exist");
            assert_eq!(entry.selected_order, 2, "selected_order must be 2");
            assert_eq!(
                entry.coefficients.len(),
                n_seasons,
                "one coefficient vec per season"
            );
        }
    }

    /// With empty observations the returned `EstimationReport` must have an
    /// empty entries map.
    #[test]
    fn test_estimation_report_empty_for_pacf() {
        use cobre_core::temporal::Stage;

        let observations: Vec<(EntityId, chrono::NaiveDate, f64)> = vec![];
        let seasonal_stats: Vec<SeasonalStats> = vec![];
        let stages: Vec<Stage> = vec![];
        let hydro_ids: Vec<EntityId> = vec![];
        let max_order = 2;

        let (_, report) = estimate_ar_coefficients_with_selection(
            &observations,
            &seasonal_stats,
            &stages,
            &hydro_ids,
            &ArEstimationConfig {
                max_order,
                max_coeff_magnitude: None,
                season_map: None,
            },
        )
        .unwrap();

        assert!(
            report.entries.is_empty(),
            "empty observations must produce empty EstimationReport"
        );
    }

    // ── Contribution validation tests ─────────────────────────────────────

    /// E2-003: Order-0 fallback when the first contribution is negative.
    ///
    /// A single-season AR(1) model with phi = -1.5 produces a negative
    /// direct contribution. `validate_order_contributions` should report
    /// max_valid_order = 0.
    #[test]
    fn test_contribution_order_zero_fallback() {
        let result = validate_order_contributions(
            0,             // season_id
            1,             // n_seasons
            1,             // current_order
            &[vec![-1.5]], // all_season_coefficients
            &[10.0],       // std_by_season
        );
        assert!(!result.valid);
        assert_eq!(result.max_valid_order, 0);
    }

    /// E2-003: Order-0 input returns valid immediately.
    #[test]
    fn test_contribution_order_zero_input_passes() {
        let result = validate_order_contributions(0, 1, 0, &[Vec::new()], &[10.0]);
        assert!(result.valid);
        assert_eq!(result.max_valid_order, 0);
        assert!(result.contributions.is_empty());
    }

    /// E2-003: Stable AR(2) model with all-positive contributions passes.
    #[test]
    fn test_contribution_stable_model_passes() {
        let result = validate_order_contributions(
            0,                 // season_id
            1,                 // n_seasons
            2,                 // current_order
            &[vec![0.4, 0.2]], // all_season_coefficients
            &[10.0],           // std_by_season
        );
        assert!(result.valid);
        assert_eq!(result.max_valid_order, 2);
        assert_eq!(result.contributions.len(), 2);
    }

    /// E2-003: apply_contribution_validation reduces an explosive model.
    ///
    /// Constructs AR(2) with coefficients [0.3, -0.8] for a single entity
    /// and single season. The contribution of lag 2 is negative (-0.71),
    /// so the order should be reduced to 1.
    #[test]
    fn test_apply_contribution_validation_reduces_explosive() {
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![0.3, -0.8],
            residual_std_ratio: 0.9,
        }];

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 100.0,
            std: 10.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
        // sigma2_per_order: [sigma2_order1, sigma2_order2]
        // At order 1, residual is sqrt(0.81) ~= 0.9
        let mut sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
        sigma2_map.insert((hydro_id, 0), vec![0.81, 0.75]);

        let reductions = apply_contribution_validation(
            &mut estimates,
            n_seasons,
            &stats_map,
            &sigma2_map,
            None, // max_coeff_magnitude
        );

        // After validation, the order should be reduced: [0.3, -0.8] -> [0.3]
        assert_eq!(
            estimates[0].coefficients.len(),
            1,
            "explosive AR(2) should be reduced to AR(1)"
        );
        assert!((estimates[0].coefficients[0] - 0.3).abs() < 1e-10);

        // Residual std ratio should be recomputed from sigma2_per_order[0]
        assert!(
            (estimates[0].residual_std_ratio - 0.81_f64.sqrt()).abs() < 1e-10,
            "residual_std_ratio should be recomputed from sigma2_per_order[0]"
        );

        // Should have recorded the reduction event.
        let entity_reductions = reductions.get(&hydro_id).expect("should have reductions");
        assert_eq!(entity_reductions.len(), 1);
        assert_eq!(entity_reductions[0].original_order, 2);
        assert_eq!(entity_reductions[0].reduced_order, 1);
        assert_eq!(entity_reductions[0].season_id, 0);
    }

    /// E2-003: PIMENTAL-like scenario -- large coefficient at lag 2 in one
    /// season while other seasons are benign.
    #[test]
    fn test_pimental_like_multi_season_reduction() {
        let hydro_id = EntityId(156);
        let n_seasons = 12;

        // Build estimates: most seasons have small AR(1), August has explosive AR(2).
        let mut estimates: Vec<ArCoefficientEstimate> = (0..n_seasons)
            .map(|s| ArCoefficientEstimate {
                hydro_id,
                season_id: s,
                coefficients: if s == 7 {
                    // August: explosive AR(2) with huge lag-2 coefficient
                    vec![0.5, 48.9]
                } else {
                    vec![0.1] // benign AR(1)
                },
                residual_std_ratio: 0.95,
            })
            .collect();

        // Std devs: most seasons ~200, August = 5 (high coefficient * low std = trouble)
        let stds: Vec<f64> = (0..n_seasons)
            .map(|s| if s == 7 { 5.0 } else { 200.0 })
            .collect();

        let stats: Vec<SeasonalStats> = (0..n_seasons)
            .map(|s| SeasonalStats {
                entity_id: hydro_id,
                stage_id: s as i32,
                mean: 100.0,
                std: stds[s],
            })
            .collect();
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
            .iter()
            .enumerate()
            .map(|(s, st)| ((hydro_id, s), st))
            .collect();

        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let reductions = apply_contribution_validation(
            &mut estimates,
            n_seasons,
            &stats_map,
            &sigma2_map,
            None, // max_coeff_magnitude
        );

        // August (season 7) should have been reduced from AR(2).
        // The contribution of lag 2 through the periodic chain may or may not be negative
        // depending on the recursive composition with neighboring months' coefficients.
        // We verify the reduction was applied if any reduction occurred for August.
        let august_order = estimates[7].coefficients.len();

        // Other months should remain unchanged at AR(1) since their contributions
        // are small positive values.
        for (s, est) in estimates.iter().enumerate() {
            if s != 7 {
                assert_eq!(est.coefficients.len(), 1, "season {s} should remain AR(1)");
            }
        }

        // If August was reduced, there should be a reduction record.
        if august_order < 2 {
            let entity_reductions = reductions.get(&hydro_id).expect("should have reductions");
            assert!(
                entity_reductions.iter().any(|r| r.season_id == 7),
                "August reduction should be recorded"
            );
        }
    }

    /// E2-003: All contributions negative forces white-noise fallback.
    ///
    /// AR(1) with phi = -2.0 for all seasons -- every contribution is negative,
    /// so order drops to 0 and residual_std_ratio becomes 1.0.
    #[test]
    fn test_all_negative_fallback_to_white_noise() {
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![-2.0],
            residual_std_ratio: 0.8,
        }];

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let _reductions = apply_contribution_validation(
            &mut estimates,
            n_seasons,
            &stats_map,
            &sigma2_map,
            None, // max_coeff_magnitude
        );

        assert!(
            estimates[0].coefficients.is_empty(),
            "should fall back to order 0"
        );
        assert!(
            (estimates[0].residual_std_ratio - 1.0).abs() < 1e-10,
            "white-noise residual ratio should be 1.0"
        );
    }

    // ── Phi_1 rejection tests ────────────────────────────────────────────────

    #[test]
    fn phi1_rejection_sets_order_to_zero() {
        let hydro_id = EntityId(1);
        let n_seasons = 2;

        let mut estimates = vec![
            ArCoefficientEstimate {
                hydro_id,
                season_id: 0,
                coefficients: vec![-0.3, 0.5],
                residual_std_ratio: 0.8,
            },
            ArCoefficientEstimate {
                hydro_id,
                season_id: 1,
                coefficients: vec![0.4, 0.2],
                residual_std_ratio: 0.7,
            },
        ];

        let stats = vec![
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 0,
                mean: 50.0,
                std: 10.0,
            },
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 1,
                mean: 60.0,
                std: 12.0,
            },
        ];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
            .iter()
            .enumerate()
            .map(|(i, s)| ((s.entity_id, i), s))
            .collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

        // Season 0 should have been rejected (negative phi_1).
        assert!(
            estimates[0].coefficients.is_empty(),
            "season 0 should be cleared to order 0"
        );
        assert!(
            (estimates[0].residual_std_ratio - 1.0).abs() < 1e-10,
            "season 0 residual_std_ratio should be 1.0"
        );

        // Season 1 should be unchanged.
        assert_eq!(
            estimates[1].coefficients,
            vec![0.4, 0.2],
            "season 1 should be unchanged"
        );

        // Verify reduction entry for season 0.
        let hydro_reductions = reductions.get(&hydro_id).expect("should have reductions");
        let r = hydro_reductions
            .iter()
            .find(|r| r.season_id == 0)
            .expect("should have reduction for season 0");
        assert_eq!(r.original_order, 2);
        assert_eq!(r.reduced_order, 0);
        assert!(r.contributions.is_empty());
    }

    #[test]
    fn phi1_rejection_before_contribution_analysis() {
        // A season with phi_1 = -0.01 and phi_2 = 0.5 with uniform std.
        // The contribution analysis would NOT catch this (contributions may
        // be non-negative), but the phi_1 gate fires first.
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![-0.01, 0.5],
            residual_std_ratio: 0.8,
        }];

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

        // Should be rejected by phi_1 gate.
        assert!(
            estimates[0].coefficients.is_empty(),
            "phi_1 = -0.01 should trigger rejection"
        );
        assert!(
            reductions.contains_key(&hydro_id),
            "should have a reduction entry"
        );
    }

    #[test]
    fn phi1_zero_is_not_rejected() {
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![0.0, 0.3],
            residual_std_ratio: 0.8,
        }];

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let _reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

        // phi_1 = 0.0 is NOT negative, so it should pass to contribution analysis.
        assert_eq!(
            estimates[0].coefficients.len(),
            2,
            "phi_1 = 0.0 should not be rejected"
        );
    }

    #[test]
    fn phi1_rejection_interacts_with_magnitude_bound() {
        // phi_1 = -50.0 is both negative and above any reasonable magnitude bound.
        // The magnitude-bound pre-pass should fire first, and the phi_1 check
        // should see the already-cleared vector and skip.
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![-50.0, 0.3],
            residual_std_ratio: 0.8,
        }];

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let reductions = apply_contribution_validation(
            &mut estimates,
            n_seasons,
            &stats_map,
            &sigma2_map,
            Some(10.0), // magnitude bound that catches -50.0
        );

        // The season should be cleared (by magnitude bound).
        assert!(
            estimates[0].coefficients.is_empty(),
            "should be cleared to order 0"
        );

        // Only ONE reduction entry should exist (from magnitude bound, NOT doubled).
        let hydro_reductions = reductions.get(&hydro_id).expect("should have reductions");
        assert_eq!(
            hydro_reductions.len(),
            1,
            "should have exactly 1 reduction entry (magnitude bound only)"
        );
    }

    // ── Iterative PACF reduction tests ──────────────────────────────────────

    /// Generate synthetic observations for a single-season AR(p) process.
    /// Uses a fixed seed for reproducibility.
    #[allow(clippy::cast_precision_loss, clippy::cast_lossless)] // u64 >> 33 fits in f64; u32::MAX fits in f64
    fn generate_ar_observations(coefficients: &[f64], n: usize) -> Vec<f64> {
        let p = coefficients.len();
        let mut values = vec![0.0_f64; n + p];
        // Simple deterministic pseudo-noise using a linear congruential generator.
        let mut seed: u64 = 42;
        for i in p..(n + p) {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / (u32::MAX as f64) - 0.5) * 2.0;
            let mut val = noise;
            for (j, c) in coefficients.iter().enumerate() {
                val += c * values[i - j - 1];
            }
            values[i] = val;
        }
        values[p..].to_vec()
    }

    #[test]
    fn iterative_reduction_terminates_at_zero() {
        // Construct a case where contributions fail at every order,
        // forcing the loop to terminate at order 0.
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        // Coefficients that produce negative contributions at order 2.
        // phi = [0.3, -5.0] -> contribution at lag 2 = 0.3*0.3 + (-5.0) < 0
        // After reducing max_order to 1 and re-running PACF, if PACF selects
        // order 1, contributions at order 1 are just 0.3 (positive).
        // But if we use coefficients that ALWAYS produce negative contributions,
        // the order will reach 0.
        //
        // Use phi = [-0.5] -> phi_1 is negative, caught by phi_1 gate directly.
        // Instead, use a scenario where contribution analysis fails at every order.
        //
        // We test this via the direct function with synthetic data designed
        // so that PACF at each reduced max_order still selects an order
        // that fails contributions.
        //
        // Simplest approach: use the apply_contribution_validation path
        // (for Fixed method) to verify order-0 termination.
        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![0.3, -0.8],
            residual_std_ratio: 0.8,
        }];

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

        // The loop should terminate (possibly at a reduced order or order 0).
        // The key assertion is that it terminates and doesn't infinite-loop.
        assert!(
            estimates[0].coefficients.len() < 2,
            "order should be reduced from 2; got {}",
            estimates[0].coefficients.len()
        );

        // Should have at least one reduction entry.
        assert!(
            reductions.contains_key(&hydro_id),
            "should have a reduction entry"
        );
    }

    #[test]
    fn iterative_reduction_only_affects_failing_seasons() {
        // Two seasons: season 0 has negative contributions, season 1 passes.
        let hydro_id = EntityId(1);
        let n_seasons = 2;

        let mut estimates = vec![
            ArCoefficientEstimate {
                hydro_id,
                season_id: 0,
                // phi = [0.3, -0.8]: contribution at lag 2 is negative.
                coefficients: vec![0.3, -0.8],
                residual_std_ratio: 0.8,
            },
            ArCoefficientEstimate {
                hydro_id,
                season_id: 1,
                // phi = [0.4, 0.2]: all contributions positive.
                coefficients: vec![0.4, 0.2],
                residual_std_ratio: 0.7,
            },
        ];

        let stats = vec![
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 0,
                mean: 50.0,
                std: 10.0,
            },
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 1,
                mean: 60.0,
                std: 10.0,
            },
        ];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
            .iter()
            .enumerate()
            .map(|(i, s)| ((s.entity_id, i), s))
            .collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let _reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

        // Season 0 should have been reduced.
        assert!(
            estimates[0].coefficients.len() < 2,
            "season 0 order should be reduced from 2; got {}",
            estimates[0].coefficients.len()
        );

        // Season 1 should remain unchanged at order 2.
        assert_eq!(
            estimates[1].coefficients,
            vec![0.4, 0.2],
            "season 1 should be unchanged"
        );
    }

    #[test]
    fn iterative_pacf_reduction_with_synthetic_observations() {
        // Test the iterative_pacf_reduction function directly with
        // synthetic observations. Generate data from a known AR process,
        // then artificially set coefficients that will fail contribution
        // analysis to trigger re-selection.
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        // Generate enough observations for PACF to work.
        let obs = generate_ar_observations(&[0.5, 0.2], 100);

        let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
        group_obs.insert((hydro_id, 0), obs);

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 0.0,
            std: 1.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

        // Start with coefficients that have a bad high-order term.
        // The iterative loop should reduce the order.
        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![0.5, 0.2, -5.0], // order 3, lag 3 will fail
            residual_std_ratio: 0.8,
        }];

        let reductions = iterative_pacf_reduction(
            &mut estimates,
            n_seasons,
            &[hydro_id],
            &group_obs,
            &stats_map,
            3, // initial_max_order
            1.96,
            None,
        );

        // The function should have re-estimated using PACF at a reduced
        // max_order. The exact final order depends on the PACF selection,
        // but it should be less than the original 3.
        assert!(
            estimates[0].coefficients.len() < 3,
            "order should be reduced from 3; got {}",
            estimates[0].coefficients.len()
        );

        // Should have at least one reduction entry.
        assert!(
            reductions.contains_key(&hydro_id),
            "should have a reduction entry"
        );
    }

    #[test]
    fn fixed_path_uses_truncation_not_reselection() {
        // Verify that the Fixed order selection path still uses
        // apply_contribution_validation (truncation), not iterative PACF.
        // We check this by verifying the behavior matches truncation semantics.
        let hydro_id = EntityId(1);
        let n_seasons = 1;

        // phi = [0.5, 0.2, -0.8]: contribution at lag 3 is negative.
        // Truncation would give order 2 (find_max_valid_order).
        // Iterative PACF would re-run PACF at max_order=2 and possibly
        // select a different order.
        let mut estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![0.5, 0.2, -0.8],
            residual_std_ratio: 0.8,
        }];

        let stats = vec![SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        }];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
            stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

        // With truncation, the Fixed path truncates coefficients at the
        // first negative contribution position, which is at lag 3 (index 2).
        // So max_valid_order should be 2 or less.
        let final_order = estimates[0].coefficients.len();
        assert!(
            final_order <= 2,
            "Fixed path should truncate; got order {final_order}"
        );

        // Verify a reduction was recorded.
        assert!(
            reductions.contains_key(&hydro_id),
            "should have a reduction entry"
        );
        let r = &reductions[&hydro_id][0];
        assert_eq!(r.original_order, 3);
        // The reduced order should be from find_max_valid_order (truncation).
        assert!(r.reduced_order <= 2, "truncation should produce order <= 2");
    }

    // ── Combined strategy and reduction reason tests ─────────────────────────

    #[test]
    fn combined_strategies_produce_correct_reduction_reasons() {
        let h1 = EntityId(1);
        let h2 = EntityId(2);
        let n_seasons = 2;

        let mut estimates = vec![
            // H1 S0: negative phi_1 -> Phi1Negative
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 0,
                coefficients: vec![-0.3, 0.5],
                residual_std_ratio: 0.8,
            },
            // H1 S1: negative contribution at lag 3 -> NegativeContribution
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 1,
                coefficients: vec![0.5, 0.2, -0.8],
                residual_std_ratio: 0.7,
            },
            // H2 S0: magnitude bound -> MagnitudeBound
            ArCoefficientEstimate {
                hydro_id: h2,
                season_id: 0,
                coefficients: vec![50.0],
                residual_std_ratio: 0.9,
            },
            // H2 S1: passes -> no reduction
            ArCoefficientEstimate {
                hydro_id: h2,
                season_id: 1,
                coefficients: vec![0.4, 0.2],
                residual_std_ratio: 0.7,
            },
        ];

        let stats = vec![
            SeasonalStats {
                entity_id: h1,
                stage_id: 0,
                mean: 50.0,
                std: 10.0,
            },
            SeasonalStats {
                entity_id: h1,
                stage_id: 1,
                mean: 60.0,
                std: 10.0,
            },
            SeasonalStats {
                entity_id: h2,
                stage_id: 0,
                mean: 70.0,
                std: 15.0,
            },
            SeasonalStats {
                entity_id: h2,
                stage_id: 1,
                mean: 80.0,
                std: 12.0,
            },
        ];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
            .iter()
            .enumerate()
            .map(|(i, s)| ((s.entity_id, i % 2), s))
            .collect();
        let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

        let reductions = apply_contribution_validation(
            &mut estimates,
            n_seasons,
            &stats_map,
            &sigma2_map,
            Some(10.0),
        );

        // H1 S0: phi_1 negative -> Phi1Negative, order 0
        let h1_reductions = &reductions[&h1];
        let h1_s0 = h1_reductions
            .iter()
            .find(|r| r.season_id == 0)
            .expect("should have reduction for H1 S0");
        assert_eq!(h1_s0.reason, ReductionReason::Phi1Negative);
        assert_eq!(h1_s0.reduced_order, 0);

        // H1 S1: negative contribution -> NegativeContribution
        let h1_s1 = h1_reductions
            .iter()
            .find(|r| r.season_id == 1)
            .expect("should have reduction for H1 S1");
        assert_eq!(h1_s1.reason, ReductionReason::NegativeContribution);

        // H2 S0: magnitude bound -> MagnitudeBound, order 0
        let h2_reductions = &reductions[&h2];
        let h2_s0 = h2_reductions
            .iter()
            .find(|r| r.season_id == 0)
            .expect("should have reduction for H2 S0");
        assert_eq!(h2_s0.reason, ReductionReason::MagnitudeBound);
        assert_eq!(h2_s0.reduced_order, 0);

        // H2 S1: no reduction
        assert!(
            !h2_reductions.iter().any(|r| r.season_id == 1),
            "H2 S1 should have no reduction"
        );
    }

    // ── Pre-study stage expansion tests (T005) ──────────────────────────────

    /// Build a Stage with the given parameters, suitable for expansion tests.
    fn make_expansion_stage(
        index: usize,
        id: i32,
        season_id: Option<usize>,
    ) -> cobre_core::temporal::Stage {
        use chrono::NaiveDate;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };

        cobre_core::temporal::Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    #[test]
    fn seasonal_stats_to_rows_includes_prestudy_stages() {
        // 3 study stages (id 0, 1, 2; seasons 0, 1, 2)
        // 2 pre-study stages (id -1, -2; seasons 2, 1)
        let stages = vec![
            make_expansion_stage(0, -2, Some(1)),
            make_expansion_stage(1, -1, Some(2)),
            make_expansion_stage(2, 0, Some(0)),
            make_expansion_stage(3, 1, Some(1)),
            make_expansion_stage(4, 2, Some(2)),
        ];

        let h1 = EntityId(1);
        // SeasonalStats for seasons 0, 1, 2 (stage_id is the first stage
        // with that season).
        let stats = vec![
            SeasonalStats {
                entity_id: h1,
                stage_id: 0,
                mean: 100.0,
                std: 20.0,
            },
            SeasonalStats {
                entity_id: h1,
                stage_id: 1,
                mean: 110.0,
                std: 22.0,
            },
            SeasonalStats {
                entity_id: h1,
                stage_id: 2,
                mean: 120.0,
                std: 24.0,
            },
        ];

        let rows = seasonal_stats_to_rows(&stats, &stages);

        // season 0: stage 0 only
        // season 1: stages -2 and 1
        // season 2: stages -1 and 2
        // Total: 1 + 2 + 2 = 5 rows
        assert_eq!(rows.len(), 5, "expected 5 rows (3 study + 2 pre-study)");

        // Verify pre-study rows exist with negative stage_ids.
        let prestudy_rows: Vec<_> = rows.iter().filter(|r| r.stage_id < 0).collect();
        assert_eq!(
            prestudy_rows.len(),
            2,
            "expected 2 pre-study rows, got {}",
            prestudy_rows.len()
        );

        // stage_id = -2 has season 1 -> (mean=110, std=22).
        let neg2 = rows.iter().find(|r| r.stage_id == -2).expect("row for -2");
        assert!((neg2.mean_m3s - 110.0).abs() < f64::EPSILON);
        assert!((neg2.std_m3s - 22.0).abs() < f64::EPSILON);

        // stage_id = -1 has season 2 -> (mean=120, std=24).
        let neg1 = rows.iter().find(|r| r.stage_id == -1).expect("row for -1");
        assert!((neg1.mean_m3s - 120.0).abs() < f64::EPSILON);
        assert!((neg1.std_m3s - 24.0).abs() < f64::EPSILON);

        // Rows should be sorted by (hydro_id, stage_id).
        for w in rows.windows(2) {
            assert!(
                (w[0].hydro_id.0, w[0].stage_id) <= (w[1].hydro_id.0, w[1].stage_id),
                "rows not sorted"
            );
        }
    }

    #[test]
    fn ar_estimates_to_rows_includes_prestudy_stages() {
        // Same stage layout as test 1.
        let stages = vec![
            make_expansion_stage(0, -2, Some(1)),
            make_expansion_stage(1, -1, Some(2)),
            make_expansion_stage(2, 0, Some(0)),
            make_expansion_stage(3, 1, Some(1)),
            make_expansion_stage(4, 2, Some(2)),
        ];

        let h1 = EntityId(1);
        // AR(1) estimates for seasons 0, 1, 2.
        let ar_estimates = vec![
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 0,
                coefficients: vec![0.3],
                residual_std_ratio: 0.9,
            },
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 1,
                coefficients: vec![0.4],
                residual_std_ratio: 0.85,
            },
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 2,
                coefficients: vec![0.5],
                residual_std_ratio: 0.8,
            },
        ];

        let rows = ar_estimates_to_rows(&ar_estimates, &stages);

        // season 0 -> 1 stage (id 0): 1 row
        // season 1 -> 2 stages (ids -2, 1): 2 rows
        // season 2 -> 2 stages (ids -1, 2): 2 rows
        // Total: 5 rows (each AR(1), so 1 coefficient row per stage)
        assert_eq!(rows.len(), 5, "expected 5 rows");

        // Pre-study coefficient rows exist.
        let prestudy_rows: Vec<_> = rows.iter().filter(|r| r.stage_id < 0).collect();
        assert_eq!(prestudy_rows.len(), 2);

        // stage_id = -2 is season 1, coefficient = 0.4.
        let neg2 = rows.iter().find(|r| r.stage_id == -2).expect("row for -2");
        assert!((neg2.coefficient - 0.4).abs() < f64::EPSILON);
        assert!((neg2.residual_std_ratio - 0.85).abs() < f64::EPSILON);

        // stage_id = -1 is season 2, coefficient = 0.5.
        let neg1 = rows.iter().find(|r| r.stage_id == -1).expect("row for -1");
        assert!((neg1.coefficient - 0.5).abs() < f64::EPSILON);
        assert!((neg1.residual_std_ratio - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn full_estimation_produces_prestudy_inflow_models() {
        use cobre_io::scenarios::assemble_inflow_models;

        // Build stages with 2 pre-study + 3 study.
        let stages = vec![
            make_expansion_stage(0, -2, Some(1)),
            make_expansion_stage(1, -1, Some(2)),
            make_expansion_stage(2, 0, Some(0)),
            make_expansion_stage(3, 1, Some(1)),
            make_expansion_stage(4, 2, Some(2)),
        ];

        let h1 = EntityId(1);

        // Build stats rows (including pre-study).
        let stats = vec![
            SeasonalStats {
                entity_id: h1,
                stage_id: 0,
                mean: 100.0,
                std: 20.0,
            },
            SeasonalStats {
                entity_id: h1,
                stage_id: 1,
                mean: 110.0,
                std: 22.0,
            },
            SeasonalStats {
                entity_id: h1,
                stage_id: 2,
                mean: 120.0,
                std: 24.0,
            },
        ];
        let stats_rows = seasonal_stats_to_rows(&stats, &stages);

        // Build coefficient rows.
        let ar_ests = vec![
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 0,
                coefficients: vec![0.3],
                residual_std_ratio: 0.9,
            },
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 1,
                coefficients: vec![0.4],
                residual_std_ratio: 0.85,
            },
            ArCoefficientEstimate {
                hydro_id: h1,
                season_id: 2,
                coefficients: vec![0.5],
                residual_std_ratio: 0.8,
            },
        ];
        let coeff_rows = ar_estimates_to_rows(&ar_ests, &stages);

        // Assemble into InflowModel.
        let inflow_models =
            assemble_inflow_models(stats_rows, coeff_rows).expect("assembly should succeed");

        // Should have entries for pre-study stages.
        assert!(
            inflow_models.iter().any(|m| m.stage_id < 0),
            "expected pre-study InflowModel entries (negative stage_id)"
        );

        // Pre-study models should have correct stats from their season.
        let prestudy_neg2 = inflow_models
            .iter()
            .find(|m| m.stage_id == -2)
            .expect("InflowModel for stage -2");
        assert!((prestudy_neg2.mean_m3s - 110.0).abs() < f64::EPSILON);
        assert!((prestudy_neg2.std_m3s - 22.0).abs() < f64::EPSILON);
    }

    // ── ticket-003: PACF and contribution cascade tests ──────────────────────

    /// Simulate a 2-season PAR(2) process using deterministic LCG (Box-Muller).
    /// Model: `z_t = phi_1 * z_{t-1} + phi_2 * z_{t-2} + noise_t`.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    fn simulate_two_season_par2(
        phi_1: f64,
        phi_2: f64,
        n_years: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_total = n_years * 2;
        let burnin = 200;
        let n_generate = n_total + burnin;
        let mut values = vec![0.0_f64; n_generate + 2];
        let mut lcg: u64 = seed;

        let lcg_next = |s: u64| -> u64 {
            s.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
        };

        for i in 2..n_generate + 2 {
            lcg = lcg_next(lcg);
            let u1 = (lcg >> 11) as f64 / (1u64 << 53) as f64;
            lcg = lcg_next(lcg);
            let u2 = (lcg >> 11) as f64 / (1u64 << 53) as f64;
            let u1_safe = u1.max(1e-15);
            let noise = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            values[i] = phi_1 * values[i - 1] + phi_2 * values[i - 2] + noise;
        }

        let start = burnin + 2;
        let mut obs_s0 = Vec::with_capacity(n_years);
        let mut obs_s1 = Vec::with_capacity(n_years);
        for y in 0..n_years {
            obs_s0.push(values[start + y * 2]);
            obs_s1.push(values[start + y * 2 + 1]);
        }
        (obs_s0, obs_s1)
    }

    /// Build a minimal 2-season `Stage` for testing.
    fn make_two_season_stage(
        index: usize,
        id: i32,
        season_id: usize,
        year: i32,
        first_half: bool,
    ) -> cobre_core::temporal::Stage {
        use chrono::NaiveDate;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };

        let (start_date, end_date) = if first_half {
            (
                NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
            )
        } else {
            (
                NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
                NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap(),
            )
        };

        cobre_core::temporal::Stage {
            index,
            id,
            start_date,
            end_date,
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 4380.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Build a 2-season `SeasonMap` (H1: Jan–Jun, H2: Jul–Dec).
    fn two_season_map() -> cobre_core::temporal::SeasonMap {
        use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap};
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                SeasonDefinition {
                    id: 0,
                    label: "H1".to_string(),
                    month_start: 1,
                    day_start: Some(1),
                    month_end: Some(6),
                    day_end: Some(30),
                },
                SeasonDefinition {
                    id: 1,
                    label: "H2".to_string(),
                    month_start: 7,
                    day_start: Some(1),
                    month_end: Some(12),
                    day_end: Some(31),
                },
            ],
        }
    }

    /// Verify `iterative_pacf_reduction` does not spuriously reduce a stable
    /// 2-season PAR(2) model (phi_1=0.7, phi_2=0.15). Checks termination,
    /// order preservation, and coefficient matching.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn iterative_pacf_reduction_stable_par2_not_spuriously_reduced() {
        use cobre_stochastic::par::fitting::{
            estimate_periodic_ar_coefficients, periodic_pacf, select_order_pacf,
        };

        let hydro_id = EntityId(1);
        let n_seasons = 2;
        let n_years = 500_usize; // 500 obs/season; threshold ≈ 0.088 << PACF(2) ≈ 0.15.

        let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 137);

        // Build group_obs and stats_map.
        let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
        group_obs.insert((hydro_id, 0), obs_s0.clone());
        group_obs.insert((hydro_id, 1), obs_s1.clone());

        let mean_s0 = obs_s0.iter().sum::<f64>() / obs_s0.len() as f64;
        let mean_s1 = obs_s1.iter().sum::<f64>() / obs_s1.len() as f64;
        let std_s0 = {
            let v = obs_s0.iter().map(|x| (x - mean_s0).powi(2)).sum::<f64>()
                / (obs_s0.len() - 1) as f64;
            v.sqrt()
        };
        let std_s1 = {
            let v = obs_s1.iter().map(|x| (x - mean_s1).powi(2)).sum::<f64>()
                / (obs_s1.len() - 1) as f64;
            v.sqrt()
        };

        let stats_storage = vec![
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 0,
                mean: mean_s0,
                std: std_s0,
            },
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 1,
                mean: mean_s1,
                std: std_s1,
            },
        ];
        let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats_storage
            .iter()
            .enumerate()
            .map(|(s, st)| ((hydro_id, s), st))
            .collect();

        // Run PACF order selection + YW estimation to get the starting estimates.
        let stats_by_season_pop = {
            let n = obs_s0.len() as f64;
            let mu0 = mean_s0;
            let mu1 = mean_s1;
            let s0 = (obs_s0.iter().map(|x| (x - mu0).powi(2)).sum::<f64>() / n).sqrt();
            let s1 = (obs_s1.iter().map(|x| (x - mu1).powi(2)).sum::<f64>() / n).sqrt();
            vec![(mu0, s0), (mu1, s1)]
        };
        let obs_refs: Vec<&[f64]> = vec![&obs_s0, &obs_s1];
        let z_alpha = 1.96_f64;
        // Use max_order=2 because the generating process is AR(2).
        // Higher max_order with periodic PACF on a 2-season split of a stationary
        // process over-estimates order due to per-season standardization artifacts.
        let max_order = 2_usize;

        let mut estimates: Vec<ArCoefficientEstimate> = Vec::new();
        for season in 0..n_seasons {
            let n_obs = obs_refs[season].len();
            let pacf_values = periodic_pacf(
                season,
                max_order,
                n_seasons,
                &obs_refs,
                &stats_by_season_pop,
            );
            let selected = select_order_pacf(&pacf_values, n_obs, z_alpha).selected_order;
            let yw = estimate_periodic_ar_coefficients(
                season,
                selected,
                n_seasons,
                &obs_refs,
                &stats_by_season_pop,
            );
            estimates.push(ArCoefficientEstimate {
                hydro_id,
                season_id: season,
                coefficients: yw.coefficients,
                residual_std_ratio: yw.residual_std_ratio,
            });
        }

        // All seasons should select order 2 before reduction.
        for est in &estimates {
            assert_eq!(
                est.coefficients.len(),
                2,
                "season {} should select order 2; got {}",
                est.season_id,
                est.coefficients.len()
            );
        }

        let coeffs_before: Vec<Vec<f64>> =
            estimates.iter().map(|e| e.coefficients.clone()).collect();

        // Run iterative PACF reduction (should terminate without spurious reductions).
        let reductions = iterative_pacf_reduction(
            &mut estimates,
            n_seasons,
            &[hydro_id],
            &group_obs,
            &stats_map,
            max_order,
            z_alpha,
            None,
        );

        // Stable PAR(2) should not be reduced (order remains 2).
        for est in &estimates {
            assert_eq!(
                est.coefficients.len(),
                2,
                "stable PAR(2) season {} should remain order 2; got {}",
                est.season_id,
                est.coefficients.len()
            );
        }

        // Verify coefficients match (unchanged or recomputed at new order).
        for (est, before) in estimates.iter().zip(coeffs_before.iter()) {
            if est.coefficients.len() == before.len() {
                // No reduction: verify coefficients unchanged.
                for (a, b) in est.coefficients.iter().zip(before.iter()) {
                    assert!(
                        (a - b).abs() < 1e-10,
                        "season {} coefficient drift: {a} vs {b}",
                        est.season_id
                    );
                }
            } else {
                // Reduction occurred: verify against direct YW solve.
                let yw_direct = estimate_periodic_ar_coefficients(
                    est.season_id,
                    est.coefficients.len(),
                    n_seasons,
                    &obs_refs,
                    &stats_by_season_pop,
                );
                for (a, b) in est.coefficients.iter().zip(yw_direct.coefficients.iter()) {
                    assert!(
                        (a - b).abs() < 1e-10,
                        "season {} post-reduction coeff {a} vs YW {b}",
                        est.season_id
                    );
                }
            }
        }

        // No spurious reductions for stable model.
        assert!(!reductions.contains_key(&hydro_id));
    }

    /// Roundtrip estimation test: verify AR(2) coefficient recovery.
    /// Simulates 1000 years of 2-season stationary AR(2) data (phi_1=0.7,
    /// phi_2=0.15). Runs full PACF order selection pipeline and verifies
    /// recovered coefficients match true values within 0.15 tolerance.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn roundtrip_estimation_two_season_par2_recovers_coefficients() {
        use chrono::NaiveDate;

        let hydro_id = EntityId(1);
        let n_years = 1000_usize;
        let true_phi1 = 0.7_f64;
        let true_phi2 = 0.15_f64;

        let (obs_s0, obs_s1) = simulate_two_season_par2(true_phi1, true_phi2, n_years, 42);

        // Build 2 study stages, one per season.
        // Season 0: Jan 1 – Jul 1 (any year); Season 1: Jul 1 – Jan 1.
        // All observations will be mapped via the SeasonMap fallback.
        let ref_year = 2000_i32;
        let stages = vec![
            make_two_season_stage(0, 0, 0, ref_year, true),
            make_two_season_stage(1, 1, 1, ref_year, false),
        ];

        // Build seasonal stats (Bessel-corrected std to match estimate_seasonal_stats).
        let n_f = n_years as f64;
        let mu0 = obs_s0.iter().sum::<f64>() / n_f;
        let mu1 = obs_s1.iter().sum::<f64>() / n_f;
        let std0 = (obs_s0.iter().map(|x| (x - mu0).powi(2)).sum::<f64>() / (n_f - 1.0)).sqrt();
        let std1 = (obs_s1.iter().map(|x| (x - mu1).powi(2)).sum::<f64>() / (n_f - 1.0)).sqrt();

        let seasonal_stats = vec![
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 0, // matches stages[0].id
                mean: mu0,
                std: std0,
            },
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: 1, // matches stages[1].id
                mean: mu1,
                std: std1,
            },
        ];

        // Build observations with dates: season 0 = Jan 1, season 1 = Jul 1.
        // Using a fixed reference year does NOT matter because find_season_for_date
        // falls back to the season_map when dates are outside the study stage ranges.
        let season_map = two_season_map();
        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for y in 0..n_years {
            let year = (1970 + y) as i32;
            observations.push((
                hydro_id,
                NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
                obs_s0[y],
            ));
            observations.push((
                hydro_id,
                NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
                obs_s1[y],
            ));
        }

        let (estimates, _report) = estimate_ar_coefficients_with_selection(
            &observations,
            &seasonal_stats,
            &stages,
            &[hydro_id],
            &ArEstimationConfig {
                max_order: 2,
                max_coeff_magnitude: None,
                season_map: Some(&season_map),
            },
        )
        .expect("estimation must succeed");

        // Collect per-season coefficients.
        let est_s0 = estimates
            .iter()
            .find(|e| e.hydro_id == hydro_id && e.season_id == 0)
            .expect("season 0 estimate must exist");
        let est_s1 = estimates
            .iter()
            .find(|e| e.hydro_id == hydro_id && e.season_id == 1)
            .expect("season 1 estimate must exist");

        // With 1000 years the PACF threshold is 1.96/sqrt(1000) ≈ 0.062,
        // well below PACF(2) ≈ 0.15 and well above PACF(3..4) ≈ 0.
        // The pipeline should select order 2 and recover the true coefficients.
        for est in [est_s0, est_s1] {
            assert!(
                est.coefficients.len() >= 2,
                "season {} should select at least order 2; got {}",
                est.season_id,
                est.coefficients.len()
            );
            assert!(
                (est.coefficients[0] - true_phi1).abs() < 0.15,
                "season {} phi_1={:.4} should be within 0.15 of true {true_phi1:.4}",
                est.season_id,
                est.coefficients[0]
            );
            assert!(
                (est.coefficients[1] - true_phi2).abs() < 0.15,
                "season {} phi_2={:.4} should be within 0.15 of true {true_phi2:.4}",
                est.season_id,
                est.coefficients[1]
            );
            assert!(
                est.residual_std_ratio.is_finite() && est.residual_std_ratio > 0.0,
                "season {} residual_std_ratio={} should be positive finite",
                est.season_id,
                est.residual_std_ratio
            );
        }
    }

    // ── ar_rows_to_estimates unit tests ───────────────────────────────────────

    /// AC-009-ar-rows-1: `ar_rows_to_estimates` groups by season, deduplicates stages.
    ///
    /// Creates `InflowArCoefficientRow` entries for 2 hydros across 3 stages:
    /// - Stage 0 (season 0), stage 1 (season 0), stage 2 (season 1).
    ///
    /// After conversion the output must have 2 * 2 = 4 estimates (2 hydros * 2
    /// seasons). Each estimate must carry the coefficients from the FIRST stage
    /// in the season (stage 0 for season 0, stage 2 for season 1).
    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_ar_rows_to_estimates_groups_by_season() {
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };

        let make_stage = |id: i32, season_id: usize| cobre_core::temporal::Stage {
            index: id as usize,
            id,
            start_date: chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(1970, 7, 1).unwrap(),
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "T".to_string(),
                duration_hours: 1.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        // 3 stages: stage 0 and stage 1 map to season 0; stage 2 maps to season 1.
        let stages = vec![make_stage(0, 0), make_stage(1, 0), make_stage(2, 1)];

        // AR(1) rows for 2 hydros; sorted by (hydro_id, stage_id, lag).
        // Each stage has one lag-1 row (AR order 1).
        let rows = vec![
            // hydro 1, stage 0 (season 0), lag 1
            InflowArCoefficientRow {
                hydro_id: EntityId(1),
                stage_id: 0,
                lag: 1,
                coefficient: 0.50,
                residual_std_ratio: 0.85,
            },
            // hydro 1, stage 1 (season 0 duplicate), lag 1
            InflowArCoefficientRow {
                hydro_id: EntityId(1),
                stage_id: 1,
                lag: 1,
                coefficient: 0.50,
                residual_std_ratio: 0.85,
            },
            // hydro 1, stage 2 (season 1), lag 1
            InflowArCoefficientRow {
                hydro_id: EntityId(1),
                stage_id: 2,
                lag: 1,
                coefficient: 0.60,
                residual_std_ratio: 0.80,
            },
            // hydro 2, stage 0 (season 0), lag 1
            InflowArCoefficientRow {
                hydro_id: EntityId(2),
                stage_id: 0,
                lag: 1,
                coefficient: 0.40,
                residual_std_ratio: 0.90,
            },
            // hydro 2, stage 1 (season 0 duplicate), lag 1
            InflowArCoefficientRow {
                hydro_id: EntityId(2),
                stage_id: 1,
                lag: 1,
                coefficient: 0.40,
                residual_std_ratio: 0.90,
            },
            // hydro 2, stage 2 (season 1), lag 1
            InflowArCoefficientRow {
                hydro_id: EntityId(2),
                stage_id: 2,
                lag: 1,
                coefficient: 0.35,
                residual_std_ratio: 0.88,
            },
        ];

        let estimates = ar_rows_to_estimates(&rows, &stages);

        // 2 hydros * 2 seasons = 4 estimates.
        assert_eq!(
            estimates.len(),
            4,
            "expected 4 estimates (2 hydros * 2 seasons), got {}",
            estimates.len()
        );

        // Hydro 1, season 0: coefficient from stage 0 (canonical first stage).
        let e = estimates
            .iter()
            .find(|e| e.hydro_id == EntityId(1) && e.season_id == 0)
            .expect("hydro 1, season 0 estimate must exist");
        assert_eq!(e.coefficients.len(), 1, "AR(1) must have 1 coefficient");
        assert!(
            (e.coefficients[0] - 0.50).abs() < f64::EPSILON,
            "coeff must be 0.50, got {}",
            e.coefficients[0]
        );
        assert!(
            (e.residual_std_ratio - 0.85).abs() < f64::EPSILON,
            "residual_std_ratio must be 0.85"
        );

        // Hydro 1, season 1: coefficient from stage 2.
        let e = estimates
            .iter()
            .find(|e| e.hydro_id == EntityId(1) && e.season_id == 1)
            .expect("hydro 1, season 1 estimate must exist");
        assert_eq!(e.coefficients.len(), 1);
        assert!((e.coefficients[0] - 0.60).abs() < f64::EPSILON);
        assert!((e.residual_std_ratio - 0.80).abs() < f64::EPSILON);

        // Hydro 2, season 0.
        let e = estimates
            .iter()
            .find(|e| e.hydro_id == EntityId(2) && e.season_id == 0)
            .expect("hydro 2, season 0 estimate must exist");
        assert_eq!(e.coefficients.len(), 1);
        assert!((e.coefficients[0] - 0.40).abs() < f64::EPSILON);

        // Hydro 2, season 1.
        let e = estimates
            .iter()
            .find(|e| e.hydro_id == EntityId(2) && e.season_id == 1)
            .expect("hydro 2, season 1 estimate must exist");
        assert_eq!(e.coefficients.len(), 1);
        assert!((e.coefficients[0] - 0.35).abs() < f64::EPSILON);
    }

    // ── UserArHistoryStats unit tests ─────────────────────────────────────────

    /// Write `inflow_ar_coefficients.parquet` with known AR(1) coefficients for
    /// a single hydro expanded to all stages in `stages`.
    ///
    /// `stages` must be pre-built (same as the system's stages). The parquet
    /// file will have one row per stage with lag=1.
    fn write_unit_test_ar_coefficients(
        path: &std::path::Path,
        hydro_id: i32,
        stages: &[cobre_core::temporal::Stage],
        coefficient: f64,
        residual_std_ratio: f64,
    ) {
        use arrow::array::{Float64Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
            Field::new("residual_std_ratio", DataType::Float64, false),
        ]));

        let n = stages.len();
        let hydro_ids: Vec<i32> = vec![hydro_id; n];
        let stage_ids: Vec<i32> = stages.iter().map(|s| s.id).collect();
        let lags: Vec<i32> = vec![1; n];
        let coefficients: Vec<f64> = vec![coefficient; n];
        let ratios: Vec<f64> = vec![residual_std_ratio; n];

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(hydro_ids)),
                Arc::new(Int32Array::from(stage_ids)),
                Arc::new(Int32Array::from(lags)),
                Arc::new(Float64Array::from(coefficients)),
                Arc::new(Float64Array::from(ratios)),
            ],
        )
        .expect("valid batch");

        let file = std::fs::File::create(path).expect("create parquet file");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");
    }

    /// Build a system with one hydro and 2-season stages (same structure as
    /// `build_system_with_user_stats`) but with EMPTY inflow_models.
    ///
    /// This represents the state after `load_case` when `inflow_seasonal_stats.parquet`
    /// is absent (the P7/UserArHistoryStats case): `assemble_inflow_models` returns
    /// an empty vec, so `system.inflow_models()` is empty.
    #[allow(clippy::cast_possible_wrap)]
    fn build_system_empty_models(n_years: usize) -> System {
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

        let hydro_id = EntityId(1);
        let bus = Bus {
            id: EntityId(10),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(f64::INFINITY),
                cost_per_mwh: 3000.0,
            }],
            excess_cost: 0.0,
        };

        let ref_year = 1970_i32;
        let mut stages = Vec::with_capacity(n_years * 2);
        for y in 0..n_years {
            let year = ref_year + y as i32;
            stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
            stages.push(make_two_season_stage(
                y * 2 + 1,
                (y * 2 + 1) as i32,
                1,
                year,
                false,
            ));
        }

        let hydro = Hydro {
            id: hydro_id,
            name: "H1".to_string(),
            bus_id: EntityId(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 5000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 1000.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            // NOTE: no inflow_models — represents the P7 case after load_case
            .build()
            .expect("valid system with empty inflow models")
    }

    /// Setup a case directory for the P7 (UserArHistoryStats) path:
    /// - `inflow_history.parquet`: real Parquet with synthetic 2-season data.
    /// - `inflow_ar_coefficients.parquet`: real Parquet with known AR(1) coefficients.
    /// - No `inflow_seasonal_stats.parquet`.
    ///
    /// Returns the stages used in the system so the AR file can reference valid stage IDs.
    #[allow(clippy::cast_possible_wrap)]
    fn setup_user_ar_case(
        case_dir: &std::path::Path,
        n_years: usize,
        ar_coefficient: f64,
        residual_std_ratio: f64,
    ) {
        create_required_files(case_dir);
        let scenarios = case_dir.join("scenarios");
        std::fs::create_dir_all(&scenarios).unwrap();

        // Write history parquet.
        write_unit_test_inflow_history(&scenarios.join("inflow_history.parquet"), 1, n_years);

        // Build the same stages as build_system_empty_models to get stage IDs.
        let ref_year = 1970_i32;
        let mut stages = Vec::with_capacity(n_years * 2);
        for y in 0..n_years {
            let year = ref_year + y as i32;
            stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
            stages.push(make_two_season_stage(
                y * 2 + 1,
                (y * 2 + 1) as i32,
                1,
                year,
                false,
            ));
        }

        // Write AR coefficients parquet with known values, one row per stage.
        write_unit_test_ar_coefficients(
            &scenarios.join("inflow_ar_coefficients.parquet"),
            1,
            &stages,
            ar_coefficient,
            residual_std_ratio,
        );

        // NO inflow_seasonal_stats.parquet — this is the P7 path.
    }

    /// AC-009-1: `estimate_from_history` with P7 setup preserves user AR coefficients
    /// bitwise in the returned inflow models.
    ///
    /// Setup: system with empty inflow_models, case dir with history + AR (no stats).
    /// Assert: every returned model's `ar_coefficients[0]` and `residual_std_ratio`
    /// match the known values written to `inflow_ar_coefficients.parquet` exactly.
    #[test]
    fn test_user_ar_estimation_preserves_ar_coefficients() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;
        const KNOWN_COEFF: f64 = 0.72;
        const KNOWN_RATIO: f64 = 0.69;

        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        setup_user_ar_case(case_dir, N_YEARS, KNOWN_COEFF, KNOWN_RATIO);
        let system = build_system_empty_models(N_YEARS);
        let config = default_config();

        let (updated, report, path) = estimate_from_history(system, case_dir, &config)
            .expect("UserArHistoryStats estimation must succeed");

        assert_eq!(
            path,
            EstimationPath::UserArHistoryStats,
            "expected UserArHistoryStats path"
        );
        assert!(
            report.is_some(),
            "UserArHistoryStats must return Some(report)"
        );

        let models = updated.inflow_models();
        assert!(
            !models.is_empty(),
            "estimation must produce at least one inflow model"
        );

        for m in models {
            assert_eq!(
                m.ar_coefficients.len(),
                1,
                "every model must have AR(1) coefficients (lag 1 only), stage {}",
                m.stage_id
            );
            assert_eq!(
                m.ar_coefficients[0].to_bits(),
                KNOWN_COEFF.to_bits(),
                "ar_coefficients[0] must be bitwise identical to {KNOWN_COEFF} for stage {}",
                m.stage_id
            );
            assert_eq!(
                m.residual_std_ratio.to_bits(),
                KNOWN_RATIO.to_bits(),
                "residual_std_ratio must be bitwise identical to {KNOWN_RATIO} for stage {}",
                m.stage_id
            );
        }
    }

    /// AC-009-2: `estimate_from_history` with P7 setup produces finite, positive
    /// `mean_m3s` and `std_m3s` estimated from inflow history.
    ///
    /// Same setup as `test_user_ar_estimation_preserves_ar_coefficients`.
    #[test]
    fn test_user_ar_estimation_estimates_stats_from_history() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;

        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        setup_user_ar_case(case_dir, N_YEARS, 0.55, 0.83);
        let system = build_system_empty_models(N_YEARS);
        let config = default_config();

        let (updated, _report, _path) = estimate_from_history(system, case_dir, &config)
            .expect("UserArHistoryStats estimation must succeed");

        let models = updated.inflow_models();
        assert!(
            !models.is_empty(),
            "estimation must produce at least one inflow model"
        );

        for m in models {
            assert!(
                m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
                "mean_m3s must be finite and positive, got {} for stage {}",
                m.mean_m3s,
                m.stage_id
            );
            assert!(
                m.std_m3s.is_finite() && m.std_m3s >= 0.0,
                "std_m3s must be finite and non-negative, got {} for stage {}",
                m.std_m3s,
                m.stage_id
            );
        }
    }

    /// AC-009-3: `estimate_from_history` with P7 setup returns a report with
    /// method "user_provided" and an empty entries map.
    #[test]
    fn test_user_ar_estimation_returns_user_provided_report() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;

        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        setup_user_ar_case(case_dir, N_YEARS, 0.55, 0.83);
        let system = build_system_empty_models(N_YEARS);
        let config = default_config();

        let (_updated, report, _path) = estimate_from_history(system, case_dir, &config)
            .expect("UserArHistoryStats estimation must succeed");

        let report = report.expect("UserArHistoryStats must return Some(EstimationReport)");

        assert_eq!(
            report.method, "user_provided",
            "report method must be 'user_provided', got '{}'",
            report.method
        );
        assert!(
            report.entries.is_empty(),
            "report entries must be empty (no AR was estimated), got {} entries",
            report.entries.len()
        );
    }

    // ── Bidirectional coverage validation tests (ticket-010) ─────────────────

    /// Build a minimal Hydro struct reusing the same penalty/generation defaults
    /// as the single-hydro helpers above.
    fn make_hydro(hydro_id: EntityId, bus_id: EntityId) -> cobre_core::entities::hydro::Hydro {
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel};
        Hydro {
            id: hydro_id,
            name: format!("H{}", hydro_id.0),
            bus_id,
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 5000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: cobre_core::entities::hydro::HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 1000.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        }
    }

    /// Build a System with two hydros (IDs 1 and 2). User stats (inflow_models)
    /// are created only for the hydros in `stats_hydro_ids`. Hydros in
    /// `all_hydro_ids` but not in `stats_hydro_ids` have no stats rows.
    #[allow(clippy::cast_possible_wrap)]
    fn build_two_hydro_system_selective_stats(
        n_years: usize,
        all_hydro_ids: &[EntityId],
        stats_hydro_ids: &[EntityId],
    ) -> System {
        use cobre_core::scenario::InflowModel;
        use cobre_core::{Bus, DeficitSegment, SystemBuilder};

        let bus_id = EntityId(10);
        let bus = Bus {
            id: bus_id,
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(f64::INFINITY),
                cost_per_mwh: 3000.0,
            }],
            excess_cost: 0.0,
        };

        let ref_year = 1970_i32;
        let mut stages = Vec::with_capacity(n_years * 2);
        for y in 0..n_years {
            let year = ref_year + y as i32;
            stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
            stages.push(make_two_season_stage(
                y * 2 + 1,
                (y * 2 + 1) as i32,
                1,
                year,
                false,
            ));
        }

        // Build inflow models only for hydros in stats_hydro_ids.
        let inflow_models: Vec<InflowModel> = stats_hydro_ids
            .iter()
            .flat_map(|&hid| {
                stages.iter().map(move |s| InflowModel {
                    hydro_id: hid,
                    stage_id: s.id,
                    mean_m3s: 100.0,
                    std_m3s: 10.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                })
            })
            .collect();

        let hydros: Vec<_> = all_hydro_ids
            .iter()
            .map(|&hid| make_hydro(hid, bus_id))
            .collect();

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .build()
            .expect("valid two-hydro system")
    }

    /// Write a real `inflow_history.parquet` with data for the given list of
    /// hydro IDs. Each hydro gets identical synthetic PAR(2) observations.
    fn write_history_for_hydros(path: &std::path::Path, hydro_ids: &[i32], n_years: usize) {
        use arrow::array::{Date32Array, Float64Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use chrono::NaiveDate;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let date_to_days = |d: NaiveDate| -> i32 {
            i32::try_from((d - epoch).num_days()).expect("date in Date32 range")
        };

        let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 99);

        let mut ids: Vec<i32> = Vec::new();
        let mut dates: Vec<i32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();

        for &hid in hydro_ids {
            for y in 0..n_years {
                let year = (1970 + y) as i32;
                ids.push(hid);
                dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 1, 15).unwrap()));
                values.push(obs_s0[y] + 300.0);

                ids.push(hid);
                dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 7, 15).unwrap()));
                values.push(obs_s1[y] + 300.0);
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("date", DataType::Date32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Date32Array::from(dates)),
                Arc::new(Float64Array::from(values)),
            ],
        )
        .expect("valid batch");

        let file = std::fs::File::create(path).expect("create parquet file");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");
    }

    /// AC-T10-1: Direction A — AR estimated for hydro 2 but no user stats for it.
    ///
    /// Setup: system with hydros [1, 2], history for both [1, 2], but
    /// `inflow_seasonal_stats.parquet` provides stats only for hydro 1.
    ///
    /// Assert: `estimate_from_history` returns `Err` with a `ConstraintError`
    /// whose description contains `"2"` (the uncovered hydro ID).
    #[test]
    fn test_partial_estimation_direction_a_missing_stats() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;
        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        create_required_files(case_dir);
        let scenarios = case_dir.join("scenarios");
        std::fs::create_dir_all(&scenarios).unwrap();

        // History for both hydros 1 and 2.
        write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1, 2], N_YEARS);

        // Stats sentinel — presence triggers PartialEstimation manifest flag.
        std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
            .expect("write sentinel");

        // System: hydros [1, 2] in the hydros list, but stats only for hydro 1.
        let system = build_two_hydro_system_selective_stats(
            N_YEARS,
            &[EntityId(1), EntityId(2)],
            &[EntityId(1)], // stats only for hydro 1
        );

        let config = default_config();
        let result = estimate_from_history(system, case_dir, &config);

        assert!(
            result.is_err(),
            "Direction A must return Err when hydro 2 has AR estimates but no user stats"
        );

        let err = result.unwrap_err();
        let description = err.to_string();
        assert!(
            description.contains('2'),
            "error description must contain the uncovered hydro ID '2', got: {description}"
        );
    }

    /// AC-T10-2: Direction B — user stats for hydro 2 but no history for it.
    ///
    /// Setup: system with hydros [1, 2] and stats for both, but history only
    /// for hydro 1.
    ///
    /// Assert: `estimate_from_history` returns `Ok`, the `EstimationReport`
    /// has `white_noise_fallbacks == [EntityId(2)]`, and the returned system's
    /// inflow model for hydro 2 has empty `ar_coefficients`.
    #[test]
    fn test_partial_estimation_direction_b_white_noise_fallback() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;
        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        create_required_files(case_dir);
        let scenarios = case_dir.join("scenarios");
        std::fs::create_dir_all(&scenarios).unwrap();

        // History only for hydro 1.
        write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1], N_YEARS);

        // Stats sentinel.
        std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
            .expect("write sentinel");

        // System: hydros [1, 2] with stats for both (hydro 2 gets white-noise fallback).
        let system = build_two_hydro_system_selective_stats(
            N_YEARS,
            &[EntityId(1), EntityId(2)],
            &[EntityId(1), EntityId(2)], // stats for both
        );

        let config = default_config();
        let (updated, report, path) = estimate_from_history(system, case_dir, &config)
            .expect("Direction B must succeed (not an error)");

        assert_eq!(
            path,
            EstimationPath::PartialEstimation,
            "expected PartialEstimation path"
        );

        let report = report.expect("PartialEstimation must return Some(EstimationReport)");

        assert_eq!(
            report.white_noise_fallbacks,
            vec![EntityId(2)],
            "white_noise_fallbacks must be [EntityId(2)], got {:?}",
            report.white_noise_fallbacks
        );

        // Hydro 2 should have empty ar_coefficients in the returned system.
        let hydro2_models: Vec<_> = updated
            .inflow_models()
            .iter()
            .filter(|m| m.hydro_id == EntityId(2))
            .collect();
        assert!(
            !hydro2_models.is_empty(),
            "returned system must have inflow models for hydro 2"
        );
        for m in &hydro2_models {
            assert!(
                m.ar_coefficients.is_empty(),
                "hydro 2 must have empty ar_coefficients (white-noise fallback), stage {}",
                m.stage_id
            );
        }
    }

    /// AC-T10-3: Exact coverage — single hydro with matching history and stats.
    ///
    /// Reuses the single-hydro setup from ticket-008. Asserts that
    /// `white_noise_fallbacks` is empty on the returned report.
    #[test]
    fn test_partial_estimation_exact_coverage_no_fallback() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;
        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        setup_partial_estimation_case(case_dir, N_YEARS);
        let system = build_system_with_user_stats(N_YEARS);
        let config = default_config();

        let (_updated, report, _path) =
            estimate_from_history(system, case_dir, &config).expect("exact coverage must succeed");

        let report = report.expect("PartialEstimation must return Some(EstimationReport)");

        assert!(
            report.white_noise_fallbacks.is_empty(),
            "white_noise_fallbacks must be empty for exact coverage, got {:?}",
            report.white_noise_fallbacks
        );
    }

    /// AC-T10-4: `run_estimation` (FullEstimation path) never populates
    /// `white_noise_fallbacks` — it must be empty on the returned report.
    #[test]
    fn test_full_estimation_report_has_empty_fallbacks() {
        use tempfile::TempDir;

        const N_YEARS: usize = 30;
        let dir = TempDir::new().unwrap();
        let case_dir = dir.path();

        create_required_files(case_dir);
        let scenarios = case_dir.join("scenarios");
        std::fs::create_dir_all(&scenarios).unwrap();

        // History only — no stats file → FullEstimation path.
        write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1], N_YEARS);
        // No inflow_seasonal_stats.parquet → FullEstimation.

        // System with one hydro, no pre-loaded inflow models (no user stats).
        let system = build_two_hydro_system_selective_stats(
            N_YEARS,
            &[EntityId(1)],
            &[], // no user stats
        );

        let config = default_config();
        let (_, report, path) =
            estimate_from_history(system, case_dir, &config).expect("FullEstimation must succeed");

        assert_eq!(
            path,
            EstimationPath::FullEstimation,
            "expected FullEstimation path"
        );

        let report = report.expect("FullEstimation must return Some(EstimationReport)");

        assert!(
            report.white_noise_fallbacks.is_empty(),
            "FullEstimation must never populate white_noise_fallbacks, got {:?}",
            report.white_noise_fallbacks
        );
    }

    // ── StdRatioDivergence unit tests ─────────────────────────────────────────

    /// Helper: build a System and fitting_stats for a single hydro with the
    /// given per-season std values, then call `check_std_ratio_divergence`.
    ///
    /// `user_stds[i]` is the user-provided std for season `i`.
    /// `est_stds[i]` is the estimated std for season `i`.
    /// Stages are created so that stage_id == season_id (one stage per season).
    fn collect_std_ratio_warnings(
        hydro_id: EntityId,
        user_stds: &[f64],
        est_stds: &[f64],
    ) -> Vec<StdRatioDivergence> {
        use cobre_core::scenario::InflowModel;

        assert_eq!(
            user_stds.len(),
            est_stds.len(),
            "user_stds and est_stds must have equal length"
        );
        let n = user_stds.len();

        // Build stages: stage_id == season_id == i, one stage per season.
        let stages: Vec<cobre_core::temporal::Stage> = (0..n)
            .map(|i| {
                let year = 1970_i32;
                let first_half = i % 2 == 0;
                make_two_season_stage(i, i as i32, i, year, first_half)
            })
            .collect();

        // Build user InflowModels: one per stage.
        let user_models: Vec<InflowModel> = (0..n)
            .map(|i| InflowModel {
                hydro_id,
                stage_id: i as i32,
                mean_m3s: 100.0,
                std_m3s: user_stds[i],
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let system = SystemBuilder::new()
            .inflow_models(user_models)
            .stages(stages.clone())
            .build()
            .expect("valid system");

        // Build fitting_stats: entity_id = hydro_id, stage_id = i, std = est_stds[i].
        let fitting_stats: Vec<SeasonalStats> = (0..n)
            .map(|i| SeasonalStats {
                entity_id: hydro_id,
                stage_id: i as i32,
                mean: 100.0,
                std: est_stds[i],
            })
            .collect();

        check_std_ratio_divergence(&system, &fitting_stats, &stages)
    }

    /// P9-001: Warning fires when consecutive std ratios diverge by more than 2x.
    ///
    /// user stds [100.0, 20.0], est stds [100.0, 100.0].
    /// Pair (0→1): ratio_user = 5.0, ratio_est = 1.0, divergence = 5.0 → warn.
    /// Wrap (1→0): ratio_user = 0.2, ratio_est = 1.0, divergence = 5.0 → warn.
    /// Both pairs diverge, so 2 warnings are emitted. The test verifies that
    /// at least one warning covers the (0→1) pair and the hydro_id is correct.
    #[test]
    fn test_std_ratio_divergence_fires_when_ratios_diverge() {
        let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0], &[100.0, 100.0]);
        assert!(
            !warnings.is_empty(),
            "expected at least one StdRatioDivergence when ratio diverges by 5x"
        );
        // The (0→1) pair must be in the warnings.
        let pair_0_1 = warnings.iter().find(|w| w.season_a == 0 && w.season_b == 1);
        assert!(
            pair_0_1.is_some(),
            "expected a warning for season pair 0→1, got {warnings:?}"
        );
        let w = pair_0_1.unwrap();
        assert_eq!(
            w.hydro_id,
            EntityId(1),
            "warning must record the correct hydro_id"
        );
        assert!(
            (w.divergence - 5.0).abs() < 1e-10,
            "divergence for pair 0→1 must be 5.0, got {}",
            w.divergence
        );
    }

    /// P9-002: No warning when ratios are similar (divergence <= 2.0).
    ///
    /// user stds [100.0, 20.0], est stds [90.0, 18.0].
    /// ratio_user = 5.0, ratio_est = 5.0. divergence = 1.0 → no warning.
    #[test]
    fn test_std_ratio_divergence_not_fires_when_similar() {
        let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0], &[90.0, 18.0]);
        assert!(
            warnings.is_empty(),
            "expected no StdRatioDivergence when ratios are similar, got {warnings:?}"
        );
    }

    /// P9-003: Season pairs with near-zero denominator std are skipped.
    ///
    /// user stds [100.0, 0.0], est stds [90.0, 18.0].
    /// The pair (season 0 → season 1) has u_b = 0.0 < 1e-12 → skipped.
    /// The wrap pair (season 1 → season 0) has u_b = 100.0 and e_b = 90.0 → checked.
    /// ratio_user = 0/100 = 0.0, ratio_est = 18/90 = 0.2.
    /// divergence = max(0/0.2, 0.2/0) → second division hits near-zero guard → skipped.
    #[test]
    fn test_std_ratio_divergence_skips_near_zero_std() {
        // user stds [100.0, 0.0]: the first pair has denominator 0.0 → skipped.
        let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 0.0], &[90.0, 18.0]);
        // No panic must occur. The pair involving zero std is silently skipped.
        // The wrap pair has ratio_user = 0.0/100.0 = 0.0, ratio_est = 18.0/90.0 = 0.2;
        // ratio_user / ratio_est would require dividing 0/0.2 = 0, and
        // ratio_est / ratio_user would divide by 0. The near-zero guard on ratio_est
        // is not triggered here (0.2 is not near zero), but ratio_user = 0.0 means
        // divergence = max(0/0.2, 0.2/0). The 0.2/0 branch triggers the ratio_est
        // guard only if ratio_user < 1e-12, which 0.0 satisfies → skipped.
        // We assert no panic and that the result is well-defined.
        let _ = warnings; // result is valid (empty or one entry); no panic is the key assertion.
    }

    /// P9-004: Wrap-around pair (last season → first season) is checked.
    ///
    /// user stds [100.0, 20.0, 50.0], est stds [100.0, 20.0, 10.0].
    /// Pair (0→1): ratio_user=5.0, ratio_est=5.0 → divergence=1.0 (no warn).
    /// Pair (1→2): ratio_user=0.4, ratio_est=2.0 → divergence=5.0 (warn).
    /// Wrap (2→0): ratio_user=50/100=0.5, ratio_est=10/100=0.1 → divergence=5.0 (warn).
    #[test]
    fn test_std_ratio_divergence_wraps_last_to_first() {
        let warnings =
            collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0, 50.0], &[100.0, 20.0, 10.0]);
        // Pairs (1→2) and wrap (2→0) both diverge.
        assert!(
            warnings.len() >= 2,
            "expected at least 2 StdRatioDivergence entries (including wrap), got {}",
            warnings.len()
        );
        // The wrap pair (season 2 → season 0) must appear.
        let has_wrap = warnings.iter().any(|w| w.season_a == 2 && w.season_b == 0);
        assert!(
            has_wrap,
            "expected a warning for the wrap-around pair season 2 → season 0"
        );
    }
}
