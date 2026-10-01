//! Automatic PAR(p) parameter estimation from historical inflow observations.
//!
//! This module bridges case loading (this crate) and PAR fitting (`cobre-stochastic`).
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
//! - **Role 2 (AR coefficients)**: `ar_coefficients` per hydro per stage. These
//!   drive the autoregressive scenario noise and can come from either the user
//!   file (`inflow_ar_coefficients.parquet`) or history estimation.
//!   `residual_std_ratio` is not a Role 2 input: it is always derived at load
//!   from the (user-provided or estimated) `ar_coefficients` via the
//!   periodic-ACF closure (`populate_derived_residual_ratios`), never read
//!   from a file.
//!
//! Rows 4-6 are the "active" paths where at least one role is estimated from
//! history. In rows 4 and 6, Role 2 is estimated via periodic Yule-Walker / PACF.
//! In row 5, Role 1 is estimated from history while Role 2's `ar_coefficients`
//! are preserved from user (`residual_std_ratio` is still derived, not preserved).
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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use chrono::{Months, NaiveDate};
use cobre_core::{EntityId, SeasonMap, Stage, System, ValidationError};
use cobre_stochastic::{
    StochasticError,
    par::aggregate::aggregate_observations_to_season,
    par::fitting::{
        ArCoefficientEstimate, ArEstimationConfig, SeasonalStats, StdRatioDivergence,
        estimate_ar_coefficients_with_selection, estimate_correlation_with_season_map,
        estimate_seasonal_stats_with_season_map,
    },
    season_cast::{
        RealizedWindow, SeasonPeriodWindow, cast, next_season_period_window, season_period_window,
    },
};

use crate::LoadError::ConstraintError;
use crate::{
    Config, FileManifest, InputFile, LoadError, OrderSelectionMethod, ValidationContext,
    parse_inflow_ar_coefficients, parse_inflow_history,
    scenarios::{
        InflowAnnualComponentRow, InflowArCoefficientRow, InflowHistoryRow, InflowSeasonalStatsRow,
        assemble_inflow_models, populate_derived_residual_ratios,
        residual_derivation::season_dense_index, resolve_stage_seasons,
    },
    validate_structure,
};

// `EstimationReport` lives in `cobre_stochastic::par::fitting`; re-exported here
// so callers resolve it alongside `EstimationPath`/`estimate_from_history`.
pub use cobre_stochastic::par::fitting::EstimationReport;

/// Classification of the estimation path taken for a given input file manifest.
///
/// Each variant is one row of the input-path matrix in the module doc, keyed on
/// the three flags H (history), S (seasonal stats), R (AR coefficients). AR
/// without history is meaningless, so `R` alone resolves to a no-history row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationPath {
    /// No history: system returned unchanged (also the fallback for AR-without-history).
    Deterministic,
    /// User-provided white-noise stats, no history: system returned unchanged.
    UserStatsWhiteNoise,
    /// User-provided complete model, no history: system returned unchanged.
    UserProvidedNoHistory,
    /// History only: both seasonal stats (Role 1) and AR coefficients (Role 2) estimated.
    FullEstimation,
    /// History + user AR: stats estimated from history (Role 1), user AR preserved bitwise (Role 2).
    UserArHistoryStats,
    /// History + user stats: user stats preserved (Role 1), AR estimated from history (Role 2).
    PartialEstimation,
    /// All parameters user-provided: system returned unchanged.
    UserProvidedAll,
}

impl EstimationPath {
    /// Resolve the estimation path from the three boolean manifest flags.
    ///
    /// Invalid combinations (AR present without history or stats) fall back to
    /// `Deterministic` because AR coefficients alone cannot drive estimation.
    #[must_use]
    pub fn resolve(manifest: &FileManifest) -> Self {
        match (
            manifest.present(InputFile::ScenariosInflowHistoryParquet),
            manifest.present(InputFile::ScenariosInflowSeasonalStatsParquet),
            manifest.present(InputFile::ScenariosInflowArCoefficientsParquet),
        ) {
            (false, false, _) => Self::Deterministic,
            (false, true, false) => Self::UserStatsWhiteNoise,
            (false, true, true) => Self::UserProvidedNoHistory,
            (true, false, false) => Self::FullEstimation,
            (true, false, true) => Self::UserArHistoryStats,
            (true, true, false) => Self::PartialEstimation,
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

    /// The replacement `inflow_models` table produced by estimation is not
    /// canonically ordered.
    #[error("scenario model validation error: {0}")]
    Validation(#[from] ValidationError),
}

/// Estimate or load PAR(p) model parameters based on the input file manifest.
///
/// Resolves the [`EstimationPath`] for `case_dir` and dispatches to the matching
/// `run_*` pipeline; pass-through paths return `system` unchanged with `None` report.
///
/// # Errors
///
/// - [`EstimationError::Load`] -- file read, parse, or validation failure.
/// - [`EstimationError::Stochastic`] -- insufficient observations for any
///   `(entity, season)` group during AR or stats estimation.
/// - [`EstimationError::Validation`] -- the estimated `inflow_models` table is
///   not canonically ordered.
pub fn estimate_from_history(
    system: System,
    case_dir: &Path,
    config: &Config,
) -> Result<(System, Option<EstimationReport>, EstimationPath), EstimationError> {
    let mut ctx = ValidationContext::new();
    let manifest = validate_structure(case_dir, &mut ctx);

    // Treat structural validation errors as a no-op deterministic path.
    if ctx.into_result().is_err() {
        return Ok((system, None, EstimationPath::Deterministic));
    }

    let path = EstimationPath::resolve(&manifest);

    match path {
        EstimationPath::Deterministic
        | EstimationPath::UserStatsWhiteNoise
        | EstimationPath::UserProvidedNoHistory
        | EstimationPath::UserProvidedAll => Ok((system, None, path)),

        EstimationPath::PartialEstimation => {
            let (system, report) = run_partial_estimation(system, case_dir, config, &manifest)?;
            Ok((system, Some(report), path))
        }

        EstimationPath::FullEstimation => {
            let (system, report) = run_estimation(system, case_dir, config, &manifest)?;
            Ok((system, Some(report), path))
        }

        EstimationPath::UserArHistoryStats => {
            let (system, report) = run_user_ar_estimation(system, case_dir, config, &manifest)?;
            Ok((system, Some(report), path))
        }
    }
}

fn run_estimation(
    system: System,
    case_dir: &Path,
    config: &Config,
    manifest: &FileManifest,
) -> Result<(System, EstimationReport), EstimationError> {
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

    // Use the system's stages, avoiding a re-parse of stages.json.
    let study_stages = system.stages();
    let season_map = system.policy_graph().season_map.as_ref();
    let max_order = config.estimation.max_order as usize;

    let prestudy = synthesize_prestudy_stages(study_stages, max_order, season_map);
    let stages: Vec<Stage> = study_stages
        .iter()
        .cloned()
        .chain(prestudy.iter().cloned())
        .collect();
    let stages = stages.as_slice();

    let observations = load_and_aggregate_observations(case_dir, study_stages, season_map)?;

    let seasonal_stats =
        estimate_seasonal_stats_with_season_map(&observations, stages, &hydro_ids, season_map)?;

    let (ar_estimates, estimation_report) = estimate_ar_coefficients_with_selection(
        &observations,
        &seasonal_stats,
        stages,
        &hydro_ids,
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: config.estimation.max_coefficient_magnitude,
            season_map,
            use_annual_component: matches!(
                config.estimation.order_selection,
                OrderSelectionMethod::PacfAnnual
            ),
        },
    )?;

    let correlation = if manifest.present(InputFile::ScenariosCorrelationJson) {
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

    let stats_rows = seasonal_stats_to_rows(&seasonal_stats, stages);
    let coeff_rows = ar_estimates_to_rows(&ar_estimates, stages);
    let annual_rows = ar_estimates_to_annual_rows(&ar_estimates, stages);

    let mut inflow_models = assemble_inflow_models(stats_rows, coeff_rows, annual_rows)?;
    // `stages` includes prestudy (see synthesize_prestudy_stages doc).
    let (stage_to_season, n_seasons) = resolve_stage_seasons(stages, season_map);
    populate_derived_residual_ratios(&mut inflow_models, &stage_to_season, n_seasons)?;

    let system = system.with_scenario_models(inflow_models, correlation)?;
    Ok((system, estimation_report))
}

/// Partial estimation: history + user seasonal stats present, AR coefficients absent.
///
/// User stats (`mean_m3s`, `std_m3s`) are preserved exactly for LP assembly; only
/// AR coefficients are estimated from history. The distinction vs [`run_estimation`]:
/// history-derived **fitting stats** drive the YW matrix construction, while
/// **user stats** drive the final `assemble_inflow_models` call.
fn run_partial_estimation(
    system: System,
    case_dir: &Path,
    config: &Config,
    manifest: &FileManifest,
) -> Result<(System, EstimationReport), EstimationError> {
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let study_stages = system.stages();
    let season_map = system.policy_graph().season_map.as_ref();
    let max_order = config.estimation.max_order as usize;

    let prestudy = synthesize_prestudy_stages(study_stages, max_order, season_map);
    let stages_owned: Vec<Stage> = study_stages
        .iter()
        .cloned()
        .chain(prestudy.iter().cloned())
        .collect();
    let stages = stages_owned.as_slice();

    // Aggregate against study_stages (the season map resolves observations there).
    let observations = load_and_aggregate_observations(case_dir, study_stages, season_map)?;

    if system.inflow_models().is_empty() {
        return Err(EstimationError::Load(ConstraintError {
            description: "manifest indicates inflow_seasonal_stats.parquet is present \
                          but system.inflow_models() is empty; \
                          no user stats available for partial estimation"
                .to_string(),
        }));
    }

    // Fitting stats: used only for the YW solve below, never for LP assembly.
    let fitting_stats =
        estimate_seasonal_stats_with_season_map(&observations, stages, &hydro_ids, season_map)?;

    let (ar_estimates, mut estimation_report) = estimate_ar_coefficients_with_selection(
        &observations,
        &fitting_stats,
        stages,
        &hydro_ids,
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: config.estimation.max_coefficient_magnitude,
            season_map,
            use_annual_component: matches!(
                config.estimation.order_selection,
                OrderSelectionMethod::PacfAnnual
            ),
        },
    )?;

    // Coverage compares over study_stages only — pre-study stages are excluded.
    let (white_noise_fallbacks, std_ratio_warnings) =
        validate_partial_estimation_coverage(&system, &fitting_stats, study_stages)?;

    let correlation = if manifest.present(InputFile::ScenariosCorrelationJson) {
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

    // LP assembly uses USER stats, not the fitting stats above.
    let mut stats_rows = user_stats_to_rows(&system);
    // Out-of-window lag seasons have no user stat; source their (mean, std) from
    // fitting_stats so PrecomputedPar::build gets a Tier-1 lag hit at the negative
    // stage_id rather than zeroing the lag. In-window wrap lags stay on the
    // Tier-2 → user-stat path. Empty for full-year.
    stats_rows.extend(prestudy_seasonal_rows(&fitting_stats, &prestudy));
    stats_rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id));
    let coeff_rows = ar_estimates_to_rows(&ar_estimates, stages);
    let annual_rows = ar_estimates_to_annual_rows(&ar_estimates, stages);
    let mut inflow_models = assemble_inflow_models(stats_rows, coeff_rows, annual_rows)?;
    // `stages` includes prestudy (see synthesize_prestudy_stages doc).
    let (stage_to_season, n_seasons) = resolve_stage_seasons(stages, season_map);
    populate_derived_residual_ratios(&mut inflow_models, &stage_to_season, n_seasons)?;

    estimation_report.white_noise_fallbacks = white_noise_fallbacks;
    estimation_report.std_ratio_warnings = std_ratio_warnings;

    let system = system.with_scenario_models(inflow_models, correlation)?;
    Ok((system, estimation_report))
}

/// Load inflow history from the case directory, gate each hydro's season
/// occurrences on full record coverage, and aggregate
/// the resulting samples to season resolution when a season map is present.
///
/// See [`resolve_coverage_gated_observations`] for the coverage gate itself
/// (record windows only — the initial-conditions conditioning layer is never
/// read here).
fn load_and_aggregate_observations(
    case_dir: &Path,
    stages: &[Stage],
    season_map: Option<&SeasonMap>,
) -> Result<Vec<(EntityId, NaiveDate, f64)>, EstimationError> {
    let history_path = case_dir.join("scenarios/inflow_history.parquet");
    let history = parse_inflow_history(&history_path)?;

    let (observations, skipped_partial) =
        resolve_coverage_gated_observations(&history, season_map, stages.first());
    log_skipped_partial_occurrences(&skipped_partial);

    if let Some(sm) = season_map {
        Ok(aggregate_observations_to_season(&observations, stages, sm)?)
    } else {
        Ok(observations)
    }
}

/// Return type of [`resolve_coverage_gated_observations`]:
/// `(observations, skipped_partial)`.
type CoverageGatedObservations = (Vec<(EntityId, NaiveDate, f64)>, BTreeMap<EntityId, usize>);

/// Coverage-gated occurrence resolution for one case's windowed inflow
/// history, over the **record layer only** — the
/// initial-conditions conditioning layer and its layered-merge helper are
/// never read here (owner gate: a conditioning window must change no fitted
/// PAR statistic).
///
/// For each hydro, record windows are grouped by season occurrence via
/// [`cast`]: an occurrence with `coverage == 1.0` contributes one sample
/// (`cast(...).value`, keyed at the occurrence's own start date); a partial
/// occurrence (`0.0 < coverage < 1.0`) is skipped and counted per hydro; a
/// zero-coverage occurrence is never enumerated (no window touches it, so it
/// never becomes a candidate).
///
/// When `season_map` or `stage_template` is unavailable, there is no season
/// occurrence to project onto: every row passes through unchanged.
// `coverage == 1.0` is an exact gate, not a tolerance shortcut: both
// `overlap` and `period.hours` are built from whole-day counts times 24.0, so
// a fully-covered occurrence's ratio is bit-exact 1.0 — see `cast`'s doc.
#[allow(clippy::float_cmp)]
fn resolve_coverage_gated_observations(
    history: &[InflowHistoryRow],
    season_map: Option<&SeasonMap>,
    stage_template: Option<&Stage>,
) -> CoverageGatedObservations {
    let (Some(season_map), Some(stage_template)) = (season_map, stage_template) else {
        let observations = history
            .iter()
            .map(|row| (row.hydro_id, row.start_date, row.value_m3s))
            .collect();
        return (observations, BTreeMap::new());
    };

    let mut windows_by_hydro: BTreeMap<EntityId, Vec<RealizedWindow>> = BTreeMap::new();
    for row in history {
        windows_by_hydro
            .entry(row.hydro_id)
            .or_default()
            .push(RealizedWindow {
                start_date: row.start_date,
                end_date: row.end_date,
                value_m3s: row.value_m3s,
            });
    }

    let mut observations = Vec::new();
    let mut skipped_partial = BTreeMap::new();

    // Cursor sweep requires ascending-by-start, disjoint windows; upstream guarantees:
    // `parse_inflow_history` sorts by (hydro_id, start_date), `validate_windowed_records` rejects overlaps.
    for (&hydro_id, windows) in &windows_by_hydro {
        debug_assert!(
            windows.is_sorted_by(|a, b| a.end_date <= b.start_date),
            "hydro {hydro_id}: windows must be ascending-by-start and disjoint for the cursor sweep"
        );
        let occurrences = discover_hydro_occurrences(season_map, stage_template, windows);
        let mut skip_count = 0usize;
        let mut lo = 0usize;

        for occurrence in &occurrences {
            while lo < windows.len() && windows[lo].end_date <= occurrence.start {
                lo += 1;
            }
            let mut hi = lo;
            while hi < windows.len() && windows[hi].start_date < occurrence.end {
                hi += 1;
            }

            let projection = cast(&windows[lo..hi], occurrence);

            if projection.coverage == 1.0 {
                observations.push((hydro_id, occurrence.start, projection.value));
            } else if projection.coverage > 0.0 {
                skip_count += 1;
            }
        }

        if skip_count > 0 {
            skipped_partial.insert(hydro_id, skip_count);
        }
    }

    observations.sort_by_key(|(id, date, _)| (id.0, *date));

    (observations, skipped_partial)
}

/// Every season-period occurrence overlapped by any of `windows`, deduplicated
/// by occurrence start date. Walks forward from each window's own occurrence
/// via [`next_season_period_window`] so a window straddling more than one
/// occurrence contributes every occurrence it touches, not just the first.
fn discover_hydro_occurrences(
    season_map: &SeasonMap,
    stage_template: &Stage,
    windows: &[RealizedWindow],
) -> Vec<SeasonPeriodWindow> {
    let mut discovered: BTreeMap<NaiveDate, SeasonPeriodWindow> = BTreeMap::new();

    for window in windows {
        let Some(mut occurrence) = occurrence_containing(season_map, stage_template, window) else {
            continue;
        };

        while occurrence.start < window.end_date {
            let key = occurrence.start;
            discovered.entry(key).or_insert_with(|| SeasonPeriodWindow {
                start: occurrence.start,
                end: occurrence.end,
                hours: occurrence.hours,
            });

            let Some(season_id) = season_map.season_for_date(occurrence.start) else {
                break;
            };
            let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
                break;
            };
            let Some(next) = next_season_period_window(season_map, season_def, &occurrence) else {
                break;
            };
            occurrence = next;
        }
    }

    discovered.into_values().collect()
}

/// The season-period occurrence containing `window.start_date`, anchored on
/// `window`'s own `[start_date, end_date)` span — the natural stage analogue
/// [`season_period_window`] expects for disambiguating a cycle-crossing
/// candidate year. `stage_template` donates every field `season_period_window`
/// does not read (only `start_date`/`end_date` are read); any real `Stage`
/// works as the donor.
fn occurrence_containing(
    season_map: &SeasonMap,
    stage_template: &Stage,
    window: &RealizedWindow,
) -> Option<SeasonPeriodWindow> {
    let season_id = season_map.season_for_date(window.start_date)?;
    let season_def = season_map.seasons.iter().find(|s| s.id == season_id)?;

    let mut probe = stage_template.clone();
    probe.start_date = window.start_date;
    probe.end_date = window.end_date;

    Some(season_period_window(season_map, season_def, &probe))
}

/// Emit one aggregate info-level diagnostic summarizing partial-coverage
/// occurrences skipped during estimation observation loading; a
/// no-op when nothing was skipped. Estimation's `run_*` pipelines carry no
/// `ValidationContext` (that lives only in [`estimate_from_history`]'s
/// structural pre-check), so `tracing::info!` is the channel already used for
/// this file's other estimation-time diagnostics (see
/// [`validate_partial_estimation_coverage`]'s `tracing::warn!`).
fn log_skipped_partial_occurrences(skipped_partial: &BTreeMap<EntityId, usize>) {
    if skipped_partial.is_empty() {
        return;
    }

    let total: usize = skipped_partial.values().sum();
    let per_hydro: Vec<String> = skipped_partial
        .iter()
        .map(|(hydro_id, count)| format!("hydro {hydro_id}: {count}"))
        .collect();

    tracing::info!(
        "estimation observation loading skipped {total} partial-coverage season \
         occurrence(s) across {} hydro(s) ({})",
        skipped_partial.len(),
        per_hydro.join(", ")
    );
}

/// Return type of [`validate_partial_estimation_coverage`]:
/// `(white_noise_fallbacks, std_ratio_warnings)`.
type CoverageCheckResult = (Vec<EntityId>, Vec<StdRatioDivergence>);

/// Validate bidirectional user-vs-estimated coverage and emit advisory warnings.
///
/// Errors only on hard coverage failures (an estimated hydro missing user stats).
fn validate_partial_estimation_coverage(
    system: &System,
    fitting_stats: &[SeasonalStats],
    stages: &[Stage],
) -> Result<CoverageCheckResult, EstimationError> {
    let estimated_hydro_ids: HashSet<EntityId> =
        fitting_stats.iter().map(|s| s.entity_id).collect();
    let user_stats_hydro_ids: HashSet<EntityId> =
        system.inflow_models().iter().map(|m| m.hydro_id).collect();

    // Direction A: AR estimated but no user stats → hard error.
    let mut missing_stats: Vec<EntityId> = estimated_hydro_ids
        .difference(&user_stats_hydro_ids)
        .copied()
        .collect();
    missing_stats.sort();
    if !missing_stats.is_empty() {
        let ids: Vec<String> = missing_stats.iter().map(|id| id.0.to_string()).collect();
        return Err(EstimationError::Load(ConstraintError {
            description: format!(
                "partial estimation: AR coefficients were estimated for hydro(s) \
                     [{ids}] but inflow_seasonal_stats.parquet has no entry for them; \
                     all hydros with estimated AR must have user-provided stats",
                ids = ids.join(", ")
            ),
        }));
    }

    // Direction B: user stats but no AR estimated → white noise fallback.
    let mut white_noise_fallbacks: Vec<EntityId> = user_stats_hydro_ids
        .difference(&estimated_hydro_ids)
        .copied()
        .collect();
    white_noise_fallbacks.sort();

    let std_ratio_warnings = check_std_ratio_divergence(system, fitting_stats, stages);
    for w in &std_ratio_warnings {
        tracing::warn!(
            "hydro {} season {}->{} std ratio diverges {:.1}x between \
             user ({:.2}) and estimated ({:.2})",
            w.hydro_id.0,
            w.season_a,
            w.season_b,
            w.divergence,
            w.user_ratio,
            w.estimated_ratio
        );
    }

    Ok((white_noise_fallbacks, std_ratio_warnings))
}

/// `UserArHistoryStats`: history + user AR coefficients present, seasonal stats absent.
///
/// Seasonal stats are estimated from history (driving both LP assembly and
/// correlation estimation); AR coefficients are loaded from the user file and
/// preserved bitwise — **no re-estimation from history is performed**. The
/// returned [`EstimationReport`] carries an empty `entries` map and method
/// `"user_provided"` to signal that no AR estimation ran.
fn run_user_ar_estimation(
    system: System,
    case_dir: &Path,
    config: &Config,
    manifest: &FileManifest,
) -> Result<(System, EstimationReport), EstimationError> {
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let stages = system.stages();
    let season_map = system.policy_graph().season_map.as_ref();
    let max_order = config.estimation.max_order as usize;

    let prestudy = synthesize_prestudy_stages(stages, max_order, season_map);
    let extended: Vec<Stage> = stages
        .iter()
        .cloned()
        .chain(prestudy.iter().cloned())
        .collect();
    let extended = extended.as_slice();

    let observations = load_and_aggregate_observations(case_dir, stages, season_map)?;

    let seasonal_stats =
        estimate_seasonal_stats_with_season_map(&observations, extended, &hydro_ids, season_map)?;

    // Read AR from file: system.inflow_models() is empty on this path (no user stats).
    let ar_path = case_dir.join("scenarios/inflow_ar_coefficients.parquet");
    let user_ar_rows = parse_inflow_ar_coefficients(&ar_path)?;

    let user_ar_estimates = ar_rows_to_estimates(&user_ar_rows, stages);

    let correlation = if manifest.present(InputFile::ScenariosCorrelationJson) {
        system.correlation().clone()
    } else {
        estimate_correlation_with_season_map(
            &observations,
            &user_ar_estimates,
            &seasonal_stats,
            extended,
            &hydro_ids,
            season_map,
        )?
    };

    // History stats drive mean_m3s/std_m3s; user AR rows drive ar_coefficients
    // (residual_std_ratio is derived below, not read from the user file).
    let stats_rows = seasonal_stats_to_rows(&seasonal_stats, extended);

    let mut inflow_models = assemble_inflow_models(stats_rows, user_ar_rows, vec![])?;
    // `extended` includes prestudy (see synthesize_prestudy_stages doc).
    let (stage_to_season, n_seasons) = resolve_stage_seasons(extended, season_map);
    populate_derived_residual_ratios(&mut inflow_models, &stage_to_season, n_seasons)?;

    let estimation_report = EstimationReport {
        entries: BTreeMap::new(),
        method: "user_provided".to_string(),
        white_noise_fallbacks: Vec::new(),
        std_ratio_warnings: Vec::new(),
    };

    let system = system.with_scenario_models(inflow_models, correlation)?;
    Ok((system, estimation_report))
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
/// lag 2, …). The innovation scale is derived downstream by
/// [`crate::scenarios::populate_derived_residual_ratios`] on the assembled
/// `InflowModel`s — it is not part of the estimate.
///
/// The result is sorted by `(hydro_id, season_id)` ascending, matching the
/// canonical ordering expected by `estimate_correlation_with_season_map`.
fn ar_rows_to_estimates(
    rows: &[InflowArCoefficientRow],
    stages: &[Stage],
) -> Vec<ArCoefficientEstimate> {
    let stage_id_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    // Rows are pre-sorted by (hydro_id, stage_id, lag), so the first stage per
    // season is canonical; later same-season stages are duplicates emitted by
    // ar_estimates_to_rows and are skipped.
    let mut first_stage: HashMap<(EntityId, usize), i32> = HashMap::new();

    // BTreeMap for deterministic (hydro_id, season_id) output ordering.
    let mut groups: BTreeMap<(EntityId, usize), Vec<f64>> = BTreeMap::new();

    for row in rows {
        let Some(&season_id) = stage_id_to_season.get(&row.stage_id) else {
            continue;
        };

        let key = (row.hydro_id, season_id);

        let canonical_stage = first_stage.entry(key).or_insert(row.stage_id);
        if *canonical_stage != row.stage_id {
            continue;
        }

        groups.entry(key).or_default().push(row.coefficient);
    }

    groups
        .into_iter()
        .map(
            |((hydro_id, season_id), coefficients)| ArCoefficientEstimate {
                hydro_id,
                season_id,
                coefficients,
                annual: None,
            },
        )
        .collect()
}

/// Extract user-provided seasonal stats from `system.inflow_models()` as
/// [`InflowSeasonalStatsRow`] entries.
///
/// Each `InflowModel` in the system contributes one row with its `mean_m3s`
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

/// Flag hydros whose consecutive-season std ratios diverge between the user and
/// estimated profiles, advisory only.
///
/// For each hydro in both user stats and `fitting_stats`, over consecutive season
/// pairs `(m, (m+1) % n)`, pushes a [`StdRatioDivergence`] when the symmetric
/// ratio-of-ratios exceeds `2.0`. Near-zero denominators (`< 1e-12`) are skipped.
fn check_std_ratio_divergence(
    system: &System,
    fitting_stats: &[SeasonalStats],
    stages: &[Stage],
) -> Vec<StdRatioDivergence> {
    let stage_id_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    // First entry wins: stages sharing a season carry the same std.
    let mut user_std: BTreeMap<(EntityId, usize), f64> = BTreeMap::new();
    for m in system.inflow_models() {
        let Some(&season_id) = stage_id_to_season.get(&m.stage_id) else {
            continue;
        };
        user_std.entry((m.hydro_id, season_id)).or_insert(m.std_m3s);
    }

    let mut est_std: BTreeMap<(EntityId, usize), f64> = BTreeMap::new();
    for s in fitting_stats {
        let Some(&season_id) = stage_id_to_season.get(&s.stage_id) else {
            continue;
        };
        est_std.entry((s.entity_id, season_id)).or_insert(s.std);
    }

    let user_hydros: BTreeSet<EntityId> = user_std.keys().map(|(h, _)| *h).collect();
    let est_hydros: BTreeSet<EntityId> = est_std.keys().map(|(h, _)| *h).collect();
    let common_hydros: Vec<EntityId> = user_hydros.intersection(&est_hydros).copied().collect();

    let mut warnings: Vec<StdRatioDivergence> = Vec::new();

    for hydro_id in common_hydros {
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

            if u_b.abs() < 1e-12 || e_b.abs() < 1e-12 {
                continue;
            }

            let ratio_user = u_a / u_b;
            let ratio_est = e_a / e_b;

            // Guard the ratio-of-ratios divergence below against division by zero.
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

    warnings.sort_by_key(|w| (w.hydro_id, w.season_a));
    warnings
}

/// The first (lowest-`id`) non-negative study stage carrying a `season_id`,
/// paired with that season — the anchor [`synthesize_prestudy_stages`] and
/// [`resolve_model_stage_seasons`] both back-walk pre-study season ids from.
fn first_study_stage_with_season(stages: &[Stage]) -> Option<(&Stage, usize)> {
    stages
        .iter()
        .filter(|s| s.id >= 0 && s.season_id.is_some())
        .min_by_key(|s| s.id)
        .and_then(|s| s.season_id.map(|season| (s, season)))
}

/// Synthesize pre-study stages covering the PAR(p) lag window for a
/// partial-year study (one whose horizon is narrower than the seasonal cycle).
///
/// For a study starting mid-cycle (e.g. a monthly model spanning September–
/// December, seasons 8–11), the first study stage's AR lags reach back into
/// months that have no study stage (August, July, …). Without a stage carrying
/// those seasons, the season-aware estimators have no place to attach the
/// out-of-window lag statistics, and the precompute silently zeroes them.
///
/// This helper emits one pre-study [`Stage`] per
/// lag `k = 1..=min(max_order, cycle_len-1)`, at the descending negative id
/// `first_study_stage.id - k` and the season [`prestudy_season_for_lag`]
/// resolves, but **only** when that season is not already among the study
/// stages' seasons. A full-year study therefore synthesizes nothing.
fn synthesize_prestudy_stages(
    stages: &[Stage],
    max_order: usize,
    season_map: Option<&SeasonMap>,
) -> Vec<Stage> {
    let Some(sm) = season_map else {
        return Vec::new();
    };
    let cycle_len = sm.seasons.len();
    if max_order == 0 || cycle_len == 0 {
        return Vec::new();
    }

    let Some((first, first_season)) = first_study_stage_with_season(stages) else {
        return Vec::new();
    };

    let study_seasons: HashSet<usize> = stages.iter().filter_map(|s| s.season_id).collect();

    let lag_window = max_order.min(cycle_len - 1);
    let mut synthetic = Vec::with_capacity(lag_window);

    for k in 1..=lag_window {
        let season_k = prestudy_season_for_lag(first_season, k, cycle_len);
        if study_seasons.contains(&season_k) {
            // In-window wrap lags are served by the cycle-correct Tier-2 / user-stat path.
            continue;
        }

        // [start_k, end_k): k and k-1 months before first.start_date give a half-open span.
        let (Some(start_k), Some(end_k)) = (
            first
                .start_date
                .checked_sub_months(Months::new(u32::try_from(k).unwrap_or(u32::MAX))),
            first
                .start_date
                .checked_sub_months(Months::new(u32::try_from(k - 1).unwrap_or(u32::MAX))),
        ) else {
            continue;
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let id = first.id - k as i32;

        // Override only identity/dates/season; estimation keys off id + season_id.
        let mut stage = first.clone();
        stage.index = 0;
        stage.id = id;
        stage.start_date = start_k;
        stage.end_date = end_k;
        stage.season_id = Some(season_k);
        synthetic.push(stage);
    }

    synthetic
}

/// The season `lag` calendar positions before `first_season`, modular on
/// `cycle_len` — the back-walk convention [`synthesize_prestudy_stages`] and
/// [`resolve_model_stage_seasons`] both stitch pre-study stage ids to.
#[must_use]
fn prestudy_season_for_lag(first_season: usize, lag: usize, cycle_len: usize) -> usize {
    (first_season + cycle_len - (lag % cycle_len)) % cycle_len
}

/// Extends [`resolve_stage_seasons`]'s map with an entry for every id in
/// `model_stage_ids` not already covered by `stages`, reproducing the
/// estimation-time stitched map [`synthesize_prestudy_stages`] builds without
/// re-running synthesis. An id that is neither a declared stage nor derivable
/// (no non-negative study stage carries a season, or the id is not below that
/// stage's) is left unmapped.
#[must_use]
pub fn resolve_model_stage_seasons(
    stages: &[Stage],
    model_stage_ids: impl Iterator<Item = i32>,
    season_map: &SeasonMap,
) -> (HashMap<i32, usize>, usize) {
    let (mut stage_to_season, n_seasons) = resolve_stage_seasons(stages, Some(season_map));
    let cycle_len = season_map.seasons.len();
    if cycle_len == 0 {
        return (stage_to_season, n_seasons);
    }

    let Some((first, first_season)) = first_study_stage_with_season(stages) else {
        return (stage_to_season, n_seasons);
    };
    let first_id = first.id;

    let (dense_index, _) = season_dense_index(stages, Some(season_map));

    for model_stage_id in model_stage_ids {
        if stage_to_season.contains_key(&model_stage_id) || model_stage_id >= first_id {
            continue;
        }
        let Ok(lag) = usize::try_from(first_id - model_stage_id) else {
            continue;
        };
        let raw_season = prestudy_season_for_lag(first_season, lag, cycle_len);
        if let Some(&ordinal) = dense_index.get(&raw_season) {
            stage_to_season.insert(model_stage_id, ordinal);
        }
    }

    (stage_to_season, n_seasons)
}

/// Emit history-derived seasonal rows for the synthetic pre-study stages of a
/// partial-year study.
///
/// Each out-of-window lag season is covered by exactly one synthetic pre-study
/// stage (negative `id`). Because the season-aware fitting keys each season to
/// its lowest-`start_date` stage, and the synthetic pre-study stages sort
/// before every study stage, the corresponding [`SeasonalStats`] entry already
/// carries that negative `stage_id`. This helper selects those entries and
/// emits one [`InflowSeasonalStatsRow`] per (hydro, pre-study stage), giving
/// `PrecomputedPar::build` a Tier-1 lag hit at the negative `stage_id`.
///
/// Returns an empty `Vec` when `prestudy` is empty (full-year studies).
fn prestudy_seasonal_rows(
    fitting_stats: &[SeasonalStats],
    prestudy: &[Stage],
) -> Vec<InflowSeasonalStatsRow> {
    if prestudy.is_empty() {
        return Vec::new();
    }
    let prestudy_ids: HashSet<i32> = prestudy.iter().map(|s| s.id).collect();
    let mut rows: Vec<InflowSeasonalStatsRow> = fitting_stats
        .iter()
        .filter(|s| prestudy_ids.contains(&s.stage_id))
        .map(|s| InflowSeasonalStatsRow {
            hydro_id: s.entity_id,
            stage_id: s.stage_id,
            mean_m3s: s.mean,
            std_m3s: s.std,
        })
        .collect();
    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id));
    rows
}

/// Index stage ids by their `season_id`, skipping stages without one.
fn build_season_to_stages(stages: &[Stage]) -> HashMap<usize, Vec<i32>> {
    let mut season_to_stages: HashMap<usize, Vec<i32>> = HashMap::new();
    for stage in stages {
        if let Some(sid) = stage.season_id {
            season_to_stages.entry(sid).or_default().push(stage.id);
        }
    }
    season_to_stages
}

/// Convert [`SeasonalStats`] to [`InflowSeasonalStatsRow`], expanding each
/// per-season estimate to every stage sharing its `season_id` so that
/// [`cobre_stochastic::PrecomputedPar`] finds a model at every stage index.
///
/// Pre-study stages (negative `id`) are included in the expansion, emitting rows
/// at their negative `stage_id` for direct lag-stage hits.
fn seasonal_stats_to_rows(
    stats: &[SeasonalStats],
    stages: &[Stage],
) -> Vec<InflowSeasonalStatsRow> {
    let stage_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    let season_to_stages = build_season_to_stages(stages);

    let mut rows = Vec::with_capacity(stats.len() * 10);
    for s in stats {
        if let Some(&season_id) = stage_to_season.get(&s.stage_id)
            && let Some(stage_ids) = season_to_stages.get(&season_id)
        {
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
        // No season mapping: emit the stat's own stage_id unexpanded.
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
/// each per-season estimate to every stage sharing its `season_id` (covering the
/// full horizon, not just the season's first occurrence).
///
/// Pre-study stages (negative `id`) are included, emitting coefficient rows at
/// their negative `stage_id` for direct lag lookups.
fn ar_estimates_to_rows(
    ar_estimates: &[ArCoefficientEstimate],
    stages: &[Stage],
) -> Vec<InflowArCoefficientRow> {
    let season_to_stages = build_season_to_stages(stages);

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
                });
            }
        }
    }

    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id, r.lag));

    rows
}

/// Convert [`ArCoefficientEstimate`] to [`InflowAnnualComponentRow`], expanding
/// per-season annual components to every stage that shares the same `season_id`.
///
/// Estimates without an `annual` field (`annual.is_none()`) are silently skipped,
/// so the function is safe to call for classical-PAR estimates (the result will be
/// an empty `Vec`).
fn ar_estimates_to_annual_rows(
    ar_estimates: &[ArCoefficientEstimate],
    stages: &[Stage],
) -> Vec<InflowAnnualComponentRow> {
    let season_to_stages = build_season_to_stages(stages);

    let mut rows: Vec<InflowAnnualComponentRow> = Vec::new();

    for est in ar_estimates {
        let Some(ref ann) = est.annual else {
            continue;
        };
        let Some(stage_ids) = season_to_stages.get(&est.season_id) else {
            continue;
        };
        for &stage_id in stage_ids {
            rows.push(InflowAnnualComponentRow {
                hydro_id: est.hydro_id,
                stage_id,
                annual_coefficient: ann.coefficient,
                annual_mean_m3s: ann.mean_m3s,
                annual_std_m3s: ann.std_m3s,
            });
        }
    }

    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id));

    rows
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
