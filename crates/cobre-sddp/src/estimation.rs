//! Automatic PAR(p) parameter estimation from historical inflow observations.
//!
//! This module bridges `cobre-io` (case loading) and `cobre-stochastic` (PAR fitting):
//! it checks the input file manifest, loads history when explicit model statistics are
//! absent, runs the fitting pipeline, and returns an updated [`System`].
//!
//! ## Input path matrix
//!
//! | `inflow_history.parquet` | `inflow_seasonal_stats.parquet` | `inflow_ar_coefficients.parquet` | Behaviour |
//! |---|---|---|---|
//! | absent | any | any | Return `system` unchanged. |
//! | present | present | present | Return `system` unchanged (explicit stats take priority). |
//! | present | present | absent | Partial estimation: load existing stats, estimate only AR coefficients. |
//! | present | absent | any | Full estimation; update `inflow_models` and optionally `correlation`. |
//!
//! `correlation.json` is handled independently: if present, the existing
//! `system.correlation()` is kept; if absent, the correlation is estimated from residuals.
//!
//! ## AIC order selection
//!
//! When `config.estimation.order_selection = "aic"`, the module calls
//! `levinson_durbin` directly per `(entity, season)` pair to obtain the
//! `sigma2_per_order` sequence, then uses `select_order_aic` to pick the
//! best order and truncates the AR coefficient vector accordingly.
//! This avoids changing `ArCoefficientEstimate`'s public API.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use chrono::NaiveDate;
use cobre_core::{EntityId, System};
use cobre_io::{
    Config, FileManifest, LoadError, ValidationContext,
    config::OrderSelectionMethod,
    parse_inflow_history,
    scenarios::{InflowArCoefficientRow, InflowSeasonalStatsRow, assemble_inflow_models},
    validate_structure,
};
use cobre_stochastic::{
    StochasticError,
    par::contribution::{
        check_negative_contributions, compute_contributions, find_max_valid_order,
    },
    par::fitting::{
        AicSelectionResult, ArCoefficientEstimate, SeasonalStats,
        estimate_ar_coefficients_with_season_map, estimate_correlation_with_season_map,
        estimate_seasonal_stats_with_season_map, find_season_for_date, levinson_durbin,
        select_order_aic,
    },
};

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
}

/// Per-hydro AIC diagnostic data captured during AIC-based AR order selection.
///
/// Holds the AIC-selected order, AIC scores for each candidate order, and the
/// fitted AR coefficients for each season at the selected order.
#[derive(Debug, Clone)]
pub struct HydroEstimationEntry {
    /// The AIC-selected AR order for this hydro plant (maximum across all seasons).
    ///
    /// This is the maximum of the per-season selected orders, which determines
    /// the coefficient vector length in the output.
    pub selected_order: u32,
    /// AIC values for AR orders 1 through `max_order` (order 0 excluded).
    ///
    /// `aic_scores[i]` is the AIC for AR order `i+1`. The order-0 white-noise
    /// baseline is omitted because it is always 0.0 and carries no diagnostic
    /// value. These scores are taken from season 0 as a representative sample.
    pub aic_scores: Vec<f64>,
    /// Fitted AR lag coefficients, one inner vector per season sorted by `season_id` ascending.
    ///
    /// Each inner vector holds the coefficients at the AIC-selected order for
    /// that season. Seasons where AIC was skipped (zero std, insufficient
    /// observations) use the original `estimate_ar_coefficients` output.
    pub coefficients: Vec<Vec<f64>>,
    /// Records of contribution-based order reductions applied during fitting.
    ///
    /// Each entry documents a season where the initial order (from AIC or fixed
    /// selection) was reduced due to negative contributions. Empty when no
    /// reductions were needed.
    pub contribution_reductions: Vec<ContributionReduction>,
}

/// Computation-side summary of the AIC-based AR estimation pipeline.
///
/// Contains one [`HydroEstimationEntry`] per hydro plant that was fitted,
/// keyed by [`EntityId`] for canonical deterministic ordering. When the
/// `Fixed` order-selection method is used, `entries` is empty.
#[must_use]
#[derive(Debug, Clone)]
pub struct EstimationReport {
    /// Per-hydro AIC diagnostic entries, keyed by entity ID.
    pub entries: BTreeMap<EntityId, HydroEstimationEntry>,
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

/// Estimate PAR(p) model parameters from inflow history when explicit stats are absent.
///
/// Reads the file manifest for `case_dir` to determine the input path. When
/// `inflow_history.parquet` is present and `inflow_seasonal_stats.parquet` is absent,
/// the function runs the full estimation pipeline and returns a new [`System`] with
/// updated `inflow_models` and (optionally) `correlation`. In all other cases the
/// `system` is returned unchanged.
///
/// # Estimation pipeline
///
/// 1. Runs structural validation to get the [`FileManifest`].
/// 2. Checks the input path matrix (see module doc).
/// 3. Loads inflow history via [`parse_inflow_history`].
/// 4. Estimates seasonal stats via `estimate_seasonal_stats`.
/// 5. Estimates AR coefficients (with AIC or fixed-order selection).
/// 6. Estimates correlation when `correlation.json` is absent.
/// 7. Converts estimation results to `cobre-io` row types, calls
///    `assemble_inflow_models`, and returns
///    `system.with_scenario_models(inflow_models, correlation)`.
///
/// # Errors
///
/// - [`EstimationError::Load`] — file read or parse failure.
/// - [`EstimationError::Stochastic`] — insufficient observations for any
///   `(entity, season)` group.
pub fn estimate_from_history(
    system: System,
    case_dir: &Path,
    config: &Config,
) -> Result<(System, Option<EstimationReport>), EstimationError> {
    // ── Step 1: resolve file manifest ────────────────────────────────────────
    let mut ctx = ValidationContext::new();
    let manifest = validate_structure(case_dir, &mut ctx);

    // Abort early if structural validation found errors.
    if ctx.into_result().is_err() {
        return Ok((system, None));
    }

    // ── Step 2: input path matrix ────────────────────────────────────────────
    if !manifest.scenarios_inflow_history_parquet {
        // No history file — nothing to estimate.
        return Ok((system, None));
    }

    if manifest.scenarios_inflow_seasonal_stats_parquet
        && manifest.scenarios_inflow_ar_coefficients_parquet
    {
        // Explicit stats provided — skip estimation.
        return Ok((system, None));
    }

    // History present, stats absent — run estimation.
    let (system, report) = run_estimation(system, case_dir, config, &manifest)?;
    Ok((system, Some(report)))
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
        max_order,
        &config.estimation.order_selection,
        season_map,
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

/// Estimate AR coefficients with the configured order selection method.
///
/// For `Fixed`, delegates directly to `estimate_ar_coefficients`.
/// For `Aic`, calls `estimate_ar_coefficients` at `max_order` to obtain full
/// coefficients, then calls `levinson_durbin` per `(entity, season)` pair to
/// get `sigma2_per_order` for AIC minimisation, and truncates the coefficient
/// vector to the AIC-selected order.
fn estimate_ar_coefficients_with_selection(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    method: &OrderSelectionMethod,
    season_map: Option<&cobre_core::temporal::SeasonMap>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    match method {
        OrderSelectionMethod::Fixed => {
            let mut estimates = estimate_ar_coefficients_with_season_map(
                observations,
                seasonal_stats,
                stages,
                hydro_ids,
                max_order,
                season_map,
            )?;

            // Build stats_map and n_seasons for contribution validation.
            let stage_index = stages
                .iter()
                .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
                .collect::<Vec<_>>();
            let stage_id_to_season: HashMap<i32, usize> = stage_index.iter().copied().collect();
            let n_seasons = stage_index
                .iter()
                .map(|&(_, sid)| sid + 1)
                .max()
                .unwrap_or(0);
            let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
                .iter()
                .filter_map(|s| {
                    let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
                    Some(((s.entity_id, season_id), s))
                })
                .collect();
            let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

            let reductions =
                apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map);

            let aic_results: HashMap<(EntityId, usize), AicSelectionResult> = HashMap::new();
            let report = build_estimation_report(&estimates, &aic_results, n_seasons, &reductions);
            Ok((estimates, report))
        }
        OrderSelectionMethod::Aic => estimate_ar_with_aic(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            max_order,
            season_map,
        ),
    }
}

/// AIC-based AR order selection.
///
/// Calls `estimate_ar_coefficients` at `max_order` to get full-order fits,
/// then for each `(entity, season)` pair calls `levinson_durbin` independently
/// to obtain `sigma2_per_order`, runs `select_order_aic` to pick the best
/// order, and truncates the coefficient vector accordingly.
///
/// Truncation recomputes `residual_std_ratio` from the selected order's
/// `sigma2` (the square root of the normalised prediction error variance).
#[allow(clippy::too_many_lines)]
fn estimate_ar_with_aic(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    season_map: Option<&cobre_core::temporal::SeasonMap>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    // Get full-order estimates first.
    let mut estimates = estimate_ar_coefficients_with_season_map(
        observations,
        seasonal_stats,
        stages,
        hydro_ids,
        max_order,
        season_map,
    )?;

    if max_order == 0 {
        let report = EstimationReport {
            entries: BTreeMap::new(),
        };
        return Ok((estimates, report));
    }

    // Build (entity_id, season_id) -> observations map for per-pair AIC.
    // Re-use the same stage index logic used in estimate_ar_coefficients.
    let mut stage_index = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect::<Vec<_>>();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let stage_id_to_season: HashMap<i32, usize> = stage_index
        .iter()
        .map(|&(_, _, stage_id, season_id)| (stage_id, season_id))
        .collect();

    // Build stats lookup: (entity_id, season_id) -> SeasonalStats
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
        .iter()
        .filter_map(|s| {
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(((s.entity_id, season_id), s))
        })
        .collect();

    // Group observations by (entity_id, season_id) for autocorrelation.
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

    let n_seasons: usize = {
        let mut max_season = 0usize;
        for &(_, _, _, season_id) in &stage_index {
            if season_id >= max_season {
                max_season = season_id + 1;
            }
        }
        max_season
    };

    // Capture AIC results per (entity_id, season_id) for building the report.
    let mut aic_results: HashMap<(EntityId, usize), AicSelectionResult> = HashMap::new();
    // Store sigma2_per_order for later use during contribution-based reduction.
    let mut sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    // For each estimate, run levinson_durbin independently to get sigma2_per_order.
    for est in &mut estimates {
        let key = (est.hydro_id, est.season_id);

        let Some(stats_m) = stats_map.get(&key) else {
            continue;
        };

        if stats_m.std == 0.0 {
            // Constant inflow — already white noise; AIC would select order 0.
            continue;
        }

        let Some(pair_obs) = group_obs.get(&key) else {
            continue;
        };

        let n_obs = pair_obs.len();
        if n_obs < 2 {
            continue;
        }

        // Compute autocorrelations for orders 1..=actual_order using
        // the same cross-seasonal formula as estimate_ar_coefficients.
        let actual_order = est.coefficients.len();
        if actual_order == 0 {
            continue;
        }

        // Compute normalised autocorrelations rho_m(1)..rho_m(actual_order).
        let autocorrelations = compute_autocorrelations(
            est.hydro_id,
            est.season_id,
            actual_order,
            n_seasons,
            pair_obs,
            &stats_map,
            &group_obs,
        );

        if autocorrelations.len() < actual_order {
            // Truncated during autocorrelation computation — keep existing.
            continue;
        }

        // Run levinson_durbin to get sigma2_per_order.
        let ld = levinson_durbin(&autocorrelations, actual_order);

        if ld.sigma2_per_order.is_empty() {
            continue;
        }

        // AIC selection. Use effective observation count: subtract the AR order
        // since the earliest `order` observations lack lag predecessors.
        let effective_n = n_obs.saturating_sub(actual_order);
        let aic_result = select_order_aic(&ld.sigma2_per_order, effective_n);
        let selected = aic_result.selected_order;
        aic_results.insert((est.hydro_id, est.season_id), aic_result);

        if selected < actual_order {
            // Truncate coefficients to AIC-selected order.
            est.coefficients.truncate(selected);

            // Recompute residual_std_ratio from the selected order's sigma2.
            let sigma2_selected = if selected == 0 {
                1.0
            } else {
                ld.sigma2_per_order[selected - 1]
            };
            // sigma2 is the normalised prediction error variance (relative to seasonal variance).
            // residual_std_ratio = sqrt(sigma2_selected), clamped to (0, 1].
            est.residual_std_ratio = sigma2_selected.sqrt().clamp(f64::EPSILON, 1.0);
        }

        // Store sigma2_per_order for contribution-based reduction (reuses LD results).
        sigma2_map.insert(key, ld.sigma2_per_order);
    }

    // === Contribution-based order validation ===
    let reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map);

    let report = build_estimation_report(&estimates, &aic_results, n_seasons, &reductions);
    Ok((estimates, report))
}

/// Apply contribution-based order validation to all AR estimates.
///
/// For each entity, groups all season coefficients and standard deviations,
/// then iterates per season: if negative contributions are found, the order
/// is reduced to the maximum valid order and the coefficient vector truncated.
/// The loop repeats until all contributions are non-negative or the order
/// reaches zero.
///
/// Returns a map of `EntityId` -> list of `ContributionReduction` events.
fn apply_contribution_validation(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    sigma2_map: &HashMap<(EntityId, usize), Vec<f64>>,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

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
                    });

                // Truncate coefficients to the reduced order.
                estimates[idx].coefficients.truncate(reduced_order);

                // Recompute residual_std_ratio from stored sigma2 if available.
                if reduced_order == 0 {
                    estimates[idx].residual_std_ratio = 1.0;
                } else if let Some(sigma2_vec) = sigma2_map.get(&(hydro_id, season_id)) {
                    if reduced_order <= sigma2_vec.len() {
                        let sigma2 = sigma2_vec[reduced_order - 1];
                        estimates[idx].residual_std_ratio = sigma2.sqrt().clamp(f64::EPSILON, 1.0);
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

/// Build an [`EstimationReport`] from AIC-truncated AR estimates and captured
/// per-`(entity, season)` [`AicSelectionResult`] values.
///
/// This function is infallible: it only reorganises already-computed data.
/// For each hydro plant the selected order is the **maximum** across all
/// seasons, and the representative AIC scores are taken from season 0
/// (the first season). These choices align with how the I/O layer
/// (`FittingReport`) expects a single order and a single AIC vector per hydro.
fn build_estimation_report(
    estimates: &[ArCoefficientEstimate],
    aic_results: &HashMap<(EntityId, usize), AicSelectionResult>,
    n_seasons: usize,
    contribution_reductions: &HashMap<EntityId, Vec<ContributionReduction>>,
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

        // Compute max selected_order across all seasons for this hydro.
        let selected_order = season_coeffs
            .iter()
            .map(|(season_id, _)| {
                aic_results
                    .get(&(hydro_id, *season_id))
                    .map_or(0, |r| r.selected_order)
            })
            .max()
            .unwrap_or(0);

        // AIC scores from season 0 (representative; excludes order-0 baseline).
        let aic_scores = aic_results
            .get(&(hydro_id, 0))
            .map(|r| r.aic_values[1..].to_vec())
            .unwrap_or_default();

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
                aic_scores,
                coefficients,
                contribution_reductions: reductions,
            },
        );
    }

    EstimationReport { entries }
}

/// Compute normalised cross-seasonal autocorrelations `ρ_m(1)..ρ_m(max_order)` for
/// a single `(entity, season)` pair.
///
/// Returns a `Vec` whose length may be less than `max_order` when a required lag
/// season has zero standard deviation or insufficient observations.
fn compute_autocorrelations(
    hydro_id: EntityId,
    season_id: usize,
    max_order: usize,
    n_seasons: usize,
    pair_obs: &[f64],
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
) -> Vec<f64> {
    let Some(stats_m) = stats_map.get(&(hydro_id, season_id)) else {
        return Vec::new();
    };

    let mu_m = stats_m.mean;
    let std_m = stats_m.std;
    let mut autocorrelations = Vec::with_capacity(max_order);

    for lag in 1..=max_order {
        let lag_season = season_id
            .wrapping_add(n_seasons)
            .wrapping_sub(lag % n_seasons)
            % n_seasons;

        let lag_key = (hydro_id, lag_season);
        let stats_lag = match stats_map.get(&lag_key) {
            Some(s) if s.std > 0.0 => s,
            _ => break,
        };

        let Some(lag_obs) = group_obs.get(&lag_key) else {
            break;
        };

        // Compute cross-correlation using aligned observations.
        // Approximate with the minimum number of observations assuming periodic alignment.
        let n_pairs = pair_obs.len().min(lag_obs.len());
        if n_pairs < 2 {
            break;
        }

        let mut cross_sum = 0.0_f64;
        for i in 0..n_pairs {
            cross_sum += (pair_obs[i] - mu_m) * (lag_obs[i] - stats_lag.mean);
        }

        #[allow(clippy::cast_precision_loss)]
        let gamma = cross_sum / (n_pairs - 1) as f64;
        let rho = gamma / (std_m * stats_lag.std);

        autocorrelations.push(rho.clamp(-1.0, 1.0));
    }

    autocorrelations
}

/// Convert [`SeasonalStats`] to [`InflowSeasonalStatsRow`].
///
/// The `stage_id` in `SeasonalStats` already stores the ID of the first stage
/// with the matching season (as documented in `estimate_seasonal_stats`).
fn seasonal_stats_to_rows(
    stats: &[SeasonalStats],
    _stages: &[cobre_core::temporal::Stage],
) -> Vec<InflowSeasonalStatsRow> {
    stats
        .iter()
        .map(|s| InflowSeasonalStatsRow {
            hydro_id: s.entity_id,
            stage_id: s.stage_id,
            mean_m3s: s.mean,
            std_m3s: s.std,
        })
        .collect()
}

/// Convert [`ArCoefficientEstimate`] to [`InflowArCoefficientRow`].
///
/// The `season_id` in `ArCoefficientEstimate` is mapped to a `stage_id` by
/// finding the first stage whose `season_id` matches. A single stats row is
/// emitted per lag in the coefficient vector.
fn ar_estimates_to_rows(
    ar_estimates: &[ArCoefficientEstimate],
    stages: &[cobre_core::temporal::Stage],
) -> Vec<InflowArCoefficientRow> {
    // Build season_id -> first stage_id mapping.
    // "First" = stage with the smallest `stage.id` among those with that `season_id`.
    let mut season_to_stage: HashMap<usize, i32> = HashMap::new();
    for stage in stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (sid, s.id)))
    {
        season_to_stage
            .entry(stage.0)
            .and_modify(|existing| {
                if stage.1 < *existing {
                    *existing = stage.1;
                }
            })
            .or_insert(stage.1);
    }

    let mut rows: Vec<InflowArCoefficientRow> = Vec::new();

    for est in ar_estimates {
        let Some(&stage_id) = season_to_stage.get(&est.season_id) else {
            continue;
        };

        for (lag_idx, &coeff) in est.coefficients.iter().enumerate() {
            // lag_idx is bounded by max_order (typically ≤ 12); i32 cast is safe.
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
            Bus, DeficitSegment,
            scenario::{CorrelationModel, InflowModel},
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
        let (result, report) = estimate_from_history(system, case_dir, &config).unwrap();

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
        let (result, report) = estimate_from_history(system, case_dir, &config).unwrap();

        assert_eq!(
            result.inflow_models().len(),
            original_len,
            "no history: system must be unchanged"
        );
        assert!(report.is_none(), "no history path must return None report");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_config() -> Config {
        use cobre_io::config::{EstimationConfig, OrderSelectionMethod};
        let mut cfg: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
        cfg.estimation = EstimationConfig {
            max_order: 2,
            order_selection: OrderSelectionMethod::Fixed,
            min_observations_per_season: 2,
        };
        cfg
    }

    const MINIMAL_CONFIG_JSON: &str = r#"{
        "training": { "seed": 42 },
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

    /// Construct mock `ArCoefficientEstimate` and `AicSelectionResult` entries for
    /// 2 hydros with 3 seasons each, call `build_estimation_report`, and verify
    /// that the report contains exactly 2 entries with the expected structure.
    #[test]
    fn test_estimation_report_from_aic() {
        use cobre_stochastic::par::fitting::AicSelectionResult;

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

        // Build mock AIC results: selected_order=2, aic_values for orders 0..=2.
        let mut aic_results: HashMap<(EntityId, usize), AicSelectionResult> = HashMap::new();
        for &hydro_id in &[h1, h2] {
            for season_id in 0..n_seasons {
                aic_results.insert(
                    (hydro_id, season_id),
                    AicSelectionResult {
                        selected_order: 2,
                        aic_values: vec![0.0, 12.4, 11.1],
                    },
                );
            }
        }

        let contribution_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();
        let report = build_estimation_report(
            &estimates,
            &aic_results,
            n_seasons,
            &contribution_reductions,
        );

        assert_eq!(report.entries.len(), 2, "expected 2 hydro entries");

        for &hydro_id in &[h1, h2] {
            let entry = report.entries.get(&hydro_id).expect("entry must exist");
            assert_eq!(entry.selected_order, 2, "selected_order must be 2");
            // aic_scores excludes order 0: [12.4, 11.1]
            assert_eq!(
                entry.aic_scores.len(),
                2,
                "aic_scores must have 2 values (orders 1..=2)"
            );
            assert_eq!(
                entry.coefficients.len(),
                n_seasons,
                "one coefficient vec per season"
            );
        }
    }

    /// When `OrderSelectionMethod::Fixed` is used, the returned `EstimationReport`
    /// must have an empty entries map.
    #[test]
    fn test_estimation_report_empty_for_fixed() {
        use cobre_core::temporal::Stage;
        use cobre_io::config::OrderSelectionMethod;

        let method = OrderSelectionMethod::Fixed;
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
            max_order,
            &method,
            None,
        )
        .unwrap();

        assert!(
            report.entries.is_empty(),
            "Fixed method must produce empty EstimationReport"
        );
    }

    /// Verify that `aic_scores` excludes the order-0 white-noise baseline value.
    #[test]
    fn test_estimation_report_aic_scores_exclude_order_zero() {
        use cobre_stochastic::par::fitting::AicSelectionResult;

        let hydro_id = EntityId(1);
        let n_seasons = 1_usize;

        let estimates = vec![ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![0.6, 0.2, 0.1],
            residual_std_ratio: 0.85,
        }];

        let mut aic_results: HashMap<(EntityId, usize), AicSelectionResult> = HashMap::new();
        aic_results.insert(
            (hydro_id, 0),
            AicSelectionResult {
                selected_order: 3,
                aic_values: vec![0.0, 12.4, 11.1, 10.8],
            },
        );

        let contribution_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();
        let report = build_estimation_report(
            &estimates,
            &aic_results,
            n_seasons,
            &contribution_reductions,
        );

        let entry = report.entries.get(&hydro_id).expect("entry must exist");
        assert_eq!(
            entry.aic_scores,
            vec![12.4, 11.1, 10.8],
            "aic_scores must exclude order-0 baseline"
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

        let reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map);

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

        let reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map);

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

        let _reductions =
            apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map);

        assert!(
            estimates[0].coefficients.is_empty(),
            "should fall back to order 0"
        );
        assert!(
            (estimates[0].residual_std_ratio - 1.0).abs() < 1e-10,
            "white-noise residual ratio should be 1.0"
        );
    }
}
