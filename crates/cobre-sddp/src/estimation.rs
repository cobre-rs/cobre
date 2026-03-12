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
//! | present | absent | any | Run estimation; update `inflow_models` and optionally `correlation`. |
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

use std::collections::{HashMap, HashSet};
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
    par::fitting::{
        ArCoefficientEstimate, SeasonalStats, estimate_ar_coefficients, estimate_correlation,
        estimate_seasonal_stats, find_season_for_date, levinson_durbin, select_order_aic,
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
) -> Result<System, EstimationError> {
    // ── Step 1: resolve file manifest ────────────────────────────────────────
    let mut ctx = ValidationContext::new();
    let manifest = validate_structure(case_dir, &mut ctx);

    // Abort early if structural validation found errors.
    if ctx.into_result().is_err() {
        return Ok(system);
    }

    // ── Step 2: input path matrix ────────────────────────────────────────────
    if !manifest.scenarios_inflow_history_parquet {
        // No history file — nothing to estimate.
        return Ok(system);
    }

    if manifest.scenarios_inflow_seasonal_stats_parquet
        && manifest.scenarios_inflow_ar_coefficients_parquet
    {
        // Explicit stats provided — skip estimation.
        return Ok(system);
    }

    // History present, stats absent — run estimation.
    run_estimation(system, case_dir, config, &manifest)
}

/// Inner function that runs the full estimation pipeline once path conditions are met.
fn run_estimation(
    system: System,
    case_dir: &Path,
    config: &Config,
    manifest: &FileManifest,
) -> Result<System, EstimationError> {
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

    // ── Step 4: estimate seasonal stats ─────────────────────────────────────
    let seasonal_stats = estimate_seasonal_stats(&observations, stages, &hydro_ids)?;

    // ── Step 5: estimate AR coefficients ────────────────────────────────────
    let max_order = config.estimation.max_order as usize;
    let ar_estimates = estimate_ar_coefficients_with_selection(
        &observations,
        &seasonal_stats,
        stages,
        &hydro_ids,
        max_order,
        &config.estimation.order_selection,
    )?;

    // ── Step 6: estimate or preserve correlation ─────────────────────────────
    let correlation = if manifest.scenarios_correlation_json {
        // Explicit correlation.json is present — keep whatever was loaded by load_case.
        system.correlation().clone()
    } else {
        estimate_correlation(
            &observations,
            &ar_estimates,
            &seasonal_stats,
            stages,
            &hydro_ids,
        )?
    };

    // ── Step 7: convert results to row types and assemble inflow models ───────
    let stats_rows = seasonal_stats_to_rows(&seasonal_stats, stages);
    let coeff_rows = ar_estimates_to_rows(&ar_estimates, stages);

    let inflow_models = assemble_inflow_models(stats_rows, coeff_rows)?;

    Ok(system.with_scenario_models(inflow_models, correlation))
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
) -> Result<Vec<ArCoefficientEstimate>, StochasticError> {
    match method {
        OrderSelectionMethod::Fixed => {
            estimate_ar_coefficients(observations, seasonal_stats, stages, hydro_ids, max_order)
        }
        OrderSelectionMethod::Aic => {
            estimate_ar_with_aic(observations, seasonal_stats, stages, hydro_ids, max_order)
        }
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
fn estimate_ar_with_aic(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
) -> Result<Vec<ArCoefficientEstimate>, StochasticError> {
    // Get full-order estimates first.
    let mut estimates =
        estimate_ar_coefficients(observations, seasonal_stats, stages, hydro_ids, max_order)?;

    if max_order == 0 {
        return Ok(estimates);
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
        let Some(season_id) = find_season_for_date(&stage_index, date) else {
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
    }

    Ok(estimates)
}

/// Compute normalised cross-seasonal autocorrelations `ρ_m(1)..ρ_m(max_order)` for
/// a single `(entity, season)` pair.
///
/// Returns a `Vec` whose length may be less than `max_order` when a required lag
/// season has zero standard deviation or insufficient observations.
#[allow(clippy::too_many_arguments)]
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
    clippy::float_cmp
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
        let result = estimate_from_history(system, case_dir, &config).unwrap();

        assert_eq!(
            result.inflow_models().len(),
            original_len,
            "explicit stats: system must be unchanged"
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
        let result = estimate_from_history(system, case_dir, &config).unwrap();

        assert_eq!(
            result.inflow_models().len(),
            original_len,
            "no history: system must be unchanged"
        );
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
}
