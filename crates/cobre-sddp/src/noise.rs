//! Shared noise transformation functions for the LP patching hot path.
//!
//! Both [`transform_inflow_noise`] and [`transform_load_noise`] convert raw
//! PAR(p) or normal noise samples into the patched RHS values that are written
//! into the stage LP before each solve.  Extracting them here eliminates the
//! class of bugs where one call site receives a fix and others are forgotten.

use cobre_stochastic::{evaluate_par_batch, solve_par_noise_batch, StochasticContext};

use crate::{
    context::{StageContext, TrainingContext},
    workspace::ScratchBuffers,
    InflowNonNegativityMethod,
};

/// Transform raw inflow noise `η` into patched water-balance RHS values.
///
/// Writes one patched RHS value per hydro plant into `scratch.noise_buf`.  The
/// patched value is:
///
/// ```text
/// noise_buf[h] = base_rhs + noise_scale[stage * n_hydros + h] * η_effective[h]
/// ```
///
/// where `η_effective[h]` equals `η[h]` when truncation is inactive, or the
/// η clamped to the floor that produces zero inflow when truncation is active.
///
/// ## Behaviour by inflow method
///
/// - `None` / `Penalty`: applies the raw η directly (no clamping).
/// - `Truncation`: evaluates the full PAR inflow, and when the result would be
///   negative for any hydro, solves for the η floor that drives inflow to zero
///   and clamps the noise before computing the RHS.
///
/// ## Allocation discipline
///
/// No heap allocations are made inside this function.  All scratch work is
/// done via pre-allocated buffers in `scratch` which are cleared and resized
/// in place.
///
/// ## Arguments
///
/// - `raw_noise` — raw η sample (length `>= n_hydros`).
/// - `stage` — 0-based stage index.
/// - `current_state` — current state vector (used to extract inflow lags for
///   truncation).
/// - `ctx` — stage LP layout: provides `noise_scale`, `n_hydros`, `base_rows`,
///   and `templates`.
/// - `training_ctx` — algorithm configuration: provides `inflow_method`,
///   `stochastic`, and `indexer`.
/// - `scratch` — pre-allocated scratch buffers; `noise_buf` receives the output.
pub(crate) fn transform_inflow_noise(
    raw_noise: &[f64],
    stage: usize,
    current_state: &[f64],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    scratch: &mut ScratchBuffers,
) {
    let n_hydros = ctx.n_hydros;
    let stage_offset = stage * n_hydros;
    let base_row = ctx.base_rows[stage];
    let template_row_lower = &ctx.templates[stage].row_lower;
    let noise_scale = ctx.noise_scale;
    let inflow_method = training_ctx.inflow_method;
    let stochastic = training_ctx.stochastic;
    let indexer = training_ctx.indexer;

    scratch.noise_buf.clear();
    match inflow_method {
        InflowNonNegativityMethod::Truncation => {
            let max_order = indexer.max_par_order;
            let lag_len = max_order * n_hydros;
            scratch.lag_matrix_buf.clear();
            scratch.lag_matrix_buf.resize(lag_len, 0.0);
            for h in 0..n_hydros {
                for l in 0..max_order {
                    scratch.lag_matrix_buf[l * n_hydros + h] =
                        current_state[indexer.inflow_lags.start + h * max_order + l];
                }
            }

            let par_lp = stochastic.par_lp();
            scratch.par_inflow_buf.clear();
            scratch.par_inflow_buf.resize(n_hydros, 0.0);
            evaluate_par_batch(
                par_lp,
                stage,
                &scratch.lag_matrix_buf,
                raw_noise,
                &mut scratch.par_inflow_buf,
            );

            let has_negative = scratch.par_inflow_buf.iter().any(|&a| a < 0.0);
            if has_negative {
                scratch.eta_floor_buf.clear();
                scratch.eta_floor_buf.resize(n_hydros, f64::NEG_INFINITY);
                let zero_targets = &scratch.zero_targets_buf[..n_hydros];
                solve_par_noise_batch(
                    par_lp,
                    stage,
                    &scratch.lag_matrix_buf,
                    zero_targets,
                    &mut scratch.eta_floor_buf,
                );
            }

            for (h, &eta) in raw_noise.iter().enumerate().take(n_hydros) {
                let clamped_eta = if has_negative && scratch.par_inflow_buf[h] < 0.0 {
                    eta.max(scratch.eta_floor_buf[h])
                } else {
                    eta
                };
                let base_rhs = template_row_lower[base_row + h];
                scratch
                    .noise_buf
                    .push(base_rhs + noise_scale[stage_offset + h] * clamped_eta);
            }
        }
        InflowNonNegativityMethod::None | InflowNonNegativityMethod::Penalty { .. } => {
            for (h, &eta) in raw_noise.iter().enumerate().take(n_hydros) {
                let base_rhs = template_row_lower[base_row + h];
                scratch
                    .noise_buf
                    .push(base_rhs + noise_scale[stage_offset + h] * eta);
            }
        }
    }
}

/// Transform raw load noise `η` into patched load-balance RHS values.
///
/// Writes `n_load_buses * block_count` values into `load_rhs_buf`.  For each
/// load bus `lb` and block `blk`:
///
/// ```text
/// load_rhs_buf[lb * block_count + blk] =
///     (mean(stage, lb) + std(stage, lb) * η[n_hydros + lb]).max(0.0)
///     * block_factor(stage, lb, blk)
/// ```
///
/// The realization is clamped to zero so that load demand is never negative.
///
/// ## Allocation discipline
///
/// No heap allocations.  `load_rhs_buf` is cleared and populated in place.
///
/// ## Arguments
///
/// - `raw_noise` — raw η sample; load bus entries start at index `n_hydros`.
/// - `n_hydros` — number of hydro plants (offset into `raw_noise`).
/// - `n_load_buses` — number of buses with stochastic load noise.
/// - `stochastic` — stochastic context providing the normal LP.
/// - `stage` — 0-based stage index.
/// - `block_count` — number of load blocks at this stage.
/// - `load_rhs_buf` — output buffer; cleared and populated with RHS values.
pub(crate) fn transform_load_noise(
    raw_noise: &[f64],
    n_hydros: usize,
    n_load_buses: usize,
    stochastic: &StochasticContext,
    stage: usize,
    block_count: usize,
    load_rhs_buf: &mut Vec<f64>,
) {
    load_rhs_buf.clear();
    if n_load_buses == 0 {
        return;
    }
    let load_lp = stochastic.normal_lp();
    for lb_idx in 0..n_load_buses {
        let eta = raw_noise[n_hydros + lb_idx];
        let mean = load_lp.mean(stage, lb_idx);
        let std = load_lp.std(stage, lb_idx);
        let realization = (mean + std * eta).max(0.0);
        for blk in 0..block_count {
            let factor = load_lp.block_factor(stage, lb_idx, blk);
            load_rhs_buf.push(realization * factor);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        LoadModel,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_solver::StageTemplate;
    use cobre_stochastic::context::build_stochastic_context;
    use cobre_stochastic::StochasticContext;
    use std::collections::BTreeMap;

    use crate::{
        context::{StageContext, TrainingContext},
        indexer::StageIndexer,
        noise::{transform_inflow_noise, transform_load_noise},
        workspace::ScratchBuffers,
        HorizonMode, InflowNonNegativityMethod,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a minimal `StageTemplate` with just `row_lower` populated.
    ///
    /// Only `row_lower` is accessed by `transform_inflow_noise`.  All other
    /// fields are set to their zero/empty defaults.
    fn make_minimal_template(row_lower: Vec<f64>) -> StageTemplate {
        let n = row_lower.len();
        StageTemplate {
            num_cols: 0,
            num_rows: n,
            num_nz: 0,
            col_starts: vec![0_i32],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![],
            col_upper: vec![],
            objective: vec![],
            row_lower,
            row_upper: vec![0.0; n],
            n_state: 0,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
        }
    }

    /// Build a `ScratchBuffers` with the given pre-filled `zero_targets_buf`.
    fn make_scratch(n_hydros: usize) -> ScratchBuffers {
        ScratchBuffers {
            noise_buf: Vec::with_capacity(n_hydros),
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: vec![0.0_f64; n_hydros],
            load_rhs_buf: Vec::new(),
            row_lower_buf: Vec::new(),
        }
    }

    /// One-hydro, one-stage `StochasticContext` with AR(0) (white noise).
    ///
    /// PAR(0): inflow = `std_m3s` * eta (no autoregressive term).
    /// With `mean_m3s = 0.0` and `std_m3s = 1.0`, inflow = eta.
    #[allow(clippy::too_many_lines)]
    fn make_one_hydro_stochastic(n_stages: usize) -> StochasticContext {
        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx as i32,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(&system, 42, &[]).unwrap()
    }

    /// One-hydro, one-load-bus, n-stage `StochasticContext`.
    ///
    /// Load bus has `mean_mw` and `std_mw`, one block per stage.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_with_load(n_stages: usize, mean_mw: f64, std_mw: f64) -> StochasticContext {
        let bus0 = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let bus1 = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(10),
            name: "H10".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(10),
                stage_id: idx as i32,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|idx| LoadModel {
                bus_id: EntityId(1),
                stage_id: idx as i32,
                mean_mw,
                std_mw,
            })
            .collect();

        let correlation = CorrelationModel {
            method: "cholesky".to_string(),
            profiles: BTreeMap::new(),
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(&system, 42, &[]).unwrap()
    }

    // ── transform_inflow_noise: None method ──────────────────────────────────

    /// None method: raw eta applied directly without clamping.
    #[test]
    fn test_transform_inflow_noise_none_method() {
        let stochastic = make_one_hydro_stochastic(1);
        // StageIndexer: 1 hydro, 0 PAR lags → n_state = 1
        let indexer = StageIndexer::new(1, 0);
        let current_state = vec![0.0; indexer.n_state];

        // noise_scale[0] = 1.0, base_rhs = 5.0, eta = -3.0
        // expected: 5.0 + 1.0 * (-3.0) = 2.0
        let raw_noise = vec![-3.0_f64];
        let noise_scale = vec![1.0_f64];
        // Template with row_lower = [0.0, 5.0]; base_row = 1.
        let template = make_minimal_template(vec![0.0, 5.0]);
        let templates = vec![template];
        let base_rows = vec![1_usize];
        let inflow_method = InflowNonNegativityMethod::None;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1],
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
        };
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            0,
            &current_state,
            &ctx,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.noise_buf.len(), 1);
        assert!((scratch.noise_buf[0] - 2.0).abs() < 1e-12);
    }

    // ── transform_inflow_noise: Truncation ───────────────────────────────────

    /// Truncation: when the PAR inflow would be negative, eta is clamped.
    ///
    /// AR(0) model: inflow = sigma * eta.  With sigma=1.0 and lag=0:
    /// inflow = 1.0 * eta.  For eta = -5.0, inflow = -5.0 < 0 → clamp.
    #[test]
    fn test_transform_inflow_noise_truncation_clamps() {
        let stochastic = make_one_hydro_stochastic(1);
        // 1 hydro, 0 PAR lags
        let indexer = StageIndexer::new(1, 0);
        let current_state = vec![0.0; indexer.n_state];

        // Very negative eta guarantees negative inflow (AR(0) with sigma=1).
        let raw_noise = vec![-5.0_f64];
        let noise_scale = vec![1.0_f64];
        // Template with row_lower = [0.0]; base_row = 0.
        let template = make_minimal_template(vec![0.0]);
        let templates = vec![template];
        let base_rows = vec![0_usize];
        let inflow_method = InflowNonNegativityMethod::Truncation;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1],
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
        };
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            0,
            &current_state,
            &ctx,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.noise_buf.len(), 1);
        // The patched RHS = base_rhs + noise_scale * clamped_eta.
        // After clamping, the inflow contribution must be >= 0: RHS >= base_rhs = 0.
        assert!(
            scratch.noise_buf[0] >= -1e-10,
            "truncation must yield non-negative RHS, got {}",
            scratch.noise_buf[0]
        );
    }

    /// Truncation passthrough: positive-inflow eta passes through unchanged.
    #[test]
    fn test_transform_inflow_noise_truncation_passthrough() {
        let stochastic = make_one_hydro_stochastic(1);
        let indexer = StageIndexer::new(1, 0);
        let current_state = vec![0.0; indexer.n_state];

        // eta = 3.0 → inflow = 1.0 * 3.0 = 3.0 > 0 → no clamping.
        let raw_noise = vec![3.0_f64];
        let noise_scale = vec![2.0_f64];
        // Template with row_lower = [5.0]; base_row = 0.
        let template = make_minimal_template(vec![5.0]);
        let templates = vec![template];
        let base_rows = vec![0_usize];
        let inflow_method = InflowNonNegativityMethod::Truncation;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1],
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
        };
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            0,
            &current_state,
            &ctx,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.noise_buf.len(), 1);
        // Expected: 5.0 + 2.0 * 3.0 = 11.0 (no clamping).
        assert!(
            (scratch.noise_buf[0] - 11.0).abs() < 1e-12,
            "expected 11.0, got {}",
            scratch.noise_buf[0]
        );
    }

    // ── transform_load_noise ──────────────────────────────────────────────────

    /// Basic load noise: verify RHS computation matches expected values.
    ///
    /// 1 hydro + 1 load bus.  Load bus is at noise index 1.
    /// eta = 0.0 → realization = (mean + std * 0.0).max(0.0) = mean.
    #[test]
    fn test_transform_load_noise_basic() {
        let mean_mw = 5.0_f64;
        let std_mw = 1.0_f64;
        let stochastic = make_stochastic_with_load(1, mean_mw, std_mw);

        // n_hydros=1 (hydro noise at index 0), load bus noise at index 1.
        // eta_load = 0.0 → realization = 5.0; block_factor = 1.0 → rhs = 5.0.
        let raw_noise = vec![0.0_f64, 0.0_f64]; // [hydro_eta, load_eta]
        let mut load_rhs_buf = Vec::new();

        transform_load_noise(&raw_noise, 1, 1, &stochastic, 0, 1, &mut load_rhs_buf);

        assert_eq!(load_rhs_buf.len(), 1);
        // The block_factor for a single Parallel block is the block duration
        // divided by total stage hours; with one block it equals 1.0.
        // Expected: 5.0 * 1.0 = 5.0.
        assert!(
            (load_rhs_buf[0] - 5.0).abs() < 1e-10,
            "expected 5.0, got {}",
            load_rhs_buf[0]
        );
    }

    /// Negative realizations are clamped to zero.
    ///
    /// Very negative eta drives `mean + std * eta` below zero; must be clamped.
    #[test]
    fn test_transform_load_noise_clamped_non_negative() {
        let mean_mw = 2.0_f64;
        let std_mw = 1.0_f64;
        let stochastic = make_stochastic_with_load(1, mean_mw, std_mw);

        // eta_load = -10.0 → realization = (2.0 - 10.0).max(0.0) = 0.0.
        let raw_noise = vec![0.0_f64, -10.0_f64];
        let mut load_rhs_buf = Vec::new();

        transform_load_noise(&raw_noise, 1, 1, &stochastic, 0, 1, &mut load_rhs_buf);

        assert_eq!(load_rhs_buf.len(), 1);
        assert!(
            load_rhs_buf[0].abs() < 1e-12,
            "expected 0.0, got {}",
            load_rhs_buf[0]
        );
    }
}
