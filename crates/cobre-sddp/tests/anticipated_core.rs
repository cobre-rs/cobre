//! Consolidated anticipated-core integration tests for `cobre-sddp`.
//!
//! Each anticipated-core domain group lives in its own inner `mod` so the suite
//! links the statically-bound solver once rather than once per file. Per-`mod`
//! scoping isolates each group's consts, helpers, and fixtures.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

mod common;

use cobre_core::HydroPenalties;
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_io::config::{SimulationSelection, TrainingSelection};

fn default_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 500.0,
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
    }
}

/// Build a [`Config`] for `iterations`-iteration training and 1-scenario
/// deterministic simulation. More iterations let cuts sharpen so the cut
/// gradients at the anticipated-state columns drive `d_t` to load level at
/// every active stage.
fn build_config(iterations: usize) -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::Penalty,
            },

            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: iterations as u32,
            }]),
            stopping_mode: cobre_io::config::StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig {
            enabled: true,
            io_channel_capacity: 8,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
            ..IoSimulationConfig::default()
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

mod anticipated_backward_cut {
    //! Analytical verification of backward-pass cut-coefficient extraction for an
    //! anticipated thermal across lead_stages K = 1, 2, 3. Each K's closed-form
    //! derivation lives on its test function.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // Numeric constants shared across all K (single source of truth).
    // ---------------------------------------------------------------------------

    const BLOCK_HOURS: f64 = 1.0;
    const C_REG: f64 = 100.0;
    const C_ANT: f64 = 50.0;

    // Every non-theta objective coefficient is divided by this, so duals and the
    // stored cut live in scaled cost units.
    const COST_SCALE_FACTOR: f64 = 1_000_000.0;

    const TOL: f64 = 1e-6;

    // System::build sorts thermals by EntityId ascending; with reg_id < ant_id,
    // thermal_idx 0 is the regular thermal and 1 is the anticipated thermal.
    const THERMAL_IDX_REG: usize = 0;
    const THERMAL_IDX_ANT: usize = 1;

    // ---------------------------------------------------------------------------
    // Per-K fixture table
    // ---------------------------------------------------------------------------

    /// Per-K parameters for the anticipated backward-cut fixtures. Each `#[test]`
    /// builds an independent `System` from one entry, so entity IDs need only be
    /// disjoint within an entry (bus 1, reg, ant); cross-K reuse is harmless.
    struct BackwardCutFixture {
        n_stages: usize,
        k_max: usize,
        /// Per-stage load, MW (length `n_stages`).
        loads_mw: &'static [f64],
        max_gen_reg: f64,
        max_gen_ant: f64,
        reg_id: EntityId,
        ant_id: EntityId,
        reg_start_date: (i32, u32, u32),
        ant_start_date: (i32, u32, u32),
        /// Anticipated ring-buffer seeds, MW (length `k_max`).
        seeds_mw: &'static [f64],
        iterations: usize,
        expected_coeff: f64,
    }

    const FIXTURE_K1: BackwardCutFixture = BackwardCutFixture {
        n_stages: 2,
        k_max: 1,
        loads_mw: &[10.0, 20.0],
        max_gen_reg: 50.0,
        max_gen_ant: 30.0,
        reg_id: EntityId(3),
        ant_id: EntityId(4),
        reg_start_date: (2024, 1, 4),
        ant_start_date: (2024, 1, 5),
        seeds_mw: &[10.0],
        iterations: 1,
        expected_coeff: -C_REG / COST_SCALE_FACTOR,
    };

    const FIXTURE_K2: BackwardCutFixture = BackwardCutFixture {
        n_stages: 3,
        k_max: 2,
        loads_mw: &[5.0, 10.0, 30.0],
        max_gen_reg: 100.0,
        max_gen_ant: 50.0,
        reg_id: EntityId(2),
        ant_id: EntityId(4),
        reg_start_date: (2024, 1, 3),
        ant_start_date: (2024, 1, 5),
        seeds_mw: &[0.0, 0.0],
        iterations: 5,
        expected_coeff: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS,
    };

    const FIXTURE_K3: BackwardCutFixture = BackwardCutFixture {
        n_stages: 4,
        k_max: 3,
        loads_mw: &[5.0, 10.0, 15.0, 30.0],
        max_gen_reg: 100.0,
        max_gen_ant: 50.0,
        reg_id: EntityId(2),
        ant_id: EntityId(5),
        reg_start_date: (2024, 1, 3),
        ant_start_date: (2024, 1, 6),
        seeds_mw: &[0.0, 0.0, 0.0],
        iterations: 5,
        expected_coeff: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    fn build_system(fixture: &BackwardCutFixture) -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        let date = |d: (i32, u32, u32)| NaiveDate::from_ymd_opt(d.0, d.1, d.2).expect("valid date");

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                // Deficit cost set safely above c_reg so the LP never prefers shedding load.
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_reg = make_thermal(
            fixture.reg_id,
            ThermalSpec {
                name: "T_reg".to_string(),
                operational_start_date: date(fixture.reg_start_date),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_reg,
                cost_per_mwh: C_REG,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            fixture.ant_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: date(fixture.ant_start_date),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_ant,
                cost_per_mwh: C_ANT,
                anticipated_config: Some(AnticipatedConfig::LeadStages(fixture.k_max as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        assert!(
            thermal_reg.id.0 < thermal_ant.id.0,
            "T_reg.id ({}) must be strictly less than T_ant.id ({}) so that \
         System::build's sort_by_key aligns thermal_idx with the bounds table",
            thermal_reg.id.0,
            thermal_ant.id.0,
        );

        // StageCalendar requires a chronologically ordered, non-overlapping
        // calendar; one calendar day per stage keeps it valid while BLOCK_HOURS
        // alone still drives every LP-facing duration.
        let stage_date = |offset_days: i64| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(offset_days)
        };
        let stages: Vec<Stage> = (0..fixture.n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i as i64),
                        end_date: stage_date(i as i64 + 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: BLOCK_HOURS,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..fixture.n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: fixture.loads_mw[i],
                std_mw: 0.0,
            })
            .collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: fixture.n_stages,
                k_max: fixture.k_max,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        // K-padded axis: fill_anticipated_columns reads delivery cells at
        // stage_idx + K_i, so overrides must cover the n_stages + k_max range.
        let thermal_axis = fixture.n_stages + fixture.k_max;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_REG, s) = ThermalStageBounds {
                cost_per_mwh: C_REG,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_REG, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_reg,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
                cost_per_mwh: C_ANT,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_ant,
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: fixture.n_stages,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Anticipated ring-buffer seeds; any feasible choice yields the same cut.
        // One windowed record per seed, tiling delivery stages 0..k_max.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: fixture
                .seeds_mw
                .iter()
                .enumerate()
                .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
                    thermal_id: fixture.ant_id,
                    start_date: stage_date(i as i64),
                    end_date: stage_date(i as i64 + 1),
                    value_mw,
                })
                .collect(),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_reg, thermal_ant])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    fn build_config(iterations: usize) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: iterations as u32,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Backward-pass cut coefficient for an anticipated thermal with `lead_stages = 1`
    /// in a 2-stage system.
    ///
    /// One anticipated thermal (K=1, cost c_ant) and one regular thermal (cost c_reg) at a
    /// single bus; loads D_0, D_1; one one-hour block per stage; max_par_order = 0 so
    /// commit_out.start = 0. The LP-builder divides every non-theta objective
    /// coefficient by COST_SCALE_FACTOR (call it K), so the stored cut lives in scaled units.
    ///
    /// Stage-1 LP (the anticipated decision column `d_ant` is inactive at the
    /// delivery stage, bounded `[0, 0]`):
    ///
    /// ```text
    ///   min  (c_reg/K) gt_reg + theta
    ///   s.t. gt_reg + gt_ant = D_1            (load balance)
    ///        gt_ant - x_state = 0             (fishing, K=1)
    ///        x_state ∈ [x_hat, x_hat]         (pinned by column bounds)
    ///        theta >= 0
    /// ```
    ///
    /// At the box optimum, fishing forces gt_ant = x_state = x_hat, so
    /// gt_reg = D_1 - x_hat and Q_scaled(x_hat) = (c_reg/K)(D_1 - x_hat). The
    /// incoming-state column's reduced cost is pi = -c_reg/K. With coefficients
    /// = dual (no sign flip), the coefficient is -c_reg/K and the intercept is
    /// Q_scaled(x_hat) - pi*x_hat = (c_reg/K)*D_1.
    #[test]
    fn two_stage_k1_anticipated_cut_coefficient_matches_analytical() {
        const K_MAX: usize = FIXTURE_K1.k_max;
        const D_1: f64 = FIXTURE_K1.loads_mw[1];
        const EXPECTED_COEFFICIENT: f64 = FIXTURE_K1.expected_coeff;
        const EXPECTED_INTERCEPT: f64 = C_REG * D_1 / COST_SCALE_FACTOR;

        let system = build_system(&FIXTURE_K1);
        let config = build_config(FIXTURE_K1.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                FIXTURE_K1.iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );

        let pool0 = &setup.fcf.pools[0];
        let active_count = pool0.active_count();
        assert_eq!(
            active_count, 1,
            "stage 0 FCF must contain exactly one active cut; got {active_count}",
        );

        let state = setup.stage_state();
        let ant_state_idx = state.commit_out.start;
        assert_eq!(
            state.n_anticipated, 1,
            "fixture must have exactly one anticipated thermal",
        );
        assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
        assert_eq!(
            ant_state_idx, 0,
            "with n_hydros=0 and max_par_order=0, commit_out.start must be 0; got {ant_state_idx}",
        );

        let (slot, intercept, coefficients) = setup
            .fcf
            .active_cuts(0)
            .next()
            .expect("exactly one active cut must be retrievable from stage 0 pool");
        assert_eq!(
            coefficients.len(),
            state.commit_out.end,
            "coefficient slice length must equal n_state",
        );

        let actual_coeff = coefficients[ant_state_idx];
        assert!(
            (actual_coeff - EXPECTED_COEFFICIENT).abs() < TOL,
            "cut coefficient at commit_out index {ant_state_idx} \
         (slot={slot}, n_state={n_state}) does not match analytical value: \
         actual = {actual_coeff}, expected = {EXPECTED_COEFFICIENT} (= -c_reg/K = -{C_REG}/{COST_SCALE_FACTOR})",
            n_state = coefficients.len(),
        );

        assert!(
            (intercept - EXPECTED_INTERCEPT).abs() < TOL,
            "cut intercept does not match analytical value: actual = {intercept}, \
         expected = {EXPECTED_INTERCEPT} (= c_reg * D_1 / K = {C_REG} * {D_1} / {COST_SCALE_FACTOR})",
        );
    }

    /// Backward-pass cut-coefficient propagation for an anticipated thermal with
    /// `lead_stages = 2` in a 3-stage system.
    ///
    /// One anticipated thermal (K=2) and one regular thermal at a single bus; loads
    /// D_0, D_1, D_2; one one-hour block per stage; zero seeds; max_par_order = 0 so
    /// commit_out.start = 0. Fishing rows are emitted at every stage in 0..n_stages.
    ///
    /// The stage-0 FCF cut is generated by backward t=0 (solving stage 1's LP), which carries
    /// the FCF cut produced earlier in the same sweep by backward t=1 (solving stage 2). Under
    /// the HOLD geometry each commitment is held at its delivery-target modular slot
    /// (`m mod k_max`): slot 1 holds delivery-1 and is fished directly at stage 1; slot 0 holds
    /// delivery-2 and is fished at stage 2, its dual flowing back through the frozen stage-1
    /// FCF cut and the same-slot carry definition row (`slot_0^out - slot_0^in = 0`) that holds
    /// slot 0 across stage 1 (`commit_out` -> `commit_in`). Both stage-1 state duals equal
    /// -c_reg/COST_SCALE_FACTOR * BLOCK_HOURS, so the stored stage-0 cut carries -0.0001 at
    /// both state slots — a uniform vector, hence unchanged under the shift-to-hold slot
    /// permutation.
    ///
    /// iterations = 5: backward t=0 consumes the cut just added to FCF[1] within the same
    /// iteration, so the propagated stage-2 subgradient reaches FCF[0] (the slot-1 cut) by
    /// iteration 1; the remaining iterations are margin and do not move the asserted cut.
    #[test]
    fn three_stage_k2_anticipated_cut_coefficient_propagates_correctly() {
        const K_MAX: usize = FIXTURE_K2.k_max;
        const N_ITERATIONS: usize = FIXTURE_K2.iterations;
        const EXPECTED_COEFF_SLOT0: f64 = FIXTURE_K2.expected_coeff;
        const EXPECTED_COEFF_SLOT1: f64 = FIXTURE_K2.expected_coeff;

        let system = build_system(&FIXTURE_K2);
        let config = build_config(FIXTURE_K2.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                FIXTURE_K2.iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error,
        );

        let pool0 = &setup.fcf.pools[0];
        let active_count = pool0.active_count();
        assert!(
            active_count >= 1,
            "stage 0 FCF must contain at least one active cut after \
         {N_ITERATIONS} iterations; got {active_count}",
        );

        // Anticipated ring layout is `start + slot * n_anticipated + plant`; with
        // n_anticipated = 1, plant = 0 the slots are consecutive from `start`.
        let state = setup.stage_state();
        let ant_state_start = state.commit_out.start;
        let slot0_idx = ant_state_start; // slot 0, plant 0
        let slot1_idx = ant_state_start + 1; // slot 1, plant 0
        assert_eq!(
            state.n_anticipated, 1,
            "fixture must have exactly one anticipated thermal",
        );
        assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
        assert_eq!(
            ant_state_start, 0,
            "with n_hydros=0 and max_par_order=0, commit_out.start must \
         be 0; got {ant_state_start}",
        );

        // The closed-form matches the iteration-1 cut only: it lands at slot 0 under
        // dense packing (iteration_base = 1, forward_passes = 1, per
        // CutPool::slot_index); later iterations add zero-subgradient cuts at shifted
        // trial points. Select slot 0 explicitly rather than the most-recent cut.
        let analytical = setup
            .fcf
            .active_cuts(0)
            .find(|(slot, _, _)| *slot == 0)
            .expect("iteration-1 cut (slot 0 under dense packing) must be present in stage 0 pool");
        let (slot, _intercept, coefficients) = analytical;

        assert_eq!(
            coefficients.len(),
            state.commit_out.end,
            "coefficient slice length must equal n_state (= commit_out.end \
         in this no-hydro fixture); got len={}, expected={}",
            coefficients.len(),
            state.commit_out.end,
        );

        let actual_coeff_slot1 = coefficients[slot1_idx];
        assert!(
            (actual_coeff_slot1 - EXPECTED_COEFF_SLOT1).abs() < TOL,
            "stage 0 cut coefficient at commit_out slot 1 \
         (state-vector index {slot1_idx}) does not match analytical value: \
         actual = {actual_coeff_slot1}, expected = {EXPECTED_COEFF_SLOT1} \
         (= -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = \
         -{C_REG}/{COST_SCALE_FACTOR}*{BLOCK_HOURS}). \
         Source: slot 1 holds delivery-1 (1 mod k_max), fished directly at stage 1 \
         under the always-active fishing predicate. \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
            n_state = coefficients.len(),
        );

        let actual_coeff_slot0 = coefficients[slot0_idx];
        assert!(
            (actual_coeff_slot0 - EXPECTED_COEFF_SLOT0).abs() < TOL,
            "stage 0 cut coefficient at commit_out slot 0 \
         (state-vector index {slot0_idx}) does not match analytical value: \
         actual = {actual_coeff_slot0}, expected = {EXPECTED_COEFF_SLOT0} \
         (= -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = \
         -{C_REG}/{COST_SCALE_FACTOR}*{BLOCK_HOURS}). \
         Source: slot 0 holds delivery-2 (2 mod k_max), fished at stage 2; its dual \
         flows back through the frozen stage-1 FCF cut and the same-slot carry \
         definition row that holds slot 0 across stage 1. \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
            n_state = coefficients.len(),
        );
    }

    /// Backward-pass cut-coefficient propagation for an anticipated thermal with
    /// `lead_stages = 3` in a 4-stage system.
    ///
    /// One anticipated thermal (K=3, cost $50/MWh, max 50 MW), one regular thermal
    /// (cost $100/MWh, max 100 MW), loads 5, 10, 15, 30 MW, zero seeds, one-hour blocks.
    /// Fishing rows are emitted at every stage in 0..n_stages. Under the HOLD geometry each
    /// commitment is held at its delivery-target modular slot (`m mod k_max`, here `k_max=3`),
    /// so all three stage-0 slots receive -c_reg/COST_SCALE_FACTOR via distinct paths:
    ///
    /// - slot 1: delivery-1 (`1 mod 3`), fished directly at stage 1;
    /// - slot 2: delivery-2 (`2 mod 3`), fished at stage 2, via one same-slot carry
    ///   definition row (slot 2 held across stage 1) and the stage-1 FCF cut;
    /// - slot 0: delivery-3 (`3 mod 3`), fished at stage 3, via two successive same-slot
    ///   carry definition rows (slot 0 held across stages 2 and 1) and the stage-2 then
    ///   stage-1 FCF cuts.
    ///
    /// The three duals are equal (-c_reg/COST_SCALE_FACTOR * BLOCK_HOURS), so the
    /// coefficient vector is uniform and unchanged under the shift-to-hold slot permutation.
    ///
    /// See `StateSpace::state_to_lp_column` for the full algebraic chain.
    #[test]
    fn four_stage_k3_anticipated_cut_coefficient_propagates_correctly() {
        const K_MAX: usize = FIXTURE_K3.k_max;
        const N_ITERATIONS: usize = FIXTURE_K3.iterations;
        const EXPECTED_COEFF_SLOT0: f64 = FIXTURE_K3.expected_coeff;
        const EXPECTED_COEFF_SLOT1: f64 = FIXTURE_K3.expected_coeff;
        const EXPECTED_COEFF_SLOT2: f64 = FIXTURE_K3.expected_coeff;

        let system = build_system(&FIXTURE_K3);
        let config = build_config(FIXTURE_K3.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                FIXTURE_K3.iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error,
        );

        let pool0 = &setup.fcf.pools[0];
        let active_count = pool0.active_count();
        assert!(
            active_count >= 1,
            "stage 0 FCF must contain at least one active cut after \
         {N_ITERATIONS} iterations; got {active_count}",
        );

        // Anticipated ring layout is `start + slot * n_anticipated + plant`; with
        // n_anticipated = 1, plant = 0 the slots are consecutive from `start`.
        let state = setup.stage_state();
        let ant_state_start = state.commit_out.start;
        let slot0_idx = ant_state_start;
        let slot1_idx = ant_state_start + 1;
        let slot2_idx = ant_state_start + 2;
        assert_eq!(
            state.n_anticipated, 1,
            "fixture must have exactly one anticipated thermal",
        );
        assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
        assert_eq!(
            ant_state_start, 0,
            "with n_hydros=0 and max_par_order=0, commit_out.start must \
         be 0; got {ant_state_start}",
        );

        // The analytical match is the iteration-1 cut (slot 0 under dense packing,
        // per CutPool::slot_index): its three-stage propagation chain completes at
        // backward t=0. Later iterations add cuts at trial points with a different
        // active basis.
        let analytical = setup
            .fcf
            .active_cuts(0)
            .find(|(slot, _, _)| *slot == 0)
            .expect("iteration-1 cut (slot 0 under dense packing) must be present in stage 0 pool");
        let (_slot, _intercept, coefficients) = analytical;

        assert_eq!(
            coefficients.len(),
            state.commit_out.end,
            "coefficient slice length must equal n_state (= commit_out.end \
         in this no-hydro fixture); got len={}, expected={}",
            coefficients.len(),
            state.commit_out.end,
        );

        let actual_coeff_slot2 = coefficients[slot2_idx];
        assert!(
            (actual_coeff_slot2 - EXPECTED_COEFF_SLOT2).abs() < TOL,
            "slot 2 coefficient {actual_coeff_slot2} != {EXPECTED_COEFF_SLOT2} \
         (slot 2 holds delivery-2 (2 mod k_max), fished at stage 2 via one same-slot \
         carry definition row and the stage-1 FCF cut)",
        );

        let actual_coeff_slot1 = coefficients[slot1_idx];
        assert!(
            (actual_coeff_slot1 - EXPECTED_COEFF_SLOT1).abs() < TOL,
            "slot 1 coefficient {actual_coeff_slot1} != {EXPECTED_COEFF_SLOT1} \
         (slot 1 holds delivery-1 (1 mod k_max), fished directly at stage 1)",
        );

        let actual_coeff_slot0 = coefficients[slot0_idx];
        assert!(
            (actual_coeff_slot0 - EXPECTED_COEFF_SLOT0).abs() < TOL,
            "slot 0 coefficient {actual_coeff_slot0} != {EXPECTED_COEFF_SLOT0} \
         (slot 0 holds delivery-3 (3 mod k_max), fished at stage 3 via two successive \
         same-slot carry definition rows and the stage-2 then stage-1 FCF cuts)",
        );
    }
}

mod hm_distribute_conservation {
    //! Runtime confirmation of the delivery-stage duration-scaling direction: the
    //! anticipated-state cut coefficient scales linearly with the delivery stage's
    //! own `block_hours_total`, not the decision stage's. Mirrors
    //! `anticipated_backward_cut`'s K=1 closed-form fixture, varying only the
    //! delivery stage's declared hours between a monthly and a weekly duration.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // Numeric constants (mirrors anticipated_backward_cut's FIXTURE_K1 exactly,
    // except for the varying delivery-stage duration).
    // ---------------------------------------------------------------------------

    const C_REG: f64 = 100.0;
    const C_ANT: f64 = 50.0;
    const COST_SCALE_FACTOR: f64 = 1_000_000.0;
    const DECISION_STAGE_HOURS: f64 = 1.0;
    /// Monthly delivery-stage duration [h].
    const H_M: f64 = 744.0;
    /// Weekly delivery-stage duration [h].
    const H_W: f64 = 168.0;
    const TOL: f64 = 1e-6;

    const REG_ID: EntityId = EntityId(3);
    const ANT_ID: EntityId = EntityId(4);
    const LOADS_MW: [f64; 2] = [10.0, 20.0];
    const MAX_GEN_REG: f64 = 50.0;
    const MAX_GEN_ANT: f64 = 30.0;
    const SEED_MW: f64 = 10.0;

    /// Per-fixture parameters: only `delivery_stage_hours` varies between the two
    /// `#[test]`-driving fixtures below; every other physical quantity (committed
    /// MW, fuel cost, load) is shared.
    struct HmFixture {
        delivery_stage_hours: f64,
        expected_coeff: f64,
    }

    const FIXTURE_MONTHLY: HmFixture = HmFixture {
        delivery_stage_hours: H_M,
        expected_coeff: -C_REG / COST_SCALE_FACTOR * H_M,
    };

    const FIXTURE_WEEKLY: HmFixture = HmFixture {
        delivery_stage_hours: H_W,
        expected_coeff: -C_REG / COST_SCALE_FACTOR * H_W,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Two-stage, K=1 system identical to `anticipated_backward_cut::FIXTURE_K1`
    /// except that the delivery (stage 1) block carries `delivery_stage_hours`
    /// instead of a fixed constant; the decision stage (stage 0) keeps
    /// `DECISION_STAGE_HOURS` unchanged across both fixtures.
    fn build_system(delivery_stage_hours: f64) -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        let stage_date = |offset_days: i64| {
            NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date") + TimeDelta::days(offset_days)
        };

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).expect("valid date"),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_reg = make_thermal(
            REG_ID,
            ThermalSpec {
                name: "T_reg".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 4).expect("valid date"),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: MAX_GEN_REG,
                cost_per_mwh: C_REG,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANT_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 5).expect("valid date"),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: MAX_GEN_ANT,
                cost_per_mwh: C_ANT,
                anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        // StageCalendar requires a chronologically ordered, non-overlapping
        // calendar; one calendar day per stage keeps it valid while each block's
        // own duration_hours alone still drives every LP-facing duration.
        let stages: Vec<Stage> = vec![
            make_stage(
                0,
                StageSpec {
                    start_date: stage_date(0),
                    end_date: stage_date(1),
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: DECISION_STAGE_HOURS,
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
                    ..Default::default()
                },
            ),
            make_stage(
                1,
                StageSpec {
                    start_date: stage_date(1),
                    end_date: stage_date(2),
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: delivery_stage_hours,
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
                    ..Default::default()
                },
            ),
        ];

        let load_models: Vec<LoadModel> = LOADS_MW
            .iter()
            .enumerate()
            .map(|(i, &mean_mw)| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw,
                std_mw: 0.0,
            })
            .collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 1,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        // K-padded axis: fill_anticipated_columns reads delivery cells at
        // stage_idx + K_i, so overrides must cover the n_stages + k_max range.
        // T_reg.id (3) < T_ant.id (4), so System::build's sort_by_key aligns
        // thermal_idx 0 with T_reg and 1 with T_ant.
        const THERMAL_AXIS: usize = 2 + 1;
        for s in 0..THERMAL_AXIS {
            *bounds.thermal_bounds_mut(0, s) = ThermalStageBounds {
                cost_per_mwh: C_REG,
            };
            *bounds.thermal_block_base_mut(0, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: MAX_GEN_REG,
            };
            *bounds.thermal_bounds_mut(1, s) = ThermalStageBounds {
                cost_per_mwh: C_ANT,
            };
            *bounds.thermal_block_base_mut(1, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: MAX_GEN_ANT,
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 2,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // One windowed pre-study seed tiling delivery stage 0 exactly (k_max = 1),
        // matching anticipated_backward_cut::FIXTURE_K1.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANT_ID,
                start_date: stage_date(0),
                end_date: stage_date(1),
                value_mw: SEED_MW,
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_reg, thermal_ant])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
                cost_scale_factor: Some(COST_SCALE_FACTOR),
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Extraction
    // ---------------------------------------------------------------------------

    /// Trains one backward pass over `fixture`'s system and returns the stage-0
    /// anticipated-state cut coefficient (the subgradient on the commit-out
    /// column), analogous to `anticipated_backward_cut`'s per-K extraction.
    fn extract_anticipated_coefficient(fixture: &HmFixture) -> f64 {
        let system = build_system(fixture.delivery_stage_hours);
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );

        let state = setup.stage_state();
        assert_eq!(
            state.n_anticipated, 1,
            "fixture must have exactly one anticipated thermal",
        );
        let ant_state_idx = state.commit_out.start;

        let (_slot, _intercept, coefficients) = setup
            .fcf
            .active_cuts(0)
            .next()
            .expect("stage 0 pool must carry a generated cut after training");
        coefficients[ant_state_idx]
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Runtime confirmation of the `÷H_M` distribute direction: the stage-0
    /// anticipated-state cut coefficient scales linearly with the delivery
    /// stage's own `block_hours_total` (the fishing-row `−H` coupling propagated
    /// backward through the incoming-state column's reduced cost), never the decision stage's. Two
    /// fixtures sharing the identical committed MW, fuel cost, and load, and
    /// differing ONLY in the delivery stage's declared hours (`H_M` = 744 vs
    /// `H_w` = 168), must produce coefficients in exactly the `H_w / H_M` ratio.
    /// A runtime that instead replicated the coefficient regardless of delivery
    /// duration would report a ratio of `≈ 1`, which the final assertion refutes
    /// loudly by naming both coefficients and both ratios.
    #[test]
    fn anticipated_state_coefficient_scales_by_delivery_stage_hours() {
        let coeff_monthly = extract_anticipated_coefficient(&FIXTURE_MONTHLY);
        let coeff_weekly = extract_anticipated_coefficient(&FIXTURE_WEEKLY);

        // The ratio check runs FIRST: a runtime that replicates the coefficient
        // regardless of delivery duration reports a ratio of ~1 here, and this is
        // the assertion whose message must name it — the per-fixture magnitude
        // checks below exist only to catch the disjoint "both wrong, ratio
        // coincidentally right" case, not to preempt this one.
        let observed_ratio = coeff_weekly / coeff_monthly;
        let expected_ratio = H_W / H_M;
        assert!(
            (observed_ratio - expected_ratio).abs()
                <= TOL * observed_ratio.abs().max(expected_ratio.abs()),
            "the anticipated-state cut coefficient must scale by the delivery stage's own \
         block_hours_total (distribute), not replicate unchanged across delivery \
         durations: coeff_weekly = {coeff_weekly}, coeff_monthly = {coeff_monthly}, \
         observed ratio (coeff_weekly/coeff_monthly) = {observed_ratio}, \
         expected ratio (H_w/H_M) = {expected_ratio}",
        );

        for (label, coeff, fixture) in [
            ("monthly", coeff_monthly, &FIXTURE_MONTHLY),
            ("weekly", coeff_weekly, &FIXTURE_WEEKLY),
        ] {
            let expected = fixture.expected_coeff;
            assert!(
                (coeff - expected).abs() <= TOL * expected.abs().max(coeff.abs()),
                "{label} coefficient does not match its closed form -C_REG/COST_SCALE_FACTOR*h: \
             actual = {coeff}, expected = {expected} \
             (= -{C_REG}/{COST_SCALE_FACTOR}*{})",
                fixture.delivery_stage_hours,
            );
        }
    }
}

mod anticipated_pre_horizon_seed_delivery {
    //! Pre-horizon seed-delivery integration tests for an anticipated thermal across
    //! lead_stages K = 1, 2, 3. Each test trains a small in-code study, runs a
    //! one-scenario simulation, and asserts that the matured ring-buffer seeds are
    //! delivered at the early stages, that anticipated decisions saturate within
    //! bounds, that the commitment-hold delivery maps committed_at(t) ≈ decision_at(t−K),
    //! and that the observed cost stays under a per-K analytical upper bound. Each
    //! K's derivation and cost bound live on its test function.

    use cobre_core::HorizonGraph;
    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };
    use super::common::run_simulation;

    // ---------------------------------------------------------------------------
    // Per-K fixture table
    // ---------------------------------------------------------------------------

    /// Per-K parameters for the pre-horizon seed-delivery fixtures. Each `#[test]`
    /// builds an independent `System` from one entry; the entity IDs sit in disjoint
    /// decades (30s / 40s / 50s), themselves disjoint from the 2..7 range the sibling
    /// anticipated tests use, so a combined nextest run attributes failures
    /// unambiguously per entity.
    struct SeedDeliveryFixture {
        n_stages: usize,
        /// Anticipated `lead_stages` K — also the ring-buffer depth `k_max` and the
        /// `kN` suffix in each entity name.
        k: usize,
        bus_id: EntityId,
        hydro_id: EntityId,
        anticipated_id: EntityId,
        backup_id: EntityId,
        /// Anticipated ring-buffer seeds, MW (length `k`).
        seeds_mw: &'static [f64],
        iterations: usize,
    }

    const FIXTURE_K1: SeedDeliveryFixture = SeedDeliveryFixture {
        n_stages: 5,
        k: 1,
        bus_id: EntityId(1),
        hydro_id: EntityId(30),
        anticipated_id: EntityId(31),
        backup_id: EntityId(32),
        seeds_mw: &[100.0],
        iterations: 1,
    };

    const FIXTURE_K2: SeedDeliveryFixture = SeedDeliveryFixture {
        n_stages: 5,
        k: 2,
        bus_id: EntityId(1),
        hydro_id: EntityId(41),
        anticipated_id: EntityId(42),
        backup_id: EntityId(43),
        seeds_mw: &[80.0, 50.0],
        iterations: 5,
    };

    const FIXTURE_K3: SeedDeliveryFixture = SeedDeliveryFixture {
        n_stages: 6,
        k: 3,
        bus_id: EntityId(1),
        hydro_id: EntityId(51),
        anticipated_id: EntityId(52),
        backup_id: EntityId(53),
        seeds_mw: &[50.0, 30.0, 10.0],
        iterations: 5,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the `System` for one seed-delivery fixture: one anticipated thermal at
    /// `fixture.anticipated_id`, one backup at `fixture.backup_id`, a trivial hydro at
    /// `fixture.hydro_id`, load 150 MW, ring-buffer seed `fixture.seeds_mw`.
    ///
    /// Constructing `System` directly via `SystemBuilder` bypasses the `cobre-io`
    /// commissioning-window validator that rejects a non-zero seed maturing outside
    /// it: the non-zero seed is the deliberate fixture; that rejection rule applies
    /// only to JSON input through `load_case`. The $10/MWh anticipated vs $5000/MWh
    /// backup asymmetry saturates anticipated dispatch at max_gen, and
    /// `annual_discount_rate = 0.0` collapses every discount factor to 1.0 so each
    /// test's analytical cost derivation is exact.
    fn build_system(fixture: &SeedDeliveryFixture) -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        let k = fixture.k;
        let n_stages = fixture.n_stages;

        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let bus = make_bus(
            fixture.bus_id,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            fixture.anticipated_id,
            ThermalSpec {
                name: format!("T_ant_seed_k{k}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: fixture.bus_id,
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(k as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            fixture.backup_id,
            ThermalSpec {
                name: format!("T_backup_seed_k{k}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: fixture.bus_id,
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
                cost_per_mwh: 5000.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        // Zero inflow and 1 MW max_gen keep the system firmly in the thermal regime.
        let hydro = make_hydro(
            fixture.hydro_id,
            HydroSpec {
                name: format!("H1_seed_k{k}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: fixture.bus_id,
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: fixture.hydro_id,
                stage_id: i as i32,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: fixture.bus_id,
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 1.0,
                max_generation_mw: 1.0,
                ..Default::default()
            }
        }

        // The padding region [n_stages, n_stages + k) is the delivery-stage axis read
        // by `fill_anticipated_columns`; it must carry per-thermal costs so the
        // decision column's objective coefficient is non-zero.
        let thermal_axis = n_stages + k;
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: k,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 200.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        // Thermal index 0 = anticipated (cheap); index 1 = backup (expensive). Without
        // these per-thermal cost overrides the LP has no incentive to commit
        // anticipated capacity and decision_at(t) collapses to zero, masking the
        // regression.
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 200.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: super::default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Seed the anticipated ring buffer; distinct seed values catch slot-swap bugs
        // that identical values would mask across the pre-horizon holds.
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: fixture.hydro_id,
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: fixture
                .seeds_mw
                .iter()
                .enumerate()
                .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
                    thermal_id: fixture.anticipated_id,
                    start_date: stage_date(i),
                    end_date: stage_date(i + 1),
                    value_mw,
                })
                .collect(),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .policy_graph(policy_graph)
            .build()
            .expect("build_system: valid system")
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Pre-horizon seed delivery at stage 0 with K=1 and the always-active fishing
    /// predicate. The always-active fishing equality at stage 0 pins the anticipated
    /// thermal to slot 0 of the ring buffer (the 100 MW seed) and the cost-zeroing
    /// predicate accepts that delivery at zero LP cost.
    ///
    /// Verifies that the 100 MW seed is delivered at stage 0 and that the commitment-hold
    /// delivery propagates in-study decisions for stages 1–4 (committed seed at stage 0,
    /// decision saturation, commitment-hold delivery, cost upper bound — derived inline at
    /// each AC block below).
    ///
    /// Cost bound: stage-0 backup carries 150 − 100 = 50 MW × $5000/MWh × 744 h =
    /// $186,000,000 (the 100 MW seed delivers at zero LP cost); the active-decision
    /// ceiling is 4 decision stages × 200 MW × $10/MWh × 744 h = $5,952,000 (the LP
    /// may commit less if cuts are loose, never more); plus a $1,000 tolerance.
    #[test]
    fn pre_horizon_seed_delivers_at_stage_zero_k1() {
        const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 100.0) * 744.0 * 5000.0;
        const MAX_DECISION_COST_USD: f64 = 4.0 * 200.0 * 744.0 * 10.0;
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 =
            STAGE_0_BACKUP_COST_USD + MAX_DECISION_COST_USD + COST_TOLERANCE_USD;

        let system = build_system(&FIXTURE_K1);
        let config = super::build_config(FIXTURE_K1.iterations);
        let mut setup = build_setup_in_code(system, &config);

        // One iteration suffices: the fishing equality pins the seed regardless of cut
        // quality; the generous cost bound absorbs a loose 1-iteration cut.
        let scenario_results = run_simulation(&mut setup, FIXTURE_K1.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            5,
            "scenario must contain one record per study stage (n_stages=5)",
        );

        let anticipated_thermal_id: i32 = 31;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };
        let committed_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_committed_mw)
        };

        let c0 = committed_at(0).expect(
            "committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 100 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at stage 0 with K=1.",
        );
        assert!(
            (c0 - 100.0).abs() < 1e-6,
            "committed_at(0) = {c0} MW, expected 100.0 MW (the seed). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 100.0 MW of the ring buffer.",
        );

        // Active-decision stages are t ∈ {0,1,2,3} (t + K < n_stages, i.e. t + 1 < 5).
        for t in 0..4_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "decision_at({t}) is None; anticipated thermal id=31 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 1 < 5)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage.",
            );
            assert!(
                dt <= 200.0 + 1e-6,
                "decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
            );
        }

        // Commitment-hold delivery invariant (K=1): the in-study decision made at
        // stage t-1 (delivered at stage t) is latched at slot 0 and read by stage t's
        // fishing equality.
        for t in 1..5_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 1,
                )
            });
            let d_prev = decision_at(t - 1).unwrap_or_else(|| {
                panic!(
                    "decision_at({}) is None (needed to check ring-buffer \
                 invariant at stage {t})",
                    t - 1,
                )
            });
            assert!(
                (ct - d_prev).abs() < 1e-6,
                "commitment-hold delivery: committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev} MW (within 1e-6 MW). \
             The ring buffer is not correctly propagating in-study decisions.",
                t - 1,
            );
        }

        // Sum immediate_cost (LP objective minus theta), not total_cost — total_cost
        // includes the theta approximation artefact.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If the seed is not delivered (legacy predicate), stage-0 backup covers \
         150 MW instead of 50 MW, producing ~$558M >> $191.95M.",
        );
    }

    /// Pre-horizon seed delivery across two pre-horizon stages with K=2 and the
    /// always-active fishing predicate.
    ///
    /// With `K = 2`, `n_stages = 5`, a single anticipated thermal (id=42), and
    /// `past_anticipated_commitments` windows tiling stage 0 and stage 1 with
    /// `value_mw` 80.0 and 50.0 respectively (`FIXTURE_K2.seeds_mw = [80.0, 50.0]`),
    /// the LP must:
    ///
    /// 1. Deliver `committed_at(0) == 80.0 MW` — the always-active fishing
    ///    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
    ///    buffer, which holds the 80.0 MW seed (`seeds_mw[0]`). The cost-zeroing
    ///    predicate zeros the per-block objective for this column so the LP
    ///    accepts the delivery at zero additional cost.
    ///
    /// 2. Deliver `committed_at(1) == 50.0 MW` — the delivery-1 seed
    ///    (`seeds_mw[1] = 50.0`) is HELD at its modular slot `1 mod k_max = 1`
    ///    across stage 0 (the same-slot carry identity), and stage 1's always-active
    ///    fishing equality reads that maturing slot (`1 mod k_max = 1`) = 50.0 MW.
    ///    This is the K=2-specific assertion that the K=1 delivery test cannot
    ///    reach: K=1 has only one pre-horizon stage, so there is no commitment held
    ///    across a stage to exercise.
    ///
    /// 3. Satisfy `committed_at(t) ≈ decision_at(t-2)` for t ∈ {2,3,4} — the
    ///    K=2 matures decisions two stages after they are made. With K=2, the
    ///    decision written at stage t is latched at its delivery-target modular slot
    ///    (`(t+2) mod k_max`) and HELD there (same-slot carry) until the fishing
    ///    equality delivers it at stage t+2. This is the t-2 offset
    ///    (compare: K=1 delivery test uses t-1 offset).
    ///
    /// 4. Saturate `decision_at(t) ≈ 200 MW` (max_gen) for t ∈ {0,1,2} (stages
    ///    where `t + K < n_stages`, i.e., `t + 2 < 5`) — the anticipated thermal
    ///    costs $10/MWh vs the backup's $5000/MWh, and the per-block cost of the
    ///    decision column is non-zero only at the decision stage, so the LP
    ///    saturates commitment to avoid future backup dispatch.
    ///
    /// 5. Satisfy the analytical cost bound:
    ///    - Stage 0: seed delivers 80 MW; backup covers 70 MW
    ///      × $5000/MWh × 744 h = $260,400,000.
    ///    - Stage 1: the held seed delivers 50 MW; backup covers 100 MW
    ///      × $5000/MWh × 744 h = $372,000,000.
    ///    - Stages 2–4 delivery: anticipated covers ≥ 150 MW load (zeroed cost).
    ///    - Decision cost ≤ 3 × 200 MW × $10/MWh × 744 h = $4,464,000.
    ///    - Total ≤ $636,865,000.
    #[test]
    fn pre_horizon_seed_delivers_pre_horizon_stages_k2() {
        const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 80.0) * 744.0 * 5000.0;
        const STAGE_1_BACKUP_COST_USD: f64 = (150.0 - 50.0) * 744.0 * 5000.0;
        const MAX_DECISION_COST_USD: f64 = 3.0 * 200.0 * 744.0 * 10.0;
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 = STAGE_0_BACKUP_COST_USD
            + STAGE_1_BACKUP_COST_USD
            + MAX_DECISION_COST_USD
            + COST_TOLERANCE_USD;

        let system = build_system(&FIXTURE_K2);
        let config = super::build_config(FIXTURE_K2.iterations);
        let mut setup = build_setup_in_code(system, &config);

        // 5 iterations: after 1, stage-1 decisions are too loose to satisfy the cost
        // bound.
        let scenario_results = run_simulation(&mut setup, FIXTURE_K2.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            5,
            "scenario must contain one record per study stage (n_stages=5)",
        );

        let anticipated_thermal_id: i32 = 42;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };
        let committed_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_committed_mw)
        };

        let c0 = committed_at(0).expect(
            "committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 80 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at pre-horizon stages.",
        );
        assert!(
            (c0 - 80.0).abs() < 1e-6,
            "committed_at(0) = {c0} MW, expected 80.0 MW (seeds_mw[0]). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 80.0 MW of the ring buffer.",
        );

        // K=2-specific: a commitment held across pre-horizon stages 0->1.
        let c1 = committed_at(1).expect(
        "committed_at(1) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 1. \
         If committed_at(1) is None, the fishing constraint is absent for stage 1.",
    );
        assert!(
            (c1 - 50.0).abs() < 1e-6,
            "committed_at(1) = {c1} MW, expected 50.0 MW (seeds_mw[1]). \
         The delivery-1 seed (50.0 MW) is held at its modular slot 1 across stage 0 \
         (same-slot carry), and stage 1's fishing equality must read that maturing \
         slot (1 mod k_max = 1). If the result is 80.0 MW, the hold is not keeping \
         the delivery-1 seed in its own slot between pre-horizon stages.",
        );

        // K=2: decisions mature 2 stages after being made.
        for t in 2..5_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 2,
                )
            });
            let d_prev2 = decision_at(t - 2).unwrap_or_else(|| {
                panic!(
                    "decision_at({}) is None (needed to check K=2 \
                 delivery-lag invariant at stage {t})",
                    t - 2,
                )
            });
            assert!(
                (ct - d_prev2).abs() < 1e-6,
                "K=2 commitment-hold delivery: committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev2} MW (within 1e-6 MW). \
             With K=2, decisions mature two stages later: the decision is latched at \
             its delivery-target modular slot and HELD there (same-slot carry) until \
             fished at delivery. The ring is not correctly propagating in-study \
             decisions.",
                t - 2,
            );
        }

        // Active-decision stages: t + 2 < 5; LP saturates on cost ratio.
        for t in 0..3_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "decision_at({t}) is None; anticipated thermal id=42 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 2 < 5)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage (stage {}).",
                t + 2,
            );
            assert!(
                dt <= 200.0 + 1e-6,
                "decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
            );
        }

        // Use immediate_cost, not total_cost (which includes the theta approximation
        // artefact).
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         STAGE_1_BACKUP_COST_USD=${STAGE_1_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If neither seed is delivered, stage-0+1 backup covers 300 MW total \
         producing ~$1,116M >> $636.865M.",
        );
    }

    /// Pre-horizon seed delivery across three pre-horizon stages with K=3 and the
    /// always-active fishing predicate.
    ///
    /// With `K = 3`, `n_stages = 6`, a single anticipated thermal (id=52), and
    /// `past_anticipated_commitments` windows tiling stages 0-2 with `value_mw`
    /// 50.0, 30.0, 10.0 respectively (`FIXTURE_K3.seeds_mw = [50.0, 30.0, 10.0]`),
    /// the LP must:
    ///
    /// 1. Deliver `committed_at(0) == 50.0 MW` — the always-active fishing
    ///    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
    ///    buffer, which holds the 50.0 MW seed (`seeds_mw[0]`). The cost-zeroing
    ///    predicate zeros the per-block objective for this column so the LP
    ///    accepts the delivery at zero additional cost.
    ///
    /// 2. Deliver `committed_at(1) == 30.0 MW` — the delivery-1 seed
    ///    (`seeds_mw[1] = 30.0`) is HELD at its modular slot `1 mod k_max = 1`
    ///    across stage 0 (same-slot carry), and stage 1's always-active fishing
    ///    equality reads that maturing slot (`1 mod k_max = 1`) = 30.0 MW. This is
    ///    one of the two K=3-specific assertions that the K=1 and K=2 delivery
    ///    tests cannot reach: K=3 has three pre-horizon stages, with commitments
    ///    held across one and two stages respectively.
    ///
    /// 3. Deliver `committed_at(2) == 10.0 MW` — the delivery-2 seed
    ///    (`seeds_mw[2] = 10.0`) is HELD at its modular slot `2 mod k_max = 2`
    ///    across stages 0 and 1, and stage 2's always-active fishing equality reads
    ///    that maturing slot (`2 mod k_max = 2`) = 10.0 MW at zero LP cost. This is
    ///    the deepest pre-horizon delivery assertion in the entire anticipated test
    ///    suite.
    ///
    /// 4. Satisfy `committed_at(t) ≈ decision_at(t-3)` for t ∈ {3, 4, 5} — the
    ///    K=3 matures decisions three stages after they are committed. With K=3,
    ///    the decision written at stage t is latched at its delivery-target modular
    ///    slot (`(t+3) mod k_max`) and HELD there (same-slot carry) until the fishing
    ///    equality delivers it at stage t+3. This is the t-3 offset
    ///    (compare: K=1 uses t-1, K=2 uses t-2).
    ///
    /// 5. Saturate `decision_at(t) > 0` and `≤ max_gen + 1e-6` for t ∈ {0, 1, 2}
    ///    (stages where `t + K < n_stages`, i.e., `t + 3 < 6`, giving t ∈ {0,1,2})
    ///    — the anticipated thermal costs $10/MWh vs the backup's $5000/MWh, and
    ///    the per-block cost of the decision column is non-zero only at the
    ///    decision stage, so the LP commits a non-trivial amount to avoid future
    ///    backup dispatch.
    ///
    /// 6. Satisfy the analytical cost bound:
    ///    - Stage 0: seed delivers 50 MW; backup covers 100 MW
    ///      × $5000/MWh × 744 h = $372,000,000.
    ///    - Stage 1: the held seed delivers 30 MW; backup covers 120 MW
    ///      × $5000/MWh × 744 h = $446,400,000.
    ///    - Stage 2: the held seed delivers 10 MW; backup covers 140 MW
    ///      × $5000/MWh × 744 h = $520,800,000.
    ///    - Stages 3–5 delivery: anticipated delivers ≥ 150 MW load (zeroed cost).
    ///    - Decision cost ≤ 3 × 200 MW × $10/MWh × 744 h = $4,464,000.
    ///    - Tolerance: $1,000.
    ///    - Total upper bound: $1,343,665,000.
    #[test]
    fn pre_horizon_seed_delivers_three_pre_horizon_stages_k3() {
        const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 50.0) * 744.0 * 5000.0;
        const STAGE_1_BACKUP_COST_USD: f64 = (150.0 - 30.0) * 744.0 * 5000.0;
        const STAGE_2_BACKUP_COST_USD: f64 = (150.0 - 10.0) * 744.0 * 5000.0;
        const MAX_DECISION_COST_USD: f64 = 3.0 * 200.0 * 744.0 * 10.0;
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 = STAGE_0_BACKUP_COST_USD
            + STAGE_1_BACKUP_COST_USD
            + STAGE_2_BACKUP_COST_USD
            + MAX_DECISION_COST_USD
            + COST_TOLERANCE_USD;

        let system = build_system(&FIXTURE_K3);
        let config = super::build_config(FIXTURE_K3.iterations);
        let mut setup = build_setup_in_code(system, &config);

        // 5 iterations let cuts sharpen so stage-2 decisions reach max_gen, covering
        // stage-5 delivery at zero backup cost.
        let scenario_results = run_simulation(&mut setup, FIXTURE_K3.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            6,
            "scenario must contain one record per study stage (n_stages=6)",
        );

        let anticipated_thermal_id: i32 = 52;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };
        let committed_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_committed_mw)
        };

        let c0 = committed_at(0).expect(
            "committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 50 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at pre-horizon stages.",
        );
        assert!(
            (c0 - 50.0).abs() < 1e-6,
            "committed_at(0) = {c0} MW, expected 50.0 MW (seeds_mw[0]). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 50.0 MW of the ring buffer.",
        );

        let c1 = committed_at(1).expect(
        "committed_at(1) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 1. \
         If committed_at(1) is None, the fishing constraint is absent for stage 1.",
    );
        assert!(
            (c1 - 30.0).abs() < 1e-6,
            "committed_at(1) = {c1} MW, expected 30.0 MW (seeds_mw[1]). \
         The delivery-1 seed (30.0 MW) is held at its modular slot 1 across stage 0 \
         (same-slot carry), and stage 1's fishing equality must read that maturing \
         slot (1 mod k_max = 1). If the result is 50.0 MW, the hold is not keeping \
         the delivery-1 seed in its own slot between pre-horizon stages 0 and 1.",
        );

        let c2 = committed_at(2).expect(
        "committed_at(2) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 2. \
         If committed_at(2) is None, the fishing constraint is absent for stage 2.",
    );
        assert!(
            (c2 - 10.0).abs() < 1e-6,
            "committed_at(2) = {c2} MW, expected 10.0 MW (seeds_mw[2]). \
         The delivery-2 seed (10.0 MW) is held at its modular slot 2 across stages 0 \
         and 1 (same-slot carry), and stage 2's fishing equality must read that \
         maturing slot (2 mod k_max = 2). If the result is 30.0 MW, the hold is \
         leaking into the delivery-1 seed; if 50.0 MW, into the delivery-0 seed.",
        );

        for t in 3..6_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 3,
                )
            });
            let d_prev3 = decision_at(t - 3).unwrap_or_else(|| {
                panic!(
                    "decision_at({}) is None (needed to check K=3 \
                 delivery-lag invariant at stage {t})",
                    t - 3,
                )
            });
            assert!(
                (ct - d_prev3).abs() < 1e-6,
                "K=3 commitment-hold delivery: committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev3} MW (within 1e-6 MW). \
             With K=3, decisions mature three stages later: the decision is latched \
             at its delivery-target modular slot and HELD there (same-slot carry) \
             until fished at delivery. The ring is not correctly propagating \
             in-study decisions.",
                t - 3,
            );
        }

        for t in 0..3_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "decision_at({t}) is None; anticipated thermal id=52 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 3 < 6)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage (stage {}).",
                t + 3,
            );
            assert!(
                dt <= 200.0 + 1e-6,
                "decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
            );
        }

        // Sum immediate_cost, not total_cost — total_cost includes the theta
        // approximation artefact.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         STAGE_1_BACKUP_COST_USD=${STAGE_1_BACKUP_COST_USD:.2}, \
         STAGE_2_BACKUP_COST_USD=${STAGE_2_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If none of the seeds are delivered, 3 pre-horizon stages use backup \
         for all 150 MW: 3 × 150 MW × 744 h × $5000/MWh = $1,674,000,000 >> \
         $1,343,665,000.",
        );
    }
}
mod anticipated_d_t_saturation {
    //! `d_t`-saturation regression tests for an anticipated thermal across lead_stages
    //! K = 2, 3. Each test trains a small in-code study, runs a one-scenario
    //! simulation, and asserts that the anticipated-decision variable `d_t` commits to
    //! the load level at every active stage (`t + K < n_stages`) and is absent at the
    //! strict-boundary inactive stages. The K=3 test additionally checks the
    //! commitment-hold delivery `committed_at(t) ≈ decision_at(t − K)`. Each K's
    //! economic-reasoning derivation lives on its test function.

    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };
    use super::common::run_simulation;

    // ---------------------------------------------------------------------------
    // Per-K fixture table
    // ---------------------------------------------------------------------------

    /// Per-K parameters for a `d_t`-saturation fixture. Each `#[test]` builds an
    /// independent `System` from one entry, so an entity id may name different roles
    /// across K (id 3 is the K=2 hydro but the K=3 anticipated thermal) without
    /// colliding — the ids are local to each built system.
    struct SaturationFixture {
        n_stages: usize,
        /// Anticipated `lead_stages` K — also the ring-buffer depth.
        k: usize,
        bus_id: EntityId,
        anticipated_id: EntityId,
        anticipated_name: &'static str,
        backup_id: EntityId,
        backup_name: &'static str,
        hydro_id: EntityId,
        /// Anticipated ring-buffer seeds, MW (length `k`); zero isolates in-horizon
        /// behaviour from any seeding artefact.
        seeds_mw: &'static [f64],
        iterations: usize,
    }

    const FIXTURE_K2: SaturationFixture = SaturationFixture {
        n_stages: 6,
        k: 2,
        bus_id: EntityId(1),
        anticipated_id: EntityId(2),
        anticipated_name: "T_ant",
        backup_id: EntityId(4),
        backup_name: "T_backup",
        hydro_id: EntityId(3),
        seeds_mw: &[0.0, 0.0],
        iterations: 10,
    };

    const FIXTURE_K3: SaturationFixture = SaturationFixture {
        n_stages: 8,
        k: 3,
        bus_id: EntityId(1),
        anticipated_id: EntityId(3),
        anticipated_name: "T_ant_k3",
        backup_id: EntityId(4),
        backup_name: "T_backup_k3",
        hydro_id: EntityId(5),
        seeds_mw: &[0.0, 0.0, 0.0],
        iterations: 15,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the thermal `System` for one `d_t`-saturation fixture: one anticipated
    /// thermal (cheap, $10/MWh, max 200 MW) at `fixture.anticipated_id`, one backup
    /// ($5000/MWh, max 500 MW) at `fixture.backup_id`, and a trivial hydro (1 hm³,
    /// zero inflow, 1 MW max_gen) at `fixture.hydro_id` that keeps the model in the
    /// thermal regime without adding an interpretable hydro state. Load is 150 MW
    /// across all stages; seeds (`fixture.seeds_mw`) are zero to isolate the
    /// in-horizon behaviour from any seeding artefact.
    fn build_system(fixture: &SaturationFixture) -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        let k = fixture.k;
        let n_stages = fixture.n_stages;

        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let bus = make_bus(
            fixture.bus_id,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = fixture.anticipated_id;
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: fixture.anticipated_name.to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: fixture.bus_id,
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(k as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            fixture.backup_id,
            ThermalSpec {
                name: fixture.backup_name.to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: fixture.bus_id,
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
                cost_per_mwh: 5000.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            fixture.hydro_id,
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: fixture.bus_id,
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: fixture.hydro_id,
                stage_id: i as i32,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: fixture.bus_id,
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 1.0,
                max_generation_mw: 1.0,
                ..Default::default()
            }
        }

        // The per-thermal costs must be patched afterwards (ResolvedBounds::new takes
        // one default for ALL thermals) so the objective distinguishes the cheap
        // anticipated thermal from the expensive backup. The patch must extend over the
        // padding region `[n_stages, n_stages + k)` — the delivery-stage axis read by
        // `fill_anticipated_columns` — or the decision column's objective coefficient
        // stays zero and the regression is masked.
        let thermal_axis = n_stages + k;
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: k,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 200.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0; // index 0 = anticipated (cheap)
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 200.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0; // index 1 = backup (expensive)
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: super::default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: fixture.hydro_id,
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: fixture
                .seeds_mw
                .iter()
                .enumerate()
                .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
                    thermal_id: anticipated_id,
                    start_date: stage_date(i),
                    end_date: stage_date(i + 1),
                    value_mw,
                })
                .collect(),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid system")
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Assert `anticipated_decision_mw` commits to load level (150 MW) for every
    /// active decision stage (`t + K < n_stages`, i.e. `t in {0,1,2,3}`) and is
    /// `None` for the boundary stages, in a K=2, 6-stage fixture.
    ///
    /// ## Economic reasoning (pins the asserted optimum)
    ///
    /// In a 6-stage K=2 study with a 500x cost asymmetry (anticipated thermal at
    /// $10/MWh vs backup at $5000/MWh), load = 150 MW < max_gen = 200 MW and excess
    /// generation costs $0. The LP optimum is therefore `d_t = load = 150 MW` at every
    /// stage `t` where `t + K < n_stages` (`t in {0,1,2,3}`): over-committing to 200 MW
    /// costs an extra 50 MW × $10/MWh × 744 h = $372k/stage with no offsetting benefit.
    /// The 500x asymmetry only forces the anticipated thermal to dispatch at all
    /// (reaching load level), not to saturate at max_gen.
    ///
    /// A regression in the anticipated-state cut-coefficient mapping
    /// (`state_to_lp_column`'s `Less` branch, for `k >= 2`) drops `d_t` to 0 at the
    /// intermediate active stages, forcing backup at $5000/MWh.
    #[test]
    fn d_t_commits_to_load_for_every_active_stage_k2() {
        let k: usize = 2;
        let n_stages: usize = 6;
        let active_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k < n_stages).collect();
        let inactive_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k >= n_stages).collect();

        let system = build_system(&FIXTURE_K2);
        let config = super::build_config(FIXTURE_K2.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let scenario_results = run_simulation(&mut setup, FIXTURE_K2.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            n_stages,
            "scenario must contain one stage record per study stage",
        );

        // The anticipated thermal has entity id=2 (see FIXTURE_K2).
        let anticipated_thermal_id: i32 = 2;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };

        let load_mw = 150.0_f64;
        let tol = 1e-3_f64;
        for t in &active_stages {
            let d_t = decision_at(*t).unwrap_or_else(|| {
                panic!(
                    "anticipated_decision_mw must be Some at active stage t={t} (t + K < n_stages)"
                )
            });
            assert!(
                (d_t - load_mw).abs() < tol,
                "d_t at stage {t} must saturate at load=150 MW: \
             got {d_t} (delta = {delta:.6} MW, tol = {tol} MW). \
             Pre-fix behaviour: d_t ≈ 0 for t >= 1 due to cut-coefficient \
             corruption in state_to_lp_column (Less branch), forcing backup \
             at $5000/MWh.",
                delta = (d_t - load_mw).abs(),
            );
        }

        for t in &inactive_stages {
            assert!(
                decision_at(*t).is_none(),
                "anticipated_decision_mw must be None at inactive stage t={t} \
             (t + K >= n_stages; strict-boundary predicate excludes this stage)",
            );
        }
    }

    /// Assert that `anticipated_decision_mw` commits to load level (150 MW) for every
    /// active decision stage in a K=3, 8-stage fixture, and that the commitment-hold delivery
    /// propagates each decision to its delivery stage:
    /// - `d_t ≈ 150.0` for `t in {0, 1, 2, 3, 4}`.
    /// - `anticipated_decision_mw` is `None` for `t in {5, 6, 7}`.
    /// - `committed_at(t) ≈ decision_at(t - 3)` for `t in {3, 4, 5, 6, 7}`.
    ///
    /// ## Bug being guarded against
    ///
    /// In an 8-stage K=3 study with a 500x cost asymmetry (anticipated thermal at
    /// $10/MWh vs backup at $5000/MWh), the LP should commit the anticipated
    /// thermal to exactly `load = 150 MW` at every stage `t` where `t + K < n_stages`
    /// (i.e. `t in {0, 1, 2, 3, 4}`). Over-committing to `max_gen = 200 MW` costs
    /// an extra 50 MW × $10/MWh × 744 h = $372k/stage with zero offsetting benefit
    /// (excess generation is free), so the LP optimum is `d_t = load = 150 MW`,
    /// not `max_gen = 200 MW`.
    ///
    /// The K=3 propagation chain is longer than K=2: cuts at stage `t` propagate
    /// to predecessor's slot 1, which is `state_col[slot 2]`, whose value at K=3
    /// is `incoming_slot_2 - d_{t-1}` if the decision-write coefficient is at
    /// slot K-1 = 2. This variant exercises the multi-step propagation through
    /// all three slot positions, confirming the fix is general and not K=2-specific.
    ///
    /// The cut coefficients for the anticipated-state columns are corrupted for
    /// `k >= 2` in `state_to_lp_column`'s `Less` branch, so the policy at those
    /// stages receives no incentive to commit. At K=3 the corruption propagates
    /// through all three slot positions, making it a stricter test than K=2.
    #[test]
    fn d_t_commits_to_load_for_every_active_stage_k3() {
        let k: usize = 3;
        let n_stages: usize = 8;
        let active_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k < n_stages).collect();
        let inactive_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k >= n_stages).collect();
        let committed_stages: Vec<usize> = (0..n_stages).filter(|&t| t >= k).collect();

        let system = build_system(&FIXTURE_K3);
        let config = super::build_config(FIXTURE_K3.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let scenario_results = run_simulation(&mut setup, FIXTURE_K3.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            n_stages,
            "scenario must contain one stage record per study stage",
        );

        // The anticipated thermal has entity id=3 (see FIXTURE_K3).
        let anticipated_thermal_id: i32 = 3;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };
        let committed_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_committed_mw)
        };

        let load_mw = 150.0_f64;
        let tol = 1e-3_f64;
        for t in &active_stages {
            let d_t = decision_at(*t).unwrap_or_else(|| {
                panic!(
                    "anticipated_decision_mw must be Some at active stage t={t} \
                 (t + K < n_stages, K=3)",
                )
            });
            assert!(
                (d_t - load_mw).abs() < tol,
                "d_t at stage {t} must saturate at load=150 MW: \
             got {d_t} (delta = {delta:.6} MW, tol = {tol} MW). \
             Pre-fix behaviour: d_t ≈ 0 for t >= 1 due to cut-coefficient \
             corruption in state_to_lp_column (Less branch). \
             At K=3 the corruption spans all three slot positions.",
                delta = (d_t - load_mw).abs(),
            );
        }

        for t in &inactive_stages {
            assert!(
                decision_at(*t).is_none(),
                "anticipated_decision_mw must be None at inactive stage t={t} \
             (t + K >= n_stages, K=3; strict-boundary predicate excludes this stage)",
            );
        }

        // The decision d_{t-3} made at stage t-3 (delivered at stage t) is latched at
        // its delivery-target modular slot `t mod k_max` and HELD there (same-slot
        // carry) across stages t-2 and t-1, where stage t's fishing constraint reads
        // it. So committed_at(t) = d_{t-3}.
        for t in &committed_stages {
            let c_t = committed_at(*t).unwrap_or_else(|| {
                panic!(
                    "anticipated_committed_mw must be Some at stage t={t} \
                 (t >= K=3; fishing constraint is active)",
                )
            });
            let d_prev = decision_at(*t - k).unwrap_or_else(|| {
                panic!(
                    "anticipated_decision_mw must be Some at stage t-K={prev} \
                 (used to verify commitment-hold delivery at delivery stage t={t})",
                    prev = *t - k,
                )
            });
            assert!(
                (c_t - d_prev).abs() < tol,
                "commitment-hold delivery invariant violated at t={t}: \
             committed_at({t}) = {c_t:.6} MW but decision_at({prev}) = {d_prev:.6} MW \
             (delta = {delta:.6} MW, tol = {tol} MW, K=3). \
             The ring buffer should propagate d_{{t-3}} through three slot \
             positions so it reaches slot 0 at stage t.",
                prev = *t - k,
                delta = (c_t - d_prev).abs(),
            );
        }
    }
}
mod anticipated_forward_pass {
    //! End-to-end integration test verifying the anticipated-state commitment-hold
    //! ring evolution across the full forward pass for a 5-stage K=2 system.
    //!
    //! Commitment-hold semantics: the LP's own in-LP definition rows resolve the
    //! ring transition — an interior slot's outgoing column is pinned to its OWN
    //! incoming value (`slot[s]_out = slot[s]_in`, the same-slot carry identity),
    //! and the deposit slot's outgoing column is pinned to the fresh decision
    //! (`slot(delivery mod k_max)_out = decision_col`) — so a commitment is HELD in
    //! its own modular slot until fished, never shifted between slots. The forward
    //! pass reads these outgoing columns by identity
    //! (`StateSpace::state_to_lp_column`), with no Rust-side shift step.

    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build a 5-stage system with one anticipated thermal (K=2, seeded
    /// `[100.0, 50.0]`) and one backup thermal that alone covers the 150 MW load, so
    /// the LP is always feasible.
    fn build_system() -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(2);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 500.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                ..Default::default()
            },
        );

        let n_stages = 5_usize;
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        let k_max: usize = 2;
        let n_st = n_stages;

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: super::default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![
                AnticipatedCommitmentHistory {
                    thermal_id: anticipated_id,
                    start_date: stage_date(0),
                    end_date: stage_date(1),
                    value_mw: 100.0,
                },
                AnticipatedCommitmentHistory {
                    thermal_id: anticipated_id,
                    start_date: stage_date(1),
                    end_date: stage_date(2),
                    value_mw: 50.0,
                },
            ],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Integration test
    // ---------------------------------------------------------------------------

    /// Verify that the anticipated-state commitment-hold ring evolves correctly
    /// across all 5 stages of the forward pass under the HOLD geometry: a
    /// commitment is HELD at its delivery-target modular slot (`m mod k_max`)
    /// until it matures — the ring never SHIFTS slots.
    ///
    /// The block layout is slot-major, plant-minor; with `n_anticipated=1`,
    /// `k_max=2` it is `[slot0_plant0, slot1_plant0]`. IC seeds: delivery-0 =
    /// 100 MW at slot `0 mod 2 = 0` (matures/is fished at stage 0), delivery-1 =
    /// 50 MW at slot `1 mod 2 = 1` (held through stage 0, matures at stage 1).
    ///
    /// `state_at_capture` is filled by two paths:
    /// - **Stage 0**: forward pass writes the outgoing state of stage 0.
    /// - **Stages 1..=4**: backward pass stores the forward outgoing of stage `t-1`
    ///   as its trial point `x_hat` — so `basis_cache[1]` (trial point of stage 1)
    ///   equals `basis_cache[0]` (forward outgoing of stage 0).
    #[test]
    fn five_stage_k2_anticipated_state_ring_buffer_evolution() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        // One iteration populates basis_cache for every stage.
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "at least 1 iteration must complete; got {}",
            outcome.result.iterations
        );

        let state = setup.stage_state();
        let n_ant = state.n_anticipated;
        let k_max = state.k_max;
        assert_eq!(n_ant, 1, "fixture must have exactly 1 anticipated thermal");
        assert_eq!(k_max, 2, "fixture must have k_max = 2");

        let ant_start = state.commit_out.start;

        let basis_cache = &outcome.result.basis_cache;
        assert_eq!(
            basis_cache.len(),
            5,
            "basis_cache must have one entry per study stage"
        );

        // Forward pass stores the outgoing state of stage 0. HOLD geometry: the
        // delivery-1 seed (50.0) is HELD at slot 1 (its modular slot, carried
        // because delivery-1 does not mature until stage 1) — under the retired
        // shift geometry it would instead move to slot 0. Slot 0 is the deposit
        // slot for the fresh decision d_0 (delivery-2, slot `2 mod 2 = 0`), so it
        // carries the LP decision d_0 ∈ [0, 100].
        let s0 = basis_cache[0]
            .as_ref()
            .expect("stage 0 basis must be captured")
            .state_at_capture
            .as_slice();
        assert!(
            (s0[ant_start + 1] - 50.0).abs() < 1e-9,
            "stage 0 slot 1 must HOLD the delivery-1 seed 50.0 (carry, not shift); got {}",
            s0[ant_start + 1]
        );
        assert!(
            (-0.01..=100.01).contains(&s0[ant_start]),
            "stage 0 slot 0 (= decision d_0) must lie in [0, 100]; got {}",
            s0[ant_start]
        );

        // Backward pass for stage 1 stores the forward outgoing of stage 0 as its
        // trial point x_hat, so basis_cache[1] equals basis_cache[0].
        let s1 = basis_cache[1]
            .as_ref()
            .expect("stage 1 basis must be captured")
            .state_at_capture
            .as_slice();
        assert!(
            (s1[ant_start] - s0[ant_start]).abs() < 1e-9,
            "stage 1 slot 0 ({}) must equal stage 0 slot 0 ({}) — both carry \
         the outgoing state of forward stage 0",
            s1[ant_start],
            s0[ant_start],
        );
        assert!(
            (s1[ant_start + 1] - s0[ant_start + 1]).abs() < 1e-9,
            "stage 1 slot 1 ({}) must equal stage 0 slot 1 ({}) — both carry \
         d_0 from the forward pass",
            s1[ant_start + 1],
            s0[ant_start + 1],
        );

        // HOLD (carry, not shift): consecutive forward outgoing states hold the
        // non-maturing slot UNCHANGED. `basis_cache[t]` holds forward outgoing of
        // stage `t-1` (with `basis_cache[1] == basis_cache[0]`); the slot NOT
        // maturing at the newer forward stage is `t mod 2`, and it must carry the
        // previous stage's SAME slot verbatim. Under the retired shift geometry
        // this comparison read `s_curr[slot 0] == s_prev[slot 1]` instead.
        for t in 2..5_usize {
            let s_curr = basis_cache[t]
                .as_ref()
                .unwrap_or_else(|| panic!("stage {t} basis must be captured"))
                .state_at_capture
                .as_slice();
            let s_prev = basis_cache[t - 1]
                .as_ref()
                .unwrap_or_else(|| panic!("stage {} basis must be captured", t - 1))
                .state_at_capture
                .as_slice();

            let carried = ant_start + (t % 2);
            assert!(
                (s_curr[carried] - s_prev[carried]).abs() < 1e-9,
                "stage {t}: commitment-hold must carry slot {} unchanged from the \
                 previous forward stage (same slot, no shift); got {} vs {}",
                t % 2,
                s_curr[carried],
                s_prev[carried],
            );
        }

        // Every captured commitment-hold value stays within the decision/seed
        // bound [0, 100] (decisions d_t, the 50 MW seed, or a frozen 0).
        for t in 0..5_usize {
            let s_t = basis_cache[t]
                .as_ref()
                .unwrap_or_else(|| panic!("stage {t} basis must be captured"))
                .state_at_capture
                .as_slice();
            for slot in 0..2_usize {
                let v = s_t[ant_start + slot];
                assert!(
                    (-0.01..=100.01).contains(&v),
                    "stage {t} slot {slot} must lie within the commitment bound [0, 100]; got {v}",
                );
            }
        }
    }
}
mod anticipated_closed_form_lb_k1_single_thermal {
    //! Closed-form lower-bound canary for anticipated thermals.
    //!
    //! Two-stage, K=1, no hydro, no stochastic noise, single deterministic opening
    //! per stage. The lower bound returned by `train` is hand-derivable from the LP
    //! coefficients written by the `lp::builder` column/row fill paths, so any
    //! deviation from `EXPECTED_LB` flags a value-correctness regression that the
    //! larger structural-only integration tests cannot pin (no closed form exists
    //! for those fixtures).
    //!
    //! ## Fixture
    //!
    //! - `n_stages = 2`, single 1-hour block per stage.
    //! - 1 bus (deficit cost `1000 $/MWh`, excess cost `0 $/MWh`).
    //! - 1 anticipated thermal `T_ant` (id=2): `cost = c_a = 10 $/MWh`,
    //!   `max_gen = M = 100 MW`, `lead_stages = K = 1`.
    //! - 1 backup thermal `T_b` (id=3): `cost = c_b = 100 $/MWh`,
    //!   `max_gen = B = 200 MW`. Strict ordering `c_a < c_b` is what makes the
    //!   anticipated commitment cheaper at delivery than backup.
    //! - Load `D = 50 MW` at both stages.
    //! - `past_anticipated_commitments` has one window tiling stage 0 with
    //!   `(thermal_id=2, value_mw=0.0)`. The past must be zero so that any
    //!   non-zero anticipated
    //!   delivery observed at stage 1 is attributable to the stage-0 decision.
    //! - 1 deterministic opening per stage.
    //! - Default `HorizonGraph::annual_discount_rate = 0.0`, so every
    //!   `discount_factors[t] = 1.0` and `cumulative_discount_factors[t] = 1.0`.
    //!
    //! ## Closed-form derivation
    //!
    //! The fishing constraint is always active for every anticipated plant at
    //! every stage, including stage 0. The fishing row `g_a_t − x_state_t = 0` is
    //! therefore emitted at stage 0 as well, and `fill_thermal_columns` skips the
    //! per-block cost of the anticipated column at stage 0 (never written, leaving
    //! it at zero; the anticipated thermal is detected via
    //! `AnticipatedPlants::local_of`, same always-active path). See the K=1
    //! sign-chain table for the cut-coefficient sign convention that applies
    //! here.
    //!
    //! Variables (all in MW; subscript denotes stage):
    //! - `g_a_t` — per-block anticipated thermal generation at stage `t`.
    //! - `g_b_t` — per-block backup thermal generation at stage `t`.
    //! - `d_ant_0` — anticipated decision placed at stage 0 (delivery at stage 1).
    //! - `θ_0` — stage-0 future-cost approximation (`≥ 0`).
    //! - `x_state_t` — anticipated-state slot 0 at stage `t`; its incoming value
    //!   is pinned by column bounds.
    //!
    //! Stage 0 (always-active fishing; decision predicate `t + K_i < n_stages` is
    //! `0 + 1 < 2` — TRUE, so `d_ant_0` is active; fishing predicate now TRUE at
    //! every stage including 0; per-block anticipated cost skipped in
    //! `fill_thermal_columns`, so it stays at zero):
    //!
    //! ```text
    //!   min  0 · g_a_0 + c_b · g_b_0 + c_a · d_ant_0 + θ_0
    //!   s.t. g_a_0 + g_b_0 + deficit_0 − excess_0 = D       (load balance)
    //!        g_a_0 − x_state_0 = 0                          (fishing row, always-active)
    //!        x_state_0 ∈ [past[0], past[0]] = [0, 0]        (pinned by column bounds, slot 0)
    //!        state_out_0 − d_ant_0 = 0                      (state-out definition row; couples decision to next-stage delivery)
    //!        θ_0 ≥ 0                                        (no cuts initially)
    //!        g_a_0 ∈ [0, M], g_b_0 ∈ [0, B], d_ant_0 ∈ [0, M]
    //!        deficit_0 ≥ 0, excess_0 ≥ 0
    //! ```
    //!
    //! The slot-0 incoming state is pinned by column bounds only — to
    //! `past[0] = 0` — with no `d_ant_0` coupling on it. The decision-vs-state
    //! coupling lives on the `state_out` definition row, which lets `d_ant_0`
    //! be optimised freely. Fishing then
    //! forces `g_a_0 = x_state_0 = 0`, so the load must be covered entirely by
    //! `g_b_0 = D` at cost `c_b · D = 5000`. `d_ant_0` is the new commitment;
    //! its objective coefficient `c_a` drives the trade-off between paying
    //! `c_a · d_ant_0` now and paying `c_b · max(0, D − d_ant_0)` at stage 1.
    //!
    //! The decision objective coefficient is set by
    //! `fill_anticipated_columns` to
    //! `c_a · TimeValue::delivery_total_hours(1) ·
    //! TimeValue::relative_delivery_discount(0, 1) = c_a · 1 · 1 = c_a`.
    //!
    //! Stage 1 (delivery; fishing always active; decision
    //! `1 + 1 < 2` — FALSE, so `d_ant_1 ∈ [0,0]` and per-block anticipated cost
    //! is skipped in `fill_thermal_columns`, so it stays at zero):
    //!
    //! ```text
    //!   min  c_b · g_b_1 + 0 · g_a_1
    //!   s.t. g_a_1 + g_b_1 + deficit_1 − excess_1 = D       (load balance)
    //!        g_a_1 − x_state_1 = 0                          (fishing row)
    //!        x_state_1 ∈ [d_ant_0, d_ant_0]                 (pinned by column bounds; incoming = d_ant_0)
    //!        g_a_1 ∈ [0, M], g_b_1 ∈ [0, B]
    //!        deficit_1 ≥ 0, excess_1 ≥ 0
    //! ```
    //!
    //! Substituting `x_state_1 = d_ant_0` and `g_a_1 = x_state_1 = d_ant_0`:
    //!
    //! - If `d_ant_0 ≤ D`: pick `g_b_1 = D − d_ant_0`, `excess_1 = deficit_1 = 0`.
    //!   Stage-1 cost = `c_b · (D − d_ant_0)`.
    //! - If `d_ant_0 > D` (≤ M): `g_b_1 = 0`, `excess_1 = d_ant_0 − D`,
    //!   `deficit_1 = 0`. Stage-1 cost = `0` (excess cost = 0).
    //!
    //! The cost-to-go function is therefore
    //! `V_1(d_ant_0) = c_b · max(0, D − d_ant_0)`.
    //!
    //! At convergence `θ_0 = V_1(d_ant_0)`, so the total objective is
    //!
    //! ```text
    //!   T(d_ant_0) = c_b · D + c_a · d_ant_0 + c_b · max(0, D − d_ant_0)
    //!              (stage-0: g_b_0=D at cost c_b·D; g_a_0=0 at cost 0·0=0)
    //! ```
    //!
    //! The variable part over `d_ant_0 ∈ [0, D]` is
    //!
    //! ```text
    //!   c_a · d_ant_0 + c_b · D − c_b · d_ant_0
    //!     = c_b · D + (c_a − c_b) · d_ant_0
    //! ```
    //!
    //! Since `c_a − c_b = 10 − 100 = −90 < 0`, this is minimised at `d_ant_0 = D`.
    //! For `d_ant_0 > D` the term becomes `c_a · d_ant_0`, strictly increasing, so
    //! the kink at `d_ant_0 = D` is the unique global minimum.
    //!
    //! ## Numerical evaluation
    //!
    //! With `c_a = 10`, `c_b = 100`, `D = 50`, `M = 100`:
    //!
    //! ```text
    //!   T*  = c_b · D                 (stage-0 backup dispatch, g_b_0 = D; g_a_0 = 0)
    //!       + c_a · D                 (stage-0 decision, d_ant_0 = D)
    //!       + 0                       (stage-1 cost, c_b · (D − D) = 0)
    //!       = (c_a + c_b) · D
    //!       = (10 + 100) · 50
    //!       = 5500.0  USD
    //! ```
    //!
    //! Sanity checks (independent traversal of the piecewise total under
    //! always-active fishing — stage-0 backup covers D; decision cost = c_a·d_ant_0):
    //!
    //! - `d_ant_0 = 0`:   `T = c_b·D + 0 + c_b·D       = 5000 + 0    + 5000 = 10000`
    //! - `d_ant_0 = D`:   `T = c_b·D + c_a·D + 0       = 5000 + 500  + 0    = 5500` ← optimum
    //! - `d_ant_0 = M`:   `T = c_b·D + c_a·M + 0       = 5000 + 1000 + 0    = 6000`
    //!
    //! The optimum is unambiguous — the kink at `d_ant_0 = D = 50` is the unique
    //! global minimum.
    //!
    //! ## Bit-for-bit determinism
    //!
    //! The fixture has no stochastic noise (one deterministic opening per stage,
    //! `std_mw = 0`, no hydro / no PAR(p) model). HiGHS is run single-threaded
    //! with `tree_seed = 42`. The LP-builder's column / row scaling is
    //! data-dependent but deterministic. We therefore assert bit-for-bit equality
    //! of `final_lb` against `EXPECTED_LB`. If the scaler introduces sub-ULP
    //! arithmetic noise on a future libhighs upgrade, switch to the relative
    //! tolerance assertion (`rel_diff < 1e-12`).

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // Closed-form fixture parameters (single source of truth).
    // ---------------------------------------------------------------------------

    const N_STAGES: usize = 2;
    const K_MAX: usize = 1;
    const BLOCK_HOURS: f64 = 1.0;

    /// MW.
    const D_LOAD: f64 = 50.0;

    /// Anticipated thermal capacity (MW). Strictly above `D_LOAD` so the optimum
    /// lies in the interior of `[0, M]` at the kink `d_ant_0 = D`.
    const M_ANT: f64 = 100.0;

    /// Backup thermal capacity (MW). Generous so backup never binds capacity.
    const B_BACK: f64 = 200.0;

    /// Anticipated marginal cost ($/MWh). Strict ordering `C_A < C_B` is what
    /// makes the anticipated commitment cheaper-than-backup at the delivery
    /// stage; the LP would otherwise be indifferent.
    const C_A: f64 = 10.0;

    /// $/MWh.
    const C_B: f64 = 100.0;

    /// Deficit cost ($/MWh). Set well above `C_B` so deficit is never preferred
    /// over backup dispatch.
    const C_DEFICIT: f64 = 1000.0;

    /// Closed-form lower bound under always-active fishing:
    /// `T* = (C_A + C_B) · D_LOAD = (10 + 100) · 50 = 5500.0` USD. See module docs
    /// for the full derivation.
    const EXPECTED_LB: f64 = (C_A + C_B) * D_LOAD; // = 5500.0

    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BACKUP_ID: EntityId = EntityId(3);
    const BUS_ID: EntityId = EntityId(1);

    // SystemBuilder::build() sorts thermals by EntityId ascending, so the
    // global thermal indices end up: 0 → id=2 (anticipated), 1 → id=3 (backup).
    const THERMAL_IDX_ANT: usize = 0;
    const THERMAL_IDX_BACKUP: usize = 1;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    fn build_system() -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        // StageCalendar requires a chronologically ordered, non-overlapping
        // calendar; one calendar day per stage keeps it valid while BLOCK_HOURS
        // alone still drives every LP-facing duration.
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(index as i64)
        };

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: C_DEFICIT,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
                cost_per_mwh: C_A,
                anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
                cost_per_mwh: C_B,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: BLOCK_HOURS,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: D_LOAD,
                std_mw: 0.0,
            })
            .collect();

        // Overrides span the full thermal axis (n_stages + k_max) so
        // `fill_anticipated_columns` reads a well-defined cost at delivery (stage 1).
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) =
                ThermalStageBounds { cost_per_mwh: C_A };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) =
                ThermalStageBounds { cost_per_mwh: C_B };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Past commitment is zero (the strict-zero precondition for
        // the K=1 ring buffer). Any non-zero delivery observed at stage 1 must
        // come from the stage-0 decision.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                start_date: stage_date(0),
                end_date: stage_date(1),
                value_mw: 0.0,
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config and setup builders
    // ---------------------------------------------------------------------------

    /// Iteration limit = 4: three iterations reach the lower envelope at the kink
    /// `d_ant_0 = D`, plus one to settle degenerate-vertex selection. Pinned by the
    /// `iterations == 4` assertion.
    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Closed-form canary: the LP-derived lower bound for the 2-stage K=1 fixture
    /// must equal `EXPECTED_LB = (C_A + C_B) · D_LOAD = 5500.0` bit-for-bit.
    #[test]
    fn anticipated_closed_form_lb_k1_single_thermal() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        assert_eq!(result.iterations, 4, "iterations mismatch");

        let actual = result.final_lb;
        assert!(actual.is_finite(), "final_lb must be finite; got {actual}");
        assert_eq!(
            actual.to_bits(),
            EXPECTED_LB.to_bits(),
            "closed-form LB mismatch: actual = {actual}, expected = {EXPECTED_LB}. \
         Under always-active fishing the answer is `(C_A + C_B) · D_LOAD = 5500.0` \
         (stage-0 backup covers load; anticipated column cost skipped in \
         fill_thermal_columns, so it stays at zero). \
         If a libhighs upgrade introduces sub-ULP arithmetic drift, switch to \
         a 1e-12 relative tolerance check.",
        );

        let gap = result.final_gap;
        assert!(
            gap.is_finite() && gap.abs() < 1e-9,
            "final_gap should be approximately zero for a fully deterministic \
         fixture; got {gap}",
        );
    }
}
mod discounted_delivery_closed_form_lb {
    use cobre_solver::ActiveSolver;

    use super::common::in_code_studies::{
        DELIVERY_ORACLE_ANNUAL_RATE, DELIVERY_ORACLE_ANTICIPATED_CAP_MW,
        DELIVERY_ORACLE_ANTICIPATED_COST, DELIVERY_ORACLE_BACKUP_COST, DELIVERY_ORACLE_LOAD_MW,
        DELIVERY_ORACLE_POST_STUDY_COST, DELIVERY_ORACLE_POST_STUDY_HOURS,
        DELIVERY_ORACLE_POST_STUDY_MIN_MW, DELIVERY_ORACLE_STAGE_DAYS,
        DELIVERY_ORACLE_STAGE0_HOURS, DELIVERY_ORACLE_STAGE1_BLOCK_HOURS,
        discounted_delivery_oracle_study,
    };
    use super::common::{StubComm, build_setup_in_code};

    const REL_TOL: f64 = 1e-9;
    const STAGE1_HOURS: f64 =
        DELIVERY_ORACLE_STAGE1_BLOCK_HOURS[0] + DELIVERY_ORACLE_STAGE1_BLOCK_HOURS[1];

    const STAGE1_ANTICIPATED_COST: f64 = DELIVERY_ORACLE_ANTICIPATED_COST[1];

    const _: () = assert!(STAGE1_HOURS != DELIVERY_ORACLE_STAGE0_HOURS);
    const _: () = assert!(STAGE1_HOURS != DELIVERY_ORACLE_POST_STUDY_HOURS);
    const _: () = assert!(DELIVERY_ORACLE_ANTICIPATED_COST[0] != STAGE1_ANTICIPATED_COST);

    /// derived: c_b·H_0·L + D_1·H_1·(c_a1·A + c_b·(L − A)) + c_p·H_2·D_2·P
    fn closed_form_total_cost(annual_rate: f64) -> f64 {
        let one_step_factor = |days: i64| {
            if annual_rate == 0.0 {
                1.0
            } else {
                (1.0 + annual_rate).powf(-(days as f64) / 365.25)
            }
        };
        let d_1 = one_step_factor(DELIVERY_ORACLE_STAGE_DAYS[0]);
        let d_2 = d_1 * one_step_factor(DELIVERY_ORACLE_STAGE_DAYS[1]);
        DELIVERY_ORACLE_BACKUP_COST * DELIVERY_ORACLE_STAGE0_HOURS * DELIVERY_ORACLE_LOAD_MW
            + d_1
                * STAGE1_HOURS
                * (STAGE1_ANTICIPATED_COST * DELIVERY_ORACLE_ANTICIPATED_CAP_MW
                    + DELIVERY_ORACLE_BACKUP_COST
                        * (DELIVERY_ORACLE_LOAD_MW - DELIVERY_ORACLE_ANTICIPATED_CAP_MW))
            + DELIVERY_ORACLE_POST_STUDY_COST
                * DELIVERY_ORACLE_POST_STUDY_HOURS
                * d_2
                * DELIVERY_ORACLE_POST_STUDY_MIN_MW
    }

    fn assert_trained_lb_matches_closed_form(annual_rate: f64) {
        let (system, config) = discounted_delivery_oracle_study(annual_rate);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let expected = closed_form_total_cost(annual_rate);
        let final_lb = outcome.result.final_lb;
        assert!(
            (final_lb - expected).abs() <= REL_TOL * expected,
            "closed-form LB mismatch at annual rate {annual_rate}: built {final_lb}, \
             expected {expected}"
        );
        let final_gap = outcome.result.final_gap;
        assert!(
            final_gap.abs() < 1e-9,
            "final_gap must be ~0 on a deterministic fixture; got {final_gap}"
        );
    }

    #[test]
    fn discounted_delivery_oracle_matches_its_closed_form() {
        assert_trained_lb_matches_closed_form(DELIVERY_ORACLE_ANNUAL_RATE);
    }

    #[test]
    fn discounted_delivery_oracle_zero_rate_twin_matches_its_own_closed_form() {
        assert_trained_lb_matches_closed_form(0.0);

        let undiscounted = closed_form_total_cost(0.0);
        let discounted = closed_form_total_cost(DELIVERY_ORACLE_ANNUAL_RATE);
        assert!(
            (discounted - undiscounted).abs() > 1e3 * REL_TOL * undiscounted,
            "the rate must move the closed form: discounted {discounted}, undiscounted \
             {undiscounted}"
        );
    }
}
mod lead_time_single_decider_end_to_end {
    //! The first true `LeadTime` parse→validate→setup→train load-path exercise:
    //! a single-decider `LeadTime` thermal (`|C(t)| <= 1` everywhere) must solve
    //! end-to-end with no panic, validating both `template.rs`'s `StateSpace`
    //! threading and the in-LP ring for `LeadTime`.
    //!
    //! ## Fixture
    //!
    //! Same topology as `anticipated_closed_form_lb_k1_single_thermal`
    //! (1 anticipated thermal, 1 backup thermal, no hydro, always-active
    //! fishing), extended to 3 uniform 744h stages with
    //! `AnticipatedConfig::LeadTime(744.0)` in place of `LeadStages(1)`. On a
    //! uniform 744h calendar a 744h lead resolves to exactly the same
    //! decider as a constant 1-stage lead (`c(m) = m - 1`), so the closed-form
    //! derivation is the direct 3-stage extension of the 2-stage K=1 case:
    //!
    //! - Stage 0's delivery is the pre-study seed (`past_anticipated_commitments
    //!   = [0.0]`), so `g_a_0 = 0` and the backup covers `D` alone.
    //! - Stage 0 decides `d_ant_0`, delivered at stage 1; stage 1 decides
    //!   `d_ant_1`, delivered at stage 2 (the last stage decides nothing
    //!   further — its own `C(2)` is empty).
    //! - Each decision's sub-problem is independent
    //!   (`min_x c_a·x + c_b·max(0, D-x)` over `x ∈ [0, M]`) and — since
    //!   `c_a < c_b` — is minimised at `x = D` for both.
    //!
    //! `T* = H·[c_b·D + c_a·D + c_b·max(0,D-D) + c_a·D + c_b·max(0,D-D)]
    //!     = H·D·(c_b + 2·c_a)`.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    const N_STAGES: usize = 3;
    const K_MAX: usize = 1;
    const LEAD_TIME_HOURS: f64 = 744.0;
    const BLOCK_HOURS: f64 = 744.0;

    const D_LOAD: f64 = 50.0;
    const M_ANT: f64 = 100.0;
    const B_BACK: f64 = 200.0;
    const C_A: f64 = 10.0;
    const C_B: f64 = 100.0;
    const C_DEFICIT: f64 = 1000.0;

    /// Closed-form lower bound: `T* = H · D · (C_B + 2 · C_A)` — see module docs.
    const EXPECTED_LB: f64 = BLOCK_HOURS * D_LOAD * (C_B + 2.0 * C_A); // = 4_464_000.0

    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BACKUP_ID: EntityId = EntityId(3);
    const BUS_ID: EntityId = EntityId(1);

    // SystemBuilder::build() sorts thermals by EntityId ascending, so the
    // global thermal indices end up: 0 → id=2 (anticipated), 1 → id=3 (backup).
    const THERMAL_IDX_ANT: usize = 0;
    const THERMAL_IDX_BACKUP: usize = 1;

    fn build_system() -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: C_DEFICIT,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
                cost_per_mwh: C_A,
                anticipated_config: Some(AnticipatedConfig::LeadTime(LEAD_TIME_HOURS)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
                cost_per_mwh: C_B,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: BLOCK_HOURS,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: D_LOAD,
                std_mw: 0.0,
            })
            .collect();

        // Overrides span the full thermal axis (n_stages + k_max) so
        // `fill_anticipated_columns` reads a well-defined cost at every delivery
        // stage.
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) =
                ThermalStageBounds { cost_per_mwh: C_A };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) =
                ThermalStageBounds { cost_per_mwh: C_B };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Past commitment is zero (the strict-zero precondition mirroring the
        // K=1 ring-buffer fixture). Any non-zero delivery observed at stage 1
        // must come from the stage-0 decision.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                start_date: stage_date(0),
                end_date: stage_date(1),
                value_mw: 0.0,
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                // 2 iterations reach the closed-form LB; 2 more settle the
                // upper-bound gap to ~0 (mirrors the K=1 2-stage fixture's margin).
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// The first true `LeadTime` parse→validate→setup→train load-path exercise:
    /// a single-decider `LeadTime(744.0)` thermal on a uniform 3×744h calendar
    /// trains to convergence with no panic, matching the closed-form optimum
    /// `EXPECTED_LB = H · D · (C_B + 2 · C_A) = 4_464_000.0` USD.
    #[test]
    fn lead_time_single_decider_solves_end_to_end() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        assert_eq!(result.iterations, 4, "iterations mismatch");

        let gap = result.final_gap;
        assert!(
            gap.is_finite() && gap.abs() < 1e-6,
            "final_gap should be approximately zero (LB == UB) for a fully \
             deterministic fixture; got {gap}",
        );

        let actual = result.final_lb;
        assert!(actual.is_finite(), "final_lb must be finite; got {actual}");
        let rel_diff = (actual - EXPECTED_LB).abs() / EXPECTED_LB.abs();
        assert!(
            rel_diff < 1e-9,
            "closed-form LB mismatch: actual = {actual}, expected = {EXPECTED_LB} \
             (rel_diff = {rel_diff}). See module docs for the derivation.",
        );
    }
}
mod anticipated_numerical_reconciliation_k2 {
    //! Numerical reconciliation test: LP total cost must match the analytical optimum
    //! for a K=2, 6-stage fixture with zero discount rate.
    //!
    //! A regression that silently zeroes intermediate-stage anticipated dispatch
    //! forces backup to carry that load, inflating the LP total by hundreds of times.
    //!
    //! ## Analytical optimum (discount rate = 0, all discount factors = 1.0)
    //!
    //! Parameters:
    //! - n_stages = 6, K = 2, load = 150 MW, block duration = 744 h/stage
    //! - Anticipated thermal: max_gen = 200 MW, cost = $10/MWh
    //! - Backup thermal: max_gen = 500 MW, cost = $5000/MWh
    //! - Excess generation cost = $0 (over-commitment is free)
    //!
    //! Stages partition into three zones:
    //!
    //! **Zone A — Active decision stages (t + K < n_stages → t ∈ {0, 1, 2, 3}):**
    //! The LP decides `d_t ∈ [0, 200]`. Because load = 150 MW < max_gen = 200 MW
    //! and excess is free, the LP commits `d_t = load = 150 MW` — not max_gen.
    //! Over-committing to 200 MW costs 50 MW × $10/MWh × 744 h = $372k/stage extra
    //! with no benefit. The per-block cost of the anticipated-decision column is
    //! $10/MWh, charged at the decision stage.
    //!
    //! Anticipated decision cost = 4 stages × 150 MW × 744 h × $10/MWh
    //!                           = **$4,464,000**
    //!
    //! **Zone B — Delivery stages with matured anticipated commitment (t ∈ {2, 3, 4, 5}):**
    //! The anticipated thermal delivers `committed_t = d_{t-K} = 150 MW = load`.
    //! Per-block cost on the anticipated thermal at delivery stages is skipped in
    //! `fill_thermal_columns` (never written; the anticipated thermal is detected
    //! via `AnticipatedPlants::local_of`), so delivered generation costs $0
    //! in the objective. No backup needed since 150 MW = load exactly.
    //!
    //! **Zone C — Pre-horizon stages (t ∈ {0, 1}):**
    //! The always-active fishing predicate pins the anticipated thermal to seed
    //! slot 0 = 0 MW (`past_anticipated_commitments = [0.0, 0.0]`). The LP must
    //! dispatch backup at $5000/MWh to meet the 150 MW load. The cost-zeroing
    //! predicate is also always-active, so the anticipated thermal column has
    //! objective 0 — but its column upper bound is fishing-pinned to 0, leaving
    //! backup as the sole feasible source.
    //!
    //! Pre-horizon backup cost = 2 stages × 150 MW × 744 h × $5000/MWh
    //!                         = **$1,116,000,000**
    //!
    //! **Total analytical optimum = $4,464,000 + $1,116,000,000 = $1,120,464,000**
    //!
    //! The 5/6/7 entity IDs are distinct from the K=2 and K=3 saturation tests so
    //! combined nextest runs give unambiguous per-entity failure attribution.

    use cobre_core::HorizonGraph;
    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };
    use super::common::run_simulation;

    // ---------------------------------------------------------------------------
    // Analytical optimum constants (documented in module-level doc comment above)
    // ---------------------------------------------------------------------------

    /// Active decision cost: 4 stages × 150 MW (load, not max_gen — over-committing
    /// is wasted when excess is free) × 744 h × $10/MWh = $4,464,000.
    const EXPECTED_DECISION_COST_USD: f64 = 4.0 * 150.0 * 744.0 * 10.0;

    /// Pre-horizon backup cost: 2 stages × 150 MW × 744 h × $5000/MWh = $1,116,000,000.
    /// At t∈{0,1} the always-active fishing equality pins the anticipated thermal to
    /// the zero seed (slot 0 = 0 MW), leaving backup as the sole feasible source for
    /// the 150 MW load. See module doc Zone C.
    const EXPECTED_PRE_HORIZON_BACKUP_COST_USD: f64 = 2.0 * 150.0 * 744.0 * 5000.0;

    /// Total = active decision cost (stages 2..=5) + pre-horizon backup cost
    /// (stages 0, 1) = $4,464,000 + $1,116,000,000 = $1,120,464,000.
    const EXPECTED_TOTAL_USD: f64 =
        EXPECTED_DECISION_COST_USD + EXPECTED_PRE_HORIZON_BACKUP_COST_USD;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 6-stage K=2 reconciliation fixture.
    ///
    /// The trivial hydro keeps the model in the thermal regime; it exists only so
    /// `n_hydros = 1` is satisfied without adding a hydro state variable that
    /// complicates interpretation. `annual_discount_rate = 0.0` collapses all
    /// discount factors to 1.0, making the analytical cost summation exact.
    fn build_system_reconciliation_k2() -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let k: usize = 2;
        let n_stages: usize = 6;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(5);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant_reconcil".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(k as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(6),
            ThermalSpec {
                name: "T_backup_reconcil".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
                cost_per_mwh: 5000.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(7),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(7),
                stage_id: i as i32,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 1.0,
                max_generation_mw: 1.0,
                ..Default::default()
            }
        }

        // The padding region [n_stages, n_stages + k) is the delivery-stage axis
        // read by `fill_anticipated_columns`; it must carry the
        // per-thermal cost so the decision column's objective coefficient is
        // non-zero.
        let thermal_axis = n_stages + k;
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: k,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 200.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        // SystemBuilder sorts by EntityId: index 0 = anticipated (id=5), index 1 =
        // backup (id=6).
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 200.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: super::default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Zero seeds: slot 0 carries 0 MW at both pre-horizon stages (Zone C).
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(7),
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![
                AnticipatedCommitmentHistory {
                    thermal_id: anticipated_id,
                    start_date: stage_date(0),
                    end_date: stage_date(1),
                    value_mw: 0.0,
                },
                AnticipatedCommitmentHistory {
                    thermal_id: anticipated_id,
                    start_date: stage_date(1),
                    end_date: stage_date(2),
                    value_mw: 0.0,
                },
            ],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        // Set explicitly (not relying on `HorizonGraph::default()`) so a future
        // default change cannot silently introduce NPV scaling into the analytical
        // derivation.
        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .policy_graph(policy_graph)
            .build()
            .expect("build_system_reconciliation_k2: valid system")
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Assert that the LP total cost equals the hand-derived analytical optimum
    /// ($1,120,464,000; derivation in module doc) for a K=2, 6-stage fixture with
    /// zero discount rate.
    #[test]
    fn lp_total_cost_matches_analytical_optimum_k2_discount_zero() {
        let system = build_system_reconciliation_k2();
        // Ten iterations let the 500x cost asymmetry produce cuts that signal the
        // value of anticipated dispatch, driving the observed cost to the optimum.
        let config = super::build_config(10);
        let mut setup = build_setup_in_code(system, &config);
        let scenario_results = run_simulation(&mut setup, 10);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            6,
            "scenario must contain one stage record per study stage (n_stages=6)",
        );

        // Sum `immediate_cost` (LP objective minus theta), excluding the future-cost
        // approximation — the realized cost the analytical optimum is derived for.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|stage| stage.costs.iter())
            .map(|cost| cost.immediate_cost)
            .sum();

        // $1000 sits comfortably above HiGHS's 1e-9 precision yet far below the
        // ~$370M pre-fix error, giving 5+ orders of magnitude of detection headroom.
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        assert!(
            (observed_total - EXPECTED_TOTAL_USD).abs() < COST_TOLERANCE_USD,
            "LP total cost {} differs from analytical optimum {} by {} \
         (tolerance ${:.2}). \
         Pre-fix behaviour: intermediate anticipated dispatch is zeroed \
         (d_1 = d_2 = d_3 ≈ 0), forcing the LP to backfill with backup at \
         $5000/MWh — producing a cost gap of approximately $370M.",
            observed_total,
            EXPECTED_TOTAL_USD,
            (observed_total - EXPECTED_TOTAL_USD).abs(),
            COST_TOLERANCE_USD,
        );

        // The named cost categories must sum to `immediate_cost` at every stage,
        // including the anticipated commitment fuel as `anticipated_thermal_cost`.
        // `hydro_violation_cost` already aggregates its six sub-components and
        // `spillage_cost` already includes diversion — sum the aggregates, not the
        // parts, or the total double-counts.
        const RECONCILE_TOLERANCE_USD: f64 = 1.0;
        let mut saw_nonzero_anticipated = false;
        for stage in &scenario.stages {
            for cost in &stage.costs {
                let category_sum = cost.thermal_cost
                    + cost.anticipated_thermal_cost
                    + cost.contract_cost
                    + cost.deficit_cost
                    + cost.excess_cost
                    + cost.storage_violation_cost
                    + cost.filling_target_cost
                    + cost.hydro_violation_cost
                    + cost.inflow_penalty_cost
                    + cost.generic_violation_cost
                    + cost.spillage_cost
                    + cost.turbined_cost
                    + cost.curtailment_cost
                    + cost.exchange_cost
                    + cost.pumping_cost;
                assert!(
                    (category_sum - cost.immediate_cost).abs() < RECONCILE_TOLERANCE_USD,
                    "stage {}: Σ(named cost categories) = {} must equal immediate_cost = {} \
                 (diff {}); the anticipated commitment fuel must be attributed to \
                 anticipated_thermal_cost, not left as an unattributed remainder",
                    cost.stage_id,
                    category_sum,
                    cost.immediate_cost,
                    (category_sum - cost.immediate_cost).abs(),
                );
                if cost.anticipated_thermal_cost.abs() > RECONCILE_TOLERANCE_USD {
                    saw_nonzero_anticipated = true;
                }
            }
        }
        // Zone A (decision stages t∈{0,1,2,3}) must book a positive
        // anticipated_thermal_cost; otherwise the new field is dead and the
        // reconciliation above would pass trivially.
        assert!(
            saw_nonzero_anticipated,
            "expected a non-zero anticipated_thermal_cost at the decision stages; \
         got zero everywhere (the GNL fuel was not attributed)",
        );
    }
}
mod anticipated_bridge_st_cruz_nova_k1 {
    //! Integration test: ST.CRUZ NOVA bridge-parity fixture with K=1 pre-horizon
    //! seed delivery.
    //!
    //! ## Analytical cost bound
    //!
    //! With `K = 1`, `n_stages = 5`, a single anticipated thermal (id=61,
    //! `ST_CRUZ_NOVA`), and a `past_anticipated_commitments` window tiling stage 0
    //! with `value_mw = 204.5647`:
    //!
    //! - Stage 0: anticipated delivers seed (204.5647 MW, zero cost) + backup
    //!   covers remaining (250.0 - 204.5647) MW at $5000/MWh × 744 h ≈ $169,019,316.
    //! - Stages 1–4 delivery: anticipated covers ≥ load (zeroed cost).
    //! - Decision cost ≤ 4 × 350 MW × $10/MWh × 744 h = $10,416,000.
    //! - Total ≤ $169,019,316 + $10,416,000 + $1,000 (tolerance).
    //!
    //! The 204.5647 MW seed is the block-fraction-weighted aggregate of ST.CRUZ NOVA
    //! per-block MW values (227.86, 238.37, 173.51) against September 2024 block
    //! fractions (0.2333, 0.2833, 0.4834); the 1e-3 MW tolerance reflects its
    //! four-decimal accuracy.
    //!
    //! The fishing constraint is always active for every anticipated plant, so a
    //! fishing row is emitted at every stage. The anticipated plant's delivery-stage
    //! per-block thermal cost is skipped in `fill_thermal_columns` (the plant is
    //! detected via `AnticipatedPlants::local_of`), so those columns are consumed
    //! at zero cost.
    //!
    //! The 60-series entity IDs are distinct from the other anticipated tests so
    //! combined nextest runs give unambiguous per-entity failure attribution.

    use cobre_core::HorizonGraph;
    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };
    use super::common::run_simulation;

    // ---------------------------------------------------------------------------
    // Analytical cost bound constants (documented in module doc comment above)
    // ---------------------------------------------------------------------------

    const STAGE_0_BACKUP_COST_USD: f64 = (250.0 - 204.5647) * 744.0 * 5000.0;
    const MAX_DECISION_COST_USD: f64 = 4.0 * 350.0 * 744.0 * 10.0;
    const COST_TOLERANCE_USD: f64 = 1_000.0;
    const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 =
        STAGE_0_BACKUP_COST_USD + MAX_DECISION_COST_USD + COST_TOLERANCE_USD;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 5-stage K=1 ST.CRUZ NOVA fixture.
    ///
    /// Builds the resolved `cobre_core::System` directly via `SystemBuilder::new()`,
    /// bypassing the `cobre-io` parse-and-validate pipeline, to keep the test
    /// self-contained; the 204.5647 MW seed is within bounds and would also pass
    /// `load_case`.
    ///
    /// The anticipated ($10/MWh) vs backup ($5000/MWh) 500× ratio makes the LP
    /// prefer anticipated dispatch. `annual_discount_rate = 0.0` collapses all
    /// discount factors to 1.0, making the analytical cost derivation exact.
    fn build_system() -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 9, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let k: usize = 1;
        let n_stages: usize = 5;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(61);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "ST_CRUZ_NOVA".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 350.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(62),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
                cost_per_mwh: 5000.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        // Trivial hydro keeps the model in the thermal regime; present only so
        // `n_hydros = 1` exercises the hydro state path without adding uncertainty.
        let hydro = make_hydro(
            EntityId(60),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(60),
                stage_id: i as i32,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 250.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 1.0,
                max_generation_mw: 1.0,
                ..Default::default()
            }
        }

        // The padding region [n_stages, n_stages + k) is the delivery-stage axis
        // read by `fill_anticipated_columns`; it must carry per-thermal
        // costs so the decision column's objective coefficient is non-zero.
        let thermal_axis = n_stages + k;
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: k,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 350.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        // SystemBuilder sorts by EntityId: index 0 = anticipated (id=61), index 1 =
        // backup (id=62). Without these per-thermal overrides the LP has no cost
        // incentive to commit anticipated capacity, so decision_at(t) collapses to
        // zero and masks the regression assertion.
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 350.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: super::default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Slot 0 = 204.5647 MW seed (block-fraction-weighted aggregate; see module
        // doc). The always-active fishing equality reads it at stage 0 and delivers
        // it at zero LP cost, leaving backup to cover the remainder.
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(60),
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                start_date: stage_date(0),
                end_date: stage_date(1),
                value_mw: 204.5647,
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .policy_graph(policy_graph)
            .build()
            .expect("build_system: valid system")
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Verify that the ST.CRUZ NOVA pre-horizon seed (204.5647 MW) is delivered at
    /// stage 0 via the always-active fishing predicate, and that the commitment-hold
    /// delivery propagates in-study decisions correctly for stages 1–4.
    #[test]
    fn pre_horizon_seed_delivers_at_stage_zero_st_cruz_nova_k1() {
        let system = build_system();
        // One iteration suffices: the stage-0 fishing equality pins the seed delivery
        // regardless of cut quality, and the cost bound is deliberately generous to
        // absorb the loose 1-iteration cut.
        let config = super::build_config(1);
        let mut setup = build_setup_in_code(system, &config);
        let scenario_results = run_simulation(&mut setup, 1);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            5,
            "scenario must contain one record per study stage (n_stages=5)",
        );

        let anticipated_thermal_id: i32 = 61;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };
        let committed_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_committed_mw)
        };

        let c0 = committed_at(0).expect(
            "AC-delivery FAIL: committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 204.5647 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at stage 0 with K=1.",
        );
        assert!(
            (c0 - 204.5647).abs() < 1e-3,
            "AC-delivery FAIL: committed_at(0) = {c0} MW, expected 204.5647 MW within \
         1e-3 MW tolerance. The fishing equality at stage 0 must pin the anticipated \
         thermal to slot 0 = 204.5647 MW of the ring buffer.",
        );

        for t in 0..4_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "AC-decision-nonzero FAIL: decision_at({t}) is None; anticipated \
                 thermal id=61 was not found in stage {t} thermals or \
                 anticipated_decision_mw is absent (stage {t} is an active-decision \
                 stage: {t} + 1 < 5)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "AC-decision-nonzero FAIL: decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage.",
            );
            assert!(
                dt <= 350.0 + 1e-6,
                "AC-decision-nonzero FAIL: decision_at({t}) = {dt} MW exceeds \
             max_gen=350 MW. This indicates a bounds violation in the LP.",
            );
        }

        // The in-study decision made at stage t-1 (delivered at stage t, K=1) is
        // latched at slot 0 and stage t's fishing equality pins generation to it.
        for t in 1..5_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "AC-ring-buffer FAIL: committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 1,
                )
            });
            let d_prev = decision_at(t - 1).unwrap_or_else(|| {
                panic!(
                    "AC-ring-buffer FAIL: decision_at({}) is None (needed to check \
                 delivery-lag invariant at stage {t})",
                    t - 1,
                )
            });
            assert!(
                (ct - d_prev).abs() < 1e-6,
                "AC-ring-buffer FAIL: committed_at({t}) = {ct} MW should equal \
             decision_at({}) = {d_prev} MW (within 1e-6 MW). The ring buffer is \
             not correctly propagating in-study decisions.",
                t - 1,
            );
        }

        // Sum immediate_cost (LP objective minus theta), not total_cost — total_cost
        // includes the theta approximation artefact.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "AC-cost-bound FAIL: observed_total = ${observed_total:.2} exceeds upper \
         bound ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If the seed is not delivered (legacy predicate), stage-0 backup covers \
         250 MW instead of 45.4353 MW, producing ~$927M >> this bound.",
        );
    }
}
mod anticipated_convergence_slow {
    //! Slow-gated convergence test for a medium anticipated-thermal case (fixture
    //! built by `build_system`).
    //!
    //! Verifies that `SamplingScheme::OutOfSample` converges to a lower bound
    //! within 10% relative tolerance of the `SamplingScheme::InSample` lower
    //! bound when anticipated thermals are present. The 10% tolerance (relaxed from
    //! 5%) accommodates the anticipated ring-buffer variance source that is absent
    //! in the pure-hydro convergence test.

    use chrono::{NaiveDate, TimeDelta};
    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel, SamplingScheme, ScenarioSource};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_sddp::{
        InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet, StudySetup,
        TrainingOutcome,
        hydro_models::PrepareHydroModelsResult,
        setup::{SimulationEnumeratedRequest, StudyParams},
    };
    use cobre_solver::ActiveSolver;
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use super::common::StubComm;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 4-stage K=2 convergence fixture (1 hydro, 1 anticipated thermal,
    /// 1 backup). The LP is always feasible: the backup thermal alone covers the
    /// 220 MW load.
    fn build_system(branching_factor: usize) -> cobre_core::System {
        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(2);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 500.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                ..Default::default()
            },
        );

        let n_stages = 4_usize;
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
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
                            branching_factor,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 220.0,
                std_mw: 0.0,
            })
            .collect();

        let k_max: usize = 2;
        let n_st = n_stages;

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: super::default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![
                AnticipatedCommitmentHistory {
                    thermal_id: anticipated_id,
                    start_date: stage_date(0),
                    end_date: stage_date(1),
                    value_mw: 40.0,
                },
                AnticipatedCommitmentHistory {
                    thermal_id: anticipated_id,
                    start_date: stage_date(1),
                    end_date: stage_date(2),
                    value_mw: 20.0,
                },
            ],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Training runner
    // ---------------------------------------------------------------------------

    /// Run the SDDP training pipeline programmatically.
    ///
    /// Builds a `StudySetup` using the low-level `from_broadcast_params` path
    /// (identical to `forward_sampler_integration.rs::run_programmatic`) so that
    /// both the `InSample` and `OutOfSample` sampling schemes can be exercised
    /// without going through the file-based config/IO path.
    fn run_training(
        system: &cobre_core::System,
        sampling_scheme: SamplingScheme,
        forward_seed: Option<i64>,
        inflow_method: InflowNonNegativityMethod,
    ) -> TrainingOutcome {
        const FORWARD_PASSES: u32 = 10;
        const MAX_ITERATIONS: u64 = 30;

        let tree_seed: u64 = 42;
        // `forward_seed` for `build_stochastic_context` is the unsigned abs of the
        // signed seed from `ScenarioSource`, matching the conversion in
        // `forward_sampler_integration.rs::run_programmatic`.
        let fwd_seed_u64 = forward_seed.map(i64::unsigned_abs);

        let stochastic = build_stochastic_context(
            system,
            tree_seed,
            fwd_seed_u64,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("build_stochastic_context must succeed");

        let hydro_models = PrepareHydroModelsResult::default_from_system(system);

        let stopping_rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit {
                limit: MAX_ITERATIONS,
            }],
            mode: StoppingMode::Any,
        };

        let config = StudyParams {
            seed: tree_seed,
            forward_passes: FORWARD_PASSES,
            training_enumerated: false,
            stopping_rule_set,
            n_scenarios: 0,
            simulation_enumerated: SimulationEnumeratedRequest::Sampled,
            io_channel_capacity: 0,
            policy_path: String::new(),
            inflow_method,
            cut_selection: None,
            cut_activity_tolerance: 0.0,
            budget: None,
            export_states: false,
            checkpoint_schedule: None,
            scalar_parameters: Vec::new(),
            training_solver_backward: None,
            training_solver_forward: None,
            simulation_solver: None,
            backward_scheduler: cobre_io::config::BackwardScheduler::default(),
            cost_scale_factor: cobre_sddp::DEFAULT_COST_SCALE_FACTOR,
            boundary: cobre_sddp::BoundaryStateRequirements::none(),
        };

        let source = ScenarioSource {
            inflow_scheme: sampling_scheme,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: forward_seed,
            historical_years: None,
        };

        let mut setup = StudySetup::from_broadcast_params(
            system,
            stochastic,
            config,
            hydro_models,
            &source,
            &source,
        )
        .expect("StudySetup::from_broadcast_params must succeed");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok")
    }

    // ---------------------------------------------------------------------------
    // Slow-gated convergence test
    // ---------------------------------------------------------------------------

    /// Verify that `SamplingScheme::OutOfSample` converges to a lower bound within
    /// 10% relative tolerance of the `SamplingScheme::InSample` lower bound on a
    /// 4-stage K=2 anticipated-thermal system with `branching_factor=5` SAA noise.
    ///
    /// The 10% tolerance (relaxed from the 5% used in the pure-hydro convergence
    /// test) accommodates the additional variance introduced by the anticipated
    /// ring-buffer's interaction with the out-of-sample noise trajectories.
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    #[test]
    fn anticipated_k2_insample_vs_out_of_sample_convergence() {
        const BRANCHING_FACTOR: usize = 5;
        const RELATIVE_TOLERANCE: f64 = 0.10;

        let system_insample = build_system(BRANCHING_FACTOR);
        let system_oos = build_system(BRANCHING_FACTOR);

        let result_insample = run_training(
            &system_insample,
            SamplingScheme::InSample,
            None,
            InflowNonNegativityMethod::None,
        );

        assert!(result_insample.error.is_none(), "InSample error");
        let lb_insample = result_insample.result.final_lb;
        assert!(lb_insample > 0.0 && lb_insample.is_finite());

        let result_oos = run_training(
            &system_oos,
            SamplingScheme::OutOfSample,
            Some(42),
            InflowNonNegativityMethod::Truncation,
        );

        assert!(result_oos.error.is_none(), "OutOfSample error");
        let lb_oos = result_oos.result.final_lb;
        assert!(lb_oos > 0.0 && lb_oos.is_finite());

        let relative_error = (lb_oos - lb_insample).abs() / lb_insample.abs().max(1e-10);
        assert!(
            relative_error < RELATIVE_TOLERANCE,
            "convergence exceeded tolerance: {:.2}% vs {:.0}%",
            relative_error * 100.0,
            RELATIVE_TOLERANCE * 100.0,
        );
    }
}
mod a1b_value_cut_identity_anchor {
    //! LeadTime == LeadStages value/cut-identity anchor on a uniform calendar.
    //!
    //! On a uniform 3x744h calendar a physical lead of exactly one stage's hours
    //! (`LeadTime(744.0)`) and a one-stage-count lead (`LeadStages(1)`) resolve to
    //! the identical delivery-anchored decider `[None, Some(0), Some(1)]`, so both
    //! build a byte-identical LP and MUST train to bit-identical value, states, and
    //! anticipated-ring cuts. This module pins that equivalence end-to-end
    //! (train + simulate) as the A-1(b) value/cut-identity anchor.

    use cobre_core::HorizonGraph;
    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_sddp::{SimulationScenarioResult, StudySetup};
    use cobre_solver::ActiveSolver;

    use std::sync::mpsc;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    const N_STAGES: usize = 3;
    const K_MAX: usize = 1;
    const STAGE_HOURS: f64 = 744.0;
    const LOAD_MW: f64 = 200.0;

    const BACKUP_COST: f64 = 100.0;
    const BACKUP_CAP: f64 = 200.0;
    const ANT_COST: f64 = 5.0;

    // Stage-varying anticipated delivery caps, indexed by delivery stage over the
    // `N_STAGES + K_MAX` axis `fill_anticipated_columns` reads. Stage 0 is the
    // seed-delivery stage AND the decision-anchored decoy for the delivery-1
    // decision; the delivery-stage-1 (150) and delivery-stage-2 (80) caps differ,
    // which is what makes the delivery-anchoring mutation non-vacuous — constant caps
    // would make a decision-anchored read indistinguishable from a delivery-anchored
    // one.
    const THERMAL_AXIS: usize = N_STAGES + K_MAX;
    const DELIVERY_CAP: [f64; THERMAL_AXIS] = [150.0, 150.0, 80.0, 80.0];

    // EntityId ordering: bus 1, backup T0 = 2, anticipated T1 = 3. System::build
    // sorts thermals by EntityId, so thermal_idx 0 = backup, 1 = anticipated.
    const BUS_ID: EntityId = EntityId(1);
    const BACKUP_ID: EntityId = EntityId(2);
    const ANT_ID: EntityId = EntityId(3);
    const THERMAL_IDX_BACKUP: usize = 0;
    const THERMAL_IDX_ANT: usize = 1;

    const ITERATIONS: usize = 20;

    fn build_system(anticipated_config: AnticipatedConfig) -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};
        let date = |m: u32, d: u32| NaiveDate::from_ymd_opt(2024, m, d).expect("valid date");
        // Each stage is a 744h (31-day) block; whole-day stage boundaries keep
        // StageCalendar's date-based window coverage exact (744 / 24 = 31).
        let stage_date = |index: usize| date(1, 1) + TimeDelta::days(31 * index as i64);

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: date(1, 1),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T0_backup".to_string(),
                operational_start_date: date(1, 2),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_CAP,
                cost_per_mwh: BACKUP_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let anticipated = make_thermal(
            ANT_ID,
            ThermalSpec {
                name: "T1_anticipated".to_string(),
                operational_start_date: date(1, 3),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_CAP,
                cost_per_mwh: ANT_COST,
                anticipated_config: Some(anticipated_config),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: STAGE_HOURS,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: LOAD_MW,
                std_mw: 0.0,
            })
            .collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        // fill_anticipated_columns reads the delivery cell at `stage_idx + K`, so
        // per-thermal bounds must cover the full `[0, N_STAGES + K_MAX)` axis. The
        // backup is constant; the anticipated plant carries the stage-varying caps.
        for s in 0..THERMAL_AXIS {
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) = ThermalStageBounds {
                cost_per_mwh: BACKUP_COST,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_CAP,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
                cost_per_mwh: ANT_COST,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: DELIVERY_CAP[s],
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // K=1 ring: exactly one pre-study delivery stage (m=0, decider None) in both
        // modes. Seed it to 0 MW so stage 0 delivers nothing from T1 and the
        // hand-derived dispatch lives entirely at delivery stages 1 and 2.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANT_ID,
                start_date: stage_date(0),
                end_date: stage_date(1),
                value_mw: 0.0,
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![backup, anticipated])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .policy_graph(policy_graph)
            .build()
            .expect("build_system: valid")
    }

    /// Converged run outputs for the two-mode identity comparison.
    struct RunOutputs {
        final_lb: f64,
        final_ub: f64,
        scenarios: Vec<SimulationScenarioResult>,
    }

    /// Train `iterations` then simulate one scenario, capturing the converged
    /// lower/upper bound before the outcome's basis cache is consumed by simulate.
    fn train_and_simulate(setup: &mut StudySetup, iterations: usize) -> RunOutputs {
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(
                &mut solver,
                &comm,
                iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );
        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("create_workspace_pool");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (tx, rx) = mpsc::sync_channel(io_capacity);
        let drain = std::thread::spawn(move || rx.into_iter().collect::<Vec<_>>());
        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");
        drop(tx);
        let scenarios = drain.join().expect("drain thread must not panic");

        RunOutputs {
            final_lb: outcome.result.final_lb,
            final_ub: outcome.result.final_ub,
            scenarios,
        }
    }

    /// Per-stage active cuts as `(slot, intercept, coefficients)`. With no hydro the
    /// full coefficient vector IS the anticipated ring (`n_state ==
    /// commit_out.end`), so equality here is anticipated-ring cut identity.
    fn collect_cuts(setup: &StudySetup) -> Vec<Vec<(usize, f64, Vec<f64>)>> {
        (0..setup.fcf.pools.len())
            .map(|stage| {
                setup
                    .fcf
                    .active_cuts(stage)
                    .map(|(slot, intercept, coeffs)| (slot, intercept, coeffs.to_vec()))
                    .collect()
            })
            .collect()
    }

    fn committed_mw(scenario: &SimulationScenarioResult, t: usize) -> f64 {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == ANT_ID.0)
            .and_then(|th| th.anticipated_committed_mw)
            .unwrap_or_else(|| panic!("committed_mw missing at stage {t}"))
    }

    fn decision_mw(scenario: &SimulationScenarioResult, t: usize) -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == ANT_ID.0)
            .and_then(|th| th.anticipated_decision_mw)
    }

    fn backup_mw(scenario: &SimulationScenarioResult, t: usize) -> f64 {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == BACKUP_ID.0)
            .map_or_else(
                || panic!("backup generation missing at stage {t}"),
                |th| th.generation_mw,
            )
    }

    /// `LeadTime(744h) == LeadStages(1)` on a uniform 3x744h calendar: the two modes
    /// resolve to the identical decider `[None, Some(0), Some(1)]`, build a
    /// byte-identical LP, and MUST train to bit-identical value, per-stage
    /// anticipated committed/decision MW, and anticipated-ring cut coefficients.
    ///
    /// This asserts SOLUTIONS (final LB/UB, committed/decision MW, cut coefficients),
    /// never LP bytes or a basis-dependent dual: the two modes' LPs are identical, so
    /// the optimum is bit-identical and `==` is the correct comparison. Runs under
    /// whichever backend the binary is compiled with, so the identity is pinned on
    /// both HiGHS and CLP via the per-backend CI matrix.
    ///
    /// Hand-derived dispatch (T1 cheap at $5/MWh runs to its delivery-stage cap; T0
    /// backup at $100/MWh fills the 200 MW load): delivery stage 1 -> T1 commits 150
    /// (its stage-1 cap), T0 supplies 50; delivery stage 2 -> T1 commits 80 (its
    /// stage-2 cap), T0 supplies 120.
    ///
    /// Delivery-anchoring is MUTATION-verified. Changing the production read in
    /// `fill_anticipated_columns` from `thermal_block_base(thermal_idx, delivery_stage)`
    /// to `thermal_block_base(thermal_idx, stage_idx)` (the decision stage) relaxes the
    /// stage-1 decision column to stage-1's 150 MW cap for delivery at stage 2; stage
    /// 2's own generation cap is 80 MW and fishing pins gen == committed, so the
    /// delivered 150 MW is undeliverable — the forward solve turns infeasible and the
    /// run errors (the capacity-drop infeasibility the delivery-anchoring contract
    /// forbids), failing this test at the `train`/`simulate` expect. The stage-1 vs
    /// stage-2 cap difference (150 vs 80) is what makes the mutation observable;
    /// constant caps would make it vacuous.
    #[test]
    fn a1b_lead_time_equals_lead_stages_uniform_calendar() {
        let mut setup_stages = build_setup_in_code(
            build_system(AnticipatedConfig::LeadStages(1)),
            &super::build_config(ITERATIONS),
        );
        let mut setup_time = build_setup_in_code(
            build_system(AnticipatedConfig::LeadTime(STAGE_HOURS)),
            &super::build_config(ITERATIONS),
        );

        let (k_max_stages, n_ant_stages) = {
            let s = setup_stages.stage_state();
            (s.k_max, s.n_anticipated)
        };
        let (k_max_time, n_ant_time) = {
            let s = setup_time.stage_state();
            (s.k_max, s.n_anticipated)
        };
        assert_eq!(n_ant_stages, 1, "one anticipated plant (LeadStages)");
        assert_eq!(n_ant_time, 1, "one anticipated plant (LeadTime)");
        assert_eq!(
            k_max_stages, k_max_time,
            "both modes must derive the same ring depth on the uniform calendar",
        );
        assert_eq!(k_max_stages, K_MAX, "ring depth must be 1");

        let run_stages = train_and_simulate(&mut setup_stages, ITERATIONS);
        let run_time = train_and_simulate(&mut setup_time, ITERATIONS);

        // ── Two-mode bit-identity ────────────────────────────────────────────────
        assert_eq!(
            run_stages.final_lb, run_time.final_lb,
            "final lower bound must be bit-identical across LeadStages/LeadTime",
        );
        assert_eq!(
            run_stages.final_ub, run_time.final_ub,
            "objective (final upper bound) must be bit-identical across modes",
        );

        let scen_stages = &run_stages.scenarios[0];
        let scen_time = &run_time.scenarios[0];
        assert_eq!(scen_stages.stages.len(), N_STAGES);
        assert_eq!(scen_time.stages.len(), N_STAGES);
        for t in 0..N_STAGES {
            assert_eq!(
                committed_mw(scen_stages, t),
                committed_mw(scen_time, t),
                "committed_mw at stage {t} must be bit-identical across modes",
            );
            assert_eq!(
                decision_mw(scen_stages, t),
                decision_mw(scen_time, t),
                "decision_mw at stage {t} must be bit-identical across modes",
            );
        }

        assert_eq!(
            collect_cuts(&setup_stages),
            collect_cuts(&setup_time),
            "anticipated-ring cut coefficients must be bit-identical across modes",
        );

        // ── Hand-derived dispatch (asserted on the LeadStages run; the identity
        //    above carries every equality to the LeadTime run) ────────────────────
        const TOL: f64 = 1e-6;
        assert!(
            (committed_mw(scen_stages, 1) - 150.0).abs() < TOL,
            "delivery stage 1: T1 must commit its 150 MW stage-1 cap; got {}",
            committed_mw(scen_stages, 1),
        );
        assert!(
            (backup_mw(scen_stages, 1) - 50.0).abs() < TOL,
            "delivery stage 1: T0 must supply 50 MW; got {}",
            backup_mw(scen_stages, 1),
        );
        assert!(
            (committed_mw(scen_stages, 2) - 80.0).abs() < TOL,
            "delivery stage 2: T1 must commit its 80 MW stage-2 cap (delivery-anchored, \
             not the 150 MW decision-stage cap); got {}",
            committed_mw(scen_stages, 2),
        );
        assert!(
            (backup_mw(scen_stages, 2) - 120.0).abs() < TOL,
            "delivery stage 2: T0 must supply 120 MW; got {}",
            backup_mw(scen_stages, 2),
        );
    }
}
mod a1c_stage_count_mode_anchor {
    //! Stage-count-mode (`LeadStages`) backwards-compatibility anchor on the d37
    //! unequal-hours monthly calendar `[730, 730, 730, 720, 744, 720]`.
    //!
    //! `LeadStages(ℓ)` is a pure index shift `c(m) = m − ℓ` whose result never
    //! reads the hour clock — the guarantee that shipped stage-count configs keep
    //! working unchanged on any calendar. The three resolver-level tests pin that
    //! the clock is never consulted in stage-count mode and contrast it with the
    //! physical `LeadTime` mode, which does consult it. The end-to-end test proves
    //! a single-decider `LeadTime` on this same unequal-hours calendar is a live
    //! solve path.
    //!
    //! Every expected decider below is hand-derived from the calendar; the
    //! cumulative stage-ends are `[730, 1460, 2190, 2910, 3654, 4374]`.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    /// The d37 unequal-hours monthly calendar (stage totals, hours).
    const D37_DURATIONS: [f64; 6] = [730.0, 730.0, 730.0, 720.0, 744.0, 720.0];
    const N_STAGES: usize = 6;

    /// `LeadStages(2)` on the d37 calendar is the pure index shift
    /// `c(m) = m − 2`: `decider = [None, None, Some(0), Some(1), Some(2), Some(3)]`,
    /// each in-horizon `C(t)` is the singleton `{t + 2}` (t = 4, 5 deliver past the
    /// horizon and are empty), and every `depth` entry is `≤ 2`.
    #[test]
    fn a1c_lead_stages_is_pure_index_shift() {
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &D37_DURATIONS,
                post_study_stage_hours: &[],
            },
        );
        let point = &resolution.per_plant[0];

        assert_eq!(
            point.decider,
            vec![None, None, Some(0), Some(1), Some(2), Some(3)],
            "LeadStages(2) must be the pure index shift c(m)=m-2",
        );

        for t in 0..N_STAGES {
            if t + 2 < N_STAGES {
                assert_eq!(
                    point.decision_sets[t],
                    vec![t + 2],
                    "in-horizon C({t}) must be the singleton {{{}}}",
                    t + 2,
                );
            } else {
                assert!(
                    point.decision_sets[t].is_empty(),
                    "C({t}) must be empty (delivery t+2 is past the horizon); got {:?}",
                    point.decision_sets[t],
                );
            }
        }

        assert!(
            point.depth.iter().all(|&d| d <= 2),
            "every depth entry must be bounded by the lead 2; got {:?}",
            point.depth,
        );
        assert_eq!(
            resolution.max_fanout(),
            1,
            "a constant lead is single-decider (|C(t)| <= 1)",
        );
    }

    /// The same `LeadStages(2)` lead resolves identically against a different
    /// 6-stage duration vector (`[672.0; 6]`) — the hour clock is never consulted,
    /// so `decider`, `decision_sets`, and `depth` are byte-for-byte identical.
    #[test]
    fn a1c_lead_stages_ignores_calendar() {
        let on_d37 = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &D37_DURATIONS,
                post_study_stage_hours: &[],
            },
        );
        let on_uniform = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &[672.0; 6],
                post_study_stage_hours: &[],
            },
        );

        let a = &on_d37.per_plant[0];
        let b = &on_uniform.per_plant[0];

        assert_eq!(
            a.decider, b.decider,
            "stage-count decider must not depend on the hour clock",
        );
        assert_eq!(
            a.decision_sets, b.decision_sets,
            "stage-count decision_sets must not depend on the hour clock",
        );
        assert_eq!(
            a.depth, b.depth,
            "stage-count depth must not depend on the hour clock",
        );
    }

    /// The physical `LeadTime(1450.0)` mode DOES consult the clock. On the d37
    /// calendar `end_1 − 1450 = 1460 − 1450 = 10`, which lands in stage 0, so the
    /// end-anchored decider decides `m = 1` at stage 0 — where the pure stage-count
    /// shift (`c(1) = 1 − 2`) is still `None`. Full physical decider:
    /// `[None, Some(0), Some(1), Some(1), Some(3), Some(4)]`.
    #[test]
    fn a1c_lead_time_consults_calendar() {
        let physical = AnticipatedResolution::resolve(
            &[LeadTime::Time(1450.0)],
            DeliveryAxis {
                study_stage_hours: &D37_DURATIONS,
                post_study_stage_hours: &[],
            },
        );
        let stage_count = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &D37_DURATIONS,
                post_study_stage_hours: &[],
            },
        );

        let phys = &physical.per_plant[0];
        let sc = &stage_count.per_plant[0];

        assert_eq!(
            phys.decider,
            vec![None, Some(0), Some(1), Some(1), Some(3), Some(4)],
            "physical end-anchored decider on the d37 calendar",
        );
        assert_eq!(
            phys.decider[1],
            Some(0),
            "LeadTime consults the clock: m=1 decides at stage 0",
        );
        assert_eq!(
            sc.decider[1], None,
            "LeadStages(2) ignores the clock: c(1)=1-2 is None",
        );
        assert_ne!(
            phys.decider[1], sc.decider[1],
            "the clock-consulted contrast at m=1 (Some(0) vs None)",
        );
    }

    // ── single-decider LeadTime end-to-end on the d37 unequal-hours calendar ──

    const LEAD_TIME_HOURS: f64 = 1440.0;
    const K_MAX: usize = 1;
    const D_LOAD: f64 = 50.0;
    const M_ANT: f64 = 100.0;
    const B_BACK: f64 = 200.0;
    const C_A: f64 = 10.0;
    const C_B: f64 = 100.0;
    const C_DEFICIT: f64 = 1000.0;
    const ITERATIONS: u32 = 12;

    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BACKUP_ID: EntityId = EntityId(3);
    const BUS_ID: EntityId = EntityId(1);

    // SystemBuilder::build() sorts thermals by EntityId ascending: 0 -> id=2
    // (anticipated), 1 -> id=3 (backup).
    const THERMAL_IDX_ANT: usize = 0;
    const THERMAL_IDX_BACKUP: usize = 1;

    /// A single-decider `LeadTime(1440.0)` variant of the d37 topology (1
    /// anticipated thermal, 1 backup thermal, no hydro) on the d37 unequal-hours
    /// calendar. `Time(1440.0)` resolves to `[None, Some(0), Some(1), Some(2),
    /// Some(3), Some(4)]` (`max_fanout = 1`), so it clears the fan-out setup guard;
    /// the single pre-study decider (`decider[0] == None`) fixes the
    /// `past_anticipated_commitments` history length at 1.
    fn build_system() -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        // Distinct 31-day (744h) stage spans keep StageCalendar's date ordering
        // invariant valid. `StageCalendar::coverage` divides overlap by each
        // stage's REAL `[start_date, end_date)` calendar span — 744h here, not
        // the declared `D37_DURATIONS[0]` (730h) — so stage 0's window must
        // span the full 744h to hit exact fraction 1.0 (K_MAX=1, so no later
        // stage boundary is ever addressed by a window).
        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * index as i64)
        };

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: C_DEFICIT,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
                cost_per_mwh: C_A,
                anticipated_config: Some(AnticipatedConfig::LeadTime(LEAD_TIME_HOURS)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
                cost_per_mwh: C_B,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        // Each stage carries one block whose hours are the d37 stage total, so
        // DeliveryCalendar::study_total_hours feeds the [730,730,730,720,744,720] calendar to the
        // point-commitment resolver.
        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: D37_DURATIONS[i],
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: D_LOAD,
                std_mw: 0.0,
            })
            .collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        // fill_anticipated_columns reads the delivery cell at each decision's own
        // delivery stage; span the full n_stages + k_max axis so every read is
        // well-defined.
        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) =
                ThermalStageBounds { cost_per_mwh: C_A };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) =
                ThermalStageBounds { cost_per_mwh: C_B };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // One pre-study delivery stage (decider[0] == None) ⇒ history length 1.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                start_date: stage_date(0),
                end_date: stage_date(1),
                value_mw: 0.0,
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: ITERATIONS,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// A single-decider `LeadTime(1440.0)` thermal on the d37 unequal-hours
    /// calendar trains to convergence (`LB == UB`) with no panic — the physical
    /// mode is a live end-to-end path, not merely a resolver unit.
    #[test]
    fn a1c_lead_time_solves_on_unequal_hours() {
        // Guard the fixture's single-decider premise before building the study: a
        // fan-out (max_fanout > 1) would trip the setup guard, not converge.
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Time(LEAD_TIME_HOURS)],
            DeliveryAxis {
                study_stage_hours: &D37_DURATIONS,
                post_study_stage_hours: &[],
            },
        );
        assert_eq!(
            resolution.per_plant[0].decider,
            vec![None, Some(0), Some(1), Some(2), Some(3), Some(4)],
            "Time(1440.0) must resolve to the single-decider chain on the d37 calendar",
        );
        assert_eq!(
            resolution.max_fanout(),
            1,
            "Time(1440.0) must be single-decider (|C(t)| == 1) to clear the fan-out guard",
        );

        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error,
        );

        let result = &outcome.result;
        assert_eq!(
            result.iterations,
            u64::from(ITERATIONS),
            "iteration count mismatch",
        );

        let gap = result.final_gap;
        assert!(
            gap.is_finite() && gap.abs() < 1e-6,
            "final_gap must be ~0 (LB == UB) for a fully deterministic fixture; got {gap}",
        );

        let lb = result.final_lb;
        assert!(
            lb.is_finite() && lb > 0.0,
            "final_lb must be finite and positive; got {lb}",
        );
    }
}

mod anticipated_ring_axis_regressions {
    //! Two executed-solve regressions pinning the numeric contracts of the
    //! gap-excised anticipated ring: ring-axis injectivity and ring-depth
    //! sizing. Both defects are silent wrong committed values that compile and
    //! solve, so each test trains and simulates and asserts on the fished
    //! `anticipated_committed_mw` (and, for the sizing test, delivered energy)
    //! — never on geometry alone.
    //!
    //! - **Ring-axis injectivity.** On the raw delivery axis an occupancy-sized
    //!   `k_max = 4` collides study-stage-3's seed (delivery 3) with the first
    //!   post-study deposit (delivery 7): `3 == 7 (mod 4)`, two definition rows
    //!   on one outgoing column. Keyed on the ring axis — the delivery axis with
    //!   the fixed post-horizon window (width `g`) excised — the two land on
    //!   distinct slots, so every study stage fishes its own seed.
    //! - **Ring-depth sizing.** `k_max = max(occupancy_max, n_none_in_study)`.
    //!   Sizing from `occupancy` alone leaves the last pre-study seed aliased
    //!   onto an earlier slot when few post-study deliveries recycle the ring.

    use chrono::NaiveDate;
    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PostStudyStage, PostStudyStages,
        PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties,
        SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_sddp::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime, PointResolution};

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };
    use super::common::run_simulation;

    const BUS_ID: EntityId = EntityId(1);
    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BACKUP_ID: EntityId = EntityId(3);
    const ANTICIPATED_THERMAL_ID: i32 = 2;

    // SystemBuilder sorts thermals ascending by EntityId, so id=2 (anticipated)
    // is thermal index 0 and id=3 (backup) is index 1.
    const THERMAL_IDX_ANTICIPATED: usize = 0;
    const THERMAL_IDX_BACKUP: usize = 1;

    /// Study calendar shared by both fixtures: three operative weeks then one
    /// month, horizon 1152 h — the reference-deck shape and the bug doc's own.
    const N_STUDY_STAGES: usize = 4;
    const STUDY_DURS: [f64; N_STUDY_STAGES] = [168.0, 168.0, 168.0, 648.0];

    const LOAD_MW: f64 = 500.0;
    const ANTICIPATED_MAX_MW: f64 = 500.0;
    const ANTICIPATED_COST: f64 = 10.0;
    const BACKUP_MAX_MW: f64 = 1000.0;
    const BACKUP_COST: f64 = 5000.0;
    const DEFICIT_COST: f64 = 100_000.0;
    const ITERATIONS: usize = 3;

    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("hardcoded fixture date is valid")
    }

    /// Study-stage boundaries (five dates for four stages), each span matching
    /// its block hours exactly so `StageCalendar` date coverage is exact:
    /// 03-14→03-21→03-28→04-04 are 168 h weeks, 04-04→05-01 is a 648 h month.
    fn study_boundaries() -> [NaiveDate; N_STUDY_STAGES + 1] {
        [
            date(2026, 3, 14),
            date(2026, 3, 21),
            date(2026, 3, 28),
            date(2026, 4, 4),
            date(2026, 5, 1),
        ]
    }

    fn study_stages() -> Vec<Stage> {
        let bounds = study_boundaries();
        (0..N_STUDY_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: bounds[i],
                        end_date: bounds[i + 1],
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: STUDY_DURS[i],
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
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    fn bus() -> cobre_core::entities::bus::Bus {
        make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: date(2026, 3, 14),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: DEFICIT_COST,
                }],
                excess_cost: 0.0,
            },
        )
    }

    fn anticipated_thermal(lead: AnticipatedConfig) -> cobre_core::entities::thermal::Thermal {
        make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: date(2026, 3, 14),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: ANTICIPATED_MAX_MW,
                cost_per_mwh: ANTICIPATED_COST,
                anticipated_config: Some(lead),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        )
    }

    fn backup_thermal() -> cobre_core::entities::thermal::Thermal {
        make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: date(2026, 3, 14),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_MAX_MW,
                cost_per_mwh: BACKUP_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        )
    }

    fn load_models() -> Vec<LoadModel> {
        (0..N_STUDY_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: LOAD_MW,
                std_mw: 0.0,
            })
            .collect()
    }

    /// Per-thermal generation and decision bounds over the full delivery-stage
    /// axis (`n_stages + k_max`): the anticipated plant's generation column must
    /// admit the largest fished seed, and the backup must cover the residual
    /// load at every study stage.
    fn bounds(k_max: usize) -> ResolvedBounds {
        let mut b = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STUDY_STAGES,
                k_max,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        for s in 0..(N_STUDY_STAGES + k_max) {
            *b.thermal_bounds_mut(THERMAL_IDX_ANTICIPATED, s) = ThermalStageBounds {
                cost_per_mwh: ANTICIPATED_COST,
            };
            *b.thermal_block_base_mut(THERMAL_IDX_ANTICIPATED, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: ANTICIPATED_MAX_MW,
            };
            *b.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) = ThermalStageBounds {
                cost_per_mwh: BACKUP_COST,
            };
            *b.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_MAX_MW,
            };
        }
        b
    }

    fn penalties() -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STUDY_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// One `past_anticipated_commitments` window per study stage, tiling the
    /// study calendar with the given (distinct) seed values — the pre-study
    /// deliveries `0..4` the ring must fish back at their own stages.
    fn seed_windows(values: [f64; N_STUDY_STAGES]) -> Vec<AnticipatedCommitmentHistory> {
        let dates = study_boundaries();
        values
            .iter()
            .enumerate()
            .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                start_date: dates[i],
                end_date: dates[i + 1],
                value_mw,
            })
            .collect()
    }

    /// Read the anticipated plant's fished commitment (MW) at study stage `t`
    /// from the one-scenario simulation result.
    fn committed_at(scenario: &cobre_sddp::SimulationScenarioResult, t: usize) -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == ANTICIPATED_THERMAL_ID)
            .and_then(|th| th.anticipated_committed_mw)
    }

    /// The reference-deck collision fixture. Four in-study seeds (deliveries
    /// `0..4`) plus a seven-stage post-study calendar whose first three stages
    /// are the fixed window (`24, 168, 168` h → `g == 3`) and whose remaining
    /// four carry the `PostStudyThermalBound` cells study stages `0..4` decide
    /// deliveries `7..11` into. The `LeadTime` lead is
    /// [`COLLISION_LEAD_HOURS`].
    const COLLISION_LEAD_HOURS: f64 = 1600.0;
    const COLLISION_SEEDS: [f64; N_STUDY_STAGES] = [110.0, 220.0, 330.0, 440.0];
    /// Post-study stage hours: fixed window `24, 168, 168` then the four
    /// commitment-carrying weeks/month.
    const COLLISION_POST_DURS: [f64; 7] = [24.0, 168.0, 168.0, 168.0, 168.0, 168.0, 648.0];

    fn collision_full_calendar() -> Vec<f64> {
        let mut cal = STUDY_DURS.to_vec();
        cal.extend_from_slice(&COLLISION_POST_DURS);
        cal
    }

    fn collision_post_study() -> PostStudyStages {
        let starts = [
            date(2026, 5, 1),
            date(2026, 5, 2),
            date(2026, 5, 9),
            date(2026, 5, 16),
            date(2026, 5, 23),
            date(2026, 5, 30),
            date(2026, 6, 6),
        ];
        let stages = starts
            .iter()
            .zip(COLLISION_POST_DURS)
            .map(|(&start_date, duration_hours)| PostStudyStage {
                start_date,
                duration_hours,
            })
            .collect();
        // The four commitment-carrying post-study stages (indices 3..7,
        // deliveries 7..11) each cost and bound the delivery study stages
        // 0..4 decide into; the fixed window (indices 0..3) carries no cell.
        let thermal_bounds = (3..7)
            .map(|post_study_stage_index| PostStudyThermalBound {
                thermal_id: ANTICIPATED_ID,
                post_study_stage_index,
                cost_per_mwh: 20.0,
                min_mw: 0.0,
                max_mw: 200.0,
            })
            .collect();
        PostStudyStages {
            stages,
            thermal_bounds,
        }
    }

    fn build_collision_system() -> cobre_core::System {
        SystemBuilder::new()
            .buses(vec![bus()])
            .thermals(vec![
                anticipated_thermal(AnticipatedConfig::LeadTime(COLLISION_LEAD_HOURS)),
                backup_thermal(),
            ])
            .stages(study_stages())
            .load_models(load_models())
            .bounds(bounds(4))
            .penalties(penalties())
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: seed_windows(COLLISION_SEEDS),
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .post_study_stages(Some(collision_post_study()))
            .build()
            .expect("build_collision_system: valid system")
    }

    /// The ring's excision keeps every delivery window on its own slot, so each
    /// study stage fishes its OWN seed and the ring depth is the true
    /// occupancy.
    ///
    /// Closed form: the reference deck resolves `decider[7 + t] = Some(t)` for
    /// `t in 0..4` (a `g == 3` fixed window at deliveries `4..7`), so the four
    /// in-study deliveries `0..4` are pre-study seeds fished at their own
    /// stages: `[110, 220, 330, 440]`. The occupancy is 4 (all four seeds are
    /// in flight at stage 0), so `k_max == 4` and the anticipated state is
    /// `n_anticipated * 4`.
    ///
    /// Pre-change (raw `m mod k_max`) code keys the ring on the full delivery
    /// axis, where `3 == 7 (mod 4)`: study-stage-3's seed (delivery 3) and the
    /// first post-study deposit (delivery 7) collide on one outgoing column —
    /// two definition rows — so stage 3 fishes the deposited value (or the
    /// solve reports a false `Infeasible`). Either symptom fails this test.
    #[test]
    fn excision_keeps_each_study_stage_fishing_its_own_seed() {
        // Assert the resolved decider directly rather than trusting the hour
        // value, so a fixture drift onto a different class boundary fails loudly
        // instead of silently testing something else. S_7 = 1512 < 1600 <=
        // 1680 = S_8 places delivery 7 (Sem 10) with study stage 0 and leaves
        // deliveries 4..7 the fixed window (g == 3).
        let calendar = collision_full_calendar();
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Time(COLLISION_LEAD_HOURS)],
            DeliveryAxis {
                study_stage_hours: &calendar[..N_STUDY_STAGES],
                post_study_stage_hours: &calendar[N_STUDY_STAGES..],
            },
        );
        let decider = &resolution.per_plant[0].decider;
        assert_eq!(
            decider,
            &vec![
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(0),
                Some(1),
                Some(2),
                Some(3),
            ],
            "collision fixture drift: study stage t must decide delivery 7 + t \
             (g == 3), so decider[7..11] == [Some(0), Some(1), Some(2), Some(3)]",
        );
        assert_eq!(
            resolution.anchored_depth(),
            4,
            "ring depth must be the true occupancy (4), not the width-inclusive \
             span (7)",
        );

        let config = super::build_config(ITERATIONS);
        let mut setup = build_setup_in_code(build_collision_system(), &config);

        // The excision must NOT pad the state with the masked fixed-window
        // slots: the anticipated contribution is n_anticipated * 4 (true
        // occupancy), never n_anticipated * 7 (width-inclusive span).
        let state = setup.stage_state();
        assert_eq!(state.n_anticipated, 1, "one anticipated plant");
        assert_eq!(
            state.k_max, 4,
            "resolved k_max must be the true occupancy depth 4, not the \
             width-inclusive span 7",
        );
        assert_eq!(
            state.commit_out.len(),
            state.n_anticipated * 4,
            "the anticipated state dimension must be n_anticipated * 4 (true \
             occupancy), not n_anticipated * 7 (width-inclusive span with \
             masked fixed-window slots)",
        );

        let results = run_simulation(&mut setup, ITERATIONS);
        assert_eq!(results.len(), 1, "one simulated scenario");
        let scenario = &results[0];
        assert_eq!(
            scenario.stages.len(),
            N_STUDY_STAGES,
            "one record per study stage",
        );

        for (t, &seed) in COLLISION_SEEDS.iter().enumerate() {
            let committed = committed_at(scenario, t).unwrap_or_else(|| {
                panic!("committed_at({t}) is None; the anticipated plant did not fish its seed")
            });
            assert!(
                (committed - seed).abs() < 1e-6,
                "study stage {t} must fish its OWN seed {seed} MW, got {committed} MW. \
                 A different value means the ring keyed on the raw delivery axis and \
                 aliased this slot onto another delivery's commitment.",
            );
        }
    }

    /// The bug doc's under-sizing reproduction: four study stages
    /// `[168, 168, 168, 648]`, one anticipated `LeadTime(1160)` thermal (lead >=
    /// horizon, so all four deliveries are pre-study), one post-study monthly
    /// stage, no boundary, four windows tiling the study at `100, 200, 300,
    /// 400` MW.
    const CLOSURE_LEAD_HOURS: f64 = 1160.0;
    const CLOSURE_SEEDS: [f64; N_STUDY_STAGES] = [100.0, 200.0, 300.0, 400.0];
    const CLOSURE_POST_DUR: f64 = 720.0;

    fn build_closure_system() -> cobre_core::System {
        let post_study = PostStudyStages {
            stages: vec![PostStudyStage {
                start_date: date(2026, 5, 1),
                duration_hours: CLOSURE_POST_DUR,
            }],
            thermal_bounds: vec![],
        };
        SystemBuilder::new()
            .buses(vec![bus()])
            .thermals(vec![
                anticipated_thermal(AnticipatedConfig::LeadTime(CLOSURE_LEAD_HOURS)),
                backup_thermal(),
            ])
            .stages(study_stages())
            .load_models(load_models())
            .bounds(bounds(4))
            .penalties(penalties())
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: seed_windows(CLOSURE_SEEDS),
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .post_study_stages(Some(post_study))
            .build()
            .expect("build_closure_system: valid system")
    }

    /// Sizing the ring `k_max = max(occupancy_max, n_none_in_study)` makes the
    /// last pre-study seed fish its own value under `n_post < n_stages`.
    ///
    /// Closed form: `decider == [None, None, None, None, Some(3)]` (the single
    /// post-study delivery decides at the last study stage), so occupancy peaks
    /// at 3 but four seeds are simultaneously in flight at stage 0
    /// (`n_none_in_study == 4`); `k_max == max(3, 4) == 4` gives every seed its
    /// own slot: `[100, 200, 300, 400]`, with stage 3 delivering `400 * 648 ==
    /// 259_200` MWh.
    ///
    /// Pre-change code sizes `k_max = occupancy_max = 3`, so the stage-0 seed
    /// write drops the overflow window and `slot(m) = m mod 3` aliases delivery
    /// 3 onto delivery 0's slot: stage 3 fishes `100` MW, delivering `100 * 648
    /// == 64_800` MWh — a silent wrong value the energy cross-check also catches.
    #[test]
    fn ring_depth_covers_every_simultaneous_pre_study_seed() {
        let calendar = {
            let mut cal = STUDY_DURS.to_vec();
            cal.push(CLOSURE_POST_DUR);
            cal
        };
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Time(CLOSURE_LEAD_HOURS)],
            DeliveryAxis {
                study_stage_hours: &calendar[..N_STUDY_STAGES],
                post_study_stage_hours: &calendar[N_STUDY_STAGES..],
            },
        );
        let point = &resolution.per_plant[0];
        assert_eq!(
            &point.decider,
            &vec![None, None, None, None, Some(3)],
            "closure fixture drift: the single post-study delivery must decide at \
             the last study stage (decider[4] == Some(3)) with four pre-study seeds",
        );
        assert_eq!(
            point.occupancy.iter().copied().max(),
            Some(3),
            "the fixture must sit in the under-sizing regime: occupancy peaks at 3",
        );
        assert_eq!(
            resolution.anchored_depth(),
            4,
            "k_max = max(occupancy_max 3, n_none_in_study 4) must widen to 4 so \
             every simultaneous pre-study seed gets its own slot",
        );

        let config = super::build_config(ITERATIONS);
        let mut setup = build_setup_in_code(build_closure_system(), &config);
        let results = run_simulation(&mut setup, ITERATIONS);
        assert_eq!(results.len(), 1, "one simulated scenario");
        let scenario = &results[0];
        assert_eq!(
            scenario.stages.len(),
            N_STUDY_STAGES,
            "one record per study stage",
        );

        for (t, &seed) in CLOSURE_SEEDS.iter().enumerate() {
            let committed = committed_at(scenario, t).unwrap_or_else(|| {
                panic!("committed_at({t}) is None; the anticipated plant did not fish its seed")
            });
            assert!(
                (committed - seed).abs() < 1e-6,
                "study stage {t} must fish its OWN seed {seed} MW, got {committed} MW. \
                 A wrong value here is the under-sizing alias (last stage delivers an \
                 earlier stage's seed).",
            );
        }

        // Energy cross-check: stage 3 delivers its own 400 MW seed over its
        // 648 h block. The aliased pre-change value would report 100 * 648.
        let stage3 = &scenario.stages[N_STUDY_STAGES - 1];
        let gen_mw = stage3
            .thermals
            .iter()
            .find(|th| th.thermal_id == ANTICIPATED_THERMAL_ID)
            .map(|th| th.generation_mw)
            .expect("anticipated thermal must appear in the stage-3 result");
        let gen_mwh = gen_mw * STUDY_DURS[N_STUDY_STAGES - 1];
        let expected_mwh = 400.0 * 648.0;
        assert!(
            (gen_mwh - expected_mwh).abs() / expected_mwh < 1e-6,
            "stage-3 delivered energy must be {expected_mwh} MWh (400 MW * 648 h), got \
             {gen_mwh} MWh; the pre-change alias would report {} MWh (100 MW * 648 h)",
            100.0 * 648.0,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Zero fixed post-horizon window: the excision is the identity map
    // ─────────────────────────────────────────────────────────────────────────
    //
    // When no delivery matures into a fixed post-horizon window, the ring-axis
    // excision is the identity: `ring_index(m) == Some(m)` everywhere,
    // `physical_target` is the identity, and `ring_depth` equals the occupancy
    // max (the widened sizing does not grow the depth). These standing anchors
    // pin that composed identity — across the slot encoder, the carry sweep,
    // the deposit latch, the ring depth, and the seed walk — on two
    // shipped-deck shapes: a study with post-study stages whose every
    // post-study delivery is decided in-study, and a study-only system. They
    // assert the built geometry only; the deterministic and parity suites carry
    // the end-to-end byte-neutrality proof over the shipped decks.

    const BN_PLANT_A_ID: EntityId = EntityId(20);
    const BN_PLANT_B_ID: EntityId = EntityId(21);
    const BN_STAGE_DUR_H: f64 = 168.0;
    const BN_PLANT_A_LEAD_STAGES: u32 = 2;
    /// Lead placing every post-study delivery on an in-study decider (no fixed
    /// window) at occupancy-max depth 4 on the extended calendar.
    const BN_WITH_POST_LEAD_HOURS: f64 = 700.0;
    /// One-stage lead: occupancy-max depth 1, heterogeneous against the
    /// `LeadStages(2)` plant's depth 2.
    const BN_STUDY_ONLY_LEAD_HOURS: f64 = 250.0;

    /// Four uniform 168 h study stages so every `LeadTime` decider resolves on a
    /// regular calendar (the study-anchored dates each span exactly 168 h).
    fn bn_uniform_study_stages() -> Vec<Stage> {
        let starts = [
            date(2026, 3, 14),
            date(2026, 3, 21),
            date(2026, 3, 28),
            date(2026, 4, 4),
            date(2026, 4, 11),
        ];
        (0..N_STUDY_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: starts[i],
                        end_date: starts[i + 1],
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: BN_STAGE_DUR_H,
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
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    fn bn_anticipated(
        id: EntityId,
        name: &str,
        lead: AnticipatedConfig,
    ) -> cobre_core::entities::thermal::Thermal {
        make_thermal(
            id,
            ThermalSpec {
                name: name.to_string(),
                operational_start_date: date(2026, 3, 14),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: ANTICIPATED_MAX_MW,
                cost_per_mwh: ANTICIPATED_COST,
                anticipated_config: Some(lead),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        )
    }

    /// The two anticipated plants, id-ascending so plant A (`LeadStages`) is
    /// anticipated-local index 0 and plant B (`LeadTime`) is index 1.
    fn bn_thermals(lead_time_hours: f64) -> Vec<cobre_core::entities::thermal::Thermal> {
        vec![
            bn_anticipated(
                BN_PLANT_A_ID,
                "T_leadstages",
                AnticipatedConfig::LeadStages(BN_PLANT_A_LEAD_STAGES),
            ),
            bn_anticipated(
                BN_PLANT_B_ID,
                "T_leadtime",
                AnticipatedConfig::LeadTime(lead_time_hours),
            ),
        ]
    }

    fn bn_leads(lead_time_hours: f64) -> [LeadTime; 2] {
        [
            LeadTime::Stages(BN_PLANT_A_LEAD_STAGES),
            LeadTime::Time(lead_time_hours),
        ]
    }

    fn bn_resolve(leads: &[LeadTime], durations: &[f64]) -> AnticipatedResolution {
        AnticipatedResolution::resolve(
            leads,
            DeliveryAxis {
                study_stage_hours: &durations[..N_STUDY_STAGES],
                post_study_stage_hours: &durations[N_STUDY_STAGES..],
            },
        )
    }

    fn bn_with_post_study_durations() -> Vec<f64> {
        vec![BN_STAGE_DUR_H; N_STUDY_STAGES + 2]
    }

    fn bn_study_only_durations() -> Vec<f64> {
        vec![BN_STAGE_DUR_H; N_STUDY_STAGES]
    }

    fn bn_with_post_study_system() -> cobre_core::System {
        let post_study = PostStudyStages {
            stages: vec![
                PostStudyStage {
                    start_date: date(2026, 4, 11),
                    duration_hours: BN_STAGE_DUR_H,
                },
                PostStudyStage {
                    start_date: date(2026, 4, 18),
                    duration_hours: BN_STAGE_DUR_H,
                },
            ],
            thermal_bounds: vec![],
        };
        SystemBuilder::new()
            .buses(vec![bus()])
            .thermals(bn_thermals(BN_WITH_POST_LEAD_HOURS))
            .stages(bn_uniform_study_stages())
            .load_models(load_models())
            .bounds(bounds(4))
            .penalties(penalties())
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: vec![],
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .post_study_stages(Some(post_study))
            .build()
            .expect("bn_with_post_study_system: valid system")
    }

    fn bn_study_only_system() -> cobre_core::System {
        SystemBuilder::new()
            .buses(vec![bus()])
            .thermals(bn_thermals(BN_STUDY_ONLY_LEAD_HOURS))
            .stages(bn_uniform_study_stages())
            .load_models(load_models())
            .bounds(bounds(4))
            .penalties(penalties())
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: vec![],
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .build()
            .expect("bn_study_only_system: valid system")
    }

    /// Assert the resolved state geometry is the identity excision: every
    /// plant's `ring_index` is `Some(m)`, `ring_depth` equals the occupancy max
    /// (no widening), `k_max` is the occupancy max over both plants (computed
    /// here, never a literal), the anticipated state is `n_anticipated * k_max`
    /// (no masked padding), and each plant's `anticipated_lead_stages` matches
    /// the declared stage count for the `LeadStages` plant and the occupancy max
    /// for the `LeadTime` plant.
    fn bn_assert_identity_geometry(
        n_anticipated: usize,
        k_max: usize,
        commit_out_len: usize,
        anticipated_lead_stages: &[usize],
        resolution: &AnticipatedResolution,
        leads: &[LeadTime],
    ) {
        assert_eq!(
            n_anticipated, 2,
            "both fixtures declare two anticipated plants"
        );
        assert_eq!(
            resolution.per_plant.len(),
            2,
            "the resolution must carry both plants",
        );

        for point in &resolution.per_plant {
            for m in 0..point.decider.len() {
                assert_eq!(
                    point.ring_index(m),
                    Some(m),
                    "ring_index must be the identity across the whole delivery range at m={m}",
                );
            }
            assert_eq!(
                point.occupancy.iter().copied().max(),
                Some(point.ring_depth()),
                "ring_depth must equal the occupancy max (no widening) with no fixed post-horizon window",
            );
        }

        // A single per-plant width applied globally is invisible with one plant
        // or two identical plants; the two leads must resolve DIFFERENT depths.
        let depths: Vec<usize> = resolution
            .per_plant
            .iter()
            .map(PointResolution::ring_depth)
            .collect();
        assert_ne!(
            depths[0], depths[1],
            "the two plants must resolve heterogeneous depths",
        );

        // The RELATION between ring_depth and occupancy is the property under
        // test, so k_max is recomputed from occupancy here rather than pinned.
        let expected_k_max = resolution
            .per_plant
            .iter()
            .filter_map(|p| p.occupancy.iter().copied().max())
            .max()
            .unwrap_or(0);
        assert_eq!(
            k_max, expected_k_max,
            "resolved k_max must equal the occupancy max over both plants",
        );
        assert_eq!(
            commit_out_len,
            n_anticipated * k_max,
            "the anticipated state dimension must be n_anticipated * k_max (no masked padding)",
        );

        let expected_lead_stages: Vec<usize> = leads
            .iter()
            .zip(&resolution.per_plant)
            .map(|(lead, point)| match lead {
                LeadTime::Stages(l) => *l as usize,
                LeadTime::Time(_) => point.occupancy.iter().copied().max().unwrap_or(0),
            })
            .collect();
        assert_eq!(
            anticipated_lead_stages,
            expected_lead_stages.as_slice(),
            "per-plant anticipated_lead_stages must be the declared count (LeadStages) or the occupancy max (LeadTime)",
        );
    }

    /// Assert the carry-slot addressing at every study stage reproduces the
    /// pre-excision identity formula `(stage_idx + depth + 1) % k_max`: with no
    /// fixed post-horizon window the excision is inert (`physical_target(r) ==
    /// r`), so the ring slot for delivery `r` is its own residue and
    /// `ring_index` inverts it — the composed carry-sweep identity.
    fn bn_assert_carry_slot_identity(
        resolution: &AnticipatedResolution,
        k_max: usize,
        n_stages: usize,
        n_delivery: usize,
    ) {
        assert!(k_max >= 1, "fixture must resolve a non-empty ring");
        for stage_idx in 0..n_stages {
            for depth in 0..k_max {
                let r = stage_idx + depth + 1;
                if r >= n_delivery {
                    continue;
                }
                let slot = r % k_max;
                for point in &resolution.per_plant {
                    let m = point.physical_target(r);
                    assert_eq!(
                        m, r,
                        "the excision must be inert at ring-axis index {r} (stage_idx={stage_idx}, depth={depth})",
                    );
                    assert_eq!(
                        m % k_max,
                        slot,
                        "the carry slot for delivery {m} must be the pre-excision (stage_idx + depth + 1) % k_max = {slot}",
                    );
                    assert_eq!(
                        point.ring_index(m),
                        Some(m),
                        "ring_index must invert physical_target on the identity axis",
                    );
                }
            }
        }
    }

    /// A study with post-study stages whose every post-study delivery is decided
    /// in-study (no fixed post-horizon window) resolves an identity excision on
    /// both plants: identity ring, occupancy-max depth, and per-plant leads.
    #[test]
    fn zero_gap_with_post_study_resolves_an_identity_ring_and_occupancy_depth() {
        let leads = bn_leads(BN_WITH_POST_LEAD_HOURS);
        let durations = bn_with_post_study_durations();
        assert!(
            durations.len() > N_STUDY_STAGES,
            "fixture must extend past the study horizon",
        );
        let resolution = bn_resolve(&leads, &durations);

        let config = super::build_config(ITERATIONS);
        let setup = build_setup_in_code(bn_with_post_study_system(), &config);
        let state = setup.stage_state();
        bn_assert_identity_geometry(
            state.n_anticipated,
            state.k_max,
            state.commit_out.len(),
            &state.anticipated_lead_stages,
            &resolution,
            &leads,
        );
    }

    /// A study-only system (no post-study stages) resolves an identity excision
    /// on both plants — the shape most shipped decks have.
    #[test]
    fn zero_gap_study_only_resolves_an_identity_ring_and_occupancy_depth() {
        let leads = bn_leads(BN_STUDY_ONLY_LEAD_HOURS);
        let durations = bn_study_only_durations();
        assert_eq!(
            durations.len(),
            N_STUDY_STAGES,
            "study-only fixture must not extend past the study horizon",
        );
        let resolution = bn_resolve(&leads, &durations);

        let config = super::build_config(ITERATIONS);
        let setup = build_setup_in_code(bn_study_only_system(), &config);
        let state = setup.stage_state();
        bn_assert_identity_geometry(
            state.n_anticipated,
            state.k_max,
            state.commit_out.len(),
            &state.anticipated_lead_stages,
            &resolution,
            &leads,
        );
    }

    /// Both zero-fixed-window fixtures address every study stage's carry slots by
    /// the pre-excision identity formula `(stage_idx + depth + 1) % k_max`.
    #[test]
    fn zero_gap_carry_slot_addressing_matches_the_open_coded_identity_formula() {
        let cases = [
            (
                bn_with_post_study_system(),
                bn_with_post_study_durations(),
                bn_leads(BN_WITH_POST_LEAD_HOURS),
            ),
            (
                bn_study_only_system(),
                bn_study_only_durations(),
                bn_leads(BN_STUDY_ONLY_LEAD_HOURS),
            ),
        ];
        for (system, durations, leads) in cases {
            let resolution = bn_resolve(&leads, &durations);
            let config = super::build_config(ITERATIONS);
            let setup = build_setup_in_code(system, &config);
            let k_max = setup.stage_state().k_max;
            bn_assert_carry_slot_identity(&resolution, k_max, N_STUDY_STAGES, durations.len());
        }
    }
}

mod anticipated_no_boundary_fixed_value_inertness {
    //! Pins `.claude/rules/sddp.md`'s "no-boundary inertness" contract: a
    //! declared class-4 fixed post-horizon value with no `config.policy.boundary`
    //! must never reach any LP or output cost.
    //!
    //! ## Fixture
    //!
    //! A `min_mw == max_mw` replay deck: the anticipated plant's own
    //! delivery-stage generation column is pinned to [`REPLAY_MW`] with no
    //! optimization freedom, so a class-4 leak into the LP would perturb
    //! `final_lb` directly rather than being reabsorbed by unrelated slack.
    //! Two stages, `LeadStages(1)`, no hydro, no stochastic noise, no
    //! post-study stages, no `config.policy.boundary`. The class-4 window
    //! starts exactly at the study horizon end (`start_date >= horizon_end`).
    //!
    //! Training two otherwise-identical decks that differ only in the class-4
    //! window's `value_mw` (non-zero vs. zeroed) and comparing `final_lb`
    //! bit-for-bit is a stronger check than a closed-form pin against a single
    //! deck: it directly measures the fixed value's effect on the solved cost
    //! rather than an absolute number that could coincidentally match.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    const N_STAGES: usize = 2;
    const K_MAX: usize = 1;
    const BLOCK_HOURS: f64 = 1.0;

    /// The single in-study decision (stage 0) targets delivery at this stage.
    const DELIVERY_STAGE: usize = 1;

    /// Forced replay generation (MW): pinned `min == max` at [`DELIVERY_STAGE`],
    /// leaving the anticipated plant zero optimization freedom.
    const REPLAY_MW: f64 = 40.0;

    /// $/MWh, booked once on the decision column.
    const C_A: f64 = 10.0;

    /// Non-zero class-4 fixed post-horizon value (MW); the study under test
    /// declares no boundary, so this must never move `final_lb`.
    const CLASS4_VALUE_MW: f64 = 77.0;

    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BUS_ID: EntityId = EntityId(1);
    const THERMAL_IDX_ANT: usize = 0;

    /// Build the deck. `class4_value_mw` is the only input that varies between
    /// the two trainings the test compares.
    fn build_system(class4_value_mw: f64) -> cobre_core::System {
        use chrono::{NaiveDate, TimeDelta};

        let stage_date = |index: usize| {
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(index as i64)
        };
        let horizon_end = stage_date(N_STAGES);

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: REPLAY_MW,
                cost_per_mwh: C_A,
                anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: BLOCK_HOURS,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        // Load matches the forced replay exactly at the delivery stage and is
        // zero at the decision stage — the always-active fishing row forces the
        // anticipated column's own generation to zero there — so no backup
        // thermal is needed to balance either stage.
        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: if i == DELIVERY_STAGE { REPLAY_MW } else { 0.0 },
                std_mw: 0.0,
            })
            .collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        // `thermal_block_base` at the delivery stage sets the DECISION column's
        // own [min, max] (delivery-anchored read); pinning it `REPLAY_MW ==
        // REPLAY_MW` is the "min_mw == max_mw replay deck" the no-reject
        // contract names. Every other axis slot stays permissive — the
        // decision stage's own generation is forced to zero by the
        // always-active fishing row regardless of its bounds.
        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) =
                ThermalStageBounds { cost_per_mwh: C_A };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = if s == DELIVERY_STAGE {
                ThermalBlockBounds {
                    min_generation_mw: REPLAY_MW,
                    max_generation_mw: REPLAY_MW,
                }
            } else {
                ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: REPLAY_MW,
                }
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // The class-4 window starts exactly at the horizon end — strictly past
        // every in-study stage, so `StageCalendar::coverage` (the ring-seed read
        // at initial-state construction) assigns it zero coverage on every
        // in-study stage: it is structurally invisible to the ring regardless of
        // `boundary_present`.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                start_date: horizon_end,
                end_date: horizon_end + TimeDelta::days(30),
                value_mw: class4_value_mw,
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    /// `PolicyConfig::default()` leaves `boundary: None`, so `boundary_present
    /// == false` and the class-4 fold path (confined to `load_boundary_cuts`)
    /// never runs.
    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    fn train_final_lb(class4_value_mw: f64) -> f64 {
        let system = build_system(class4_value_mw);
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        outcome.result.final_lb
    }

    /// `.claude/rules/sddp.md`'s "no-boundary inertness": a `min_mw == max_mw`
    /// replay-shaped study with a declared non-zero class-4 fixed post-horizon
    /// value and no boundary trains to the SAME `final_lb` as the identical
    /// study with that value zeroed.
    #[test]
    fn no_boundary_nonzero_fixed_value_leaves_solved_cost_untouched() {
        let lb_nonzero = train_final_lb(CLASS4_VALUE_MW);
        let lb_zeroed = train_final_lb(0.0);

        assert!(
            lb_nonzero.is_finite(),
            "final_lb must be finite; got {lb_nonzero}"
        );
        assert_eq!(
            lb_nonzero.to_bits(),
            lb_zeroed.to_bits(),
            "a declared class-4 fixed post-horizon value with no boundary must \
             leave the solved cost untouched: nonzero = {lb_nonzero}, zeroed = {lb_zeroed}",
        );
    }
}

mod fixed_delivery_output {
    //! End-to-end pins for the run-level fixed post-horizon commitment table:
    //! accessor -> rows -> file, against a real `StudySetup`/`System`. Extends
    //! the no-boundary class-4 fixture shape above to two anticipated plants
    //! for the canonical-order and declaration-order-invariance regressions.
    //! No training solve: [`build_fixed_delivery_rows`] reads only the
    //! resolved input the setup already carries.

    use chrono::{NaiveDate, TimeDelta};

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::{LoadModel, SamplingScheme};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_io::{FixedDeliveryRow, write_fixed_delivery};
    use cobre_sddp::hydro_models::PrepareHydroModelsResult;
    use cobre_sddp::{StudySetup, build_fixed_delivery_rows};
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
    use tempfile::tempdir;

    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    const N_STAGES: usize = 2;
    const K_MAX: usize = 1;
    const BLOCK_HOURS: f64 = 1.0;
    const BUS_ID: EntityId = EntityId(1);

    /// Same `operational_start_date` as [`PLANT_HI`] — declared canonically
    /// first only by the lower id, so a fixture that declares plants/windows
    /// in the OTHER order still probes the accessor's own `(date, id)` sort
    /// rather than the input declaration order.
    const PLANT_LO: EntityId = EntityId(3);
    const PLANT_HI: EntityId = EntityId(7);

    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("valid date")
    }

    fn stage_date(index: usize) -> NaiveDate {
        date(2030, 1, 1) + TimeDelta::days(index as i64)
    }

    /// A declared class-4 window, field-for-field with
    /// [`AnticipatedCommitmentHistory`], real calendar dates well past the
    /// study horizon (`stage_date(N_STAGES)`).
    #[derive(Clone, Copy)]
    struct DeclaredWindow {
        thermal_id: EntityId,
        start_date: NaiveDate,
        end_date: NaiveDate,
        value_mw: f64,
    }

    fn window_lo() -> DeclaredWindow {
        DeclaredWindow {
            thermal_id: PLANT_LO,
            start_date: date(2030, 4, 1),
            end_date: date(2030, 4, 15),
            value_mw: 88.0,
        }
    }

    fn window_hi() -> DeclaredWindow {
        DeclaredWindow {
            thermal_id: PLANT_HI,
            start_date: date(2030, 5, 1),
            end_date: date(2030, 5, 31),
            value_mw: 120.5,
        }
    }

    /// Build a deck declaring exactly `thermal_ids` (each an anticipated
    /// `LeadStages(1)` plant on the shared bus) and exactly `windows` (each a
    /// class-4 `past_anticipated_commitments` entry), in the given order —
    /// the caller controls declaration order to probe the accessor's own
    /// canonicalization rather than relying on it.
    fn build_system(thermal_ids: &[EntityId], windows: &[DeclaredWindow]) -> System {
        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: stage_date(0),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermals = thermal_ids
            .iter()
            .map(|&id| {
                make_thermal(
                    id,
                    ThermalSpec {
                        name: format!("T{}", id.0),
                        operational_start_date: stage_date(0),
                        bus_id: BUS_ID,
                        min_generation_mw: 0.0,
                        max_generation_mw: 10.0,
                        cost_per_mwh: 10.0,
                        anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                        entry_stage_id: None,
                        exit_stage_id: None,
                        ..Default::default()
                    },
                )
            })
            .collect();

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_date(i),
                        end_date: stage_date(i + 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: BLOCK_HOURS,
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
                        ..Default::default()
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: 0.0,
                std_mw: 0.0,
            })
            .collect();

        let n_thermals = thermal_ids.len();
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let thermal_axis = N_STAGES + K_MAX;
        for idx in 0..n_thermals {
            for s in 0..thermal_axis {
                *bounds.thermal_bounds_mut(idx, s) = ThermalStageBounds { cost_per_mwh: 10.0 };
                *bounds.thermal_block_base_mut(idx, s) = ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 10.0,
                };
            }
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let past_anticipated_commitments = windows
            .iter()
            .map(|w| AnticipatedCommitmentHistory {
                thermal_id: w.thermal_id,
                start_date: w.start_date,
                end_date: w.end_date,
                value_mw: w.value_mw,
            })
            .collect();

        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments,
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(thermals)
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// Builds the setup the same way [`StudySetup::train`] callers do.
    /// [`StudySetup`] does not own `system`, so callers keep their own
    /// `&System` alongside the returned setup for [`build_fixed_delivery_rows`].
    fn build_setup_and_system(system: &System, config: &Config) -> StudySetup {
        let stochastic = build_stochastic_context(
            system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("build_stochastic_context");

        let hydro_models = PrepareHydroModelsResult::default_from_system(system);

        StudySetup::new(system, config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup::new")
    }

    fn assert_row_matches_window(row: &FixedDeliveryRow, expected: &DeclaredWindow) {
        assert_eq!(row.thermal_id, expected.thermal_id.0, "thermal_id");
        assert_eq!(row.start_date, expected.start_date, "start_date");
        assert_eq!(row.end_date, expected.end_date, "end_date");
        assert_eq!(row.value_mw, expected.value_mw, "value_mw");
    }

    fn assert_rows_equal(a: &[FixedDeliveryRow], b: &[FixedDeliveryRow]) {
        assert_eq!(a.len(), b.len(), "row count must match");
        for (ra, rb) in a.iter().zip(b) {
            assert_eq!(ra.thermal_id, rb.thermal_id, "thermal_id");
            assert_eq!(ra.start_date, rb.start_date, "start_date");
            assert_eq!(ra.end_date, rb.end_date, "end_date");
            assert_eq!(ra.value_mw, rb.value_mw, "value_mw");
        }
    }

    /// Real declared dates, verbatim (never mirror-shifted or
    /// horizon-anchored), in canonical-plant then ascending-`start_date`
    /// order — pinned by declaring both plants and windows in the REVERSE of
    /// that order, so a broken canonicalization would surface here.
    #[test]
    fn fixed_delivery_rows_carry_declared_dates_in_canonical_order() {
        let lo = window_lo();
        let hi = window_hi();

        let system = build_system(&[PLANT_HI, PLANT_LO], &[hi, lo]);
        let config = build_config();
        let setup = build_setup_and_system(&system, &config);

        let rows = build_fixed_delivery_rows(&setup, &system);

        assert_eq!(rows.len(), 2, "one row per declared window");
        assert_row_matches_window(&rows[0], &lo);
        assert_row_matches_window(&rows[1], &hi);
    }

    #[test]
    fn fixed_delivery_file_exists_after_declared_rows_are_written() {
        let lo = window_lo();
        let hi = window_hi();
        let system = build_system(&[PLANT_LO, PLANT_HI], &[lo, hi]);
        let config = build_config();
        let setup = build_setup_and_system(&system, &config);
        let rows = build_fixed_delivery_rows(&setup, &system);
        assert_eq!(rows.len(), 2);

        let root = tempdir().expect("tempdir");
        write_fixed_delivery(root.path(), &rows).expect("write must succeed");

        let path = root
            .path()
            .join("anticipated")
            .join("fixed_deliveries.parquet");
        assert!(path.exists(), "file must exist after a non-empty write");
    }

    /// The absence check is against a concrete fact — no file, no directory —
    /// never a `load(&[])`-vs-itself self-comparison.
    #[test]
    fn fixed_delivery_absence_is_byte_neutral_no_file_no_directory() {
        let system = build_system(&[PLANT_LO], &[]);
        let config = build_config();
        let setup = build_setup_and_system(&system, &config);

        let rows = build_fixed_delivery_rows(&setup, &system);
        assert!(rows.is_empty(), "no class-4 window declared, so no rows");

        let root = tempdir().expect("tempdir");
        write_fixed_delivery(root.path(), &rows).expect("an empty write must succeed");

        let anticipated_dir = root.path().join("anticipated");
        assert!(
            !anticipated_dir.exists(),
            "an empty declaration must not create the anticipated/ directory"
        );
        assert!(
            !anticipated_dir.join("fixed_deliveries.parquet").exists(),
            "an empty declaration must not write the parquet file"
        );
    }

    /// The workspace declaration-order-invariance hard rule: two decks
    /// differing only in the order plants and windows were declared produce
    /// byte-identical rows, via the accessor's own canonical sort.
    #[test]
    fn fixed_delivery_rows_are_declaration_order_invariant() {
        let lo = window_lo();
        let hi = window_hi();
        let config = build_config();

        let system_forward = build_system(&[PLANT_LO, PLANT_HI], &[lo, hi]);
        let setup_forward = build_setup_and_system(&system_forward, &config);
        let rows_forward = build_fixed_delivery_rows(&setup_forward, &system_forward);

        let system_reversed = build_system(&[PLANT_HI, PLANT_LO], &[hi, lo]);
        let setup_reversed = build_setup_and_system(&system_reversed, &config);
        let rows_reversed = build_fixed_delivery_rows(&setup_reversed, &system_reversed);

        assert_eq!(rows_forward.len(), 2);
        assert_rows_equal(&rows_forward, &rows_reversed);
    }
}
