//! Audits the two in-class state families the admissible-box mechanism
//! analysis flags: inflow lags and the PreFilling frozen storage identity.
//!
//! Inflow lags carry an unbounded box. The incoming AR-lag column enters only
//! the z-inflow definition row (`fill_z_inflow_entries`) and the water-balance
//! row's own inflow term (`fill_parallel_water_entries` /
//! `fill_chronological_water_entries`); no row couples it to a bounded
//! generation column, and the water-balance row's own slack columns absorb
//! any magnitude on either side, so driving a lag to an extreme value never
//! produces a false infeasibility.
//!
//! The PreFilling frozen identity `v_h - v_h_in = 0` (`hydro_phase`,
//! `fill_parallel_water_entries`) couples only the outgoing storage column,
//! which `fill_storage_columns`'s own relaxed `[0, max_storage_hm3]` floor
//! already covers — PreFilling needs no separate admissible box.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::needless_update,
    clippy::too_many_lines
)]

mod common;

mod inflow_lag_audit {
    use chrono::{NaiveDate, TimeDelta};
    use cobre_core::entities::bus::DeficitSegment;
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage, InitialConditions,
        LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig,
        TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::setup::{NodeId, StageIdx};
    use cobre_sddp::test_support::{patch_backward_opening_for_probe, solve_stage_for_probe};
    use cobre_sddp::workspace::SolverWorkspace;
    use cobre_sddp::{SddpError, StudySetup};
    use cobre_solver::{ActiveSolver, SolverInterface};

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };

    const N_STAGES: usize = 2;
    const TERMINAL_STAGE: usize = N_STAGES - 1;
    const BUS_ID: EntityId = EntityId(1);
    const HYDRO_ID: EntityId = EntityId(2);
    const MAX_STORAGE_HM3: f64 = 200.0;
    const EXTREME_LAG_M3S: f64 = 1.0e8;

    fn hydro_penalties() -> HydroPenalties {
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

    /// One bus (unbounded deficit), one AR(1) hydro, two stages — the minimal
    /// fixture with a reachable inflow-lag state dimension.
    fn build_system() -> cobre_core::System {
        let anchor = NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date");

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: anchor,
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let hydro = make_hydro(
            HYDRO_ID,
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: anchor,
                bus_id: BUS_ID,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_storage_hm3: 0.0,
                max_storage_hm3: MAX_STORAGE_HM3,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                penalties: hydro_penalties(),
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: anchor + TimeDelta::days(31 * i as i64),
                        end_date: anchor + TimeDelta::days(31 * (i as i64 + 1)),
                        season_id: Some(0),
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 744.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: true,
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

        let inflow_models: Vec<InflowModel> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id: HYDRO_ID,
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.5],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: MAX_STORAGE_HM3,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
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

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: HYDRO_ID,
                value_hm3: 100.0,
            }],
            ..Default::default()
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("inflow-lag audit fixture: system must build")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: InflowNonNegativityMethod::Penalty,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: SimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// Solve the terminal stage's own structural template (no cuts exist yet
    /// on a freshly built setup) with the incoming state vector pinned to
    /// `pinned_state`.
    fn solve_terminal_with_pin(setup: &StudySetup, pinned_state: &[f64]) -> Result<(), SddpError> {
        let comm = super::common::StubComm;
        let mut workspace_pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("create_workspace_pool must succeed");
        let ws: &mut SolverWorkspace<ActiveSolver> = &mut workspace_pool.workspaces[0];
        let ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let raw_noise = vec![0.0_f64; setup.stage_state().hydro_count];
        let pool = &setup.fcf.pools[TERMINAL_STAGE];

        ws.solver.reset_solver_state();
        ws.solver.load_model(&ctx.templates[TERMINAL_STAGE]);
        patch_backward_opening_for_probe(
            ws,
            &ctx,
            &training_ctx,
            StageIdx(TERMINAL_STAGE),
            pinned_state,
            &raw_noise,
        );

        solve_stage_for_probe(
            ws,
            &ctx,
            pool,
            None,
            StageIdx(TERMINAL_STAGE),
            0,
            NodeId(TERMINAL_STAGE as i32),
        )
        .map(|_| ())
    }

    #[test]
    fn state_family_audit_inflow_lag_extreme_is_feasible() {
        let setup = build_setup_in_code(build_system(), &build_config());
        let state = setup.stage_state();
        assert_eq!(
            state.max_par_order, 1,
            "fixture must declare a PAR(1) inflow lag"
        );
        assert_eq!(state.hydro_count, 1, "fixture declares exactly one hydro");

        let storage_col = state.storage.start;
        let lag_col = state.inflow_lags.start;

        for &extreme in &[EXTREME_LAG_M3S, -EXTREME_LAG_M3S] {
            let mut pinned_state = vec![0.0_f64; state.n_state];
            pinned_state[storage_col] = 100.0;
            pinned_state[lag_col] = extreme;

            let result = solve_terminal_with_pin(&setup, &pinned_state);
            assert!(
                result.is_ok(),
                "inflow lag pinned to {extreme:e} must not produce a false infeasibility: {:?}",
                result.err()
            );
        }
    }
}

mod prefilling_audit {
    use chrono::{NaiveDate, TimeDelta};
    use cobre_core::entities::bus::DeficitSegment;
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage, InitialConditions,
        LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig,
        TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::test_support::stage_state_box_bounds;

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };

    const N_STAGES: usize = 2;
    const PREFILLING_STAGE: usize = 0;
    const ENTRY_STAGE_ID: i32 = 1;
    const BUS_ID: EntityId = EntityId(1);
    const HYDRO_ID: EntityId = EntityId(2);
    const MIN_STORAGE_HM3: f64 = 20.0;
    const MAX_STORAGE_HM3: f64 = 150.0;

    fn hydro_penalties() -> HydroPenalties {
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

    /// One bus (unbounded deficit), one non-filling hydro commissioning at
    /// `ENTRY_STAGE_ID` (`PreFilling` at stage 0, `Operating` from stage 1),
    /// a nonzero declared `min_storage_hm3` so the PreFilling floor relax is
    /// observable rather than incidental.
    fn build_system() -> cobre_core::System {
        let anchor = NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date");

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: anchor,
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let hydro = make_hydro(
            HYDRO_ID,
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: anchor,
                bus_id: BUS_ID,
                entry_stage_id: Some(ENTRY_STAGE_ID),
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_storage_hm3: MIN_STORAGE_HM3,
                max_storage_hm3: MAX_STORAGE_HM3,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                penalties: hydro_penalties(),
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: anchor + TimeDelta::days(31 * i as i64),
                        end_date: anchor + TimeDelta::days(31 * (i as i64 + 1)),
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

        let inflow_models: Vec<InflowModel> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id: HYDRO_ID,
                stage_id: i as i32,
                mean_m3s: 40.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: 50.0,
                std_mw: 0.0,
            })
            .collect();

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: MIN_STORAGE_HM3,
                    max_storage_hm3: MAX_STORAGE_HM3,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 200.0,
                    ..Default::default()
                },
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

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: HYDRO_ID,
                value_hm3: 100.0,
            }],
            ..Default::default()
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("PreFilling audit fixture: system must build")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: InflowNonNegativityMethod::Penalty,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: SimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    #[test]
    fn state_family_audit_prefilling_is_covered_by_storage_box() {
        let setup = build_setup_in_code(build_system(), &build_config());
        let state = setup.stage_state();
        assert_eq!(state.hydro_count, 1, "fixture declares exactly one hydro");

        let storage_col = state.storage.start;
        assert!(
            state.storage.contains(&storage_col),
            "the PreFilling hydro's outgoing storage column must lie within layout.storage"
        );

        let (lower, upper) = stage_state_box_bounds(&setup, PREFILLING_STAGE);
        assert_eq!(
            lower[storage_col], 0.0,
            "PreFilling's relaxed floor must read 0.0 from the storage box, not the declared \
             min_storage_hm3 ({MIN_STORAGE_HM3})"
        );
        assert_eq!(
            upper[storage_col], MAX_STORAGE_HM3,
            "the storage box's upper bound must equal the declared max_storage_hm3"
        );
    }
}
