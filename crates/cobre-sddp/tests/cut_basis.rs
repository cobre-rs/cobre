//! Consolidated cut / basis / warm-start integration tests for `cobre-sddp`.
//!
//! Each cut/basis domain group lives in its own inner `mod` so the suite links
//! the statically-bound solver once rather than once per file. Per-`mod` scoping
//! isolates each group's consts, helpers, and fixtures.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

mod common;

mod boundary_cuts {
    //! Terminal-stage boundary cuts: a consumer study loads boundary cuts from a
    //! source checkpoint; they must inject into the terminal pool and the lower
    //! bound must not degrade versus the same study without them.

    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::STAGE_CUTS_PRICED_STATE_DATE_SENTINEL;
    use cobre_io::config::StoppingRuleConfig;
    use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    fn d01_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d01-thermal-dispatch")
    }

    /// Ascending, distinct `end_date` per pool (`2030-01-01` plus the pool
    /// index in months) — the study-stage calendar [`write_test_checkpoint`]
    /// feeds [`build_stage_cuts_payloads`] to self-describe each pool's
    /// `priced_state_date`.
    fn ascending_stage_end_dates(n_pools: usize) -> Vec<chrono::NaiveDate> {
        let base = chrono::NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        (0..n_pools as u32)
            .map(|i| base.checked_add_months(chrono::Months::new(i)).unwrap())
            .collect()
    }

    fn write_test_checkpoint(
        policy_dir: &Path,
        setup: &StudySetup,
        result: &cobre_sddp::TrainingResult,
        seed: u64,
    ) {
        use cobre_sddp::policy_export::{
            build_active_indices, build_stage_basis_records, build_stage_cut_records,
            build_stage_cuts_payloads, convert_basis_cache,
        };
        let fcf = &setup.fcf;
        let stage_records = build_stage_cut_records(fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![Vec::new(); fcf.pools.len()];
        let study_stage_ids: Vec<i32> = (0..fcf.pools.len() as i32).collect();
        let study_stage_end_dates = ascending_stage_end_dates(fcf.pools.len());
        let stage_cuts = build_stage_cuts_payloads(
            fcf,
            &setup.inputs.node_graph,
            &study_stage_ids,
            &study_stage_end_dates,
            1_000_000.0,
            &stage_records,
            &stage_active_indices,
            &stage_manifests,
        );
        let (basis_col, basis_row) = convert_basis_cache(&result.basis_cache);
        let stage_bases = build_stage_basis_records(
            &result.basis_cache,
            result.iterations,
            &basis_col,
            &basis_row,
        );
        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
        let metadata = cobre_sddp::test_support::checkpoint_metadata(
            fcf.pools.len() as u32,
            setup.build_graph_manifest(),
            cobre_io::ProducerBlock {
                completed_iterations: result.iterations as u32,
                final_lower_bound: result.final_lb,
                best_upper_bound: Some(result.final_ub),
                max_iterations: setup.loop_params.max_iterations as u32,
                forward_passes: setup.loop_params.forward_passes,
                warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
                warm_start_counts,
                rng_seed: seed,
                total_visited_states: 0,
                training_block_mode: "parallel".to_string(),
                training_block_mode_per_stage: vec![],
                cost_scale_factor: None,
                lower_bound_history: Vec::new(),
            },
        );
        write_policy_checkpoint(policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write checkpoint");
    }

    fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> (StudySetup, cobre_core::System) {
        let system = cobre_io::load_case(case_dir).expect("load_case");
        let prep = prepare_stochastic(
            system,
            case_dir,
            config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic");
        let hydro_models =
            prepare_hydro_models(&prep.system, case_dir, false).expect("prepare_hydro_models");
        let setup = StudySetup::new(
            &prep.system,
            config,
            prep.stochastic,
            hydro_models,
            Vec::new(),
        )
        .expect("StudySetup::new");
        (setup, prep.system)
    }

    #[test]
    fn boundary_cuts_improve_terminal_stage_objective() {
        let case_dir = d01_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let mut config_5iter = config;
        config_5iter.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 5 }]);

        let (mut setup_a, _system_a) = build_setup(&case_dir, &config_5iter);
        let comm = StubComm;
        let mut solver_a = ActiveSolver::new().expect("solver");
        let outcome_a = setup_a
            .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
            .expect("train A");
        assert!(outcome_a.error.is_none());

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let source_policy_dir = tmpdir.path().join("source_policy");
        write_test_checkpoint(&source_policy_dir, &setup_a, &outcome_a.result, 42);

        let num_stages = setup_a.fcf.pools.len();
        // terminal stage has no backward-pass cuts
        let boundary_date = ascending_stage_end_dates(num_stages)[num_stages - 2];

        let (mut setup_b, _system_b) = build_setup(&case_dir, &config_5iter);
        let mut solver_b = ActiveSolver::new().expect("solver");
        let outcome_b = setup_b
            .train(&mut solver_b, &comm, 1, ActiveSolver::new, None, None)
            .expect("train B");
        assert!(outcome_b.error.is_none());
        let lb_no_boundary = outcome_b.result.final_lb;

        let (mut setup_c, system_c) = build_setup(&case_dir, &config_5iter);
        let state_dim = setup_c.fcf.state_dimension as u32;
        let current_manifest = setup_c.build_terminal_entity_manifest(&system_c);
        let boundary_records =
            cobre_sddp::load_boundary_cuts(&cobre_sddp::BoundaryLoadRequest::new(
                &source_policy_dir,
                boundary_date,
                state_dim,
                &current_manifest,
                1_000_000.0,
            ))
            .expect("load_boundary_cuts");
        assert!(
            !boundary_records.is_empty(),
            "source stage must have cuts after training"
        );
        cobre_sddp::inject_boundary_cuts(&mut setup_c, &boundary_records).unwrap();

        let terminal_pool_id = setup_c.inputs.node_graph.terminal_pool(num_stages).unwrap();
        let terminal_pool = &setup_c.fcf.pools[terminal_pool_id];
        assert!(
            terminal_pool.has_warm_start_cuts(),
            "terminal pool must have boundary cuts"
        );
        assert!(
            terminal_pool.populated() >= terminal_pool.warm_start_count as usize,
            "populated_count must include boundary cuts"
        );

        let mut solver_c = ActiveSolver::new().expect("solver");
        let outcome_c = setup_c
            .train(&mut solver_c, &comm, 1, ActiveSolver::new, None, None)
            .expect("train C");
        assert!(outcome_c.error.is_none());
        let lb_with_boundary = outcome_c.result.final_lb;

        assert!(
            lb_with_boundary >= lb_no_boundary - 1e-6,
            "boundary LB ({lb_with_boundary}) must be >= baseline LB ({lb_no_boundary})"
        );
    }

    /// Every pool [`write_test_checkpoint`] writes carries a non-sentinel
    /// `priced_state_date`, and no two pools share one.
    #[test]
    fn write_test_checkpoint_pools_carry_distinct_priced_dates() {
        let case_dir = d01_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let mut config_5iter = config;
        config_5iter.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 5 }]);

        let (mut setup, _system) = build_setup(&case_dir, &config_5iter);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("solver");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train");
        assert!(outcome.error.is_none());

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let policy_dir = tmpdir.path().join("policy");
        write_test_checkpoint(&policy_dir, &setup, &outcome.result, 42);

        let checkpoint = read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint");
        let dates: Vec<i32> = checkpoint
            .stage_cuts
            .iter()
            .map(|sr| sr.priced_state_date)
            .collect();
        assert!(
            dates
                .iter()
                .all(|&d| d != STAGE_CUTS_PRICED_STATE_DATE_SENTINEL)
        );
        let mut sorted = dates.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), dates.len());
    }
}

mod cut_subgradient_parity {
    //! KKT parity test: `reduced_costs[storage_in.start]` matches the hand-computed
    //! closed-form value derived from row duals of the same LP solve.
    //!
    //! ## What is being verified
    //!
    //! Cut-subgradient extraction reads the reduced cost of the incoming-state column
    //! (`view.reduced_costs[state_to_lp_incoming_column(j)]`), not the dual of a
    //! state-fixing equality row.
    //!
    //! By KKT optimality, the reduced cost of a column with `lb == ub` equals the
    //! dual of the equivalent equality row — but the same identity holds for the
    //! multi-row case via the full KKT stationarity condition:
    //!
    //! ```text
    //! rc[j] = c_j - sum_i( a_{i,j} * y_i )
    //! ```
    //!
    //! where `c_j` is the column objective coefficient, `a_{i,j}` are the
    //! constraint-matrix coefficients, and `y_i` are the row dual variables.
    //!
    //! For the `storage_in[0]` column (incoming storage, `c_j = 0`):
    //! - Water-balance row: coefficient `-1.0`
    //! - FPHA row:          coefficient `-gamma_v / 2`
    //! - Evaporation row:   coefficient `-volume_slope_m3s_per_hm3 / 2`
    //!
    //! Therefore: `rc[storage_in[0]] = y_wb + (gamma_v/2)*y_fpha + (volume_slope_m3s_per_hm3/2)*y_evap`
    //!
    //! ## System layout
    //!
    //! - N=1 FPHA hydro, L=0 (no lags), A=0 (no anticipated thermals)
    //! - 1 bus, 1 block (K=1)
    //! - No thermal plant — deficit variable absorbs any load-generation gap
    //! - FPHA: 1 plane with `gamma_v=0.002, gamma_q=0.8, gamma_0=0.0`
    //! - Evaporation: `volume_slope_m3s_per_hm3=0.01, intercept_m3s=0.0`
    //!
    //! ## Column and row indices (storage_fixing = 0..0)
    //!
    //! Column layout for N=1, L=0, A=0, K=1 (no penalty, no thermal):
    //! ```text
    //! col 0: storage (v_out)
    //! col 1: z_inflow
    //! col 2: storage_in (v_in) ← pinned via set_col_bounds; reduced cost tested here
    //! col 3: theta
    //! col 4: turbine
    //! col 5: spillage
    //! col 6: diversion
    //! col 7: deficit
    //! col 8: excess
    //! col 9: withdrawal_slack_neg
    //! col 10: withdrawal_slack_pos
    //! cols 11-14: 4 × operational violation slacks
    //! col 15: evaporation outflow
    //! col 16: f_evap_plus
    //! col 17: f_evap_minus
    //! col 18: g_fpha (FPHA generation variable)
    //! ```
    //!
    //! Row layout (storage_fixing = 0..0, z_inflow at row 0):
    //! ```text
    //! row 0: z_inflow definition (z = RHS)
    //! row 1: water_balance       ← dual = y_wb
    //! row 2: load_balance
    //! row 3: FPHA plane 0        ← dual = y_fpha
    //! row 4: evaporation equality ← dual = y_evap
    //! rows 5-8: operational violation constraints
    //! ```

    use chrono::NaiveDate;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds, LineBlockBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
        ThermalStageBounds,
        entities::hydro::HydroGenerationModel,
        scenario::{InflowModel, LoadModel},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        },
    };
    use cobre_sddp::{
        build_stage_templates_resolving_layout,
        hydro_models::{
            EvaporationModel, EvaporationModelSet, FphaPlane, LinearizedEvaporation,
            PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel,
        },
        inflow_method::InflowNonNegativityMethod,
        resolved_parameters::ResolvedParameters,
    };
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};
    use cobre_stochastic::{PrecomputedPar, normal::precompute::PrecomputedNormal};

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };

    /// FPHA plane coefficients for the test: modest gamma_v so the FPHA constraint
    /// can be active with reasonable storage values.
    const GAMMA_V: f64 = 0.002;
    const GAMMA_Q: f64 = 0.8;
    const GAMMA_0: f64 = 0.0;

    const VOLUME_SLOPE_M3S_PER_HM3: f64 = 0.01;
    const INTERCEPT_M3S: f64 = 0.0;

    const V_IN_HM3: f64 = 100.0;

    /// Indices into the column/row layout documented in the module doc; the
    /// closed-form KKT reference and the post-solve assertions both depend on these.
    const COL_STORAGE_IN: usize = 2;
    const ROW_WATER_BALANCE: usize = 1;
    const ROW_FPHA: usize = 3;
    const ROW_EVAP: usize = 4;

    /// Build a minimal 1-FPHA-hydro, 1-bus system with linearized evaporation.
    ///
    /// Parameters chosen so the LP is feasible and the FPHA constraint is active
    /// at the optimum (maximising turbine use to minimise thermal/deficit cost).
    fn fpha_evap_system() -> cobre_core::System {
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

        let hydro = make_hydro(
            EntityId(2),
            HydroSpec {
                name: "H_FPHA".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::Fpha,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

        let stages = vec![make_stage(
            0,
            StageSpec {
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 730.0,
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
        )];

        let inflow_models = vec![InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 0.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }];

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 50.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 50.0,
                    max_generation_mw: 100.0,
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
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
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
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("fpha_evap_system: valid")
    }

    fn fpha_production() -> ProductionModelSet {
        let plane = FphaPlane {
            intercept: GAMMA_0,
            gamma_v: GAMMA_V,
            gamma_q: GAMMA_Q,
            gamma_s: 0.0,
        };
        let models = vec![vec![ResolvedProductionModel::Fpha {
            planes: vec![plane],
        }]];
        ProductionModelSet::new(models, &cobre_sddp::test_support::minimal_hydros(1), 1)
    }

    fn fpha_evap_evaporation() -> EvaporationModelSet {
        let models = vec![EvaporationModel::Linearized {
            coefficients: vec![LinearizedEvaporation {
                intercept_m3s: INTERCEPT_M3S,
                volume_slope_m3s_per_hm3: VOLUME_SLOPE_M3S_PER_HM3,
            }],
            reference_volumes_hm3: vec![100.0],
        }];
        EvaporationModelSet::new(models)
    }

    /// KKT parity: `reduced_costs[storage_in.start]` equals the closed-form
    /// reference `y_wb + (gamma_v/2)*y_fpha + (volume_slope_m3s_per_hm3/2)*y_evap`
    /// from the same LP solve.
    ///
    /// This confirms that `state_to_lp_incoming_column` returns the correct column
    /// for cut-subgradient extraction in the presence of both FPHA and evaporation
    /// contributions to the incoming-storage column's KKT stationarity condition.
    #[test]
    fn cut_subgradient_parity_with_fpha_and_evaporation() {
        let system = fpha_evap_system();
        let production = fpha_production();
        let evaporation = fpha_evap_evaporation();

        let result = build_stage_templates_resolving_layout(
            &system,
            InflowNonNegativityMethod::None,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evaporation,
            &ResolvedParameters::default(),
        )
        .expect("FPHA+evap template must build");

        let template = &result.templates[0];

        // Guards the hardcoded COL_STORAGE_IN: storage_in.start = N*(2+L) + A*K = 2.
        assert_eq!(
            template.n_state, 1,
            "N=1, L=0: n_state must be 1 (storage only, no lags)"
        );

        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        solver.load_model(template);

        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        // Pin storage_in[0] via column bounds (lb == ub) — NOT a state-fixing row.
        solver.set_col_bounds(&[COL_STORAGE_IN], &[V_IN_HM3], &[V_IN_HM3]);

        let view = solver
            .solve(None)
            .expect("FPHA+evap LP must be feasible and optimal");

        let rc_storage_in = view.reduced_costs[COL_STORAGE_IN];
        let y_wb = view.dual[ROW_WATER_BALANCE];
        let y_fpha = view.dual[ROW_FPHA];
        let y_evap = view.dual[ROW_EVAP];
        let obj = view.objective;

        // Closed-form KKT reference.
        //
        // storage_in[0] participates in:
        //   water-balance row: coefficient  -1.0
        //   FPHA row:          coefficient  -gamma_v / 2
        //   evap row:          coefficient  -volume_slope_m3s_per_hm3 / 2
        //
        // KKT stationarity: rc[j] = c_j - sum_i( a_{i,j} * y_i )
        //   = 0 - ( (-1.0)*y_wb + (-GAMMA_V/2)*y_fpha + (-VOLUME_SLOPE_M3S_PER_HM3/2)*y_evap )
        //   = y_wb + (GAMMA_V/2)*y_fpha + (VOLUME_SLOPE_M3S_PER_HM3/2)*y_evap
        let kkt_ref = y_wb + (GAMMA_V / 2.0) * y_fpha + (VOLUME_SLOPE_M3S_PER_HM3 / 2.0) * y_evap;

        assert!(
            obj.is_finite(),
            "LP objective must be finite (LP must be feasible and bounded)"
        );

        assert!(
            (rc_storage_in - kkt_ref).abs() < 1e-8,
            "KKT parity failed:\n  \
             rc[storage_in[0]]  = {rc_storage_in:.12}\n  \
             KKT reference      = {kkt_ref:.12}\n  \
             |delta|            = {:.2e}\n  \
             y_wb               = {y_wb:.12}\n  \
             y_fpha             = {y_fpha:.12}\n  \
             y_evap             = {y_evap:.12}\n  \
             gamma_v/2          = {:.6}\n  \
             volume_slope_m3s_per_hm3/2 = {:.6}",
            (rc_storage_in - kkt_ref).abs(),
            GAMMA_V / 2.0,
            VOLUME_SLOPE_M3S_PER_HM3 / 2.0,
        );
    }

    /// Structural guard: `PrepareHydroModelsResult::default_from_system` produces
    /// a model set that does NOT include FPHA for a system declared with
    /// `HydroGenerationModel::ConstantProductivity`. This confirms that the FPHA
    /// plane override in `fpha_production()` is required and meaningful.
    #[test]
    fn default_production_is_not_fpha_for_this_system() {
        let system = fpha_evap_system();
        let default_prod = PrepareHydroModelsResult::default_from_system(&system);
        assert!(
            matches!(
                default_prod.production.model(0, 0),
                ResolvedProductionModel::ConstantProductivity { .. }
            ),
            "default_from_system must return ConstantProductivity as fallback"
        );
    }
}

mod cut_selection_determinism_realistic {
    //! `CutSelectionStrategy::select` produces bit-identical `CutActivityUpdates`
    //! across thread counts. On hosts with fewer logical CPUs than a requested pool
    //! size, rayon caps the worker count; this is acceptable — the assertions
    //! compare bitmaps across whatever thread counts rayon actually instantiates.

    use cobre_sddp::cut::CutPool;
    use cobre_sddp::cut_selection::{CutActivityUpdates, CutMetadata, CutSelectionStrategy};
    use cobre_sddp::setup::NodeId;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill_f64(buf: &mut [f64], seed: u64) {
        let mut state = seed;
        for slot in buf.iter_mut() {
            let r = splitmix64(&mut state);
            let bits = (r >> 12) & ((1u64 << 52) - 1);
            *slot = f64::from_bits((1023u64 << 52) | bits) - 1.5;
        }
    }

    fn make_pool(k: usize, d: usize, seed: u64) -> CutPool {
        let mut pool = CutPool::new(k, d, 1, 0);
        let mut state = seed;
        for slot in 0..k {
            let r = splitmix64(&mut state);
            let bits = (r >> 12) & ((1u64 << 52) - 1);
            let intercept = f64::from_bits((1023u64 << 52) | bits) - 1.5;
            let mut coeffs = vec![0.0_f64; d];
            fill_f64(&mut coeffs, state.wrapping_add(slot as u64));
            pool.add_cut(NodeId(0), 0, slot as u32, intercept, &coeffs);
        }
        // Force all cuts eligible: iteration_generated < the current_iteration (5)
        // the tests below pass to select.
        let metadata: Vec<CutMetadata> = (0..k)
            .map(|slot| {
                let mut meta = pool.metadata(slot).clone();
                meta.iteration_generated = 1;
                meta
            })
            .collect();
        let active: Vec<bool> = (0..k).map(|slot| pool.is_active(slot)).collect();
        pool.replace_selection(&metadata, &active);
        pool
    }

    fn make_states(m: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut buf = vec![0.0_f64; m * d];
        fill_f64(&mut buf, seed);
        buf
    }

    fn run_in_pool<F>(threads: usize, f: F) -> CutActivityUpdates
    where
        F: FnOnce() -> CutActivityUpdates + Send,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon thread pool must build");
        pool.install(f)
    }

    #[test]
    fn select_for_stage_deterministic_aggregated_scale() {
        const K: usize = 945;
        const D: usize = 155;
        const M: usize = 384;

        let pool = make_pool(K, D, 0xAAAA_BBBB_CCCC_DDDD);
        let states = make_states(M, D, 0x1111_2222_3333_4444);

        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };

        let r1 = run_in_pool(1, || strategy.select(&pool, &states, 5));
        let r16 = run_in_pool(16, || strategy.select(&pool, &states, 5));
        let r96 = run_in_pool(96, || strategy.select(&pool, &states, 5));

        assert_eq!(
            r1.updates, r16.updates,
            "deactivations differ between 1- and 16-thread runs at aggregated scale"
        );
        assert_eq!(
            r1.updates, r96.updates,
            "deactivations differ between 1- and 96-thread runs at aggregated scale"
        );
        assert_eq!(
            r1.reactivations, r16.reactivations,
            "reactivations differ between 1- and 16-thread runs at aggregated scale"
        );
        assert_eq!(
            r1.reactivations, r96.reactivations,
            "reactivations differ between 1- and 96-thread runs at aggregated scale"
        );
    }

    /// Exercises Level1 and Dominated to confirm determinism is not Lml1-specific;
    /// sized to run in <100 ms so it stays in the default (non-slow) suite.
    #[test]
    fn select_for_stage_deterministic_level1_and_dominated_medium() {
        const K: usize = 200;
        const D: usize = 155;
        const M: usize = 64;

        let pool = make_pool(K, D, 0x0001_0002_0003_0004);
        let states = make_states(M, D, 0x0005_0006_0007_0008);

        for strategy in [
            CutSelectionStrategy::Level1 {
                check_frequency: 1,
                tie_tolerance: 1e-10,
            },
            CutSelectionStrategy::Dominated {
                threshold: 0.01,
                check_frequency: 1,
            },
        ] {
            let r1 = run_in_pool(1, || strategy.select(&pool, &states, 5));
            let r8 = run_in_pool(8, || strategy.select(&pool, &states, 5));
            assert_eq!(
                r1.updates, r8.updates,
                "deactivations differ for strategy {strategy:?}"
            );
            assert_eq!(
                r1.reactivations, r8.reactivations,
                "reactivations differ for strategy {strategy:?}"
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn select_for_stage_deterministic_disaggregated_scale() {
        // TODO(measured-K-probe): replace K=1000 with the measured disaggregated K.
        const K: usize = 1000;
        const D: usize = 2080;
        const M: usize = 384;

        let pool = make_pool(K, D, 0xFEED_FACE_CAFE_BABE);
        let states = make_states(M, D, 0xDEAD_BEEF_BAAD_F00D);

        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };

        let r1 = run_in_pool(1, || strategy.select(&pool, &states, 5));
        let r16 = run_in_pool(16, || strategy.select(&pool, &states, 5));
        let r96 = run_in_pool(96, || strategy.select(&pool, &states, 5));

        assert_eq!(
            r1.updates, r16.updates,
            "deactivations differ between 1- and 16-thread runs at disaggregated scale"
        );
        assert_eq!(
            r1.updates, r96.updates,
            "deactivations differ between 1- and 96-thread runs at disaggregated scale"
        );
        assert_eq!(
            r1.reactivations, r16.reactivations,
            "reactivations differ between 1- and 16-thread runs at disaggregated scale"
        );
        assert_eq!(
            r1.reactivations, r96.reactivations,
            "reactivations differ between 1- and 96-thread runs at disaggregated scale"
        );
    }
}

mod basis_reconstruct_churn {
    //! Cross-churn regression tests for [`cobre_sddp::basis_reconstruct`]: guards
    //! slot reconciliation, `state_at_capture` routing, and the iteration-wide
    //! hot-path allocation invariants against LML1 deactivation, budget eviction,
    //! and new-cut churn.

    use cobre_io::config::{SimulationSelection, TrainingSelection};
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::{SelectionMethod, StoppingRuleConfig};
    use cobre_sddp::{
        CutPool, FutureCostFunction, SolverStatsDelta, SolverStatsLogEntry, StudySetup,
        hydro_models::prepare_hydro_models, setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    // ---------------------------------------------------------------------------
    // Path helpers
    // ---------------------------------------------------------------------------

    fn d03_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<crate> must have a parent")
            .parent()
            .expect("crates/ must have a parent (repo root)")
            .join("examples/deterministic/d03-two-hydro-cascade")
    }

    fn d01_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<crate> must have a parent")
            .parent()
            .expect("crates/ must have a parent (repo root)")
            .join("examples/deterministic/d01-thermal-dispatch")
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn sum_forward_deltas(log: &[SolverStatsLogEntry]) -> SolverStatsDelta {
        SolverStatsDelta::aggregate(
            log.iter()
                .filter(|e| e.phase == "forward")
                .map(|e| &e.delta),
        )
    }

    // ---------------------------------------------------------------------------
    // Test 1: Cross-churn — LML1 + budget + new cuts
    // ---------------------------------------------------------------------------

    /// All three churn types active at once; the simplex-iteration pin catches a
    /// `padding_state = x_hat` regression (it raises the count beyond the ±5 % band).
    #[test]
    fn basis_reconstruct_churn() {
        // Simplex pin and lower-bound pin are calibrated against HiGHS's simplex path
        // and last-ULP accumulation, so they gate to the `highs` backend; the
        // structural checks below run on both. Regenerate the pin only if the HiGHS
        // major version or the fixture parameters change.
        #[cfg(feature = "highs")]
        const PINNED_SIMPLEX_ITERS: u64 = 30;
        #[cfg(feature = "highs")]
        const LO_SIMPLEX: u64 = PINNED_SIMPLEX_ITERS * 95 / 100;
        #[cfg(feature = "highs")]
        const HI_SIMPLEX: u64 = PINNED_SIMPLEX_ITERS * 105 / 100;

        #[cfg(feature = "highs")]
        const PINNED_FINAL_LB: f64 = 1_391_697.766_666_666_8;

        let case_dir = d03_case_dir();
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 3 });
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 8 }]);

        config.training.cut_selection.selection = Some(SelectionMethod::Lml1 {
            tie_tolerance: 1e-10,
            check_frequency: 1,
        });

        // Budget = 6 = 2 iters × 3 fwd-passes, so eviction first fires after iteration 2.
        config.training.cut_selection.max_active_per_stage = Some(6);

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result = prepare_stochastic(
            system,
            &case_dir,
            &config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup must build");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "basis_reconstruct_churn: expected no training error, got: {:?}",
            outcome.error
        );
        let result = outcome.result;
        assert_eq!(
            result.iterations, 8,
            "basis_reconstruct_churn: expected 8 iterations, got {}",
            result.iterations
        );

        let fwd = sum_forward_deltas(&result.solver_stats_log);

        // basis_reconstructions is excluded from forward-log MPI packing, so it is
        // always 0 in these entries; basis_consistency_failures == 0 is the proxy
        // for "reconstruction is active and producing valid bases".
        assert_eq!(
            fwd.basis_consistency_failures, 0,
            "basis_reconstruct_churn: expected 0 basis rejections, got {} \
             (reconstructed bases must always be accepted by HiGHS)",
            fwd.basis_consistency_failures
        );

        #[cfg(feature = "highs")]
        {
            let observed_simplex = fwd.simplex_iterations;
            assert!(
                (LO_SIMPLEX..=HI_SIMPLEX).contains(&observed_simplex),
                "basis_reconstruct_churn: simplex_iterations={observed_simplex} is outside \
                 the ±5 % band [{LO_SIMPLEX}, {HI_SIMPLEX}] around pin={PINNED_SIMPLEX_ITERS}"
            );
        }

        // final_lb is deterministic for a fixed seed and scenario count; its last ULP
        // shifts with the HiGHS build, so pin to a relative tolerance that catches
        // real drift (~9 significant figures) while absorbing float noise.
        #[cfg(feature = "highs")]
        {
            let lb_rel_err = (result.final_lb - PINNED_FINAL_LB).abs() / PINNED_FINAL_LB.abs();
            assert!(
                lb_rel_err <= 1e-9,
                "basis_reconstruct_churn: final_lb={} deviates from pin={PINNED_FINAL_LB:.15e} \
                 by relative error {lb_rel_err:.3e} (> 1e-9); the lower bound must be \
                 deterministic for a fixed seed and scenario count",
                result.final_lb
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Test 2: No-churn — happy path, reconstruction active across iterations
    // ---------------------------------------------------------------------------

    /// No-churn happy path: cuts only accumulate. `basis_consistency_failures == 0`
    /// confirms the always-frozen reconstruction stays active across iterations.
    #[test]
    fn test_basis_reconstruct_no_churn_full_preservation() {
        let case_dir = d03_case_dir();
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 2 });
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]);

        config.training.cut_selection.selection = None;
        config.training.cut_selection.max_active_per_stage = None;

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result = prepare_stochastic(
            system,
            &case_dir,
            &config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup must build");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "test_basis_reconstruct_no_churn_full_preservation: unexpected training error: {:?}",
            outcome.error
        );
        let result = outcome.result;
        assert_eq!(
            result.iterations, 3,
            "no_churn: expected 3 iterations, got {}",
            result.iterations
        );

        let fwd = sum_forward_deltas(&result.solver_stats_log);

        // basis_reconstructions is excluded from forward-log MPI packing (always 0
        // here); basis_consistency_failures == 0 is the proxy for "reconstruction is
        // active and producing valid bases".
        assert_eq!(
            fwd.basis_consistency_failures, 0,
            "no_churn: expected 0 basis rejections, got {}",
            fwd.basis_consistency_failures
        );

        assert!(
            result.final_lb.is_finite() && result.final_lb > 0.0,
            "no_churn: expected finite positive lower bound, got {}",
            result.final_lb
        );
    }

    // ---------------------------------------------------------------------------
    // Test 3: Full churn — all iteration-1 cuts deactivated before iteration 2
    // ---------------------------------------------------------------------------

    /// Full-churn edge case: every iteration-1 cut deactivated before the
    /// iteration-2 forward pass, so the reconstruction path meets an empty FCF.
    /// Phase 2 rebuilds a fresh cold-start `setup2` with no stored basis, so
    /// `basis_reconstructions == 0` is expected and the classification assertion is
    /// dropped on purpose; the safety goal is that an empty FCF neither panics nor
    /// produces an invalid basis (`basis_consistency_failures == 0`).
    #[test]
    fn test_basis_reconstruct_full_churn_no_rows_preserved() {
        let case_dir = d01_case_dir();
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

        // Two IterationLimit rules in "any" mode: limit 2 sizes the FCF pool to
        // (2+1) × forward_passes slots/stage (capacity headroom); limit 1 fires first
        // and stops after iteration 1.
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 2 });
        config.training.stopping_rules = Some(vec![
            StoppingRuleConfig::IterationLimit { limit: 2 },
            StoppingRuleConfig::IterationLimit { limit: 1 },
        ]);
        let system = cobre_io::load_case(&case_dir).expect("load_case phase1 must succeed");
        let prepare_result = prepare_stochastic(
            system,
            &case_dir,
            &config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic phase1 must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup phase1 must build");

        let comm = StubComm;
        let mut solver1 = ActiveSolver::new().expect("ActiveSolver phase1 must succeed");

        let outcome1 = setup
            .train(&mut solver1, &comm, 1, ActiveSolver::new, None, None)
            .expect("phase1 train must return Ok");
        assert!(
            outcome1.error.is_none(),
            "full_churn: phase 1 unexpected error: {:?}",
            outcome1.error
        );
        assert_eq!(
            outcome1.result.iterations, 1,
            "full_churn: phase 1 must complete exactly 1 iteration"
        );

        let cuts_after_iter1: Vec<usize> =
            setup.fcf.pools.iter().map(CutPool::active_count).collect();
        let total_cuts_iter1: usize = cuts_after_iter1.iter().sum();
        assert!(
            total_cuts_iter1 > 0,
            "full_churn: phase 1 must generate at least 1 cut, got 0; \
             test cannot exercise the deactivation path otherwise"
        );

        {
            let fcf = &mut setup.fcf;
            for (stage, pool) in fcf.pools.iter_mut().enumerate() {
                let active_indices: Vec<u32> = (0..pool.populated())
                    .filter(|&i| pool.is_active(i))
                    .map(|i| i as u32)
                    .collect();
                if !active_indices.is_empty() {
                    pool.deactivate(&active_indices);
                }
                assert_eq!(
                    pool.active_count(),
                    0,
                    "full_churn: stage {stage} pool must have 0 active cuts after deactivation, \
                     got {}",
                    pool.active_count()
                );
            }
        }

        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 2 }]);

        setup.set_resume_point(1, Vec::new());

        let mut solver2 = ActiveSolver::new().expect("ActiveSolver phase2 must succeed");

        // The config is locked into setup at construction, so phase 2 cannot reuse the
        // phase-1 setup with the new stopping rule: rebuild a fresh setup from the same
        // system and transplant the fully-deactivated FCF into it.
        {
            let system2 = cobre_io::load_case(&case_dir).expect("load_case phase2 must succeed");
            let prepare2 = prepare_stochastic(
                system2,
                &case_dir,
                &config,
                42,
                &ScenarioSource::default(),
                None,
            )
            .expect("prepare_stochastic phase2 must succeed");
            let system2 = prepare2.system;
            let stochastic2 = prepare2.stochastic;
            let hydro2 = prepare_hydro_models(&system2, &case_dir, false)
                .expect("prepare_hydro_models phase2");

            let mut setup2 = StudySetup::new(&system2, &config, stochastic2, hydro2, Vec::new())
                .expect("StudySetup phase2");

            // Read the placeholder-FCF metadata before the mutable borrow below
            // (borrow checker).
            let n_stages = setup.fcf.pools.len();
            let state_dim = setup.fcf.state_dimension;
            let fwd_passes = setup.loop_params.forward_passes;
            let max_iters = setup.loop_params.max_iterations;
            let placeholder_fcf = FutureCostFunction::new(
                n_stages,
                state_dim,
                fwd_passes,
                max_iters,
                &vec![0u32; n_stages],
            );
            let deactivated_fcf = std::mem::replace(&mut setup.fcf, placeholder_fcf);
            setup2.replace_fcf(deactivated_fcf);
            setup2.set_resume_point(1, Vec::new());

            let outcome2 = setup2
                .train(&mut solver2, &comm, 1, ActiveSolver::new, None, None)
                .expect("phase2 train must return Ok");
            assert!(
                outcome2.error.is_none(),
                "full_churn: phase 2 unexpected error: {:?}",
                outcome2.error
            );
            assert_eq!(
                outcome2.result.iterations, 2,
                "full_churn: phase 2 must report 2 total iterations, got {}",
                outcome2.result.iterations
            );

            let result2 = outcome2.result;

            let iter2_fwd: Vec<&SolverStatsDelta> = result2
                .solver_stats_log
                .iter()
                .filter(|e| e.iteration == 2 && e.phase == "forward")
                .map(|e| &e.delta)
                .collect();

            assert!(
                !iter2_fwd.is_empty(),
                "full_churn: must have at least one forward log entry for iteration 2"
            );

            let iter2_fwd_agg = SolverStatsDelta::aggregate(iter2_fwd.into_iter());

            assert_eq!(
                iter2_fwd_agg.basis_consistency_failures, 0,
                "full_churn: iteration 2 forward must have 0 basis rejections, got {} \
                 (empty FCF must not cause HiGHS to reject the warm-start basis)",
                iter2_fwd_agg.basis_consistency_failures
            );

            assert!(
                result2.final_lb.is_finite(),
                "full_churn: final_lb must be finite after full-churn iteration, got {}",
                result2.final_lb
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Test 4: Simulation smoke — frozen-path simulate completes with zero failures
    // ---------------------------------------------------------------------------

    /// Smoke test: after 2-iteration training on D03, frozen-path simulation with the
    /// trained `basis_cache` reconstructs a basis per stage/scenario with zero
    /// `basis_consistency_failures` (no reconstructed basis is rejected by `HiGHS`).
    #[test]
    fn simulate_frozen_path_zero_consistency_failures() {
        let case_dir = d03_case_dir();
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

        // 2 iterations so iter 2's forward pass captures a basis with non-empty
        // cut_row_slots into basis_cache for the simulation warm-start.
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 2 });
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 2 }]);

        config.training.cut_selection.selection = None;
        config.training.cut_selection.max_active_per_stage = None;

        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 2 });

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result = prepare_stochastic(
            system,
            &case_dir,
            &config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup must build");

        let comm = StubComm;
        let mut train_solver = ActiveSolver::new().expect("ActiveSolver for training must succeed");

        let outcome = setup
            .train(&mut train_solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "simulate_warm_start: expected no training error, got: {:?}",
            outcome.error
        );
        assert_eq!(
            outcome.result.iterations, 2,
            "simulate_warm_start: expected 2 iterations, got {}",
            outcome.result.iterations
        );

        let frozen_templates = outcome.result.frozen_templates.as_deref().expect(
            "simulate_warm_start: training must produce frozen_templates after >= 2 iterations",
        );
        let basis_cache = &outcome.result.basis_cache;
        assert!(
            basis_cache
                .iter()
                .any(|cb| cb.as_ref().is_some_and(|b| !b.cut_row_slots.is_empty())),
            "simulate_warm_start: at least one stage must have a CapturedBasis with \
             non-empty cut_row_slots in basis_cache after 2 iterations"
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("simulation workspace pool must build");

        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

        // Drain results on a background thread to avoid channel backpressure.
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().count());

        let sim_result = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                Some(frozen_templates),
                basis_cache,
            )
            .expect("simulate must return Ok");

        drop(result_tx);
        drain_handle.join().expect("drain thread must not panic");

        let total_rejections: u64 = sim_result
            .solver_stats
            .iter()
            .map(|(_, _, delta)| delta.basis_consistency_failures)
            .sum();

        assert_eq!(
            total_rejections, 0,
            "simulate_warm_start: expected 0 basis_consistency_failures in frozen-path simulation, \
             got {total_rejections} (reconstructed bases must always be accepted by HiGHS)"
        );
    }
}

mod hybrid_reconstruction {
    //! Integration smoke for the slot-identity basis-reconstruction path.

    use cobre_sddp::basis_reconstruct::{
        ReconstructionStats, ReconstructionTarget, reconstruct_basis,
    };
    use cobre_sddp::setup::NodeId;
    use cobre_sddp::workspace::CapturedBasis;
    use cobre_solver::Basis;
    use cobre_solver::BasisStatus::{Basic as B, Lower as L};

    /// All columns are `BASIC` so the column block does not perturb the test focus.
    fn make_stored(
        base_rows: usize,
        num_cols: usize,
        slots: &[u32],
        cut_statuses: &[cobre_solver::BasisStatus],
        state_at_capture: &[f64],
    ) -> CapturedBasis {
        assert_eq!(slots.len(), cut_statuses.len());
        let total_rows = base_rows + cut_statuses.len();
        let mut cb = CapturedBasis::new(
            num_cols,
            total_rows,
            base_rows,
            slots.len(),
            state_at_capture.len(),
            NodeId(0),
        );
        cb.basis.row_status.clear();
        cb.basis.row_status.resize(base_rows, B);
        cb.basis.row_status.extend_from_slice(cut_statuses);
        cb.basis.col_status.clear();
        cb.basis.col_status.resize(num_cols, B);
        cb.cut_row_slots.extend_from_slice(slots);
        cb.state_at_capture.extend_from_slice(state_at_capture);
        cb
    }

    #[test]
    fn mixed_case_5_rows() {
        let stored = make_stored(2, 3, &[10, 20, 30, 40], &[L, B, L, B], &[1.0, 2.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0, 0.0]),
            (25, 0.0, vec![0.0, 0.0]),
            (30, 0.0, vec![0.0, 0.0]),
            (45, 0.0, vec![0.0, 0.0]),
            (50, 0.0, vec![0.0, 0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 2,
            num_cols: 3,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_slack: 3,
            },
            "preserved={{10, 30}} (2), new_slack={{25, 45, 50}} (3)",
        );
        assert_eq!(out.row_status.len(), 7, "2 template + 5 cut rows");
        assert_eq!(&out.row_status[..2], &[B, B], "template rows preserved");
        assert_eq!(out.row_status[2], L, "slot 10 → stored LOWER");
        assert_eq!(out.row_status[3], B, "slot 25 → new → BASIC");
        assert_eq!(out.row_status[4], L, "slot 30 → stored LOWER");
        assert_eq!(out.row_status[5], B, "slot 45 → new → BASIC");
        assert_eq!(out.row_status[6], B, "slot 50 → new → BASIC");
    }

    #[test]
    fn all_preserved() {
        let stored = make_stored(1, 2, &[10, 20, 30], &[L, B, L], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0]),
            (20, 0.0, vec![0.0]),
            (30, 0.0, vec![0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 3,
                new_slack: 0,
            },
        );
        assert_eq!(&out.row_status[1..], &[L, B, L]);
    }

    #[test]
    fn all_new() {
        let stored = make_stored(1, 2, &[], &[], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (5, 0.0, vec![0.0]),
            (6, 0.0, vec![0.0]),
            (7, 0.0, vec![0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 0,
                new_slack: 3,
            },
        );
        assert_eq!(&out.row_status[1..], &[B, B, B]);
    }
}

mod warm_start {
    //! Warm-start training: a run resumed from saved cuts reuses them
    //! (`warm_start_count > 0`) and yields a lower bound no worse than fresh training.

    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::StoppingRuleConfig;
    use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
    use cobre_sddp::{
        FutureCostFunction, SolverStatsDelta, StoredBasisMisfit, StudySetup,
        build_basis_cache_from_checkpoint, hydro_models::prepare_hydro_models,
        setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    fn d01_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d01-thermal-dispatch")
    }

    /// Ascending, distinct `end_date` per pool (`2030-01-01` plus the pool
    /// index in months) — the study-stage calendar [`write_test_checkpoint`]
    /// feeds [`build_stage_cuts_payloads`] to self-describe each pool's
    /// `priced_state_date`.
    fn ascending_stage_end_dates(n_pools: usize) -> Vec<chrono::NaiveDate> {
        let base = chrono::NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        (0..n_pools as u32)
            .map(|i| base.checked_add_months(chrono::Months::new(i)).unwrap())
            .collect()
    }

    fn write_test_checkpoint(
        policy_dir: &Path,
        setup: &StudySetup,
        result: &cobre_sddp::TrainingResult,
        seed: u64,
    ) {
        use cobre_sddp::policy_export::{
            build_active_indices, build_stage_basis_records, build_stage_cut_records,
            build_stage_cuts_payloads, convert_basis_cache,
        };
        let fcf = &setup.fcf;
        let stage_records = build_stage_cut_records(fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![Vec::new(); fcf.pools.len()];
        let study_stage_ids: Vec<i32> = (0..fcf.pools.len() as i32).collect();
        let study_stage_end_dates = ascending_stage_end_dates(fcf.pools.len());
        let stage_cuts = build_stage_cuts_payloads(
            fcf,
            &setup.inputs.node_graph,
            &study_stage_ids,
            &study_stage_end_dates,
            1_000_000.0,
            &stage_records,
            &stage_active_indices,
            &stage_manifests,
        );
        let (basis_col, basis_row) = convert_basis_cache(&result.basis_cache);
        let stage_bases = build_stage_basis_records(
            &result.basis_cache,
            result.iterations,
            &basis_col,
            &basis_row,
        );
        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
        let metadata = cobre_sddp::test_support::checkpoint_metadata(
            fcf.pools.len() as u32,
            setup.build_graph_manifest(),
            cobre_io::ProducerBlock {
                completed_iterations: result.iterations as u32,
                final_lower_bound: result.final_lb,
                best_upper_bound: Some(result.final_ub),
                max_iterations: setup.loop_params.max_iterations as u32,
                forward_passes: setup.loop_params.forward_passes,
                warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
                warm_start_counts,
                rng_seed: seed,
                total_visited_states: 0,
                training_block_mode: "parallel".to_string(),
                training_block_mode_per_stage: vec![],
                cost_scale_factor: None,
                lower_bound_history: Vec::new(),
            },
        );
        write_policy_checkpoint(policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write checkpoint");
    }

    fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> StudySetup {
        let system = cobre_io::load_case(case_dir).expect("load_case");
        let prep = prepare_stochastic(
            system,
            case_dir,
            config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic");
        let hydro_models =
            prepare_hydro_models(&prep.system, case_dir, false).expect("prepare_hydro_models");
        StudySetup::new(
            &prep.system,
            config,
            prep.stochastic,
            hydro_models,
            Vec::new(),
        )
        .expect("StudySetup::new")
    }

    #[test]
    fn resume_training_from_checkpoint() {
        let case_dir = d01_case_dir();
        let config_path = case_dir.join("config.json");
        let config_full = cobre_io::parse_config(&config_path).expect("config must parse");

        let mut config_phase1 = config_full.clone();
        config_phase1.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 5 }]);

        let mut setup_phase1 = build_setup(&case_dir, &config_phase1);
        let comm = StubComm;
        let mut solver_phase1 = ActiveSolver::new().expect("ActiveSolver");
        let outcome_phase1 = setup_phase1
            .train(&mut solver_phase1, &comm, 1, ActiveSolver::new, None, None)
            .expect("train phase1");
        assert!(outcome_phase1.error.is_none());
        let result_phase1 = outcome_phase1.result;
        assert_eq!(
            result_phase1.iterations, 5,
            "phase 1 must complete exactly 5 iterations"
        );
        let lb_phase1 = result_phase1.final_lb;

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let policy_dir = tmpdir.path().join("policy");
        write_test_checkpoint(&policy_dir, &setup_phase1, &result_phase1, 42);

        let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");
        let mut setup_phase2 = build_setup(&case_dir, &config_full);

        let proof = cobre_sddp::test_support::trivial_full_fcf_proof(
            checkpoint.stage_cuts[0].state_dimension,
            checkpoint.metadata.num_stages,
        );
        let pool_state_dimensions: Vec<usize> = setup_phase2
            .fcf
            .pools
            .iter()
            .map(|p| p.state_dimension)
            .collect();
        let visit_bounds: Vec<u64> = setup_phase2
            .fcf
            .pools
            .iter()
            .map(|p| u64::from(p.visit_stride))
            .collect();
        let warm_fcf = FutureCostFunction::new_with_warm_start(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
            &visit_bounds,
            setup_phase2.loop_params.forward_passes,
            setup_phase2.loop_params.max_iterations.saturating_add(1),
        )
        .expect("warm-start FCF");
        setup_phase2.replace_fcf(warm_fcf);
        setup_phase2.set_resume_point(
            u64::from(checkpoint.metadata.producer.completed_iterations),
            checkpoint.metadata.producer.lower_bound_history.clone(),
        );

        let mut solver_phase2 = ActiveSolver::new().expect("ActiveSolver");
        let outcome_phase2 = setup_phase2
            .train(&mut solver_phase2, &comm, 1, ActiveSolver::new, None, None)
            .expect("train phase2");
        assert!(outcome_phase2.error.is_none());
        let result_phase2 = outcome_phase2.result;

        assert_eq!(
            result_phase2.iterations, 10,
            "resumed run must report 10 total iterations (not 5 delta)"
        );

        assert!(
            result_phase2.final_lb >= lb_phase1 - 1e-6,
            "resumed LB ({}) must be >= phase-1 LB ({})",
            result_phase2.final_lb,
            lb_phase1
        );
    }

    #[test]
    fn exported_basis_records_count_the_cut_rows_each_basis_was_captured_with() {
        use cobre_sddp::policy_export::{build_stage_basis_records, convert_basis_cache};

        let case_dir = d01_case_dir();
        let mut config =
            cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 5 }]);

        let mut setup = build_setup(&case_dir, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train");
        assert!(outcome.error.is_none());
        let result = outcome.result;

        let (basis_col, basis_row) = convert_basis_cache(&result.basis_cache);
        let records = build_stage_basis_records(
            &result.basis_cache,
            result.iterations,
            &basis_col,
            &basis_row,
        );
        assert!(
            !records.is_empty(),
            "training must capture at least one basis"
        );

        for rec in &records {
            let cb = result.basis_cache[rec.stage_id as usize]
                .as_ref()
                .expect("every record comes from a captured basis");
            assert_eq!(
                rec.num_cut_rows as usize,
                cb.basis.row_status.len() - cb.base_row_count,
                "node {} must record the cut rows its basis was captured with",
                rec.stage_id
            );
        }

        let root = records
            .iter()
            .find(|rec| rec.stage_id == 0)
            .expect("the root node's basis is exported");
        let root_pool_populated = setup.fcf.pools[0].populated();
        assert!(
            (root.num_cut_rows as usize) < root_pool_populated,
            "the root basis is captured before the last backward pass, so it must carry fewer \
             cut rows ({}) than its pool holds at export ({root_pool_populated})",
            root.num_cut_rows
        );
    }

    #[test]
    fn warm_start_training_preserves_cuts_and_trains_further() {
        let case_dir = d01_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");

        let mut setup_fresh = build_setup(&case_dir, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver");
        let fresh_outcome = setup_fresh
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train");
        assert!(fresh_outcome.error.is_none());
        let fresh_result = fresh_outcome.result;
        let fresh_lb = fresh_result.final_lb;
        let fresh_active = setup_fresh.fcf.total_active_cuts();

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let policy_dir = tmpdir.path().join("policy");
        write_test_checkpoint(&policy_dir, &setup_fresh, &fresh_result, 42);

        let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");
        let mut setup_warm = build_setup(&case_dir, &config);

        // Capacity must match from_broadcast_params's saturating_add(1) or the slot
        // layout diverges.
        let proof = cobre_sddp::test_support::trivial_full_fcf_proof(
            checkpoint.stage_cuts[0].state_dimension,
            checkpoint.metadata.num_stages,
        );
        let pool_state_dimensions: Vec<usize> = setup_warm
            .fcf
            .pools
            .iter()
            .map(|p| p.state_dimension)
            .collect();
        let visit_bounds: Vec<u64> = setup_warm
            .fcf
            .pools
            .iter()
            .map(|p| u64::from(p.visit_stride))
            .collect();
        let warm_fcf = FutureCostFunction::new_with_warm_start(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
            &visit_bounds,
            setup_warm.loop_params.forward_passes,
            setup_warm.loop_params.max_iterations.saturating_add(1),
        )
        .expect("warm-start FCF");
        setup_warm.replace_fcf(warm_fcf);

        let warm_start_count = setup_warm.fcf.pools[0].warm_start_count;
        assert!(warm_start_count > 0, "warm_start_count should be > 0");
        assert_eq!(
            setup_warm.fcf.total_active_cuts(),
            fresh_active,
            "warm-start FCF should have same active cuts as fresh training"
        );

        let mut solver_warm = ActiveSolver::new().expect("ActiveSolver");
        let warm_outcome = setup_warm
            .train(&mut solver_warm, &comm, 1, ActiveSolver::new, None, None)
            .expect("warm-start train");
        assert!(warm_outcome.error.is_none());
        let warm_result = warm_outcome.result;

        assert!(
            warm_result.final_lb >= fresh_lb - 1e-6,
            "warm-start LB ({}) should be >= fresh LB ({})",
            warm_result.final_lb,
            fresh_lb
        );

        let total_active_after = setup_warm.fcf.total_active_cuts();
        assert!(
            total_active_after > fresh_active,
            "warm-start training should produce more total cuts"
        );
    }

    #[test]
    fn warm_start_skips_a_stored_basis_with_too_few_basic_entries_instead_of_aborting() {
        const MISFIT_NODE: u32 = 1;
        const MISFIT_STAGE: i32 = 1;
        const OTHER_STAGE: i32 = 0;

        let case_dir = d01_case_dir();
        let config =
            cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");

        let mut setup_fresh = build_setup(&case_dir, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver");
        let fresh_outcome = setup_fresh
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train");
        assert!(fresh_outcome.error.is_none());

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let policy_dir = tmpdir.path().join("policy");
        write_test_checkpoint(&policy_dir, &setup_fresh, &fresh_outcome.result, 42);

        let mut checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");
        assert!(
            checkpoint.stage_bases.len() >= 2,
            "the checkpoint must store a basis for the root and for the misfit node"
        );
        let misfit = checkpoint
            .stage_bases
            .iter_mut()
            .find(|record| record.stage_id == MISFIT_NODE)
            .expect("the misfit node's basis is stored");
        misfit.column_status.fill(0);
        misfit.row_status.fill(0);

        let mut setup_warm = build_setup(&case_dir, &config);
        let proof = cobre_sddp::test_support::trivial_full_fcf_proof(
            checkpoint.stage_cuts[0].state_dimension,
            checkpoint.metadata.num_stages,
        );
        let pool_state_dimensions: Vec<usize> = setup_warm
            .fcf
            .pools
            .iter()
            .map(|p| p.state_dimension)
            .collect();
        let visit_bounds: Vec<u64> = setup_warm
            .fcf
            .pools
            .iter()
            .map(|p| u64::from(p.visit_stride))
            .collect();
        let warm_fcf = FutureCostFunction::new_with_warm_start(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
            &visit_bounds,
            setup_warm.loop_params.forward_passes,
            setup_warm.loop_params.max_iterations.saturating_add(1),
        )
        .expect("warm-start FCF");
        setup_warm.replace_fcf(warm_fcf);

        let load = build_basis_cache_from_checkpoint(
            &checkpoint.stage_bases,
            &checkpoint.stage_cuts,
            &setup_warm,
        );
        let unused = load.unused.expect("the misfit record must be reported");
        assert_eq!(unused.count, 1);
        assert!(
            matches!(unused.first_reason, StoredBasisMisfit::BasicCount { .. }),
            "{:?}",
            unused.first_reason
        );
        setup_warm.set_warm_start_basis_cache(load.cache);

        let mut solver_warm = ActiveSolver::new().expect("ActiveSolver");
        let warm_outcome = setup_warm
            .train(&mut solver_warm, &comm, 1, ActiveSolver::new, None, None)
            .expect("warm-start train");
        assert!(warm_outcome.error.is_none(), "{:?}", warm_outcome.error);

        let first_iteration_forward_at = |stage_id: i32| {
            SolverStatsDelta::aggregate(
                warm_outcome
                    .result
                    .solver_stats_log
                    .iter()
                    .filter(|e| {
                        e.phase == "forward" && e.iteration == 1 && e.stage_id == Some(stage_id)
                    })
                    .map(|e| &e.delta),
            )
        };
        let at_misfit = first_iteration_forward_at(MISFIT_STAGE);
        let at_other = first_iteration_forward_at(OTHER_STAGE);
        assert_eq!(
            at_misfit.basis_offered, 0,
            "the dropped node must cold-start in iteration 1"
        );
        assert!(
            at_other.basis_offered > 0,
            "the admitted node must warm-start in iteration 1"
        );
        assert_eq!(
            at_misfit.basis_consistency_failures + at_other.basis_consistency_failures,
            0,
            "no basis offered to the solver may be rejected"
        );
    }
}

mod test_backward_cache_reduces_pivots {
    //! Backward-basis-cache sanity test.
    //!
    //! Verifies that the backward-basis cache plumbing (capture → broadcast →
    //! swap → read) is operational end-to-end on D03 and does not cause
    //! catastrophic regressions.
    //!
    //! ## Why this is a sanity test, not a performance test
    //!
    //! At ω=0 the forward basis has an exact state match (same iter, same trial
    //! point `x_hat`) and so outperforms the backward cache's state-drift warm-start;
    //! on D03 the cache adds roughly +4× pivots at ω=0 iter ≥ 2. The cache's net win
    //! comes from a secondary effect at scale: its richer ω=0 pivoting leaves the
    //! `HiGHS` retained LU in a state that accelerates the subsequent ω ≥ 1 chain,
    //! and with ω ≥ 1 LPs ≈ 19× the ω=0 count the amortization produces a net
    //! backward wall-time win.
    //!
    //! D03 has too few cuts for that ω ≥ 1 amortization, so any strict "pivots reduce
    //! on iter ≥ 2" assertion on D03 is empirically wrong and would fail on working
    //! code. This test instead verifies a loose property: backward ω=0 pivot counts
    //! stay within a generous bound, catching only catastrophic failures (basis
    //! rejected every iteration, capture path never fires, wire-format corruption).
    //! The real performance metric is total backward wall time, measured separately.
    //!
    //! The 20× bound on the baseline mean sits well above the observed ~4× state-drift
    //! regression while still catching order-of-magnitude blowups.

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    // ---------------------------------------------------------------------------
    // Baseline constant
    //
    // Default D03 config (10 iterations, 1 forward pass, no cut selection):
    //   backward ω=0 (opening == 0) iter ≥ 2 rows: 18, sum(simplex_iterations): 2,
    //   mean: 2.0 / 18.0 ≈ 0.1111.
    // Set to 0.112 — just above the measured mean — for a tight but noise-tolerant
    // threshold; with the cache the mean drops to 0.0, so the assertion has margin.
    // ---------------------------------------------------------------------------

    const D03_PRE_PLAN_BWD_OMEGA0_MEAN_PIVOTS: f64 = 0.112;

    // ---------------------------------------------------------------------------
    // Path helpers
    // ---------------------------------------------------------------------------

    fn d03_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<crate> must have a parent")
            .parent()
            .expect("crates/ must have a parent (repo root)")
            .join("examples/deterministic/d03-two-hydro-cascade")
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Asserts that the backward-basis cache keeps iter-≥2 ω=0 backward-pass simplex
    /// iterations on D03 within the sanity bound (see module docs for why this is a
    /// loose bound, not a strict reduction). From iteration 2 the ω=0 backward solve
    /// warm-starts from the cached rank-0 m=0 basis rather than the forward basis.
    ///
    /// On failure, the assertion message lists the triage steps; the key read site is
    /// `process_by_scenario_backward`, which must prefer `succ.backward_store[stage]`
    /// over `succ.basis_store.get(m, s)` when the stored backward basis is `Some`.
    #[test]
    fn test_backward_cache_reduces_pivots() {
        let case_dir = d03_case_dir();
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

        // Clear cut-selection overrides so the default no-selection path the baseline
        // was measured on is exercised.
        config.training.cut_selection.selection = None;
        config.training.cut_selection.max_active_per_stage = None;

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result = prepare_stochastic(
            system,
            &case_dir,
            &config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup must build");

        let comm = StubComm;
        let (tx, _rx) = mpsc::channel();
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, Some(tx), None)
            .expect("train must succeed");

        assert!(
            outcome.error.is_none(),
            "test_backward_cache_reduces_pivots: unexpected training error: {:?}",
            outcome.error
        );

        let result = outcome.result;

        assert_eq!(
            result.iterations, 10,
            "test_backward_cache_reduces_pivots: expected 10 iterations (D03 default), got {}",
            result.iterations
        );

        // The iter≥2 ω=0 rows are the ones whose warm-start quality the cache affects.
        let bwd_omega0_ge2: Vec<u64> = result
            .solver_stats_log
            .iter()
            .filter(|e| e.phase == "backward" && e.opening == Some(0) && e.iteration >= 2)
            .map(|e| e.delta.simplex_iterations)
            .collect();

        assert!(
            !bwd_omega0_ge2.is_empty(),
            "test_backward_cache_reduces_pivots: no backward ω=0 iter≥2 entries found in \
             solver_stats_log — check that the backward pass emits per-opening entries"
        );

        let n = bwd_omega0_ge2.len();
        let sum: u64 = bwd_omega0_ge2.iter().sum();
        // Cast is lossless for u64 < 2^53.
        #[allow(clippy::cast_precision_loss)]
        let observed_mean = sum as f64 / n as f64;

        let upper_bound = D03_PRE_PLAN_BWD_OMEGA0_MEAN_PIVOTS * 20.0;
        assert!(
            observed_mean < upper_bound,
            "test_backward_cache_reduces_pivots: observed backward ω=0 iter≥2 mean \
             simplex iterations ({observed_mean:.6}) exceeds the sanity bound \
             ({upper_bound:.6} = 20× pre-plan baseline {D03_PRE_PLAN_BWD_OMEGA0_MEAN_PIVOTS:.6}).\n\
             n={n}, sum={sum}\n\
             This indicates a catastrophic failure in the backward-basis cache \
             pipeline.\n\
             Triage: (1) confirm backward log entries have opening==0; \
             (2) verify stored backward basis is Some from iter 2; \
             (3) check that the backward-pass read site correctly constructs stored_basis; \
             (4) verify the backward-pass basis cache is updated at end of each iteration; \
             (5) inspect basis_consistency_failures counter in solver_iterations.parquet."
        );
    }
}

mod group_state_invariance {
    //! Declaring `unit_groups`, or splitting a plant across buses, must change
    //! only the per-cell turbine/FPHA-generation column count. The recourse
    //! state (one storage coordinate per reservoir), the cut coefficients
    //! keyed off it, and the terminal entity manifest must stay identical.
    //! Compared here as two live runs (no committed golden): the real
    //! two-bus `d51-split-plant-two-bus` deck, an in-code single-mirror-group
    //! equivalent of the same plant, and an in-code same-bus multi-group
    //! variant.

    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_core::{EntityId, HydroUnitGroup, System, SystemBuilder};
    use cobre_sddp::indexer::{HydroCellIndex, HydroSys};
    use cobre_sddp::policy_export::build_stage_cut_records;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::{StubComm, build_setup_for_case};

    fn d51_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<crate> must have a parent")
            .parent()
            .expect("crates/ must have a parent (repo root)")
            .join("examples/deterministic/d51-split-plant-two-bus")
    }

    /// Rebuilds `system` with hydro-position-0's `unit_groups` replaced by
    /// `groups`, cloning every other resolved table verbatim. Safe because the
    /// group-bound overlay addresses `(hydro_idx, group POSITION, stage,
    /// block)`, never a group's own `id` -- changing which or how many groups
    /// occupy that position only shifts the per-cell column bound the LP
    /// resolves (deliberately not asserted in this module), never the state
    /// layout, which depends only on hydro count and per-stage state config.
    fn rebuild_with_hydro0_groups(system: &System, groups: Vec<HydroUnitGroup>) -> System {
        let mut hydros = system.hydros().to_vec();
        hydros[0].unit_groups = groups;
        SystemBuilder::new()
            .buses(system.buses().to_vec())
            .lines(system.lines().to_vec())
            .hydros(hydros)
            .thermals(system.thermals().to_vec())
            .pumping_stations(system.pumping_stations().to_vec())
            .contracts(system.contracts().to_vec())
            .non_controllable_sources(system.non_controllable_sources().to_vec())
            .stages(system.stages().to_vec())
            .policy_graph(system.policy_graph().clone())
            .penalties(system.penalties().clone())
            .bounds(system.bounds().clone())
            .resolved_generic_bounds(system.resolved_generic_bounds().clone())
            .resolved_load_factors(system.resolved_load_factors().clone())
            .resolved_ncs_bounds(system.resolved_ncs_bounds().clone())
            .resolved_ncs_factors(system.resolved_ncs_factors().clone())
            .inflow_models(system.inflow_models().to_vec())
            .load_models(system.load_models().to_vec())
            .ncs_models(system.ncs_models().to_vec())
            .correlation(system.correlation().clone())
            .initial_conditions(system.initial_conditions().clone())
            .generic_constraints(system.generic_constraints().to_vec())
            .inflow_history(system.inflow_history().to_vec())
            .external_scenarios(system.external_scenarios().to_vec())
            .external_load_scenarios(system.external_load_scenarios().to_vec())
            .external_ncs_scenarios(system.external_ncs_scenarios().to_vec())
            .build()
            .expect("mutated d51 system must still build")
    }

    /// One group mirroring H0's own declared envelope onto `bus_id`: every
    /// hydro must declare at least one unit group, so this is the
    /// single-group equivalent of a groupless deck.
    fn single_mirror_group(system: &System, bus_id: i32) -> Vec<HydroUnitGroup> {
        let mut h0 = system.hydros()[0].clone();
        h0.unit_groups.clear();
        h0.declare_mirror_unit_group(EntityId(bus_id));
        h0.unit_groups
    }

    /// H0's own two declared groups, unmodified except both now share one
    /// bus -- the same-bus collapse of two groups into one cell.
    fn same_bus_groups(system: &System) -> Vec<HydroUnitGroup> {
        let mut groups = system.hydros()[0].unit_groups.clone();
        assert_eq!(
            groups.len(),
            2,
            "d51's H0 must declare exactly 2 groups before the same-bus collapse"
        );
        let shared_bus = groups[0].bus_id;
        groups[1].bus_id = shared_bus;
        groups
    }

    /// Number of bus-partitioned cells H0 (hydro position 0) resolves into.
    fn n_cells_h0(system: &System) -> usize {
        HydroCellIndex::build(system.hydros())
            .cells_of(HydroSys::new(0))
            .len()
    }

    /// Trains a short run over `system`, returning the trained setup alongside
    /// the stochastic-prepared system it was built against — needed for the
    /// terminal-manifest and cell-count reads.
    fn train_short_run(
        system: System,
        config: &cobre_io::Config,
        case_dir: &Path,
    ) -> (StudySetup, System) {
        let prepare_result = prepare_stochastic(
            system,
            case_dir,
            config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let hydro_models = prepare_hydro_models(&system, case_dir, false)
            .expect("prepare_hydro_models must succeed");
        let mut setup = build_setup_for_case(
            case_dir,
            config,
            &system,
            prepare_result.stochastic,
            hydro_models,
        );

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must succeed");
        assert!(
            outcome.error.is_none(),
            "training must not error: {:?}",
            outcome.error
        );
        (setup, system)
    }

    /// State/cut/manifest-derived quantities this module's invariance claim
    /// covers. Deliberately excludes column counts: those are cell-sized and
    /// grow with cells, the additive change this module must not flatten into
    /// an equality.
    struct StateCutManifestSnapshot {
        state_dimension: usize,
        pool_state_dimensions: Vec<usize>,
        populated_cut_coefficient_lengths: Vec<Option<usize>>,
        terminal_slot_keys: Vec<(u8, i32, u32)>,
    }

    fn snapshot(setup: &StudySetup, system: &System) -> StateCutManifestSnapshot {
        let pool_state_dimensions: Vec<usize> = setup
            .fcf
            .pools
            .iter()
            .map(|pool| pool.state_dimension)
            .collect();

        let stage_records = build_stage_cut_records(&setup.fcf);
        let populated_cut_coefficient_lengths: Vec<Option<usize>> = stage_records
            .iter()
            .map(|records| records.first().map(|record| record.coefficients.len()))
            .collect();

        let terminal_manifest = setup.build_terminal_entity_manifest(system);
        let terminal_slot_keys: Vec<(u8, i32, u32)> = terminal_manifest
            .iter()
            .map(|slot| (slot.entity_type, slot.entity_id, slot.subindex))
            .collect();

        StateCutManifestSnapshot {
            state_dimension: setup.fcf.state_dimension,
            pool_state_dimensions,
            populated_cut_coefficient_lengths,
            terminal_slot_keys,
        }
    }

    fn assert_state_cut_manifest_identical(
        a: &StateCutManifestSnapshot,
        b: &StateCutManifestSnapshot,
        label_a: &str,
        label_b: &str,
    ) {
        assert_eq!(
            a.state_dimension, b.state_dimension,
            "{label_a} vs {label_b}: state dimension must be identical"
        );
        assert_eq!(
            a.pool_state_dimensions, b.pool_state_dimensions,
            "{label_a} vs {label_b}: per-stage cut-pool state dimension must be identical"
        );
        assert_eq!(
            a.populated_cut_coefficient_lengths, b.populated_cut_coefficient_lengths,
            "{label_a} vs {label_b}: per-stage cut-record coefficient-vector length must be identical"
        );
        assert_eq!(
            a.terminal_slot_keys, b.terminal_slot_keys,
            "{label_a} vs {label_b}: terminal manifest entity/subindex slot keys must be identical"
        );
    }

    /// A two-bus, two-group, two-cell split and its single-mirror-group,
    /// one-cell equivalent train to identical state dimension, cut-record
    /// shape, and terminal manifest -- even though the cell count, and with
    /// it the turbine/FPHA-generation column count, differs.
    #[test]
    fn two_bus_split_matches_single_mirror_group_state_cut_manifest() {
        let case_dir = d51_case_dir();
        let config =
            cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");

        let two_bus_system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let mirror_groups = single_mirror_group(&two_bus_system, 0);
        let mirror_system = rebuild_with_hydro0_groups(&two_bus_system, mirror_groups);

        let (setup_two_bus, system_two_bus) = train_short_run(two_bus_system, &config, &case_dir);
        let (setup_mirror, system_mirror) = train_short_run(mirror_system, &config, &case_dir);

        let cells_two_bus = n_cells_h0(&system_two_bus);
        let cells_mirror = n_cells_h0(&system_mirror);
        assert_eq!(
            cells_two_bus, 2,
            "the real d51 fixture must partition H0 into 2 cells"
        );
        assert_eq!(
            cells_mirror, 1,
            "the single-mirror-group equivalent must partition H0 into 1 cell"
        );
        assert_ne!(
            cells_two_bus, cells_mirror,
            "the cell count, and with it the turbine/FPHA-generation column \
             count, must differ between the two-bus split and its \
             single-mirror-group equivalent, or this test stops exercising \
             the additive change it is meant to pin"
        );

        let snapshot_two_bus = snapshot(&setup_two_bus, &system_two_bus);
        let snapshot_mirror = snapshot(&setup_mirror, &system_mirror);
        assert_state_cut_manifest_identical(
            &snapshot_two_bus,
            &snapshot_mirror,
            "two-bus split",
            "single-mirror-group equivalent",
        );
    }

    /// A same-bus, two-group, one-cell collapse and its single-mirror-group,
    /// one-cell equivalent train to identical state dimension, cut-record
    /// shape, and terminal manifest: the collapse changes the resolved
    /// per-cell bound (fold-then-sum of two groups versus one group's own
    /// box), never the cell count or the state.
    #[test]
    fn same_bus_collapse_matches_single_mirror_group_state_cut_manifest() {
        let case_dir = d51_case_dir();
        let config =
            cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");

        let base_system = cobre_io::load_case(&case_dir).expect("load_case must succeed");

        let same_bus_groups_h0 = same_bus_groups(&base_system);
        let same_bus_system = rebuild_with_hydro0_groups(&base_system, same_bus_groups_h0);
        let (setup_same_bus, system_same_bus) =
            train_short_run(same_bus_system, &config, &case_dir);

        let mirror_groups = single_mirror_group(&base_system, 0);
        let mirror_system = rebuild_with_hydro0_groups(&base_system, mirror_groups);
        let (setup_mirror, system_mirror) = train_short_run(mirror_system, &config, &case_dir);

        assert_eq!(
            n_cells_h0(&system_same_bus),
            1,
            "two groups sharing one bus must still collapse to 1 cell"
        );
        assert_eq!(
            n_cells_h0(&system_mirror),
            1,
            "the single-mirror-group equivalent must partition H0 into 1 cell"
        );

        let snapshot_same_bus = snapshot(&setup_same_bus, &system_same_bus);
        let snapshot_mirror = snapshot(&setup_mirror, &system_mirror);
        assert_state_cut_manifest_identical(
            &snapshot_same_bus,
            &snapshot_mirror,
            "same-bus multi-group collapse",
            "single-mirror-group equivalent",
        );
    }
}

mod range_warm_start_determinism {
    //! Empirical evidence for the range-row basis-reconstruction safety argument
    //! (the highest-risk coverage gap): `BasisStatus::Upper` on a real,
    //! solved range row surviving `reconstruct_basis`, warm-start/cold objective
    //! agreement, run-to-run reproducibility, `generic_constraints.json`
    //! declaration-order invariance, and range-vs-two-row solution equivalence.
    //!
    //! Every fixture pins the range row's upper bound below the cheapest
    //! generator's unconstrained dispatch and below its own column bound, so the
    //! LP is economically forced to rest exactly at the row's upper bound — a
    //! degenerate coincident-bound row (`bound_lower == bound_upper`) could never
    //! demonstrate this economic forcing, since it would pin there trivially.

    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;

    use cobre_core::scenario::{LoadModel, ScenarioSource};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ConstraintExpression,
        ContractBlockBounds, DeficitSegment, EntityId, GenericConstraint, HydroBlockBounds,
        HydroPenalties, HydroStageBounds, LineBlockBounds, LineStagePenalties, LinearTerm,
        NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds,
        ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties, SlackConfig,
        SystemBuilder, ThermalBlockBounds, ThermalStageBounds, VariableRef,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, ParallelismConfig,
        PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
        SimulationSelection, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
        TrainingSolverConfig,
    };
    use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
    use cobre_sddp::basis_reconstruct::{
        ReconstructionTarget, enforce_basic_count_invariant, reconstruct_basis,
    };
    use cobre_sddp::hydro_models::{PrepareHydroModelsResult, prepare_hydro_models};
    use cobre_sddp::inflow_method::InflowNonNegativityMethod;
    use cobre_sddp::policy_export::{
        build_active_indices, build_stage_basis_records, build_stage_cut_records,
        build_stage_cuts_payloads, convert_basis_cache,
    };
    use cobre_sddp::resolved_parameters::ResolvedParameters;
    use cobre_sddp::setup::prepare_stochastic;
    use cobre_sddp::{
        FutureCostFunction, SolverStatsDelta, StudySetup, build_stage_templates_resolving_layout,
    };
    use cobre_solver::{ActiveSolver, Basis, BasisStatus};

    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };
    use super::common::{StubComm, build_setup_in_code, run_simulation};

    /// `(constraint_id, stage_id, block_id, bound_lower, bound_upper)` — the
    /// raw-row shape [`ResolvedGenericConstraintBounds::new`] consumes.
    type RawBoundRow = (i32, i32, Option<i32>, Option<f64>, Option<f64>);

    // ---------------------------------------------------------------------------
    // Shared fixture identities
    // ---------------------------------------------------------------------------

    fn range_bus_id() -> EntityId {
        EntityId(0)
    }

    fn range_thermal_id() -> EntityId {
        EntityId(0)
    }

    fn thermal_generation_expr() -> ConstraintExpression {
        ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: range_thermal_id(),
                    block_id: None,
                },
            )],
        }
    }

    fn no_slack() -> SlackConfig {
        SlackConfig {
            enabled: false,
            penalty: None,
        }
    }

    // ---------------------------------------------------------------------------
    // 3-constraint fixture: a range row (binds at upper) plus two loose siblings
    // ---------------------------------------------------------------------------

    /// `cap_loose (<= 25)`, `band (range [3, 8])`, `floor_trivial (>= 0)` — all on
    /// `thermal_generation(0)`. T0's cost (50/MWh) is far below the bus deficit
    /// cost (1000/MWh) and demand (20 MW) exceeds the band's upper bound, so the
    /// optimum pushes T0 to exactly 8 MW: `cap_loose` and `floor_trivial` stay
    /// slack (`Basic`), only `band`'s row rests at its upper bound.
    fn build_range_constraints() -> Vec<GenericConstraint> {
        vec![
            GenericConstraint {
                id: EntityId(1),
                name: "cap_loose".to_string(),
                description: None,
                expression: thermal_generation_expr(),
                slack: no_slack(),
                bound_lower_affine: None,
                bound_upper_affine: None,
            },
            GenericConstraint {
                id: EntityId(2),
                name: "band".to_string(),
                description: None,
                expression: thermal_generation_expr(),
                slack: no_slack(),
                bound_lower_affine: None,
                bound_upper_affine: None,
            },
            GenericConstraint {
                id: EntityId(3),
                name: "floor_trivial".to_string(),
                description: None,
                expression: thermal_generation_expr(),
                slack: no_slack(),
                bound_lower_affine: None,
                bound_upper_affine: None,
            },
        ]
    }

    fn build_range_bounds() -> ResolvedGenericConstraintBounds {
        let id_map: HashMap<i32, usize> = [(1, 0), (2, 1), (3, 2)].into_iter().collect();
        let rows: Vec<RawBoundRow> = vec![
            (1, 0, None, None, Some(25.0)),
            (1, 1, None, None, Some(25.0)),
            (2, 0, None, Some(3.0), Some(8.0)),
            (2, 1, None, Some(3.0), Some(8.0)),
            (3, 0, None, Some(0.0), None),
            (3, 1, None, Some(0.0), None),
        ];
        ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter())
    }

    // ---------------------------------------------------------------------------
    // Band-only fixture (range vs. two-row equivalence)
    // ---------------------------------------------------------------------------

    /// The same `[3, 8]` band on `thermal_generation(0)`, expressed either as one
    /// `range` constraint or as a `>= 3` plus a `<= 8` constraint.
    fn build_band_constraints(
        as_range: bool,
    ) -> (Vec<GenericConstraint>, ResolvedGenericConstraintBounds) {
        if as_range {
            let constraints = vec![GenericConstraint {
                id: EntityId(1),
                name: "band".to_string(),
                description: None,
                expression: thermal_generation_expr(),
                slack: no_slack(),
                bound_lower_affine: None,
                bound_upper_affine: None,
            }];
            let id_map: HashMap<i32, usize> = [(1, 0)].into_iter().collect();
            let rows: Vec<RawBoundRow> = vec![
                (1, 0, None, Some(3.0), Some(8.0)),
                (1, 1, None, Some(3.0), Some(8.0)),
            ];
            (
                constraints,
                ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter()),
            )
        } else {
            let constraints = vec![
                GenericConstraint {
                    id: EntityId(1),
                    name: "band_lower".to_string(),
                    description: None,
                    expression: thermal_generation_expr(),
                    slack: no_slack(),
                    bound_lower_affine: None,
                    bound_upper_affine: None,
                },
                GenericConstraint {
                    id: EntityId(2),
                    name: "band_upper".to_string(),
                    description: None,
                    expression: thermal_generation_expr(),
                    slack: no_slack(),
                    bound_lower_affine: None,
                    bound_upper_affine: None,
                },
            ];
            let id_map: HashMap<i32, usize> = [(1, 0), (2, 1)].into_iter().collect();
            let rows: Vec<RawBoundRow> = vec![
                (1, 0, None, Some(3.0), None),
                (1, 1, None, Some(3.0), None),
                (2, 0, None, None, Some(8.0)),
                (2, 1, None, None, Some(8.0)),
            ];
            (
                constraints,
                ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter()),
            )
        }
    }

    // ---------------------------------------------------------------------------
    // System / config builders (in-code, no case directory)
    // ---------------------------------------------------------------------------

    /// A 2-stage, 1-bus, 1-thermal, 0-hydro system: bus deficit cost 1000/MWh,
    /// thermal T0 [0, 30] MW at 50/MWh, demand 20 MW (`std = 0`) both stages.
    /// Economics mirror `examples/deterministic/d13-generic-constraint`.
    fn build_system(
        constraints: Vec<GenericConstraint>,
        bounds: ResolvedGenericConstraintBounds,
    ) -> cobre_core::System {
        let bus = make_bus(
            range_bus_id(),
            BusSpec {
                name: "B0".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2020, 1, 1).expect("valid date"),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal = make_thermal(
            range_thermal_id(),
            ThermalSpec {
                name: "T0".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2020, 1, 1).expect("valid date"),
                bus_id: range_bus_id(),
                min_generation_mw: 0.0,
                max_generation_mw: 30.0,
                cost_per_mwh: 50.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let stage_dates = [
            (
                NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
                NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
            ),
            (
                NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
                NaiveDate::from_ymd_opt(2024, 3, 1).expect("valid date"),
            ),
        ];
        let stages: Vec<cobre_core::temporal::Stage> = stage_dates
            .iter()
            .enumerate()
            .map(|(i, (start, end))| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: *start,
                        end_date: *end,
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "SINGLE".to_string(),
                            duration_hours: 730.0,
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
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..2)
            .map(|i| LoadModel {
                bus_id: range_bus_id(),
                stage_id: i,
                mean_mw: 20.0,
                std_mw: 0.0,
            })
            .collect();

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 0.0,
                    max_generation_mw: 0.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 30.0,
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

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .generic_constraints(constraints)
            .resolved_generic_bounds(bounds)
            .build()
            .expect("build_system: valid")
    }

    fn build_config(limit: u32) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::None,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: cobre_io::config::UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// The row index in stage 0's structural template whose `(row_lower,
    /// row_upper)` is exactly `(3.0, 8.0)` — the range row's unique bound pair
    /// (no other row in the fixture shares it). Located structurally rather than
    /// by a hand-derived offset so the test stays correct if the LP layout shifts.
    fn find_range_row_stage0(system: &cobre_core::System) -> usize {
        let prepared = PrepareHydroModelsResult::default_from_system(system);
        let templates = build_stage_templates_resolving_layout(
            system,
            InflowNonNegativityMethod::None,
            &cobre_stochastic::par::precompute::PrecomputedPar::default(),
            &cobre_stochastic::normal::precompute::PrecomputedNormal::default(),
            &prepared.production,
            &prepared.evaporation,
            &ResolvedParameters::default(),
        )
        .expect("build_stage_templates_resolving_layout: valid")
        .templates;

        templates[0]
            .row_lower
            .iter()
            .zip(templates[0].row_upper.iter())
            .position(|(&lo, &hi)| lo == 3.0 && hi == 8.0)
            .expect("the range row [3, 8] must be present in stage 0's structural template")
    }

    /// Ascending, distinct `end_date` per pool (`2030-01-01` plus the pool
    /// index in months) — the study-stage calendar [`write_test_checkpoint`]
    /// feeds [`build_stage_cuts_payloads`] to self-describe each pool's
    /// `priced_state_date`.
    fn ascending_stage_end_dates(n_pools: usize) -> Vec<chrono::NaiveDate> {
        let base = chrono::NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        (0..n_pools as u32)
            .map(|i| base.checked_add_months(chrono::Months::new(i)).unwrap())
            .collect()
    }

    /// Local mirror of the `warm_start` mod's checkpoint writer (module-scoped
    /// helpers are not shared across `mod` blocks in this file).
    fn write_test_checkpoint(
        policy_dir: &Path,
        setup: &StudySetup,
        result: &cobre_sddp::TrainingResult,
        seed: u64,
    ) {
        let fcf = &setup.fcf;
        let stage_records = build_stage_cut_records(fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![Vec::new(); fcf.pools.len()];
        let study_stage_ids: Vec<i32> = (0..fcf.pools.len() as i32).collect();
        let study_stage_end_dates = ascending_stage_end_dates(fcf.pools.len());
        let stage_cuts = build_stage_cuts_payloads(
            fcf,
            &setup.inputs.node_graph,
            &study_stage_ids,
            &study_stage_end_dates,
            1_000_000.0,
            &stage_records,
            &stage_active_indices,
            &stage_manifests,
        );
        let (basis_col, basis_row) = convert_basis_cache(&result.basis_cache);
        let stage_bases = build_stage_basis_records(
            &result.basis_cache,
            result.iterations,
            &basis_col,
            &basis_row,
        );
        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
        let metadata = cobre_sddp::test_support::checkpoint_metadata(
            fcf.pools.len() as u32,
            setup.build_graph_manifest(),
            cobre_io::ProducerBlock {
                completed_iterations: result.iterations as u32,
                final_lower_bound: result.final_lb,
                best_upper_bound: Some(result.final_ub),
                max_iterations: setup.loop_params.max_iterations as u32,
                forward_passes: setup.loop_params.forward_passes,
                warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
                warm_start_counts,
                rng_seed: seed,
                total_visited_states: 0,
                training_block_mode: "parallel".to_string(),
                training_block_mode_per_stage: vec![],
                cost_scale_factor: None,
                lower_bound_history: Vec::new(),
            },
        );
        write_policy_checkpoint(policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write checkpoint");
    }

    // ---------------------------------------------------------------------------
    // BasisStatus::Upper round-trips through reconstruct_basis
    // ---------------------------------------------------------------------------

    /// A range row whose optimum rests on its upper bound
    /// captures `BasisStatus::Upper`, and `reconstruct_basis` (the sole hot-path
    /// entry point over `reconstruct_template_row_statuses`) preserves it
    /// verbatim. Runs on both backends: HiGHS validates a warm-start basis and
    /// would reject a corrupted one loudly; CLP would accept it silently, so this
    /// test alone cannot prove backend-acceptance — the HiGHS test below does.
    #[test]
    fn range_row_rests_at_upper_and_survives_reconstruction() {
        let system = build_system(build_range_constraints(), build_range_bounds());
        let range_row = find_range_row_stage0(&system);

        let config = build_config(2);
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

        let stored = outcome.result.basis_cache[0]
            .as_ref()
            .expect("stage 0 must have a captured basis after training");

        assert_eq!(
            stored.basis.row_status[range_row],
            BasisStatus::Upper,
            "the range row [3, 8] must rest at BasisStatus::Upper: T0's own [0, 30] \
             column bound and the loose <=25 sibling row are both slack at the optimum \
             (T0 = 8, deficit = 12), so only the range row's own upper bound forces the \
             remaining demand onto the bus deficit"
        );

        // Round-trip through the production entry point (reconstruct_basis calls
        // reconstruct_template_row_statuses internally — never bypass it).
        let target = ReconstructionTarget {
            base_row_count: stored.base_row_count,
            num_cols: stored.basis.col_status.len(),
        };
        let mut out = Basis::new(0, 0);
        let mut slot_lookup: Vec<Option<u32>> = vec![None; 16];
        let stats = reconstruct_basis(
            stored,
            target,
            std::iter::empty::<(usize, f64, &[f64])>(),
            &mut out,
            &mut slot_lookup,
        );
        assert_eq!(
            stats.preserved, 0,
            "this round-trip carries zero cut rows by construction — it isolates the \
             template-row path"
        );
        assert_eq!(
            out.row_status[range_row],
            BasisStatus::Upper,
            "reconstruct_basis must preserve the range row's Upper status verbatim — \
             it is not folded to Lower, Fixed, or Basic"
        );

        let num_row = out.row_status.len();
        enforce_basic_count_invariant(&mut out, num_row, stored.base_row_count).expect(
            "enforce_basic_count_invariant must accept a same-shape (no-churn) \
                 reconstruction of a basis captured from a real HiGHS/CLP solve",
        );
        assert_eq!(
            out.row_status[range_row],
            BasisStatus::Upper,
            "enforce_basic_count_invariant demotes only cut rows (index >= \
             base_row_count), never a template row"
        );
    }

    // ---------------------------------------------------------------------------
    // basic-count invariant survives; warm objective matches cold
    // ---------------------------------------------------------------------------

    /// With a range row present, training accepts every
    /// reconstructed basis (zero `basis_consistency_failures`, meaningful on
    /// HiGHS — CLP would accept a corrupted basis silently), and a checkpoint
    /// warm-started resume reaches the same converged objective as an
    /// equal-total-iteration cold run, within tolerance. Never `hot == cold`:
    /// this compares two independent runs' scalar `final_lb`, never a per-solve
    /// basis or dual.
    #[test]
    fn warm_start_matches_cold_objective_with_range_row() {
        const TOTAL_ITERS: u32 = 3;
        let comm = StubComm;

        let cold_config = build_config(TOTAL_ITERS);
        let mut setup_cold = build_setup_in_code(
            build_system(build_range_constraints(), build_range_bounds()),
            &cold_config,
        );
        let mut solver_cold = ActiveSolver::new().expect("ActiveSolver::new");
        let cold_outcome = setup_cold
            .train(&mut solver_cold, &comm, 1, ActiveSolver::new, None, None)
            .expect("cold train must not return Err");
        assert!(cold_outcome.error.is_none(), "cold training error");
        let cold_lb = cold_outcome.result.final_lb;
        let cold_stats = SolverStatsDelta::aggregate(
            cold_outcome
                .result
                .solver_stats_log
                .iter()
                .map(|e| &e.delta),
        );
        assert_eq!(
            cold_stats.basis_consistency_failures, 0,
            "cold training with a range row present must never have the solver \
             reject a reconstructed basis"
        );

        let checkpoint_config = build_config(1);
        let mut setup_checkpoint = build_setup_in_code(
            build_system(build_range_constraints(), build_range_bounds()),
            &checkpoint_config,
        );
        let mut solver_checkpoint = ActiveSolver::new().expect("ActiveSolver::new");
        let checkpoint_outcome = setup_checkpoint
            .train(
                &mut solver_checkpoint,
                &comm,
                1,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("checkpoint train must not return Err");
        assert!(
            checkpoint_outcome.error.is_none(),
            "checkpoint training error"
        );

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let policy_dir = tmpdir.path().join("policy");
        write_test_checkpoint(
            &policy_dir,
            &setup_checkpoint,
            &checkpoint_outcome.result,
            42,
        );
        let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");

        let mut setup_warm = build_setup_in_code(
            build_system(build_range_constraints(), build_range_bounds()),
            &cold_config,
        );
        let proof = cobre_sddp::test_support::trivial_full_fcf_proof(
            checkpoint.stage_cuts[0].state_dimension,
            checkpoint.metadata.num_stages,
        );
        let pool_state_dimensions: Vec<usize> = setup_warm
            .fcf
            .pools
            .iter()
            .map(|p| p.state_dimension)
            .collect();
        let visit_bounds: Vec<u64> = setup_warm
            .fcf
            .pools
            .iter()
            .map(|p| u64::from(p.visit_stride))
            .collect();
        let warm_fcf = FutureCostFunction::new_with_warm_start(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
            &visit_bounds,
            setup_warm.loop_params.forward_passes,
            setup_warm.loop_params.max_iterations.saturating_add(1),
        )
        .expect("warm-start FCF");
        setup_warm.replace_fcf(warm_fcf);
        setup_warm.set_resume_point(
            u64::from(checkpoint.metadata.producer.completed_iterations),
            checkpoint.metadata.producer.lower_bound_history.clone(),
        );

        let mut solver_warm = ActiveSolver::new().expect("ActiveSolver::new");
        let warm_outcome = setup_warm
            .train(&mut solver_warm, &comm, 1, ActiveSolver::new, None, None)
            .expect("warm-start train must not return Err");
        assert!(warm_outcome.error.is_none(), "warm-start training error");
        assert_eq!(
            warm_outcome.result.iterations,
            u64::from(TOTAL_ITERS),
            "the warm-started resume must reach the same total iteration count as \
             the cold run"
        );

        let warm_stats = SolverStatsDelta::aggregate(
            warm_outcome
                .result
                .solver_stats_log
                .iter()
                .map(|e| &e.delta),
        );
        assert_eq!(
            warm_stats.basis_consistency_failures, 0,
            "warm-started re-solve with a range row present must never have the \
             solver reject the reconstructed basis"
        );

        assert!(
            (warm_outcome.result.final_lb - cold_lb).abs() < 1e-6,
            "a warm-started re-solve must reach the same objective as an \
             equal-total-iteration cold solve within tolerance: warm={}, cold={}",
            warm_outcome.result.final_lb,
            cold_lb
        );
    }

    // ---------------------------------------------------------------------------
    // run-to-run reproducibility
    // ---------------------------------------------------------------------------

    /// Two fresh solver instances over the same
    /// range-constraint study produce bit-identical results — compared via
    /// `to_bits()`/`==`, never a tolerance (mirrors
    /// `crates/cobre-solver/tests/clp_determinism.rs`'s determinism harness).
    #[test]
    fn range_study_run_to_run_reproducibility() {
        let config = build_config(3);
        let comm = StubComm;

        let mut setup1 = build_setup_in_code(
            build_system(build_range_constraints(), build_range_bounds()),
            &config,
        );
        let mut solver1 = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome1 = setup1
            .train(&mut solver1, &comm, 1, ActiveSolver::new, None, None)
            .expect("run 1 must not return Err");
        assert!(outcome1.error.is_none());

        let mut setup2 = build_setup_in_code(
            build_system(build_range_constraints(), build_range_bounds()),
            &config,
        );
        let mut solver2 = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome2 = setup2
            .train(&mut solver2, &comm, 1, ActiveSolver::new, None, None)
            .expect("run 2 must not return Err");
        assert!(outcome2.error.is_none());

        assert_eq!(
            outcome1.result.iterations, outcome2.result.iterations,
            "both fresh runs must converge in the same iteration count"
        );
        assert_eq!(
            outcome1.result.final_lb.to_bits(),
            outcome2.result.final_lb.to_bits(),
            "two fresh solver instances over the same range-constraint study must \
             produce a bit-identical final_lb: {} vs {}",
            outcome1.result.final_lb,
            outcome2.result.final_lb
        );

        let templates1 = outcome1
            .result
            .frozen_templates
            .expect("run 1 frozen_templates");
        let templates2 = outcome2
            .result
            .frozen_templates
            .expect("run 2 frozen_templates");
        assert_eq!(templates1.len(), templates2.len());
        for (t1, t2) in templates1.iter().zip(templates2.iter()) {
            assert_eq!(
                t1.row_lower, t2.row_lower,
                "row_lower must be bit-identical"
            );
            assert_eq!(
                t1.row_upper, t2.row_upper,
                "row_upper must be bit-identical"
            );
            assert_eq!(
                t1.objective, t2.objective,
                "objective must be bit-identical"
            );
            assert_eq!(
                t1.col_starts, t2.col_starts,
                "col_starts must be bit-identical"
            );
            assert_eq!(
                t1.row_indices, t2.row_indices,
                "row_indices must be bit-identical"
            );
            assert_eq!(t1.values, t2.values, "CSC values must be bit-identical");
        }
    }

    // ---------------------------------------------------------------------------
    // declaration-order invariance for generic_constraints.json
    // ---------------------------------------------------------------------------

    /// `(id, name)` for the three constraints declared in every permutation test.
    const CONSTRAINT_DEFS: [(i32, &str); 3] = [(1, "cap_loose"), (2, "band"), (3, "floor_trivial")];

    /// Every ordering of 3 elements (property-style — a single hand-picked
    /// reordering can be a tautology).
    const ALL_ORDERINGS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    fn generic_constraints_json(order: &[usize; 3]) -> String {
        let entries: Vec<String> = order
            .iter()
            .map(|&i| {
                let (id, name) = CONSTRAINT_DEFS[i];
                format!(
                    "    {{\n      \"id\": {id},\n      \"name\": \"{name}\",\n      \
                     \"expression\": \"thermal_generation(0)\",\n      \
                     \"slack\": {{ \"enabled\": false }}\n    }}"
                )
            })
            .collect();
        format!(
            "{{\n  \"constraints\": [\n{}\n  ]\n}}\n",
            entries.join(",\n")
        )
    }

    /// Sort-contract tier: `cobre_io::parse_generic_constraints`
    /// sorts by id regardless of the JSON array's declared order — the mechanism
    /// that makes declaration-order invariance hold. Exhaustive over every
    /// ordering of 3 elements, not one hand-picked reordering.
    #[test]
    fn generic_constraints_json_sorts_by_id_for_every_ordering() {
        let mut canonical: Option<Vec<cobre_core::GenericConstraint>> = None;
        for order in ALL_ORDERINGS {
            let tmp = tempfile::tempdir().expect("tempdir");
            let path = tmp.path().join("generic_constraints.json");
            std::fs::write(&path, generic_constraints_json(&order)).expect("write json");
            let parsed = cobre_io::parse_generic_constraints(
                &path,
                &HashMap::new(),
                &cobre_io::LineBusPairIndex::default(),
            )
            .unwrap_or_else(|e| panic!("parse_generic_constraints failed for {order:?}: {e}"));
            match &canonical {
                None => canonical = Some(parsed),
                Some(expected) => assert_eq!(
                    &parsed, expected,
                    "ordering {order:?} must parse to the same canonical, id-sorted \
                     Vec<GenericConstraint> as ordering {:?}",
                    ALL_ORDERINGS[0]
                ),
            }
        }
        let ids: Vec<i32> = canonical
            .expect("at least one ordering ran")
            .iter()
            .map(|c| c.id.0)
            .collect();
        assert_eq!(
            ids,
            vec![1, 2, 3],
            "parse_generic_constraints must sort by id ascending"
        );
    }

    fn d13_case_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<crate> must have a parent")
            .parent()
            .expect("crates/ must have a parent (repo root)")
            .join("examples/deterministic/d13-generic-constraint")
    }

    fn copy_dir_recursive(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).expect("create_dir_all");
        for entry in std::fs::read_dir(src).expect("read_dir") {
            let entry = entry.expect("dir entry");
            let file_type = entry.file_type().expect("file_type");
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            if file_type.is_dir() {
                if entry.file_name() == "output" {
                    continue;
                }
                copy_dir_recursive(&src_path, &dst_path);
            } else {
                std::fs::copy(&src_path, &dst_path).expect("copy file");
            }
        }
    }

    /// Overwrite `constraints/generic_constraint_bounds.parquet` with bound rows
    /// for all 3 [`CONSTRAINT_DEFS`] across both d13 stages: `cap_loose <= 25`,
    /// `band` in `[3, 8]`, `floor_trivial >= 0`.
    fn write_generic_constraint_bounds_parquet(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("block_id", DataType::Int32, true),
            Field::new("bound_lower", DataType::Float64, true),
            Field::new("bound_upper", DataType::Float64, true),
        ]));

        let constraint_ids = vec![1, 1, 2, 2, 3, 3];
        let stage_ids = vec![0, 1, 0, 1, 0, 1];
        let block_ids: Int32Array = vec![None, None, None, None, None, None]
            .into_iter()
            .collect();
        // constraint 1 = cap_loose (<=25, upper-only); constraint 2 = band
        // ([3, 8], two-sided); constraint 3 = floor_trivial (>=0, lower-only).
        let bound_lowers: Float64Array =
            vec![None, None, Some(3.0), Some(3.0), Some(0.0), Some(0.0)]
                .into_iter()
                .collect();
        let bound_uppers: Float64Array =
            vec![Some(25.0), Some(25.0), Some(8.0), Some(8.0), None, None]
                .into_iter()
                .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(constraint_ids)),
                Arc::new(Int32Array::from(stage_ids)),
                Arc::new(block_ids),
                Arc::new(bound_lowers),
                Arc::new(bound_uppers),
            ],
        )
        .expect("valid record batch");

        let file = std::fs::File::create(path).expect("create parquet file");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter::try_new");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");
    }

    /// A fresh copy of `d13-generic-constraint` with `constraints/generic_constraints.json`
    /// (3 constraints in `order`) and `constraints/generic_constraint_bounds.parquet`
    /// (fixed content, order-independent) replaced. Never mutates the committed
    /// example — everything is written into a fresh `TempDir`.
    fn build_permuted_case(order: &[usize; 3]) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        copy_dir_recursive(&d13_case_dir(), tmp.path());
        std::fs::write(
            tmp.path().join("constraints/generic_constraints.json"),
            generic_constraints_json(order),
        )
        .expect("write generic_constraints.json");
        write_generic_constraint_bounds_parquet(
            &tmp.path()
                .join("constraints/generic_constraint_bounds.parquet"),
        );
        tmp
    }

    /// End-to-end tier: every ordering of
    /// `generic_constraints.json`'s 3 constraints (including `band`, the range
    /// one) produces a bit-identical LP (`frozen_templates`' CSC arrays and row
    /// bounds) and a bit-identical result (`final_lb`) through the real
    /// `cobre_io::load_case -> train` pipeline. Gated behind `slow-tests`: it
    /// builds and trains 6 full case directories.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn declaration_order_invariance_full_pipeline() {
        #[allow(clippy::type_complexity)]
        let mut baseline: Option<(
            u64,
            Vec<f64>,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
            Vec<Vec<i32>>,
            Vec<Vec<i32>>,
            Vec<Vec<f64>>,
        )> = None;

        for order in ALL_ORDERINGS {
            let tmp = build_permuted_case(&order);
            let dir = tmp.path();

            let config = cobre_io::parse_config(&dir.join("config.json")).expect("config");
            let system = cobre_io::load_case(dir).expect("load_case must succeed");

            let ids: Vec<i32> = system
                .generic_constraints()
                .iter()
                .map(|c| c.id.0)
                .collect();
            assert_eq!(
                ids,
                vec![1, 2, 3],
                "system.generic_constraints() must be canonically id-sorted regardless \
                 of generic_constraints.json's declared order {order:?}"
            );

            let pr = prepare_stochastic(system, dir, &config, 42, &ScenarioSource::default(), None)
                .expect("prepare_stochastic must succeed");
            let system = pr.system;
            let stochastic = pr.stochastic;
            let hydro_models = prepare_hydro_models(&system, dir, false)
                .expect("prepare_hydro_models must succeed");
            let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
                .expect("StudySetup::new must succeed");

            let comm = StubComm;
            let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
            let outcome = setup
                .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
                .expect("train must not return Err");
            assert!(
                outcome.error.is_none(),
                "training error for ordering {order:?}: {:?}",
                outcome.error
            );

            let templates = outcome
                .result
                .frozen_templates
                .expect("frozen_templates must be Some");
            let signature = (
                outcome.result.iterations,
                vec![outcome.result.final_lb],
                templates
                    .iter()
                    .map(|t| t.row_lower.clone())
                    .collect::<Vec<_>>(),
                templates
                    .iter()
                    .map(|t| t.row_upper.clone())
                    .collect::<Vec<_>>(),
                templates
                    .iter()
                    .map(|t| t.col_starts.clone())
                    .collect::<Vec<_>>(),
                templates
                    .iter()
                    .map(|t| t.row_indices.clone())
                    .collect::<Vec<_>>(),
                templates
                    .iter()
                    .map(|t| t.values.clone())
                    .collect::<Vec<_>>(),
            );

            match &baseline {
                None => baseline = Some(signature),
                Some(base) => {
                    assert_eq!(
                        base.0, signature.0,
                        "iterations differ for ordering {order:?}"
                    );
                    assert_eq!(
                        base.1[0].to_bits(),
                        signature.1[0].to_bits(),
                        "final_lb differs for ordering {order:?}: {} vs baseline {}",
                        signature.1[0],
                        base.1[0]
                    );
                    assert_eq!(
                        base.2, signature.2,
                        "row_lower differs for ordering {order:?}"
                    );
                    assert_eq!(
                        base.3, signature.3,
                        "row_upper differs for ordering {order:?}"
                    );
                    assert_eq!(
                        base.4, signature.4,
                        "col_starts differs for ordering {order:?}"
                    );
                    assert_eq!(
                        base.5, signature.5,
                        "row_indices differs for ordering {order:?}"
                    );
                    assert_eq!(
                        base.6, signature.6,
                        "CSC values differ for ordering {order:?}"
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // range vs. two-row solution equivalence
    // ---------------------------------------------------------------------------

    /// The same `[3, 8]` band expressed as one `range`
    /// constraint versus a `>= 3` plus a `<= 8` constraint reaches equal optimal
    /// objective and equal primals within the suite's tolerance. Duals and row
    /// counts are never compared — the two formulations have different row
    /// counts by construction and may settle on a different-but-equally-valid
    /// vertex.
    #[test]
    fn range_matches_two_row_formulation_objective_and_primals() {
        let mut config = build_config(2);
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 1 });

        let (range_constraints, range_bounds) = build_band_constraints(true);
        let (two_row_constraints, two_row_bounds) = build_band_constraints(false);

        let mut setup_range =
            build_setup_in_code(build_system(range_constraints, range_bounds), &config);
        let mut setup_two_row =
            build_setup_in_code(build_system(two_row_constraints, two_row_bounds), &config);

        let range_scenarios = run_simulation(&mut setup_range, 1);
        let two_row_scenarios = run_simulation(&mut setup_two_row, 1);

        assert_eq!(range_scenarios.len(), 1);
        assert_eq!(two_row_scenarios.len(), 1);

        let range_total = range_scenarios[0].total_cost;
        let two_row_total = two_row_scenarios[0].total_cost;
        assert!(
            (range_total - two_row_total).abs() < 1e-6,
            "range and two-row formulations of the same band must reach the same \
             optimal objective: range={range_total}, two_row={two_row_total}"
        );

        assert_eq!(
            range_scenarios[0].stages.len(),
            two_row_scenarios[0].stages.len()
        );
        for (rs, ts) in range_scenarios[0]
            .stages
            .iter()
            .zip(two_row_scenarios[0].stages.iter())
        {
            assert_eq!(rs.thermals.len(), 1);
            assert_eq!(ts.thermals.len(), 1);
            let r_gen = rs.thermals[0].generation_mw;
            let t_gen = ts.thermals[0].generation_mw;
            assert!(
                (r_gen - t_gen).abs() < 1e-6,
                "stage {}: range T0 generation {r_gen} must match two-row T0 \
                 generation {t_gen} within tolerance",
                rs.stage_id
            );
            assert!(
                (r_gen - 8.0).abs() < 1e-6,
                "stage {}: both formulations must dispatch T0 at the band's upper \
                 bound (8 MW); got {r_gen}",
                rs.stage_id
            );
        }
    }
}
