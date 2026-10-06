//! Consolidated hydro-simulation and stage-structure integration tests for
//! `cobre-sddp`.
//!
//! Each source domain lives in its own inner `mod` so the suite links the
//! statically-bound solver once rather than once per file. Per-`mod` scoping
//! isolates each group's consts, helpers, and fixtures.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

mod common;

mod simulation_only {
    //! Integration test for simulation-only round-trip: the FCF reconstructed from
    //! a written-then-reloaded policy checkpoint must evaluate identically.

    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
    use cobre_sddp::{
        FutureCostFunction, StudySetup, build_basis_cache_from_checkpoint,
        hydro_models::prepare_hydro_models,
        policy_export::{
            build_active_indices, build_stage_basis_records, build_stage_cut_records,
            build_stage_cuts_payloads, convert_basis_cache,
        },
        setup::{NodePos, prepare_stochastic},
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    /// Ascending, distinct `end_date` per pool (`2030-01-01` plus the pool
    /// index in months) — the study-stage calendar fed to
    /// `build_stage_cuts_payloads` to self-describe each pool's
    /// `priced_state_date`.
    fn ascending_stage_end_dates(n_pools: usize) -> Vec<chrono::NaiveDate> {
        let base = chrono::NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        (0..n_pools as u32)
            .map(|i| base.checked_add_months(chrono::Months::new(i)).unwrap())
            .collect()
    }

    #[test]
    fn simulation_only_fcf_round_trip() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d01-thermal-dispatch");

        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result = prepare_stochastic(
            system,
            &case_dir,
            &config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        let training_result = outcome.result;

        let original_active_cuts = setup.fcf.total_active_cuts();
        assert!(original_active_cuts > 0, "training should produce cuts");

        let n_stages = setup.inputs.stage_data.stage_templates.templates.len();
        let state_dim = setup.fcf.state_dimension;

        let test_state: Vec<f64> = vec![50.0; state_dim];
        let mut original_evals = Vec::with_capacity(n_stages);
        for stage in 0..n_stages {
            original_evals.push(setup.fcf.evaluate_at_state(stage, &test_state));
        }

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let policy_dir = tmpdir.path().join("policy");

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

        let (basis_col_u8, basis_row_u8) = convert_basis_cache(&training_result.basis_cache);
        let stage_bases = build_stage_basis_records(
            &training_result.basis_cache,
            training_result.iterations,
            &basis_col_u8,
            &basis_row_u8,
        );

        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
        let metadata = cobre_sddp::test_support::checkpoint_metadata(
            n_stages as u32,
            setup.build_graph_manifest(),
            cobre_io::ProducerBlock {
                completed_iterations: training_result.iterations as u32,
                final_lower_bound: training_result.final_lb,
                best_upper_bound: Some(training_result.final_ub),
                max_iterations: setup.loop_params.max_iterations as u32,
                forward_passes: setup.loop_params.forward_passes,
                warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
                warm_start_counts,
                rng_seed: 42,
                total_visited_states: 0,
                training_block_mode: "parallel".to_string(),
                training_block_mode_per_stage: vec![],
                cost_scale_factor: None,
                lower_bound_history: Vec::new(),
            },
        );

        write_policy_checkpoint(&policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write checkpoint");

        let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");

        assert_eq!(
            checkpoint.stage_cuts[0].state_dimension, state_dim as u32,
            "per-pool state_dimension must round-trip"
        );
        assert_eq!(
            checkpoint.metadata.num_stages, n_stages as u32,
            "num_stages must round-trip"
        );
        assert_eq!(
            checkpoint.stage_cuts.len(),
            n_stages,
            "stage_cuts count must match"
        );

        let proof = cobre_sddp::test_support::trivial_full_fcf_proof(
            checkpoint.stage_cuts[0].state_dimension,
            checkpoint.metadata.num_stages,
        );
        let pool_state_dimensions: Vec<usize> =
            setup.fcf.pools.iter().map(|p| p.state_dimension).collect();
        let loaded_fcf = FutureCostFunction::from_deserialized(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
        )
        .expect("from_deserialized");

        assert_eq!(
            loaded_fcf.total_active_cuts(),
            original_active_cuts,
            "active cut count must match after round-trip"
        );

        for (stage, &expected_eval) in original_evals.iter().enumerate() {
            let loaded_eval = loaded_fcf.evaluate_at_state(stage, &test_state);
            assert_eq!(
                loaded_eval, expected_eval,
                "evaluate_at_state mismatch at stage {stage}"
            );
        }

        let load = build_basis_cache_from_checkpoint(
            &checkpoint.stage_bases,
            &checkpoint.stage_cuts,
            &setup,
        );
        assert!(
            load.unused.is_none(),
            "every stored basis of a current-build checkpoint fits its own study"
        );
        let loaded_basis_cache = load.cache;
        assert_eq!(
            loaded_basis_cache.len(),
            n_stages,
            "basis cache length must match (chain: n_nodes == n_stages)"
        );
        let has_basis = loaded_basis_cache.iter().any(Option::is_some);
        assert!(has_basis, "at least one stage should have basis data");

        for (pos, cb) in loaded_basis_cache.iter().enumerate() {
            let Some(cb) = cb else { continue };
            let stage = setup.inputs.node_graph.nodes[NodePos(pos)].stage;
            let expected_rows = setup.inputs.stage_data.stage_templates.templates[stage.0].num_rows;
            assert_eq!(
                cb.base_row_count, expected_rows,
                "node {pos} base_row_count must equal the study's own template row count"
            );
            assert_eq!(
                cb.cut_row_slots.len(),
                cb.basis.row_status.len() - cb.base_row_count,
                "node {pos} cut_row_slots length must equal the trailing cut-row count"
            );
        }
    }
}

mod d17_signed_evaporation {
    //! Focused semantic regression test for the signed evaporation-outflow variable.
    //!
    //! Runs D17 (`d17-evaporation-mixed-sign`) end-to-end and asserts that the
    //! simulation output reflects the mixed-sign monthly evaporation coefficients
    //! defined in the fixture:
    //!
    //! - Stages 0, 1, 2 land on Oct/Nov/Dec where `coefficients_mm` is negative
    //!   (net rainfall input on the lake surface); the simulated
    //!   `evaporation_m3s` must be **negative**.
    //! - Stage 3 lands on January where `coefficients_mm` is positive (true
    //!   evaporation loss); the simulated `evaporation_m3s` must be **positive**.
    //! - In every stage the symmetric `[-q_max, +q_max]` magnitude bound must
    //!   absorb the linearized target, so neither the positive nor the negative
    //!   evaporation violation slack should fire.
    //!
    //! The companion `parity_hash_d17` test guards bit-for-bit byte stability of
    //! the same case; this test guards the **sign semantics** explicitly so a
    //! future regression that flips the sign convention is caught by an
    //! actionable assertion rather than an opaque hash diff.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::{
        SimulationWeighting, StudySetup, aggregate_simulation,
        hydro_models::prepare_hydro_models,
        setup::{StudyParams, prepare_stochastic},
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d17_evaporation_is_signed_per_month() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d17-evaporation-mixed-sign");

        let config_path = dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(&dir).expect("load_case must succeed");

        let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default(), None)
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

        let mut config_with_sim = config.clone();
        config_with_sim.simulation.enabled = true;
        config_with_sim.simulation.selection =
            Some(SimulationSelection::Sampled { num_scenarios: 1 });

        let sentinel = Path::new("config.json");
        let training_source = config_with_sim
            .training_scenario_source(sentinel)
            .expect("training_scenario_source must parse");
        let simulation_source = config_with_sim
            .simulation_scenario_source(sentinel)
            .expect("simulation_scenario_source must parse");

        let construction = StudyParams::from_config(&config_with_sim, Vec::new())
            .expect("StudyParams::from_config must succeed");

        let mut setup = StudySetup::from_broadcast_params(
            &system,
            stochastic,
            construction,
            hydro_models,
            &training_source,
            &simulation_source,
        )
        .expect("StudySetup must build");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "D17: expected no training error");
        let result = outcome.result;

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("simulation workspace pool must build");

        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let local_costs = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                result.frozen_templates.as_deref(),
                &result.basis_cache,
            )
            .expect("simulate must return Ok");

        drop(result_tx);
        let mut scenario_results = drain_handle.join().expect("drain thread must not panic");

        let sim_config = setup.simulation_config();
        let (_summary, _gathered) = aggregate_simulation(
            &local_costs.costs,
            sim_config,
            &comm,
            SimulationWeighting::Uniform,
        )
        .expect("aggregate_simulation must succeed");

        assert_eq!(
            scenario_results.len(),
            1,
            "expected exactly one simulation scenario for D17, got {}",
            scenario_results.len()
        );
        let scenario = scenario_results.remove(0);
        assert_eq!(
            scenario.stages.len(),
            4,
            "expected 4 stages in D17 simulation result, got {}",
            scenario.stages.len()
        );

        for stage in &scenario.stages {
            assert_eq!(
                stage.hydros.len(),
                1,
                "stage {} should have exactly one hydro record",
                stage.stage_id
            );
            let h = &stage.hydros[0];
            let evap = h
                .evaporation_m3s
                .expect("evaporation must be modeled for D17");

            match stage.stage_id {
                0..=2 => assert!(
                    evap < 0.0,
                    "stage {} (Oct/Nov/Dec) expected net-rainfall evaporation_m3s < 0, got {evap}",
                    stage.stage_id
                ),
                3 => assert!(
                    evap > 0.0,
                    "stage {} (Jan) expected true-evaporation evaporation_m3s > 0, got {evap}",
                    stage.stage_id
                ),
                other => panic!("unexpected stage_id in D17 simulation: {other}"),
            }

            assert_eq!(
                h.evaporation_violation_pos_m3s, 0.0,
                "stage {}: positive evaporation slack must not fire, got {}",
                stage.stage_id, h.evaporation_violation_pos_m3s
            );
            assert_eq!(
                h.evaporation_violation_neg_m3s, 0.0,
                "stage {}: negative evaporation slack must not fire, got {}",
                stage.stage_id, h.evaporation_violation_neg_m3s
            );
        }
    }
}

mod d41_energy_contracts_simulation {
    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::simulation::accumulate_category_costs;
    use cobre_sddp::simulation::types::ScenarioCategoryCosts;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    /// Energy-contract gating and dispatch must reach the simulation output.
    ///
    /// The parity hash does NOT digest contract-output columns, so a gating bug — a
    /// commissioning-dormant contract that dispatches, a wrong-signed load-balance
    /// term that flips import/export, a mis-stamped `contract_id`, a stage override
    /// that fails to change dispatch — would still hash-match and ship silently. This
    /// test asserts the dispatch values directly through the full train+simulate
    /// pipeline; they are the only numerical guard.
    ///
    /// The commissioning gate is `entry <= stage.id && stage.id < exit`. Under the
    /// dense layout a dormant contract keeps its column but is pinned to `[0, 0]`,
    /// emitting a ZERO row rather than being absent.
    ///
    /// D41 topology: 3 stages, single 730h block, one bus, two hydros H0->H1, one
    /// thermal at $50/MWh, deficit at $1000/MWh. Two contracts on bus 0: import id 0
    /// (price $200/MWh, always active, stage-2 `min_mw` + `price_per_mwh` override)
    /// and export id 1 (price -$150/MWh revenue, commissioned at stage 1 only).
    #[test]
    fn energy_contract_gating_reaches_simulation_output() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d41-energy-contracts");

        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        // The shipped parity case trains only; enable one sim scenario so the contract extraction path runs.
        config.simulation = SimulationConfig {
            enabled: true,
            io_channel_capacity: 8,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
            ..SimulationConfig::default()
        };

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
            .expect("StudySetup::new");

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

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            scenario_results.len(),
            1,
            "exactly one deterministic scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            3,
            "one stage record per study stage (horizon 0, 1, 2)",
        );

        // Each stage keeps a dense row per contract (single block): import id 0 then
        // export id 1, regardless of commissioning.
        for (t, stage) in scenario.stages.iter().enumerate() {
            assert_eq!(
                stage.contracts.len(),
                2,
                "stage {t} keeps the dense column for both contracts, got {:?}",
                stage.contracts,
            );
        }

        let import_row = |t: usize| {
            scenario.stages[t]
                .contracts
                .iter()
                .find(|r| r.contract_id == 0)
                .unwrap_or_else(|| panic!("stage {t} must have an import (id 0) row"))
        };
        let export_row = |t: usize| {
            scenario.stages[t]
                .contracts
                .iter()
                .find(|r| r.contract_id == 1)
                .unwrap_or_else(|| panic!("stage {t} must have an export (id 1) row"))
        };

        // Commissioning gate, dormant zero: export id 1 has entry_stage_id = 1,
        // so stage 0 (t < E) is dormant: zero power, operative_state_code == 1.
        let exp0 = export_row(0);
        assert_eq!(
            exp0.power_mw, 0.0,
            "stage 0 is before the export entry window: dormant column pinned to 0, got {}",
            exp0.power_mw,
        );
        assert_eq!(
            exp0.operative_state_code, 1,
            "dormant export row still carries operative_state_code 1",
        );

        // Commissioning gate, active dispatch + post-exit zero: export id 1 has
        // exit_stage_id = 2, active only at stage 1 (E <= t < X). It is the surplus
        // stage, so the optimizer sells the revenue export at a positive power.
        let exp1 = export_row(1);
        assert!(
            exp1.power_mw > 0.0,
            "stage 1 is inside the export window and a surplus stage: export must dispatch, got {}",
            exp1.power_mw,
        );
        assert!(
            exp1.total_cost < 0.0,
            "an active export (negative price) nets negative total_cost (revenue), got {}",
            exp1.total_cost,
        );
        let exp2 = export_row(2);
        assert_eq!(
            exp2.power_mw, 0.0,
            "stage 2 is at/after the export exit boundary (decommissioned): pinned to 0, got {}",
            exp2.power_mw,
        );

        // Stage 0 is a shortage stage — import id 0 is pulled (high load, scarce hydro,
        // thermal capped): import at $200 beats the $1000 deficit.
        let imp0 = import_row(0);
        assert!(
            imp0.power_mw > 0.0,
            "stage 0 is a shortage stage: import must be pulled, got {}",
            imp0.power_mw,
        );
        assert!(
            (imp0.price_per_mwh - 200.0).abs() < 1e-6,
            "stage 0 import uses the base price 200.0, got {}",
            imp0.price_per_mwh,
        );

        // Take-or-pay floor binds: the contract_bounds override sets import id 0
        // min_mw = 10.0 at stage 2, where importing is uneconomic (hydro+thermal cover
        // the load and the override price is 999.0). The floor forces power_mw == 10.0.
        let imp2 = import_row(2);
        assert!(
            (imp2.power_mw - 10.0).abs() < 1e-6,
            "stage 2 import take-or-pay floor (min_mw = 10.0) must bind, got {}",
            imp2.power_mw,
        );

        // Stage override changes price and cost: the same override sets import id
        // 0 price_per_mwh = 999.0 at stage 2, differing from the base 200.0 used at
        // stage 0, which visibly changes total_cost relative to the non-overridden
        // stage 0 import row.
        assert!(
            (imp2.price_per_mwh - 999.0).abs() < 1e-6,
            "stage 2 import price override (999.0) must be reflected, got {}",
            imp2.price_per_mwh,
        );
        assert!(
            (imp2.price_per_mwh - imp0.price_per_mwh).abs() > 1e-6,
            "override price (999.0) must differ from base price (200.0)",
        );
        let block_hours = 730.0;
        assert!(
            (imp2.total_cost - 999.0 * 10.0 * block_hours).abs() < 1.0,
            "stage 2 import total_cost = price * power * hours = 999 * 10 * 730, got {}",
            imp2.total_cost,
        );
        assert!(
            (imp2.total_cost - imp0.total_cost).abs() > 1.0,
            "overridden stage 2 import total_cost must differ from non-overridden stage 0",
        );

        // Cost-breakdown invariant with an active contract: the five macro
        // categories sum to immediate_cost at the export-active stage 1.
        let cost = &scenario.stages[1].costs[0];
        let mut accum = ScenarioCategoryCosts {
            resource_cost: 0.0,
            recourse_cost: 0.0,
            violation_cost: 0.0,
            regularization_cost: 0.0,
            imputed_cost: 0.0,
        };
        accumulate_category_costs(cost, &mut accum);
        let macro_sum = accum.resource_cost
            + accum.recourse_cost
            + accum.violation_cost
            + accum.regularization_cost
            + accum.imputed_cost;
        assert!(
            (macro_sum - cost.immediate_cost).abs() < 1.0,
            "Sigma(macro categories) ({macro_sum}) must equal immediate_cost ({}) with an active contract",
            cost.immediate_cost,
        );
    }
}

mod multi_resolution_integration {
    //! End-to-end integration test for the multi-resolution stage-transition pipeline
    //! on the D30 case (6 monthly stages followed by 4 quarterly stages).

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::temporal::SeasonCycleType;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn d30_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d30-multi-resolution-monthly-quarterly")
    }

    /// Uses `config.training_scenario_source()`, not [`ScenarioSource::default()`]:
    /// the default silently falls back to `InSample` and bypasses the PAR(1) pipeline.
    fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> StudySetup {
        let system = cobre_io::load_case(case_dir).expect("load_case");
        let config_path = case_dir.join("config.json");
        let source = config
            .training_scenario_source(&config_path)
            .expect("training_scenario_source");
        let prep = prepare_stochastic(system, case_dir, config, 42, &source, None)
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

    // ---------------------------------------------------------------------------
    // Test: structural properties, training, and simulation
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_resolution_structural_properties_and_training() {
        let case_dir = d30_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let system = cobre_io::load_case(&case_dir).expect("load_case D30");

        // Pre-study stages carry negative IDs; the study count is the non-negative set.
        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

        assert_eq!(
            study_stages.len(),
            10,
            "D30: expected 10 study stages; got {}",
            study_stages.len()
        );

        for (t, stage) in study_stages.iter().enumerate().take(6) {
            assert_eq!(
                stage.scenario_config.branching_factor, 5,
                "D30: stage {t} (monthly) must have num_scenarios == 5"
            );
        }

        for (t, stage) in study_stages.iter().enumerate().skip(6) {
            assert_eq!(
                stage.scenario_config.branching_factor, 5,
                "D30: stage {t} (quarterly) must have num_scenarios == 5"
            );
        }

        let season_map = system
            .policy_graph()
            .season_map
            .as_ref()
            .expect("D30 must have season_map");
        assert_eq!(
            season_map.cycle_type,
            SeasonCycleType::Custom,
            "D30: season map must have Custom cycle type"
        );
        assert_eq!(
            season_map.seasons.len(),
            16,
            "D30: Custom season map must have 16 seasons (12 monthly + 4 quarterly); got {}",
            season_map.seasons.len()
        );

        for expected_id in 0..12_usize {
            assert!(
                season_map.seasons.iter().any(|s| s.id == expected_id),
                "D30: monthly season id {expected_id} must be present in the season map"
            );
        }

        for expected_id in 12..16_usize {
            assert!(
                season_map.seasons.iter().any(|s| s.id == expected_id),
                "D30: quarterly season id {expected_id} must be present in the season map"
            );
        }

        let mut setup = build_setup(&case_dir, &config);

        let groups = &setup.inputs.stage_data.noise_group_ids;
        assert_eq!(
            groups.len(),
            10,
            "D30: noise_group_ids must have 10 elements (one per study stage); got {}",
            groups.len()
        );

        // Monthly stages 0-5 each pair a unique season_id with a unique year, so each
        // gets its own group ID — hence 6 distinct values.
        let monthly_groups = &groups[..6];
        let mut unique_monthly: Vec<u32> = monthly_groups.to_vec();
        unique_monthly.sort_unstable();
        unique_monthly.dedup();
        assert_eq!(
            unique_monthly.len(),
            6,
            "D30: monthly stages 0-5 must have 6 distinct noise group IDs; got {monthly_groups:?}"
        );

        let stage_ctx = setup.stage_ctx();
        let lag_transitions = stage_ctx.stage_lag_transitions;

        assert_eq!(
            lag_transitions.len(),
            10,
            "D30: stage_lag_transitions must have 10 entries (one per study stage); got {}",
            lag_transitions.len()
        );

        // downstream_par_order == 1 → the pre-transition window is the 1*3 = 3 monthly
        // stages (3, 4, 5) immediately before the quarterly boundary.
        for (t, transition) in lag_transitions.iter().enumerate().take(6).skip(3) {
            assert!(
                transition.accumulate_downstream,
                "D30: monthly stage {t} (index in pre-transition window) must have              accumulate_downstream == true; got false. Full transition: {transition:?}"
            );
        }

        for (t, transition) in lag_transitions.iter().enumerate().take(3) {
            assert!(
                !transition.accumulate_downstream,
                "D30: monthly stage {t} (outside pre-transition window) must have              accumulate_downstream == false; got true. Full transition: {transition:?}"
            );
        }

        assert!(
            lag_transitions[6].rebuild_from_downstream,
            "D30: first quarterly stage (index 6) must have rebuild_from_downstream == true; \
             got false. Full transition: {:?}",
            lag_transitions[6]
        );

        for (t, transition) in lag_transitions.iter().enumerate().skip(7) {
            assert!(
                !transition.rebuild_from_downstream,
                "D30: quarterly stage {t} (not first) must have rebuild_from_downstream == false;              got true. Full transition: {transition:?}"
            );
        }

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("solver");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("D30: train must not return Err");

        assert!(
            outcome.error.is_none(),
            "D30: training outcome must have no error; got: {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "D30: must complete at least 1 iteration; got {}",
            outcome.result.iterations
        );
        assert!(
            outcome.result.final_lb > 0.0,
            "D30: lower bound must be positive; got {}",
            outcome.result.final_lb
        );
        assert!(
            outcome.result.final_lb.is_finite(),
            "D30: lower bound must be finite; got {}",
            outcome.result.final_lb
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("D30: workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

        // Drain the channel in a background thread to avoid blocking simulate().
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("D30: simulate must not return Err");

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            scenario_results.len(),
            1,
            "D30: expected 1 simulation scenario result; got {}",
            scenario_results.len()
        );
    }
}

mod sparse_dense {
    //! `build_cut_row_batch_into` emits one `(lp_column, coefficient)` pair per
    //! nonzero-state-mask entry in ascending order, then the theta column. For a
    //! proper-subset mask (mixed AR orders), the emitted columns and values must
    //! equal the remapped LP columns and the negated coefficients.

    use cobre_sddp::FutureCostFunction;
    use cobre_sddp::build_cut_row_batch_into;
    use cobre_sddp::indexer::StateSpace;
    use cobre_sddp::lead_time::AnticipatedResolution;
    use cobre_sddp::setup::NodeId;
    use cobre_sddp::test_support::cut_state_projection;
    use cobre_solver::RowBatch;

    #[test]
    fn sparse_partial_mask_produces_correct_subset() {
        let n_hydro = 3;
        let max_par_order = 2;
        let n_state = n_hydro * (1 + max_par_order);

        // The [0, 1, 2] arg is per-hydro effective_lag_count; mixed orders zero out
        // some lag slots. StateSpace::new finalizes the mask as production setup does.
        let state = StateSpace::new(
            n_hydro,
            max_par_order,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &[0, 1, 2],
        );
        // Expected mask = storage [0,1,2] + lag0 of h1,h2 [4,5] + lag1 of h2 [8].
        let mask = &state.nonzero_state_indices;
        assert_eq!(
            mask.len(),
            6,
            "mixed-order mask has 3 + 0 + 1 + 2 = 6 entries"
        );

        let mut fcf = FutureCostFunction::new(2, n_state, 1, 1, &[0; 2]);
        let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        fcf.add_cut(NodeId(0), 0, 0, 0, 50.0, &coeffs);

        let col_scale: Vec<f64> = Vec::new();

        let mut batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        let cut_state = cut_state_projection(&state);
        build_cut_row_batch_into(&mut batch, &fcf, 0, &state, &cut_state, &col_scale);

        assert_eq!(batch.num_rows, 1);
        let theta_col = state.theta;
        let expected_cols: Vec<i32> = mask
            .iter()
            .map(|&j| state.state_to_lp_column(j).get() as i32)
            .chain(std::iter::once(theta_col as i32))
            .collect();
        assert_eq!(
            batch.col_indices, expected_cols,
            "sparse col_indices mismatch"
        );

        let expected_values: Vec<f64> = mask
            .iter()
            .map(|&j| -coeffs[j.get()])
            .chain(std::iter::once(1.0))
            .collect();
        for (i, (actual, expected)) in batch.values.iter().zip(expected_values.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "values[{i}] differ: actual={actual}, expected={expected}"
            );
        }
    }
}

mod decomp_integration {
    //! End-to-end integration test for the mixed-resolution pipeline on the D28
    //! case (5 weekly stages + 1 monthly terminal stage).

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::temporal::SeasonCycleType;
    use cobre_io::output::policy::write_policy_checkpoint;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    fn d28_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d28-decomp-weekly-monthly")
    }

    /// Ascending, distinct `end_date` per pool (`2030-01-01` plus the pool
    /// index in months) — the study-stage calendar fed to
    /// `build_stage_cuts_payloads` to self-describe each pool's
    /// `priced_state_date`.
    fn ascending_stage_end_dates(n_pools: usize) -> Vec<chrono::NaiveDate> {
        let base = chrono::NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
        (0..n_pools as u32)
            .map(|i| base.checked_add_months(chrono::Months::new(i)).unwrap())
            .collect()
    }

    /// Write a policy checkpoint to `policy_dir` from the given setup and training result.
    fn write_test_checkpoint(
        policy_dir: &Path,
        setup: &StudySetup,
        result: &cobre_sddp::TrainingResult,
        seed: u64,
    ) {
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> =
            vec![Vec::new(); setup.fcf.pools.len()];
        write_test_checkpoint_with_manifests(policy_dir, setup, result, seed, &stage_manifests);
    }

    /// Like [`write_test_checkpoint`] but attaches `stage_manifests` (one per pool) to
    /// the per-stage cut payloads, so a test can write a checkpoint whose entity
    /// identity deliberately diverges from the current study's.
    fn write_test_checkpoint_with_manifests(
        policy_dir: &Path,
        setup: &StudySetup,
        result: &cobre_sddp::TrainingResult,
        seed: u64,
        stage_manifests: &[Vec<cobre_io::EntitySlot>],
    ) {
        use cobre_sddp::policy_export::{
            build_active_indices, build_stage_basis_records, build_stage_cut_records,
            build_stage_cuts_payloads, convert_basis_cache,
        };
        let fcf = &setup.fcf;
        let stage_records = build_stage_cut_records(fcf);
        let stage_active_indices = build_active_indices(&stage_records);
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
            stage_manifests,
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

    /// Build a `StudySetup` for the D28 case, calling `config.training_scenario_source()`
    /// so the External inflow library is loaded; `ScenarioSource::default()` (`InSample`)
    /// would skip the external parquet and produce a wrong scenario pipeline.
    fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> (StudySetup, cobre_core::System) {
        let system = cobre_io::load_case(case_dir).expect("load_case");
        let config_path = case_dir.join("config.json");
        let source = config
            .training_scenario_source(&config_path)
            .expect("training_scenario_source");
        let prep = prepare_stochastic(system, case_dir, config, 42, &source, None)
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

    /// Verify structural properties, training correctness, lag accumulation, and
    /// simulation completion for the D28 mixed-resolution case.
    ///
    /// D28 has 5 weekly stages (indices 0-4, `num_scenarios == 1`) followed by
    /// 1 monthly terminal stage (index 5, `num_scenarios == 5`). The monthly
    /// season map must have 12 seasons.
    #[test]
    fn structural_properties_and_training() {
        let case_dir = d28_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let system = cobre_io::load_case(&case_dir).expect("load_case D28");
        let stages = system.stages();

        assert_eq!(
            stages.len(),
            6,
            "D28 must have exactly 6 study stages; got {}",
            stages.len()
        );

        for (t, stage) in stages.iter().enumerate().take(5) {
            assert_eq!(
                stage.scenario_config.branching_factor, 1,
                "stage {t} (weekly) must have num_scenarios == 1"
            );
        }

        assert_eq!(
            stages[5].scenario_config.branching_factor, 5,
            "stage 5 (monthly) must have num_scenarios == 5"
        );

        let season_map = system
            .policy_graph()
            .season_map
            .as_ref()
            .expect("D28 must have season_map");
        assert_eq!(
            season_map.cycle_type,
            SeasonCycleType::Monthly,
            "season map must have monthly cycle type"
        );
        assert_eq!(
            season_map.seasons.len(),
            12,
            "monthly season map must have 12 seasons"
        );

        let (mut setup, _system) = build_setup(&case_dir, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("solver");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training outcome must have no error; got: {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "must complete at least 1 iteration; got {}",
            outcome.result.iterations
        );
        assert!(
            outcome.result.final_lb.is_finite(),
            "lower bound must be finite; got {}",
            outcome.result.final_lb
        );
        assert!(
            !outcome.result.final_lb.is_nan(),
            "lower bound must not be NaN"
        );

        // The backward pass writes cuts into pools[0..T-2] only; the terminal pool
        // (pools[T-1]) stays empty unless boundary cuts are injected.
        let fcf = &setup.fcf;
        let n_pools = fcf.pools.len();
        for (t, pool) in fcf.pools[..n_pools - 1].iter().enumerate() {
            assert!(
                pool.populated() >= 1,
                "stage {t} (non-terminal) cut pool must have at least 1 cut after training; got {}",
                pool.populated()
            );
        }
        assert_eq!(
            fcf.pools[n_pools - 1].populated(),
            0,
            "terminal pool must have 0 backward-pass cuts (boundary-cut injection is a separate test)"
        );

        let stage_ctx = setup.stage_ctx();
        let lag_transitions = stage_ctx.stage_lag_transitions;

        assert_eq!(
            lag_transitions.len(),
            6,
            "stage_lag_transitions must have 6 entries (one per study stage)"
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

        // Drain the channel in a background thread to avoid blocking simulate().
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");

        drop(result_tx);
        let _scenario_results = drain_handle.join().expect("drain thread must not panic");

        let sim_config = setup.simulation_config();
        assert_eq!(
            sim_config.n_scenarios, 5,
            "simulation must be configured for 5 scenarios"
        );
    }

    /// Verify that boundary cuts produced by a trained D28 source study compose
    /// correctly with a consumer D28 study: injecting them must not degrade the
    /// lower bound relative to the no-boundary-cut baseline.
    #[test]
    fn decomp_boundary_cuts_compose_with_weekly_monthly() {
        let case_dir = d28_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let comm = StubComm;

        let (mut setup_a, _system_a) = build_setup(&case_dir, &config);
        let mut solver_a = ActiveSolver::new().expect("solver A");
        let outcome_a = setup_a
            .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
            .expect("train A");
        assert!(outcome_a.error.is_none(), "source training must not error");

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let source_policy_dir = tmpdir.path().join("source_policy");
        write_test_checkpoint(&source_policy_dir, &setup_a, &outcome_a.result, 42);

        let num_stages = setup_a.fcf.pools.len();
        // second-to-last stage has backward-pass cuts
        let boundary_date = ascending_stage_end_dates(num_stages)[num_stages - 2];

        let (mut setup_b, _system_b) = build_setup(&case_dir, &config);
        let mut solver_b = ActiveSolver::new().expect("solver B");
        let outcome_b = setup_b
            .train(&mut solver_b, &comm, 1, ActiveSolver::new, None, None)
            .expect("train B");
        assert!(
            outcome_b.error.is_none(),
            "baseline training must not error"
        );
        let lb_no_boundary = outcome_b.result.final_lb;

        let (mut setup_c, system_c) = build_setup(&case_dir, &config);
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
            "terminal pool must have boundary cuts after injection; warm_start_count == 0"
        );
        assert!(
            terminal_pool.populated() >= terminal_pool.warm_start_count as usize,
            "populated_count ({}) must include boundary cuts (warm_start_count = {})",
            terminal_pool.populated(),
            terminal_pool.warm_start_count
        );

        let mut solver_c = ActiveSolver::new().expect("solver C");
        let outcome_c = setup_c
            .train(&mut solver_c, &comm, 1, ActiveSolver::new, None, None)
            .expect("train C");
        assert!(
            outcome_c.error.is_none(),
            "consumer training must not error"
        );
        let lb_with_boundary = outcome_c.result.final_lb;

        assert!(
            lb_with_boundary >= lb_no_boundary - 1e-6,
            "boundary LB ({lb_with_boundary}) must be >= baseline LB ({lb_no_boundary})"
        );
    }

    /// A boundary checkpoint whose source-stage manifest carries a permuted hydro
    /// `entity_id` at slot 0 must be rejected by `load_boundary_cuts` against the
    /// current study's terminal manifest, naming the offending slot — the fail-loud
    /// catch of the silent wrong-bound mis-load the integer-only check misses.
    #[test]
    fn decomp_boundary_cuts_reject_permuted_entity_identity() {
        let case_dir = d28_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let comm = StubComm;

        let (mut setup_a, system_a) = build_setup(&case_dir, &config);
        let mut solver_a = ActiveSolver::new().expect("solver A");
        let outcome_a = setup_a
            .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
            .expect("train A");
        assert!(outcome_a.error.is_none(), "source training must not error");

        // D28 has one hydro, so the terminal manifest carries at least one storage
        // slot to permute.
        let current_manifest = setup_a.build_terminal_entity_manifest(&system_a);
        assert!(
            !current_manifest.is_empty(),
            "D28 must have at least one state slot for this test"
        );

        // Write the source checkpoint with a manifest whose slot 0 entity_id is
        // shifted away from the current study's, on every stage (the uniform cut
        // dimension makes per-stage manifests identical).
        let mut permuted = current_manifest.clone();
        permuted[0].entity_id += 1000;
        let n_pools = setup_a.fcf.pools.len();
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![permuted; n_pools];

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let source_policy_dir = tmpdir.path().join("source_policy");
        write_test_checkpoint_with_manifests(
            &source_policy_dir,
            &setup_a,
            &outcome_a.result,
            42,
            &stage_manifests,
        );

        let boundary_date = ascending_stage_end_dates(n_pools)[n_pools - 2];
        let state_dim = setup_a.fcf.state_dimension as u32;
        let result = cobre_sddp::load_boundary_cuts(&cobre_sddp::BoundaryLoadRequest::new(
            &source_policy_dir,
            boundary_date,
            state_dim,
            &current_manifest,
            1_000_000.0,
        ));

        assert!(
            result.is_err(),
            "a permuted slot-0 identity must be rejected, not silently loaded"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("does not price hydro"),
            "rejection must name the unpriced hydro: {msg}"
        );
        assert!(
            msg.contains("different set of plants"),
            "rejection must describe the identity mismatch (trained on a different set of plants): {msg}"
        );
    }
}

mod transit_seed_output {
    //! Integration coverage for the rolling-seed output (`simulation/transit_seed`):
    //! a declared upstream->downstream travel-time arc emits release windows
    //! matching the scenario's own realized turbined+spillage, riding the shared
    //! per-scenario writer for automatic CLI+Python parity.

    use chrono::{Duration, NaiveDate};
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::InflowModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroPastDefluence, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
        SimulationConfig as IoSimulationConfig, SimulationSelection, StoppingMode,
        StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
        UpperBoundEvaluationConfig,
    };
    use cobre_io::output::simulation_writer::{ScenarioWritePayload, SimulationParquetWriter};
    use cobre_sddp::SimulationScenarioResult;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{build_setup_in_code, run_simulation};

    const UPSTREAM_ID: i32 = 2;
    const DOWNSTREAM_ID: i32 = 1;
    const BUS_ID: i32 = 1;
    /// 2 stages of 24h each; a `t_v` at exactly this width stays inside the
    /// horizon (the `t_v <= H` case). The stitch (`t_v > H`) is unit-tested
    /// directly against the emitter in `simulation::extraction::transit_seed_tests`
    /// and the round-trip fidelity claim in
    /// `setup::transit_seed_round_trip_tests`, where a hand-built scenario gives
    /// full control over the horizon/`t_v` relationship with no solver dependency.
    const TRAVEL_TIME_HOURS: f64 = 48.0;

    fn study_start() -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
    }

    fn stages() -> Vec<Stage> {
        let start = study_start();
        (0..2_usize)
            .map(|i| {
                let s = start + Duration::days(i64::try_from(i).unwrap_or(0));
                make_stage(
                    i,
                    StageSpec {
                        start_date: s,
                        end_date: s + Duration::days(1),
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 24.0,
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
            .collect()
    }

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

    /// Two hydros on one bus: upstream `U` (id 2) with a mandatory 50 m³/s
    /// minimum outflow (a deterministic nonzero release every stage, independent
    /// of load economics) feeding downstream `D` (id 1). `with_arc` toggles the
    /// declared `downstream_id`/`travel_time_hours` (the inert, no-declared-arc case).
    fn build_system(with_arc: bool) -> System {
        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );
        let downstream = make_hydro(
            EntityId(DOWNSTREAM_ID),
            HydroSpec {
                bus_id: EntityId(BUS_ID),
                max_storage_hm3: 1_000.0,
                max_turbined_m3s: 200.0,
                max_generation_mw: 500.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );
        let upstream = make_hydro(
            EntityId(UPSTREAM_ID),
            HydroSpec {
                bus_id: EntityId(BUS_ID),
                downstream_id: with_arc.then_some(EntityId(DOWNSTREAM_ID)),
                travel_time_hours: with_arc.then_some(TRAVEL_TIME_HOURS),
                min_outflow_m3s: 50.0,
                max_storage_hm3: 1_000.0,
                max_turbined_m3s: 200.0,
                max_generation_mw: 500.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );

        let stages = stages();
        let n_stages = stages.len();
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(UPSTREAM_ID),
                stage_id: i32::try_from(i).unwrap_or(0),
                mean_m3s: 50.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 1_000.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 200.0,
                    max_generation_mw: 500.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds {
                    cost_per_mwh: 500.0,
                },
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
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
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

        let past_defluences = if with_arc {
            vec![HydroPastDefluence {
                hydro_id: EntityId(UPSTREAM_ID),
                start_date: study_start() - Duration::days(2),
                end_date: study_start(),
                value_m3s: 50.0,
            }]
        } else {
            vec![]
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![downstream, upstream])
            .stages(stages)
            .inflow_models(inflow_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(InitialConditions {
                storage: vec![
                    HydroStorage {
                        hydro_id: EntityId(DOWNSTREAM_ID),
                        value_hm3: 100.0,
                    },
                    HydroStorage {
                        hydro_id: EntityId(UPSTREAM_ID),
                        value_hm3: 100.0,
                    },
                ],
                past_defluences,
                ..InitialConditions::default()
            })
            .build()
            .expect("transit_seed_output: valid two-hydro cascade")
    }

    fn config(num_scenarios: u32) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: InflowNonNegativityMethod::Penalty,
                },
                cost_scale_factor: Some(1.0),
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: StoppingMode::Any,
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
                io_channel_capacity: 16,
                selection: Some(SimulationSelection::Sampled { num_scenarios }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    fn simulate_one(system: System) -> SimulationScenarioResult {
        let mut setup = build_setup_in_code(system, &config(1));
        let mut results = run_simulation(&mut setup, 1);
        assert_eq!(results.len(), 1, "exactly one scenario must be produced");
        results.remove(0)
    }

    /// A declared arc emits `transit_seed` windows keyed to the upstream
    /// hydro, each `value_m3s` matching the scenario's own realized release.
    #[test]
    fn declared_arc_emits_transit_seed_windows_for_the_upstream_hydro() {
        let scenario = simulate_one(build_system(true));

        assert!(
            !scenario.transit_seed.is_empty(),
            "a declared travel-time arc must emit transit_seed windows"
        );
        for window in &scenario.transit_seed {
            assert_eq!(
                window.hydro_id, UPSTREAM_ID,
                "every window must key on the arc's upstream hydro"
            );
            assert!(
                window.value_m3s >= 0.0,
                "a release rate must be non-negative, got {}",
                window.value_m3s
            );
        }
    }

    /// A study with no declared arc emits no `transit_seed` rows and the
    /// writer creates no `transit_seed` directory at all.
    #[test]
    fn no_declared_arc_leaves_transit_seed_empty_and_writer_creates_no_directory() {
        let scenario = simulate_one(build_system(false));
        assert!(
            scenario.transit_seed.is_empty(),
            "no declared arc must produce no transit_seed windows"
        );

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let mut writer = SimulationParquetWriter::new(tmp.path(), &build_system(false))
            .expect("SimulationParquetWriter::new must succeed");
        writer
            .write_scenario(ScenarioWritePayload::from(scenario))
            .expect("write_scenario must succeed");

        assert!(
            !tmp.path().join("simulation/transit_seed").exists(),
            "transit_seed/ must not exist for a study with no declared travel-time arc"
        );
    }

    /// Two independent runs of the identical fixture, converted and
    /// written through the exact production path both the CLI and Python
    /// bindings call (`ScenarioWritePayload::from` then
    /// `SimulationParquetWriter::write_scenario`), must produce byte-identical
    /// `transit_seed` Parquet — the shared-writer mechanism that makes
    /// CLI/Python parity automatic, mirroring
    /// `right_boundary_output.rs::identical_runs_produce_byte_identical_anticipated_lanes_parquet`.
    #[test]
    fn identical_runs_produce_byte_identical_transit_seed_parquet() {
        let write_once = |tmp_path: &std::path::Path| {
            let scenario = simulate_one(build_system(true));
            let mut writer = SimulationParquetWriter::new(tmp_path, &build_system(true))
                .expect("SimulationParquetWriter::new must succeed");
            writer
                .write_scenario(ScenarioWritePayload::from(scenario))
                .expect("write_scenario must succeed");
        };

        let tmp_a = tempfile::tempdir().expect("tempdir must succeed");
        let tmp_b = tempfile::tempdir().expect("tempdir must succeed");
        write_once(tmp_a.path());
        write_once(tmp_b.path());

        let rel = "simulation/transit_seed/scenario_id=0000/data.parquet";
        let path_a = tmp_a.path().join(rel);
        let path_b = tmp_b.path().join(rel);
        assert!(path_a.exists(), "transit_seed parquet must exist in run A");
        assert!(path_b.exists(), "transit_seed parquet must exist in run B");

        let bytes_a = std::fs::read(&path_a).expect("run A parquet must be readable");
        let bytes_b = std::fs::read(&path_b).expect("run B parquet must be readable");
        assert_eq!(
            bytes_a, bytes_b,
            "two independent runs of the identical fixture through the shared writer path \
             must produce byte-identical transit_seed output"
        );
    }
}

mod transit_seed_round_trip {
    //! End-to-end fidelity of the rolling-seed emitter: the windows a
    //! water-travel-time study emits, fed as a second study's own
    //! `past_defluences`, must reproduce the first study's true terminal
    //! in-transit bucket state. Exercises both the arc-fits-within-horizon
    //! regime and the leftover-seed regime, where the emitted windows must
    //! stitch in the run's own pre-study seed, plus the boundary-present
    //! keep-live behaviour the round trip's ground truth depends on.
    //!
    //! Ground truth for "the first study's true terminal in-transit state" is
    //! read with `config.policy.boundary` set on the same fixture: no cut is
    //! ever injected there, so it changes nothing about the optimal release
    //! (theta carries no cost pressure either way) — it only stops the
    //! terminal deep-lag slots from being masked to `0.0`, exposing the value
    //! the LP's own shift-ring equations already force deterministically.

    use std::collections::BTreeMap;

    use chrono::{Duration, NaiveDate};
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::InflowModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroPastDefluence, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        BoundaryPolicy, Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
        SimulationConfig as IoSimulationConfig, SimulationSelection, StoppingMode,
        StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
        UpperBoundEvaluationConfig,
    };
    use cobre_sddp::SimulationScenarioResult;
    use cobre_sddp::simulation::types::SimulationTransitSeedResult;
    use cobre_sddp::test_support::oracle_initial_state;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{build_setup_in_code, run_simulation};

    const BUS_ID: i32 = 1;
    const UPSTREAM_ID: i32 = 2;
    const DOWNSTREAM_ID: i32 = 1;
    const N_STAGES: usize = 2;
    const STAGE_HOURS: f64 = 24.0;
    const FORCED_RELEASE_M3S: f64 = 50.0;
    /// Storage level pinned identically (`min == max == initial`) on both
    /// hydros, so each one's water balance forces `outflow == inflow` exactly
    /// every stage — a hard column bound, never the soft min/max-outflow row
    /// pair (whose own violation slack is free at zero penalty and would
    /// leave the release economically undetermined).
    const PINNED_STORAGE_HM3: f64 = 1_000.0;
    const TOL: f64 = 1e-9;
    /// The receiving study's own stage count for the round-trip reconstruction:
    /// must be `>=` every plant's own run length in
    /// `StateSpace::transit_bucket_plants`, or `build_initial_transit_bucket_state`'s
    /// `StageCalendar` (built from the receiving study's own, un-padded
    /// stages) truncates `hour_window_shares`'s returned weight vector before
    /// it reaches the deepest lags — losing an in-study-release contribution
    /// that has nothing to do with the leftover-seed stitch. `8` comfortably
    /// exceeds every such run length this module's fixtures produce, so it
    /// isolates the stitch-specific behavior from this unrelated,
    /// receiver-sizing precondition.
    const SEED_RECEIVER_STAGES: usize = 8;

    fn stages_from(base: NaiveDate, n_stages: usize) -> Vec<Stage> {
        (0..n_stages)
            .map(|i| {
                let s = base + Duration::days(i64::try_from(i).unwrap_or(0));
                make_stage(
                    i,
                    StageSpec {
                        start_date: s,
                        end_date: s + Duration::days(1),
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: STAGE_HOURS,
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
            .collect()
    }

    fn zero_hydro_stage_penalties() -> HydroPenalties {
        HydroPenalties {
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
        }
    }

    /// A minimal upstream `U` -> downstream `J` cascade declaring a travel-time
    /// arc of `travel_time_hours`, its study starting at `base`, seeded from
    /// `past_defluences`. Both hydros' storage is pinned constant
    /// (`min_storage_hm3 == max_storage_hm3 == PINNED_STORAGE_HM3`), forcing
    /// `outflow == inflow` exactly, every stage, on a hard column bound —
    /// independent of the `config.policy.boundary` gate under test: no cut is
    /// ever injected, so theta carries no cost pressure that could change the
    /// release decision.
    fn build_system(
        base: NaiveDate,
        n_stages: usize,
        travel_time_hours: f64,
        past_defluences: Vec<HydroPastDefluence>,
    ) -> System {
        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let downstream = make_hydro(
            EntityId(DOWNSTREAM_ID),
            HydroSpec {
                bus_id: EntityId(BUS_ID),
                min_storage_hm3: PINNED_STORAGE_HM3,
                max_storage_hm3: PINNED_STORAGE_HM3,
                max_turbined_m3s: 500.0,
                max_generation_mw: 1_000.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );

        let upstream = make_hydro(
            EntityId(UPSTREAM_ID),
            HydroSpec {
                bus_id: EntityId(BUS_ID),
                downstream_id: Some(EntityId(DOWNSTREAM_ID)),
                travel_time_hours: Some(travel_time_hours),
                min_storage_hm3: PINNED_STORAGE_HM3,
                max_storage_hm3: PINNED_STORAGE_HM3,
                max_turbined_m3s: 500.0,
                max_generation_mw: 1_000.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );

        let stages = stages_from(base, n_stages);
        let n_stages = stages.len();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(UPSTREAM_ID),
                stage_id: i32::try_from(i).unwrap_or(0),
                mean_m3s: FORCED_RELEASE_M3S,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: PINNED_STORAGE_HM3,
                    max_storage_hm3: PINNED_STORAGE_HM3,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 500.0,
                    max_generation_mw: 1_000.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds {
                    cost_per_mwh: 500.0,
                },
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
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![downstream, upstream])
            .stages(stages)
            .inflow_models(inflow_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(InitialConditions {
                storage: vec![
                    HydroStorage {
                        hydro_id: EntityId(DOWNSTREAM_ID),
                        value_hm3: PINNED_STORAGE_HM3,
                    },
                    HydroStorage {
                        hydro_id: EntityId(UPSTREAM_ID),
                        value_hm3: PINNED_STORAGE_HM3,
                    },
                ],
                past_defluences,
                ..InitialConditions::default()
            })
            .build()
            .expect("transit_seed_round_trip: valid two-hydro cascade")
    }

    fn config(boundary_on: bool) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: InflowNonNegativityMethod::Penalty,
                },
                cost_scale_factor: Some(1.0),
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig {
                boundary: boundary_on.then(|| BoundaryPolicy {
                    path: "unused".to_string(),
                    strict: false,
                }),
                ..PolicyConfig::default()
            },
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 16,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// Train + simulate one scenario; `boundary_on` toggles `config.policy.
    /// boundary` (no boundary cut is ever injected — only the gate matters).
    fn simulate_one(system: System, boundary_on: bool) -> SimulationScenarioResult {
        let mut setup = build_setup_in_code(system, &config(boundary_on));
        let mut results = run_simulation(&mut setup, 1);
        assert_eq!(results.len(), 1, "exactly one scenario must be produced");
        results.remove(0)
    }

    /// The terminal stage's in-transit bucket state for the downstream plant,
    /// keyed by lag.
    fn terminal_bucket_state(scenario: &SimulationScenarioResult) -> BTreeMap<u32, f64> {
        let terminal = scenario
            .stages
            .last()
            .expect("scenario must have at least one stage");
        terminal
            .transit_buckets
            .iter()
            .map(|b| (b.lag, b.in_transit_volume_hm3))
            .collect()
    }

    /// Build a fresh study seeded from `past_defluences`, starting at `base`
    /// (never trained or simulated — the seed lands during `StudySetup::new`,
    /// before any solve), and read its stage-0 in-transit bucket vector by lag.
    fn stage_0_seed(
        base: NaiveDate,
        n_stages: usize,
        travel_time_hours: f64,
        past_defluences: Vec<HydroPastDefluence>,
    ) -> BTreeMap<u32, f64> {
        let system = build_system(base, n_stages, travel_time_hours, past_defluences);
        let setup = build_setup_in_code(system, &config(false));
        let state = setup.stage_state();
        let initial_state = oracle_initial_state(&setup);
        state
            .transit_bucket_column_order
            .iter()
            .enumerate()
            .map(|(pos, &(_, lag))| {
                (
                    u32::try_from(lag).unwrap_or(0),
                    initial_state[state.transit_buckets_out.start + pos],
                )
            })
            .collect()
    }

    fn to_past_defluences(windows: &[SimulationTransitSeedResult]) -> Vec<HydroPastDefluence> {
        windows
            .iter()
            .map(|w| HydroPastDefluence {
                hydro_id: EntityId(w.hydro_id),
                start_date: w.start_date,
                end_date: w.end_date,
                value_m3s: w.value_m3s,
            })
            .collect()
    }

    fn assert_maps_close(
        actual: &BTreeMap<u32, f64>,
        expected: &BTreeMap<u32, f64>,
        context: &str,
    ) {
        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "{context}: lag key sets must match"
        );
        for (&lag, &expected_value) in expected {
            let actual_value = actual[&lag];
            assert!(
                (actual_value - expected_value).abs() < TOL,
                "{context}: lag {lag} mismatch: actual={actual_value}, expected={expected_value}"
            );
        }
    }

    /// `t_v <= H` — the arc's travel time fits entirely within a single
    /// study's own horizon, so the emitted windows are all in-study (no
    /// leftover-seed stitch is exercised here; the `t_v > H` case below is).
    #[test]
    fn round_trip_continuity_holds_when_travel_time_fits_within_horizon() {
        let study_1_start = NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date");
        let travel_time_hours = STAGE_HOURS; // 24h <= H = 48h.

        let system_1 = build_system(study_1_start, N_STAGES, travel_time_hours, vec![]);
        let scenario_1 = simulate_one(system_1, true);

        let ground_truth = terminal_bucket_state(&scenario_1);
        assert!(
            ground_truth.values().any(|&v| v.abs() > 1e-6),
            "fixture has no power unless at least one terminal in-transit lag is nonzero"
        );

        let study_2_start = study_1_start + Duration::days(i64::try_from(N_STAGES).unwrap_or(0));
        let past_defluences = to_past_defluences(&scenario_1.transit_seed);
        let seeded = stage_0_seed(
            study_2_start,
            SEED_RECEIVER_STAGES,
            travel_time_hours,
            past_defluences,
        );

        assert_maps_close(&seeded, &ground_truth, "t_v <= H round trip");
    }

    /// `t_v > H` — the arc's travel time exceeds the study's own
    /// horizon, so part of the terminal in-transit state is leftover mass
    /// that entered before the study started (study 1's own
    /// `past_defluences`), not from any in-study release.
    ///
    /// Documented scope boundary: the rolling seed is faithful
    /// only for `t_v <= horizon`; the `t_v > horizon` deep in-transit mass beyond
    /// the horizon is not represented — the "Terminal credit deferred" imprecision
    /// (see `sddp.md`) surfacing at the rolling seam. This test pins the
    /// reproduction for the day that limitation is lifted; it is `#[ignore]`d, not
    /// weakened. Root cause: `build_initial_transit_bucket_state` derives its
    /// `StageCalendar` from the CURRENT study's own (un-padded)
    /// stage list, so `hour_window_shares` can never populate a lag deeper
    /// than that study's own stage count — for ANY declared arc whose
    /// `t_v > H` (by definition, that plant's `transit_bucket_plants` run
    /// is longer than `n_stages` whenever stage widths are uniform), a wide
    /// pre-study window's seed is silently
    /// truncated at both ends of the rolling seam: study 1 itself can only
    /// ever seed the window's first `n_stages` lags (delivering, not
    /// carrying, the remainder before the terminal), while re-emitting that
    /// SAME un-sliced window verbatim as study 2's `past_defluences` (today's
    /// stitch behaviour) re-populates lags study 1 already delivered. The
    /// mismatch below lands exactly on those shallow lags; the deeper lags
    /// (populated by each study's own in-study release, which resolves
    /// through a padded calendar and has no such truncation) match exactly —
    /// isolating the gap to the pre-study leftover-seed path specifically.
    #[test]
    #[ignore = "known gap: build_initial_transit_bucket_state's un-padded \
                StageCalendar truncates a pre-study window's reach to the \
                study's own stage count, so re-emitting that window verbatim \
                across the rolling seam double-represents already-delivered \
                mass at the shallow lags whenever a declared arc's travel \
                time exceeds the study's own horizon; see the module/function \
                doc above for the isolated reproduction"]
    fn round_trip_continuity_needs_the_leftover_seed_stitch_when_travel_time_exceeds_horizon() {
        let study_1_start = NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date");
        let travel_time_hours = 4.0 * STAGE_HOURS; // 96h > H = 48h.

        let leftover_seed = vec![HydroPastDefluence {
            hydro_id: EntityId(UPSTREAM_ID),
            start_date: study_1_start - Duration::days(4),
            end_date: study_1_start,
            value_m3s: 80.0,
        }];

        let system_1 = build_system(study_1_start, N_STAGES, travel_time_hours, leftover_seed);
        let scenario_1 = simulate_one(system_1, true);

        let ground_truth = terminal_bucket_state(&scenario_1);
        assert!(
            ground_truth.values().any(|&v| v.abs() > 1e-6),
            "fixture has no power unless at least one terminal in-transit lag is nonzero"
        );

        let real_windows = &scenario_1.transit_seed;
        assert!(
            real_windows.iter().any(|w| w.end_date <= study_1_start),
            "fixture has no power on the stitch unless the emitter actually includes a \
             pre-study window in its output"
        );

        let study_2_start = study_1_start + Duration::days(i64::try_from(N_STAGES).unwrap_or(0));

        let seeded = stage_0_seed(
            study_2_start,
            SEED_RECEIVER_STAGES,
            travel_time_hours,
            to_past_defluences(real_windows),
        );
        assert_maps_close(&seeded, &ground_truth, "t_v > H round trip (with stitch)");

        let naive_windows: Vec<SimulationTransitSeedResult> = real_windows
            .iter()
            .filter(|w| w.end_date > study_1_start)
            .cloned()
            .collect();
        assert!(
            naive_windows.len() < real_windows.len(),
            "the naive in-study-only reconstruction must actually drop the stitched window(s), \
             or this fixture has no power over the stitch"
        );
        let naive_seeded = stage_0_seed(
            study_2_start,
            SEED_RECEIVER_STAGES,
            travel_time_hours,
            to_past_defluences(&naive_windows),
        );

        let naive_gap: f64 = ground_truth
            .keys()
            .map(|lag| (naive_seeded.get(lag).copied().unwrap_or(0.0) - ground_truth[lag]).abs())
            .sum();
        assert!(
            naive_gap > 1e-3,
            "dropping the leftover-seed stitch must visibly diverge from ground truth \
             (naive reconstruction gap: {naive_gap}); otherwise this fixture has no power \
             over the stitch"
        );
    }

    /// With `config.policy.boundary` set, the terminal
    /// deep-lag bucket slots stay live and nonzero exactly where the
    /// gated-off build masks them to `0.0`, and the solve completes with no
    /// infeasibility.
    #[test]
    fn boundary_present_keeps_terminal_bucket_state_live_past_the_horizon() {
        let study_1_start = NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date");
        let travel_time_hours = 4.0 * STAGE_HOURS; // 96h > H = 48h.
        let leftover_seed = vec![HydroPastDefluence {
            hydro_id: EntityId(UPSTREAM_ID),
            start_date: study_1_start - Duration::days(4),
            end_date: study_1_start,
            value_m3s: 80.0,
        }];

        let scenario_off = simulate_one(
            build_system(
                study_1_start,
                N_STAGES,
                travel_time_hours,
                leftover_seed.clone(),
            ),
            false,
        );
        let scenario_on = simulate_one(
            build_system(study_1_start, N_STAGES, travel_time_hours, leftover_seed),
            true,
        );

        let masked = terminal_bucket_state(&scenario_off);
        let live = terminal_bucket_state(&scenario_on);

        assert!(
            masked.values().all(|&v| v == 0.0),
            "gated-off (pre-boundary) build must mask every terminal deep-lag slot to 0.0, got {masked:?}"
        );
        assert!(
            !live.is_empty() && live.values().any(|&v| v.abs() > 1e-6),
            "boundary-present build must keep at least one terminal deep-lag slot live and \
             nonzero, got {live:?}"
        );
        assert_eq!(
            masked.keys().collect::<Vec<_>>(),
            live.keys().collect::<Vec<_>>(),
            "the gate must not change which lags are structurally reachable, only whether \
             they are masked"
        );
    }
}

mod diversion_outflow_bounds {
    //! Both per-hydro outflow bounds bind the NON-DIVERTED river-remnant flow
    //! `q + s`, never `q + s + d`. The minimum reproduces the Belo Monte / Volta
    //! Grande defect (a defluência-mínima must be routed down the plant's own
    //! channel, not satisfied by the diversion canal); the maximum is its mirror
    //! (a defluência-máxima caps the natural channel, while the diversion canal —
    //! bounded separately — carries the surplus past it).

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{DiversionChannel, HydroGenerationModel};
    use cobre_core::scenario::InflowModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
        SimulationConfig as IoSimulationConfig, SimulationSelection, StoppingMode,
        StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
        UpperBoundEvaluationConfig,
    };
    use cobre_sddp::SimulationScenarioResult;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{build_setup_in_code, run_simulation};

    const SOURCE_ID: i32 = 1;
    const TARGET_ID: i32 = 2;
    const BUS_ID: i32 = 1;
    /// The defluência-mínima on the source's natural reach (Volta Grande analog).
    const MIN_OUTFLOW_M3S: f64 = 400.0;
    /// The defluência-máxima on the source's natural reach; below the inflow, so
    /// the surplus must leave through the diversion canal past the cap.
    const MAX_OUTFLOW_M3S: f64 = 300.0;
    /// Turbine cap below the floor, so meeting `q + s >= MIN_OUTFLOW` forces spill.
    const MAX_TURBINED_M3S: f64 = 100.0;
    /// Diversion capacity large enough to absorb every surplus m³/s.
    const DIVERSION_CAP_M3S: f64 = 900.0;
    /// Deterministic run-of-river inflow (std 0): the source must release exactly
    /// this every block, split among turbine / spill / diversion.
    const INFLOW_M3S: f64 = 500.0;

    /// Which outflow bound the fixture exercises. Both bind `q + s`; each needs a
    /// cost structure that would push the diverted flow across the bound if the
    /// diversion column were (wrongly) coupled into the row.
    #[derive(Clone, Copy)]
    enum Bound {
        /// Diversion is cheaper than spill, so the LP would meet the floor with
        /// diversion if `d` counted toward the minimum.
        Min,
        /// Diversion is dearer than spill, so the LP pushes `q + s` up to the cap
        /// and routes only the forced surplus through the canal — the canal flow
        /// would evade the maximum if `d` counted toward it.
        Max,
    }

    fn study_start() -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
    }

    fn stages() -> Vec<Stage> {
        let start = study_start();
        (0..2_usize)
            .map(|i| {
                let s = start + chrono::Duration::days(30 * i64::try_from(i).unwrap_or(0));
                make_stage(
                    i,
                    StageSpec {
                        start_date: s,
                        end_date: s + chrono::Duration::days(30),
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
                )
            })
            .collect()
    }

    /// `Min`: spill dearer than diversion, so absent the floor the LP would route
    /// the surplus through the cheaper canal, leaving `q + s` below `MIN_OUTFLOW`.
    /// `Max`: spill cheaper than diversion, so the LP fills the natural channel to
    /// the cap and diverts only the forced surplus. The high violation costs make
    /// each bound bind rather than be paid off.
    fn hydro_penalties(bound: Bound) -> HydroPenalties {
        let (spillage_cost, diversion_cost) = match bound {
            Bound::Min => (0.5, 0.0),
            Bound::Max => (0.01, 10.0),
        };
        HydroPenalties {
            spillage_cost,
            diversion_cost,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 1000.0,
            outflow_violation_above_cost: 1000.0,
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

    /// Source `S` (id 1) is run-of-river with a `MIN_OUTFLOW_M3S` floor and a
    /// diversion channel to target `T` (id 2). `T` is run-of-river with turbine
    /// headroom to absorb the whole diverted flow. Neither generates (default
    /// zero productivity): the operative economics are spill-vs-divert cost, which
    /// is all the floor semantics need to be exercised.
    fn build_system(bound: Bound) -> System {
        let (entity_min, entity_max) = match bound {
            Bound::Min => (MIN_OUTFLOW_M3S, None),
            Bound::Max => (0.0, Some(MAX_OUTFLOW_M3S)),
        };
        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );
        // Source: id 1 → canonical position 0 (smallest id, shared start date);
        // asserted below before the per-hydro bounds override is trusted.
        let source = make_hydro(
            EntityId(SOURCE_ID),
            HydroSpec {
                bus_id: EntityId(BUS_ID),
                min_outflow_m3s: entity_min,
                max_outflow_m3s: entity_max,
                max_turbined_m3s: MAX_TURBINED_M3S,
                max_generation_mw: 500.0,
                diversion: Some(DiversionChannel {
                    downstream_id: EntityId(TARGET_ID),
                    max_flow_m3s: DIVERSION_CAP_M3S,
                }),
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );
        let target = make_hydro(
            EntityId(TARGET_ID),
            HydroSpec {
                bus_id: EntityId(BUS_ID),
                max_turbined_m3s: DIVERSION_CAP_M3S,
                max_generation_mw: 500.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );

        let stages = stages();
        let n_stages = stages.len();
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(SOURCE_ID),
                stage_id: i32::try_from(i).unwrap_or(0),
                mean_m3s: INFLOW_M3S,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
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
                    max_turbined_m3s: DIVERSION_CAP_M3S,
                    max_generation_mw: 500.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds {
                    cost_per_mwh: 500.0,
                },
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
        // Source-only resolved overrides at canonical position 0 (the exercised
        // outflow bound, the diversion cap, and the sub-inflow turbine cap); the
        // target keeps the defaults (no bounds, no diversion column).
        for stage_idx in 0..n_stages {
            let src = bounds.hydro_block_base_mut(0, stage_idx);
            src.max_turbined_m3s = MAX_TURBINED_M3S;
            src.max_diversion_m3s = Some(DIVERSION_CAP_M3S);
            match bound {
                Bound::Min => src.min_outflow_m3s = MIN_OUTFLOW_M3S,
                Bound::Max => src.max_outflow_m3s = Some(MAX_OUTFLOW_M3S),
            }
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: hydro_penalties(bound),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![source, target])
            .stages(stages)
            .inflow_models(inflow_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(InitialConditions {
                storage: vec![
                    HydroStorage {
                        hydro_id: EntityId(SOURCE_ID),
                        value_hm3: 0.0,
                    },
                    HydroStorage {
                        hydro_id: EntityId(TARGET_ID),
                        value_hm3: 0.0,
                    },
                ],
                ..InitialConditions::default()
            })
            .build()
            .expect("diversion_outflow_bounds: valid two-hydro diverting system");

        assert_eq!(
            system.hydros()[0].id,
            EntityId(SOURCE_ID),
            "the source must occupy canonical position 0 for the per-hydro bounds override"
        );
        system
    }

    fn config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: InflowNonNegativityMethod::Penalty,
                },
                cost_scale_factor: Some(1.0),
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: StoppingMode::Any,
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
                io_channel_capacity: 16,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// The floor forces `q + s >= MIN_OUTFLOW` down the source's own channel even
    /// though diverting the surplus is strictly cheaper. Under the pre-fix
    /// coefficient set (`q + s + d >= MIN_OUTFLOW`) the LP would meet the floor
    /// with diversion, leaving `q + s = q = MAX_TURBINED_M3S < MIN_OUTFLOW` at
    /// zero below-slack — the silent Volta Grande under-release.
    #[test]
    fn min_outflow_binds_the_non_diverted_flow_on_a_diverter() {
        let mut setup = build_setup_in_code(build_system(Bound::Min), &config());
        let results = run_simulation(&mut setup, 1);
        assert_eq!(results.len(), 1, "exactly one scenario must be produced");
        let scenario: &SimulationScenarioResult = &results[0];

        let source_blocks: Vec<_> = scenario
            .stages
            .iter()
            .flat_map(|s| &s.hydros)
            .filter(|h| h.hydro_id == SOURCE_ID)
            .collect();
        assert!(
            !source_blocks.is_empty(),
            "the source must appear in the simulation output"
        );

        for h in source_blocks {
            let non_diverted = h.turbined_m3s + h.spillage_m3s;
            assert!(
                non_diverted >= MIN_OUTFLOW_M3S - 1e-3,
                "non-diverted outflow (q + s = {non_diverted}) must meet the {MIN_OUTFLOW_M3S} \
                 floor; including diversion in the floor would let it drop to \
                 turbined={} with zero slack",
                h.turbined_m3s
            );
            assert!(
                h.outflow_slack_below_m3s < 1e-3,
                "the floor is met without a below-violation, got slack {}",
                h.outflow_slack_below_m3s
            );
            // Non-vacuous: the plant genuinely diverts, so the floor is satisfied
            // by `q + s` DESPITE water leaving through the canal — not because no
            // diversion occurs.
            let diverted = h
                .diverted_outflow_m3s
                .expect("a diverting plant reports its diverted outflow");
            assert!(
                diverted > 1e-3,
                "the source must actively divert (else the test is vacuous), got {diverted}"
            );
        }
    }

    /// The cap binds `q + s <= MAX_OUTFLOW` on the source's own channel while the
    /// diversion canal carries the forced surplus past it (total release
    /// `q + s + d = INFLOW > MAX_OUTFLOW`), with zero above-violation. Under the
    /// pre-symmetry coefficient set (`q + s + d <= MAX_OUTFLOW`) the run-of-river
    /// balance would instead force an above-slack of `INFLOW − MAX_OUTFLOW` and no
    /// diversion — the total-release cap this change removes.
    #[test]
    fn max_outflow_binds_the_non_diverted_flow_on_a_diverter() {
        let mut setup = build_setup_in_code(build_system(Bound::Max), &config());
        let results = run_simulation(&mut setup, 1);
        assert_eq!(results.len(), 1, "exactly one scenario must be produced");
        let scenario: &SimulationScenarioResult = &results[0];

        let source_blocks: Vec<_> = scenario
            .stages
            .iter()
            .flat_map(|s| &s.hydros)
            .filter(|h| h.hydro_id == SOURCE_ID)
            .collect();
        assert!(
            !source_blocks.is_empty(),
            "the source must appear in the simulation output"
        );

        for h in source_blocks {
            let non_diverted = h.turbined_m3s + h.spillage_m3s;
            assert!(
                non_diverted <= MAX_OUTFLOW_M3S + 1e-3,
                "non-diverted outflow (q + s = {non_diverted}) must honor the {MAX_OUTFLOW_M3S} \
                 cap on the natural channel"
            );
            assert!(
                h.outflow_slack_above_m3s < 1e-3,
                "the cap is met without an above-violation, got slack {}; including diversion \
                 in the maximum would force it to {} on this run-of-river balance",
                h.outflow_slack_above_m3s,
                INFLOW_M3S - MAX_OUTFLOW_M3S
            );
            let diverted = h
                .diverted_outflow_m3s
                .expect("a diverting plant reports its diverted outflow");
            // Non-vacuous: the canal carries the surplus past the cap, so total
            // release exceeds MAX_OUTFLOW even though `q + s` honors it — which is
            // impossible if `d` counts toward the maximum.
            assert!(
                diverted > 1e-3,
                "the source must actively divert (else the test is vacuous), got {diverted}"
            );
            assert!(
                non_diverted + diverted > MAX_OUTFLOW_M3S + 1e-3,
                "total release (q + s + d = {}) must exceed the {MAX_OUTFLOW_M3S} cap via the \
                 diversion canal, proving the maximum bounds only the non-diverted flow",
                non_diverted + diverted
            );
        }
    }
}

mod water_arc_and_post_study_anticipated_coexist_on_extended_layout {
    //! The water in-transit bucket ring and the anticipated-commitment ring
    //! are the two `DeliveryRing` instantiations, and `commit_out`/`commit_in`
    //! sit ahead of every water incoming column in the state layout
    //! (`storage -> inflow_lags -> transit_buckets_out -> commit_out ->
    //! z_inflow -> storage_in -> transit_buckets_in -> commit_in -> theta`).
    //! No existing fixture combines a water travel-time arc, a post-study
    //! anticipated delivery, and a boundary policy on one deck, so nothing
    //! catches an off-by-region incoming-column resolution, a broken
    //! per-arc `k`-weight conservation identity, a per-family terminal
    //! pricing arm, or a basis-rejecting warm start under this specific
    //! two-family width shift. This module pins all four against the deck.

    use std::path::Path;

    use chrono::{Duration, NaiveDate};
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::entities::thermal::AnticipatedConfig;
    use cobre_core::scenario::InflowModel;
    use cobre_core::temporal::{Block, Stage};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, HydroBlockBounds,
        HydroPastDefluence, HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds,
        PostStudyStage, PostStudyStages, PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds,
        System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        BoundaryPolicy, Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig,
        TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::indexer::{CutStateProjection, StateDim};
    use cobre_sddp::setup::{NodeId, StageIdx};
    use cobre_sddp::test_support::{patch_backward_opening_for_probe, solve_stage_for_probe};
    use cobre_sddp::workspace::SolverWorkspace;
    use cobre_sddp::{
        BoundaryLoadRequest, SolverStatsDelta, inject_boundary_cuts, load_boundary_cuts,
    };
    use cobre_solver::{
        ActiveSolver, FreezeScratch, RowBatch, SolverInterface, StageTemplate,
        freeze_rows_into_template,
    };

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    const BUS_ID: EntityId = EntityId(1);
    const THERMAL_ID: EntityId = EntityId(2);
    const DOWNSTREAM_ID: EntityId = EntityId(3);
    const UPSTREAM_ID: EntityId = EntityId(4);
    const N_STAGES: usize = 2;
    const N_TRAIN_ITERATIONS: u32 = 8;

    /// Wide enough that every anchor's own release reaches TWO stages ahead
    /// (`stage_reach == 2`, a lag-1 AND a lag-2 bucket), so lag 1's outgoing
    /// definition genuinely depends on lag 2's incoming state
    /// (`b_1^out = b_2^in + k_1*D`) — the shift dependency property 4 pins
    /// via a delta comparison; a depth-1 ring's sole lag has no incoming
    /// dependency at all (`b_1^out = k_1*D` alone), so it cannot demonstrate
    /// this. Still well short of the study's own 1464h horizon —
    /// coexistence with the anticipated ring, not an interaction with the
    /// `t_v > horizon` rolling-seed limitation.
    const TRAVEL_TIME_HOURS: f64 = 900.0;
    /// `LeadTime` delta (hours): resolves the thermal's own stage-1 delivery
    /// to a pre-study decider (dormant) while the post-study target resolves
    /// to a stage-0 decider, ring-sizing `k_max == 2` — the same fixture
    /// shape `right_boundary_pricing.rs` pins for the anticipated family
    /// alone.
    const DELTA_HOURS: f64 = 1500.0;
    const POST_STUDY_HOURS: f64 = 720.0;
    const ALPHA: f64 = 5.0;
    const BETA_WATER: f64 = 3.0;
    const BETA_ANT: f64 = 4.0;
    const TOL: f64 = 1e-6;

    fn study_start() -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
    }

    /// Study end date (stage 1 end) — the first `post_study_stages` stage and
    /// the post-horizon delivery window both anchor here.
    fn study_end() -> NaiveDate {
        study_start() + Duration::days(61)
    }

    fn stages() -> Vec<Stage> {
        let start = study_start();
        let stage0_end = start + Duration::days(31);
        vec![
            make_stage(
                0,
                StageSpec {
                    start_date: start,
                    end_date: stage0_end,
                    blocks: vec![Block {
                        index: 0,
                        name: "S0".to_string(),
                        duration_hours: 744.0,
                    }],
                    ..Default::default()
                },
            ),
            make_stage(
                1,
                StageSpec {
                    start_date: stage0_end,
                    end_date: study_end(),
                    blocks: vec![Block {
                        index: 0,
                        name: "S1".to_string(),
                        duration_hours: 720.0,
                    }],
                    ..Default::default()
                },
            ),
        ]
    }

    /// Per-stage total hours, mirroring `DeliveryCalendar::study_total_hours`
    /// for the [`resolve_spread`](cobre_sddp::lead_time::resolve_spread) calls
    /// property 3 drives directly.
    fn study_durations() -> Vec<f64> {
        stages()
            .iter()
            .map(|s| s.blocks.iter().map(|b| b.duration_hours).sum())
            .collect()
    }

    /// The one declared post-study stage: the span the anticipated
    /// `LeadTime(DELTA_HOURS)` thermal's post-study-targeted delivery resolves
    /// onto, so `StageCalendar::resolve_window` covers it at destination index 0.
    fn post_study_stages() -> PostStudyStages {
        PostStudyStages {
            stages: vec![PostStudyStage {
                start_date: study_end(),
                duration_hours: POST_STUDY_HOURS,
            }],
            thermal_bounds: vec![PostStudyThermalBound {
                thermal_id: THERMAL_ID,
                post_study_stage_index: 0,
                cost_per_mwh: 37.5,
                min_mw: 0.0,
                max_mw: 100.0,
            }],
        }
    }

    fn bounds() -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 1,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 1_000.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 200.0,
                    max_generation_mw: 500.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 1.0 },
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
        )
    }

    fn penalties() -> cobre_core::resolved::ResolvedPenalties {
        use cobre_core::HydroPenalties;
        use cobre_core::resolved::{
            BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
            PenaltiesDefaults, ResolvedPenalties,
        };
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
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
                    inflow_nonnegativity_cost: 1_000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// The combined deck: a water travel-time arc (`UPSTREAM_ID` ->
    /// `DOWNSTREAM_ID`), a `LeadTime` anticipated thermal with a post-study
    /// target, and the `post_study_stages` calendar that target resolves
    /// into. Reuses the water-arc `HydroSpec` shape from
    /// `transit_seed_output` and the anticipated/post-study shape from
    /// `right_boundary_pricing.rs`.
    fn build_system() -> System {
        let bus = make_bus(BUS_ID, BusSpec::default());

        let downstream = make_hydro(
            DOWNSTREAM_ID,
            HydroSpec {
                bus_id: BUS_ID,
                max_storage_hm3: 1_000.0,
                max_turbined_m3s: 200.0,
                max_generation_mw: 500.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );
        let upstream = make_hydro(
            UPSTREAM_ID,
            HydroSpec {
                bus_id: BUS_ID,
                downstream_id: Some(DOWNSTREAM_ID),
                travel_time_hours: Some(TRAVEL_TIME_HOURS),
                min_outflow_m3s: 50.0,
                max_storage_hm3: 1_000.0,
                max_turbined_m3s: 200.0,
                max_generation_mw: 500.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );
        let thermal = make_thermal(
            THERMAL_ID,
            ThermalSpec {
                bus_id: BUS_ID,
                cost_per_mwh: 1.0,
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                anticipated_config: Some(AnticipatedConfig::LeadTime(DELTA_HOURS)),
                ..Default::default()
            },
        );

        let inflow_models: Vec<InflowModel> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id: UPSTREAM_ID,
                stage_id: i32::try_from(i).unwrap_or(0),
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let initial_conditions = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: DOWNSTREAM_ID,
                    value_hm3: 100.0,
                },
                HydroStorage {
                    hydro_id: UPSTREAM_ID,
                    value_hm3: 100.0,
                },
            ],
            past_defluences: vec![HydroPastDefluence {
                hydro_id: UPSTREAM_ID,
                // 38 days (912h) over-covers TRAVEL_TIME_HOURS (900h, not a
                // whole day count); the windowed IC seed only requires the
                // window to COVER [start_0 - t_v, start_0), not match it
                // exactly.
                start_date: study_start() - Duration::days(38),
                end_date: study_start(),
                value_m3s: 50.0,
            }],
            ..Default::default()
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![downstream, upstream])
            .thermals(vec![thermal])
            .stages(stages())
            .inflow_models(inflow_models)
            .bounds(bounds())
            .penalties(penalties())
            .initial_conditions(initial_conditions)
            .post_study_stages(Some(post_study_stages()))
            .build()
            .expect("combined deck: water arc + post-study anticipated thermal must build")
    }

    fn config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: InflowNonNegativityMethod::Penalty,
                },
                // Unscaled: the priced `theta` and the propagated cut
                // coefficient compare directly against ALPHA/BETA_* with no
                // COST_SCALE_FACTOR back-out.
                cost_scale_factor: Some(1.0),
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: N_TRAIN_ITERATIONS,
                }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig {
                boundary: Some(BoundaryPolicy {
                    path: "unused-boundary-checkpoint".to_string(),
                    strict: false,
                }),
                ..PolicyConfig::default()
            },
            simulation: SimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// State-vector index of the ring slot carrying the post-study-targeted
    /// delivery: `commitment_hold_in_study_offset(plant, m)` for the sole
    /// anticipated plant and delivery target `m = num_stages()` (the first
    /// post-study stage) — mirrors `StateSpace::commitment_hold_in_study_offset`
    /// byte-for-byte (`pub(crate)`, unreachable from this integration-test
    /// binary), never a hand-rolled `commit_out.end`-relative guess.
    fn post_study_ring_slot(setup: &cobre_sddp::StudySetup) -> usize {
        let state = setup.stage_state();
        let m = setup.num_stages();
        state.commit_out.start + (m % state.k_max) * state.n_anticipated
    }

    /// State-vector index of the water arc's lag-1 bucket — the slot property
    /// 4 prices with `BETA_WATER`. Masked at the terminal stage without a
    /// declared boundary, kept live with one (the "Delivery-family
    /// right-boundary pricing" contract). Its OUTGOING definition is
    /// `b_1^out = b_2^in + k_1*D` ([`water_pin_slot`]'s incoming state plus
    /// this stage's own deposit share).
    fn water_priced_slot(setup: &cobre_sddp::StudySetup) -> usize {
        let state = setup.stage_state();
        assert_eq!(
            state.n_buckets, 2,
            "fixture must declare exactly two water bucket lags (lag 1 and lag 2)"
        );
        state.transit_buckets_out.start
    }

    /// State-vector index of the water arc's lag-2 (deepest) bucket — pinning
    /// its INCOMING column controls [`water_priced_slot`]'s outgoing value by
    /// the shift row `b_1^out = b_2^in + k_1*D`, up to the deposit share
    /// `k_1*D` (identical across pin values, so it cancels in a delta
    /// comparison). Unlike lag 1, lag 2's own outgoing has no incoming
    /// dependency (`b_2^out = k_2*D` alone, the deepest lag), so it is not
    /// itself pin-controllable — only usable as the CONTROL for lag 1.
    fn water_pin_slot(setup: &cobre_sddp::StudySetup) -> usize {
        let state = setup.stage_state();
        assert_eq!(
            state.n_buckets, 2,
            "fixture must declare exactly two water bucket lags (lag 1 and lag 2)"
        );
        state.transit_buckets_out.start + 1
    }

    /// Pin the full state vector at zero except `entries`.
    fn pin_state(setup: &cobre_sddp::StudySetup, entries: &[(usize, f64)]) -> Vec<f64> {
        let mut pin = vec![0.0_f64; setup.stage_state().n_state];
        for &(slot, v) in entries {
            pin[slot] = v;
        }
        pin
    }

    /// Pool `pool`'s fixture `priced_state_date`: `2030-01-01` plus `pool`
    /// months.
    fn fixture_priced_date(pool: u32) -> NaiveDate {
        cobre_sddp::test_support::fixture_priced_date(
            cobre_sddp::test_support::ymd(2030, 1, 1),
            pool,
        )
    }

    /// Load a boundary carrying `BETA_WATER` on `water_priced_slot` and
    /// `BETA_ANT` on `ant_slot`, zero elsewhere, intercept `ALPHA`, and
    /// inject it into `setup`'s terminal pool.
    fn inject_two_family_boundary(
        setup: &mut cobre_sddp::StudySetup,
        dir: &Path,
        water_priced_slot: usize,
        ant_slot: usize,
    ) {
        let state_dimension = setup.fcf.state_dimension as u32;
        let mut coefficients = vec![0.0_f64; state_dimension as usize];
        coefficients[water_priced_slot] = BETA_WATER;
        coefficients[ant_slot] = BETA_ANT;
        cobre_sddp::test_support::write_synthetic_boundary(
            dir,
            state_dimension,
            ALPHA,
            &coefficients,
            fixture_priced_date(0),
        );

        let boundary_cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            dir,
            fixture_priced_date(0),
            state_dimension,
            &[],
            1.0,
        ))
        .expect("boundary cut must load");
        inject_boundary_cuts(setup, &boundary_cuts).unwrap();
    }

    /// Build the terminal pool's frozen LP template: the base structural
    /// template plus one literal row per active cut in the pool. The terminal
    /// pool always projects the full global state (`build_cut_state_layouts`),
    /// so the all-enabled projection reproduces exactly the coefficients the
    /// boundary cut carries, regardless of this study's own per-stage
    /// `StageStateConfig`.
    fn freeze_terminal_template(setup: &cobre_sddp::StudySetup, pool_id: usize) -> StageTemplate {
        let state = setup.stage_state();
        let terminal_stage = setup.num_stages() - 1;
        let ctx = setup.stage_ctx();
        let base = &ctx.templates[terminal_stage];

        let cut_state = CutStateProjection::new(
            state,
            cobre_core::temporal::StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
        );

        let mut batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        cobre_sddp::build_cut_row_batch_into(
            &mut batch,
            &setup.fcf,
            pool_id,
            state,
            &cut_state,
            &base.col_scale,
        );

        let mut frozen = StageTemplate::empty();
        let mut scratch = FreezeScratch::new();
        freeze_rows_into_template(base, &batch, &mut frozen, &mut scratch);
        frozen
    }

    /// Solve the terminal stage's frozen `template` against `pool` with the
    /// state vector pinned to `pinned_state`, returning `theta`'s primal
    /// value. `raw_noise` is zeroed per hydro (deterministic mean inflow):
    /// the probe's pinned state alone drives `theta`, independent of the
    /// realized noise draw.
    fn terminal_theta(
        setup: &cobre_sddp::StudySetup,
        template: &StageTemplate,
        pool: &cobre_sddp::CutPool,
        node_id: NodeId,
        pinned_state: &[f64],
    ) -> f64 {
        let comm = super::common::StubComm;
        let mut workspace_pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("create_workspace_pool");
        let ws: &mut SolverWorkspace<ActiveSolver> = &mut workspace_pool.workspaces[0];
        let ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let theta_col = setup.stage_state().theta;
        let terminal_stage = setup.num_stages() - 1;
        let raw_noise = vec![0.0_f64; setup.stage_state().hydro_count];

        ws.solver.reset_solver_state();
        ws.solver.load_model(template);
        patch_backward_opening_for_probe(
            ws,
            &ctx,
            &training_ctx,
            StageIdx(terminal_stage),
            pinned_state,
            &raw_noise,
        );

        let view =
            solve_stage_for_probe(ws, &ctx, pool, None, StageIdx(terminal_stage), 0, node_id)
                .expect("terminal stage solve must not error");
        view.primal[theta_col]
    }

    // ── Property 2: every water incoming bucket resolves through the
    // transit_buckets_in arm, never the commitment-hold catch-all ──────────

    #[test]
    fn every_water_incoming_bucket_resolves_through_the_transit_buckets_in_arm() {
        let setup = build_setup_in_code(build_system(), &config());
        let state = setup.stage_state();

        assert!(
            state.n_buckets > 0,
            "fixture must declare a water travel-time arc"
        );
        assert!(
            state.n_anticipated > 0 && state.k_max > 0,
            "fixture must also carry the anticipated ring so both families widen \
             commit_out/commit_in"
        );

        for j in state.transit_buckets_out.clone() {
            let incoming = state.state_to_lp_incoming_column(StateDim::new(j)).get();
            assert!(
                state.transit_buckets_in.contains(&incoming),
                "bucket state dim {j} must resolve through the transit_buckets_in arm \
                 [{:?}); got column {incoming}",
                state.transit_buckets_in
            );
            assert!(
                !state.commit_in.contains(&incoming),
                "bucket state dim {j} must never fall through to the commitment-hold \
                 catch-all commit_in [{:?}); got column {incoming}",
                state.commit_in
            );
        }
    }

    // ── Property 3: Sigma_d k_d = 1 per arc per anchor stage ────────────────

    #[test]
    fn bucket_k_weight_conservation_holds_per_arc_per_anchor_stage() {
        let durations = study_durations();

        // Mirrors `bucket_topology::extend_for_resolution`: pad the calendar
        // with copies of the trailing stage so `resolve_spread` never sees a
        // window it cannot fully absorb.
        let mut extended = durations.clone();
        let last = *durations.last().expect("at least one stage");
        let mut padded_hours = 0.0_f64;
        while padded_hours < TRAVEL_TIME_HOURS {
            extended.push(last);
            padded_hours += last;
        }

        for anchor in 0..durations.len() {
            let resolution =
                cobre_sddp::lead_time::resolve_spread(TRAVEL_TIME_HOURS, anchor, &extended, None);
            let sum: f64 = resolution.stage_weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "stage-clock weights must sum to 1.0 at anchor {anchor}; got {sum} from {:?}",
                resolution.stage_weights
            );
        }
    }

    // ── Property 4: the post-study carry and the water terminal buckets
    // price through the shared cut-state projection, additively ─────────────

    #[test]
    fn post_study_carry_and_water_terminal_buckets_price_through_the_shared_projection() {
        // Part A: the terminal probe. `water_priced_slot`'s outgoing value
        // (lag 1) is `water_pin_slot`'s (lag 2) pinned incoming value plus
        // this stage's own deposit share (`b_1^out = b_2^in + k_1*D`) — a
        // constant, identical across every probe below since none of them
        // touch the release decision. Comparing theta by DELTA against the
        // zero-pin baseline cancels that constant, proving theta prices both
        // families additively through the shared projection with no
        // per-family pricing arm.
        let mut setup = build_setup_in_code(build_system(), &config());
        let water_priced = water_priced_slot(&setup);
        let water_pin = water_pin_slot(&setup);
        let ant_slot = post_study_ring_slot(&setup);
        assert_ne!(
            water_priced, ant_slot,
            "the water and anticipated families must price distinct state dims"
        );
        assert_ne!(
            water_pin, ant_slot,
            "the water pin slot and the anticipated slot must be distinct state dims"
        );

        let tmp = tempfile::tempdir().expect("tempdir");
        inject_two_family_boundary(
            &mut setup,
            &tmp.path().join("boundary"),
            water_priced,
            ant_slot,
        );

        let terminal_pool_id = setup
            .inputs
            .node_graph
            .terminal_pool(setup.num_stages())
            .unwrap();
        let template = freeze_terminal_template(&setup, terminal_pool_id);
        let pool = &setup.fcf.pools[terminal_pool_id];
        let node_id = NodeId(i32::try_from(setup.num_stages() - 1).unwrap_or(0));

        let x_water = 4.0;
        let x_ant = 6.0;

        let theta_baseline =
            terminal_theta(&setup, &template, pool, node_id, &pin_state(&setup, &[]));
        let theta_water = terminal_theta(
            &setup,
            &template,
            pool,
            node_id,
            &pin_state(&setup, &[(water_pin, x_water)]),
        );
        let theta_ant = terminal_theta(
            &setup,
            &template,
            pool,
            node_id,
            &pin_state(&setup, &[(ant_slot, x_ant)]),
        );
        let theta_both = terminal_theta(
            &setup,
            &template,
            pool,
            node_id,
            &pin_state(&setup, &[(water_pin, x_water), (ant_slot, x_ant)]),
        );

        let delta_water = theta_water - theta_baseline;
        let delta_ant = theta_ant - theta_baseline;
        let delta_both = theta_both - theta_baseline;

        assert!(
            (delta_water - BETA_WATER * x_water).abs() < TOL,
            "theta must price the water terminal bucket alone by BETA_WATER*x_water via the \
             b_1^out = b_2^in + k_1*D shift: expected delta {}, got {delta_water}",
            BETA_WATER * x_water
        );
        assert!(
            (delta_ant - BETA_ANT * x_ant).abs() < TOL,
            "theta must price the post-study carry alone by BETA_ANT*x_ant: expected delta {}, \
             got {delta_ant}",
            BETA_ANT * x_ant
        );
        let expected_delta_both = BETA_WATER * x_water + BETA_ANT * x_ant;
        assert!(
            (delta_both - expected_delta_both).abs() < TOL,
            "theta must price both families additively through the shared projection, with no \
             per-family pricing arm: expected delta {expected_delta_both}, got {delta_both}"
        );

        // Part B: the solved training bound reflects the injected post-study
        // carry — training with the boundary cut raises final_lb over an
        // identical run with no boundary cut injected (ALPHA > 0 alone lifts
        // the terminal floor, independent of which state the optimum visits).
        let mut baseline = build_setup_in_code(build_system(), &config());
        let mut treated = build_setup_in_code(build_system(), &config());
        let treated_water_priced_slot = water_priced_slot(&treated);
        let treated_ant_slot = post_study_ring_slot(&treated);
        let tmp2 = tempfile::tempdir().expect("tempdir");
        inject_two_family_boundary(
            &mut treated,
            &tmp2.path().join("boundary"),
            treated_water_priced_slot,
            treated_ant_slot,
        );

        let comm = super::common::StubComm;
        let mut solver_baseline = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome_baseline = baseline
            .train(
                &mut solver_baseline,
                &comm,
                1,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome_baseline.error.is_none(),
            "baseline training error: {:?}",
            outcome_baseline.error
        );

        let mut solver_treated = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome_treated = treated
            .train(&mut solver_treated, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome_treated.error.is_none(),
            "treated training error: {:?}",
            outcome_treated.error
        );

        assert!(
            outcome_treated.result.final_lb > outcome_baseline.result.final_lb + TOL,
            "training with the injected post-study boundary must raise the lower bound: \
             treated final_lb={}, baseline final_lb={}",
            outcome_treated.result.final_lb,
            outcome_baseline.result.final_lb
        );
    }

    // ── Property 5: slot-identity reconstruction absorbs the two-family
    // width shift — zero basis rejections across warm-started iterations ────

    #[test]
    fn two_family_width_shift_warm_starts_with_zero_basis_rejections() {
        let mut setup = build_setup_in_code(build_system(), &config());
        let comm = super::common::StubComm;
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
        assert_eq!(
            result.iterations,
            u64::from(N_TRAIN_ITERATIONS),
            "warm-start regression needs the full iteration run to exercise \
             reconstruct_basis across iterations"
        );

        let total_rejections =
            SolverStatsDelta::aggregate(result.solver_stats_log.iter().map(|entry| &entry.delta))
                .basis_consistency_failures;
        assert_eq!(
            total_rejections, 0,
            "two-family width shift: expected 0 basis rejections, got {total_rejections} \
             (reconstruct_basis must match cut rows by CutPool slot identity, not absolute \
             column index, so the commit_out/commit_in width shift stays transparent)"
        );
    }
}
