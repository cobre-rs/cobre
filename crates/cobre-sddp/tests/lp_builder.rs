//! Consolidated LP-builder / indexer / stage-geometry integration tests for
//! `cobre-sddp`.
//!
//! Groups the non-uniform-block extraction backstop, the indexer-slim migration
//! rejection gate, the operational-start-date fixture-ordering guard, the policy
//! entity-manifest coverage, and the bus-partition collapse/multi-bus/
//! production-row boundary gates into one binary so the statically-linked solver
//! links once rather than once per file. The extraction backstop is HiGHS-only, so
//! it carries a per-`mod` `#[cfg(feature = "highs")]`; the other four mods compile
//! on every backend. The extraction, policy, and cell-partition mods reach the
//! shared `tests/common` harness.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

mod common;

#[cfg(feature = "highs")]
mod extraction_nonuniform_block_bases {
    //! Non-uniform-block extraction-correctness assertions.
    //!
    //! These tests pin the simulation read-path against the equipment-base /
    //! cost-range bug that surfaces only when a stage's block count differs from
    //! stage 0's. Both D33 (`[1, 3, 2]`, no anticipated thermals) and D34
    //! (`[1, 3, 2]` plus an anticipated thermal) declare such a schedule, so any
    //! family read with the global stage-0 base/length misreads the interior stages.
    //!
    //! Two assertions, each FAILS against the pre-fix stage-0-base/length read and
    //! PASSES once extraction resolves the per-stage `StageGeometry`:
    //!
    //! 1. **Per-block equipment shape** — at every stage the simulation must emit one
    //!    hydro record per (block, hydro) pair, i.e. exactly `n_blks(stage)` records
    //!    per hydro. A wrong per-stage stride/base does not change the record *count*
    //!    but does change which column each record reads; this assertion pins the
    //!    count as a coarse structural guard and the reconciliation below pins the
    //!    *values*.
    //! 2. **Cost-breakdown reconciliation** — `Σ(cost categories)` must equal
    //!    `immediate_cost` at every stage, including the non-uniform interior stages.
    //!    Pre-fix, `compute_cost_result` sums the stage-0 ranges (wrong base AND
    //!    length) for the interior stages, so the breakdown no longer reconciles to
    //!    the solved objective; post-fix it sums the stage-correct ranges and
    //!    reconciles exactly. D33/D34 declare no generic constraints and no NCS, so
    //!    every cost category is an objective·primal·scale sum and the breakdown is
    //!    expected to reconcile to the LP objective to within floating-point round-off.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::{TrainingEvent, scenario::ScenarioSource};
    use cobre_sddp::{
        SimulationScenarioResult, SimulationWeighting, StudySetup, aggregate_simulation,
        hydro_models::prepare_hydro_models,
        setup::{StudyParams, prepare_stochastic},
    };
    use cobre_solver::highs::HighsSolver;

    use super::common::StubComm;

    fn case_dir(suffix: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic")
            .join(suffix)
    }

    fn train_and_simulate(suffix: &str) -> Vec<SimulationScenarioResult> {
        let dir = case_dir(suffix);
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

        let params = StudyParams::from_config(&config_with_sim, Vec::new())
            .expect("StudyParams::from_config must succeed");

        let sentinel = Path::new("config.json");
        let training_source = config_with_sim
            .training_scenario_source(sentinel)
            .expect("training_scenario_source must parse");
        let simulation_source = config_with_sim
            .simulation_scenario_source(sentinel)
            .expect("simulation_scenario_source must parse");

        let mut setup = StudySetup::from_broadcast_params(
            &system,
            stochastic,
            params,
            hydro_models,
            &training_source,
            &simulation_source,
        )
        .expect("StudySetup must build");

        let comm = StubComm;
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                1,
                HighsSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "{suffix}: training error");
        let result = outcome.result;

        let mut pool = setup
            .create_workspace_pool(&comm, 1, HighsSolver::new)
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
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        let sim_config = setup.simulation_config();
        let (_summary, _gathered) = aggregate_simulation(
            &local_costs.costs,
            sim_config,
            &comm,
            SimulationWeighting::Uniform,
        )
        .expect("aggregate_simulation must succeed");

        scenario_results
    }

    /// The non-uniform block schedule shared by D33 and D34: the interior stages
    /// differ from stage 0's single block, which is the bug trigger.
    const BLOCK_COUNTS: [usize; 3] = [1, 3, 2];

    /// Assert one hydro record per (block, hydro) pair at every stage: a wrong
    /// per-stage stride does not change the count, but the count is the coarse shape
    /// guard the value reconciliation below complements.
    fn assert_per_block_equipment_shape(scenarios: &[SimulationScenarioResult], label: &str) {
        let scenario = scenarios.first().expect("one simulation scenario");
        for stage in &scenario.stages {
            let s = stage.stage_id as usize;
            let expected_blocks = BLOCK_COUNTS[s];
            // One hydro in both fixtures; per-block branch emits `n_blks` records.
            let n_block_records = stage.hydros.iter().filter(|h| h.block_id.is_some()).count();
            assert_eq!(
                n_block_records, expected_blocks,
                "{label}: stage {s} must emit {expected_blocks} per-block hydro records",
            );
            // Cross-check the two spillage read paths against each other. The
            // per-record `spillage_cost` (a Part-1 per-block `grid.flat` read of the
            // spillage column) summed across blocks must equal the cost result's
            // `spillage_cost` (a Part-2 `range_sum` over the spillage range). D33/D34
            // declare no diversion, so the cost result's `spillage_cost` is the pure
            // spillage range_sum (diversion contributes 0). Pre-fix, at the
            // non-uniform interior stages the per-block base and the range base/length
            // disagree (one strides off stage-0's block width per block, the other
            // sums stage-0's range), so the two paths read different columns and the
            // cross-check diverges; post-fix both read the stage-correct columns and
            // agree to round-off.
            let per_record_spillage_cost: f64 = stage.hydros.iter().map(|h| h.spillage_cost).sum();
            let cost = stage.costs.first().expect("one cost record per stage");
            let scale = cost.spillage_cost.abs().max(1.0);
            let abs_err = (per_record_spillage_cost - cost.spillage_cost).abs();
            assert!(
                abs_err <= 1e-6 * scale,
                "{label}: stage {s} per-block spillage cost {per_record_spillage_cost} \
                 disagrees with the cost-result spillage_cost {} (abs err {abs_err}); \
                 the per-block column read and the range_sum addressed different \
                 columns at a non-uniform-block stage",
                cost.spillage_cost,
            );
            for h in &stage.hydros {
                assert!(
                    h.spillage_m3s.is_finite(),
                    "{label}: stage {s} spillage must be finite",
                );
            }
        }
    }

    /// Reconciliation invariant: the sum of every cost-breakdown category equals
    /// `immediate_cost` at every stage. The interior stages (1, 2) have block counts
    /// differing from stage 0's, so this fails pre-fix (stage-0 ranges misbook the
    /// cost) and passes post-fix (stage-correct ranges).
    fn assert_cost_reconciliation(scenarios: &[SimulationScenarioResult], label: &str) {
        let scenario = scenarios.first().expect("one simulation scenario");
        for stage in &scenario.stages {
            let s = stage.stage_id as usize;
            let cost = stage.costs.first().expect("one cost record per stage");

            // `hydro_violation_cost` is the total of the per-constraint violation
            // costs (outflow/turbined/generation/evaporation/withdrawal); summing it
            // here together with the standalone categories covers every priced column
            // exactly once. `generic_violation_cost` and `curtailment_cost` use a
            // non-range formula, but D33/D34 declare neither generic constraints nor
            // NCS, so both are zero and do not perturb the reconciliation.
            let breakdown = cost.thermal_cost
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

            // Relative tolerance scaled by the immediate cost magnitude: the
            // breakdown and `immediate_cost` are the same objective·primal·scale sum
            // grouped differently, so they agree to floating-point round-off.
            let scale = cost.immediate_cost.abs().max(1.0);
            let abs_err = (breakdown - cost.immediate_cost).abs();
            assert!(
                abs_err <= 1e-6 * scale,
                "{label}: stage {s} cost breakdown {breakdown} does not reconcile to \
                 immediate_cost {} (abs err {abs_err}); a stage-0 equipment range was \
                 summed at a non-uniform-block stage",
                cost.immediate_cost,
            );
        }
    }

    /// Pin the reported `anticipated_decision_mw` at the active interior delivery
    /// stage. The decision column's base is the per-stage `thermal.end`
    /// (`n_blks`-dependent); reading it off the global stage-0 base lands on a
    /// thermal-generation column at a non-uniform stage, so the reported decision MW
    /// equals one of the thermal's per-block `generation_mw` values instead of the
    /// distinct decision primal. Post-fix the decision is a single per-plant scalar
    /// (identical across blocks) distinct from every per-block generation; pre-fix it
    /// collapses onto a generation column. D34's anticipated thermal (`K = 1`)
    /// commits at stage 0 and re-commits at stage 1 (3 blocks ≠ stage 0's 1), so
    /// stage 1 is the bug-exposing active delivery stage.
    fn assert_anticipated_decision_mw(scenarios: &[SimulationScenarioResult], label: &str) {
        let scenario = scenarios.first().expect("one simulation scenario");
        let active_stage = 1u32;
        let stage = scenario
            .stages
            .iter()
            .find(|s| s.stage_id == active_stage)
            .expect("D34 has an interior stage 1");
        assert_eq!(
            BLOCK_COUNTS[active_stage as usize], 3,
            "stage 1 must carry 3 blocks (≠ stage 0's 1) to exercise the bug",
        );

        let antic: Vec<_> = stage.thermals.iter().filter(|t| t.is_anticipated).collect();
        assert!(
            !antic.is_empty(),
            "{label}: stage {active_stage} must report an anticipated thermal",
        );

        // The decision is a per-plant-per-stage scalar: identical across all block
        // records of the same anticipated thermal, present, finite, and positive.
        let decision = antic[0]
            .anticipated_decision_mw
            .expect("active anticipated decision must be Some at the delivery stage");
        assert!(
            decision.is_finite() && decision > 0.0,
            "{label}: anticipated_decision_mw must be a positive finite scalar, got {decision}",
        );
        for t in &antic {
            let d = t
                .anticipated_decision_mw
                .expect("anticipated decision present for every block record at the active stage");
            assert!(
                (d - decision).abs() <= 1e-9,
                "{label}: anticipated_decision_mw must be identical across blocks of the \
                 same thermal (per-plant scalar), got {d} vs {decision}",
            );
            // The decisive base-correctness check: a stage-0-based read lands on a
            // thermal-generation column at this non-uniform stage, so the reported
            // decision would equal one of the per-block generation values. The
            // stage-correct decision column is distinct from every per-block
            // generation of the same thermal.
            assert!(
                (d - t.generation_mw).abs() > 1e-6,
                "{label}: anticipated_decision_mw {d} coincides with this thermal's \
                 per-block generation_mw {} at stage {active_stage} — the decision was \
                 read off the global stage-0 base onto a generation column",
                t.generation_mw,
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d33_non_uniform_blocks_per_block_equipment_shape() {
        let scenarios = train_and_simulate("d33-per-stage-block-counts");
        assert_per_block_equipment_shape(&scenarios, "D33");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d33_non_uniform_blocks_cost_reconciles() {
        let scenarios = train_and_simulate("d33-per-stage-block-counts");
        assert_cost_reconciliation(&scenarios, "D33");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d34_anticipated_non_uniform_blocks_per_block_equipment_shape() {
        let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
        assert_per_block_equipment_shape(&scenarios, "D34");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d34_anticipated_non_uniform_blocks_cost_reconciles() {
        // D34 exercises the anticipated-decision column range too: its base is the
        // per-stage `thermal.end`, so the reconciliation here additionally pins the
        // anticipated-decision range repoint at the interior delivery stages.
        let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
        assert_cost_reconciliation(&scenarios, "D34");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d34_anticipated_decision_mw_reads_stage_correct_column() {
        let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
        assert_anticipated_decision_mw(&scenarios, "D34");
    }
}

mod indexer_slim_migration_rejection {
    //! Deletion-regression grep gate for the removed indexer types.
    //!
    //! The role-(b) geometry descriptor `StageIndexer` and its `EquipmentCounts`
    //! constructor input were deleted: the state-vector concern lives on
    //! `StateSpace`, the non-state study shape on `StudyDimensions`, and the
    //! per-stage equipment geometry on `StageLayout`/`StageGeometry`. This gate scans
    //! the production sources under `src/` and asserts those deleted types do not
    //! reappear — a regression guard that the deletion stays deleted.
    //!
    //! The forbidden tokens are assembled from character arrays so this gate file
    //! does not match itself (the project grep-gate convention).

    use std::path::{Path, PathBuf};

    fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read_dir src") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                collect_rs(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// Assemble a forbidden token from chars so the gate does not self-match.
    fn token(chars: &[char]) -> String {
        chars.iter().collect()
    }

    #[test]
    fn deleted_indexer_types_stay_deleted() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect_rs(&src, &mut files);
        assert!(!files.is_empty(), "must scan at least one source file");

        // `StageIndexer` — the deleted role-(b) geometry descriptor type.
        let stage_indexer = token(&['S', 't', 'a', 'g', 'e', 'I', 'n', 'd', 'e', 'x', 'e', 'r']);
        // `EquipmentCounts` — the deleted constructor input bag.
        let equipment_counts = token(&[
            'E', 'q', 'u', 'i', 'p', 'm', 'e', 'n', 't', 'C', 'o', 'u', 'n', 't', 's',
        ]);

        let mut offenders: Vec<String> = Vec::new();
        for path in &files {
            let body = std::fs::read_to_string(path).expect("read source file");
            for (lineno, line) in body.lines().enumerate() {
                if line.contains(&stage_indexer) {
                    offenders.push(format!(
                        "{}:{}: deleted type StageIndexer reappeared",
                        path.display(),
                        lineno + 1
                    ));
                }
                if line.contains(&equipment_counts) {
                    offenders.push(format!(
                        "{}:{}: deleted type EquipmentCounts reappeared",
                        path.display(),
                        lineno + 1
                    ));
                }
            }
        }

        assert!(
            offenders.is_empty(),
            "deletion regression — a deleted indexer type reappeared in production \
             sources (the role-(b) geometry now lives on StageLayout/StageGeometry, \
             the state vector on StateSpace, the study shape on StudyDimensions):\n{}",
            offenders.join("\n")
        );
    }
}

mod fixture_operational_start_date_order {
    //! Guards that every shipped deterministic-case entity fixture assigns
    //! `operational_start_date` so the `(operational_start_date, name)` build order
    //! reproduces the ascending-`id` order. A future fixture edit that breaks the
    //! date-monotonic-with-id property (e.g. a shared sentinel date that lets the
    //! `name` tiebreak reorder a collection) fails here instead of silently moving a
    //! parity hash.

    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::Value;

    const ENTITY_FILES: &[&str] = &[
        "buses.json",
        "hydros.json",
        "thermals.json",
        "lines.json",
        "non_controllable_sources.json",
        "pumping_stations.json",
        "energy_contracts.json",
    ];

    fn deterministic_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic")
    }

    fn entity_array(doc: &Value) -> Option<&Vec<Value>> {
        doc.as_object()?
            .iter()
            .find(|(key, value)| key.as_str() != "$schema" && value.is_array())
            .and_then(|(_, value)| value.as_array())
    }

    fn check_file(path: &Path) {
        let text = fs::read_to_string(path).expect("read entity fixture");
        let doc: Value = serde_json::from_str(&text).expect("parse entity fixture");
        let Some(entities) = entity_array(&doc) else {
            return;
        };
        if entities.is_empty() {
            return;
        }

        let mut by_id: Vec<&Value> = entities.iter().collect();
        by_id.sort_by_key(|e| e["id"].as_i64().expect("entity id"));

        let mut by_build: Vec<&Value> = entities.iter().collect();
        by_build.sort_by(|a, b| {
            let da = a["operational_start_date"]
                .as_str()
                .expect("operational_start_date present");
            let db = b["operational_start_date"]
                .as_str()
                .expect("operational_start_date present");
            let na = a["name"].as_str().expect("entity name");
            let nb = b["name"].as_str().expect("entity name");
            (da, na).cmp(&(db, nb))
        });

        let ids_by_id: Vec<i64> = by_id.iter().map(|e| e["id"].as_i64().unwrap()).collect();
        let ids_by_build: Vec<i64> = by_build.iter().map(|e| e["id"].as_i64().unwrap()).collect();
        assert_eq!(
            ids_by_build,
            ids_by_id,
            "{}: (operational_start_date, name) order must equal ascending-id order",
            path.display()
        );
    }

    #[test]
    fn deterministic_fixtures_preserve_id_order_under_build_sort() {
        let root = deterministic_root();
        let mut checked = 0usize;
        for case in fs::read_dir(&root).expect("read deterministic root") {
            let system = case.expect("case dir entry").path().join("system");
            if !system.is_dir() {
                continue;
            }
            for fname in ENTITY_FILES {
                let path = system.join(fname);
                if path.exists() {
                    check_file(&path);
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 0,
            "no deterministic entity fixtures found under {}",
            root.display()
        );
    }
}

mod policy_entity_manifest {
    //! Integration coverage for the embedded per-slot entity manifest written into
    //! `policy/cuts/<pool>.bin` by the shared `write_checkpoint`.
    //!
    //! Trains a deterministic case to a policy checkpoint through the same
    //! `write_checkpoint` both front ends call, then reads the cut files back and
    //! asserts the manifest classification, identity, and per-stage length — including
    //! the reduced-stage (`inflow_lags: false`) d43 case, whose pool drops its
    //! inflow-lag slots.

    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::StateFamily;
    use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    fn case_dir(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic")
            .join(name)
    }

    /// Train a case to a policy checkpoint via the shared `write_checkpoint`, then read
    /// it back. Returns `(checkpoint, per-pool cut_state_layout n_slots)`.
    fn train_and_read_checkpoint(name: &str) -> (cobre_io::PolicyCheckpoint, Vec<usize>) {
        let dir = case_dir(name);
        let config = cobre_io::parse_config(&dir.join("config.json")).expect("config must parse");
        let system = cobre_io::load_case(&dir).expect("load_case must succeed");

        let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default(), None)
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup must build");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        let result = outcome.result;

        let _training_output = setup.build_training_output(&result, &[]);

        // Each pool is sized to its stage's cut-state dimension at construction
        // (`pool_state_dimensions[t] == cut_state_layouts[t].n_slots()`), so the pool's
        // own `state_dimension` is the authoritative per-stage manifest length.
        let pool_n_state: Vec<usize> = setup.fcf.pools.iter().map(|p| p.state_dimension).collect();

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let policy_dir = tmp.path().join("policy");
        write_checkpoint(
            &policy_dir,
            &setup,
            &system,
            &result,
            &CheckpointParams {
                max_iterations: 100,
                forward_passes: 1,
                seed: 42,
                export_states: false,
            },
        )
        .expect("write_checkpoint must succeed");

        let checkpoint = cobre_io::read_policy_checkpoint(&policy_dir)
            .expect("read_policy_checkpoint must succeed");
        (checkpoint, pool_n_state)
    }

    /// All-enabled study (d03, two-hydro cascade, storage state only): every stage's
    /// manifest length equals that pool's cut-state dimension, and the storage slots
    /// carry `entity_type == 0` with the hydro ids in `system.hydros()` order and
    /// `subindex == 0`.
    #[test]
    fn all_enabled_manifest_storage_slots_carry_hydro_identity() {
        let (checkpoint, pool_n_state) = train_and_read_checkpoint("d03-two-hydro-cascade");

        assert!(
            !checkpoint.stage_cuts.is_empty(),
            "checkpoint must contain stage cut files"
        );
        for stage in &checkpoint.stage_cuts {
            let t = stage.stage_id as usize;
            assert_eq!(
                stage.entity_manifest.len(),
                pool_n_state[t],
                "stage {t} manifest length must equal the pool cut-state dimension"
            );
            // d03 is storage-only state (no inflow lags, no anticipated): every slot is
            // a storage slot, in system.hydros() order, subindex 0.
            for (i, slot) in stage.entity_manifest.iter().enumerate() {
                assert_eq!(
                    slot.entity_type, 0,
                    "stage {t} slot {i} must be HydroStorage"
                );
                assert_eq!(slot.subindex, 0, "stage {t} slot {i} storage subindex 0");
            }
            // The two cascade hydros have ids 0 and 1, in `system.hydros()` order.
            assert_eq!(stage.entity_manifest[0].entity_id, 0);
            assert_eq!(stage.entity_manifest[1].entity_id, 1);
        }
    }

    /// d43 reduced stage: the pool sized by a stage with `inflow_lags: false` produces
    /// a manifest whose length equals that pool's reduced cut-state dimension and which
    /// carries NO `HydroInflowLag` (type 1) slot, while the full-state pools do carry
    /// lag slots. The reduced pool is located empirically by its `cut_state_layouts`
    /// dimension (the pool-sizing off-by-one means the reduced projection is not at the
    /// `inflow_lags: false` stage's own index).
    #[test]
    fn d43_reduced_stage_manifest_drops_inflow_lag_slots() {
        let (checkpoint, pool_n_state) = train_and_read_checkpoint("d43-storage-only-cut");

        // Per-stage manifest length always equals the pool's cut-state dimension.
        for stage in &checkpoint.stage_cuts {
            let t = stage.stage_id as usize;
            assert_eq!(
                stage.entity_manifest.len(),
                pool_n_state[t],
                "stage {t} manifest length must equal cut_state_layouts[{t}].n_slots()"
            );
        }

        // d43 has one hydro; the full-state pools carry one storage + lag slots, the
        // reduced pool carries one storage slot only. So the reduced pool is the one
        // with the strictly smallest cut-state dimension, and exactly one pool is
        // reduced.
        let min_dim = *pool_n_state.iter().min().expect("at least one pool");
        let max_dim = *pool_n_state.iter().max().expect("at least one pool");
        assert!(
            min_dim < max_dim,
            "d43 must have a reduced pool (min {min_dim}) and a full pool (max {max_dim})"
        );

        let reduced_stage = checkpoint
            .stage_cuts
            .iter()
            .find(|s| s.entity_manifest.len() == min_dim)
            .expect("a reduced-dimension stage must exist");
        assert!(
            reduced_stage
                .entity_manifest
                .iter()
                .all(|s| s.entity_type != StateFamily::HydroInflowLag.code()),
            "the reduced stage manifest must contain no HydroInflowLag slot"
        );

        // A full-state stage must carry at least one inflow-lag slot (d43 fits a PAR
        // model, so the full pools have lag dimensions).
        let full_stage = checkpoint
            .stage_cuts
            .iter()
            .find(|s| s.entity_manifest.len() == max_dim)
            .expect("a full-dimension stage must exist");
        assert!(
            full_stage
                .entity_manifest
                .iter()
                .any(|s| s.entity_type == StateFamily::HydroInflowLag.code()),
            "a full-state stage manifest must contain an inflow-lag slot"
        );
    }

    /// Anticipated K=2 study (d37): the manifest carries `AnticipatedThermalState`
    /// (type 2) slots whose `entity_id` is the anticipated plant (id 1) and whose
    /// `subindex` ring slots cover `0..k_max` (`{0, 1}` for K=2).
    #[test]
    fn anticipated_k2_manifest_has_thermal_state_slots() {
        let (checkpoint, _pool_n_state) =
            train_and_read_checkpoint("d37-anticipated-commissioning");

        let stage0 = checkpoint
            .stage_cuts
            .iter()
            .find(|s| s.stage_id == 0)
            .expect("stage 0 cut file must exist");

        let anticipated_slots: Vec<_> = stage0
            .entity_manifest
            .iter()
            .filter(|s| s.entity_type == StateFamily::AnticipatedThermalState.code())
            .collect();

        assert_eq!(
            anticipated_slots.len(),
            2,
            "K=2 yields two anticipated ring slots on the single anticipated plant"
        );
        for slot in &anticipated_slots {
            assert_eq!(slot.entity_id, 1, "anticipated slot must own plant id 1");
        }
        let ring: Vec<u32> = anticipated_slots.iter().map(|s| s.subindex).collect();
        assert_eq!(ring, vec![0, 1], "ring slots must cover 0..k_max");
    }
}

mod cell_partition_gates {
    //! Bus-partition boundary gates: collapse, multi-bus behaviour, and the
    //! production-row cap/attribution properties.
    //!
    //! Turbine/generation columns are indexed, bounded, credited to buses, and
    //! related by FPHA production rows per cell — and every deterministic
    //! golden still reproduces byte-for-byte, because every shipped fixture
    //! is single-bus, so its cell index is the identity map (each plant is
    //! its own single cell, `cells_of(h) == h..h + 1`). That is a cost pin,
    //! not a structure pin: a wrong cell index and a right one
    //! are numerically identical under identity staging. These tests are what
    //! actually distinguishes them, on a synthetic multi-cell fixture built
    //! in-process (no case directory).
    //!
    //! One shared fixture family: an FPHA plant (`H_FPHA`) whose `unit_groups`
    //! vary per test, alongside an unrelated `ConstantProductivity` hydro
    //! (`H_OTHER`) and four buses — so a cell index, a hydro index, and a bus
    //! position are never forced to coincide. `H_FPHA` carries two known FPHA
    //! planes: [`GENERIC_PLANE`] (nonzero on every term, for the aggregate-cap
    //! identity) and [`ORIGIN_PLANE`] (`γ0 = 0 ∧ γV = 0`, for the per-cell
    //! zero-flow attribution gate) — both declared directly rather than fit,
    //! since [`hull_fit_emits_exactly_one_origin_plane`] and
    //! [`reduce_planes_huge_tolerance_preserves_origin_plane_both_methods`]
    //! (`crates/cobre-sddp/src/production/fpha_fitting/tests.rs`) already pin
    //! that a real fit emits and preserves exactly this plane.
    //!
    //! Column/row identity is never read via hardcoded layout offsets
    //! (`StageLayout` is `pub(crate)`, unreachable from this external
    //! integration-test binary, and offsets would silently drift with an
    //! unrelated layout change anyway). Instead: turbine/generation columns
    //! are given mutually distinct `col_upper` bounds per cell and located by
    //! scanning for them ([`find_col_by_upper`]); the rows a column
    //! participates in are read directly off its own CSC slice
    //! ([`col_rows`]); and a specific row (an FPHA plane row, a load-balance
    //! row, the shared water-balance row) is identified by set
    //! intersection/difference against a sibling column's row set, never by
    //! number.

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::InflowModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ContractBlockBounds,
        DeficitSegment, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        HydroUnitGroup, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_sddp::hydro_models::{
        EvaporationModelSet, FphaPlane, PrepareHydroModelsResult, ProductionModelSet,
        ResolvedProductionModel,
    };
    use cobre_sddp::inflow_method::InflowNonNegativityMethod;
    use cobre_sddp::resolved_parameters::ResolvedParameters;
    use cobre_sddp::test_support::{assert_templates_byte_identical, make_unit_group};
    use cobre_sddp::{StageTemplates, build_stage_templates_resolving_layout};
    use cobre_solver::StageTemplate;
    use cobre_stochastic::normal::precompute::PrecomputedNormal;
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };

    /// Nonzero on every term — the plane the aggregate-cap identity check
    /// is evaluated against.
    const GENERIC_PLANE: FphaPlane = FphaPlane {
        intercept: 40.0,
        gamma_v: 0.02,
        gamma_q: 0.8,
        gamma_s: 0.05,
    };

    /// The plane through the origin (`γ0 = 0 ∧ γV = 0`), `gamma_s` also zero:
    /// its per-cell row reduces to `g_c <= gamma_q * q_c` regardless of
    /// storage/spillage, which is what the per-cell zero-flow attribution
    /// gate rests on. See the module doc for the two tests that pin this
    /// shape survives a real fit.
    const ORIGIN_PLANE: FphaPlane = FphaPlane {
        intercept: 0.0,
        gamma_v: 0.0,
        gamma_q: 0.6,
        gamma_s: 0.0,
    };

    const BUS_PLANT_OWN: EntityId = EntityId(1);
    const BUS_OTHER_HYDRO: EntityId = EntityId(2);
    const BUS_CELL_Y: EntityId = EntityId(3);
    const BUS_CELL_Z: EntityId = EntityId(4);

    const HYDRO_OTHER_ID: EntityId = EntityId(10);
    const HYDRO_FPHA_ID: EntityId = EntityId(20);

    /// `H_FPHA`'s own declared bounds — every variant's `unit_groups` sums to
    /// exactly these (the single-mirror-group variant declares one group
    /// carrying them directly), so the resolved-bounds plant term
    /// (`cell_max_turbined`/`cell_max_generation`'s `hb` clamp, set far larger
    /// below) never binds and the group term is what a test observes.
    const PLANT_MAX_TURBINED_M3S: f64 = 750.0;
    const PLANT_MAX_GENERATION_MW: f64 = 12_000.0;

    const BLOCK_HOURS: f64 = 730.0;
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;

    fn neutral_date() -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd_opt(2024, 1, 1).expect("2024-01-01 is valid")
    }

    fn buses() -> Vec<Bus> {
        let start = neutral_date();
        [BUS_PLANT_OWN, BUS_OTHER_HYDRO, BUS_CELL_Y, BUS_CELL_Z]
            .into_iter()
            .map(|id| {
                make_bus(
                    id,
                    BusSpec {
                        name: format!("B{id}"),
                        operational_start_date: start,
                        deficit_segments: vec![DeficitSegment {
                            depth_mw: None,
                            cost_per_mwh: 500.0,
                        }],
                        excess_cost: 0.0,
                    },
                )
            })
            .collect()
    }

    fn other_hydro() -> cobre_core::Hydro {
        make_hydro(
            HYDRO_OTHER_ID,
            HydroSpec {
                name: "H_OTHER".to_string(),
                bus_id: BUS_OTHER_HYDRO,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                ..Default::default()
            },
        )
    }

    fn fpha_hydro(unit_groups: Vec<HydroUnitGroup>) -> cobre_core::Hydro {
        make_hydro(
            HYDRO_FPHA_ID,
            HydroSpec {
                name: "H_FPHA".to_string(),
                bus_id: BUS_PLANT_OWN,
                generation_model: HydroGenerationModel::Fpha,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: PLANT_MAX_TURBINED_M3S,
                min_generation_mw: 0.0,
                max_generation_mw: PLANT_MAX_GENERATION_MW,
                unit_groups,
                ..Default::default()
            },
        )
    }

    /// A single group carrying the plant's own bus and four bounds — the
    /// mirror group a plant now declares explicitly, since an empty
    /// `unit_groups` is rejected — 1 cell.
    fn groups_single_mirror() -> Vec<HydroUnitGroup> {
        vec![make_unit_group(
            EntityId(0),
            BUS_PLANT_OWN,
            0.0,
            PLANT_MAX_GENERATION_MW,
            0.0,
            PLANT_MAX_TURBINED_M3S,
        )]
    }

    /// Three groups, all on `BUS_PLANT_OWN` — the plant's own declared
    /// `bus_id`, so the collapsed cell lands on the SAME bus the single-
    /// mirror-group variant's group does. A same-bus group on any OTHER bus
    /// would collapse to a different cell than the mirror group's, comparing
    /// two different physical configurations rather than testing collapse.
    /// Deliberately UNEQUAL thirds summing exactly to the plant's declared
    /// totals: equal thirds would make a first-group, largest-group, or mean
    /// implementation indistinguishable from the sum, which is the entire
    /// content of the collapse claim.
    fn groups_three_same_bus() -> Vec<HydroUnitGroup> {
        vec![
            make_unit_group(EntityId(101), BUS_PLANT_OWN, 0.0, 1_000.0, 0.0, 100.0),
            make_unit_group(EntityId(102), BUS_PLANT_OWN, 0.0, 3_000.0, 0.0, 250.0),
            make_unit_group(EntityId(103), BUS_PLANT_OWN, 0.0, 8_000.0, 0.0, 400.0),
        ]
    }

    /// Two groups on two distinct buses, neither the plant's own `bus_id` —
    /// 2 cells, ordered `[BUS_CELL_Y, BUS_CELL_Z]` ascending by bus id.
    fn groups_two_buses() -> Vec<HydroUnitGroup> {
        vec![
            make_unit_group(EntityId(201), BUS_CELL_Y, 0.0, 7_000.0, 0.0, 450.0),
            make_unit_group(EntityId(202), BUS_CELL_Z, 0.0, 5_000.0, 0.0, 300.0),
        ]
    }

    /// Three groups on three distinct buses — 3 cells, ordered
    /// `[BUS_OTHER_HYDRO, BUS_CELL_Y, BUS_CELL_Z]` ascending by bus id. Only
    /// the aggregate-cap identity check's 1/2/3-cell partition triple uses
    /// this variant; it is not part of the A/B/C collapse/state-space
    /// family.
    fn groups_three_buses() -> Vec<HydroUnitGroup> {
        vec![
            make_unit_group(EntityId(301), BUS_OTHER_HYDRO, 0.0, 3_000.0, 0.0, 200.0),
            make_unit_group(EntityId(302), BUS_CELL_Y, 0.0, 4_000.0, 0.0, 250.0),
            make_unit_group(EntityId(303), BUS_CELL_Z, 0.0, 5_000.0, 0.0, 300.0),
        ]
    }

    /// A `ResolvedBounds` whose hydro clamp (`100_000.0` on every axis) never
    /// binds against any group/raw sum this module's fixtures declare — a
    /// binding clamp would mask the summing/folding helper each test exists
    /// to exercise. Every non-hydro family is zeroed (no thermals/lines/
    /// pumping/contracts declared anywhere in this module).
    fn generous_resolved_bounds(counts: &BoundsCountsSpec) -> ResolvedBounds {
        ResolvedBounds::new(
            counts,
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 100_000.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100_000.0,
                    max_generation_mw: 100_000.0,
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
        )
    }

    /// All-zero `ResolvedPenalties` — no test in this module solves, so cost
    /// coefficients are inert; only the shape (counts) matters.
    fn zero_resolved_penalties(counts: &PenaltiesCountsSpec) -> ResolvedPenalties {
        ResolvedPenalties::new(
            counts,
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

    fn build_system(fpha_groups: Vec<HydroUnitGroup>) -> System {
        let bounds = generous_resolved_bounds(&BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        });

        let penalties = zero_resolved_penalties(&PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 4,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        });

        let stage = make_stage(
            0,
            StageSpec {
                blocks: vec![Block {
                    index: 0,
                    name: "B0".to_string(),
                    duration_hours: BLOCK_HOURS,
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
        );

        let inflow_models = vec![
            InflowModel {
                hydro_id: HYDRO_OTHER_ID,
                stage_id: 0,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: HYDRO_FPHA_ID,
                stage_id: 0,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];

        SystemBuilder::new()
            .buses(buses())
            .hydros(vec![other_hydro(), fpha_hydro(fpha_groups)])
            .stages(vec![stage])
            .inflow_models(inflow_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("cell_partition_gates fixture must be a valid System")
    }

    /// `H_OTHER` (`ConstantProductivity`) at hydro system-index 0, `H_FPHA`
    /// (`Fpha`, two known planes) at index 1 — stable across every variant
    /// (only `unit_groups` differs), so this one `ProductionModelSet` is
    /// shared by every variant's template build.
    fn production_models() -> ProductionModelSet {
        ProductionModelSet::new(
            vec![
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 2.0 }],
                vec![ResolvedProductionModel::Fpha {
                    planes: vec![GENERIC_PLANE, ORIGIN_PLANE],
                }],
            ],
            &cobre_sddp::test_support::minimal_hydros(2),
            1,
        )
    }

    fn evaporation_models(system: &System) -> EvaporationModelSet {
        PrepareHydroModelsResult::default_from_system(system).evaporation
    }

    fn build_stage_templates(fpha_groups: Vec<HydroUnitGroup>) -> StageTemplates {
        let system = build_system(fpha_groups);
        let production = production_models();
        let evaporation = evaporation_models(&system);
        build_stage_templates_resolving_layout(
            &system,
            InflowNonNegativityMethod::None,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evaporation,
            &ResolvedParameters::default(),
        )
        .expect("cell_partition_gates template must build")
    }

    fn build_template(fpha_groups: Vec<HydroUnitGroup>) -> StageTemplate {
        build_stage_templates(fpha_groups)
            .templates
            .into_iter()
            .next()
            .expect("exactly one stage")
    }

    // ---- CSC probes --------------------------------------------------------

    /// Sum of every raw push at `(col, row)`. CSC assembly does not
    /// pre-merge duplicate `(row, col)` pushes (mirrors this crate's own
    /// `csc_entry_sum` unit-test convention) — summing is what "the
    /// coefficient at this position" means.
    fn csc_entry_sum(tmpl: &StageTemplate, col: usize, row: usize) -> f64 {
        let start = tmpl.col_starts[col] as usize;
        let end = tmpl.col_starts[col + 1] as usize;
        (start..end)
            .filter(|&pos| tmpl.row_indices[pos] as usize == row)
            .map(|pos| tmpl.values[pos])
            .sum()
    }

    /// Every `(row, value)` `col` participates in, rows summed.
    fn col_rows(tmpl: &StageTemplate, col: usize) -> Vec<(usize, f64)> {
        let start = tmpl.col_starts[col] as usize;
        let end = tmpl.col_starts[col + 1] as usize;
        let mut rows: Vec<(usize, f64)> = Vec::new();
        for pos in start..end {
            let row = tmpl.row_indices[pos] as usize;
            if let Some(entry) = rows.iter_mut().find(|(r, _)| *r == row) {
                entry.1 += tmpl.values[pos];
            } else {
                rows.push((row, tmpl.values[pos]));
            }
        }
        rows
    }

    /// Every `(col, value)` present at `row` — a full column scan (this
    /// binary has no CSR view), fine at this fixture's size. Presence is
    /// determined by whether `col`'s CSC slice contains `row` at all, not by
    /// the summed value being nonzero (the origin plane's zero-valued
    /// pushes must still show up here).
    fn row_entries(tmpl: &StageTemplate, row: usize) -> Vec<(usize, f64)> {
        (0..tmpl.num_cols)
            .filter_map(|col| {
                let start = tmpl.col_starts[col] as usize;
                let end = tmpl.col_starts[col + 1] as usize;
                let hits: Vec<f64> = (start..end)
                    .filter(|&pos| tmpl.row_indices[pos] as usize == row)
                    .map(|pos| tmpl.values[pos])
                    .collect();
                if hits.is_empty() {
                    None
                } else {
                    Some((col, hits.into_iter().sum()))
                }
            })
            .collect()
    }

    /// Rows common to both `(row, value)` sets, by row index.
    fn common_rows(a: &[(usize, f64)], b: &[(usize, f64)]) -> Vec<usize> {
        a.iter()
            .map(|&(r, _)| r)
            .filter(|r| b.iter().any(|&(r2, _)| r2 == *r))
            .collect()
    }

    /// The unique column whose `col_upper` matches `target`. Turbine/
    /// generation columns are given mutually distinct bounds per cell
    /// precisely so a column is identifiable this way instead of via a
    /// hardcoded, layout-order-dependent offset (see the module doc).
    fn find_col_by_upper(tmpl: &StageTemplate, target: f64, tol: f64) -> usize {
        let matches: Vec<usize> = tmpl
            .col_upper
            .iter()
            .enumerate()
            .filter(|&(_, &v)| (v - target).abs() < tol)
            .map(|(c, _)| c)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one column with col_upper ~= {target}, found {matches:?}"
        );
        matches[0]
    }

    // ---- Collapse gate -----------------------------------------------------

    /// The collapse claim: a plant declaring three same-bus groups (unequal,
    /// summing EXACTLY to the plant's own declared totals) yields a
    /// byte-identical `StageTemplate` — identity of the generated LP and its
    /// index-addressed positions, not of the `HydroCellIndex` object itself
    /// (`groups_of(cell)` necessarily differs between the variants, which is
    /// expected, not a gap) — to the same plant declaring a single mirror
    /// group, its bounds summed then clamped to the plant's own resolved
    /// bound.
    ///
    /// This equality holds only under three hypotheses, satisfied here but
    /// not in general: (H1) capacity conservation is EXACT
    /// (`Σ_g ū_g == ū_plant`; the semantic-validation rule this fixture
    /// stands in for only enforces `≤`); (H2) fold consistency — see below;
    /// (H3) the float sum realizes the plant total exactly, which is why the
    /// resolved-bounds plant clamp is set far above the sum (a binding clamp
    /// would mask the summing helper's own behaviour, defeating the gate).
    ///
    /// H2 is the non-obvious one and is VACUOUS for this fixture, not
    /// generally true: `cell_max_turbined`/`cell_max_generation` fold each
    /// group's own bound pair before summing
    /// (`min(q̄_g, p̄_g/ρ)` for `ConstantProductivity`), while the single-
    /// mirror-group path folds once over the plant's own totals. `min` is
    /// concave, so `Σ_g min(a_g, b_g) ≤ min(Σa_g, Σb_g)`, equality iff every
    /// group binds on the SAME side. `H_FPHA`'s production model makes the
    /// fold the identity (`_ => turbined` in `cell_max_turbined`'s `fold`
    /// closure), so H2 holds trivially here regardless of binding side — this
    /// gate proves collapse for fold-identity production models, not in
    /// general. A `ConstantProductivity` plant with MIXED-binding-side
    /// same-bus groups does NOT collapse under exact arithmetic, and that is
    /// correct behaviour (per-group limits are the more faithful model) —
    /// pinned as a documented negative by
    /// `test_mixed_binding_same_bus_groups_do_not_collapse`.
    #[test]
    fn test_three_same_bus_groups_collapse_to_a_single_mirror_group() {
        let variant_a = build_template(groups_three_same_bus());
        let variant_b = build_template(groups_single_mirror());
        assert_templates_byte_identical(
            &variant_a,
            &variant_b,
            "three-same-bus-groups vs single-mirror-group",
        );
    }

    /// A minimal 1-bus, 1-hydro `ConstantProductivity` system whose single
    /// hydro's `unit_groups` are set by the caller — isolated from the
    /// shared FPHA/`H_OTHER` fixture family above (this test's own H1/H2
    /// arithmetic needs bounds only this test controls).
    fn build_mixed_binding_template(unit_groups: Vec<HydroUnitGroup>) -> StageTemplate {
        const MIXED_BUS: EntityId = EntityId(500);
        const MIXED_HYDRO: EntityId = EntityId(510);
        // The raw sum of the three groups' own bounds below (100+10+60,
        // 50+100+200) -- H1, capacity conservation, exact.
        const PLANT_MAX_TURBINED: f64 = 170.0;
        const PLANT_MAX_GENERATION: f64 = 350.0;

        let bus = make_bus(
            MIXED_BUS,
            BusSpec {
                name: "B_MIXED".to_string(),
                operational_start_date: neutral_date(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );
        let hydro = make_hydro(
            MIXED_HYDRO,
            HydroSpec {
                name: "H_MIXED".to_string(),
                bus_id: MIXED_BUS,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: PLANT_MAX_TURBINED,
                min_generation_mw: 0.0,
                max_generation_mw: PLANT_MAX_GENERATION,
                unit_groups,
                ..Default::default()
            },
        );

        let bounds = generous_resolved_bounds(&BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        });
        let penalties = zero_resolved_penalties(&PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        });

        let stage = make_stage(
            0,
            StageSpec {
                blocks: vec![Block {
                    index: 0,
                    name: "B0".to_string(),
                    duration_hours: BLOCK_HOURS,
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
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(vec![stage])
            .inflow_models(vec![InflowModel {
                hydro_id: MIXED_HYDRO,
                stage_id: 0,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            }])
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("mixed-binding fixture must be a valid System");

        let production = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::ConstantProductivity {
                productivity: 1.0,
            }]],
            system.hydros(),
            1,
        );
        let evaporation = PrepareHydroModelsResult::default_from_system(&system).evaporation;

        build_stage_templates_resolving_layout(
            &system,
            InflowNonNegativityMethod::None,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evaporation,
            &ResolvedParameters::default(),
        )
        .expect("mixed-binding template must build")
        .templates
        .into_iter()
        .next()
        .expect("exactly one stage")
    }

    /// The H2 negative, documented rather than left as an unexercised
    /// assumption: a `ConstantProductivity` plant declaring three same-bus
    /// groups whose binding sides are MIXED does NOT collapse to the same
    /// cell bound as the plant declaring none, under exact arithmetic.
    /// `cell_max_turbined` folds each group's own `min(turbined,
    /// generation/rho)` before summing (fold-then-sum); `min` is concave, so
    /// this is strictly less than folding once over the raw totals whenever
    /// the groups bind on different sides
    /// (`Σ_g min(a_g,b_g) < min(Σa_g, Σb_g)`). This is correct behaviour, not
    /// a defect: per-group limits are the more faithful model. See
    /// `test_three_same_bus_groups_collapse_to_a_single_mirror_group`'s doc
    /// for why the FPHA collapse gate above cannot exercise this case (its
    /// production model's fold is the identity).
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_mixed_binding_same_bus_groups_do_not_collapse() {
        // rho = 1.0. G1 (turb 100, gen 50) binds on generation/MW; G2 (turb
        // 10, gen 100) and G3 (turb 60, gen 200) both bind on turbined/flow
        // -- a genuinely mixed set, not merely two-sided by construction.
        // fold-then-sum = min(100,50) + min(10,100) + min(60,200) = 120.0.
        let groups = vec![
            make_unit_group(EntityId(511), EntityId(500), 0.0, 50.0, 0.0, 100.0),
            make_unit_group(EntityId(512), EntityId(500), 0.0, 100.0, 0.0, 10.0),
            make_unit_group(EntityId(513), EntityId(500), 0.0, 200.0, 0.0, 60.0),
        ];
        let with_groups = build_mixed_binding_template(groups);
        let without_groups = build_mixed_binding_template(Vec::new());

        let fold_then_sum_col = find_col_by_upper(&with_groups, 120.0, 1e-9);
        assert_eq!(
            with_groups.col_upper[fold_then_sum_col], 120.0,
            "three mixed-binding groups: the cell bound must be the fold-then-sum \
             (sum of each group's own min), not the raw turbined sum (170.0)"
        );

        // The no-groups plant folds ONCE over its own totals (the raw group
        // sums, 170/350 -- H1 capacity conservation): min(170, 350/1) =
        // 170.0, strictly ABOVE the fold-then-sum value above.
        let single_fold_col = find_col_by_upper(&without_groups, 170.0, 1e-9);
        assert_eq!(
            without_groups.col_upper[single_fold_col], 170.0,
            "the no-groups plant's single fold over its own (raw-sum) totals"
        );

        assert!(
            with_groups.col_upper[fold_then_sum_col] < without_groups.col_upper[single_fold_col],
            "mixed-binding same-bus groups must NOT collapse: {} must be strictly \
             less than {}",
            with_groups.col_upper[fold_then_sum_col],
            without_groups.col_upper[single_fold_col]
        );
    }

    // ---- State-space invariance --------------------------------------------

    /// Declaring groups (including the two-bus variant, which DOES add
    /// columns) never moves `n_state`, and an independently-built
    /// `StateSpace` (unrelated to `unit_groups` at all) agrees. Checkpoint
    /// bytes and the terminal entity manifest are a separate artifact-level
    /// concern this test does not assert.
    #[test]
    fn test_declaring_groups_does_not_move_the_state_space() {
        let tmpl_a = build_template(groups_three_same_bus());
        let tmpl_b = build_template(groups_single_mirror());
        let tmpl_c = build_template(groups_two_buses());

        for (label, t) in [("A", &tmpl_a), ("B", &tmpl_b), ("C", &tmpl_c)] {
            assert_eq!(t.n_state, tmpl_b.n_state, "{label}: n_state");
        }

        // Independent cross-check via a totally separate code path: a
        // StateSpace built directly from (hydro_count, max_par_order) alone
        // -- unit_groups plays no role in state-layout resolution at all.
        let independent = cobre_sddp::test_support::state_layout(2, 0);
        assert_eq!(
            independent.n_state, tmpl_b.n_state,
            "an independently-built StateSpace must agree with the template's n_state"
        );

        // Variant C is the load-bearing case: splitting a plant across two buses
        // adds an extra turbine and an extra FPHA generation column, and this is
        // the proof that neither becomes a state column.
        assert!(
            tmpl_c.num_cols > tmpl_b.num_cols,
            "variant C must add columns relative to B for this check to have power"
        );
    }

    // ---- Multi-bus behavioural gate ------------------------------------------

    /// The multi-bus exit criterion: a plant declaring two groups on two
    /// distinct (non-own) buses credits each cell's generation to its OWN
    /// bus's load-balance row and to no other, while the plant's
    /// water-balance row still sums both cells' turbined flow at the same
    /// `tau`. `num_cols` grows by one turbine + one FPHA generation column
    /// for the new cell, plus one `turbine_below_slack` + one
    /// `generation_below_slack` column (`OperViolationRanges` sizes both
    /// power-family slacks per cell, not per hydro — `n_blks = 1` here).
    #[test]
    fn test_two_bus_plant_credits_each_bus_and_still_balances_water() {
        let tau_h = BLOCK_HOURS * M3S_TO_HM3;

        let tmpl_b = build_template(groups_single_mirror());
        let tmpl_c = build_template(groups_two_buses());

        assert_eq!(
            tmpl_c.num_cols,
            tmpl_b.num_cols + 4,
            "variant C must add 1 turbine + 1 FPHA generation + 1 turbine_below_slack \
             + 1 generation_below_slack column over B (n_blks=1)"
        );

        let cell_first_gen = find_col_by_upper(&tmpl_c, 7_000.0, 1e-9);
        let cell_second_gen = find_col_by_upper(&tmpl_c, 5_000.0, 1e-9);
        let cell_first_turb = find_col_by_upper(&tmpl_c, 450.0, 1e-9);
        let cell_second_turb = find_col_by_upper(&tmpl_c, 300.0, 1e-9);
        assert_ne!(
            cell_first_gen, cell_second_gen,
            "the two cells' generation columns must differ"
        );
        assert_ne!(
            cell_first_turb, cell_second_turb,
            "the two cells' turbine columns must differ"
        );

        let gen_first_rows = col_rows(&tmpl_c, cell_first_gen);
        let gen_second_rows = col_rows(&tmpl_c, cell_second_gen);
        let turb_first_rows = col_rows(&tmpl_c, cell_first_turb);
        let turb_second_rows = col_rows(&tmpl_c, cell_second_turb);

        let fpha_rows_first = common_rows(&gen_first_rows, &turb_first_rows);
        let fpha_rows_second = common_rows(&gen_second_rows, &turb_second_rows);
        assert_eq!(
            fpha_rows_first.len(),
            2,
            "cell Y must own exactly 2 FPHA plane rows (generic + origin)"
        );
        assert_eq!(
            fpha_rows_second.len(),
            2,
            "cell Z must own exactly 2 FPHA plane rows"
        );

        // The load-balance entry: in the generation column's own row set,
        // not one of its cell's FPHA rows (excluded via the turbine
        // column's row set), and not its own cell's min_generation_rows
        // row (a soft floor, `row_upper == +INF`; load-balance is an
        // equality row, `row_upper` always finite -- see
        // `fill_load_balance_rows`/`fill_operational_violation_rows`).
        let load_balance_entry = |tmpl: &StageTemplate,
                                  gen_rows: &[(usize, f64)],
                                  fpha_rows: &[usize]|
         -> (usize, f64) {
            let candidates: Vec<(usize, f64)> = gen_rows
                .iter()
                .copied()
                .filter(|(r, _)| !fpha_rows.contains(r))
                .filter(|(r, _)| tmpl.row_upper[*r].is_finite())
                .collect();
            assert_eq!(
                candidates.len(),
                1,
                "expected exactly one load-balance entry, got {candidates:?}"
            );
            candidates[0]
        };

        let (lb_row_first, lb_coeff_first) =
            load_balance_entry(&tmpl_c, &gen_first_rows, &fpha_rows_first);
        let (lb_row_second, lb_coeff_second) =
            load_balance_entry(&tmpl_c, &gen_second_rows, &fpha_rows_second);

        assert_ne!(
            lb_row_first, lb_row_second,
            "the two cells must credit DIFFERENT load-balance rows (different buses)"
        );
        assert!(
            (lb_coeff_first - 1.0).abs() < 1e-12,
            "cell Y's load-balance credit must be +1.0, got {lb_coeff_first}"
        );
        assert!(
            (lb_coeff_second - 1.0).abs() < 1e-12,
            "cell Z's load-balance credit must be +1.0, got {lb_coeff_second}"
        );
        assert!(
            !gen_second_rows.iter().any(|&(r, _)| r == lb_row_first),
            "cell Z's column must be absent from cell Y's load-balance row"
        );
        assert!(
            !gen_first_rows.iter().any(|&(r, _)| r == lb_row_second),
            "cell Y's column must be absent from cell Z's load-balance row"
        );

        // Water balance: both cells' turbine columns credit the SAME
        // plant-level row (shared, unlike the per-cell FPHA/load-balance
        // rows), at the SAME tau. The two turbine columns share THREE
        // plant-keyed rows in total -- water-balance (coefficient tau_h) and
        // the two hydro-keyed outflow soft floors that also sum every cell's
        // turbine column (min_outflow_rows, max_outflow_rows, each
        // coefficient exactly 1.0, `fill_operational_violation_entries`).
        // min_turbine_rows is NOT among them: it is cell-keyed, so each
        // cell owns its own row rather than sharing the plant's. Disambiguated
        // by value since tau_h != 1.0 for this fixture's block duration.
        let shared_turbine_rows = common_rows(&turb_first_rows, &turb_second_rows);
        assert_eq!(
            shared_turbine_rows.len(),
            3,
            "turbine columns must share exactly 3 plant-keyed rows"
        );
        let water_balance_row = shared_turbine_rows
            .into_iter()
            .find(|&r| (csc_entry_sum(&tmpl_c, cell_first_turb, r) - tau_h).abs() < 1e-9)
            .expect("one shared row must carry tau_h");
        assert!(
            (csc_entry_sum(&tmpl_c, cell_second_turb, water_balance_row) - tau_h).abs() < 1e-9,
            "cell Z's water-balance entry must be the SAME tau_h"
        );
    }

    // ---- Production-row gates: aggregate cap and per-cell attribution ------
    //
    // Both gates read row_upper[r] minus every OTHER column's own rendered
    // coefficient times its value -- NEVER re-derived from the plane literal
    // -- so a coefficient bug the builder actually renders is what these
    // functions observe. They are deliberately encoding-agnostic: nothing
    // here assumes a row belongs to exactly one cell, which is what lets the
    // aggregate-cap check stay meaningful (not merely crash) against an
    // aggregate-only encoding that shares a row across cells.

    /// Which of `H_FPHA`'s two planes to select rows for.
    #[derive(Clone, Copy)]
    enum WhichPlane {
        /// `GENERIC_PLANE`'s row(s): `row_upper > 0` (`sigma_c * 40.0`, any
        /// `sigma_c > 0`).
        Generic,
        /// `ORIGIN_PLANE`'s row(s): `row_upper == 0` (`sigma_c * 0.0`,
        /// always exactly zero regardless of `sigma_c`).
        Origin,
    }

    /// Resolve each `(gen_upper, turb_upper, q_c)` descriptor to its actual
    /// `(gen_col, turb_col, q_c)`.
    fn resolve_cells(tmpl: &StageTemplate, cells: &[(f64, f64, f64)]) -> Vec<(usize, usize, f64)> {
        cells
            .iter()
            .map(|&(gen_upper, turb_upper, q_c)| {
                (
                    find_col_by_upper(tmpl, gen_upper, 1e-6),
                    find_col_by_upper(tmpl, turb_upper, 1e-6),
                    q_c,
                )
            })
            .collect()
    }

    /// The plant's rows for the requested plane, deduplicated across every
    /// cell's own `gen_col ∩ turb_col` intersection. Deduplication is what
    /// keeps a partition-independence sum from over-counting a row that
    /// several cells happen to reference (the shape an aggregate-only
    /// encoding would take): each cell contributes its own 2-row FPHA set
    /// (asserted below), and a row two cells both name is counted once.
    fn plant_plane_rows(
        tmpl: &StageTemplate,
        resolved: &[(usize, usize, f64)],
        which: WhichPlane,
    ) -> Vec<usize> {
        let mut rows: Vec<usize> = Vec::new();
        for &(gen_col, turb_col, _) in resolved {
            let fpha_rows = common_rows(&col_rows(tmpl, gen_col), &col_rows(tmpl, turb_col));
            assert_eq!(
                fpha_rows.len(),
                2,
                "each cell must own exactly 2 FPHA rows (generic + origin planes)"
            );
            for r in fpha_rows {
                if !rows.contains(&r) {
                    rows.push(r);
                }
            }
        }
        rows.into_iter()
            .filter(|&r| match which {
                WhichPlane::Generic => tmpl.row_upper[r] > 1e-9,
                WhichPlane::Origin => tmpl.row_upper[r].abs() <= 1e-9,
            })
            .collect()
    }

    /// `row`'s allowance on `Σ (this plant's generation columns)`:
    /// `row_upper[r]` less every OTHER column's own rendered coefficient
    /// times its value -- turbine columns via the caller's known
    /// `(turb_col, q_c)` map (so a row referencing more than one cell's
    /// turbine column, as an aggregate-only encoding's would, is evaluated
    /// correctly rather than silently mis-attributed), storage/spillage via
    /// pairing equal coefficients (`v_each` stands for BOTH `V_in` and
    /// `V_out`, so which physical column is which cannot matter here --
    /// `chronological_d06_gamma_v_on_both_block_columns` is the dedicated
    /// pin for that contract, not this test's job).
    fn row_allowance_excluding_generation(
        tmpl: &StageTemplate,
        row: usize,
        gen_cols: &[usize],
        turb_q: &[(usize, f64)],
        v_each: f64,
        s: f64,
    ) -> f64 {
        let mut non_flow_terms: Vec<f64> = Vec::new();
        let mut q_contribution = 0.0;
        for (col, val) in row_entries(tmpl, row) {
            if gen_cols.contains(&col) {
                continue;
            }
            if let Some(&(_, q_c)) = turb_q.iter().find(|&&(tc, _)| tc == col) {
                q_contribution += val * q_c;
            } else {
                non_flow_terms.push(val);
            }
        }
        assert_eq!(
            non_flow_terms.len(),
            3,
            "row {row}: expected 2 storage + 1 spillage entry after excluding this plant's \
             generation/turbine columns, got {non_flow_terms:?}"
        );
        let mut sorted = non_flow_terms.clone();
        sorted.sort_by(f64::total_cmp);
        // Two of the three share a coefficient -- V_in and V_out under the
        // average-storage contract; the remaining, distinct one is spillage.
        let (pair_val, odd_val) = if (sorted[0] - sorted[1]).abs() < 1e-9 {
            (sorted[0], sorted[2])
        } else if (sorted[1] - sorted[2]).abs() < 1e-9 {
            (sorted[1], sorted[0])
        } else {
            panic!(
                "row {row}: no two of the three non-flow entries match \
                 (average-storage contract broken?): {non_flow_terms:?}"
            );
        };
        tmpl.row_upper[row] - (pair_val * 2.0 * v_each + q_contribution + odd_val * s)
    }

    /// The aggregate cap: sum `row_allowance_excluding_generation`
    /// over every one of the plant's (deduplicated) generic-plane rows.
    fn aggregate_cap(tmpl: &StageTemplate, cells: &[(f64, f64, f64)], v_each: f64, s: f64) -> f64 {
        let resolved = resolve_cells(tmpl, cells);
        let gen_cols: Vec<usize> = resolved.iter().map(|&(g, _, _)| g).collect();
        let turb_q: Vec<(usize, f64)> = resolved.iter().map(|&(_, t, q)| (t, q)).collect();
        plant_plane_rows(tmpl, &resolved, WhichPlane::Generic)
            .into_iter()
            .map(|row| row_allowance_excluding_generation(tmpl, row, &gen_cols, &turb_q, v_each, s))
            .sum()
    }

    /// For one `(block, plane)` (the generic plane, block 0),
    /// the sum over every one of the plant's FPHA rows of that plane's
    /// implied bound on `Σ_c g_c` is independent of the cell partition, at
    /// the SAME total turbined flow, storage, and spillage. Three partitions
    /// (1, 2, 3 cells) of the same plant: a partition-count-scaling bug
    /// (`(n-1)*A` inflation) is caught by any pair, but a constant-offset bug
    /// needs three to distinguish from a coincidence.
    #[test]
    fn test_production_row_block_cap_is_independent_of_the_cell_partition() {
        const V_EACH: f64 = 80.0;
        const SPILLAGE: f64 = 5.0;
        // Relative, not to_bits: three partitions sum a different number of
        // terms, so floating-point association legitimately differs by a
        // handful of ULPs. 1e-9 relative is generous versus that noise floor
        // yet three orders of magnitude tighter than the smallest bug this
        // gate exists to catch (dropping/duplicating one plant-level term,
        // ~40 out of ~442 -- a ~9% relative error).
        const REL_TOL: f64 = 1e-9;

        let tmpl_1 = build_template(groups_single_mirror());
        let tmpl_2 = build_template(groups_two_buses());
        let tmpl_3 = build_template(groups_three_buses());

        // Same total turbined flow (500.0 m3/s), split differently per
        // partition; (gen_upper, turb_upper) identify each cell.
        let cap_1 = aggregate_cap(&tmpl_1, &[(12_000.0, 750.0, 500.0)], V_EACH, SPILLAGE);
        let cap_2 = aggregate_cap(
            &tmpl_2,
            &[(7_000.0, 450.0, 200.0), (5_000.0, 300.0, 300.0)],
            V_EACH,
            SPILLAGE,
        );
        let cap_3 = aggregate_cap(
            &tmpl_3,
            &[
                (3_000.0, 200.0, 150.0),
                (4_000.0, 250.0, 200.0),
                (5_000.0, 300.0, 150.0),
            ],
            V_EACH,
            SPILLAGE,
        );

        // Hand-derived cross-check: the 1-cell cap is exactly its own
        // single row's allowance, gamma0 + gammaV*Vbar + gammaQ*Sigma_q +
        // gammaS*s = 40.0 + 0.02*80.0 + 0.8*500.0 + 0.05*5.0.
        let expected = GENERIC_PLANE.intercept
            + GENERIC_PLANE.gamma_v * V_EACH
            + GENERIC_PLANE.gamma_q * 500.0
            + GENERIC_PLANE.gamma_s * SPILLAGE;
        assert!(
            (cap_1 - expected).abs() < 1e-6,
            "1-cell cap must equal the closed-form gamma0 + gammaV*Vbar + gammaQ*Sq + gammaS*s: \
             got {cap_1}, expected {expected}"
        );

        for (label, other) in [("2-cell", cap_2), ("3-cell", cap_3)] {
            assert!(
                (other - cap_1).abs() <= REL_TOL * cap_1.abs(),
                "aggregate cap must be partition-independent: 1-cell={cap_1}, {label}={other}"
            );
        }

        // The per-cell attribution gate: the ORIGIN plane pins a zero-flow cell to zero
        // generation -- a DIFFERENT property from the aggregate-cap identity
        // above, which an aggregate-only (one-row-per-plant) encoding
        // satisfies exactly (as proven above) while providing no per-cell
        // attribution at all. The attribution gate is therefore two parts:
        // (a) a row exclusively naming this cell's own generation column
        // must exist among the origin-plane rows -- an aggregate-only
        // encoding, where every row names every cell's generation column,
        // has NO such row, so this is where that encoding is caught; (b)
        // among the rows that DO exclusively attribute, the implied bound at
        // q_c=0 must be <=0.
        let resolved_2 = resolve_cells(&tmpl_2, &[(7_000.0, 450.0, 0.0), (5_000.0, 300.0, 0.0)]);
        let gen_cols_2: Vec<usize> = resolved_2.iter().map(|&(g, _, _)| g).collect();
        let origin_rows_2 = plant_plane_rows(&tmpl_2, &resolved_2, WhichPlane::Origin);
        assert_eq!(
            origin_rows_2.len(),
            2,
            "the two cells' origin-plane rows must be distinct (2 total), not shared"
        );

        for &(gen_col, turb_col, _) in &resolved_2 {
            let exclusive: Vec<usize> = origin_rows_2
                .iter()
                .copied()
                .filter(|&r| {
                    let present: Vec<usize> = row_entries(&tmpl_2, r)
                        .into_iter()
                        .map(|(c, _)| c)
                        .filter(|c| gen_cols_2.contains(c))
                        .collect();
                    present == [gen_col]
                })
                .collect();
            assert_eq!(
                exclusive.len(),
                1,
                "cell (gen col {gen_col}) must have exactly one origin-plane row that names \
                 ONLY its own generation column, got {exclusive:?} among {origin_rows_2:?}"
            );
            let row = exclusive[0];
            // q_c = 0.0 for this cell; the other cell's turbine column, if
            // this row named it, would poison the bound with an unrelated
            // flow -- excluded by construction since this row is exclusive.
            let bound = row_allowance_excluding_generation(
                &tmpl_2,
                row,
                &gen_cols_2,
                &[(turb_col, 0.0)],
                V_EACH,
                SPILLAGE,
            );
            assert!(
                bound <= 1e-9,
                "cell (gen col {gen_col}) at q_c=0 must be capped at <=0 by the origin plane, \
                 got {bound}"
            );
        }
    }
}
