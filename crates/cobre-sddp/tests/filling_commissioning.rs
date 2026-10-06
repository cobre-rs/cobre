//! Consolidated filling / commissioning integration tests for `cobre-sddp`.
//!
//! Each filling/commissioning domain group lives in its own inner `mod` so the
//! suite links the statically-bound solver once rather than once per file.
//! Per-`mod` scoping isolates each group's consts, helpers, and fixtures.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

mod common;

mod d38_dead_volume_filling_simulation {
    //! Hydro dead-volume filling must reach the simulation output and bind under
    //! short inflow.
    //!
    //! ## What the parity hash cannot see
    //!
    //! The declaration-order parity baseline hashes hydro storage / water values and
    //! cuts; it does NOT hash the filling soft-penalty slacks. The per-stage
    //! `σ_fill` (`filling_target_violation_hm3`) and the operating soft-floor
    //! `σ^{v-}` (`storage_violation_below_hm3`) are invisible to the hash, as is the
    //! PreFilling cascade short-circuit that routes an absent dam's water onto its
    //! real downstream's water-balance row. A gating bug that let the slacks stay
    //! zero under a short fill, or dropped the routed water into a sink, would still
    //! hash-match. This test exercises those paths directly through the full
    //! train+simulate pipeline.
    //!
    //! ## Fixture (`d38-dead-volume-filling`)
    //!
    //! Cascade `H1 (id 0) → H2 (id 1, filling) → H3 (id 2, real fed downstream)` plus
    //! an off-cascade non-filling control `H4 (id 3)`. `H2` carries
    //! `entry_stage_id = 4` and `filling { start_stage_id = 2, filling_min_rate_m3s = 12 }`,
    //! so over the 6-stage horizon it is PreFilling at ids 0–1, Filling at ids 2–3,
    //! and Operating at ids 4–5. Block counts change at BOTH phase boundaries
    //! (schedule 1/1/3/2/3/1: 1→3 at stage 1→2, 2→3 at stage 3→4), exercising the
    //! per-stage geometry and per-stage `τ`. All hydros use `constant_productivity`
    //! (FPHA-during-filling exclusion is out of scope here; it is unit-tested
    //! separately). A single filling hydro is the design baseline — the chained
    //! transitive short-circuit walk is covered by `filling_cut_validity`.
    //!
    //! The volume-target filling model gives each Filling stage a per-stage SOFT
    //! floor `v_out[t] + σ_fill[t] ≥ V_target[t]` (`fill_filling_target_rows`:
    //! row_upper = +∞), with the target anchored backward from the dead volume:
    //! `V_target[3] = min_storage_hm3 = 60` and `V_target[2] = 60 − ζ·rate = 28.896`
    //! hm³. The end-of-Filling storage at id 3 hands off to the incoming storage at id
    //! 4 with no reset (the pin chain). Because the floor is soft and Filling spillage
    //! is FREE, H2 is not forced to hold its water across Filling: with the downstream
    //! H3 starved (the PreFilling phantom-spill removed), the LP optimally releases the
    //! impounded water at the last Filling stage (id 3) to feed H3, driving `σ_fill[3]`
    //! to the full dead-volume shortfall. d40 (byte-identical penalties) suppresses the
    //! same release only via a downstream turbine cap that d38 lacks. The inflow
    //! schedule (`std_m3s = 0` everywhere ⇒ deterministic) keeps `σ_fill[t] > 0` at
    //! BOTH filling stages (ids 2 and 3); inflows RECOVER in Operating (60 m³/s) so H2
    //! climbs above the dead volume and `σ^{v-} → 0` by id 5. `H1` is starved before
    //! Operating (seed 0, inflow 2/2/0/0 m³/s over ids 0–3).

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;
    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::{
        SolverStatsDelta, StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    // ---------------------------------------------------------------------------
    // Fixture constants (study stage ids == indices for this single-resolution horizon)
    // ---------------------------------------------------------------------------

    /// PreFilling → Filling boundary for `H2`: filling begins at this stage id.
    const START_STAGE_ID: i32 = 2;
    /// Filling → Operating boundary for `H2`: it becomes a normal plant at this id.
    const ENTRY_STAGE_ID: i32 = 4;
    const H1_ID: i32 = 0;
    const H2_ID: i32 = 1;
    const H3_ID: i32 = 2;
    const H4_ID: i32 = 3;

    /// `H2`'s dead volume (filling target / soft operating floor), hm³. Mirrors
    /// `system/hydros.json` `H2.reservoir.min_storage_hm3`.
    const H2_MIN_STORAGE_HM3: f64 = 60.0;
    /// `H2`'s per-stage minimum accumulation rate (m³/s). Mirrors
    /// `system/hydros.json` `H2.filling.filling_min_rate_m3s`.
    const H2_FILLING_MIN_RATE_M3S: f64 = 12.0;
    /// `H4`'s reservoir bounds (hm³), mirroring `system/hydros.json`.
    const H4_MIN_STORAGE_HM3: f64 = 0.0;
    const H4_MAX_STORAGE_HM3: f64 = 200.0;
    /// `H4`'s stage-0 seed (hm³), mirroring `initial_conditions.json`.
    const H4_SEED_HM3: f64 = 100.0;

    /// m³/s → hm³ per hour. Mirrors `crate::lp::builder::M3S_TO_HM3` (a private const
    /// not exported to integration tests). `ζ = total_stage_hours · M3S_TO_HM3`.
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    /// Every stage in `stages.json` totals 720 h, so `ζ` is uniform.
    const STAGE_HOURS: f64 = 720.0;

    /// Per-(stage, hydro) view collapsing the stage's per-block rows to the scalars the
    /// assertions need: storage and the σ_fill / σ^{v-} slacks are stage-level
    /// (block-invariant), read from the first block; generation and turbined flow are
    /// summed across blocks. The water balance reads the per-block rows directly (it
    /// needs per-block τ), not this aggregate.
    struct StageHydro {
        storage_initial_hm3: f64,
        storage_final_hm3: f64,
        incremental_inflow_m3s: f64,
        generation_mw: f64,
        turbined_total_m3s: f64,
        filling_target_violation_hm3: f64,
        storage_violation_below_hm3: f64,
    }

    /// Collect the per-stage hydro view for `hydro_id` from a scenario result.
    fn stage_hydro(
        scenario: &cobre_sddp::SimulationScenarioResult,
        hydro_id: i32,
        stage_index: usize,
    ) -> StageHydro {
        let rows: Vec<_> = scenario.stages[stage_index]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == hydro_id)
            .collect();
        assert!(
            !rows.is_empty(),
            "hydro {hydro_id} must have at least one row at stage {stage_index}"
        );
        let first = rows[0];
        StageHydro {
            storage_initial_hm3: first.storage_initial_hm3,
            storage_final_hm3: first.storage_final_hm3,
            incremental_inflow_m3s: first.incremental_inflow_m3s,
            generation_mw: rows.iter().map(|r| r.generation_mw).sum(),
            turbined_total_m3s: rows.iter().map(|r| r.turbined_m3s).sum(),
            filling_target_violation_hm3: first.filling_target_violation_hm3,
            storage_violation_below_hm3: first.storage_violation_below_hm3,
        }
    }

    /// Train the d38 dead-volume-filling case, simulate one deterministic scenario,
    /// and assert the obligations the parity hash cannot see.
    // Rationale: splitting the body into per-criterion helpers would re-run the one
    // expensive train+simulate pipeline per helper (or thread the whole scenario
    // through opaque args).
    #[allow(clippy::too_many_lines)]
    #[test]
    fn dead_volume_filling_binds_and_routes_in_simulation() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d38-dead-volume-filling");

        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system_for_check = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let h2 = system_for_check
            .hydros()
            .iter()
            .find(|h| h.id.0 == H2_ID)
            .expect("H2 must be present");
        let filling = h2.filling.as_ref().expect("H2.filling must be Some");
        assert_eq!(
            filling.start_stage_id, START_STAGE_ID,
            "H2 filling start_stage_id"
        );
        assert_eq!(h2.entry_stage_id, Some(ENTRY_STAGE_ID), "H2 entry_stage_id");
        assert_eq!(
            h2.downstream_id.map(|d| d.0),
            Some(H3_ID),
            "H2 must short-circuit onto H3 (its real fed downstream)"
        );

        // The shipped parity case trains only; enable one sim scenario so the extraction path runs.
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

        let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(
                &mut solver,
                &comm,
                1,
                ActiveSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error (an infeasible filling stage is a real data finding, not \
         a bound to relax): {:?}",
            outcome.error
        );

        let lower_bounds: Vec<f64> = event_rx
            .try_iter()
            .filter_map(|e| {
                if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                    Some(lower_bound)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !lower_bounds.is_empty(),
            "must capture at least one ConvergenceUpdate lower bound"
        );
        for window in lower_bounds.windows(2) {
            let (prev, next) = (window[0], window[1]);
            let tol = 1e-6 * prev.abs().max(1.0);
            assert!(
                next >= prev - tol,
                "lower bound must be monotone within FP tolerance: {prev} -> {next} \
             (allowed slack {tol})"
            );
        }

        // Aggregate ONLY the two filling boundary stages (id == index here): a global
        // aggregate would dilute a boundary spike.
        let boundary_stages = [START_STAGE_ID, ENTRY_STAGE_ID];
        let boundary_rejections = SolverStatsDelta::aggregate(
            outcome
                .result
                .solver_stats_log
                .iter()
                .filter(|entry| {
                    matches!(entry.phase, "forward" | "backward")
                        && entry.stage_id.is_some_and(|s| boundary_stages.contains(&s))
                })
                .map(|entry| &entry.delta),
        )
        .basis_consistency_failures;
        assert_eq!(
            boundary_rejections, 0,
            "filling boundaries: expected 0 basis rejections at PreFilling->Filling \
         (id {START_STAGE_ID}) and Filling->Operating (id {ENTRY_STAGE_ID}); got \
         {boundary_rejections} (a rejection means a cut row/column was relocated \
         across the phase change, breaking reconstruct_basis slot-identity matching)"
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
            .expect("simulate must not return Err (the soft slacks keep every stage feasible)");

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
            6,
            "one stage record per study stage (horizon ids 0..=5)",
        );

        let zeta = STAGE_HOURS * M3S_TO_HM3;

        // H2's turbine/generation columns are pinned [0, 0] until Operating (PreFilling
        // 0–1, Filling 2–3).
        for stage in 0..(ENTRY_STAGE_ID as usize) {
            let h2_s = stage_hydro(scenario, H2_ID, stage);
            assert!(
                h2_s.generation_mw.abs() < 1e-6,
                "H2 must not generate before entry (stage {stage}); got {}",
                h2_s.generation_mw
            );
            assert!(
                h2_s.turbined_total_m3s.abs() < 1e-6,
                "H2 must not turbine before entry (stage {stage}); got {}",
                h2_s.turbined_total_m3s
            );
        }

        // H2 spillage is frozen [0,0] only while PreFilling (the "spillage frozen during
        // PreFilling" contract, D38/D40); it is FREE during Filling, so the LP may
        // legitimately release the impounded water at the last Filling stage (id 3) to
        // feed the starved downstream H3. The σ_fill row is a SOFT floor
        // (`fill_filling_target_rows`: row_upper = +∞), so storage is NOT required to be
        // monotone across Filling.
        let h2_s0 = stage_hydro(scenario, H2_ID, 0);
        let h2_s1 = stage_hydro(scenario, H2_ID, 1);
        let seed = h2_s0.storage_initial_hm3;
        assert!(
            seed.abs() < 1e-6,
            "H2 PreFilling seed must be the empty pit 0.0; got {seed}"
        );
        for (stage, h2) in [(0_usize, &h2_s0), (1, &h2_s1)] {
            assert!(
                (h2.storage_final_hm3 - seed).abs() < 1e-6,
                "H2 storage must be frozen at the seed through PreFilling stage {stage}; got {}",
                h2.storage_final_hm3
            );
        }

        // Per-stage Filling water balance (ids 2–3), every term pinned to FP tolerance:
        // v_out = v_in + ζ·incr_H2 − τ·spillage_H2 + routed_upstream(H1→H2). H2 turbine
        // and diversion are 0 while Filling, so only its spillage subtracts. The upstream
        // H1 release enters with per-block τ_k = block_hours·M3S_TO_HM3, summed across
        // the stage's blocks so the balance is exact regardless of block count.
        let h2_incr_filling_m3s = 5.0; // scenarios/inflow_seasonal_stats.parquet, H2 ids 2–3
        for stage in (START_STAGE_ID as usize)..(ENTRY_STAGE_ID as usize) {
            let h2 = stage_hydro(scenario, H2_ID, stage);
            let h2_spill_release_hm3: f64 = scenario.stages[stage]
                .hydros
                .iter()
                .filter(|r| r.hydro_id == H2_ID)
                .map(|r| r.spillage_m3s * block_hours(stage, r.block_id) * M3S_TO_HM3)
                .sum();
            let routed_upstream_hm3: f64 = scenario.stages[stage]
                .hydros
                .iter()
                .filter(|r| r.hydro_id == H1_ID)
                .map(|r| {
                    (r.turbined_m3s + r.spillage_m3s) * block_hours(stage, r.block_id) * M3S_TO_HM3
                })
                .sum();
            let expected_v_out = h2.storage_initial_hm3 + zeta * h2_incr_filling_m3s
                - h2_spill_release_hm3
                + routed_upstream_hm3;
            assert!(
                (h2.storage_final_hm3 - expected_v_out).abs() < 1e-3,
                "Filling stage {stage}: H2's closed water balance must hold — v_out {} = v_in {} + \
             ζ·incr {} − spill_release {h2_spill_release_hm3} + routed_upstream {routed_upstream_hm3}; \
             expected {expected_v_out}",
                h2.storage_final_hm3,
                h2.storage_initial_hm3,
                zeta * h2_incr_filling_m3s
            );
        }

        // When H2 releases at the last Filling stage (v_out[3] < v_out[2] with
        // spillage[3] > 0), the released water must reach the starved downstream H3, and
        // σ_fill[3] must book the dead-volume shortfall exactly: σ_fill[3] = V_target[3]
        // − v_out[3].
        let h2_s2 = stage_hydro(scenario, H2_ID, 2);
        let h2_s3 = stage_hydro(scenario, H2_ID, 3);
        let h2_s3_spill_total: f64 = scenario.stages[3]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == H2_ID)
            .map(|r| r.spillage_m3s)
            .sum();
        if h2_s3.storage_final_hm3 < h2_s2.storage_final_hm3 - 1e-6 {
            assert!(
                h2_s3_spill_total > 1e-6,
                "H2 storage fell across Filling ids 2→3 ({} → {}); the only free release path while \
             Filling is spillage, so spillage[3] must be > 0; got {h2_s3_spill_total}",
                h2_s2.storage_final_hm3,
                h2_s3.storage_final_hm3
            );
            let h3_s3 = stage_hydro(scenario, H3_ID, 3);
            let h3_release_hm3: f64 = scenario.stages[3]
                .hydros
                .iter()
                .filter(|r| r.hydro_id == H3_ID)
                .map(|r| {
                    (r.turbined_m3s + r.spillage_m3s) * block_hours(3, r.block_id) * M3S_TO_HM3
                })
                .sum();
            let h3_delta = h3_s3.storage_final_hm3 - h3_s3.storage_initial_hm3;
            let h3_incremental_balance = zeta * h3_s3.incremental_inflow_m3s - h3_release_hm3;
            let h3_routed_gap = h3_delta - h3_incremental_balance;
            let h2_released_hm3: f64 = scenario.stages[3]
                .hydros
                .iter()
                .filter(|r| r.hydro_id == H2_ID)
                .map(|r| r.spillage_m3s * block_hours(3, r.block_id) * M3S_TO_HM3)
                .sum();
            assert!(
                h3_routed_gap >= h2_released_hm3 - 1e-3,
                "H2's last-Filling release ({h2_released_hm3:.6} hm³ of spillage) must reach H3: H3's \
             routed-water gap ({h3_routed_gap:.6} hm³) must be at least H2's released volume — \
             conservation, not a vanished dump"
            );
            let v_target_3 = H2_MIN_STORAGE_HM3;
            assert!(
                (h2_s3.filling_target_violation_hm3 - (v_target_3 - h2_s3.storage_final_hm3)).abs()
                    < 1e-3,
                "σ_fill[3] must book the dead-volume shortfall exactly: σ_fill {} = V_target[3] {} − \
             v_out[3] {}",
                h2_s3.filling_target_violation_hm3,
                v_target_3,
                h2_s3.storage_final_hm3
            );
        }

        // Water balance on H3, re-anchored (does NOT use inflow_m3s): during
        // PreFilling the dam at H2 does not exist, so H2's water short-circuits onto
        // H3's real water-balance row. The closed water balance on H3 is
        //     Δstorage = ζ·incr_H3 − release_H3 + ROUTED,
        // where ROUTED = ζ·incr_H2 + (any H1 release re-routed H1→H3) + H2's withdrawal.
        // H1 is starved before Operating, so its PreFilling release is small or zero;
        // the asserted floor therefore rests on H2's OWN incremental alone. Rearranged,
        // the GAP between H3's actual Δstorage and its incremental-only balance IS the
        // routed contribution: it must be strictly positive, proving the water lands on
        // H3's row (H3 is a fed reservoir, not a sink). It must moreover be at least
        // H2's own routed incremental ζ·incr_H2 — a known, deterministic quantity
        // (std=0) — which is a quantitative floor on the routing, not just
        // `> incremental_only`.
        let h2_prefilling_incr_m3s = 15.0; // mirrors scenarios/inflow_seasonal_stats.parquet, H2 ids 0–1
        let routed_floor_hm3 = zeta * h2_prefilling_incr_m3s;
        for prefilling_stage in [0_usize, 1] {
            let h3_s = stage_hydro(scenario, H3_ID, prefilling_stage);
            let delta_storage = h3_s.storage_final_hm3 - h3_s.storage_initial_hm3;
            // Release uses per-block τ_k, not a stage-level ζ: a single ζ mis-weights the
            // multi-block 1/1/3/2/3/1 schedule.
            let release_hm3: f64 = scenario.stages[prefilling_stage]
                .hydros
                .iter()
                .filter(|r| r.hydro_id == H3_ID)
                .map(|r| {
                    (r.turbined_m3s + r.spillage_m3s)
                        * block_hours(prefilling_stage, r.block_id)
                        * M3S_TO_HM3
                })
                .sum();
            let incremental_balance = zeta * h3_s.incremental_inflow_m3s - release_hm3;
            let routed_gap = delta_storage - incremental_balance;
            assert!(
                routed_gap >= routed_floor_hm3 - 1e-3,
                "PreFilling stage {prefilling_stage}: the routed-water gap on H3 \
             ({routed_gap:.6} hm³ = Δstorage {delta_storage:.6} − incremental-only balance \
             {incremental_balance:.6}) must be at least H2's own routed incremental \
             ζ·incr_H2 = {routed_floor_hm3:.6} hm³; a smaller gap means short-circuited water \
             was dropped before reaching H3's row (a sink bug). H3 incremental_inflow = {} m³/s, \
             release = {release_hm3:.6} hm³",
                h3_s.incremental_inflow_m3s
            );
        }

        // σ_fill binds at BOTH Filling stages (ids 2 and 3) under the short inflow, and
        // σ^{v-} → 0 by the last Operating stage (id 5) on recovery. The
        // volume-target model imposes a per-stage soft floor v_out[t] + σ_fill[t] ≥
        // V_target[t]; with the short inflow the inflow-only accumulation (12.96 hm³
        // by id 2, 25.92 hm³ by id 3) stays below the backward-anchored targets
        // (V_target[2] = min_storage − ζ·rate = 28.896, V_target[3] = min_storage =
        // 60.0), so σ_fill is strictly positive at each filling stage. A terminal-only
        // check would miss a per-stage floor regression that binds only at the last
        // Filling stage.
        let v_target_2 = H2_MIN_STORAGE_HM3 - zeta * H2_FILLING_MIN_RATE_M3S;
        for (filling_stage, v_target) in [(2_usize, v_target_2), (3, H2_MIN_STORAGE_HM3)] {
            let sigma_fill =
                stage_hydro(scenario, H2_ID, filling_stage).filling_target_violation_hm3;
            assert!(
                sigma_fill > 1e-6,
                "H2 σ_fill must be strictly positive at Filling stage {filling_stage} (the \
             short inflow leaves storage below V_target[{filling_stage}] = {v_target}); got \
             {sigma_fill}"
            );
        }
        let sigma_floor_s5 = stage_hydro(scenario, H2_ID, 5).storage_violation_below_hm3;
        assert!(
            sigma_floor_s5 < 1e-6,
            "H2 σ^{{v-}} must be ~0 at the last Operating stage (id 5) after the inflow \
         recovers and storage climbs above the dead volume {H2_MIN_STORAGE_HM3}; got \
         {sigma_floor_s5}"
        );

        let h2_s3_final = stage_hydro(scenario, H2_ID, 3).storage_final_hm3;
        let h2_s4_initial = stage_hydro(scenario, H2_ID, 4).storage_initial_hm3;
        assert!(
            (h2_s3_final - h2_s4_initial).abs() < 1e-6,
            "end-of-Filling storage at id 3 ({h2_s3_final}) must equal incoming storage at \
         id 4 ({h2_s4_initial}) via the pin chain"
        );

        let mut h4_moved = false;
        for stage in 0..6 {
            let h4_s = stage_hydro(scenario, H4_ID, stage);
            for v in [h4_s.storage_initial_hm3, h4_s.storage_final_hm3] {
                assert!(
                    (H4_MIN_STORAGE_HM3 - 1e-6..=H4_MAX_STORAGE_HM3 + 1e-6).contains(&v),
                    "control H4 storage must stay within [{H4_MIN_STORAGE_HM3}, \
                 {H4_MAX_STORAGE_HM3}] at stage {stage}; got {v}"
                );
            }
            if (h4_s.storage_final_hm3 - H4_SEED_HM3).abs() > 1e-6 {
                h4_moved = true;
            }
        }
        assert!(
            h4_moved,
            "control H4 must dispatch (its storage moves off the {H4_SEED_HM3} hm³ seed), not \
         stay frozen like a PreFilling hydro"
        );
    }

    /// Block duration in hours for `(stage_index, block_id)`, mirroring `stages.json`.
    /// `block_id` is always `Some` for a turbine-branch hydro row.
    fn block_hours(stage_index: usize, block_id: Option<u32>) -> f64 {
        let blk = block_id.expect("turbine-branch hydro rows carry a block id") as usize;
        match stage_index {
            0 | 1 | 5 => 720.0,
            2 | 4 => [240.0, 240.0, 240.0][blk],
            3 => [360.0, 360.0][blk],
            _ => unreachable!("d38 has stages 0..=5"),
        }
    }
}

mod d40_filling_cascade_simulation {
    //! A filling cascade — two reservoirs in the Filling phase at the same stages —
    //! needs no special handling: each carries its OWN per-stage soft floor and the
    //! two couple ONLY through normal cascade releases.
    //!
    //! ## What the parity hash cannot see
    //!
    //! The declaration-order parity baseline hashes hydro storage / water values and
    //! cuts; it does NOT hash the filling soft-penalty slacks. The per-stage `σ_fill`
    //! (`filling_target_violation_hm3`) is invisible to the hash. A regression that
    //! coupled the two filling floors through a shared inter-reservoir target term —
    //! instead of letting each floor read its OWN `V_target` trajectory — could still
    //! hash-match while silently producing a wrong (shared) shortfall. This test
    //! exercises the per-floor independence and the release-only cascade coupling
    //! directly through the full train+simulate pipeline.
    //!
    //! ## Fixture (`d40-filling-cascade`)
    //!
    //! Cascade `H_up (id 0, filling) → H_down (id 1, filling) → H_sink (id 2, real
    //! outlet)` plus an off-cascade non-filling control `H_ctrl (id 3)`. BOTH filling
    //! hydros carry `entry_stage_id = 4` and `filling { start_stage_id = 1,
    //! filling_min_rate_m3s = 12 }`, so over the 6-stage horizon both are PreFilling
    //! at id 0, Filling at ids 1–3, and Operating at ids 4–5 — the one topology with
    //! two reservoirs in the Filling phase simultaneously. Block counts change across
    //! the 1/1/3/2/3/1 schedule, exercising the per-stage geometry and per-stage `τ`.
    //! All hydros use `constant_productivity`.
    //!
    //! The volume-target model gives each Filling stage a per-stage soft floor
    //! `v_out[t] + σ_fill[t] ≥ V_target[t]`, with the target anchored backward from
    //! the dead volume: `V_target[3] = min_storage = 60`, `V_target[2] = 60 − ζ·rate =
    //! 28.896` hm³, and `V_target[1] = 28.896 − ζ·rate = −2.208` clipped negative
    //! (trivially satisfied ⇒ `σ_fill[1] = 0`). With `std_m3s = 0` everywhere the
    //! trajectory is deterministic. The own incremental inflows are DISTINCT (`H_up`
    //! 5 m³/s, `H_down` 3 m³/s over Filling) so each hydro's inflow-only accumulation
    //! falls short of its `V_target` at the binding stages 2 and 3 by a DIFFERENT
    //! amount — the per-floor independence is observable as two distinct `σ_fill`
    //! values. During Filling both impound (turbine pinned `[0,0]`), so `H_up`'s
    //! release onto `H_down` is ~0 at the Filling stages: the cascade couples
    //! release-only, with the gap between `H_down`'s Δstorage and its incremental-only
    //! balance equal to `H_up`'s routed release (an exact identity, no phantom
    //! inter-floor term). The cascade outlet `H_sink` carries a small turbine
    //! (`max_turbined 20 m³/s`) so the finite, zero-discount terminal horizon cannot
    //! monetize a last-Filling-stage water dump as cheap hydro generation; without it
    //! the LP would optimally spill the dead-volume water at id 3 (driving `σ_fill[3]`
    //! to the full dead volume for BOTH and collapsing the independence) instead of
    //! holding it. Inflows recover to 60 m³/s in Operating so both climb above the dead
    //! volume and `σ^{v-} → 0`.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;
    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    // ---------------------------------------------------------------------------
    // Fixture constants (study stage ids == indices for this single-resolution horizon)
    // ---------------------------------------------------------------------------

    /// PreFilling → Filling boundary shared by both filling hydros.
    const START_STAGE_ID: i32 = 1;
    /// Filling → Operating boundary shared by both filling hydros.
    const ENTRY_STAGE_ID: i32 = 4;
    const H_UP_ID: i32 = 0;
    const H_DOWN_ID: i32 = 1;
    const H_SINK_ID: i32 = 2;
    const H_CTRL_ID: i32 = 3;

    /// Shared dead volume (filling target / soft operating floor), hm³. Mirrors
    /// `system/hydros.json` `min_storage_hm3` for both filling hydros.
    const MIN_STORAGE_HM3: f64 = 60.0;
    /// Shared per-stage minimum accumulation rate (m³/s). Mirrors `system/hydros.json`
    /// `filling.filling_min_rate_m3s` for both filling hydros.
    const FILLING_MIN_RATE_M3S: f64 = 12.0;
    /// `H_up`'s own incremental inflow over the Filling stages (m³/s). Mirrors
    /// `scenarios/inflow_seasonal_stats.parquet`, id 0 at stages 1–3.
    const H_UP_FILLING_INCR_M3S: f64 = 5.0;
    /// `H_down`'s own incremental inflow over the Filling stages (m³/s). Distinct from
    /// `H_up`'s so the two `σ_fill` shortfalls differ. Mirrors id 1 at stages 1–3.
    const H_DOWN_FILLING_INCR_M3S: f64 = 3.0;
    /// `H_ctrl`'s reservoir bounds (hm³), mirroring `system/hydros.json`.
    const H_CTRL_MIN_STORAGE_HM3: f64 = 0.0;
    const H_CTRL_MAX_STORAGE_HM3: f64 = 200.0;
    /// `H_ctrl`'s stage-0 seed (hm³), mirroring `initial_conditions.json`.
    const H_CTRL_SEED_HM3: f64 = 100.0;

    /// m³/s → hm³ per hour. Mirrors `crate::lp::builder::M3S_TO_HM3` (a private const
    /// not exported to integration tests). `ζ = total_stage_hours · M3S_TO_HM3`.
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    /// Every stage in `stages.json` totals 720 h, so `ζ` is uniform.
    const STAGE_HOURS: f64 = 720.0;

    /// Per-(stage, hydro) view collapsing the stage's per-block rows to the scalars
    /// the assertions need. Storage and the σ_fill / σ^{v-} slacks are stage-level
    /// (block-invariant), so they are read from the first block; turbined flow is
    /// summed across blocks. The cascade water balance reads the per-block rows
    /// directly (it needs per-block τ), not this aggregate.
    struct StageHydro {
        storage_initial_hm3: f64,
        storage_final_hm3: f64,
        incremental_inflow_m3s: f64,
        generation_mw: f64,
        turbined_total_m3s: f64,
        filling_target_violation_hm3: f64,
        storage_violation_below_hm3: f64,
    }

    /// Collect the per-stage hydro view for `hydro_id` from a scenario result.
    fn stage_hydro(
        scenario: &cobre_sddp::SimulationScenarioResult,
        hydro_id: i32,
        stage_index: usize,
    ) -> StageHydro {
        let rows: Vec<_> = scenario.stages[stage_index]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == hydro_id)
            .collect();
        assert!(
            !rows.is_empty(),
            "hydro {hydro_id} must have at least one row at stage {stage_index}"
        );
        let first = rows[0];
        StageHydro {
            storage_initial_hm3: first.storage_initial_hm3,
            storage_final_hm3: first.storage_final_hm3,
            incremental_inflow_m3s: first.incremental_inflow_m3s,
            generation_mw: rows.iter().map(|r| r.generation_mw).sum(),
            turbined_total_m3s: rows.iter().map(|r| r.turbined_m3s).sum(),
            filling_target_violation_hm3: first.filling_target_violation_hm3,
            storage_violation_below_hm3: first.storage_violation_below_hm3,
        }
    }

    /// `H_up`'s whole-stage release (turbine + spillage) routed onto `H_down`'s
    /// water-balance row at `stage_index`, in hm³. Uses per-block τ_k, not a stage-level
    /// ζ: a single ζ mis-weights the multi-block 1/1/3/2/3/1 schedule.
    fn routed_release_hm3(
        scenario: &cobre_sddp::SimulationScenarioResult,
        upstream_id: i32,
        stage_index: usize,
    ) -> f64 {
        scenario.stages[stage_index]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == upstream_id)
            .map(|r| {
                (r.turbined_m3s + r.spillage_m3s)
                    * block_hours(stage_index, r.block_id)
                    * M3S_TO_HM3
            })
            .sum()
    }

    /// Train the d40 filling-cascade case, simulate one deterministic scenario, and
    /// assert the per-floor independence and release-only cascade coupling the parity
    /// hash cannot see.
    // Rationale: splitting the body into per-criterion helpers would re-run the one
    // expensive train+simulate pipeline per helper (or thread the whole scenario
    // through opaque args).
    #[allow(clippy::too_many_lines)]
    #[test]
    fn filling_cascade_floors_are_independent_and_couple_release_only() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d40-filling-cascade");

        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system_for_check = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        for (id, downstream) in [(H_UP_ID, H_DOWN_ID), (H_DOWN_ID, H_SINK_ID)] {
            let h = system_for_check
                .hydros()
                .iter()
                .find(|h| h.id.0 == id)
                .unwrap_or_else(|| panic!("filling hydro {id} must be present"));
            let filling = h
                .filling
                .as_ref()
                .unwrap_or_else(|| panic!("hydro {id} filling must be Some"));
            assert_eq!(
                filling.start_stage_id, START_STAGE_ID,
                "hydro {id} filling start_stage_id"
            );
            assert_eq!(
                h.entry_stage_id,
                Some(ENTRY_STAGE_ID),
                "hydro {id} entry_stage_id"
            );
            assert_eq!(
                h.downstream_id.map(|d| d.0),
                Some(downstream),
                "hydro {id} cascade edge"
            );
        }

        // The shipped parity case trains only; enable one sim scenario so the extraction path runs.
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

        let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(
                &mut solver,
                &comm,
                1,
                ActiveSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error (an infeasible filling stage is a real data finding, not \
         a bound to relax): {:?}",
            outcome.error
        );

        let lower_bounds: Vec<f64> = event_rx
            .try_iter()
            .filter_map(|e| {
                if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                    Some(lower_bound)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !lower_bounds.is_empty(),
            "must capture at least one ConvergenceUpdate lower bound"
        );
        for window in lower_bounds.windows(2) {
            let (prev, next) = (window[0], window[1]);
            let tol = 1e-6 * prev.abs().max(1.0);
            assert!(
                next >= prev - tol,
                "lower bound must be monotone within FP tolerance: {prev} -> {next} \
             (allowed slack {tol})"
            );
        }

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
            .expect("simulate must not return Err (the soft slacks keep every stage feasible)");

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
            6,
            "one stage record per study stage (horizon ids 0..=5)",
        );

        let zeta = STAGE_HOURS * M3S_TO_HM3;
        // Shared V_target trajectory (anchored backward from the dead volume). Both
        // filling hydros share min_storage and rate, so the targets coincide; the
        // shortfalls differ only because realized storage differs (distinct inflows).
        let v_target_3 = MIN_STORAGE_HM3;
        let v_target_2 = MIN_STORAGE_HM3 - zeta * FILLING_MIN_RATE_M3S;

        // ── Per-floor INDEPENDENCE ────────────────────────────────────────────────
        // Both H_up and H_down have a strictly-positive σ_fill at the binding Filling
        // stages 2 and 3, and each σ_fill equals its OWN independently-computed
        // shortfall against its OWN realized storage and the shared V_target. The two
        // hydros' shortfalls DIFFER (distinct own incrementals), proving the floors
        // read each hydro's own trajectory rather than a single shared floor. A
        // terminal-only check would miss a per-stage floor regression that binds only
        // at the last Filling stage; a single-hydro check would miss a shared-floor
        // bug that fuses the two reservoirs' shortfalls.
        for (filling_stage, v_target) in [(2_usize, v_target_2), (3, v_target_3)] {
            let h_up_s = stage_hydro(scenario, H_UP_ID, filling_stage);
            let h_down_s = stage_hydro(scenario, H_DOWN_ID, filling_stage);
            let sigma_up = h_up_s.filling_target_violation_hm3;
            let sigma_down = h_down_s.filling_target_violation_hm3;

            assert!(
                sigma_up > 1e-6,
                "H_up σ_fill must be strictly positive at Filling stage {filling_stage} \
             (storage {} below V_target {v_target}); got {sigma_up}",
                h_up_s.storage_final_hm3
            );
            assert!(
                sigma_down > 1e-6,
                "H_down σ_fill must be strictly positive at Filling stage {filling_stage} \
             (storage {} below V_target {v_target}); got {sigma_down}",
                h_down_s.storage_final_hm3
            );

            // Each σ_fill equals its OWN shortfall V_target − v_out against its own
            // realized storage: σ_fill = max(0, V_target − v_out). This is the
            // per-floor contract — the floor row reads this hydro's storage, not the
            // sibling's.
            let shortfall_up = (v_target - h_up_s.storage_final_hm3).max(0.0);
            let shortfall_down = (v_target - h_down_s.storage_final_hm3).max(0.0);
            assert!(
                (sigma_up - shortfall_up).abs() < 1e-3,
                "H_up σ_fill must equal its OWN shortfall at stage {filling_stage}: \
             σ_fill {sigma_up} vs V_target − v_out = {shortfall_up} (storage {})",
                h_up_s.storage_final_hm3
            );
            assert!(
                (sigma_down - shortfall_down).abs() < 1e-3,
                "H_down σ_fill must equal its OWN shortfall at stage {filling_stage}: \
             σ_fill {sigma_down} vs V_target − v_out = {shortfall_down} (storage {})",
                h_down_s.storage_final_hm3
            );

            // The two shortfalls DIFFER: a shared floor would force identical σ_fill.
            assert!(
                (sigma_up - sigma_down).abs() > 1e-3,
                "H_up and H_down σ_fill must DIFFER at stage {filling_stage} (distinct own \
             incrementals {H_UP_FILLING_INCR_M3S} vs {H_DOWN_FILLING_INCR_M3S} m³/s leave \
             distinct storages and so distinct shortfalls); a shared floor would force them \
             equal: σ_up {sigma_up} vs σ_down {sigma_down}"
            );
        }

        for id in [H_UP_ID, H_DOWN_ID] {
            let sigma_floor_s5 = stage_hydro(scenario, id, 5).storage_violation_below_hm3;
            assert!(
                sigma_floor_s5 < 1e-6,
                "hydro {id} σ^{{v-}} must be ~0 at the last Operating stage (id 5) after the \
             inflow recovers and storage climbs above the dead volume {MIN_STORAGE_HM3}; got \
             {sigma_floor_s5}"
            );
        }

        // ── Release-only cascade coupling ─────────────────────────────────────────
        // At each Filling stage the closed water balance on H_down is
        //     Δstorage = ζ·incr_down − release_down + ROUTED,
        // where ROUTED = release_up routed from the upstream filling reservoir (the
        // only coupling). Rearranged, the GAP between H_down's actual Δstorage and its
        // incremental-only balance (ζ·incr_down − release_down) IS exactly H_up's
        // routed release. Asserting gap == release_up (within tolerance) proves the
        // reservoirs couple ONLY through the cascade release: a phantom inter-floor
        // routing term would make the gap exceed release_up. The criterion floor is
        // `gap ≥ release_up − tol`; the tight equality is the stronger statement.
        for filling_stage in [1_usize, 2, 3] {
            let h_down_s = stage_hydro(scenario, H_DOWN_ID, filling_stage);
            let delta_storage = h_down_s.storage_final_hm3 - h_down_s.storage_initial_hm3;
            let release_down_hm3: f64 = scenario.stages[filling_stage]
                .hydros
                .iter()
                .filter(|r| r.hydro_id == H_DOWN_ID)
                .map(|r| {
                    (r.turbined_m3s + r.spillage_m3s)
                        * block_hours(filling_stage, r.block_id)
                        * M3S_TO_HM3
                })
                .sum();
            let incremental_balance = zeta * h_down_s.incremental_inflow_m3s - release_down_hm3;
            let routed_gap = delta_storage - incremental_balance;
            let release_up = routed_release_hm3(scenario, H_UP_ID, filling_stage);
            assert!(
                routed_gap >= release_up - 1e-3,
                "Filling stage {filling_stage}: H_down's routed-water gap ({routed_gap:.6} hm³ = \
             Δstorage {delta_storage:.6} − incremental-only balance {incremental_balance:.6}) \
             must be at least H_up's routed release ({release_up:.6} hm³)"
            );
            assert!(
                (routed_gap - release_up).abs() < 1e-3,
                "Filling stage {filling_stage}: H_down's routed-water gap ({routed_gap:.6} hm³) must \
             EQUAL H_up's routed release ({release_up:.6} hm³) — a larger gap means a phantom \
             inter-floor coupling term landed on H_down's balance row"
            );
        }

        // ── Continuous handoff for BOTH filling hydros ────────────────────────────
        for id in [H_UP_ID, H_DOWN_ID] {
            let s3_final = stage_hydro(scenario, id, 3).storage_final_hm3;
            let s4_initial = stage_hydro(scenario, id, 4).storage_initial_hm3;
            assert!(
                (s3_final - s4_initial).abs() < 1e-6,
                "hydro {id}: end-of-Filling storage at id 3 ({s3_final}) must equal incoming storage \
             at id 4 ({s4_initial}) via the pin chain"
            );
        }

        // ── Off-cascade control dispatches as a normal Operating plant ────────────
        let mut h_ctrl_moved = false;
        for stage in 0..6 {
            let h_ctrl_s = stage_hydro(scenario, H_CTRL_ID, stage);
            for v in [h_ctrl_s.storage_initial_hm3, h_ctrl_s.storage_final_hm3] {
                assert!(
                    (H_CTRL_MIN_STORAGE_HM3 - 1e-6..=H_CTRL_MAX_STORAGE_HM3 + 1e-6).contains(&v),
                    "control H_ctrl storage must stay within [{H_CTRL_MIN_STORAGE_HM3}, \
                 {H_CTRL_MAX_STORAGE_HM3}] at stage {stage}; got {v}"
                );
            }
            if (h_ctrl_s.storage_final_hm3 - H_CTRL_SEED_HM3).abs() > 1e-6 {
                h_ctrl_moved = true;
            }
        }
        assert!(
            h_ctrl_moved,
            "control H_ctrl must dispatch (its storage moves off the {H_CTRL_SEED_HM3} hm³ seed), not \
         stay frozen like a PreFilling hydro"
        );

        // Before entry both filling hydros neither generate nor turbine: the
        // turbine/generation columns are pinned [0, 0] until Operating.
        for id in [H_UP_ID, H_DOWN_ID] {
            for stage in 0..(ENTRY_STAGE_ID as usize) {
                let s = stage_hydro(scenario, id, stage);
                assert!(
                    s.generation_mw.abs() < 1e-6,
                    "hydro {id} must not generate before entry (stage {stage}); got {}",
                    s.generation_mw
                );
                assert!(
                    s.turbined_total_m3s.abs() < 1e-6,
                    "hydro {id} must not turbine before entry (stage {stage}); got {}",
                    s.turbined_total_m3s
                );
            }
        }
    }

    /// Block duration in hours for `(stage_index, block_id)`, mirroring `stages.json`.
    /// `block_id` is always `Some` for a turbine-branch hydro row.
    fn block_hours(stage_index: usize, block_id: Option<u32>) -> f64 {
        let blk = block_id.expect("turbine-branch hydro rows carry a block id") as usize;
        match stage_index {
            0 | 1 | 5 => 720.0,
            2 | 4 => [240.0, 240.0, 240.0][blk],
            3 => [360.0, 360.0][blk],
            _ => unreachable!("d40 has stages 0..=5"),
        }
    }
}

mod prefilling_spillage_frozen {
    //! A `PreFilling` hydro upstream of an active reservoir must have spillage exactly
    //! `0` and contribute zero phantom water to the downstream balance row.
    //!
    //! ## The bug the one-sided floor cannot catch
    //!
    //! `fill_spillage_columns` once left every hydro's spillage free `[0, +∞)`. A
    //! `PreFilling` hydro's spillage column is decoupled from its frozen storage (its
    //! real inflow already left via the short-circuit), so a free column lets the
    //! optimizer inject water the hydro does not have onto the first active downstream
    //! hydro's water-balance row — a conservation violation. The downstream hydro
    //! turbines that phantom water cheaply and displaces thermal generation.
    //!
    //! The `d38` simulation regression checks H3's routed inflow with a ONE-SIDED floor
    //! (`routed_gap >= ζ·incr_H2`): phantom spillage from H2 only inflates the gap, so
    //! the floor still passes and the bug hides. This test pins the structural fix
    //! directly — H2's spillage is `0` while `PreFilling` — and pins the downstream gap
    //! with a TWO-SIDED equality that no phantom water can satisfy.
    //!
    //! ## Fixture (`d38-dead-volume-filling`)
    //!
    //! Cascade `H1 (id 0) → H2 (id 1, filling) → H3 (id 2, real fed downstream)`. H2's
    //! `filling.start_stage_id = 2` makes it `PreFilling` at stage ids 0–1 (the dam is
    //! not built yet), with 15 m³/s of its own incremental inflow at each. Stages 0 and
    //! 1 are single FLAT blocks of 720 h. H2 is upstream of the active reservoir H3, so
    //! these two stages are exactly the "PreFilling hydro upstream of an active
    //! reservoir" geometry the "spillage frozen [0,0] during PreFilling" contract targets.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;
    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    const H1_ID: i32 = 0;
    const H2_ID: i32 = 1;
    const H3_ID: i32 = 2;

    /// Last PreFilling stage index for H2 (exclusive upper bound of the loop):
    /// `filling.start_stage_id = 2` ⇒ PreFilling at ids 0 and 1.
    const H2_START_STAGE_ID: usize = 2;

    /// H2's own incremental inflow (m³/s) at PreFilling stages 0–1. Mirrors
    /// `scenarios/inflow_seasonal_stats.parquet`, hydro 1 ids 0–1.
    const H2_PREFILLING_INCR_M3S: f64 = 15.0;

    /// H3's own incremental inflow (m³/s), constant across stages. Mirrors
    /// `scenarios/inflow_seasonal_stats.parquet`, hydro 2.
    const H3_INCR_M3S: f64 = 20.0;

    /// m³/s → hm³ per hour. Mirrors `crate::lp::builder::M3S_TO_HM3` (private). Stages
    /// 0 and 1 are single FLAT blocks totalling 720 h, so `ζ = 720 · M3S_TO_HM3`.
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const STAGE_HOURS: f64 = 720.0;

    /// Sum a hydro's release (turbine + spillage, m³/s) across the blocks of one stage.
    fn release_m3s(
        scenario: &cobre_sddp::SimulationScenarioResult,
        hydro_id: i32,
        stage_index: usize,
    ) -> (f64, f64) {
        let turbined = scenario.stages[stage_index]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == hydro_id)
            .map(|r| r.turbined_m3s)
            .sum();
        let spillage = scenario.stages[stage_index]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == hydro_id)
            .map(|r| r.spillage_m3s)
            .sum();
        (turbined, spillage)
    }

    /// First-block scalar (storage is block-invariant) for a hydro at one stage.
    fn storage_initial_final(
        scenario: &cobre_sddp::SimulationScenarioResult,
        hydro_id: i32,
        stage_index: usize,
    ) -> (f64, f64, f64) {
        let first = scenario.stages[stage_index]
            .hydros
            .iter()
            .find(|r| r.hydro_id == hydro_id)
            .unwrap_or_else(|| panic!("hydro {hydro_id} must have a row at stage {stage_index}"));
        (
            first.storage_initial_hm3,
            first.storage_final_hm3,
            first.incremental_inflow_m3s,
        )
    }

    /// Train + simulate the d38 fixture and assert the PreFilling spillage freeze: H2
    /// (PreFilling at ids 0–1, upstream of the active H3) spills exactly `0`, and H3's
    /// routed-water gap equals ζ·incr_H2 PLUS H1's actual re-routed release — a
    /// two-sided equality no phantom spillage can satisfy.
    #[test]
    fn prefilling_hydro_upstream_of_active_reservoir_has_zero_spillage_and_no_phantom_water() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d38-dead-volume-filling");

        let mut config =
            cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
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

        let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(
                &mut solver,
                &comm,
                1,
                ActiveSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must not return Err");
        assert!(outcome.error.is_none(), "training must be feasible");

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
        assert_eq!(scenario_results.len(), 1, "one deterministic scenario");
        let scenario = &scenario_results[0];

        let zeta = STAGE_HOURS * M3S_TO_HM3;

        for stage in 0..H2_START_STAGE_ID {
            // The PreFilling hydro spills exactly 0 — the assertion the d38 one-sided
            // routed-water floor cannot make (it never inspects H2's own spillage, only
            // the aggregate gap on H3).
            let (h2_turb, h2_spill) = release_m3s(scenario, H2_ID, stage);
            assert!(
                h2_spill.abs() < 1e-6,
                "PreFilling H2 must not spill at stage {stage} (no dam exists to spill from); got \
             {h2_spill} m³/s — a positive value is phantom water injected downstream"
            );
            assert!(
                h2_turb.abs() < 1e-6,
                "PreFilling H2 must not turbine at stage {stage}; got {h2_turb} m³/s"
            );

            // No phantom water downstream: H3's closed water balance over the stage
            // is Δstorage = ζ·incr_H3 − release_H3 + ROUTED, where the only routed terms
            // onto H3's row while H2 is PreFilling are H2's own incremental inflow
            // (ζ·incr_H2) and H1's re-routed release (H1→H3, since H1→H2→H3 collapses
            // while H2 is absent). Stages 0–1 are single FLAT blocks, so ζ weights every
            // release uniformly. The gap must EQUAL that sum — a TWO-SIDED equality.
            // Phantom spillage from H2 would add ζ·spill_H2 to the gap and break it; the
            // d38 floor (`gap >= ζ·incr_H2`) would still pass.
            let (h3_init, h3_final, h3_incr) = storage_initial_final(scenario, H3_ID, stage);
            let (h3_turb, h3_spill) = release_m3s(scenario, H3_ID, stage);
            let (h1_turb, h1_spill) = release_m3s(scenario, H1_ID, stage);

            let release_h3 = (h3_turb + h3_spill) * zeta;
            let routed_h1 = (h1_turb + h1_spill) * zeta;
            let delta_storage = h3_final - h3_init;
            let incremental_balance = zeta * h3_incr - release_h3;
            let routed_gap = delta_storage - incremental_balance;
            let expected_routed = zeta * H2_PREFILLING_INCR_M3S + routed_h1;

            assert!(
                (routed_gap - expected_routed).abs() < 1e-3,
                "stage {stage}: H3's routed-water gap ({routed_gap:.6} hm³ = Δstorage \
             {delta_storage:.6} − incremental-only balance {incremental_balance:.6}) must EQUAL \
             ζ·incr_H2 + H1's re-routed release ({expected_routed:.6} hm³); a LARGER gap means H2 \
             injected phantom spillage onto H3's row (the bug the one-sided floor hides)"
            );

            assert!(
                (h3_incr - H3_INCR_M3S).abs() < 1e-6,
                "stage {stage}: H3 incremental inflow must be {H3_INCR_M3S} m³/s (own inflow must \
             not be stripped by PreFilling routing); got {h3_incr}"
            );
        }
    }
}

mod filling_cut_validity {
    //! Cut- and warm-start-validity regression across the two hydro filling phase
    //! boundaries (PreFilling -> Filling at `id == start_stage_id`, Filling ->
    //! Operating at `id == entry_stage_id`).
    //!
    //! The fixture drives a CHAINED cascade `Hf1 -> Hf2 -> Hop` (two consecutive
    //! PreFilling hydros draining into an Operating plant), not a single filling
    //! hydro: a PreFilling water-balance row collapses to the frozen identity
    //! `v_h - v_h_in = 0` and short-circuits onto the first non-PreFilling downstream
    //! resolved by `resolve_shortcircuit_target`, which walks *through* any PreFilling
    //! downstream whose row is itself a frozen identity. A single-hop fixture never
    //! exercises that transitive walk, where the silent corruption hides: `Hf1`'s
    //! water can land on `Hf2`'s frozen-identity RHS, producing a wrong-but-compiling
    //! cut.

    use cobre_io::config::TrainingSelection;
    use std::sync::mpsc;

    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{FillingConfig, HydroGenerationModel},
    };
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
        ThermalBlockBounds, ThermalStageBounds, TrainingEvent,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::SolverStatsDelta;
    use cobre_sddp::setup::NodePos;
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // Fixture topology constants (study stage ids; id == index for this horizon)
    // ---------------------------------------------------------------------------

    const N_STAGES: usize = 7;

    /// PreFilling -> Filling boundary: both filling hydros are PreFilling at every
    /// id `< START_STAGE_ID` and Filling from here.
    const START_STAGE_ID: i32 = 2;

    /// Filling -> Operating boundary, interior to the horizon
    /// (`START_STAGE_ID < ENTRY_STAGE_ID < N_STAGES`).
    const ENTRY_STAGE_ID: i32 = 4;

    /// Iteration 1 captures the first basis; iterations 2.. warm-start through
    /// `reconstruct_basis` across both boundaries.
    const N_ITERATIONS: u64 = 8;

    const HF1_ID: i32 = 1;
    const HF2_ID: i32 = 2;
    const HOP_ID: i32 = 3; // Operating downstream — the transitive short-circuit target
    const HCTL_ID: i32 = 4; // off-cascade control hydro (Operating every stage)

    // ---------------------------------------------------------------------------
    // System builder — chained filling cascade + off-cascade control
    // ---------------------------------------------------------------------------

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

    /// Build the chained filling cascade system: `Hf1 -> Hf2 -> Hop` (both filling,
    /// both PreFilling at ids 0,1; `Hop` the transitive short-circuit target),
    /// an off-cascade control `Hctl`, plus a bus deficit segment + backup thermal so
    /// the LP stays feasible regardless of the filling hydros' frozen storage.
    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(5),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 6).unwrap(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 500.0,
                min_generation_mw: 0.0,
                max_generation_mw: 400.0,
                anticipated_config: None,
                ..Default::default()
            },
        );

        let filling = || {
            Some(FillingConfig {
                start_stage_id: START_STAGE_ID,
                filling_min_rate_m3s: 50.0,
            })
        };

        // A filling hydro impounds toward a non-zero dead volume (min_storage_hm3 = 50)
        // and requires entry_stage_id = Some (the system builder rejects `filling`
        // without it); operating/control hydros use a 0 floor and no entry stage.
        let cascade_hydro =
            |id: i32, name: &str, downstream: Option<i32>, filling: Option<FillingConfig>| {
                let min_storage_hm3 = if filling.is_some() { 50.0 } else { 0.0 };
                make_hydro(
                    EntityId(id),
                    HydroSpec {
                        name: name.to_string(),
                        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                            .unwrap()
                            .checked_add_signed(chrono::Duration::days(i64::from(id)))
                            .unwrap(),
                        bus_id: EntityId(1),
                        downstream_id: downstream.map(EntityId),
                        entry_stage_id: filling.as_ref().map(|_| ENTRY_STAGE_ID),
                        exit_stage_id: None,
                        min_storage_hm3,
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
                        filling,
                        penalties: hydro_penalties(),
                        ..Default::default()
                    },
                )
            };

        let hydros = vec![
            cascade_hydro(HF1_ID, "Hf1", Some(HF2_ID), filling()),
            cascade_hydro(HF2_ID, "Hf2", Some(HOP_ID), filling()),
            cascade_hydro(HOP_ID, "Hop", None, None),
            cascade_hydro(HCTL_ID, "Hctl", None, None),
        ];

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2020, ((i % 12 + 1) % 12 + 1) as u32, 1)
                            .unwrap(),
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

        let inflow_models: Vec<InflowModel> = [HF1_ID, HF2_ID, HOP_ID, HCTL_ID]
            .into_iter()
            .flat_map(|hid| {
                (0..N_STAGES).map(move |i| InflowModel {
                    hydro_id: EntityId(hid),
                    stage_id: i as i32,
                    mean_m3s: 80.0,
                    std_m3s: 20.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                })
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
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

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 4,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds {
                    cost_per_mwh: 500.0,
                },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 400.0,
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

        // The per-stage Filling impound cap is read from the resolved bounds table
        // (`hydro_bounds(h_idx, stage_idx).filling_min_rate_m3s`), NOT the
        // `FillingConfig` scalar; set it on the two filling hydros (positional
        // indices 0 = Hf1, 1 = Hf2) at every stage so the Filling-phase row impounds.
        for h_idx in [0_usize, 1] {
            for stage_idx in 0..N_STAGES {
                bounds
                    .hydro_bounds_mut(h_idx, stage_idx)
                    .filling_min_rate_m3s = 50.0;
            }
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 4,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Filling hydros carry their stage-0 seed in `filling_storage`, operating /
        // control hydros in `storage`; the filling seed is `0.0` (empty pit, frozen
        // through the PreFilling phase that exists because `start_stage_id > 0`).
        let initial_conditions = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(HOP_ID),
                    value_hm3: 100.0,
                },
                HydroStorage {
                    hydro_id: EntityId(HCTL_ID),
                    value_hm3: 100.0,
                },
            ],
            filling_storage: vec![
                HydroStorage {
                    hydro_id: EntityId(HF1_ID),
                    value_hm3: 0.0,
                },
                HydroStorage {
                    hydro_id: EntityId(HF2_ID),
                    value_hm3: 0.0,
                },
            ],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_backup])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid chained filling cascade")
    }

    // ---------------------------------------------------------------------------
    // Config + setup builders
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
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: N_ITERATIONS as u32,
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
    // Integration test
    // ---------------------------------------------------------------------------

    /// Train the chained filling cascade and assert the cut-/warm-start-validity
    /// contract across both filling phase boundaries: monotone lower bound, zero
    /// basis-rejection spike at the boundary stages, and an off-cascade control
    /// hydro that still dispatches as a normal Operating plant.
    #[test]
    fn filling_cascade_lower_bound_monotone_no_basis_spike() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        // Populate the visited-states archive so the control hydro's forward-pass
        // storage trajectory is observable for the parity-neutrality assertion below.
        setup.set_export_states(true);

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(
                &mut solver,
                &comm,
                // 3rd arg is the forward-thread count, NOT the iteration limit (that
                // comes from the IterationLimit stopping rule).
                1,
                ActiveSolver::new,
                Some(tx),
                None,
            )
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error (an infeasible filling stage is a genuine cut-validity \
         finding, not a tolerance to relax): {:?}",
            outcome.error
        );
        let result = &outcome.result;
        assert_eq!(
            result.iterations, N_ITERATIONS,
            "regression needs the full {N_ITERATIONS}-iteration run to exercise \
         reconstruct_basis across both filling boundaries on iterations 2..",
        );

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let lower_bounds: Vec<f64> = events
            .iter()
            .filter_map(|e| {
                if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                    Some(*lower_bound)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !lower_bounds.is_empty(),
            "must capture at least one ConvergenceUpdate lower bound"
        );
        for window in lower_bounds.windows(2) {
            let (prev, next) = (window[0], window[1]);
            // The minorant property is exact in real arithmetic, but HiGHS/CLP FP
            // divergence makes a strict `>=` flaky across backends; the
            // absolute-relative tolerance absorbs that noise while still catching a
            // genuine non-monotone regression (an understated cut drops the bound by
            // far more than 1e-6 relative). Do not relax it to a large absolute slack.
            let tol = 1e-6 * prev.abs().max(1.0);
            assert!(
                next >= prev - tol,
                "lower bound must be monotone within FP tolerance: {prev} -> {next} \
             (allowed slack {tol})"
            );
        }

        // Aggregate `basis_consistency_failures` at exactly the two boundary stages
        // (id == index here) — NOT a global aggregate, which would dilute a boundary
        // spike behind otherwise-clean stages.
        let boundary_stages = [START_STAGE_ID, ENTRY_STAGE_ID];
        let boundary_rejections = SolverStatsDelta::aggregate(
            result
                .solver_stats_log
                .iter()
                .filter(|entry| {
                    matches!(entry.phase, "forward" | "backward")
                        && entry.stage_id.is_some_and(|s| boundary_stages.contains(&s))
                })
                .map(|entry| &entry.delta),
        )
        .basis_consistency_failures;
        assert_eq!(
            boundary_rejections, 0,
            "filling boundaries: expected 0 basis rejections at the PreFilling->Filling \
         (id {START_STAGE_ID}) and Filling->Operating (id {ENTRY_STAGE_ID}) stages, \
         got {boundary_rejections} (a rejection means a cut row/column was relocated \
         across the phase change, breaking reconstruct_basis slot-identity matching)"
        );

        // Read the control hydro's forward-pass storage from the visited archive at
        // state index `state.storage.start + ctl_pos`; `ctl_pos` is its positional
        // index in canonical id order (id 4 -> last of four hydros -> 3).
        let state = setup.stage_state();
        let ctl_pos = 3_usize;
        let ctl_storage_col = state.storage.start + ctl_pos;
        let archive = result
            .visited_archive
            .as_ref()
            .expect("export_states was enabled, so the visited archive is populated");

        let init_storage = 100.0_f64;
        let mut control_moved = false;
        for stage in 0..N_STAGES {
            let stage_states = archive.states_for_node(NodePos(stage));
            if stage_states.is_empty() {
                continue;
            }
            let dim = archive.node(NodePos(stage)).state_dimension();
            assert!(
                ctl_storage_col < dim,
                "control storage column {ctl_storage_col} must lie within the state \
             dimension {dim}"
            );
            for s in stage_states.chunks_exact(dim) {
                let v = s[ctl_storage_col];
                assert!(
                    (-1e-6..=200.0 + 1e-6).contains(&v),
                    "control hydro storage must stay within bounds [0, 200] hm3 at \
                 stage {stage}, got {v} (the filling chain must not perturb a \
                 normal Operating hydro's feasible region)"
                );
                if (v - init_storage).abs() > 1e-6 {
                    control_moved = true;
                }
            }
        }
        assert!(
            control_moved,
            "control hydro must dispatch as a normal Operating plant (its storage \
         moves off the initial {init_storage} hm3 across the horizon), not stay \
         frozen like a PreFilling hydro"
        );

        // Continuous handoff (no reset) across the Filling -> Operating boundary.
        // The end-of-filling outgoing storage flows into the first Operating stage's
        // incoming-state pin via the SAME pin chain every other stage uses (no
        // entry-boundary reset / RHS-fold / `initial_operating_volume`); the no-reset
        // contract is owned by `builder_setup_never_references_entry_boundary_reset`.
        // Column identity is structural — one study-level `StateSpace` maps each
        // storage coordinate to a stage-invariant dense column, so a same-vs-same
        // assertion would be tautological; this block adds the value-continuity check.
        let entry = ENTRY_STAGE_ID as usize;
        let last_filling = entry - 1;
        let hf1_pos = 0_usize;

        let out_col_last_filling = state.storage.start + hf1_pos;
        let out_col_entry = state.storage.start + hf1_pos;

        let last_filling_states = archive.states_for_node(NodePos(last_filling));
        let entry_states = archive.states_for_node(NodePos(entry));
        assert!(
            !last_filling_states.is_empty() && !entry_states.is_empty(),
            "the visited archive must hold outgoing states at the last Filling stage \
         ({last_filling}) and the entry stage ({entry}) for the handoff check"
        );
        let dim_lf = archive.node(NodePos(last_filling)).state_dimension();
        let dim_e = archive.node(NodePos(entry)).state_dimension();
        assert_eq!(
            dim_lf, dim_e,
            "state dimension must be identical across the boundary (the storage \
         coordinate is not relocated): {dim_lf} vs {dim_e}"
        );
        let filling_seed = 0.0_f64;
        for s in last_filling_states.chunks_exact(dim_lf) {
            let v_out_last_filling = s[out_col_last_filling];
            assert!(
                (filling_seed - 1e-6..=200.0 + 1e-6).contains(&v_out_last_filling),
                "end-of-filling outgoing storage at stage {last_filling} must be within \
             [0, 200] hm3, got {v_out_last_filling}"
            );
            assert!(
                v_out_last_filling > filling_seed + 1e-6,
                "the Filling phase impounds water, so the end-of-filling outgoing storage \
             at stage {last_filling} must be strictly above the 0.0 seed (a frozen / \
             reset-to-seed value would fail this), got {v_out_last_filling}"
            );
        }
        for s in entry_states.chunks_exact(dim_e) {
            let v_out_entry = s[out_col_entry];
            assert!(
                (-1e-6..=200.0 + 1e-6).contains(&v_out_entry),
                "entry-stage (id {entry}) outgoing storage must stay within [0, 200] hm3 \
             under the continuous handoff, got {v_out_entry}"
            );
        }
    }

    /// No-reset source guard: the LP builder and study-setup source must contain NO
    /// entry-boundary reset, RHS-fold, or `initial_operating_volume` symbol.
    ///
    /// The Filling -> Operating handoff is CONTINUOUS: the end-of-filling outgoing
    /// storage flows into the first Operating stage through the existing incoming-state
    /// pin chain (`build_initial_state` seeds only stage 0; `filling_phase` flips column
    /// BOUNDS at `entry`, never the column index). Re-introducing an entry-stage reset,
    /// an RHS-fold, or a pin to `initial_operating_volume` would land one of the
    /// forbidden symbols in the builder/setup source and fail this guard.
    ///
    /// The needles are assembled from char fragments (the
    /// `lower_bound_never_references_filling_gating` idiom) so the literals are absent
    /// from this file's own bytes — else the scan would flag itself.
    #[test]
    fn builder_setup_never_references_entry_boundary_reset() {
        let needles: [String; 5] = [
            ["initial", "_operating_volume"].concat(),
            ["entry", "_reset"].concat(),
            ["rhs", "_fold"].concat(),
            ["reset", "_storage"].concat(),
            ["operating", "_volume_reset"].concat(),
        ];

        // The builder/setup sources that own the entry-boundary handoff; a reset draft
        // would land in one of these.
        let sources: [(&str, &str); 5] = [
            ("setup/mod.rs", include_str!("../src/setup/mod.rs")),
            (
                "lp/builder/mod.rs",
                include_str!("../src/lp/builder/mod.rs"),
            ),
            (
                "lp/builder/columns.rs",
                include_str!("../src/lp/builder/columns.rs"),
            ),
            (
                "lp/builder/rows.rs",
                include_str!("../src/lp/builder/rows.rs"),
            ),
            (
                "lp/builder/patch.rs",
                include_str!("../src/lp/builder/patch.rs"),
            ),
        ];

        let mut offenders: Vec<String> = Vec::new();
        for (path, src) in &sources {
            for needle in &needles {
                if src.contains(needle.as_str()) {
                    offenders.push(format!("{path}: {needle}"));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "the Filling->Operating handoff is CONTINUOUS via the incoming-state pin \
         chain; the builder/setup source must contain NO entry-boundary reset / \
         RHS-fold / initial_operating_volume symbol (re-introducing the abandoned \
         reset draft re-introduces one of these); offenders: {offenders:?}"
        );
    }
}

mod d35_pumping_commissioning_simulation {
    //! Pumping-station commissioning gating must reach the simulation output.
    //!
    //! The parity hash does NOT hash pumping-output columns, so a gating bug that let
    //! a commissioning-dormant station pump, or stamped the wrong station id, would
    //! still hash-match. This test exercises that path through the full train+simulate
    //! pipeline.
    //!
    //! The commissioning gate is `entry <= stage.id && stage.id < exit`. Under the
    //! dense layout a dormant station keeps its column but is pinned to `[0, 0]`,
    //! emitting a ZERO row rather than being absent.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    #[test]
    fn pumping_commissioning_window_gates_simulation_output() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d35-pumping-commissioning");

        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        // The shipped parity case trains only; enable one sim scenario so the pumping extraction path runs.
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
            .expect("simulate must not return Err (omitting the forced transfer only relaxes 0/2)");

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

        let stage0 = &scenario.stages[0].pumping_stations;
        assert_eq!(
            stage0.len(),
            1,
            "stage 0 keeps the dense pumping column: one (zeroed) row, got {stage0:?}",
        );
        assert_eq!(
            stage0[0].pumping_station_id, 0,
            "dormant stage 0 row must still map to system station id 0",
        );
        assert_eq!(
            stage0[0].pumped_flow_m3s, 0.0,
            "stage 0 is before the entry window: dormant column pinned to 0, got {}",
            stage0[0].pumped_flow_m3s,
        );

        let stage2 = &scenario.stages[2].pumping_stations;
        assert_eq!(
            stage2.len(),
            1,
            "stage 2 keeps the dense pumping column: one (zeroed) row, got {stage2:?}",
        );
        assert_eq!(
            stage2[0].pumping_station_id, 0,
            "decommissioned stage 2 row must still map to system station id 0",
        );
        assert_eq!(
            stage2[0].pumped_flow_m3s, 0.0,
            "stage 2 is at the exit boundary (decommissioned): dormant column pinned to 0, got {}",
            stage2[0].pumped_flow_m3s,
        );

        let stage1 = &scenario.stages[1].pumping_stations;
        assert_eq!(
            stage1.len(),
            1,
            "stage 1 is active with a single station and a single block: one pumping row",
        );
        let row = &stage1[0];
        assert_eq!(
            row.pumping_station_id, 0,
            "active-local index 0 must map back to system station id 0",
        );
        assert_eq!(row.stage_id, 1, "row must be stamped with stage 1");
        assert!(
            row.pumped_flow_m3s >= 2.0 - 1e-6,
            "the forced minimum flow (2.0 m³/s) must bind at the active stage; got {}",
            row.pumped_flow_m3s,
        );
        // The negative-injection sign lives in the LP coupling; the output reports
        // the positive magnitude (pumped flow × the 0.5 MW per m³/s rate).
        assert!(
            (row.power_consumption_mw - row.pumped_flow_m3s * 0.5).abs() < 1e-6,
            "power consumption must equal pumped_flow * 0.5; got {} for flow {}",
            row.power_consumption_mw,
            row.pumped_flow_m3s,
        );
    }
}

mod d36_thermal_line_commissioning_simulation {
    //! Thermal and line commissioning gating must reach the simulation output.
    //!
    //! The parity hash covers hydro/water values and cuts but NOT thermal-generation
    //! or line-flow columns, so the zero-influence gating convention is invisible to
    //! it: a commissioning-dormant thermal pins BOTH bounds to `[0, 0]` (zeroing its
    //! `min_generation_mw` must-run floor too, so the LP stays feasible), and a dormant
    //! line pins both directional caps to 0. The commissioning gate is
    //! `entry <= stage.id && stage.id < exit`. This test exercises those paths
    //! directly through the full train+simulate pipeline.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    #[test]
    fn thermal_line_commissioning_window_gates_simulation_output() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d36-thermal-line-commissioning");

        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        // The shipped parity case trains only; enable one sim scenario so the thermal/line
        // extraction paths run (`StudySetup::new` reads `n_scenarios` from this).
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
            .expect(
                "simulate must not return Err (a windowed-out must-run thermal must stay feasible)",
            );

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

        // ── Thermal gating ────────────────────────────────────────────────────────

        let t_stage0 = &scenario.stages[0].thermals;
        assert_eq!(
            t_stage0.len(),
            1,
            "stage 0 keeps the dense thermal column: one (zeroed) row, got {t_stage0:?}",
        );
        assert_eq!(
            t_stage0[0].thermal_id, 0,
            "dormant row maps to system thermal 0"
        );
        assert_eq!(
            t_stage0[0].generation_mw, 0.0,
            "stage 0 is before entry: the must-run floor must be zeroed, got {}",
            t_stage0[0].generation_mw,
        );

        let t_stage2 = &scenario.stages[2].thermals;
        assert_eq!(
            t_stage2.len(),
            1,
            "stage 2 keeps the dense thermal column: one (zeroed) row, got {t_stage2:?}",
        );
        assert_eq!(
            t_stage2[0].generation_mw, 0.0,
            "stage 2 is at the exit boundary (decommissioned): must-run floor zeroed, got {}",
            t_stage2[0].generation_mw,
        );

        let t_stage1 = &scenario.stages[1].thermals;
        assert_eq!(t_stage1.len(), 1, "stage 1 is active: one thermal row");
        assert_eq!(
            t_stage1[0].thermal_id, 0,
            "active row maps to system thermal 0"
        );
        assert_eq!(t_stage1[0].stage_id, 1, "row stamped with stage 1");
        assert!(
            t_stage1[0].generation_mw >= 10.0 - 1e-6,
            "the must-run floor (10 MW) must bind at the active stage; got {}",
            t_stage1[0].generation_mw,
        );

        // ── Line gating ─────────────────────────────────────────────────────────────

        let l_stage0 = &scenario.stages[0].exchanges;
        assert_eq!(
            l_stage0.len(),
            1,
            "stage 0 keeps the dense line column: one (zeroed) row, got {l_stage0:?}",
        );
        assert_eq!(l_stage0[0].line_id, 0, "dormant row maps to system line 0");
        assert_eq!(
            l_stage0[0].direct_flow_mw, 0.0,
            "stage 0 line dormant: direct flow must be 0, got {}",
            l_stage0[0].direct_flow_mw,
        );
        assert_eq!(
            l_stage0[0].reverse_flow_mw, 0.0,
            "stage 0 line dormant: reverse flow must be 0, got {}",
            l_stage0[0].reverse_flow_mw,
        );

        // B1's 40 MW load exceeds H1's 20 MW cap, so B0 must export across the line —
        // direct flow is positive and within the restored 15 MW cap.
        let l_stage1 = &scenario.stages[1].exchanges;
        assert_eq!(l_stage1.len(), 1, "stage 1 line active: one exchange row");
        assert_eq!(l_stage1[0].line_id, 0, "active row maps to system line 0");
        assert!(
            l_stage1[0].direct_flow_mw > 1e-6,
            "stage 1 line active: B0 must export to the deficit-prone B1, got {}",
            l_stage1[0].direct_flow_mw,
        );
        assert!(
            l_stage1[0].direct_flow_mw <= 15.0 + 1e-6,
            "stage 1 direct flow must respect the restored 15 MW cap, got {}",
            l_stage1[0].direct_flow_mw,
        );

        let l_stage2 = &scenario.stages[2].exchanges;
        assert_eq!(l_stage2.len(), 1, "stage 2 line active: one exchange row");
        assert!(
            l_stage2[0].direct_flow_mw <= 15.0 + 1e-6 && l_stage2[0].reverse_flow_mw <= 15.0 + 1e-6,
            "stage 2 flows must respect the active 15 MW caps, got direct={} reverse={}",
            l_stage2[0].direct_flow_mw,
            l_stage2[0].reverse_flow_mw,
        );
    }
}

mod d42_nonfilling_hydro_commissioning {
    //! A non-filling hydro that commissions mid-horizon reuses the `PreFilling`
    //! short-circuit reformulation: while commissioning-dormant its dam is not built,
    //! so its turbine/spillage/generation columns are pinned `[0, 0]`, its storage is
    //! frozen at the inert initial-condition value, and its incremental inflow flows
    //! past the un-built site — onto the first active downstream reservoir, or to the
    //! sink at a cascade tail. The danger this case guards against is the trapped-water
    //! trap: zeroing the flow columns while leaving the inflow on the hydro's OWN
    //! balance row makes the LP infeasible whenever the site has inflow.
    //!
    //! ## Fixture (`d42-nonfilling-hydro-commissioning`)
    //!
    //! Three NON-filling hydros (ids 0, 1, 2) over a 4-stage horizon, single FLAT
    //! block, 720 h each, deterministic (std = 0):
    //!
    //! - `H_new` (id 0): `entry_stage_id = 2`, downstream = `H_down` (id 1). Dormant at
    //!   stages 0-1 (the trapped-water catcher: nonzero 40 m³/s local inflow routed onto
    //!   a downstream reservoir), Operating at 2-3.
    //! - `H_down` (id 1): no commissioning window, real reservoir outlet
    //!   (`downstream_id = null`). Operating every stage; receives `H_new`'s routed
    //!   inflow while `H_new` is dormant.
    //! - `H_tail` (id 2): `entry_stage_id = 2`, cascade tail (`downstream_id = null`).
    //!   Dormant at 0-1 (the sink-discard fixture: its inflow exits the system), then
    //!   Operating at 2-3.
    //!
    //! A 300 MW thermal backstop keeps every stage feasible while the hydros are
    //! dormant, so an infeasibility here is a real trapped-water regression, not a
    //! tuning artifact.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;
    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    /// Hydro entity ids in canonical order. Mirrors `system/hydros.json`.
    const H_NEW_ID: i32 = 0;
    const H_DOWN_ID: i32 = 1;
    const H_TAIL_ID: i32 = 2;

    /// First commissioned stage for `H_new` and `H_tail` (`entry_stage_id`). Mirrors
    /// `system/hydros.json`.
    const ENTRY_STAGE_ID: usize = 2;

    /// `H_new`'s and `H_tail`'s own incremental inflow (m³/s), constant across stages.
    /// Mirrors `scenarios/inflow_seasonal_stats.parquet` ids 0 and 2.
    const H_NEW_INCR_M3S: f64 = 40.0;
    /// `H_down`'s own incremental inflow (m³/s). Mirrors id 1.
    const H_DOWN_INCR_M3S: f64 = 20.0;
    /// Initial-condition storage seeds (hm³). Mirrors `initial_conditions.json`; the
    /// dormant hydros pin to their IC value (NOT `[0, 0]`, NOT `min_storage`).
    const H_NEW_SEED_HM3: f64 = 50.0;
    const H_TAIL_SEED_HM3: f64 = 30.0;

    /// m³/s → hm³ per hour. Mirrors `crate::lp::builder::M3S_TO_HM3` (private). `ζ =
    /// total_stage_hours · M3S_TO_HM3`; every stage totals 720 h.
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const STAGE_HOURS: f64 = 720.0;

    /// Per-(stage, hydro) view: single FLAT block per stage, so each scalar reads from
    /// the lone block row.
    struct StageHydro {
        turbined_m3s: f64,
        spillage_m3s: f64,
        generation_mw: f64,
        incremental_inflow_m3s: f64,
        storage_initial_hm3: f64,
        storage_final_hm3: f64,
    }

    fn hydro_rows(
        scenario: &cobre_sddp::SimulationScenarioResult,
        hydro_id: i32,
        stage_index: usize,
    ) -> Vec<&cobre_sddp::SimulationHydroResult> {
        scenario.stages[stage_index]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == hydro_id)
            .collect()
    }

    fn stage_hydro(
        scenario: &cobre_sddp::SimulationScenarioResult,
        hydro_id: i32,
        stage_index: usize,
    ) -> StageHydro {
        let rows = hydro_rows(scenario, hydro_id, stage_index);
        assert!(
            !rows.is_empty(),
            "hydro {hydro_id} must have at least one row at stage {stage_index}"
        );
        let first = rows[0];
        StageHydro {
            turbined_m3s: rows.iter().map(|r| r.turbined_m3s).sum(),
            spillage_m3s: rows.iter().map(|r| r.spillage_m3s).sum(),
            generation_mw: rows.iter().map(|r| r.generation_mw).sum(),
            incremental_inflow_m3s: first.incremental_inflow_m3s,
            storage_initial_hm3: first.storage_initial_hm3,
            storage_final_hm3: first.storage_final_hm3,
        }
    }

    /// Train the d42 case, simulate one deterministic scenario, and assert the
    /// dormant-non-filling reformulation: feasibility (no trapped water), the routed
    /// inflow landing on the downstream reservoir, the sink discard, and the
    /// dormant-vs-active output transition at `entry`.
    // Rationale: the four assertion blocks share one train+simulate run; splitting them
    // would re-run the pipeline per block or thread the scenario through opaque args.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn nonfilling_commissioning_routes_inflow_and_is_feasible() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d42-nonfilling-hydro-commissioning");

        // Load + parse succeed: a non-filling hydro carrying a commissioning window is
        // valid (the relaxed filling⟹entry guard), and the three windows round-trip.
        let system_for_check = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        for (id, entry, downstream) in [
            (H_NEW_ID, Some(ENTRY_STAGE_ID as i32), Some(H_DOWN_ID)),
            (H_DOWN_ID, None, None),
            (H_TAIL_ID, Some(ENTRY_STAGE_ID as i32), None),
        ] {
            let h = system_for_check
                .hydros()
                .iter()
                .find(|h| h.id.0 == id)
                .unwrap_or_else(|| panic!("hydro {id} must be present"));
            assert!(h.filling.is_none(), "hydro {id} must be non-filling");
            assert_eq!(h.entry_stage_id, entry, "hydro {id} entry_stage_id");
            assert_eq!(
                h.downstream_id.map(|d| d.0),
                downstream,
                "hydro {id} cascade edge"
            );
        }

        let mut config =
            cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
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

        let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(
                &mut solver,
                &comm,
                1,
                ActiveSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training must be feasible — an infeasibility here means the dormant hydro's \
         inflow was trapped on its own balance row instead of short-circuited: {:?}",
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
        .expect(
            "simulate must not return Err (the dormant reformulation keeps every stage feasible)",
        );

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");
        assert_eq!(scenario_results.len(), 1, "one deterministic scenario");
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            4,
            "one record per study stage (0..=3)"
        );

        let zeta = STAGE_HOURS * M3S_TO_HM3;

        // ── Dormant outputs before entry (the [0,0] flow columns + frozen IC storage) ─
        // H_new and H_tail neither turbine nor generate while dormant, and their storage
        // stays pinned at the inert IC seed (frozen identity v = v_in = seed). A nonzero
        // turbine/generation here would mean the window was not applied; a storage drift
        // would mean the storage column was not decoupled.
        for stage in 0..ENTRY_STAGE_ID {
            for (id, seed) in [(H_NEW_ID, H_NEW_SEED_HM3), (H_TAIL_ID, H_TAIL_SEED_HM3)] {
                // The simulation read path emits a dormant output row (not an absent
                // entity): exactly one row per single-FLAT-block stage, with the
                // non-line `operative_state_code == 1`, mirroring the existing
                // commissioning_active simulation gate for thermals/lines/NCS.
                let rows = hydro_rows(scenario, id, stage);
                assert_eq!(
                    rows.len(),
                    1,
                    "dormant hydro {id} must emit exactly one output row at stage {stage} (present, not absent)"
                );
                assert_eq!(
                    rows[0].operative_state_code, 1,
                    "dormant hydro {id} output row at stage {stage} carries operative_state_code 1"
                );
                let s = stage_hydro(scenario, id, stage);
                assert!(
                    s.turbined_m3s.abs() < 1e-6,
                    "dormant hydro {id} must not turbine at stage {stage}; got {}",
                    s.turbined_m3s
                );
                assert!(
                    s.spillage_m3s.abs() < 1e-6,
                    "dormant hydro {id} must not spill at stage {stage}; got {}",
                    s.spillage_m3s
                );
                assert!(
                    s.generation_mw.abs() < 1e-6,
                    "dormant hydro {id} must not generate at stage {stage}; got {}",
                    s.generation_mw
                );
                assert!(
                    (s.storage_final_hm3 - seed).abs() < 1e-6,
                    "dormant hydro {id} storage must stay frozen at the IC seed {seed} at stage \
                 {stage} (decoupled, NOT [0,0] / min_storage); got {}",
                    s.storage_final_hm3
                );
            }
        }

        // ── Trapped-water catcher: H_new's inflow lands on H_down's balance row ───────
        // While H_new (id 0) is dormant, the river flows past its site onto the active
        // downstream reservoir H_down (id 1). The closed water balance on H_down is
        //     Δstorage = ζ·incr_down − release_down + ROUTED,
        // and ROUTED is exactly ζ·incr_new (H_new's incremental inflow, since H_new's own
        // release is 0 while dormant). Rearranged, the GAP between H_down's actual
        // Δstorage and its incremental-only balance IS ζ·incr_new. A trapped-water bug
        // (inflow stranded on H_new's own row) would make the gap 0 instead.
        for stage in 0..ENTRY_STAGE_ID {
            let h_down = stage_hydro(scenario, H_DOWN_ID, stage);
            let release_down_hm3 = (h_down.turbined_m3s + h_down.spillage_m3s) * zeta;
            let delta_storage = h_down.storage_final_hm3 - h_down.storage_initial_hm3;
            let incremental_balance = zeta * h_down.incremental_inflow_m3s - release_down_hm3;
            let routed_gap = delta_storage - incremental_balance;
            let expected_routed = zeta * H_NEW_INCR_M3S;
            assert!(
                (routed_gap - expected_routed).abs() < 1e-3,
                "stage {stage}: H_down's routed-water gap ({routed_gap:.6} hm³ = Δstorage \
             {delta_storage:.6} − incremental-only balance {incremental_balance:.6}) must EQUAL \
             ζ·incr_new ({expected_routed:.6} hm³); a smaller gap means H_new's inflow was \
             trapped on its own frozen row instead of routed downstream"
            );
            assert_eq!(
                h_down.incremental_inflow_m3s, H_DOWN_INCR_M3S,
                "stage {stage}: H_down keeps its own incremental inflow"
            );
        }

        // ── Sink discard: H_tail's dormant inflow exits the system ────────────────────
        // H_tail (id 2) has no downstream while dormant, so its incremental inflow is
        // discarded at the sink and its storage stays frozen — already asserted above as
        // part of the dormant-output block. The feasibility of the whole solve (asserted
        // via outcome.error.is_none()) is the sink case's "no infeasibility" guarantee:
        // a trapped sink inflow would have made stage 0/1 infeasible.
        for stage in 0..ENTRY_STAGE_ID {
            let h_tail = stage_hydro(scenario, H_TAIL_ID, stage);
            assert_eq!(
                h_tail.incremental_inflow_m3s, H_NEW_INCR_M3S,
                "stage {stage}: H_tail carries its own (discarded) incremental inflow"
            );
        }

        // ── Active from entry: no Filling phase, normal operation ─────────────────────
        // From entry onward H_new and H_tail are Operating: they can turbine/generate
        // (the columns are no longer pinned [0,0]) and their storage is no longer frozen
        // at the seed. At least one commissioned stage must show the plant dispatching to
        // prove the window opened straight to Operating, and storage must be able to move
        // off the dormant seed (the frozen identity is released at entry).
        let mut new_dispatched = false;
        let mut tail_dispatched = false;
        let mut new_storage_moved = false;
        for stage in ENTRY_STAGE_ID..4 {
            let s_new = stage_hydro(scenario, H_NEW_ID, stage);
            let s_tail = stage_hydro(scenario, H_TAIL_ID, stage);
            if s_new.turbined_m3s > 1e-6 || s_new.generation_mw > 1e-6 {
                new_dispatched = true;
            }
            if s_tail.turbined_m3s > 1e-6 || s_tail.generation_mw > 1e-6 {
                tail_dispatched = true;
            }
            if (s_new.storage_final_hm3 - H_NEW_SEED_HM3).abs() > 1e-6 {
                new_storage_moved = true;
            }
        }
        assert!(
            new_dispatched,
            "H_new must dispatch at some commissioned stage (>= entry {ENTRY_STAGE_ID}) — the window \
         opens straight to Operating with no Filling phase"
        );
        assert!(
            tail_dispatched,
            "H_tail must dispatch at some commissioned stage (>= entry {ENTRY_STAGE_ID})"
        );
        assert!(
            new_storage_moved,
            "H_new's storage must move off the dormant IC seed {H_NEW_SEED_HM3} hm³ once Operating — \
         the frozen-identity storage decoupling is released at entry, not held like a PreFilling pin"
        );
    }
}

mod backwater_reference_volume {
    //! A downstream plant's declared `reference_volume` reaches the upstream plant's
    //! backwater family selection.
    //!
    //! In `d31-backwater-reference-volume`, upstream computed-FPHA plant U
    //! (`hydro_id = 0`) discharges into downstream plant D (`hydro_id = 1`). U's
    //! tailrace carries two backwater families keyed at distinct downstream levels, so
    //! the resolved downstream level — D's forebay surface at D's reference operating
    //! volume — moves the interpolated tailrace elevation and therefore U's fitted FPHA
    //! planes. D declares `reference_volume: { percentile: 0.95 }`; the two tests prove
    //! (a) clearing it (falling back to the 0.65 default) shifts U's planes, so the
    //! value genuinely reaches `resolve_downstream_level`, and (b) reversing the
    //! auxiliary-row declaration order leaves U's planes bit-identical
    //! (declaration-order invariance).

    use std::path::Path;

    use cobre_core::EntityId;
    use cobre_io::FphaHyperplaneRow;
    use cobre_io::extensions::{SelectionMode, load_tailrace_curves};
    use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;

    fn case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic")
            .join("d31-backwater-reference-volume")
    }

    /// `load_case_with_artifacts` does not parse `tailrace_curves.parquet` (it defaults
    /// the field to empty); `prepare_hydro_models` fills it from disk in production. This
    /// helper mirrors that step so `prepare_hydro_models_from_artifacts` below sees the
    /// same backwater families the CLI/training path does.
    fn load_d31() -> (cobre_core::System, cobre_io::CaseArtifacts) {
        let dir = case_dir();
        let loaded = cobre_io::load_case_with_artifacts(&dir).expect("d31 must load");
        let mut artifacts = loaded.artifacts;
        let tailrace_path = dir.join("system").join("tailrace_curves.parquet");
        artifacts.tailrace_curves =
            load_tailrace_curves(Some(&tailrace_path)).expect("tailrace curves must load");
        (loaded.system, artifacts)
    }

    /// FPHA export rows for upstream plant U (`hydro_id = 0`), sorted into a stable
    /// order so two runs compare element-by-element.
    fn upstream_planes(
        system: &cobre_core::System,
        artifacts: &cobre_io::CaseArtifacts,
    ) -> Vec<FphaHyperplaneRow> {
        let prepared = prepare_hydro_models_from_artifacts(system, artifacts, false, None)
            .expect("hydro models must prepare");
        let mut rows: Vec<FphaHyperplaneRow> = prepared
            .fpha_export_rows
            .into_iter()
            .filter(|r| r.hydro_id == EntityId::from(0))
            .collect();
        rows.sort_by(|a, b| {
            (a.stage_id, a.plane_id)
                .cmp(&(b.stage_id, b.plane_id))
                .then(a.gamma_0.total_cmp(&b.gamma_0))
        });
        rows
    }

    /// Clears downstream plant D's (`hydro_id = 1`) `reference_volume` so the resolver
    /// falls back to the 0.65 default fraction; U's plane fit is the only observable
    /// that may change.
    fn clear_downstream_reference_volume(artifacts: &mut cobre_io::CaseArtifacts) {
        for config in &mut artifacts.production_models {
            if config.hydro_id != EntityId::from(1) {
                continue;
            }
            match &mut config.selection_mode {
                SelectionMode::StageRanges { ranges } => {
                    for range in ranges {
                        range.reference_volume = None;
                    }
                }
                SelectionMode::Seasonal { seasons, .. } => {
                    for season in seasons {
                        season.reference_volume = None;
                    }
                }
            }
        }
    }

    /// D's declared `reference_volume` must shift U's exported FPHA planes versus the
    /// 0.65-default fallback, proving the value reaches `resolve_downstream_level`. An
    /// unchanged result would prove nothing (the backwater family bracket was not
    /// crossed), so the test asserts an actual bit-level difference.
    #[test]
    fn backwater_reference_volume_shifts_upstream_planes() {
        let (system, declared_artifacts) = load_d31();

        let with_reference_volume = upstream_planes(&system, &declared_artifacts);

        let mut default_artifacts = declared_artifacts.clone();
        clear_downstream_reference_volume(&mut default_artifacts);
        let without_reference_volume = upstream_planes(&system, &default_artifacts);

        assert!(
            !with_reference_volume.is_empty(),
            "the upstream computed-FPHA plant must export at least one plane"
        );
        assert_eq!(
            with_reference_volume.len(),
            without_reference_volume.len(),
            "the plane count must not depend on the downstream reference volume"
        );

        let differs = with_reference_volume
            .iter()
            .zip(&without_reference_volume)
            .any(|(declared, default)| {
                declared.gamma_0.to_bits() != default.gamma_0.to_bits()
                    || declared.gamma_v.to_bits() != default.gamma_v.to_bits()
                    || declared.gamma_q.to_bits() != default.gamma_q.to_bits()
                    || declared.gamma_s.to_bits() != default.gamma_s.to_bits()
            });
        assert!(
            differs,
            "U's planes must differ between the declared reference volume and the \
         0.65 default — the backwater family bracket was not crossed:\n\
         declared = {with_reference_volume:?}\n\
         default  = {without_reference_volume:?}"
        );
    }

    /// Reversing the declaration order of the auxiliary rows (geometry, tailrace,
    /// and production-model configs) leaves U's exported planes bit-identical. The
    /// resolve path re-sorts geometry by volume, groups tailrace families by plant,
    /// and keys production configs by `hydro_id`, so input ordering must not perturb
    /// the fitted planes — the declaration-order-invariance hard rule.
    #[test]
    fn backwater_upstream_planes_are_declaration_order_invariant() {
        let (system, forward_artifacts) = load_d31();
        let forward = upstream_planes(&system, &forward_artifacts);

        let mut reversed_artifacts = forward_artifacts.clone();
        reversed_artifacts.hydro_geometry.reverse();
        reversed_artifacts.tailrace_curves.reverse();
        reversed_artifacts.production_models.reverse();
        let reversed = upstream_planes(&system, &reversed_artifacts);

        assert_eq!(
            forward.len(),
            reversed.len(),
            "plane counts must match across input orderings"
        );
        for (a, b) in forward.iter().zip(&reversed) {
            assert_eq!(a.stage_id, b.stage_id);
            assert_eq!(a.plane_id, b.plane_id);
            assert_eq!(
                a.gamma_0.to_bits(),
                b.gamma_0.to_bits(),
                "gamma_0 must be bit-identical across input orderings"
            );
            assert_eq!(a.gamma_v.to_bits(), b.gamma_v.to_bits());
            assert_eq!(a.gamma_q.to_bits(), b.gamma_q.to_bits());
            assert_eq!(a.gamma_s.to_bits(), b.gamma_s.to_bits());
        }
    }
}

mod exited_plant_transit {
    //! Water in transit toward a hydro plant that retires before it arrives must
    //! reach the next operating plant downstream.
    //!
    //! ## Fixture (`crates/cobre-sddp/tests/fixtures/exited_plant_transit`)
    //!
    //! Cascade `U (id 0) -> J (id 1) -> D (id 2)`, two single-block 720 h stages, no
    //! storage anywhere, productivity 0.0036 MW/(m³/s) on every plant (1 MWh per
    //! hm³). `U` declares `travel_time_hours = 360` on its arc to `J` (`k_0 = k_1 =
    //! 1/2`), and `J` has `exit_stage_id = 1`: `J` is `PreFilling` at stage 1, the
    //! stage its incoming bucket matures. `U`'s stage-0 inflow of 25 m³/s (64.8 hm³)
    //! is the only water. One thermal at 10 $/MWh serves 200 MWh per stage.
    //!
    //! ## Hand-derived optimum
    //!
    //! Stage 0: `U` passes its 64.8 hm³; `J` receives the same-stage half (32.4 hm³)
    //! and passes it to `D`; hydro 129.6 MWh, thermal 70.4 MWh, 704 $. Stage 1: the
    //! 32.4 hm³ bucket reaches `D`; thermal 167.6 MWh, 1676 $. Lower bound 2380 $;
    //! dropping the bucket gives 704 + 2000 = 2704 $. The stage-1 thermal is
    //! marginal, so `D`'s water and the bucket are both worth -10 $/hm³.

    use std::path::PathBuf;

    use cobre_core::EntityId;
    use cobre_sddp::{SimulationHydroResult, SimulationScenarioResult};
    use cobre_solver::ActiveSolver;

    use super::common::{StubComm, fresh_system_and_setup_with, run_simulation};

    /// Undoes the builder's objective scaling on a stored cut coefficient (mirrors
    /// `extraction.rs`'s `water_value = dual * COST_SCALE_FACTOR`).
    const COST_SCALE_FACTOR: f64 = 1_000_000.0;
    /// hm³ per m³/s over one 720 h stage.
    const ZETA: f64 = 720.0 * 3_600.0 / 1_000_000.0;
    const TOL: f64 = 1e-6;

    const U_ID: i32 = 0;
    const J_ID: i32 = 1;
    const D_ID: i32 = 2;
    /// `J`'s `exit_stage_id`, the stage its bucket matures while it is `PreFilling`.
    const J_EXIT_STAGE: usize = 1;
    const EXPECTED_LOWER_BOUND: f64 = 2380.0;
    const TRANSIT_AT_EXIT_HM3: f64 = 32.4;
    const D_WATER_VALUE_PER_HM3: f64 = -10.0;

    fn case_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/exited_plant_transit")
    }

    fn hydro_row(
        scenario: &SimulationScenarioResult,
        stage: usize,
        hydro_id: i32,
    ) -> &SimulationHydroResult {
        let mut rows = scenario.stages[stage]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == hydro_id);
        let row = rows
            .next()
            .unwrap_or_else(|| panic!("hydro {hydro_id} must have a row at stage {stage}"));
        assert!(
            rows.next().is_none(),
            "hydro {hydro_id} stage {stage}: every fixture stage is a single block"
        );
        row
    }

    /// `ζ·(turbined + spillage − incremental inflow) + Δstorage`: the water a plant's
    /// balance row received from outside the plant over one stage.
    fn received_from_upstream(row: &SimulationHydroResult) -> f64 {
        ZETA * (row.turbined_m3s + row.spillage_m3s - row.incremental_inflow_m3s)
            + (row.storage_final_hm3 - row.storage_initial_hm3)
    }

    fn release(row: &SimulationHydroResult) -> f64 {
        ZETA * (row.turbined_m3s + row.spillage_m3s)
    }

    #[test]
    fn exited_plant_transit_reaches_the_next_operating_plant_at_the_hand_derived_cost() {
        let (system, mut setup) = fresh_system_and_setup_with(&case_dir(), |_| {});
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(outcome.error.is_none(), "training must be feasible");

        let final_lb = outcome.result.final_lb;
        assert!(
            (final_lb - EXPECTED_LOWER_BOUND).abs() < TOL,
            "lower bound must be the hand-derived {EXPECTED_LOWER_BOUND} $ (2704 $ means the \
             water in transit at J's exit was dropped); got {final_lb}"
        );

        let state = setup.stage_state();
        assert_eq!(
            state.n_buckets, 1,
            "power guard: one depth-1 arc gives one bucket dimension"
        );
        let bucket_idx = state.transit_buckets_out.start;
        let d_pos = system
            .hydros()
            .iter()
            .position(|h| h.id == EntityId(D_ID))
            .expect("D (hydro id 2) must be in the canonical hydro order");
        let storage_d_idx = state.storage.start + d_pos;

        let pool0 = &setup.fcf.pools[0];
        assert!(
            pool0.active_count() > 0,
            "pool 0 must hold at least one trained cut to inspect"
        );
        for (slot, _intercept, coefficients) in pool0.active_cuts() {
            let bucket_value = coefficients[bucket_idx] * COST_SCALE_FACTOR;
            assert!(
                (bucket_value - D_WATER_VALUE_PER_HM3).abs() < TOL,
                "pool-0 slot {slot}: J's maturing bucket must be worth D's water value \
                 {D_WATER_VALUE_PER_HM3} $/hm³ (0 means the bucket sits on no row); got \
                 {bucket_value}"
            );
            let storage_d_value = coefficients[storage_d_idx] * COST_SCALE_FACTOR;
            assert!(
                (storage_d_value - D_WATER_VALUE_PER_HM3).abs() < TOL,
                "pool-0 slot {slot}: D's storage must be worth {D_WATER_VALUE_PER_HM3} $/hm³; \
                 got {storage_d_value}"
            );
        }
    }

    #[test]
    fn exited_plant_transit_conserves_every_released_hm3() {
        let (system, mut setup) = fresh_system_and_setup_with(&case_dir(), |_| {});
        let scenarios = run_simulation(&mut setup, 1);
        assert_eq!(scenarios.len(), 1, "one deterministic scenario");
        let scenario = &scenarios[0];
        let last_stage = scenario.stages.len() - 1;

        let released_u: f64 = (0..=last_stage)
            .map(|t| release(hydro_row(scenario, t, U_ID)))
            .sum();
        let delivered_j: f64 = (0..J_EXIT_STAGE)
            .map(|t| received_from_upstream(hydro_row(scenario, t, J_ID)))
            .sum();
        let routed_to_d: f64 = (J_EXIT_STAGE..=last_stage)
            .map(|t| {
                received_from_upstream(hydro_row(scenario, t, D_ID))
                    - ZETA * hydro_row(scenario, t, J_ID).incremental_inflow_m3s
                    - release(hydro_row(scenario, t, U_ID))
            })
            .sum();

        let j = system
            .hydros()
            .iter()
            .find(|h| h.id == EntityId(J_ID))
            .expect("J (hydro id 1) must exist");
        assert_eq!(
            j.exit_stage_id,
            Some(i32::try_from(J_EXIT_STAGE).unwrap()),
            "J must retire at stage {J_EXIT_STAGE}"
        );
        assert_eq!(
            last_stage, J_EXIT_STAGE,
            "J must be PreFilling at the last stage, so no deposit into its bucket is written \
             there and nothing is dropped past the horizon"
        );
        let terminal_drop = 0.0;

        let residual = released_u - (delivered_j + routed_to_d + terminal_drop);
        assert!(
            residual.abs() <= TOL,
            "every hm³ U releases toward J must be accounted for: released {released_u}, \
             delivered to J {delivered_j}, routed to D {routed_to_d}, residual {residual}"
        );
        assert!(
            (routed_to_d - TRANSIT_AT_EXIT_HM3).abs() <= TOL,
            "the {TRANSIT_AT_EXIT_HM3} hm³ in transit at J's exit must reach D; got {routed_to_d}"
        );
        let matured = scenario.stages[J_EXIT_STAGE]
            .transit_buckets
            .iter()
            .find(|b| b.hydro_id == J_ID && b.lag == 1)
            .expect("J's lag-1 bucket must be reported at its exit stage")
            .delayed_arrival_hm3;
        assert!(
            (routed_to_d - matured).abs() <= TOL,
            "D must receive exactly J's matured bucket ({matured} hm³); got {routed_to_d}"
        );
    }
}
