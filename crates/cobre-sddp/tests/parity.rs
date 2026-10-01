//! Consolidated parity / determinism / reproducibility integration tests for
//! `cobre-sddp`.
//!
//! Groups the golden parity-hash regression, the self-reproducibility regression,
//! the b6a hydro-inflow parity, and the determinism conformance suite into one
//! binary so the statically-linked solver links once rather than once per file.
//! The two parity-hash sources carry mutually exclusive solver-backend gates, so
//! each is a per-`mod` `#[cfg(feature = …)]`: `parity_hash_highs` compiles under
//! HiGHS and `parity_hash_clp` under CLP.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

mod common;

/// Fixed base seed for the declaration-order-shuffle axis. Permutation `i`'s
/// seed is `SHUFFLE_BASE_SEED.wrapping_add(i as u64)`; the single permutation
/// folded into each `parity_hash_<case>` test uses `i = 0`. See
/// [`common::permute`]'s module doc for the classification map and coverage
/// bound the permutations exercise.
const SHUFFLE_BASE_SEED: u64 = 0x5EED_C0BE_5EED_C0BE;

/// Permutation count for the `shuffle_matrix_<case>` tests, dispatched only via
/// `.github/workflows/invariance-shuffle.yml` (`--run-ignored ignored-only`).
const FULL_SHUFFLE_PERMUTATIONS: usize = 8;

#[cfg(feature = "highs")]
mod parity_hash_highs {
    //! HiGHS golden parity-hash regression. Each case's train + simulate output
    //! is digested over the whitelist owned by
    //! [`compute_parity_hash`](super::common::parity_hash::compute_parity_hash);
    //! the hash pins bit-for-bit determinism and declaration-order invariance, so a
    //! changed hash means a real output change. The `parity_regen_*` tests below
    //! write the baseline instead of asserting it; they are unconditionally
    //! `#[ignore]`d and must be run explicitly. The selection rationale for the
    //! five golden cases lives on [`case_dir`](super::common::parity_hash::case_dir).
    //!
    //! Each `parity_hash_<case>` folds in ONE seeded declaration-order
    //! permutation (asserted against the in-memory base hash, not a committed
    //! baseline); the sibling `shuffle_matrix_<case>` tests run the full
    //! `FULL_SHUFFLE_PERMUTATIONS`-permutation matrix and are unconditionally
    //! `#[ignore]`d — see [`super::SHUFFLE_BASE_SEED`]/[`super::FULL_SHUFFLE_PERMUTATIONS`].

    use cobre_solver::highs::HighsSolver;

    fn run_case(label: &str) -> String {
        super::common::parity_hash::run_golden_case("parity_baselines", label, HighsSolver::new)
    }

    fn regen_case(label: &str) {
        super::common::parity_hash::regen_golden_case("parity_baselines", label, HighsSolver::new);
    }

    /// The base case plus one seeded permutation, asserted against the
    /// in-memory base hash — the default-CI fold-in of the shuffle axis.
    fn run_case_with_permutation(label: &str) {
        let base_hash = run_case(label);
        super::common::parity_hash::assert_permutation_hash(
            label,
            super::SHUFFLE_BASE_SEED,
            &base_hash,
            HighsSolver::new,
        );
    }

    /// The base case plus [`super::FULL_SHUFFLE_PERMUTATIONS`] seeded
    /// permutations, each asserted against the in-memory base hash.
    fn run_shuffle_matrix(label: &str) {
        let base_hash = run_case(label);
        for i in 0..super::FULL_SHUFFLE_PERMUTATIONS {
            let seed = super::SHUFFLE_BASE_SEED.wrapping_add(i as u64);
            super::common::parity_hash::assert_permutation_hash(
                label,
                seed,
                &base_hash,
                HighsSolver::new,
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d06() {
        run_case_with_permutation("D06");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d15() {
        run_case_with_permutation("D15");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d30() {
        run_case_with_permutation("D30");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d34() {
        run_case_with_permutation("D34");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d41() {
        run_case_with_permutation("D41");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d06() {
        run_shuffle_matrix("D06");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d15() {
        run_shuffle_matrix("D15");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d30() {
        run_shuffle_matrix("D30");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d34() {
        run_shuffle_matrix("D34");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d41() {
        run_shuffle_matrix("D41");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d06() {
        regen_case("D06");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d15() {
        regen_case("D15");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d30() {
        regen_case("D30");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d34() {
        regen_case("D34");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d41() {
        regen_case("D41");
    }
}

#[cfg(feature = "clp")]
mod parity_hash_clp {
    //! CLP golden parity-hash regression — the same harness and golden cases as
    //! `parity_hash_highs`, over an independent baseline set.
    //!
    //! CLP's simplex legitimately reaches **different-but-valid** optima, so its
    //! digests differ from other backends': the CLP baselines live under
    //! `tests/fixtures/parity_baselines_clp/`. The committed `*.sha256` files are
    //! machine/CI-canonical — regenerated on the canonical environment to assert
    //! run-to-run reproducibility there, not bit-for-bit reproduction on arbitrary
    //! machines.

    use cobre_solver::clp::ClpSolver;

    fn run_case(label: &str) -> String {
        super::common::parity_hash::run_golden_case("parity_baselines_clp", label, ClpSolver::new)
    }

    fn regen_case(label: &str) {
        super::common::parity_hash::regen_golden_case(
            "parity_baselines_clp",
            label,
            ClpSolver::new,
        );
    }

    /// The base case plus one seeded permutation, asserted against the
    /// in-memory base hash — the default-CI fold-in of the shuffle axis.
    fn run_case_with_permutation(label: &str) {
        let base_hash = run_case(label);
        super::common::parity_hash::assert_permutation_hash(
            label,
            super::SHUFFLE_BASE_SEED,
            &base_hash,
            ClpSolver::new,
        );
    }

    /// The base case plus [`super::FULL_SHUFFLE_PERMUTATIONS`] seeded
    /// permutations, each asserted against the in-memory base hash.
    fn run_shuffle_matrix(label: &str) {
        let base_hash = run_case(label);
        for i in 0..super::FULL_SHUFFLE_PERMUTATIONS {
            let seed = super::SHUFFLE_BASE_SEED.wrapping_add(i as u64);
            super::common::parity_hash::assert_permutation_hash(
                label,
                seed,
                &base_hash,
                ClpSolver::new,
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d06() {
        run_case_with_permutation("D06");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d15() {
        run_case_with_permutation("D15");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d30() {
        run_case_with_permutation("D30");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d34() {
        run_case_with_permutation("D34");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn parity_hash_d41() {
        run_case_with_permutation("D41");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d06() {
        run_shuffle_matrix("D06");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d15() {
        run_shuffle_matrix("D15");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d30() {
        run_shuffle_matrix("D30");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d34() {
        run_shuffle_matrix("D34");
    }

    #[test]
    #[ignore = "full shuffle matrix — dispatched via invariance-shuffle.yml"]
    fn shuffle_matrix_d41() {
        run_shuffle_matrix("D41");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d06() {
        regen_case("D06");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d15() {
        regen_case("D15");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d30() {
        regen_case("D30");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d34() {
        regen_case("D34");
    }

    #[test]
    #[ignore = "rewrites committed parity baselines — run explicitly on the canonical machine"]
    fn parity_regen_d41() {
        regen_case("D41");
    }
}

mod self_reproducibility_regression {
    //! Self-reproducibility regression test.
    //!
    //! Runs the SDDP training + simulation pipeline **twice in the same process**
    //! on the D02 single-hydro fixture and asserts that the parity hash of run-1
    //! equals the parity hash of run-2.
    //!
    //! Distinct from the `parity_hash_highs` module (which compares against committed
    //! SHA-256 baselines on disk) and the `cut_subgradient_parity` module in
    //! `cut_basis.rs` (which verifies the KKT identity for a single isolated LP
    //! solve). This test guards against
    //! future non-determinism vectors — HashMap iteration order, parallel
    //! floating-point reductions, RNG re-seeding drift, scheduler ordering — that
    //! would surface as a hash that drifts between consecutive runs of the same
    //! `(seed, config, input)`.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::{TrainingEvent, scenario::ScenarioSource};
    use cobre_sddp::{
        SimulationWeighting, StudySetup, aggregate_simulation,
        hydro_models::prepare_hydro_models,
        setup::{StudyParams, prepare_stochastic},
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    use super::common::parity_hash::compute_parity_hash;

    fn d02_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic/d02-single-hydro")
    }

    /// Drive one full train + simulate pass and return the parity hash.
    ///
    /// Mirrors the body of `run_case` in the `parity_hash_highs` module but skips the
    /// baseline read/write — this test compares two in-process invocations against
    /// each other.
    fn run_d02_once() -> String {
        let dir = d02_case_dir();
        let config_path = dir.join("config.json");

        let config = cobre_io::parse_config(&config_path).expect("config must parse");

        let system = cobre_io::load_case(&dir).expect("load_case must succeed");

        let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default(), None)
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

        let mut config_with_sim = config;
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

        let params = StudyParams::from_config(&config_with_sim, Vec::new())
            .expect("StudyParams::from_config must succeed");

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
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

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
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
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
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        let sim_config = setup.simulation_config();
        let (_summary, _gathered) = aggregate_simulation(
            &local_costs.costs,
            sim_config,
            &comm,
            SimulationWeighting::Uniform,
        )
        .expect("aggregate_simulation must succeed");

        compute_parity_hash(&setup, scenario_results)
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d02_self_reproducibility() {
        let hash_run1 = run_d02_once();
        let hash_run2 = run_d02_once();
        assert_eq!(
            hash_run1, hash_run2,
            "self-reproducibility violation on d02-single-hydro:\n  \
         run-1 hash: {hash_run1}\n  \
         run-2 hash: {hash_run2}"
        );
    }
}

#[cfg(any(feature = "highs", feature = "clp"))]
mod b6a_hydro_inflow_parity {
    //! B6a end-to-end parity test: a `>=` generic constraint on the total realized
    //! inflow `hydro_inflow(h)` solves on both LP backends, and the B6a keyword is
    //! inert for studies that do not use it.
    //!
    //! ## What this file proves (and what it deliberately does not)
    //!
    //! `hydro_inflow(h[,blk])` resolves to the **total** inflow column set:
    //! `z_inflow(h)` plus, for each immediately-upstream plant `i`, the turbine and
    //! spillage columns at the referenced block (plus any diversion-into columns),
    //! each with coefficient `+1.0`. The exhaustive **resolver-level** assertion of
    //! that full column set — that the row references `z_inflow` AND each upstream
    //! plant's turbine+spillage (and diversion-into) at `+1.0`, NOT merely that
    //! `z_inflow` appears — is a crate-internal property of `resolve_variable_ref`,
    //! which is `pub(crate)` to `cobre-sddp` and therefore unreachable from an
    //! integration test under `tests/`. That assertion lives in the crate-internal
    //! unit tests in `lp::builder::generic_constraints` (e.g.
    //! `hydro_inflow_two_upstream_canonical_order`,
    //! `hydro_inflow_diversion_into_appends_diversion_column`,
    //! `hydro_inflow_headwater_resolves_to_z_inflow_only`,
    //! `hydro_inflow_is_block_dependent`), which assert the exact `(col, +1.0)`
    //! pair list for a multi-upstream cascade. This file relies on that coverage and
    //! proves the complementary, genuinely-integration property the unit tests
    //! cannot: the full public `load → train → simulate` pipeline materializes the
    //! cascade `hydro_inflow` row and **solves** it on HiGHS and on CLP.
    //!
    //! ## Fixture
    //!
    //! `tests/fixtures/b6a_hydro_inflow_cascade/` is a two-hydro cascade (H0 → H1)
    //! with a `hydro_inflow(1) >= 12.0` generic constraint. H1's standalone
    //! incremental inflow is 10/5/2 m³/s by stage, so the `12.0` lower bound is
    //! above the incremental inflow in every stage: a slack-free solution must count
    //! the upstream H0 turbine+spillage release routed into H1, making the cascade
    //! columns load-bearing. Slack is enabled (so the LP is never infeasible) and
    //! the bound is well below the feasible total-inflow ceiling (~40 m³/s when H0
    //! releases at capacity), so the row participates without forcing infeasibility.
    //!
    //! ## Inertness (no new baseline)
    //!
    //! B6a adds no LP columns and no rows for any study without a `hydro_inflow`
    //! constraint, so the existing parity baselines are byte-identical and this file
    //! adds **no** new `.sha256` baseline to `tests/fixtures/parity_baselines*`.

    use cobre_io::config::SimulationSelection;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_core::{CoefficientRef, EntityId, VariableRef};
    use cobre_sddp::{
        SimulationWeighting, aggregate_simulation, hydro_models::prepare_hydro_models,
        setup::prepare_stochastic,
    };

    use super::common::{StubComm, build_setup_for_case};

    /// Path to the multi-plant cascade fixture carrying the `hydro_inflow(1)`
    /// generic constraint.
    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/b6a_hydro_inflow_cascade")
    }

    /// Pin the fixture's intent at the parser boundary.
    ///
    /// The cascade target is H1 (`EntityId(1)`), the downstream plant whose upstream
    /// is H0 — so the resolved row references the total-inflow column set (z_inflow
    /// plus H0's turbine+spillage), not the headwater z_inflow-only case. The
    /// exhaustive `(col, +1.0)` resolver-level check is the crate-internal
    /// responsibility of `lp::builder::generic_constraints`'s unit tests (see this file's
    /// module docs); this assertion only guards that the fixture references the
    /// cascade target so the end-to-end solve genuinely exercises B6a.
    fn assert_cascade_inflow_constraint(system: &cobre_core::System) {
        let constraints = system.generic_constraints();
        assert_eq!(
            constraints.len(),
            1,
            "fixture must declare exactly one generic constraint, got {}",
            constraints.len()
        );
        let gc = &constraints[0];
        // Shape derives from the resolved bounds row, not an authored sense: the
        // fixture must be lower-only (`bound_lower` present, `bound_upper` absent),
        // matching the historical `>=` bound.
        let bounds = system.resolved_generic_bounds().bounds_for_stage(0, 0);
        assert_eq!(
            bounds.len(),
            1,
            "fixture must carry exactly one bound entry at stage 0, got {}",
            bounds.len()
        );
        assert_eq!(
            bounds[0].bound_lower,
            Some(12.0),
            "fixture constraint must have bound_lower = 12.0"
        );
        assert_eq!(
            bounds[0].bound_upper, None,
            "fixture constraint must be lower-only (no bound_upper)"
        );
        assert_eq!(
            gc.expression.terms.len(),
            1,
            "fixture constraint must be a single hydro_inflow term, got {} terms",
            gc.expression.terms.len()
        );
        let term = &gc.expression.terms[0];
        assert_eq!(
            term.variable,
            VariableRef::HydroInflow {
                hydro_id: EntityId(1),
                block_id: None,
            },
            "fixture constraint must reference hydro_inflow(1) (cascade target H1) with no explicit block"
        );
        assert_eq!(
            term.coefficient,
            CoefficientRef::Literal(1.0),
            "the hydro_inflow term coefficient must be the literal +1.0"
        );
        assert!(
            (term.scale - 1.0).abs() < f64::EPSILON,
            "the hydro_inflow term scale must be +1.0, got {}",
            term.scale
        );
    }

    /// Run the public `load → train → simulate` pipeline for the cascade
    /// `hydro_inflow` fixture under the given LP solver, asserting the LP solves
    /// without error end-to-end.
    ///
    /// Generic over the solver so the HiGHS and CLP entry points share one body;
    /// the closure `make_solver` supplies fresh worker solvers for the workspace
    /// pool exactly as the production training/simulation paths do.
    fn run_cascade_inflow_case<S, F>(make_solver: F)
    where
        S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
        F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
    {
        let dir = fixture_dir();
        let config_path = dir.join("config.json");

        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(&dir).expect("load_case must succeed");

        assert_cascade_inflow_constraint(&system);

        let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default(), None)
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

        // Enable simulation so the cascade `hydro_inflow` row is exercised on the
        // simulation LP as well as the training LP, mirroring the D-case harness.
        let mut config_with_sim = config;
        config_with_sim.simulation.enabled = true;
        config_with_sim.simulation.selection =
            Some(SimulationSelection::Sampled { num_scenarios: 1 });

        let mut setup =
            build_setup_for_case(&dir, &config_with_sim, &system, stochastic, hydro_models);

        let comm = StubComm;
        let mut solver = make_solver().expect("solver construction must succeed");

        let (event_tx, _event_rx) = mpsc::channel();
        let outcome = setup
            .train(&mut solver, &comm, 1, make_solver, Some(event_tx), None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "training must not surface a solver error for the cascade hydro_inflow row: {:?}",
            outcome.error
        );
        let result = outcome.result;

        let mut pool = setup
            .create_workspace_pool(&comm, 1, make_solver)
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
        assert!(
            !scenario_results.is_empty(),
            "simulation must produce at least one scenario result"
        );

        let sim_config = setup.simulation_config();
        aggregate_simulation(
            &local_costs.costs,
            sim_config,
            &comm,
            SimulationWeighting::Uniform,
        )
        .expect("aggregate_simulation must succeed");
    }

    /// The cascade `hydro_inflow(1) >= 12.0` constraint solves end-to-end on the
    /// default HiGHS backend.
    #[test]
    #[cfg(feature = "highs")]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn b6a_hydro_inflow_cascade_solves_highs() {
        use cobre_solver::highs::HighsSolver;
        run_cascade_inflow_case(HighsSolver::new);
    }

    /// The same cascade `hydro_inflow(1) >= 12.0` constraint solves end-to-end
    /// on the CLP backend (`--no-default-features --features "clp slow-tests"`).
    #[test]
    #[cfg(feature = "clp")]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn b6a_hydro_inflow_cascade_solves_clp() {
        use cobre_solver::clp::ClpSolver;
        run_cascade_inflow_case(ClpSolver::new);
    }
}

mod determinism {
    //! Determinism verification tests for the SDDP training and simulation loops.
    //!
    //! Verifies that the rayon-parallelized forward pass, backward pass, and
    //! simulation produce bit-identical results regardless of thread count.
    //! This property is guaranteed by:
    //!
    //! 1. Deterministic SipHash-1-3 seed derivation per scenario.
    //! 2. Declaration-order invariance in entity processing.
    //! 3. Static work partitioning (not rayon default chunking).
    //! 4. Deterministic cut merging order (sorted by trial point index).
    //!
    //! Each test runs with 1 workspace, then with 4 workspaces, and asserts
    //! bit-exact equality on all outputs.

    use std::collections::{BTreeMap, HashMap};
    use std::sync::mpsc;

    use chrono::NaiveDate;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::{
        DeficitSegment, EntityId,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            SamplingScheme,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_sddp::{
        Phase, SolverProfiles, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
        config::{CutManagementConfig, EventConfig, LoopConfig},
        context::TrainingContext,
        cut::FutureCostFunction,
        energy_conversion::{EnergyConversion, EnergyConversionSet},
        forward::{ForwardBound, ForwardResult, sync_forward},
        horizon_mode::HorizonMode,
        indexer::{CutStateProjection, StateSpace, StudyDimensions},
        inflow_method::InflowNonNegativityMethod,
        lead_time::AnticipatedResolution,
        lp::builder::{PatchBuffer, StageGeometry},
        risk_measure::RiskMeasure,
        setup::node_graph::Traversal,
        simulate,
        simulation::{EntityCounts, SimulationConfig, SimulationOutputSpec},
        test_support::{GeometryDims, StageContextFixture, geometry, permissive_state_boxes},
        train,
        workspace::{SolverWorkspace, WorkspaceSizing},
    };
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::{
        ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
    };

    // ===========================================================================
    // Shared communicator stub
    // ===========================================================================

    /// Mirrors the gated `test_support::state_layout_for` via the public
    /// [`StateSpace::new`] constructor: this external test crate cannot see the
    /// parent crate's `#[cfg(test)]` surface, so it rebuilds byte-identical patch
    /// columns on the default feature set.
    fn state_layout_for(hydro_count: usize, max_par_order: usize) -> StateSpace {
        StateSpace::new(
            hydro_count,
            max_par_order,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &vec![max_par_order; hydro_count],
        )
    }

    fn study_dims() -> StudyDimensions {
        StudyDimensions::default()
    }

    /// Build the [`StageGeometry`] `template_3h` addresses: N=3 hydros, 1 bus,
    /// 1 block — the production layout `test_support::geometry` builds for
    /// those dims.
    fn three_hydro_one_bus_geometry() -> StageGeometry {
        geometry(
            &GeometryDims {
                hydro_count: 3,
                n_buses: 1,
                n_blks: 1,
                ..GeometryDims::default()
            },
            vec![],
            &[],
            vec![],
        )
    }

    // ===========================================================================
    // Mock solver for N=3 hydros, L=0 PAR
    //
    // Column layout for the N=3, L=0 state vector:
    //   storage      = 0..3
    //   inflow_lags  = 3..3  (empty, L=0)
    //   z_inflow     = 3..6
    //   storage_in   = 6..9
    //   theta        = 9
    //   then the decoupled hydro+bus equipment padding `test_support::geometry`
    //   addresses for N=3 hydros, 1 bus, 1 block (cols 10-38), so per-block
    //   hydro/bus extraction addresses real columns instead of an empty family.
    //   num_cols     = 39
    //
    // The primal must have 39 entries so simulation's per-block extraction can
    // address every equipment column. The dual must have at least 7 entries
    // to cover the widened geometry's water-balance/load-balance row family.
    // ===========================================================================

    const PRIMAL_3H: &[f64] = &[0.0; 39];
    // The dual must cover the widened row family plus max cuts per stage
    // (10 iterations × 1 pass = 10). Use 64 to cover any reasonable iteration
    // count without tight sizing.
    const DUAL_3H: &[f64] = &[0.0; 64];
    const REDUCED_COSTS_3H: &[f64] = &[0.0; 39];

    /// Mock solver returning a fixed objective on every solve, so any output
    /// variation across thread counts comes from the orchestration layer alone.
    struct MockSolver3H {
        objective: f64,
    }

    impl MockSolver3H {
        fn new(objective: f64) -> Self {
            Self { objective }
        }
    }

    impl SolverInterface for MockSolver3H {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _cuts: &RowBatch) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            Ok(cobre_solver::SolutionView {
                objective: self.objective,
                primal: PRIMAL_3H,
                dual: DUAL_3H,
                reduced_costs: REDUCED_COSTS_3H,
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }

        fn get_basis(&mut self, out: &mut Basis) {
            cobre_sddp::test_support::fill_consistent_basis(out);
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            *out = self.statistics();
        }

        fn name(&self) -> &'static str {
            "MockDeterminism3H"
        }
    }

    // ===========================================================================
    // Fixture construction
    // ===========================================================================

    /// 3-hydro `StochasticContext` (seed 42, PAR(0)) — small but exercising more
    /// paths than a 1-hydro fixture.
    fn make_stochastic_context_3h_branching(
        n_stages: usize,
        branching_factor: usize,
    ) -> StochasticContext {
        use cobre_core::SystemBuilder;
        use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::InflowModel;

        let zero_penalties = || HydroPenalties {
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
            inflow_nonnegativity_cost: 1000.0,
        };

        let bus = make_bus(
            EntityId(0),
            BusSpec {
                name: "B0".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let build_hydro = |id_val: i32, name: &str| {
            make_hydro(
                EntityId(id_val),
                HydroSpec {
                    name: name.to_string(),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(0),
                    downstream_id: None,
                    entry_stage_id: None,
                    exit_stage_id: None,
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
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
                    penalties: zero_penalties(),
                    ..Default::default()
                },
            )
        };

        let hydros = vec![
            build_hydro(1, "H1"),
            build_hydro(2, "H2"),
            build_hydro(3, "H3"),
        ];

        let stages: Vec<Stage> = (0..n_stages)
            .map(|idx| {
                make_stage(
                    idx,
                    StageSpec {
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
                            branching_factor,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect();

        let mut inflow_models: Vec<InflowModel> = Vec::new();
        for hydro_id in [1i32, 2, 3] {
            for stage_idx in 0..n_stages {
                inflow_models.push(InflowModel {
                    hydro_id: EntityId(hydro_id),
                    stage_id: stage_idx as i32,
                    mean_m3s: 50.0 + f64::from(hydro_id) * 10.0,
                    std_m3s: 15.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                });
            }
        }

        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(1),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(2),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(3),
                        },
                    ],
                    matrix: vec![
                        vec![1.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 1.0],
                    ],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(
            &system,
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
        .unwrap()
    }

    /// Build a `StageTemplate` for a 3-hydro, PAR(0) stage LP.
    ///
    /// Column layout (N=3, L=0):
    /// ```text
    /// 0..3   storage_out  (outgoing storage, N=3)
    /// 3..6   z_inflow     (realized inflow variables, N=3, free)
    /// 6..9   storage_in   (incoming storage, N=3, L=0 → no lag cols)
    /// 9      theta
    /// 10..39 decoupled hydro+bus equipment padding `test_support::geometry`
    ///        addresses for N=3 hydros, 1 bus, 1 block (zero cost, zero NZ):
    ///        `MockSolver3H` never reads the coefficients.
    /// ```
    ///
    /// Row layout (N=3, L=0):
    /// ```text
    /// 0..3  z_inflow rows        (one per hydro, at StateSpace::z_inflow_rows())
    /// 3..6  storage_in pin rows (one per hydro)
    /// 6     the widened geometry's load-balance row (unused, no NZ)
    /// ```
    ///
    /// The matrix has one nonzero per z_inflow row (column = `z_inflow[h]`) and one
    /// per storage_in pin row (column = `storage_in[h]`), each coefficient 1.0, so
    /// the patch buffer has something to patch.
    fn template_3h() -> StageTemplate {
        let num_cols = 39;
        let num_rows = 7;
        // CSC col_starts: 39 columns + 1 sentinel; z_inflow and storage_in cols
        // each carry one NZ, cols 9 (theta) onward carry none.
        let col_starts = {
            let mut v = vec![6_i32; num_cols + 1];
            v[0] = 0; // col 0 (storage_out[0])
            v[1] = 0; // col 1 (storage_out[1])
            v[2] = 0; // col 2 (storage_out[2])
            v[3] = 0; // col 3 (z_inflow[0]) — NZ starts here
            v[4] = 1; // col 4 (z_inflow[1])
            v[5] = 2; // col 5 (z_inflow[2])
            v[6] = 3; // col 6 (storage_in[0])
            v[7] = 4; // col 7 (storage_in[1])
            v[8] = 5; // col 8 (storage_in[2])
            v
        };
        let row_indices = vec![0_i32, 1, 2, 3, 4, 5];
        let values = vec![1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut objective = vec![0.0_f64; num_cols];
        objective[9] = 1.0; // theta at col 9

        let mut col_lower = vec![0.0_f64; num_cols];
        col_lower[3..6].fill(f64::NEG_INFINITY); // z_inflow cols are free

        StageTemplate {
            num_cols,
            num_rows,
            num_nz: 6,
            col_starts,
            row_indices,
            values,
            col_lower,
            col_upper: vec![f64::INFINITY; num_cols],
            objective,
            row_lower: vec![0.0; num_rows],
            row_upper: vec![0.0; num_rows],
            n_state: 3,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn make_fcf_3h(n_stages: usize) -> FutureCostFunction {
        // state_dimension = 3, forward_passes = 1, capacity = 50 iterations, 0 warm-start cuts
        FutureCostFunction::new(n_stages, 3, 1, 50, &vec![0; n_stages])
    }

    fn iteration_limit(limit: u64) -> StoppingRuleSet {
        StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit }],
            mode: StoppingMode::Any,
        }
    }

    /// All training parameters for the 3-hydro, 5-stage test system.
    struct Fixture3H {
        n_stages: usize,
        templates: Vec<StageTemplate>,
        state: StateSpace,
        initial_state: Vec<f64>,
        stochastic: StochasticContext,
        horizon: HorizonMode,
        risk_measures: Vec<RiskMeasure>,
    }

    impl Fixture3H {
        fn new() -> Self {
            Self::with_branching(1)
        }

        /// Build the 3-hydro / 5-stage fixture with `branching_factor` openings per
        /// stage. `branching_factor > 1` is used by the opening-reorder determinism
        /// test so the per-trial-point solve loop visits multiple openings.
        fn with_branching(branching_factor: usize) -> Self {
            let n_stages = 5;
            let state = state_layout_for(3, 0);
            let templates = vec![template_3h(); n_stages];
            let initial_state = vec![0.0_f64; state.n_state];
            let stochastic = make_stochastic_context_3h_branching(n_stages, branching_factor);
            let horizon = HorizonMode::Finite {
                num_stages: n_stages,
            };
            let risk_measures = vec![RiskMeasure::Expectation; n_stages];

            Self {
                n_stages,
                templates,
                state,
                initial_state,
                stochastic,
                horizon,
                risk_measures,
            }
        }
    }

    // ===========================================================================
    // Helper: run training with a given number of forward-pass workspaces
    // ===========================================================================

    /// Run `train()` on the 3-hydro fixture with `n_workspaces` forward-pass threads,
    /// returning `(TrainingResult, FutureCostFunction)`. Uses an isolated rayon pool
    /// to avoid interaction with the global pool or other parallel tests.
    ///
    /// Cuts must be bit-identical across thread counts: the backward pass solves each
    /// trial point's openings in the run-constant precomputed solve order (identity by
    /// default; a non-identity order can be installed via
    /// [`StochasticContext::set_solve_order`]), and per-ω outcomes are aggregated by
    /// canonical ω.
    fn run_training(
        n_workspaces: usize,
        fx: &Fixture3H,
        n_iterations: u64,
    ) -> (cobre_sddp::TrainingResult, FutureCostFunction) {
        let mut fcf = make_fcf_3h(fx.n_stages);
        let mut primary_solver = MockSolver3H::new(100.0);
        let comm = StubComm;

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: n_iterations,
                start_iteration: 0,
                n_fwd_threads: 1,
                stopping_rules: iteration_limit(n_iterations),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workspaces)
            .build()
            .unwrap();

        let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
        let geometry = vec![three_hydro_one_bus_geometry(); fx.n_stages];
        let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
        let stage_ctx = stage_ctx_fixture.ctx();
        let result = pool
            .install(|| {
                train(
                    &mut primary_solver,
                    config,
                    &mut fcf,
                    &stage_ctx,
                    &TrainingContext {
                        node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
                        horizon: &fx.horizon,
                        state: &fx.state,
                        cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
                        study_dims: &study_dims(),
                        inflow_method: &InflowNonNegativityMethod::None,
                        stochastic: &fx.stochastic,
                        initial_state: &fx.initial_state,
                        inflow_scheme: SamplingScheme::InSample,
                        load_scheme: SamplingScheme::InSample,
                        ncs_scheme: SamplingScheme::InSample,
                        historical_library: None,
                        external_inflow_library: None,
                        external_load_library: None,
                        external_ncs_library: None,
                        stages: &[],
                        lag_accum_seed: &[],
                        lag_weight_seed: &[],
                        dcs: None,
                    },
                    &comm,
                    || Ok(MockSolver3H::new(100.0)),
                    None,
                    SolverProfiles::default(),
                )
            })
            .unwrap();

        (result.result, fcf)
    }

    // ===========================================================================
    // Helper: run simulation with a given number of workspaces
    // ===========================================================================

    /// Run `simulate()` on the trained FCF with `n_workspaces` worker threads.
    ///
    /// Returns the sorted cost buffer `Vec<(scenario_id, total_cost, category_costs)>`.
    fn run_simulation(
        n_workspaces: usize,
        fx: &Fixture3H,
        fcf: &FutureCostFunction,
        n_scenarios: u32,
    ) -> Vec<(u32, f64, cobre_sddp::ScenarioCategoryCosts)> {
        let sim_config = SimulationConfig {
            n_scenarios,
            io_channel_capacity: 64,
            profile: Phase::Simulation.profile(),
        };
        let entity_counts = EntityCounts {
            hydro_ids: vec![1, 2, 3],
            hydro_productivities: vec![1.0, 1.0, 1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![0],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let zero_ec = EnergyConversion {
            equivalent_productivity_mw_per_m3s: 0.0,
            reference_volume_hm3: 0.0,
            reference_outflow_m3s: 0.0,
        };
        let ec = EnergyConversionSet::new(
            vec![vec![zero_ec; fx.n_stages]; 3],
            vec![vec![0.0_f64; fx.n_stages]; 3],
            &cobre_sddp::test_support::minimal_hydros(3),
            fx.n_stages,
        );

        let sim_training_ctx = TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };
        let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
        let geometry = vec![three_hydro_one_bus_geometry(); fx.n_stages];
        let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
        let mut workspaces: Vec<SolverWorkspace<MockSolver3H>> = (0..n_workspaces)
            .map(|idx| {
                SolverWorkspace::new(
                    0,
                    i32::try_from(idx).expect("worker_id fits in i32"),
                    MockSolver3H::new(100.0),
                    PatchBuffer::new(&fx.state, &[], &[]),
                    &sim_training_ctx,
                    &stage_ctx_fixture.ctx(),
                    WorkspaceSizing::default(),
                )
            })
            .collect();

        let comm = StubComm;
        let (result_tx, result_rx) = mpsc::sync_channel(64);

        // Drain the channel in a background thread to avoid blocking simulate().
        let drain_thread = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workspaces)
            .build()
            .unwrap();

        let cost_buffer = pool
            .install(|| {
                simulate(
                    &mut workspaces,
                    &stage_ctx_fixture.ctx(),
                    fcf,
                    &sim_training_ctx,
                    &sim_config,
                    SimulationOutputSpec {
                        result_tx: &result_tx,
                        hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
                        block_hours_per_stage: &vec![vec![744.0]; fx.n_stages],
                        entity_counts: &entity_counts,
                        generic_constraint_row_entries: &vec![Vec::new(); fx.n_stages],
                        pumping_consumption_mw_per_m3s: &[],
                        contract_prices_per_stage: &vec![Vec::new(); fx.n_stages],
                        contract_slots: &[],
                        diversion_upstream: &HashMap::new(),
                        hydro_productivities_per_stage: &vec![vec![1.0, 1.0, 1.0]; fx.n_stages],
                        energy_conversion: &ec,
                        hydro_min_storage_hm3: &[0.0; 3],
                        event_sender: None,
                        extended_delivery_anchors: &[],
                        transit_seed_arcs: &[],
                        past_defluences: &[],
                        study_stage_dates: &[],
                    },
                    None,
                    &[],
                    &comm,
                    &Traversal::default(),
                )
            })
            .unwrap();

        drop(result_tx);
        let _ = drain_thread.join().unwrap();

        cost_buffer.costs
    }

    // ===========================================================================
    // Test: training determinism across thread counts
    // ===========================================================================

    /// Verify that `train()` produces bit-identical outputs when run with 1
    /// workspace versus 4 workspaces.
    ///
    /// The fixture uses 3 hydros and 5 stages, which exercises enough
    /// parallelisation code paths (forward pass partitioning, backward pass
    /// synchronisation, cut merging order) to catch ordering bugs.
    #[test]
    fn test_training_determinism_across_thread_counts() {
        const N_ITERATIONS: u64 = 10;

        let fx = Fixture3H::new();
        let (result_1, fcf_1) = run_training(1, &fx, N_ITERATIONS);
        let (result_4, fcf_4) = run_training(4, &fx, N_ITERATIONS);

        assert_eq!(result_1.iterations, result_4.iterations);
        assert_eq!(result_1.final_lb.to_bits(), result_4.final_lb.to_bits());
        assert_eq!(result_1.final_ub.to_bits(), result_4.final_ub.to_bits());
        assert_eq!(result_1.final_gap.to_bits(), result_4.final_gap.to_bits());

        assert_eq!(fcf_1.pools.len(), fcf_4.pools.len());
        for t in 0..fcf_1.pools.len() {
            let pool_1 = &fcf_1.pools[t];
            let pool_4 = &fcf_4.pools[t];
            assert_eq!(pool_1.populated(), pool_4.populated());

            for s in 0..pool_1.populated() {
                assert_eq!(pool_1.is_active(s), pool_4.is_active(s));
                assert_eq!(pool_1.intercept(s).to_bits(), pool_4.intercept(s).to_bits());
                let c1 = pool_1.coefficient_row(s);
                let c4 = pool_4.coefficient_row(s);
                assert_eq!(c1.len(), c4.len());
                for (&coeff_1, &coeff_4) in c1.iter().zip(c4.iter()) {
                    assert_eq!(coeff_1.to_bits(), coeff_4.to_bits());
                }
            }
        }
    }

    /// Work-distribution invariance with a NON-IDENTITY backward opening solve order.
    ///
    /// This is the in-crate proof gate for the reorder feature: with a NON-IDENTITY
    /// per-stage solve order installed on the opening tree (the backward pass always
    /// honors the solve order), `train()` must still produce bit-identical FCF cuts
    /// when run with 1 workspace versus 4 workspaces. The solve order is run-constant
    /// (precomputed once, identical on every worker), so every worker solves a
    /// given trial point's openings in the same order — yielding the same warm-start
    /// chain and the same per-ω values regardless of thread count. We do NOT compare
    /// against an identity-order run: reordering the solves legitimately changes the
    /// warm-start bases (marginal numerical differences are accepted); the invariant
    /// is across work distributions for the SAME version + config.
    #[test]
    fn test_training_determinism_across_thread_counts_with_reorder() {
        const N_ITERATIONS: u64 = 10;
        const BRANCHING: usize = 4;

        let mut fx = Fixture3H::with_branching(BRANCHING);

        // Keys are arbitrary: the determinism property needs the order to be
        // run-constant, not noise-derived.
        let keys: Vec<Vec<f64>> = (0..fx.n_stages)
            .map(|s| {
                assert_eq!(
                    fx.stochastic.tree_view().n_openings(s),
                    BRANCHING,
                    "fixture must have BRANCHING openings per stage"
                );
                vec![3.0, 1.0, 4.0, 2.0]
            })
            .collect();
        fx.stochastic
            .set_solve_order(&keys)
            .expect("solve-order key dims match the tree");
        assert_eq!(
            fx.stochastic.tree_view().solve_order(0),
            &[2u32, 0, 3, 1],
            "expected a non-identity descending solve order"
        );

        let (result_1, fcf_1) = run_training(1, &fx, N_ITERATIONS);
        let (result_4, fcf_4) = run_training(4, &fx, N_ITERATIONS);

        assert_eq!(result_1.iterations, result_4.iterations);
        assert_eq!(result_1.final_lb.to_bits(), result_4.final_lb.to_bits());
        assert_eq!(result_1.final_ub.to_bits(), result_4.final_ub.to_bits());
        assert_eq!(result_1.final_gap.to_bits(), result_4.final_gap.to_bits());

        assert_eq!(fcf_1.pools.len(), fcf_4.pools.len());
        for t in 0..fcf_1.pools.len() {
            let pool_1 = &fcf_1.pools[t];
            let pool_4 = &fcf_4.pools[t];
            assert_eq!(pool_1.populated(), pool_4.populated());

            for s in 0..pool_1.populated() {
                assert_eq!(pool_1.is_active(s), pool_4.is_active(s));
                assert_eq!(pool_1.intercept(s).to_bits(), pool_4.intercept(s).to_bits());
                let c1 = pool_1.coefficient_row(s);
                let c4 = pool_4.coefficient_row(s);
                assert_eq!(c1.len(), c4.len());
                for (&coeff_1, &coeff_4) in c1.iter().zip(c4.iter()) {
                    assert_eq!(coeff_1.to_bits(), coeff_4.to_bits());
                }
            }
        }
    }

    // ===========================================================================
    // Multi-rank mock communicator
    // ===========================================================================

    use std::any::Any;
    use std::cell::RefCell;
    use std::sync::Arc;

    use super::common::StubComm;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };

    /// Type-erased carrier for the pre-assembled gather buffer: stored as
    /// `Arc<dyn Any>`, recovered in `allgatherv<T>` via `downcast_ref` so the test
    /// simulates multi-rank gather with no unsafe transmutation.
    struct GatherBuffer<T>(Vec<T>);

    thread_local! {
        static MOCK_GATHER_BUFFER: RefCell<Arc<dyn Any + Send + Sync>> =
            RefCell::new(Arc::new(GatherBuffer::<f64>(Vec::new())));
    }

    /// Mock communicator simulating a single rank within a multi-rank group.
    /// `allgatherv` fills the entire `recv` from the buffer pre-loaded via
    /// [`MultiRankMockComm::set_gather_buffer`], reproducing real MPI `allgatherv`
    /// where every rank receives all data.
    struct MultiRankMockComm {
        rank: usize,
        total_size: usize,
    }

    impl MultiRankMockComm {
        fn new(rank: usize, total_size: usize) -> Self {
            Self { rank, total_size }
        }

        /// Pre-load `allgatherv`'s gather buffer. `global_costs` must hold every
        /// rank's scenario costs in rank order (rank 0 first), as real MPI delivers.
        /// Must run on the thread that will call `sync_forward` (thread-local).
        fn set_gather_buffer(global_costs: Vec<f64>) {
            MOCK_GATHER_BUFFER.with(|cell| {
                *cell.borrow_mut() = Arc::new(GatherBuffer(global_costs));
            });
        }
    }

    impl Communicator for MultiRankMockComm {
        /// Fills `recv` from the pre-loaded `GatherBuffer<T>` (`T = f64` in these
        /// tests). The downcast-failure branch fills only the local rank's slot and
        /// is unreachable here.
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            counts: &[usize],
            displs: &[usize],
        ) -> Result<(), CommError> {
            MOCK_GATHER_BUFFER.with(|cell| {
                let arc = cell.borrow();
                if let Some(buf) = arc.downcast_ref::<GatherBuffer<T>>() {
                    for rank in 0..self.total_size {
                        let start = displs[rank];
                        let count = counts[rank];
                        recv[start..start + count].clone_from_slice(&buf.0[start..start + count]);
                    }
                } else {
                    let local_start = displs[self.rank];
                    let local_count = counts[self.rank];
                    recv[local_start..local_start + local_count].clone_from_slice(send);
                }
            });
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv.clone_from_slice(send);
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            self.rank
        }

        fn size(&self) -> usize {
            self.total_size
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    // ===========================================================================
    // Test: canonical upper bound determinism across rank counts
    // ===========================================================================

    /// `sync_forward` produces bit-identical `SyncResult` statistics when the same 8
    /// scenario costs are partitioned across 1, 2, and 4 virtual ranks: it
    /// `allgatherv`s a flat cost vector in global scenario-index order (rank 0's
    /// costs first) and sums it sequentially, so the summation order is independent
    /// of rank count. Uses [`MultiRankMockComm`] to simulate multi-rank gather
    /// without an MPI installation.
    #[test]
    fn test_canonical_ub_determinism_across_rank_counts() {
        // Values chosen so partial sums group differently across partition
        // boundaries, exercising the canonical summation property.
        const ALL_COSTS: &[f64] = &[100.0, 200.0, 150.0, 175.0, 125.0, 190.0, 160.0, 180.0];
        const N: usize = 8;
        const TOTAL_FWD_PASSES: usize = N;

        let result_1rank = {
            let local = ForwardResult {
                scenario_costs: ALL_COSTS.to_vec(),
                elapsed_ms: 0,
                lp_solves: 0,
                setup_time_ms: 0,
                load_imbalance_ms: 0,
                scheduling_overhead_ms: 0,
                stage_stats: Vec::new(),
            };
            sync_forward(
                &local,
                &StubComm,
                TOTAL_FWD_PASSES,
                ForwardBound::Statistical,
            )
            .unwrap()
        };

        let result_2rank = {
            MultiRankMockComm::set_gather_buffer(ALL_COSTS.to_vec());
            let comm = MultiRankMockComm::new(0, 2);
            let local = ForwardResult {
                scenario_costs: ALL_COSTS[..4].to_vec(),
                elapsed_ms: 0,
                lp_solves: 0,
                setup_time_ms: 0,
                load_imbalance_ms: 0,
                scheduling_overhead_ms: 0,
                stage_stats: Vec::new(),
            };
            sync_forward(&local, &comm, TOTAL_FWD_PASSES, ForwardBound::Statistical).unwrap()
        };

        let result_4rank = {
            MultiRankMockComm::set_gather_buffer(ALL_COSTS.to_vec());
            let comm = MultiRankMockComm::new(0, 4);
            let local = ForwardResult {
                scenario_costs: ALL_COSTS[..2].to_vec(),
                elapsed_ms: 0,
                lp_solves: 0,
                setup_time_ms: 0,
                load_imbalance_ms: 0,
                scheduling_overhead_ms: 0,
                stage_stats: Vec::new(),
            };
            sync_forward(&local, &comm, TOTAL_FWD_PASSES, ForwardBound::Statistical).unwrap()
        };

        assert_eq!(
            result_1rank.global_ub_mean.to_bits(),
            result_2rank.global_ub_mean.to_bits()
        );
        assert_eq!(
            result_1rank.global_ub_mean.to_bits(),
            result_4rank.global_ub_mean.to_bits()
        );
        assert_eq!(
            result_1rank.global_ub_std.to_bits(),
            result_2rank.global_ub_std.to_bits()
        );
        assert_eq!(
            result_1rank.global_ub_std.to_bits(),
            result_4rank.global_ub_std.to_bits()
        );
        assert_eq!(
            result_1rank.ci_95_half_width.to_bits(),
            result_2rank.ci_95_half_width.to_bits()
        );
        assert_eq!(
            result_1rank.ci_95_half_width.to_bits(),
            result_4rank.ci_95_half_width.to_bits()
        );
    }

    // ===========================================================================
    // Test: simulation determinism across thread counts
    // ===========================================================================

    /// Verify that `simulate()` produces bit-identical cost buffers when run with
    /// 1 workspace versus 4 workspaces on the same trained FCF.
    ///
    /// Uses the same 3-hydro, 5-stage fixture with 20 scenarios, which is enough
    /// to require work distribution across multiple workers when 4 workspaces are
    /// active.
    #[test]
    fn test_simulation_determinism_across_thread_counts() {
        const N_ITERATIONS: u64 = 10;
        const N_SCENARIOS: u32 = 20;

        let fx = Fixture3H::new();
        let (_training_result, fcf) = run_training(1, &fx, N_ITERATIONS);

        let costs_1 = run_simulation(1, &fx, &fcf, N_SCENARIOS);
        let costs_4 = run_simulation(4, &fx, &fcf, N_SCENARIOS);

        assert_eq!(costs_1.len(), costs_4.len());
        for ((id_1, cost_1, cats_1), (id_4, cost_4, cats_4)) in costs_1.iter().zip(costs_4.iter()) {
            assert_eq!(id_1, id_4);
            assert_eq!(cost_1.to_bits(), cost_4.to_bits());
            assert_eq!(
                cats_1.resource_cost.to_bits(),
                cats_4.resource_cost.to_bits()
            );
            assert_eq!(
                cats_1.recourse_cost.to_bits(),
                cats_4.recourse_cost.to_bits()
            );
            assert_eq!(
                cats_1.violation_cost.to_bits(),
                cats_4.violation_cost.to_bits()
            );
            assert_eq!(
                cats_1.regularization_cost.to_bits(),
                cats_4.regularization_cost.to_bits()
            );
            assert_eq!(cats_1.imputed_cost.to_bits(), cats_4.imputed_cost.to_bits());
        }
    }

    /// Simulation work-distribution invariance when the policy was trained with a
    /// NON-IDENTITY backward opening solve order.
    ///
    /// Trains a policy with a non-identity solve order installed (multi-opening
    /// fixture), then verifies the simulation cost buffer is bit-identical across
    /// 1 vs 4 simulation workspaces. The reorder is a backward-pass-only setting;
    /// this exercises that a reorder-trained FCF still simulates invariantly across
    /// thread counts.
    #[test]
    fn test_simulation_determinism_across_thread_counts_with_reorder() {
        const N_ITERATIONS: u64 = 10;
        const N_SCENARIOS: u32 = 20;
        const BRANCHING: usize = 4;

        let mut fx = Fixture3H::with_branching(BRANCHING);
        let keys: Vec<Vec<f64>> = (0..fx.n_stages).map(|_| vec![3.0, 1.0, 4.0, 2.0]).collect();
        fx.stochastic
            .set_solve_order(&keys)
            .expect("solve-order key dims match the tree");

        let (_training_result, fcf) = run_training(1, &fx, N_ITERATIONS);

        let costs_1 = run_simulation(1, &fx, &fcf, N_SCENARIOS);
        let costs_4 = run_simulation(4, &fx, &fcf, N_SCENARIOS);

        assert_eq!(costs_1.len(), costs_4.len());
        for ((id_1, cost_1, cats_1), (id_4, cost_4, cats_4)) in costs_1.iter().zip(costs_4.iter()) {
            assert_eq!(id_1, id_4);
            assert_eq!(cost_1.to_bits(), cost_4.to_bits());
            assert_eq!(
                cats_1.resource_cost.to_bits(),
                cats_4.resource_cost.to_bits()
            );
            assert_eq!(
                cats_1.recourse_cost.to_bits(),
                cats_4.recourse_cost.to_bits()
            );
            assert_eq!(
                cats_1.violation_cost.to_bits(),
                cats_4.violation_cost.to_bits()
            );
            assert_eq!(
                cats_1.regularization_cost.to_bits(),
                cats_4.regularization_cost.to_bits()
            );
            assert_eq!(cats_1.imputed_cost.to_bits(), cats_4.imputed_cost.to_bits());
        }
    }

    /// Local mirror of the gated `test_support::all_enabled_cut_state_layouts`
    /// via the public `CutStateProjection::new`, so this external test crate (which cannot
    /// see the parent crate's `#[cfg(test)]` surface) builds the default all-enabled
    /// per-pool projection. Every pool projects the full global state, keeping the
    /// extracted subgradient bit-identical to the global-loop result.
    fn all_enabled_cut_state_layouts(
        global: &StateSpace,
        n_stages: usize,
    ) -> Vec<CutStateProjection> {
        let full = StageStateConfig {
            storage: true,
            inflow_lags: true,
        };
        (0..n_stages)
            .map(|_| CutStateProjection::new(global, full))
            .collect()
    }
}

mod water_travel_time_no_arc_byte_identity {
    //! With the water travel-time feature compiled in but no arc declared on any
    //! hydro, `n_buckets == 0` and the `StateSpace`/LP/cuts/outputs must collapse
    //! to the pre-bucket baseline byte-for-byte (the `n_buckets` == 0
    //! byte-identity anchor). This module makes that guarantee an explicit
    //! regression at two scales:
    //!
    //! - [`synthetic_no_arc_state_layout_matches_pre_transit_bucket_formula`] and
    //!   [`k1_chronological_byte_identical_to_parallel_with_no_arc_declared`]
    //!   build a tiny in-code system (no solver, no baseline) and check the
    //!   `StateSpace` dimensions and the `K = 1` chronological/parallel
    //!   templates directly.
    //! - [`d06_state_layout_matches_pre_transit_bucket_formula`] and
    //!   `d06_parity_hash_matches_existing_baseline_{highs,clp}` exercise a real
    //!   golden deterministic case (D06, which declares no arc): its
    //!   `StateSpace` and its full train+simulate parity hash, reusing
    //!   [`common::parity_hash::run_golden_case`](super::common::parity_hash::run_golden_case)
    //!   against the EXISTING committed baseline — no new baseline is written.

    use cobre_io::config::TrainingSelection;
    use std::path::{Path, PathBuf};

    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{BlockMode, Stage};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroGenerationModel, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_sddp::{
        StudySetup,
        hydro_models::prepare_hydro_models,
        indexer::StateSpace,
        setup::{StudyParams, prepare_stochastic},
    };
    use cobre_solver::StageTemplate;

    use cobre_sddp::test_support::assert_all_templates_byte_identical;

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    const N_STAGES: usize = 3;
    const HYDRO_ID: i32 = 1;

    fn zero_hydro_penalties() -> HydroPenalties {
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

    /// One bus, one standalone hydro (no `downstream_id`/`travel_time_hours` —
    /// no arc declared) with a backup thermal, `N_STAGES` stages each carrying a
    /// single default-length block (`StageSpec::default()`'s block: the `K = 1`
    /// case under test).
    fn build_system(block_mode: BlockMode) -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(2),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let hydro = make_hydro(
            EntityId(HYDRO_ID),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(2),
                min_storage_hm3: 0.0,
                max_storage_hm3: 500.0,
                max_turbined_m3s: 100.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                specific_productivity_mw_per_m3s_per_m: Some(0.5),
                max_generation_mw: 250.0,
                penalties: zero_hydro_penalties(),
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1)
                            .unwrap(),
                        season_id: Some(0),
                        block_mode,
                        ..StageSpec::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id: EntityId(HYDRO_ID),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_m3s: 60.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: EntityId(2),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_mw: 120.0,
                std_mw: 0.0,
            })
            .collect();

        let default_hydro_bounds = || HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let default_hydro_bounds_block = || HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_bounds_block(),
                thermal: ThermalStageBounds {
                    cost_per_mwh: 100.0,
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

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(HYDRO_ID),
                value_hm3: 200.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![make_thermal(
                EntityId(3),
                ThermalSpec {
                    name: "T_backup".to_string(),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(2),
                    cost_per_mwh: 100.0,
                    min_generation_mw: 0.0,
                    max_generation_mw: 400.0,
                    anticipated_config: None,
                    ..Default::default()
                },
            )])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid no-arc single-block study")
    }

    fn build_config() -> cobre_io::Config {
        use cobre_io::config::{
            Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
            InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
            RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
            TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
        };

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

    fn build_templates(block_mode: BlockMode) -> Vec<StageTemplate> {
        let system = build_system(block_mode);
        let config = build_config();
        let setup = build_setup_in_code(system, &config);
        setup.inputs.stage_data.stage_templates.templates.clone()
    }

    /// Shared pre-bucket-formula assertion: `n_buckets == 0`, `transit_buckets_out` /
    /// `transit_buckets_in` / `transit_bucket_column_order` empty, and `n_state` equal to the
    /// pre-bucket `N*(1+L) + A*k_max` — computed from the layout's OWN public
    /// dimensions (`hydro_count`, `max_par_order`, `n_anticipated`, `k_max`), not
    /// a hand-picked literal, so the check holds for any no-arc case.
    fn assert_no_arc_state_layout(state: &StateSpace) {
        assert_eq!(state.n_buckets, 0, "no arc declared: n_buckets must be 0");
        assert!(
            state.transit_buckets_out.is_empty(),
            "transit_buckets_out must be empty when n_buckets == 0"
        );
        assert!(
            state.transit_buckets_in.is_empty(),
            "transit_buckets_in must be empty when n_buckets == 0"
        );
        assert!(
            state.transit_bucket_column_order.is_empty(),
            "transit_bucket_column_order must be empty when n_buckets == 0"
        );

        let pre_transit_bucket_n_state =
            state.hydro_count * (1 + state.max_par_order) + state.n_anticipated * state.k_max;
        assert_eq!(
            state.n_state, pre_transit_bucket_n_state,
            "n_state must equal the pre-bucket formula N*(1+L) + A*k_max when B == 0"
        );
    }

    /// Synthetic no-arc system: `StateSpace` collapses to the pre-bucket
    /// formula with no `.sha256` baseline involved.
    #[test]
    fn synthetic_no_arc_state_layout_matches_pre_transit_bucket_formula() {
        let system = build_system(BlockMode::Parallel);
        let config = build_config();
        let setup = build_setup_in_code(system, &config);
        assert_no_arc_state_layout(setup.stage_state());
    }

    /// `K = 1` chronological build collapses to the parallel LP with travel
    /// time off: no in-transit arc is declared on the hydro, so — independent of
    /// the chronological/parallel structural claim itself — every stage
    /// template must be byte-identical between the two block modes.
    #[test]
    fn k1_chronological_byte_identical_to_parallel_with_no_arc_declared() {
        let parallel = build_templates(BlockMode::Parallel);
        let chronological = build_templates(BlockMode::Chronological);

        assert_all_templates_byte_identical(
            &parallel,
            &chronological,
            "parallel vs chronological (no arc declared)",
        );
    }

    /// D06 (`d06-fpha-variable-head`) is one of the pinned golden parity-hash
    /// cases (`common::parity_hash::case_dir`) and declares no travel-time arc;
    /// the directory suffix here duplicates that private mapping because it is
    /// not reachable from this module.
    fn d06_case_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d06-fpha-variable-head")
    }

    /// Build D06's `StudySetup` (mirrors `common::parity_hash::run_golden_case`'s
    /// construction) without training/simulating, so the caller can inspect the
    /// built `StateSpace` directly.
    fn build_d06_setup() -> StudySetup {
        let dir = d06_case_dir();
        let config_path = dir.join("config.json");

        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(&dir).expect("load_case must succeed");

        let prep_source = config
            .training_scenario_source(&config_path)
            .expect("training_scenario_source must parse");
        let pr = prepare_stochastic(system, &dir, &config, 42, &prep_source, None)
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

        let sentinel = Path::new("config.json");
        let training_source = config
            .training_scenario_source(sentinel)
            .expect("training_scenario_source must parse");
        let simulation_source = config
            .simulation_scenario_source(sentinel)
            .expect("simulation_scenario_source must parse");

        let params = StudyParams::from_config(&config, Vec::new())
            .expect("StudyParams::from_config must succeed");

        StudySetup::from_broadcast_params(
            &system,
            stochastic,
            params,
            hydro_models,
            &training_source,
            &simulation_source,
        )
        .expect("StudySetup must build")
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d06_state_layout_matches_pre_transit_bucket_formula() {
        let setup = build_d06_setup();
        assert_no_arc_state_layout(setup.stage_state());
    }

    /// Reuses [`common::parity_hash::run_golden_case`](super::common::parity_hash::run_golden_case)
    /// against the EXISTING committed D06 baseline — no new baseline is
    /// written; a mismatch here means the no-arc build is no longer
    /// byte-identical to the pre-feature baseline.
    #[cfg(feature = "highs")]
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d06_parity_hash_matches_existing_baseline_highs() {
        use cobre_solver::highs::HighsSolver;
        super::common::parity_hash::run_golden_case("parity_baselines", "D06", HighsSolver::new);
    }

    /// CLP counterpart of
    /// [`d06_parity_hash_matches_existing_baseline_highs`] against the
    /// EXISTING committed CLP baseline.
    #[cfg(feature = "clp")]
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d06_parity_hash_matches_existing_baseline_clp() {
        use cobre_solver::clp::ClpSolver;
        super::common::parity_hash::run_golden_case("parity_baselines_clp", "D06", ClpSolver::new);
    }
}

mod water_travel_time_gate_byte_neutrality {
    //! Byte-neutrality of two independent gates on the same DECLARED-ARC deck
    //! (D44, distinct from [`super::water_travel_time_no_arc_byte_identity`]'s
    //! no-arc D06).
    //!
    //! The first is the water-travel-time terminal keep-live gate when
    //! `config.policy.boundary` is absent: a gated-off study must reproduce
    //! `final_lb` bit-for-bit across two independent, freshly-constructed
    //! runs, and the gated-off state layout must keep every terminal deep-lag
    //! bucket slot masked exactly as the pre-keep-live layout — the "Terminal
    //! credit deferred" contract the gate must preserve when no boundary is
    //! loaded. The existing water goldens' own `.sha256` reproduction
    //! (`d06_parity_hash_matches_existing_baseline_{highs,clp}` above) is the
    //! companion evidence that no baseline moved; this module adds the
    //! run-to-run reproducibility and mask-invariance checks a golden hash
    //! alone does not pin.
    //!
    //! The second is the arrival-calendar gate: the topology sizing derived
    //! from a declared post-study calendar only diverges from today's
    //! synthetic pad when the deck actually declares one. This deck declares
    //! none, so that gate is structurally incapable of moving these goldens —
    //! asserted directly below rather than assumed.

    use std::path::Path;

    use cobre_io::config::BoundaryPolicy;
    use cobre_solver::ActiveSolver;

    use super::common::{StubComm, fresh_setup_with};

    fn case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d44-travel-time-substage")
    }

    fn train_gated_off() -> f64 {
        let mut setup = fresh_setup_with(&case_dir(), |_cfg| {});
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        outcome.result.final_lb
    }

    /// A water-travel-time study with no `config.policy.boundary` reproduces
    /// `final_lb` bit-for-bit across two independent, freshly constructed
    /// runs.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn gated_off_declared_arc_study_reproduces_final_lb_across_independent_runs() {
        let lb_a = train_gated_off();
        let lb_b = train_gated_off();
        assert_eq!(
            lb_a.to_bits(),
            lb_b.to_bits(),
            "two independent runs of a gated-off declared-arc water-travel-time study \
             must reproduce final_lb bit-for-bit (run A: {lb_a}, run B: {lb_b})"
        );
    }

    /// With no `config.policy.boundary`, `n_state` and the declared bucket
    /// column order stay gate-invariant, and every terminal bucket slot that
    /// is masked `[0, 0]` with the gate off stays live once
    /// `boundary_present` is true — proving the gated-off path stays
    /// byte-neutral.
    #[test]
    fn gated_off_declared_arc_study_keeps_the_terminal_mask_and_state_dimension() {
        let setup_off = fresh_setup_with(&case_dir(), |_cfg| {});
        let setup_on = fresh_setup_with(&case_dir(), |cfg| {
            cfg.policy.boundary = Some(BoundaryPolicy {
                path: "unused".to_string(),
                strict: false,
            });
        });

        let state_off = setup_off.stage_state();
        let state_on = setup_on.stage_state();
        assert!(
            state_off.n_buckets > 0,
            "fixture has no power unless it declares at least one travel-time bucket"
        );
        assert_eq!(
            state_off.n_state, state_on.n_state,
            "n_state must stay gate-invariant"
        );
        assert_eq!(
            state_off.transit_bucket_column_order, state_on.transit_bucket_column_order,
            "bucket column order/depth must stay gate-invariant"
        );

        let terminal_stage = setup_off.num_stages() - 1;
        let template_off = &setup_off.stage_ctx().templates[terminal_stage];
        let template_on = &setup_on.stage_ctx().templates[terminal_stage];

        let mut any_masked_off = false;
        for pos in 0..state_off.n_buckets {
            let col = state_off.transit_buckets_out.start + pos;
            if template_off.col_lower[col] == 0.0 && template_off.col_upper[col] == 0.0 {
                any_masked_off = true;
                assert!(
                    template_on.col_upper[col] > 0.0,
                    "bucket column {col} (pos {pos}) masked [0,0] with the gate off must be \
                     live once boundary_present is true"
                );
            }
        }
        assert!(
            any_masked_off,
            "fixture has no power unless at least one terminal bucket slot is masked \
             [0,0] with no boundary present"
        );
    }

    #[test]
    fn declared_arc_golden_deck_declares_no_post_study_calendar() {
        let system = cobre_io::load_case(&case_dir()).expect("load_case must succeed");
        assert!(
            system.post_study_stages().is_none(),
            "the water goldens' deck declares no post-study calendar, so the \
             arrival-calendar extension cannot move them"
        );
    }
}

mod sacred_chain_parity_roster {
    //! The ten chain-parity obligations and their break-one-obligation
    //! verification table. No executable code — a module doc only, the
    //! auditable index every chain-parity correctness claim (absent
    //! `nodes[]`) resolves to. Mirrors `tests/mpi_wire.rs`'s
    //! `branching_gate_roster` in shape and intent; the two rosters are
    //! disjoint (see the closing section below).
    //!
    //! # The roster — obligation → named test → introduced alongside
    //!
    //! | # | Obligation | Named test | File | Introduced alongside |
    //! |---|---|---|---|---|
    //! | 1 | Pool indexing on a chain (`pool_id == stage`) | `pool_stage_chain_is_identity` | `setup/node_graph.rs` | the pool re-key to pool-id addressing |
    //! | 2 | Draw-sequence tuples pinned at sampler level (no test-only env vars) | `transition_draw_call_does_not_perturb_subsequent_within_node_noise` | `cobre-stochastic` `sampling/class_sampler.rs` | the sampled root-to-leaf graph walk |
    //! | 3 | Per-pool cut append order (checkpoint bytes) | `read_policy_checkpoint_full_round_trip` | `cobre-io` `output/policy/mod.rs` | the versioned value-function-artifact checkpoint schema |
    //! | 4 | Basis addressing + constant node tag | `opening_order_determinism` | `mpi_wire.rs` | basis node-tagging (cross-node warm-start rejection) |
    //! | 5 | Full golden-case output parity, both backends | `parity_hash_d06`/`d15`/`d30`/`d34`/`d41` (`parity_hash_highs` and `parity_hash_clp`) | `parity.rs` | pre-existing; the harness was re-keyed by pool id when pools became node-indexed |
    //! | 6 | The existing `mpi_wire.rs` gates | `opening_order_determinism`, `by_node_scheduler_determinism_expectation`, `by_node_scheduler_determinism_cvar`, `hardest_first_claim_order_is_result_neutral`, `retry_armed_determinism_expectation`, `retry_armed_determinism_cvar`, `derived_inflow_seeds_rank_invariant`, `four_rank_basis_broadcast_round_trip`, `k_fan_thread_shape_invariance`, `k_fan_weighted_aggregation_canonical_order_invariance`, `enumerated_k_fan_thread_and_declaration_shapes_agree` | `mpi_wire.rs` | various (see each gate's own module doc) |
    //! | 7 | CVaR `(outcomes, probabilities)` index-order identity | `cvar_aggregation_tie_break_follows_canonical_child_order` | `training/backward_pass_state.rs` | threading successor product weights through backward cut aggregation |
    //! | 8 | Uniform-weight bit pattern `1.0/(n as f64)`, left-to-right reduction | `aggregate_simulation_uniform_mean_matches_left_to_right_per_term_weighted_sum` | `mpi_wire.rs` | `aggregate_simulation`'s formula predates this pin; the dedicated bit-pattern-plus-reduction gate is added here |
    //! | 9 | Unchanged pool capacities on chains | `pool_cut_stride_chain_matches_forward_passes_over_a_sweep` | `setup/node_graph.rs` | per-node pool capacity replacing the flat per-stage formula |
    //! | 10 | Checkpoint round-trip | `d12_checkpoint_round_trip` | `deterministic.rs` | pre-existing (the D-series deterministic suite) |
    //!
    //! Item 8 is the one addition this change makes: the pre-existing
    //! `aggregate_uniform_mean_matches_risk_measure_expectation`
    //! (`simulation/aggregation.rs`) exercises the same formula but its
    //! literal costs (`[100.0, 200.0, 150.0]`, `n = 3`) round to the
    //! identical `f64` bit pattern under sum-then-divide by coincidence —
    //! documented as a coverage hole in `mpi_wire.rs`'s own
    //! `branching_gate_roster` (row d) for the branching suite, and
    //! independently reconfirmed below for the chain suite. The new gate's
    //! literals are chosen so the two formulas provably diverge (self-checked
    //! in the test body) — it closes the hole rather than duplicating the
    //! existing test's blind spot. Every other item resolves to an existing
    //! named test; none needed a second addition.
    //!
    //! # Break-one-obligation verification (real, observed results)
    //!
    //! Each row's obligation was broken with a single scratch mutation to
    //! production code, the mapped test (plus enough of the surrounding
    //! suite to bound the blast radius) was run against the mutated binary,
    //! the observed pass/fail outcome was recorded below, and the edit was
    //! reverted before the next row. No row is recorded without having been
    //! run.
    //!
    //! | # | Scratch mutation | Observed result |
    //! |---|---|---|
    //! | 1 | `setup/node_graph.rs`: `build_chain_node_graph`'s `let pool_id = t;` changed to `n_stages - 1 - t` (reversed) | **FAILS** 22 tests across `cobre-sddp --lib`, including the mapped `pool_stage_chain_is_identity` and `chain_degeneracy_one_node_per_stage_1to1_pools_uniform_q_bit_pattern` — pool-id identity is a deeply load-bearing invariant with broad existing coverage (over-determined, not a hole). |
    //! | 2 | `cobre-stochastic` `sampling/class_sampler.rs`: introduced a shared `static AtomicU64` call counter, XORed into both `select_transition_child`'s seed and `ClassSampler::fill`'s `InSample` seed (simulating a stateful, call-order-dependent draw) | **FAILS** exactly the mapped test within `class_sampler`'s own 29-test module (28 pass, 1 fails), plus 2 more at the crate level (`sampling::tests::test_composite_in_sample_fills_correct_segments`, `sampling::tests::test_in_sample_sample_is_deterministic`) that also exercise `InSample::fill` repeatability — over-determined, not a hole. The current architecture has no shared mutable state between the two draws (each independently re-derives a pure-function seed and a fresh RNG), so this mutation had to introduce the state the obligation forbids rather than merely reorder existing code. |
    //! | 3 | `cobre-io` `output/policy/codec.rs`: `deserialize_stage_cuts`'s cut-vector read loop changed from `nested_positions.iter().enumerate()` to `.iter().rev().enumerate()` | **FAILS** exactly 2 tests in `cobre-io --lib`: the mapped `read_policy_checkpoint_full_round_trip` and the lower-level `deserialize_stage_cuts_three_cuts_all_match` — over-determined, not a hole. |
    //! | 4 | `training/forward/basis_capture.rs`: `write_capture_metadata`'s `captured.node_id = node_id;` hardcoded to `NodeId(0)` (every capture mistagged to node 0 regardless of the real node being solved) | **COVERAGE HOLE.** The mapped `opening_order_determinism`, all 39 other `mpi_wire.rs` tests, `parity_hash_d06`, and `d12_checkpoint_round_trip` all stayed green. On a chain the mismatch (`0 != t` for every `t > 0`) forces every non-root basis capture to cold-start — uniformly and deterministically, so it is bit-reproducible across every thread/rank shape these gates compare (a shape-invariance gate cannot see a shape-invariant defect), and the D02/D06 LPs these fixtures use converge to the same unique optimal vertex regardless of warm- or cold-start, so even the final numeric output is unmoved (the parity hash deliberately excludes iteration counts, so a hot-vs-cold difference in convergence speed alone would not move it either). `run_stage_solve_cross_node_stored_basis_is_treated_as_cold` (`solve/stage_solve.rs`) proves the reject mechanism fires correctly given an already-mismatched tag, but nothing in the suite proves `write_capture_metadata` populates the tag correctly from the real node in the first place, on an end-to-end chain run. Reported, not papered over. |
    //! | 5 | `simulation/extraction.rs`: the per-block `SimulationHydroResult` builder's `storage_final_hm3: storage_final,` changed to `storage_final + 1e-6` | **FAILS** `parity_hash_d06`/`d30`/`d34`/`d41` (D15 unaffected — its fixture has no hydro storage exercising this field) plus `water_travel_time_no_arc_byte_identity::d06_parity_hash_matches_existing_baseline_highs`, a second D06-hash-checking test — over-determined, not a hole. `d12_checkpoint_round_trip` (item 10) stays green under the same mutation (its cost comparison tolerance, 1e-2, absorbs a 1e-6 perturbation) — confirming items 5 and 10 are genuinely independent obligations, not accidental duplicates. |
    //! | 6 | Three independent mutations tried against the item-6 roster: (a) the same seed perturbation as row 8 below applied to `cobre-stochastic`'s `derive_inflow_seeds`; (b) and (c) `training/forward/stats_aggregation.rs`'s `weighted_cost_reduction` reduction loop reversed (twice, isolating the K-fan aggregation path) | **COVERAGE HOLE for the roster as a whole.** All 40 `mpi_wire.rs` tests stayed green under every one of the three mutations. This is a structural property, not a fluke: every named gate in item 6 tests invariance across thread/rank/claim shape, which is orthogonal to value correctness — a uniformly-wrong-but-deterministic formula or order change produces the identical wrong value under every shape, so shape-comparison gates cannot see it by design (several backward-pass claim orderings are explicitly claim-order-neutral by contract — e.g. hardest-first, pinned by `hardest_first_claim_order_is_result_neutral`). Mutation (a) WAS caught elsewhere in the suite (`d16_par1_lag_shift`, `deterministic.rs`, a value-pinned behavioral-tier test) and mutation (b)/(c) reproduces the exact coverage hole `mpi_wire.rs`'s own `branching_gate_roster` row (c) already documents for the branching suite (Neumaier-compensated summation over these fixtures' literals happens to reorder to the same bit pattern). Reported, not papered over; the system as a whole has power, the item-6 roster alone does not. |
    //! | 7 | `setup/node_graph.rs`: `assemble_outcome_weights`'s `for succ in successors` changed to `.iter().rev()` | **FAILS** 5 tests in `cobre-sddp --lib`, including the mapped `cvar_aggregation_tie_break_follows_canonical_child_order` and its sibling at the shared owner's other call site (`assemble_successor_outcome_weights_k_fan_canonical_order_and_product_weights`, `assemble_outcome_weights_k_fan_canonical_order_and_product_weights`, `root_outcome_weights_cvar_matches_evaluate_risk`, `root_outcome_weights_expectation_matches_analytical_sum`) — over-determined, not a hole; `assemble_outcome_weights` is deliberately the single owner both call sites delegate to. |
    //! | 8 | `simulation/aggregation.rs`: `aggregate_simulation`'s `mean_cost` changed from `RiskMeasure::Expectation.evaluate_risk(&cost_recv, &weights)` to `cost_recv.iter().sum::<f64>() / n as f64` (sum-then-divide) | **FAILS** exactly the mapped (new) test, 39/40 other `mpi_wire.rs` tests stay green, and `parity_hash_d06`/`d15`/`d30`/`d34`/`d41` plus `d12_checkpoint_round_trip` are unaffected (the golden hash excludes simulation summary statistics; `d12`'s tolerance absorbs the difference). Confirms the new test closes a real, previously-undetectable gap rather than duplicating existing coverage. |
    //! | 9 | `cut/fcf.rs`: `pool_capacity`'s `warm_start_count + max_iterations * visit_bound` changed to `warm_start_count + 1 + max_iterations * visit_bound` | **FAILS** 8 tests in `cobre-sddp --lib`, including the mapped `pool_cut_stride_chain_matches_forward_passes_over_a_sweep` (whose own assertion message states the chain-parity contract verbatim: "chain capacity must equal warm_start + max_iterations * forward_passes exactly") and its `cut::fcf` siblings — over-determined, not a hole. |
    //! | 10 | `cobre-io` `output/policy/checkpoint.rs`: `bin_file_name`'s `format!("{id:03}.bin")` changed to 4-digit padding | **FAILS** exactly the mapped `d12_checkpoint_round_trip` in `deterministic.rs` (99 other tests unaffected) plus 4 sibling file-naming assertions in `cobre-io --lib` — over-determined, not a hole. The reader itself is unaffected (`read_sorted_bin_files` globs `*.bin` and derives identity from inside each buffer, never from the name), confirming the doc comment's own claim. |
    //!
    //! Every mutation above was reverted immediately after being run; `git
    //! diff` on each touched production file was confirmed empty before
    //! moving to the next row. No scratch mutation survives in the tree.
    //!
    //! # The two branching gates (not among the ten)
    //!
    //! Two branching-graph gates were deliberately deferred to land alongside
    //! the branching-engine work rather than here, and are cited by name
    //! below rather than re-implemented:
    //!
    //! - **By-node scheduler equivalence on the branching graph** —
    //!   consolidated in `mpi_wire.rs`'s `branching_gate_roster` under
    //!   "By-node-on-branching equivalence"
    //!   (`by_node_k_fan_thread_shape_invariance`,
    //!   `interior_sibling_generated_fan_by_node_matches_oracle`,
    //!   `water_binding_external_fan_by_node_matches_extensive_form`,
    //!   `external_distinct_fan_by_node_matches_by_scenario`).
    //! - **Real 2-rank D-2 rank-invariance on a branching graph** — the test
    //!   is
    //!   `k_fan_branching_rank_invariance::k_fan_final_lb_bitwise_invariant_across_world_size`
    //!   in `tests/test_mpi_sync_cuts_invariant.rs`, cited in
    //!   `branching_gate_roster` under "Rank-shape: genuine 2-rank real MPI".
    //! - **Both are consolidated** in `mpi_wire.rs`'s `branching_gate_roster`
    //!   module (value + invariance, both schedulers, uniform and
    //!   non-uniform cut-state axis, single- and multi-rank), verified to
    //!   exist by inspecting the live tree rather than re-derived here.
    //!
    //! Neither gate is among the ten (all ten are chain obligations, absent
    //! `nodes[]`); neither is an eleventh or twelfth item.
}
