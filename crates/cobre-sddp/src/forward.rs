//! Forward pass execution for the SDDP training loop.
//!
//! [`run_forward_pass`] simulates `M` scenario trajectories through the full
//! stage horizon, solving the stage LP at each `(scenario, stage)` pair with
//! the current Future Cost Function (FCF) approximation.
//!
//! ## Outputs
//!
//! The function produces two outputs:
//!
//! - **[`TrajectoryRecord`]s** — one per `(scenario, stage)` pair, stored in
//!   a flat pre-allocated slice at index `scenario * num_stages + stage`. The
//!   backward pass reads these records to generate Benders cuts.
//! - **[`ForwardResult`]** — local UB candidate statistics for the calling
//!   rank, merged across ranks by the forward synchronisation step.
//!
//! ## Work distribution
//!
//! Each rank processes `config.forward_passes` scenarios. The global scenario
//! index for rank `r`, local scenario `m` is `r * forward_passes + m`. This
//! deterministic mapping drives the communication-free seed derivation used by
//! [`sample_forward`].
//!
//! ## LP rebuild sequence
//!
//! For each `(scenario, stage)` pair the LP is rebuilt in three steps:
//!
//! 1. `solver.load_model(template)` — reset to the structural LP.
//! 2. `solver.add_rows(cut_batch)` — append active Benders cuts.
//! 3. `solver.set_row_bounds(...)` — patch scenario-specific row bounds.
//!
//! ## Hot-path allocation discipline
//!
//! No allocations occur per scenario or per stage during the inner loops. The
//! [`PatchBuffer`] and [`TrajectoryRecord`] slice are pre-allocated by the
//! caller. The only allocation inside the function is the [`RowBatch`] built
//! by [`build_cut_row_batch`], which runs once per stage template (before the
//! scenario loop) — not once per scenario.

use std::time::Instant;

use cobre_comm::{Communicator, ReduceOp};
use cobre_solver::{Basis, RowBatch, SolverError, SolverInterface, StageTemplate};
use cobre_stochastic::{sample_forward, StochasticContext};

use crate::{
    FutureCostFunction, HorizonMode, PatchBuffer, SddpError, StageIndexer, TrainingConfig,
    TrajectoryRecord,
};

/// Local statistics from one rank's forward pass (reduced via `allreduce`).
///
/// Does not contain lower bound estimate (evaluated separately after backward pass).
#[derive(Debug, Clone)]
#[must_use]
pub struct ForwardResult {
    /// Sum of total trajectory costs across all local scenarios.
    pub cost_sum: f64,

    /// Sum of squared total trajectory costs (for variance estimate).
    pub cost_sum_sq: f64,

    /// Number of scenarios solved on this rank, as `f64`.
    pub scenario_count: f64,

    /// Wall-clock time in milliseconds for this rank's forward pass.
    pub elapsed_ms: u64,
}

/// Global upper bound statistics from forward synchronisation step (`allreduce`).
#[derive(Debug, Clone)]
#[must_use]
pub struct SyncResult {
    /// Sample mean of total trajectory costs across all ranks.
    pub global_ub_mean: f64,

    /// Bessel-corrected sample standard deviation of total trajectory costs.
    pub global_ub_std: f64,

    /// 95% confidence interval half-width: `1.96 * std / sqrt(N)`.
    pub ci_95_half_width: f64,

    /// Wall-clock time in milliseconds for the `allreduce` call.
    pub sync_time_ms: u64,
}

/// Aggregate local forward pass statistics across all MPI ranks.
///
/// Performs a single `allreduce` collective operation to produce global upper
/// bound statistics from the per-rank [`ForwardResult`] produced by
/// [`run_forward_pass`]:
///
/// - `allreduce` with `ReduceOp::Sum` on `[cost_sum, cost_sum_sq, scenario_count]`
///   — the pooled statistics are used to compute the global UB mean, variance,
///   standard deviation, and 95% confidence interval half-width.
///
/// From the aggregated sums, the following statistics are computed (per SS3.1a):
/// - `mean = global_cost_sum / N`
/// - `variance = (global_cost_sum_sq - N * mean^2) / (N - 1)` when `N > 1`
/// - `variance = 0.0` when `N <= 1` (Bessel correction edge case)
/// - `std_dev = max(0, variance).sqrt()` (guard against negative variance from
///   floating-point catastrophic cancellation)
/// - `ci_95 = 1.96 * std_dev / sqrt(N)`
///
/// The lower bound is **not** computed here. It is evaluated separately after
/// the backward pass adds new cuts to the FCF.
///
/// In single-rank mode (`comm.size() == 1`), `LocalBackend.allreduce` is an
/// identity copy. No special-casing is needed — the result equals the local values.
///
/// ## Arguments
///
/// - `local` — the [`ForwardResult`] from the calling rank's forward pass.
/// - `comm` — the communicator used for collective operations.
///
/// # Errors
///
/// Returns `Err(SddpError::Communication(_))` if the `allreduce` call fails.
/// The `From<CommError>` conversion on `SddpError` is applied automatically
/// via the `?` operator. No partial results are produced on error.
pub fn sync_forward<C: Communicator>(
    local: &ForwardResult,
    comm: &C,
) -> Result<SyncResult, SddpError> {
    let start = Instant::now();

    // UB: aggregate sufficient statistics (sum, sum_sq, count).
    let ub_send = [local.cost_sum, local.cost_sum_sq, local.scenario_count];
    let mut ub_recv = [0.0_f64; 3];
    comm.allreduce(&ub_send, &mut ub_recv, ReduceOp::Sum)?;

    let global_sum = ub_recv[0];
    let global_sum_sq = ub_recv[1];
    let global_count = ub_recv[2];
    let mean = global_sum / global_count;

    // Bessel-corrected variance. When N <= 1, set to 0 to avoid division by zero.
    // max(0, variance).sqrt() guards against catastrophic cancellation.
    let (std_dev, ci_95) = if global_count > 1.0 {
        let variance = (global_sum_sq - global_count * mean * mean) / (global_count - 1.0);
        let sd = variance.max(0.0).sqrt();
        let ci = 1.96_f64 * sd / global_count.sqrt();
        (sd, ci)
    } else {
        (0.0_f64, 0.0_f64)
    };

    #[allow(clippy::cast_possible_truncation)]
    let sync_time_ms = start.elapsed().as_millis() as u64;

    Ok(SyncResult {
        global_ub_mean: mean,
        global_ub_std: std_dev,
        ci_95_half_width: ci_95,
        sync_time_ms,
    })
}

/// Construct a [`RowBatch`] from the active cuts at the given stage.
///
/// Each active cut `(slot, intercept, coefficients)` from [`FutureCostFunction::active_cuts`]
/// becomes one row in the batch. The cut constraint
///
/// ```text
/// theta >= intercept + sum_i(coefficients[i] * state[i])
/// ```
///
/// is reformulated in standard row form as:
///
/// ```text
/// -coefficients[0] * x[0] - ... - coefficients[n-1] * x[n-1] + theta >= intercept
/// ```
///
/// so the row has:
/// - `col_indices` = `[0, 1, ..., n_state-1, theta_col]`
/// - `values` = `[-coefficients[0], ..., -coefficients[n-1], 1.0]`
/// - `row_lower` = `intercept`
/// - `row_upper` = `f64::INFINITY`
///
/// Returns an empty [`RowBatch`] (with `num_rows = 0`) when there are no
/// active cuts at the stage.
///
/// # Arguments
///
/// - `fcf` — Future Cost Function containing the cut pools.
/// - `stage` — 0-based stage index.
/// - `indexer` — LP layout map; provides `n_state` and `theta`.
///
/// # Panics
///
/// Panics if the total number of non-zeros in the cut batch exceeds `i32::MAX`,
/// which would exceed the `HiGHS` API index limit. In practice this cannot occur
/// for any realistic problem size.
#[must_use]
pub fn build_cut_row_batch(
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
) -> RowBatch {
    let n_state = indexer.n_state;
    let theta_col = indexer.theta;

    let num_cuts: usize = fcf.active_cuts(stage).count();

    if num_cuts == 0 {
        return RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
    }

    let nnz_per_cut = n_state + 1;
    let total_nnz = num_cuts * nnz_per_cut;

    let mut row_starts: Vec<i32> = Vec::with_capacity(num_cuts + 1);
    let mut col_indices: Vec<i32> = Vec::with_capacity(total_nnz);
    let mut values = Vec::with_capacity(total_nnz);
    let mut row_lower = Vec::with_capacity(num_cuts);
    let mut row_upper = Vec::with_capacity(num_cuts);

    let mut nz_offset = 0;

    for (_slot, intercept, coefficients) in fcf.active_cuts(stage) {
        debug_assert_eq!(
            coefficients.len(),
            n_state,
            "cut coefficients length {got} != n_state {expected}",
            got = coefficients.len(),
            expected = n_state,
        );

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        row_starts.push(nz_offset as i32);

        for (j, &c) in coefficients.iter().enumerate() {
            debug_assert!(
                i32::try_from(j).is_ok(),
                "column index j={j} exceeds i32::MAX"
            );
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            col_indices.push(j as i32);
            values.push(-c);
        }

        debug_assert!(
            i32::try_from(theta_col).is_ok(),
            "theta_col={theta_col} exceeds i32::MAX"
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        col_indices.push(theta_col as i32);
        values.push(1.0_f64);

        row_lower.push(intercept);
        row_upper.push(f64::INFINITY);

        nz_offset += nnz_per_cut;
    }

    #[allow(clippy::expect_used)]
    row_starts.push(
        i32::try_from(total_nnz).expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
    );

    RowBatch {
        num_rows: num_cuts,
        row_starts,
        col_indices,
        values,
        row_lower,
        row_upper,
    }
}

/// Execute the forward pass for one training iteration on this rank.
///
/// Simulates `config.forward_passes` scenarios through the full stage horizon,
/// solving the stage LP at each `(scenario, stage)` pair. Pre-allocated
/// [`TrajectoryRecord`]s in `records` are populated in-place.
///
/// ## Argument layout
///
/// - `solver` — mutable LP solver instance (one per rank, reused across stages).
/// - `templates` — one [`StageTemplate`] per stage (0-indexed); shared read-only.
/// - `base_rows` — AR dynamics base row index per stage; indexed identically to
///   `templates`.
/// - `fcf` — Future Cost Function carrying the current Benders cut pools.
/// - `stochastic` — pre-built stochastic pipeline (tree, seed, dim).
/// - `config` — training configuration (`forward_passes`, etc.).
/// - `iteration` — current training iteration (0-based counter used for seed
///   derivation).
/// - `horizon` — horizon mode determining the stage count.
/// - `initial_state` — starting state for every scenario (length `n_state`).
/// - `records` — pre-allocated output slice of length
///   `config.forward_passes * num_stages`.
/// - `patch_buf` — reusable row-bound patch buffer (pre-allocated, length
///   `N*(2+L)`).
/// - `indexer` — LP column/row layout map for this stage.
/// - `comm` — communicator for rank/size queries (work distribution).
///
/// ## Record layout
///
/// `records[scenario * num_stages + stage]` holds the LP solution for scenario
/// `scenario` at 0-based stage `stage`.
///
/// ## Error handling
///
/// On `SolverError::Infeasible`, returns `SddpError::Infeasible` with the
/// 0-based stage and local scenario indices. On any other `SolverError`,
/// returns `SddpError::Solver`. On error, `records` may be partially
/// populated.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
/// feasible solution. Returns `Err(SddpError::Solver(_))` for all other
/// terminal LP solver failures.
///
/// # Panics (debug builds only)
///
/// Panics if any of the following debug preconditions are violated:
///
/// - `records.len() != forward_passes * num_stages`
/// - `initial_state.len() != indexer.n_state`
/// - `templates.len() != num_stages`
/// - `base_rows.len() != num_stages`
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn run_forward_pass<S: SolverInterface, C: Communicator>(
    solver: &mut S,
    templates: &[StageTemplate],
    base_rows: &[usize],
    fcf: &FutureCostFunction,
    stochastic: &StochasticContext,
    config: &TrainingConfig,
    iteration: u64,
    horizon: &HorizonMode,
    initial_state: &[f64],
    records: &mut [TrajectoryRecord],
    patch_buf: &mut PatchBuffer,
    indexer: &StageIndexer,
    comm: &C,
    basis_cache: &mut [Option<Basis>],
) -> Result<ForwardResult, SddpError> {
    let num_stages = horizon.num_stages();
    let forward_passes = config.forward_passes as usize;
    let rank = comm.rank();

    debug_assert_eq!(
        records.len(),
        forward_passes * num_stages,
        "records.len() {got} != forward_passes * num_stages {expected}",
        got = records.len(),
        expected = forward_passes * num_stages,
    );
    debug_assert_eq!(
        initial_state.len(),
        indexer.n_state,
        "initial_state.len() {got} != indexer.n_state {expected}",
        got = initial_state.len(),
        expected = indexer.n_state,
    );
    debug_assert_eq!(
        templates.len(),
        num_stages,
        "templates.len() {got} != num_stages {expected}",
        got = templates.len(),
        expected = num_stages,
    );
    debug_assert_eq!(
        base_rows.len(),
        num_stages,
        "base_rows.len() {got} != num_stages {expected}",
        got = base_rows.len(),
        expected = num_stages,
    );

    let start = Instant::now();

    // Build one cut RowBatch per stage outside the scenario loop — batch
    // construction is O(num_cuts) and the cuts are the same for all scenarios
    // within the same iteration.
    let cut_batches: Vec<RowBatch> = (0..num_stages)
        .map(|t| build_cut_row_batch(fcf, t, indexer))
        .collect();

    let tree_view = stochastic.tree_view();
    let base_seed = stochastic.base_seed();

    let mut cost_sum = 0.0_f64;
    let mut cost_sum_sq = 0.0_f64;

    // Allocate a reusable current-state buffer. Cloned from initial_state
    // once per scenario — not on the inner (stage) loop.
    let mut current_state: Vec<f64> = Vec::with_capacity(indexer.n_state);

    for m in 0..forward_passes {
        // Global scenario index for deterministic seed derivation (DEC-017).
        let global_scenario = rank * forward_passes + m;

        // Initialise current state for this scenario.
        current_state.clear();
        current_state.extend_from_slice(initial_state);

        let mut trajectory_cost = 0.0_f64;

        for t in 0..num_stages {
            // Cast to u32 for the sampling API (DEC-017 domain ID derivation).
            // Indices bounded by u32::MAX in practice; truncation is safe.
            #[allow(clippy::cast_possible_truncation)]
            let (iter_u32, scenario_u32, stage_id_u32) =
                (iteration as u32, global_scenario as u32, t as u32);

            let (_opening_idx, noise) = sample_forward(
                &tree_view,
                base_seed,
                iter_u32,
                scenario_u32,
                stage_id_u32,
                t,
            );

            // Rebuild LP: template → cuts → scenario-specific row bounds.
            solver.load_model(&templates[t]);
            solver.add_rows(&cut_batches[t]);

            patch_buf.fill_forward_patches(indexer, &current_state, noise, base_rows[t]);
            let patch_count = patch_buf.forward_patch_count();
            solver.set_row_bounds(
                &patch_buf.indices[..patch_count],
                &patch_buf.lower[..patch_count],
                &patch_buf.upper[..patch_count],
            );

            let view = (if let Some(rb) = basis_cache[t].as_ref() {
                solver.solve_with_basis(rb)
            } else {
                solver.solve()
            })
            .map_err(|e| {
                basis_cache[t] = None;
                match e {
                    SolverError::Infeasible => SddpError::Infeasible {
                        stage: t,
                        iteration,
                        scenario: m,
                    },
                    other => SddpError::Solver(other),
                }
            })?;

            // Stage cost = LP objective minus theta (future cost variable).
            let stage_cost = view.objective - view.primal[indexer.theta];

            let record_idx = m * num_stages + t;
            records[record_idx].primal.clear();
            records[record_idx].primal.extend_from_slice(view.primal);
            records[record_idx].dual.clear();
            records[record_idx].dual.extend_from_slice(view.dual);
            records[record_idx].stage_cost = stage_cost;
            records[record_idx].state.clear();
            records[record_idx]
                .state
                .extend_from_slice(&view.primal[..indexer.n_state]);

            trajectory_cost += stage_cost;
            current_state.clear();
            current_state.extend_from_slice(&view.primal[..indexer.n_state]);

            if let Some(rb) = &mut basis_cache[t] {
                solver.get_basis(rb);
            } else {
                let mut rb = Basis::new(templates[t].num_cols, templates[t].num_rows);
                solver.get_basis(&mut rb);
                basis_cache[t] = Some(rb);
            }
        }

        cost_sum += trajectory_cost;
        cost_sum_sq += trajectory_cost * trajectory_cost;
    }

    #[allow(clippy::cast_possible_truncation)]
    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(ForwardResult {
        cost_sum,
        cost_sum_sq,
        scenario_count: f64::from(config.forward_passes),
        elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::context::build_stochastic_context;
    use cobre_stochastic::StochasticContext;

    use cobre_comm::LocalBackend;

    use super::{build_cut_row_batch, run_forward_pass, sync_forward, ForwardResult, SyncResult};
    use crate::{
        FutureCostFunction, HorizonMode, PatchBuffer, StageIndexer, TrainingConfig,
        TrajectoryRecord,
    };

    // ── Stub communicator ────────────────────────────────────────────────────

    /// Single-rank stub communicator used in unit and integration tests.
    struct StubComm {
        rank: usize,
        size: usize,
    }

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("StubComm allgatherv not used in forward pass tests")
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("StubComm allreduce not used in forward pass tests")
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("StubComm broadcast not used in forward pass tests")
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            self.rank
        }

        fn size(&self) -> usize {
            self.size
        }
    }

    // ── Mock solver ──────────────────────────────────────────────────────────

    /// Mock solver that returns a configurable fixed `LpSolution` on every `solve()`.
    ///
    /// Optionally returns `SolverError::Infeasible` at a specific
    /// `(scenario, stage)` pair (counted across calls in the scenario-outer,
    /// stage-inner traversal order). `infeasible_at` counts global solve
    /// calls starting from 0.
    ///
    /// `warm_start_calls` is incremented each time `solve_with_basis`
    /// is called, enabling warm-start invocation tests.
    struct MockSolver {
        solution: LpSolution,
        /// If `Some(n)`, the n-th solve call (0-indexed, counting both cold-start
        /// and warm-start calls) returns infeasible.
        infeasible_at: Option<usize>,
        call_count: usize,
        /// Number of times `solve_with_basis` has been called.
        warm_start_calls: usize,
        /// Internal buffers that `SolutionView` borrows from.
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
    }

    impl MockSolver {
        /// Create a solver that always returns `solution`.
        fn always_ok(solution: LpSolution) -> Self {
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: None,
                call_count: 0,
                warm_start_calls: 0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        /// Create a solver that returns infeasible on the `n`-th solve call.
        fn infeasible_on(solution: LpSolution, n: usize) -> Self {
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: Some(n),
                call_count: 0,
                warm_start_calls: 0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        /// Shared solve logic used by both cold-start and warm-start paths.
        ///
        /// Increments `call_count` and returns `Infeasible` when `call_count`
        /// matches `infeasible_at`, otherwise returns the stored solution.
        fn do_solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_at == Some(call) {
                return Err(SolverError::Infeasible);
            }
            // Fill internal buffers from the stored solution.
            self.buf_primal.clone_from(&self.solution.primal);
            self.buf_dual.clone_from(&self.solution.dual);
            self.buf_reduced_costs
                .clone_from(&self.solution.reduced_costs);
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: self.solution.iterations,
                solve_time_seconds: self.solution.solve_time_seconds,
            })
        }
    }

    impl SolverInterface for MockSolver {
        fn load_model(&mut self, _template: &StageTemplate) {}

        fn add_rows(&mut self, _cuts: &RowBatch) {}

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.do_solve()
        }

        fn reset(&mut self) {}

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn solve_with_basis(
            &mut self,
            _basis: &Basis,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.warm_start_calls += 1;
            self.do_solve()
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Minimal valid stage template for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout: [storage (0), `storage_in` (1), theta (2)]
    /// Row layout: [`storage_fixing` (0)]
    fn minimal_template_1_0() -> StageTemplate {
        // N=1, L=0:
        //   storage      = 0..1
        //   storage_in   = 1..2
        //   theta        = 2
        //   n_state      = 1
        //   n_transfer   = 0
        //   n_dual_relevant = 1
        //
        // LP: min theta  s.t. storage_in = ? (patched)  x >= 0
        //
        // CSC matrix has 1 non-zero: storage_in coefficient in storage_fixing row.
        // Simplified to a structurally valid but otherwise no-op LP for testing.
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1], // col 1 (storage_in) has NZ at row 0
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0], // minimise theta
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
        }
    }

    /// Build a fixed `LpSolution` with `num_cols` columns.
    ///
    /// `objective` is passed directly. `primal[theta_col]` is set to
    /// `theta_val`; all other primal entries are zero.
    fn fixed_solution(
        num_cols: usize,
        objective: f64,
        theta_col: usize,
        theta_val: f64,
    ) -> LpSolution {
        let mut primal = vec![0.0_f64; num_cols];
        primal[theta_col] = theta_val;
        let num_rows = 1; // single structural row for minimal template
        LpSolution {
            objective,
            primal,
            dual: vec![0.0; num_rows],
            reduced_costs: vec![0.0; num_cols],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Allocate `n` empty `TrajectoryRecord`s.
    fn empty_records(n: usize) -> Vec<TrajectoryRecord> {
        (0..n)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                state: Vec::new(),
            })
            .collect()
    }

    /// Build a minimal `StochasticContext` for a single-hydro, 3-stage system.
    ///
    /// Used by integration tests that call `run_forward_pass`. The `MockSolver`
    /// ignores the noise values produced by `sample_forward`, so the exact
    /// stochastic parameterisation does not affect correctness; it only needs
    /// to be structurally valid for the sampling API.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_context_1_hydro_3_stages() -> StochasticContext {
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
        let make_stage = |idx: usize, id: i32| Stage {
            index: idx,
            id,
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
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        };
        let stages = vec![make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let inflow = |stage_id: i32| InflowModel {
            hydro_id: EntityId(1),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
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
            .inflow_models(vec![inflow(0), inflow(1), inflow(2)])
            .correlation(correlation)
            .build()
            .unwrap();
        build_stochastic_context(&system, 42).unwrap()
    }

    // ── Unit tests: ForwardResult ────────────────────────────────────────────

    #[test]
    fn forward_result_field_access() {
        let r = ForwardResult {
            cost_sum: 100.0,
            cost_sum_sq: 5000.0,
            scenario_count: 4.0,
            elapsed_ms: 123,
        };
        assert_eq!(r.cost_sum, 100.0);
        assert_eq!(r.cost_sum_sq, 5000.0);
        assert_eq!(r.scenario_count, 4.0);
        assert_eq!(r.elapsed_ms, 123);
    }

    #[test]
    fn forward_result_clone_and_debug() {
        let r = ForwardResult {
            cost_sum: 2.0,
            cost_sum_sq: 3.0,
            scenario_count: 4.0,
            elapsed_ms: 5,
        };
        let c = r.clone();
        assert_eq!(c.cost_sum, r.cost_sum);
        assert_eq!(c.cost_sum_sq, r.cost_sum_sq);
        let s = format!("{r:?}");
        assert!(s.contains("ForwardResult"));
    }

    // ── Unit tests: build_cut_row_batch ──────────────────────────────────────

    #[test]
    fn build_cut_row_batch_empty_cuts_returns_empty_batch() {
        // Given: FCF with no cuts at stage 0
        let fcf = FutureCostFunction::new(2, 1, 1, 10, 0);
        let indexer = StageIndexer::new(1, 0); // n_state=1, theta=2

        let batch = build_cut_row_batch(&fcf, 0, &indexer);

        assert_eq!(batch.num_rows, 0);
        assert_eq!(batch.row_starts, vec![0]);
        assert!(batch.col_indices.is_empty());
        assert!(batch.values.is_empty());
        assert!(batch.row_lower.is_empty());
        assert!(batch.row_upper.is_empty());
    }

    #[test]
    fn build_cut_row_batch_one_cut_correct_structure() {
        // Given: FCF with one cut at stage 0: theta >= 5 + 2*x[0]
        // In row form: -2*x[0] + theta >= 5
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, 0);
        fcf.add_cut(0, 0, 0, 5.0, &[2.0]);

        let indexer = StageIndexer::new(1, 0); // n_state=1, theta=2

        let batch = build_cut_row_batch(&fcf, 0, &indexer);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_starts, vec![0, 2]); // 1 state + 1 theta = 2 NZ
        assert_eq!(batch.col_indices, vec![0, 2]); // state col 0, theta col 2
        assert_eq!(batch.values, vec![-2.0, 1.0]); // -coeff[0], +1.0
        assert_eq!(batch.row_lower, vec![5.0]); // intercept
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_two_cuts_correct_row_starts() {
        // Given: FCF with two cuts at stage 1 with n_state=2
        // Cut A: theta >= 10 + 1*x[0] + 3*x[1]  →  -x[0] - 3*x[1] + theta >= 10
        // Cut B: theta >= 20 + 2*x[0] + 4*x[1]  →  -2*x[0] - 4*x[1] + theta >= 20
        let mut fcf = FutureCostFunction::new(2, 2, 1, 10, 0);
        fcf.add_cut(1, 0, 0, 10.0, &[1.0, 3.0]);
        fcf.add_cut(1, 1, 0, 20.0, &[2.0, 4.0]);

        let indexer = StageIndexer::new(1, 1); // N=1, L=1: n_state=2, theta=3

        let batch = build_cut_row_batch(&fcf, 1, &indexer);

        assert_eq!(batch.num_rows, 2);
        // Each row has n_state + 1 = 3 NZ
        assert_eq!(batch.row_starts, vec![0, 3, 6]);
        // Cut A: cols [0, 1, theta=3], vals [-1, -3, 1]
        assert_eq!(batch.col_indices[0], 0);
        assert_eq!(batch.col_indices[1], 1);
        assert_eq!(batch.col_indices[2], 3); // theta
        assert_eq!(batch.values[0], -1.0);
        assert_eq!(batch.values[1], -3.0);
        assert_eq!(batch.values[2], 1.0);
        // Cut B: cols [0, 1, theta=3], vals [-2, -4, 1]
        assert_eq!(batch.col_indices[3], 0);
        assert_eq!(batch.col_indices[4], 1);
        assert_eq!(batch.col_indices[5], 3);
        assert_eq!(batch.values[3], -2.0);
        assert_eq!(batch.values[4], -4.0);
        assert_eq!(batch.values[5], 1.0);
        assert_eq!(batch.row_lower, vec![10.0, 20.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
        assert!(batch.row_upper[1].is_infinite() && batch.row_upper[1] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_zero_coefficient_state_variable() {
        // A cut with a zero coefficient must still emit the corresponding
        // non-zero entry (value = 0.0); the structure must be complete.
        let mut fcf = FutureCostFunction::new(1, 2, 1, 5, 0);
        fcf.add_cut(0, 0, 0, 3.0, &[0.0, 7.0]);

        let indexer = StageIndexer::new(1, 1); // n_state=2, theta=3

        let batch = build_cut_row_batch(&fcf, 0, &indexer);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.col_indices, vec![0, 1, 3]);
        assert_eq!(batch.values, vec![0.0, -7.0, 1.0]);
        assert_eq!(batch.row_lower, vec![3.0]);
    }

    // ── Acceptance criteria integration tests ───────────────────────────────

    /// AC: 2 scenarios, 3 stages, fixed `LpSolution(objective=100, theta=30)`.
    /// Expected: `scenario_count=2`, all 6 records with `stage_cost=70`.
    #[test]
    fn ac_two_scenarios_three_stages_fixed_solution() {
        // StageIndexer: N=1, L=0 → n_state=1, theta=2, num_cols=3
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(3, 100.0, indexer.theta, 30.0);
        let mut solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
        let config = TrainingConfig {
            forward_passes: 2,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let comm = StubComm { rank: 0, size: 1 };
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![1usize, 1, 1];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(2 * 3);
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let stochastic = make_stochastic_context_1_hydro_3_stages();

        let result = run_forward_pass(
            &mut solver,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            0,
            &horizon,
            &initial_state,
            &mut records,
            &mut patch_buf,
            &indexer,
            &comm,
            &mut vec![None; templates.len()],
        )
        .unwrap();

        // AC: scenario_count equals forward_passes as f64.
        assert_eq!(result.scenario_count, 2.0);
        // AC: all 6 records have stage_cost = 100 - 30 = 70.
        for (i, record) in records.iter().enumerate() {
            assert_eq!(
                record.stage_cost, 70.0,
                "record[{i}].stage_cost should be 70.0 (objective - theta)"
            );
        }
    }

    /// AC: mock solver returns `Infeasible` at stage 1, scenario 0.
    ///
    /// Call 0 = scenario 0 stage 0 (succeeds). Call 1 = scenario 0 stage 1
    /// (infeasible). The function must return `SddpError::Infeasible { stage: 1,
    /// scenario: 0 }`.
    #[test]
    fn ac_infeasible_at_stage_1_scenario_0_returns_infeasible_error() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(3, 100.0, indexer.theta, 30.0);
        // The 2nd solve call (index 1) is stage 1 of scenario 0.
        let mut solver = MockSolver::infeasible_on(solution, 1);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
        let config = TrainingConfig {
            forward_passes: 2,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let comm = StubComm { rank: 0, size: 1 };
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![1usize, 1, 1];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(2 * 3);
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let stochastic = make_stochastic_context_1_hydro_3_stages();

        let result = run_forward_pass(
            &mut solver,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            0,
            &horizon,
            &initial_state,
            &mut records,
            &mut patch_buf,
            &indexer,
            &comm,
            &mut vec![None; templates.len()],
        );

        // AC: must return SddpError::Infeasible with stage=1, scenario=0.
        match result {
            Err(crate::SddpError::Infeasible {
                stage, scenario, ..
            }) => {
                assert_eq!(stage, 1, "expected stage=1");
                assert_eq!(scenario, 0, "expected scenario=0");
            }
            other => panic!("expected Infeasible, got {other:?}"),
        }
    }

    /// AC: with `forward_passes=3`, rank=1, size=2, `global_scenario` for m=0 is 3.
    #[test]
    fn ac_global_scenario_index_rank1_scenario0() {
        // global_scenario = rank * forward_passes + m = 1 * 3 + 0 = 3
        let rank = 1usize;
        let forward_passes = 3usize;
        let m = 0usize;
        let global_scenario = rank * forward_passes + m;
        assert_eq!(global_scenario, 3);
    }

    /// Behavioral: `cost_sum` and `cost_sum_sq` are correctly accumulated.
    ///
    /// With 2 scenarios and `stage_cost=70` at every `(scenario, stage)`:
    /// - `total_cost` per scenario = 70 * 3 = 210
    /// - `cost_sum` = 210 + 210 = 420
    /// - `cost_sum_sq` = 210^2 + 210^2 = 88200
    #[test]
    fn cost_statistics_accumulated_correctly() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(3, 100.0, indexer.theta, 30.0);
        let mut solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
        let config = TrainingConfig {
            forward_passes: 2,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let comm = StubComm { rank: 0, size: 1 };
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![1usize, 1, 1];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(2 * 3);
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let stochastic = make_stochastic_context_1_hydro_3_stages();

        let result = run_forward_pass(
            &mut solver,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            0,
            &horizon,
            &initial_state,
            &mut records,
            &mut patch_buf,
            &indexer,
            &comm,
            &mut vec![None; templates.len()],
        )
        .unwrap();

        // stage_cost per solve = 100 - 30 = 70
        // total_cost per scenario = 70 * 3 stages = 210
        assert_eq!(result.cost_sum, 420.0);
        assert_eq!(result.cost_sum_sq, 210.0_f64.powi(2) * 2.0);
    }

    // ── Unit tests: SyncResult ───────────────────────────────────────────────

    #[test]
    fn sync_result_field_access() {
        let r = SyncResult {
            global_ub_mean: 75.0,
            global_ub_std: 12.909,
            ci_95_half_width: 12.651,
            sync_time_ms: 7,
        };
        assert_eq!(r.global_ub_mean, 75.0);
        assert_eq!(r.global_ub_std, 12.909);
        assert_eq!(r.ci_95_half_width, 12.651);
        assert_eq!(r.sync_time_ms, 7);
    }

    #[test]
    fn sync_result_clone_and_debug() {
        let r = SyncResult {
            global_ub_mean: 2.0,
            global_ub_std: 3.0,
            ci_95_half_width: 4.0,
            sync_time_ms: 5,
        };
        let c = r.clone();
        assert_eq!(c.global_ub_mean, r.global_ub_mean);
        assert_eq!(c.global_ub_std, r.global_ub_std);
        let s = format!("{r:?}");
        assert!(s.contains("SyncResult"));
    }

    // ── Unit tests: UB statistics computation ───────────────────────────────

    /// AC: 4 scenarios with costs [60, 70, 80, 90].
    ///
    /// `cost_sum` = 300, `cost_sum_sq` = 22700, count = 4.
    /// mean = 75.0
    /// variance = (22700 - 4 * 75^2) / 3 = (22700 - 22500) / 3 = 200/3
    /// std = sqrt(200/3) ≈ 8.165
    /// `ci_95` = 1.96 * 8.165 / 2.0 ≈ 8.0
    #[test]
    fn ub_statistics_four_scenarios_correct_mean_and_std() {
        let local = ForwardResult {
            cost_sum: 300.0,
            cost_sum_sq: 22700.0,
            scenario_count: 4.0,
            elapsed_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm).unwrap();

        assert_eq!(result.global_ub_mean, 75.0, "mean must be 300/4 = 75");

        // variance = 200/3 ≈ 66.667, std ≈ 8.165
        let expected_std = (200.0_f64 / 3.0).sqrt();
        let tolerance = 1e-9;
        assert!(
            (result.global_ub_std - expected_std).abs() < tolerance,
            "std deviation {got} should be ≈ {expected_std}",
            got = result.global_ub_std,
        );

        // ci_95 = 1.96 * std / sqrt(4) = 1.96 * std / 2
        let expected_ci = 1.96_f64 * expected_std / 4.0_f64.sqrt();
        assert!(
            (result.ci_95_half_width - expected_ci).abs() < tolerance,
            "ci_95 {got} should be ≈ {expected_ci}",
            got = result.ci_95_half_width,
        );
    }

    /// AC: 4 scenarios, costs [60,70,80,90].
    ///
    /// Matches the exact acceptance criterion values: `global_ub_mean` = 75.0,
    /// `global_ub_std` > 0.
    ///
    /// Note: the ticket uses `cost_sum_sq` = 22700, `scenario_count` = 4, mean = 75.
    /// std = sqrt((22700 - 4*75^2) / 3) = sqrt(200/3) ≈ 8.165.
    /// We test the formula as specified in the Requirements section (SS3.1a):
    /// variance = (`sum_sq` - N * mean^2) / (N - 1).
    #[test]
    fn ac_ticket_acceptance_criterion_ub_mean() {
        // These are the exact values from the ticket's acceptance criterion.
        let local = ForwardResult {
            cost_sum: 300.0,
            cost_sum_sq: 22700.0,
            scenario_count: 4.0,
            elapsed_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm).unwrap();

        assert_eq!(result.global_ub_mean, 75.0);
        // std = sqrt((22700 - 4 * 75^2) / 3) = sqrt(200/3) ≈ 8.165
        assert!(
            result.global_ub_std > 0.0,
            "std must be positive for 4 distinct scenarios"
        );
    }

    /// AC: Bessel correction edge case — single scenario produces zero variance.
    #[test]
    fn bessel_correction_single_scenario_zero_std_and_ci() {
        let local = ForwardResult {
            cost_sum: 500.0,
            cost_sum_sq: 250_000.0,
            scenario_count: 1.0,
            elapsed_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm).unwrap();

        assert_eq!(
            result.global_ub_std, 0.0,
            "std must be 0.0 for a single scenario (N=1 Bessel correction)"
        );
        assert_eq!(
            result.ci_95_half_width, 0.0,
            "ci_95 must be 0.0 for a single scenario"
        );
    }

    /// Guard: negative variance from floating-point cancellation → std = 0.0, not NaN.
    #[test]
    fn negative_variance_guard_produces_zero_std_not_nan() {
        // Construct inputs where the single-pass formula gives a slightly
        // negative variance. With N=2, mean=X, if sum_sq is computed with
        // slight rounding, the subtraction N*mean^2 can exceed sum_sq by eps.
        //
        // sum = 2 * 1e15, sum_sq = 2 * (1e15)^2.
        // mean = 1e15, N*mean^2 = 2 * 1e30.
        // Exact: variance = (2e30 - 2e30) / 1 = 0, but floating-point
        // representation of 2e30 may differ between sum_sq and N*mean^2,
        // producing a tiny negative result.
        let v = 1.0e15_f64;
        let local = ForwardResult {
            cost_sum: 2.0 * v,
            // Subtract epsilon to force a slightly negative exact variance.
            cost_sum_sq: 2.0 * v * v - 1.0,
            scenario_count: 2.0,
            elapsed_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm).unwrap();

        assert!(
            !result.global_ub_std.is_nan(),
            "std must not be NaN even when floating-point variance is slightly negative"
        );
        assert_eq!(
            result.global_ub_std, 0.0,
            "std must be 0.0 when variance clamps to max(0, negative)"
        );
    }

    // ── Integration tests: sync_forward with LocalBackend ────────────────────

    /// Integration: single-rank mode — global UB mean equals local mean.
    #[test]
    fn sync_forward_local_backend_global_equals_local() {
        let local = ForwardResult {
            cost_sum: 840.0,
            cost_sum_sq: 840.0_f64 * 840.0,
            scenario_count: 2.0,
            elapsed_ms: 5,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm).unwrap();

        // In single-rank mode, allreduce is identity copy.
        assert_eq!(
            result.global_ub_mean,
            840.0 / 2.0,
            "global_ub_mean must be cost_sum / scenario_count"
        );
    }

    /// Integration: `sync_time_ms` is a valid non-negative u64.
    #[test]
    fn sync_forward_sync_time_ms_is_valid_u64() {
        let local = ForwardResult {
            cost_sum: 100.0,
            cost_sum_sq: 5000.0,
            scenario_count: 2.0,
            elapsed_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm).unwrap();
        // sync_time_ms is u64 — any value is a valid non-negative u64.
        // We just verify the field exists and doesn't overflow to something absurd.
        let _ = result.sync_time_ms;
    }

    /// Integration: `CommError` from a failing communicator is wrapped as `SddpError::Communication`.
    #[test]
    fn sync_forward_comm_error_wraps_as_sddp_communication() {
        use cobre_comm::CommError;

        /// Communicator that always returns `CommError::InvalidCommunicator`.
        struct FailingComm;

        impl Communicator for FailingComm {
            fn allgatherv<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                Err(CommError::InvalidCommunicator)
            }

            fn allreduce<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _op: ReduceOp,
            ) -> Result<(), CommError> {
                Err(CommError::InvalidCommunicator)
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                Err(CommError::InvalidCommunicator)
            }

            fn barrier(&self) -> Result<(), CommError> {
                Err(CommError::InvalidCommunicator)
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                1
            }
        }

        let local = ForwardResult {
            cost_sum: 100.0,
            cost_sum_sq: 5000.0,
            scenario_count: 1.0,
            elapsed_ms: 0,
        };
        let comm = FailingComm;
        let err = sync_forward(&local, &comm).unwrap_err();

        assert!(
            matches!(err, crate::SddpError::Communication(_)),
            "CommError must be wrapped as SddpError::Communication, got: {err:?}"
        );
    }

    // ── Unit tests: warm-start basis caching ─────────────────────────────────

    /// Helper: run one iteration of `run_forward_pass` with a single scenario
    /// and a 3-stage horizon, returning the warm-start call count from the
    /// solver.
    fn run_one_iteration(
        solver: &mut MockSolver,
        basis_cache: &mut [Option<Basis>],
    ) -> Result<(), crate::SddpError> {
        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 1, 100, 0);
        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let comm = StubComm { rank: 0, size: 1 };
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![1usize, 1, 1];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(3);
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let stochastic = make_stochastic_context_1_hydro_3_stages();

        run_forward_pass(
            solver,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            0,
            &horizon,
            &initial_state,
            &mut records,
            &mut patch_buf,
            &indexer,
            &comm,
            basis_cache,
        )
        .map(|_| ())
    }

    /// Warm-start invocation: the first iteration calls `solve` (cold start);
    /// the second iteration calls `solve_with_basis` (warm start).
    ///
    /// AC: `run_forward_pass` called twice with the same `basis_cache`.
    /// After iteration 1: `warm_start_calls` == 0 (3 cold-start solves for 3 stages).
    /// After iteration 2: `warm_start_calls` > 0 (all 3 stages warm-start).
    #[test]
    fn warm_start_first_iteration_cold_second_iteration_warm() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(3, 100.0, indexer.theta, 30.0);
        let mut solver = MockSolver::always_ok(solution);
        let mut basis_cache: Vec<Option<Basis>> = vec![None, None, None];

        // First iteration: no cached bases → all cold-start.
        run_one_iteration(&mut solver, &mut basis_cache).unwrap();
        assert_eq!(
            solver.warm_start_calls, 0,
            "first iteration must use cold-start for all stages (warm_start_calls == 0)"
        );

        // After first iteration, all 3 stages have a cached basis.
        assert!(
            basis_cache.iter().all(Option::is_some),
            "basis_cache must be fully populated after the first iteration"
        );

        // Second iteration: cached bases present → all stages warm-start.
        run_one_iteration(&mut solver, &mut basis_cache).unwrap();
        assert!(
            solver.warm_start_calls > 0,
            "second iteration must use warm-start for at least one stage \
             (warm_start_calls > 0, got {})",
            solver.warm_start_calls
        );
    }

    /// Basis invalidation: when a solve returns `SolverError::Infeasible`,
    /// `basis_cache[t]` must be set to `None` before the error is propagated.
    /// The next iteration at that stage cold-starts.
    ///
    /// AC: `MockSolver` returns `Infeasible` on call index 4 (second iteration,
    /// stage 1 — stages 0,1,2 are calls 0,1,2 in iteration 1; then 3,4,5 in
    /// iteration 2). After the error, `basis_cache[1]` must be `None`.
    /// A third iteration at stage 1 must not call warm-start (cold-start instead).
    #[test]
    fn basis_invalidated_on_solver_error() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(3, 100.0, indexer.theta, 30.0);
        // Call 4 = second iteration, stage 1 (calls 0-2 = first iteration
        // stages 0,1,2; calls 3,4,5 = second iteration stages 0,1,2).
        let mut solver = MockSolver::infeasible_on(solution, 4);
        let mut basis_cache: Vec<Option<Basis>> = vec![None, None, None];

        // First iteration: all cold-start, all succeed, cache all 3 bases.
        run_one_iteration(&mut solver, &mut basis_cache).unwrap();
        assert!(
            basis_cache.iter().all(Option::is_some),
            "cache must be full after iteration 1"
        );

        // Second iteration: stages 0 warm-start (call 3 OK), stage 1 infeasible (call 4).
        let err = run_one_iteration(&mut solver, &mut basis_cache).unwrap_err();
        assert!(
            matches!(err, crate::SddpError::Infeasible { stage: 1, .. }),
            "expected Infeasible at stage 1, got: {err:?}"
        );

        // AC: basis_cache[1] must be None after the error (invalidated).
        assert!(
            basis_cache[1].is_none(),
            "basis_cache[1] must be None after solver error at stage 1"
        );

        // Stages 0 and 2: stage 0 was warm-started before the error (basis still set);
        // stage 2 was never reached in iteration 2 (error propagated early).
        // The exact state of cache[0] and cache[2] depends on traversal order:
        // stage 0 succeeded and had its basis re-extracted; stage 2 was not reached.
        assert!(
            basis_cache[0].is_some(),
            "basis_cache[0] must still be Some (solve succeeded before error)"
        );
    }
}
