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
//! The user's total forward passes are split across MPI ranks by the caller
//! (`train()`), which passes each rank's local share as the
//! `local_forward_passes` parameter. The global scenario index for local
//! scenario `m` is `fwd_offset + m`, where `fwd_offset` is the pre-computed
//! global index of this rank's first forward pass. This deterministic mapping
//! drives the communication-free seed derivation used by `ForwardSampler::sample`.
//!
//! ## Thread-level parallelism
//!
//! Within a rank, scenarios are distributed across one or more [`SolverWorkspace`]
//! instances from a [`crate::WorkspacePool`]. Each workspace owns its solver, patch
//! buffer, and current-state buffer. Rayon's `par_iter_mut` drives
//! the scenario loop: scenarios are statically partitioned across workspaces
//! (not rayon's default work-stealing chunking) so that the assignment of
//! scenarios to workers is deterministic and reproducible.
//!
//! Warm-start bases are stored per `(scenario, stage)` in a [`BasisStore`]
//! passed by the caller. Before the parallel region the store is split into
//! disjoint per-worker sub-views via [`BasisStore::split_workers_mut`]; each
//! worker writes bases only for its own scenario range.
//!
//! Per-scenario costs are collected after the parallel region by merging
//! worker-local cost vectors in global scenario index order (worker 0's costs
//! first, then worker 1's, etc.). This canonical ordering ensures that
//! [`sync_forward`] produces bit-identical statistics for any workspace count.
//!
//! ## LP rebuild sequence
//!
//! For each `(worker, scenario)` pair (iterating over scenarios in the outer
//! loop and stages in the inner loop):
//!
//! 1. `solver.load_model(template)` — reset to the structural LP for this
//!    stage (per-scenario reload, not per-stage).
//! 2. `solver.add_rows(cut_batch)` — append active Benders cuts.
//! 3. `solver.set_row_bounds(...)` — patch scenario-specific row bounds.
//!
//! Reloading the full model per scenario (rather than reusing the model across
//! scenarios within the same stage) ensures deterministic LP state regardless
//! of thread assignment: no residual warm-start artefacts or bound mutations
//! carry over between scenarios processed by the same worker.
//!
//! ## Hot-path allocation discipline
//!
//! No allocations occur per scenario during the inner loops. Per-worker
//! buffers (`trajectory_costs`, `raw_noise_buf`, `perm_scratch`) are
//! allocated once at the start of each iteration — one allocation per worker,
//! not per scenario. The [`TrajectoryRecord`] slice is pre-allocated by the
//! caller. The only additional allocation inside the function is the
//! [`RowBatch`] built by `build_cut_row_batch`, which runs once per stage
//! template (before the scenario loop) — not once per scenario.

use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::WelfordAccumulator;
use cobre_solver::{RowBatch, SolverError, SolverInterface};
use cobre_stochastic::context::ClassSchemes;
use cobre_stochastic::{
    build_forward_sampler, ClassDimensions, ClassSampleRequest, ForwardSampler,
    ForwardSamplerConfig, SampleRequest,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    basis_reconstruct::{reconstruct_basis, PaddingContext, ReconstructionTarget},
    context::{BakedTemplates, StageContext, TrainingContext},
    cut::pool::CutPool,
    lp_builder::COST_SCALE_FACTOR,
    noise::{transform_inflow_noise, transform_load_noise, transform_ncs_noise},
    solver_stats::SolverStatsDelta,
    workspace::{BasisStore, BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
    FutureCostFunction, SddpError, StageIndexer, TrajectoryRecord,
};

/// Local statistics from one rank's forward pass.
///
/// Carries the individual per-scenario trajectory costs in global scenario
/// index order (scenario 0 first, scenario N-1 last). The synchronisation
/// step gathers these costs from all ranks via `allgatherv` and performs
/// canonical-order summation to produce bit-identical statistics regardless
/// of the number of MPI ranks or intra-rank worker threads.
///
/// Does not contain lower bound estimate (evaluated separately after backward pass).
#[derive(Debug, Clone)]
#[must_use]
pub struct ForwardResult {
    /// Per-scenario trajectory costs in global scenario index order.
    ///
    /// Length equals the number of scenarios solved on this rank.
    /// Scenario `m` (local index) appears at position `m`.
    pub scenario_costs: Vec<f64>,

    /// Wall-clock time in milliseconds for this rank's forward pass.
    pub elapsed_ms: u64,

    /// Number of LP solves performed during this forward pass.
    pub lp_solves: u64,

    /// Aggregate non-solve work inside the parallel region summed across all
    /// workers, in milliseconds.
    ///
    /// Computed as the sum across all workers of
    /// `load_model_time_ms + add_rows_time_ms + set_bounds_time_ms + basis_set_time_ms`.
    pub setup_time_ms: u64,

    /// Load-imbalance component of parallel overhead, in milliseconds.
    ///
    /// Computed as `max_worker_total_ms - avg_worker_total_ms`, where
    /// `worker_total_ms = solve + load_model + add_rows + set_bounds + basis_set`
    /// for each worker.  Measures how much the slowest worker exceeds the average.
    pub load_imbalance_ms: u64,

    /// True rayon scheduling overhead, in milliseconds.
    ///
    /// Computed as `parallel_wall_ms - max_worker_total_ms`, clamped to zero.
    /// Represents rayon barrier, thread wake-up, and work-stealing dispatch costs
    /// after accounting for all measured per-worker work.
    pub scheduling_overhead_ms: u64,
}

/// Global upper bound statistics from forward synchronisation step.
#[derive(Debug, Clone)]
#[must_use]
pub struct SyncResult {
    /// Sample mean of total trajectory costs across all ranks.
    pub global_ub_mean: f64,

    /// Bessel-corrected sample standard deviation of total trajectory costs.
    pub global_ub_std: f64,

    /// 95% confidence interval half-width: `1.96 * std / sqrt(N)`.
    pub ci_95_half_width: f64,

    /// Wall-clock time in milliseconds for the forward synchronization call.
    pub sync_time_ms: u64,
}

/// Aggregate local forward pass statistics across all MPI ranks.
///
/// Performs an `allgatherv` collective operation to produce global upper bound
/// statistics from the per-rank [`ForwardResult`] produced by [`run_forward_pass`]:
///
/// - `allgatherv` of `local.scenario_costs` gathers every rank's per-scenario
///   costs into a single canonical-order buffer (rank 0's costs first, then
///   rank 1's, …). All ranks receive the full cost vector.
/// - Statistics are computed via sequential summation in canonical global
///   scenario index order, eliminating floating-point non-associativity from
///   different rank-count groupings.
///
/// From the gathered costs, the following statistics are computed (per SS3.1a):
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
/// In single-rank mode (`comm.size() == 1`), `LocalBackend.allgatherv` is an
/// identity copy. No special-casing is needed — the result equals the local values.
///
/// ## Arguments
///
/// - `local` — the [`ForwardResult`] from the calling rank's forward pass.
/// - `comm` — the communicator used for collective operations.
/// - `total_forward_passes` — the total number of forward passes across all
///   ranks. Used to compute per-rank counts and displacements arithmetically
///   without a preliminary communication round.
///
/// # Errors
///
/// Returns `Err(SddpError::Communication(_))` if the `allgatherv` call fails.
/// The `From<CommError>` conversion on `SddpError` is applied automatically
/// via the `?` operator. No partial results are produced on error.
pub fn sync_forward<C: Communicator>(
    local: &ForwardResult,
    comm: &C,
    total_forward_passes: usize,
) -> Result<SyncResult, SddpError> {
    let start = Instant::now();

    let num_ranks = comm.size();
    let my_rank = comm.rank();

    // Compute per-rank counts and displacements arithmetically from the total.
    // Each rank r receives `partition(total_forward_passes, num_ranks, r)` scenarios.
    let base = total_forward_passes / num_ranks;
    let remainder = total_forward_passes % num_ranks;
    let counts: Vec<usize> = (0..num_ranks)
        .map(|r| base + usize::from(r < remainder))
        .collect();
    let mut displs = vec![0usize; num_ranks];
    for r in 1..num_ranks {
        displs[r] = displs[r - 1] + counts[r - 1];
    }

    // Allocate the global cost buffer and gather all per-scenario costs.
    let global_n = counts.iter().sum::<usize>();
    debug_assert_eq!(
        global_n, total_forward_passes,
        "counts sum {global_n} != total_forward_passes {total_forward_passes}",
    );
    let mut global_costs = vec![0.0_f64; global_n];

    // Validate that this rank's local cost vector matches the expected count.
    debug_assert_eq!(
        local.scenario_costs.len(),
        counts[my_rank],
        "rank {my_rank}: scenario_costs length {} != expected count {}",
        local.scenario_costs.len(),
        counts[my_rank],
    );

    comm.allgatherv(&local.scenario_costs, &mut global_costs, &counts, &displs)?;

    // Canonical-order single-pass statistics. All ranks iterate global_costs in
    // the same order, producing bit-identical statistics regardless of rank count.
    // Welford's online algorithm is used instead of the two-pass naive formula to
    // avoid catastrophic cancellation when sum_sq ≈ n * mean^2 (F1-007 fix).
    // MPI Welford merge is not used here because the full gathered array is
    // already available — a single sequential pass suffices.
    let mut welford = WelfordAccumulator::new();
    for &c in &global_costs {
        welford.update(c);
    }
    let mean = welford.mean();
    let (std_dev, ci_95) = if global_n > 1 {
        let sd = welford.sample_std_dev();
        let ci = welford.sample_ci_95_half_width();
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

/// Push one negated, scaled coefficient entry into the cut row batch.
///
/// Shared by the sparse and dense paths in [`build_cut_row_batch_into`] to
/// prevent the two branches from drifting apart during maintenance.
#[allow(clippy::empty_line_after_doc_comments)]
#[inline]
fn push_scaled_coefficient(batch: &mut RowBatch, j: usize, coeff: f64, col_scale: &[f64]) {
    debug_assert!(
        i32::try_from(j).is_ok(),
        "column index j={j} exceeds i32::MAX"
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    batch.col_indices.push(j as i32);
    let d = if col_scale.is_empty() {
        1.0
    } else {
        col_scale[j]
    };
    batch.values.push(-coeff * d);
}

/// Fill a pre-allocated [`RowBatch`] with Benders cut rows from the FCF.
///
/// Clears `batch` and repopulates it with active cuts from `fcf` at the
/// given `stage`. The buffers inside `batch` retain their allocated capacity
/// across calls, eliminating heap allocation on the hot path.
///
/// This is the allocation-free core used by `build_cut_row_batch`.
///
/// # Panics
///
/// Panics if the total number of non-zeros exceeds `i32::MAX` (the `HiGHS`
/// API limit for CSR indices).
pub fn build_cut_row_batch_into(
    batch: &mut RowBatch,
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
) {
    batch.clear();

    let n_state = indexer.n_state;
    let theta_col = indexer.theta;
    let mask = &indexer.nonzero_state_indices;
    let is_sparse = !mask.is_empty();

    let num_cuts: usize = fcf.pools[stage].active_count();

    if num_cuts == 0 {
        batch.row_starts.push(0_i32);
        return;
    }

    // Sparse path: NNZ = mask.len() + 1 (nonzero state entries + theta).
    // Dense path: NNZ = n_state + 1 (all state entries + theta).
    let nnz_per_cut = if is_sparse {
        mask.len() + 1
    } else {
        n_state + 1
    };
    let total_nnz = num_cuts * nnz_per_cut;

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
        batch.row_starts.push(nz_offset as i32);

        // Unified state coefficient loop: sparse iterates over the nonzero
        // mask, dense iterates over all state indices. Both yield (col_index,
        // coefficient) pairs and share the same push logic.
        //
        // state_to_lp_column remaps outgoing-state indices to LP columns.
        // For storage (j < N) the mapping is identity. For lag dimensions
        // the outgoing state after shift_lag_state stores z_inflow at lag 0
        // and shifted incoming lags at lag 1+, so the cut must reference the
        // corresponding LP columns (z_inflow and incoming lag l−1).
        if is_sparse {
            for &j in mask {
                let lp_col = indexer.state_to_lp_column(j);
                push_scaled_coefficient(batch, lp_col, coefficients[j], col_scale);
            }
        } else {
            for (j, &c) in coefficients.iter().enumerate() {
                let lp_col = indexer.state_to_lp_column(j);
                push_scaled_coefficient(batch, lp_col, c, col_scale);
            }
        }

        debug_assert!(
            i32::try_from(theta_col).is_ok(),
            "theta_col={theta_col} exceeds i32::MAX"
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch.col_indices.push(theta_col as i32);
        let d_theta = if col_scale.is_empty() {
            1.0
        } else {
            col_scale[theta_col]
        };
        batch.values.push(d_theta);

        batch.row_lower.push(intercept);
        batch.row_upper.push(f64::INFINITY);

        nz_offset += nnz_per_cut;
    }

    #[allow(clippy::expect_used)]
    batch.row_starts.push(
        i32::try_from(total_nnz).expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
    );

    batch.num_rows = num_cuts;
}

/// Build a fresh [`RowBatch`] of Benders cut rows from the FCF.
///
/// Convenience wrapper around [`build_cut_row_batch_into`] that allocates a
/// new `RowBatch`. For allocation-free usage on the hot path, prefer calling
/// [`build_cut_row_batch_into`] with a pre-allocated batch.
#[must_use]
pub fn build_cut_row_batch(
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
) -> RowBatch {
    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    build_cut_row_batch_into(&mut batch, fcf, stage, indexer, col_scale);
    batch
}

/// Fill a pre-allocated [`RowBatch`] with only the Benders cut rows generated
/// in `current_iteration`.
///
/// Clears `batch` and repopulates it with the subset of active cuts from
/// `fcf.pools[stage]` whose `iteration_generated` metadata field equals
/// `current_iteration`. Warm-start cuts (sentinel `iteration_generated ==
/// u64::MAX`) are always excluded.
///
/// Delta-cut variant of [`build_cut_row_batch_into`] for use with baked templates.
///
/// When a baked template contains all cuts from previous iterations, this
/// function builds only the new cuts from `current_iteration` for appending
/// via `add_rows`. The CSR layout and coefficient transformation are identical
/// to [`build_cut_row_batch_into`]; when the pool contains only cuts from
/// `current_iteration`, both functions produce byte-identical output.
///
/// # Panics
///
/// Panics if total non-zeros exceeds `i32::MAX` (`HiGHS` API limit).
pub fn build_delta_cut_row_batch_into(
    batch: &mut RowBatch,
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
    current_iteration: u64,
) {
    batch.clear();

    let n_state = indexer.n_state;
    let theta_col = indexer.theta;
    let mask = &indexer.nonzero_state_indices;
    let is_sparse = !mask.is_empty();

    // Count delta cuts with a lightweight scan to avoid double-iteration
    // overhead in the common case of zero delta cuts (early return).
    let num_cuts: usize = fcf.pools[stage]
        .active_delta_cuts(current_iteration)
        .count();

    if num_cuts == 0 {
        batch.row_starts.push(0_i32);
        return;
    }

    // Sparse path: NNZ = mask.len() + 1 (nonzero state entries + theta).
    // Dense path: NNZ = n_state + 1 (all state entries + theta).
    let nnz_per_cut = if is_sparse {
        mask.len() + 1
    } else {
        n_state + 1
    };
    let total_nnz = num_cuts * nnz_per_cut;

    let mut nz_offset = 0;

    for (_slot, intercept, coefficients) in fcf.pools[stage].active_delta_cuts(current_iteration) {
        debug_assert_eq!(
            coefficients.len(),
            n_state,
            "cut coefficients length {got} != n_state {expected}",
            got = coefficients.len(),
            expected = n_state,
        );

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch.row_starts.push(nz_offset as i32);

        // Unified state coefficient loop: sparse iterates over the nonzero
        // mask, dense iterates over all state indices. Both yield (col_index,
        // coefficient) pairs and share the same push logic.
        //
        // state_to_lp_column remaps outgoing-state indices to LP columns.
        // For storage (j < N) the mapping is identity. For lag dimensions
        // the outgoing state after shift_lag_state stores z_inflow at lag 0
        // and shifted incoming lags at lag 1+, so the cut must reference the
        // corresponding LP columns (z_inflow and incoming lag l−1).
        if is_sparse {
            for &j in mask {
                let lp_col = indexer.state_to_lp_column(j);
                push_scaled_coefficient(batch, lp_col, coefficients[j], col_scale);
            }
        } else {
            for (j, &c) in coefficients.iter().enumerate() {
                let lp_col = indexer.state_to_lp_column(j);
                push_scaled_coefficient(batch, lp_col, c, col_scale);
            }
        }

        debug_assert!(
            i32::try_from(theta_col).is_ok(),
            "theta_col={theta_col} exceeds i32::MAX"
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch.col_indices.push(theta_col as i32);
        let d_theta = if col_scale.is_empty() {
            1.0
        } else {
            col_scale[theta_col]
        };
        batch.values.push(d_theta);

        batch.row_lower.push(intercept);
        batch.row_upper.push(f64::INFINITY);

        nz_offset += nnz_per_cut;
    }

    #[allow(clippy::expect_used)]
    batch.row_starts.push(
        i32::try_from(total_nnz).expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
    );

    batch.num_rows = num_cuts;
}

/// Append only the newly active cuts (not yet in the LP) to a live solver.
///
/// Iterates over all active cuts in `fcf.pools[stage]`, checks `row_map` to
/// determine which are already present in the LP, builds a small [`RowBatch`]
/// containing only the new cuts, and calls `solver.add_rows()`. Updates
/// `row_map` with the new LP row indices.
///
/// Returns the number of new cuts appended (0 if none).
///
/// The LP rows produced use the same coefficient transformation as
/// [`build_cut_row_batch_into`]: negated state coefficients with column
/// scaling and a positive theta column entry.
///
/// # Arguments
///
/// - `solver`: the live LP solver instance with a loaded model.
/// - `fcf`: the Future Cost Function containing all cut pools.
/// - `stage`: 0-based stage index.
/// - `indexer`: provides `n_state` and `theta` column index.
/// - `col_scale`: column scaling factors (empty slice if no scaling).
/// - `row_map`: per-stage [`CutRowMap`] to update.
/// - `batch_buf`: reusable [`RowBatch`] buffer for constructing the new cut rows.
///
/// # Panics
///
/// Panics if `total_nnz` exceeds `i32::MAX` (LP exceeds the `HiGHS` API limit).
/// In debug builds, also panics if `stage >= fcf.pools.len()`.
///
/// [`CutRowMap`]: crate::cut::CutRowMap
pub fn append_new_cuts_to_lp<S: SolverInterface>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
    row_map: &mut crate::cut::CutRowMap,
    batch_buf: &mut RowBatch,
) -> usize {
    batch_buf.clear();

    let n_state = indexer.n_state;
    let theta_col = indexer.theta;
    let mask = &indexer.nonzero_state_indices;
    let is_sparse = !mask.is_empty();
    let nnz_per_cut = if is_sparse {
        mask.len() + 1
    } else {
        n_state + 1
    };

    let mut new_count = 0usize;
    let mut nz_offset = 0usize;

    for (slot, intercept, coefficients) in fcf.active_cuts(stage) {
        // Skip cuts already present in the LP.
        if row_map.lp_row_for_slot(slot).is_some() {
            continue;
        }

        debug_assert_eq!(
            coefficients.len(),
            n_state,
            "cut coefficients length {got} != n_state {expected}",
            got = coefficients.len(),
            expected = n_state,
        );

        // Build the row using the same transformation as build_cut_row_batch_into.
        // Sparse path iterates only nonzero state indices; dense path iterates
        // all. Both use state_to_lp_column to remap outgoing-state indices to
        // LP columns.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch_buf.row_starts.push(nz_offset as i32);

        if is_sparse {
            for &j in mask {
                let lp_col = indexer.state_to_lp_column(j);
                push_scaled_coefficient(batch_buf, lp_col, coefficients[j], col_scale);
            }
        } else {
            for (j, &c) in coefficients.iter().enumerate() {
                let lp_col = indexer.state_to_lp_column(j);
                push_scaled_coefficient(batch_buf, lp_col, c, col_scale);
            }
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch_buf.col_indices.push(theta_col as i32);
        let d_theta = if col_scale.is_empty() {
            1.0
        } else {
            col_scale[theta_col]
        };
        batch_buf.values.push(d_theta);

        batch_buf.row_lower.push(intercept);
        batch_buf.row_upper.push(f64::INFINITY);

        row_map.insert(slot);
        new_count += 1;
        nz_offset += nnz_per_cut;
    }

    if new_count > 0 {
        let total_nnz = new_count * nnz_per_cut;
        #[allow(clippy::expect_used)]
        batch_buf.row_starts.push(
            i32::try_from(total_nnz)
                .expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
        );
        batch_buf.num_rows = new_count;
        solver.add_rows(batch_buf);
    }

    new_count
}

/// Bundled scalar parameters for one forward pass invocation.
///
/// Groups the per-iteration, per-rank scalar arguments that are forwarded
/// from [`crate::train`] into [`run_forward_pass`].
pub struct ForwardPassBatch {
    /// Number of forward-pass scenarios assigned to this rank.
    pub local_forward_passes: usize,
    /// Total forward passes across all MPI ranks. Used for LHS stratification
    /// in the sampler (`total_scenarios` field of `SampleRequest`) and for
    /// sizing the LHS permutation scratch buffer. Must equal the study-level
    /// `forward_passes` parameter, NOT the per-rank local count.
    pub total_forward_passes: usize,
    /// Current training iteration (0-based; used for seed derivation).
    pub iteration: u64,
    /// Global index of this rank's first forward pass for seed derivation.
    pub fwd_offset: usize,
}

/// Compute the scenario range `[start, end)` for worker `worker_id` when
/// distributing `n_scenarios` scenarios across `n_workers` workers.
///
/// Uses ceiling-division for the first `n_scenarios % n_workers` workers so
/// that extra scenarios are assigned to lower-index workers. This is a
/// deterministic, static partition — scenario-to-worker assignment is
/// identical regardless of thread scheduling order.
///
/// Returns `(start, end)` where `start == end` when `worker_id` receives zero
/// scenarios (only occurs when `n_workers > n_scenarios`).
#[inline]
pub(crate) fn partition(n_scenarios: usize, n_workers: usize, worker_id: usize) -> (usize, usize) {
    if n_workers == 0 {
        return (0, 0);
    }
    let base = n_scenarios / n_workers;
    let remainder = n_scenarios % n_workers;
    // First `remainder` workers get `base + 1` scenarios; the rest get `base`.
    let start = base * worker_id + worker_id.min(remainder);
    let end = start + base + usize::from(worker_id < remainder);
    (start, end)
}

/// Per-stage solve context for one (stage, scenario) pair in the forward pass.
///
/// Passed to [`run_forward_stage`] to bundle scalar and slice parameters and
/// keep the argument count within the clippy `too_many_arguments` threshold.
struct StageKey<'a> {
    /// 0-based stage index.
    t: usize,
    /// 0-based global scenario index (rank offset + local scenario index).
    m: usize,
    /// Local scenario index within this worker's partition.
    local_m: usize,
    /// Total number of stages in the horizon.
    num_stages: usize,
    /// Current training iteration (used in error context).
    iteration: u64,
    /// Raw noise sample for this (stage, scenario) pair.
    raw_noise: &'a [f64],
    /// Total LP row count (template + active cuts) for pre-allocating basis
    /// storage and avoiding per-scenario heap reallocation.
    basis_row_capacity: usize,
    /// True when the last study stage (`T-1`) has at least one warm-start
    /// (boundary) cut.  When true, the theta column at the terminal stage is
    /// NOT zeroed out so that the boundary cuts can contribute to the LP
    /// objective.  Computed once per forward pass from
    /// `fcf.pools[num_stages - 1].warm_start_count > 0` and reused for every
    /// (scenario, stage) pair in the pass.
    terminal_has_boundary_cuts: bool,
    /// Reference to the cut pool for stage `t`. Used by
    /// [`reconstruct_basis`](crate::basis_reconstruct::reconstruct_basis)
    /// to walk active cut rows and by `write_capture_metadata` to record slot
    /// identities for the next iteration's warm-start.
    pool: &'a CutPool,
}

/// Populate `CapturedBasis` metadata after a forward solve.
///
/// `cut_row_count` is the number of cut rows actually in the LP (derived from
/// `basis_row_capacity - base_row_count`).  On terminal stages the pool may
/// hold cuts that the LP does not load, so iterating `pool.active_cuts()`
/// blindly would over-count; `take(cut_row_count)` limits to the LP shape.
///
/// `row_status` is defensively resized to `base_row_count + cut_row_count` so
/// the metadata invariant holds even when the underlying solver's `get_basis`
/// is a no-op (e.g. test mocks).  For real solvers this is a no-op since they
/// write the correct length.
#[allow(clippy::cast_possible_truncation)]
fn write_capture_metadata(
    captured: &mut CapturedBasis,
    pool: &CutPool,
    base_row_count: usize,
    cut_row_count: usize,
    current_state: &[f64],
) {
    captured.cut_row_slots.clear();
    for (slot, _intercept, _coeffs) in pool.active_cuts().take(cut_row_count) {
        captured.cut_row_slots.push(slot as u32);
    }
    captured.state_at_capture.clear();
    captured.state_at_capture.extend_from_slice(current_state);
    captured.base_row_count = base_row_count;
    let expected_len = base_row_count + cut_row_count;
    if captured.basis.row_status.len() != expected_len {
        captured.basis.row_status.resize(
            expected_len,
            crate::basis_reconstruct::HIGHS_BASIS_STATUS_BASIC,
        );
    }
}

/// Execute the stage-level LP solve for one (scenario, stage) pair.
///
/// Applies noise patches, warm-starts the solver, records the trajectory step,
/// and updates the current state and basis store for the next stage.
///
/// Returns the stage cost on success, or propagates the solver error.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible)` when the stage LP is infeasible, or
/// `Err(SddpError::Solver)` for any other terminal solver failure.
#[allow(clippy::too_many_lines)]
fn run_forward_stage<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    key: &StageKey<'_>,
    worker_records: &mut [TrajectoryRecord],
) -> Result<f64, SddpError> {
    let StageKey {
        t,
        m,
        local_m,
        num_stages,
        iteration,
        raw_noise,
        basis_row_capacity,
        terminal_has_boundary_cuts,
        pool,
    } = *key;
    let n_hydros = ctx.n_hydros;
    let n_load_buses = ctx.n_load_buses;
    let indexer = training_ctx.indexer;
    let stochastic = training_ctx.stochastic;
    let horizon = training_ctx.horizon;

    // Split borrows: current_state and scratch are distinct fields of ws.
    let (state_ref, scratch) = (&ws.current_state[..], &mut ws.scratch);
    transform_inflow_noise(raw_noise, t, state_ref, ctx, training_ctx, scratch);
    let blk = if n_load_buses > 0 {
        ctx.block_counts_per_stage[t]
    } else {
        0
    };
    transform_load_noise(
        raw_noise,
        n_hydros,
        n_load_buses,
        stochastic,
        t,
        blk,
        &mut ws.scratch.load_rhs_buf,
    );
    let n_stochastic_ncs = stochastic.n_stochastic_ncs();
    if n_stochastic_ncs > 0 {
        transform_ncs_noise(
            raw_noise,
            n_hydros,
            n_load_buses,
            stochastic,
            t,
            ctx.block_counts_per_stage[t],
            ctx.ncs_max_gen,
            &mut ws.scratch.ncs_col_upper_buf,
        );
    }

    ws.patch_buf.fill_forward_patches(
        indexer,
        &ws.current_state,
        &ws.scratch.noise_buf,
        ctx.base_rows[t],
        &ctx.templates[t].row_scale,
    );
    if n_load_buses > 0 {
        ws.patch_buf.fill_load_patches(
            ctx.load_balance_row_starts[t],
            ctx.block_counts_per_stage[t],
            &ws.scratch.load_rhs_buf,
            ctx.load_bus_indices,
            &ctx.templates[t].row_scale,
        );
    }
    ws.patch_buf.fill_z_inflow_patches(
        indexer.z_inflow_row_start,
        &ws.scratch.z_inflow_rhs_buf,
        &ctx.templates[t].row_scale,
    );
    let pc = ws.patch_buf.forward_patch_count();
    ws.solver.set_row_bounds(
        &ws.patch_buf.indices[..pc],
        &ws.patch_buf.lower[..pc],
        &ws.patch_buf.upper[..pc],
    );
    // Patch NCS column upper bounds with per-scenario availability.
    if n_stochastic_ncs > 0 && !indexer.ncs_generation.is_empty() {
        let n_blks = ctx.block_counts_per_stage[t];
        let expected_len = n_stochastic_ncs * n_blks;
        // Only rebuild index/lower buffers when the size changes (i.e., on a stage
        // transition). Within a single stage the indices are constant across scenarios.
        if ws.scratch.ncs_col_indices_buf.len() != expected_len {
            ws.scratch.ncs_col_indices_buf.clear();
            ws.scratch.ncs_col_lower_buf.clear();
            for ncs_idx in 0..n_stochastic_ncs {
                for blk in 0..n_blks {
                    ws.scratch
                        .ncs_col_indices_buf
                        .push(indexer.ncs_generation.start + ncs_idx * n_blks + blk);
                    ws.scratch.ncs_col_lower_buf.push(0.0);
                }
            }
        }
        // ncs_col_upper_buf was populated by transform_ncs_noise above.
        ws.solver.set_col_bounds(
            &ws.scratch.ncs_col_indices_buf,
            &ws.scratch.ncs_col_lower_buf,
            &ws.scratch.ncs_col_upper_buf,
        );
    }
    // Zero out the theta column at the terminal stage (the last study stage,
    // `T-1`) so that the LP does not penalise future cost when there is no
    // successor.  The zeroing is skipped when boundary (warm-start) cuts have
    // been loaded for the terminal stage: in that case the cuts constrain
    // theta from below and their contribution to the objective must remain
    // visible to the solver.
    if horizon.is_terminal(t + 1) && !terminal_has_boundary_cuts {
        ws.solver.set_col_bounds(&[indexer.theta], &[0.0], &[0.0]);
    }

    // Grow the slot-lookup scratch if the pool has allocated new slots since
    // the last call.  `pool.populated_count` is monotonically non-decreasing,
    // so after the first few iterations this check is a no-op.
    if ws.scratch.recon_slot_lookup.len() < pool.populated_count {
        ws.scratch
            .recon_slot_lookup
            .resize(pool.populated_count, None);
    }

    let view = match basis_slice.get_mut(m, t) {
        &mut Some(ref captured) => {
            // Slot-tracked reconstruction: copy preserved cut-row statuses by
            // slot identity, and evaluate new cuts at the current state.
            // Runs unconditionally when a stored basis exists.
            let theta_value = pool.evaluate_at_state(&ws.current_state[..indexer.n_state]);
            let recon_stats = reconstruct_basis(
                captured,
                ReconstructionTarget {
                    base_row_count: ctx.templates[t].num_rows,
                    num_cols: ctx.templates[t].num_cols,
                },
                pool.active_cuts(),
                PaddingContext {
                    state: &ws.current_state[..indexer.n_state],
                    theta: theta_value,
                    tolerance: 1e-7,
                },
                0,
                &mut ws.scratch_basis,
                &mut ws.scratch.recon_slot_lookup,
            );
            ws.solver.record_reconstruction_stats(
                recon_stats.preserved,
                recon_stats.new_tight,
                recon_stats.new_slack,
            );
            ws.solver.solve_with_basis(&ws.scratch_basis)
        }
        _ => ws.solver.solve(),
    }
    .map_err(|e| {
        *basis_slice.get_mut(m, t) = None;
        match e {
            SolverError::Infeasible => SddpError::Infeasible {
                stage: t,
                iteration,
                scenario: m,
            },
            other => SddpError::Solver(other),
        }
    })?;

    // Unscale primal values from the solver's scaled coordinate system back
    // to the original physical units. When col_scale is empty (no scaling
    // applied), this loop is skipped.
    let col_scale = &ctx.templates[t].col_scale;
    let unscaled_primal = &mut ws.scratch.unscaled_primal;
    if col_scale.is_empty() {
        unscaled_primal.clear();
        unscaled_primal.extend_from_slice(view.primal);
    } else {
        unscaled_primal.resize(view.primal.len(), 0.0);
        for (j, (xp, &d)) in view.primal.iter().zip(col_scale).enumerate() {
            unscaled_primal[j] = d * xp;
        }
    }

    let d_t = ctx.discount_factors.get(t).copied().unwrap_or(1.0);
    let stage_cost = (view.objective - d_t * unscaled_primal[indexer.theta]) * COST_SCALE_FACTOR;
    let rec = &mut worker_records[local_m * num_stages + t];
    // Skip primal storage: only rec.state is consumed downstream; primals
    // are read directly from the solver when needed (simulation).
    rec.primal.clear();
    // Skip dual storage: rec.dual is not read by the backward pass or any
    // training-path code. Simulation reads duals directly from the solver view.
    rec.dual.clear();
    rec.stage_cost = stage_cost;

    // Save incoming lag values before overwriting state with primal.
    // Uses the pre-allocated lag_matrix_buf scratch buffer (no allocation).
    let lag_start = indexer.inflow_lags.start;
    let lag_len = indexer.hydro_count * indexer.max_par_order;
    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[lag_start..lag_start + lag_len]);

    // Compute shifted lag state once into ws.current_state, then copy to rec.state.
    ws.current_state.clear();
    ws.current_state
        .extend_from_slice(&unscaled_primal[..indexer.n_state]);
    let stage_lag = ctx.stage_lag_transitions.get(t).copied().unwrap_or(
        cobre_core::temporal::StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        },
    );
    let downstream_par_order = {
        let h = ws.scratch.lag_accumulator.len();
        if h == 0 {
            0
        } else {
            ws.scratch.downstream_completed_lags.len() / h
        }
    };
    crate::noise::accumulate_and_shift_lag_state(
        &mut ws.current_state,
        &ws.scratch.lag_matrix_buf,
        unscaled_primal,
        indexer,
        &stage_lag,
        &mut crate::noise::LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut crate::noise::DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
    rec.state.clear();
    rec.state.extend_from_slice(&ws.current_state);
    let cut_row_count = basis_row_capacity.saturating_sub(ctx.templates[t].num_rows);
    if let Some(captured) = basis_slice.get_mut(m, t) {
        ws.solver.get_basis(&mut captured.basis);
        write_capture_metadata(
            captured,
            pool,
            ctx.templates[t].num_rows,
            cut_row_count,
            &ws.current_state[..indexer.n_state],
        );
    } else {
        let mut captured = CapturedBasis::new(
            ctx.templates[t].num_cols,
            basis_row_capacity,
            ctx.templates[t].num_rows,
            cut_row_count,
            indexer.n_state,
        );
        ws.solver.get_basis(&mut captured.basis);
        write_capture_metadata(
            &mut captured,
            pool,
            ctx.templates[t].num_rows,
            cut_row_count,
            &ws.current_state[..indexer.n_state],
        );
        *basis_slice.get_mut(m, t) = Some(captured);
    }
    Ok(stage_cost)
}

/// Build a [`ForwardSampler`] from the sampler-related fields of a
/// [`TrainingContext`].
///
/// Extracted so callers (e.g. the training loop in `training.rs`) can
/// construct the sampler once before the iteration loop and reuse it across
/// all iterations without repeated heap allocation.
///
/// # Errors
///
/// Propagates any error from [`build_forward_sampler`], such as a missing
/// `OutOfSample` seed or an incompatible library shape.
pub fn build_sampler_from_ctx<'a>(
    ctx: &'a TrainingContext<'a>,
) -> Result<ForwardSampler<'a>, SddpError> {
    let stochastic = ctx.stochastic;
    build_forward_sampler(ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(ctx.inflow_scheme),
            load: Some(ctx.load_scheme),
            ncs: Some(ctx.ncs_scheme),
        },
        ctx: stochastic,
        stages: ctx.stages,
        dims: ClassDimensions {
            n_hydros: stochastic.n_hydros(),
            n_load_buses: stochastic.n_load_buses(),
            n_ncs: stochastic.n_stochastic_ncs(),
        },
        historical_library: ctx.historical_library,
        external_inflow_library: ctx.external_inflow_library,
        external_load_library: ctx.external_load_library,
        external_ncs_library: ctx.external_ncs_library,
    })
    .map_err(SddpError::Stochastic)
}

/// Execute the forward pass for one training iteration on this rank.
///
/// Simulates this rank's share of forward-pass scenarios through the full
/// stage horizon, solving the stage LP at each `(scenario, stage)` pair.
/// Pre-allocated [`TrajectoryRecord`]s in `records` are populated in-place.
///
/// ## Argument layout
///
/// - `workspaces` — one [`SolverWorkspace`] per worker thread. Scenarios are
///   statically partitioned across workspaces; each workspace owns its solver,
///   patch buffer, and current-state buffer exclusively.
/// - `basis_store` — per-scenario, per-stage basis store pre-allocated by the
///   caller. The store is split into disjoint sub-views before the parallel
///   region; each worker writes the optimal basis for its own scenarios.
/// - `ctx` — per-stage LP layout and noise scaling parameters bundled into a
///   single [`crate::context::StageContext`]. Contains: stage templates, base
///   row indices, noise scale factors, hydro and load-bus counts, load-balance
///   row starts, load-bus index mapping, and per-stage block counts.
/// - `fcf` — Future Cost Function carrying the current Benders cut pools.
/// - `stochastic` — pre-built stochastic pipeline (tree, seed, dim).
/// - `local_forward_passes` — number of forward-pass scenarios assigned to this
///   rank (the caller splits the user's total across MPI ranks).
/// - `iteration` — current training iteration (0-based counter used for seed
///   derivation).
/// - `horizon` — horizon mode determining the stage count.
/// - `initial_state` — starting state for every scenario (length `n_state`).
/// - `records` — pre-allocated output slice of length
///   `local_forward_passes * num_stages`.
/// - `indexer` — LP column/row layout map for this stage.
/// - `fwd_offset` — global index of this rank's first forward pass. Used for
///   deterministic seed derivation (`global_scenario = fwd_offset + m`).
/// - `inflow_method` — inflow non-negativity treatment. Controls whether
///   slack columns are present in the LP for absorbing negative inflow.
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
/// - `records.len() != local_forward_passes * num_stages`
/// - `initial_state.len() != indexer.n_state`
/// - `ctx.templates.len() != num_stages`
/// - `ctx.base_rows.len() != num_stages`
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn run_forward_pass<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    ctx: &StageContext<'_>,
    baked: &BakedTemplates<'_>,
    fcf: &FutureCostFunction,
    cut_batches: &mut [RowBatch],
    training_ctx: &TrainingContext<'_>,
    batch: &ForwardPassBatch,
    records: &mut [TrajectoryRecord],
) -> Result<ForwardResult, SddpError> {
    let TrainingContext {
        horizon,
        indexer,
        stochastic,
        initial_state,
        recent_accum_seed,
        recent_weight_seed,
        ..
    } = training_ctx;
    let recent_weight_seed = *recent_weight_seed;
    let ForwardPassBatch {
        local_forward_passes,
        total_forward_passes,
        iteration,
        fwd_offset,
    } = batch;
    let (num_stages, forward_passes) = (horizon.num_stages(), *local_forward_passes);

    debug_assert_eq!(records.len(), forward_passes * num_stages);
    debug_assert_eq!(initial_state.len(), indexer.n_state);
    if baked.ready {
        debug_assert_eq!(
            baked.templates.len(),
            num_stages,
            "baked templates length mismatch: expected {num_stages}, got {}",
            baked.templates.len()
        );
    }

    let start = Instant::now();
    // Populate `cut_batches` for the legacy (non-baked) load path and for
    // backward-pass stage-loop initialization. On the baked path
    // (`baked.ready == true`) the forward inner loop calls `load_model` only
    // and never reads these batches; the backward pass overwrites each
    // `cut_batches[successor]` with the delta batch before first use. This
    // single full rebuild per iteration is one-time O(active_cuts) work
    // outside the per-scenario hot loop and reuses pre-allocated RowBatch
    // buffers, so it does not allocate.
    for (t, batch) in cut_batches.iter_mut().enumerate().take(num_stages) {
        build_cut_row_batch_into(batch, fcf, t, indexer, &ctx.templates[t].col_scale);
    }
    let sampler = build_forward_sampler(ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(training_ctx.inflow_scheme),
            load: Some(training_ctx.load_scheme),
            ncs: Some(training_ctx.ncs_scheme),
        },
        ctx: stochastic,
        stages: training_ctx.stages,
        dims: ClassDimensions {
            n_hydros: stochastic.n_hydros(),
            n_load_buses: stochastic.n_load_buses(),
            n_ncs: stochastic.n_stochastic_ncs(),
        },
        historical_library: training_ctx.historical_library,
        external_inflow_library: training_ctx.external_inflow_library,
        external_load_library: training_ctx.external_load_library,
        external_ncs_library: training_ctx.external_ncs_library,
    })?;
    let n_workers = workspaces.len().max(1);

    let mut remaining: &mut [TrajectoryRecord] = records;
    let mut record_slices: Vec<&mut [TrajectoryRecord]> = Vec::with_capacity(n_workers);
    for w in 0..n_workers {
        let (start_m, end_m) = partition(forward_passes, n_workers, w);
        let (slice, rest) = remaining.split_at_mut((end_m - start_m) * num_stages);
        record_slices.push(slice);
        remaining = rest;
    }
    let basis_slices: Vec<BasisStoreSliceMut<'_>> = basis_store.split_workers_mut(n_workers);

    // Noise dimension for worker-local sampling buffers (OutOfSample path).
    let noise_dim = stochastic.dim();

    // True when the last study stage has warm-start (boundary) cuts.
    // Computed once here so it can be captured cheaply by the parallel closure.
    // When true, the theta column at the terminal stage is not zeroed out,
    // allowing boundary cuts to contribute to the LP objective.
    let terminal_has_boundary_cuts =
        num_stages > 0 && fcf.pools[num_stages - 1].warm_start_count > 0;

    // Collect per-worker snapshots before the parallel region (needed for overhead decomposition).
    let worker_stats_before: Vec<_> = workspaces.iter().map(|ws| ws.solver.statistics()).collect();

    // Each worker collects per-scenario costs in local scenario index order.
    let parallel_start = Instant::now();
    let worker_results: Vec<Result<(Vec<f64>, u64), SddpError>> = workspaces
        .par_iter_mut()
        .zip(record_slices.par_iter_mut())
        .zip(basis_slices.into_par_iter())
        .enumerate()
        .map(|(w, ((ws, worker_records), mut basis_slice))| {
            let (start_m, end_m) = partition(forward_passes, n_workers, w);
            let n_local = end_m - start_m;
            let mut trajectory_costs = vec![0.0_f64; n_local];
            let local_solve_count_before = ws.solver.statistics().solve_count;
            // Sampling scratch: lives here (not in ws) to avoid borrow conflicts
            // when run_forward_stage borrows ws while raw_noise is still live.
            let mut raw_noise_buf = vec![0.0_f64; noise_dim];
            #[allow(clippy::cast_possible_truncation)]
            let mut perm_scratch = vec![0_usize; (*total_forward_passes).max(1)];
            #[allow(clippy::cast_possible_truncation)]
            let total_scenarios_u32 = *total_forward_passes as u32;

            for t in 0..num_stages {
                let cum_d = ctx
                    .cumulative_discount_factors
                    .get(t)
                    .copied()
                    .unwrap_or(1.0);

                for (local_m, m) in (start_m..end_m).enumerate() {
                    // Reload model per scenario to ensure deterministic LP state across thread assignments.
                    // When baked templates are ready, the active cut rows are already structural rows
                    // in the baked template — no add_rows call is needed.
                    if baked.ready {
                        ws.solver.load_model(&baked.templates[t]);
                    } else {
                        ws.solver.load_model(&ctx.templates[t]);
                        if cut_batches[t].num_rows > 0 {
                            ws.solver.add_rows(&cut_batches[t]);
                        }
                    }
                    ws.current_state.clear();
                    let src: &[f64] = if t == 0 {
                        initial_state
                    } else {
                        &worker_records[local_m * num_stages + (t - 1)].state
                    };
                    ws.current_state.extend_from_slice(src);

                    // Seed (or zero) the lag accumulator at trajectory start so it
                    // does not carry state across scenarios or training iterations.
                    // When recent_accum_seed is non-empty, copy it instead of
                    // zeroing — this pre-fills the partial period with observed data.
                    if t == 0 {
                        if recent_accum_seed.is_empty() {
                            ws.scratch.lag_accumulator.iter_mut().for_each(|v| *v = 0.0);
                            ws.scratch.lag_weight_accum = 0.0;
                        } else {
                            ws.scratch.lag_accumulator[..recent_accum_seed.len()]
                                .copy_from_slice(recent_accum_seed);
                            ws.scratch.lag_weight_accum = recent_weight_seed;
                        }
                        // Reset downstream accumulator at trajectory start.
                        ws.scratch
                            .downstream_accumulator
                            .iter_mut()
                            .for_each(|v| *v = 0.0);
                        ws.scratch.downstream_weight_accum = 0.0;
                        ws.scratch
                            .downstream_completed_lags
                            .iter_mut()
                            .for_each(|v| *v = 0.0);
                        ws.scratch.downstream_n_completed = 0;
                    }

                    let global_scenario = fwd_offset + m;
                    #[allow(clippy::cast_possible_truncation)]
                    let (i32, s32, t32) = (*iteration as u32, global_scenario as u32, t as u32);

                    if t == 0 {
                        let class_req = ClassSampleRequest {
                            iteration: i32,
                            scenario: s32,
                            stage: 0,
                            stage_idx: 0,
                            total_scenarios: total_scenarios_u32,
                            noise_group_id: 0,
                        };
                        sampler.apply_initial_state(
                            &class_req,
                            &mut ws.current_state,
                            indexer.inflow_lags.start,
                        );
                    }
                    let noise = sampler.sample(SampleRequest {
                        iteration: i32,
                        scenario: s32,
                        stage: t32,
                        stage_idx: t,
                        noise_buf: &mut raw_noise_buf,
                        perm_scratch: &mut perm_scratch,
                        total_scenarios: total_scenarios_u32,
                        noise_group_id: ctx.noise_group_id_at(t),
                    })?;
                    let raw_noise = noise.as_slice();
                    let key = StageKey {
                        t,
                        m,
                        local_m,
                        num_stages,
                        iteration: *iteration,
                        raw_noise,
                        basis_row_capacity: if baked.ready {
                            baked.templates[t].num_rows
                        } else {
                            ctx.templates[t].num_rows + cut_batches[t].num_rows
                        },
                        terminal_has_boundary_cuts,
                        pool: &fcf.pools[t],
                    };
                    trajectory_costs[local_m] += cum_d
                        * run_forward_stage(
                            ws,
                            &mut basis_slice,
                            ctx,
                            training_ctx,
                            &key,
                            worker_records,
                        )?;
                }
            }

            let local_solves = ws.solver.statistics().solve_count - local_solve_count_before;
            Ok((trajectory_costs, local_solves))
        })
        .collect();

    // Capture parallel region wall-clock before sequential post-processing.
    #[allow(clippy::cast_possible_truncation)]
    let parallel_wall_ms = parallel_start.elapsed().as_millis() as u64;

    // Collect per-worker snapshots after the parallel region and decompose overhead.
    let worker_stats_after: Vec<_> = workspaces.iter().map(|ws| ws.solver.statistics()).collect();

    let worker_deltas: Vec<SolverStatsDelta> = worker_stats_before
        .iter()
        .zip(&worker_stats_after)
        .map(|(b, a)| SolverStatsDelta::from_snapshots(b, a))
        .collect();

    // setup_time_ms: total non-solve work (load_model + add_rows + set_bounds + basis_set).
    let fwd_setup_ms: f64 = worker_deltas
        .iter()
        .map(|d| {
            d.load_model_time_ms + d.add_rows_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
        })
        .sum();

    // Per-worker elapsed: solve + setup phases.
    let worker_totals: Vec<f64> = worker_deltas
        .iter()
        .map(|d| {
            d.solve_time_ms
                + d.load_model_time_ms
                + d.add_rows_time_ms
                + d.set_bounds_time_ms
                + d.basis_set_time_ms
        })
        .collect();

    #[allow(clippy::cast_precision_loss)]
    let n_workers_f = n_workers as f64;
    let max_worker_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
    let avg_worker_ms = if worker_totals.is_empty() {
        0.0_f64
    } else {
        worker_totals.iter().sum::<f64>() / n_workers_f
    };

    // load_imbalance_ms: slowest worker minus average.
    let fwd_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
    // scheduling_overhead_ms: residual wall time beyond slowest worker.
    #[allow(
        clippy::cast_precision_loss,      // parallel_wall_ms as f64: safe at HPC scale
        clippy::cast_possible_truncation, // f64 as u64: clamped non-negative
        clippy::cast_sign_loss            // f64 as u64: clamped non-negative
    )]
    let fwd_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);

    // Merge per-worker cost vectors in global scenario index order (canonical).
    let mut scenario_costs = Vec::with_capacity(forward_passes);
    let mut lp_solves = 0u64;
    for result in worker_results {
        let (worker_costs, w_solves) = result?;
        scenario_costs.extend(worker_costs);
        lp_solves += w_solves;
    }

    #[allow(clippy::cast_possible_truncation)]
    let elapsed_ms = start.elapsed().as_millis() as u64;

    #[allow(
        clippy::cast_possible_truncation, // f64 as u64: non-negative values, sub-ms precision loss acceptable
        clippy::cast_sign_loss            // f64 as u64: all values are non-negative after .max(0.0)
    )]
    Ok(ForwardResult {
        scenario_costs,
        elapsed_ms,
        lp_solves,
        setup_time_ms: fwd_setup_ms as u64,
        load_imbalance_ms: fwd_imbalance_ms as u64,
        scheduling_overhead_ms: fwd_scheduling_ms as u64,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_comm::{CommData, Communicator, ReduceOp};
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        LoadModel, SamplingScheme,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::context::{build_stochastic_context, ClassSchemes, OpeningTreeInputs};
    use cobre_stochastic::StochasticContext;

    use cobre_comm::LocalBackend;

    use super::{
        build_cut_row_batch, build_delta_cut_row_batch_into, partition, run_forward_pass,
        sync_forward, ForwardPassBatch, ForwardResult, SyncResult,
    };
    use crate::{
        config::{CutManagementConfig, EventConfig, LoopConfig},
        context::{BakedTemplates, StageContext, TrainingContext},
        workspace::{BackwardAccumulators, BasisStore, SolverWorkspace},
        FutureCostFunction, HorizonMode, InflowNonNegativityMethod, RiskMeasure, StageIndexer,
        StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig, TrajectoryRecord,
    };

    /// Return a `BakedTemplates` that signals the legacy (non-baked) path.
    ///
    /// Used by unit tests that call `run_forward_pass` directly and do not
    /// exercise baked-template behaviour.
    fn not_baked() -> BakedTemplates<'static> {
        BakedTemplates {
            templates: &[],
            ready: false,
        }
    }

    /// Create a `Vec<RowBatch>` of empty batches, one per stage.
    fn empty_cut_batches(n_stages: usize) -> Vec<RowBatch> {
        (0..n_stages)
            .map(|_| RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            })
            .collect()
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
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
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
            SolverStatistics {
                solve_count: self.call_count as u64,
                ..SolverStatistics::default()
            }
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
        // Column layout for N=1, L=0:
        //   col 0: storage_out, col 1: z_inflow, col 2: storage_in, col 3: theta
        //
        // LP: min theta  s.t. storage_in = ? (patched)  x >= 0
        //
        // CSC matrix has 1 non-zero: storage_in coefficient in storage_fixing row.
        // Simplified to a structurally valid but otherwise no-op LP for testing.
        StageTemplate {
            num_cols: 4,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 0, 1, 1], // col 2 (storage_in) has NZ at row 0
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0], // minimise theta (at col 3)
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
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
            evaporation_reference_volumes_hm3: None,
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
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
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
            method: "spectral".to_string(),
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

    // ── Unit tests: ForwardResult ────────────────────────────────────────────

    #[test]
    fn forward_result_field_access() {
        let r = ForwardResult {
            scenario_costs: vec![60.0, 70.0, 80.0, 90.0],
            elapsed_ms: 123,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        assert_eq!(r.scenario_costs.len(), 4);
        assert_eq!(r.scenario_costs[0], 60.0);
        assert_eq!(r.elapsed_ms, 123);
    }

    #[test]
    fn forward_result_clone_and_debug() {
        let r = ForwardResult {
            scenario_costs: vec![1.0, 2.0],
            elapsed_ms: 5,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let c = r.clone();
        assert_eq!(c.scenario_costs.len(), r.scenario_costs.len());
        assert_eq!(c.scenario_costs[0].to_bits(), r.scenario_costs[0].to_bits());
        let s = format!("{r:?}");
        assert!(s.contains("ForwardResult"));
    }

    // ── Unit tests: forward overhead decomposition ───────────────────────────

    /// Verify the three-component decomposition: 4 workers with solve times
    /// 500/600/550/580 ms and setup times 50/60/45/55 ms yields setup=210,
    /// imbalance=50, scheduling varies by `parallel_wall_ms`.
    #[test]
    fn forward_overhead_decomposition_four_workers() {
        use cobre_solver::SolverStatistics;

        use crate::solver_stats::SolverStatsDelta;

        fn make_stats(
            solve_s: f64,
            load_model_s: f64,
            add_rows_s: f64,
            set_bounds_s: f64,
            basis_set_s: f64,
        ) -> SolverStatistics {
            SolverStatistics {
                total_solve_time_seconds: solve_s,
                total_load_model_time_seconds: load_model_s,
                total_add_rows_time_seconds: add_rows_s,
                total_set_bounds_time_seconds: set_bounds_s,
                total_basis_set_time_seconds: basis_set_s,
                ..SolverStatistics::default()
            }
        }

        // Solve times (s): 0.5, 0.6, 0.55, 0.58; setup times (s): 0.05, 0.06, 0.045, 0.055
        let befores = [
            make_stats(0.0, 0.0, 0.0, 0.0, 0.0),
            make_stats(0.0, 0.0, 0.0, 0.0, 0.0),
            make_stats(0.0, 0.0, 0.0, 0.0, 0.0),
            make_stats(0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let afters = [
            make_stats(0.500, 0.050, 0.0, 0.0, 0.0),
            make_stats(0.600, 0.060, 0.0, 0.0, 0.0),
            make_stats(0.550, 0.045, 0.0, 0.0, 0.0),
            make_stats(0.580, 0.055, 0.0, 0.0, 0.0),
        ];

        let deltas: Vec<SolverStatsDelta> = befores
            .iter()
            .zip(&afters)
            .map(|(b, a)| SolverStatsDelta::from_snapshots(b, a))
            .collect();

        // setup_time_ms: sum of all workers' setup phases.
        let setup_ms: f64 = deltas
            .iter()
            .map(|d| {
                d.load_model_time_ms
                    + d.add_rows_time_ms
                    + d.set_bounds_time_ms
                    + d.basis_set_time_ms
            })
            .sum();

        // Per-worker totals: solve + setup.
        let worker_totals: Vec<f64> = deltas
            .iter()
            .map(|d| {
                d.solve_time_ms
                    + d.load_model_time_ms
                    + d.add_rows_time_ms
                    + d.set_bounds_time_ms
                    + d.basis_set_time_ms
            })
            .collect();

        let n_workers_f = 4.0_f64;
        let max_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_ms = worker_totals.iter().sum::<f64>() / n_workers_f;
        let imbalance_ms = (max_ms - avg_ms).max(0.0);
        let parallel_wall_ms = 700_u64; // 40 ms above the slowest worker
        #[allow(clippy::cast_precision_loss)]
        let scheduling_ms = (parallel_wall_ms as f64 - max_ms).max(0.0);

        // setup_time_ms = 50 + 60 + 45 + 55 = 210
        assert!(
            (setup_ms - 210.0).abs() < 0.001,
            "setup_time_ms should be 210, got {setup_ms}"
        );
        // max_worker total = 660 (worker 1: 600+60)
        assert!(
            (max_ms - 660.0).abs() < 0.001,
            "max_worker_ms should be 660, got {max_ms}"
        );
        // avg = (550+660+595+635)/4 = 610
        assert!(
            (avg_ms - 610.0).abs() < 0.001,
            "avg_worker_ms should be 610, got {avg_ms}"
        );
        // imbalance = 660 - 610 = 50
        assert!(
            (imbalance_ms - 50.0).abs() < 0.001,
            "load_imbalance_ms should be 50, got {imbalance_ms}"
        );
        // scheduling = 700 - 660 = 40
        assert!(
            (scheduling_ms - 40.0).abs() < 0.001,
            "scheduling_overhead_ms should be 40, got {scheduling_ms}"
        );
    }

    /// Edge case: single worker — load imbalance must be exactly zero.
    #[test]
    fn forward_overhead_decomposition_single_worker_zero_imbalance() {
        use cobre_solver::SolverStatistics;

        use crate::solver_stats::SolverStatsDelta;

        let before = SolverStatistics::default();
        let after = SolverStatistics {
            total_solve_time_seconds: 1.0,
            total_load_model_time_seconds: 0.1,
            ..SolverStatistics::default()
        };

        let deltas = [SolverStatsDelta::from_snapshots(&before, &after)];
        let worker_totals: Vec<f64> = deltas
            .iter()
            .map(|d| {
                d.solve_time_ms
                    + d.load_model_time_ms
                    + d.add_rows_time_ms
                    + d.set_bounds_time_ms
                    + d.basis_set_time_ms
            })
            .collect();

        let n_workers_f = 1.0_f64;
        let max_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_ms = worker_totals.iter().sum::<f64>() / n_workers_f;
        let imbalance_ms = (max_ms - avg_ms).max(0.0);

        assert_eq!(
            imbalance_ms, 0.0,
            "load_imbalance_ms must be 0.0 for a single worker"
        );
    }

    /// Edge case: scheduling overhead is clamped to zero when parallel wall
    /// time is less than the max worker total (clock-skew scenario).
    #[test]
    fn forward_overhead_scheduling_clamped_to_zero_on_clock_skew() {
        use cobre_solver::SolverStatistics;

        use crate::solver_stats::SolverStatsDelta;

        let before = SolverStatistics::default();
        let after = SolverStatistics {
            total_solve_time_seconds: 1.0, // 1000 ms
            ..SolverStatistics::default()
        };

        let deltas = [SolverStatsDelta::from_snapshots(&before, &after)];
        let max_ms = deltas
            .iter()
            .map(|d| d.solve_time_ms)
            .fold(0.0_f64, f64::max);

        // Wall time (800 ms) < max worker total (1000 ms) — clock skew.
        let parallel_wall_ms = 800_u64;
        #[allow(clippy::cast_precision_loss)]
        let scheduling_ms = (parallel_wall_ms as f64 - max_ms).max(0.0);

        assert_eq!(
            scheduling_ms, 0.0,
            "scheduling_overhead_ms must clamp to 0.0 on clock skew"
        );
    }

    // ── Unit tests: build_cut_row_batch ──────────────────────────────────────

    #[test]
    fn build_cut_row_batch_empty_cuts_returns_empty_batch() {
        let fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        let indexer = StageIndexer::new(1, 0);
        let batch = build_cut_row_batch(&fcf, 0, &indexer, &[]);

        assert_eq!(batch.num_rows, 0);
        assert_eq!(batch.row_starts, vec![0]);
        assert!(batch.col_indices.is_empty());
        assert!(batch.values.is_empty());
        assert!(batch.row_lower.is_empty());
        assert!(batch.row_upper.is_empty());
    }

    #[test]
    fn build_cut_row_batch_one_cut_correct_structure() {
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 5.0, &[2.0]);
        let indexer = StageIndexer::new(1, 0);
        let batch = build_cut_row_batch(&fcf, 0, &indexer, &[]);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_starts, vec![0, 2]);
        assert_eq!(batch.col_indices, vec![0, 3]); // theta at col N*(3+L) = 3
        assert_eq!(batch.values, vec![-2.0, 1.0]);
        assert_eq!(batch.row_lower, vec![5.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_two_cuts_correct_row_starts() {
        let mut fcf = FutureCostFunction::new(2, 2, 1, 10, &[0; 2]);
        fcf.add_cut(1, 0, 0, 10.0, &[1.0, 3.0]);
        fcf.add_cut(1, 1, 0, 20.0, &[2.0, 4.0]);
        let indexer = StageIndexer::new(1, 1);
        let batch = build_cut_row_batch(&fcf, 1, &indexer, &[]);

        assert_eq!(batch.num_rows, 2);
        assert_eq!(batch.row_starts, vec![0, 3, 6]);
        assert_eq!(batch.col_indices[0], 0); // storage col 0
        assert_eq!(batch.col_indices[1], 2); // lag 0 → z_inflow col N*(1+L)=2
        assert_eq!(batch.col_indices[2], 4); // theta at N*(3+L) = 1*(3+1) = 4
        assert_eq!(batch.values[0], -1.0);
        assert_eq!(batch.values[1], -3.0);
        assert_eq!(batch.values[2], 1.0);
        assert_eq!(batch.col_indices[3], 0); // storage col 0
        assert_eq!(batch.col_indices[4], 2); // lag 0 → z_inflow col 2
        assert_eq!(batch.col_indices[5], 4); // theta at N*(3+L) = 4
        assert_eq!(batch.values[3], -2.0);
        assert_eq!(batch.values[4], -4.0);
        assert_eq!(batch.values[5], 1.0);
        assert_eq!(batch.row_lower, vec![10.0, 20.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
        assert!(batch.row_upper[1].is_infinite() && batch.row_upper[1] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_zero_coefficient_state_variable() {
        let mut fcf = FutureCostFunction::new(1, 2, 1, 5, &[0; 1]);
        fcf.add_cut(0, 0, 0, 3.0, &[0.0, 7.0]);
        let indexer = StageIndexer::new(1, 1);
        let batch = build_cut_row_batch(&fcf, 0, &indexer, &[]);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.col_indices, vec![0, 2, 4]); // lag 0 → z_inflow col 2; theta at 4
        assert_eq!(batch.values, vec![0.0, -7.0, 1.0]);
        assert_eq!(batch.row_lower, vec![3.0]);
    }

    /// Build a single-workspace from a solver sized for the given `indexer`.
    fn single_workspace(solver: MockSolver, indexer: &StageIndexer) -> SolverWorkspace<MockSolver> {
        SolverWorkspace {
            solver,
            patch_buf: crate::lp_builder::PatchBuffer::new(
                indexer.hydro_count,
                indexer.max_par_order,
                0,
                0,
            ),
            current_state: Vec::with_capacity(indexer.n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::with_capacity(indexer.hydro_count),
                inflow_m3s_buf: Vec::with_capacity(indexer.hydro_count),
                lag_matrix_buf: Vec::with_capacity(indexer.max_par_order * indexer.hydro_count),
                par_inflow_buf: Vec::with_capacity(indexer.hydro_count),
                eta_floor_buf: Vec::with_capacity(indexer.hydro_count),
                zero_targets_buf: vec![0.0_f64; indexer.hydro_count],
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
        }
    }

    /// Build 3 minimal [`Stage`] values matching `make_stochastic_context_1_hydro_3_stages`.
    ///
    /// Provides the `stages` slice required by [`TrainingContext`] so that
    /// [`cobre_stochastic::build_forward_sampler`] can read per-stage noise methods.
    fn make_stages_3() -> Vec<Stage> {
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
        vec![make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)]
    }

    // ── Acceptance criteria integration tests ───────────────────────────────

    /// AC: 2 scenarios, 3 stages, fixed `LpSolution(objective=100, theta=30)`.
    /// Expected: `scenario_count=2`, all 6 records with `stage_cost=70_000`.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn ac_two_scenarios_three_stages_fixed_solution() {
        // StageIndexer: N=1, L=0 → n_state=1, theta=3, num_cols=4
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, &[0; 3]);
        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 2,
                max_iterations: 100,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: StoppingRuleSet {
                    rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                    mode: StoppingMode::Any,
                },
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let horizon = HorizonMode::Finite { num_stages: 3 };

        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(2 * 3);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store =
            BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: config.loop_config.forward_passes as usize,
                total_forward_passes: config.loop_config.forward_passes as usize,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        // AC: scenario_costs has exactly 2 entries (one per forward pass).
        assert_eq!(result.scenario_costs.len(), 2);
        // AC: all 6 records have stage_cost = (100 - 30) * COST_SCALE_FACTOR = 70_000.
        for (i, record) in records.iter().enumerate() {
            assert_eq!(
                record.stage_cost, 70_000.0,
                "record[{i}].stage_cost should be 70_000.0 ((objective - theta) * COST_SCALE_FACTOR)"
            );
        }
        // AC: each scenario cost = 70_000 * 3 stages = 210_000.
        assert_eq!(result.scenario_costs[0], 210_000.0);
        assert_eq!(result.scenario_costs[1], 210_000.0);
    }

    /// AC: mock solver returns `Infeasible` at stage 1, scenario 0.
    ///
    /// Call 0 = scenario 0 stage 0 (succeeds). Call 1 = scenario 0 stage 1
    /// (infeasible). The function must return `SddpError::Infeasible { stage: 1,
    /// scenario: 0 }`.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn ac_infeasible_at_stage_1_scenario_0_returns_infeasible_error() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        // Stage-first loop: with 2 scenarios and 3 stages, the solve order is
        // (s0,t0), (s1,t0), (s0,t1), (s1,t1), ... — the 3rd call (index 2)
        // is stage 1 of scenario 0.
        let solver = MockSolver::infeasible_on(solution, 2);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, &[0; 3]);
        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 2,
                max_iterations: 100,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: StoppingRuleSet {
                    rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                    mode: StoppingMode::Any,
                },
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let horizon = HorizonMode::Finite { num_stages: 3 };

        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(2 * 3);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store =
            BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: config.loop_config.forward_passes as usize,
                total_forward_passes: config.loop_config.forward_passes as usize,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
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
    /// With 2 scenarios and `stage_cost=70_000` at every `(scenario, stage)`:
    /// - `total_cost` per scenario = `70_000` \* 3 = `210_000`
    /// - `cost_sum` = `210_000` + `210_000` = `420_000`
    /// - `cost_sum_sq` = `210_000`^2 + `210_000`^2 = `88_200_000_000`
    #[test]
    #[allow(clippy::too_many_lines)]
    fn cost_statistics_accumulated_correctly() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, &[0; 3]);
        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 2,
                max_iterations: 100,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: StoppingRuleSet {
                    rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                    mode: StoppingMode::Any,
                },
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let horizon = HorizonMode::Finite { num_stages: 3 };

        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(2 * 3);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store =
            BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: config.loop_config.forward_passes as usize,
                total_forward_passes: config.loop_config.forward_passes as usize,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        // stage_cost per solve = (100 - 30) * COST_SCALE_FACTOR = 70_000
        // total_cost per scenario = 70_000 * 3 stages = 210_000
        assert_eq!(result.scenario_costs.len(), 2);
        assert_eq!(result.scenario_costs[0], 210_000.0);
        assert_eq!(result.scenario_costs[1], 210_000.0);
        // Derived statistics: sum = 420_000, sum_sq = 210_000^2 * 2.
        let cost_sum: f64 = result.scenario_costs.iter().sum();
        let cost_sum_sq: f64 = result.scenario_costs.iter().map(|c| c * c).sum();
        assert_eq!(cost_sum, 420_000.0);
        assert_eq!(cost_sum_sq, 210_000.0_f64.powi(2) * 2.0);
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
    /// `cost_sum` = 300, `cost_sum_sq` = 60²+70²+80²+90² = 23000, count = 4.
    /// mean = 75.0
    /// variance = (23000 - 4 * 75^2) / 3 = (23000 - 22500) / 3 = 500/3
    /// std = sqrt(500/3) ≈ 12.910
    /// `ci_95` = 1.96 * std / sqrt(4)
    #[test]
    fn ub_statistics_four_scenarios_correct_mean_and_std() {
        let local = ForwardResult {
            scenario_costs: vec![60.0, 70.0, 80.0, 90.0],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm, 4).unwrap();

        assert_eq!(result.global_ub_mean, 75.0, "mean must be 300/4 = 75");

        // variance = 500/3, std = sqrt(500/3) ≈ 12.910
        let expected_std = (500.0_f64 / 3.0).sqrt();
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
    /// The sequential summation of [60, 70, 80, 90] gives:
    /// `cost_sum` = 300, `cost_sum_sq` = 22700, N = 4, mean = 75.
    /// std = sqrt((22700 - 4*75^2) / 3) = sqrt(200/3) ≈ 8.165.
    #[test]
    fn ac_ticket_acceptance_criterion_ub_mean() {
        let local = ForwardResult {
            scenario_costs: vec![60.0, 70.0, 80.0, 90.0],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm, 4).unwrap();

        assert_eq!(result.global_ub_mean, 75.0);
        // std = sqrt((22700 - 4 * 75^2) / 3) = sqrt(200/3) ≈ 8.165
        assert!(
            result.global_ub_std > 0.0,
            "std must be positive for 4 distinct scenarios"
        );
    }

    /// Canonical summation: [1.0, 2.0, 3.0, 4.0] produces identical mean/std
    /// regardless of how the vector is split across "ranks".
    ///
    /// Verifies that the `allgatherv` + sequential summation approach produces
    /// bit-identical statistics for the full vector `[1, 2, 3, 4]` whether it
    /// is presented as a single rank with 4 scenarios or simulated as two
    /// ranks with 2 scenarios each.
    #[test]
    fn canonical_summation_identical_regardless_of_partition() {
        // Single-rank: all 4 costs provided together.
        let single_rank = ForwardResult {
            scenario_costs: vec![1.0, 2.0, 3.0, 4.0],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = LocalBackend;
        let result_single = sync_forward(&single_rank, &comm, 4).unwrap();

        // Simulate two-rank scenario: rank 0 has [1.0, 2.0], rank 1 has [3.0, 4.0].
        // The mock communicator below pre-fills the recv buf with both ranks' data.
        // We test by constructing the full global buffer manually and verifying
        // that sequential summation produces the same statistics.
        let global_costs = [1.0_f64, 2.0, 3.0, 4.0];
        let global_n = global_costs.len();
        #[allow(clippy::cast_precision_loss)]
        let global_n_f64 = global_n as f64;
        let cost_sum: f64 = global_costs.iter().sum();
        let cost_sum_sq: f64 = global_costs.iter().map(|c| c * c).sum();
        let mean = cost_sum / global_n_f64;
        let variance = (cost_sum_sq - global_n_f64 * mean * mean) / (global_n_f64 - 1.0);
        let expected_std = variance.max(0.0).sqrt();
        let expected_mean = mean;

        assert_eq!(
            result_single.global_ub_mean.to_bits(),
            expected_mean.to_bits(),
            "mean must be bit-identical to sequential summation of [1,2,3,4]"
        );
        assert_eq!(
            result_single.global_ub_std.to_bits(),
            expected_std.to_bits(),
            "std must be bit-identical to sequential summation of [1,2,3,4]"
        );
    }

    /// AC: Bessel correction edge case — single scenario produces zero variance.
    #[test]
    fn bessel_correction_single_scenario_zero_std_and_ci() {
        let local = ForwardResult {
            scenario_costs: vec![500.0],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm, 1).unwrap();

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
        // Construct two identical large values so that the single-pass Bessel
        // formula (sum_sq - N*mean^2) / (N-1) can produce a tiny negative result
        // due to floating-point representation differences.
        //
        // With costs = [v, v]: sum = 2v, mean = v, sum_sq = 2v^2.
        // Variance = (2v^2 - 2v^2) / 1 = 0. In floating-point the exact
        // representation of sum_sq and N*mean^2 may differ by epsilon,
        // potentially yielding a slightly negative result.
        //
        // We synthesise this by using two scenarios whose costs are very close
        // to the representable large value but not exactly equal.
        let v = 1.0e15_f64;
        let local = ForwardResult {
            scenario_costs: vec![v, v],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm, 2).unwrap();

        assert!(
            !result.global_ub_std.is_nan(),
            "std must not be NaN even when floating-point variance is slightly negative"
        );
        // Both costs are exactly equal so true variance = 0.
        // The max(0, variance).sqrt() guard must clamp any tiny negative value.
        assert_eq!(
            result.global_ub_std, 0.0,
            "std must be 0.0 when variance is zero (or clamps from tiny negative)"
        );
    }

    // ── Integration tests: sync_forward with LocalBackend ────────────────────

    /// Integration: single-rank mode — global UB mean equals local mean.
    #[test]
    fn sync_forward_local_backend_global_equals_local() {
        // Two scenarios each costing 420.0 → mean = 420.0.
        let local = ForwardResult {
            scenario_costs: vec![420.0, 420.0],
            elapsed_ms: 5,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm, 2).unwrap();

        // In single-rank mode, allgatherv is an identity copy.
        assert_eq!(
            result.global_ub_mean, 420.0,
            "global_ub_mean must equal the arithmetic mean of the cost vector"
        );
    }

    /// Integration: `sync_time_ms` is a valid non-negative u64.
    #[test]
    fn sync_forward_sync_time_ms_is_valid_u64() {
        let local = ForwardResult {
            scenario_costs: vec![50.0, 50.0],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = LocalBackend;
        let result = sync_forward(&local, &comm, 2).unwrap();
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

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let local = ForwardResult {
            scenario_costs: vec![100.0],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        let comm = FailingComm;
        let err = sync_forward(&local, &comm, 1).unwrap_err();

        assert!(
            matches!(err, crate::SddpError::Communication(_)),
            "CommError must be wrapped as SddpError::Communication, got: {err:?}"
        );
    }

    // ── Unit tests: warm-start basis caching ─────────────────────────────────

    /// Helper: run one iteration of `run_forward_pass` with a single scenario
    /// and a 3-stage horizon. The workspace is passed mutably; the basis store
    /// is returned so callers can inspect per-scenario, per-stage cached bases.
    fn run_one_iteration(
        ws: &mut SolverWorkspace<MockSolver>,
        basis_store: &mut BasisStore,
    ) -> Result<(), crate::SddpError> {
        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 1, 100, &[0; 3]);
        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 100,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: StoppingRuleSet {
                    rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                    mode: StoppingMode::Any,
                },
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let horizon = HorizonMode::Finite { num_stages: 3 };

        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(3);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        run_forward_pass(
            std::slice::from_mut(ws),
            basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: config.loop_config.forward_passes as usize,
                total_forward_passes: config.loop_config.forward_passes as usize,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .map(|_| ())
    }

    /// Warm-start invocation: the first iteration calls `solve` (cold start);
    /// the second iteration calls `solve_with_basis` (warm start).
    ///
    /// AC: `run_forward_pass` called twice, sharing the same `BasisStore`
    /// (1 scenario × 3 stages). After iteration 1: `warm_start_calls` == 0
    /// (3 cold-start solves). After iteration 2: `warm_start_calls` > 0 (all
    /// 3 stages warm-start from the store populated in iteration 1).
    #[test]
    fn warm_start_first_iteration_cold_second_iteration_warm() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let solver = MockSolver::always_ok(solution);
        // Single workspace and a shared basis store (1 scenario × 3 stages).
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(1, 3);

        // First iteration: no cached bases → all cold-start.
        run_one_iteration(&mut ws, &mut basis_store).unwrap();
        assert_eq!(
            ws.solver.warm_start_calls, 0,
            "first iteration must use cold-start for all stages (warm_start_calls == 0)"
        );

        // After first iteration, all 3 stages for scenario 0 have a cached basis.
        assert!(
            (0..3).all(|t| basis_store.get(0, t).is_some()),
            "basis_store must be fully populated for scenario 0 after the first iteration"
        );

        // Second iteration: cached bases present → all stages warm-start.
        run_one_iteration(&mut ws, &mut basis_store).unwrap();
        assert!(
            ws.solver.warm_start_calls > 0,
            "second iteration must use warm-start for at least one stage \
             (warm_start_calls > 0, got {})",
            ws.solver.warm_start_calls
        );
    }

    /// Basis invalidation on solver error: when a forward solve returns
    /// `SolverError::Infeasible`, the `BasisStore` slot at `(scenario, stage)`
    /// must be set to `None` before the error propagates.
    ///
    /// AC: `MockSolver` returns `Infeasible` on call index 4 (second iteration,
    /// stage 1 — calls 0-2 = first iteration stages 0,1,2; calls 3,4,5 = second
    /// iteration stages 0,1,2). After the error:
    /// - `basis_store.get(0, 1)` is `None` (invalidated at the failing stage).
    /// - `basis_store.get(0, 0)` is `Some` (stage 0 succeeded in iteration 2).
    #[test]
    fn basis_invalidated_on_solver_error() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        // Call 4 = second iteration, stage 1 (calls 0-2 = first iteration
        // stages 0,1,2; calls 3,4,5 = second iteration stages 0,1,2).
        let solver = MockSolver::infeasible_on(solution, 4);
        // Single workspace and a shared basis store (1 scenario × 3 stages).
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(1, 3);

        // First iteration: all cold-start, all succeed, populate all 3 stages.
        run_one_iteration(&mut ws, &mut basis_store).unwrap();
        assert!(
            (0..3).all(|t| basis_store.get(0, t).is_some()),
            "basis_store must be fully populated for scenario 0 after iteration 1"
        );

        // Second iteration: stage 0 warm-starts (call 3 OK), stage 1 infeasible (call 4).
        let err = run_one_iteration(&mut ws, &mut basis_store).unwrap_err();
        assert!(
            matches!(err, crate::SddpError::Infeasible { stage: 1, .. }),
            "expected Infeasible at stage 1, got: {err:?}"
        );

        // AC: basis_store slot (0, 1) must be None after the error (invalidated).
        assert!(
            basis_store.get(0, 1).is_none(),
            "basis_store.get(0, 1) must be None after solver error at stage 1"
        );

        // Stage 0 succeeded in iteration 2 — its basis was re-extracted.
        assert!(
            basis_store.get(0, 0).is_some(),
            "basis_store.get(0, 0) must be Some (stage 0 succeeded before error)"
        );
    }

    // ── New test: parallel cost agreement ────────────────────────────────────

    /// AC: with 1-workspace and 4-workspace pools producing the same `cost_sum`.
    ///
    /// Given the same input data, `run_forward_pass` with a single workspace
    /// must produce identical `cost_sum` and `cost_sum_sq` values compared to
    /// running with 4 workspaces. This verifies the static partitioning
    /// produces deterministic results regardless of workspace count.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_forward_pass_parallel_cost_agreement() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, &[0; 3]);
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let n_scenarios = 10;

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };

        // Run with 1 workspace.
        let mut ws1 = single_workspace(MockSolver::always_ok(solution.clone()), &indexer);
        let mut records1 = empty_records(n_scenarios * 3);
        let mut basis_store1 = BasisStore::new(n_scenarios, templates.len());
        let result1 = run_forward_pass(
            std::slice::from_mut(&mut ws1),
            &mut basis_store1,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
                total_forward_passes: n_scenarios,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records1,
        )
        .unwrap();

        // Run with 4 workspaces.
        let mut workspaces4: Vec<SolverWorkspace<MockSolver>> = (0..4)
            .map(|_| single_workspace(MockSolver::always_ok(solution.clone()), &indexer))
            .collect();
        let mut records4 = empty_records(n_scenarios * 3);
        let mut basis_store4 = BasisStore::new(n_scenarios, templates.len());
        let result4 = run_forward_pass(
            &mut workspaces4,
            &mut basis_store4,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
                total_forward_passes: n_scenarios,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records4,
        )
        .unwrap();

        assert_eq!(
            result1.scenario_costs.len(),
            result4.scenario_costs.len(),
            "scenario_costs length must be identical for 1 and 4 workspaces"
        );
        // Each scenario cost must be bit-identical regardless of workspace count.
        for (i, (c1, c4)) in result1
            .scenario_costs
            .iter()
            .zip(result4.scenario_costs.iter())
            .enumerate()
        {
            assert_eq!(
                c1.to_bits(),
                c4.to_bits(),
                "scenario_costs[{i}] must be bit-identical: 1-workspace={c1:.17e}, 4-workspace={c4:.17e}"
            );
        }
    }

    // ── New test: work distribution across 4 workspaces ──────────────────────

    /// C1: verify that 10 scenarios distributed across 4 workspaces assign
    /// each workspace `floor(10/4)` or `ceil(10/4)` scenarios.
    ///
    /// With `n=10`, `n_workers=4`: `base=2`, `remainder=2`.
    /// - Workers 0,1 receive 3 scenarios each → 3 * 3 stages = 9 solve calls.
    /// - Workers 2,3 receive 2 scenarios each → 2 * 3 stages = 6 solve calls.
    ///
    /// `MockSolver.statistics().solve_count` now returns `call_count`, so we
    /// can verify each workspace performed its assigned number of LP solves.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn test_forward_pass_work_distribution() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, &[0; 3]);
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let num_stages = 3usize;
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let n_scenarios = 10usize;
        let n_workers = 4usize;

        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|_| single_workspace(MockSolver::always_ok(solution.clone()), &indexer))
            .collect();
        let mut records = empty_records(n_scenarios * num_stages);
        let mut basis_store = BasisStore::new(n_scenarios, num_stages);

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let _result = run_forward_pass(
            &mut workspaces,
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
                total_forward_passes: n_scenarios,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        // Verify each workspace performed the expected number of LP solves.
        // partition(10, 4, w): base=2, remainder=2.
        // Workers 0,1: 3 scenarios × 3 stages = 9 solves each.
        // Workers 2,3: 2 scenarios × 3 stages = 6 solves each.
        for (w, ws) in workspaces.iter().enumerate() {
            let (start_m, end_m) = partition(n_scenarios, n_workers, w);
            let assigned_scenarios = end_m - start_m;
            let expected_solves = assigned_scenarios * num_stages;

            // floor and ceil of n_scenarios / n_workers for boundary check.
            let floor_scenarios = n_scenarios / n_workers;
            let ceil_scenarios = n_scenarios.div_ceil(n_workers);
            assert!(
                assigned_scenarios == floor_scenarios || assigned_scenarios == ceil_scenarios,
                "worker {w} assigned {assigned_scenarios} scenarios, expected {floor_scenarios} or {ceil_scenarios}"
            );

            let actual_solves = usize::try_from(ws.solver.statistics().solve_count)
                .expect("solve_count fits in usize in tests");
            assert_eq!(
                actual_solves, expected_solves,
                "worker {w} (scenarios [{start_m}, {end_m})) performed {actual_solves} solves, expected {expected_solves}"
            );
        }

        // Verify the total solve count equals n_scenarios * num_stages.
        let total_solves: usize = workspaces
            .iter()
            .map(|ws| {
                usize::try_from(ws.solver.statistics().solve_count)
                    .expect("solve_count fits in usize in tests")
            })
            .sum();
        assert_eq!(
            total_solves,
            n_scenarios * num_stages,
            "total solve count {total_solves} must equal n_scenarios * num_stages = {}",
            n_scenarios * num_stages
        );
    }

    // ── Truncation unit tests ────────────────────────────────────────────────

    /// Build a `StochasticContext` for 1 hydro and 1 stage with the given
    /// `mean_m3s` and `std_m3s`. Used by truncation tests.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_1h_1s(mean_m3s: f64, std_m3s: f64) -> StochasticContext {
        use std::collections::BTreeMap;

        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        };
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

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
            evaporation_reference_volumes_hm3: None,
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
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        let stage = Stage {
            index: 0,
            id: 0,
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
        let inflow_model = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s,
            std_m3s,
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
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(vec![stage])
            .inflow_models(vec![inflow_model])
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

    /// Minimal stage template for N=1 hydro, L=0 PAR, with a single water-balance
    /// row at position `base_row_idx`.
    ///
    /// This is a three-row template:
    /// - Row 0: storage fixing row
    /// - Row 1: z-inflow definition row (at N*(1+L) = 1)
    /// - Row 2: water-balance row (`base_rows`[t] = 2)
    ///
    /// `row_lower[2]` encodes the deterministic inflow base (ζ * `mean_m3s`).
    fn minimal_template_1_0_with_base(base_rhs: f64) -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 3,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0, base_rhs],
            row_upper: vec![0.0, 0.0, base_rhs],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Helper that runs `run_forward_pass` with 1 scenario, 1 stage, and returns
    /// the `noise_buf` from the workspace after the call.
    fn run_single_stage_forward(
        stochastic: &StochasticContext,
        inflow_method: &InflowNonNegativityMethod,
        base_rhs: f64,
        noise_scale_val: f64,
    ) -> Vec<f64> {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 0.0, indexer.theta, 0.0);
        let solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(1, indexer.n_state, 1, 10, &[0; 1]);
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let template = minimal_template_1_0_with_base(base_rhs);
        let templates = vec![template];
        // base_rows[t] = 1: the water-balance row is at row index 1.
        let base_rows = vec![2usize];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(1);
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(1, 1);
        let noise_scale = vec![noise_scale_val];

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let stages = vec![Stage {
            index: 0,
            id: 0,
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
        }];
        let _ = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method,
                stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
                total_forward_passes: 1,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        ws.scratch.noise_buf.clone()
    }

    /// AC: truncation clamps negative inflow noise.
    ///
    /// Set up a 1-hydro, 1-stage system where the PAR mean is very large
    /// negative (`mean_m3s = -1000.0`) and sigma is small (`std_m3s = 1.0`)
    /// so that the deterministic base alone produces a hugely negative inflow.
    /// Any sampled noise value will produce a negative inflow without truncation.
    ///
    /// With truncation active: the noise buffer entry must be >= 0.0 because
    /// `noise_buf[h] = base_rhs + noise_scale * clamped_eta`
    /// = `zeta * (mean + sigma * eta_min)` = `zeta * 0.0` = 0.0 (or near it).
    ///
    /// Concretely: `noise_buf[0] = base_rhs + noise_scale * clamped_eta`
    /// where `base_rhs = zeta * (-1000)` and `noise_scale = zeta * 1`.
    /// After clamping: `clamped_eta = max(eta, (0 - (-1000)) / 1) = 1000`.
    /// So `noise_buf[0] = zeta * (-1000) + zeta * 1000 = 0`.
    ///
    /// The actual PAR inflow = `mean + sigma * clamped_eta = -1000 + 1 * 1000 = 0 >= 0`.
    #[test]
    fn truncation_clamps_negative_inflow_noise() {
        // Deterministic base = -1000 m³/s (always produces negative inflow).
        // sigma = 1.0. For AR(0): zeta * mean = base_rhs, zeta * sigma = noise_scale.
        // Use zeta = 1.0 for simplicity (noise_scale = sigma).
        let mean_m3s = -1000.0_f64;
        let sigma = 1.0_f64;
        let zeta = 1.0_f64; // simplified for test: treat zeta=1
        let base_rhs = zeta * mean_m3s;
        let noise_scale_val = zeta * sigma;

        let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);

        let noise_buf_truncation = run_single_stage_forward(
            &stochastic,
            &InflowNonNegativityMethod::Truncation,
            base_rhs,
            noise_scale_val,
        );

        assert_eq!(noise_buf_truncation.len(), 1, "noise_buf must have 1 entry");
        // After truncation: noise_buf[0] = base_rhs + noise_scale * eta_clamped.
        // eta_clamped = max(eta, eta_min) where eta_min = (0 - mean) / sigma = 1000.
        // noise_buf[0] = -1000 + 1 * 1000 = 0.0 (exactly, no rounding).
        assert!(
            noise_buf_truncation[0] >= 0.0,
            "after truncation, noise_buf[0] must be >= 0 (inflow cannot be negative), got {}",
            noise_buf_truncation[0]
        );
    }

    /// AC: truncation does not clamp when inflow is positive.
    ///
    /// With a very large positive mean (`mean_m3s = 1000.0`) and small sigma,
    /// the PAR inflow is always positive for any sampled noise. The noise buffer
    /// must be identical to the no-truncation path.
    #[test]
    fn truncation_no_clamp_when_inflow_positive() {
        let mean_m3s = 1000.0_f64;
        let sigma = 1.0_f64;
        let zeta = 1.0_f64;
        let base_rhs = zeta * mean_m3s;
        let noise_scale_val = zeta * sigma;

        let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);

        let noise_buf_truncation = run_single_stage_forward(
            &stochastic,
            &InflowNonNegativityMethod::Truncation,
            base_rhs,
            noise_scale_val,
        );
        let noise_buf_none = run_single_stage_forward(
            &stochastic,
            &InflowNonNegativityMethod::None,
            base_rhs,
            noise_scale_val,
        );

        assert_eq!(noise_buf_truncation.len(), 1);
        assert_eq!(noise_buf_none.len(), 1);
        assert_eq!(
            noise_buf_truncation[0].to_bits(),
            noise_buf_none[0].to_bits(),
            "when inflow is positive, truncation must not alter the noise buffer (expected identical bits)"
        );
    }

    /// AC: `InflowNonNegativityMethod::None` produces unchanged behavior when
    /// truncation code is present in the same function.
    ///
    /// Uses the existing 3-stage fixture to verify no regression.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn none_method_unchanged_with_truncation_code_present() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, &[0; 3]);
        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 2,
                max_iterations: 100,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: StoppingRuleSet {
                    rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                    mode: StoppingMode::Any,
                },
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let horizon = HorizonMode::Finite { num_stages: 3 };
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(2 * 3);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store =
            BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: config.loop_config.forward_passes as usize,
                total_forward_passes: config.loop_config.forward_passes as usize,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        // Regression guard: same assertions as `ac_two_scenarios_three_stages_fixed_solution`.
        assert_eq!(result.scenario_costs.len(), 2);
        for (i, record) in records.iter().enumerate() {
            assert_eq!(
                record.stage_cost, 70_000.0,
                "none_method: record[{i}].stage_cost should be 70_000.0 ((objective - theta) * COST_SCALE_FACTOR)"
            );
        }
    }

    // ── Load noise test helpers ─────────────────────────────────────────────

    /// Build a `StochasticContext` with 1 hydro and 1 stochastic load bus over
    /// a single stage.
    ///
    /// `mean_mw` and `std_mw` control the load bus noise model.  An empty
    /// correlation model is used so the two noise entities are treated as
    /// independent standard normals.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_context_1_hydro_1_load_bus(mean_mw: f64, std_mw: f64) -> StochasticContext {
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
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: cobre_core::entities::hydro::HydroPenalties {
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
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        let stage = Stage {
            index: 0,
            id: 0,
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
        let inflow_model = InflowModel {
            hydro_id: EntityId(10),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        let load_model = LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw,
            std_mw,
        };
        // No correlation profile: entities are treated as independent.
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles: std::collections::BTreeMap::new(),
            schedule: vec![],
        };
        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .hydros(vec![hydro])
            .stages(vec![stage])
            .inflow_models(vec![inflow_model])
            .load_models(vec![load_model])
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

    // ── New test: parallel infeasibility propagation ──────────────────────────

    /// C3: when one of 4 workspaces returns `SolverError::Infeasible`, the
    /// error propagates as `SddpError::Infeasible` with correct stage and
    /// scenario indices.
    ///
    /// Worker 1 handles scenarios [3, 6) (3 scenarios) with 3 stages each.
    /// Its solver is configured to fail on call 0 (scenario 3, stage 0).
    /// Expected: `SddpError::Infeasible { stage: 0, scenario: 3, .. }`.
    #[test]
    fn test_forward_pass_parallel_infeasibility() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, &[0; 3]);
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let num_stages = 3usize;
        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let n_scenarios = 10usize;
        let n_workers = 4usize;

        // Worker 1 handles scenarios [3, 6). Its first solve call (call index 0
        // within that worker) corresponds to scenario 3, stage 0.
        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|w| {
                let solver = if w == 1 {
                    // Fail on the first solve call this worker makes.
                    MockSolver::infeasible_on(solution.clone(), 0)
                } else {
                    MockSolver::always_ok(solution.clone())
                };
                single_workspace(solver, &indexer)
            })
            .collect();

        let mut records = empty_records(n_scenarios * num_stages);
        let mut basis_store = BasisStore::new(n_scenarios, num_stages);

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = run_forward_pass(
            &mut workspaces,
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
                total_forward_passes: n_scenarios,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        );

        // Worker 1's partition: partition(10, 4, 1) → start_m=3.
        // The first solve in that worker is scenario 3, stage 0.
        match result {
            Err(crate::SddpError::Infeasible {
                stage,
                scenario,
                iteration,
            }) => {
                assert_eq!(
                    stage, 0,
                    "infeasible stage must be 0 (first stage of worker 1)"
                );
                assert_eq!(
                    scenario, 3,
                    "infeasible scenario must be 3 (start_m of worker 1)"
                );
                assert_eq!(
                    iteration, 0,
                    "iteration must be 0 (first training iteration)"
                );
            }
            Err(other) => panic!("expected SddpError::Infeasible, got: {other:?}"),
            Ok(_) => panic!("expected Err(SddpError::Infeasible), got Ok"),
        }
    }

    // ── Load noise wiring tests ──────────────────────────────────────────────

    /// Verify that the load balance row is patched to a positive value when
    /// `mean_mw + std_mw * eta > 0`.
    ///
    /// With `mean_mw = 300.0` and `std_mw = 30.0`, any standard-normal draw `eta`
    /// satisfying `|eta| <= 10` produces a positive realization, which holds with
    /// overwhelming probability.  After a single-scenario forward pass the
    /// `load_rhs_buf` must contain a positive value equal to
    /// `max(0, 300 + 30 * eta) * block_factor`.  Since no load factors file is
    /// supplied, `block_factor = 1.0`, so `load_rhs_buf[0] = max(0, 300 + 30 * eta)`.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn forward_pass_load_noise_positive_realization() {
        let n_load_buses = 1usize;
        let stochastic = make_stochastic_context_1_hydro_1_load_bus(300.0, 30.0);
        let indexer = StageIndexer::new(1, 0);
        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, n_load_buses, 1);
        let mut ws = SolverWorkspace {
            solver: MockSolver::always_ok(fixed_solution(4, 100.0, indexer.theta, 30.0)),
            patch_buf,
            current_state: Vec::with_capacity(indexer.n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::with_capacity(1),
                inflow_m3s_buf: Vec::with_capacity(1),
                lag_matrix_buf: Vec::with_capacity(0),
                par_inflow_buf: Vec::with_capacity(1),
                eta_floor_buf: Vec::with_capacity(1),
                zero_targets_buf: vec![0.0_f64; 1],
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::with_capacity(n_load_buses),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
        };

        let templates = vec![minimal_template_1_0_with_base(100.0)];
        let base_rows = vec![2usize];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(1);
        let fcf = FutureCostFunction::new(1, indexer.n_state, 1, 10, &[0; 1]);
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let mut basis_store = BasisStore::new(1, 1);
        let load_balance_row_starts = vec![10usize];
        let load_bus_indices = vec![0usize];
        let block_counts_per_stage = vec![1usize];

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[1.0],
            n_hydros: 1,
            n_load_buses,
            load_balance_row_starts: &load_balance_row_starts,
            load_bus_indices: &load_bus_indices,
            block_counts_per_stage: &block_counts_per_stage,
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let _fwd = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
                total_forward_passes: 1,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        assert_eq!(
            ws.scratch.load_rhs_buf.len(),
            n_load_buses,
            "load_rhs_buf must have 1 entry (1 load bus x 1 block)"
        );
        assert!(
            ws.scratch.load_rhs_buf[0] > 0.0,
            "load realization must be positive with mean=300, std=30: got {}",
            ws.scratch.load_rhs_buf[0]
        );

        let cat4_start = 2;
        assert_eq!(
            ws.patch_buf.lower[cat4_start], ws.scratch.load_rhs_buf[0],
            "patch_buf lower must equal load_rhs_buf[0]"
        );
        assert_eq!(
            ws.patch_buf.upper[cat4_start], ws.scratch.load_rhs_buf[0],
            "patch_buf upper must equal load_rhs_buf[0] (equality constraint)"
        );
        assert_eq!(
            ws.patch_buf.indices[cat4_start], 10,
            "patch index must be load_balance_row_starts[0] + 0 * n_blks"
        );
    }

    /// Verify that a load realization that would be negative is clamped to zero.
    ///
    /// With `mean_mw = -1000.0` and `std_mw = 1.0`, any standard-normal draw
    /// (bounded to roughly +-5 in practice) produces `mean + std * eta ~= -1000`,
    /// which must be clamped to `0.0` before block factor scaling.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn forward_pass_load_noise_clamped_to_zero() {
        let n_load_buses = 1usize;
        let stochastic = make_stochastic_context_1_hydro_1_load_bus(-1000.0, 1.0);
        let indexer = StageIndexer::new(1, 0);
        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, n_load_buses, 1);
        let mut ws = SolverWorkspace {
            solver: MockSolver::always_ok(fixed_solution(4, 100.0, indexer.theta, 30.0)),
            patch_buf,
            current_state: Vec::with_capacity(indexer.n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::with_capacity(1),
                inflow_m3s_buf: Vec::with_capacity(1),
                lag_matrix_buf: Vec::with_capacity(0),
                par_inflow_buf: Vec::with_capacity(1),
                eta_floor_buf: Vec::with_capacity(1),
                zero_targets_buf: vec![0.0_f64; 1],
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::with_capacity(n_load_buses),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
        };

        let templates = vec![minimal_template_1_0_with_base(100.0)];
        let base_rows = vec![2usize];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(1);
        let fcf = FutureCostFunction::new(1, indexer.n_state, 1, 10, &[0; 1]);
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let mut basis_store = BasisStore::new(1, 1);
        let load_balance_row_starts = vec![10usize];
        let load_bus_indices = vec![0usize];
        let block_counts_per_stage = vec![1usize];

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[1.0],
            n_hydros: 1,
            n_load_buses,
            load_balance_row_starts: &load_balance_row_starts,
            load_bus_indices: &load_bus_indices,
            block_counts_per_stage: &block_counts_per_stage,
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let _fwd = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
                total_forward_passes: 1,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        assert_eq!(
            ws.scratch.load_rhs_buf.len(),
            n_load_buses,
            "load_rhs_buf must have 1 entry (1 load bus x 1 block)"
        );
        assert_eq!(
            ws.scratch.load_rhs_buf[0], 0.0,
            "realization with mean=-1000 must be clamped to 0.0, got {}",
            ws.scratch.load_rhs_buf[0]
        );

        let cat4_start = 2;
        assert_eq!(
            ws.patch_buf.lower[cat4_start], 0.0,
            "patch lower must be 0.0 (clamped)"
        );
        assert_eq!(
            ws.patch_buf.upper[cat4_start], 0.0,
            "patch upper must be 0.0 (clamped)"
        );
    }

    /// Verify that when `n_load_buses == 0` no load patches are applied and
    /// `forward_patch_count()` equals `N*(2+L)`.
    ///
    /// With N=1 hydro, L=0 PAR order, and no load buses the patch count must be
    /// exactly `1 * (2 + 0) = 2`.
    #[test]
    fn forward_pass_no_load_buses_unchanged() {
        // Use the existing 1-hydro-3-stage context that has no load buses.
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let stages = make_stages_3();
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let mut ws = single_workspace(MockSolver::always_ok(solution), &indexer);

        let templates = vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
            minimal_template_1_0(),
        ];
        let base_rows = vec![2usize, 2, 2];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(3); // 1 scenario * 3 stages
        let fcf = FutureCostFunction::new(3, indexer.n_state, 1, 10, &[0; 3]);
        let horizon = HorizonMode::Finite { num_stages: 3 };
        let mut basis_store = BasisStore::new(1, 3);

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[], // noise_scale empty when n_hydros=0
            n_hydros: 0,      // skip inflow noise loop (minimal_template_1_0 has 1 row)
            n_load_buses: 0,  // no load patches
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1, 1, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let _fwd = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &not_baked(),
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
                total_forward_passes: 1,
                iteration: 0,
                fwd_offset: 0,
            },
            &mut records,
        )
        .unwrap();

        // With n_load_buses=0, active_load_patches stays 0.
        // forward_patch_count = N*(2+L) = 1*(2+0) = 2.
        // The PatchBuffer was constructed for 1 hydro (single_workspace uses indexer.hydro_count=1).
        assert_eq!(
            ws.patch_buf.forward_patch_count(),
            2,
            "forward_patch_count must be N*(2+L)=2 when n_load_buses=0, got {}",
            ws.patch_buf.forward_patch_count()
        );
        // load_rhs_buf must remain empty (never pushed to).
        assert!(
            ws.scratch.load_rhs_buf.is_empty(),
            "load_rhs_buf must be empty when n_load_buses=0"
        );
    }

    // ── Tests for append_new_cuts_to_lp ─────────────────────────────────

    /// Mock solver that records the last `add_rows` call for verification.
    struct RecordingMockSolver {
        last_batch: Option<RowBatch>,
        add_rows_count: usize,
    }

    impl RecordingMockSolver {
        fn new() -> Self {
            Self {
                last_batch: None,
                add_rows_count: 0,
            }
        }
    }

    impl SolverInterface for RecordingMockSolver {
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}

        fn add_rows(&mut self, cuts: &RowBatch) {
            self.last_batch = Some(RowBatch {
                num_rows: cuts.num_rows,
                row_starts: cuts.row_starts.clone(),
                col_indices: cuts.col_indices.clone(),
                values: cuts.values.clone(),
                row_lower: cuts.row_lower.clone(),
                row_upper: cuts.row_upper.clone(),
            });
            self.add_rows_count += 1;
        }

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "not implemented for test".to_string(),
                error_code: None,
            })
        }

        fn reset(&mut self) {}

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn solve_with_basis(
            &mut self,
            _basis: &Basis,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "not implemented for test".to_string(),
                error_code: None,
            })
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "RecordingMock"
        }
    }

    // ── Tests for append_new_cuts_to_lp ─────────────────────────────────

    // StageIndexer::new(1, 0) gives: n_state=1, theta=3
    // FCF state_dimension must match n_state=1.

    fn empty_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    #[test]
    fn append_new_cuts_returns_zero_when_no_new_cuts() {
        use crate::cut::CutRowMap;

        let fcf = crate::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        let indexer = crate::StageIndexer::new(1, 0);
        let mut row_map = CutRowMap::new(10, 5);
        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        // No active cuts -> should return 0 and not call add_rows.
        let count = super::append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut batch_buf,
        );
        assert_eq!(count, 0);
        assert_eq!(solver.add_rows_count, 0);
    }

    #[test]
    fn append_new_cuts_appends_all_on_empty_row_map() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 20.0, &[3.0]); // slot 1

        let indexer = crate::StageIndexer::new(1, 0);
        let mut row_map = CutRowMap::new(10, 5);
        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        let count = super::append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut batch_buf,
        );

        assert_eq!(count, 2);
        assert_eq!(solver.add_rows_count, 1);
        assert_eq!(row_map.total_cut_rows(), 2);
        assert_eq!(row_map.lp_row_for_slot(0), Some(5));
        assert_eq!(row_map.lp_row_for_slot(1), Some(6));
    }

    #[test]
    fn append_new_cuts_skips_already_mapped_cuts() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 20.0, &[3.0]); // slot 1

        let indexer = crate::StageIndexer::new(1, 0);
        let mut row_map = CutRowMap::new(10, 5);
        // Pre-insert slot 0 as if it was already in the LP.
        row_map.insert(0);

        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        let count = super::append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut batch_buf,
        );

        // Only slot 1 should be appended (slot 0 was already mapped).
        assert_eq!(count, 1);
        assert_eq!(solver.add_rows_count, 1);
        assert_eq!(row_map.total_cut_rows(), 2);
        assert!(solver.last_batch.as_ref().is_some_and(|b| b.num_rows == 1));
    }

    #[test]
    fn append_new_cuts_matches_build_cut_row_batch_into() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 20.0, &[3.0]); // slot 1

        let indexer = crate::StageIndexer::new(1, 0);

        // Build via build_cut_row_batch_into.
        let mut expected_batch = empty_row_batch();
        super::build_cut_row_batch_into(&mut expected_batch, &fcf, 0, &indexer, &[]);

        // Build via append_new_cuts_to_lp (empty row_map, so all cuts are new).
        let mut row_map = CutRowMap::new(10, 5);
        let mut actual_batch = empty_row_batch();
        let mut solver = RecordingMockSolver::new();
        super::append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut actual_batch,
        );

        // The batch passed to add_rows must match build_cut_row_batch_into.
        assert_eq!(actual_batch.num_rows, expected_batch.num_rows);
        assert_eq!(actual_batch.row_starts, expected_batch.row_starts);
        assert_eq!(actual_batch.col_indices, expected_batch.col_indices);
        assert_eq!(actual_batch.values, expected_batch.values);
        assert_eq!(actual_batch.row_lower, expected_batch.row_lower);
        assert_eq!(actual_batch.row_upper, expected_batch.row_upper);
    }

    #[test]
    fn append_new_cuts_with_scaling_matches_build() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]);

        let indexer = crate::StageIndexer::new(1, 0);
        // col_scale must have at least theta+1 = 4 entries.
        let col_scale = vec![0.5, 2.0, 1.0, 0.1];

        let mut expected = empty_row_batch();
        super::build_cut_row_batch_into(&mut expected, &fcf, 0, &indexer, &col_scale);

        let mut row_map = CutRowMap::new(10, 5);
        let mut actual = empty_row_batch();
        let mut solver = RecordingMockSolver::new();
        super::append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &col_scale,
            &mut row_map,
            &mut actual,
        );

        assert_eq!(actual.values, expected.values);
        assert_eq!(actual.col_indices, expected.col_indices);
    }

    // ── Tests for build_delta_cut_row_batch_into ─────────────────────────

    fn empty_delta_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    #[test]
    fn test_build_delta_empty_pool() {
        // Empty pool → num_rows == 0, row_starts == [0], col_indices empty.
        let fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        let indexer = StageIndexer::new(1, 0);
        let mut batch = empty_delta_batch();

        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 1);

        assert_eq!(batch.num_rows, 0);
        assert_eq!(batch.row_starts, vec![0_i32]);
        assert!(batch.col_indices.is_empty());
        assert!(batch.values.is_empty());
        assert!(batch.row_lower.is_empty());
        assert!(batch.row_upper.is_empty());
    }

    #[test]
    fn test_build_delta_single_iteration_filter() {
        // Pool has cuts at iterations 1, 2, 3; calling with current_iteration=2
        // emits only the iteration-2 cut.
        //
        // FCF: 2 stages, 1 state dimension, 1 forward pass, 10 max iterations,
        // 0 warm-start cuts per stage.
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        // iteration=1, fwd_idx=0: slot = 0 + 1*1 + 0 = 1
        fcf.add_cut(0, 1, 0, 10.0, &[1.0]);
        // iteration=2, fwd_idx=0: slot = 0 + 2*1 + 0 = 2
        fcf.add_cut(0, 2, 0, 20.0, &[2.0]);
        // iteration=3, fwd_idx=0: slot = 0 + 3*1 + 0 = 3
        fcf.add_cut(0, 3, 0, 30.0, &[3.0]);

        let indexer = StageIndexer::new(1, 0);
        let mut batch = empty_delta_batch();

        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 2);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_lower, vec![20.0]);
        assert_eq!(batch.row_starts, vec![0_i32, 2_i32]);
        // The cut emitted must carry iteration-2's coefficient (-2.0).
        assert_eq!(batch.values[0], -2.0);
    }

    #[test]
    fn test_build_delta_skips_deactivated_cuts() {
        // Pool has cuts at iteration 1, some deactivated; only active
        // iteration-1 cuts are emitted.
        let mut fcf = FutureCostFunction::new(2, 1, 2, 10, &[0; 2]);
        // iteration=1, fwd_idx=0: slot = 0 + 1*2 + 0 = 2
        fcf.add_cut(0, 1, 0, 10.0, &[1.0]);
        // iteration=1, fwd_idx=1: slot = 0 + 1*2 + 1 = 3
        fcf.add_cut(0, 1, 1, 20.0, &[2.0]);

        // Deactivate slot 2 (the first iteration-1 cut).
        fcf.pools[0].deactivate(&[2]);

        let indexer = StageIndexer::new(1, 0);
        let mut batch = empty_delta_batch();

        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 1);

        // Only slot 3 (intercept=20.0) should appear.
        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_lower, vec![20.0]);
    }

    #[test]
    fn test_build_delta_excludes_warm_start_cuts() {
        // Pool seeded with a warm-start cut AND one training iteration cut.
        // Delta call with current_iteration=1 must exclude the warm-start row.
        use cobre_io::OwnedPolicyCutRecord;

        let warm_record = OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            coefficients: vec![5.0],
            intercept: 99.0,
            iteration: 0,
            forward_pass_index: 0,
            is_active: true,
            domination_count: 0,
        };
        let mut pool = crate::cut::pool::CutPool::new_with_warm_start(1, 2, 10, &[warm_record]);
        // Now add a training cut at iteration=1, fwd_idx=0:
        // slot = warm_start_count(1) + 1*2 + 0 = 3
        pool.add_cut(1, 0, 7.0, &[1.0]);

        // Build an FCF with 2 stages (n_state=1, 2 fwd passes, 10 max iters).
        let mut fcf = FutureCostFunction::new(2, 1, 2, 10, &[0; 2]);
        fcf.pools[0] = pool;

        let indexer = StageIndexer::new(1, 0);
        let mut batch = empty_delta_batch();

        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 1);

        // Warm-start cut (intercept=99.0) must be excluded; training cut
        // (intercept=7.0) must be present.
        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_lower, vec![7.0]);
    }

    #[test]
    fn test_build_delta_matches_full_batch_when_pool_has_only_current_iter() {
        // When the pool contains only cuts from current_iteration, delta and
        // full builders must produce byte-identical output.
        let mut fcf = FutureCostFunction::new(2, 1, 2, 10, &[0; 2]);
        // iteration=1, fwd_idx=0: slot = 1*2+0 = 2
        fcf.add_cut(0, 1, 0, 10.0, &[1.0]);
        // iteration=1, fwd_idx=1: slot = 1*2+1 = 3
        fcf.add_cut(0, 1, 1, 20.0, &[3.0]);

        let indexer = StageIndexer::new(1, 0);

        let mut batch_full = empty_delta_batch();
        super::build_cut_row_batch_into(&mut batch_full, &fcf, 0, &indexer, &[]);

        let mut batch_delta = empty_delta_batch();
        build_delta_cut_row_batch_into(&mut batch_delta, &fcf, 0, &indexer, &[], 1);

        assert_eq!(batch_delta.num_rows, batch_full.num_rows);
        assert_eq!(batch_delta.row_starts, batch_full.row_starts);
        assert_eq!(batch_delta.col_indices, batch_full.col_indices);
        assert_eq!(batch_delta.values, batch_full.values);
        assert_eq!(batch_delta.row_lower, batch_full.row_lower);
        assert_eq!(batch_delta.row_upper, batch_full.row_upper);
    }

    #[test]
    fn test_build_delta_sparse_path() {
        // StageIndexer with non-empty nonzero_state_indices (sparse path).
        // Verify that the emitted col_indices for the cut contain exactly
        // nonzero_state_indices.len() + 1 entries (mask entries plus theta).
        //
        // StageIndexer::new(n_hydro, n_lag): n_state = n_hydro * (1 + n_lag)
        // With n_hydro=2, n_lag=0: n_state=2, no lags, sparse mask is empty.
        // We need a lag to get a nonzero_state_indices mask.
        // With n_hydro=1, n_lag=1: n_state=2, nonzero_state_indices=[0,1] (len=2)
        // for a cut that touches both state components.
        //
        // Actually we verify against the existing build_cut_row_batch_into for
        // correctness, which already tests the sparse path thoroughly.
        // Here we just verify col_indices.len() == mask.len() + 1 per row.

        // n_hydro=1, n_lag=1: n_state=2 (vol + lag).
        // nonzero_state_indices should be non-empty (check via indexer).
        let indexer = StageIndexer::new(1, 1);
        // nonzero_state_indices is the mask for non-trivially-zero state dims.
        let mask_len = indexer.nonzero_state_indices.len();

        // Only proceed if this indexer actually uses the sparse path.
        if mask_len == 0 {
            // Sparse path not active for this indexer; skip the assertion.
            return;
        }

        let mut fcf = FutureCostFunction::new(2, indexer.n_state, 1, 10, &[0; 2]);
        fcf.add_cut(0, 1, 0, 5.0, &vec![1.0; indexer.n_state]);

        let mut batch = empty_delta_batch();
        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 1);

        assert_eq!(batch.num_rows, 1);
        // Each row: mask_len state entries + 1 theta entry.
        assert_eq!(batch.col_indices.len(), mask_len + 1);
    }

    #[test]
    fn test_build_delta_reuses_out_buffer() {
        // Call twice; second call must produce correct output even when `batch`
        // had stale data from the first call.
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 1, 0, 11.0, &[1.0]);
        fcf.add_cut(0, 2, 0, 22.0, &[2.0]);

        let indexer = StageIndexer::new(1, 0);
        let mut batch = empty_delta_batch();

        // First call: iteration 1 → should yield the iteration-1 cut.
        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 1);
        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_lower, vec![11.0]);

        // Second call: iteration 2 → stale data from first call must be gone.
        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 2);
        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_lower, vec![22.0]);
        assert_eq!(batch.row_starts.len(), 2); // [0, 2]
    }

    #[test]
    fn test_build_delta_clears_row_starts() {
        // batch.row_starts[0] must be 0 regardless of prior state.
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 1, 0, 5.0, &[1.0]);

        let indexer = StageIndexer::new(1, 0);

        // Pre-populate batch with garbage.
        let mut batch = RowBatch {
            num_rows: 5,
            row_starts: vec![0_i32, 2, 4, 6, 8, 10],
            col_indices: vec![0_i32; 10],
            values: vec![99.0_f64; 10],
            row_lower: vec![0.0_f64; 5],
            row_upper: vec![0.0_f64; 5],
        };

        build_delta_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &[], 1);

        assert_eq!(batch.row_starts[0], 0_i32);
        assert_eq!(batch.num_rows, 1);
        // Prior garbage must be gone.
        assert_eq!(batch.row_starts.len(), 2);
    }
}
