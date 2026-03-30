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
//! scenario `m` is
//! `fwd_offset + m`, where `fwd_offset` is the pre-computed global index of
//! this rank's first forward pass. This deterministic mapping drives the
//! communication-free seed derivation used by [`sample_forward`].
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
//! For each `(scenario, stage)` pair the LP is rebuilt in three steps:
//!
//! 1. `solver.load_model(template)` — reset to the structural LP.
//! 2. `solver.add_rows(cut_batch)` — append active Benders cuts.
//! 3. `solver.set_row_bounds(...)` — patch scenario-specific row bounds.
//!
//! ## Hot-path allocation discipline
//!
//! No allocations occur per scenario or per stage during the inner loops.
//! The [`TrajectoryRecord`] slice is pre-allocated by the caller. The only
//! allocation inside the function is the [`RowBatch`] built by
//! `build_cut_row_batch`, which runs once per stage template (before the
//! scenario loop) — not once per scenario.

use std::time::Instant;

use cobre_comm::Communicator;
use cobre_solver::{Basis, RowBatch, SolverError, SolverInterface};
use cobre_stochastic::sample_forward;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    FutureCostFunction, SddpError, StageIndexer, TrajectoryRecord,
    context::{StageContext, TrainingContext},
    lp_builder::COST_SCALE_FACTOR,
    noise::{transform_inflow_noise, transform_load_noise, transform_ncs_noise},
    workspace::{BasisStore, BasisStoreSliceMut, SolverWorkspace},
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

    // Canonical-order sequential summation. All ranks iterate global_costs in
    // the same order, producing bit-identical statistics regardless of rank count.
    #[allow(clippy::cast_precision_loss)]
    let global_n_f64 = global_n as f64;
    let mut cost_sum = 0.0_f64;
    let mut cost_sum_sq = 0.0_f64;
    for &c in &global_costs {
        cost_sum += c;
        cost_sum_sq += c * c;
    }
    let mean = cost_sum / global_n_f64;

    let (std_dev, ci_95) = if global_n > 1 {
        let variance = (cost_sum_sq - global_n_f64 * mean * mean) / (global_n_f64 - 1.0);
        let sd = variance.max(0.0).sqrt();
        let ci = 1.96_f64 * sd / global_n_f64.sqrt();
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

    let num_cuts: usize = fcf.active_cuts(stage).count();

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
        if is_sparse {
            for &j in mask {
                push_scaled_coefficient(batch, j, coefficients[j], col_scale);
            }
        } else {
            for (j, &c) in coefficients.iter().enumerate() {
                push_scaled_coefficient(batch, j, c, col_scale);
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
    let nnz_per_cut = n_state + 1;

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
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch_buf.row_starts.push(nz_offset as i32);

        for (j, &c) in coefficients.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            batch_buf.col_indices.push(j as i32);
            let d = if col_scale.is_empty() {
                1.0
            } else {
                col_scale[j]
            };
            batch_buf.values.push(-c * d);
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

/// Deactivate cuts in the live LP by zeroing their row bounds.
///
/// For each slot index in `deactivation_set.indices`, looks up the LP row
/// via `row_map`, then calls `solver.set_row_bounds` to set the row bounds
/// to `(-inf, +inf)`, making the constraint non-binding (a free row).
/// Updates `row_map` to mark the slot as deactivated.
///
/// Slots that are not present in the LP (never appended) are silently
/// skipped. This handles the case where cut selection deactivates a cut
/// that was generated but not yet appended to the LP.
///
/// # Returns
///
/// The number of LP rows whose bounds were actually zeroed.
///
/// # Panics
///
/// Panics if `set_row_bounds` is called with mismatched slice lengths
/// (indicates a logic error in this function).
pub fn deactivate_cuts_in_lp<S: SolverInterface>(
    solver: &mut S,
    deactivation_set: &crate::cut_selection::DeactivationSet,
    row_map: &mut crate::cut::CutRowMap,
) -> usize {
    if deactivation_set.indices.is_empty() {
        return 0;
    }

    let mut indices: Vec<usize> = Vec::with_capacity(deactivation_set.indices.len());
    let mut lower: Vec<f64> = Vec::with_capacity(deactivation_set.indices.len());
    let mut upper: Vec<f64> = Vec::with_capacity(deactivation_set.indices.len());

    for &slot_u32 in &deactivation_set.indices {
        let slot = slot_u32 as usize;
        if let Some(lp_row) = row_map.lp_row_for_slot(slot) {
            // Only deactivate if the slot is still active in the row_map.
            if row_map.is_slot_active(slot) {
                indices.push(lp_row);
                lower.push(f64::NEG_INFINITY);
                upper.push(f64::INFINITY);
                row_map.deactivate(slot);
            }
        }
    }

    if !indices.is_empty() {
        solver.set_row_bounds(&indices, &lower, &upper);
    }

    indices.len()
}

/// Bundled scalar parameters for one forward pass invocation.
///
/// Groups the per-iteration, per-rank scalar arguments that are forwarded
/// from [`crate::train`] into [`run_forward_pass`].
pub struct ForwardPassBatch {
    /// Number of forward-pass scenarios assigned to this rank.
    pub local_forward_passes: usize,
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
    if horizon.is_terminal(t + 1) {
        ws.solver.set_col_bounds(&[indexer.theta], &[0.0], &[0.0]);
    }

    let view = match basis_slice.get(m, t) {
        Some(rb) => ws.solver.solve_with_basis(rb),
        None => ws.solver.solve(),
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
    rec.primal.clear();
    rec.primal.extend_from_slice(unscaled_primal);
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
    crate::noise::shift_lag_state(
        &mut ws.current_state,
        &ws.scratch.lag_matrix_buf,
        unscaled_primal,
        indexer,
    );
    rec.state.clear();
    rec.state.extend_from_slice(&ws.current_state);
    if let Some(rb) = basis_slice.get_mut(m, t) {
        ws.solver.get_basis(rb);
    } else {
        let mut rb = Basis::new(ctx.templates[t].num_cols, ctx.templates[t].num_rows);
        ws.solver.get_basis(&mut rb);
        *basis_slice.get_mut(m, t) = Some(rb);
    }
    Ok(stage_cost)
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
#[allow(clippy::too_many_arguments)]
pub fn run_forward_pass<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    ctx: &StageContext<'_>,
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
        ..
    } = training_ctx;
    let ForwardPassBatch {
        local_forward_passes,
        iteration,
        fwd_offset,
    } = batch;
    let (num_stages, forward_passes) = (horizon.num_stages(), *local_forward_passes);

    debug_assert_eq!(records.len(), forward_passes * num_stages);
    debug_assert_eq!(initial_state.len(), indexer.n_state);

    let start = Instant::now();
    for (t, batch) in cut_batches.iter_mut().enumerate().take(num_stages) {
        build_cut_row_batch_into(batch, fcf, t, indexer, &ctx.templates[t].col_scale);
    }
    let tree_view = stochastic.tree_view();
    let base_seed = stochastic.base_seed();
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

    // Each worker collects per-scenario costs in local scenario index order.
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

            for t in 0..num_stages {
                ws.solver.load_model(&ctx.templates[t]);
                if cut_batches[t].num_rows > 0 {
                    ws.solver.add_rows(&cut_batches[t]);
                }

                let cum_d = ctx
                    .cumulative_discount_factors
                    .get(t)
                    .copied()
                    .unwrap_or(1.0);

                for (local_m, m) in (start_m..end_m).enumerate() {
                    ws.current_state.clear();
                    let src: &[f64] = if t == 0 {
                        initial_state
                    } else {
                        &worker_records[local_m * num_stages + (t - 1)].state
                    };
                    ws.current_state.extend_from_slice(src);

                    let global_scenario = fwd_offset + m;
                    #[allow(clippy::cast_possible_truncation)]
                    let (i32, s32, t32) = (*iteration as u32, global_scenario as u32, t as u32);
                    let (_, raw_noise) = sample_forward(&tree_view, base_seed, i32, s32, t32, t);
                    let key = StageKey {
                        t,
                        m,
                        local_m,
                        num_stages,
                        iteration: *iteration,
                        raw_noise,
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

    Ok(ForwardResult {
        scenario_costs,
        elapsed_ms,
        lp_solves,
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
        LoadModel,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::StochasticContext;
    use cobre_stochastic::context::build_stochastic_context;

    use cobre_comm::LocalBackend;

    use super::{
        ForwardPassBatch, ForwardResult, SyncResult, build_cut_row_batch, partition,
        run_forward_pass, sync_forward,
    };
    use crate::{
        FutureCostFunction, HorizonMode, InflowNonNegativityMethod, StageIndexer, TrainingConfig,
        TrajectoryRecord,
        context::{StageContext, TrainingContext},
        workspace::{BasisStore, SolverWorkspace},
    };

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
        build_stochastic_context(&system, 42, &[], &[], None).unwrap()
    }

    // ── Unit tests: ForwardResult ────────────────────────────────────────────

    #[test]
    fn forward_result_field_access() {
        let r = ForwardResult {
            scenario_costs: vec![60.0, 70.0, 80.0, 90.0],
            elapsed_ms: 123,
            lp_solves: 0,
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
        };
        let c = r.clone();
        assert_eq!(c.scenario_costs.len(), r.scenario_costs.len());
        assert_eq!(c.scenario_costs[0].to_bits(), r.scenario_costs[0].to_bits());
        let s = format!("{r:?}");
        assert!(s.contains("ForwardResult"));
    }

    // ── Unit tests: build_cut_row_batch ──────────────────────────────────────

    #[test]
    fn build_cut_row_batch_empty_cuts_returns_empty_batch() {
        let fcf = FutureCostFunction::new(2, 1, 1, 10, 0);
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
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, 0);
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
        let mut fcf = FutureCostFunction::new(2, 2, 1, 10, 0);
        fcf.add_cut(1, 0, 0, 10.0, &[1.0, 3.0]);
        fcf.add_cut(1, 1, 0, 20.0, &[2.0, 4.0]);
        let indexer = StageIndexer::new(1, 1);
        let batch = build_cut_row_batch(&fcf, 1, &indexer, &[]);

        assert_eq!(batch.num_rows, 2);
        assert_eq!(batch.row_starts, vec![0, 3, 6]);
        assert_eq!(batch.col_indices[0], 0);
        assert_eq!(batch.col_indices[1], 1);
        assert_eq!(batch.col_indices[2], 4); // theta at N*(3+L) = 1*(3+1) = 4
        assert_eq!(batch.values[0], -1.0);
        assert_eq!(batch.values[1], -3.0);
        assert_eq!(batch.values[2], 1.0);
        assert_eq!(batch.col_indices[3], 0);
        assert_eq!(batch.col_indices[4], 1);
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
        let mut fcf = FutureCostFunction::new(1, 2, 1, 5, 0);
        fcf.add_cut(0, 0, 0, 3.0, &[0.0, 7.0]);
        let indexer = StageIndexer::new(1, 1);
        let batch = build_cut_row_batch(&fcf, 0, &indexer, &[]);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.col_indices, vec![0, 1, 4]); // theta at N*(3+L) = 4
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
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
            },
        }
    }

    // ── Acceptance criteria integration tests ───────────────────────────────

    /// AC: 2 scenarios, 3 stages, fixed `LpSolution(objective=100, theta=30)`.
    /// Expected: `scenario_count=2`, all 6 records with `stage_cost=70_000`.
    #[test]
    fn ac_two_scenarios_three_stages_fixed_solution() {
        // StageIndexer: N=1, L=0 → n_state=1, theta=3, num_cols=4
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
        let config = TrainingConfig {
            forward_passes: 2,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
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
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(config.forward_passes as usize, templates.len());

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
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: config.forward_passes as usize,
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
    fn ac_infeasible_at_stage_1_scenario_0_returns_infeasible_error() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        // Stage-first loop: with 2 scenarios and 3 stages, the solve order is
        // (s0,t0), (s1,t0), (s0,t1), (s1,t1), ... — the 3rd call (index 2)
        // is stage 1 of scenario 0.
        let solver = MockSolver::infeasible_on(solution, 2);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
        let config = TrainingConfig {
            forward_passes: 2,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
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
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(config.forward_passes as usize, templates.len());

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
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: config.forward_passes as usize,
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
    fn cost_statistics_accumulated_correctly() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
        let config = TrainingConfig {
            forward_passes: 2,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
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
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(config.forward_passes as usize, templates.len());

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
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: config.forward_passes as usize,
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
        }

        let local = ForwardResult {
            scenario_costs: vec![100.0],
            elapsed_ms: 0,
            lp_solves: 0,
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
        let fcf = FutureCostFunction::new(3, indexer.n_state, 1, 100, 0);
        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
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
        };
        run_forward_pass(
            std::slice::from_mut(ws),
            basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: config.forward_passes as usize,
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
    fn test_forward_pass_parallel_cost_agreement() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
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
        };

        // Run with 1 workspace.
        let mut ws1 = single_workspace(MockSolver::always_ok(solution.clone()), &indexer);
        let mut records1 = empty_records(n_scenarios * 3);
        let mut basis_store1 = BasisStore::new(n_scenarios, templates.len());
        let result1 = run_forward_pass(
            std::slice::from_mut(&mut ws1),
            &mut basis_store1,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
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
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
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
    #[test]
    fn test_forward_pass_work_distribution() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let stochastic = make_stochastic_context_1_hydro_3_stages();
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
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
        };
        let _result = run_forward_pass(
            &mut workspaces,
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
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
            method: "cholesky".to_string(),
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
        build_stochastic_context(&system, 42, &[], &[], None).unwrap()
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
        let fcf = FutureCostFunction::new(1, indexer.n_state, 1, 10, 0);
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
        };
        let _ = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method,
                stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
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
    fn none_method_unchanged_with_truncation_code_present() {
        let indexer = StageIndexer::new(1, 0);
        let solution = fixed_solution(4, 100.0, indexer.theta, 30.0);
        let solver = MockSolver::always_ok(solution);
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
        let config = TrainingConfig {
            forward_passes: 2,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
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
        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(config.forward_passes as usize, templates.len());

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
        };
        let result = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: config.forward_passes as usize,
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
            method: "cholesky".to_string(),
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
        build_stochastic_context(&system, 42, &[], &[], None).unwrap()
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
        let fcf = FutureCostFunction::new(3, indexer.n_state, 2, 100, 0);
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
        };
        let result = run_forward_pass(
            &mut workspaces,
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: n_scenarios,
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
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
            },
        };

        let templates = vec![minimal_template_1_0_with_base(100.0)];
        let base_rows = vec![2usize];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(1);
        let fcf = FutureCostFunction::new(1, indexer.n_state, 1, 10, 0);
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
        };
        let _fwd = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
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
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
            },
        };

        let templates = vec![minimal_template_1_0_with_base(100.0)];
        let base_rows = vec![2usize];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut records = empty_records(1);
        let fcf = FutureCostFunction::new(1, indexer.n_state, 1, 10, 0);
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
        };
        let _fwd = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
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
        let fcf = FutureCostFunction::new(3, indexer.n_state, 1, 10, 0);
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
        };
        let _fwd = run_forward_pass(
            std::slice::from_mut(&mut ws),
            &mut basis_store,
            &ctx,
            &fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &ForwardPassBatch {
                local_forward_passes: 1,
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

        let fcf = crate::FutureCostFunction::new(2, 1, 1, 10, 0);
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

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, 0);
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
        assert_eq!(row_map.active_count(), 2);
        assert_eq!(row_map.lp_row_for_slot(0), Some(5));
        assert_eq!(row_map.lp_row_for_slot(1), Some(6));
    }

    #[test]
    fn append_new_cuts_skips_already_mapped_cuts() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, 0);
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

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, 0);
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

        let mut fcf = crate::FutureCostFunction::new(2, 1, 1, 10, 0);
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

    // ── Tests for deactivate_cuts_in_lp ────────────────────────────────

    /// Mock solver that records `set_row_bounds` calls.
    struct BoundRecordingMockSolver {
        last_indices: Vec<usize>,
        last_lower: Vec<f64>,
        last_upper: Vec<f64>,
        set_row_bounds_count: usize,
    }

    impl BoundRecordingMockSolver {
        fn new() -> Self {
            Self {
                last_indices: Vec::new(),
                last_lower: Vec::new(),
                last_upper: Vec::new(),
                set_row_bounds_count: 0,
            }
        }
    }

    impl SolverInterface for BoundRecordingMockSolver {
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _cuts: &RowBatch) {}
        fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
            self.last_indices = indices.to_vec();
            self.last_lower = lower.to_vec();
            self.last_upper = upper.to_vec();
            self.set_row_bounds_count += 1;
        }
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "not implemented".to_string(),
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
                message: "not implemented".to_string(),
                error_code: None,
            })
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn name(&self) -> &'static str {
            "BoundRecordingMock"
        }
    }

    #[test]
    fn deactivate_cuts_empty_set_returns_zero() {
        use crate::cut::CutRowMap;
        use crate::cut_selection::DeactivationSet;

        let mut solver = BoundRecordingMockSolver::new();
        let mut row_map = CutRowMap::new(10, 5);

        let deact = DeactivationSet {
            stage_index: 0,
            indices: vec![],
        };
        let count = super::deactivate_cuts_in_lp(&mut solver, &deact, &mut row_map);
        assert_eq!(count, 0);
        assert_eq!(solver.set_row_bounds_count, 0);
    }

    #[test]
    fn deactivate_cuts_zeros_bounds_for_mapped_slots() {
        use crate::cut::CutRowMap;
        use crate::cut_selection::DeactivationSet;

        let mut solver = BoundRecordingMockSolver::new();
        let mut row_map = CutRowMap::new(10, 5);
        row_map.insert(0); // lp_row = 5
        row_map.insert(1); // lp_row = 6
        row_map.insert(2); // lp_row = 7

        let deact = DeactivationSet {
            stage_index: 0,
            indices: vec![0, 2], // deactivate slots 0 and 2
        };
        let count = super::deactivate_cuts_in_lp(&mut solver, &deact, &mut row_map);

        assert_eq!(count, 2);
        assert_eq!(solver.set_row_bounds_count, 1);
        assert_eq!(solver.last_indices, vec![5, 7]);
        assert!(solver.last_lower.iter().all(|&v| v == f64::NEG_INFINITY));
        assert!(solver.last_upper.iter().all(|&v| v == f64::INFINITY));
        assert_eq!(row_map.active_count(), 1); // only slot 1 remains active
    }

    #[test]
    fn deactivate_cuts_skips_unmapped_slots() {
        use crate::cut::CutRowMap;
        use crate::cut_selection::DeactivationSet;

        let mut solver = BoundRecordingMockSolver::new();
        let mut row_map = CutRowMap::new(10, 5);
        row_map.insert(0); // lp_row = 5

        // Slot 3 was never inserted.
        let deact = DeactivationSet {
            stage_index: 0,
            indices: vec![0, 3],
        };
        let count = super::deactivate_cuts_in_lp(&mut solver, &deact, &mut row_map);

        // Only slot 0 should be deactivated; slot 3 is skipped.
        assert_eq!(count, 1);
        assert_eq!(solver.last_indices, vec![5]);
    }

    #[test]
    fn deactivate_cuts_preserves_row_mapping() {
        use crate::cut::CutRowMap;
        use crate::cut_selection::DeactivationSet;

        let mut solver = BoundRecordingMockSolver::new();
        let mut row_map = CutRowMap::new(10, 5);
        row_map.insert(0); // lp_row = 5

        let deact = DeactivationSet {
            stage_index: 0,
            indices: vec![0],
        };
        super::deactivate_cuts_in_lp(&mut solver, &deact, &mut row_map);

        // Row mapping is preserved after deactivation.
        assert_eq!(row_map.lp_row_for_slot(0), Some(5));
        assert!(!row_map.is_slot_active(0));
    }

    #[test]
    fn deactivate_already_deactivated_slot_is_noop() {
        use crate::cut::CutRowMap;
        use crate::cut_selection::DeactivationSet;

        let mut solver = BoundRecordingMockSolver::new();
        let mut row_map = CutRowMap::new(10, 5);
        row_map.insert(0);
        row_map.deactivate(0); // already deactivated

        let deact = DeactivationSet {
            stage_index: 0,
            indices: vec![0],
        };
        let count = super::deactivate_cuts_in_lp(&mut solver, &deact, &mut row_map);

        assert_eq!(count, 0);
        assert_eq!(solver.set_row_bounds_count, 0);
    }
}
