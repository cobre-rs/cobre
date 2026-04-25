//! Per-phase delta of solver counters for LP statistics collection.
//!
//! [`SolverStatsDelta`] is computed from before/after [`SolverStatistics`] snapshots
//! taken around each training phase (forward pass, backward pass, lower bound evaluation).
//! The deltas are stored per-iteration and per-phase for later Parquet output and
//! CLI display.

use cobre_solver::SolverStatistics;

/// Delta of solver counters between two snapshots.
///
/// All fields represent the difference: after minus before snapshot.
#[derive(Debug, Clone, Default)]
pub struct SolverStatsDelta {
    /// Number of LP solves in this phase.
    pub lp_solves: u64,

    /// Solves that returned optimal (including retried solves that eventually succeeded).
    pub lp_successes: u64,

    /// Solves that returned optimal on first attempt (before any retry).
    pub first_try_successes: u64,

    /// Solves that exhausted all retry levels.
    pub lp_failures: u64,

    /// Total retry attempts across all retried solves.
    pub retry_attempts: u64,

    /// Number of warm-start `solve(Some(&basis))` calls.
    pub basis_offered: u64,

    /// Times the offered basis was rejected because `isBasisConsistent` returned false.
    pub basis_consistency_failures: u64,

    /// Total simplex iterations across all solves.
    pub simplex_iterations: u64,

    /// Cumulative wall-clock solve time in milliseconds.
    pub solve_time_ms: f64,

    /// Number of `load_model` calls in this phase.
    pub load_model_count: u64,

    /// Cumulative wall-clock time spent in `load_model` calls, in milliseconds.
    pub load_model_time_ms: f64,

    /// Cumulative wall-clock time spent in `set_row_bounds`/`set_col_bounds` calls, in milliseconds.
    pub set_bounds_time_ms: f64,

    /// Cumulative wall-clock time spent in `set_basis` FFI calls, in milliseconds.
    pub basis_set_time_ms: f64,

    /// Number of `reconstruct_basis` invocations with a non-empty stored basis.
    /// Application-level counter: not included in MPI packing or allreduce.
    /// A non-zero value indicates basis reconstruction is active.
    pub basis_reconstructions: u64,

    /// Per-level retry success histogram delta. Length depends on the solver
    /// backend (e.g. 12 for `HiGHS`).
    pub retry_level_histogram: Vec<u64>,
}

/// Resize histogram on first use to match the source histogram length.
fn ensure_histogram_capacity(result: &mut Vec<u64>, source: &[u64]) {
    if result.is_empty() && !source.is_empty() {
        result.resize(source.len(), 0);
    }
}

impl SolverStatsDelta {
    /// Compute the delta between two monotonic [`SolverStatistics`] snapshots.
    ///
    /// `before` must be taken before the phase starts, `after` immediately after.
    /// All counters are monotonically increasing, so the difference is non-negative.
    #[must_use]
    pub fn from_snapshots(before: &SolverStatistics, after: &SolverStatistics) -> Self {
        Self {
            lp_solves: after.solve_count - before.solve_count,
            lp_successes: after.success_count - before.success_count,
            first_try_successes: after.first_try_successes - before.first_try_successes,
            lp_failures: after.failure_count - before.failure_count,
            retry_attempts: after.retry_count - before.retry_count,
            basis_offered: after.basis_offered - before.basis_offered,
            basis_consistency_failures: after.basis_consistency_failures
                - before.basis_consistency_failures,
            simplex_iterations: after.total_iterations - before.total_iterations,
            solve_time_ms: (after.total_solve_time_seconds - before.total_solve_time_seconds)
                * 1000.0,
            load_model_count: after.load_model_count - before.load_model_count,
            load_model_time_ms: (after.total_load_model_time_seconds
                - before.total_load_model_time_seconds)
                * 1000.0,
            set_bounds_time_ms: (after.total_set_bounds_time_seconds
                - before.total_set_bounds_time_seconds)
                * 1000.0,
            basis_set_time_ms: (after.total_basis_set_time_seconds
                - before.total_basis_set_time_seconds)
                * 1000.0,
            basis_reconstructions: after
                .basis_reconstructions
                .saturating_sub(before.basis_reconstructions),
            retry_level_histogram: after
                .retry_level_histogram
                .iter()
                .zip(&before.retry_level_histogram)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Add `rhs` element-wise into `dst` in place.
    ///
    /// Equivalent to `*dst = aggregate([dst, rhs])` but avoids an intermediate
    /// allocation. Used on the hot backward-pass path to accumulate per-opening
    /// deltas across trial points within a single worker.
    pub fn accumulate_into(dst: &mut Self, rhs: &Self) {
        dst.lp_solves += rhs.lp_solves;
        dst.lp_successes += rhs.lp_successes;
        dst.first_try_successes += rhs.first_try_successes;
        dst.lp_failures += rhs.lp_failures;
        dst.retry_attempts += rhs.retry_attempts;
        dst.basis_offered += rhs.basis_offered;
        dst.basis_consistency_failures += rhs.basis_consistency_failures;
        dst.simplex_iterations += rhs.simplex_iterations;
        dst.solve_time_ms += rhs.solve_time_ms;
        dst.load_model_count += rhs.load_model_count;
        dst.load_model_time_ms += rhs.load_model_time_ms;
        dst.set_bounds_time_ms += rhs.set_bounds_time_ms;
        dst.basis_set_time_ms += rhs.basis_set_time_ms;
        dst.basis_reconstructions += rhs.basis_reconstructions;
        ensure_histogram_capacity(&mut dst.retry_level_histogram, &rhs.retry_level_histogram);
        for (d, s) in dst
            .retry_level_histogram
            .iter_mut()
            .zip(&rhs.retry_level_histogram)
        {
            *d += s;
        }
    }

    /// Copy all fields from `self` into `dst`, reusing the destination's
    /// existing `retry_level_histogram` allocation.
    ///
    /// All 14 scalar fields are copied by value. The destination's histogram
    /// `Vec<u64>` is resized (via `resize`) to match `self`, then overwritten
    /// with `copy_from_slice`. When `dst` already has a histogram of the same
    /// length (the common case on iteration ≥ 2), no heap allocation occurs.
    ///
    /// Prefer this over `dst = self.clone()` or `dst = self.clone_into_reuse(..)`
    /// on hot paths where the histogram length is stable across calls.
    pub fn clone_into_reuse(&self, dst: &mut Self) {
        dst.lp_solves = self.lp_solves;
        dst.lp_successes = self.lp_successes;
        dst.first_try_successes = self.first_try_successes;
        dst.lp_failures = self.lp_failures;
        dst.retry_attempts = self.retry_attempts;
        dst.basis_offered = self.basis_offered;
        dst.basis_consistency_failures = self.basis_consistency_failures;
        dst.simplex_iterations = self.simplex_iterations;
        dst.solve_time_ms = self.solve_time_ms;
        dst.load_model_count = self.load_model_count;
        dst.load_model_time_ms = self.load_model_time_ms;
        dst.set_bounds_time_ms = self.set_bounds_time_ms;
        dst.basis_set_time_ms = self.basis_set_time_ms;
        dst.basis_reconstructions = self.basis_reconstructions;
        let n = self.retry_level_histogram.len();
        dst.retry_level_histogram.resize(n, 0);
        dst.retry_level_histogram
            .copy_from_slice(&self.retry_level_histogram);
    }

    /// Reset all scalar fields to zero and clear the histogram in place.
    ///
    /// Equivalent to `*self = SolverStatsDelta::default()` but retains the
    /// existing `retry_level_histogram` capacity. Use this on hot paths where
    /// the histogram is known to be refilled immediately after the reset.
    pub fn reset_in_place(&mut self) {
        self.lp_solves = 0;
        self.lp_successes = 0;
        self.first_try_successes = 0;
        self.lp_failures = 0;
        self.retry_attempts = 0;
        self.basis_offered = 0;
        self.basis_consistency_failures = 0;
        self.simplex_iterations = 0;
        self.solve_time_ms = 0.0;
        self.load_model_count = 0;
        self.load_model_time_ms = 0.0;
        self.set_bounds_time_ms = 0.0;
        self.basis_set_time_ms = 0.0;
        self.basis_reconstructions = 0;
        self.retry_level_histogram.clear();
    }

    /// Sum an iterator of deltas element-wise into a single aggregate.
    ///
    /// Returns `Default` (all zeros) for an empty iterator.
    #[must_use]
    pub fn aggregate<'a>(deltas: impl Iterator<Item = &'a Self>) -> Self {
        let mut result = Self::default();
        for d in deltas {
            result.lp_solves += d.lp_solves;
            result.lp_successes += d.lp_successes;
            result.first_try_successes += d.first_try_successes;
            result.lp_failures += d.lp_failures;
            result.retry_attempts += d.retry_attempts;
            result.basis_offered += d.basis_offered;
            result.basis_consistency_failures += d.basis_consistency_failures;
            result.simplex_iterations += d.simplex_iterations;
            result.solve_time_ms += d.solve_time_ms;
            result.load_model_count += d.load_model_count;
            result.load_model_time_ms += d.load_model_time_ms;
            result.set_bounds_time_ms += d.set_bounds_time_ms;
            result.basis_set_time_ms += d.basis_set_time_ms;
            result.basis_reconstructions += d.basis_reconstructions;
            ensure_histogram_capacity(&mut result.retry_level_histogram, &d.retry_level_histogram);
            for (dst, src) in result
                .retry_level_histogram
                .iter_mut()
                .zip(&d.retry_level_histogram)
            {
                *dst += src;
            }
        }
        result
    }
}

/// Aggregate [`SolverStatistics`] from all solver instances in a workspace pool.
///
/// Sums all scalar counters across the given iterator of statistics. This is used
/// to get a single snapshot before/after a phase that distributes work across
/// multiple solver instances.
#[must_use]
pub fn aggregate_solver_statistics(
    stats: impl Iterator<Item = SolverStatistics>,
) -> SolverStatistics {
    let mut result = SolverStatistics::default();
    for s in stats {
        result.solve_count += s.solve_count;
        result.success_count += s.success_count;
        result.failure_count += s.failure_count;
        result.total_iterations += s.total_iterations;
        result.retry_count += s.retry_count;
        result.total_solve_time_seconds += s.total_solve_time_seconds;
        result.basis_consistency_failures += s.basis_consistency_failures;
        result.first_try_successes += s.first_try_successes;
        result.basis_offered += s.basis_offered;
        result.load_model_count += s.load_model_count;
        result.total_load_model_time_seconds += s.total_load_model_time_seconds;
        result.total_set_bounds_time_seconds += s.total_set_bounds_time_seconds;
        result.total_basis_set_time_seconds += s.total_basis_set_time_seconds;
        result.basis_reconstructions += s.basis_reconstructions;
        ensure_histogram_capacity(&mut result.retry_level_histogram, &s.retry_level_histogram);
        for (dst, src) in result
            .retry_level_histogram
            .iter_mut()
            .zip(&s.retry_level_histogram)
        {
            *dst += src;
        }
    }
    result
}

/// A single row in the solver stats log:
/// `(iteration, phase, stage, opening, rank, worker_id, delta)`.
///
/// - `iteration`: 1-based iteration number.
/// - `phase`: `"forward"`, `"backward"`, or `"lower_bound"` — a `&'static str`
///   to avoid per-iteration heap allocation (F1-005 fix).
/// - `stage`: stage index for backward phase (per-stage), `-1` for forward/LB.
/// - `opening`: opening index `0..n_openings` for backward rows; `-1` for
///   forward, lower-bound, and simulation rows (no opening dimension).
///   The parquet writer maps `-1` to a NULL `Int32` column.
/// - `rank`: MPI rank that produced this row. `-1` maps to NULL at the writer
///   boundary. Forward and lower-bound rows carry the local rank; backward rows
///   carry the actual rank from the allgatherv unpack.
/// - `worker_id`: rayon worker that produced this row. `-1` maps to NULL at the
///   writer boundary. Forward and lower-bound rows carry `-1` (no per-worker
///   dimension yet); backward rows carry the actual `worker_id` from the `allgatherv`
///   unpack.
/// - `delta`: the solver counter delta for this entry.
pub type SolverStatsEntry = (u64, &'static str, i32, i32, i32, i32, SolverStatsDelta);

/// Number of scalar fields in [`SolverStatsDelta`] (excludes histogram `Vec`).
/// Buffer size for MPI allreduce/allgatherv: 8 `u64` fields cast to `f64` + 5 native `f64` fields.
pub const SOLVER_STATS_DELTA_SCALAR_FIELDS: usize = 13;

/// Packed `f64` stride per scenario: `scenario_id` + 13 scalar fields.
pub const SCENARIO_STATS_STRIDE: usize = 1 + SOLVER_STATS_DELTA_SCALAR_FIELDS;

/// Packed `f64` stride per entry: `worker_id`, `slot_idx`, + 13 scalar fields (total 15).
pub const WORKER_STATS_ENTRY_STRIDE: usize = 2 + SOLVER_STATS_DELTA_SCALAR_FIELDS;

/// Required `f64` buffer length for a per-worker per-slot pack payload.
///
/// Returns `n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE`. Use to preallocate
/// MPI send/recv buffers at training setup.
#[must_use]
#[inline]
pub fn worker_opening_stats_buffer_size(n_workers: usize, n_slots: usize) -> usize {
    n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE
}

/// Pack the 13 scalar fields of a [`SolverStatsDelta`] into a fixed-size `f64` array.
/// Indices 0–7: eight `u64` fields cast to `f64`. Indices 8–12: five native `f64` fields.
/// The `retry_level_histogram` is excluded from MPI packing and Parquet output.
///
/// # Precision contract
///
/// `u64` casts are exact for event counts up to `2^53` (9,007,199,254,740,992). Beyond
/// that threshold `f64` cannot represent consecutive integers exactly, and an
/// `allreduce(Sum)` across MPI ranks would silently lose precision.
///
/// **The caller is responsible for enforcing the `2^53` ceiling before calling this
/// function.** This function does not validate counter values. For the CLI training
/// path the guard is implemented in `check_stats_overflow` in
/// `crates/cobre-cli/src/commands/run.rs`, which runs before the `[f64; 7]` pack and
/// returns `Err(CliError::Internal)` if any counter exceeds the limit.
///
/// # Timing precision
///
/// `solve_time_ms` (index 9) and the other `f64` timing fields are summed natively
/// without a cast. `f64` addition is non-associative, so rank-order changes across
/// MPI runs can cause ULP-level drift in `total_solve_time_seconds`. This is
/// acceptable: timing metrics require semantic parity (same order of magnitude),
/// not bit-for-bit reproducibility. Solve-count correctness is the strict invariant;
/// timing totals across ranks are informational.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn pack_delta_scalars(delta: &SolverStatsDelta) -> [f64; SOLVER_STATS_DELTA_SCALAR_FIELDS] {
    [
        delta.lp_solves as f64,                  // index 0
        delta.lp_successes as f64,               // index 1
        delta.first_try_successes as f64,        // index 2
        delta.lp_failures as f64,                // index 3
        delta.retry_attempts as f64,             // index 4
        delta.basis_offered as f64,              // index 5
        delta.basis_consistency_failures as f64, // index 6
        delta.simplex_iterations as f64,         // index 7
        delta.load_model_count as f64,           // index 8
        delta.solve_time_ms,                     // index 9
        delta.load_model_time_ms,                // index 10
        delta.set_bounds_time_ms,                // index 11
        delta.basis_set_time_ms,                 // index 12
    ]
}

/// Unpack a fixed-size `f64` array (from [`pack_delta_scalars`]) back into a [`SolverStatsDelta`].
/// The `retry_level_histogram` and `basis_reconstructions` are excluded from MPI packing and reset to defaults.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn unpack_delta_scalars(buf: &[f64; SOLVER_STATS_DELTA_SCALAR_FIELDS]) -> SolverStatsDelta {
    SolverStatsDelta {
        lp_solves: buf[0] as u64,
        lp_successes: buf[1] as u64,
        first_try_successes: buf[2] as u64,
        lp_failures: buf[3] as u64,
        retry_attempts: buf[4] as u64,
        basis_offered: buf[5] as u64,
        basis_consistency_failures: buf[6] as u64,
        simplex_iterations: buf[7] as u64,
        load_model_count: buf[8] as u64,
        solve_time_ms: buf[9],
        load_model_time_ms: buf[10],
        set_bounds_time_ms: buf[11],
        basis_set_time_ms: buf[12],
        basis_reconstructions: 0,
        retry_level_histogram: Vec::new(),
    }
}

/// Pack per-scenario `(scenario_id, delta)` pairs into a flat `f64` buffer for `allgatherv`.
/// Each scenario contributes [`SCENARIO_STATS_STRIDE`] values. Histogram is excluded.
#[must_use]
pub fn pack_scenario_stats(stats: &[(u32, SolverStatsDelta)]) -> Vec<f64> {
    let mut buf = Vec::with_capacity(stats.len() * SCENARIO_STATS_STRIDE);
    for (scenario_id, delta) in stats {
        buf.push(f64::from(*scenario_id));
        buf.extend_from_slice(&pack_delta_scalars(delta));
    }
    buf
}

/// Unpack a flat `f64` buffer (from [`pack_scenario_stats`]) back into a `Vec<(u32, SolverStatsDelta)>`.
/// Buffer length must be a multiple of [`SCENARIO_STATS_STRIDE`].
/// Panics in debug if length is not a multiple of `SCENARIO_STATS_STRIDE`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn unpack_scenario_stats(buf: &[f64]) -> Vec<(u32, SolverStatsDelta)> {
    debug_assert_eq!(
        buf.len() % SCENARIO_STATS_STRIDE,
        0,
        "buffer length must be a multiple of SCENARIO_STATS_STRIDE"
    );
    buf.chunks_exact(SCENARIO_STATS_STRIDE)
        .map(|chunk| {
            let scenario_id = chunk[0] as u32;
            // `chunks_exact(SCENARIO_STATS_STRIDE)` guarantees chunk.len() ==
            // SCENARIO_STATS_STRIDE = 1 + SOLVER_STATS_DELTA_SCALAR_FIELDS = 14.
            // Index 1..=13 therefore covers exactly the 13 scalar field slots.
            let arr = [
                chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7], chunk[8],
                chunk[9], chunk[10], chunk[11], chunk[12], chunk[13],
            ];
            (scenario_id, unpack_delta_scalars(&arr))
        })
        .collect()
}

/// Pack per-worker per-slot buffer into a flat `f64` buffer for `allgatherv`.
/// Fixed stride per entry: `[w as f64, k as f64, <SOLVER_STATS_DELTA_SCALAR_FIELDS scalar fields>]` in row-major order.
/// `out` length must be `n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE`.
/// `stats` length must be `n_workers * n_slots` (row-major: `stats[w * n_slots + k]`).
/// Panics in debug if sizes don't match.
#[allow(clippy::cast_precision_loss)]
pub fn pack_worker_opening_stats(
    out: &mut [f64],
    stats: &[SolverStatsDelta],
    n_workers: usize,
    n_slots: usize,
) {
    debug_assert_eq!(stats.len(), n_workers * n_slots);
    debug_assert_eq!(out.len(), n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE);
    for w in 0..n_workers {
        for k in 0..n_slots {
            let entry_base = (w * n_slots + k) * WORKER_STATS_ENTRY_STRIDE;
            out[entry_base] = w as f64;
            out[entry_base + 1] = k as f64;
            let scalars = pack_delta_scalars(&stats[w * n_slots + k]);
            out[entry_base + 2..entry_base + WORKER_STATS_ENTRY_STRIDE].copy_from_slice(&scalars);
        }
    }
}

/// Unpack a flat `f64` buffer (from [`pack_worker_opening_stats`]) into `out`.
/// `buf` must be `n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE` floats.
/// `out` must be a slice of length `n_workers * n_slots` (row-major order; contents overwritten).
/// The prefix `[worker_id, slot_idx]` per entry is informational (not asserted on unpack for ranks > 0).
/// Panics in debug if buffer sizes don't match.
///
/// # Panics
///
/// Panics (in debug builds) if buffer lengths don't match the expected sizes.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used
)]
pub fn unpack_worker_opening_stats(
    buf: &[f64],
    out: &mut [SolverStatsDelta],
    n_workers: usize,
    n_slots: usize,
) {
    debug_assert_eq!(buf.len(), n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE);
    debug_assert_eq!(out.len(), n_workers * n_slots);
    for w in 0..n_workers {
        for k in 0..n_slots {
            let entry_base = (w * n_slots + k) * WORKER_STATS_ENTRY_STRIDE;
            // The prefix fields buf[entry_base] (worker_id) and
            // buf[entry_base + 1] (slot_idx) store the LOCAL rank-relative
            // indices written by pack_worker_opening_stats on each rank.
            // When n_workers = n_ranks * n_workers_local, the combined flat
            // index w does NOT equal the local worker_id for ranks > 0, so
            // these fields are informational only and not asserted here.
            let scalars: [f64; SOLVER_STATS_DELTA_SCALAR_FIELDS] = buf
                [entry_base + 2..entry_base + WORKER_STATS_ENTRY_STRIDE]
                .try_into()
                .expect("slice length equals SOLVER_STATS_DELTA_SCALAR_FIELDS");
            out[w * n_slots + k] = unpack_delta_scalars(&scalars);
        }
    }
}

/// Flat per-(worker, slot) gather buffer for a single training pass.
///
/// Shape: `n_workers × n_slots`. Indexed by `worker_id * n_slots + slot`.
///
/// Used in three configurations:
/// - **Backward pass**: `n_slots = max_openings` — each worker writes its
///   per-opening stats for the current stage. Cleared via [`Self::reset`] between
///   stages (one stage per backward iteration step).
/// - **Forward pass**: `n_slots = n_stages` — each worker writes its
///   per-stage total for the full forward pass. Cleared between iterations.
/// - **Lower-bound pass**: `n_slots = 1` — each worker writes a single root-stage
///   delta. Cleared between iterations.
///
/// Allocated once at training setup (`train`) and reused across stages and
/// iterations without any hot-path allocation.
///
/// # Zero-allocation guarantee
///
/// `new` performs exactly one allocation. `reset`, `get`, `set`, and
/// `as_slice` perform no allocations. There are no `Vec::push` or
/// `Vec::new` calls on any path that touches this buffer after
/// construction.
///
/// # Reset cadence
///
/// - Backward buffer: call `reset()` at the **start of each backward stage**
///   (before the parallel region), so that stale data from the previous stage
///   does not accumulate.
/// - Forward and lower-bound buffers: call `reset()` at the **start of each
///   training iteration** (before the parallel region), so that data from the
///   previous iteration does not accumulate.
#[derive(Debug)]
pub struct StageWorkerStatsBuffer {
    data: Vec<SolverStatsDelta>,
    n_workers: usize,
    n_slots: usize,
}

impl StageWorkerStatsBuffer {
    /// Allocate a new buffer of shape `n_workers × n_slots`, initialised to
    /// `SolverStatsDelta::default()` (all zeros).
    #[must_use]
    pub fn new(n_workers: usize, n_slots: usize) -> Self {
        Self {
            data: vec![SolverStatsDelta::default(); n_workers * n_slots],
            n_workers,
            n_slots,
        }
    }

    /// Compute the flat index for `(worker_id, slot)`.
    ///
    /// Equivalent to `worker_id * n_slots + slot`. Used by [`Self::get`] and
    /// [`Self::set`] internally; exposed so callers can do bulk slice arithmetic
    /// without re-deriving the formula.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `worker_id >= self.n_workers` or `slot >= self.n_slots`.
    #[inline]
    #[must_use]
    pub fn index(&self, worker_id: usize, slot: usize) -> usize {
        debug_assert!(
            worker_id < self.n_workers,
            "worker_id {worker_id} >= n_workers {}",
            self.n_workers
        );
        debug_assert!(
            slot < self.n_slots,
            "slot {slot} >= n_slots {}",
            self.n_slots
        );
        worker_id * self.n_slots + slot
    }

    /// Return a shared reference to the delta at `(worker_id, slot)`.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if indices are out of bounds.
    #[must_use]
    pub fn get(&self, worker_id: usize, slot: usize) -> &SolverStatsDelta {
        let idx = self.index(worker_id, slot);
        &self.data[idx]
    }

    /// Write `delta` into the slot at `(worker_id, slot)`.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, worker_id: usize, slot: usize, delta: SolverStatsDelta) {
        let idx = self.index(worker_id, slot);
        self.data[idx] = delta;
    }

    /// Zero all slots.
    ///
    /// Call between stages (backward path) or between iterations (forward and
    /// lower-bound paths). Does not re-allocate; always `O(n_workers * n_slots)`.
    pub fn reset(&mut self) {
        for slot in &mut self.data {
            *slot = SolverStatsDelta::default();
        }
    }

    /// Return the flat backing slice of length `n_workers * n_slots`.
    ///
    /// Layout: `data[worker_id * n_slots + slot]`.
    #[must_use]
    pub fn as_slice(&self) -> &[SolverStatsDelta] {
        &self.data
    }

    /// Number of workers in the buffer (first dimension).
    #[must_use]
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    /// Number of slots per worker (second dimension).
    ///
    /// Equal to `max_openings` on the backward path, `n_stages` on the
    /// forward path, and `1` on the lower-bound path.
    #[must_use]
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    #[test]
    fn test_from_snapshots_all_deltas() {
        let before = SolverStatistics {
            solve_count: 10,
            success_count: 9,
            failure_count: 1,
            total_iterations: 500,
            retry_count: 3,
            total_solve_time_seconds: 2.0,
            basis_consistency_failures: 1,
            first_try_successes: 7,
            basis_offered: 8,
            load_model_count: 5,
            total_load_model_time_seconds: 1.0,
            total_set_bounds_time_seconds: 0.25,
            total_basis_set_time_seconds: 0.1,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        };
        let after = SolverStatistics {
            solve_count: 20,
            success_count: 18,
            failure_count: 2,
            total_iterations: 1100,
            retry_count: 5,
            total_solve_time_seconds: 4.5,
            basis_consistency_failures: 3,
            first_try_successes: 15,
            basis_offered: 17,
            load_model_count: 12,
            total_load_model_time_seconds: 3.0,
            total_set_bounds_time_seconds: 0.75,
            total_basis_set_time_seconds: 0.3,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        };

        let delta = SolverStatsDelta::from_snapshots(&before, &after);
        assert_eq!(delta.lp_solves, 10);
        assert_eq!(delta.lp_successes, 9);
        assert_eq!(delta.first_try_successes, 8);
        assert_eq!(delta.lp_failures, 1);
        assert_eq!(delta.retry_attempts, 2);
        assert_eq!(delta.basis_offered, 9);
        assert_eq!(delta.basis_consistency_failures, 2);
        assert_eq!(delta.simplex_iterations, 600);
        assert!((delta.solve_time_ms - 2500.0).abs() < 1e-6);
        assert_eq!(delta.load_model_count, 7);
        assert!((delta.load_model_time_ms - 2000.0).abs() < 1e-6);
        assert!((delta.set_bounds_time_ms - 500.0).abs() < 1e-6);
        assert!((delta.basis_set_time_ms - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_snapshots_zero_delta() {
        let snap = SolverStatistics {
            solve_count: 5,
            success_count: 5,
            failure_count: 0,
            total_iterations: 200,
            retry_count: 0,
            total_solve_time_seconds: 1.0,
            basis_consistency_failures: 0,
            first_try_successes: 5,
            basis_offered: 3,
            load_model_count: 3,
            total_load_model_time_seconds: 0.1,
            total_set_bounds_time_seconds: 0.02,
            total_basis_set_time_seconds: 0.01,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        };
        let delta = SolverStatsDelta::from_snapshots(&snap, &snap);
        assert_eq!(delta.lp_solves, 0);
        assert_eq!(delta.lp_successes, 0);
        assert_eq!(delta.first_try_successes, 0);
        assert_eq!(delta.lp_failures, 0);
        assert_eq!(delta.retry_attempts, 0);
        assert_eq!(delta.basis_offered, 0);
        assert_eq!(delta.basis_consistency_failures, 0);
        assert_eq!(delta.simplex_iterations, 0);
        assert!((delta.solve_time_ms).abs() < 1e-10);
        assert!((delta.load_model_time_ms).abs() < 1e-10);
        assert!((delta.set_bounds_time_ms).abs() < 1e-10);
        assert!((delta.basis_set_time_ms).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_empty_returns_default() {
        let agg = SolverStatsDelta::aggregate(std::iter::empty());
        assert_eq!(agg.lp_solves, 0);
        assert_eq!(agg.solve_time_ms, 0.0);
    }

    #[test]
    fn test_aggregate_sums_all_fields() {
        let d1 = SolverStatsDelta {
            lp_solves: 10,
            lp_successes: 9,
            first_try_successes: 8,
            lp_failures: 1,
            retry_attempts: 2,
            basis_offered: 7,
            basis_consistency_failures: 1,
            simplex_iterations: 500,
            solve_time_ms: 100.0,
            load_model_count: 5,
            load_model_time_ms: 10.0,
            set_bounds_time_ms: 2.0,
            basis_set_time_ms: 1.0,
            basis_reconstructions: 3,
            retry_level_histogram: vec![0; 12],
        };
        let d2 = SolverStatsDelta {
            lp_solves: 20,
            lp_successes: 19,
            first_try_successes: 17,
            lp_failures: 1,
            retry_attempts: 3,
            basis_offered: 15,
            basis_consistency_failures: 2,
            simplex_iterations: 800,
            solve_time_ms: 200.0,
            load_model_count: 10,
            load_model_time_ms: 20.0,
            set_bounds_time_ms: 4.0,
            basis_set_time_ms: 2.0,
            basis_reconstructions: 5,
            retry_level_histogram: vec![0; 12],
        };

        let agg = SolverStatsDelta::aggregate([d1, d2].iter());
        assert_eq!(agg.lp_solves, 30);
        assert_eq!(agg.lp_successes, 28);
        assert_eq!(agg.first_try_successes, 25);
        assert_eq!(agg.lp_failures, 2);
        assert_eq!(agg.retry_attempts, 5);
        assert_eq!(agg.basis_offered, 22);
        assert_eq!(agg.basis_consistency_failures, 3);
        assert_eq!(agg.simplex_iterations, 1300);
        assert!((agg.solve_time_ms - 300.0).abs() < 1e-6);
        assert_eq!(agg.load_model_count, 15);
        assert!((agg.load_model_time_ms - 30.0).abs() < 1e-6);
        assert!((agg.set_bounds_time_ms - 6.0).abs() < 1e-6);
        assert!((agg.basis_set_time_ms - 3.0).abs() < 1e-6);
        assert_eq!(agg.basis_reconstructions, 8);
    }

    #[test]
    fn test_aggregate_solver_statistics_sums_all_fields() {
        let s1 = SolverStatistics {
            solve_count: 10,
            success_count: 9,
            failure_count: 1,
            total_iterations: 500,
            retry_count: 3,
            total_solve_time_seconds: 2.0,
            basis_consistency_failures: 1,
            first_try_successes: 7,
            basis_offered: 8,
            load_model_count: 5,
            total_load_model_time_seconds: 1.0,
            total_set_bounds_time_seconds: 0.25,
            total_basis_set_time_seconds: 0.05,
            basis_reconstructions: 4,
            retry_level_histogram: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let s2 = SolverStatistics {
            solve_count: 20,
            success_count: 18,
            failure_count: 2,
            total_iterations: 1100,
            retry_count: 5,
            total_solve_time_seconds: 4.5,
            basis_consistency_failures: 3,
            first_try_successes: 15,
            basis_offered: 17,
            load_model_count: 12,
            total_load_model_time_seconds: 3.0,
            total_set_bounds_time_seconds: 0.75,
            total_basis_set_time_seconds: 0.15,
            basis_reconstructions: 10,
            retry_level_histogram: vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        let agg = aggregate_solver_statistics([s1, s2].into_iter());
        assert_eq!(agg.solve_count, 30);
        assert_eq!(agg.success_count, 27);
        assert_eq!(agg.failure_count, 3);
        assert_eq!(agg.total_iterations, 1600);
        assert_eq!(agg.retry_count, 8);
        assert!((agg.total_solve_time_seconds - 6.5).abs() < 1e-10);
        assert_eq!(agg.basis_consistency_failures, 4);
        assert_eq!(agg.first_try_successes, 22);
        assert_eq!(agg.basis_offered, 25);
        assert_eq!(agg.load_model_count, 17);
        assert!((agg.total_load_model_time_seconds - 4.0).abs() < 1e-10);
        assert!((agg.total_set_bounds_time_seconds - 1.0).abs() < 1e-10);
        assert!((agg.total_basis_set_time_seconds - 0.2).abs() < 1e-10);
        assert_eq!(agg.retry_level_histogram[0], 1);
        assert_eq!(agg.retry_level_histogram[1], 2);
        assert_eq!(agg.retry_level_histogram[2], 0);
        assert_eq!(agg.basis_reconstructions, 14);
    }

    fn make_delta(lp_solves: u64) -> SolverStatsDelta {
        SolverStatsDelta {
            lp_solves,
            lp_successes: lp_solves,
            first_try_successes: lp_solves / 2,
            lp_failures: 0,
            retry_attempts: 1,
            basis_offered: lp_solves,
            basis_consistency_failures: 2,
            simplex_iterations: lp_solves * 10,
            solve_time_ms: lp_solves as f64 * 0.5,
            load_model_count: 3,
            load_model_time_ms: 1.5,
            set_bounds_time_ms: 0.25,
            basis_set_time_ms: 0.125,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        }
    }

    #[test]
    fn test_pack_unpack_delta_scalars_round_trip() {
        let delta = make_delta(600);
        let packed = pack_delta_scalars(&delta);
        assert_eq!(packed.len(), SOLVER_STATS_DELTA_SCALAR_FIELDS);
        let unpacked = unpack_delta_scalars(&packed);

        assert_eq!(unpacked.lp_solves, delta.lp_solves);
        assert_eq!(unpacked.lp_successes, delta.lp_successes);
        assert_eq!(unpacked.first_try_successes, delta.first_try_successes);
        assert_eq!(unpacked.lp_failures, delta.lp_failures);
        assert_eq!(unpacked.retry_attempts, delta.retry_attempts);
        assert_eq!(unpacked.basis_offered, delta.basis_offered);
        assert_eq!(
            unpacked.basis_consistency_failures,
            delta.basis_consistency_failures
        );
        assert_eq!(unpacked.simplex_iterations, delta.simplex_iterations);
        assert_eq!(unpacked.load_model_count, delta.load_model_count);
        assert!((unpacked.solve_time_ms - delta.solve_time_ms).abs() < 1e-10);
        assert!((unpacked.load_model_time_ms - delta.load_model_time_ms).abs() < 1e-10);
        assert!((unpacked.set_bounds_time_ms - delta.set_bounds_time_ms).abs() < 1e-10);
        assert!((unpacked.basis_set_time_ms - delta.basis_set_time_ms).abs() < 1e-10);
        // histogram is excluded from pack/unpack
        assert!(unpacked.retry_level_histogram.is_empty());
    }

    #[test]
    fn test_pack_unpack_delta_scalars_identity_for_lp_solves_600() {
        // Acceptance criterion: identity property for allreduce with LocalBackend.
        let delta = make_delta(600);
        let packed = pack_delta_scalars(&delta);
        let unpacked = unpack_delta_scalars(&packed);
        assert_eq!(unpacked.lp_solves, 600);
    }

    #[test]
    fn test_pack_unpack_scenario_stats_round_trip_three_entries() {
        // Acceptance criterion: pack/unpack round-trip for Vec<(u32, SolverStatsDelta)>
        // with scenario IDs 7, 12, 25.
        let stats = vec![
            (7u32, make_delta(100)),
            (12u32, make_delta(200)),
            (25u32, make_delta(300)),
        ];
        let buf = pack_scenario_stats(&stats);
        assert_eq!(buf.len(), 3 * SCENARIO_STATS_STRIDE);

        let unpacked = unpack_scenario_stats(&buf);
        assert_eq!(unpacked.len(), 3);

        // Verify scenario IDs
        assert_eq!(unpacked[0].0, 7);
        assert_eq!(unpacked[1].0, 12);
        assert_eq!(unpacked[2].0, 25);

        // Verify field values for each scenario
        assert_eq!(unpacked[0].1.lp_solves, 100);
        assert_eq!(unpacked[1].1.lp_solves, 200);
        assert_eq!(unpacked[2].1.lp_solves, 300);

        assert!((unpacked[0].1.solve_time_ms - 50.0).abs() < 1e-10);
        assert!((unpacked[1].1.solve_time_ms - 100.0).abs() < 1e-10);
        assert!((unpacked[2].1.solve_time_ms - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_pack_scenario_stats_empty_round_trip() {
        let buf = pack_scenario_stats(&[]);
        assert!(buf.is_empty());
        let unpacked = unpack_scenario_stats(&buf);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_pack_delta_scalars_field_count() {
        // Regression: pack_delta_scalars must produce a 13-element array.
        let delta = SolverStatsDelta::default();
        let packed = pack_delta_scalars(&delta);
        assert_eq!(
            packed.len(),
            13,
            "pack_delta_scalars must return 13 elements"
        );
        assert_eq!(packed.len(), SOLVER_STATS_DELTA_SCALAR_FIELDS);
        assert_eq!(SOLVER_STATS_DELTA_SCALAR_FIELDS, 13);
    }

    #[test]
    fn test_solver_stats_delta_includes_reconstruction_fields() {
        // Acceptance criterion: from_snapshots correctly computes the delta for
        // basis_reconstructions.
        let before = SolverStatistics {
            basis_reconstructions: 10,
            ..SolverStatistics::default()
        };
        let after = SolverStatistics {
            basis_reconstructions: 25,
            ..SolverStatistics::default()
        };

        let delta = SolverStatsDelta::from_snapshots(&before, &after);
        assert_eq!(delta.basis_reconstructions, 15);
    }

    #[test]
    fn test_accumulate_into_all_fields() {
        // AC-001: accumulate_into sums every scalar field and extends the histogram.
        let mut dst = make_delta(10);
        dst.retry_level_histogram = vec![1, 0, 2, 0];
        let rhs = make_delta(5);
        let mut rhs_full = rhs.clone();
        rhs_full.retry_level_histogram = vec![0, 3, 0, 1];

        SolverStatsDelta::accumulate_into(&mut dst, &rhs_full);

        assert_eq!(dst.lp_solves, 15);
        assert_eq!(dst.lp_successes, 15);
        assert_eq!(dst.first_try_successes, 7); // 5 + 2
        assert_eq!(dst.lp_failures, 0);
        assert_eq!(dst.retry_attempts, 2); // 1 + 1
        assert_eq!(dst.basis_offered, 15);
        assert_eq!(dst.basis_consistency_failures, 4); // 2 + 2
        assert_eq!(dst.simplex_iterations, 150); // 100 + 50
        assert!((dst.solve_time_ms - 7.5).abs() < 1e-10); // 5.0 + 2.5
        assert_eq!(dst.load_model_count, 6); // 3 + 3
        assert!((dst.load_model_time_ms - 3.0).abs() < 1e-10);
        assert!((dst.set_bounds_time_ms - 0.5).abs() < 1e-10);
        assert!((dst.basis_set_time_ms - 0.25).abs() < 1e-10);
        // Histogram: [1+0, 0+3, 2+0, 0+1]
        assert_eq!(dst.retry_level_histogram, vec![1, 3, 2, 1]);
    }

    #[test]
    fn test_solver_stats_log_per_opening_shape() {
        // AC-002: SolverStatsEntry is a 7-tuple (iteration, phase, stage, opening,
        // rank, worker_id, delta). Forward entries use a real stage index (>= 0)
        // and opening == -1, worker_id == -1 (no per-worker dimension yet).
        // LB entries continue to use stage == -1, opening == -1, worker_id == -1.
        // Backward entries carry real (rank, worker_id) from allgatherv unpack.
        let fwd_entry: SolverStatsEntry = (1, "forward", 0, -1, 0, -1, make_delta(4));
        let bwd_entry_0: SolverStatsEntry = (1, "backward", 2, 0, 0, 0, make_delta(2));
        let bwd_entry_1: SolverStatsEntry = (1, "backward", 2, 1, 0, 1, make_delta(3));
        let lb_entry: SolverStatsEntry = (1, "lower_bound", -1, -1, 0, -1, make_delta(1));

        let log: Vec<SolverStatsEntry> = vec![fwd_entry, bwd_entry_0, bwd_entry_1, lb_entry];

        // Verify the forward entry has a real stage index, opening == -1, worker_id == -1.
        let (_, phase_fwd, stage_fwd, opening_fwd, _rank_fwd, worker_id_fwd, _) = &log[0];
        assert_eq!(*phase_fwd, "forward");
        assert!(
            *stage_fwd >= 0,
            "forward stage must be a real stage index, got {stage_fwd}"
        );
        assert_eq!(*opening_fwd, -1);
        assert_eq!(
            *worker_id_fwd, -1,
            "forward rows have no per-worker dimension"
        );

        // Verify backward entries carry correct opening indices and worker ids.
        let (_, _, stage0, opening0, rank0, worker_id0, delta0) = &log[1];
        assert_eq!(*stage0, 2);
        assert_eq!(*opening0, 0);
        assert_eq!(*rank0, 0);
        assert_eq!(*worker_id0, 0);
        assert_eq!(delta0.lp_solves, 2);

        let (_, _, stage1, opening1, rank1, worker_id1, delta1) = &log[2];
        assert_eq!(*stage1, 2);
        assert_eq!(*opening1, 1);
        assert_eq!(*rank1, 0);
        assert_eq!(*worker_id1, 1);
        assert_eq!(delta1.lp_solves, 3);

        // Verify that collapsing across openings/workers yields the per-stage total.
        let backward_entries: Vec<&SolverStatsDelta> = log
            .iter()
            .filter(|(_, ph, _, _, _, _, _)| *ph == "backward")
            .map(|(_, _, _, _, _, _, d)| d)
            .collect();
        let collapsed = SolverStatsDelta::aggregate(backward_entries.into_iter());
        assert_eq!(collapsed.lp_solves, 5); // 2 + 3

        // Verify that LB entry has opening == -1, worker_id == -1.
        let (_, _, _, opening_lb, _, worker_id_lb, _) = &log[3];
        assert_eq!(*opening_lb, -1);
        assert_eq!(*worker_id_lb, -1);
    }

    /// per-stage forward `stage_stats` summed element-wise across workers.
    ///
    /// Simulates 2 workers × 3 stages, verifying that the post-parallel reduction
    /// produces the correct element-wise sum without hot-path allocations.
    #[test]
    fn test_forward_stage_stats_summed_across_workers() {
        // Worker 0 processed some scenarios at each stage.
        let worker0: Vec<SolverStatsDelta> = vec![
            make_delta(10), // stage 0: 10 lp_solves
            make_delta(20), // stage 1: 20 lp_solves
            make_delta(30), // stage 2: 30 lp_solves
        ];

        // Worker 1 processed the remaining scenarios.
        let worker1: Vec<SolverStatsDelta> = vec![
            make_delta(5),  // stage 0: 5 lp_solves
            make_delta(15), // stage 1: 15 lp_solves
            make_delta(25), // stage 2: 25 lp_solves
        ];

        // Simulate the post-parallel reduction (mirrors run_forward_pass merge code).
        let n_stages = 3;
        let mut stage_stats: Vec<SolverStatsDelta> =
            (0..n_stages).map(|_| SolverStatsDelta::default()).collect();

        for worker_stage_stats in [&worker0, &worker1] {
            for (dst, src) in stage_stats.iter_mut().zip(worker_stage_stats) {
                SolverStatsDelta::accumulate_into(dst, src);
            }
        }

        // Element-wise sum must equal worker0[t] + worker1[t] for each stage.
        assert_eq!(
            stage_stats[0].lp_solves, 15,
            "stage 0: 10 + 5 = 15 lp_solves"
        );
        assert_eq!(
            stage_stats[1].lp_solves, 35,
            "stage 1: 20 + 15 = 35 lp_solves"
        );
        assert_eq!(
            stage_stats[2].lp_solves, 55,
            "stage 2: 30 + 25 = 55 lp_solves"
        );

        // Verify simplex_iterations also sum correctly (10× lp_solves in make_delta).
        assert_eq!(stage_stats[0].simplex_iterations, 150); // (10 + 5) * 10
        assert_eq!(stage_stats[1].simplex_iterations, 350); // (20 + 15) * 10
        assert_eq!(stage_stats[2].simplex_iterations, 550); // (30 + 25) * 10

        // Verify the log shape: one SolverStatsEntry per stage with stage index 0..3.
        // forward rows use real stage index, opening sentinel -1 → NULL)
        // 7-tuple: (iter, phase, stage, opening, rank, worker_id, delta)
        // Forward rows carry rank = local rank, worker_id = -1 (no per-worker dimension).
        let log: Vec<SolverStatsEntry> = stage_stats
            .iter()
            .enumerate()
            .map(|(t, delta)| {
                (
                    1u64,
                    "forward",
                    i32::try_from(t).expect("stage fits i32"),
                    -1_i32,
                    0_i32,  // rank
                    -1_i32, // worker_id sentinel → NULL at writer
                    delta.clone(),
                )
            })
            .collect();

        assert_eq!(log.len(), 3, "one entry per stage");
        for (t, entry) in log.iter().enumerate() {
            let (_, phase, stage, opening, _rank, worker_id, _) = entry;
            assert_eq!(*phase, "forward");
            assert_eq!(
                *stage,
                i32::try_from(t).expect("stage fits i32"),
                "stage index must match loop variable"
            );
            assert_eq!(*opening, -1, "forward rows have no opening dimension");
            assert_eq!(*worker_id, -1, "forward rows have no per-worker dimension");
        }
    }

    /// Verify `index(w, k) = w * n_slots + k` for several values.
    #[test]
    fn test_stage_worker_stats_buffer_index_layout() {
        let buf = StageWorkerStatsBuffer::new(3, 4);
        assert_eq!(buf.index(0, 0), 0);
        assert_eq!(buf.index(0, 3), 3);
        assert_eq!(buf.index(1, 0), 4);
        assert_eq!(buf.index(2, 3), 11);
        assert_eq!(buf.as_slice().len(), 12);
        assert_eq!(buf.n_workers(), 3);
        assert_eq!(buf.n_slots(), 4);
    }

    /// Verify `reset()` zeros every slot, even after non-default writes.
    #[test]
    fn test_stage_worker_stats_buffer_reset_zeroes_all_slots() {
        let mut buf = StageWorkerStatsBuffer::new(2, 3);
        for w in 0..2 {
            for k in 0..3 {
                let d = SolverStatsDelta {
                    lp_solves: 7,
                    ..SolverStatsDelta::default()
                };
                buf.set(w, k, d);
            }
        }
        for slot in buf.as_slice() {
            assert_eq!(slot.lp_solves, 7);
        }
        buf.reset();
        for slot in buf.as_slice() {
            assert_eq!(slot.lp_solves, 0);
        }
    }

    /// Round-trip pack→unpack must preserve every scalar field for every (w,k) pair.
    #[test]
    fn test_pack_worker_opening_stats_roundtrip() {
        let n_workers = 3;
        let n_slots = 4;
        let mut input: Vec<SolverStatsDelta> = Vec::with_capacity(n_workers * n_slots);
        for w in 0..n_workers {
            for k in 0..n_slots {
                input.push(SolverStatsDelta {
                    lp_solves: (w * 10 + k) as u64,
                    ..SolverStatsDelta::default()
                });
            }
        }
        let mut buf = vec![0.0_f64; worker_opening_stats_buffer_size(n_workers, n_slots)];
        assert_eq!(buf.len(), n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE);
        pack_worker_opening_stats(&mut buf, &input, n_workers, n_slots);

        let mut recovered = vec![SolverStatsDelta::default(); n_workers * n_slots];
        unpack_worker_opening_stats(&buf, &mut recovered, n_workers, n_slots);

        for w in 0..n_workers {
            for k in 0..n_slots {
                assert_eq!(
                    recovered[w * n_slots + k].lp_solves,
                    (w * 10 + k) as u64,
                    "lp_solves mismatch at (w={w}, k={k})"
                );
            }
        }
    }

    /// Helper returns the precise `stride * n_workers * n_slots` size in `f64` units.
    #[test]
    fn test_pack_worker_opening_stats_buffer_size() {
        assert_eq!(worker_opening_stats_buffer_size(10, 20), 10 * 20 * 15);
        assert_eq!(worker_opening_stats_buffer_size(10, 20), 3000);
    }

    /// Layout invariants — `[w as f64, k as f64, ...]` per entry, row-major.
    #[test]
    fn test_pack_worker_opening_stats_layout_invariant() {
        let n_workers = 2;
        let n_slots = 4;
        let input = vec![SolverStatsDelta::default(); n_workers * n_slots];
        let mut buf = vec![0.0_f64; worker_opening_stats_buffer_size(n_workers, n_slots)];
        pack_worker_opening_stats(&mut buf, &input, n_workers, n_slots);

        // entry(w=0, k=0) at offset 0
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[1], 0.0);
        // entry(w=0, k=1) at offset WORKER_STATS_ENTRY_STRIDE (= 15)
        assert_eq!(buf[WORKER_STATS_ENTRY_STRIDE], 0.0);
        assert_eq!(buf[WORKER_STATS_ENTRY_STRIDE + 1], 1.0);
        // entry(w=1, k=0) at offset WORKER_STATS_ENTRY_STRIDE * n_slots (= 15 * 4)
        let w1_k0 = WORKER_STATS_ENTRY_STRIDE * n_slots;
        assert_eq!(buf[w1_k0], 1.0);
        assert_eq!(buf[w1_k0 + 1], 0.0);
    }

    /// MPI wire-format pin: `SolverStatsDelta` pack/unpack uses a 13-element `f64` array.
    /// Uses distinct nonzero values for every field so that field-order swaps are caught
    /// at both pack and unpack.
    #[test]
    fn test_solver_stats_delta_mpi_wire_format_13_fields() {
        // Wire-format constant assertions.
        assert_eq!(SOLVER_STATS_DELTA_SCALAR_FIELDS, 13);
        assert_eq!(SCENARIO_STATS_STRIDE, 14);
        assert_eq!(WORKER_STATS_ENTRY_STRIDE, 15);

        // Construct a populated delta. Use distinct nonzero values
        // to catch field-order swaps at pack/unpack.
        let delta = SolverStatsDelta {
            lp_solves: 1,
            lp_successes: 2,
            first_try_successes: 3,
            lp_failures: 4,
            retry_attempts: 5,
            basis_offered: 6,
            basis_consistency_failures: 7,
            simplex_iterations: 8,
            load_model_count: 9,
            solve_time_ms: 10.5,
            load_model_time_ms: 11.25,
            set_bounds_time_ms: 12.125,
            basis_set_time_ms: 13.0625,
            basis_reconstructions: 42, // application-level; not in wire
            retry_level_histogram: vec![1, 2, 3],
        };

        let packed = pack_delta_scalars(&delta);
        assert_eq!(packed.len(), 13);

        // Cross-check packed field ordering via explicit index reads.
        // Indices 0..=8 are u64 fields cast to f64; indices 9..=12 are f64.
        assert_eq!(packed[0], 1.0); // lp_solves
        assert_eq!(packed[8], 9.0); // load_model_count
        assert!((packed[9] - 10.5).abs() < 1e-10); // solve_time_ms
        assert!((packed[12] - 13.0625).abs() < 1e-10); // basis_set_time_ms

        let unpacked = unpack_delta_scalars(&packed);

        // Wire-carried fields must match.
        assert_eq!(unpacked.lp_solves, 1);
        assert_eq!(unpacked.lp_successes, 2);
        assert_eq!(unpacked.first_try_successes, 3);
        assert_eq!(unpacked.lp_failures, 4);
        assert_eq!(unpacked.retry_attempts, 5);
        assert_eq!(unpacked.basis_offered, 6);
        assert_eq!(unpacked.basis_consistency_failures, 7);
        assert_eq!(unpacked.simplex_iterations, 8);
        assert_eq!(unpacked.load_model_count, 9);
        assert!((unpacked.solve_time_ms - 10.5).abs() < 1e-10);
        assert!((unpacked.load_model_time_ms - 11.25).abs() < 1e-10);
        assert!((unpacked.set_bounds_time_ms - 12.125).abs() < 1e-10);
        assert!((unpacked.basis_set_time_ms - 13.0625).abs() < 1e-10);

        // Application-level fields are NOT in the wire — unpack zeros them.
        assert_eq!(unpacked.basis_reconstructions, 0);
        assert!(unpacked.retry_level_histogram.is_empty());
    }

    /// Pins the wire-format array length at the type level. The function signature
    /// is `unpack_delta_scalars(&[f64; SOLVER_STATS_DELTA_SCALAR_FIELDS])`, so the
    /// compiler enforces the length. This test confirms the constant equals 13 and
    /// that the `size_of` the array is exactly `13 * 8` bytes.
    #[test]
    fn test_unpack_delta_scalars_array_length_is_compile_time() {
        assert_eq!(
            std::mem::size_of::<[f64; SOLVER_STATS_DELTA_SCALAR_FIELDS]>(),
            13 * std::mem::size_of::<f64>()
        );
        // Also confirm that a valid 13-element buffer round-trips without panic.
        let buf: [f64; 13] = [0.0; 13];
        let _ = unpack_delta_scalars(&buf);
    }

    /// `clone_into_reuse` must copy all scalar fields and resize+overwrite
    /// the histogram without reallocating when the destination already has
    /// sufficient capacity (tested via distinct source and destination
    /// histogram lengths).
    #[test]
    fn solver_stats_delta_clone_into_reuse_preserves_values() {
        let src = SolverStatsDelta {
            lp_solves: 42,
            lp_successes: 40,
            first_try_successes: 38,
            lp_failures: 2,
            retry_attempts: 5,
            basis_offered: 35,
            basis_consistency_failures: 3,
            simplex_iterations: 1200,
            solve_time_ms: 99.5,
            load_model_count: 10,
            load_model_time_ms: 7.25,
            set_bounds_time_ms: 1.5,
            basis_set_time_ms: 0.75,
            basis_reconstructions: 4,
            retry_level_histogram: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };

        // dst starts with a different histogram length to verify resize.
        let mut dst = SolverStatsDelta {
            lp_solves: 0,
            retry_level_histogram: vec![0; 5], // different length from src
            ..SolverStatsDelta::default()
        };

        src.clone_into_reuse(&mut dst);

        assert_eq!(dst.lp_solves, 42);
        assert_eq!(dst.lp_successes, 40);
        assert_eq!(dst.first_try_successes, 38);
        assert_eq!(dst.lp_failures, 2);
        assert_eq!(dst.retry_attempts, 5);
        assert_eq!(dst.basis_offered, 35);
        assert_eq!(dst.basis_consistency_failures, 3);
        assert_eq!(dst.simplex_iterations, 1200);
        assert!((dst.solve_time_ms - 99.5).abs() < 1e-10);
        assert_eq!(dst.load_model_count, 10);
        assert!((dst.load_model_time_ms - 7.25).abs() < 1e-10);
        assert!((dst.set_bounds_time_ms - 1.5).abs() < 1e-10);
        assert!((dst.basis_set_time_ms - 0.75).abs() < 1e-10);
        assert_eq!(dst.basis_reconstructions, 4);
        assert_eq!(
            dst.retry_level_histogram,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        // Verify reuse: clone_into_reuse onto a dst that already has the
        // right histogram length must produce the same result.
        let src2 = SolverStatsDelta {
            lp_solves: 7,
            retry_level_histogram: vec![10; 12],
            ..SolverStatsDelta::default()
        };
        src2.clone_into_reuse(&mut dst);
        assert_eq!(dst.lp_solves, 7);
        assert_eq!(dst.retry_level_histogram, vec![10; 12]);
    }

    /// `reset_in_place` must zero all scalar fields and clear the histogram,
    /// leaving the existing `Vec` capacity intact.
    #[test]
    fn solver_stats_delta_reset_in_place_zeroes_all_fields() {
        let mut d = SolverStatsDelta {
            lp_solves: 99,
            lp_successes: 88,
            first_try_successes: 77,
            lp_failures: 11,
            retry_attempts: 6,
            basis_offered: 50,
            basis_consistency_failures: 4,
            simplex_iterations: 500,
            solve_time_ms: 42.0,
            load_model_count: 8,
            load_model_time_ms: 3.0,
            set_bounds_time_ms: 1.0,
            basis_set_time_ms: 0.5,
            basis_reconstructions: 2,
            retry_level_histogram: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };

        d.reset_in_place();

        assert_eq!(d.lp_solves, 0);
        assert_eq!(d.lp_successes, 0);
        assert_eq!(d.first_try_successes, 0);
        assert_eq!(d.lp_failures, 0);
        assert_eq!(d.retry_attempts, 0);
        assert_eq!(d.basis_offered, 0);
        assert_eq!(d.basis_consistency_failures, 0);
        assert_eq!(d.simplex_iterations, 0);
        assert_eq!(d.solve_time_ms, 0.0);
        assert_eq!(d.load_model_count, 0);
        assert_eq!(d.load_model_time_ms, 0.0);
        assert_eq!(d.set_bounds_time_ms, 0.0);
        assert_eq!(d.basis_set_time_ms, 0.0);
        assert_eq!(d.basis_reconstructions, 0);
        assert!(d.retry_level_histogram.is_empty());
    }
}
