//! Per-phase delta of solver counters for LP statistics collection.
//!
//! [`SolverStatsDelta`] is computed from before/after [`SolverStatistics`] snapshots
//! taken around each training phase (forward pass, backward pass, lower bound evaluation).
//! The deltas are stored per-iteration and per-phase for later Parquet output and
//! CLI display.

use cobre_solver::SolverStatistics;

/// Per-phase delta of solver counters, computed from before/after snapshots.
///
/// All integer fields represent the difference between the after and before snapshots.
/// `solve_time_ms` is the wall-clock time delta in milliseconds.
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

    /// Number of `solve_with_basis` calls.
    pub basis_offered: u64,

    /// Times the basis was rejected (cold-start fallback).
    pub basis_rejections: u64,

    /// Total simplex iterations across all solves.
    pub simplex_iterations: u64,

    /// Cumulative wall-clock solve time in milliseconds.
    pub solve_time_ms: f64,

    /// Number of `load_model` calls in this phase.
    pub load_model_count: u64,

    /// Number of `add_rows` calls in this phase.
    pub add_rows_count: u64,

    /// Cumulative wall-clock time spent in `load_model` calls, in milliseconds.
    pub load_model_time_ms: f64,

    /// Cumulative wall-clock time spent in `add_rows` calls, in milliseconds.
    pub add_rows_time_ms: f64,

    /// Cumulative wall-clock time spent in `set_row_bounds`/`set_col_bounds` calls, in milliseconds.
    pub set_bounds_time_ms: f64,

    /// Cumulative wall-clock time spent in `set_basis` FFI calls, in milliseconds.
    pub basis_set_time_ms: f64,

    /// Number of new cut rows assigned `NONBASIC_LOWER` by basis-aware padding
    /// (Strategy S3) during this phase.
    ///
    /// Application-level counter: not included in MPI packing or allreduce.
    pub basis_padding_tight: u64,

    /// Number of new cut rows assigned `BASIC` by basis-aware padding
    /// (Strategy S3) during this phase.
    ///
    /// Application-level counter: not included in MPI packing or allreduce.
    pub basis_padding_slack: u64,

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
            basis_rejections: after.basis_rejections - before.basis_rejections,
            simplex_iterations: after.total_iterations - before.total_iterations,
            solve_time_ms: (after.total_solve_time_seconds - before.total_solve_time_seconds)
                * 1000.0,
            load_model_count: after.load_model_count - before.load_model_count,
            add_rows_count: after.add_rows_count - before.add_rows_count,
            load_model_time_ms: (after.total_load_model_time_seconds
                - before.total_load_model_time_seconds)
                * 1000.0,
            add_rows_time_ms: (after.total_add_rows_time_seconds
                - before.total_add_rows_time_seconds)
                * 1000.0,
            set_bounds_time_ms: (after.total_set_bounds_time_seconds
                - before.total_set_bounds_time_seconds)
                * 1000.0,
            basis_set_time_ms: (after.total_basis_set_time_seconds
                - before.total_basis_set_time_seconds)
                * 1000.0,
            basis_padding_tight: after.basis_padding_tight - before.basis_padding_tight,
            basis_padding_slack: after.basis_padding_slack - before.basis_padding_slack,
            retry_level_histogram: after
                .retry_level_histogram
                .iter()
                .zip(&before.retry_level_histogram)
                .map(|(a, b)| a - b)
                .collect(),
        }
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
            result.basis_rejections += d.basis_rejections;
            result.simplex_iterations += d.simplex_iterations;
            result.solve_time_ms += d.solve_time_ms;
            result.load_model_count += d.load_model_count;
            result.add_rows_count += d.add_rows_count;
            result.load_model_time_ms += d.load_model_time_ms;
            result.add_rows_time_ms += d.add_rows_time_ms;
            result.set_bounds_time_ms += d.set_bounds_time_ms;
            result.basis_set_time_ms += d.basis_set_time_ms;
            result.basis_padding_tight += d.basis_padding_tight;
            result.basis_padding_slack += d.basis_padding_slack;
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
        result.basis_rejections += s.basis_rejections;
        result.first_try_successes += s.first_try_successes;
        result.basis_offered += s.basis_offered;
        result.load_model_count += s.load_model_count;
        result.add_rows_count += s.add_rows_count;
        result.total_load_model_time_seconds += s.total_load_model_time_seconds;
        result.total_add_rows_time_seconds += s.total_add_rows_time_seconds;
        result.total_set_bounds_time_seconds += s.total_set_bounds_time_seconds;
        result.total_basis_set_time_seconds += s.total_basis_set_time_seconds;
        result.basis_padding_tight += s.basis_padding_tight;
        result.basis_padding_slack += s.basis_padding_slack;
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

/// A single row in the solver stats log: (iteration, phase, stage, delta).
///
/// - `iteration`: 1-based iteration number.
/// - `phase`: `"forward"`, `"backward"`, or `"lower_bound"` — a `&'static str`
///   to avoid per-iteration heap allocation (F1-005 fix).
/// - `stage`: stage index for backward phase (per-stage), `-1` for forward/LB.
/// - `delta`: the solver counter delta for this entry.
pub type SolverStatsEntry = (u64, &'static str, i32, SolverStatsDelta);

/// Number of scalar fields in [`SolverStatsDelta`] (excludes the histogram `Vec`).
///
/// This constant defines the size of the fixed-size buffer used for MPI allreduce
/// and allgatherv operations. The 15 fields are packed in declaration order:
/// 10 `u64` fields (cast to `f64`) followed by 5 native `f64` fields.
pub const SOLVER_STATS_DELTA_SCALAR_FIELDS: usize = 15;

/// Number of `f64` values packed per scenario in [`pack_scenario_stats`].
///
/// Each scenario occupies `scenario_id_as_f64` + the 15 scalar fields = 16 values.
pub const SCENARIO_STATS_STRIDE: usize = 1 + SOLVER_STATS_DELTA_SCALAR_FIELDS;

/// Pack the 15 scalar fields of a [`SolverStatsDelta`] into a fixed-size `f64` array.
///
/// The packing order matches the declaration order of [`SolverStatsDelta`]:
/// - Indices 0–9: the ten `u64` fields cast to `f64` (exact for values ≤ 2^53).
/// - Indices 10–14: the five native `f64` fields.
///
/// The `retry_level_histogram` (`Vec<u64>`) is excluded: it is not part of the
/// summary allreduce and is not included in the per-scenario Parquet schema.
///
/// # Precision note
///
/// All `u64` fields represent event counts (LP solve counts, iteration counts).
/// At realistic workloads (≤ 10^6 scenarios × 10^3 stages), these counts fit
/// comfortably within the 53-bit mantissa of `f64`, so no precision is lost.
/// All `u64` fields represent event counts. At realistic workloads
/// (≤ 10^6 scenarios × 10^3 stages), counts stay well below 2^53, so the
/// cast is exact.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn pack_delta_scalars(delta: &SolverStatsDelta) -> [f64; SOLVER_STATS_DELTA_SCALAR_FIELDS] {
    [
        delta.lp_solves as f64,
        delta.lp_successes as f64,
        delta.first_try_successes as f64,
        delta.lp_failures as f64,
        delta.retry_attempts as f64,
        delta.basis_offered as f64,
        delta.basis_rejections as f64,
        delta.simplex_iterations as f64,
        delta.load_model_count as f64,
        delta.add_rows_count as f64,
        delta.solve_time_ms,
        delta.load_model_time_ms,
        delta.add_rows_time_ms,
        delta.set_bounds_time_ms,
        delta.basis_set_time_ms,
    ]
}

/// Unpack a fixed-size `f64` array (produced by [`pack_delta_scalars`]) back into
/// a [`SolverStatsDelta`].
///
/// The `retry_level_histogram` field is set to an empty `Vec` because the histogram
/// is excluded from both the allreduce summary and the per-scenario Parquet gather.
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
        basis_rejections: buf[6] as u64,
        simplex_iterations: buf[7] as u64,
        load_model_count: buf[8] as u64,
        add_rows_count: buf[9] as u64,
        solve_time_ms: buf[10],
        load_model_time_ms: buf[11],
        add_rows_time_ms: buf[12],
        set_bounds_time_ms: buf[13],
        basis_set_time_ms: buf[14],
        // Padding counters are application-level and excluded from MPI packing.
        basis_padding_tight: 0,
        basis_padding_slack: 0,
        retry_level_histogram: Vec::new(),
    }
}

/// Pack a slice of per-scenario `(scenario_id, delta)` pairs into a flat `f64`
/// buffer for use with `allgatherv`.
///
/// Each scenario contributes [`SCENARIO_STATS_STRIDE`] values:
/// `[scenario_id as f64, lp_solves as f64, ..., basis_set_time_ms]`.
/// The histogram is excluded.
#[must_use]
pub fn pack_scenario_stats(stats: &[(u32, SolverStatsDelta)]) -> Vec<f64> {
    let mut buf = Vec::with_capacity(stats.len() * SCENARIO_STATS_STRIDE);
    for (scenario_id, delta) in stats {
        buf.push(f64::from(*scenario_id));
        buf.extend_from_slice(&pack_delta_scalars(delta));
    }
    buf
}

/// Unpack a flat `f64` buffer (produced by [`pack_scenario_stats`]) back into a
/// `Vec<(u32, SolverStatsDelta)>`.
///
/// The buffer length must be a multiple of [`SCENARIO_STATS_STRIDE`].
/// Returns an empty `Vec` for an empty buffer.
///
/// # Panics
///
/// Panics in debug builds if `buf.len()` is not a multiple of `SCENARIO_STATS_STRIDE`.
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
            // SCENARIO_STATS_STRIDE = 1 + SOLVER_STATS_DELTA_SCALAR_FIELDS = 16.
            // Index 1..=15 therefore covers exactly the 15 scalar field slots.
            let arr = [
                chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7], chunk[8],
                chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14], chunk[15],
            ];
            (scenario_id, unpack_delta_scalars(&arr))
        })
        .collect()
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
            basis_rejections: 1,
            first_try_successes: 7,
            basis_offered: 8,
            load_model_count: 5,
            add_rows_count: 5,
            total_load_model_time_seconds: 1.0,
            total_add_rows_time_seconds: 0.5,
            total_set_bounds_time_seconds: 0.25,
            total_basis_set_time_seconds: 0.1,
            basis_padding_tight: 0,
            basis_padding_slack: 0,
            retry_level_histogram: vec![0; 12],
        };
        let after = SolverStatistics {
            solve_count: 20,
            success_count: 18,
            failure_count: 2,
            total_iterations: 1100,
            retry_count: 5,
            total_solve_time_seconds: 4.5,
            basis_rejections: 3,
            first_try_successes: 15,
            basis_offered: 17,
            load_model_count: 12,
            add_rows_count: 12,
            total_load_model_time_seconds: 3.0,
            total_add_rows_time_seconds: 1.5,
            total_set_bounds_time_seconds: 0.75,
            total_basis_set_time_seconds: 0.3,
            basis_padding_tight: 0,
            basis_padding_slack: 0,
            retry_level_histogram: vec![0; 12],
        };

        let delta = SolverStatsDelta::from_snapshots(&before, &after);
        assert_eq!(delta.lp_solves, 10);
        assert_eq!(delta.lp_successes, 9);
        assert_eq!(delta.first_try_successes, 8);
        assert_eq!(delta.lp_failures, 1);
        assert_eq!(delta.retry_attempts, 2);
        assert_eq!(delta.basis_offered, 9);
        assert_eq!(delta.basis_rejections, 2);
        assert_eq!(delta.simplex_iterations, 600);
        assert!((delta.solve_time_ms - 2500.0).abs() < 1e-6);
        assert_eq!(delta.load_model_count, 7);
        assert_eq!(delta.add_rows_count, 7);
        assert!((delta.load_model_time_ms - 2000.0).abs() < 1e-6);
        assert!((delta.add_rows_time_ms - 1000.0).abs() < 1e-6);
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
            basis_rejections: 0,
            first_try_successes: 5,
            basis_offered: 3,
            load_model_count: 3,
            add_rows_count: 3,
            total_load_model_time_seconds: 0.1,
            total_add_rows_time_seconds: 0.05,
            total_set_bounds_time_seconds: 0.02,
            total_basis_set_time_seconds: 0.01,
            basis_padding_tight: 0,
            basis_padding_slack: 0,
            retry_level_histogram: vec![0; 12],
        };
        let delta = SolverStatsDelta::from_snapshots(&snap, &snap);
        assert_eq!(delta.lp_solves, 0);
        assert_eq!(delta.lp_successes, 0);
        assert_eq!(delta.first_try_successes, 0);
        assert_eq!(delta.lp_failures, 0);
        assert_eq!(delta.retry_attempts, 0);
        assert_eq!(delta.basis_offered, 0);
        assert_eq!(delta.basis_rejections, 0);
        assert_eq!(delta.simplex_iterations, 0);
        assert!((delta.solve_time_ms).abs() < 1e-10);
        assert!((delta.load_model_time_ms).abs() < 1e-10);
        assert!((delta.add_rows_time_ms).abs() < 1e-10);
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
            basis_rejections: 1,
            simplex_iterations: 500,
            solve_time_ms: 100.0,
            load_model_count: 5,
            add_rows_count: 5,
            load_model_time_ms: 10.0,
            add_rows_time_ms: 5.0,
            set_bounds_time_ms: 2.0,
            basis_set_time_ms: 1.0,
            basis_padding_tight: 3,
            basis_padding_slack: 7,
            retry_level_histogram: vec![0; 12],
        };
        let d2 = SolverStatsDelta {
            lp_solves: 20,
            lp_successes: 19,
            first_try_successes: 17,
            lp_failures: 1,
            retry_attempts: 3,
            basis_offered: 15,
            basis_rejections: 2,
            simplex_iterations: 800,
            solve_time_ms: 200.0,
            load_model_count: 10,
            add_rows_count: 10,
            load_model_time_ms: 20.0,
            add_rows_time_ms: 10.0,
            set_bounds_time_ms: 4.0,
            basis_set_time_ms: 2.0,
            basis_padding_tight: 5,
            basis_padding_slack: 2,
            retry_level_histogram: vec![0; 12],
        };

        let agg = SolverStatsDelta::aggregate([d1, d2].iter());
        assert_eq!(agg.lp_solves, 30);
        assert_eq!(agg.lp_successes, 28);
        assert_eq!(agg.first_try_successes, 25);
        assert_eq!(agg.lp_failures, 2);
        assert_eq!(agg.retry_attempts, 5);
        assert_eq!(agg.basis_offered, 22);
        assert_eq!(agg.basis_rejections, 3);
        assert_eq!(agg.simplex_iterations, 1300);
        assert!((agg.solve_time_ms - 300.0).abs() < 1e-6);
        assert_eq!(agg.load_model_count, 15);
        assert_eq!(agg.add_rows_count, 15);
        assert!((agg.load_model_time_ms - 30.0).abs() < 1e-6);
        assert!((agg.add_rows_time_ms - 15.0).abs() < 1e-6);
        assert!((agg.set_bounds_time_ms - 6.0).abs() < 1e-6);
        assert!((agg.basis_set_time_ms - 3.0).abs() < 1e-6);
        assert_eq!(agg.basis_padding_tight, 8);
        assert_eq!(agg.basis_padding_slack, 9);
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
            basis_rejections: 1,
            first_try_successes: 7,
            basis_offered: 8,
            load_model_count: 5,
            add_rows_count: 5,
            total_load_model_time_seconds: 1.0,
            total_add_rows_time_seconds: 0.5,
            total_set_bounds_time_seconds: 0.25,
            total_basis_set_time_seconds: 0.05,
            basis_padding_tight: 4,
            basis_padding_slack: 6,
            retry_level_histogram: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let s2 = SolverStatistics {
            solve_count: 20,
            success_count: 18,
            failure_count: 2,
            total_iterations: 1100,
            retry_count: 5,
            total_solve_time_seconds: 4.5,
            basis_rejections: 3,
            first_try_successes: 15,
            basis_offered: 17,
            load_model_count: 12,
            add_rows_count: 12,
            total_load_model_time_seconds: 3.0,
            total_add_rows_time_seconds: 1.5,
            total_set_bounds_time_seconds: 0.75,
            total_basis_set_time_seconds: 0.15,
            basis_padding_tight: 10,
            basis_padding_slack: 20,
            retry_level_histogram: vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        let agg = aggregate_solver_statistics([s1, s2].into_iter());
        assert_eq!(agg.solve_count, 30);
        assert_eq!(agg.success_count, 27);
        assert_eq!(agg.failure_count, 3);
        assert_eq!(agg.total_iterations, 1600);
        assert_eq!(agg.retry_count, 8);
        assert!((agg.total_solve_time_seconds - 6.5).abs() < 1e-10);
        assert_eq!(agg.basis_rejections, 4);
        assert_eq!(agg.first_try_successes, 22);
        assert_eq!(agg.basis_offered, 25);
        assert_eq!(agg.load_model_count, 17);
        assert_eq!(agg.add_rows_count, 17);
        assert!((agg.total_load_model_time_seconds - 4.0).abs() < 1e-10);
        assert!((agg.total_add_rows_time_seconds - 2.0).abs() < 1e-10);
        assert!((agg.total_set_bounds_time_seconds - 1.0).abs() < 1e-10);
        assert!((agg.total_basis_set_time_seconds - 0.2).abs() < 1e-10);
        assert_eq!(agg.retry_level_histogram[0], 1);
        assert_eq!(agg.retry_level_histogram[1], 2);
        assert_eq!(agg.retry_level_histogram[2], 0);
        assert_eq!(agg.basis_padding_tight, 14);
        assert_eq!(agg.basis_padding_slack, 26);
    }

    fn make_delta(lp_solves: u64) -> SolverStatsDelta {
        SolverStatsDelta {
            lp_solves,
            lp_successes: lp_solves,
            first_try_successes: lp_solves / 2,
            lp_failures: 0,
            retry_attempts: 1,
            basis_offered: lp_solves,
            basis_rejections: 2,
            simplex_iterations: lp_solves * 10,
            solve_time_ms: lp_solves as f64 * 0.5,
            load_model_count: 3,
            add_rows_count: 4,
            load_model_time_ms: 1.5,
            add_rows_time_ms: 2.5,
            set_bounds_time_ms: 0.25,
            basis_set_time_ms: 0.125,
            basis_padding_tight: 0,
            basis_padding_slack: 0,
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
        assert_eq!(unpacked.basis_rejections, delta.basis_rejections);
        assert_eq!(unpacked.simplex_iterations, delta.simplex_iterations);
        assert_eq!(unpacked.load_model_count, delta.load_model_count);
        assert_eq!(unpacked.add_rows_count, delta.add_rows_count);
        assert!((unpacked.solve_time_ms - delta.solve_time_ms).abs() < 1e-10);
        assert!((unpacked.load_model_time_ms - delta.load_model_time_ms).abs() < 1e-10);
        assert!((unpacked.add_rows_time_ms - delta.add_rows_time_ms).abs() < 1e-10);
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
    fn test_solver_stats_delta_includes_padding_fields() {
        // Acceptance criterion: from_snapshots correctly computes the delta for
        // basis_padding_tight and basis_padding_slack.
        let before = SolverStatistics {
            basis_padding_tight: 3,
            basis_padding_slack: 5,
            ..SolverStatistics::default()
        };
        let after = SolverStatistics {
            basis_padding_tight: 10,
            basis_padding_slack: 12,
            ..SolverStatistics::default()
        };

        let delta = SolverStatsDelta::from_snapshots(&before, &after);
        assert_eq!(delta.basis_padding_tight, 7);
        assert_eq!(delta.basis_padding_slack, 7);
    }
}
