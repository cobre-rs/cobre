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

    /// Per-level retry success histogram delta. Length depends on the solver
    /// backend (e.g. 12 for `HiGHS`).
    pub retry_level_histogram: Vec<u64>,
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
            retry_level_histogram: after
                .retry_level_histogram
                .iter()
                .zip(&before.retry_level_histogram)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Sum a slice of deltas element-wise into a single aggregate.
    ///
    /// Returns `Default` (all zeros) for an empty slice.
    #[must_use]
    pub fn aggregate(deltas: &[Self]) -> Self {
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
            if result.retry_level_histogram.is_empty() && !d.retry_level_histogram.is_empty() {
                result
                    .retry_level_histogram
                    .resize(d.retry_level_histogram.len(), 0);
            }
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
/// Sums all scalar counters across the given slice of statistics. This is used
/// to get a single snapshot before/after a phase that distributes work across
/// multiple solver instances.
#[must_use]
pub fn aggregate_solver_statistics(stats: &[SolverStatistics]) -> SolverStatistics {
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
        if result.retry_level_histogram.is_empty() && !s.retry_level_histogram.is_empty() {
            result
                .retry_level_histogram
                .resize(s.retry_level_histogram.len(), 0);
        }
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
/// - `phase`: `"forward"`, `"backward"`, or `"lower_bound"`.
/// - `stage`: stage index for backward phase (per-stage), `-1` for forward/LB.
/// - `delta`: the solver counter delta for this entry.
pub type SolverStatsEntry = (u64, String, i32, SolverStatsDelta);

#[cfg(test)]
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
        let agg = SolverStatsDelta::aggregate(&[]);
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
            retry_level_histogram: vec![0; 12],
        };

        let agg = SolverStatsDelta::aggregate(&[d1, d2]);
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
            retry_level_histogram: vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        let agg = aggregate_solver_statistics(&[s1, s2]);
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
    }
}
