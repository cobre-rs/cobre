//! Scaling report data structures and computation for LP conditioning diagnostics.
//!
//! Captures the pre-scaling and post-scaling coefficient ranges of the stage template
//! LPs. Written once after template build as a JSON diagnostic artifact.

use cobre_solver::StageTemplate;
use serde::Serialize;

/// Complete scaling report for all stages.
#[derive(Debug, Clone, Serialize)]
pub struct ScalingReport {
    /// The cost scale factor applied to objective coefficients.
    pub cost_scale_factor: f64,

    /// Per-stage scaling reports.
    pub stages: Vec<StageScalingReport>,

    /// Cross-stage summary statistics.
    pub summary: ScalingReportSummary,
}

/// Scaling report for a single stage.
#[derive(Debug, Clone, Serialize)]
pub struct StageScalingReport {
    /// Stage index (0-based).
    pub stage_id: usize,

    /// LP dimensions.
    pub dimensions: LpDimensions,

    /// Coefficient ranges before column/row scaling.
    pub pre_scaling: CoefficientRange,

    /// Coefficient ranges after column/row scaling (and cost scaling).
    pub post_scaling: CoefficientRange,

    /// Column scale factor summary.
    pub col_scale: ScaleFactorSummary,

    /// Row scale factor summary.
    pub row_scale: ScaleFactorSummary,
}

/// LP matrix and objective dimensions.
#[derive(Debug, Clone, Serialize)]
pub struct LpDimensions {
    /// Number of columns (decision variables).
    pub num_cols: usize,

    /// Number of structural rows (constraints).
    pub num_rows: usize,

    /// Number of nonzero entries in the constraint matrix.
    pub num_nz: usize,
}

/// Coefficient range for matrix and objective.
#[derive(Debug, Clone, Serialize)]
pub struct CoefficientRange {
    /// `[min|A_ij|, max|A_ij|]` over nonzero matrix entries.
    pub matrix_coeff_range: [f64; 2],

    /// `max / min` of absolute nonzero matrix values.
    pub matrix_coeff_ratio: f64,

    /// `[min|c_j|, max|c_j|]` over nonzero objective coefficients.
    pub objective_range: [f64; 2],

    /// `max / min` of absolute nonzero objective values.
    pub objective_ratio: f64,
}

/// Summary of a scale factor vector.
#[derive(Debug, Clone, Serialize)]
pub struct ScaleFactorSummary {
    /// Minimum scale factor.
    pub min: f64,

    /// Maximum scale factor.
    pub max: f64,

    /// Median scale factor.
    pub median: f64,

    /// Number of scale factors (equal to `num_cols` or `num_rows`).
    pub count: usize,
}

/// Cross-stage summary of scaling effectiveness.
#[derive(Debug, Clone, Serialize)]
pub struct ScalingReportSummary {
    /// Maximum pre-scaling matrix coefficient ratio across all stages.
    pub worst_pre_scaling_matrix_ratio: f64,

    /// Maximum post-scaling matrix coefficient ratio across all stages.
    pub worst_post_scaling_matrix_ratio: f64,

    /// Improvement factor: `worst_pre / worst_post`.
    pub improvement_factor: f64,

    /// Number of stages.
    pub num_stages: usize,
}

/// Compute the min and max absolute values over nonzero entries.
///
/// Returns `(min_abs, max_abs)`. For an empty or all-zero slice, returns
/// `(f64::INFINITY, 0.0)` which produces a ratio of 0.
fn compute_abs_range(values: &[f64]) -> (f64, f64) {
    let mut min_abs = f64::INFINITY;
    let mut max_abs = 0.0_f64;
    for &v in values {
        if v != 0.0 {
            let abs_v = v.abs();
            min_abs = min_abs.min(abs_v);
            max_abs = max_abs.max(abs_v);
        }
    }
    (min_abs, max_abs)
}

/// Compute the coefficient range for a stage template's matrix and objective.
#[must_use]
pub fn compute_coefficient_range(tmpl: &StageTemplate) -> CoefficientRange {
    let (mat_min, mat_max) = compute_abs_range(&tmpl.values);
    let (obj_min, obj_max) = compute_abs_range(&tmpl.objective);

    let matrix_ratio = if mat_min > 0.0 && mat_min.is_finite() {
        mat_max / mat_min
    } else {
        0.0
    };
    let objective_ratio = if obj_min > 0.0 && obj_min.is_finite() {
        obj_max / obj_min
    } else {
        0.0
    };

    CoefficientRange {
        matrix_coeff_range: [mat_min, mat_max],
        matrix_coeff_ratio: matrix_ratio,
        objective_range: [obj_min, obj_max],
        objective_ratio,
    }
}

/// Compute the median of a slice by sorting a copy.
///
/// Returns 0.0 for an empty slice.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    // `total_cmp` provides IEEE-754 total ordering on f64; required so that
    // NaN inputs land in a deterministic position and the sort is
    // declaration-order-invariant (Cobre hard rule).
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 0 {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

/// Summarize a scale factor vector.
///
/// Returns a summary with zeros for an empty vector.
#[must_use]
pub fn summarize_scale_factors(factors: &[f64]) -> ScaleFactorSummary {
    if factors.is_empty() {
        return ScaleFactorSummary {
            min: 0.0,
            max: 0.0,
            median: 0.0,
            count: 0,
        };
    }
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in factors {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    ScaleFactorSummary {
        min: min_val,
        max: max_val,
        median: median(factors),
        count: factors.len(),
    }
}

/// Build a complete [`ScalingReport`] from per-stage pre/post data.
///
/// `cost_scale_factor` is the factor that was applied to objective coefficients
/// during template building.
#[must_use]
pub fn build_scaling_report(
    cost_scale_factor: f64,
    stage_reports: Vec<StageScalingReport>,
) -> ScalingReport {
    let num_stages = stage_reports.len();
    let worst_pre = stage_reports
        .iter()
        .map(|s| s.pre_scaling.matrix_coeff_ratio)
        .fold(0.0_f64, f64::max);
    let worst_post = stage_reports
        .iter()
        .map(|s| s.post_scaling.matrix_coeff_ratio)
        .fold(0.0_f64, f64::max);
    let improvement = if worst_post > 0.0 {
        worst_pre / worst_post
    } else {
        0.0
    };

    ScalingReport {
        cost_scale_factor,
        stages: stage_reports,
        summary: ScalingReportSummary {
            worst_pre_scaling_matrix_ratio: worst_pre,
            worst_post_scaling_matrix_ratio: worst_post,
            improvement_factor: improvement,
            num_stages,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_abs_range_basic() {
        let values = [1.0, -3.0, 0.0, 2.0, -0.5];
        let (min_abs, max_abs) = compute_abs_range(&values);
        assert!((min_abs - 0.5).abs() < 1e-15);
        assert!((max_abs - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_compute_abs_range_all_zero() {
        let values = [0.0, 0.0];
        let (min_abs, max_abs) = compute_abs_range(&values);
        assert!(min_abs.is_infinite());
        assert!((max_abs - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_compute_abs_range_empty() {
        let (min_abs, max_abs) = compute_abs_range(&[]);
        assert!(min_abs.is_infinite());
        assert!((max_abs - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_median_odd() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_median_even() {
        assert!((median(&[4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < 1e-15);
    }

    #[test]
    fn test_median_single() {
        assert!((median(&[7.0]) - 7.0).abs() < 1e-15);
    }

    #[test]
    fn test_median_empty() {
        assert!((median(&[]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_summarize_scale_factors() {
        let factors = [0.1, 1.0, 10.0, 1.0, 0.5];
        let summary = summarize_scale_factors(&factors);
        assert!((summary.min - 0.1).abs() < 1e-15);
        assert!((summary.max - 10.0).abs() < 1e-15);
        assert!((summary.median - 1.0).abs() < 1e-15);
        assert_eq!(summary.count, 5);
    }

    #[test]
    fn test_summarize_scale_factors_empty() {
        let summary = summarize_scale_factors(&[]);
        assert_eq!(summary.count, 0);
        assert!((summary.min - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_coefficient_range_from_template() {
        let tmpl = StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0_i32, 2, 2, 3],
            row_indices: vec![0_i32, 1, 1],
            values: vec![1.0, 2.0, 0.5],
            col_lower: vec![0.0; 3],
            col_upper: vec![10.0; 3],
            objective: vec![0.0, 1.0, 50.0],
            row_lower: vec![0.0; 2],
            row_upper: vec![0.0; 2],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let range = compute_coefficient_range(&tmpl);
        assert!((range.matrix_coeff_range[0] - 0.5).abs() < 1e-15);
        assert!((range.matrix_coeff_range[1] - 2.0).abs() < 1e-15);
        assert!((range.matrix_coeff_ratio - 4.0).abs() < 1e-15);
        assert!((range.objective_range[0] - 1.0).abs() < 1e-15);
        assert!((range.objective_range[1] - 50.0).abs() < 1e-15);
        assert!((range.objective_ratio - 50.0).abs() < 1e-15);
    }

    #[test]
    fn test_build_scaling_report_summary() {
        let stage1 = StageScalingReport {
            stage_id: 0,
            dimensions: LpDimensions {
                num_cols: 10,
                num_rows: 5,
                num_nz: 20,
            },
            pre_scaling: CoefficientRange {
                matrix_coeff_range: [0.01, 1000.0],
                matrix_coeff_ratio: 100_000.0,
                objective_range: [1.0, 100.0],
                objective_ratio: 100.0,
            },
            post_scaling: CoefficientRange {
                matrix_coeff_range: [0.5, 5.0],
                matrix_coeff_ratio: 10.0,
                objective_range: [0.001, 0.1],
                objective_ratio: 100.0,
            },
            col_scale: ScaleFactorSummary {
                min: 0.01,
                max: 100.0,
                median: 1.0,
                count: 10,
            },
            row_scale: ScaleFactorSummary {
                min: 0.1,
                max: 10.0,
                median: 1.0,
                count: 5,
            },
        };
        let report = build_scaling_report(1000.0, vec![stage1]);
        assert_eq!(report.summary.num_stages, 1);
        assert!((report.summary.worst_pre_scaling_matrix_ratio - 100_000.0).abs() < 1e-6);
        assert!((report.summary.worst_post_scaling_matrix_ratio - 10.0).abs() < 1e-6);
        assert!((report.summary.improvement_factor - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn total_cmp_handles_nan_deterministically() {
        // Verify that sorting with total_cmp produces a stable, NaN-safe order.
        // NaN values must not cause a panic or non-deterministic ordering.
        let mut values = [f64::NAN, 3.0, 1.0, f64::NAN, 2.0];
        values.sort_by(f64::total_cmp);
        // total_cmp places NaN after all finite values (NaN > everything in
        // total order). The finite elements must be in ascending order.
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());

        // Descending order (as used in compute_cvar / aggregation sort).
        // total_cmp places +NaN > +inf > finite, so reversing it (b.total_cmp(a))
        // puts +NaN at the front. The deterministic placement is what matters
        // for declaration-order-invariance; downstream code should not see NaN
        // in production, but if it does, the position is reproducible.
        let mut desc = [f64::NAN, 3.0, 1.0, f64::NAN, 2.0];
        desc.sort_by(|a, b| b.total_cmp(a));
        assert!(desc[0].is_nan());
        assert!(desc[1].is_nan());
        assert_eq!(desc[2], 3.0);
        assert_eq!(desc[3], 2.0);
        assert_eq!(desc[4], 1.0);

        // Determinism check: re-sorting the same input twice produces the
        // same permutation by-bits.
        let mut a = [f64::NAN, 3.0, 1.0, f64::NAN, 2.0];
        let mut b = [f64::NAN, 3.0, 1.0, f64::NAN, 2.0];
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits());
        }
    }
}
