//! Trajectory records produced by the SDDP forward pass.
//!
//! A [`TrajectoryRecord`] captures the complete LP solution for one scenario
//! at one stage. It is the unit of data that flows from the forward pass to
//! the backward pass: the forward pass produces one record per
//! `(scenario, stage)` pair and stores them in a flat `Vec` indexed as
//!
//! ```text
//! records[scenario * n_stages + stage]
//! ```
//!
//! The backward pass reads the `state` and `dual` fields from each record to
//! generate Benders cuts.
//!
//! ## Ownership
//!
//! `TrajectoryRecord`s are owned by the forward pass workspace. They are
//! **not** stored inside the [`FutureCostFunction`]. Serialization support
//! will be added when checkpoint/restart is implemented.
//!
//! [`FutureCostFunction`]: crate::cut::fcf::FutureCostFunction

/// LP solution for one scenario at one stage, produced by the forward pass.
///
/// All four fields are taken directly from the LP solver output:
///
/// - `primal` is the full primal solution vector (all LP variables in column
///   order).
/// - `dual` is the full dual solution vector (one value per LP constraint).
/// - `stage_cost` is the LP objective value at this stage, excluding the
///   future-cost theta variable contribution.
/// - `state` is the end-of-stage state vector (length `n_state`), extracted
///   from the primal solution by the stage indexer.
///
/// The backward pass uses `state` and `dual` to compute cut coefficients.
/// The training loop monitors `stage_cost` for convergence statistics.
#[derive(Debug, Clone)]
pub struct TrajectoryRecord {
    /// Full primal solution vector (all LP variables in column order).
    pub primal: Vec<f64>,

    /// Full dual solution vector (one value per LP constraint).
    pub dual: Vec<f64>,

    /// LP objective value at this stage (excluding the future-cost
    /// theta variable contribution).
    pub stage_cost: f64,

    /// End-of-stage state vector (length `n_state`), extracted from the
    /// primal solution by the stage indexer.
    pub state: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::TrajectoryRecord;

    #[test]
    fn construct_and_access_all_fields() {
        let record = TrajectoryRecord {
            primal: vec![1.0, 2.0, 3.0],
            dual: vec![0.5, 0.6],
            stage_cost: 42.0,
            state: vec![9.0, 8.0],
        };

        assert_eq!(record.primal, vec![1.0, 2.0, 3.0]);
        assert_eq!(record.dual, vec![0.5, 0.6]);
        assert_eq!(record.stage_cost, 42.0);
        assert_eq!(record.state, vec![9.0, 8.0]);
    }

    #[test]
    fn stage_cost_value_is_accessible() {
        // Acceptance criterion: given stage_cost: 42.0, accessing stage_cost
        // returns 42.0.
        let record = TrajectoryRecord {
            primal: vec![],
            dual: vec![],
            stage_cost: 42.0,
            state: vec![],
        };
        assert_eq!(record.stage_cost, 42.0);
    }

    #[test]
    fn clone_produces_identical_independent_copy() {
        let original = TrajectoryRecord {
            primal: vec![1.0, 2.0],
            dual: vec![3.0],
            stage_cost: 7.5,
            state: vec![4.0, 5.0],
        };

        let mut cloned = original.clone();

        // Field equality.
        assert_eq!(cloned.primal, original.primal);
        assert_eq!(cloned.dual, original.dual);
        assert_eq!(cloned.stage_cost, original.stage_cost);
        assert_eq!(cloned.state, original.state);

        // Independence: mutating the clone does not affect the original.
        cloned.stage_cost = 0.0;
        cloned.primal.push(99.0);
        assert_eq!(original.stage_cost, 7.5);
        assert_eq!(original.primal.len(), 2);
    }

    #[test]
    fn debug_format_is_non_empty() {
        let record = TrajectoryRecord {
            primal: vec![1.0],
            dual: vec![2.0],
            stage_cost: 3.0,
            state: vec![4.0],
        };
        let s = format!("{record:?}");
        assert!(!s.is_empty());
    }

    #[test]
    fn flat_vec_indexing_pattern_works_correctly() {
        // Demonstrates the canonical storage layout used by the forward pass:
        // records[scenario * n_stages + stage]
        let n_stages: usize = 5;
        let scenario: usize = 1;

        // n_scenarios * n_stages = 15, well within u8 range.
        let mut records: Vec<TrajectoryRecord> = (0_u8..15_u8)
            .map(|i| TrajectoryRecord {
                primal: vec![],
                dual: vec![],
                stage_cost: f64::from(i),
                state: vec![],
            })
            .collect();

        // Write to scenario=1, stage=3
        records[scenario * n_stages + 3].stage_cost = 999.0;

        assert_eq!(records[scenario * n_stages + 3].stage_cost, 999.0);
        // Adjacent entries are unaffected. Initial stage_cost for each record
        // was set to its flat index (scenario * n_stages + stage).
        // scenario=1, stage=2 → index=7, scenario=1, stage=4 → index=9.
        assert_eq!(records[scenario * n_stages + 2].stage_cost, 7.0);
        assert_eq!(records[scenario * n_stages + 4].stage_cost, 9.0);
    }
}
