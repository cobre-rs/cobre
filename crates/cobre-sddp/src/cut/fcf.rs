//! Future Cost Function (FCF) — all-stages container for Benders cuts.
//!
//! [`FutureCostFunction`] wraps one [`CutPool`] per stage and provides the
//! high-level interface consumed by the training loop:
//!
//! - [`add_cut`] — insert a Benders cut at a given stage.
//! - [`active_cuts`] — iterate over active cuts at a given stage for LP
//!   construction.
//! - [`evaluate_at_state`] — evaluate the FCF at a given state (max over
//!   active cuts).
//! - [`total_active_cuts`] — aggregate count across all stages.
//! - [`deactivate`] — delegate cut deactivation to a specific stage pool.
//!
//! ## Stage indexing
//!
//! The FCF uses **0-based internal stage indices** throughout its API.
//! The SDDP algorithm specification uses 1-based stage numbers (`t = 1..T`).
//! Callers are responsible for converting: pass `stage = t - 1` when calling
//! FCF methods.
//!
//! ## Memory allocation
//!
//! All pool storage is pre-allocated at construction time via
//! [`FutureCostFunction::new`]. No heap allocation occurs during the training
//! loop hot path.
//!
//! [`add_cut`]: FutureCostFunction::add_cut
//! [`active_cuts`]: FutureCostFunction::active_cuts
//! [`evaluate_at_state`]: FutureCostFunction::evaluate_at_state
//! [`total_active_cuts`]: FutureCostFunction::total_active_cuts
//! [`deactivate`]: FutureCostFunction::deactivate
//!
//! ## Example
//!
//! ```rust
//! use cobre_sddp::cut::fcf::FutureCostFunction;
//!
//! let mut fcf = FutureCostFunction::new(3, 4, 10, 50, 0);
//! assert_eq!(fcf.pools.len(), 3);
//!
//! let coeffs = vec![1.0, 0.0, 0.0, 0.0];
//! fcf.add_cut(1, 0, 0, 5.0, &coeffs);
//!
//! let cuts: Vec<_> = fcf.active_cuts(1).collect();
//! assert_eq!(cuts.len(), 1);
//! assert_eq!(fcf.total_active_cuts(), 1);
//! ```

use super::pool::CutPool;

/// All-stages container for the Future Cost Function (FCF).
///
/// Holds one [`CutPool`] per stage. All pools share the same
/// `state_dimension`. The FCF itself is a thin orchestration layer; all
/// per-cut logic is delegated to [`CutPool`].
///
/// ## Stage indexing
///
/// FCF methods accept **0-based stage indices** (`0..num_stages`). The SDDP
/// spec uses 1-based stage numbers. Convert with `stage = t - 1` before
/// calling.
#[derive(Debug, Clone)]
pub struct FutureCostFunction {
    /// One cut pool per stage, indexed 0-based.
    pub pools: Vec<CutPool>,

    /// Length of the state vector shared across all stages.
    pub state_dimension: usize,

    /// Number of forward passes per training iteration. Immutable after
    /// construction.
    pub forward_passes: u32,
}

impl FutureCostFunction {
    /// Construct a new `FutureCostFunction` with pre-allocated pools.
    ///
    /// Creates `num_stages` [`CutPool`]s, each with capacity:
    ///
    /// ```text
    /// capacity = warm_start_count + max_iterations * forward_passes
    /// ```
    ///
    /// The capacity arithmetic uses `u64` to avoid overflow for large
    /// `max_iterations` and `forward_passes` values before converting to
    /// `usize`.
    ///
    /// # Parameters
    ///
    /// - `num_stages`: number of stages in the planning horizon.
    /// - `state_dimension`: length of the state vector (number of coefficients
    ///   per Benders cut).
    /// - `forward_passes`: number of forward passes per training iteration.
    /// - `max_iterations`: maximum number of training iterations.
    /// - `warm_start_count`: number of warm-start cuts pre-loaded before
    ///   training begins.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::fcf::FutureCostFunction;
    ///
    /// let fcf = FutureCostFunction::new(5, 9, 10, 100, 0);
    /// assert_eq!(fcf.pools.len(), 5);
    /// // capacity = 0 + 100 * 10 = 1000
    /// assert_eq!(fcf.pools[0].capacity, 1000);
    /// ```
    #[must_use]
    pub fn new(
        num_stages: usize,
        state_dimension: usize,
        forward_passes: u32,
        max_iterations: u64,
        warm_start_count: u32,
    ) -> Self {
        // Use u64 arithmetic to prevent overflow before converting to usize.
        // The cast cannot realistically truncate on any 64-bit platform:
        // pool capacity is bounded by available memory, which fits in usize on
        // all supported targets. Clippy's truncation warning is suppressed here
        // because the same pattern is used by the underlying CutPool (see
        // pool.rs slot_index), and both are guarded by debug_assert at
        // insertion time.
        #[allow(clippy::cast_possible_truncation)]
        let capacity: usize =
            (u64::from(warm_start_count) + max_iterations * u64::from(forward_passes)) as usize;

        let pools = (0..num_stages)
            .map(|_| CutPool::new(capacity, state_dimension, forward_passes, warm_start_count))
            .collect();

        Self {
            pools,
            state_dimension,
            forward_passes,
        }
    }

    /// Reconstruct a `FutureCostFunction` from deserialized policy checkpoint data.
    ///
    /// Builds one [`CutPool::from_deserialized`] per stage. The pools are
    /// tightly sized to hold exactly the loaded cuts, with no capacity for
    /// additional training cuts.
    ///
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] if:
    /// - `stage_results` is empty
    /// - `state_dimension` is inconsistent across stages
    ///
    /// [`SddpError::Validation`]: crate::SddpError::Validation
    pub fn from_deserialized(
        stage_results: &[cobre_io::StageCutsReadResult],
    ) -> Result<Self, crate::SddpError> {
        if stage_results.is_empty() {
            return Err(crate::SddpError::Validation(
                "from_deserialized: stage_results is empty".to_string(),
            ));
        }

        let state_dimension = stage_results[0].state_dimension as usize;
        for sr in &stage_results[1..] {
            if sr.state_dimension as usize != state_dimension {
                return Err(crate::SddpError::Validation(format!(
                    "from_deserialized: inconsistent state_dimension: stage {} has {}, \
                     expected {} (from stage {})",
                    sr.stage_id, sr.state_dimension, state_dimension, stage_results[0].stage_id
                )));
            }
        }

        let pools = stage_results
            .iter()
            .map(|sr| CutPool::from_deserialized(state_dimension, &sr.cuts))
            .collect();

        Ok(Self {
            pools,
            state_dimension,
            forward_passes: 0,
        })
    }

    /// Build an FCF with warm-start cuts plus capacity for new training cuts.
    ///
    /// Each pool is constructed with [`CutPool::new_with_warm_start`], giving
    /// capacity for both the loaded cuts and `max_iterations * forward_passes`
    /// new training cuts.
    ///
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] if `stage_results` is empty or if
    /// `state_dimension` is inconsistent across stages.
    ///
    /// [`SddpError::Validation`]: crate::SddpError::Validation
    pub fn new_with_warm_start(
        stage_results: &[cobre_io::StageCutsReadResult],
        forward_passes: u32,
        max_iterations: u64,
    ) -> Result<Self, crate::SddpError> {
        if stage_results.is_empty() {
            return Err(crate::SddpError::Validation(
                "new_with_warm_start: stage_results is empty".to_string(),
            ));
        }

        let state_dimension = stage_results[0].state_dimension as usize;
        for sr in &stage_results[1..] {
            if sr.state_dimension as usize != state_dimension {
                return Err(crate::SddpError::Validation(format!(
                    "new_with_warm_start: inconsistent state_dimension: stage {} has {}, \
                     expected {} (from stage {})",
                    sr.stage_id, sr.state_dimension, state_dimension, stage_results[0].stage_id
                )));
            }
        }

        let pools = stage_results
            .iter()
            .map(|sr| {
                CutPool::new_with_warm_start(
                    state_dimension,
                    forward_passes,
                    max_iterations,
                    &sr.cuts,
                )
            })
            .collect();

        Ok(Self {
            pools,
            state_dimension,
            forward_passes,
        })
    }

    /// Insert a Benders cut at the given stage.
    ///
    /// Delegates to `pools[stage].add_cut(...)`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()` or if `coefficients.len() !=
    /// state_dimension`.
    ///
    /// # Parameters
    ///
    /// - `stage`: 0-based stage index.
    /// - `iteration`: training iteration counter (0-based).
    /// - `forward_pass_index`: index of the forward pass within the iteration.
    /// - `intercept`: cut intercept value.
    /// - `coefficients`: cut gradient with respect to the state; must have
    ///   length `state_dimension`.
    pub fn add_cut(
        &mut self,
        stage: usize,
        iteration: u64,
        forward_pass_index: u32,
        intercept: f64,
        coefficients: &[f64],
    ) {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].add_cut(iteration, forward_pass_index, intercept, coefficients);
    }

    /// Iterate over active cuts at the given stage.
    ///
    /// Yields `(slot_index, intercept, coefficient_slice)` for every active
    /// cut in `pools[stage]`. Used by LP construction to add cut constraints.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()`.
    ///
    /// # Parameters
    ///
    /// - `stage`: 0-based stage index.
    pub fn active_cuts(&self, stage: usize) -> impl Iterator<Item = (usize, f64, &[f64])> {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].active_cuts()
    }

    /// Evaluate the FCF at a given state for the specified stage.
    ///
    /// Returns the maximum over all active cuts of
    /// `intercept + coefficients · state`. Returns [`f64::NEG_INFINITY`] if
    /// the stage has no active cuts.
    ///
    /// Delegates to `pools[stage].evaluate_at_state(values)`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()` or if `values.len() !=
    /// state_dimension`.
    ///
    /// # Parameters
    ///
    /// - `stage`: 0-based stage index.
    /// - `values`: state vector; must have length `state_dimension`.
    #[must_use]
    pub fn evaluate_at_state(&self, stage: usize, values: &[f64]) -> f64 {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].evaluate_at_state(values)
    }

    /// Return the total number of active cuts across all stages.
    ///
    /// Sums `active_count()` over every pool in the FCF.
    #[must_use]
    pub fn total_active_cuts(&self) -> usize {
        self.pools.iter().map(CutPool::active_count).sum()
    }

    /// Deactivate the cuts at the given slot indices for the specified stage.
    ///
    /// Delegates to `pools[stage].deactivate(indices)`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()`.
    ///
    /// # Parameters
    ///
    /// - `stage`: 0-based stage index.
    /// - `indices`: slice of slot indices (0-based) to deactivate.
    pub fn deactivate(&mut self, stage: usize, indices: &[u32]) {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].deactivate(indices);
    }

    /// Compute sparsity reports for all stages.
    ///
    /// Returns a vector of [`SparsityReport`] values, one per stage, in
    /// stage order. Delegates to [`CutPool::sparsity_report`] for each pool.
    ///
    /// [`SparsityReport`]: super::pool::SparsityReport
    #[must_use]
    pub fn sparsity_reports(&self) -> Vec<super::pool::SparsityReport> {
        self.pools.iter().map(CutPool::sparsity_report).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::FutureCostFunction;

    #[test]
    fn new_creates_correct_number_of_pools() {
        let fcf = FutureCostFunction::new(5, 9, 10, 100, 0);
        assert_eq!(fcf.pools.len(), 5);
    }

    #[test]
    fn new_each_pool_has_correct_capacity_no_warmstart() {
        // capacity = 0 + 100 * 10 = 1000
        let fcf = FutureCostFunction::new(5, 9, 10, 100, 0);
        for pool in &fcf.pools {
            assert_eq!(pool.capacity, 1000);
            assert_eq!(pool.state_dimension, 9);
            assert_eq!(pool.forward_passes, 10);
            assert_eq!(pool.warm_start_count, 0);
        }
    }

    #[test]
    fn new_each_pool_has_correct_capacity_with_warmstart() {
        // capacity = 5 + 100 * 10 = 1005
        let fcf = FutureCostFunction::new(3, 4, 10, 100, 5);
        for pool in &fcf.pools {
            assert_eq!(pool.capacity, 1005);
            assert_eq!(pool.warm_start_count, 5);
        }
    }

    #[test]
    fn new_all_pools_start_with_zero_active_cuts() {
        let fcf = FutureCostFunction::new(4, 3, 5, 20, 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn new_zero_stages_is_valid() {
        let fcf = FutureCostFunction::new(0, 4, 5, 10, 0);
        assert_eq!(fcf.pools.len(), 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn add_cut_and_active_cuts_round_trip_at_specific_stage() {
        let mut fcf = FutureCostFunction::new(5, 2, 1, 10, 0);
        let coeffs = [3.0, 7.0];
        fcf.add_cut(2, 0, 0, 42.0, &coeffs);

        let active: Vec<_> = fcf.active_cuts(2).collect();
        assert_eq!(active.len(), 1);
        let (_, intercept, c) = active[0];
        assert_eq!(intercept, 42.0);
        assert_eq!(c, &[3.0, 7.0]);
    }

    #[test]
    fn active_cuts_at_other_stage_returns_empty() {
        let mut fcf = FutureCostFunction::new(5, 2, 1, 10, 0);
        fcf.add_cut(2, 0, 0, 42.0, &[1.0, 2.0]);

        let active: Vec<_> = fcf.active_cuts(3).collect();
        assert!(active.is_empty());
    }

    #[test]
    fn add_cut_multiple_stages_are_independent() {
        let mut fcf = FutureCostFunction::new(4, 1, 1, 10, 0);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(3, 0, 0, 4.0, &[4.0]);

        assert_eq!(fcf.active_cuts(0).count(), 1);
        assert_eq!(fcf.active_cuts(1).count(), 1);
        assert_eq!(fcf.active_cuts(2).count(), 0);
        assert_eq!(fcf.active_cuts(3).count(), 1);
    }

    #[test]
    fn evaluate_at_state_delegates_to_correct_pool() {
        let mut fcf = FutureCostFunction::new(3, 2, 1, 10, 0);
        // stage 1: cut with intercept=10, coeffs=[1,0]
        fcf.add_cut(1, 0, 0, 10.0, &[1.0, 0.0]);
        // stage 2: cut with intercept=5, coeffs=[0,2]
        fcf.add_cut(2, 0, 0, 5.0, &[0.0, 2.0]);

        // stage 1: 10 + 1*3 + 0*4 = 13
        assert_eq!(fcf.evaluate_at_state(1, &[3.0, 4.0]), 13.0);
        // stage 2: 5 + 0*3 + 2*4 = 13
        assert_eq!(fcf.evaluate_at_state(2, &[3.0, 4.0]), 13.0);
        // stage 0: no cuts → NEG_INFINITY
        assert_eq!(fcf.evaluate_at_state(0, &[3.0, 4.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn total_active_cuts_sums_across_stages() {
        let mut fcf = FutureCostFunction::new(4, 1, 1, 20, 0);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(1, 1, 0, 3.0, &[3.0]);
        fcf.add_cut(3, 0, 0, 4.0, &[4.0]);

        // stage 0: 1, stage 1: 2, stage 2: 0, stage 3: 1 → total = 4
        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn total_active_cuts_reflects_deactivation() {
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, 0);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 2.0, &[2.0]); // slot 1
        fcf.add_cut(1, 0, 0, 3.0, &[3.0]); // slot 0

        assert_eq!(fcf.total_active_cuts(), 3);
        fcf.deactivate(0, &[0]);
        assert_eq!(fcf.total_active_cuts(), 2);
    }

    #[test]
    fn deactivate_delegates_to_correct_pool() {
        let mut fcf = FutureCostFunction::new(3, 1, 1, 10, 0);
        fcf.add_cut(1, 0, 0, 10.0, &[1.0]); // slot 0 of pool[1]
        fcf.add_cut(1, 1, 0, 20.0, &[2.0]); // slot 1 of pool[1]
        fcf.add_cut(2, 0, 0, 30.0, &[3.0]); // slot 0 of pool[2]

        fcf.deactivate(1, &[0]);

        // pool[1] should now have 1 active cut
        assert_eq!(fcf.active_cuts(1).count(), 1);
        // pool[2] should still have 1 active cut
        assert_eq!(fcf.active_cuts(2).count(), 1);
    }

    #[test]
    fn ac_new_5_stages_pools_len_is_5() {
        // Given FutureCostFunction::new(5, 9, 10, 100, 0),
        // when inspecting pools.len(), then it is 5.
        let fcf = FutureCostFunction::new(5, 9, 10, 100, 0);
        assert_eq!(fcf.pools.len(), 5);
    }

    #[test]
    fn ac_active_cuts_at_stage_with_cut_yields_it() {
        // Given an FCF with a cut added at stage 2,
        // when active_cuts(2) is called, then it yields the added cut.
        let mut fcf = FutureCostFunction::new(5, 3, 1, 10, 0);
        let coeffs = [1.0, 2.0, 3.0];
        fcf.add_cut(2, 0, 0, 99.0, &coeffs);

        let active: Vec<_> = fcf.active_cuts(2).collect();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn ac_active_cuts_at_different_stage_yields_none() {
        // Given an FCF with a cut added at stage 2,
        // when active_cuts(3) is called, then it yields no cuts.
        let mut fcf = FutureCostFunction::new(5, 3, 1, 10, 0);
        fcf.add_cut(2, 0, 0, 99.0, &[1.0, 2.0, 3.0]);

        let active: Vec<_> = fcf.active_cuts(3).collect();
        assert!(active.is_empty());
    }

    #[test]
    fn ac_total_active_cuts_is_sum_across_stages() {
        // Given an FCF with cuts at multiple stages,
        // when total_active_cuts() is called, then it returns the sum.
        let mut fcf = FutureCostFunction::new(5, 1, 1, 10, 0);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(1, 1, 0, 3.0, &[3.0]);
        fcf.add_cut(4, 0, 0, 4.0, &[4.0]);

        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn fcf_derives_debug_and_clone() {
        let mut fcf = FutureCostFunction::new(2, 2, 1, 5, 0);
        fcf.add_cut(0, 0, 0, 7.0, &[1.0, 2.0]);

        let cloned = fcf.clone();
        assert_eq!(cloned.total_active_cuts(), 1);
        assert_eq!(cloned.evaluate_at_state(0, &[0.0, 0.0]), 7.0);

        let debug_str = format!("{fcf:?}");
        assert!(!debug_str.is_empty());
    }

    // ── from_deserialized tests ──────────────────────────────────────────

    fn make_record(
        intercept: f64,
        coefficients: Vec<f64>,
        is_active: bool,
    ) -> cobre_io::OwnedPolicyCutRecord {
        cobre_io::OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active,
            domination_count: 0,
        }
    }

    fn make_stage(
        stage_id: u32,
        state_dimension: u32,
        cuts: Vec<cobre_io::OwnedPolicyCutRecord>,
    ) -> cobre_io::StageCutsReadResult {
        let populated_count = cuts.len() as u32;
        cobre_io::StageCutsReadResult {
            stage_id,
            state_dimension,
            capacity: populated_count,
            warm_start_count: 0,
            populated_count,
            cuts,
        }
    }

    #[test]
    fn from_deserialized_empty_input_returns_err() {
        let result = FutureCostFunction::from_deserialized(&[]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty"), "{msg}");
    }

    #[test]
    fn from_deserialized_inconsistent_dimensions_returns_err() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 3, vec![make_record(2.0, vec![1.0, 0.0, 0.0], true)]),
        ];
        let result = FutureCostFunction::from_deserialized(&stages);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("inconsistent"), "{msg}");
    }

    #[test]
    fn from_deserialized_preserves_active_flags() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], false), // inactive
                make_record(3.0, vec![1.0, 1.0], true),
            ],
        )];

        let fcf = FutureCostFunction::from_deserialized(&stages).unwrap();
        assert_eq!(fcf.pools.len(), 1);
        assert_eq!(fcf.total_active_cuts(), 2);
        assert_eq!(fcf.pools[0].populated_count, 3);
    }

    #[test]
    fn from_deserialized_evaluate_at_state_matches_original() {
        // Build original FCF with known cuts, then reconstruct via deserialized.
        let mut original = FutureCostFunction::new(2, 2, 1, 10, 0);
        original.add_cut(0, 0, 0, 10.0, &[1.0, 0.0]);
        original.add_cut(0, 1, 0, 5.0, &[0.0, 2.0]);
        original.add_cut(1, 0, 0, 3.0, &[1.0, 1.0]);

        let state = [3.0, 4.0];
        let orig_val_s0 = original.evaluate_at_state(0, &state);
        let orig_val_s1 = original.evaluate_at_state(1, &state);

        // Construct deserialized data matching the original cuts.
        let stages = vec![
            make_stage(
                0,
                2,
                vec![
                    make_record(10.0, vec![1.0, 0.0], true),
                    make_record(5.0, vec![0.0, 2.0], true),
                ],
            ),
            make_stage(1, 2, vec![make_record(3.0, vec![1.0, 1.0], true)]),
        ];

        let reconstructed = FutureCostFunction::from_deserialized(&stages).unwrap();
        assert_eq!(reconstructed.evaluate_at_state(0, &state), orig_val_s0);
        assert_eq!(reconstructed.evaluate_at_state(1, &state), orig_val_s1);
    }

    #[test]
    fn from_deserialized_empty_stage_is_valid() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 2, vec![]), // empty stage
        ];

        let fcf = FutureCostFunction::from_deserialized(&stages).unwrap();
        assert_eq!(fcf.pools.len(), 2);
        assert_eq!(fcf.pools[1].capacity, 0);
        assert_eq!(fcf.pools[1].active_count(), 0);
        assert_eq!(fcf.evaluate_at_state(1, &[1.0, 1.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn from_deserialized_single_cut_stage() {
        let stages = vec![make_stage(
            0,
            3,
            vec![make_record(7.0, vec![1.0, 2.0, 3.0], true)],
        )];

        let fcf = FutureCostFunction::from_deserialized(&stages).unwrap();
        assert_eq!(fcf.state_dimension, 3);
        assert_eq!(fcf.total_active_cuts(), 1);
        // 7 + 1*1 + 2*2 + 3*3 = 7 + 1 + 4 + 9 = 21
        assert_eq!(fcf.evaluate_at_state(0, &[1.0, 2.0, 3.0]), 21.0);
    }

    // ── new_with_warm_start tests ────────────────────────────────────────

    #[test]
    fn warm_start_capacity_includes_training_slots() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], true),
            ],
        )];

        let fcf = FutureCostFunction::new_with_warm_start(&stages, 4, 10).unwrap();
        assert_eq!(fcf.pools.len(), 1);
        // capacity = 2 warm-start + 10*4 training = 42
        assert_eq!(fcf.pools[0].capacity, 42);
        assert_eq!(fcf.pools[0].warm_start_count, 2);
        assert_eq!(fcf.pools[0].forward_passes, 4);
        assert_eq!(fcf.pools[0].populated_count, 2);
        assert_eq!(fcf.total_active_cuts(), 2);
    }

    #[test]
    fn warm_start_training_cuts_at_correct_offset() {
        let stages = vec![make_stage(0, 1, vec![make_record(10.0, vec![1.0], true)])];

        let mut fcf = FutureCostFunction::new_with_warm_start(&stages, 2, 5).unwrap();
        // warm_start_count = 1, forward_passes = 2
        // Training cut at iteration=0, fwd_idx=0: slot = 1 + 0*2 + 0 = 1
        fcf.add_cut(0, 0, 0, 20.0, &[2.0]);
        // Training cut at iteration=0, fwd_idx=1: slot = 1 + 0*2 + 1 = 2
        fcf.add_cut(0, 0, 1, 30.0, &[3.0]);

        assert_eq!(fcf.total_active_cuts(), 3);
        assert_eq!(fcf.pools[0].populated_count, 3);
        // Warm-start cut at slot 0
        assert_eq!(fcf.pools[0].intercepts[0], 10.0);
        // Training cuts at slots 1 and 2
        assert_eq!(fcf.pools[0].intercepts[1], 20.0);
        assert_eq!(fcf.pools[0].intercepts[2], 30.0);
    }

    #[test]
    fn warm_start_empty_stage_has_training_capacity() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 2, vec![]),
        ];

        let fcf = FutureCostFunction::new_with_warm_start(&stages, 3, 5).unwrap();
        // Stage 0: warm_start=1, capacity=1+15=16
        assert_eq!(fcf.pools[0].capacity, 16);
        assert_eq!(fcf.pools[0].warm_start_count, 1);
        // Stage 1: warm_start=0, capacity=0+15=15
        assert_eq!(fcf.pools[1].capacity, 15);
        assert_eq!(fcf.pools[1].warm_start_count, 0);
    }

    #[test]
    fn warm_start_preserves_inactive_flags() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], false), // inactive
            ],
        )];

        let fcf = FutureCostFunction::new_with_warm_start(&stages, 1, 5).unwrap();
        assert_eq!(fcf.pools[0].warm_start_count, 2);
        assert_eq!(fcf.total_active_cuts(), 1); // only the active one
    }
}
