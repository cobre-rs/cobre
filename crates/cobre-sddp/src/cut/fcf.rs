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
}
