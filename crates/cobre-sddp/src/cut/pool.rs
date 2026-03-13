//! Per-stage cut pool for the Future Cost Function (FCF).
//!
//! The [`CutPool`] is the central data structure for cut storage in the SDDP
//! training loop. Each stage owns one pool. Cuts are stored in pre-allocated
//! slots with a deterministic assignment formula so that results are
//! bit-for-bit identical regardless of execution timing or ordering.
//!
//! ## Slot assignment
//!
//! Each cut inserted during the training loop is assigned a deterministic
//! slot index:
//!
//! ```text
//! slot = warm_start_count + iteration * forward_passes + forward_pass_index
//! ```
//!
//! This guarantees that every run with the same parameters produces the same
//! pool layout, which is required for reproducibility and checkpointing.
//!
//! ## Activity tracking
//!
//! Each slot carries an `active` flag. Inactive cuts are retained in the pool
//! for reproducibility but excluded from LP construction and from
//! [`evaluate_at_state`] queries. The [`CutSelectionStrategy`] determines
//! which cuts to deactivate; [`deactivate`] applies the decision.
//!
//! [`evaluate_at_state`]: CutPool::evaluate_at_state
//! [`deactivate`]: CutPool::deactivate
//! [`CutSelectionStrategy`]: crate::cut_selection::CutSelectionStrategy
//!
//! ## Example
//!
//! ```rust
//! use cobre_sddp::cut::pool::CutPool;
//!
//! // 100-slot pool, 9-dimensional state, 10 forward passes per iteration,
//! // no warm-start cuts.
//! let mut pool = CutPool::new(100, 9, 10, 0);
//! assert_eq!(pool.active_count(), 0);
//!
//! let coeffs = vec![1.0; 9];
//! pool.add_cut(0, 0, 5.0, &coeffs);
//! assert_eq!(pool.active_count(), 1);
//! ```

use crate::cut_selection::CutMetadata;

/// Pre-allocated per-stage cut pool for the Future Cost Function (FCF).
///
/// All storage is allocated at construction time to avoid heap allocation
/// during the training loop hot path. The pool holds `capacity` slots;
/// each slot stores:
///
/// - A coefficient vector of length `state_dimension`
/// - A scalar intercept
/// - [`CutMetadata`] for cut selection bookkeeping
/// - An `active` flag that controls LP participation
///
/// Slots are addressed by a deterministic formula derived from the iteration
/// counter and forward pass index. The pool tracks a `populated_count`
/// high-water mark to avoid scanning unpopulated slots.
#[derive(Debug, Clone)]
pub struct CutPool {
    /// Per-slot coefficient arrays. Each inner `Vec` has length `state_dimension`.
    pub coefficients: Vec<Vec<f64>>,

    /// Per-slot intercept values.
    pub intercepts: Vec<f64>,

    /// Per-slot cut tracking metadata for cut selection strategies.
    pub metadata: Vec<CutMetadata>,

    /// Per-slot activity flags. `false` means the cut is excluded from LP
    /// construction and evaluations. Inactive cuts are still retained in the
    /// pool so that their slots remain deterministic.
    pub active: Vec<bool>,

    /// High-water mark: the number of slots that have been populated at least
    /// once. Iteration over the pool uses this bound to skip trailing
    /// unpopulated slots. Updated by [`add_cut`] when the new slot index
    /// is at or beyond the current mark.
    ///
    /// [`add_cut`]: CutPool::add_cut
    pub populated_count: usize,

    /// Total number of pre-allocated slots. Fixed after construction.
    pub capacity: usize,

    /// Length of each coefficient vector. Fixed after construction.
    pub state_dimension: usize,

    /// Number of forward passes per SDDP iteration. Used by the slot
    /// assignment formula. Fixed after construction.
    pub forward_passes: u32,

    /// Number of warm-start cuts loaded before training begins. Acts as a
    /// base offset in the slot assignment formula. Fixed after construction.
    pub warm_start_count: u32,
}

impl CutPool {
    /// Create a new `CutPool` with all slots pre-allocated and initialized
    /// to zero / inactive.
    ///
    /// # Parameters
    ///
    /// - `capacity`: total number of cut slots to allocate.
    /// - `state_dimension`: length of the state vector (number of coefficients
    ///   per cut).
    /// - `forward_passes`: number of forward passes per training iteration.
    ///   Used by the slot formula: `slot = warm_start_count + iteration *
    ///   forward_passes + forward_pass_index`.
    /// - `warm_start_count`: number of warm-start cuts that occupy the first
    ///   slots. Offsets slot indices for iteration-generated cuts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let pool = CutPool::new(50, 4, 5, 0);
    /// assert_eq!(pool.capacity, 50);
    /// assert_eq!(pool.state_dimension, 4);
    /// assert_eq!(pool.active_count(), 0);
    /// assert_eq!(pool.populated_count, 0);
    /// ```
    #[must_use]
    pub fn new(
        capacity: usize,
        state_dimension: usize,
        forward_passes: u32,
        warm_start_count: u32,
    ) -> Self {
        let default_meta = CutMetadata {
            iteration_generated: 0,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: 0,
            domination_count: 0,
        };

        Self {
            coefficients: vec![vec![0.0; state_dimension]; capacity],
            intercepts: vec![0.0; capacity],
            metadata: vec![default_meta; capacity],
            active: vec![false; capacity],
            populated_count: 0,
            capacity,
            state_dimension,
            forward_passes,
            warm_start_count,
        }
    }

    /// Compute the deterministic slot index for a cut.
    ///
    /// Formula:
    /// ```text
    /// slot = warm_start_count + iteration * forward_passes + forward_pass_index
    /// ```
    #[inline]
    fn slot_index(&self, iteration: u64, forward_pass_index: u32) -> usize {
        // Slot indices are bounded by `capacity` (a usize), so the result
        // always fits in usize. The intermediate cast of `iteration` to usize
        // cannot realistically truncate: any platform capable of running SDDP
        // at scale is 64-bit, and pool capacity is enforced to be < usize::MAX.
        #[allow(clippy::cast_possible_truncation)]
        let iter_usize = iteration as usize;
        self.warm_start_count as usize
            + iter_usize * self.forward_passes as usize
            + forward_pass_index as usize
    }

    /// Insert a Benders cut into the pool at the deterministic slot.
    ///
    /// The slot is computed from `warm_start_count + iteration *
    /// forward_passes + forward_pass_index`. The cut is marked active
    /// immediately. [`CutMetadata`] is initialized with `iteration_generated`
    /// and `forward_pass_index`; activity tracking fields start at zero /
    /// `iteration_generated`.
    ///
    /// `populated_count` is updated if the slot index is at or beyond the
    /// current high-water mark.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the computed slot is >= `capacity` or if
    /// `coefficients.len() != state_dimension`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let mut pool = CutPool::new(20, 3, 5, 0);
    /// pool.add_cut(1, 2, 10.0, &[1.0, 2.0, 3.0]);
    /// // slot = 0 + 1*5 + 2 = 7
    /// assert!(pool.active[7]);
    /// assert_eq!(pool.intercepts[7], 10.0);
    /// ```
    pub fn add_cut(
        &mut self,
        iteration: u64,
        forward_pass_index: u32,
        intercept: f64,
        coefficients: &[f64],
    ) {
        let slot = self.slot_index(iteration, forward_pass_index);

        debug_assert!(
            slot < self.capacity,
            "cut slot {slot} is out of bounds (capacity = {})",
            self.capacity
        );
        debug_assert!(
            coefficients.len() == self.state_dimension,
            "coefficients length {} != state_dimension {}",
            coefficients.len(),
            self.state_dimension
        );

        self.intercepts[slot] = intercept;
        self.coefficients[slot].copy_from_slice(coefficients);
        self.active[slot] = true;
        self.metadata[slot] = CutMetadata {
            iteration_generated: iteration,
            forward_pass_index,
            active_count: 0,
            last_active_iter: iteration,
            domination_count: 0,
        };

        if slot >= self.populated_count {
            self.populated_count = slot + 1;
        }
    }

    /// Iterate over active cuts in the populated range.
    ///
    /// Yields `(slot_index, intercept, coefficient_slice)` for every slot
    /// where `active[slot]` is `true`. Only scans up to `populated_count`
    /// to avoid touching uninitialized slots.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let mut pool = CutPool::new(10, 2, 1, 0);
    /// pool.add_cut(0, 0, 3.0, &[1.0, 2.0]);
    /// pool.add_cut(1, 0, 7.0, &[3.0, 4.0]);
    ///
    /// let active: Vec<_> = pool.active_cuts().collect();
    /// assert_eq!(active.len(), 2);
    /// ```
    pub fn active_cuts(&self) -> impl Iterator<Item = (usize, f64, &[f64])> {
        self.active[..self.populated_count]
            .iter()
            .enumerate()
            .filter(|&(_, &is_active)| is_active)
            .map(|(i, _)| (i, self.intercepts[i], self.coefficients[i].as_slice()))
    }

    /// Count the number of active cuts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// assert_eq!(pool.active_count(), 0);
    /// pool.add_cut(0, 0, 1.0, &[1.0]);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active[..self.populated_count]
            .iter()
            .filter(|&&a| a)
            .count()
    }

    /// Deactivate the cuts at the given slot indices.
    ///
    /// Sets `active[i] = false` for each index in `indices`. Indices are
    /// zero-based slot positions. Out-of-bounds indices are silently ignored
    /// in release builds; a debug assertion fires for out-of-bounds access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(0, 0, 1.0, &[1.0]);
    /// pool.add_cut(1, 0, 2.0, &[2.0]);
    /// pool.deactivate(&[0]);
    /// assert_eq!(pool.active_count(), 1);
    /// assert!(!pool.active[0]);
    /// assert!(pool.active[1]);
    /// ```
    pub fn deactivate(&mut self, indices: &[u32]) {
        for &idx in indices {
            let i = idx as usize;
            debug_assert!(i < self.capacity, "deactivate index {i} out of bounds");
            if i < self.capacity {
                self.active[i] = false;
            }
        }
    }

    /// Evaluate the FCF at the given state vector.
    ///
    /// Returns the maximum over all active cuts of `intercept + coefficients
    /// · state`. Returns [`f64::NEG_INFINITY`] if no active cuts exist (the
    /// pool is empty or all cuts have been deactivated).
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `state.len() != state_dimension`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let mut pool = CutPool::new(10, 2, 1, 0);
    /// pool.add_cut(0, 0, 10.0, &[1.0, 0.0]);
    /// pool.add_cut(1, 0,  5.0, &[0.0, 2.0]);
    ///
    /// // max(10 + 1*3 + 0*4, 5 + 0*3 + 2*4) = max(13, 13) = 13
    /// assert_eq!(pool.evaluate_at_state(&[3.0, 4.0]), 13.0);
    ///
    /// // Empty pool returns NEG_INFINITY.
    /// let empty = CutPool::new(10, 2, 1, 0);
    /// assert_eq!(empty.evaluate_at_state(&[1.0, 1.0]), f64::NEG_INFINITY);
    /// ```
    #[must_use]
    pub fn evaluate_at_state(&self, state: &[f64]) -> f64 {
        debug_assert!(
            state.len() == self.state_dimension,
            "state length {} != state_dimension {}",
            state.len(),
            self.state_dimension
        );

        self.active_cuts()
            .map(|(_, intercept, coeffs)| {
                let dot: f64 = coeffs.iter().zip(state).map(|(a, b)| a * b).sum();
                intercept + dot
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::CutPool;

    #[test]
    fn new_creates_pool_with_correct_capacity_and_all_inactive() {
        let pool = CutPool::new(100, 9, 10, 0);
        assert_eq!(pool.capacity, 100);
        assert_eq!(pool.state_dimension, 9);
        assert_eq!(pool.forward_passes, 10);
        assert_eq!(pool.warm_start_count, 0);
        assert_eq!(pool.populated_count, 0);
        assert_eq!(pool.active_count(), 0);
        assert!(pool.active.iter().all(|&a| !a));
        assert_eq!(pool.coefficients.len(), 100);
        assert!(
            pool.coefficients
                .iter()
                .all(|c| c.iter().all(|&v| v == 0.0))
        );
        assert!(pool.intercepts.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn new_zero_capacity_is_valid() {
        let pool = CutPool::new(0, 4, 5, 0);
        assert_eq!(pool.capacity, 0);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn add_cut_at_slot_zero_stores_intercept_coefficients_and_active_flag() {
        let mut pool = CutPool::new(100, 9, 10, 0);
        let coeffs = vec![1.0; 9];
        pool.add_cut(0, 0, 5.0, &coeffs);

        assert_eq!(pool.active_count(), 1);
        assert!(pool.active[0]);
        assert_eq!(pool.intercepts[0], 5.0);
        assert_eq!(pool.coefficients[0], vec![1.0; 9]);
        assert_eq!(pool.metadata[0].iteration_generated, 0);
        assert_eq!(pool.metadata[0].forward_pass_index, 0);
        assert_eq!(pool.populated_count, 1);
    }

    #[test]
    fn add_cut_deterministic_slot_formula_no_warmstart() {
        // slot = 0 + iteration * forward_passes + forward_pass_index
        let mut pool = CutPool::new(200, 2, 10, 0);

        pool.add_cut(0, 0, 1.0, &[1.0, 2.0]); // slot = 0
        pool.add_cut(0, 3, 2.0, &[3.0, 4.0]); // slot = 3
        pool.add_cut(1, 0, 3.0, &[5.0, 6.0]); // slot = 10
        pool.add_cut(2, 5, 4.0, &[7.0, 8.0]); // slot = 25

        assert!(pool.active[0]);
        assert_eq!(pool.intercepts[0], 1.0);

        assert!(pool.active[3]);
        assert_eq!(pool.intercepts[3], 2.0);

        assert!(pool.active[10]);
        assert_eq!(pool.intercepts[10], 3.0);

        assert!(pool.active[25]);
        assert_eq!(pool.intercepts[25], 4.0);
    }

    #[test]
    fn add_cut_warm_start_count_offsets_slot() {
        // slot = 5 + 0*10 + 0 = 5
        let mut pool = CutPool::new(100, 9, 10, 5);
        let coeffs = vec![0.0; 9];
        pool.add_cut(0, 0, 42.0, &coeffs);

        assert!(pool.active[5]);
        assert_eq!(pool.intercepts[5], 42.0);
        assert_eq!(pool.populated_count, 6);
    }

    #[test]
    fn add_cut_metadata_initialized_correctly() {
        let mut pool = CutPool::new(50, 3, 5, 0);
        pool.add_cut(3, 2, 7.0, &[1.0, 2.0, 3.0]);
        // slot = 0 + 3*5 + 2 = 17
        let meta = &pool.metadata[17];
        assert_eq!(meta.iteration_generated, 3);
        assert_eq!(meta.forward_pass_index, 2);
        assert_eq!(meta.active_count, 0);
        assert_eq!(meta.last_active_iter, 3);
        assert_eq!(meta.domination_count, 0);
    }

    #[test]
    fn populated_count_tracks_high_water_mark() {
        let mut pool = CutPool::new(50, 1, 5, 0);

        pool.add_cut(0, 0, 1.0, &[1.0]); // slot 0 → populated_count = 1
        assert_eq!(pool.populated_count, 1);

        pool.add_cut(1, 0, 2.0, &[2.0]); // slot 5 → populated_count = 6
        assert_eq!(pool.populated_count, 6);

        pool.add_cut(0, 2, 3.0, &[3.0]); // slot 2 → no change (2 < 6)
        assert_eq!(pool.populated_count, 6);
    }

    #[test]
    fn active_cuts_returns_only_active_cuts() {
        let mut pool = CutPool::new(20, 2, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0, 2.0]); // slot 0
        pool.add_cut(1, 0, 2.0, &[3.0, 4.0]); // slot 1
        pool.add_cut(2, 0, 3.0, &[5.0, 6.0]); // slot 2

        pool.deactivate(&[1]);

        let active: Vec<_> = pool.active_cuts().collect();
        assert_eq!(active.len(), 2);

        let slots: Vec<usize> = active.iter().map(|(s, _, _)| *s).collect();
        assert!(slots.contains(&0));
        assert!(slots.contains(&2));
        assert!(!slots.contains(&1));
    }

    #[test]
    fn active_cuts_empty_pool_returns_empty_iterator() {
        let pool = CutPool::new(10, 3, 5, 0);
        let active: Vec<_> = pool.active_cuts().collect();
        assert!(active.is_empty());
    }

    #[test]
    fn active_count_is_correct_after_add_and_deactivate() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(2, 0, 3.0, &[3.0]); // slot 2

        assert_eq!(pool.active_count(), 3);
        pool.deactivate(&[1]);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn deactivate_sets_flags_correctly() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(2, 0, 3.0, &[3.0]); // slot 2

        pool.deactivate(&[1]);

        assert!(pool.active[0]);
        assert!(!pool.active[1]);
        assert!(pool.active[2]);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn deactivate_multiple_indices() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(2, 0, 3.0, &[3.0]); // slot 2

        pool.deactivate(&[0, 2]);

        assert!(!pool.active[0]);
        assert!(pool.active[1]);
        assert!(!pool.active[2]);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn deactivate_empty_slice_is_noop() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0]);
        pool.deactivate(&[]);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn evaluate_at_state_returns_max_cut_value() {
        // cuts: (intercept=10, coeffs=[1, 0]) and (intercept=5, coeffs=[0, 2])
        // state = [3, 4]
        // cut 0: 10 + 1*3 + 0*4 = 13
        // cut 1:  5 + 0*3 + 2*4 = 13
        // max = 13
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(0, 0, 10.0, &[1.0, 0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0, 2.0]);

        let result = pool.evaluate_at_state(&[3.0, 4.0]);
        assert_eq!(result, 13.0);
    }

    #[test]
    fn evaluate_at_state_selects_correct_max() {
        // cut 0: intercept=2, coeffs=[1] → at state [10]: 2 + 10 = 12
        // cut 1: intercept=5, coeffs=[2] → at state [10]: 5 + 20 = 25
        // max = 25
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(0, 0, 2.0, &[1.0]);
        pool.add_cut(1, 0, 5.0, &[2.0]);

        let result = pool.evaluate_at_state(&[10.0]);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn evaluate_at_state_empty_pool_returns_neg_infinity() {
        let pool = CutPool::new(10, 3, 5, 0);
        assert_eq!(pool.evaluate_at_state(&[1.0, 2.0, 3.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn evaluate_at_state_all_deactivated_returns_neg_infinity() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(0, 0, 100.0, &[1.0]);
        pool.deactivate(&[0]);
        assert_eq!(pool.evaluate_at_state(&[5.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn evaluate_at_state_ignores_deactivated_cuts() {
        // slot 0: active, intercept=10, coeff=[1]  → at state [3]: 13
        // slot 1: INACTIVE, intercept=100, coeff=[1] → would be 103, but ignored
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(0, 0, 10.0, &[1.0]);
        pool.add_cut(1, 0, 100.0, &[1.0]);
        pool.deactivate(&[1]);

        assert_eq!(pool.evaluate_at_state(&[3.0]), 13.0);
    }

    #[test]
    fn ac_add_cut_stores_at_slot_zero_and_active_count_is_one() {
        // Given CutPool::new(100, 9, 10, 0), when add_cut(0, 0, ...) is called,
        // then the cut is stored at slot 0 and active_count() returns 1.
        let mut pool = CutPool::new(100, 9, 10, 0);
        let coeffs = vec![0.0; 9];
        pool.add_cut(0, 0, 5.0, &coeffs);

        assert!(pool.active[0]);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn ac_deactivate_reduces_active_count_correctly() {
        // Given a pool with 3 cuts at slots 0, 1, 2, when deactivate(&[1]) is
        // called, then active_count() returns 2 and slot 1 is inactive.
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0]);
        pool.add_cut(1, 0, 2.0, &[2.0]);
        pool.add_cut(2, 0, 3.0, &[3.0]);

        pool.deactivate(&[1]);

        assert_eq!(pool.active_count(), 2);
        assert!(!pool.active[1]);
    }

    #[test]
    fn ac_evaluate_at_state_returns_correct_max() {
        // cuts: (intercept=10, coeffs=[1,0]) and (intercept=5, coeffs=[0,2])
        // state=[3,4] → max(10+3, 5+8) = max(13, 13) = 13
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(0, 0, 10.0, &[1.0, 0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0, 2.0]);

        assert_eq!(pool.evaluate_at_state(&[3.0, 4.0]), 13.0);
    }

    #[test]
    fn ac_warm_start_count_offsets_slot() {
        // Given CutPool::new(100, 9, 10, 5), when add_cut(0, 0, ...) is called,
        // then slot = 5 + 0*10 + 0 = 5.
        let mut pool = CutPool::new(100, 9, 10, 5);
        let coeffs = vec![0.0; 9];
        pool.add_cut(0, 0, 1.0, &coeffs);

        assert!(pool.active[5]);
        assert!(!pool.active[0]);
    }

    #[test]
    fn ac_empty_pool_evaluate_returns_neg_infinity() {
        // Given an empty pool, evaluate_at_state returns NEG_INFINITY.
        let pool = CutPool::new(10, 2, 1, 0);
        assert_eq!(pool.evaluate_at_state(&[1.0, 2.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn cut_pool_derives_debug_and_clone() {
        let mut pool = CutPool::new(5, 2, 1, 0);
        pool.add_cut(0, 0, 3.0, &[1.0, 2.0]);

        let cloned = pool.clone();
        assert_eq!(cloned.active_count(), 1);
        assert_eq!(cloned.intercepts[0], 3.0);

        let debug_str = format!("{pool:?}");
        assert!(!debug_str.is_empty());
    }
}
