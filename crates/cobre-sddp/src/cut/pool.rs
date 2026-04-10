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
///
/// A `cached_active_count` field is maintained incrementally by [`add_cut`]
/// and [`deactivate`], making [`active_count`] an O(1) query.
///
/// [`add_cut`]: CutPool::add_cut
/// [`deactivate`]: CutPool::deactivate
/// [`active_count`]: CutPool::active_count
#[derive(Debug, Clone)]
pub struct CutPool {
    /// Flat coefficient storage. Coefficients for slot `i` occupy the range
    /// `i * state_dimension .. (i + 1) * state_dimension`. Length is always
    /// `capacity * state_dimension`.
    pub coefficients: Vec<f64>,

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

    /// Cached count of active cuts, maintained incrementally by [`add_cut`]
    /// (increment) and [`deactivate`] (decrement). Makes [`active_count`]
    /// O(1) instead of O(`populated_count`).
    ///
    /// [`add_cut`]: CutPool::add_cut
    /// [`deactivate`]: CutPool::deactivate
    /// [`active_count`]: CutPool::active_count
    pub cached_active_count: usize,
}

impl CutPool {
    /// Create a new `CutPool` with lazily-grown storage.
    ///
    /// No coefficient, intercept, or metadata arrays are allocated at
    /// construction. Storage grows on demand via [`ensure_capacity`] when
    /// [`add_cut`] is called. The `capacity` parameter is stored as a
    /// theoretical maximum for debug assertions only.
    ///
    /// [`ensure_capacity`]: CutPool::ensure_capacity
    /// [`add_cut`]: CutPool::add_cut
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let pool = CutPool::new(50, 4, 5, 0);
    /// assert_eq!(pool.capacity, 50);
    /// assert_eq!(pool.coefficients.len(), 0);
    /// assert_eq!(pool.active_count(), 0);
    /// ```
    #[must_use]
    pub fn new(
        capacity: usize,
        state_dimension: usize,
        forward_passes: u32,
        warm_start_count: u32,
    ) -> Self {
        Self {
            coefficients: Vec::new(),
            intercepts: Vec::new(),
            metadata: Vec::new(),
            active: Vec::new(),
            populated_count: 0,
            capacity,
            state_dimension,
            forward_passes,
            warm_start_count,
            cached_active_count: 0,
        }
    }

    /// Grow all parallel arrays to accommodate at least `slot + 1` entries.
    ///
    /// Uses a doubling strategy with a minimum of 16 slots. No-op if current
    /// allocation is already sufficient.
    fn ensure_capacity(&mut self, slot: usize) {
        let needed = slot + 1;
        if needed <= self.intercepts.len() {
            return;
        }
        let new_len = needed.max(self.intercepts.len() * 2).max(16);
        let default_meta = CutMetadata {
            iteration_generated: 0,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: 0,
            domination_count: 0,
        };
        self.coefficients
            .resize(new_len * self.state_dimension, 0.0);
        self.intercepts.resize(new_len, 0.0);
        self.metadata.resize(new_len, default_meta);
        self.active.resize(new_len, false);
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
            "cut slot {slot} exceeds theoretical capacity {}",
            self.capacity
        );
        debug_assert!(
            coefficients.len() == self.state_dimension,
            "coefficients length {} != state_dimension {}",
            coefficients.len(),
            self.state_dimension
        );
        self.ensure_capacity(slot);

        self.intercepts[slot] = intercept;
        let start = slot * self.state_dimension;
        self.coefficients[start..start + self.state_dimension].copy_from_slice(coefficients);
        debug_assert!(
            !self.active[slot],
            "add_cut: slot {slot} is already active (double-insert)"
        );
        self.active[slot] = true;
        self.cached_active_count += 1;
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
            .map(|(i, _)| {
                let start = i * self.state_dimension;
                (
                    i,
                    self.intercepts[i],
                    &self.coefficients[start..start + self.state_dimension],
                )
            })
    }

    /// Count the number of active cuts.
    ///
    /// Returns the cached count in O(1). A debug assertion verifies consistency
    /// with the computed count in debug builds.
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
        debug_assert_eq!(
            self.cached_active_count,
            self.active[..self.populated_count]
                .iter()
                .filter(|&&a| a)
                .count(),
            "cached active_count {} != computed {}",
            self.cached_active_count,
            self.active[..self.populated_count]
                .iter()
                .filter(|&&a| a)
                .count(),
        );
        self.cached_active_count
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
            let allocated = self.intercepts.len();
            debug_assert!(i < allocated, "deactivate index {i} out of bounds");
            if i < allocated && self.active[i] {
                self.active[i] = false;
                self.cached_active_count -= 1;
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

    /// Compute a report of exact-zero sparsity across all active cuts.
    ///
    /// Scans every coefficient of every active cut and counts exact zeros
    /// (`value == 0.0`). Returns a [`SparsityReport`] with aggregate and
    /// per-dimension statistics. Only exact zeros are counted to preserve
    /// bit-for-bit reproducibility when sparse representations are used.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let mut pool = CutPool::new(10, 3, 1, 0);
    /// pool.add_cut(0, 0, 1.0, &[1.0, 0.0, 2.0]);
    /// pool.add_cut(1, 0, 2.0, &[0.0, 0.0, 3.0]);
    ///
    /// let report = pool.sparsity_report();
    /// assert_eq!(report.total_coefficients, 6);   // 2 cuts * 3 dims
    /// assert_eq!(report.exact_zero_count, 3);      // (0,1), (1,0), (1,1)
    /// assert!((report.sparsity_fraction - 0.5).abs() < 1e-10);
    /// assert_eq!(report.per_dimension_zeros, vec![1, 2, 0]);
    /// ```
    #[must_use]
    pub fn sparsity_report(&self) -> SparsityReport {
        let active_count = self.active_count();
        let mut exact_zero_count = 0usize;
        let mut per_dimension_zeros = vec![0usize; self.state_dimension];

        for (_slot, _intercept, coeffs) in self.active_cuts() {
            for (j, &c) in coeffs.iter().enumerate() {
                if c == 0.0 {
                    exact_zero_count += 1;
                    per_dimension_zeros[j] += 1;
                }
            }
        }

        let total = active_count * self.state_dimension;
        #[allow(clippy::cast_precision_loss)]
        let fraction = if total > 0 {
            exact_zero_count as f64 / total as f64
        } else {
            0.0
        };

        SparsityReport {
            total_coefficients: total,
            exact_zero_count,
            sparsity_fraction: fraction,
            per_dimension_zeros,
        }
    }

    /// Construct a `CutPool` from deserialized cut records.
    ///
    /// The pool capacity is set to `records.len()` (no room for new training
    /// cuts). All loaded cuts are populated and their active flags are set
    /// from the deserialized records.
    ///
    /// `forward_passes` is set to 0 and `warm_start_count` is set to
    /// `records.len()` since this pool is not intended for incremental
    /// training addition (only for FCF evaluation during simulation).
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_io::OwnedPolicyCutRecord;
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let records = vec![
    ///     OwnedPolicyCutRecord {
    ///         cut_id: 0,
    ///         slot_index: 0,
    ///         iteration: 0,
    ///         forward_pass_index: 0,
    ///         intercept: 5.0,
    ///         coefficients: vec![1.0, 2.0],
    ///         is_active: true,
    ///         domination_count: 0,
    ///     },
    /// ];
    ///
    /// let pool = CutPool::from_deserialized(2, &records);
    /// assert_eq!(pool.capacity, 1);
    /// assert_eq!(pool.populated_count, 1);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    pub fn from_deserialized(
        state_dimension: usize,
        records: &[cobre_io::OwnedPolicyCutRecord],
    ) -> Self {
        let capacity = records.len();
        let mut coefficients = Vec::with_capacity(capacity * state_dimension);
        let mut intercepts = Vec::with_capacity(capacity);
        let mut active = Vec::with_capacity(capacity);
        let mut metadata = Vec::with_capacity(capacity);
        let mut cached_active_count = 0usize;

        for record in records {
            debug_assert!(
                record.coefficients.len() == state_dimension,
                "from_deserialized: coefficients length {} != state_dimension {}",
                record.coefficients.len(),
                state_dimension
            );
            coefficients.extend_from_slice(&record.coefficients);
            intercepts.push(record.intercept);
            active.push(record.is_active);
            if record.is_active {
                cached_active_count += 1;
            }
            metadata.push(CutMetadata {
                iteration_generated: u64::from(record.iteration),
                forward_pass_index: record.forward_pass_index,
                active_count: 0,
                last_active_iter: u64::from(record.iteration),
                domination_count: u64::from(record.domination_count),
            });
        }

        #[allow(clippy::cast_possible_truncation)]
        Self {
            coefficients,
            intercepts,
            metadata,
            active,
            populated_count: capacity,
            capacity,
            state_dimension,
            forward_passes: 0,
            warm_start_count: capacity as u32,
            cached_active_count,
        }
    }

    /// Construct a `CutPool` with warm-start cuts plus capacity for training.
    ///
    /// The loaded cuts occupy the first `records.len()` slots. The remaining
    /// `max_iterations * forward_passes` slots are allocated for new training
    /// cuts. The slot formula `warm_start_count + iteration * forward_passes +
    /// forward_pass_index` correctly offsets training cuts past the warm-start
    /// region.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_io::OwnedPolicyCutRecord;
    /// use cobre_sddp::cut::pool::CutPool;
    ///
    /// let records = vec![
    ///     OwnedPolicyCutRecord {
    ///         cut_id: 0, slot_index: 0, iteration: 0, forward_pass_index: 0,
    ///         intercept: 5.0, coefficients: vec![1.0, 2.0],
    ///         is_active: true, domination_count: 0,
    ///     },
    /// ];
    /// let pool = CutPool::new_with_warm_start(2, 4, 10, &records);
    /// assert_eq!(pool.warm_start_count, 1);
    /// assert_eq!(pool.capacity, 1 + 10 * 4); // 41
    /// assert_eq!(pool.populated_count, 1);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    pub fn new_with_warm_start(
        state_dimension: usize,
        forward_passes: u32,
        max_iterations: u64,
        records: &[cobre_io::OwnedPolicyCutRecord],
    ) -> Self {
        let warm_start_count = records.len();
        #[allow(clippy::cast_possible_truncation)]
        let capacity = warm_start_count + (max_iterations as usize) * (forward_passes as usize);

        let mut coefficients = Vec::with_capacity(warm_start_count * state_dimension);
        let mut intercepts = Vec::with_capacity(warm_start_count);
        let mut active = Vec::with_capacity(warm_start_count);
        let mut metadata = Vec::with_capacity(warm_start_count);
        let mut cached_active_count = 0usize;

        for record in records {
            debug_assert!(
                record.coefficients.len() == state_dimension,
                "new_with_warm_start: coefficients length {} != state_dimension {}",
                record.coefficients.len(),
                state_dimension
            );
            coefficients.extend_from_slice(&record.coefficients);
            intercepts.push(record.intercept);
            active.push(record.is_active);
            if record.is_active {
                cached_active_count += 1;
            }
            // Use u64::MAX as iteration_generated sentinel so warm-start cuts
            // are never matched by pack_local_cuts (which filters on the
            // current training iteration). This prevents double-counting
            // warm-start cuts as new training cuts in cut sync.
            metadata.push(CutMetadata {
                iteration_generated: u64::MAX,
                forward_pass_index: record.forward_pass_index,
                active_count: 0,
                last_active_iter: u64::from(record.iteration),
                domination_count: u64::from(record.domination_count),
            });
        }

        #[allow(clippy::cast_possible_truncation)]
        Self {
            coefficients,
            intercepts,
            metadata,
            active,
            populated_count: warm_start_count,
            capacity,
            state_dimension,
            forward_passes,
            warm_start_count: warm_start_count as u32,
            cached_active_count,
        }
    }
}

/// Report of exact-zero sparsity across active cuts in a [`CutPool`].
///
/// Produced by [`CutPool::sparsity_report`]. Only exact zeros (`value == 0.0`)
/// are counted -- near-zero values are not included to preserve bit-for-bit
/// reproducibility.
#[derive(Debug, Clone)]
pub struct SparsityReport {
    /// Total number of coefficients scanned (`active_count * state_dimension`).
    pub total_coefficients: usize,
    /// Number of exact-zero coefficients (`value == 0.0`).
    pub exact_zero_count: usize,
    /// Fraction of exact-zero coefficients (0.0 to 1.0).
    pub sparsity_fraction: f64,
    /// Per-dimension zero count (length = `state_dimension`). Entry `j` is the
    /// number of active cuts where `coefficient[j] == 0.0`.
    pub per_dimension_zeros: Vec<usize>,
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
        assert_eq!(pool.active.len(), 0);
        assert_eq!(pool.coefficients.len(), 0);
        assert_eq!(pool.intercepts.len(), 0);
    }

    #[test]
    fn new_zero_capacity_is_valid() {
        let pool = CutPool::new(0, 4, 5, 0);
        assert_eq!(pool.capacity, 0);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn lazy_growth_starts_empty() {
        let pool = CutPool::new(100, 3, 1, 0);
        assert_eq!(pool.coefficients.len(), 0);
        assert_eq!(pool.intercepts.len(), 0);
        assert_eq!(pool.active.len(), 0);
    }

    #[test]
    fn lazy_growth_allocates_on_add() {
        let mut pool = CutPool::new(100, 3, 1, 0);
        pool.add_cut(0, 0, 7.0, &[1.0, 2.0, 3.0]);
        assert!(pool.coefficients.len() >= 3);
        let cuts: Vec<_> = pool.active_cuts().collect();
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].1, 7.0);
        assert_eq!(cuts[0].2, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_cut_at_slot_zero_stores_intercept_coefficients_and_active_flag() {
        let mut pool = CutPool::new(100, 9, 10, 0);
        let coeffs = vec![1.0; 9];
        pool.add_cut(0, 0, 5.0, &coeffs);

        assert_eq!(pool.active_count(), 1);
        assert!(pool.active[0]);
        assert_eq!(pool.intercepts[0], 5.0);
        assert_eq!(&pool.coefficients[0..9], vec![1.0; 9].as_slice());
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

    // ── SparsityReport tests ──────────────────────────────────────────

    #[test]
    fn sparsity_report_empty_pool() {
        let pool = CutPool::new(10, 3, 1, 0);
        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 0);
        assert_eq!(report.exact_zero_count, 0);
        assert!((report.sparsity_fraction - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![0, 0, 0]);
    }

    #[test]
    fn sparsity_report_all_nonzero() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0, 2.0, 3.0]);
        pool.add_cut(1, 0, 2.0, &[4.0, 5.0, 6.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 0);
        assert!((report.sparsity_fraction - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![0, 0, 0]);
    }

    #[test]
    fn sparsity_report_all_zero() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0, 0.0, 0.0]);
        pool.add_cut(1, 0, 2.0, &[0.0, 0.0, 0.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 6);
        assert!((report.sparsity_fraction - 1.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![2, 2, 2]);
    }

    #[test]
    fn sparsity_report_mixed() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0, 0.0, 2.0]);
        pool.add_cut(1, 0, 2.0, &[0.0, 0.0, 3.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 3);
        assert!((report.sparsity_fraction - 0.5).abs() < 1e-10);
        assert_eq!(report.per_dimension_zeros, vec![1, 2, 0]);
    }

    #[test]
    fn sparsity_report_excludes_inactive_cuts() {
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0, 0.0]); // all zero, then deactivate
        pool.add_cut(1, 0, 2.0, &[1.0, 2.0]); // all non-zero
        pool.deactivate(&[0]);

        let report = pool.sparsity_report();
        // Only the second cut is active.
        assert_eq!(report.total_coefficients, 2);
        assert_eq!(report.exact_zero_count, 0);
        assert!((report.sparsity_fraction - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sparsity_report_per_dimension_zeros_correct() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        // Cut 0: dims 0,2 are zero
        pool.add_cut(0, 0, 1.0, &[0.0, 1.0, 0.0, 3.0]);
        // Cut 1: dims 0,3 are zero
        pool.add_cut(1, 0, 2.0, &[0.0, 2.0, 4.0, 0.0]);
        // Cut 2: no zeros
        pool.add_cut(2, 0, 3.0, &[5.0, 6.0, 7.0, 8.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 12);
        assert_eq!(report.exact_zero_count, 4);
        assert_eq!(report.per_dimension_zeros, vec![2, 0, 1, 1]);
    }
}
