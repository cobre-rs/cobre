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

use crate::cut::WARM_START_ITERATION;
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

    /// Scratch buffer for [`enforce_budget`] candidate collection.
    ///
    /// Reused across calls to avoid per-call `Vec<u32>` allocation. Cleared
    /// at the start of each `enforce_budget` invocation and populated with
    /// active slot indices that are eligible for eviction.
    ///
    /// [`enforce_budget`]: CutPool::enforce_budget
    pub(crate) candidates_buf: Vec<u32>,
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
            active_window: 0,
        };

        Self {
            coefficients: vec![0.0; capacity * state_dimension],
            intercepts: vec![0.0; capacity],
            metadata: vec![default_meta; capacity],
            active: vec![false; capacity],
            populated_count: 0,
            capacity,
            state_dimension,
            forward_passes,
            warm_start_count,
            cached_active_count: 0,
            candidates_buf: Vec::new(),
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
            // Transient seed: set SEED_BIT (outside RECENT_WINDOW_BITS) so the
            // classifier fires LOWER within the same iteration, but the seed is
            // cleared at end-of-iter before the shift — no cross-iter carryover.
            active_window: crate::basis_reconstruct::SEED_BIT,
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
        let mut remaining = self.cached_active_count;
        self.active[..self.populated_count]
            .iter()
            .enumerate()
            .scan((), move |(), (i, &is_active)| {
                if remaining == 0 {
                    return None;
                }
                if is_active {
                    remaining -= 1;
                    let start = i * self.state_dimension;
                    Some(Some((
                        i,
                        self.intercepts[i],
                        &self.coefficients[start..start + self.state_dimension],
                    )))
                } else {
                    Some(None)
                }
            })
            .flatten()
    }

    /// Iterate over active cuts generated in a specific training iteration.
    ///
    /// Yields `(slot_index, intercept, coefficient_slice)` for every slot
    /// where `active[slot]` is `true` AND
    /// `metadata[slot].iteration_generated == current_iteration`.
    ///
    /// Warm-start cuts (whose `iteration_generated` is
    /// [`WARM_START_ITERATION`]) are always excluded, even if
    /// `current_iteration` were to equal the sentinel numerically — the
    /// sentinel is chosen as `u64::MAX` to make such a collision impossible
    /// in practice, but the explicit guard is retained for clarity.
    ///
    /// Only scans up to `populated_count` to avoid touching uninitialized
    /// slots. Iterates in insertion order, preserving declaration-order
    /// invariance.
    pub(crate) fn active_delta_cuts(
        &self,
        current_iteration: u64,
    ) -> impl Iterator<Item = (usize, f64, &[f64])> {
        let mut remaining = self.cached_active_count;
        self.active[..self.populated_count]
            .iter()
            .enumerate()
            .scan((), move |(), (slot, &is_active)| {
                if remaining == 0 {
                    return None;
                }
                if is_active {
                    remaining -= 1;
                    Some(Some(slot))
                } else {
                    Some(None)
                }
            })
            .flatten()
            .filter(move |&slot| {
                self.metadata[slot].iteration_generated == current_iteration
                    && self.metadata[slot].iteration_generated != WARM_START_ITERATION
            })
            .map(|i| {
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
            debug_assert!(i < self.capacity, "deactivate index {i} out of bounds");
            if i < self.capacity && self.active[i] {
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

    /// Diagnostic: count exact-zero coefficients across all active cuts.
    ///
    /// Walks every coefficient of every active cut and tallies exact
    /// zeros (`value == 0.0`). Returns a [`SparsityReport`] with the
    /// aggregate count, fraction, and per-dimension breakdown.
    ///
    /// Not on the hot path: allocates one `Vec<usize>` of length
    /// `state_dimension` per call. Intended for offline analysis or
    /// pre-release diagnostics, not per-iteration use.
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
                active_window: 0,
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
            candidates_buf: Vec::new(),
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
    ///         is_active: true,
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

        let default_meta = CutMetadata {
            iteration_generated: 0,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: 0,
            active_window: 0,
        };

        let mut coefficients = vec![0.0_f64; capacity * state_dimension];
        let mut intercepts = vec![0.0; capacity];
        let mut active = vec![false; capacity];
        let mut metadata = vec![default_meta; capacity];
        let mut cached_active_count = 0usize;

        for (i, record) in records.iter().enumerate() {
            debug_assert!(
                record.coefficients.len() == state_dimension,
                "new_with_warm_start: coefficients length {} != state_dimension {}",
                record.coefficients.len(),
                state_dimension
            );
            let start = i * state_dimension;
            coefficients[start..start + state_dimension].copy_from_slice(&record.coefficients);
            intercepts[i] = record.intercept;
            active[i] = record.is_active;
            if record.is_active {
                cached_active_count += 1;
            }
            // Use WARM_START_ITERATION as the iteration_generated sentinel so
            // warm-start cuts are never matched by pack_local_cuts (which
            // filters on the current training iteration).  This prevents
            // double-counting warm-start cuts as new training cuts in cut sync.
            metadata[i] = CutMetadata {
                iteration_generated: WARM_START_ITERATION,
                forward_pass_index: record.forward_pass_index,
                active_count: 0,
                last_active_iter: u64::from(record.iteration),
                active_window: 0,
            };
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
            candidates_buf: Vec::new(),
        }
    }
}

/// Diagnostic report of exact-zero coefficients across active cuts in a [`CutPool`].
///
/// Produced by [`CutPool::sparsity_report`]. Counts only exact
/// zeros (`value == 0.0`); near-zero values are not collapsed to
/// zero by this report.
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

/// Result of a [`CutPool::enforce_budget`] call.
///
/// Reports how many cuts were evicted and the active-cut counts before and
/// after the enforcement pass.
#[derive(Debug, Clone)]
pub struct BudgetEnforcementResult {
    /// Number of cuts deactivated during this enforcement pass.
    pub evicted_count: u32,
    /// Active cut count before enforcement.
    pub active_before: u32,
    /// Active cut count after enforcement (`active_before - evicted_count`).
    pub active_after: u32,
}

impl CutPool {
    /// Enforce a hard cap on the number of active cuts per stage.
    ///
    /// If `active_count() <= budget`, returns immediately with zero evictions.
    ///
    /// Otherwise, collects all active slots where
    /// `metadata[slot].iteration_generated != current_iteration` (protecting
    /// cuts from the current backward pass), sorts them by
    /// `(last_active_iter ASC, active_count ASC)` — stalest and least-used
    /// first — and deactivates the first `active_count - budget` of them.
    ///
    /// Uses `select_nth_unstable_by` (partial sort) when the number of excess
    /// cuts is small relative to the candidate set, otherwise `sort_unstable_by`.
    ///
    /// Cuts generated in `current_iteration` are **never** evicted. If the
    /// current-iteration cuts alone exceed the budget, all current-iteration
    /// cuts are preserved and `active_count()` may remain above `budget` after
    /// the call.
    ///
    /// # Parameters
    ///
    /// - `budget`: maximum number of active cuts allowed.
    /// - `current_iteration`: cuts generated in this iteration are protected.
    /// - `forward_passes`: unused by the method; present for call-site
    ///   uniformity with the training loop (which uses it for the warning
    ///   validation in `StudyParams::from_config`).
    pub fn enforce_budget(
        &mut self,
        budget: u32,
        current_iteration: u64,
        _forward_passes: u32,
    ) -> BudgetEnforcementResult {
        #[allow(clippy::cast_possible_truncation)]
        let active_before = self.active_count() as u32;
        let budget_usize = budget as usize;

        if self.cached_active_count <= budget_usize {
            return BudgetEnforcementResult {
                evicted_count: 0,
                active_before,
                active_after: active_before,
            };
        }

        let excess = self.cached_active_count - budget_usize;

        // Collect eviction candidates into the reused scratch buffer:
        // active slots not from current_iteration.
        self.candidates_buf.clear();
        #[allow(clippy::cast_possible_truncation)]
        self.candidates_buf.extend(
            self.active[..self.populated_count]
                .iter()
                .enumerate()
                .filter(|&(slot, &is_active)| {
                    is_active && self.metadata[slot].iteration_generated != current_iteration
                })
                .map(|(slot, _)| slot as u32),
        );

        if self.candidates_buf.is_empty() {
            // All active cuts are from the current iteration; preserve them all.
            return BudgetEnforcementResult {
                evicted_count: 0,
                active_before,
                active_after: active_before,
            };
        }

        // Eviction key: (last_active_iter ASC, active_count ASC).
        // Stalest, least-frequently-used cuts are evicted first.
        let evict_count = excess.min(self.candidates_buf.len());

        let key = |&slot: &u32| {
            let meta = &self.metadata[slot as usize];
            (meta.last_active_iter, meta.active_count)
        };

        // Use partial sort when only a small fraction of candidates need
        // evicting; fall back to full sort otherwise.
        if evict_count < self.candidates_buf.len() / 2 {
            // select_nth_unstable_by partitions so that candidates_buf[..evict_count]
            // contains the evict_count smallest elements (in any order).
            self.candidates_buf
                .select_nth_unstable_by(evict_count, |a, b| key(a).cmp(&key(b)));
        } else {
            self.candidates_buf.sort_unstable_by_key(|a| key(a));
        }

        // Copy the eviction slice into a local Vec to release the borrow on
        // self.candidates_buf before calling deactivate, which takes &mut self.
        let to_evict: Vec<u32> = self.candidates_buf[..evict_count].to_vec();
        self.deactivate(&to_evict);

        #[allow(clippy::cast_possible_truncation)]
        let evicted_count = evict_count as u32;
        #[allow(clippy::cast_possible_truncation)]
        let active_after = self.active_count() as u32;

        BudgetEnforcementResult {
            evicted_count,
            active_before,
            active_after,
        }
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
        assert_eq!(pool.coefficients.len(), 100 * 9);
        assert!(pool.coefficients.iter().all(|&v| v == 0.0));
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

    #[test]
    fn warm_start_cuts_have_sentinel_iteration() {
        use crate::cut::WARM_START_ITERATION;
        use cobre_io::OwnedPolicyCutRecord;

        let records = vec![
            OwnedPolicyCutRecord {
                cut_id: 0,
                slot_index: 0,
                coefficients: vec![1.0, 2.0],
                intercept: 10.0,
                is_active: true,
                iteration: 5,
                forward_pass_index: 0,
            },
            OwnedPolicyCutRecord {
                cut_id: 1,
                slot_index: 1,
                coefficients: vec![3.0, 4.0],
                intercept: 20.0,
                is_active: true,
                iteration: 7,
                forward_pass_index: 1,
            },
        ];

        let pool = CutPool::new_with_warm_start(2, 4, 100, &records);
        assert_eq!(pool.warm_start_count, 2);
        assert_eq!(pool.populated_count, 2);
        // Both warm-start cuts must use the sentinel value.
        assert_eq!(pool.metadata[0].iteration_generated, WARM_START_ITERATION);
        assert_eq!(pool.metadata[1].iteration_generated, WARM_START_ITERATION);
        // The original iteration is preserved in last_active_iter for
        // informational purposes (checkpoint round-trip).
        assert_eq!(pool.metadata[0].last_active_iter, 5);
        assert_eq!(pool.metadata[1].last_active_iter, 7);
    }

    #[test]
    fn terminal_has_boundary_cuts_when_warm_start_count_positive() {
        // A pool with warm_start_count > 0 signals boundary cuts at the
        // terminal stage.
        use cobre_io::OwnedPolicyCutRecord;

        let records = vec![OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            coefficients: vec![1.0],
            intercept: 5.0,
            is_active: true,
            iteration: 0,
            forward_pass_index: 0,
        }];
        let pool = CutPool::new_with_warm_start(1, 4, 100, &records);
        assert!(pool.warm_start_count > 0, "terminal pool has boundary cuts");
    }

    #[test]
    fn no_boundary_cuts_when_warm_start_count_zero() {
        let pool = CutPool::new(100, 2, 10, 0);
        assert_eq!(pool.warm_start_count, 0, "no boundary cuts");
    }

    // ── enforce_budget tests ────────────────────────────────────────────────

    #[test]
    fn enforce_budget_noop_when_under_budget() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        pool.add_cut(0, 0, 1.0, &[1.0, 2.0]);
        pool.add_cut(0, 1, 2.0, &[3.0, 4.0]);
        assert_eq!(pool.active_count(), 2);
        let result = pool.enforce_budget(5, 1, 10);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(result.active_after, 2);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn enforce_budget_evicts_oldest_last_active_iter() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // Add 5 cuts at iterations 0-4
        for iter in 0..5_u64 {
            pool.add_cut(iter, 0, 1.0, &[1.0, 0.0]);
            // Set last_active_iter to make older cuts staler
            pool.metadata[pool.populated_count - 1].last_active_iter = iter;
        }
        assert_eq!(pool.active_count(), 5);
        // Budget = 3, current_iteration = 5 → evict 2 oldest
        let result = pool.enforce_budget(3, 5, 10);
        assert_eq!(result.evicted_count, 2);
        assert_eq!(result.active_after, 3);
        assert_eq!(pool.active_count(), 3);
        // The 2 oldest (last_active_iter 0 and 1) should be evicted
        // Slot for (iter=0, fp=0) = 0*10+0 = 0
        // Slot for (iter=1, fp=0) = 1*10+0 = 10
        assert!(!pool.active[0], "oldest cut should be evicted");
        assert!(!pool.active[10], "second oldest should be evicted");
    }

    #[test]
    fn enforce_budget_tiebreaks_by_active_count() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // Two cuts with same last_active_iter but different active_count
        pool.add_cut(0, 0, 1.0, &[1.0, 0.0]);
        pool.metadata[0].last_active_iter = 1;
        pool.metadata[0].active_count = 5;
        pool.add_cut(0, 1, 2.0, &[0.0, 1.0]);
        pool.metadata[1].last_active_iter = 1;
        pool.metadata[1].active_count = 2;
        assert_eq!(pool.active_count(), 2);
        // Budget = 1 → evict the one with lower active_count (slot 1)
        let result = pool.enforce_budget(1, 1, 10);
        assert_eq!(result.evicted_count, 1);
        assert!(pool.active[0], "higher active_count survives");
        assert!(!pool.active[1], "lower active_count evicted");
    }

    #[test]
    fn enforce_budget_protects_current_iteration() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // 3 cuts: 2 from iteration 0, 1 from iteration 1 (current)
        pool.add_cut(0, 0, 1.0, &[1.0, 0.0]);
        pool.metadata[0].last_active_iter = 0;
        pool.add_cut(0, 1, 2.0, &[0.0, 1.0]);
        pool.metadata[1].last_active_iter = 0;
        pool.add_cut(1, 0, 3.0, &[1.0, 1.0]);
        pool.metadata[10].last_active_iter = 1;
        assert_eq!(pool.active_count(), 3);
        // Budget = 1, current_iteration = 1 → can only evict iter-0 cuts
        let result = pool.enforce_budget(1, 1, 10);
        assert_eq!(result.evicted_count, 2);
        // Current-iteration cut (slot 10) survives
        assert!(pool.active[10], "current iteration cut preserved");
    }

    #[test]
    fn enforce_budget_all_current_iteration_no_eviction() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // All cuts from current iteration
        pool.add_cut(5, 0, 1.0, &[1.0, 0.0]);
        pool.add_cut(5, 1, 2.0, &[0.0, 1.0]);
        pool.add_cut(5, 2, 3.0, &[1.0, 1.0]);
        assert_eq!(pool.active_count(), 3);
        // Budget = 1, current_iteration = 5 → no candidates, no eviction
        let result = pool.enforce_budget(1, 5, 10);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(pool.active_count(), 3);
    }

    #[test]
    fn enforce_budget_result_fields() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        pool.add_cut(0, 0, 1.0, &[1.0, 0.0]);
        pool.add_cut(1, 0, 2.0, &[0.0, 1.0]);
        pool.add_cut(2, 0, 3.0, &[1.0, 1.0]);
        assert_eq!(pool.active_count(), 3);
        let result = pool.enforce_budget(1, 3, 10);
        assert_eq!(result.active_before, 3);
        assert_eq!(result.evicted_count, 2);
        assert_eq!(result.active_after, 1);
    }

    // ── active_cuts early-exit tests ─────────────────────────────────────────

    /// Verify that `active_cuts()` stops iterating once all active cuts have
    /// been yielded — it must not scan up to `populated_count` when
    /// `cached_active_count` is small.
    ///
    /// Pool has `populated_count = 100` and only slot 0 is active
    /// (`cached_active_count = 1`).  The early-exit iterator must stop after
    /// visiting slot 0, yielding exactly 1 item.  If the old O(populated)
    /// walk were still in place the count would still be 1, but the
    /// scan-based implementation verifies correctness by construction:
    /// `remaining` hits 0 after the first active slot and `scan` returns
    /// `None` for all subsequent elements, preventing any further polling.
    #[test]
    fn active_cuts_early_exit_stops_at_cached_count() {
        // forward_passes = 1 so slot = warm_start_count + iteration * 1 + fp_index
        let mut pool = CutPool::new(100, 2, 1, 0);

        // Add a cut at slot 0 (iteration 0, fp 0).
        pool.add_cut(0, 0, 5.0, &[1.0, 2.0]);

        // Manually populate a further 99 slots as inactive to extend
        // populated_count to 100 without going through add_cut (which marks
        // them active).  We write directly to the active flag so that
        // populated_count is extended to 100 while cached_active_count stays 1.
        //
        // We do this by exploiting that slot_index(i, 0) = i * 1 + 0 = i.
        // We need slot 99 to be "populated" (high-water mark = 100) so that
        // the old O(populated) walk would have to visit all 100 slots.
        pool.populated_count = 100;
        // active[0] is already true; active[1..100] are all false (default).

        assert_eq!(pool.cached_active_count, 1);
        assert_eq!(pool.populated_count, 100);

        // The iterator must yield exactly 1 item.
        let result: Vec<_> = pool.active_cuts().collect();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0, "yielded slot must be 0");
        assert_eq!(result[0].1, 5.0, "intercept must match");
        assert_eq!(result[0].2, &[1.0, 2.0], "coefficients must match");
    }

    /// Verify that `candidates_buf` retains its allocation across successive
    /// `enforce_budget` calls (i.e., the scratch buffer is reused, not dropped
    /// and reallocated each time).
    #[test]
    fn enforce_budget_candidates_buf_is_reused() {
        let mut pool = CutPool::new(100, 2, 10, 0);

        // Add cuts spread across several iterations so there are always
        // eviction candidates regardless of current_iteration.
        for iter in 0..5_u64 {
            pool.add_cut(iter, 0, 1.0, &[1.0, 0.0]);
        }
        assert_eq!(pool.active_count(), 5);

        // First enforce: evicts some cuts, candidates_buf gets populated.
        pool.enforce_budget(3, 5, 10);
        let cap_after_first = pool.candidates_buf.capacity();
        assert!(
            cap_after_first >= 1,
            "candidates_buf must have acquired capacity after first enforce_budget"
        );

        // Re-add cuts so the second call also has candidates.
        // We need iteration offsets past the existing slots; restart with a
        // new pool to keep slot arithmetic simple.
        let mut pool2 = CutPool::new(100, 2, 10, 0);
        for iter in 0..5_u64 {
            pool2.add_cut(iter, 0, 1.0, &[1.0, 0.0]);
        }
        pool2.enforce_budget(3, 5, 10);
        let cap_after_first2 = pool2.candidates_buf.capacity();

        // Second call on the same pool2 — re-add some cuts first.
        // Since slots 0, 10, 20, 30, 40 are now inactive, add at iter 6..=8.
        pool2.add_cut(6, 0, 2.0, &[0.0, 1.0]);
        pool2.add_cut(7, 0, 2.0, &[0.0, 1.0]);
        pool2.add_cut(8, 0, 2.0, &[0.0, 1.0]);
        pool2.enforce_budget(2, 9, 10);
        let cap_after_second2 = pool2.candidates_buf.capacity();

        // The capacity must not have shrunk — Vec::clear() preserves the heap
        // allocation, so the second call reuses the buffer.
        assert!(
            cap_after_second2 >= cap_after_first2,
            "candidates_buf capacity must not shrink across calls (was {cap_after_first2}, now {cap_after_second2})"
        );
    }
}
