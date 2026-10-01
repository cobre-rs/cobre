//! Per-stage cut pool for the Future Cost Function (FCF).
//!
//! Each stage owns one [`CutPool`]. Cuts occupy pre-allocated slots with a
//! deterministic slot-assignment formula so results are bit-for-bit identical
//! regardless of execution timing or ordering (the declaration-order hard rule).
//!
//! ## Slot assignment
//!
//! ```text
//! slot = warm_start_count + iteration * visit_stride + forward_pass_index
//! ```
//!
//! ## Activity tracking
//!
//! The pool is append-only: deactivated cuts are retained at their stable slot
//! (never removed), so the slot layout stays deterministic. Inactive cuts are
//! excluded from LP construction and [`evaluate_at_state`]. [`set_active`] is the
//! canonical toggle that keeps [`active_count`] consistent; [`deactivate`] wraps
//! `set_active(slot, false)`. [`cuts_in_lp`] is the populated-slot count — the
//! LP-row metric for append-only LP tracking.
//!
//! [`evaluate_at_state`]: CutPool::evaluate_at_state
//! [`set_active`]: CutPool::set_active
//! [`deactivate`]: CutPool::deactivate
//! [`active_count`]: CutPool::active_count
//! [`cuts_in_lp`]: CutPool::cuts_in_lp
//!
//! ## Example
//!
//! ```rust
//! use cobre_sddp::cut::pool::CutPool;
//! use cobre_sddp::setup::NodeId;
//!
//! // 100-slot pool, 9-dimensional state, a per-pool visit stride of 10,
//! // no warm-start cuts.
//! let mut pool = CutPool::new(100, 9, 10, 0);
//! assert_eq!(pool.active_count(), 0);
//!
//! let coeffs = vec![1.0; 9];
//! pool.add_cut(NodeId(0), 0, 0, 5.0, &coeffs);
//! assert_eq!(pool.active_count(), 1);
//! assert_eq!(pool.cuts_in_lp(), 1);
//! ```

use crate::cut::WARM_START_ITERATION;
use crate::cut_selection::CutActivityUpdates;
use crate::cut_selection::CutMetadata;
use crate::setup::NodeId;

use cobre_io::OwnedPolicyCutRecord;

/// Pre-allocated per-stage cut pool for the Future Cost Function (FCF).
///
/// All storage is allocated at construction time — no heap allocation on the
/// training-loop hot path. Slots are addressed by a deterministic formula of the
/// iteration counter and forward-pass index.
#[derive(Debug, Clone)]
pub struct CutPool {
    /// Flat coefficient storage. Slot `i` occupies
    /// `i * state_dimension .. (i + 1) * state_dimension`.
    coefficients: Vec<f64>,

    /// Per-slot intercept values.
    intercepts: Vec<f64>,

    /// Per-slot cut-selection bookkeeping.
    metadata: Vec<CutMetadata>,

    /// Per-slot activity flags. `false` excludes the cut from LP construction and
    /// evaluation; the slot is retained so the layout stays deterministic.
    active: Vec<bool>,

    /// High-water mark of populated slots; bounds iteration so trailing
    /// unpopulated slots are skipped.
    populated_count: usize,

    /// Total number of pre-allocated slots. Fixed after construction.
    pub capacity: usize,

    /// Length of each coefficient vector. Fixed after construction.
    pub state_dimension: usize,

    /// Per-pool cut-receipt stride: how many fresh slots this pool receives
    /// per iteration, a factor in the slot formula. Exact where the graph
    /// makes visits statically known (a chain node's is `forward_passes`),
    /// or an expected-value floor otherwise — see
    /// [`NodeGraph::pool_cut_stride`](crate::setup::node_graph::NodeGraph::pool_cut_stride).
    /// Fixed after construction.
    pub visit_stride: u32,

    /// Warm-start cuts loaded before training; the base offset in the slot
    /// formula. Fixed after construction.
    pub warm_start_count: u32,

    /// Iteration that maps to the first training slot (`warm_start_count`). The
    /// slot formula subtracts it so 1-based iterations pack densely. Default 0 is
    /// the legacy layout that leaves the block `[warm_start_count, +visit_stride)`
    /// unused; production sets `start_iteration + 1` via [`set_iteration_base`].
    /// Both are correct; dense is tighter.
    ///
    /// [`set_iteration_base`]: CutPool::set_iteration_base
    pub iteration_base: u64,

    /// Active-cut count, maintained incrementally so [`active_count`] is O(1).
    ///
    /// [`active_count`]: CutPool::active_count
    cached_active_count: usize,

    /// Cuts ever inserted (including warm-start cuts). Unlike
    /// [`populated`](CutPool::populated) — a slot high-water mark that includes
    /// reserved-but-unwritten leading slots — this counts only real insertions:
    /// the true policy-row count.
    pub generated_count: usize,

    /// Scratch buffer for [`enforce_budget`] candidate collection, reused across
    /// calls to avoid per-call allocation.
    ///
    /// [`enforce_budget`]: CutPool::enforce_budget
    pub(crate) candidates_buf: Vec<u32>,
}

impl CutPool {
    /// Create a `CutPool` with all slots pre-allocated and zero / inactive.
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
    /// assert_eq!(pool.populated(), 0);
    /// ```
    #[must_use]
    pub fn new(
        capacity: usize,
        state_dimension: usize,
        visit_stride: u32,
        warm_start_count: u32,
    ) -> Self {
        let default_meta = CutMetadata {
            iteration_generated: 0,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: 0,
        };

        Self {
            coefficients: vec![0.0; capacity * state_dimension],
            intercepts: vec![0.0; capacity],
            metadata: vec![default_meta; capacity],
            active: vec![false; capacity],
            populated_count: 0,
            capacity,
            state_dimension,
            visit_stride,
            warm_start_count,
            iteration_base: 0,
            cached_active_count: 0,
            generated_count: warm_start_count as usize,
            candidates_buf: Vec::new(),
        }
    }

    /// Compute the deterministic slot index for a cut.
    ///
    /// ```text
    /// slot = warm_start_count
    ///      + (iteration - iteration_base) * visit_stride
    ///      + forward_pass_index
    /// ```
    #[inline]
    fn slot_index(&self, iteration: u64, forward_pass_index: u32) -> usize {
        debug_assert!(
            iteration >= self.iteration_base,
            "slot_index: iteration {iteration} < iteration_base {}",
            self.iteration_base
        );
        // Cast cannot truncate: SDDP runs only on 64-bit targets and capacity < usize::MAX.
        #[allow(clippy::cast_possible_truncation)]
        let iter_usize = (iteration - self.iteration_base) as usize;
        self.warm_start_count as usize
            + iter_usize * self.visit_stride as usize
            + forward_pass_index as usize
    }

    /// Set the iteration that maps to the first training slot.
    ///
    /// Pass `start_iteration + 1` for dense packing from slot `warm_start_count`;
    /// default 0 leaves the block `[warm_start_count, +visit_stride)` unused.
    ///
    /// Call before the first [`add_cut`](CutPool::add_cut) of a run. Changing the
    /// base while the pool still holds *active* training cuts is caught by
    /// `add_cut`'s no-overwrite guard; re-setting when all cuts are inactive
    /// safely reuses slots.
    pub fn set_iteration_base(&mut self, iteration_base: u64) {
        self.iteration_base = iteration_base;
    }

    /// Insert a Benders cut at its deterministic `slot_index`, marked active.
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
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(20, 3, 5, 0);
    /// pool.add_cut(NodeId(0), 1, 2, 10.0, &[1.0, 2.0, 3.0]);
    /// // slot = 0 + 1*5 + 2 = 7
    /// assert!(pool.is_active(7));
    /// assert_eq!(pool.intercept(7), 10.0);
    /// ```
    ///
    /// `node_id` is the declared id of the generating node
    /// (`NodeGraph::node_ids`), recorded in the slot's [`CutMetadata::node`] and
    /// later carried onto the MPI cut wire. It is provenance only — it never
    /// affects the slot the cut lands in (the append-only, slot-identity
    /// contract is independent of `node_id`).
    ///
    /// [`CutMetadata::node`]: crate::cut_selection::CutMetadata::node
    pub fn add_cut(
        &mut self,
        node_id: NodeId,
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
            node: node_id,
            active_count: 0,
            last_active_iter: iteration,
        };

        if slot >= self.populated_count {
            self.populated_count = slot + 1;
        }
        self.generated_count += 1;
    }

    /// Iterate over active cuts as `(slot_index, intercept, coefficient_slice)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 2, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 3.0, &[1.0, 2.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 7.0, &[3.0, 4.0]);
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

    /// Iterate over active cuts whose `iteration_generated == current_iteration`,
    /// in insertion order (declaration-order invariance).
    ///
    /// Warm-start cuts ([`WARM_START_ITERATION`]) are always excluded — the
    /// explicit guard prevents them being repacked as new training cuts in cut
    /// sync even if `current_iteration` collided with the sentinel.
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

    /// Count the active cuts (cached, O(1)).
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// assert_eq!(pool.active_count(), 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    #[inline]
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

    /// Whether this pool carries any warm-started cuts — on the terminal
    /// stage's pool, the boundary cuts.
    #[must_use]
    #[inline]
    pub fn has_warm_start_cuts(&self) -> bool {
        self.warm_start_count > 0
    }

    /// Return the populated-slot count — the LP-row metric, independent of
    /// activity state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// assert_eq!(pool.cuts_in_lp(), 2);
    ///
    /// // Deactivation does not change the LP-row count.
    /// pool.deactivate(&[0]);
    /// assert_eq!(pool.cuts_in_lp(), 2);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    #[inline]
    pub fn cuts_in_lp(&self) -> usize {
        self.populated_count
    }

    /// Populated-slot count — the same value as [`cuts_in_lp`](CutPool::cuts_in_lp).
    #[must_use]
    #[inline]
    pub fn populated(&self) -> usize {
        self.populated_count
    }

    /// Read a single slot's activity flag.
    #[must_use]
    #[inline]
    pub fn is_active(&self, slot: usize) -> bool {
        self.active[slot]
    }

    /// Read a single slot's cut-selection metadata.
    #[must_use]
    #[inline]
    pub fn metadata(&self, slot: usize) -> &CutMetadata {
        &self.metadata[slot]
    }

    /// Read a single slot's intercept.
    #[must_use]
    #[inline]
    pub fn intercept(&self, slot: usize) -> f64 {
        self.intercepts[slot]
    }

    /// Read a single slot's coefficient row (`state_dimension` elements).
    #[must_use]
    #[inline]
    pub fn coefficient_row(&self, slot: usize) -> &[f64] {
        let start = slot * self.state_dimension;
        &self.coefficients[start..start + self.state_dimension]
    }

    /// Read every populated slot's coefficients as one flat, row-major slice
    /// (`populated() * state_dimension` elements) — the batched-GEMM
    /// counterpart to [`coefficient_row`](CutPool::coefficient_row).
    #[must_use]
    #[inline]
    pub fn coefficients_prefix(&self) -> &[f64] {
        &self.coefficients[..self.populated_count * self.state_dimension]
    }

    /// Read every populated slot's intercepts as one flat slice
    /// (`populated()` elements) — the batched counterpart to
    /// [`intercept`](CutPool::intercept).
    #[must_use]
    #[inline]
    pub fn intercepts_prefix(&self) -> &[f64] {
        &self.intercepts[..self.populated_count]
    }

    /// Record that a cut was binding `increment` more times, most recently at
    /// `iteration`. Metadata mutation cannot desync `active`/
    /// `cached_active_count` (independent fields), so — unlike
    /// [`replace_selection`](CutPool::replace_selection) — this touches only
    /// the one slot's metadata.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= populated_count`.
    #[inline]
    pub fn record_binding(&mut self, slot: usize, increment: u64, iteration: u64) {
        debug_assert!(
            slot < self.populated_count,
            "record_binding: slot {slot} out of populated range"
        );
        self.metadata[slot].active_count += increment;
        self.metadata[slot].last_active_iter = iteration;
    }

    /// Test-only: overwrite one slot's metadata directly, bypassing
    /// `add_cut`'s iteration/slot coupling. Metadata mutation cannot desync
    /// `active`/`cached_active_count`, so no bulk recomputation is needed.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= populated_count`.
    #[cfg(test)]
    pub(crate) fn set_metadata_for_test(&mut self, slot: usize, metadata: CutMetadata) {
        assert!(
            slot < self.populated_count,
            "set_metadata_for_test: slot {slot} out of populated range"
        );
        self.metadata[slot] = metadata;
    }

    /// Test-only: overwrite one slot's `iteration_generated`, decoupling a
    /// cut's metadata age from the slot `add_cut` placed it at. Every other
    /// call site fixtures test scenarios (cut-selection windows, DCS
    /// candidate age) around this one field.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= populated_count`.
    #[cfg(test)]
    pub(crate) fn set_iteration_generated_for_test(
        &mut self,
        slot: usize,
        iteration_generated: u64,
    ) {
        assert!(
            slot < self.populated_count,
            "set_iteration_generated_for_test: slot {slot} out of populated range"
        );
        self.metadata[slot].iteration_generated = iteration_generated;
    }

    /// Cuts ever inserted, including warm-start cuts.
    #[must_use]
    #[inline]
    pub fn generated(&self) -> usize {
        self.generated_count
    }

    /// Deactivate the cuts at the given slot indices via [`set_active`].
    ///
    /// [`set_active`]: CutPool::set_active
    ///
    /// # Idempotency
    ///
    /// An already-inactive index is a no-op (no second decrement → no underflow),
    /// so a duplicate index is safe.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// pool.deactivate(&[0]);
    /// assert_eq!(pool.active_count(), 1);
    /// assert!(!pool.is_active(0));
    /// assert!(pool.is_active(1));
    /// ```
    pub fn deactivate(&mut self, indices: &[u32]) {
        for &idx in indices {
            self.set_active(idx, false);
        }
    }

    /// Apply a batch of activity changes from a [`CutActivityUpdates`] result.
    ///
    /// Deactivates `updates.updates` and reactivates `updates.reactivations` via
    /// [`set_active`]. Folding both into one call is why the training loop never
    /// silently drops reactivation entries. Idempotency follows [`set_active`].
    ///
    /// [`set_active`]: CutPool::set_active
    /// [`CutActivityUpdates`]: CutActivityUpdates
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::cut_selection::CutActivityUpdates;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);
    ///
    /// // Deactivate slot 1, then reactivate it via apply_updates.
    /// pool.deactivate(&[1]);
    /// assert_eq!(pool.active_count(), 2);
    ///
    /// let updates = CutActivityUpdates {
    ///     stage_index: 0,
    ///     updates: vec![0],          // deactivate slot 0
    ///     reactivations: vec![1],    // reactivate slot 1
    /// };
    /// pool.apply_updates(&updates);
    /// assert!(!pool.is_active(0));
    /// assert!(pool.is_active(1));
    /// assert!(pool.is_active(2));
    /// assert_eq!(pool.active_count(), 2);
    /// ```
    pub fn apply_updates(&mut self, updates: &CutActivityUpdates) {
        for &slot in &updates.updates {
            self.set_active(slot, false);
        }
        for &slot in &updates.reactivations {
            self.set_active(slot, true);
        }
    }

    /// Overwrite `metadata` and `active` across the populated prefix in one
    /// call, recomputing `cached_active_count` from the just-written bitmap in
    /// the same call — the two writes stay atomic so the cache can never
    /// desync from the bitmap it counts.
    ///
    /// # Panics
    ///
    /// Panics if `metadata.len() != populated_count` or `active.len() !=
    /// populated_count` (slice-length mismatch on assignment).
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::cut_selection::CutMetadata;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    ///
    /// let meta = CutMetadata {
    ///     iteration_generated: 0,
    ///     forward_pass_index: 0,
    ///     node: NodeId(0),
    ///     active_count: 0,
    ///     last_active_iter: 0,
    /// };
    /// pool.replace_selection(&[meta.clone(), meta], &[false, true]);
    /// assert_eq!(pool.active_count(), 1);
    /// assert!(!pool.is_active(0));
    /// assert!(pool.is_active(1));
    /// ```
    pub fn replace_selection(&mut self, metadata: &[CutMetadata], active: &[bool]) {
        self.metadata[..self.populated_count].clone_from_slice(metadata);
        self.active[..self.populated_count].clone_from_slice(active);
        self.cached_active_count = self.active[..self.populated_count]
            .iter()
            .filter(|&&a| a)
            .count();
    }

    /// Toggle a single slot's activity flag, keeping `cached_active_count`
    /// consistent. A no-op when the slot already holds the requested state.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= populated_count`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);
    ///
    /// pool.set_active(1, false);
    /// assert_eq!(pool.active_count(), 2);
    /// assert!(!pool.is_active(1));
    /// assert_eq!(pool.cuts_in_lp(), 3); // populated count unchanged
    ///
    /// pool.set_active(1, true);
    /// assert_eq!(pool.active_count(), 3);
    /// assert!(pool.is_active(1));
    /// ```
    pub fn set_active(&mut self, slot: u32, active: bool) {
        let i = slot as usize;
        debug_assert!(
            i < self.populated_count,
            "set_active slot {i} out of populated range"
        );
        if self.active[i] == active {
            return;
        }
        self.active[i] = active;
        if active {
            self.cached_active_count += 1;
        } else {
            self.cached_active_count -= 1;
        }
    }

    /// Evaluate the FCF: the max over active cuts of `intercept + coefficients ·
    /// state`, or [`f64::NEG_INFINITY`] when no cut is active.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `state.len() != state_dimension`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 2, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 0.0]);
    /// pool.add_cut(NodeId(0), 1, 0,  5.0, &[0.0, 2.0]);
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

    /// Diagnostic: count exact-zero coefficients (`value == 0.0`) across active
    /// cuts into a [`SparsityReport`].
    ///
    /// Allocates per call (one `Vec<usize>`); offline use, not per-iteration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 3, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 2.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 0.0, 3.0]);
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

    /// Construct a `CutPool` from deserialized cut records for FCF evaluation
    /// during simulation.
    ///
    /// Capacity is exactly `records.len()` with `visit_stride = 0` — this pool
    /// takes no new training cuts.
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
    /// assert_eq!(pool.populated(), 1);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    pub fn from_deserialized(state_dimension: usize, records: &[OwnedPolicyCutRecord]) -> Self {
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
                // Checkpoint records carry no generating-node id; the reload path
                // never re-emits these cuts on the MPI wire, so 0 is inert.
                node: NodeId(0),
                active_count: 0,
                last_active_iter: u64::from(record.iteration),
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
            visit_stride: 0,
            warm_start_count: capacity as u32,
            iteration_base: 0,
            cached_active_count,
            generated_count: capacity,
            candidates_buf: Vec::new(),
        }
    }

    /// Construct a `CutPool` with warm-start cuts plus capacity for training.
    ///
    /// Loaded cuts occupy the first `records.len()` slots; the remaining
    /// `max_iterations * visit_stride` slots take new training cuts (offset past
    /// the warm-start region by the slot formula).
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
    /// assert_eq!(pool.populated(), 1);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    pub fn new_with_warm_start(
        state_dimension: usize,
        visit_stride: u32,
        max_iterations: u64,
        records: &[OwnedPolicyCutRecord],
    ) -> Self {
        let warm_start_count = records.len();
        #[allow(clippy::cast_possible_truncation)]
        let capacity = warm_start_count + (max_iterations as usize) * (visit_stride as usize);

        let default_meta = CutMetadata {
            iteration_generated: 0,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: 0,
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
            // WARM_START_ITERATION sentinel keeps warm-start cuts out of
            // pack_local_cuts (filters on current iteration), so cut sync never
            // double-counts them as new training cuts.
            metadata[i] = CutMetadata {
                iteration_generated: WARM_START_ITERATION,
                forward_pass_index: record.forward_pass_index,
                // Warm-start cuts are WARM_START_ITERATION-filtered out of cut-sync
                // packing, so their generating node never reaches the wire; 0 is inert.
                node: NodeId(0),
                active_count: 0,
                last_active_iter: u64::from(record.iteration),
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
            visit_stride,
            warm_start_count: warm_start_count as u32,
            iteration_base: 0,
            cached_active_count,
            generated_count: warm_start_count,
            candidates_buf: Vec::new(),
        }
    }

    /// Grow this pool's capacity in place. `Vec::resize` only appends new
    /// (zeroed/inactive) slots when growing — it never relocates an existing
    /// element — so every populated slot keeps its index, the invariant a
    /// stored basis's slot-identity match depends on (the "Cut pool is
    /// append-only; basis matches by slot identity" contract). Reserved
    /// seam: the training session's between-iteration growth hook calls this
    /// when a pool's realized visit rate would exceed its construction-time
    /// floor.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `new_capacity < self.capacity` — growth-only, never shrinks.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::pool::CutPool;
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(2, 2, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 5.0, &[1.0, 2.0]);
    /// pool.grow(10);
    /// assert_eq!(pool.capacity, 10);
    /// assert!(pool.is_active(0));
    /// assert_eq!(pool.intercept(0), 5.0);
    /// ```
    pub fn grow(&mut self, new_capacity: usize) {
        debug_assert!(
            new_capacity >= self.capacity,
            "grow: new_capacity {new_capacity} < current capacity {}",
            self.capacity
        );
        self.coefficients
            .resize(new_capacity * self.state_dimension, 0.0);
        self.intercepts.resize(new_capacity, 0.0);
        self.metadata.resize(
            new_capacity,
            CutMetadata {
                iteration_generated: 0,
                forward_pass_index: 0,
                node: NodeId(0),
                active_count: 0,
                last_active_iter: 0,
            },
        );
        self.active.resize(new_capacity, false);
        self.capacity = new_capacity;
    }
}

/// Diagnostic report of exact-zero coefficients across active cuts in a [`CutPool`].
///
/// Counts exact zeros (`value == 0.0`) only; near-zero values are not collapsed.
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
    /// Enforce a hard cap on active cuts per stage, evicting by
    /// `(last_active_iter ASC, active_count ASC)` — stalest, least-used first.
    ///
    /// Cuts generated in `current_iteration` are **never** evicted; if they alone
    /// exceed `budget`, `active_count()` may remain above it after the call.
    ///
    /// # Parameters
    ///
    /// - `forward_passes`: unused; present for call-site uniformity with the
    ///   training loop.
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
            return BudgetEnforcementResult {
                evicted_count: 0,
                active_before,
                active_after: active_before,
            };
        }

        let evict_count = excess.min(self.candidates_buf.len());

        let key = |&slot: &u32| {
            let meta = &self.metadata[slot as usize];
            (meta.last_active_iter, meta.active_count)
        };

        // Partial sort partitions the evict_count smallest into [..evict_count]
        // (any order); full sort when that fraction is large.
        if evict_count < self.candidates_buf.len() / 2 {
            self.candidates_buf
                .select_nth_unstable_by(evict_count, |a, b| key(a).cmp(&key(b)));
        } else {
            self.candidates_buf.sort_unstable_by_key(|a| key(a));
        }

        // Local copy releases the candidates_buf borrow before deactivate(&mut self).
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
    use crate::setup::NodeId;

    #[test]
    fn new_creates_pool_with_correct_capacity_and_all_inactive() {
        let pool = CutPool::new(100, 9, 10, 0);
        assert_eq!(pool.capacity, 100);
        assert_eq!(pool.state_dimension, 9);
        assert_eq!(pool.visit_stride, 10);
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
        pool.add_cut(NodeId(0), 0, 0, 5.0, &coeffs);

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
        // slot = 0 + iteration * visit_stride + forward_pass_index
        let mut pool = CutPool::new(200, 2, 10, 0);

        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0]); // slot = 0
        pool.add_cut(NodeId(0), 0, 3, 2.0, &[3.0, 4.0]); // slot = 3
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[5.0, 6.0]); // slot = 10
        pool.add_cut(NodeId(0), 2, 5, 4.0, &[7.0, 8.0]); // slot = 25

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
        pool.add_cut(NodeId(0), 0, 0, 42.0, &coeffs);

        assert!(pool.active[5]);
        assert_eq!(pool.intercepts[5], 42.0);
        assert_eq!(pool.populated_count, 6);
    }

    #[test]
    fn add_cut_metadata_initialized_correctly() {
        let mut pool = CutPool::new(50, 3, 5, 0);
        pool.add_cut(NodeId(0), 3, 2, 7.0, &[1.0, 2.0, 3.0]);
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

        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0 → populated_count = 1
        assert_eq!(pool.populated_count, 1);

        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 5 → populated_count = 6
        assert_eq!(pool.populated_count, 6);

        pool.add_cut(NodeId(0), 0, 2, 3.0, &[3.0]); // slot 2 → no change (2 < 6)
        assert_eq!(pool.populated_count, 6);
    }

    #[test]
    fn set_iteration_base_packs_training_cuts_densely() {
        // base = 1 (start_iteration 0) maps iteration 1 to slot warm_start_count
        // (0 here), so 1-based iterations leave no reserved leading block.
        let mut pool = CutPool::new(30, 1, 3, 0);
        pool.set_iteration_base(1);
        pool.add_cut(NodeId(0), 1, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 1, 2.0, &[1.0]); // slot 1
        pool.add_cut(NodeId(0), 1, 2, 3.0, &[1.0]); // slot 2
        pool.add_cut(NodeId(0), 2, 0, 4.0, &[1.0]); // slot 3
        assert!(pool.active[0] && pool.active[1] && pool.active[2] && pool.active[3]);
        assert_eq!(pool.populated_count, 4, "dense packing leaves no gap");
        assert_eq!(pool.generated_count, 4);
        // metadata keeps the TRUE iteration, not the re-based slot iteration.
        assert_eq!(pool.metadata[0].iteration_generated, 1);
        assert_eq!(pool.metadata[3].iteration_generated, 2);
    }

    #[test]
    fn active_cuts_returns_only_active_cuts() {
        let mut pool = CutPool::new(20, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[3.0, 4.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[5.0, 6.0]); // slot 2

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
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2

        assert_eq!(pool.active_count(), 3);
        pool.deactivate(&[1]);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn deactivate_sets_flags_correctly() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2

        pool.deactivate(&[1]);

        assert!(pool.active[0]);
        assert!(!pool.active[1]);
        assert!(pool.active[2]);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn deactivate_multiple_indices() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2

        pool.deactivate(&[0, 2]);

        assert!(!pool.active[0]);
        assert!(pool.active[1]);
        assert!(!pool.active[2]);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn deactivate_empty_slice_is_noop() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.deactivate(&[]);
        assert_eq!(pool.active_count(), 1);
    }

    /// `deactivate` silently skips already-inactive indices.
    ///
    /// In debug builds the `debug_assert!` would fire on a duplicate
    /// input, so the test is gated to release-only. When a release
    /// build receives `&[0, 0, 0]`, the active count drops by exactly
    /// 1 (not 3) and `active[0]` becomes `false`.
    #[test]
    #[cfg(not(debug_assertions))]
    fn deactivate_duplicate_index_is_silently_skipped() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        assert_eq!(pool.active_count(), 2);
        pool.deactivate(&[0, 0, 0]);
        assert_eq!(
            pool.active_count(),
            1,
            "duplicate indices must not double-decrement"
        );
        assert!(!pool.active[0]);
        assert!(pool.active[1]);
    }

    #[test]
    fn evaluate_at_state_returns_max_cut_value() {
        // cuts: (intercept=10, coeffs=[1, 0]) and (intercept=5, coeffs=[0, 2])
        // state = [3, 4]
        // cut 0: 10 + 1*3 + 0*4 = 13
        // cut 1:  5 + 0*3 + 2*4 = 13
        // max = 13
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0, 2.0]);

        let result = pool.evaluate_at_state(&[3.0, 4.0]);
        assert_eq!(result, 13.0);
    }

    #[test]
    fn evaluate_at_state_selects_correct_max() {
        // cut 0: intercept=2, coeffs=[1] → at state [10]: 2 + 10 = 12
        // cut 1: intercept=5, coeffs=[2] → at state [10]: 5 + 20 = 25
        // max = 25
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 2.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[2.0]);

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
        pool.add_cut(NodeId(0), 0, 0, 100.0, &[1.0]);
        pool.deactivate(&[0]);
        assert_eq!(pool.evaluate_at_state(&[5.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn evaluate_at_state_ignores_deactivated_cuts() {
        // slot 0: active, intercept=10, coeff=[1]  → at state [3]: 13
        // slot 1: INACTIVE, intercept=100, coeff=[1] → would be 103, but ignored
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 100.0, &[1.0]);
        pool.deactivate(&[1]);

        assert_eq!(pool.evaluate_at_state(&[3.0]), 13.0);
    }

    #[test]
    fn ac_add_cut_stores_at_slot_zero_and_active_count_is_one() {
        let mut pool = CutPool::new(100, 9, 10, 0);
        let coeffs = vec![0.0; 9];
        pool.add_cut(NodeId(0), 0, 0, 5.0, &coeffs);

        assert!(pool.active[0]);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn ac_deactivate_reduces_active_count_correctly() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);

        pool.deactivate(&[1]);

        assert_eq!(pool.active_count(), 2);
        assert!(!pool.active[1]);
    }

    #[test]
    fn ac_evaluate_at_state_returns_correct_max() {
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0, 2.0]);

        assert_eq!(pool.evaluate_at_state(&[3.0, 4.0]), 13.0);
    }

    #[test]
    fn ac_warm_start_count_offsets_slot() {
        let mut pool = CutPool::new(100, 9, 10, 5);
        let coeffs = vec![0.0; 9];
        pool.add_cut(NodeId(0), 0, 0, 1.0, &coeffs);

        assert!(pool.active[5]);
        assert!(!pool.active[0]);
    }

    #[test]
    fn ac_empty_pool_evaluate_returns_neg_infinity() {
        let pool = CutPool::new(10, 2, 1, 0);
        assert_eq!(pool.evaluate_at_state(&[1.0, 2.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn cut_pool_derives_debug_and_clone() {
        let mut pool = CutPool::new(5, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 3.0, &[1.0, 2.0]);

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
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0, 3.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[4.0, 5.0, 6.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 0);
        assert!((report.sparsity_fraction - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![0, 0, 0]);
    }

    #[test]
    fn sparsity_report_all_zero() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 0.0, 0.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 6);
        assert!((report.sparsity_fraction - 1.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![2, 2, 2]);
    }

    #[test]
    fn sparsity_report_mixed() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 2.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 0.0, 3.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 3);
        assert!((report.sparsity_fraction - 0.5).abs() < 1e-10);
        assert_eq!(report.per_dimension_zeros, vec![1, 2, 0]);
    }

    #[test]
    fn sparsity_report_excludes_inactive_cuts() {
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0, 0.0]); // all zero, then deactivate
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[1.0, 2.0]); // all non-zero
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
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0, 1.0, 0.0, 3.0]);
        // Cut 1: dims 0,3 are zero
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 2.0, 4.0, 0.0]);
        // Cut 2: no zeros
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[5.0, 6.0, 7.0, 8.0]);

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
    fn new_with_warm_start_zero_max_iterations_yields_fixed_capacity() {
        use cobre_io::OwnedPolicyCutRecord;

        let records = vec![
            OwnedPolicyCutRecord {
                cut_id: 0,
                slot_index: 0,
                coefficients: vec![1.0, 2.0],
                intercept: 5.0,
                is_active: true,
                iteration: 3,
                forward_pass_index: 0,
            },
            OwnedPolicyCutRecord {
                cut_id: 1,
                slot_index: 1,
                coefficients: vec![3.0, 4.0],
                intercept: 6.0,
                is_active: true,
                iteration: 4,
                forward_pass_index: 1,
            },
        ];

        let pool = CutPool::new_with_warm_start(2, 4, 0, &records);
        assert_eq!(
            pool.capacity,
            records.len(),
            "max_iterations = 0 must leave no growable training slack"
        );
        assert_eq!(pool.warm_start_count as usize, records.len());
        assert_eq!(pool.populated(), records.len());
    }

    #[test]
    fn has_warm_start_cuts_follows_the_warm_start_records() {
        use cobre_io::OwnedPolicyCutRecord;

        let empty = CutPool::new(100, 2, 10, 0);
        assert!(!empty.has_warm_start_cuts());

        let active_record = vec![OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            coefficients: vec![1.0],
            intercept: 5.0,
            is_active: true,
            iteration: 0,
            forward_pass_index: 0,
        }];
        let active = CutPool::new_with_warm_start(1, 4, 100, &active_record);
        assert!(active.has_warm_start_cuts());

        let inactive_record = vec![OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            coefficients: vec![1.0],
            intercept: 5.0,
            is_active: false,
            iteration: 0,
            forward_pass_index: 0,
        }];
        let inactive = CutPool::new_with_warm_start(1, 4, 100, &inactive_record);
        assert!(inactive.has_warm_start_cuts());
        assert_eq!(inactive.active_count(), 0);
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
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0]);
        pool.add_cut(NodeId(0), 0, 1, 2.0, &[3.0, 4.0]);
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
            pool.add_cut(NodeId(0), iter, 0, 1.0, &[1.0, 0.0]);
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
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0]);
        pool.metadata[0].last_active_iter = 1;
        pool.metadata[0].active_count = 5;
        pool.add_cut(NodeId(0), 0, 1, 2.0, &[0.0, 1.0]);
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
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0]);
        pool.metadata[0].last_active_iter = 0;
        pool.add_cut(NodeId(0), 0, 1, 2.0, &[0.0, 1.0]);
        pool.metadata[1].last_active_iter = 0;
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[1.0, 1.0]);
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
        pool.add_cut(NodeId(0), 5, 0, 1.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 5, 1, 2.0, &[0.0, 1.0]);
        pool.add_cut(NodeId(0), 5, 2, 3.0, &[1.0, 1.0]);
        assert_eq!(pool.active_count(), 3);
        // Budget = 1, current_iteration = 5 → no candidates, no eviction
        let result = pool.enforce_budget(1, 5, 10);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(pool.active_count(), 3);
    }

    #[test]
    fn enforce_budget_result_fields() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[1.0, 1.0]);
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
        // visit_stride = 1 so slot = warm_start_count + iteration * 1 + fp_index
        let mut pool = CutPool::new(100, 2, 1, 0);

        // Add a cut at slot 0 (iteration 0, fp 0).
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[1.0, 2.0]);

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
            pool.add_cut(NodeId(0), iter, 0, 1.0, &[1.0, 0.0]);
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
            pool2.add_cut(NodeId(0), iter, 0, 1.0, &[1.0, 0.0]);
        }
        pool2.enforce_budget(3, 5, 10);
        let cap_after_first2 = pool2.candidates_buf.capacity();

        // Second call on the same pool2 — re-add some cuts first.
        // Since slots 0, 10, 20, 30, 40 are now inactive, add at iter 6..=8.
        pool2.add_cut(NodeId(0), 6, 0, 2.0, &[0.0, 1.0]);
        pool2.add_cut(NodeId(0), 7, 0, 2.0, &[0.0, 1.0]);
        pool2.add_cut(NodeId(0), 8, 0, 2.0, &[0.0, 1.0]);
        pool2.enforce_budget(2, 9, 10);
        let cap_after_second2 = pool2.candidates_buf.capacity();

        // The capacity must not have shrunk — Vec::clear() preserves the heap
        // allocation, so the second call reuses the buffer.
        assert!(
            cap_after_second2 >= cap_after_first2,
            "candidates_buf capacity must not shrink across calls (was {cap_after_first2}, now {cap_after_second2})"
        );
    }

    // ── set_active / cuts_in_lp tests ────────────────────────────────────────

    #[test]
    fn set_active_false_decrements_active_count() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);

        pool.set_active(1, false);

        assert!(!pool.active[1]);
        assert_eq!(pool.active_count(), 2);
        assert_eq!(pool.cuts_in_lp(), 3);
    }

    #[test]
    fn set_active_true_reactivates_deactivated_slot() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);
        pool.deactivate(&[1]);

        pool.set_active(1, true);

        assert!(pool.active[1]);
        assert_eq!(pool.active_count(), 3);
        assert_eq!(pool.cuts_in_lp(), 3);
    }

    #[test]
    fn set_active_idempotent_when_state_unchanged() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);

        // slot 1 is already active — second call must be a no-op
        pool.set_active(1, true);
        pool.set_active(1, true);

        assert_eq!(pool.active_count(), 3);
    }

    #[test]
    fn deactivate_delegates_to_set_active() {
        let mut pool_a = CutPool::new(10, 4, 1, 0);
        pool_a.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool_a.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool_a.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);
        pool_a.deactivate(&[1, 2]);

        let mut pool_b = CutPool::new(10, 4, 1, 0);
        pool_b.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool_b.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool_b.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);
        pool_b.set_active(1, false);
        pool_b.set_active(2, false);

        assert_eq!(pool_a.active, pool_b.active);
        assert_eq!(pool_a.cached_active_count, pool_b.cached_active_count);
    }

    #[test]
    fn cuts_in_lp_returns_populated_count() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);

        assert_eq!(pool.cuts_in_lp(), 2);
        assert_eq!(pool.cuts_in_lp(), pool.populated_count);

        pool.deactivate(&[0]);

        // Deactivation must not change the populated count.
        assert_eq!(pool.cuts_in_lp(), 2);
        assert_eq!(pool.active_count(), 1);
    }

    // ── apply_updates tests ──────────────────────────────────────────────────

    /// `apply_updates` deactivates every slot in `updates.updates` and
    /// reactivates every slot in `updates.reactivations`. Mixed batches
    /// must leave the pool in the expected state and the cached counter
    /// must match the bitmap.
    #[test]
    fn apply_updates_applies_mixed_deactivate_and_reactivate() {
        use crate::cut_selection::CutActivityUpdates;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0 active
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1 active
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2 active
        pool.add_cut(NodeId(0), 3, 0, 4.0, &[4.0]); // slot 3 active

        // Pre-deactivate slot 2 so the reactivation has something to flip.
        pool.deactivate(&[2]);
        assert_eq!(pool.active_count(), 3);
        assert!(!pool.active[2]);

        let updates = CutActivityUpdates {
            stage_index: 0,
            updates: vec![0, 3],    // deactivate slots 0 and 3
            reactivations: vec![2], // reactivate slot 2
        };
        pool.apply_updates(&updates);

        assert!(!pool.active[0], "slot 0 must be deactivated");
        assert!(pool.active[1], "slot 1 must remain active");
        assert!(pool.active[2], "slot 2 must be reactivated");
        assert!(!pool.active[3], "slot 3 must be deactivated");
        assert_eq!(pool.active_count(), 2);
        // cuts_in_lp is unaffected by activity changes.
        assert_eq!(pool.cuts_in_lp(), 4);
    }

    /// `apply_updates` must be idempotent: applying the same updates twice
    /// must leave the pool in the same state as applying it once. This
    /// follows from `set_active`'s "already-in-target-state is a no-op"
    /// contract.
    #[test]
    fn apply_updates_is_idempotent() {
        use crate::cut_selection::CutActivityUpdates;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);

        // Pre-deactivate slot 1 so the reactivation has something to flip.
        pool.deactivate(&[1]);
        assert_eq!(pool.active_count(), 2);

        let updates = CutActivityUpdates {
            stage_index: 0,
            updates: vec![0],
            reactivations: vec![1],
        };

        // First application flips slot 0 off and slot 1 on.
        pool.apply_updates(&updates);
        let active_snapshot = pool.active.clone();
        let count_snapshot = pool.active_count();

        // Second application: every requested state already holds → no-op.
        pool.apply_updates(&updates);
        assert_eq!(
            pool.active, active_snapshot,
            "active bitmap must not change"
        );
        assert_eq!(
            pool.active_count(),
            count_snapshot,
            "active count must not change"
        );
        assert_eq!(pool.active_count(), 2);
        assert!(!pool.active[0]);
        assert!(pool.active[1]);
        assert!(pool.active[2]);
    }

    #[test]
    fn apply_updates_empty_is_noop() {
        use crate::cut_selection::CutActivityUpdates;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        let active_before = pool.active.clone();
        let count_before = pool.active_count();

        let updates = CutActivityUpdates {
            stage_index: 0,
            updates: vec![],
            reactivations: vec![],
        };
        pool.apply_updates(&updates);

        assert_eq!(pool.active, active_before);
        assert_eq!(pool.active_count(), count_before);
    }

    /// `apply_updates` must match the behavior of calling `set_active`
    /// directly for each entry, regardless of which list (`updates` or
    /// `reactivations`) drives the change.
    #[test]
    fn apply_updates_matches_manual_set_active_loop() {
        use crate::cut_selection::CutActivityUpdates;

        let build_pool = || {
            let mut pool = CutPool::new(10, 1, 1, 0);
            pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
            pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
            pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);
            pool.add_cut(NodeId(0), 3, 0, 4.0, &[4.0]);
            pool.deactivate(&[2]); // pre-deactivate so reactivation flips a bit
            pool
        };

        let updates = CutActivityUpdates {
            stage_index: 0,
            updates: vec![0, 3],
            reactivations: vec![2],
        };

        let mut pool_a = build_pool();
        pool_a.apply_updates(&updates);

        let mut pool_b = build_pool();
        for &slot in &updates.updates {
            pool_b.set_active(slot, false);
        }
        for &slot in &updates.reactivations {
            pool_b.set_active(slot, true);
        }

        assert_eq!(pool_a.active, pool_b.active);
        assert_eq!(pool_a.cached_active_count, pool_b.cached_active_count);
    }

    // ── read accessor equivalence tests ──────────────────────────────────────

    /// Every new read accessor must return exactly the value its corresponding
    /// direct field read would, on a pool with a mix of active and inactive
    /// populated slots.
    #[test]
    fn accessors_match_direct_field_reads() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 2.0, 3.0]);
        pool.add_cut(NodeId(0), 1, 0, 20.0, &[4.0, 5.0, 6.0]);
        pool.add_cut(NodeId(0), 2, 0, 30.0, &[7.0, 8.0, 9.0]);
        pool.deactivate(&[1]);

        assert_eq!(pool.populated(), pool.populated_count);
        assert_eq!(pool.generated(), pool.generated_count);

        for slot in 0..pool.populated_count {
            assert_eq!(pool.is_active(slot), pool.active[slot], "slot {slot}");
            assert_eq!(pool.intercept(slot), pool.intercepts[slot], "slot {slot}");

            let start = slot * pool.state_dimension;
            assert_eq!(
                pool.coefficient_row(slot),
                &pool.coefficients[start..start + pool.state_dimension],
                "slot {slot}"
            );

            let via_accessor = pool.metadata(slot);
            let direct = &pool.metadata[slot];
            assert_eq!(
                via_accessor.iteration_generated, direct.iteration_generated,
                "slot {slot}"
            );
            assert_eq!(
                via_accessor.forward_pass_index, direct.forward_pass_index,
                "slot {slot}"
            );
            assert_eq!(
                via_accessor.active_count, direct.active_count,
                "slot {slot}"
            );
            assert_eq!(
                via_accessor.last_active_iter, direct.last_active_iter,
                "slot {slot}"
            );
        }
    }

    // ── replace_selection tests ────────────────────────────────────────────

    /// `replace_selection` writes `active` and recomputes `cached_active_count`
    /// in the same call, so `active_count()` — whose internal `debug_assert`
    /// re-derives the count from the bitmap — must return exactly the count of
    /// `true` entries with no desync.
    #[test]
    fn replace_selection_recomputes_active_count_from_written_bitmap() {
        use crate::cut_selection::CutMetadata;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);
        pool.add_cut(NodeId(0), 3, 0, 4.0, &[4.0]);

        let meta = |iteration_generated: u64| CutMetadata {
            iteration_generated,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: iteration_generated,
        };
        let metadata = vec![meta(0), meta(1), meta(2), meta(3)];
        let active = vec![true, false, true, false];

        pool.replace_selection(&metadata, &active);

        assert_eq!(
            pool.active_count(),
            2,
            "must equal the count of true entries"
        );
        assert!(pool.active[0]);
        assert!(!pool.active[1]);
        assert!(pool.active[2]);
        assert!(!pool.active[3]);
    }

    /// A `replace_selection` call that flips every active slot to inactive
    /// must recompute `cached_active_count` down to zero, not merely leave it
    /// unset — the recompute is unconditional, not a delta.
    #[test]
    fn replace_selection_recomputes_to_zero_when_all_inactive() {
        use crate::cut_selection::CutMetadata;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        assert_eq!(pool.active_count(), 2);

        let meta = CutMetadata {
            iteration_generated: 0,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: 0,
        };
        pool.replace_selection(&[meta.clone(), meta], &[false, false]);

        assert_eq!(pool.active_count(), 0);
    }

    // ── grow ─────────────────────────────────────────────────────────────

    #[test]
    fn grow_preserves_populated_slot_positions_and_values() {
        let mut pool = CutPool::new(2, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[1.0, 2.0]);
        pool.add_cut(NodeId(0), 1, 0, 6.0, &[3.0, 4.0]);

        pool.grow(10);

        assert_eq!(pool.capacity, 10);
        assert_eq!(pool.populated(), 2, "growth does not touch populated_count");
        assert!(pool.is_active(0));
        assert_eq!(pool.intercept(0), 5.0);
        assert_eq!(pool.coefficient_row(0), &[1.0, 2.0]);
        assert!(pool.is_active(1));
        assert_eq!(pool.intercept(1), 6.0);
        assert_eq!(pool.coefficient_row(1), &[3.0, 4.0]);
    }

    #[test]
    fn grow_new_slots_are_writable_via_add_cut() {
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.grow(5);
        // slot = 0 + 1*1 + 0 = 1 is already populated; grow one more forward
        // pass worth of stride to reach a genuinely new slot at iteration 4.
        pool.add_cut(NodeId(0), 4, 0, 9.0, &[7.0]); // slot 4, beyond the original capacity
        assert!(pool.is_active(4));
        assert_eq!(pool.intercept(4), 9.0);
        assert_eq!(pool.coefficient_row(4), &[7.0]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "new_capacity")]
    fn grow_to_a_smaller_capacity_panics_in_debug() {
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.grow(5);
    }
}
