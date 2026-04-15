//! Bidirectional mapping between [`CutPool`] slot indices and LP row indices.
//!
//! [`CutRowMap`] is used by the incremental cut management system to track
//! which cuts from the [`CutPool`] are currently present as rows in the live
//! LP solver instance. It supports O(1) slot-to-row and O(1) row-to-slot
//! lookups.
//!
//! ## Row index stability
//!
//! `HiGHS` assigns row indices sequentially when `add_rows` is called: if the
//! LP has R rows and N rows are appended, the new rows get indices
//! R, R+1, ..., R+N-1. Row indices are stable across subsequent `add_rows`
//! calls. [`CutRowMap`] relies on this property.
//!
//! ## Deactivation semantics
//!
//! When a cut is deactivated (bound-zeroed), the LP row remains allocated.
//! The slot-to-row mapping is preserved so the row can still be identified.
//! Deactivated rows are "phantom" rows -- they exist in the LP but are
//! non-binding. Periodic rebuilds purge phantom rows by resetting the map
//! and doing a full `load_model + add_rows` cycle.
//!
//! [`CutPool`]: super::pool::CutPool
//!
//! ## Example
//!
//! ```rust
//! use cobre_sddp::cut::row_map::CutRowMap;
//!
//! let mut map = CutRowMap::new(100, 50);
//! assert_eq!(map.total_cut_rows(), 0);
//! assert_eq!(map.next_row(), 50);
//!
//! let row = map.insert(7);
//! assert_eq!(row, 50);
//! assert_eq!(map.lp_row_for_slot(7), Some(50));
//! assert_eq!(map.slot_for_lp_row(50), Some(7));
//! assert_eq!(map.active_count(), 1);
//! ```

/// Bidirectional mapping between [`CutPool`](super::pool::CutPool) slot
/// indices and LP row indices for incrementally managed cuts.
///
/// Supports O(1) slot-to-row and O(1) row-to-slot lookups. All storage is
/// pre-allocated at construction time.
///
/// A slot can be in one of three states:
/// - **Unmapped**: never inserted (or cleared by [`reset`](Self::reset)).
/// - **Active**: inserted and contributing to the LP objective.
/// - **Deactivated**: inserted but bound-zeroed (phantom row).
#[derive(Debug, Clone)]
pub struct CutRowMap {
    /// `slot_to_row[slot] = Some(lp_row)` if the cut at `slot` has been
    /// inserted into the LP (whether active or deactivated), or `None` if
    /// it has never been added.
    slot_to_row: Vec<Option<usize>>,

    /// `row_to_slot[lp_row - base_row_offset] = Some(slot)` for reverse
    /// lookup. Only covers the dynamic cut row region (rows >= `base_row_offset`).
    row_to_slot: Vec<Option<usize>>,

    /// `is_active[slot] = true` if the cut was inserted and has NOT been
    /// deactivated. Used to distinguish active from deactivated slots without
    /// scanning the LP.
    is_active: Vec<bool>,

    /// The LP row index where cut rows begin. Equal to the number of template
    /// (structural) rows in the LP.
    base_row_offset: usize,

    /// The next LP row index to assign when a new cut is appended.
    next_row: usize,

    /// Number of currently active (non-deactivated) cut rows.
    active_count: usize,
}

impl CutRowMap {
    /// Create a new `CutRowMap`.
    ///
    /// # Parameters
    ///
    /// - `pool_capacity`: total number of slots in the [`CutPool`](super::pool::CutPool).
    ///   Determines the size of the slot-indexed vectors.
    /// - `base_row_offset`: the number of template (structural) rows in the
    ///   LP. Cut rows start at this index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::row_map::CutRowMap;
    ///
    /// let map = CutRowMap::new(100, 50);
    /// assert_eq!(map.base_row_offset(), 50);
    /// assert_eq!(map.next_row(), 50);
    /// assert_eq!(map.active_count(), 0);
    /// assert_eq!(map.total_cut_rows(), 0);
    /// ```
    #[must_use]
    pub fn new(pool_capacity: usize, base_row_offset: usize) -> Self {
        Self {
            slot_to_row: vec![None; pool_capacity],
            row_to_slot: Vec::new(),
            is_active: vec![false; pool_capacity],
            base_row_offset,
            next_row: base_row_offset,
            active_count: 0,
        }
    }

    /// Record that the cut at `slot` has been appended to the LP.
    ///
    /// Assigns the next available LP row index and updates both mappings.
    /// Returns the assigned LP row index.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= pool_capacity` or if `slot` is already mapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::row_map::CutRowMap;
    ///
    /// let mut map = CutRowMap::new(10, 20);
    /// let row = map.insert(3);
    /// assert_eq!(row, 20);
    /// assert_eq!(map.lp_row_for_slot(3), Some(20));
    /// assert_eq!(map.slot_for_lp_row(20), Some(3));
    /// ```
    pub fn insert(&mut self, slot: usize) -> usize {
        debug_assert!(
            slot < self.slot_to_row.len(),
            "CutRowMap::insert: slot {slot} >= pool_capacity {}",
            self.slot_to_row.len()
        );
        debug_assert!(
            self.slot_to_row[slot].is_none(),
            "CutRowMap::insert: slot {slot} is already mapped to LP row {:?}",
            self.slot_to_row[slot]
        );

        let lp_row = self.next_row;
        self.slot_to_row[slot] = Some(lp_row);
        self.is_active[slot] = true;

        // Grow row_to_slot to cover the new row index.
        let relative = lp_row - self.base_row_offset;
        if relative >= self.row_to_slot.len() {
            self.row_to_slot.resize(relative + 1, None);
        }
        self.row_to_slot[relative] = Some(slot);

        self.next_row += 1;
        self.active_count += 1;
        lp_row
    }

    /// Mark the cut at `slot` as deactivated (bound-zeroed).
    ///
    /// The LP row remains allocated (the row index is preserved), but the
    /// slot is marked as deactivated. The row-to-slot mapping is preserved
    /// so that LP row duals can still be attributed to the original slot.
    ///
    /// Does nothing if the slot is already deactivated.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot` is not currently mapped (never inserted).
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::row_map::CutRowMap;
    ///
    /// let mut map = CutRowMap::new(10, 20);
    /// map.insert(3);
    /// assert_eq!(map.active_count(), 1);
    ///
    /// map.deactivate(3);
    /// assert_eq!(map.active_count(), 0);
    /// // Row mapping is preserved:
    /// assert_eq!(map.lp_row_for_slot(3), Some(20));
    /// ```
    pub fn deactivate(&mut self, slot: usize) {
        debug_assert!(
            slot < self.slot_to_row.len() && self.slot_to_row[slot].is_some(),
            "CutRowMap::deactivate: slot {slot} is not mapped"
        );

        if self.is_active[slot] {
            self.is_active[slot] = false;
            self.active_count -= 1;
        }
    }

    /// Look up the LP row index for a [`CutPool`](super::pool::CutPool) slot.
    ///
    /// Returns `Some(lp_row)` if the cut at `slot` has been inserted into
    /// the LP (whether active or deactivated), or `None` if the slot has
    /// never been added.
    #[must_use]
    #[inline]
    pub fn lp_row_for_slot(&self, slot: usize) -> Option<usize> {
        self.slot_to_row.get(slot).copied().flatten()
    }

    /// Look up the [`CutPool`](super::pool::CutPool) slot for an LP row index.
    ///
    /// Returns `Some(slot)` if the LP row corresponds to a cut, or `None`
    /// if the row is a structural template row (index < `base_row_offset`)
    /// or is not mapped.
    #[must_use]
    #[inline]
    pub fn slot_for_lp_row(&self, lp_row: usize) -> Option<usize> {
        if lp_row < self.base_row_offset {
            return None;
        }
        let relative = lp_row - self.base_row_offset;
        self.row_to_slot.get(relative).copied().flatten()
    }

    /// Return the number of LP rows currently assigned to cuts (both active
    /// and deactivated).
    #[must_use]
    #[inline]
    pub fn total_cut_rows(&self) -> usize {
        self.next_row - self.base_row_offset
    }

    /// Return the number of active (non-deactivated) cut rows.
    #[must_use]
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Return the base row offset (template row count).
    #[must_use]
    #[inline]
    pub fn base_row_offset(&self) -> usize {
        self.base_row_offset
    }

    /// Return the next row index that will be assigned.
    #[must_use]
    #[inline]
    pub fn next_row(&self) -> usize {
        self.next_row
    }

    /// Check whether the cut at `slot` is currently active (inserted and not
    /// deactivated).
    #[must_use]
    #[inline]
    pub fn is_slot_active(&self, slot: usize) -> bool {
        slot < self.is_active.len() && self.is_active[slot]
    }

    /// Reset the map for a full LP rebuild.
    ///
    /// Clears all mappings and resets `next_row` to `new_base_row_offset`.
    /// Called when a periodic full rebuild purges phantom rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::row_map::CutRowMap;
    ///
    /// let mut map = CutRowMap::new(10, 20);
    /// map.insert(0);
    /// map.insert(1);
    /// assert_eq!(map.total_cut_rows(), 2);
    ///
    /// map.reset(25);
    /// assert_eq!(map.total_cut_rows(), 0);
    /// assert_eq!(map.active_count(), 0);
    /// assert_eq!(map.next_row(), 25);
    /// assert_eq!(map.base_row_offset(), 25);
    /// ```
    pub fn reset(&mut self, new_base_row_offset: usize) {
        self.slot_to_row.fill(None);
        self.row_to_slot.clear();
        self.is_active.fill(false);
        self.base_row_offset = new_base_row_offset;
        self.next_row = new_base_row_offset;
        self.active_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::CutRowMap;

    #[test]
    fn new_creates_empty_map() {
        let map = CutRowMap::new(100, 50);
        assert_eq!(map.base_row_offset(), 50);
        assert_eq!(map.next_row(), 50);
        assert_eq!(map.active_count(), 0);
        assert_eq!(map.total_cut_rows(), 0);
    }

    #[test]
    fn new_zero_capacity_is_valid() {
        let map = CutRowMap::new(0, 10);
        assert_eq!(map.total_cut_rows(), 0);
        assert_eq!(map.active_count(), 0);
    }

    #[test]
    fn insert_assigns_sequential_rows_from_base_offset() {
        let mut map = CutRowMap::new(100, 50);

        let r0 = map.insert(7);
        assert_eq!(r0, 50);

        let r1 = map.insert(3);
        assert_eq!(r1, 51);

        let r2 = map.insert(42);
        assert_eq!(r2, 52);

        assert_eq!(map.next_row(), 53);
        assert_eq!(map.total_cut_rows(), 3);
        assert_eq!(map.active_count(), 3);
    }

    #[test]
    fn insert_updates_both_mappings() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(7);

        assert_eq!(map.lp_row_for_slot(7), Some(50));
        assert_eq!(map.slot_for_lp_row(50), Some(7));
    }

    #[test]
    fn lp_row_for_slot_returns_none_for_unmapped_slot() {
        let map = CutRowMap::new(100, 50);
        assert_eq!(map.lp_row_for_slot(7), None);
    }

    #[test]
    fn lp_row_for_slot_returns_none_for_out_of_range() {
        let map = CutRowMap::new(10, 50);
        assert_eq!(map.lp_row_for_slot(999), None);
    }

    #[test]
    fn slot_for_lp_row_returns_none_for_template_rows() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0);
        // Template rows (below base_row_offset) always return None.
        assert_eq!(map.slot_for_lp_row(0), None);
        assert_eq!(map.slot_for_lp_row(49), None);
    }

    #[test]
    fn slot_for_lp_row_returns_none_for_unmapped_cut_row() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0); // row 50
                       // Row 51 was never assigned.
        assert_eq!(map.slot_for_lp_row(51), None);
    }

    #[test]
    fn deactivate_decrements_active_count() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0);
        map.insert(1);
        assert_eq!(map.active_count(), 2);

        map.deactivate(0);
        assert_eq!(map.active_count(), 1);
        assert_eq!(map.total_cut_rows(), 2); // total unchanged
    }

    #[test]
    fn deactivate_preserves_row_mapping() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0);
        let row = map.lp_row_for_slot(0);

        map.deactivate(0);
        // Row mapping is preserved after deactivation.
        assert_eq!(map.lp_row_for_slot(0), row);
        assert_eq!(map.slot_for_lp_row(row.expect("must be Some")), Some(0));
    }

    #[test]
    fn deactivate_is_idempotent() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0);
        map.deactivate(0);
        assert_eq!(map.active_count(), 0);

        // Second deactivation should not underflow.
        map.deactivate(0);
        assert_eq!(map.active_count(), 0);
    }

    #[test]
    fn is_slot_active_reflects_state() {
        let mut map = CutRowMap::new(100, 50);
        assert!(!map.is_slot_active(0));

        map.insert(0);
        assert!(map.is_slot_active(0));

        map.deactivate(0);
        assert!(!map.is_slot_active(0));
    }

    #[test]
    fn is_slot_active_returns_false_for_out_of_range() {
        let map = CutRowMap::new(10, 50);
        assert!(!map.is_slot_active(999));
    }

    #[test]
    fn reset_clears_all_state() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0);
        map.insert(1);
        map.insert(2);
        map.deactivate(1);

        map.reset(60);
        assert_eq!(map.total_cut_rows(), 0);
        assert_eq!(map.active_count(), 0);
        assert_eq!(map.next_row(), 60);
        assert_eq!(map.base_row_offset(), 60);
        assert_eq!(map.lp_row_for_slot(0), None);
        assert_eq!(map.lp_row_for_slot(1), None);
        assert_eq!(map.lp_row_for_slot(2), None);
        assert!(!map.is_slot_active(0));
    }

    #[test]
    fn reset_allows_reinsertion() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0);
        assert_eq!(map.lp_row_for_slot(0), Some(50));

        map.reset(60);
        let row = map.insert(0);
        assert_eq!(row, 60);
        assert_eq!(map.lp_row_for_slot(0), Some(60));
    }

    #[test]
    fn multiple_inserts_and_deactivations() {
        let mut map = CutRowMap::new(100, 10);

        // Insert slots 5, 10, 15 -> rows 10, 11, 12
        map.insert(5);
        map.insert(10);
        map.insert(15);
        assert_eq!(map.active_count(), 3);
        assert_eq!(map.total_cut_rows(), 3);

        // Deactivate slot 10 (row 11)
        map.deactivate(10);
        assert_eq!(map.active_count(), 2);
        assert_eq!(map.total_cut_rows(), 3);

        // Insert slot 20 -> row 13
        let row = map.insert(20);
        assert_eq!(row, 13);
        assert_eq!(map.active_count(), 3);
        assert_eq!(map.total_cut_rows(), 4);

        // Verify all mappings
        assert_eq!(map.lp_row_for_slot(5), Some(10));
        assert_eq!(map.lp_row_for_slot(10), Some(11)); // deactivated but mapped
        assert_eq!(map.lp_row_for_slot(15), Some(12));
        assert_eq!(map.lp_row_for_slot(20), Some(13));

        assert_eq!(map.slot_for_lp_row(10), Some(5));
        assert_eq!(map.slot_for_lp_row(11), Some(10));
        assert_eq!(map.slot_for_lp_row(12), Some(15));
        assert_eq!(map.slot_for_lp_row(13), Some(20));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "already mapped")]
    fn insert_same_slot_twice_panics_in_debug() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(0);
        map.insert(0); // should panic
    }

    #[test]
    fn cut_row_map_derives_debug_and_clone() {
        let mut map = CutRowMap::new(10, 5);
        map.insert(0);
        map.insert(3);

        let cloned = map.clone();
        assert_eq!(cloned.active_count(), 2);
        assert_eq!(cloned.lp_row_for_slot(0), Some(5));
        assert_eq!(cloned.lp_row_for_slot(3), Some(6));

        let debug_str = format!("{map:?}");
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn base_row_offset_zero_works() {
        let mut map = CutRowMap::new(10, 0);
        let row = map.insert(0);
        assert_eq!(row, 0);
        assert_eq!(map.slot_for_lp_row(0), Some(0));
    }
}
