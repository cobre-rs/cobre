//! Bidirectional mapping between [`CutPool`] slot indices and LP row indices.
//!
//! [`CutRowMap`] is used by the append-only lower-bound LP to skip cuts that
//! are already present as LP rows. Supports O(1) slot-to-row lookup.
//!
//! ## Row index stability
//!
//! `HiGHS` assigns row indices sequentially when `add_rows` is called: if the
//! LP has R rows and N rows are appended, the new rows get indices
//! R, R+1, ..., R+N-1. Row indices are stable across subsequent `add_rows`
//! calls. [`CutRowMap`] relies on this property.
//!
//! ## Append-only semantics
//!
//! Cuts are added to the LB LP via [`CutRowMap::insert`] and never removed.
//! This keeps the lower bound monotonically non-decreasing across iterations.
//! Cut selection (strategy-based or budget-based) runs on the shared cut pool
//! and does not affect the LB LP; pool-deactivated cuts remain as LP rows in
//! the LB solver.
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
//! ```

/// Slot-to-row map for the append-only lower-bound LP.
///
/// Supports O(1) slot-to-row lookup. All storage is pre-allocated at
/// construction time.
#[derive(Debug, Clone)]
pub struct CutRowMap {
    /// `slot_to_row[slot] = Some(lp_row)` when the cut at `slot` has been
    /// inserted into the LP, or `None` otherwise.
    slot_to_row: Vec<Option<usize>>,

    /// The LP row index where cut rows begin. Equal to the number of template
    /// (structural) rows in the LP.
    base_row_offset: usize,

    /// The next LP row index to assign when a new cut is appended.
    next_row: usize,
}

impl CutRowMap {
    /// Create a new `CutRowMap`.
    ///
    /// # Parameters
    ///
    /// - `pool_capacity`: total number of slots in the [`CutPool`](super::pool::CutPool).
    ///   Determines the size of the slot-indexed vector.
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
    /// assert_eq!(map.total_cut_rows(), 0);
    /// ```
    #[must_use]
    pub fn new(pool_capacity: usize, base_row_offset: usize) -> Self {
        Self {
            slot_to_row: vec![None; pool_capacity],
            base_row_offset,
            next_row: base_row_offset,
        }
    }

    /// Record that the cut at `slot` has been appended to the LP.
    ///
    /// Assigns the next available LP row index and returns it.
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
        self.next_row += 1;
        lp_row
    }

    /// Look up the LP row index for a [`CutPool`](super::pool::CutPool) slot.
    ///
    /// Returns `Some(lp_row)` when the cut at `slot` has been inserted into
    /// the LP, or `None` when the slot has never been added.
    #[must_use]
    #[inline]
    pub fn lp_row_for_slot(&self, slot: usize) -> Option<usize> {
        self.slot_to_row.get(slot).copied().flatten()
    }

    /// Return the number of LP rows currently assigned to cuts.
    #[must_use]
    #[inline]
    pub fn total_cut_rows(&self) -> usize {
        self.next_row - self.base_row_offset
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
}

#[cfg(test)]
mod tests {
    use super::CutRowMap;

    #[test]
    fn new_creates_empty_map() {
        let map = CutRowMap::new(100, 50);
        assert_eq!(map.base_row_offset(), 50);
        assert_eq!(map.next_row(), 50);
        assert_eq!(map.total_cut_rows(), 0);
    }

    #[test]
    fn new_zero_capacity_is_valid() {
        let map = CutRowMap::new(0, 10);
        assert_eq!(map.total_cut_rows(), 0);
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
    }

    #[test]
    fn insert_records_slot_to_row_mapping() {
        let mut map = CutRowMap::new(100, 50);
        map.insert(7);

        assert_eq!(map.lp_row_for_slot(7), Some(50));
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
    fn multiple_inserts_preserve_mappings() {
        let mut map = CutRowMap::new(100, 10);

        map.insert(5); // row 10
        map.insert(10); // row 11
        map.insert(15); // row 12
        assert_eq!(map.total_cut_rows(), 3);

        let row = map.insert(20);
        assert_eq!(row, 13);
        assert_eq!(map.total_cut_rows(), 4);

        assert_eq!(map.lp_row_for_slot(5), Some(10));
        assert_eq!(map.lp_row_for_slot(10), Some(11));
        assert_eq!(map.lp_row_for_slot(15), Some(12));
        assert_eq!(map.lp_row_for_slot(20), Some(13));
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
        assert_eq!(cloned.total_cut_rows(), 2);
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
    }
}
