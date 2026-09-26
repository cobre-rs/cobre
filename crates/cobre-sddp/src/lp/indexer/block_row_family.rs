//! The typed [`BlockRowFamily`] address primitive for a block-major row
//! family: one row per entity, collapsing the block, or one row per
//! `(entity, block)` pair, strided by the owner's `n_blks`.
//!
//! The family stores no block count of its own — every [`Self::row`] call
//! takes the owner's `n_blks` as an argument, so the stride lives in exactly
//! one place, the family's owner (e.g.
//! [`StageGeometry`](crate::lp::builder::StageGeometry)'s `n_blks`).

use std::ops::Range;

use super::BlockIdx;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum RowsPerEntity {
    #[default]
    One,
    PerBlock,
}

/// Typed block-major row-family address calculator for one SDDP stage LP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockRowFamily {
    start: usize,
    end: usize,
    rows_per_entity: RowsPerEntity,
}

impl BlockRowFamily {
    /// A family with one row per entity; every block reads the same row.
    #[must_use]
    pub fn one_per_entity(rows: Range<usize>) -> Self {
        Self {
            start: rows.start,
            end: rows.end,
            rows_per_entity: RowsPerEntity::One,
        }
    }

    /// A family with one row per `(entity, block)` pair, block-major.
    #[must_use]
    pub fn per_block(rows: Range<usize>) -> Self {
        Self {
            start: rows.start,
            end: rows.end,
            rows_per_entity: RowsPerEntity::PerBlock,
        }
    }

    /// `entity`'s row for `blk`, striding by the owner's `n_blks`.
    pub(crate) fn row(self, entity: usize, blk: BlockIdx, n_blks: usize) -> usize {
        let (stride, blk_offset) = match self.rows_per_entity {
            RowsPerEntity::One => (1, 0),
            RowsPerEntity::PerBlock => {
                debug_assert!(blk.get() < n_blks, "block {} out of 0..{n_blks}", blk.get());
                (n_blks, blk.get())
            }
        };
        let row = self.start + entity * stride + blk_offset;
        debug_assert!(
            row < self.end,
            "row {row} out of {}..{}",
            self.start,
            self.end
        );
        row
    }

    /// The family's first row.
    #[inline]
    #[must_use]
    pub fn start(self) -> usize {
        self.start
    }

    /// The family's one-past-last row.
    #[inline]
    #[must_use]
    pub fn end(self) -> usize {
        self.end
    }

    /// The family's row range.
    #[inline]
    #[must_use]
    pub fn range(self) -> Range<usize> {
        self.start..self.end
    }

    /// Rows per entity: `1` for [`RowsPerEntity::One`], `n_blks` for
    /// [`RowsPerEntity::PerBlock`].
    #[inline]
    #[must_use]
    pub fn rows_per_entity(self, n_blks: usize) -> usize {
        match self.rows_per_entity {
            RowsPerEntity::One => 1,
            RowsPerEntity::PerBlock => n_blks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BlockRowFamily;
    use crate::indexer::BlockIdx;

    #[test]
    fn one_row_family_ignores_the_block() {
        let family = BlockRowFamily::one_per_entity(10..13);
        for k in 0..4 {
            assert_eq!(family.row(1, BlockIdx::new(k), 4), 11);
        }
    }

    #[test]
    fn per_block_family_strides_entities_by_the_block_count() {
        let family = BlockRowFamily::per_block(10..22);
        assert_eq!(family.row(2, BlockIdx::new(1), 4), 10 + 2 * 4 + 1);
    }

    #[test]
    fn family_reports_its_end_and_row_count() {
        let one = BlockRowFamily::one_per_entity(10..13);
        assert_eq!(one.start(), 10);
        assert_eq!(one.end(), 13);
        assert_eq!(one.range(), 10..13);
        assert_eq!(one.rows_per_entity(4), 1);

        let per_block = BlockRowFamily::per_block(10..22);
        assert_eq!(per_block.end(), 22);
        assert_eq!(per_block.rows_per_entity(4), 4);
    }

    #[test]
    fn default_family_is_empty() {
        let family = BlockRowFamily::default();
        assert_eq!(family.range(), 0..0);
        assert_eq!(family.rows_per_entity(4), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "block")]
    fn per_block_family_rejects_an_out_of_range_block() {
        let family = BlockRowFamily::per_block(10..22);
        family.row(0, BlockIdx::new(4), 4);
    }
}
