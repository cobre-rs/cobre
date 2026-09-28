//! The typed [`BlockGrid`] address primitive shared by all block-stride LP fills.
//!
//! The block-stride operation appears in three nesting shapes across the LP
//! build, patch, resolver, and extraction paths. [`BlockGrid`] is the single
//! owner of every production block-major stride for these shapes
//! ([`flat`](BlockGrid::flat), [`fpha_plane`](BlockGrid::fpha_plane),
//! [`deficit`](BlockGrid::deficit)): every production site addresses through one
//! of them, never open-coding `start + entity * n_blks + blk`. The
//! wrong-but-compiling alternative is a hand-rolled stride that drifts from the
//! grid — a transposed nesting, or a stride read from a different `n_blks` than
//! the LP was built with.
//!
//! Each shape gets its own method with distinct parameter names rather than one
//! generic `(a, b, c)` calculator: because the shapes nest in opposite orders
//! (flat: entity OUTER / block INNER; FPHA-plane: block OUTER / plane INNER), a
//! shared method would let a caller pass one shape's arguments in another's order
//! and still compile, silently addressing the wrong cell. No method can express
//! the transpose (`blk * n_entities + entity`) — the entity count it needs is not
//! carried. Pinned by `block_grid_forbids_transposed_shape`.
//!
//! Two open-coded site classes are NOT violations: the `#[cfg(test)]`
//! differential-oracle tests that compute the address by hand to verify a
//! `BlockGrid`-routed accessor, and doc comments mirroring the layout formula.

use super::BlockIdx;

/// Typed block-stride address calculator for one SDDP stage LP.
///
/// A cheap `Copy` value carrying the two stage constants (`n_blks`,
/// `max_deficit_segments`) the three shapes need beyond their per-call arguments.
///
/// `pub` because the public
/// [`PatchBuffer::fill_load_patches`](super::super::builder::PatchBuffer) accepts
/// it by value; narrowing to `pub(crate)` would break that API.
#[derive(Debug, Clone, Copy)]
pub struct BlockGrid {
    /// Operating blocks per stage (K).
    n_blks: usize,
    /// Maximum deficit segments across all buses (S).
    max_deficit_segments: usize,
}

impl BlockGrid {
    /// Construct a [`BlockGrid`] from its two stride constants.
    ///
    /// Source `n_blks` from the per-stage block count the LP was built with
    /// (`StageLayout` / [`StageGeometry::n_blks`](crate::lp::builder::StageGeometry::n_blks)),
    /// never a study-global value, so the grid cannot disagree with the LP it
    /// addresses.
    #[inline]
    #[must_use]
    pub fn new(n_blks: usize, max_deficit_segments: usize) -> Self {
        Self {
            n_blks,
            max_deficit_segments,
        }
    }

    /// The per-stage block count `n_blks` this grid strides by.
    #[inline]
    #[must_use]
    pub fn n_blks(&self) -> usize {
        self.n_blks
    }

    /// Flat block-major address: `start + entity * n_blks + blk`.
    ///
    /// Entity OUTER (stride `n_blks`), block INNER. A swapped
    /// `flat(start, blk, entity)` call passing a bare `usize` in the block
    /// slot fails to compile — the block operand requires [`BlockIdx`]:
    ///
    /// ```compile_fail
    /// use cobre_sddp::indexer::BlockGrid;
    ///
    /// BlockGrid::new(2, 1).flat(0, 1, 2usize); // block operand requires BlockIdx, not usize
    /// ```
    #[inline]
    #[must_use]
    pub fn flat(&self, start: usize, entity: usize, blk: BlockIdx) -> usize {
        let blk = blk.get();
        start + entity * self.n_blks + blk
    }

    /// FPHA-plane address: `fpha_block_start + blk * n_planes + p_idx`.
    ///
    /// Block OUTER (stride `n_planes`), plane INNER — the OPPOSITE nesting of
    /// [`flat`](Self::flat). Advance the base with
    /// [`advance_fpha_base`](Self::advance_fpha_base) after each cell.
    #[inline]
    #[must_use]
    pub fn fpha_plane(
        &self,
        fpha_block_start: usize,
        blk: BlockIdx,
        p_idx: usize,
        n_planes: usize,
    ) -> usize {
        let blk = blk.get();
        fpha_block_start + blk * n_planes + p_idx
    }

    /// Advance the FPHA row-block base past one cell's `n_blks * n_planes` rows
    /// (`n_planes` is caller-supplied, since plane counts vary per hydro).
    #[inline]
    #[must_use]
    pub fn advance_fpha_base(&self, fpha_block_start: usize, n_planes: usize) -> usize {
        fpha_block_start + self.n_blks * n_planes
    }

    /// Deficit 3-term address:
    /// `deficit_start + b_pos * S * n_blks + seg * n_blks + blk`.
    ///
    /// Bus `b_pos` OUTER (stride `S * n_blks`, `S = max_deficit_segments`),
    /// segment `seg` MIDDLE (stride `n_blks`), block `blk` INNER.
    #[inline]
    #[must_use]
    pub fn deficit(&self, deficit_start: usize, b_pos: usize, seg: usize, blk: BlockIdx) -> usize {
        let blk = blk.get();
        deficit_start + b_pos * self.max_deficit_segments * self.n_blks + seg * self.n_blks + blk
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockGrid, BlockIdx};

    #[test]
    fn flat_block_major_address() {
        let grid = BlockGrid::new(3, 1);
        assert_eq!(grid.flat(9, 1, BlockIdx::new(2)), 14);
    }

    #[test]
    fn fpha_plane_address() {
        let grid = BlockGrid::new(3, 1);
        assert_eq!(grid.fpha_plane(100, BlockIdx::new(1), 2, 5), 107);
    }

    #[test]
    fn fpha_base_advance() {
        let grid = BlockGrid::new(3, 1);

        assert_eq!(grid.advance_fpha_base(100, 5), 115);
    }

    #[test]
    fn deficit_three_term_address() {
        let grid = BlockGrid::new(3, 2);
        assert_eq!(grid.deficit(61, 0, 1, BlockIdx::new(0)), 64);
    }

    #[test]
    fn block_grid_forbids_transposed_shape() {
        let grid = BlockGrid::new(3, 2);
        let (n_entities, entity, blk) = (4, 1, 2);
        let correct_flat = grid.flat(0, entity, BlockIdx::new(blk));
        let transposed_flat = blk * n_entities + entity; // NOT expressible via BlockGrid
        assert_eq!(correct_flat, 5);
        assert_ne!(correct_flat, transposed_flat);

        let grid = BlockGrid::new(2, 2);
        let (blk, p_idx, n_planes) = (1, 3, 5);
        let correct_fpha = grid.fpha_plane(0, BlockIdx::new(blk), p_idx, n_planes);
        let transposed_fpha = p_idx * grid.n_blks + blk; // wrong nesting
        assert_eq!(correct_fpha, 8);
        assert_ne!(correct_fpha, transposed_fpha);

        let grid = BlockGrid::new(3, 2);
        let (b_pos, seg, blk) = (1, 1, 0);
        let correct_def = grid.deficit(0, b_pos, seg, BlockIdx::new(blk));
        let transposed_def =
            b_pos * grid.max_deficit_segments * grid.n_blks + seg * grid.max_deficit_segments + blk; // wrong segment stride
        assert_eq!(correct_def, 9);
        assert_ne!(correct_def, transposed_def);
    }
}
