use crate::hydro_models::{FphaPlane, ResolvedProductionModel};
use crate::indexer::{BlockIdx, FphaCellLocal, FphaLocal, HydroCell, HydroSys};

use super::layout::{StageLayout, TemplateBuildCtx};

/// One `(plant, cell, block, plane)` visit emitted by [`for_each_fpha_plane`],
/// bundled into a `Copy` struct rather than passed as separate closure arguments.
#[derive(Debug, Clone, Copy)]
pub(super) struct FphaVisit {
    /// The plant this cell belongs to; owns the plant-level storage/spillage columns.
    pub(super) plant: HydroSys,
    /// The visited cell, in [`crate::indexer::HydroCellIndex`]'s own indexing.
    pub(super) cell: HydroCell,
    /// The cell's FPHA-cell-local index; resolves its own generation column.
    pub(super) cell_local: FphaCellLocal,
    pub(super) blk: BlockIdx,
    /// LP row for this `(cell, block, plane)` triple.
    pub(super) row: usize,
}

/// Walk every FPHA hyperplane row of the stage, invoking `visit` once per
/// `(FPHA plant, cell, block, plane)` quadruple with the resolved plane and its
/// LP row, in plant -> cell -> block -> plane nesting order.
///
/// The single owner of the FPHA row-cursor arithmetic: both the bounds fill
/// (`rows::fill_fpha_rows`) and the coefficient fill
/// ([`super::entries::fill_fpha_entries`]) drive off this walker, so a one-sided
/// edit that lands the bounds and the coefficients on different rows is impossible.
///
/// The row-block base advances once per CELL, not once per plant: each cell owns
/// its own contiguous `n_blks * n_planes` row block. This nesting is load-bearing
/// for CSC row-sortedness: the plant-level storage/spillage columns
/// ([`super::entries::fill_fpha_entries`]) get one push per cell, and stay
/// row-ascending only because cells are visited in ascending order with
/// `blk`/plane innermost. Nesting `blk` outside `cell` would revisit an earlier
/// cell's row range after a later cell's, tripping `assemble_csc`'s
/// row-sortedness assertion. Matches `StageLayout::row_fpha_start`.
///
/// The closure is a monomorphised `FnMut` borrowing its target buffer (no
/// `Box<dyn>`, no intermediate `Vec`), so the build allocates nothing and the
/// byte-identical `(row, value)` push order is preserved.
pub(super) fn for_each_fpha_plane<F>(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    mut visit: F,
) where
    F: FnMut(FphaVisit, &FphaPlane),
{
    let n_blks = layout.clock.n_blks();
    let grid = layout.block_grid();
    let mut fpha_block_start = layout.geometry.fpha.start;
    for (local_idx, &h) in layout.geometry.fpha_hydro_indices.iter().enumerate() {
        let planes = match ctx.production_models.model(h.get(), stage_idx) {
            ResolvedProductionModel::Fpha { planes, .. } => planes,
            ResolvedProductionModel::ConstantProductivity { .. } => {
                debug_assert!(
                    false,
                    "fpha_hydro_indices contains hydro {} but model is ConstantProductivity",
                    h.get()
                );
                continue;
            }
        };
        let n_planes = planes.len();
        let local_idx = FphaLocal::new(local_idx);
        let cell_base = layout.fpha_cell_local_start[local_idx.get()];
        for (offset, c) in ctx.hydro_cell_index.cells_of(h).enumerate() {
            let cell = HydroCell::new(c);
            let cell_local = FphaCellLocal::new(cell_base + offset);
            for blk in (0..n_blks).map(BlockIdx::new) {
                for (p_idx, plane) in planes.iter().enumerate() {
                    let row = grid.fpha_plane(fpha_block_start, blk, p_idx, n_planes);
                    visit(
                        FphaVisit {
                            plant: h,
                            cell,
                            cell_local,
                            blk,
                            row,
                        },
                        plane,
                    );
                }
            }
            fpha_block_start = grid.advance_fpha_base(fpha_block_start, n_planes);
        }
    }
}
