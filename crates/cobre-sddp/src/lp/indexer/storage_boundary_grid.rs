//! The typed [`StorageBoundaryGrid`] address primitive for the chronological
//! storage-boundary family, single-owning `block_storage_col`'s three-arm
//! formula beside [`BlockGrid`](super::BlockGrid).
//!
//! Every `block_storage_col` copy (`StageLayout`, [`StageGeometry`
//! ](crate::lp::builder::StageGeometry), the generic-constraint resolver) delegates
//! to [`StorageBoundaryGrid::col`] rather than re-deriving the endpoint/interior
//! split — the wrong-but-compiling alternative is a hand-rolled copy of the match
//! whose arm order silently drifts from the others. The [`Boundary`](super::Boundary)
//! operand's exhaustive match (no `_` arm) makes the endpoint/interior split
//! structural rather than order-dependent.

use super::Boundary;

/// Typed storage-boundary address calculator for one SDDP stage LP.
#[derive(Debug, Clone, Copy, Default)]
pub struct StorageBoundaryGrid {
    storage_in_base: usize,
    storage_out_base: usize,
    storage_internal_start: usize,
    n_blks: usize,
}

impl StorageBoundaryGrid {
    /// Construct a [`StorageBoundaryGrid`] from its state bases and stride
    /// constants.
    ///
    /// Source `storage_in_base`/`storage_out_base` from the owning stage's
    /// `StateSpace` (`storage_in.start`/`storage.start`), never a hard-coded
    /// `0`, so the grid cannot disagree with a future region reorder.
    #[inline]
    #[must_use]
    pub fn new(
        storage_in_base: usize,
        storage_out_base: usize,
        storage_internal_start: usize,
        n_blks: usize,
    ) -> Self {
        Self {
            storage_in_base,
            storage_out_base,
            storage_internal_start,
            n_blks,
        }
    }

    /// Storage column at chronological boundary `boundary` for hydro `h`. The
    /// two endpoints are STATE columns — [`Boundary::Incoming`] → `S⁰` and
    /// [`Boundary::Outgoing`] → `Sᴷ` — while [`Boundary::Interior`] is a
    /// CONTROL column (stride `n_blks − 1`, not `n_blks`).
    ///
    /// This match never gains a `_` arm — [`Boundary`]'s own doc-pinned
    /// example proves a catch-all would silently absorb the `Outgoing`
    /// endpoint.
    #[inline]
    #[must_use]
    pub fn col(&self, h: usize, boundary: Boundary) -> usize {
        match boundary {
            Boundary::Incoming => self.storage_in_base + h,
            Boundary::Outgoing => self.storage_out_base + h,
            Boundary::Interior(k) => {
                debug_assert!(
                    (1..self.n_blks).contains(&k),
                    "interior boundary {k} out of 1..{}",
                    self.n_blks
                );
                self.storage_internal_start + h * (self.n_blks - 1) + (k - 1)
            }
        }
    }

    /// The four private fields, in declaration order, for the canonical
    /// byte-encoding snapshot (ADR-041) — the no-`..` destructure fails to
    /// compile the moment a field is added, so the digest cannot silently
    /// drop it.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn canonical_fields(self) -> [usize; 4] {
        let Self {
            storage_in_base,
            storage_out_base,
            storage_internal_start,
            n_blks,
        } = self;
        [
            storage_in_base,
            storage_out_base,
            storage_internal_start,
            n_blks,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::{Boundary, StorageBoundaryGrid};

    #[test]
    fn storage_boundary_grid_resolves_endpoints_and_interior() {
        let grid = StorageBoundaryGrid::new(100, 200, 50, 3);
        let h = 2;

        assert_eq!(grid.col(h, Boundary::Incoming), 100 + h, "S⁰ endpoint");
        assert_eq!(grid.col(h, Boundary::Outgoing), 200 + h, "Sᴷ endpoint");
        assert_eq!(
            grid.col(h, Boundary::Interior(1)),
            50 + h * 2,
            "interior k=1"
        );
        assert_eq!(
            grid.col(h, Boundary::Interior(2)),
            50 + h * 2 + 1,
            "interior k=2"
        );
    }
}
