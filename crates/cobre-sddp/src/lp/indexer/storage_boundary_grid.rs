//! The typed [`StorageBoundaryGrid`] address primitive for the chronological
//! storage-boundary family, single-owning `block_storage_col`'s three-arm
//! formula beside [`BlockGrid`](super::BlockGrid).
//!
//! Every `block_storage_col` copy (`StageLayout`, [`StageGeometry`
//! ](crate::lp::builder::StageGeometry), the generic-constraint resolver) delegates
//! to [`StorageBoundaryGrid::col`] rather than re-deriving the endpoint/interior
//! split — the wrong-but-compiling alternative is a hand-rolled copy of the match
//! whose arm order silently drifts from the others. The [`Boundary`]
//! operand's exhaustive match (no `_` arm) makes the endpoint/interior split
//! structural rather than order-dependent.

use super::{Boundary, HydroSys, StateSpace};

/// Typed storage-boundary address calculator for one SDDP stage LP.
#[derive(Debug, Clone, Copy, Default)]
pub struct StorageBoundaryGrid {
    storage_internal_start: usize,
    n_blks: usize,
}

impl StorageBoundaryGrid {
    /// Construct a [`StorageBoundaryGrid`] from its interior anchor and
    /// block-count stride.
    #[inline]
    #[must_use]
    pub fn new(storage_internal_start: usize, n_blks: usize) -> Self {
        Self {
            storage_internal_start,
            n_blks,
        }
    }

    /// Storage column at chronological boundary `boundary` for hydro `h`. The
    /// two endpoints are STATE columns, resolved through `state`'s own
    /// [`StateSpace::storage_incoming_col`]/[`StateSpace::storage_outgoing_col`]
    /// — [`Boundary::Incoming`] → `S⁰` and [`Boundary::Outgoing`] → `Sᴷ` —
    /// while [`Boundary::Interior`] is a CONTROL column (stride `n_blks − 1`,
    /// not `n_blks`) owned by this grid alone.
    ///
    /// This match never gains a `_` arm — [`Boundary`]'s own doc-pinned
    /// example proves a catch-all would silently absorb the `Outgoing`
    /// endpoint.
    #[inline]
    #[must_use]
    pub fn col(&self, state: &StateSpace, h: HydroSys, boundary: Boundary) -> usize {
        match boundary {
            Boundary::Incoming => state.storage_incoming_col(h).get(),
            Boundary::Outgoing => state.storage_outgoing_col(h).get(),
            Boundary::Interior(k) => {
                debug_assert!(
                    (1..self.n_blks).contains(&k),
                    "interior boundary {k} out of 1..{}",
                    self.n_blks
                );
                self.storage_internal_start + h.get() * (self.n_blks - 1) + (k - 1)
            }
        }
    }

    /// The two private fields, in declaration order, for the canonical
    /// byte-encoding snapshot — the no-`..` destructure fails to
    /// compile the moment a field is added, so the digest cannot silently
    /// drop it.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn canonical_fields(self) -> [usize; 2] {
        let Self {
            storage_internal_start,
            n_blks,
        } = self;
        [storage_internal_start, n_blks]
    }
}

#[cfg(test)]
mod tests {
    use super::{Boundary, HydroSys, StateSpace, StorageBoundaryGrid};
    use crate::lead_time::AnticipatedResolution;

    #[test]
    fn storage_boundary_grid_resolves_endpoints_and_interior() {
        let state = StateSpace::new(
            4,
            0,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &[0, 0, 0, 0],
        );
        let grid = StorageBoundaryGrid::new(50, 3);

        for h in 0..state.hydro_count {
            let hsys = HydroSys::new(h);
            assert_eq!(
                grid.col(&state, hsys, Boundary::Incoming),
                state.storage_in.start + h,
                "S⁰ endpoint"
            );
            assert_eq!(
                grid.col(&state, hsys, Boundary::Outgoing),
                state.storage.start + h,
                "Sᴷ endpoint"
            );
            assert_eq!(
                grid.col(&state, hsys, Boundary::Interior(1)),
                50 + h * 2,
                "interior k=1"
            );
            assert_eq!(
                grid.col(&state, hsys, Boundary::Interior(2)),
                50 + h * 2 + 1,
                "interior k=2"
            );
        }
    }
}
