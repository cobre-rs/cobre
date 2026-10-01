//! The [`RangeCursor`] running column/row offset allocator shared by
//! [`StageLayout::new`](crate::lp::builder::StageLayout)'s per-stage equipment
//! column/row chains and [`StateSpace::build`](super::StateSpace)'s
//! stage-invariant state-vector chain.

use std::ops::Range;

/// Running column/row offset allocator. [`Self::alloc`] returns `pos..pos + len`
/// and advances the cursor by `len`.
///
/// `alloc(0)` returns `pos..pos`, never `0..0` — empty ranges must preserve their
/// position for empty-block fields and `n_h == 0` fallbacks.
pub(crate) struct RangeCursor {
    pos: usize,
}

impl RangeCursor {
    pub(crate) fn new(start: usize) -> Self {
        Self { pos: start }
    }

    pub(crate) fn alloc(&mut self, len: usize) -> Range<usize> {
        let start = self.pos;
        self.pos += len;
        start..self.pos
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }
}
