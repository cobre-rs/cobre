//! Slot-tracked basis reconstruction for cut-set-aware warm-start (Strategy S3+).
//!
//! This module provides [`reconstruct_basis`] — the slot-identity-based helper that
//! correctly handles cut-set churn (drops, reorders, adds) between iterations.
//!
//! ## Why slot identity matters
//!
//! The legacy `pad_basis_for_cuts` helper used the LP row count as the only
//! reconciliation key.  If cut selection replaced one cut with another of equal
//! count, the two bases were the same length but positionally misaligned — `HiGHS`
//! received a basis with mismatched row statuses and crashed back to cold start
//! (or, worse, warm-started with a corrupt basis).
//!
//! `reconstruct_basis` takes [`CapturedBasis::cut_row_slots`] as its key: each
//! stored cut row carries the [`CutPool`](crate::cut::pool::CutPool) slot that
//! generated it.  The reconstruction walks the current LP's cut rows in order,
//! looks each slot up in an O(1) scratch map, and either copies the stored status
//! (slot found → preserved) or evaluates the cut at `padding_state` to assign
//! tight/slack (slot not found → new).
//!
//! ## Forward-path basic-count invariant (ticket-009)
//!
//! On the forward path, cut selection may drop cuts whose stored row status was
//! `HIGHS_BASIS_STATUS_LOWER`.  Each such LOWER drop reduces the running basic
//! count by zero (LOWER is non-basic), but each preserved BASIC drop reduces it
//! by 1.  `reconstruct_basis` assigns BASIC unconditionally to new cuts, so new
//! cuts never cause a deficit.  However, if `excess = col_basic + row_basic -
//! num_row > 0` after reconstruction, it means dropped cuts were all BASIC, and
//! the reconstructed basis has too many BASIC statuses.
//!
//! [`enforce_basic_count_invariant`] corrects this by scanning from the end of
//! `out.row_status` (the cut rows, indices `>= base_row_count`) and demoting
//! trailing `HIGHS_BASIS_STATUS_BASIC` entries to `HIGHS_BASIS_STATUS_LOWER`
//! until `col_basic + row_basic == num_row` exactly.  The demotion count equals
//! `excess = dropped_lower >= 0` and is always non-negative (never a deficit).
//!
//! This pass is applied **only on the forward path**; the backward path produces
//! `delta == 0` by construction (ticket-008) and does not need it.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::basis_reconstruct::{
//!     PromotionScratch, PaddingContext, ReconstructionSource, ReconstructionStats,
//!     ReconstructionTarget, reconstruct_basis,
//! };
//! use cobre_sddp::workspace::CapturedBasis;
//! use cobre_solver::Basis;
//!
//! let stored = CapturedBasis::new(4, 3, 3, 0, 0); // empty — shim state
//! let source = ReconstructionSource {
//!     target: ReconstructionTarget { base_row_count: 3, num_cols: 4 },
//!     cut_metadata: &[],
//! };
//! let mut out = Basis::new(0, 0);
//! let mut lookup: Vec<Option<u32>> = vec![None; 16];
//! let mut promotion_scratch = PromotionScratch::default();
//! let cuts: Vec<(usize, f64, Vec<f64>)> = vec![];
//! let padding = PaddingContext { state: &[], theta: 0.0, tolerance: 1e-7 };
//! let stats = reconstruct_basis(
//!     &stored,
//!     source,
//!     cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
//!     padding,
//!     &mut out,
//!     &mut lookup,
//!     &mut promotion_scratch,
//! );
//! assert_eq!(stats, ReconstructionStats::default());
//! ```

pub use cobre_solver::ffi::{HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER};

use cobre_solver::Basis;

use crate::cut_selection::CutMetadata;
use crate::workspace::CapturedBasis;

// ---------------------------------------------------------------------------
// Activity-classifier constants (Epic 06 AD-2)
// ---------------------------------------------------------------------------

/// Window size for the activity-driven new-cut classifier (Epic 06 AD-7).
///
/// The classifier looks at the low `RECENT_WINDOW_K` bits of
/// `CutMetadata::active_window`. A cut that was binding in any of the last
/// `RECENT_WINDOW_K` iterations has `active_window & RECENT_WINDOW_BITS != 0`
/// and is classified LOWER (tight guess). Cuts with a zero mask are classified
/// BASIC (slack guess, the safe default).
pub const RECENT_WINDOW_K: u32 = 5;

/// Mask for the low [`RECENT_WINDOW_K`] bits (Epic 06 AD-2).
///
/// Derived from [`RECENT_WINDOW_K`]: `(1 << RECENT_WINDOW_K) - 1 = 0b11111`.
pub const RECENT_WINDOW_BITS: u32 = (1u32 << RECENT_WINDOW_K) - 1;

// ---------------------------------------------------------------------------
// Target LP shape (absorbs two scalar parameters to stay within clippy budget)
// ---------------------------------------------------------------------------

/// Dimensions of the target LP that `reconstruct_basis` populates a basis for.
///
/// Grouping these two related scalars keeps `reconstruct_basis` within the
/// clippy `too_many_arguments` threshold while making the relationship between
/// template rows and column count explicit at call sites.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReconstructionTarget {
    /// Number of template (non-cut) rows in the target LP.
    pub base_row_count: usize,
    /// Total column count of the target LP.
    pub num_cols: usize,
}

// ---------------------------------------------------------------------------
// Source (target + cut metadata grouped to stay within clippy arg budget)
// ---------------------------------------------------------------------------

/// Target LP dimensions plus the cut-activity metadata slice.
///
/// Grouping [`ReconstructionTarget`] and the `CutMetadata` slice into a single
/// struct keeps [`reconstruct_basis`] within the `clippy::too_many_arguments`
/// threshold (workspace-level `deny`) while making the relationship between the
/// LP shape and the per-cut activity data explicit at call sites.
///
/// The `cut_metadata` slice is indexed by **cut pool slot**. It is safe to pass
/// `&[]` when no activity metadata is available; the classifier falls back to
/// `active_window = 0` (BASIC) for every slot beyond the slice bounds.
#[derive(Clone, Copy, Debug)]
pub struct ReconstructionSource<'a> {
    /// Dimensions of the target LP.
    pub target: ReconstructionTarget,
    /// Per-slot activity metadata for the activity-driven classifier.
    ///
    /// Indexed by cut pool slot. Use `&[]` when the caller does not have
    /// activity data (e.g. simulation, or tests with empty pools).
    pub cut_metadata: &'a [CutMetadata],
}

// ---------------------------------------------------------------------------
// Padding context (state, theta, tolerance grouped to keep arg count <= 7)
// ---------------------------------------------------------------------------

/// Inputs for evaluating new cuts (those not present in the stored basis).
///
/// Grouped together so [`reconstruct_basis`] stays under clippy's
/// `too_many_arguments` threshold while making the relationship between the
/// state vector, the θ proxy, and the slack tolerance explicit at call sites.
#[derive(Clone, Copy, Debug)]
pub struct PaddingContext<'a> {
    /// State vector at which to evaluate newly-added cuts.
    /// Forward path: `ws.current_state[..n_state]`.
    /// Backward path: `stored.state_at_capture` (the fix from ticket 004).
    pub state: &'a [f64],
    /// θ proxy used for the tight/slack decision.
    pub theta: f64,
    /// Slack threshold; cuts with `theta - cut_value <= tolerance` get
    /// `NONBASIC_LOWER`.
    pub tolerance: f64,
}

// ---------------------------------------------------------------------------
// Promotion scratch (grouped to keep reconstruct_basis arg count <= 7)
// ---------------------------------------------------------------------------

/// Scratch buffers for the Scheme 1 symmetric promotion and Scheme 2 tail
/// fallback inside [`reconstruct_basis`].
///
/// **Scheme 1 symmetric promotion** (Epic 06 AD-3, corrected by T3a): for each
/// new cut classified `LOWER` by the activity-driven classifier, one
/// preserved-`LOWER` cut-row is promoted to `BASIC`.  The two cuts swap
/// roles — the new cut takes the "active/binding" `LOWER` slot; the stale
/// preserved cut takes the "non-binding slack" `BASIC` slot.  Net change to
/// `col_basic + row_basic`: zero.  Invariant preserved by construction.
///
/// Grouped into a single struct so [`reconstruct_basis`] stays within the
/// workspace-level `clippy::too_many_arguments` deny threshold (7 args max).
///
/// Pre-allocate with [`PromotionScratch::with_capacity`] at workspace
/// construction time and pass `&mut promotion_scratch` on every call.  The
/// function clears both vecs at entry, so stale data from a previous call
/// is never visible.
///
/// Use [`PromotionScratch::default()`] in tests and one-off callers where
/// allocation cost is acceptable.
///
/// (Renamed from the previous Scheme 1 scratch type in T3a — see
/// `ticket-003a-fix-scheme-1-direction.md`.)
#[derive(Debug, Default)]
pub struct PromotionScratch {
    /// Preserved-LOWER cut rows that are candidates for Scheme 1 symmetric
    /// promotion (LOWER → BASIC).  Each entry is `(out_row_index, popcount)`.
    /// Sorted by popcount ascending after the consumer loop so the least-active
    /// candidates are promoted first (safest to move into the basis as BASIC).
    pub candidates: Vec<(usize, u32)>,
    /// Output row indices of new cuts classified as LOWER by the activity-
    /// driven classifier (Epic 06 AD-2).  Used by the Scheme 2 tail fallback
    /// to override the most-recently-classified LOWER cuts back to BASIC when
    /// the Scheme 1 preserved-LOWER pool is exhausted.
    pub new_lower_indices: Vec<usize>,
}

impl PromotionScratch {
    /// Allocate both scratch vecs with the given initial capacity.
    ///
    /// Pass `initial_pool_capacity` (the `CutPool` initial size) so that the
    /// hot path avoids reallocation for typical pool sizes.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(cap),
            new_lower_indices: Vec::with_capacity(cap),
        }
    }
}

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

/// Counters returned by [`reconstruct_basis`].
///
/// The invariant `preserved + new_tight + new_slack` equals the number of
/// elements the iterator passed to `reconstruct_basis` yielded (i.e. the
/// cut-row count of the target LP).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReconstructionStats {
    /// Cut rows whose slot was found in the stored basis and whose status was
    /// copied directly.
    pub preserved: u32,
    /// Cut rows classified as LOWER because their
    /// `active_window & RECENT_WINDOW_BITS != 0` (Epic 06 AD-2).
    ///
    /// These are new cuts (slot not in stored basis) for which the sliding
    /// activity bitmap shows binding in at least one of the last
    /// [`RECENT_WINDOW_K`] iterations. The simplex may pivot them back to BASIC
    /// if the guess was wrong; a post-solve telemetry delta (`basis_offered` vs
    /// `basis_consistency_failures`) surfaces this.
    pub new_tight: u32,
    /// New cut rows evaluated as slack at `padding_state`
    /// (`theta - cut_value > tolerance`).
    pub new_slack: u32,
}

// ---------------------------------------------------------------------------
// reconstruct_basis
// ---------------------------------------------------------------------------

/// Reconstruct a full [`Basis`] for the target LP using slot identity.
///
/// ## Parameters
///
/// - `stored` — read-only stored metadata from the previous iteration.
/// - `source` — target LP dimensions and per-slot activity metadata.
///   Pass `cut_metadata: &[]` when no activity data is available; the
///   classifier falls back to `active_window = 0` (BASIC) for all slots.
/// - `current_cut_rows` — iterator of `(slot, intercept, coefficients)` in
///   target LP row order.  The state dimension is inferred from
///   `padding.state.len()`.
/// - `padding` — state, θ proxy, and tolerance for evaluating new cut rows.
///   Forward path: `padding.state = ws.current_state[..n_state]`.
///   Backward path: `padding.state = captured.state_at_capture` (the fix from
///   ticket 004 — preserved rows were captured at that state so new rows must
///   be evaluated at the same state for a consistent basis).
/// - `out` — destination basis (caller owns; cleared and refilled in place).
/// - `slot_lookup` — scratch `Vec<Option<u32>>` pre-sized by the caller to
///   at least `max_slot + 1`.  Grown in place if undersized (hot path should
///   avoid this via `ScratchBuffers::recon_slot_lookup`).
/// - `promotion_scratch` — scratch buffers for Scheme 1 symmetric promotion
///   and Scheme 2 tail fallback. Both vecs are cleared at function entry;
///   stale data from a previous call is never visible. Pre-sized by the
///   caller via `ScratchBuffers::promotion_scratch`.
///
/// ## Returns
///
/// [`ReconstructionStats`] with `preserved + new_tight + new_slack` equal to
/// the number of items yielded by the iterator.
///
/// ## Allocation contract
///
/// Allocation-free on the hot path when `slot_lookup.len() >= max_slot + 1`
/// and `promotion_scratch` has sufficient capacity.
/// The growth path triggers a `debug_assert!(false)` to surface caller
/// under-sizing, but does not panic in release.
#[allow(clippy::too_many_lines)]
pub fn reconstruct_basis<'a, I>(
    stored: &CapturedBasis,
    source: ReconstructionSource<'_>,
    current_cut_rows: I,
    padding: PaddingContext<'_>,
    out: &mut Basis,
    slot_lookup: &mut Vec<Option<u32>>,
    promotion_scratch: &mut PromotionScratch,
) -> ReconstructionStats
where
    I: Iterator<Item = (usize, f64, &'a [f64])>,
{
    let target = source.target;
    let cut_metadata = source.cut_metadata;

    // Clear the promotion scratch at function entry.
    // CRITICAL: both vecs are reused across stages; a stale list from a
    // previous stage would contaminate the current stage's promotion.
    promotion_scratch.candidates.clear();
    promotion_scratch.new_lower_indices.clear();

    debug_assert!(
        padding.state.len() == stored.state_at_capture.len() || stored.state_at_capture.is_empty(),
        "padding.state.len() {} != stored.state_at_capture.len() {}",
        padding.state.len(),
        stored.state_at_capture.len(),
    );
    if !stored.cut_row_slots.is_empty() {
        debug_assert!(
            stored.basis.row_status.len() == stored.base_row_count + stored.cut_row_slots.len(),
            "CapturedBasis invariant violated: row_status.len() {} != base_row_count {} + \
             cut_row_slots.len() {}",
            stored.basis.row_status.len(),
            stored.base_row_count,
            stored.cut_row_slots.len(),
        );
    }

    let reconcilable_slots = stored.cut_row_slots.as_slice();

    // (a) Column statuses — copy stored, then resize to target if lengths differ.
    out.col_status.clear();
    out.col_status.extend_from_slice(&stored.basis.col_status);
    if out.col_status.len() != target.num_cols {
        out.col_status
            .resize(target.num_cols, HIGHS_BASIS_STATUS_BASIC);
    }

    // (b) Template row statuses — copy the first `target.base_row_count` entries.
    out.row_status.clear();
    if stored.basis.row_status.len() >= target.base_row_count {
        out.row_status
            .extend_from_slice(&stored.basis.row_status[..target.base_row_count]);
    } else {
        // Stored basis has fewer template rows than the target — fill missing with BASIC.
        out.row_status.extend_from_slice(&stored.basis.row_status);
        out.row_status
            .resize(target.base_row_count, HIGHS_BASIS_STATUS_BASIC);
    }

    // (c) Build slot → reconcilable-position lookup.
    //
    // Grow the scratch if it is too small for the current stored slots.  In
    // normal operation the caller pre-sizes this to `initial_pool_capacity`
    // (via `ScratchBuffers`), so growth only occurs on cold paths.  When
    // reconcilable_slots is empty there is nothing to look up — skip the
    // size check (an empty lookup is correct in that case).
    if let Some(max_slot_val) = reconcilable_slots.iter().copied().max() {
        let max_slot = max_slot_val as usize;
        if slot_lookup.len() <= max_slot {
            debug_assert!(
                false,
                "slot_lookup undersized ({} <= max_slot {}); caller must pre-size to \
                 initial_pool_capacity",
                slot_lookup.len(),
                max_slot,
            );
            // Defensive growth in release so the function remains safe.
            slot_lookup.resize(max_slot + 1, None);
        }
    }
    slot_lookup.fill(None);
    // `position` here is the index within reconcilable_slots (0-based), so the
    // stored row index is `stored.base_row_count + position`.
    #[allow(clippy::cast_possible_truncation)]
    for (position, &slot) in reconcilable_slots.iter().enumerate() {
        slot_lookup[slot as usize] = Some(position as u32);
    }

    // (d) Walk current cut rows and assign statuses.
    //
    // Activity-driven classifier (Epic 06 AD-2):
    // - Preserved slots: copy stored status. If LOWER, push onto promotion
    //   candidates for potential Scheme 1 symmetric promotion.
    // - New slots: consult active_window bitmap. If any of the last
    //   RECENT_WINDOW_K bits are set → LOWER (tight guess, count new_tight).
    //   Otherwise → BASIC (slack default, count new_slack).
    //
    // Scheme 1 symmetric promotion (after loop, Epic 06 AD-3 corrected by T3a):
    // for each LOWER-guessed new cut, promote one preserved-LOWER candidate
    // (lowest popcount first) to BASIC to maintain basic_count == num_row.
    // If the preserved-LOWER pool is exhausted (Scheme 2 tail), flip the
    // latest activity-driven LOWER guesses back to BASIC instead.
    let mut stats = ReconstructionStats::default();
    let mut lower_deficit: usize = 0;

    for (target_slot, intercept, coefficients) in current_cut_rows {
        // Capture the output row index BEFORE the push.
        let out_row_index = out.row_status.len();
        debug_assert!(
            out_row_index >= target.base_row_count,
            "cut row index {out_row_index} must be >= base_row_count {}",
            target.base_row_count,
        );

        let row_status_byte = if let Some(pos) = slot_lookup.get(target_slot).and_then(|o| *o) {
            // Slot was in the stored basis — copy its row status.
            // The stored row index is `stored.base_row_count + position`.
            let stored_row_idx = stored.base_row_count + pos as usize;
            let stored_status = stored.basis.row_status[stored_row_idx];
            stats.preserved += 1;
            if stored_status == HIGHS_BASIS_STATUS_LOWER {
                // Candidate for Scheme 1 symmetric promotion (preserved-LOWER → BASIC).
                // See docs/design/epic-06-classifier-refinement-gaps.md.
                let popcount = cut_metadata
                    .get(target_slot)
                    .map_or(0, |m| m.active_window.count_ones());
                promotion_scratch.candidates.push((out_row_index, popcount));
            }
            stored_status
        } else {
            // New cut — consult the activity window (Epic 06 AD-2).
            let active_window = cut_metadata.get(target_slot).map_or(0, |m| m.active_window);
            let _ = (intercept, coefficients); // suppress unused-variable warnings
            if active_window & RECENT_WINDOW_BITS != 0 {
                // Activity-guided tight guess: classify LOWER.
                stats.new_tight += 1;
                lower_deficit += 1;
                promotion_scratch.new_lower_indices.push(out_row_index);
                HIGHS_BASIS_STATUS_LOWER
            } else {
                // No recent activity: classify BASIC (safe default).
                stats.new_slack += 1;
                HIGHS_BASIS_STATUS_BASIC
            }
        };
        out.row_status.push(row_status_byte);
    }

    // (e) Scheme 1 symmetric promotion: restore basic_count == num_row.
    //
    // Each LOWER-classified new cut reduced total_basic by 1 relative to
    // the BASIC-default HiGHS dimension extension. Offset by promoting
    // the preserved-LOWER candidate with the lowest activity popcount
    // (least likely to still be binding → safest to move into the basis
    // as BASIC). Net change to total_basic: zero.
    if lower_deficit > 0 {
        // Stable sort: equal-popcount ties preserve insertion order
        // (declaration-order invariance). Do NOT use sort_unstable_by_key.
        promotion_scratch
            .candidates
            .sort_by_key(|&(_, popcount)| popcount);

        let available_candidates = promotion_scratch.candidates.len();
        let scheme1_count = lower_deficit.min(available_candidates);

        // Apply Scheme 1: promote the `scheme1_count` preserved-LOWER
        // candidates with the lowest popcounts to BASIC.
        for &(row_idx, _) in promotion_scratch.candidates.iter().take(scheme1_count) {
            debug_assert!(
                row_idx >= target.base_row_count,
                "Scheme 1 promotion target row_idx {row_idx} must be >= base_row_count {}",
                target.base_row_count,
            );
            debug_assert_eq!(
                out.row_status[row_idx], HIGHS_BASIS_STATUS_LOWER,
                "Scheme 1 promotion target must currently be LOWER",
            );
            out.row_status[row_idx] = HIGHS_BASIS_STATUS_BASIC;
        }

        // Scheme 2 tail fallback: if more LOWER guesses than preserved-LOWER
        // candidates, override the most-recently-classified new LOWER cuts
        // back to BASIC. This keeps the invariant under adversarial inputs.
        let remaining_deficit = lower_deficit - scheme1_count;
        if remaining_deficit > 0 {
            // Walk the new-LOWER indices from latest to earliest.
            for &row_idx in promotion_scratch
                .new_lower_indices
                .iter()
                .rev()
                .take(remaining_deficit)
            {
                debug_assert_eq!(
                    out.row_status[row_idx], HIGHS_BASIS_STATUS_LOWER,
                    "Scheme 2 override target must be LOWER",
                );
                out.row_status[row_idx] = HIGHS_BASIS_STATUS_BASIC;
            }
            // Adjust telemetry to reflect the actual classifier outcome.
            #[allow(clippy::cast_possible_truncation)]
            let n = remaining_deficit as u32;
            stats.new_tight -= n;
            stats.new_slack += n;
        }
    }

    stats
}

// ---------------------------------------------------------------------------
// enforce_basic_count_invariant
// ---------------------------------------------------------------------------

/// Restore `col_basic + row_basic == num_row` after [`reconstruct_basis`]
/// on the forward path.
///
/// ## Algebraic invariant
///
/// Let `excess = col_basic + row_basic - num_row` after reconstruction.
///
/// On the forward path, cut selection may drop cuts whose stored row status
/// was `HIGHS_BASIS_STATUS_BASIC`.  Each such BASIC drop leaves one fewer
/// BASIC row, but `reconstruct_basis` assigns BASIC unconditionally to new
/// cuts.  If more cuts were preserved with BASIC status than the LP now has
/// room for, `excess > 0`.
///
/// The investigation proved that `excess = dropped_basic >= 0` always.
/// There is never a deficit.  Therefore:
///
/// - If `excess == 0`: no-op, return 0.
/// - If `excess > 0`: scan `out.row_status` from the end, flipping
///   `HIGHS_BASIS_STATUS_BASIC` to `HIGHS_BASIS_STATUS_LOWER` only for
///   cut rows (indices `>= base_row_count`) until exactly `excess` demotions
///   have been applied.
///
/// ## Parameters
///
/// - `out` — the [`Basis`] returned by [`reconstruct_basis`].
/// - `num_row` — the total row count of the target LP
///   (`base_row_count + active_cut_count`).
/// - `base_row_count` — the number of template (non-cut) rows.  Only rows
///   at indices `>= base_row_count` are eligible for demotion.
///
/// ## Returns
///
/// The number of demotions applied (0 in the no-op case).
///
/// ## Assertions
///
/// `debug_assert!` checks that `num_row == out.row_status.len()` and
/// `base_row_count <= num_row`.
pub fn enforce_basic_count_invariant(
    out: &mut Basis,
    num_row: usize,
    base_row_count: usize,
) -> u32 {
    debug_assert_eq!(
        num_row,
        out.row_status.len(),
        "enforce_basic_count_invariant: num_row ({num_row}) != out.row_status.len() ({})",
        out.row_status.len(),
    );
    debug_assert!(
        base_row_count <= num_row,
        "enforce_basic_count_invariant: base_row_count ({base_row_count}) > num_row ({num_row})",
    );

    let col_basic = out
        .col_status
        .iter()
        .filter(|&&s| s == HIGHS_BASIS_STATUS_BASIC)
        .count();
    let row_basic = out
        .row_status
        .iter()
        .filter(|&&s| s == HIGHS_BASIS_STATUS_BASIC)
        .count();

    let total_basic = col_basic + row_basic;
    if total_basic <= num_row {
        // Invariant holds (or excess == 0): nothing to do.
        return 0;
    }

    let mut excess = total_basic - num_row;
    let mut demotions: u32 = 0;

    // Scan from the end of row_status, touching only cut rows (>= base_row_count).
    for idx in (base_row_count..out.row_status.len()).rev() {
        if excess == 0 {
            break;
        }
        if out.row_status[idx] == HIGHS_BASIS_STATUS_BASIC {
            out.row_status[idx] = HIGHS_BASIS_STATUS_LOWER;
            excess -= 1;
            demotions += 1;
        }
    }

    demotions
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod tests {
    use cobre_solver::Basis;

    use super::{
        enforce_basic_count_invariant, reconstruct_basis, PaddingContext, PromotionScratch,
        ReconstructionSource, ReconstructionStats, ReconstructionTarget,
        HIGHS_BASIS_STATUS_BASIC as B, HIGHS_BASIS_STATUS_LOWER as L, RECENT_WINDOW_K,
    };
    use crate::cut::pool::CutPool;
    use crate::cut_selection::CutMetadata;
    use crate::workspace::CapturedBasis;

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    /// Build a `CapturedBasis` from explicit slices.
    ///
    /// - `base_rows` template rows are filled with `B` (BASIC).
    /// - `cut_statuses` are appended after the template rows.
    /// - `slots` must have the same length as `cut_statuses`.
    fn make_stored_basis(
        base_rows: usize,
        num_cols: usize,
        slots: &[u32],
        cut_statuses: &[i32],
        state_at_capture: &[f64],
    ) -> CapturedBasis {
        assert_eq!(slots.len(), cut_statuses.len());
        let total_rows = base_rows + cut_statuses.len();
        let mut cb = CapturedBasis::new(
            num_cols,
            total_rows,
            base_rows,
            slots.len(),
            state_at_capture.len(),
        );

        // Fill basis row_status: template rows = BASIC, then cut statuses.
        cb.basis.row_status.clear();
        cb.basis.row_status.resize(base_rows, B);
        cb.basis.row_status.extend_from_slice(cut_statuses);

        // Fill col_status: all BASIC.
        cb.basis.col_status.clear();
        cb.basis.col_status.resize(num_cols, B);

        // Slot and state metadata.
        cb.cut_row_slots.extend_from_slice(slots);
        cb.state_at_capture.extend_from_slice(state_at_capture);

        cb
    }

    /// Convenience: build a `ReconstructionSource` with no activity metadata.
    ///
    /// Used by tests that exercise the preserved/slack paths only; the
    /// classifier falls back to `active_window = 0` (BASIC) for all slots.
    fn source_no_metadata(base_row_count: usize, num_cols: usize) -> ReconstructionSource<'static> {
        ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count,
                num_cols,
            },
            cut_metadata: &[],
        }
    }

    /// Build a minimal `CutMetadata` with only `active_window` set.
    ///
    /// All other fields default to zero / `iteration_generated = 0`.
    fn meta_with_window(active_window: u32) -> CutMetadata {
        CutMetadata {
            iteration_generated: 0,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: 0,
            active_window,
        }
    }

    // -----------------------------------------------------------------------
    // reconstruct_basis — unit tests (offset=0 path, the common case)
    // -----------------------------------------------------------------------

    /// AC1: empty stored basis + 3 new cuts — all classified BASIC regardless
    /// of cut-value evaluation (ticket-008: unconditional-BASIC for new cuts).
    #[test]
    fn test_empty_stored_all_new_cuts() {
        // stored: shim state — no cut rows, base_rows=2, num_cols=3.
        let stored = CapturedBasis::new(3, 2, 2, 0, 0);

        // Cuts at padding_state=[1.0, 2.0], theta=10.0, tolerance=1e-7.
        // Post ticket-008: cut-value evaluation is no longer performed for new
        // cuts; all three slots are BASIC regardless of their slack/tightness.
        //   slot 5: would have been BASIC (slack=2)
        //   slot 6: would have been LOWER (tight, slack=0) — now BASIC
        //   slot 7: would have been BASIC (slack=10)
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (5, 3.0, vec![1.0, 2.0]),
            (6, 8.0, vec![1.0, 0.5]),
            (7, 0.0, vec![0.0, 0.0]),
        ];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 16];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(2, 3),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 10.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 0,
                new_tight: 0,
                new_slack: 3
            }
        );
        // Template rows come first (2 BASIC), then all 3 new cuts are BASIC.
        assert_eq!(out.row_status.len(), 5);
        assert_eq!(out.row_status[2], B, "slot 5 -> BASIC");
        assert_eq!(
            out.row_status[3], B,
            "slot 6 -> BASIC (tight by value, but BASIC by invariant)"
        );
        assert_eq!(out.row_status[4], B, "slot 7 -> BASIC");
    }

    /// AC2: all 5 cuts preserved — same slots, same order.
    #[test]
    fn test_all_preserved_same_slots_same_order() {
        let stored = make_stored_basis(3, 4, &[10, 11, 12, 13, 14], &[L, B, L, B, L], &[1.0, 2.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 1.0, vec![0.5, 0.5]),
            (11, 2.0, vec![0.5, 0.5]),
            (12, 3.0, vec![0.5, 0.5]),
            (13, 4.0, vec![0.5, 0.5]),
            (14, 5.0, vec![0.5, 0.5]),
        ];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 5,
                new_tight: 0,
                new_slack: 0
            }
        );
        assert_eq!(&out.row_status[3..8], &stored.basis.row_status[3..8]);
    }

    /// AC3: drops — target has only slots [10, 12, 14] (11 and 13 dropped).
    #[test]
    fn test_drops_only() {
        let stored = make_stored_basis(3, 4, &[10, 11, 12, 13, 14], &[L, B, L, B, L], &[1.0, 2.0]);
        // Target has only slots 10, 12, 14 (positions 0, 2, 4 in stored).
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0, 0.0]),
            (12, 0.0, vec![0.0, 0.0]),
            (14, 0.0, vec![0.0, 0.0]),
        ];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 3,
                new_tight: 0,
                new_slack: 0
            }
        );
        // out.row_status[3] == stored[3] (slot 10 -> stored pos 0 -> row 3)
        // out.row_status[4] == stored[5] (slot 12 -> stored pos 2 -> row 5)
        // out.row_status[5] == stored[7] (slot 14 -> stored pos 4 -> row 7)
        assert_eq!(out.row_status[3], stored.basis.row_status[3], "slot 10");
        assert_eq!(out.row_status[4], stored.basis.row_status[5], "slot 12");
        assert_eq!(out.row_status[5], stored.basis.row_status[7], "slot 14");
    }

    /// AC4: reorder — target has slots [11, 13, 10, 12, 14].
    #[test]
    fn test_reorder() {
        let stored = make_stored_basis(3, 4, &[10, 11, 12, 13, 14], &[L, B, L, B, L], &[1.0, 2.0]);
        // Reordered: slot 11 (stored pos 1), slot 13 (stored pos 3), etc.
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (11, 0.0, vec![0.0, 0.0]),
            (13, 0.0, vec![0.0, 0.0]),
            (10, 0.0, vec![0.0, 0.0]),
            (12, 0.0, vec![0.0, 0.0]),
            (14, 0.0, vec![0.0, 0.0]),
        ];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 5,
                new_tight: 0,
                new_slack: 0
            }
        );
        // slot 11 was stored at position 1 -> stored row 3+1=4
        assert_eq!(
            out.row_status[3], stored.basis.row_status[4],
            "slot 11 -> stored pos 1"
        );
        // slot 13 -> stored pos 3 -> stored row 6
        assert_eq!(
            out.row_status[4], stored.basis.row_status[6],
            "slot 13 -> stored pos 3"
        );
        // slot 10 -> stored pos 0 -> stored row 3
        assert_eq!(
            out.row_status[5], stored.basis.row_status[3],
            "slot 10 -> stored pos 0"
        );
        // slot 12 -> stored pos 2 -> stored row 5
        assert_eq!(
            out.row_status[6], stored.basis.row_status[5],
            "slot 12 -> stored pos 2"
        );
        // slot 14 -> stored pos 4 -> stored row 7
        assert_eq!(
            out.row_status[7], stored.basis.row_status[7],
            "slot 14 -> stored pos 4"
        );
    }

    /// AC5: adds only — target has slots [10, 11, 12, 13]; 10 and 11 preserved,
    /// 12 and 13 are new cuts classified BASIC unconditionally (ticket-008).
    #[test]
    fn test_adds_only() {
        // Stored has only slots 10 and 11.
        let stored = make_stored_basis(3, 4, &[10, 11], &[L, B], &[1.0, 2.0]);
        // Post ticket-008: slots 12 and 13 are new; both get BASIC regardless of
        // cut-value evaluation (slot 12 would have been LOWER, slot 13 BASIC).
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0, 0.0]),
            (11, 0.0, vec![0.0, 0.0]),
            (12, 9.0, vec![0.5, 0.5]),
            (13, 0.0, vec![0.0, 0.0]),
        ];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 10.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_tight: 0,
                new_slack: 2
            }
        );
        // Template rows 0..3; out[3..5] = stored[3..5] = [L, B] (preserved).
        assert_eq!(&out.row_status[3..5], &stored.basis.row_status[3..5]);
        // New cuts are unconditionally BASIC (no activity metadata, active_window=0).
        assert_eq!(
            out.row_status[5], B,
            "slot 12 -> BASIC (no metadata, active_window=0)"
        );
        assert_eq!(out.row_status[6], B, "slot 13 -> BASIC");
    }

    /// AC6: mixed drop+add — stored has [10, 11], target has [11, 12].
    #[test]
    fn test_mixed_drop_and_add() {
        let stored = make_stored_basis(3, 4, &[10, 11], &[L, B], &[1.0, 2.0]);
        // slot 11 -> preserved (stored pos 1 -> stored row 4 = B)
        // slot 12 -> new, padding_state=[1.0,2.0], theta=10.0:
        //   value=0.0+0.0+0.0=0.0, slack=10 -> BASIC
        let cuts: Vec<(usize, f64, Vec<f64>)> =
            vec![(11, 0.0, vec![0.0, 0.0]), (12, 0.0, vec![0.0, 0.0])];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 10.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 1,
                new_tight: 0,
                new_slack: 1
            }
        );
        // out[3] = stored[4] (slot 11 at stored pos 1 -> row index 3+1=4)
        assert_eq!(
            out.row_status[3], stored.basis.row_status[4],
            "slot 11 preserved"
        );
        assert_eq!(out.row_status[4], B, "slot 12 new slack");
    }

    /// AC7: empty iterator — no cut rows in target LP.
    #[test]
    fn test_empty_iterator_preserves_template_rows() {
        let stored = make_stored_basis(3, 4, &[10, 11, 12], &[L, B, L], &[1.0, 2.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(stats, ReconstructionStats::default());
        assert_eq!(out.row_status.len(), 3, "only template rows");
    }

    /// AC8: stored basis with a high-numbered slot + undersized lookup —
    /// release-mode defensive growth kicks in.
    ///
    /// Only meaningful in release mode: in debug mode the `debug_assert!(false)`
    /// inside `reconstruct_basis` panics, which is the intended signal that the
    /// caller under-sized the scratch buffer. Growth is triggered by the
    /// `reconcilable_slots.max()` branch — i.e. stored-slot max, not target
    /// slot — which is why the stored basis carries slot 999 here.
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_slot_lookup_growth_safe_in_release() {
        // Stored basis has one cut row at slot 999 (LOWER). Undersized lookup
        // (len=5) triggers release-mode growth when reconcilable max = 999.
        let stored = make_stored_basis(3, 3, &[999], &[L], &[]);
        // Current target has a different, fresh slot (500) — classified new slack
        // via the empty-coefficients slack path (theta=100, value=0).
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![(500, 0.0, vec![])];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 5]; // too small for slot 999

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(3, 3),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        // Growth path: lookup must have grown to >= max_stored_slot + 1 = 1000.
        assert!(lookup.len() >= 1000, "lookup must have grown");
        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 0,
                new_tight: 0,
                new_slack: 1
            }
        );
    }

    // -----------------------------------------------------------------------
    // Integration: forward apply path with cut churn (ticket-003 AC #5)
    // -----------------------------------------------------------------------

    /// AC #5 (ticket-003): a stored basis whose `cut_row_slots` are
    /// `[0, 1, 2]` and a current `pool.active_cuts()` that yields
    /// `[0, 2]` (slot 1 deactivated by cut selection) must reconstruct with
    /// `preserved == 2` and the preserved row statuses must match the stored
    /// basis at the right positions.  This mirrors the wiring in
    /// `forward.rs::run_forward_stage` after ticket-003.
    #[test]
    fn test_forward_reconstruct_preserves_slots_after_churn() {
        // Build a pool with three cuts at slots 0, 1, 2.
        let mut pool = CutPool::new(16, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0]);
        pool.add_cut(1, 0, 2.0, &[2.0]);
        pool.add_cut(2, 0, 3.0, &[3.0]);

        // Stored basis (from a previous iteration): all 3 cut slots active,
        // row statuses [L, B, L].
        let stored = make_stored_basis(2, 3, &[0, 1, 2], &[L, B, L], &[5.0]);

        // Cut selection deactivates slot 1 between iterations.
        pool.deactivate(&[1]);
        let active_after_churn: Vec<(usize, f64, Vec<f64>)> = pool
            .active_cuts()
            .map(|(s, i, c)| (s, i, c.to_vec()))
            .collect();
        assert_eq!(
            active_after_churn.len(),
            2,
            "two cuts remain active after deactivating slot 1"
        );

        // Reconstruct using the same call shape that `forward.rs` uses.
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 16];
        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(2, 3),
            active_after_churn
                .iter()
                .map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[5.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        // Both surviving cuts (slots 0 and 2) should be preserved, no new cuts.
        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_tight: 0,
                new_slack: 0,
            },
            "churn must preserve surviving slots without new-cut churn",
        );

        // Verify the preserved row statuses came from the right stored slots:
        //   target row 2 (after 2 template rows) = slot 0 = stored row 2 (= L)
        //   target row 3                          = slot 2 = stored row 4 (= L)
        assert_eq!(out.row_status[2], stored.basis.row_status[2], "slot 0");
        assert_eq!(out.row_status[3], stored.basis.row_status[4], "slot 2");
    }

    /// AC #6 (ticket-003): three new cuts at slots beyond the stored basis,
    /// all evaluating to slack at the padding state, must produce
    /// `stats.new_slack == 3`, `stats.new_tight == 0`, and the corresponding
    /// row statuses must all be `HIGHS_BASIS_STATUS_BASIC`.
    #[test]
    fn test_forward_reconstruct_three_new_slack_cuts() {
        // Stored basis has 2 pre-existing cuts at slots [0, 1] with statuses [L, B].
        let stored = make_stored_basis(2, 3, &[0, 1], &[L, B], &[1.0]);

        // Pool now contains the original 2 cuts plus three new cuts at
        // slots 20, 21, 22.  All three new cuts evaluate to slack at the
        // padding state (theta=100, cut_value low → slack > tolerance).
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (0, 1.0, vec![1.0]),  // preserved
            (1, 2.0, vec![2.0]),  // preserved
            (20, 0.0, vec![0.0]), // new, value=0, slack=100  → BASIC
            (21, 1.0, vec![1.0]), // new, value=2, slack=98   → BASIC
            (22, 2.0, vec![0.5]), // new, value=2.5, slack=97 → BASIC
        ];

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];
        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(2, 3),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_tight: 0,
                new_slack: 3,
            },
            "two preserved + three slack new",
        );
        // Rows 4, 5, 6 are the three new cuts (after 2 template + 2 preserved).
        assert_eq!(out.row_status[4], B, "slot 20 new slack");
        assert_eq!(out.row_status[5], B, "slot 21 new slack");
        assert_eq!(out.row_status[6], B, "slot 22 new slack");
    }

    /// AC #7 (ticket-003): the capture-site metadata writes must populate
    /// `cut_row_slots`, `state_at_capture`, and `base_row_count` so the next
    /// iteration's reconstruct sees a well-formed `CapturedBasis`.  This
    /// test exercises the same logic as `forward.rs::write_capture_metadata`
    /// directly on a `CapturedBasis` to verify the three invariants without
    /// requiring a full forward-pass setup.
    #[test]
    fn test_capture_metadata_invariants() {
        // Build a pool with three cuts.
        let mut pool = CutPool::new(16, 2, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0, 2.0]);
        pool.add_cut(1, 0, 2.0, &[1.5, 2.5]);
        pool.add_cut(2, 0, 3.0, &[2.0, 3.0]);

        let n_state = 2;
        let base_row_count = 4;
        let cut_row_count = 3;
        let mut captured = CapturedBasis::new(
            5, // num_cols
            base_row_count + cut_row_count,
            base_row_count,
            cut_row_count,
            n_state,
        );

        // Mimic the write_capture_metadata implementation (since it lives in
        // forward.rs and is private).  This test verifies the contract.
        captured.cut_row_slots.clear();
        #[allow(clippy::cast_possible_truncation)]
        for (slot, _intercept, _coeffs) in pool.active_cuts().take(cut_row_count) {
            captured.cut_row_slots.push(slot as u32);
        }
        captured.state_at_capture.clear();
        let current_state = vec![10.0, 20.0];
        captured.state_at_capture.extend_from_slice(&current_state);
        captured.base_row_count = base_row_count;

        // Verify invariants:
        assert_eq!(
            captured.cut_row_slots.len(),
            pool.active_cuts().count(),
            "cut_row_slots.len() must equal pool.active_cuts().count()",
        );
        assert_eq!(
            captured.state_at_capture.len(),
            n_state,
            "state_at_capture.len() must equal n_state",
        );
        assert_eq!(
            captured.base_row_count, base_row_count,
            "base_row_count must equal templates[t].num_rows",
        );
        // The full metadata invariant: row_status.len() == base + cut_row_slots.len().
        assert_eq!(
            captured.basis.row_status.len(),
            captured.base_row_count + captured.cut_row_slots.len(),
            "metadata invariant: row_status.len() == base_row_count + cut_row_slots.len()",
        );
    }

    // -----------------------------------------------------------------------
    // Ticket-008: basic-count invariant on the baked-template backward path
    //
    // HiGHS isBasisConsistent requires: basic_count == num_row, where
    //   basic_count = col_status.count(BASIC) + row_status.count(BASIC)
    //   num_row     = base_row_count + delta_count
    //
    // The stored basis is captured from a fully solved baked-template LP, so it
    // already satisfies the invariant for its own shape.  reconstruct_basis must
    // preserve it after (a) all cuts preserved, and (b) new cuts appended.
    //
    // Setup chosen so the invariant holds exactly for sub-scenario (a):
    //   baked_base_rows=10, num_cols=6, stored delta [100,101,102] with [B,L,B]
    //   col_basic_count=1 (1 BASIC + 5 LOWER cols)
    //   row_basic_count = 10 base (B) + 2 delta (B) = 12
    //   1 + 12 = 13 == 10 + 3  (invariant holds for (a))
    // -----------------------------------------------------------------------

    // Builds a baked-style stored basis whose basic count satisfies the HiGHS
    // invariant for 3 delta cuts: 1 BASIC col + (10 base + 2 delta) BASIC rows
    // = 13 == base_rows(10) + delta(3).
    fn make_baked_consistent_stored() -> CapturedBasis {
        let mut s = make_stored_basis(10, 6, &[100, 101, 102], &[B, L, B], &[1.0, 2.0, 3.0]);
        s.basis.col_status.clear();
        s.basis.col_status.extend_from_slice(&[B, L, L, L, L, L]);
        s
    }

    /// (a) Stored basis exactly matches the target delta set — all cuts preserved.
    ///
    /// Verifies `col_basic_count + row_basic_count == base_row_count + delta_count`
    /// when no new cuts are appended (all 3 delta slots are in the stored basis).
    #[test]
    fn reconstructed_basis_preserves_basic_count_invariant_backward_all_preserved() {
        let stored = make_baked_consistent_stored();
        let baked_base_rows = 10usize;
        let num_cols = 6usize;
        let col_basic = stored.basis.col_status.iter().filter(|&&s| s == B).count();

        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (100, 0.0, vec![0.0, 0.0, 0.0]),
            (101, 0.0, vec![0.0, 0.0, 0.0]),
            (102, 0.0, vec![0.0, 0.0, 0.0]),
        ];
        let delta_count = delta_cuts.len();
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 200];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(baked_base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0, 3.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(stats.preserved, 3, "all delta cuts preserved");
        assert_eq!(
            stats.new_tight, 0,
            "no activity metadata — all new cuts fall back to BASIC"
        );
        assert_eq!(stats.new_slack, 0, "no new cuts");

        let row_basic = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic,
            baked_base_rows + delta_count,
            "basic_count invariant: col_basic({col_basic}) + row_basic({row_basic}) \
             == base_rows({baked_base_rows}) + delta({delta_count})",
        );
    }

    /// (b) Target adds 2 new delta cuts beyond the stored set.
    ///
    /// Verifies that each new cut increments `row_basic_count` by 1 (unconditional
    /// BASIC classification), keeping `col_basic_count + row_basic_count ==
    /// base_row_count + delta_count` exactly.
    #[test]
    fn reconstructed_basis_preserves_basic_count_invariant_backward_with_new_cuts() {
        let stored = make_baked_consistent_stored();
        let baked_base_rows = 10usize;
        let num_cols = 6usize;
        let col_basic = stored.basis.col_status.iter().filter(|&&s| s == B).count();

        // 3 preserved slots + 2 new cuts (slot 200 would be tight by value, 201 slack).
        // Both must become BASIC unconditionally (ticket-008 invariant).
        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (100, 0.0, vec![0.0, 0.0, 0.0]),
            (101, 0.0, vec![0.0, 0.0, 0.0]),
            (102, 0.0, vec![0.0, 0.0, 0.0]),
            (200, 5.0, vec![1.0, 0.0, 0.0]), // new; would be tight by value
            (201, 0.0, vec![0.0, 0.0, 0.0]), // new; slack
        ];
        let delta_count = delta_cuts.len();
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 300];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(baked_base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0, 3.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );

        assert_eq!(stats.preserved, 3, "three preserved delta cuts");
        assert_eq!(
            stats.new_tight, 0,
            "no activity metadata — new cuts fall back to BASIC"
        );
        assert_eq!(
            stats.new_slack, 2,
            "two new cuts classified BASIC (no metadata)"
        );

        let row_basic = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic,
            baked_base_rows + delta_count,
            "basic_count invariant: col_basic({col_basic}) + row_basic({row_basic}) \
             == base_rows({baked_base_rows}) + delta({delta_count})",
        );
        // New delta cuts sit at rows [base+3] and [base+4]; both must be BASIC.
        assert_eq!(
            out.row_status[baked_base_rows + 3],
            B,
            "slot 200 must be BASIC"
        );
        assert_eq!(
            out.row_status[baked_base_rows + 4],
            B,
            "slot 201 must be BASIC"
        );
    }

    // -----------------------------------------------------------------------
    // Ticket-009: basic-count invariant on the forward path
    //
    // On the forward path, cut selection can drop cuts whose stored status was
    // BASIC, causing col_basic + row_basic to exceed num_row after
    // reconstruct_basis.  enforce_basic_count_invariant restores equality by
    // demoting trailing cut-row BASIC statuses to LOWER.
    //
    // Three sub-scenarios (split per clippy::too_many_lines):
    //   (a) all_preserved — no drops, delta == 0, demotions == 0
    //   (b) drops_with_lower — drops include LOWER statuses; only BASIC drops
    //       cause excess; demotions == dropped_basic
    //   (c) new_cuts_after_drops — new cuts are always BASIC (ticket-008),
    //       so they never cause excess; only preserved BASIC drops do
    //
    // Setup for (a) and (b):
    //   base_rows=3, num_cols=4
    //   stored 5 cuts [10,11,12,13,14] with statuses [B,L,B,L,B]
    //   col_status = [B,L,L,L] => col_basic = 1
    //   stored row_basic = 3 (template) + 3 (cut B: slots 10,12,14) = 6
    //   invariant for stored LP: col_basic(1) + row_basic(6) = 7 == 3+3+1=7 ✓
    //   (stored LP had 5 delta cuts: num_row = 3+5 = 8, but we use a subset
    //    in the reconstruction targets below)
    // -----------------------------------------------------------------------

    // (a) All cuts preserved — no drops, excess == 0, demotions must be 0.
    //
    // Stored: 3 cuts [10,12,14] with statuses [B,B,B], col_basic=1.
    // Target: same 3 cuts => all preserved, 0 new.
    // num_row = base(3) + cuts(3) = 6
    // col_basic=1, row_basic = 3 (template B) + 3 (cuts B) = 6 => total=7 > 6 ?
    //
    // To make (a) a clean no-op, choose stored so that invariant holds:
    //   col_basic=1, template_basic=2 (two template rows BASIC, one LOWER),
    //   cut_basic=3 => total = 1+2+3 = 6 == num_row(6). ✓
    #[test]
    fn reconstructed_basis_preserves_basic_count_invariant_forward_all_preserved() {
        // stored: base_rows=3 (2 BASIC, 1 LOWER via explicit row_status),
        //         col=[B,L,L,L], cuts [10,12,14] all BASIC.
        // We set template rows explicitly: first build then override row_status.
        let num_cols = 4usize;
        let base_rows = 3usize;
        let mut stored = make_stored_basis(base_rows, num_cols, &[10, 12, 14], &[B, B, B], &[1.0]);
        // Override col_status: only col 0 is BASIC (col_basic=1).
        stored.basis.col_status.clear();
        stored.basis.col_status.extend_from_slice(&[B, L, L, L]);
        // Override template rows so row_basic for template = 2 (rows 0,1 = BASIC, row 2 = LOWER).
        stored.basis.row_status[0] = B;
        stored.basis.row_status[1] = B;
        stored.basis.row_status[2] = L;
        // Total: col_basic=1, row_basic(template)=2, row_basic(cuts)=3 => 1+2+3=6 == 3+3 ✓

        let col_basic = stored.basis.col_status.iter().filter(|&&s| s == B).count();

        // Target: same 3 cuts, all preserved.
        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0]),
            (12, 0.0, vec![0.0]),
            (14, 0.0, vec![0.0]),
        ];
        let num_cut = delta_cuts.len();
        let num_row = base_rows + num_cut;

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );
        assert_eq!(stats.preserved, 3, "all 3 cuts preserved");
        assert_eq!(stats.new_tight, 0);
        assert_eq!(stats.new_slack, 0);

        let demotions = enforce_basic_count_invariant(&mut out, num_row, base_rows);

        assert_eq!(
            demotions, 0,
            "no excess when all preserved and invariant holds"
        );
        let row_basic_after = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic_after,
            num_row,
            "invariant holds: col_basic({col_basic}) + row_basic({row_basic_after}) \
             == num_row({num_row})",
        );
    }

    // (b) Cuts dropped include LOWER statuses — excess equals count of dropped
    // BASIC cuts (not the LOWER drops), demotions restore the invariant.
    //
    // Concrete working fixture (derived from ticket-009 algebraic proof):
    //   base_rows=1, col=[B] => col_basic=1, row_basic_tmpl=1.
    //   stored: 4 cuts [10,11,12,13] statuses [B,B,L,B] (3 BASIC, 1 LOWER).
    //   stored LP: 1+1+3=5 == 1+4=5 (consistent)
    //   Target: 2 cuts [10,11] (slots 12,13 dropped — statuses L,B).
    //   preserved: [B,B] => preserved_basic=2.
    //   col(1)+row(1+2)=4; num_row=1+2=3; excess=4-3=1.
    //   dropped_lower count = 1 (slot 12 was L), dropped_basic=1 (slot 13).
    //   The ticket's proof: excess = dropped_lower = 1. demotions expected = 1.
    #[test]
    fn reconstructed_basis_preserves_basic_count_invariant_forward_drops_with_lower() {
        let base_rows = 1usize;
        let num_cols = 1usize;
        // stored: 4 cuts, statuses [B,B,L,B]; col=[B]; template row=[B].
        let mut stored = make_stored_basis(
            base_rows,
            num_cols,
            &[10, 11, 12, 13],
            &[B, B, L, B],
            &[1.0],
        );
        stored.basis.col_status.clear();
        stored.basis.col_status.extend_from_slice(&[B]);
        stored.basis.row_status[0] = B; // template row

        let col_basic = stored.basis.col_status.iter().filter(|&&s| s == B).count();
        assert_eq!(col_basic, 1);

        // Stored LP invariant check: 1 + (1 template + 3 cut BASIC) = 5 == 1+4=5 ✓
        let stored_row_basic = stored.basis.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + stored_row_basic,
            1 + 4,
            "stored LP invariant sanity"
        );

        // Target: only slots [10, 11] — slots 12 (L) and 13 (B) are dropped.
        // dropped_lower = 1 (slot 12), dropped_basic = 1 (slot 13).
        // excess = dropped_lower = 1 by the ticket's proof.
        let delta_cuts: Vec<(usize, f64, Vec<f64>)> =
            vec![(10, 0.0, vec![0.0]), (11, 0.0, vec![0.0])];
        let num_cut = delta_cuts.len();
        let num_row = base_rows + num_cut; // 1+2=3

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 20];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );
        assert_eq!(stats.preserved, 2, "both surviving cuts preserved");
        assert_eq!(stats.new_tight, 0);
        assert_eq!(stats.new_slack, 0);

        // Before enforce: col(1)+row(1+2)=4 > num_row(3) => excess=1
        let row_basic_before = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic_before,
            4,
            "pre-enforce: total basic = 4"
        );

        let demotions = enforce_basic_count_invariant(&mut out, num_row, base_rows);

        // excess=1 => 1 demotion; demotions == dropped_lower = 1 ✓
        assert_eq!(demotions, 1, "demotions == dropped_lower == 1");

        let row_basic_after = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic_after,
            num_row,
            "invariant restored: col_basic({col_basic}) + row_basic({row_basic_after}) \
             == num_row({num_row})",
        );
    }

    // (d) Baked-path truncation case — regression for convertido_a iter-4 crash.
    //
    // When cut selection deactivates cuts between iterations, baked[t].num_rows
    // shrinks from L_old to L_new < L_old. The stored forward basis was captured
    // against L_old; reconstruct_basis truncates row_status to L_new. If the
    // dropped tail [L_new, L_old) contained any LOWER entries, the remaining
    // basis carries excess BASIC equal to that count. The caller (stage_solve.rs)
    // must pass base_row_count = n_state (not baked.num_rows) so the enforcer's
    // scan range [n_state, num_row) is non-empty and can demote the excess.
    #[test]
    fn reconstructed_basis_preserves_invariant_on_baked_truncation() {
        // L_old = 5, L_new = 3, tail [3..5) has one LOWER entry.
        let num_cols = 3usize;
        let n_state = 1usize;
        let l_old = 5usize;
        let l_new = 3usize;

        // Build a stored basis with the chosen row pattern; make_stored_basis
        // initialises template rows to BASIC, so we override row_status below.
        let mut stored = make_stored_basis(l_old, num_cols, &[], &[], &[1.0]);
        stored.basis.col_status.clear();
        stored.basis.col_status.extend_from_slice(&[B, B, L]);
        // Stored row pattern [L, B, B, B, L]: truncated tail [3..5) = [B, L]
        // has one LOWER entry → excess = 1 after truncation to L_new = 3.
        stored.basis.row_status.clear();
        stored.basis.row_status.extend_from_slice(&[L, B, B, B, L]);

        let col_basic = stored.basis.col_status.iter().filter(|&&s| s == B).count();
        let stored_row_basic = stored.basis.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(col_basic + stored_row_basic, l_old, "stored LP invariant");

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 16];

        // Baked path: empty cut iterator — all cuts live in the baked template.
        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(l_new, num_cols),
            std::iter::empty::<(usize, f64, &[f64])>(),
            PaddingContext {
                state: &[1.0],
                theta: 0.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );
        assert_eq!(stats, ReconstructionStats::default());
        assert_eq!(out.row_status.len(), l_new);

        let row_basic_before = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic_before,
            l_new + 1,
            "truncation leaves excess = 1 (one LOWER dropped from the tail)",
        );

        // The buggy call (base_row_count == num_row) is a no-op.
        let mut out_noop = out.clone();
        let noop_demotions = enforce_basic_count_invariant(&mut out_noop, l_new, l_new);
        assert_eq!(
            noop_demotions, 0,
            "old caller args (base_row_count == num_row) cannot demote",
        );
        let noop_total = col_basic + out_noop.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            noop_total,
            l_new + 1,
            "old caller args leave the excess in place — this is the bug",
        );

        // The fixed call (base_row_count == n_state) demotes one BASIC.
        let demotions = enforce_basic_count_invariant(&mut out, l_new, n_state);
        assert_eq!(demotions, 1, "fixed caller args demote one trailing BASIC");

        let row_basic_after = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic_after,
            l_new,
            "invariant restored after the fix",
        );
    }

    // (c) New cuts added after drops.
    //
    // New cuts are always classified BASIC (ticket-008). They grow num_row by 1
    // each, so they cannot create excess — each new cut adds 1 to row_basic and
    // 1 to num_row simultaneously. Excess is derived solely from drops.
    //
    // Fixture extends (b): same stored/dropped setup but the target includes
    // 2 new cuts (slots 20, 21) in addition to the 2 preserved ones.
    //   base_rows=1, col=[B], template=[B].
    //   stored: 4 cuts [10,11,12,13] statuses [B,B,L,B].
    //   Target: slots [10,11,20,21] — slots 12(L),13(B) dropped + 2 new.
    //   preserved: [B,B], new: [B,B] (ticket-008 unconditional BASIC).
    //   col(1)+row(1+2+2)=6; num_row=1+4=5; excess=6-5=1.
    //   demotions expected = 1 (new cuts do not contribute to excess).
    #[test]
    fn reconstructed_basis_preserves_basic_count_invariant_forward_new_cuts_after_drops() {
        let base_rows = 1usize;
        let num_cols = 1usize;
        let mut stored = make_stored_basis(
            base_rows,
            num_cols,
            &[10, 11, 12, 13],
            &[B, B, L, B],
            &[1.0],
        );
        stored.basis.col_status.clear();
        stored.basis.col_status.extend_from_slice(&[B]);
        stored.basis.row_status[0] = B;

        let col_basic = stored.basis.col_status.iter().filter(|&&s| s == B).count();

        // Target: 2 preserved + 2 new.
        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0]),
            (11, 0.0, vec![0.0]),
            (20, 0.0, vec![0.0]), // new
            (21, 0.0, vec![0.0]), // new
        ];
        let num_cut = delta_cuts.len();
        let num_row = base_rows + num_cut; // 1+4=5

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];

        let stats = reconstruct_basis(
            &stored,
            source_no_metadata(base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut PromotionScratch::default(),
        );
        assert_eq!(stats.preserved, 2, "slots 10,11 preserved");
        assert_eq!(stats.new_tight, 0);
        assert_eq!(stats.new_slack, 2, "slots 20,21 new (BASIC)");

        // Verify pre-enforce excess: col(1)+row(1+2+2)=6 > num_row(5) => excess=1.
        let row_basic_before = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic_before,
            6,
            "pre-enforce total basic = 6"
        );

        let demotions = enforce_basic_count_invariant(&mut out, num_row, base_rows);

        // New cuts do not contribute to excess; only dropped_lower=1 does.
        assert_eq!(
            demotions, 1,
            "demotions == dropped_lower == 1 (new cuts don't add excess)"
        );

        let row_basic_after = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic_after,
            num_row,
            "invariant restored: col_basic({col_basic}) + row_basic({row_basic_after}) \
             == num_row({num_row})",
        );
    }

    // ── Activity-driven classifier tests (Epic 06 T2) ─────────────────────────

    /// A new cut whose `active_window` is all-zero (never seen as binding) gets
    /// classified BASIC (new_slack += 1).
    #[test]
    fn classifier_returns_basic_when_active_window_is_zero() {
        let base_rows = 1usize;
        let num_cols = 1usize;
        let stored = make_stored_basis(base_rows, num_cols, &[], &[], &[1.0]);

        // One new cut, slot 99, active_window = 0 (never tight).
        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![(99, 0.0, vec![0.0])];
        let meta = vec![meta_with_window(0)];

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 128];
        let mut promotion_scratch = PromotionScratch::default();

        let source = ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &meta,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 0.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );
        assert_eq!(stats.new_tight, 0, "window=0 → BASIC (new_slack)");
        assert_eq!(stats.new_slack, 1);
    }

    /// A new cut whose `active_window` has bit 0 set (tight in the most recent
    /// iteration) gets classified LOWER (new_tight += 1).
    ///
    /// Includes one preserved-LOWER cut (slot 5) so Scheme 1 symmetric promotion
    /// can absorb the LOWER guess — otherwise Scheme 2 would override it back to
    /// BASIC and the new_tight increment would be cancelled.
    ///
    /// Invariant sizing: num_row=3 (1 base + 1 preserved + 1 new).
    /// After Scheme 1: row_basic = template(B) + slot5(B promoted) + slot0(L) = 2.
    /// col_basic = num_cols = 1. Total = 3 = num_row. ✓
    #[test]
    fn classifier_returns_lower_when_active_window_has_recent_bit() {
        let base_rows = 1usize;
        let num_cols = 1usize;
        // stored has slot 5 = LOWER so Scheme 1 has a promotion candidate.
        let stored = make_stored_basis(base_rows, num_cols, &[5], &[L], &[1.0]);

        // Preserved cut at slot 5, new cut at slot 0 with recent-bit activity.
        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (5, 0.0, vec![0.0]), // preserved
            (0, 0.0, vec![0.0]), // new, tight
        ];
        let mut meta = vec![meta_with_window(0); 6];
        meta[0] = meta_with_window(0b0000_0001); // new cut — recent bit
                                                 // meta[5] left at 0 → popcount=0, will be picked for promotion.

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 128];
        let mut promotion_scratch = PromotionScratch::default();

        let source = ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &meta,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 0.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );
        assert_eq!(stats.preserved, 1, "slot 5 preserved");
        assert_eq!(stats.new_tight, 1, "window bit 0 set → LOWER (new_tight)");
        assert_eq!(stats.new_slack, 0);
        assert_eq!(
            out.row_status.last().copied(),
            Some(L),
            "new cut slot 0 must have LOWER status",
        );
    }

    /// A new cut whose `active_window` has only bits outside the recent window
    /// (bits 5+) set is treated as dormant — classified BASIC.
    #[test]
    fn classifier_ignores_active_window_outside_recent_window() {
        let base_rows = 1usize;
        let num_cols = 1usize;
        let stored = make_stored_basis(base_rows, num_cols, &[], &[], &[1.0]);

        // Bit 5 set → outside RECENT_WINDOW_K=5, so masked away → BASIC.
        // cut_metadata indexed by pool slot; use slot 0 to align with meta[0].
        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![(0, 0.0, vec![0.0])];
        let meta = vec![meta_with_window(1u32 << RECENT_WINDOW_K)];

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 128];
        let mut promotion_scratch = PromotionScratch::default();

        let source = ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &meta,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 0.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );
        assert_eq!(stats.new_tight, 0, "bits outside window → BASIC");
        assert_eq!(stats.new_slack, 1);
    }

    /// Scheme 1 symmetric promotion (T3a, Epic 06 AD-3 corrected):
    /// when 2 new cuts are classified LOWER (both have recent bits), the 2
    /// preserved-LOWER candidates with the lowest popcounts are promoted to
    /// BASIC, keeping `col_basic + row_basic == num_row`.
    ///
    /// Fixture (post-T3a direction: preserved cuts are LOWER, not BASIC):
    ///   base_rows=1 (col [B] × 4, row 0 = B).
    ///   stored cuts: slots [10, 11, 12, 13] — all LOWER (promotion candidates).
    ///   Target: all 4 preserved + 2 new (slots 20, 21), both with recent activity.
    ///   meta for preserved 10..13 has monotonic popcounts 1, 3, 5, 7; stable
    ///   sort picks the two lowest (10 and 11) for promotion LOWER → BASIC.
    ///
    /// num_cols=4 chosen so col_basic(4) + row_basic(3) = 7 = num_row. ✓
    /// After promotion: template(B) + slot10(B) + slot11(B) → row_basic=3.
    /// The HiGHS invariant col_basic + row_basic == num_row holds exactly.
    #[test]
    fn scheme_1_symmetric_promotion_preserves_total_basic() {
        let base_rows = 1usize;
        // num_cols=4: col_basic(4) + row_basic(3 after promotion) = 7 = num_row.
        // This satisfies the HiGHS invariant col_basic + row_basic == num_row.
        let num_cols = 4usize;
        // Stored cuts are LOWER — these are the promotion candidates.
        let stored = make_stored_basis(
            base_rows,
            num_cols,
            &[10, 11, 12, 13],
            &[L, L, L, L],
            &[1.0],
        );

        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0]),
            (11, 0.0, vec![0.0]),
            (12, 0.0, vec![0.0]),
            (13, 0.0, vec![0.0]),
            (20, 0.0, vec![0.0]), // new, tight → LOWER
            (21, 0.0, vec![0.0]), // new, tight → LOWER
        ];
        let num_row = base_rows + delta_cuts.len(); // 1+6=7

        // meta indexed by cut-pool slot. Size=22 so slots 20/21 are in bounds.
        // Preserved slot popcounts: 1, 3, 5, 7 (monotonic); stable sort picks
        // the two lowest (10 and 11) for promotion.
        let mut meta = vec![meta_with_window(0); 22];
        meta[10] = meta_with_window(0b0000_0001); // popcount=1 — promoted first
        meta[11] = meta_with_window(0b0000_0111); // popcount=3 — promoted second
        meta[12] = meta_with_window(0b0001_1111); // popcount=5 — stays LOWER
        meta[13] = meta_with_window(0b0111_1111); // popcount=7 — stays LOWER
        meta[20] = meta_with_window(0b0000_0001); // new, recent → LOWER
        meta[21] = meta_with_window(0b0000_0001); // new, recent → LOWER

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];
        let mut promotion_scratch = PromotionScratch::default();

        let source = ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &meta,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 0.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );

        assert_eq!(stats.preserved, 4, "4 slots preserved");
        assert_eq!(stats.new_tight, 2, "2 new LOWER cuts");
        assert_eq!(stats.new_slack, 0);

        // Verify that the 2 preserved slots with the LOWEST popcounts (10, 11)
        // were promoted from LOWER to BASIC. Preserved slots 12 and 13 remain
        // LOWER because they had higher popcounts (more recently active).
        // Row layout:
        //   [template(B), slot10(B-promoted), slot11(B-promoted),
        //    slot12(L), slot13(L), slot20(L-new), slot21(L-new)]
        assert_eq!(out.row_status[0], B, "template row unchanged");
        assert_eq!(
            out.row_status[1], B,
            "slot 10 (popcount=1) promoted LOWER → BASIC"
        );
        assert_eq!(
            out.row_status[2], B,
            "slot 11 (popcount=3) promoted LOWER → BASIC"
        );
        assert_eq!(out.row_status[3], L, "slot 12 (popcount=5) stays LOWER");
        assert_eq!(out.row_status[4], L, "slot 13 (popcount=7) stays LOWER");
        assert_eq!(out.row_status[5], L, "new slot 20 classified LOWER");
        assert_eq!(out.row_status[6], L, "new slot 21 classified LOWER");

        // Critical invariant assertion absent from T2 — the primary reason the
        // direction bug survived review. Scheme 1 symmetric promotion keeps
        // total_basic unchanged: 2 new LOWER each consume one promotion (net ±0).
        // col_basic=4 (all 4 cols BASIC) + row_basic=3 (template + slot10 + slot11)
        // = 7 = num_row. ✓
        let row_basic = out.row_status.iter().filter(|&&s| s == B).count();
        let col_basic = out.col_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic,
            num_row,
            "Epic 06 AD-3: HiGHS invariant col_basic + row_basic == num_row"
        );

        // Scheme 1 keeps col_basic + row_basic == num_row by construction, so
        // the safety-net enforcer is a no-op.
        let demoted = enforce_basic_count_invariant(&mut out, num_row, base_rows);
        assert_eq!(demoted, 0, "Scheme 1 promotion makes enforcer a no-op");
    }

    /// Scheme 2 fallback (T3a updated fixture):
    /// when the LOWER-classified new cuts exceed the preserved-LOWER promotion
    /// candidates, the excess LOWERs are overridden back to BASIC.
    ///
    /// Fixture (post-T3a: preserved cuts are LOWER, not BASIC):
    ///   base_rows=1 (col [B] × 2, row 0 = B).
    ///   stored: 2 preserved LOWER cuts at slots 5 and 6 (promotion candidates).
    ///   Target: 2 preserved + 3 new (all with recent activity).
    ///   Classifier: 3 LOWER guesses (lower_deficit=3).
    ///   Scheme 1: promotes both preserved LOWERs → BASIC (scheme1_count=2).
    ///   Scheme 2: remaining_deficit=1 → overrides the latest new LOWER back to BASIC.
    ///   Final: slot5(B-promoted), slot6(B-promoted), slot20(L), slot21(L),
    ///          slot22(B-overridden) → new_tight=2, new_slack=1.
    ///
    /// num_cols=2: col_basic(2) + row_basic(4) = 6 = num_row. ✓
    #[test]
    fn scheme_2_fallback_when_deficit_exceeds_candidates() {
        let base_rows = 1usize;
        // num_cols=2: col_basic(2) + row_basic(4 after promotion+override) = 6 = num_row.
        let num_cols = 2usize;
        // Preserved cuts are LOWER — promotion candidates for Scheme 1.
        let stored = make_stored_basis(base_rows, num_cols, &[5, 6], &[L, L], &[1.0]);

        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (5, 0.0, vec![0.0]),  // preserved (LOWER) → promoted to BASIC by Scheme 1
            (6, 0.0, vec![0.0]),  // preserved (LOWER) → promoted to BASIC by Scheme 1
            (20, 0.0, vec![0.0]), // new, tight → LOWER
            (21, 0.0, vec![0.0]), // new, tight → LOWER
            (22, 0.0, vec![0.0]), // new, tight → LOWER; overridden to BASIC by Scheme 2
        ];
        let num_row = base_rows + delta_cuts.len(); // 1+5=6

        // meta indexed by pool slot; size=23 covers slots 20..22.
        let mut meta = vec![meta_with_window(0); 23];
        meta[5] = meta_with_window(0b0000_0001); // popcount=1 — promoted first
        meta[6] = meta_with_window(0b0000_0011); // popcount=2 — promoted second
        meta[20] = meta_with_window(0b0000_0001); // new, recent
        meta[21] = meta_with_window(0b0000_0001); // new, recent
        meta[22] = meta_with_window(0b0000_0001); // new, recent — latest, overridden

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];
        let mut promotion_scratch = PromotionScratch::default();

        let source = ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &meta,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 0.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );

        assert_eq!(stats.preserved, 2, "2 slots preserved");
        assert_eq!(
            stats.new_tight, 2,
            "3 LOWER guesses − 1 Scheme 2 override = 2 net new_tight"
        );
        assert_eq!(
            stats.new_slack, 1,
            "1 LOWER overridden to BASIC by Scheme 2 tail"
        );

        // Verify final row statuses. Row layout (post-T3a promotion):
        // [template(B), slot5(B-promoted), slot6(B-promoted),
        //  slot20(L), slot21(L), slot22(B-overridden)]
        assert_eq!(out.row_status[0], B, "template row unchanged");
        assert_eq!(
            out.row_status[1], B,
            "slot 5 promoted LOWER → BASIC by Scheme 1"
        );
        assert_eq!(
            out.row_status[2], B,
            "slot 6 promoted LOWER → BASIC by Scheme 1"
        );
        assert_eq!(out.row_status[3], L, "slot 20 classified LOWER (tight)");
        assert_eq!(out.row_status[4], L, "slot 21 classified LOWER (tight)");
        assert_eq!(
            out.row_status[5], B,
            "slot 22 overridden to BASIC by Scheme 2"
        );
        assert_eq!(out.row_status.len(), num_row);

        // Invariant assertion: Scheme 1 promotion + Scheme 2 override together
        // keep col_basic + row_basic == num_row (Epic 06 AD-3).
        // col_basic=2 + row_basic=4 (template, slot5, slot6, slot22) = 6 = num_row.
        let row_basic = out.row_status.iter().filter(|&&s| s == B).count();
        let col_basic = out.col_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic,
            num_row,
            "Epic 06 AD-3: HiGHS invariant col_basic + row_basic == num_row"
        );

        // Scheme 1 + Scheme 2 keep the invariant by construction, so the enforcer
        // is a no-op.
        let demoted = enforce_basic_count_invariant(&mut out, num_row, base_rows);
        assert_eq!(demoted, 0, "Scheme 1+2 together make enforcer a no-op");
    }

    /// Scheme 1 promotion with no preserved-LOWER candidates falls back to Scheme 2
    /// (T3a Requirement 8, Epic 06 AD-3).
    ///
    /// Fixture: stored basis has 0 preserved LOWER cuts — all preserved are BASIC
    /// (the pre-T3a configuration where Scheme 1 would have acted on BASIC candidates).
    /// Under T3a, the preserved BASIC cuts are NOT promotion candidates, so Scheme 1
    /// finds 0 candidates and the entire deficit is handled by Scheme 2.
    ///
    ///   base_rows=1 (0 cols → col_basic=0, row 0 = B).
    ///   stored: 1 preserved BASIC cut at slot 3 (not a promotion candidate).
    ///   new: slot 10 with recent activity → LOWER (lower_deficit=1).
    ///   Scheme 1: 0 LOWER candidates → scheme1_count=0.
    ///   Scheme 2: remaining_deficit=1 → overrides slot 10 LOWER → BASIC.
    ///   Final: template(B), slot3(B), slot10(B-overridden). new_tight=0, new_slack=1.
    ///
    /// num_cols=0: col_basic(0) + row_basic(3) = 3 = num_row. ✓
    #[test]
    fn scheme_1_promotion_with_no_preserved_lowers_fallback_to_scheme_2() {
        let base_rows = 1usize;
        // num_cols=0: col_basic(0) + row_basic(num_row) = num_row. ✓
        // All preserved cuts are BASIC — none qualify as promotion candidates.
        let num_cols = 0usize;
        let stored = make_stored_basis(base_rows, num_cols, &[3], &[B], &[1.0]);

        let delta_cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (3, 0.0, vec![]),  // preserved (BASIC) — not a Scheme 1 candidate
            (10, 0.0, vec![]), // new, tight → LOWER; Scheme 2 overrides to BASIC
        ];
        let num_row = base_rows + delta_cuts.len(); // 1+2=3

        // meta: slot 10 has recent activity; slot 3 (preserved) has no impact on
        // promotion (it's BASIC — not selected as a candidate regardless of popcount).
        let mut meta = vec![meta_with_window(0); 11];
        meta[10] = meta_with_window(0b0000_0001); // recent bit set → LOWER classified

        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 16];
        let mut promotion_scratch = PromotionScratch::default();

        let source = ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &meta,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 0.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );

        // Scheme 1 found 0 LOWER candidates → Scheme 2 overrode the 1 new LOWER.
        assert_eq!(stats.preserved, 1, "slot 3 preserved");
        assert_eq!(
            stats.new_tight, 0,
            "Scheme 2 adjusted new_tight to 0 (1 LOWER overridden)"
        );
        assert_eq!(
            stats.new_slack, 1,
            "Scheme 2 override counts as new_slack += 1"
        );

        // Verify final row statuses:
        // [template(B), slot3(B-preserved), slot10(B-overridden-by-Scheme2)]
        assert_eq!(out.row_status[0], B, "template row unchanged");
        assert_eq!(out.row_status[1], B, "slot 3 preserved BASIC unchanged");
        assert_eq!(
            out.row_status[2], B,
            "slot 10 overridden to BASIC by Scheme 2 (no Scheme 1 candidates)"
        );
        assert_eq!(out.row_status.len(), num_row);

        // Invariant assertion: Scheme 2 alone keeps col_basic + row_basic == num_row.
        // col_basic=0 (num_cols=0) + row_basic=3 (all rows BASIC) = 3 = num_row. ✓
        let row_basic = out.row_status.iter().filter(|&&s| s == B).count();
        let col_basic = out.col_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic,
            num_row,
            "Epic 06 AD-3: HiGHS invariant col_basic + row_basic == num_row"
        );

        // Scheme 2 kept the invariant, so the enforcer is a no-op.
        let demoted = enforce_basic_count_invariant(&mut out, num_row, base_rows);
        assert_eq!(demoted, 0, "Scheme 2 alone makes enforcer a no-op");
    }

    // -----------------------------------------------------------------------
    // Epic 06 T3: production-path activation tests
    //
    // These tests exercise the slot-identity preservation and activity-driven
    // classifier on the production baked path, where stage_solve.rs passes
    // inputs.pool.active_cuts() instead of std::iter::empty().
    // -----------------------------------------------------------------------

    /// T3 Requirement 6: slot-identity preservation across cut deactivation.
    ///
    /// Build a pool with 10 cuts at slots 0..9. Capture a basis with
    /// cut_row_slots=[0..9] and alternating row_statuses [L,B,L,B,...].
    /// Deactivate slots 3 and 7. Call reconstruct_basis with pool.active_cuts()
    /// (yields 8 slots: 0,1,2,4,5,6,8,9). Assert all 8 surviving slots are
    /// preserved and their statuses match the stored status for each slot,
    /// NOT the positional status that a truncation-based approach would copy.
    #[test]
    fn baked_path_preserves_slot_identity_across_cut_deactivation() {
        let base_rows = 1usize;
        let num_cols = 1usize;
        let state_dim = 1usize;

        // Stored basis: 10 cuts at slots 0..9 with alternating [L,B,...].
        // slot 0=L, 1=B, 2=L, 3=B, 4=L, 5=B, 6=L, 7=B, 8=L, 9=B.
        #[allow(clippy::cast_possible_truncation)]
        let stored_slots: Vec<u32> = (0u32..10).collect();
        let stored_statuses: Vec<i32> = (0..10).map(|i| if i % 2 == 0 { L } else { B }).collect();
        let stored =
            make_stored_basis(base_rows, num_cols, &stored_slots, &stored_statuses, &[1.0]);

        // Pool: 10 cuts at slots 0..9, then deactivate 3 and 7.
        // With forward_passes=1, warm_start_count=0: slot = iteration.
        let mut pool = CutPool::new(16, state_dim, 1, 0);
        for i in 0u64..10 {
            #[allow(clippy::cast_precision_loss)]
            let intercept = i as f64;
            pool.add_cut(i, 0, intercept, &[1.0]);
        }
        pool.deactivate(&[3, 7]);

        // active_cuts() now yields 8 slots: 0,1,2,4,5,6,8,9.
        assert_eq!(pool.active_count(), 8, "8 active cuts after deactivation");

        let source = ReconstructionSource {
            target: ReconstructionTarget {
                // base_row_count = template rows only (not baked rows).
                // Mirrors what stage_solve.rs does after T3.
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &[], // no activity metadata needed for preservation
        };

        let mut out = Basis::new(0, 0);
        // slot_lookup must cover slot 9 (max stored slot).
        let mut lookup: Vec<Option<u32>> = vec![None; 16];
        let mut promotion_scratch = PromotionScratch::default();

        let padding = PaddingContext {
            state: &[1.0],
            theta: 100.0,
            tolerance: 1e-7,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            pool.active_cuts(),
            padding,
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );

        // All 8 surviving slots must be preserved (slot identity).
        assert_eq!(stats.preserved, 8, "all 8 surviving slots preserved");
        assert_eq!(stats.new_tight, 0, "no new tight cuts");
        assert_eq!(stats.new_slack, 0, "no new slack cuts");

        // Output has base_rows (template) + 8 (cut) rows.
        let num_cut_rows = 8usize;
        assert_eq!(out.row_status.len(), base_rows + num_cut_rows);

        // Verify each surviving slot's status matches its stored status.
        // active_cuts() yields in slot order: 0,1,2,4,5,6,8,9.
        // Stored statuses (slot 0=L,1=B,2=L,3=B[deactivated],4=L,5=B,6=L,
        //                   7=B[deactivated],8=L,9=B).
        // Output cut rows (indices base_rows..base_rows+8):
        //   row 1 → slot 0 → L
        //   row 2 → slot 1 → B
        //   row 3 → slot 2 → L
        //   row 4 → slot 4 → L  (NOT slot 3's B — identity, not position)
        //   row 5 → slot 5 → B
        //   row 6 → slot 6 → L
        //   row 7 → slot 8 → L  (NOT slot 7's B — identity, not position)
        //   row 8 → slot 9 → B
        let expected = [L, B, L, L, B, L, L, B];
        for (i, &expected_status) in expected.iter().enumerate() {
            assert_eq!(
                out.row_status[base_rows + i],
                expected_status,
                "cut row {i}: expected {expected_status}",
            );
        }
    }

    /// T3 Requirement 7: activity-driven classifier fires on the production path.
    ///
    /// Build a pool with 8 cuts: 5 new cuts at slots 0..4 (slot 2 has
    /// active_window=0b1) and 3 preserved cuts at slots 100..102 (stored
    /// LOWER). The stored basis has only the 3 high-slot cuts as cut rows.
    /// Calling reconstruct_basis with pool.active_cuts() exercises:
    /// - Slot identity preservation for stored slots 100,101,102 (preserved=3).
    /// - Activity-driven LOWER guess for new slot 2 (new_tight=1).
    /// - Scheme 1 symmetric promotion: the LOWER guess triggers promotion of one
    ///   preserved-LOWER candidate (slot 100, lowest popcount, first in insertion
    ///   order) from LOWER → BASIC.
    /// - enforce_basic_count_invariant is a no-op (Scheme 1 kept the invariant).
    ///
    /// Invariant sizing: num_row = 1 + 8 = 9.
    /// After Scheme 1: row_basic = template(B) + slots 0,1,3,4(B) + slot100(B)
    ///   = 6. col_basic = num_cols = 3. Total = 9 = num_row. ✓
    #[test]
    fn baked_path_classifier_fires_on_recent_activity() {
        let base_rows = 1usize;
        // num_cols=3: col_basic(3) + row_basic(6) = 9 = num_row(9). ✓
        let num_cols = 3usize;
        let state_dim = 1usize;

        // Stored basis: 3 cuts at slots 100,101,102 all LOWER (promotion candidates).
        let stored = make_stored_basis(base_rows, num_cols, &[100, 101, 102], &[L, L, L], &[1.0]);

        // Pool: 5 new cuts at slots 0..4 + 3 preserved at slots 100..102.
        // Capacity must be >= 103 to hold slot 102.
        let mut pool = CutPool::new(128, state_dim, 1, 0);
        for i in 0u64..5 {
            pool.add_cut(i, 0, 0.0, &[0.0]);
        }
        for i in 100u64..103 {
            pool.add_cut(i, 0, 0.0, &[0.0]);
        }
        // Seed activity on slot 2: bit 0 set → recent activity → LOWER guess.
        pool.metadata[2].active_window = 0b0000_0001;
        // Slots 100,101,102 have active_window=0 (popcount=0); they will be
        // sorted first by Scheme 1 so slot 100 gets promoted LOWER → BASIC.

        assert_eq!(pool.active_count(), 8, "5 new + 3 preserved cuts active");

        // Build metadata slice for the source. cut_metadata indexed by slot.
        let source = ReconstructionSource {
            target: ReconstructionTarget {
                base_row_count: base_rows,
                num_cols,
            },
            cut_metadata: &pool.metadata,
        };

        let mut out = Basis::new(0, 0);
        // slot_lookup must cover slot 102.
        let mut lookup: Vec<Option<u32>> = vec![None; 128];
        let mut promotion_scratch = PromotionScratch::default();

        let padding = PaddingContext {
            state: &[1.0],
            theta: 100.0,
            tolerance: 1e-7,
        };

        let stats = reconstruct_basis(
            &stored,
            source,
            pool.active_cuts(),
            padding,
            &mut out,
            &mut lookup,
            &mut promotion_scratch,
        );

        // 5 new cuts (0..4) + 3 preserved (100..102).
        assert_eq!(stats.preserved, 3, "slots 100,101,102 preserved");
        // Slot 2 has active_window bit 0 set → classifier guesses LOWER.
        // Scheme 1 absorbs the deficit (promotes preserved-LOWER slot 100).
        assert_eq!(stats.new_tight, 1, "slot 2 classified LOWER by classifier");
        assert_eq!(stats.new_slack, 4, "slots 0,1,3,4 classified BASIC");

        // active_cuts() yields in slot order: 0,1,2,3,4,100,101,102.
        // Output cut rows (base_rows=1):
        //   row 1 → slot 0 → B (new, no activity)
        //   row 2 → slot 1 → B (new, no activity)
        //   row 3 → slot 2 → L (new, active_window=0b1 → LOWER)
        //   row 4 → slot 3 → B (new, no activity)
        //   row 5 → slot 4 → B (new, no activity)
        //   row 6 → slot 100 → B (preserved L, promoted by Scheme 1)
        //   row 7 → slot 101 → L (preserved L, not promoted)
        //   row 8 → slot 102 → L (preserved L, not promoted)
        assert_eq!(out.row_status[base_rows], B, "slot 0 → BASIC");
        assert_eq!(out.row_status[base_rows + 1], B, "slot 1 → BASIC");
        assert_eq!(
            out.row_status[base_rows + 2],
            L,
            "slot 2 → LOWER (classifier)"
        );
        assert_eq!(out.row_status[base_rows + 3], B, "slot 3 → BASIC");
        assert_eq!(out.row_status[base_rows + 4], B, "slot 4 → BASIC");
        assert_eq!(
            out.row_status[base_rows + 5],
            B,
            "slot 100 → BASIC (Scheme 1 promoted from LOWER)"
        );
        assert_eq!(
            out.row_status[base_rows + 6],
            L,
            "slot 101 → LOWER (preserved, not promoted)"
        );
        assert_eq!(
            out.row_status[base_rows + 7],
            L,
            "slot 102 → LOWER (preserved, not promoted)"
        );

        // Scheme 1 kept the invariant: enforce_basic_count_invariant is a no-op.
        let num_row = base_rows + 8; // 1 template + 8 cut rows
        let col_basic = out.col_status.iter().filter(|&&s| s == B).count();
        let row_basic = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic,
            num_row,
            "Epic 06 AD-3: HiGHS invariant col_basic + row_basic == num_row"
        );
        let demoted = enforce_basic_count_invariant(&mut out, num_row, base_rows);
        assert_eq!(
            demoted, 0,
            "Scheme 1 maintained the invariant; enforcer is no-op"
        );
    }
}
