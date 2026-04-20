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
//!     PaddingContext, ReconstructionStats, ReconstructionTarget, reconstruct_basis,
//! };
//! use cobre_sddp::workspace::CapturedBasis;
//! use cobre_solver::Basis;
//!
//! let stored = CapturedBasis::new(4, 3, 3, 0, 0); // empty — shim state
//! let target = ReconstructionTarget { base_row_count: 3, num_cols: 4 };
//! let mut out = Basis::new(0, 0);
//! let mut lookup: Vec<Option<u32>> = vec![None; 16];
//! let cuts: Vec<(usize, f64, Vec<f64>)> = vec![];
//! let padding = PaddingContext { state: &[], theta: 0.0, tolerance: 1e-7 };
//! let stats = reconstruct_basis(
//!     &stored,
//!     target,
//!     cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
//!     padding,
//!     &mut out,
//!     &mut lookup,
//! );
//! assert_eq!(stats, ReconstructionStats::default());
//! ```

pub use cobre_solver::ffi::{HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER};

use cobre_solver::Basis;

use crate::workspace::CapturedBasis;

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
    /// New cut rows (slot not in stored basis) evaluated as tight or violated
    /// at `padding_state` (`theta - cut_value <= tolerance`).
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
/// - `target` — template row count and column count of the target LP.
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
///
/// ## Returns
///
/// [`ReconstructionStats`] with `preserved + new_tight + new_slack` equal to
/// the number of items yielded by the iterator.
///
/// ## Allocation contract
///
/// Allocation-free on the hot path when `slot_lookup.len() >= max_slot + 1`.
/// The growth path triggers a `debug_assert!(false)` to surface caller
/// under-sizing, but does not panic in release.
pub fn reconstruct_basis<'a, I>(
    stored: &CapturedBasis,
    target: ReconstructionTarget,
    current_cut_rows: I,
    padding: PaddingContext<'_>,
    out: &mut Basis,
    slot_lookup: &mut Vec<Option<u32>>,
) -> ReconstructionStats
where
    I: Iterator<Item = (usize, f64, &'a [f64])>,
{
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
    let mut stats = ReconstructionStats::default();
    for (target_slot, intercept, coefficients) in current_cut_rows {
        let row_status_byte = if let Some(pos) = slot_lookup.get(target_slot).and_then(|o| *o) {
            // Slot was in the stored basis — copy its row status.
            // The stored row index is `stored.base_row_count + position`.
            let stored_row_idx = stored.base_row_count + pos as usize;
            stats.preserved += 1;
            stored.basis.row_status[stored_row_idx]
        } else {
            // New cut — classify as BASIC (slack) unconditionally to preserve
            // basic_count == num_row. Each appended cut grows num_row by 1, so
            // marking it BASIC grows basic_count by 1 and keeps the invariant
            // exact. The simplex discovers tight cuts via pivots during the solve.
            //
            // NOTE: `new_tight` is always 0 after ticket-008 (backward path no
            // longer uses cut-value evaluation for new cuts). Scheduled for
            // removal in a follow-up cleanup ticket.
            let _ = (intercept, coefficients); // suppress unused-variable warnings
            stats.new_slack += 1;
            HIGHS_BASIS_STATUS_BASIC
        };
        out.row_status.push(row_status_byte);
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
mod tests {
    use cobre_solver::Basis;

    use super::{
        HIGHS_BASIS_STATUS_BASIC as B, HIGHS_BASIS_STATUS_LOWER as L, PaddingContext,
        ReconstructionStats, ReconstructionTarget, enforce_basic_count_invariant,
        reconstruct_basis,
    };
    use crate::cut::pool::CutPool;
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

    fn target(base_row_count: usize, num_cols: usize) -> ReconstructionTarget {
        ReconstructionTarget {
            base_row_count,
            num_cols,
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
            target(2, 3),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 10.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 10.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
        // New cuts are unconditionally BASIC.
        assert_eq!(
            out.row_status[5], B,
            "slot 12 -> BASIC (tight by value, BASIC by invariant)"
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
            target(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 10.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(3, 4),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(3, 3),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(2, 3),
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
            target(2, 3),
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(baked_base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0, 3.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
        );

        assert_eq!(stats.preserved, 3, "all delta cuts preserved");
        assert_eq!(stats.new_tight, 0, "new_tight is always 0 post ticket-008");
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
            target(baked_base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0, 2.0, 3.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
        );

        assert_eq!(stats.preserved, 3, "three preserved delta cuts");
        assert_eq!(stats.new_tight, 0, "new_tight is always 0 post ticket-008");
        assert_eq!(stats.new_slack, 2, "two new cuts classified BASIC");

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
            target(base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
            target(base_rows, num_cols),
            delta_cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            PaddingContext {
                state: &[1.0],
                theta: 100.0,
                tolerance: 1e-7,
            },
            &mut out,
            &mut lookup,
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
}
