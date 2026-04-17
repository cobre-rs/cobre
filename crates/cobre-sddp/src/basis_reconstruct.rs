//! Slot-tracked basis reconstruction for cut-set-aware warm-start (Strategy S3+).
//!
//! This module provides two helpers:
//!
//! - [`pad_basis_for_cuts`] — the original length-based padding helper (retained
//!   for backwards compatibility while tickets 003 and 004 rewire the callers).
//! - [`reconstruct_basis`] — the slot-identity-based replacement that correctly
//!   handles cut-set churn (drops, reorders, adds) between iterations.
//!
//! ## Why slot identity matters
//!
//! `pad_basis_for_cuts` uses the LP row count as the only reconciliation key.
//! If cut selection replaces one cut with another of equal count, the two bases
//! are the same length but positionally misaligned — `HiGHS` receives a basis with
//! mismatched row statuses and crashes back to cold start (or, worse, warm-starts
//! with a corrupt basis).
//!
//! `reconstruct_basis` takes [`CapturedBasis::cut_row_slots`] as its key: each
//! stored cut row carries the [`CutPool`](crate::cut::pool::CutPool) slot that
//! generated it.  The reconstruction walks the current LP's cut rows in order,
//! looks each slot up in an O(1) scratch map, and either copies the stored status
//! (slot found → preserved) or evaluates the cut at `padding_state` to assign
//! tight/slack (slot not found → new).
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

use crate::cut::pool::CutPool;
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
///   `padding_state.len()`.
/// - `padding_state` — state at which to evaluate newly-added cuts.
///   Forward path: `ws.current_state[..n_state]`.
///   Backward path: `stored.state_at_capture` (the fix implemented by ticket 004).
/// - `theta_padding_value` — θ proxy for the tight/slack decision.
/// - `tolerance` — slack threshold; cuts with `theta - cut_value <= tolerance`
///   get `NONBASIC_LOWER`.
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

    // (c) Build slot → stored-position lookup.
    //
    // Grow the scratch if it is too small for the current stored slots.  In
    // normal operation the caller pre-sizes this to `initial_pool_capacity`
    // (via `ScratchBuffers`), so growth only occurs on cold paths.  When
    // stored.cut_row_slots is empty there is nothing to look up — skip the
    // size check (an empty lookup is correct in that case).
    if let Some(max_slot_val) = stored.cut_row_slots.iter().copied().max() {
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
    #[allow(clippy::cast_possible_truncation)]
    for (position, &slot) in stored.cut_row_slots.iter().enumerate() {
        slot_lookup[slot as usize] = Some(position as u32);
    }

    // (d) Walk current cut rows and assign statuses.
    let mut stats = ReconstructionStats::default();
    for (target_slot, intercept, coefficients) in current_cut_rows {
        let row_status_byte = if let Some(pos) = slot_lookup.get(target_slot).and_then(|o| *o) {
            // Slot was in the stored basis — copy its row status.
            let stored_row_idx = stored.base_row_count + pos as usize;
            stats.preserved += 1;
            stored.basis.row_status[stored_row_idx]
        } else {
            // New cut — evaluate at padding.state.
            let cut_value: f64 = intercept
                + coefficients
                    .iter()
                    .zip(padding.state)
                    .map(|(c, x)| c * x)
                    .sum::<f64>();
            let slack = padding.theta - cut_value;
            if slack <= padding.tolerance {
                stats.new_tight += 1;
                HIGHS_BASIS_STATUS_LOWER
            } else {
                stats.new_slack += 1;
                HIGHS_BASIS_STATUS_BASIC
            }
        };
        out.row_status.push(row_status_byte);
    }

    stats
}

// ---------------------------------------------------------------------------
// pad_basis_for_cuts — retained for forward.rs / backward.rs callers.
// Tickets 003 and 004 will remove this import and delete this function.
// ---------------------------------------------------------------------------

/// Extend `basis.row_status` with informed status codes for new cut rows.
///
/// For each active cut in `pool` whose row is not yet covered by the existing
/// `basis.row_status`, evaluates the cut at the warm-start state and assigns:
///
/// - `NONBASIC_LOWER` if `theta_value - cut_value <= tolerance` (tight or violated),
/// - `BASIC` if `theta_value - cut_value > tolerance` (slack).
///
/// # Parameters
///
/// - `basis`: the cached basis to extend. Only `row_status` is modified;
///   `col_status` is untouched.
/// - `pool`: the active cut pool for the current stage.
/// - `state`: warm-start state vector, length `pool.state_dimension`. Provides
///   the state-variable values `x̂` at which each cut is evaluated.
/// - `theta_value`: the future-cost variable value from the warm-start solution,
///   already unscaled.
/// - `base_row_count`: number of structural (template) rows. The basis is
///   expected to already have entries for these rows plus any previously
///   appended cut rows from earlier iterations.
/// - `tolerance`: slack threshold. Cuts with `theta_value - cut_value <= tolerance`
///   are assigned `NONBASIC_LOWER`.
///
/// # Returns
///
/// `(tight_count, slack_count)` — the number of new rows assigned
/// `NONBASIC_LOWER` and `BASIC` respectively. Both are 0 when no new rows
/// are added.
///
/// # Panics
///
/// Panics (in debug builds only) if `state.len() != pool.state_dimension` or
/// if `basis.row_status.len() < base_row_count`.
pub fn pad_basis_for_cuts(
    basis: &mut Basis,
    pool: &CutPool,
    state: &[f64],
    theta_value: f64,
    base_row_count: usize,
    tolerance: f64,
) -> (usize, usize) {
    debug_assert!(
        state.len() == pool.state_dimension,
        "state length {} != pool.state_dimension {}",
        state.len(),
        pool.state_dimension,
    );
    debug_assert!(
        basis.row_status.len() >= base_row_count,
        "basis.row_status.len() {} < base_row_count {}",
        basis.row_status.len(),
        base_row_count,
    );

    let target_len = base_row_count + pool.active_count();

    if basis.row_status.len() >= target_len {
        return (0, 0);
    }

    let old_len = basis.row_status.len();

    // Resize with conservative default BASIC. Entries for previously seen cuts
    // (between old_len and target_len) will be overwritten below; the resize
    // ensures the Vec has the correct final length before the loop writes into it.
    basis
        .row_status
        .resize(target_len, HIGHS_BASIS_STATUS_BASIC);

    // The offset of the first cut row from base_row_count in the basis vector.
    // Previously appended cut rows occupy [base_row_count, old_len).
    // New cut rows occupy [old_len, target_len).
    //
    // We iterate all active cuts in slot order (matching LP row order) and
    // assign statuses only for those that fall in the new region.
    let already_padded_cuts = old_len.saturating_sub(base_row_count);

    let mut tight_count: usize = 0;
    let mut slack_count: usize = 0;
    let mut cut_index: usize = 0;

    for (_slot, intercept, coefficients) in pool.active_cuts() {
        if cut_index < already_padded_cuts {
            // This cut row was already in the basis from a previous padding call.
            cut_index += 1;
            continue;
        }

        let cut_value: f64 = intercept
            + coefficients
                .iter()
                .zip(state)
                .map(|(c, x)| c * x)
                .sum::<f64>();
        let slack = theta_value - cut_value;

        let row_idx = base_row_count + cut_index;
        if slack <= tolerance {
            basis.row_status[row_idx] = HIGHS_BASIS_STATUS_LOWER;
            tight_count += 1;
        } else {
            // Already set to BASIC by the resize fill — increment counter only.
            slack_count += 1;
        }

        cut_index += 1;
    }

    (tight_count, slack_count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use cobre_solver::Basis;

    use super::{
        pad_basis_for_cuts, reconstruct_basis, PaddingContext, ReconstructionStats,
        ReconstructionTarget, HIGHS_BASIS_STATUS_BASIC as B, HIGHS_BASIS_STATUS_LOWER as L,
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
    // reconstruct_basis — 8 unit tests
    // -----------------------------------------------------------------------

    /// AC1: empty stored basis + 3 new cuts (1 tight, 2 slack).
    #[test]
    fn test_empty_stored_all_new_cuts() {
        // stored: shim state — no cut rows, base_rows=2, num_cols=3.
        let stored = CapturedBasis::new(3, 2, 2, 0, 0);

        // Cuts at padding_state=[1.0, 2.0], theta=10.0, tolerance=1e-7:
        //   slot 5: value = 3.0 + 1.0*1.0 + 2.0*2.0 = 8.0,  slack = 10-8=2   -> BASIC
        //   slot 6: value = 8.0 + 1.0*1.0 + 0.5*2.0 = 10.0, slack = 10-10=0  -> LOWER (tight)
        //   slot 7: value = 0.0 + 0.0*1.0 + 0.0*2.0 = 0.0,  slack = 10-0=10  -> BASIC
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
                new_tight: 1,
                new_slack: 2
            }
        );
        // Template rows come first (2 BASIC), then the 3 cut statuses.
        assert_eq!(out.row_status.len(), 5);
        assert_eq!(out.row_status[2], B, "slot 5 slack=2 -> BASIC");
        assert_eq!(out.row_status[3], L, "slot 6 tight -> LOWER");
        assert_eq!(out.row_status[4], B, "slot 7 slack=10 -> BASIC");
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

    /// AC5: adds only — target has slots [10, 11, 12, 13] with 12 tight, 13 slack.
    #[test]
    fn test_adds_only() {
        // Stored has only slots 10 and 11.
        let stored = make_stored_basis(3, 4, &[10, 11], &[L, B], &[1.0, 2.0]);
        // padding_state=[1.0, 2.0], theta=10.0:
        //   slot 12: value=9.0+0.5*1+0.5*2=10.5, slack=10-10.5=-0.5 -> tight
        //   slot 13: value=0.0+0.0*1+0.0*2=0.0,  slack=10-0=10       -> slack
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
                new_tight: 1,
                new_slack: 1
            }
        );
        // Template rows 0..3; out[3..5] = stored[3..5] = [L, B]
        assert_eq!(&out.row_status[3..5], &stored.basis.row_status[3..5]);
        assert_eq!(out.row_status[5], L, "slot 12 tight");
        assert_eq!(out.row_status[6], B, "slot 13 slack");
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

    /// AC8: slot 999 with undersized lookup — grown in place, treated as new (slack).
    ///
    /// Only meaningful in release mode: in debug mode the `debug_assert!(false)`
    /// inside `reconstruct_basis` would panic, which is the intended signal that
    /// the caller under-sized the scratch buffer.
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_slot_lookup_growth_safe_in_release() {
        // Stored has no cut rows (shim). Undersized lookup triggers growth.
        let stored = CapturedBasis::new(3, 3, 3, 0, 0);
        // slot 999 is a new cut evaluated as slack (theta=100, value=0).
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![(999, 0.0, vec![])];
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

        // Growth path: lookup must have grown to >= 1000 entries.
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
    // pad_basis_for_cuts — 6 original tests (kept until tickets 003/004 land)
    // -----------------------------------------------------------------------

    fn make_pool_with_cuts(cuts: &[(f64, Vec<f64>)], state_dim: usize) -> CutPool {
        let mut pool = CutPool::new(cuts.len().max(1) * 10, state_dim, 1, 0);
        for (i, (intercept, coeffs)) in cuts.iter().enumerate() {
            pool.add_cut(i as u64, 0, *intercept, coeffs);
        }
        pool
    }

    #[test]
    fn test_tight_and_slack_cuts_get_correct_status() {
        let pool = make_pool_with_cuts(
            &[(10.0, vec![1.0]), (20.0, vec![2.0]), (30.0, vec![3.0])],
            1,
        );

        let mut basis = Basis::new(5, 2);
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[5.0], 25.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 5, "basis must grow to base+active");
        assert_eq!(basis.row_status[2], B, "cut 0 slack=10 -> BASIC");
        assert_eq!(basis.row_status[3], L, "cut 1 slack=-5 -> NONBASIC_LOWER");
        assert_eq!(basis.row_status[4], L, "cut 2 slack=-20 -> NONBASIC_LOWER");
        assert_eq!(tight, 2, "two tight/violated cuts");
        assert_eq!(slack, 1, "one slack cut");
    }

    #[test]
    fn test_exactly_tight_cut_is_nonbasic_lower() {
        let pool = make_pool_with_cuts(&[(5.0, vec![1.0, 2.0])], 2);

        let mut basis = Basis::new(3, 0);
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0, 1.0], 8.0, 0, 1e-7);

        assert_eq!(basis.row_status.len(), 1);
        assert_eq!(basis.row_status[0], L);
        assert_eq!(tight, 1);
        assert_eq!(slack, 0);
    }

    #[test]
    fn test_empty_pool_is_noop() {
        let pool = CutPool::new(10, 2, 1, 0);
        let mut basis = Basis::new(3, 2);
        basis.row_status[0] = L;
        basis.row_status[1] = B;

        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0, 1.0], 5.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 2, "row_status unchanged");
        assert_eq!(basis.row_status[0], L);
        assert_eq!(basis.row_status[1], B);
        assert_eq!(tight, 0);
        assert_eq!(slack, 0);
    }

    #[test]
    fn test_already_padded_basis_is_noop() {
        let pool = make_pool_with_cuts(&[(10.0, vec![1.0]), (20.0, vec![2.0])], 1);

        let mut basis = Basis::new(3, 4);
        basis.row_status[2] = L;
        basis.row_status[3] = L;

        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0], 5.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 4, "row_status unchanged");
        assert_eq!(basis.row_status[2], L, "prior status preserved");
        assert_eq!(basis.row_status[3], L, "prior status preserved");
        assert_eq!(tight, 0);
        assert_eq!(slack, 0);
    }

    #[test]
    fn test_all_slack_cuts_get_basic() {
        let pool = make_pool_with_cuts(&[(1.0, vec![1.0]), (2.0, vec![2.0])], 1);

        let mut basis = Basis::new(3, 2);
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0], 1000.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 4);
        assert_eq!(basis.row_status[2], B);
        assert_eq!(basis.row_status[3], B);
        assert_eq!(tight, 0);
        assert_eq!(slack, 2);
    }

    #[test]
    fn test_negative_slack_is_tight() {
        let pool = make_pool_with_cuts(&[(100.0, vec![10.0])], 1);

        let mut basis = Basis::new(3, 1);
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[5.0], 1.0, 1, 1e-7);

        assert_eq!(basis.row_status.len(), 2);
        assert_eq!(basis.row_status[1], L);
        assert_eq!(tight, 1);
        assert_eq!(slack, 0);
    }
}
