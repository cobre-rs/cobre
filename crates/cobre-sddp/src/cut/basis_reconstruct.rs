//! Slot-tracked basis reconstruction for cut-set-aware warm-start.
//!
//! [`reconstruct_basis`] handles cut-set churn (drops, reorders, adds) between
//! iterations by keying on [`CapturedBasis::cut_row_slots`] — the
//! [`CutPool`](crate::cut::pool::CutPool) slot each stored cut row came from.
//!
//! ## Why slot identity matters
//!
//! Matching stored row statuses to the current LP by row **count** breaks under
//! churn: replacing one cut with another of equal count yields same-length but
//! positionally misaligned bases, so `HiGHS` rejects the basis (cold start) or
//! warm-starts a corrupted one. Keying on slot identity, not count, is the
//! contract.
//!
//! ## Why new cuts default to BASIC
//!
//! `HiGHS` requires `col_basic + row_basic == num_row` for any warm-start basis.
//! A new cut adds one row and one `BASIC`, balancing the equality by
//! construction; classifying it `LOWER` would break the equality and force a
//! compensating demotion elsewhere.
//!
//! ## DCS path: uniform-BASIC, slot-identity-free
//!
//! [`reconstruct_basis_uniform_basic`] is the Dynamic Cut Selection (DCS) variant
//! for the initial solve of each (stage, solve). It takes no `slot_lookup` and
//! reads none of [`CapturedBasis::cut_row_slots`]: DCS adds its cut rows fresh
//! each solve and does not guess which will bind, so slot alignment is
//! unnecessary and every resident cut row is seeded BASIC.
//!
//! ## Basic-count invariant
//!
//! [`enforce_basic_count_invariant`] runs unconditionally after every
//! reconstruction, demoting trailing BASIC cut rows until
//! `excess = col_basic + row_basic - num_row` reaches zero.
//!
//! `excess >= 0` holds only under three premises the reconstruction assumes and
//! never verifies: `stored.basis.col_status.len() == target.num_cols`;
//! `stored.base_row_count == target.base_row_count`; and `stored` satisfied
//! `col_basic + row_basic == num_row` for its own LP. A deficit therefore proves
//! `stored` was captured against a differently-shaped LP — it is rejected with
//! [`SddpError::BasisShapeMismatch`], never
//! repaired: demotion cannot create the missing basics, and promotion would
//! fabricate a basis `stored` never described.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::basis_reconstruct::{
//!     ReconstructionStats, ReconstructionTarget, reconstruct_basis,
//! };
//! use cobre_sddp::setup::NodeId;
//! use cobre_sddp::workspace::CapturedBasis;
//! use cobre_solver::Basis;
//!
//! let stored = CapturedBasis::new(4, 3, 3, 0, 0, NodeId(0)); // empty — shim state
//! let target = ReconstructionTarget { base_row_count: 3, num_cols: 4 };
//! let mut out = Basis::new(0, 0);
//! let mut lookup: Vec<Option<u32>> = vec![None; 16];
//! let cuts: Vec<(usize, f64, Vec<f64>)> = vec![];
//! let stats = reconstruct_basis(
//!     &stored,
//!     target,
//!     cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
//!     &mut out,
//!     &mut lookup,
//! );
//! assert_eq!(stats, ReconstructionStats::default());
//! ```

use cobre_solver::{Basis, BasisStatus, StageTemplate};

use crate::error::SddpError;
use crate::workspace::CapturedBasis;

// ---------------------------------------------------------------------------
// Target LP shape
// ---------------------------------------------------------------------------

/// Dimensions of the target LP [`reconstruct_basis`] populates a basis for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReconstructionTarget {
    /// Template (non-cut) row count of the target LP.
    pub base_row_count: usize,
    /// Total column count of the target LP.
    pub num_cols: usize,
}

impl ReconstructionTarget {
    /// Build a [`ReconstructionTarget`] from a stage's structural LP shape.
    #[must_use]
    #[inline]
    pub fn from_template(template: &StageTemplate) -> Self {
        Self {
            base_row_count: template.num_rows,
            num_cols: template.num_cols,
        }
    }
}

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

/// Counters returned by [`reconstruct_basis`].
///
/// `preserved + new_slack` equals the cut-row count of the target LP
/// (the number of items the iterator yielded).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReconstructionStats {
    /// Cut rows whose slot was found in the stored basis; status copied directly.
    pub preserved: u32,
    /// Cut rows whose slot was absent from the stored basis; each seeded BASIC.
    pub new_slack: u32,
}

// ---------------------------------------------------------------------------
// reconstruct_basis
// ---------------------------------------------------------------------------

/// Reconstruct a full [`Basis`] for the target LP using slot identity: stored
/// cut rows keep their status verbatim, new cut rows are seeded
/// [`BasisStatus::Basic`].
///
/// ## Parameters
///
/// - `current_cut_rows` — `(slot, intercept, coefficients)` in target LP row
///   order. `intercept`/`coefficients` are accepted for iterator parity but not
///   consulted — classification is by slot identity alone.
/// - `slot_lookup` — caller-presized scratch (`>= max_slot + 1`, via
///   `ScratchBuffers::recon_slot_lookup`); grown in place if undersized.
///
/// ## Allocation contract
///
/// Allocation-free when `slot_lookup.len() >= max_slot + 1`; the growth branch
/// `debug_assert!(false)`s to surface caller under-sizing without panicking in
/// release.
pub fn reconstruct_basis<'a, I>(
    stored: &CapturedBasis,
    target: ReconstructionTarget,
    current_cut_rows: I,
    out: &mut Basis,
    slot_lookup: &mut Vec<Option<u32>>,
) -> ReconstructionStats
where
    I: Iterator<Item = (usize, f64, &'a [f64])>,
{
    debug_assert!(
        stored.basis.row_status.len() == stored.base_row_count + stored.cut_row_slots.len(),
        "CapturedBasis invariant violated: row_status.len() {} != base_row_count {} + \
         cut_row_slots.len() {}",
        stored.basis.row_status.len(),
        stored.base_row_count,
        stored.cut_row_slots.len(),
    );

    reconstruct_col_statuses(stored, target, out);
    reconstruct_template_row_statuses(stored, target, out);
    build_slot_lookup(stored.cut_row_slots.as_slice(), slot_lookup);

    let mut stats = ReconstructionStats::default();
    for (target_slot, _intercept, _coefficients) in current_cut_rows {
        let row_status = if let Some(pos) = slot_lookup.get(target_slot).and_then(|o| *o) {
            let stored_row_idx = stored.base_row_count + pos as usize;
            stats.preserved += 1;
            stored.basis.row_status[stored_row_idx]
        } else {
            stats.new_slack += 1;
            BasisStatus::Basic
        };
        out.row_status.push(row_status);
    }

    stats
}

// ---------------------------------------------------------------------------
// reconstruct_basis_uniform_basic (DCS path)
// ---------------------------------------------------------------------------

/// Reconstruct a [`Basis`] for the **Dynamic Cut Selection (DCS)** initial solve,
/// seeding every cut row uniform [`BasisStatus::Basic`] — slot-identity-free, see
/// the module docs.
///
/// Does **not** repair the basic count: the caller must pair this with
/// [`enforce_basic_count_invariant`]`(out, target.base_row_count + cut_row_count,
/// target.base_row_count)` to restore `col_basic + row_basic == num_row` and to
/// reject a shape-mismatched `stored`.
pub fn reconstruct_basis_uniform_basic(
    stored: &CapturedBasis,
    target: ReconstructionTarget,
    cut_row_count: usize,
    out: &mut Basis,
) {
    reconstruct_col_statuses(stored, target, out);
    reconstruct_template_row_statuses(stored, target, out);

    debug_assert_eq!(
        out.row_status.len(),
        target.base_row_count,
        "reconstruct_template_row_statuses must leave exactly base_row_count entries before \
         cut-row seeding"
    );

    out.row_status
        .resize(target.base_row_count + cut_row_count, BasisStatus::Basic);
}

// ---------------------------------------------------------------------------
// Phase helpers (private — not part of the public API)
// ---------------------------------------------------------------------------

/// Copy column statuses from the stored basis into `out`, resized to
/// `target.num_cols` (padded with [`BasisStatus::Basic`] if wider).
fn reconstruct_col_statuses(stored: &CapturedBasis, target: ReconstructionTarget, out: &mut Basis) {
    debug_assert!(
        stored.basis.col_status.len() <= target.num_cols,
        "stored basis wider than target LP: col_status.len() {} > num_cols {}",
        stored.basis.col_status.len(),
        target.num_cols,
    );
    out.col_status.clear();
    out.col_status.extend_from_slice(&stored.basis.col_status);
    out.col_status.resize(target.num_cols, BasisStatus::Basic);
}

/// Copy the first `target.base_row_count` template row statuses from the stored
/// basis (missing rows filled [`BasisStatus::Basic`]).
///
/// Cut rows (indices `>= base_row_count`) are not written here — they belong to
/// the slot-identity loop in [`reconstruct_basis`].
fn reconstruct_template_row_statuses(
    stored: &CapturedBasis,
    target: ReconstructionTarget,
    out: &mut Basis,
) {
    out.row_status.clear();
    if stored.basis.row_status.len() >= target.base_row_count {
        out.row_status
            .extend_from_slice(&stored.basis.row_status[..target.base_row_count]);
    } else {
        out.row_status.extend_from_slice(&stored.basis.row_status);
        out.row_status
            .resize(target.base_row_count, BasisStatus::Basic);
    }
}

/// Fill `slot_lookup[slot] = Some(position)` for each slot in
/// `reconcilable_slots` (`position` = 0-based index within the slice).
///
/// Grows defensively if undersized — should not happen when the caller pre-sizes
/// via `ScratchBuffers`.
fn build_slot_lookup(reconcilable_slots: &[u32], slot_lookup: &mut Vec<Option<u32>>) {
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
            slot_lookup.resize(max_slot + 1, None);
        }
    }
    slot_lookup.fill(None);
    #[allow(clippy::cast_possible_truncation)]
    for (position, &slot) in reconcilable_slots.iter().enumerate() {
        slot_lookup[slot as usize] = Some(position as u32);
    }
}

// ---------------------------------------------------------------------------
// enforce_basic_count_invariant
// ---------------------------------------------------------------------------

/// Restore `col_basic + row_basic == num_row` after a reconstruction. Returns the
/// number of demotions applied.
///
/// Demotes **only cut rows** (indices `>= base_row_count`), never a template row.
///
/// # Errors
///
/// Returns [`SddpError::BasisShapeMismatch`] when `excess` is negative — a deficit
/// is unreachable under the module docs' three premises, so it is a shape mismatch
/// to surface, never a condition to repair.
pub fn enforce_basic_count_invariant(
    out: &mut Basis,
    num_row: usize,
    base_row_count: usize,
) -> Result<u32, SddpError> {
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
        .filter(|&&s| s == BasisStatus::Basic)
        .count();
    let row_basic = out
        .row_status
        .iter()
        .filter(|&&s| s == BasisStatus::Basic)
        .count();

    let total_basic = col_basic + row_basic;
    if total_basic < num_row {
        return Err(SddpError::BasisShapeMismatch {
            num_row,
            total_basic,
            col_basic,
            row_basic,
        });
    }

    let mut excess = total_basic - num_row;
    let mut demotions: u32 = 0;

    for idx in (base_row_count..out.row_status.len()).rev() {
        if excess == 0 {
            break;
        }
        if out.row_status[idx] == BasisStatus::Basic {
            out.row_status[idx] = BasisStatus::Lower;
            excess -= 1;
            demotions += 1;
        }
    }

    Ok(demotions)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod tests {
    use cobre_solver::Basis;
    use cobre_solver::BasisStatus::{Basic as B, Lower as L};

    use super::{
        ReconstructionStats, ReconstructionTarget, enforce_basic_count_invariant,
        reconstruct_basis, reconstruct_basis_uniform_basic,
    };
    use crate::error::SddpError;
    use crate::setup::NodeId;
    use crate::workspace::CapturedBasis;

    /// Build a `CapturedBasis` populated with the requested slot list and
    /// cut-row status sequence. Template rows are all `BASIC`; columns are
    /// all `BASIC` so the column block does not perturb the test focus.
    fn make_stored_basis(
        base_rows: usize,
        num_cols: usize,
        slots: &[u32],
        cut_statuses: &[cobre_solver::BasisStatus],
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
            NodeId(0),
        );

        cb.basis.row_status.clear();
        cb.basis.row_status.resize(base_rows, B);
        cb.basis.row_status.extend_from_slice(cut_statuses);

        cb.basis.col_status.clear();
        cb.basis.col_status.resize(num_cols, B);

        cb.cut_row_slots.extend_from_slice(slots);
        cb.state_at_capture.extend_from_slice(state_at_capture);

        cb
    }

    // -----------------------------------------------------------------------
    // reconstruct_basis — unit tests
    // -----------------------------------------------------------------------

    /// Empty stored slot list + 3 new cut rows. Every new row is classified
    /// `BASIC` and `stats.new_slack == 3`.
    #[test]
    fn returns_basic_for_all_new_cuts() {
        let stored = make_stored_basis(1, 2, &[], &[], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (5, 0.0, vec![0.0, 0.0]),
            (6, 0.0, vec![0.0, 0.0]),
            (7, 0.0, vec![0.0, 0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 16];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 0,
                new_slack: 3,
            },
        );
        // 1 template row + 3 cut rows.
        assert_eq!(out.row_status.len(), 4);
        assert_eq!(&out.row_status[1..], &[B, B, B]);
    }

    /// Two preserved slots — their stored row statuses must be copied
    /// verbatim even when the stored value is `LOWER`. The reconstruction
    /// performs no classifier work, so it must not promote `LOWER` to
    /// `BASIC` itself.
    #[test]
    fn copies_stored_status_for_preserved_slots() {
        let stored = make_stored_basis(1, 2, &[10, 20], &[B, L], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> =
            vec![(10, 0.0, vec![0.0, 0.0]), (20, 0.0, vec![0.0, 0.0])];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_slack: 0,
            },
        );
        assert_eq!(out.row_status.len(), 3);
        assert_eq!(out.row_status[1], B, "slot 10 → stored BASIC");
        assert_eq!(out.row_status[2], L, "slot 20 → stored LOWER");
    }

    /// Mixed case: stored preserves slots `{10, 20, 30, 40}` with statuses
    /// `[L, B, L, B]`; the target LP has 5 cut rows for slots
    /// `{10, 25, 30, 45, 50}`. Slots 10 and 30 are preserved (their stored
    /// statuses copied); slots 25, 45, 50 are new and receive `BASIC`.
    #[test]
    fn mixed_case_preserved_and_new() {
        let stored = make_stored_basis(2, 3, &[10, 20, 30, 40], &[L, B, L, B], &[1.0, 2.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0, 0.0]),
            (25, 0.0, vec![0.0, 0.0]),
            (30, 0.0, vec![0.0, 0.0]),
            (45, 0.0, vec![0.0, 0.0]),
            (50, 0.0, vec![0.0, 0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 2,
            num_cols: 3,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_slack: 3,
            },
            "preserved={{10, 30}}, new_slack={{25, 45, 50}}",
        );
        // 2 template rows + 5 cut rows.
        assert_eq!(out.row_status.len(), 7);
        // Cut row block starts at index 2.
        assert_eq!(out.row_status[2], L, "slot 10 → stored LOWER");
        assert_eq!(out.row_status[3], B, "slot 25 → new → BASIC");
        assert_eq!(out.row_status[4], L, "slot 30 → stored LOWER");
        assert_eq!(out.row_status[5], B, "slot 45 → new → BASIC");
        assert_eq!(out.row_status[6], B, "slot 50 → new → BASIC");
    }

    /// Empty iterator: the cut-row block must be empty and stats must remain
    /// at zero. The template-row block is still populated from the stored
    /// basis.
    #[test]
    fn empty_iterator_preserves_template_rows() {
        let stored = make_stored_basis(3, 2, &[10, 20], &[B, L], &[1.0]);
        let target = ReconstructionTarget {
            base_row_count: 3,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];

        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![];
        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(stats, ReconstructionStats::default());
        // Only the 3 template rows remain in the output basis.
        assert_eq!(out.row_status.len(), 3);
        assert!(
            out.row_status.iter().all(|&s| s == B),
            "template rows must be copied verbatim (all BASIC in this fixture)",
        );
    }

    // -----------------------------------------------------------------------
    // reconstruct_basis_uniform_basic (DCS path) — unit tests
    // -----------------------------------------------------------------------

    /// All 4 cut rows are seeded BASIC; the column block and template rows are
    /// copied verbatim.
    #[test]
    fn uniform_basic_appends_all_basic_cut_rows() {
        // Template rows must be LOWER here, so build the CapturedBasis directly
        // rather than via make_stored_basis (which forces template rows BASIC).
        let mut stored = CapturedBasis::new(3, 2, 2, 0, 0, NodeId(0));
        stored.basis.col_status.clear();
        stored.basis.col_status.extend_from_slice(&[B, B, L]);
        stored.basis.row_status.clear();
        stored.basis.row_status.extend_from_slice(&[L, L]);

        let target = ReconstructionTarget {
            base_row_count: 2,
            num_cols: 3,
        };
        let mut out = Basis::new(0, 0);

        reconstruct_basis_uniform_basic(&stored, target, 4, &mut out);

        assert_eq!(out.col_status, vec![B, B, L]);
        assert_eq!(out.row_status.len(), 6);
        assert_eq!(&out.row_status[0..2], &[L, L]);
        assert_eq!(&out.row_status[2..6], &[B, B, B, B]);
    }

    /// `col_basic = 3` plus 4 BASIC cut rows against `num_row = 6` is an excess of
    /// one, so the repair demotes exactly one trailing BASIC cut row.
    #[test]
    fn uniform_basic_then_invariant_repair_balances() {
        let mut stored = CapturedBasis::new(3, 2, 2, 0, 0, NodeId(0));
        stored.basis.col_status.clear();
        stored.basis.col_status.extend_from_slice(&[B, B, B]); // col_basic = 3
        stored.basis.row_status.clear();
        stored.basis.row_status.extend_from_slice(&[L, L]); // template rows LOWER

        let target = ReconstructionTarget {
            base_row_count: 2,
            num_cols: 3,
        };
        let mut out = Basis::new(0, 0);

        reconstruct_basis_uniform_basic(&stored, target, 4, &mut out);
        assert_eq!(out.col_status, vec![B, B, B]);
        assert_eq!(out.row_status, vec![L, L, B, B, B, B]);

        let num_row = target.base_row_count + 4; // 6
        let demotions = enforce_basic_count_invariant(&mut out, num_row, target.base_row_count)
            .expect("excess is repairable, never a deficit");
        assert_eq!(demotions, 1, "exactly one excess BASIC cut row demoted");

        let col_basic = out.col_status.iter().filter(|&&s| s == B).count();
        let row_basic = out.row_status.iter().filter(|&&s| s == B).count();
        assert_eq!(
            col_basic + row_basic,
            num_row,
            "col_basic + row_basic must equal num_row after repair"
        );
    }

    /// The helper consults none of `stored.cut_row_slots`: a non-empty slot list
    /// must give the same result as one that is cleared.
    #[test]
    fn uniform_basic_ignores_cut_row_slots() {
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };

        // Stored with a populated (but to-be-ignored) cut_row_slots list.
        let with_slots = make_stored_basis(1, 2, &[10, 20, 30], &[B, L, B], &[1.0]);
        assert!(
            !with_slots.cut_row_slots.is_empty(),
            "fixture must have non-empty cut_row_slots to make the test meaningful"
        );
        let mut out_with = Basis::new(0, 0);
        reconstruct_basis_uniform_basic(&with_slots, target, 3, &mut out_with);

        // Same stored basis with the slots cleared.
        let mut without_slots = make_stored_basis(1, 2, &[10, 20, 30], &[B, L, B], &[1.0]);
        without_slots.cut_row_slots.clear();
        let mut out_without = Basis::new(0, 0);
        reconstruct_basis_uniform_basic(&without_slots, target, 3, &mut out_without);

        assert_eq!(out_with.col_status, out_without.col_status);
        assert_eq!(out_with.row_status, out_without.row_status);
    }

    /// `cut_row_count = 0` appends no cut rows; only the template rows remain.
    #[test]
    fn uniform_basic_zero_cut_rows() {
        let stored = make_stored_basis(3, 2, &[10, 20], &[B, L], &[1.0]);
        let target = ReconstructionTarget {
            base_row_count: 3,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);

        reconstruct_basis_uniform_basic(&stored, target, 0, &mut out);

        assert_eq!(out.row_status.len(), 3);
        assert_eq!(out.col_status.len(), 2);
    }

    // -----------------------------------------------------------------------
    // enforce_basic_count_invariant — deficit detection
    // -----------------------------------------------------------------------

    /// A basis whose basic count falls short of `num_row` is rejected with
    /// `BasisShapeMismatch` carrying all four counters — never silently accepted,
    /// never "repaired" by promotion.
    #[test]
    fn deficit_is_rejected_as_shape_mismatch() {
        let mut out = Basis::new(0, 0);
        out.col_status.extend_from_slice(&[B, L, L]);
        out.row_status.extend_from_slice(&[L, L, L, B]);

        let err = enforce_basic_count_invariant(&mut out, 4, 2)
            .expect_err("total_basic = 2 against num_row = 4 is a deficit");

        match err {
            SddpError::BasisShapeMismatch {
                num_row,
                total_basic,
                col_basic,
                row_basic,
            } => {
                assert_eq!(num_row, 4);
                assert_eq!(col_basic, 1);
                assert_eq!(row_basic, 1);
                assert_eq!(total_basic, 2);
            }
            other => panic!("expected SddpError::BasisShapeMismatch, got {other:?}"),
        }
        assert_eq!(
            out.row_status,
            vec![L, L, L, B],
            "a rejected basis must not be mutated"
        );
    }

    /// `stored` is a *valid* basis for its own 6-row LP (`col_basic(4) +
    /// row_basic(2) == 6`), but the target LP carries two extra template rows, so
    /// the template copy swallows two of stored's slack cut rows and the
    /// reconstruction lands two basics short of the target's 8 rows.
    #[test]
    fn reconstruct_from_grown_base_row_count_yields_deficit() {
        let mut stored = CapturedBasis::new(5, 6, 2, 4, 0, NodeId(0));
        stored.basis.col_status.clear();
        stored.basis.col_status.extend_from_slice(&[B, B, B, B, L]);
        stored.basis.row_status.clear();
        stored
            .basis
            .row_status
            .extend_from_slice(&[B, B, L, L, L, L]);
        stored.cut_row_slots.extend_from_slice(&[10, 20, 30, 40]);

        let target = ReconstructionTarget {
            base_row_count: 4,
            num_cols: 5,
        };
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0; 5]),
            (20, 0.0, vec![0.0; 5]),
            (30, 0.0, vec![0.0; 5]),
            (40, 0.0, vec![0.0; 5]),
        ];
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        let num_row = out.row_status.len();
        assert_eq!(num_row, 8, "4 template rows + 4 cut rows");

        let err = enforce_basic_count_invariant(&mut out, num_row, target.base_row_count)
            .expect_err("a base_row_count divergence must surface as a shape mismatch");
        match err {
            SddpError::BasisShapeMismatch {
                num_row,
                total_basic,
                ..
            } => {
                assert_eq!(num_row, 8);
                assert_eq!(total_basic, 6, "stored's own LP row count, two short");
            }
            other => panic!("expected SddpError::BasisShapeMismatch, got {other:?}"),
        }
    }

    /// `total_basic == num_row` is the balanced case: no demotions, no error, and
    /// the basis is returned untouched.
    #[test]
    fn balanced_basic_count_is_a_no_op() {
        let mut out = Basis::new(0, 0);
        out.col_status.extend_from_slice(&[B, B]);
        out.row_status.extend_from_slice(&[L, B, L]);

        let demotions = enforce_basic_count_invariant(&mut out, 3, 1)
            .expect("total_basic == num_row is balanced, not a deficit");

        assert_eq!(demotions, 0);
        assert_eq!(out.row_status, vec![L, B, L]);
    }
}
