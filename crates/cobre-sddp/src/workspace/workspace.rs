//! Per-thread solver workspace, workspace pool, and per-scenario basis store.
//!
//! [`SolverWorkspace`] bundles all mutable per-thread resources needed for one LP solve sequence.
//! [`WorkspacePool`] allocates one workspace per worker thread.
//! [`BasisStore`] provides per-scenario, per-stage basis storage for warm-starting LP solves.

use std::ops::Range;

use cobre_core::WorkerPhaseTimings;
use cobre_solver::{Basis, BasisStatus, ProfiledSolver, SolverInterface, SolverStatistics};

use crate::SddpError;
use crate::SddpError::Validation;
use crate::backward::{OpeningOutcome, StagedCut};
use crate::context::{StageContext, TrainingContext};
use crate::dcs::DcsSolveScratch;
use crate::lp::builder::{PatchBuffer, StageGeometry};
use crate::risk_measure::{BackwardOutcome, RiskMeasureScratch};
use crate::setup::{NodeId, NodePos};
use crate::solve::partition;
use crate::solver_stats::SolverStatsDelta;

// ---------------------------------------------------------------------------
// CapturedBasis
// ---------------------------------------------------------------------------

/// A solver basis augmented with slot-tracking metadata for cut-set-aware
/// warm-start reconstruction.
///
/// `cut_row_slots` maps each captured cut row to the cut pool slot it occupied
/// at capture, so reconstruction binds stored statuses to current LP cut rows by
/// slot identity rather than row count. Pre-sized via [`CapturedBasis::new`] so
/// the forward capture site reuses one `CapturedBasis` across iterations without
/// reallocation.
///
/// `node_id` tags the node this basis was captured at; the apply sites
/// (`run_stage_solve`, `lazy_solve_preloaded`) treat a stored basis whose
/// `node_id` differs from the node being solved as cold, never warm-started
/// from it — a resampled path visiting a different node than the one that
/// captured the basis. Carried on the broadcast wire (see
/// [`CapturedBasis::to_broadcast_payload`]), so the decode side is
/// self-describing rather than positionally filled.
#[derive(Clone, Debug)]
pub struct CapturedBasis {
    /// Row and column statuses from the solver.
    pub basis: Basis,
    /// Number of template (non-cut) LP rows at capture time; row statuses at
    /// `0..base_row_count` are template rows.
    pub base_row_count: usize,
    /// Cut pool slot for each cut row in `basis.row_status[base_row_count..]`;
    /// length equals `basis.row_status.len() - base_row_count` when populated.
    pub cut_row_slots: Vec<u32>,
    /// State vector `x_hat` at capture, the operating point for evaluating newly
    /// added cuts on the backward warm-start.
    pub state_at_capture: Vec<f64>,
    /// Node this basis was captured at (`NodeGraph::node_ids[node]`).
    pub node_id: NodeId,
}

/// Wire-format version for `CapturedBasis` broadcast payloads.
///
/// Stored as the second `i32` in every `Some`-path payload, immediately after
/// the presence sentinel (`1_i32`). Bump this constant and update
/// `try_from_broadcast_payload` whenever the field layout changes.
///
/// Version 2 carries: the generating/capture `node_id` (making each entry
/// self-describing rather than positionally filled by the caller), column
/// statuses, template-row statuses, cut-row statuses, `cut_row_slots`, and
/// `state_at_capture`. `cut_row_slots` is the load-bearing reconstruction key —
/// `build_slot_lookup` binds stored cut-row statuses to target-LP cut rows by
/// slot id. `state_at_capture` is not read by any current reconstruction
/// reader; it is retained for diagnostics and to allow re-introducing a
/// state-dependent reuse policy without a wire-format change. The DCS arm
/// captures no basis by design (a captured basis would describe the frozen
/// layout, not the DCS resident subset — see `StageOpeningSolver::solve_lazy`),
/// so there the field is never part of a consumed warm-start.
pub const BASIS_BROADCAST_FORMAT_TAG: i32 = 2;

/// Converts via [`BasisStatus::to_discriminant_code`], the single owner of the numeric
/// mapping (injective for round-trip). Payload byte layout owned by the
/// `to_broadcast_payload`/`try_from_broadcast_payload` pair.
fn basis_status_to_wire_code(status: BasisStatus) -> i32 {
    i32::from(status.to_discriminant_code())
}

/// Out-of-range codes decode to [`BasisStatus::Nonbasic`] via
/// [`BasisStatus::from_discriminant_code`].
fn basis_status_from_wire_code(code: i32) -> BasisStatus {
    u8::try_from(code).map_or(BasisStatus::Nonbasic, BasisStatus::from_discriminant_code)
}

impl CapturedBasis {
    /// Pre-sizes `cut_row_slots` and `state_at_capture` but leaves them empty (length 0).
    #[must_use]
    pub fn new(
        num_cols: usize,
        num_rows: usize,
        base_row_count: usize,
        cut_slot_capacity: usize,
        n_state: usize,
        node_id: NodeId,
    ) -> Self {
        Self {
            basis: Basis::new(num_cols, num_rows),
            base_row_count,
            cut_row_slots: Vec::with_capacity(cut_slot_capacity),
            state_at_capture: Vec::with_capacity(n_state),
            node_id,
        }
    }

    /// Retains capacity. Does not touch `basis` (next `get_basis` overwrites it).
    pub fn clear(&mut self) {
        self.cut_row_slots.clear();
        self.state_at_capture.clear();
    }

    /// Append this basis's wire-format payload to the output buffers.
    ///
    /// Single owner of the wire format; any change must update both this method and
    /// [`CapturedBasis::try_from_broadcast_payload`] together.
    ///
    /// Pushes the following into `i32_buf` in order:
    /// - `1_i32` sentinel (present)
    /// - [`BASIS_BROADCAST_FORMAT_TAG`] as `i32` (format tag)
    /// - `node_id` as `i32` (the generating/capture node — self-describing
    ///   header, so the decode side no longer fills it positionally)
    /// - `col_status.len()` as `i32`
    /// - `row_status.len()` as `i32`
    /// - `base_row_count` as `i32`
    /// - `cut_row_slots.len()` as `i32`
    /// - `state_at_capture.len()` as `i32`
    /// - `col_status[..]` mapped through `basis_status_to_wire_code`
    /// - `row_status[..]` mapped through `basis_status_to_wire_code`
    /// - `cut_row_slots[..]` cast to `i32`
    ///
    /// Pushes `state_at_capture[..]` into `f64_buf`.
    ///
    /// The callers (currently `broadcast_basis_cache`) are
    /// responsible for writing the `0_i32` "no basis" sentinel
    /// when `Option<CapturedBasis>` is `None`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn to_broadcast_payload(&self, i32_buf: &mut Vec<i32>, f64_buf: &mut Vec<f64>) {
        i32_buf.push(1_i32);
        i32_buf.push(BASIS_BROADCAST_FORMAT_TAG);
        i32_buf.push(self.node_id.0);
        i32_buf.push(self.basis.col_status.len() as i32);
        i32_buf.push(self.basis.row_status.len() as i32);
        i32_buf.push(self.base_row_count as i32);
        i32_buf.push(self.cut_row_slots.len() as i32);
        i32_buf.push(self.state_at_capture.len() as i32);
        for &status in &self.basis.col_status {
            i32_buf.push(basis_status_to_wire_code(status));
        }
        for &status in &self.basis.row_status {
            i32_buf.push(basis_status_to_wire_code(status));
        }
        // u32 -> i32: slot values are LP pool indices (always
        // non-negative) that fit comfortably in i32.
        for &slot in &self.cut_row_slots {
            i32_buf.push(slot as i32);
        }
        f64_buf.extend_from_slice(&self.state_at_capture);
    }

    /// Deserialise one stage's payload from the two wire-format
    /// buffers, advancing the cursors past the consumed bytes.
    ///
    /// Returns `Ok(None)` when the sentinel read is `0` (no basis
    /// for this stage). Returns `Ok(Some(captured))` when the
    /// sentinel is `1`, the format tag matches [`BASIS_BROADCAST_FORMAT_TAG`],
    /// and the payload is complete.
    ///
    /// # Layout (`Some` path)
    ///
    /// Reads from `i32_buf` in order:
    /// - `1_i32` sentinel (present)
    /// - [`BASIS_BROADCAST_FORMAT_TAG`] as `i32` (format tag)
    /// - `node_id` as `i32` (the generating/capture node, read straight into
    ///   the reconstructed [`CapturedBasis::node_id`])
    /// - `col_status.len()` as `i32`
    /// - `row_status.len()` as `i32`
    /// - `base_row_count` as `i32`
    /// - `cut_row_slots.len()` as `i32`
    /// - `state_at_capture.len()` as `i32`
    /// - `col_status[..]` decoded through `basis_status_from_wire_code`
    /// - `row_status[..]` decoded through `basis_status_from_wire_code`
    /// - `cut_row_slots[..]` (stored as `i32`, cast back to `u32`)
    ///
    /// Reads `state_at_capture[..]` from `f64_buf`.
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Validation` if the `i32_buf` or
    /// `f64_buf` is truncated at any of the bounded reads
    /// (sentinel, format tag, `node_id` + five length fields, `col_status`,
    /// `row_status`, `cut_row_slots`, `state_at_capture`), or if the format-tag
    /// field does not match [`BASIS_BROADCAST_FORMAT_TAG`]. The error message
    /// names the affected stage and the expected vs. available byte count.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    pub fn try_from_broadcast_payload(
        stage: usize,
        i32_buf: &[i32],
        i32_cursor: &mut usize,
        f64_buf: &[f64],
        f64_cursor: &mut usize,
    ) -> Result<Option<Self>, SddpError> {
        if *i32_cursor >= i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated at stage {stage} \
                 (pos={}, len={})",
                *i32_cursor,
                i32_buf.len()
            )));
        }
        let sentinel = i32_buf[*i32_cursor];
        *i32_cursor += 1;

        if sentinel == 0 {
            return Ok(None);
        }

        // Format tag is present only on the Some path, after the sentinel.
        if *i32_cursor >= i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading version at stage {stage}"
            )));
        }
        let version = i32_buf[*i32_cursor];
        *i32_cursor += 1;
        if version != BASIS_BROADCAST_FORMAT_TAG {
            return Err(Validation(format!(
                "try_from_broadcast_payload: unsupported wire version {version} at stage \
                 {stage} (expected {BASIS_BROADCAST_FORMAT_TAG})"
            )));
        }

        if *i32_cursor + 6 > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading node_id + lengths at stage \
                 {stage}"
            )));
        }
        let node_id = NodeId(i32_buf[*i32_cursor]);
        *i32_cursor += 1;
        let col_len = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;
        let row_len = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;
        let base_row_count = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;
        let cut_slot_count = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;
        let state_len = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;

        if *i32_cursor + col_len > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading col_status at stage \
                 {stage} (need {col_len}, have {})",
                i32_buf.len() - *i32_cursor
            )));
        }
        let col_status: Vec<BasisStatus> = i32_buf[*i32_cursor..*i32_cursor + col_len]
            .iter()
            .map(|&code| basis_status_from_wire_code(code))
            .collect();
        *i32_cursor += col_len;

        if *i32_cursor + row_len > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading row_status at stage \
                 {stage} (need {row_len}, have {})",
                i32_buf.len() - *i32_cursor
            )));
        }
        let row_status: Vec<BasisStatus> = i32_buf[*i32_cursor..*i32_cursor + row_len]
            .iter()
            .map(|&code| basis_status_from_wire_code(code))
            .collect();
        *i32_cursor += row_len;

        if *i32_cursor + cut_slot_count > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading cut_row_slots at stage \
                 {stage} (need {cut_slot_count}, have {})",
                i32_buf.len() - *i32_cursor
            )));
        }
        // cast_sign_loss: values were originally u32 LP pool indices cast to
        // i32 on the pack side; casting back to u32 is lossless.
        let cut_row_slots: Vec<u32> = i32_buf[*i32_cursor..*i32_cursor + cut_slot_count]
            .iter()
            .map(|&v| v as u32)
            .collect();
        *i32_cursor += cut_slot_count;

        if *f64_cursor + state_len > f64_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: f64 buffer truncated reading state_at_capture at \
                 stage {stage} (need {state_len}, have {})",
                f64_buf.len() - *f64_cursor
            )));
        }
        let state_at_capture = f64_buf[*f64_cursor..*f64_cursor + state_len].to_vec();
        *f64_cursor += state_len;

        Ok(Some(Self {
            basis: Basis {
                col_status,
                row_status,
            },
            base_row_count,
            cut_row_slots,
            state_at_capture,
            node_id,
        }))
    }
}

/// Execution-count sizing hints shared by [`SolverWorkspace`], [`WorkspacePool`],
/// and `ScratchBuffers` constructors.
///
/// Grouping these dimensions into one struct keeps constructor argument counts
/// within the clippy budget. Every study-shape count these constructors need
/// (hydro count, PAR order, load buses, block counts, bucket/anticipated
/// counts, `n_state`, the downstream PAR order) is read from the
/// `training_ctx`/`stage_ctx` owners each constructor also takes; this struct
/// carries only counts with no other owner.
///
/// Any dimension may be `0`: the corresponding buffer starts empty and grows
/// on-demand via the growth-only resize semantics. Pass `0` for
/// `max_openings`/`initial_pool_capacity` on simulation-only workspaces, and
/// for `max_local_fwd` on backward-only or simulation-only workspaces.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkspaceSizing {
    /// Maximum openings across all successor stages; pre-sizes
    /// `BackwardAccumulators::outcomes`.
    pub max_openings: usize,
    /// Initial cut pool capacity; pre-sizes
    /// `BackwardAccumulators::slot_increments`.
    pub initial_pool_capacity: usize,
    /// Maximum forward-pass scenarios assigned to this rank; pre-sizes
    /// `ScratchBuffers::trajectory_costs_buf`.
    pub max_local_fwd: usize,
    /// Whether `ScratchBuffers::raw_noise_buf`/`corr_scratch` pre-reserve to
    /// the study's noise dimension, or start empty and grow on demand.
    pub noise: NoisePreallocation,
}

/// [`WorkspaceSizing::noise`]'s two preallocation strategies for
/// `ScratchBuffers::raw_noise_buf`/`corr_scratch`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NoisePreallocation {
    /// Start empty; grow on first use.
    #[default]
    OnDemand,
    /// Pre-reserve to `TrainingContext::stochastic`'s noise dimension.
    StochasticDim,
}

/// Pre-allocated accumulators for the backward pass trial-point loop.
///
/// All buffers grow monotonically (never shrink) and are reused across stages
/// and trial points without per-call allocation. Each rayon worker owns an
/// exclusive [`SolverWorkspace`] and therefore an exclusive instance, so the
/// per-worker fields need no synchronisation; the sequential merge phase after
/// the parallel region sums their contributions.
#[derive(Default)]
pub(crate) struct BackwardAccumulators {
    /// Per-opening backward outcomes, grown to the maximum `n_openings` seen.
    pub(crate) outcomes: Vec<BackwardOutcome>,
    /// Per-slot binding count over the node's successor pool regions, concatenated
    /// per child (`SuccessorChild::metadata_offset + slot`), so a fan's distinct
    /// sibling pools never collide on a shared slot index; zeroed per trial point.
    pub(crate) slot_increments: Vec<u64>,
    /// Aggregated cut coefficients (`n_state` entries), written by
    /// `aggregate_weighted_into` then copied into [`agg_arena`](Self::agg_arena)
    /// at the trial point's slot.
    pub(crate) agg_coefficients: Vec<f64>,
    /// Per-worker flat arena holding every staged cut's coefficient vector for
    /// the current stage.
    ///
    /// Sized lazily to at least `(end_m - start_m) * n_state` `f64`s; slot `i` of
    /// the trial point with local index `local_idx = m - start_m` lives at
    /// `agg_arena[local_idx * n_state + i]`. Each [`StagedCut`] stores a
    /// `Range<usize>` into this arena instead of an owned `Vec<f64>`; the FCF
    /// merge reads the slice after the parallel region returns. Overwritten per
    /// trial point before any read, so no zero-fill is required.
    pub(crate) agg_arena: Vec<f64>,
    /// Per-worker metadata sync contribution over the node's successor pool
    /// regions, concatenated per child (same layout as `slot_increments`). Zeroed
    /// once per stage (not per trial point); each pool's own region slice is merged
    /// into `metadata_sync_buf` and synced separately.
    pub(crate) metadata_sync_contribution: Vec<u64>,
    /// Per-worker per-opening solver-statistics accumulator (length
    /// `n_openings`). Re-initialised once per stage; merged into the per-stage
    /// `Vec<SolverStatsDelta>` in `BackwardResult::stage_stats`.
    pub(crate) per_opening_stats: Vec<SolverStatsDelta>,
    /// Per-block `simplex_iterations` sum for the by-node
    /// scheduler's pivot accumulator, index `b` in `[0, n_blocks)` for
    /// the current stage. Re-initialised once per stage (never per trial
    /// point); merged into `ByNodeScratch::block_pivots` after the
    /// parallel region.
    pub(crate) block_pivot_sum: Vec<u64>,
    /// Per-block solved-opening count, parallel to `block_pivot_sum`.
    pub(crate) block_pivot_count: Vec<u64>,
    /// Per-opening scratch for the incoming-state columns' reduced costs; grows to `indexer.n_state`.
    pub(crate) state_duals_buf: Vec<f64>,
    /// Per-opening scratch for cut-row duals; grows to
    /// `succ.num_cuts_at_successor`.
    pub(crate) cut_duals_buf: Vec<f64>,
    /// Reusable `[hydro | load-bus | NCS]` opening-noise buffer for an External
    /// successor node, assembled from the external libraries at the node's
    /// column. Empty and never touched on the generated/sampled path.
    pub(crate) external_noise_buf: Vec<f64>,
    /// Per-worker staging buffer for cuts produced within one stage; drained at
    /// the rayon closure boundary so ownership can cross the closure return.
    pub(crate) staged_cuts_buf: Vec<StagedCut>,
    /// Per-worker by-node scheduler out-buffer (`process_stage_backward_by_node`'s
    /// claim loop). Starts empty — zero footprint under the default `ByScenario`
    /// scheduler, which never touches it. The outer `Vec` grows once, per worker,
    /// to a stage's claimed-unit high-water mark (indexed by a per-stage count,
    /// never `clear`ed/`drain`ed); each slot's `coefficients` resizes to the
    /// CURRENT stage's `cut_n_state` before every write, so capacity — not
    /// length — is the no-reallocation invariant once every `cut_n_state` value
    /// in the run has been visited (trivially within iteration 1).
    pub(crate) opening_outcomes_buf: Vec<OpeningOutcome>,
    /// Scratch for `CVaR` weight computation; the internal `Vec`s grow to
    /// `n_openings` on the first `CVaR` call. Never accessed under
    /// `RiskMeasure::Expectation`.
    pub(crate) risk_scratch: RiskMeasureScratch,
    /// Reusable scratch for the dynamic-cut-selection lazy solve, only touched
    /// when that method is active.
    pub(crate) dcs_solve: DcsSolveScratch,
    /// Metadata-seeded initial resident-cut set for the dynamic-cut-selection
    /// lazy solve; refilled per (trial point, opening).
    pub(crate) dcs_initial_resident: Vec<u32>,
    /// Per-opening before-solve statistics snapshot, filled via `statistics_into`
    /// (reusing the histogram allocation) and diffed against `stats_after_buf`.
    pub(crate) stats_before_buf: SolverStatistics,
    /// Per-opening after-solve statistics snapshot; counterpart to
    /// `stats_before_buf`.
    pub(crate) stats_after_buf: SolverStatistics,
}

impl BackwardAccumulators {
    /// Allocate accumulators pre-sized from the given dimensions; any may be
    /// `0`, in which case those buffers start empty and grow lazily.
    pub(crate) fn new(max_openings: usize, initial_pool_capacity: usize, n_state: usize) -> Self {
        let outcomes = (0..max_openings)
            .map(|_| BackwardOutcome {
                intercept: 0.0,
                coefficients: vec![0.0_f64; n_state],
                objective_value: 0.0,
            })
            .collect();
        let mut dcs_solve = DcsSolveScratch::default();
        dcs_solve.reserve(n_state, initial_pool_capacity);
        Self {
            outcomes,
            slot_increments: vec![0u64; initial_pool_capacity],
            agg_coefficients: vec![0.0_f64; n_state],
            agg_arena: Vec::new(),
            metadata_sync_contribution: vec![0u64; initial_pool_capacity],
            per_opening_stats: Vec::new(),
            block_pivot_sum: Vec::new(),
            block_pivot_count: Vec::new(),
            state_duals_buf: Vec::new(),
            cut_duals_buf: Vec::new(),
            external_noise_buf: Vec::new(),
            staged_cuts_buf: Vec::new(),
            opening_outcomes_buf: Vec::new(),
            risk_scratch: RiskMeasureScratch::new(),
            dcs_solve,
            dcs_initial_resident: Vec::with_capacity(initial_pool_capacity),
            stats_before_buf: SolverStatistics::default(),
            stats_after_buf: SolverStatistics::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// ByNodeScratch
// ---------------------------------------------------------------------------

/// Rank-level pre-allocated scratch for the by-node scheduler
/// (`training.parallelism.backward_scheduler`), held on `BackwardPassState`
/// and reused across every backward stage for the lifetime of a training run.
///
/// Empty (`arena` capacity `0`) unless `ByNode` could be the scheduler a run
/// actually dispatches to: sized by `BackwardPassState::resize_by_node_scratch`
/// (called from `set_scheduler`) whenever the configured scheduler is
/// literally `ByNode`. Untouched by the enumerated traversal — its per-node
/// solve (`BackwardPassState::run_enumerated_backward`) never dispatches
/// through the by-node scheduler at all. Never resized on the hot path
/// (sddp.md "By-node scheduler is warm-start-only").
#[derive(Default)]
pub(crate) struct ByNodeScratch {
    /// Per-`(m, ω)` outcome arena, addressed `arena[m * n_openings + omega]` for
    /// the CURRENT stage's `n_openings` — a prefix of the full
    /// `max_local_fwd * bwd_max_openings`-outcome allocation; a stage with fewer
    /// openings than the run's max simply uses less of it. Mirrors
    /// `BackwardAccumulators::agg_arena`'s grow-once/overwrite-before-read
    /// discipline: every `(m, ω)` in the active region is written by
    /// `by_node_finish`'s scatter before the aggregation loop reads it, so no clear
    /// pass is required between stages. Each entry's `coefficients` starts at
    /// `n_state` length but is resized (never reallocated — capacity is the
    /// invariant) to the CURRENT stage's `cut_n_state` during that scatter, a
    /// stage whose successor disables a state group projecting to fewer slots.
    pub(crate) arena: Vec<BackwardOutcome>,
    /// `CVaR` weight-computation scratch for `by_node_finish`'s per-trial-point
    /// aggregation loop.
    pub(crate) risk_scratch: RiskMeasureScratch,
    /// Aggregated cut coefficients buffer for `by_node_finish`'s per-trial-point
    /// aggregation loop; starts at `n_state` length, resized (within reserved
    /// capacity) to each stage's `cut_n_state` — see [`Self::arena`].
    pub(crate) coeffs_buf: Vec<f64>,
    /// Per-`(generating node, block-index)` `(simplex_iterations sum,
    /// solved-opening count)` pivot accumulator, addressed via
    /// [`Self::block_pivot_row`] — never re-derived at a call site. Keyed by
    /// the GENERATING node's canonical position, never the successor stage
    /// (`merge_block_pivots`'s own contract): sibling fan nodes at one level
    /// own distinct rows, so each computes its hardest-first order from its
    /// own history. Aggregated over every trial point `m` claimed for that
    /// block, never keyed per-`m` (the wrong-but-plausible variant, since
    /// resampled trial points make per-`m` hardness noise where the
    /// opening-block component is iteration-stable). Reset to `(0, 0)` at the
    /// start of every `BackwardPassState::run` call, never per stage.
    pub(crate) block_pivots: Vec<(u64, u64)>,
    /// Double-buffer companion to [`Self::block_pivots`]: holds the
    /// PREVIOUS iteration's fully-merged row, swapped in at the top of
    /// `BackwardPassState::run` (never per stage) so the hardest-first claim-order
    /// computation reads a row the current sweep has not yet started
    /// overwriting. Reading `block_pivots` directly during the sweep is the
    /// wrong-but-compiling alternative it rules out — that buffer is
    /// reset-then-partially-filled mid-run, yielding zeros or a half-filled
    /// row. Same shape as `block_pivots`.
    pub(crate) block_pivots_prev: Vec<(u64, u64)>,
    /// Row width of `block_pivots`/`block_pivots_prev` (block slots per
    /// generating node) — the run's max opening count, an upper bound on any
    /// node's actual block count (block size >= 1 implies
    /// `n_blocks <= n_openings <= n_blocks_max`).
    pub(crate) n_blocks_max: usize,
    /// Hardest-first claim-order scratch: the successor stage's sorted (or identity)
    /// `0..n_blocks` block-index permutation, recomputed into this buffer
    /// once per stage from [`Self::block_pivots_prev`] — never a fresh `Vec`
    /// per stage. Length `n_blocks_max`, an upper bound on any stage's
    /// `n_blocks` (see [`Self::n_blocks_max`]).
    pub(crate) block_order: Vec<u32>,
}

impl ByNodeScratch {
    /// Allocate the arena and aggregation scratch for `ByNode`: `arena`
    /// holds `max_local_fwd * bwd_max_openings` outcomes, each with an
    /// `n_state`-length coefficient vector; `block_pivots` and
    /// `block_pivots_prev` each hold `row_axis_len * bwd_max_openings`
    /// accumulator cells, `row_axis_len` an upper bound on the GENERATING
    /// node count (see [`Self::block_pivots`] — never a stage count, though
    /// it coincides with one on a chain); `block_order` holds
    /// `bwd_max_openings` claim-order scratch slots.
    #[must_use]
    pub(crate) fn sized(
        max_local_fwd: usize,
        bwd_max_openings: usize,
        n_state: usize,
        row_axis_len: usize,
    ) -> Self {
        let arena = (0..max_local_fwd * bwd_max_openings)
            .map(|_| BackwardOutcome {
                intercept: 0.0,
                coefficients: vec![0.0_f64; n_state],
                objective_value: 0.0,
            })
            .collect();
        Self {
            arena,
            risk_scratch: RiskMeasureScratch::new(),
            coeffs_buf: vec![0.0_f64; n_state],
            block_pivots: vec![(0u64, 0u64); row_axis_len * bwd_max_openings],
            block_pivots_prev: vec![(0u64, 0u64); row_axis_len * bwd_max_openings],
            n_blocks_max: bwd_max_openings,
            block_order: vec![0u32; bwd_max_openings],
        }
    }

    /// `block_pivots`/`block_pivots_prev`'s row range for the GENERATING
    /// node `node_pos` — single owner of the `node_pos * n_blocks_max`
    /// row-start arithmetic (`NodePos` carries no arithmetic of its own), so
    /// a caller never recomputes it independently of this method.
    #[must_use]
    pub(crate) fn block_pivot_row(&self, node_pos: NodePos, n_blocks: usize) -> Range<usize> {
        let start = node_pos.0 * self.n_blocks_max;
        start..start + n_blocks
    }
}

/// Pre-allocated scratch buffers for noise transformation and simulation,
/// passed by `&mut` to the transformation functions in `noise.rs`.
///
/// `pub` type with every field kept `pub(crate)`: [`crate::lower_bound::evaluate_lower_bound`]'s
/// public signature must name the type, but an external caller must still go
/// through [`ScratchBuffers::new`] rather than construct or mutate one field-by-field.
#[allow(clippy::struct_field_names)]
pub struct ScratchBuffers {
    pub(crate) inflow_m3s_buf: Vec<f64>,
    pub(crate) lag_matrix_buf: Vec<f64>,
    pub(crate) par_inflow_buf: Vec<f64>,
    pub(crate) eta_floor_buf: Vec<f64>,
    pub(crate) zero_targets_buf: Vec<f64>,
    pub(crate) ncs_col_upper_buf: Vec<f64>,
    pub(crate) ncs_col_lower_buf: Vec<f64>,
    pub(crate) ncs_col_indices_buf: Vec<usize>,
    // Active-subset gather targets, parallel to `ncs_col_indices_buf`:
    // `transform_ncs_noise` writes `ncs_col_{lower,upper}_buf` in full slot order
    // (incl. slots inactive at this stage); the patch gathers only active slots'
    // bounds here so set-bounds gets equal-length buffers. Passing the slot-order
    // buffers verbatim would over-feed the call at a strict-subset stage.
    pub(crate) ncs_col_lower_active_buf: Vec<f64>,
    pub(crate) ncs_col_upper_active_buf: Vec<f64>,
    /// Per-stage NCS column start `ncs_col_indices_buf` was last built for;
    /// `usize::MAX` initially so the first patch always rebuilds.
    ///
    /// The lazy rebuild must key on the **start**, not the length: two stages can
    /// share the same `active_count * n_blks` length yet address different columns
    /// (mid-horizon commissioning / differing block counts), so keying on length
    /// would write `set_col_bounds` onto the wrong LP columns. Comparing
    /// `ncs_col_indices_buf[0]` is also insufficient — a simultaneous change of
    /// start and first-active-local can collide on the first index while the rest
    /// of the buffer is stale.
    pub(crate) last_ncs_col_start: usize,
    // NCS upper bounds for simulation extraction, length `n_ncs * n_blks` in
    // active-local column order (NOT slot order): template `col_upper` overwritten
    // per stochastic active column's block with the realized availability from
    // `ncs_col_upper_buf`. `ncs_col_upper_buf` (slot order) cannot be consumed
    // directly because extraction indexes by active-local column.
    pub(crate) ncs_col_upper_extract_buf: Vec<f64>,
    pub(crate) load_rhs_buf: Vec<f64>,
    pub(crate) row_lower_buf: Vec<f64>,
    pub(crate) z_inflow_rhs_buf: Vec<f64>,
    pub(crate) effective_eta_buf: Vec<f64>,
    pub(crate) unscaled_primal: Vec<f64>,
    pub(crate) unscaled_dual: Vec<f64>,
    pub(crate) lag_accumulator: Vec<f64>,
    pub(crate) lag_weight_accum: Vec<f64>,
    pub(crate) downstream_accumulator: Vec<f64>,
    pub(crate) downstream_weight_accum: f64,
    // Slot-major `[slot * hydro_count + hydro]`; slot 0 = oldest completed
    // quarter, slot n-1 = most recent.
    pub(crate) downstream_completed_lags: Vec<f64>,
    pub(crate) downstream_n_completed: usize,
    /// Maps each cut pool slot to its position in the stored
    /// `CapturedBasis::cut_row_slots`, so reconstruction locates any active cut's
    /// row status in O(1). Pre-filled with `None` to `initial_pool_capacity`.
    pub(crate) recon_slot_lookup: Vec<Option<u32>>,

    /// Per-worker forward-pass trajectory-cost accumulator, taken via
    /// `std::mem::take` at the worker boundary. Named `..._buf` (not
    /// `trajectory_costs`) to avoid collision with the identically-named field on
    /// `ForwardWorkerResult`.
    pub(crate) trajectory_costs_buf: Vec<f64>,

    /// Per-worker raw-noise scratch for the forward-pass sampler and simulation
    /// worker loop.
    pub(crate) raw_noise_buf: Vec<f64>,

    /// Per-worker gather and correlate scratch for a correlation group wider
    /// than the applier's stack bound, sized `2 * noise_dim`: a group's width
    /// never exceeds its class dimension, hence never `noise_dim`, and the
    /// applier needs one gather array and one correlate array of that width.
    pub(crate) corr_scratch: Vec<f64>,

    /// Per-trajectory sampled-walk node carrier: `current_node_buf[local_m]` is
    /// trajectory `local_m`'s [`crate::setup::node_graph::NodeGraph`] position at
    /// the stage the forward loop is about to visit, root-initialized then
    /// advanced by the transition draw. Resized (never reallocated once at
    /// capacity) at each `run_forward_worker` call.
    pub(crate) current_node_buf: Vec<NodePos>,
}

/// All per-thread mutable resources required for one LP solve sequence.
///
/// Each instance is exclusively owned by one worker thread — no shared state
/// between workspaces — and is distributed by mutable reference from a
/// [`WorkspacePool`]. `rank` and `worker_id` are assigned once at
/// [`WorkspacePool::new`] and never change.
pub struct SolverWorkspace<S: SolverInterface> {
    /// MPI rank that owns this workspace. Stable across the run.
    pub rank: i32,
    /// Rayon worker index within this rank's pool, range `0..n_workers_local`.
    /// Stable across the run.
    pub worker_id: i32,
    /// LP solver, wrapped in [`ProfiledSolver`] so per-phase configuration
    /// applies via [`ProfiledSolver::set_profile`] without touching call sites
    /// that use the [`SolverInterface`] trait.
    pub solver: ProfiledSolver<S>,
    /// Pre-allocated row-bound patch buffer.
    pub patch_buf: PatchBuffer,
    /// Scratch buffer for the current state vector.
    pub current_state: Vec<f64>,
    /// Pre-allocated scratch buffers for noise transformation and simulation.
    pub(crate) scratch: ScratchBuffers,
    /// Pre-allocated destination basis for [`reconstruct_basis`], filled in-place
    /// from the read-only [`CapturedBasis`] in [`BasisStore`] then passed to
    /// `solve(Some(&basis))`. Sized via [`WorkspacePool::resize_scratch_bases`]
    /// so reconstruction never reallocates on the hot path.
    ///
    /// [`reconstruct_basis`]: crate::basis_reconstruct::reconstruct_basis
    pub(crate) scratch_basis: Basis,
    /// Pre-allocated accumulators for the backward pass trial-point loop.
    /// Empty on simulation-only workspaces (`max_openings = 0`), which the
    /// backward pass never touches.
    pub(crate) backward_accum: BackwardAccumulators,

    /// Zero-allocation timing payload for [`cobre_core::TrainingEvent::WorkerTiming`],
    /// accumulated in the parallel region and moved into the event payload after
    /// it completes. Reset to default at each iteration boundary.
    ///
    /// Field-to-slot mapping for the writer record:
    /// `forward_wall_ms` → `WORKER_TIMING_SLOT_FWD_WALL`,
    /// `backward_wall_ms` → `WORKER_TIMING_SLOT_BWD_WALL`,
    /// `bwd_setup_ms` → `WORKER_TIMING_SLOT_BWD_SETUP`,
    /// `fwd_setup_ms` → `WORKER_TIMING_SLOT_FWD_SETUP`.
    pub worker_timing_buf: WorkerPhaseTimings,
}

impl<S: SolverInterface> SolverWorkspace<S> {
    /// Construct a workspace with explicit identity, solver, and patch buffer.
    ///
    /// `scratch_basis` starts empty; call `WorkspacePool::resize_scratch_bases`
    /// after construction to pre-allocate it for in-place basis reconstruction.
    #[must_use]
    pub fn new(
        rank: i32,
        worker_id: i32,
        solver: S,
        patch_buf: PatchBuffer,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
    ) -> Self {
        Self {
            rank,
            worker_id,
            solver: ProfiledSolver::new(solver),
            patch_buf,
            current_state: Vec::with_capacity(training_ctx.state.n_state),
            scratch: ScratchBuffers::new(training_ctx, stage_ctx, sizing),
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::new(
                sizing.max_openings,
                sizing.initial_pool_capacity,
                training_ctx.state.n_state,
            ),
            worker_timing_buf: WorkerPhaseTimings::default(),
        }
    }
}

impl ScratchBuffers {
    /// Allocate scratch buffers sized from the study owners `training_ctx`/
    /// `stage_ctx` carry, plus the execution-count hints in `sizing`.
    #[must_use]
    pub fn new(
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
    ) -> Self {
        let hydro_count = training_ctx.state.hydro_count;
        let max_par_order = training_ctx.state.max_par_order;
        let downstream_par_order = training_ctx.study_dims.downstream_par_order;
        let load_row_capacity = stage_ctx.load_bus_indices.len()
            * StageGeometry::max_blocks(stage_ctx.geometry_per_stage);
        let WorkspaceSizing {
            initial_pool_capacity,
            max_local_fwd,
            noise,
            // `max_openings` sizes `BackwardAccumulators`, not `ScratchBuffers`.
            ..
        } = sizing;
        let noise_dim = match noise {
            NoisePreallocation::StochasticDim => training_ctx.stochastic.dim(),
            NoisePreallocation::OnDemand => 0,
        };
        Self {
            inflow_m3s_buf: Vec::with_capacity(hydro_count),
            lag_matrix_buf: Vec::with_capacity(max_par_order * hydro_count),
            par_inflow_buf: Vec::with_capacity(hydro_count),
            eta_floor_buf: Vec::with_capacity(hydro_count),
            zero_targets_buf: vec![0.0_f64; hydro_count],
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::with_capacity(load_row_capacity),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::with_capacity(hydro_count),
            effective_eta_buf: Vec::with_capacity(hydro_count),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![0.0_f64; hydro_count],
            lag_weight_accum: vec![0.0_f64; hydro_count],
            downstream_accumulator: if downstream_par_order > 0 {
                vec![0.0_f64; hydro_count]
            } else {
                Vec::new()
            },
            downstream_weight_accum: 0.0,
            downstream_completed_lags: if downstream_par_order > 0 {
                vec![0.0_f64; hydro_count * downstream_par_order]
            } else {
                Vec::new()
            },
            downstream_n_completed: 0,
            recon_slot_lookup: vec![None; initial_pool_capacity],
            trajectory_costs_buf: Vec::with_capacity(max_local_fwd),
            raw_noise_buf: Vec::with_capacity(noise_dim),
            corr_scratch: Vec::with_capacity(2 * noise_dim),
            current_node_buf: Vec::with_capacity(max_local_fwd),
        }
    }
}

/// A pool of [`SolverWorkspace`] instances, one per worker thread.
///
/// Create once at algorithm startup via [`WorkspacePool::new`] and distribute
/// workspaces to threads by indexing into [`workspaces`](WorkspacePool::workspaces).
///
/// The pool size equals the number of worker threads. Each workspace is
/// independently allocated and does not share any mutable state with the others.
pub struct WorkspacePool<S: SolverInterface> {
    /// The individual workspaces, indexed by thread number.
    pub workspaces: Vec<SolverWorkspace<S>>,
}

impl<S: SolverInterface> WorkspacePool<S> {
    /// Build one workspace: size its [`PatchBuffer`] from `training_ctx`/`stage_ctx`
    /// and construct the [`SolverWorkspace`]. Shared tail of [`WorkspacePool::new`]
    /// and [`WorkspacePool::try_new`], which differ only in how they obtain
    /// `worker_id` and `solver`.
    fn build_workspace(
        rank: i32,
        worker_id: i32,
        solver: S,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
    ) -> SolverWorkspace<S> {
        let patch_buf = PatchBuffer::new(
            training_ctx.state,
            stage_ctx.load_bus_indices,
            stage_ctx.geometry_per_stage,
        );
        SolverWorkspace::new(
            rank,
            worker_id,
            solver,
            patch_buf,
            training_ctx,
            stage_ctx,
            sizing,
        )
    }

    /// Construct a pool of `n_threads` independently allocated workspaces, each
    /// with a sequentially assigned `worker_id` in `0..n_threads` and a fresh
    /// solver from `solver_factory` (called once per thread).
    ///
    /// # Panics
    ///
    /// Panics if `n_threads > i32::MAX` — physically impossible since rayon pools
    /// are bounded by CPU count.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new(
        rank: i32,
        n_threads: usize,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
        solver_factory: impl Fn() -> S,
    ) -> Self {
        let workspaces = (0..n_threads)
            .map(|idx| {
                let worker_id =
                    i32::try_from(idx).expect("worker_id fits in i32 (rayon pools are small)");
                let solver = solver_factory();
                Self::build_workspace(rank, worker_id, solver, training_ctx, stage_ctx, sizing)
            })
            .collect();
        Self { workspaces }
    }

    /// Like [`WorkspacePool::new`] but with a fallible `solver_factory`; the
    /// first factory error is returned immediately and no partial pool is built.
    ///
    /// # Errors
    ///
    /// Returns `Err(E)` if any call to `solver_factory` fails.
    ///
    /// # Panics
    ///
    /// Panics if `n_threads > i32::MAX` — physically impossible since rayon pools
    /// are bounded by CPU count.
    #[allow(clippy::expect_used)]
    pub fn try_new<E>(
        rank: i32,
        n_threads: usize,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
        solver_factory: impl Fn() -> Result<S, E>,
    ) -> Result<Self, E> {
        let mut workspaces = Vec::with_capacity(n_threads);
        for idx in 0..n_threads {
            let worker_id =
                i32::try_from(idx).expect("worker_id fits in i32 (rayon pools are small)");
            let solver = solver_factory()?;
            workspaces.push(Self::build_workspace(
                rank,
                worker_id,
                solver,
                training_ctx,
                stage_ctx,
                sizing,
            ));
        }
        Ok(Self { workspaces })
    }

    /// Pre-allocate each workspace's `scratch_basis` to the given LP dimensions.
    ///
    /// The allocation happens once during setup; basis reconstruction on the
    /// hot path then reuses the existing capacity without reallocating.
    pub(crate) fn resize_scratch_bases(&mut self, max_cols: usize, max_rows: usize) {
        for ws in &mut self.workspaces {
            ws.scratch_basis = Basis::new(max_cols, max_rows);
        }
    }

    /// Re-reserve every workspace's DCS scratch ([`DcsSolveScratch`]) to
    /// `pool_capacity`, growth-only (`DcsSolveScratch::reserve`) — a no-op once
    /// every workspace already covers it.
    ///
    /// `pool_capacity` must be the LARGEST capacity across every pool a
    /// workspace's sweep may touch, never a single pool's own value: the
    /// scratch is shared per WORKER, not per pool, so sizing it from one
    /// pool under-covers every other pool that worker also solves against.
    pub(crate) fn reserve_dcs_scratch(&mut self, n_state: usize, pool_capacity: usize) {
        for ws in &mut self.workspaces {
            ws.backward_accum.dcs_solve.reserve(n_state, pool_capacity);
        }
    }
}

// ---------------------------------------------------------------------------
// BasisStore
// ---------------------------------------------------------------------------

/// Per-scenario, per-node basis storage for warm-starting LP solves.
///
/// The store is indexed as `store[scenario_index][node_position]`. The forward
/// pass writes each visit at its stage position and the backward pass
/// warm-starts a cut's successor at that successor's canonical node position;
/// absent `nodes[]` node position == stage, so the two coincide. During the
/// forward pass, each worker writes to its disjoint scenario range; during the
/// backward pass, all workers read from any scenario's basis.
///
/// Internally, data is stored flat as
/// `bases[scenario * num_nodes + node]` for cache-friendly sequential
/// access within a single scenario's node loop.
///
/// # Cut selection interaction
///
/// Row statuses are positional (`row_status[i]` is LP row `i`). When cut
/// selection changes the active cut set, the cut-row count changes and the
/// stored cut-row statuses become stale; the degraded warm-start is accepted —
/// `HiGHS` detects the dimension mismatch on `solve(Some(&basis))`, crash-starts,
/// and records a `basis_rejection` in [`SolverStatistics`]. Template (non-cut)
/// row statuses remain valid.
///
/// [`SolverStatistics`]: cobre_solver::SolverStatistics
pub struct BasisStore {
    /// Flat storage: `bases[scenario * num_nodes + node]`.
    bases: Vec<Option<CapturedBasis>>,
    /// Node-axis length (canonical node count; equals `num_stages` on the chain).
    num_nodes: usize,
}

impl BasisStore {
    /// Allocate a new store for `num_scenarios` scenarios and `num_nodes`
    /// node positions, with every slot initialised to `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::setup::NodePos;
    /// use cobre_sddp::workspace::BasisStore;
    ///
    /// let store = BasisStore::new(4, 10);
    /// assert_eq!(store.num_scenarios(), 4);
    /// assert_eq!(store.num_nodes(), 10);
    /// assert!(store.get(0, NodePos(0)).is_none());
    /// ```
    #[must_use]
    pub fn new(num_scenarios: usize, num_nodes: usize) -> Self {
        let len = num_scenarios * num_nodes;
        Self {
            bases: vec![None; len],
            num_nodes,
        }
    }

    /// Return the number of scenarios this store was allocated for.
    #[must_use]
    pub fn num_scenarios(&self) -> usize {
        self.bases.len().checked_div(self.num_nodes).unwrap_or(0)
    }

    /// Return the node-axis length.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get an immutable reference to the basis at `[scenario][node]`.
    ///
    /// Returns `None` if the slot has not yet been populated.
    #[must_use]
    pub fn get(&self, scenario: usize, node: NodePos) -> Option<&CapturedBasis> {
        self.bases[scenario * self.num_nodes + node.0].as_ref()
    }

    /// Get a mutable reference to the basis slot at `[scenario][node]`.
    pub fn get_mut(&mut self, scenario: usize, node: NodePos) -> &mut Option<CapturedBasis> {
        &mut self.bases[scenario * self.num_nodes + node.0]
    }

    /// Split the store into `n_workers` disjoint mutable sub-views by scenario
    /// range, one per worker.
    ///
    /// Worker `w` receives the scenarios in the range produced by
    /// `partition(num_scenarios, n_workers, w)`. Each [`BasisStoreSliceMut`]
    /// carries its scenario offset so that callers can index using absolute
    /// scenario indices.
    ///
    /// When `n_workers` exceeds `num_scenarios`, some workers receive empty
    /// slices (`start == end`), which is valid — their slice covers zero
    /// scenarios.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `n_workers == 0`.
    #[must_use]
    pub fn split_workers_mut(&mut self, n_workers: usize) -> Vec<BasisStoreSliceMut<'_>> {
        debug_assert!(n_workers > 0, "n_workers must be > 0");
        let total_scenarios = self.num_scenarios();
        let mut slices = Vec::with_capacity(n_workers);
        let mut bases_rem = self.bases.as_mut_slice();

        for w in 0..n_workers {
            let (start, end) = partition(total_scenarios, n_workers, w);
            let count = end - start;
            let chunk = count * self.num_nodes;
            let (bases_left, bases_rest) = bases_rem.split_at_mut(chunk);
            bases_rem = bases_rest;
            slices.push(BasisStoreSliceMut {
                bases: bases_left,
                scenario_offset: start,
                num_nodes: self.num_nodes,
            });
        }
        slices
    }
}

/// A mutable sub-view of a [`BasisStore`] covering a contiguous range of
/// scenarios.
///
/// Produced by [`BasisStore::split_workers_mut`]. Each slice is exclusive
/// to one worker thread; multiple slices can coexist because they cover
/// disjoint memory regions.
pub struct BasisStoreSliceMut<'a> {
    /// Sub-slice of the flat basis array for this worker's scenario range.
    bases: &'a mut [Option<CapturedBasis>],
    /// Absolute scenario index of the first scenario in this slice.
    scenario_offset: usize,
    /// Node-axis length (see [`BasisStore`]).
    num_nodes: usize,
}

impl BasisStoreSliceMut<'_> {
    /// Get an immutable reference to the basis at absolute scenario index
    /// `scenario` and node position `node`.
    ///
    /// Returns `None` if the slot has not yet been populated.
    ///
    /// # Panics
    ///
    /// Panics if `scenario < self.scenario_offset` (scenario not in this slice).
    #[must_use]
    pub fn get(&self, scenario: usize, node: NodePos) -> Option<&CapturedBasis> {
        let local = scenario - self.scenario_offset;
        self.bases[local * self.num_nodes + node.0].as_ref()
    }

    /// This slice's global scenario window `[start, end)` — the scenarios
    /// `get`/`get_mut` address in-bounds. The backward trial-point router assigns
    /// each routed scenario to the worker whose window contains it, so its
    /// `get(m, node)` never indexes outside this slice.
    #[must_use]
    pub fn scenario_window(&self) -> (usize, usize) {
        let count = self.bases.len() / self.num_nodes;
        (self.scenario_offset, self.scenario_offset + count)
    }

    /// Get a mutable reference to the basis slot at absolute scenario index
    /// `scenario` and node position `node`.
    ///
    /// # Panics
    ///
    /// Panics if `scenario < self.scenario_offset` (scenario not in this slice).
    pub fn get_mut(&mut self, scenario: usize, node: NodePos) -> &mut Option<CapturedBasis> {
        let local = scenario - self.scenario_offset;
        &mut self.bases[local * self.num_nodes + node.0]
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "highs")]
    use super::PatchBuffer;
    use super::{
        BasisStore, ByNodeScratch, CapturedBasis, NodeId, NodePos, ScratchBuffers, SolverWorkspace,
        WorkspacePool, WorkspaceSizing,
    };
    use crate::context::StageContext;
    use crate::test_support::{TrainingContextFixture, state_layout};
    use cobre_solver::{
        Basis, BasisStatus, SolutionView, SolverError, SolverInterface, SolverStatistics,
        types::{RowBatch, StageTemplate},
    };

    /// Minimal no-op solver for workspace tests.
    struct MockSolver;

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "mock".into(),
                error_code: None,
            })
        }
        fn get_basis(&mut self, out: &mut Basis) {
            crate::test_support::fill_consistent_basis(out);
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }
        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    /// Compile-time assertion that `SolverWorkspace<MockSolver>` is `Send`.
    fn assert_send<T: Send>() {}

    #[test]
    fn test_workspace_send_bound() {
        assert_send::<SolverWorkspace<MockSolver>>();
    }

    /// A [`StageContext`] over empty slices, for pool-sizing tests that never
    /// exercise load buses or geometry.
    fn empty_stage_ctx() -> StageContext<'static> {
        StageContext {
            templates: &[],
            state_boxes: &[],
            geometry_per_stage: &[],
            cost_scale_factor: 1.0,
            load_bus_indices: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
        }
    }

    #[test]
    fn test_workspace_pool_size() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let pool = WorkspacePool::new(
            0,
            4,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 4);
    }

    #[test]
    fn test_workspace_buffer_dimensions() {
        // N=3, L=2, M=0, B=0 → patch_buf length = M*B + N = 0 + 3 = 3
        // n_state=9 → current_state capacity = 9
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let pool = WorkspacePool::new(
            0,
            4,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert_eq!(ws.patch_buf.indices.len(), 3, "patch_buf length");
            assert_eq!(ws.current_state.capacity(), 9, "current_state capacity");
            assert_eq!(ws.current_state.len(), 0, "current_state starts empty");
        }
    }

    #[test]
    fn test_workspace_pool_zero_threads() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let pool = WorkspacePool::new(
            0,
            0,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 0);
    }

    #[test]
    fn test_workspace_pool_single_thread() {
        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            0,
            1,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 1);
        assert_eq!(pool.workspaces[0].patch_buf.indices.len(), 0);
    }

    #[test]
    fn test_workspace_pool_each_solver_independent() {
        // Factory is called n_threads times; each workspace gets its own instance.
        // Verify by checking pool size matches factory call expectation.
        let n = 6;
        let ctx = TrainingContextFixture::new(state_layout(1, 0));
        let pool = WorkspacePool::new(
            0,
            n,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), n);
    }

    #[test]
    fn test_backward_accumulators_opening_outcomes_buf_starts_empty() {
        // A nonzero max_openings must not eagerly build opening_outcomes_buf —
        // the default ByScenario scheduler never touches it (zero opening-block footprint).
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing {
                max_openings: 5,
                ..WorkspaceSizing::default()
            },
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert!(
                ws.backward_accum.opening_outcomes_buf.is_empty(),
                "opening_outcomes_buf must start empty regardless of max_openings"
            );
        }
    }

    #[test]
    fn test_scratch_buffers_zero_downstream_par_order_empty_buffers() {
        let ctx = TrainingContextFixture::new(state_layout(5, 2)).downstream_par_order(0);
        let scratch = ScratchBuffers::new(
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
        );
        assert!(
            scratch.downstream_accumulator.is_empty(),
            "downstream_accumulator must be empty when downstream_par_order=0"
        );
        assert!(
            scratch.downstream_completed_lags.is_empty(),
            "downstream_completed_lags must be empty when downstream_par_order=0"
        );
        assert_eq!(
            scratch.downstream_weight_accum, 0.0,
            "downstream_weight_accum must be 0.0"
        );
        assert_eq!(
            scratch.downstream_n_completed, 0,
            "downstream_n_completed must be 0"
        );
    }

    #[test]
    fn test_scratch_buffers_nonzero_downstream_par_order_allocates_correctly() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2)).downstream_par_order(2);
        let scratch = ScratchBuffers::new(
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
        );
        assert_eq!(
            scratch.downstream_accumulator.len(),
            3,
            "downstream_accumulator.len() must equal hydro_count"
        );
        assert_eq!(
            scratch.downstream_completed_lags.len(),
            6,
            "downstream_completed_lags.len() must equal hydro_count * downstream_par_order"
        );
        assert!(
            scratch.downstream_accumulator.iter().all(|&v| v == 0.0),
            "downstream_accumulator must be initialized to 0.0"
        );
        assert!(
            scratch.downstream_completed_lags.iter().all(|&v| v == 0.0),
            "downstream_completed_lags must be initialized to 0.0"
        );
        assert_eq!(scratch.downstream_weight_accum, 0.0);
        assert_eq!(scratch.downstream_n_completed, 0);
    }

    #[test]
    fn test_workspace_pool_propagates_downstream_par_order() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2)).downstream_par_order(2);
        let pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert_eq!(
                ws.scratch.downstream_accumulator.len(),
                3,
                "downstream_accumulator.len() per workspace"
            );
            assert_eq!(
                ws.scratch.downstream_completed_lags.len(),
                6,
                "downstream_completed_lags.len() per workspace"
            );
            assert_eq!(ws.scratch.downstream_weight_accum, 0.0);
            assert_eq!(ws.scratch.downstream_n_completed, 0);
        }
    }

    // ---------------------------------------------------------------------------
    // ByNodeScratch tests
    // ---------------------------------------------------------------------------

    #[test]
    fn backward_by_node_scratch_default_is_empty() {
        let scratch = ByNodeScratch::default();
        assert_eq!(scratch.arena.capacity(), 0, "arena must start capacity 0");
        assert!(scratch.arena.is_empty());
        assert!(scratch.coeffs_buf.is_empty());
        assert!(scratch.block_pivots.is_empty());
        assert!(scratch.block_pivots_prev.is_empty());
        assert_eq!(scratch.n_blocks_max, 0);
        assert!(scratch.block_order.is_empty());
    }

    #[test]
    fn backward_by_node_scratch_sized_matches_study_dims() {
        let max_local_fwd = 3_usize;
        let bwd_max_openings = 4_usize;
        let n_state = 5_usize;
        let num_stages = 6_usize;

        let scratch = ByNodeScratch::sized(max_local_fwd, bwd_max_openings, n_state, num_stages);

        assert_eq!(
            scratch.arena.len(),
            max_local_fwd * bwd_max_openings,
            "arena must hold max_local_fwd * bwd_max_openings outcomes"
        );
        assert!(
            scratch
                .arena
                .iter()
                .all(|outcome| outcome.coefficients.len() == n_state),
            "every outcome's coefficients must be pre-sized to n_state"
        );
        assert_eq!(scratch.coeffs_buf.len(), n_state);
        assert_eq!(
            scratch.block_pivots.len(),
            num_stages * bwd_max_openings,
            "block_pivots must hold num_stages * bwd_max_openings accumulator cells"
        );
        assert!(
            scratch
                .block_pivots
                .iter()
                .all(|&(sum, count)| sum == 0 && count == 0),
            "a freshly sized accumulator must start zeroed"
        );
        assert_eq!(
            scratch.block_pivots_prev.len(),
            num_stages * bwd_max_openings,
            "block_pivots_prev must mirror block_pivots' shape"
        );
        assert!(
            scratch
                .block_pivots_prev
                .iter()
                .all(|&(sum, count)| sum == 0 && count == 0),
            "a freshly sized double-buffer companion must start zeroed"
        );
        assert_eq!(scratch.n_blocks_max, bwd_max_openings);
        assert_eq!(
            scratch.block_order.len(),
            bwd_max_openings,
            "block_order must hold n_blocks_max claim-order scratch slots"
        );
    }

    #[test]
    fn backward_by_node_scratch_sized_zero_dims_yields_empty_arena() {
        let scratch = ByNodeScratch::sized(0, 0, 0, 0);
        assert!(scratch.arena.is_empty());
        assert!(scratch.coeffs_buf.is_empty());
        assert!(scratch.block_pivots.is_empty());
        assert!(scratch.block_pivots_prev.is_empty());
        assert!(scratch.block_order.is_empty());
    }

    // ---------------------------------------------------------------------------
    // BasisStore tests
    // ---------------------------------------------------------------------------

    #[test]
    fn basis_store_new_all_none() {
        let store = BasisStore::new(3, 5);
        assert_eq!(store.num_scenarios(), 3);
        assert_eq!(store.num_nodes(), 5);
        for s in 0..3 {
            for t in 0..5 {
                assert!(
                    store.get(s, NodePos(t)).is_none(),
                    "slot [{s}][{t}] must start as None"
                );
            }
        }
    }

    #[test]
    fn basis_store_get_mut_set_and_retrieve() {
        let mut store = BasisStore::new(2, 3);
        *store.get_mut(1, NodePos(2)) = Some(CapturedBasis::new(4, 2, 0, 0, 0, NodeId(0)));
        assert!(store.get(1, NodePos(2)).is_some());
        assert!(store.get(0, NodePos(0)).is_none());
        assert!(store.get(1, NodePos(0)).is_none());
    }

    #[test]
    fn basis_store_zero_scenarios() {
        let store = BasisStore::new(0, 5);
        assert_eq!(store.num_scenarios(), 0);
        assert_eq!(store.num_nodes(), 5);
    }

    #[test]
    fn basis_store_zero_stages() {
        let store = BasisStore::new(3, 0);
        assert_eq!(store.num_scenarios(), 0);
        assert_eq!(store.num_nodes(), 0);
    }

    #[test]
    fn basis_store_split_workers_mut_disjoint_writes() {
        // 4 scenarios, 3 stages, 2 workers.
        // Worker 0 covers scenarios 0..2; worker 1 covers scenarios 2..4.
        let mut store = BasisStore::new(4, 3);
        let mut slices = store.split_workers_mut(2);

        // Worker 0 writes to scenario 0 stage 1.
        *slices[0].get_mut(0, NodePos(1)) = Some(CapturedBasis::new(2, 1, 0, 0, 0, NodeId(0)));
        // Worker 1 writes to scenario 3 stage 2.
        *slices[1].get_mut(3, NodePos(2)) = Some(CapturedBasis::new(2, 1, 0, 0, 0, NodeId(0)));

        // Drop slices to release the borrow on store.
        drop(slices);

        assert!(
            store.get(0, NodePos(1)).is_some(),
            "scenario 0 stage 1 must be populated"
        );
        assert!(
            store.get(3, NodePos(2)).is_some(),
            "scenario 3 stage 2 must be populated"
        );
        assert!(store.get(0, NodePos(0)).is_none());
        assert!(store.get(3, NodePos(0)).is_none());
    }

    #[test]
    fn basis_store_split_single_worker() {
        let mut store = BasisStore::new(3, 2);
        let mut slices = store.split_workers_mut(1);
        *slices[0].get_mut(2, NodePos(1)) = Some(CapturedBasis::new(1, 0, 0, 0, 0, NodeId(0)));
        drop(slices);
        assert!(store.get(2, NodePos(1)).is_some());
    }

    #[test]
    fn basis_store_split_more_workers_than_scenarios() {
        // 2 scenarios, 4 workers → workers 2 and 3 receive empty slices.
        let mut store = BasisStore::new(2, 3);
        let slices = store.split_workers_mut(4);
        assert_eq!(slices.len(), 4);
        // Workers 0 and 1 cover 1 scenario each; workers 2 and 3 cover 0.
        assert_eq!(slices[0].bases.len(), 3); // 1 scenario × 3 stages
        assert_eq!(slices[1].bases.len(), 3);
        assert_eq!(slices[2].bases.len(), 0);
        assert_eq!(slices[3].bases.len(), 0);
    }

    #[test]
    fn basis_store_slice_offset_correct() {
        // 6 scenarios, 2 stages, 3 workers → 2 scenarios each.
        let mut store = BasisStore::new(6, 2);
        let mut slices = store.split_workers_mut(3);

        // Worker 1 covers absolute scenarios 2..4.
        *slices[1].get_mut(2, NodePos(0)) = Some(CapturedBasis::new(1, 0, 0, 0, 0, NodeId(0)));
        *slices[1].get_mut(3, NodePos(1)) = Some(CapturedBasis::new(1, 0, 0, 0, 0, NodeId(0)));
        drop(slices);

        assert!(store.get(2, NodePos(0)).is_some());
        assert!(store.get(3, NodePos(1)).is_some());
        assert!(store.get(0, NodePos(0)).is_none());
        assert!(store.get(4, NodePos(0)).is_none());
    }

    #[test]
    fn test_captured_basis_new_capacities() {
        let cb = CapturedBasis::new(4, 6, 3, 10, 2, NodeId(9));
        assert_eq!(cb.basis.row_status.len(), 6, "row_status length");
        assert_eq!(cb.base_row_count, 3, "base_row_count");
        assert_eq!(cb.node_id, NodeId(9), "node_id");
        assert!(
            cb.cut_row_slots.capacity() >= 10,
            "cut_row_slots capacity must be >= 10 (got {})",
            cb.cut_row_slots.capacity()
        );
        assert_eq!(cb.cut_row_slots.len(), 0, "cut_row_slots starts empty");
        assert!(
            cb.state_at_capture.capacity() >= 2,
            "state_at_capture capacity must be >= 2 (got {})",
            cb.state_at_capture.capacity()
        );
        assert_eq!(
            cb.state_at_capture.len(),
            0,
            "state_at_capture starts empty"
        );
    }

    #[test]
    fn test_basis_store_holds_captured_basis() {
        let mut store = BasisStore::new(3, 5);
        // All 15 slots start as None.
        for s in 0..3 {
            for t in 0..5 {
                assert!(
                    store.get(s, NodePos(t)).is_none(),
                    "slot [{s}][{t}] must be None before any write"
                );
            }
        }
        // Write a CapturedBasis at [1][3]; read it back.
        *store.get_mut(1, NodePos(3)) = Some(CapturedBasis::new(4, 6, 3, 10, 2, NodeId(0)));
        let retrieved = store.get(1, NodePos(3));
        assert!(retrieved.is_some(), "slot [1][3] must be Some after write");
        let cb = retrieved.expect("just checked is_some");
        assert_eq!(cb.base_row_count, 3);
        // All other slots remain None.
        for s in 0..3 {
            for t in 0..5 {
                if s == 1 && t == 3 {
                    continue;
                }
                assert!(
                    store.get(s, NodePos(t)).is_none(),
                    "slot [{s}][{t}] must remain None"
                );
            }
        }
    }

    #[test]
    fn test_recon_slot_lookup_presized() {
        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            0,
            4,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing {
                initial_pool_capacity: 50,
                ..WorkspaceSizing::default()
            },
            || MockSolver,
        );
        for (i, ws) in pool.workspaces.iter().enumerate() {
            assert_eq!(
                ws.scratch.recon_slot_lookup.len(),
                50,
                "workspace {i}: recon_slot_lookup.len() must equal initial_pool_capacity (50)"
            );
            assert!(
                ws.scratch.recon_slot_lookup.iter().all(Option::is_none),
                "workspace {i}: all recon_slot_lookup entries must be None"
            );
        }
        // Verify zero initial_pool_capacity produces an empty vec.
        let pool_empty = WorkspacePool::new(
            0,
            1,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(
            pool_empty.workspaces[0].scratch.recon_slot_lookup.len(),
            0,
            "initial_pool_capacity=0 must produce empty recon_slot_lookup"
        );
    }

    #[test]
    fn test_workspace_pool_assigns_sequential_worker_ids() {
        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            /* rank = */ 3,
            /* n_workers = */ 5,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        let ws_slice = &pool.workspaces;
        assert_eq!(ws_slice.len(), 5);
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for ws in ws_slice {
            assert_eq!(ws.rank, 3);
            assert!(ws.worker_id >= 0 && ws.worker_id < 5);
            assert!(seen.insert(ws.worker_id), "worker_id duplicated");
        }
    }

    // ---------------------------------------------------------------------------
    // CapturedBasis wire-format round-trip tests
    // ---------------------------------------------------------------------------

    /// Round-trip test: single stage with fully populated metadata.
    ///
    /// Constructs a `CapturedBasis` with known fields, packs via
    /// `to_broadcast_payload`, then unpacks via `try_from_broadcast_payload`.
    /// Asserts field-by-field equality with the original.
    #[test]
    fn test_captured_basis_round_trip_populated() {
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower, BasisStatus::Basic, BasisStatus::Upper],
                row_status: vec![BasisStatus::Zero, BasisStatus::Nonbasic],
            },
            base_row_count: 1,
            cut_row_slots: vec![10_u32, 20],
            state_at_capture: vec![1.5_f64, 2.5, 3.5],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let result = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail");
        let recovered = result.expect("sentinel is 1; must return Some");

        assert_eq!(
            recovered.basis.col_status, original.basis.col_status,
            "col_status"
        );
        assert_eq!(
            recovered.basis.row_status, original.basis.row_status,
            "row_status"
        );
        assert_eq!(
            recovered.base_row_count, original.base_row_count,
            "base_row_count"
        );
        assert_eq!(
            recovered.cut_row_slots, original.cut_row_slots,
            "cut_row_slots"
        );
        assert_eq!(
            recovered.state_at_capture, original.state_at_capture,
            "state_at_capture"
        );
        // Cursors must have advanced past the full payload.
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
        assert_eq!(f64_cursor, f64_buf.len(), "f64_cursor must be at end");
    }

    /// Round-trip test: single stage with empty `cut_row_slots` and
    /// empty `state_at_capture`.
    #[test]
    fn test_captured_basis_round_trip_empty_metadata() {
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Superbasic, BasisStatus::Fixed],
                row_status: vec![BasisStatus::Lower],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            state_at_capture: vec![],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let result = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail");
        let recovered = result.expect("sentinel is 1; must return Some");

        assert_eq!(recovered.basis.col_status, original.basis.col_status);
        assert_eq!(recovered.basis.row_status, original.basis.row_status);
        assert_eq!(recovered.base_row_count, original.base_row_count);
        assert!(
            recovered.cut_row_slots.is_empty(),
            "cut_row_slots must be empty"
        );
        assert!(
            recovered.state_at_capture.is_empty(),
            "state_at_capture must be empty"
        );
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
        assert_eq!(f64_cursor, f64_buf.len(), "f64_cursor must be at end");
    }

    /// Multi-stage round-trip: one `Some` stage followed by one `None` stage.
    ///
    /// Packs the `Some` stage via `to_broadcast_payload`, writes a `0_i32`
    /// sentinel for the `None` stage, then loops `try_from_broadcast_payload`
    /// twice and asserts the recovered `Option`s match.
    #[test]
    fn test_captured_basis_round_trip_multi_stage() {
        let populated = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic, BasisStatus::Upper, BasisStatus::Zero],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                ],
            },
            base_row_count: 2,
            cut_row_slots: vec![100_u32, 200, 300],
            state_at_capture: vec![0.1_f64, 0.2],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();

        // Stage 0: Some — pack via method.
        populated.to_broadcast_payload(&mut i32_buf, &mut f64_buf);
        // Stage 1: None — caller writes the 0 sentinel directly.
        i32_buf.push(0_i32);

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;

        // Unpack stage 0.
        let stage0 = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("stage 0 must not fail")
        .expect("stage 0 sentinel is 1; must return Some");

        assert_eq!(stage0.basis.col_status, populated.basis.col_status);
        assert_eq!(stage0.basis.row_status, populated.basis.row_status);
        assert_eq!(stage0.base_row_count, populated.base_row_count);
        assert_eq!(stage0.cut_row_slots, populated.cut_row_slots);
        assert_eq!(stage0.state_at_capture, populated.state_at_capture);

        // Unpack stage 1 — must return None, advancing cursor by 1.
        let stage1 = CapturedBasis::try_from_broadcast_payload(
            1,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("stage 1 must not fail");
        assert!(stage1.is_none(), "stage 1 sentinel is 0; must return None");

        assert_eq!(
            i32_cursor,
            i32_buf.len(),
            "i32_cursor must be at end after both stages"
        );
        assert_eq!(
            f64_cursor,
            f64_buf.len(),
            "f64_cursor must be at end after both stages"
        );
    }

    /// Truncated i32 buffer: truncate the packed buffer by 1 element, assert
    /// `Err(SddpError::Validation(msg))` where `msg` contains "truncated" and
    /// the stage index.
    #[test]
    fn test_captured_basis_truncated_i32_buffer() {
        use crate::SddpError;

        let cb = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower, BasisStatus::Basic],
                row_status: vec![BasisStatus::Upper],
            },
            base_row_count: 1,
            cut_row_slots: vec![5_u32],
            state_at_capture: vec![9.9_f64],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        cb.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // Truncate i32 buffer by 1.
        i32_buf.pop();

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let err = CapturedBasis::try_from_broadcast_payload(
            7,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect_err("truncated buffer must return Err");

        match err {
            SddpError::Validation(ref msg) => {
                assert!(
                    msg.contains("truncated"),
                    "error message must contain 'truncated', got: {msg}"
                );
                assert!(
                    msg.contains('7'),
                    "error message must contain stage index 7, got: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    /// Truncated f64 buffer: pack a basis with non-empty `state_at_capture`,
    /// truncate the f64 buffer by 1, assert `Err(SddpError::Validation(msg))`
    /// where `msg` mentions `state_at_capture` and the stage index.
    #[test]
    fn test_captured_basis_truncated_f64_buffer() {
        use crate::SddpError;

        let cb = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower],
                row_status: vec![BasisStatus::Basic],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            state_at_capture: vec![1.0_f64, 2.0, 3.0],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        cb.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // Truncate f64 buffer by 1.
        f64_buf.pop();

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let err = CapturedBasis::try_from_broadcast_payload(
            3,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect_err("truncated f64 buffer must return Err");

        match err {
            SddpError::Validation(ref msg) => {
                assert!(
                    msg.contains("state_at_capture"),
                    "error message must contain 'state_at_capture', got: {msg}"
                );
                assert!(
                    msg.contains('3'),
                    "error message must contain stage index 3, got: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------------------
    // Version-byte tests
    // ---------------------------------------------------------------------------

    /// Round-trip verification that `to_broadcast_payload` emits
    /// `BASIS_BROADCAST_FORMAT_TAG` at offset 1 of the `i32_buf` (immediately
    /// after the presence sentinel).
    #[test]
    fn to_broadcast_payload_emits_version_byte() {
        use super::BASIS_BROADCAST_FORMAT_TAG;

        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![
                    BasisStatus::Lower,
                    BasisStatus::Basic,
                    BasisStatus::Upper,
                    BasisStatus::Zero,
                ],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                    BasisStatus::Lower,
                ],
            },
            base_row_count: 2,
            cut_row_slots: vec![10_u32, 20],
            state_at_capture: vec![0.5_f64, 1.5],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // Offset 0 is the presence sentinel (1_i32).
        assert_eq!(i32_buf[0], 1_i32, "offset 0 must be the presence sentinel");
        // Offset 1 must be the wire version.
        assert_eq!(
            i32_buf[1], BASIS_BROADCAST_FORMAT_TAG,
            "offset 1 must be BASIS_BROADCAST_FORMAT_TAG"
        );

        // Full round-trip must return a bit-equal CapturedBasis.
        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");

        assert_eq!(recovered.basis.col_status, original.basis.col_status);
        assert_eq!(recovered.basis.row_status, original.basis.row_status);
        assert_eq!(recovered.base_row_count, original.base_row_count);
        assert_eq!(recovered.cut_row_slots, original.cut_row_slots);
        assert_eq!(recovered.state_at_capture, original.state_at_capture);
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
        assert_eq!(f64_cursor, f64_buf.len(), "f64_cursor must be at end");
    }

    /// Manually overwrite the version field (offset 1) to `1_i32` (the removed
    /// v1 format) and assert that `try_from_broadcast_payload` returns
    /// `Err(SddpError::Validation)` whose message contains
    /// `"unsupported wire version 1"`.
    #[test]
    fn try_from_broadcast_payload_rejects_wrong_version() {
        use crate::SddpError;

        let cb = CapturedBasis {
            basis: Basis {
                col_status: vec![
                    BasisStatus::Lower,
                    BasisStatus::Basic,
                    BasisStatus::Upper,
                    BasisStatus::Zero,
                ],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                    BasisStatus::Lower,
                ],
            },
            base_row_count: 2,
            cut_row_slots: vec![10_u32, 20],
            state_at_capture: vec![0.5_f64, 1.5],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        cb.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // Corrupt the version field (offset 1) to the removed v1 to simulate a
        // stale-version peer.
        i32_buf[1] = 1_i32;

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let err = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect_err("mismatched version must return Err");

        match err {
            SddpError::Validation(ref msg) => {
                assert!(
                    msg.contains("unsupported wire version 1"),
                    "error must contain 'unsupported wire version 1', got: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    /// A `None` payload (sentinel `0_i32`) returns `Ok(None)` and advances the
    /// i32 cursor by exactly 1 — the version byte is never consumed.
    #[test]
    fn try_from_broadcast_payload_none_does_not_consume_version_byte() {
        // Build a buffer that starts with a 0 sentinel followed by sentinel=1
        // data for a second stage.  After unpacking stage 0 the cursor must
        // sit at offset 1 (i.e. the 0 sentinel was the only consumed element).
        let populated = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Superbasic],
                row_status: vec![BasisStatus::Fixed],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            state_at_capture: vec![],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        // Stage 0: None sentinel.
        i32_buf.push(0_i32);
        // Stage 1: Some.
        populated.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;

        // Unpack stage 0 — must return None.
        let stage0 = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("stage 0 must not fail");
        assert!(stage0.is_none(), "stage 0 sentinel is 0; must return None");
        // Only the sentinel was consumed (1 element).
        assert_eq!(
            i32_cursor, 1,
            "None path must advance cursor by exactly 1 (only the sentinel)"
        );

        // Unpack stage 1 — must still succeed (version byte is intact).
        let stage1 = CapturedBasis::try_from_broadcast_payload(
            1,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("stage 1 must not fail")
        .expect("stage 1 sentinel is 1; must return Some");
        assert_eq!(stage1.basis.col_status, populated.basis.col_status);
        assert_eq!(stage1.basis.row_status, populated.basis.row_status);
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
        assert_eq!(f64_cursor, f64_buf.len(), "f64_cursor must be at end");
    }

    /// `node_id` rides the wire (version 2): it is written at a fixed header
    /// slot and recovered by `try_from_broadcast_payload`, so the caller no
    /// longer fills it out-of-band. A distinct `node_id` changes exactly one
    /// wire word (the byte count is unchanged), and the value round-trips.
    #[test]
    fn captured_basis_node_id_round_trips_on_the_wire() {
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower],
                row_status: vec![BasisStatus::Basic],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            state_at_capture: vec![],
            node_id: NodeId(77),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);
        let bytes_with_node_id = i32_buf.len();

        let mut other = original.clone();
        other.node_id = NodeId(-1);
        let mut i32_buf2: Vec<i32> = Vec::new();
        let mut f64_buf2: Vec<f64> = Vec::new();
        other.to_broadcast_payload(&mut i32_buf2, &mut f64_buf2);

        assert_ne!(
            i32_buf, i32_buf2,
            "a distinct node_id must change the payload (it is on the wire)"
        );
        assert_eq!(
            bytes_with_node_id,
            i32_buf2.len(),
            "node_id is one fixed-width word; it must not change the wire byte count"
        );

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");

        assert_eq!(
            recovered.node_id, original.node_id,
            "node_id must round-trip through the wire (version 2)"
        );
    }

    // ---------------------------------------------------------------------------
    // Anticipated-state roundtrip tests
    // ---------------------------------------------------------------------------

    /// Roundtrip a basis whose `state_at_capture` has the
    /// `N*(1+L) + n_anticipated*k_max` layout introduced by the
    /// anticipated-thermals feature, with numerically distinct regions.
    #[test]
    fn test_captured_basis_round_trip_includes_anticipated_state() {
        // Layout: N=2 hydros, L=1 PAR lag, n_anticipated=1, k_max=2.
        // n_state = 2 * (1 + 1) + 1 * 2 = 6.
        //
        // Region populations (numerically distinct so truncation shows up):
        //   storage      [0..2)  = [1.0, 2.0]
        //   lags         [2..4)  = [100.0, 200.0]
        //   anticipated  [4..6)  = [1000.0, 2000.0]
        let state_at_capture = vec![1.0_f64, 2.0, 100.0, 200.0, 1000.0, 2000.0];

        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![
                    BasisStatus::Lower,
                    BasisStatus::Basic,
                    BasisStatus::Upper,
                    BasisStatus::Zero,
                ],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                ],
            },
            base_row_count: 2,
            cut_row_slots: vec![10_u32, 20],
            state_at_capture: state_at_capture.clone(),
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");

        assert_eq!(
            recovered.state_at_capture, state_at_capture,
            "full state must roundtrip bit-exactly"
        );
        assert_eq!(
            &recovered.state_at_capture[4..6],
            &[1000.0, 2000.0],
            "anticipated slice must roundtrip bit-exactly"
        );
        assert_eq!(
            &recovered.state_at_capture[2..4],
            &[100.0, 200.0],
            "lag slice must roundtrip bit-exactly"
        );
        assert_eq!(
            &recovered.state_at_capture[0..2],
            &[1.0, 2.0],
            "storage slice must roundtrip bit-exactly"
        );

        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor at end");
        assert_eq!(f64_cursor, f64_buf.len(), "f64_cursor at end");
    }

    /// Explicit length-field inspection: verify that the recorded
    /// `state_at_capture` length field in the wire payload equals `n_state`
    /// for three layouts (`n_anticipated=0` baseline, `n_anticipated` > 0 small,
    /// and a larger realistic layout).
    #[test]
    fn test_captured_basis_state_at_capture_length_is_recorded_correctly() {
        // Layout 1: small with anticipated.
        // n_state = 2 * (1+1) + 1 * 2 = 6.
        let small = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic; 4],
                row_status: vec![BasisStatus::Basic; 3],
            },
            base_row_count: 2,
            cut_row_slots: vec![],
            state_at_capture: vec![0.0_f64; 6],
            node_id: NodeId(0),
        };
        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        small.to_broadcast_payload(&mut i32_buf, &mut f64_buf);
        // i32_buf layout (Some path):
        //   [0] = 1 (sentinel)
        //   [1] = BASIS_BROADCAST_FORMAT_TAG = 2
        //   [2] = node_id = 0
        //   [3] = col_len = 4
        //   [4] = row_len = 3
        //   [5] = base_row_count = 2
        //   [6] = cut_slot_count = 0
        //   [7] = state_len = 6
        assert_eq!(
            i32_buf[7], 6_i32,
            "state_at_capture length field must be 6 for N=2 L=1 A=1 K=2"
        );

        // Layout 2: n_state == 0 boundary case.
        let empty_state = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic],
                row_status: vec![BasisStatus::Basic],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            state_at_capture: vec![],
            node_id: NodeId(0),
        };
        let mut i32_buf2: Vec<i32> = Vec::new();
        let mut f64_buf2: Vec<f64> = Vec::new();
        empty_state.to_broadcast_payload(&mut i32_buf2, &mut f64_buf2);
        assert_eq!(
            i32_buf2[7], 0_i32,
            "state_at_capture length field must be 0 for empty state"
        );

        // Layout 3: larger realistic layout N=3 L=2 A=2 K_max=3.
        // n_state = 3 * (1+2) + 2 * 3 = 9 + 6 = 15.
        let large_state: Vec<f64> = (0..15).map(|i| f64::from(i) * 10.0).collect();
        let large = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic; 5],
                row_status: vec![BasisStatus::Basic; 5],
            },
            base_row_count: 3,
            cut_row_slots: vec![1_u32, 2],
            state_at_capture: large_state.clone(),
            node_id: NodeId(0),
        };
        let mut i32_buf3: Vec<i32> = Vec::new();
        let mut f64_buf3: Vec<f64> = Vec::new();
        large.to_broadcast_payload(&mut i32_buf3, &mut f64_buf3);
        assert_eq!(
            i32_buf3[7], 15_i32,
            "state_at_capture length field must be 15 for N=3 L=2 A=2 K=3"
        );

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf3,
            &mut i32_cursor,
            &f64_buf3,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");
        assert_eq!(
            &recovered.state_at_capture[9..15],
            &large_state[9..15],
            "anticipated slice (last 6 entries) must roundtrip bit-exactly"
        );
    }

    /// Pre-horizon seeding roundtrip: slot 0 carries sentinel 12345.5.
    #[test]
    fn test_captured_basis_round_trip_with_pre_horizon_seed_in_slot_zero() {
        let state_at_capture = vec![1.0_f64, 2.0, 100.0, 200.0, 12345.5, 0.0];
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![
                    BasisStatus::Lower,
                    BasisStatus::Basic,
                    BasisStatus::Upper,
                    BasisStatus::Zero,
                ],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                    BasisStatus::Lower,
                ],
            },
            base_row_count: 3,
            cut_row_slots: vec![10_u32, 20],
            state_at_capture: state_at_capture.clone(),
            node_id: NodeId(0),
        };

        let mut i32_buf = Vec::new();
        let mut f64_buf = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        let mut i32_cursor = 0;
        let mut f64_cursor = 0;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");

        assert_eq!(recovered.state_at_capture[4], 12345.5);
        assert_eq!(recovered.basis.row_status.len(), 4);
        assert_eq!(recovered.state_at_capture, state_at_capture);
        assert_eq!(i32_cursor, i32_buf.len());
        assert_eq!(f64_cursor, f64_buf.len());
    }

    // ---------------------------------------------------------------------------
    // Layout-invariance tests
    // ---------------------------------------------------------------------------

    /// Locks `state_at_capture.len() == n_state`: the anticipated ring's
    /// outgoing/incoming blocks live inside `n_state`, so widening the ring
    /// does not change `state_at_capture`'s length contract.
    #[test]
    fn test_state_at_capture_length_equals_n_state_after_layout_change() {
        // Layout: N=2 hydros, L=1 PAR lag, n_anticipated=1, k_max=2.
        // n_state = 2*(1+1) + 1*2 = 6.
        //
        // col_status length includes the anticipated-ring column slots;
        // those slots live outside the n_state prefix and must not affect
        // state_at_capture recovery.
        let state_at_capture = vec![1.0_f64, 2.0, 100.0, 200.0, 1000.0, 2000.0];
        let original = CapturedBasis {
            basis: Basis {
                // col_status length includes the anticipated-ring columns.
                // 12 is a representative LP num_cols.
                col_status: vec![BasisStatus::Basic; 12],
                row_status: vec![BasisStatus::Nonbasic; 8],
            },
            base_row_count: 6,
            cut_row_slots: vec![10_u32, 20],
            state_at_capture: state_at_capture.clone(),
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");

        assert_eq!(
            recovered.state_at_capture.len(),
            6,
            "state_at_capture.len() must equal n_state regardless of new LP columns"
        );
        assert_eq!(
            recovered.state_at_capture, state_at_capture,
            "state_at_capture round-trips bit-identically"
        );
        assert_eq!(
            recovered.basis.col_status.len(),
            12,
            "col_status length round-trips bit-identically (including the anticipated-ring column slots)"
        );
        assert_eq!(
            recovered.cut_row_slots, original.cut_row_slots,
            "cut_row_slots round-trips bit-identically"
        );
    }

    /// Locks `BASIS_BROADCAST_FORMAT_TAG` at 2: widening the anticipated
    /// ring does not bump the wire version.
    #[test]
    fn test_basis_broadcast_wire_version_stays_stable_with_state_out_column() {
        use super::BASIS_BROADCAST_FORMAT_TAG;

        // Representative basis with enough col_status entries to include the
        // anticipated-ring columns.
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic; 16], // includes the anticipated-ring block
                row_status: vec![BasisStatus::Basic; 10],
            },
            base_row_count: 8,
            cut_row_slots: vec![],
            state_at_capture: vec![0.0_f64; 6],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // i32_buf layout per workspace.rs to_broadcast_payload:
        //   [0]: sentinel (1)
        //   [1]: BASIS_BROADCAST_FORMAT_TAG
        //   [2]: node_id
        //   [3]: col_status.len()
        //   [4]: row_status.len()
        //   [5]: base_row_count
        //   [6]: cut_row_slots.len()
        //   [7]: state_at_capture.len()
        //   [8..]: col_status elements, then row_status, then cut_row_slots
        assert_eq!(i32_buf[0], 1, "sentinel must be 1");
        assert_eq!(
            i32_buf[1], BASIS_BROADCAST_FORMAT_TAG,
            "wire version field must equal BASIS_BROADCAST_FORMAT_TAG (= 2)"
        );
        assert_eq!(
            BASIS_BROADCAST_FORMAT_TAG, 2,
            "broadcast wire-format version constant must remain stable across releases"
        );

        // Full round-trip confirms no data drift.
        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");

        assert_eq!(recovered.basis.col_status, original.basis.col_status);
        assert_eq!(recovered.basis.row_status, original.basis.row_status);
        assert_eq!(recovered.base_row_count, original.base_row_count);
        assert_eq!(recovered.cut_row_slots, original.cut_row_slots);
        assert_eq!(recovered.state_at_capture, original.state_at_capture);
    }

    /// Full-`Basis` round-trip on a wide-anticipated-ring solve: the
    /// `col_status` count carries the anticipated ring's outgoing+incoming
    /// blocks in the state region (column count is wider by `2*n_anticipated*k_max`
    /// than the no-anticipated baseline), and `cut_row_slots` carries a slot that
    /// the reconstruction path will match by identity. The wire format serialises
    /// `col_status.len()`/`row_status.len()` as live counts, so the wider column
    /// block round-trips byte-for-byte with no format change.
    #[test]
    fn test_captured_basis_round_trip_relocated_anticipated_column_layout() {
        // Anticipated layout sketch: N=2, L=1, A=2, k_max=2.
        //   n_state             = N*(1+L) + A*k_max = 2*2 + 2*2 = 8.
        //   col_status models a relocated-layout LP whose column count grew by
        //   A=2 relative to the no-anticipated layout (z_inflow onward shifted).
        // Two full BasisStatus cycles (7 variants each) so a within-cycle
        // positional mix-up (e.g. cut row written onto a state column) shows as
        // a diff; a shift by exactly 7 is the one class this fixture cannot
        // discriminate.
        let col_status: Vec<BasisStatus> = [
            BasisStatus::Lower,
            BasisStatus::Basic,
            BasisStatus::Upper,
            BasisStatus::Zero,
            BasisStatus::Nonbasic,
            BasisStatus::Superbasic,
            BasisStatus::Fixed,
        ]
        .iter()
        .copied()
        .cycle()
        .take(14)
        .collect();
        // 6 template rows (uniform) then 2 distinct cut rows.
        let row_status: Vec<BasisStatus> = vec![
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Basic,
            BasisStatus::Zero,
        ];
        let state_at_capture = vec![1.0_f64, 2.0, 100.0, 200.0, 1000.0, 2000.0, 3000.0, 4000.0];

        let original = CapturedBasis {
            basis: Basis {
                col_status: col_status.clone(),
                row_status: row_status.clone(),
            },
            base_row_count: 6,
            cut_row_slots: vec![7_u32, 42],
            state_at_capture: state_at_capture.clone(),
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        )
        .expect("round-trip must not fail")
        .expect("sentinel is 1; must return Some");

        // The wider column block (incl. the anticipated-ring columns) round-trips
        // byte-for-byte.
        assert_eq!(
            recovered.basis.col_status, col_status,
            "relocated-layout col_status must round-trip bit-identically"
        );
        assert_eq!(
            recovered.basis.row_status, row_status,
            "template+cut row_status must round-trip bit-identically"
        );
        assert_eq!(recovered.base_row_count, 6);
        assert_eq!(
            recovered.cut_row_slots,
            vec![7_u32, 42],
            "cut_row_slots (the slot-identity key) must round-trip bit-identically"
        );
        assert_eq!(
            recovered.state_at_capture, state_at_capture,
            "state_at_capture must round-trip bit-identically"
        );
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
        assert_eq!(f64_cursor, f64_buf.len(), "f64_cursor must be at end");
    }

    // ---------------------------------------------------------------------------
    // ProfiledSolver integration
    // ---------------------------------------------------------------------------

    /// A freshly constructed workspace must expose a solver whose
    /// `current_profile()` equals `HighsProfile::default()`.
    ///
    /// Confirms that `ProfiledSolver::new` wraps the inner solver without
    /// issuing any FFI calls and that `WorkspacePool::new` correctly initialises
    /// every workspace.
    ///
    /// Gated to `feature = "highs"` because it embeds the backend-specific
    /// default-profile value `HighsProfile::default()`; a CLP analog is a
    /// separate assertion.
    #[cfg(all(test, feature = "highs"))]
    #[test]
    fn workspace_solver_initialised_with_default_profile() {
        use cobre_solver::HighsProfile;

        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert_eq!(
                ws.solver.current_profile(),
                &HighsProfile::default(),
                "solver.current_profile() must equal HighsProfile::default() after construction"
            );
        }

        // Also verify SolverWorkspace::new directly.
        let ws = SolverWorkspace::new(
            0,
            0,
            MockSolver,
            PatchBuffer::new(ctx.training_ctx().state, &[], &[]),
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
        );
        assert_eq!(
            ws.solver.current_profile(),
            &HighsProfile::default(),
            "SolverWorkspace::new must initialise solver with default profile"
        );
    }
}
