//! Per-iteration scratch buffers for the SDDP training loop.
//!
//! [`IterationScratch`] holds reusable scratch fields allocated once and cleared
//! each iteration, avoiding per-iteration heap allocation.
//!
//! ## Startup allocations
//!
//! [`IterationScratch::new`] performs `O(max_local_fwd * num_stages)`
//! allocations at training-run startup:
//! - N × [`TrajectoryRecord`](crate::TrajectoryRecord) with `state: Vec<f64>` of length `n_state`.
//! - `num_stages` × `RowBatch` for cut-batch scratch.
//! - `num_stages` × `RowBatch` for bake scratch.
//! - `num_stages` × `StageTemplate` for baked templates.
//! - One `PatchBuffer`, one `lb_cut_batch`, one `CutRowMap`.
//!
//! These are amortized across all iterations of a training run.

use cobre_solver::{RowBatch, StageTemplate};

use crate::{
    context::StageContext, cut::CutRowMap, lower_bound::LbEvalScratch, lp_builder::PatchBuffer,
    TrajectoryRecord,
};

/// Per-training-run iteration scratch owned by `TrainingSession`.
///
/// All fields are allocated once in `IterationScratch::new` at the start
/// of a training run and are cleared/reused across iterations. The
/// iteration loop body (forward pass, backward pass, cut management,
/// lower-bound evaluation) writes into these buffers in-place; no
/// per-iteration heap allocation is permitted on this struct's fields.
///
/// This struct intentionally excludes the backward-pass-specific scratch
/// (currently the `bwd_*_buf` fields on `TrainingSession`) because those
/// are owned by `BackwardPassState`.
pub(crate) struct IterationScratch {
    /// Patch buffer for lower-bound LP patching (single solver path).
    pub patch_buf: PatchBuffer,
    /// Per-scenario per-stage trajectory records; sized `max_local_fwd * num_stages`.
    pub records: Vec<TrajectoryRecord>,
    /// Cut row batches built during the backward pass, one per stage.
    pub cut_batches: Vec<RowBatch>,
    /// Cut row batch used exclusively for lower-bound evaluation (stage 0).
    pub lb_cut_batch: RowBatch,
    /// Baked stage templates (structural copies of base templates + active cuts).
    pub baked_templates: Vec<StageTemplate>,
    /// Row batches used to build the active-cut rows before baking, one per stage.
    pub bake_row_batches: Vec<RowBatch>,
    /// Cut row map for the lower-bound LP (tracks row positions within stage 0 template).
    pub lb_cut_row_map: CutRowMap,
    /// Per-evaluation scratch buffers for lower-bound evaluation (reused across iterations).
    pub lb_scratch: LbEvalScratch,
}

impl IterationScratch {
    /// Allocate and initialise all iteration scratch buffers.
    ///
    /// Performs the exact seven allocations previously inlined in
    /// `TrainingSession::new` and applies the pre-bake loop so that
    /// `baked_templates[t]` is a structural copy of
    /// `stage_ctx.templates[t]` with an empty cut batch applied before
    /// the first iteration begins.
    ///
    /// # Arguments
    ///
    /// * `max_local_fwd` — maximum number of forward passes assigned to this rank.
    /// * `num_stages` — number of study stages.
    /// * `n_state` — state-vector dimension (used to size `records[i].state`).
    /// * `fcf_pool_0_capacity` — capacity of the stage-0 FCF pool (for `CutRowMap`).
    /// * `template_0_num_rows` — number of rows in the stage-0 template (for `CutRowMap`).
    /// * `hydro_count` — number of hydro plants (for `PatchBuffer`).
    /// * `max_par_order` — maximum PAR model order (for `PatchBuffer`).
    /// * `stage_ctx` — stage context providing base templates for the pre-bake loop.
    pub(crate) fn new(
        max_local_fwd: usize,
        num_stages: usize,
        n_state: usize,
        fcf_pool_0_capacity: usize,
        template_0_num_rows: usize,
        hydro_count: usize,
        max_par_order: usize,
        stage_ctx: &StageContext<'_>,
    ) -> Self {
        // ── Trajectory records ─────────────────────────────────────────────
        // Each record is constructed fresh to avoid cloning empty Vecs;
        // `Vec::new()` is strictly cheaper than cloning a capacity-0 Vec.
        let records: Vec<TrajectoryRecord> = (0..max_local_fwd * num_stages)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                state: vec![0.0; n_state],
            })
            .collect();

        // ── Patch buffer ───────────────────────────────────────────────────
        // Standalone patch buffer for the lower bound evaluation which uses
        // the single `solver` argument directly. Trailing `0, 0` preserve
        // the pre-refactor argument list verbatim.
        let patch_buf = PatchBuffer::new(hydro_count, max_par_order, 0, 0);

        // ── Cut row batch buffers (reused across iterations) ───────────────
        let cut_batches: Vec<RowBatch> = (0..num_stages)
            .map(|_| RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            })
            .collect();
        let lb_cut_batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };

        // ── Template baking buffers ────────────────────────────────────────
        let mut baked_templates: Vec<StageTemplate> =
            (0..num_stages).map(|_| StageTemplate::empty()).collect();
        let bake_row_batches: Vec<RowBatch> = (0..num_stages)
            .map(|_| RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            })
            .collect();

        // Pre-bake every stage template with an empty cut batch so that
        // iteration 1's forward and backward passes can use the baked
        // load path. Empty-batch bake is a structural copy of the base
        // template.
        for t in 0..num_stages {
            cobre_solver::bake_rows_into_template(
                &stage_ctx.templates[t],
                &bake_row_batches[t],
                &mut baked_templates[t],
            );
        }

        // ── Lower-bound cut row map ────────────────────────────────────────
        let lb_cut_row_map = CutRowMap::new(fcf_pool_0_capacity, template_0_num_rows);

        // ── Lower-bound evaluation scratch ────────────────────────────────
        // Allocated empty; populated on the first evaluate_lower_bound call
        // and reused (without reallocation) on every subsequent iteration.
        let lb_scratch = LbEvalScratch::new();

        Self {
            patch_buf,
            records,
            cut_batches,
            lb_cut_batch,
            baked_templates,
            bake_row_batches,
            lb_cut_row_map,
            lb_scratch,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop
)]
mod tests {
    use cobre_solver::StageTemplate;

    use super::IterationScratch;
    use crate::context::StageContext;

    fn minimal_template() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn make_stage_ctx(templates: &[StageTemplate]) -> StageContext<'_> {
        StageContext {
            templates,
            base_rows: &[],
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        }
    }

    /// Verify that `IterationScratch::new` sizes all Vecs correctly.
    #[test]
    fn iteration_scratch_new_sizes_vecs_correctly() {
        let max_local_fwd = 2;
        let num_stages = 3;
        let n_state = 4;
        let fcf_pool_0_capacity = 10;
        let template_0_num_rows = 5;
        let hydro_count = 1;
        let max_par_order = 1;

        let templates = vec![minimal_template(); num_stages];
        let stage_ctx = make_stage_ctx(&templates);

        let scratch = IterationScratch::new(
            max_local_fwd,
            num_stages,
            n_state,
            fcf_pool_0_capacity,
            template_0_num_rows,
            hydro_count,
            max_par_order,
            &stage_ctx,
        );

        assert_eq!(
            scratch.records.len(),
            max_local_fwd * num_stages,
            "records must be pre-sized to max_local_fwd * num_stages"
        );
        assert_eq!(
            scratch.records[0].state.len(),
            n_state,
            "each record state must have n_state elements"
        );
        assert_eq!(
            scratch.cut_batches.len(),
            num_stages,
            "cut_batches must have one RowBatch per stage"
        );
        assert_eq!(
            scratch.bake_row_batches.len(),
            num_stages,
            "bake_row_batches must have one RowBatch per stage"
        );
        assert_eq!(
            scratch.baked_templates.len(),
            num_stages,
            "baked_templates must have one StageTemplate per stage"
        );
    }

    /// Verify that `IterationScratch::new` pre-bakes all templates so that
    /// each `baked_templates[t]` matches the structural shape of
    /// `stage_ctx.templates[t]`.
    #[test]
    fn iteration_scratch_new_pre_bakes_templates() {
        let max_local_fwd = 2;
        let num_stages = 3;
        let n_state = 4;
        let fcf_pool_0_capacity = 10;
        let template_0_num_rows = 5;
        let hydro_count = 1;
        let max_par_order = 1;

        let templates = vec![minimal_template(); num_stages];
        let stage_ctx = make_stage_ctx(&templates);

        let scratch = IterationScratch::new(
            max_local_fwd,
            num_stages,
            n_state,
            fcf_pool_0_capacity,
            template_0_num_rows,
            hydro_count,
            max_par_order,
            &stage_ctx,
        );

        for t in 0..num_stages {
            assert_eq!(
                scratch.baked_templates[t].num_rows, stage_ctx.templates[t].num_rows,
                "baked_templates[{t}].num_rows must match stage_ctx template"
            );
            assert_eq!(
                scratch.baked_templates[t].num_cols, stage_ctx.templates[t].num_cols,
                "baked_templates[{t}].num_cols must match stage_ctx template"
            );
        }
    }
}
