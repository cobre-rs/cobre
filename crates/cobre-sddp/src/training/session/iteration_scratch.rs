//! Per-iteration scratch buffers for the SDDP training loop.
//!
//! Allocated once at training-run startup and reused every iteration to avoid
//! per-iteration heap allocation.

use cobre_solver::FreezeScratch;
use cobre_solver::freeze_rows_into_template;
use cobre_solver::{RowBatch, StageTemplate};

use crate::{
    context::{StageContext, TrainingContext},
    cut::CutRowMap,
    lower_bound::LbEvalScratch,
    lp::builder::PatchBuffer,
    setup::{NodeId, node_graph::StageIdx},
    solver_stats::SolverStatsDelta,
    trajectory::TrajectoryRecord,
    workspace::{ScratchBuffers, WorkspaceSizing},
};

/// Per-training-run iteration scratch owned by `TrainingSession`.
///
/// Excludes backward-pass-specific scratch, which `BackwardPassState` owns.
pub(crate) struct IterationScratch {
    /// Patch buffer for lower-bound LP patching (single solver path).
    pub patch_buf: PatchBuffer,
    /// Per-scenario per-stage trajectory records; sized `max_local_fwd * num_stages`.
    pub records: Vec<TrajectoryRecord>,
    /// Cut row batches built during the backward pass, one per pool.
    pub cut_batches: Vec<RowBatch>,
    /// Cut row batch used exclusively for lower-bound evaluation (stage 0).
    pub lb_cut_batch: RowBatch,
    /// Frozen LP templates, one per POOL (not per stage): each is its pool's
    /// base stage template plus that pool's own active cuts. A per-stage overlay
    /// is the wrong-but-compiling alternative — on a branching graph one stage
    /// holds several nodes with distinct pools, so a per-stage frozen row batch
    /// would bake one node's cuts into every sibling's LP.
    pub frozen_templates: Vec<StageTemplate>,
    /// Row batches used to build the active-cut rows before freeze, one per pool.
    pub freeze_row_batches: Vec<RowBatch>,
    /// Cut row map for the lower-bound LP.
    pub lb_cut_row_map: CutRowMap,
    /// Per-evaluation scratch buffers for lower-bound evaluation.
    pub lb_scratch: LbEvalScratch,
    /// Noise/NCS-transform scratch for the lower-bound path, the same shape
    /// [`StageSolvePrep::run`] reads on every other solve site.
    ///
    /// [`StageSolvePrep::run`]: crate::training::stage_solve_prep::StageSolvePrep::run
    pub lb_noise_scratch: ScratchBuffers,
    /// Scratch buffers for `freeze_rows_into_template` (count/emit-pass temporaries).
    pub(crate) freeze_scratch: FreezeScratch,
    /// Per-path probability weights for the exact upper-bound reduction, filled
    /// only on an enumerated forward; empty on the sampled path.
    pub(crate) ub_path_weights: Vec<f64>,
    /// Per-path per-stage immediate costs (this rank's paths, path-major) fed to
    /// the nested risk-adjusted upper bound. Filled only on an enumerated forward
    /// under an effective `CVaR`; empty otherwise.
    pub(crate) ub_stage_costs: Vec<f64>,
    /// Packed per-stage forward solver-stat scalars, the cross-rank allreduce
    /// input in `run_forward_phase` (empty until the first forward phase).
    pub(crate) fwd_stats_pack_local: Vec<f64>,
    /// Allreduced (summed) counterpart of [`Self::fwd_stats_pack_local`].
    pub(crate) fwd_stats_pack_global: Vec<f64>,
    /// Per-stage forward stats unpacked from [`Self::fwd_stats_pack_global`] for
    /// the rank-0 solver-stats log.
    pub(crate) fwd_stats_unpacked: Vec<SolverStatsDelta>,
}

impl IterationScratch {
    /// Allocate all iteration scratch buffers; pre-freeze each `frozen_templates[p]`
    /// as an empty-cut-batch structural copy of pool `p`'s base stage template so
    /// iteration 1 can use the frozen load path.
    pub(crate) fn new(
        max_local_fwd: usize,
        pool_stage: &[StageIdx],
        lb_root_pool_capacity: usize,
        template_0_num_rows: usize,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
    ) -> Self {
        let n_pools = pool_stage.len();
        let n_state = training_ctx.state.n_state;
        let num_stages = training_ctx.horizon.num_stages();
        let records: Vec<TrajectoryRecord> = (0..max_local_fwd * num_stages)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: vec![0.0; n_state],
            })
            .collect();

        let patch_buf = crate::lower_bound::lower_bound_patch_buffer(training_ctx.state, stage_ctx);

        let empty_row_batch = || RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        let cut_batches: Vec<RowBatch> = (0..n_pools).map(|_| empty_row_batch()).collect();
        let lb_cut_batch = empty_row_batch();

        let mut frozen_templates: Vec<StageTemplate> =
            (0..n_pools).map(|_| StageTemplate::empty()).collect();
        let freeze_row_batches: Vec<RowBatch> = (0..n_pools).map(|_| empty_row_batch()).collect();

        let mut freeze_scratch = FreezeScratch::new();

        for p in 0..n_pools {
            let t = pool_stage[p];
            freeze_rows_into_template(
                stage_ctx.template(t),
                &freeze_row_batches[p],
                &mut frozen_templates[p],
                &mut freeze_scratch,
            );
        }

        let lb_cut_row_map = CutRowMap::new(lb_root_pool_capacity, template_0_num_rows);

        let lb_scratch = LbEvalScratch::new();

        let lb_noise_scratch =
            ScratchBuffers::new(training_ctx, stage_ctx, WorkspaceSizing::default());

        Self {
            patch_buf,
            records,
            cut_batches,
            lb_cut_batch,
            frozen_templates,
            freeze_row_batches,
            lb_cut_row_map,
            lb_scratch,
            lb_noise_scratch,
            freeze_scratch,
            ub_path_weights: Vec::with_capacity(max_local_fwd),
            ub_stage_costs: Vec::with_capacity(max_local_fwd * num_stages),
            fwd_stats_pack_local: Vec::new(),
            fwd_stats_pack_global: Vec::new(),
            fwd_stats_unpacked: Vec::new(),
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
    use crate::lp::builder::StageGeometry;
    use crate::lp::indexer::HydroSys;
    use crate::setup::node_graph::StageIdx;
    use crate::test_support::{
        StageContextFixture, TrainingContextFixture, equipment_free_geometry, state_layout,
        state_layout_full, state_layout_with_transit_buckets,
    };

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
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn make_stage_ctx<'a>(
        templates: &'a [StageTemplate],
        geometry_per_stage: &'a [StageGeometry],
    ) -> StageContextFixture<'a> {
        StageContextFixture::new(templates, &[], geometry_per_stage)
    }

    #[test]
    fn iteration_scratch_new_sizes_vecs_correctly() {
        let max_local_fwd = 2;
        let num_stages = 3;
        let lb_root_pool_capacity = 10;
        let template_0_num_rows = 5;

        let templates = vec![minimal_template(); num_stages];
        let geometry = equipment_free_geometry(&vec![0; num_stages]);
        let fixture = make_stage_ctx(&templates, &geometry);
        let stage_ctx = fixture.ctx();
        let training_fixture =
            TrainingContextFixture::new(state_layout(1, 1)).num_stages(num_stages);
        let training_ctx = training_fixture.training_ctx();
        let n_state = training_ctx.state.n_state;

        let scratch = IterationScratch::new(
            max_local_fwd,
            &(0..num_stages).map(StageIdx).collect::<Vec<StageIdx>>(),
            lb_root_pool_capacity,
            template_0_num_rows,
            &training_ctx,
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
            scratch.freeze_row_batches.len(),
            num_stages,
            "freeze_row_batches must have one RowBatch per stage"
        );
        assert_eq!(
            scratch.frozen_templates.len(),
            num_stages,
            "frozen_templates must have one StageTemplate per stage"
        );
    }

    #[test]
    fn iteration_scratch_new_pre_freezes_templates() {
        let max_local_fwd = 2;
        let num_stages = 3;
        let lb_root_pool_capacity = 10;
        let template_0_num_rows = 5;

        let templates = vec![minimal_template(); num_stages];
        let geometry = equipment_free_geometry(&vec![0; num_stages]);
        let fixture = make_stage_ctx(&templates, &geometry);
        let stage_ctx = fixture.ctx();
        let training_fixture =
            TrainingContextFixture::new(state_layout(1, 1)).num_stages(num_stages);
        let training_ctx = training_fixture.training_ctx();

        let scratch = IterationScratch::new(
            max_local_fwd,
            &(0..num_stages).map(StageIdx).collect::<Vec<StageIdx>>(),
            lb_root_pool_capacity,
            template_0_num_rows,
            &training_ctx,
            &stage_ctx,
        );

        for t in 0..num_stages {
            assert_eq!(
                scratch.frozen_templates[t].num_rows, stage_ctx.templates[t].num_rows,
                "frozen_templates[{t}].num_rows must match stage_ctx template"
            );
            assert_eq!(
                scratch.frozen_templates[t].num_cols, stage_ctx.templates[t].num_cols,
                "frozen_templates[{t}].num_cols must match stage_ctx template"
            );
        }
    }

    /// Regression: `IterationScratch::new` must size the lower-bound
    /// patch buffer's anticipated capacity to `n_anticipated * k_max`.
    /// Undersizing panics in `fill_col_state_patches`.
    #[test]
    fn iteration_scratch_new_sizes_patch_buffer_for_anticipated_thermals() {
        let max_local_fwd = 1;
        let num_stages = 2;
        let lb_root_pool_capacity = 4;
        let template_0_num_rows = 4;
        let hydro_count = 2;
        let max_par_order = 1;
        let n_anticipated = 3;
        let k_max = 2;

        let templates = vec![minimal_template(); num_stages];
        let geometry = equipment_free_geometry(&vec![0; num_stages]);
        let fixture = make_stage_ctx(&templates, &geometry);
        let stage_ctx = fixture.ctx();
        let state = state_layout_full(hydro_count, max_par_order, vec![k_max; n_anticipated]);
        assert_eq!(
            state.k_max, k_max,
            "ring_size must resolve to k_max for uniform leads"
        );
        let training_fixture = TrainingContextFixture::new(state).num_stages(num_stages);
        let training_ctx = training_fixture.training_ctx();

        let scratch = IterationScratch::new(
            max_local_fwd,
            &(0..num_stages).map(StageIdx).collect::<Vec<StageIdx>>(),
            lb_root_pool_capacity,
            template_0_num_rows,
            &training_ctx,
            &stage_ctx,
        );

        // Forward-pass layout capacity:
        //   M*B + N
        // with M = 0 and B = 0 in the LB patch buffer (no load patches); the A*K
        // anticipated-state slots size the column region only, never this row region.
        let expected_capacity = hydro_count;
        assert_eq!(
            scratch.patch_buf.indices.len(),
            expected_capacity,
            "patch_buf indices length must equal M*B + N regardless of A*K",
        );
        assert_eq!(
            scratch.patch_buf.lower.len(),
            expected_capacity,
            "patch_buf lower length must equal M*B + N regardless of A*K",
        );
        assert_eq!(
            scratch.patch_buf.upper.len(),
            expected_capacity,
            "patch_buf upper length must equal M*B + N regardless of A*K",
        );

        // forward_patch_count is 0 before any load or z-inflow patch has been filled.
        assert_eq!(
            scratch.patch_buf.forward_patch_count(),
            0,
            "forward_patch_count must be zero before any fill call",
        );
    }

    /// Regression: with `n_anticipated == 0`, the patch buffer must size
    /// identically to the pre-anticipated layout.
    #[test]
    fn iteration_scratch_new_patch_buffer_zero_anticipated_unchanged() {
        let max_local_fwd = 1;
        let num_stages = 2;
        let lb_root_pool_capacity = 4;
        let template_0_num_rows = 4;
        let hydro_count = 2;
        let max_par_order = 1;

        let templates = vec![minimal_template(); num_stages];
        let geometry = equipment_free_geometry(&vec![0; num_stages]);
        let fixture = make_stage_ctx(&templates, &geometry);
        let stage_ctx = fixture.ctx();
        let training_fixture =
            TrainingContextFixture::new(state_layout(hydro_count, max_par_order))
                .num_stages(num_stages);
        let training_ctx = training_fixture.training_ctx();

        let scratch = IterationScratch::new(
            max_local_fwd,
            &(0..num_stages).map(StageIdx).collect::<Vec<StageIdx>>(),
            lb_root_pool_capacity,
            template_0_num_rows,
            &training_ctx,
            &stage_ctx,
        );

        // With A=0 and K=0, row capacity is still M*B + N = 0 + N = N.
        let expected_capacity = hydro_count;
        assert_eq!(
            scratch.patch_buf.indices.len(),
            expected_capacity,
            "zero-anticipated patch_buf must match the pre-anticipated layout",
        );
    }

    /// Regression: `IterationScratch::new` must size the lower-bound patch
    /// buffer's column region to include `n_buckets` travel-time bucket slots;
    /// omitting them panics in `fill_col_state_patches`.
    #[test]
    fn iteration_scratch_new_sizes_patch_buffer_for_transit_buckets() {
        let max_local_fwd = 1;
        let num_stages = 2;
        let lb_root_pool_capacity = 4;
        let template_0_num_rows = 4;
        let hydro_count = 2;
        let max_par_order = 1;
        let n_buckets = 3;

        let templates = vec![minimal_template(); num_stages];
        let geometry = equipment_free_geometry(&vec![0; num_stages]);
        let fixture = make_stage_ctx(&templates, &geometry);
        let stage_ctx = fixture.ctx();
        let bucket_order = (0..n_buckets).map(|d| (HydroSys::new(0), d)).collect();
        let state =
            state_layout_with_transit_buckets(hydro_count, max_par_order, bucket_order, vec![]);
        let training_fixture = TrainingContextFixture::new(state).num_stages(num_stages);
        let training_ctx = training_fixture.training_ctx();

        let scratch = IterationScratch::new(
            max_local_fwd,
            &(0..num_stages).map(StageIdx).collect::<Vec<StageIdx>>(),
            lb_root_pool_capacity,
            template_0_num_rows,
            &training_ctx,
            &stage_ctx,
        );

        // Column-bound region: N*(1+L) + n_buckets + A*K + W = 2*2 + 3 + 0 + 0 = 7.
        assert_eq!(
            scratch.patch_buf.state_col_patch_count(),
            hydro_count * (1 + max_par_order) + n_buckets,
            "patch_buf column region must include n_buckets bucket slots",
        );
    }
}
