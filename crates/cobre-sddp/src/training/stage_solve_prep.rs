//! Shared stage-LP solve-preparation pipeline (`pin → row_patches → ncs_patch →
//! commit`): the single owner the forward pass, backward pass, simulation
//! pipeline, and lower-bound evaluation all route through, so every solve site
//! patches the LP identically (the "patch NCS identically" contract, D15).
//!
//! Home is `training/`, not `lp/builder/`: the pipeline needs `crate::context`,
//! `crate::workspace`, and `crate::noise`, which `training/` already depends on,
//! whereas `lp/builder/` sits below `training/` in the crate layering — hosting it
//! there would invert the dependency direction.

use cobre_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    lp::builder::{PatchBuffer, StateBox},
    lp::indexer::BlockGrid,
    noise::{
        apply_ncs_col_bounds, transform_inflow_noise, transform_load_noise, transform_ncs_noise,
    },
    setup::node_graph::StageIdx,
    workspace::ScratchBuffers,
};

/// The state slice this stage solve pins as incoming state; the caller resolves
/// it per site (forward `current_state`, backward `x_hat`, lower-bound
/// `initial_state`, simulation `current_state`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct StateSource<'a>(pub &'a [f64]);

/// How the water-balance noise buffer this solve reads
/// (`scratch.z_inflow_rhs_buf`) is populated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InflowNoise {
    /// Fill the buffers from `raw_noise` (forward, backward, simulation).
    Transform,
    /// The caller has already filled the buffers (the lower bound's PAR-batch
    /// precompute).
    PreBuilt,
}

/// Per-call variation points for [`StageSolvePrep::run`] — the divergences
/// among the four solve sites.
///
/// The NCS availability patch is not a variation point: [`StageSolvePrep::run`]
/// gates it on `n_stochastic_ncs() > 0` internally, the same gate every solve
/// site applies.
pub(crate) struct StageSolvePrepParams<'a> {
    /// Which slice this solve pins as incoming state.
    pub state_source: StateSource<'a>,
    /// How the inflow-noise buffers are populated.
    pub inflow_noise: InflowNoise,
    /// This solve's realized noise draw, laid out `[hydro | load-bus | NCS]`.
    pub raw_noise: &'a [f64],
}

/// The single owner of the shared stage-LP solve-preparation pipeline.
pub(crate) struct StageSolvePrep;

impl StageSolvePrep {
    /// Executes `pin → row_patches → ncs_patch → commit` over caller-owned `&mut`
    /// scratch: allocates nothing.
    pub(crate) fn run<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        params: &StageSolvePrepParams<'_>,
    ) where
        S: SolverInterface,
    {
        let producer_box = if stage.0 == 0 {
            None
        } else {
            Some(ctx.state_box(StageIdx(stage.0 - 1)))
        };
        Self::run_with_producer_box(
            solver,
            patch_buf,
            scratch,
            ctx,
            training_ctx,
            stage,
            params,
            producer_box,
        );
    }

    /// Test-support-only variant of [`run`](Self::run) for a deliberate
    /// counterfactual pin that lies outside its producer's admissible box on
    /// purpose (an LP-sensitivity probe): same pipeline, but never checks the
    /// pinned state against a producer box.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn run_ignoring_producer_box<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        params: &StageSolvePrepParams<'_>,
    ) where
        S: SolverInterface,
    {
        Self::run_with_producer_box(
            solver,
            patch_buf,
            scratch,
            ctx,
            training_ctx,
            stage,
            params,
            None,
        );
    }

    // Rationale (too_many_arguments): solver, patch_buf, and scratch are three
    // disjoint mutable borrows of the workspace that must stay separate
    // parameters — a bundling struct would reborrow the whole workspace and
    // defeat the split; ctx/training_ctx/params are distinct immutable contexts.
    #[allow(clippy::too_many_arguments)]
    fn run_with_producer_box<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        params: &StageSolvePrepParams<'_>,
        producer_box: Option<&StateBox>,
    ) where
        S: SolverInterface,
    {
        let pinned_state = params.state_source.0;

        if params.inflow_noise == InflowNoise::Transform {
            transform_inflow_noise(params.raw_noise, stage, pinned_state, training_ctx, scratch);
        }

        let load_blocks = if ctx.load_bus_indices.is_empty() {
            0
        } else {
            ctx.block_count(stage)
        };
        transform_load_noise(
            params.raw_noise,
            training_ctx.stochastic,
            stage,
            load_blocks,
            &mut scratch.load_rhs_buf,
        );

        patch_buf.fill_col_state_patches(
            training_ctx.state,
            pinned_state,
            &ctx.template(stage).col_scale,
            producer_box,
        );
        if !ctx.load_bus_indices.is_empty() {
            let grid = BlockGrid::new(load_blocks, training_ctx.study_dims.max_deficit_segments);
            patch_buf.fill_load_patches(
                ctx.geometry_per_stage[stage.0].load_balance,
                grid,
                &scratch.load_rhs_buf,
                ctx.load_bus_indices,
                &ctx.template(stage).row_scale,
            );
        }
        patch_buf.fill_z_inflow_patches(
            training_ctx.state,
            &scratch.z_inflow_rhs_buf,
            &ctx.template(stage).row_scale,
        );

        let cp = patch_buf.state_col_patch_count();
        solver.set_col_bounds(
            &patch_buf.col_indices[..cp],
            &patch_buf.col_lower[..cp],
            &patch_buf.col_upper[..cp],
        );

        let pc = patch_buf.forward_patch_count();
        solver.set_row_bounds(
            &patch_buf.indices[..pc],
            &patch_buf.lower[..pc],
            &patch_buf.upper[..pc],
        );

        if training_ctx.stochastic.n_stochastic_ncs() > 0 {
            transform_ncs_noise(
                params.raw_noise,
                training_ctx.stochastic,
                stage,
                ctx.block_count(stage),
                ctx.ncs_max_gen,
                ctx.ncs_allow_curtailment,
                &mut scratch.ncs_col_lower_buf,
                &mut scratch.ncs_col_upper_buf,
            );
            // Stage id is the dormancy key (NOT the index `stage`; filtered
            // placeholder stages can shift the id off the index).
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let stage_id = training_ctx
                .stages
                .get(stage.0)
                .map_or(stage.0 as i32, |s| s.id);
            apply_ncs_col_bounds(
                solver,
                scratch,
                &ctx.geometry_per_stage[stage.0],
                ctx.ncs_stochastic_dense_col,
                ctx.ncs_stochastic_windows,
                stage_id,
            );
        }
    }
}

#[cfg(test)]
mod tests;
