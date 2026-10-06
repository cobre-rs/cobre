//! Forward pass execution for SDDP training.
//!
//! `run_forward_pass` simulates scenario trajectories via stage LPs with the
//! current Future Cost Function. Outputs `TrajectoryRecord`s and [`ForwardResult`]
//! for the backward pass and synchronisation step. Parallelised across workers
//! with deterministic scenario assignment; the per-scenario hot loop lives in
//! `forward_pass_state::ForwardPassState::run`.
//!
//! This `mod.rs` owns the result structs ([`ForwardResult`], [`SyncResult`]), the
//! per-call parameter bundles (`ForwardPassBatch`, `StageKey`), and the thin
//! `run_forward_pass` shim that delegates to `ForwardPassState::run`.

use crate::{
    cut::pool::CutPool,
    dcs::DcsParams,
    setup::node_graph::{NodePos, StageIdx},
    solver_stats::SolverStatsDelta,
};

// Consumed only by the test-support `run_forward_pass` shim / `ForwardPassBatch`
// below; the import gate mirrors their `cfg` so a featureless build stays clean.
#[cfg(any(test, feature = "test-support"))]
use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    error::SddpError,
    trajectory::TrajectoryRecord,
    workspace::{BasisStore, SolverWorkspace},
};
#[cfg(any(test, feature = "test-support"))]
use cobre_core::TrainingEvent;
#[cfg(any(test, feature = "test-support"))]
use cobre_solver::{ActiveProfile, SolverInterface, StageTemplate};
#[cfg(any(test, feature = "test-support"))]
use std::sync::mpsc::Sender;

mod basis_capture;
mod delta_cut_batch;
mod enumerated;
mod sampler;
mod stage_solve;
mod stats_aggregation;

#[cfg(test)]
mod tests;

pub use delta_cut_batch::build_delta_cut_row_batch_into;
pub use sampler::build_sampler_from_ctx;
pub use stats_aggregation::{ForwardBound, NestedUbScratch, sync_forward};

pub(crate) use basis_capture::write_capture_metadata;
pub use enumerated::EnumeratedForwardScratch;
pub(crate) use enumerated::{EnumeratedForwardResult, EnumeratedParams, run_enumerated_forward};
pub(crate) use stage_solve::run_forward_stage;

/// Local statistics from one rank's forward pass.
///
/// Carries the individual per-scenario trajectory costs in global scenario
/// index order (scenario 0 first, scenario N-1 last). The synchronisation
/// step gathers these costs from all ranks via `allgatherv` and performs
/// canonical-order summation to produce bit-identical statistics regardless
/// of the number of MPI ranks or intra-rank worker threads.
///
/// Does not contain lower bound estimate (evaluated separately after backward pass).
#[derive(Debug, Clone)]
#[must_use]
pub struct ForwardResult {
    /// Per-scenario trajectory costs in global scenario index order.
    pub scenario_costs: Vec<f64>,

    /// Wall-clock time in milliseconds for this rank's forward pass.
    pub elapsed_ms: u64,

    /// Number of LP solves performed during this forward pass.
    pub lp_solves: u64,

    /// Aggregate non-solve work (load-model + set-bounds + basis-set) inside the
    /// parallel region, summed across workers, in milliseconds.
    pub setup_time_ms: u64,

    /// Load-imbalance component of parallel overhead (slowest worker minus
    /// average worker total), in milliseconds.
    pub load_imbalance_ms: u64,

    /// True rayon scheduling overhead (parallel wall minus the slowest worker
    /// total, clamped to zero), in milliseconds.
    pub scheduling_overhead_ms: u64,

    /// Per-stage solver-counter deltas, summed across all workers and scenarios
    /// this rank processed; length equals the study stage count.
    pub stage_stats: Vec<SolverStatsDelta>,
}

/// Global upper bound statistics from forward synchronisation step.
#[derive(Debug, Clone)]
#[must_use]
pub struct SyncResult {
    /// Sample mean of total trajectory costs across all ranks.
    pub global_ub_mean: f64,

    /// Bessel-corrected sample standard deviation of total trajectory costs.
    pub global_ub_std: f64,

    /// 95% confidence interval half-width: `1.96 * std / sqrt(N)`.
    pub ci_95_half_width: f64,

    /// Wall-clock time in milliseconds for the forward synchronization call.
    pub sync_time_ms: u64,
}

/// Bundled scalar parameters for one forward pass invocation.
///
/// Groups the per-iteration, per-rank scalar arguments that are forwarded
/// from [`crate::train`] into [`run_forward_pass`].
#[cfg(any(test, feature = "test-support"))]
pub struct ForwardPassBatch<'a> {
    /// Number of forward-pass scenarios assigned to this rank.
    pub local_forward_passes: usize,
    /// Total forward passes across all MPI ranks. Used for LHS stratification
    /// in the sampler (`total_scenarios` field of `SampleRequest`) and for
    /// sizing the LHS permutation scratch buffer. Must equal the study-level
    /// `forward_passes` parameter, NOT the per-rank local count.
    pub total_forward_passes: usize,
    /// Current training iteration (0-based; used for seed derivation).
    pub iteration: u64,
    /// Global index of this rank's first forward pass for seed derivation.
    pub fwd_offset: usize,
    /// Optional channel for emitting one [`TrainingEvent::WorkerTiming`] per
    /// rayon worker after the parallel region.
    pub event_sender: Option<&'a Sender<TrainingEvent>>,
}

/// Per-stage solve context for one (stage, scenario) pair in the forward pass.
///
/// Passed to [`run_forward_stage`] to bundle scalar and slice parameters and
/// keep the argument count within the clippy `too_many_arguments` threshold.
pub(crate) struct StageKey<'a> {
    /// 0-based stage index.
    pub(crate) t: StageIdx,
    /// 0-based global scenario index (rank offset + local scenario index).
    pub(crate) m: usize,
    /// Local scenario index within this worker's partition.
    pub(crate) local_m: usize,
    /// Current training iteration (used in error context).
    pub(crate) iteration: u64,
    /// Raw noise sample for this (stage, scenario) pair.
    pub(crate) raw_noise: &'a [f64],
    /// Total LP row count, equal to `frozen[t].num_rows` (the frozen template
    /// absorbs all active cut rows as structural rows); sizes basis storage.
    pub(crate) basis_row_capacity: usize,
    /// Cut pool for stage `t`.
    pub(crate) pool: &'a CutPool,
    /// Dynamic Cut Selection hyperparameters, `Some` only when the dynamic method
    /// is configured AND active at this iteration. When `Some`, the stage is
    /// solved lazily against the cut pool from the cut-free base template; when
    /// `None`, the frozen all-cuts path is used.
    pub(crate) dcs: Option<DcsParams>,
    /// Canonical node-graph position this visit resolved to
    /// (`NodeGraph::nodes` index) — the pool/node-id resolution site, never
    /// `t` itself once a stage carries more than one alive node.
    pub(crate) node: NodePos,
}

/// Execute the forward pass for one training iteration on this rank.
///
/// Simulates this rank's share of forward-pass scenarios through the full stage
/// horizon, solving the stage LP at each `(scenario, stage)` pair and populating
/// the pre-allocated `records` in place.
/// `records[scenario * num_stages + stage]` holds the LP solution for that pair;
/// on error `records` may be partially populated.
///
/// `run_forward_pass` is a thin shim over `ForwardPassState::run`; production
/// callers use `TrainingSession::run_forward_phase`, which drives
/// `ForwardPassState::run` directly and bypasses this shim.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
/// feasible solution. Returns `Err(SddpError::Solver(_))` for all other
/// terminal LP solver failures.
///
/// # Panics (debug builds only)
///
/// Panics on a violated debug precondition (the assertions fire inside
/// `ForwardPassState::run`):
///
/// - `records.len() != batch.local_forward_passes * num_stages`
/// - `training_ctx.initial_state.len() != state.n_state`
#[cfg(any(test, feature = "test-support"))]
pub fn run_forward_pass<S>(
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    ctx: &StageContext<'_>,
    frozen: &[StageTemplate],
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    batch: &ForwardPassBatch<'_>,
    records: &mut [TrajectoryRecord],
) -> Result<ForwardResult, SddpError>
where
    S: SolverInterface<Profile = ActiveProfile> + Send,
{
    use crate::forward_pass_state::{ForwardPassInputs, ForwardPassState};
    let n_workers = workspaces.len().max(1);
    let num_stages = training_ctx.horizon.num_stages();
    let mut state = ForwardPassState::new(n_workers, num_stages, batch.local_forward_passes);
    let mut inputs = ForwardPassInputs {
        workspaces,
        basis_store,
        ctx,
        frozen,
        fcf,
        training_ctx,
        records,
        local_forward_passes: batch.local_forward_passes,
        total_forward_passes: batch.total_forward_passes,
        iteration: batch.iteration,
        fwd_offset: batch.fwd_offset,
        event_sender: batch.event_sender,
    };
    state.run(&mut inputs)
}
