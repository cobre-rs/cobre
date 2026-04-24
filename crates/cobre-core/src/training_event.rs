//! Typed event system for iterative optimization training loops and simulation runners.
//!
//! This module defines the [`TrainingEvent`] enum and its companion [`StoppingRuleResult`]
//! struct. Events are emitted at each step of the iterative optimization lifecycle
//! (forward pass, backward pass, convergence update, etc.) and consumed by runtime
//! observers: text loggers, JSON-lines writers, TUI renderers, MCP progress
//! notifications, and Parquet convergence writers.
//!
//! ## Design principles
//!
//! - **Zero-overhead when unused.** The event channel uses
//!   `Option<std::sync::mpsc::Sender<TrainingEvent>>`: when `None`, no events are
//!   emitted and no allocation occurs. When `Some(sender)`, events are moved into
//!   the channel at each lifecycle step boundary.
//! - **No per-event timestamps.** Consumers capture wall-clock time upon receipt.
//!   This avoids `clock_gettime` syscall overhead on the hot path. The single
//!   exception is [`TrainingEvent::TrainingStarted::timestamp`], which records the
//!   run-level start time once at entry.
//! - **Consumer-agnostic.** This module is defined in `cobre-core` (not in the
//!   algorithm crate) so that interface crates (`cobre-cli`, `cobre-tui`,
//!   `cobre-mcp`) can consume events without depending on the algorithm crate.
//!
//! ## Event channel pattern
//!
//! ```rust
//! use std::sync::mpsc;
//! use cobre_core::TrainingEvent;
//!
//! let (tx, rx) = mpsc::channel::<TrainingEvent>();
//! // Pass `Some(tx)` to the training loop; pass `rx` to the consumer thread.
//! drop(tx);
//! drop(rx);
//! ```
//!
//! See [`TrainingEvent`] for the full variant catalogue.

use std::borrow::Cow;

/// Phase discriminant for [`TrainingEvent::WorkerTiming`].
///
/// Distinguishes whether timing was captured during forward or backward pass.
#[derive(Clone, Debug)]
pub enum WorkerTimingPhase {
    /// Timing captured from the forward-pass parallel region.
    Forward,
    /// Timing captured from the backward-pass parallel region.
    Backward,
}

/// Number of timing slots in the `[f64; 16]` payload of
/// [`TrainingEvent::WorkerTiming`].
pub const WORKER_TIMING_SLOT_COUNT: usize = 16;

/// Slot index for `forward_wall_ms` (populated only on `Forward` phase events).
pub const WORKER_TIMING_SLOT_FWD_WALL: usize = 0;

/// Slot index for `backward_wall_ms` (populated only on `Backward` phase events).
pub const WORKER_TIMING_SLOT_BWD_WALL: usize = 1;

/// Slot index for `bwd_setup_ms` (populated only on `Backward` phase events).
pub const WORKER_TIMING_SLOT_BWD_SETUP: usize = 8;

/// Slot index for `fwd_setup_ms` (populated only on `Forward` phase events).
pub const WORKER_TIMING_SLOT_FWD_SETUP: usize = 11;

/// Result of evaluating a single stopping rule at a given iteration.
///
/// The [`TrainingEvent::ConvergenceUpdate`] variant carries a [`Vec`] of these,
/// one per configured stopping rule. Each element reports the rule's identifier,
/// whether its condition is satisfied, and a human-readable description of the
/// current state (e.g., `"gap 0.42% <= 1.00%"`).
#[derive(Clone, Debug)]
pub struct StoppingRuleResult {
    /// Rule identifier matching the variant name in the stopping rules config
    /// (e.g., `"gap_tolerance"`, `"bound_stalling"`, `"iteration_limit"`,
    /// `"time_limit"`, `"simulation"`).
    ///
    /// Always a compile-time `&'static str` constant. No heap allocation
    /// occurs on the hot path.
    pub rule_name: &'static str,
    /// Whether this rule's condition is satisfied at the current iteration.
    pub triggered: bool,
    /// Human-readable description of the rule's current state
    /// (e.g., `"gap 0.42% <= 1.00%"`, `"LB stable for 12/10 iterations"`).
    ///
    /// Use [`Cow::Borrowed`] for compile-time string literals and
    /// [`Cow::Owned`] for runtime `format!(...)` results.
    pub detail: Cow<'static, str>,
}

/// Per-stage row-selection statistics for one iteration.
///
/// Each instance describes the row-selection lifecycle at a single stage after a
/// selection step: how many rows existed, how many were active before
/// selection, how many were deactivated, and how many remain active.
///
/// The two optional fields capture the budget-enforcement pipeline state:
/// `budget_evicted` and `active_after_budget` are set after Step 4b (budget
/// enforcement). Both are `None` when budget enforcement is disabled.
#[derive(Debug, Clone)]
pub struct StageRowSelectionRecord {
    /// 0-based stage index.
    pub stage: u32,
    /// Total rows ever generated at this stage (high-water mark).
    pub rows_populated: u32,
    /// Active rows before selection ran.
    pub rows_active_before: u32,
    /// Rows deactivated by selection at this stage.
    pub rows_deactivated: u32,
    /// Active rows after selection.
    pub rows_active_after: u32,
    /// Wall-clock time for selection at this stage, in milliseconds.
    pub selection_time_ms: f64,
    /// Rows evicted by budget enforcement (Step 4b) at this stage.
    ///
    /// `None` when budget enforcement is disabled.
    pub budget_evicted: Option<u32>,
    /// Active cuts after budget enforcement (Step 4b).
    ///
    /// `None` when budget enforcement is disabled.
    pub active_after_budget: Option<u32>,
}

/// Typed events emitted by an iterative optimization training loop and
/// simulation runner.
///
/// The enum has 15 variants: 11 per-iteration events (one per lifecycle step)
/// and 4 lifecycle events (emitted once per training or simulation run).
///
/// ## Per-iteration events (steps 1–7 + 4a + 4b + 4c + per-worker)
///
/// | Step | Variant                  | When emitted                                           |
/// |------|--------------------------|--------------------------------------------------------|
/// | 1    | [`Self::ForwardPassComplete`]  | Local forward pass done                                |
/// | 2    | [`Self::ForwardSyncComplete`]  | Global allreduce of bounds done                        |
/// | 3    | [`Self::BackwardPassComplete`] | Backward sweep done                                    |
/// | 4    | [`Self::PolicySyncComplete`]      | Row-sync allgatherv done                                    |
/// | 4a   | [`Self::PolicySelectionComplete`] | Row-selection done (conditional on `should_run`)       |
/// | 4b   | [`Self::PolicyBudgetEnforcementComplete`] | Budget cap enforcement done (every iteration when budget is set) |
/// | 4c   | [`Self::PolicyTemplateBakeComplete`] | Per-stage baked template rebuild done (every iteration) |
/// | 5    | [`Self::ConvergenceUpdate`]    | Stopping rules evaluated                               |
/// | 6    | [`Self::CheckpointComplete`]   | Checkpoint written (conditional on checkpoint interval)|
/// | 7    | [`Self::IterationSummary`]     | End-of-iteration aggregated summary                    |
/// | pw   | [`Self::WorkerTiming`]         | Per-worker timing (2 × n\_workers per iteration)       |
///
/// ## Lifecycle events
///
/// | Variant                      | When emitted                        |
/// |------------------------------|-------------------------------------|
/// | [`Self::TrainingStarted`]    | Training loop entry                 |
/// | [`Self::TrainingFinished`]   | Training loop exit                  |
/// | [`Self::SimulationStarted`]  | Simulation loop entry               |
/// | [`Self::SimulationProgress`] | Simulation batch completion         |
/// | [`Self::SimulationFinished`] | Simulation completion               |
#[derive(Clone, Debug)]
pub enum TrainingEvent {
    // ── Per-iteration events (8) ─────────────────────────────────────────────
    /// Step 1: Forward pass completed for this iteration on the local rank.
    ForwardPassComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of forward scenarios evaluated on this rank.
        scenarios: u32,
        /// Mean total forward cost across local scenarios.
        ub_mean: f64,
        /// Standard deviation of total forward cost across local scenarios.
        ub_std: f64,
        /// Wall-clock time for the forward pass on this rank, in milliseconds.
        elapsed_ms: u64,
    },

    /// Step 2: Forward synchronization (allreduce) completed.
    ///
    /// Emitted after the global reduction of local bound estimates across all
    /// participating ranks.
    ForwardSyncComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Global upper bound mean after allreduce.
        global_ub_mean: f64,
        /// Global upper bound standard deviation after allreduce.
        global_ub_std: f64,
        /// Wall-clock time for the synchronization, in milliseconds.
        sync_time_ms: u64,
    },

    /// Step 3: Backward pass completed for this iteration.
    ///
    /// Emitted after the full backward sweep that generates new rows for each
    /// stage.
    BackwardPassComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of new rows generated across all stages.
        rows_generated: u32,
        /// Number of stages processed in the backward sweep.
        stages_processed: u32,
        /// Wall-clock time for the backward pass, in milliseconds.
        elapsed_ms: u64,
        /// Wall-clock time for state exchange (`allgatherv`) across all stages,
        /// in milliseconds.
        state_exchange_time_ms: u64,
        /// Wall-clock time for row-batch assembly (`build_row_batch_into`)
        /// across all stages, in milliseconds.
        row_batch_build_time_ms: u64,
        /// Aggregate non-solve work inside the parallel region accumulated across
        /// all stages and all workers, in milliseconds.
        ///
        /// Computed per stage as the sum over all workers of
        /// `load_model + add_rows + set_bounds + basis_set` times. Because it
        /// sums across workers, this value can exceed `parallel_wall_ms` when
        /// there are multiple workers. It is an aggregate cost metric, not a
        /// wall-time slice.
        setup_time_ms: u64,
        /// Estimated load imbalance across worker threads (wall minus ideal
        /// parallel work), in milliseconds.
        load_imbalance_ms: u64,
        /// Scheduling and synchronisation overhead not attributable to solve
        /// work or load imbalance, in milliseconds.
        scheduling_overhead_ms: u64,
    },

    /// Step 4: Policy row synchronization (allgatherv) completed.
    ///
    /// Emitted after new rows from all ranks have been gathered and distributed
    /// to every rank via allgatherv.
    PolicySyncComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of rows distributed to all ranks via allgatherv.
        rows_distributed: u32,
        /// Total number of active rows in the approximation after synchronization.
        rows_active: u32,
        /// Number of rows removed during synchronization.
        rows_removed: u32,
        /// Wall-clock time for the synchronization, in milliseconds.
        sync_time_ms: u64,
    },

    /// Step 4a: Policy row selection completed.
    ///
    /// Only emitted on iterations where row selection runs (i.e., when
    /// `should_run(iteration)` returns `true`). On non-selection iterations
    /// this variant is skipped entirely.
    PolicySelectionComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of rows deactivated across all stages.
        rows_deactivated: u32,
        /// Number of stages processed during row selection.
        stages_processed: u32,
        /// Wall-clock time for the local row-selection phase, in milliseconds.
        selection_time_ms: u64,
        /// Wall-clock time for the allgatherv deactivation-set exchange, in
        /// milliseconds.
        allgatherv_time_ms: u64,
        /// Per-stage breakdown of selection results.
        per_stage: Vec<StageRowSelectionRecord>,
    },

    /// Step 4b: Active-row budget enforcement completed.
    ///
    /// Emitted every iteration when `budget` is set in `TrainingConfig`.
    /// When `budget` is `None`, this variant is never emitted. Unlike Step
    /// 4a, budget enforcement is not gated by `check_frequency` because the
    /// budget is a hard cap that must be maintained at all times.
    PolicyBudgetEnforcementComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Total number of rows evicted across all stages in this iteration.
        rows_evicted: u32,
        /// Number of stages processed during budget enforcement.
        stages_processed: u32,
        /// Wall-clock time for the budget enforcement pass, in milliseconds.
        enforcement_time_ms: u64,
    },

    /// Step 4c: Template baking completed.
    ///
    /// Emitted every iteration after all per-stage baked templates have been
    /// rebuilt from the current active row set (after Steps 4a and 4b). Baking
    /// runs sequentially over stages and is outside the forward/backward hot
    /// paths. The baked templates are consumed by the forward and backward
    /// passes in the *next* iteration.
    PolicyTemplateBakeComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of stages for which baked templates were rebuilt.
        stages_processed: u32,
        /// Total number of rows baked across all stages
        /// (sum of `active_count()` over all stage pools at the emit instant).
        total_rows_baked: u64,
        /// Wall-clock time for the baking pass across all stages, in milliseconds.
        bake_time_ms: u64,
    },

    /// Step 5: Convergence check completed.
    ///
    /// Emitted after all configured stopping rules have been evaluated for the
    /// current iteration. Contains the current bounds, gap, and per-rule results.
    ConvergenceUpdate {
        /// Iteration number (1-based).
        iteration: u64,
        /// Current lower bound (non-decreasing across iterations).
        lower_bound: f64,
        /// Current upper bound (statistical estimate from forward costs).
        upper_bound: f64,
        /// Standard deviation of the upper bound estimate.
        upper_bound_std: f64,
        /// Relative optimality gap: `(upper_bound - lower_bound) / |upper_bound|`.
        gap: f64,
        /// Evaluation result for each configured stopping rule.
        rules_evaluated: Vec<StoppingRuleResult>,
    },

    /// Step 6: Checkpoint written.
    ///
    /// Only emitted when the checkpoint interval triggers (i.e., when
    /// `iteration % checkpoint_interval == 0`). Not emitted on every iteration.
    CheckpointComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Filesystem path where the checkpoint was written.
        checkpoint_path: String,
        /// Wall-clock time for the checkpoint write, in milliseconds.
        elapsed_ms: u64,
    },

    /// Step 7: Full iteration summary with aggregated timings.
    ///
    /// Emitted at the end of every iteration as the final per-iteration event.
    /// Contains all timing breakdowns for the completed iteration.
    IterationSummary {
        /// Iteration number (1-based).
        iteration: u64,
        /// Current lower bound.
        lower_bound: f64,
        /// Current upper bound.
        upper_bound: f64,
        /// Relative optimality gap: `(upper_bound - lower_bound) / |upper_bound|`.
        gap: f64,
        /// Cumulative wall-clock time since training started, in milliseconds.
        wall_time_ms: u64,
        /// Wall-clock time for this iteration only, in milliseconds.
        iteration_time_ms: u64,
        /// Forward pass wall-clock time for this iteration, in milliseconds.
        forward_ms: u64,
        /// Backward pass wall-clock time for this iteration, in milliseconds.
        backward_ms: u64,
        /// Total number of LP solves in this iteration (forward + backward stages).
        lp_solves: u64,
        /// Cumulative LP solve wall-clock time for this iteration, in milliseconds.
        solve_time_ms: f64,
        /// Wall-clock time for lower bound evaluation, in milliseconds.
        lower_bound_eval_ms: u64,
        /// Aggregate non-solve work inside the forward pass parallel region
        /// accumulated across all workers, in milliseconds.
        ///
        /// Computed as the sum over all workers of
        /// `load_model + add_rows + set_bounds + basis_set` times. Because it
        /// sums across workers, this value can exceed `forward_ms` when there
        /// are multiple workers. It is an aggregate cost metric, not a
        /// wall-time slice.
        fwd_setup_time_ms: u64,
        /// Estimated load imbalance across worker threads in the forward pass,
        /// in milliseconds.
        fwd_load_imbalance_ms: u64,
        /// Scheduling and synchronisation overhead in the forward pass not
        /// attributable to solve work or load imbalance, in milliseconds.
        fwd_scheduling_overhead_ms: u64,
    },

    // ── Lifecycle events (4) ─────────────────────────────────────────────────
    /// Emitted once when the training loop begins.
    ///
    /// Carries run-level metadata describing the problem size and parallelism
    /// configuration for this training run.
    TrainingStarted {
        /// Case study name from the input data directory.
        case_name: String,
        /// Total number of stages in the optimization horizon.
        stages: u32,
        /// Number of hydro plants in the system.
        hydros: u32,
        /// Number of thermal plants in the system.
        thermals: u32,
        /// Number of distributed ranks participating in training.
        ranks: u32,
        /// Number of threads per rank.
        threads_per_rank: u32,
        /// Wall-clock time at training start as an ISO 8601 string
        /// (run-level metadata, not a per-event timestamp).
        timestamp: String,
    },

    /// Emitted once when the training loop exits (converged or limit reached).
    TrainingFinished {
        /// Termination reason (e.g., `"gap_tolerance"`, `"iteration_limit"`,
        /// `"time_limit"`).
        reason: String,
        /// Total number of iterations completed.
        iterations: u64,
        /// Final lower bound at termination.
        final_lb: f64,
        /// Final upper bound at termination.
        final_ub: f64,
        /// Total wall-clock time for the training run, in milliseconds.
        total_time_ms: u64,
        /// Total number of rows in the approximation at termination.
        total_rows: u64,
    },

    /// Emitted once per rank when the simulation loop begins, before any
    /// scenario runs. Mirrors [`Self::TrainingStarted`] for the simulation
    /// phase. Progress consumers use this to display a "starting..." banner
    /// and capture run-level metadata (scenario count, parallelism layout).
    SimulationStarted {
        /// Case study name from the input data directory.
        case_name: String,
        /// Total number of simulation scenarios across all ranks.
        n_scenarios: u32,
        /// Total number of stages in the optimization horizon.
        n_stages: u32,
        /// Number of distributed ranks participating in simulation.
        ranks: u32,
        /// Number of threads per rank.
        threads_per_rank: u32,
        /// Wall-clock time at simulation start as an ISO 8601 string
        /// (run-level metadata, not a per-event timestamp).
        timestamp: String,
    },

    /// Emitted periodically during policy simulation (not during training).
    ///
    /// Consumers can use this to display a progress indicator during the
    /// simulation phase. Each event carries the cost of the most recently
    /// completed scenario; the progress thread accumulates statistics across
    /// events.
    SimulationProgress {
        /// Number of simulation scenarios completed so far, as a global
        /// estimate.
        ///
        /// Only rank 0 emits displayable progress events (non-root ranks are
        /// `--quiet`). Rank 0 knows only its own local completion count, so
        /// the global estimate is computed as `local_completed × ranks`,
        /// clamped to `scenarios_total`. The estimate is exact when work is
        /// evenly distributed and ranks finish at similar rates; it assumes
        /// the balanced-workload invariant from [`assign_scenarios`].
        scenarios_complete: u32,
        /// Total number of simulation scenarios to run across all ranks.
        scenarios_total: u32,
        /// Wall-clock time since simulation started, in milliseconds.
        elapsed_ms: u64,
        /// Total cost of the most recently completed simulation scenario,
        /// in cost units.
        scenario_cost: f64,
        /// Cumulative LP solve time for this scenario, in milliseconds.
        solve_time_ms: f64,
        /// Number of LP solves in this scenario.
        lp_solves: u64,
    },

    /// Emitted once when policy simulation completes.
    SimulationFinished {
        /// Total number of simulation scenarios evaluated.
        scenarios: u32,
        /// Directory where simulation output files were written.
        output_dir: String,
        /// Total wall-clock time for the simulation run, in milliseconds.
        elapsed_ms: u64,
    },

    /// Per-worker timing for one phase of one iteration.
    ///
    /// Emitted `n_workers_local` times per `(iteration, phase)` pair — once
    /// for every rayon worker in the local pool — after the parallel region
    /// completes. For a 10-worker / 50-iteration run this produces
    /// `2 × 10 × 50 = 1 000` extra events; the fixed-size `[f64; 16]`
    /// payload (128 bytes) is moved by value so no heap allocation occurs
    /// per event.
    ///
    /// ## Slot mapping
    ///
    /// Slot indices map to columns of `iteration_timing_schema` in declaration
    /// order (skipping `iteration`, which is the row key):
    ///
    /// | Slot | Column                       | Per-worker?                         |
    /// |------|------------------------------|-------------------------------------|
    /// | 0    | `forward_wall_ms`            | yes — Forward emit only; 0 on Backward |
    /// | 1    | `backward_wall_ms`           | yes — Backward emit only; 0 on Forward |
    /// | 2    | `cut_selection_ms`           | NO (rank-only); always 0            |
    /// | 3    | `mpi_allreduce_ms`           | NO (rank-only); always 0            |
    /// | 4    | `cut_sync_ms`                | NO (rank-only); always 0            |
    /// | 5    | `lower_bound_ms`             | NO (rank-only sequential); always 0 |
    /// | 6    | `state_exchange_ms`          | NO (rank-only); always 0            |
    /// | 7    | `cut_batch_build_ms`         | NO (rank-only sequential); always 0 |
    /// | 8    | `bwd_setup_ms`               | yes — Backward emit only; 0 on Forward |
    /// | 9    | `bwd_load_imbalance_ms`      | NO (synthetic rank-level); always 0 |
    /// | 10   | `bwd_scheduling_overhead_ms` | NO (synthetic rank-level); always 0 |
    /// | 11   | `fwd_setup_ms`               | yes — Forward emit only; 0 on Backward |
    /// | 12   | `fwd_load_imbalance_ms`      | NO (synthetic rank-level); always 0 |
    /// | 13   | `fwd_scheduling_overhead_ms` | NO (synthetic rank-level); always 0 |
    /// | 14   | `overhead_ms`                | NO (rank-only residual); always 0   |
    /// | 15   | reserved                     | always 0 (future use)               |
    ///
    /// ## Recovery invariant
    ///
    /// For every slot that carries a per-worker value (0, 1, 8, 11):
    /// `SUM(slot_value) GROUP BY (iteration, slot)` over the `WorkerTiming`
    /// events for a given phase equals the corresponding field on the
    /// rank-level `BackwardPassComplete` / `IterationSummary` event for the
    /// same iteration. Slots 2–7, 9–10, 12–15 are always zero on
    /// `WorkerTiming` events; the rank-level events carry the authoritative
    /// values there.
    WorkerTiming {
        /// MPI rank that owns this worker.
        rank: i32,
        /// Rayon worker index within this rank's pool (`0..n_workers_local`).
        worker_id: i32,
        /// Training iteration (1-based), matching the rank-level events.
        iteration: u64,
        /// Forward or Backward, distinguishing the two per-iteration emissions.
        phase: WorkerTimingPhase,
        /// Fixed-size timing payload; slot mapping documented in the variant
        /// doc comment above and in [`WORKER_TIMING_SLOT_FWD_WALL`],
        /// [`WORKER_TIMING_SLOT_BWD_WALL`], [`WORKER_TIMING_SLOT_BWD_SETUP`],
        /// [`WORKER_TIMING_SLOT_FWD_SETUP`].
        timings: [f64; WORKER_TIMING_SLOT_COUNT],
    },
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{
        StageRowSelectionRecord, StoppingRuleResult, TrainingEvent, WORKER_TIMING_SLOT_COUNT,
        WorkerTimingPhase,
    };

    // Helper: build one of each variant with representative values.
    fn make_all_variants() -> Vec<TrainingEvent> {
        vec![
            TrainingEvent::ForwardPassComplete {
                iteration: 1,
                scenarios: 10,
                ub_mean: 110.0,
                ub_std: 5.0,
                elapsed_ms: 42,
            },
            TrainingEvent::ForwardSyncComplete {
                iteration: 1,
                global_ub_mean: 110.0,
                global_ub_std: 5.0,
                sync_time_ms: 3,
            },
            TrainingEvent::BackwardPassComplete {
                iteration: 1,
                rows_generated: 48,
                stages_processed: 12,
                elapsed_ms: 87,
                state_exchange_time_ms: 0,
                row_batch_build_time_ms: 0,
                setup_time_ms: 0,
                load_imbalance_ms: 0,
                scheduling_overhead_ms: 0,
            },
            TrainingEvent::PolicySyncComplete {
                iteration: 1,
                rows_distributed: 48,
                rows_active: 200,
                rows_removed: 0,
                sync_time_ms: 2,
            },
            TrainingEvent::PolicySelectionComplete {
                iteration: 10,
                rows_deactivated: 15,
                stages_processed: 12,
                selection_time_ms: 20,
                allgatherv_time_ms: 1,
                per_stage: vec![],
            },
            TrainingEvent::PolicyBudgetEnforcementComplete {
                iteration: 10,
                rows_evicted: 2,
                stages_processed: 12,
                enforcement_time_ms: 1,
            },
            TrainingEvent::PolicyTemplateBakeComplete {
                iteration: 10,
                stages_processed: 12,
                total_rows_baked: 48,
                bake_time_ms: 2,
            },
            TrainingEvent::ConvergenceUpdate {
                iteration: 1,
                lower_bound: 100.0,
                upper_bound: 110.0,
                upper_bound_std: 5.0,
                gap: 0.0909,
                rules_evaluated: vec![StoppingRuleResult {
                    rule_name: "gap_tolerance",
                    triggered: false,
                    detail: Cow::Borrowed("gap 9.09% > 1.00%"),
                }],
            },
            TrainingEvent::CheckpointComplete {
                iteration: 5,
                checkpoint_path: "/tmp/checkpoint.bin".to_string(),
                elapsed_ms: 150,
            },
            TrainingEvent::IterationSummary {
                iteration: 1,
                lower_bound: 100.0,
                upper_bound: 110.0,
                gap: 0.0909,
                wall_time_ms: 1000,
                iteration_time_ms: 200,
                forward_ms: 80,
                backward_ms: 100,
                lp_solves: 240,
                solve_time_ms: 45.2,
                lower_bound_eval_ms: 10,
                fwd_setup_time_ms: 2,
                fwd_load_imbalance_ms: 2,
                fwd_scheduling_overhead_ms: 1,
            },
            TrainingEvent::TrainingStarted {
                case_name: "test_case".to_string(),
                stages: 60,
                hydros: 5,
                thermals: 10,
                ranks: 4,
                threads_per_rank: 8,
                timestamp: "2026-01-01T00:00:00Z".to_string(),
            },
            TrainingEvent::TrainingFinished {
                reason: "gap_tolerance".to_string(),
                iterations: 50,
                final_lb: 105.0,
                final_ub: 106.0,
                total_time_ms: 300_000,
                total_rows: 2400,
            },
            TrainingEvent::SimulationProgress {
                scenarios_complete: 50,
                scenarios_total: 200,
                elapsed_ms: 5_000,
                scenario_cost: 45_230.0,
                solve_time_ms: 0.0,
                lp_solves: 0,
            },
            TrainingEvent::SimulationFinished {
                scenarios: 200,
                output_dir: "/tmp/output".to_string(),
                elapsed_ms: 20_000,
            },
            TrainingEvent::WorkerTiming {
                rank: 0,
                worker_id: 2,
                iteration: 1,
                phase: WorkerTimingPhase::Backward,
                timings: [0.0; WORKER_TIMING_SLOT_COUNT],
            },
        ]
    }

    #[test]
    fn all_fifteen_variants_construct() {
        let variants = make_all_variants();
        assert_eq!(
            variants.len(),
            15,
            "expected exactly 15 TrainingEvent variants"
        );
    }

    #[test]
    fn all_variants_clone() {
        for variant in make_all_variants() {
            let cloned = variant.clone();
            // Verify the clone produces a non-empty debug string (proxy for equality).
            assert!(!format!("{cloned:?}").is_empty());
        }
    }

    #[test]
    fn all_variants_debug_non_empty() {
        for variant in make_all_variants() {
            let debug = format!("{variant:?}");
            assert!(!debug.is_empty(), "debug output must not be empty");
        }
    }

    #[test]
    fn forward_pass_complete_fields_accessible() {
        let event = TrainingEvent::ForwardPassComplete {
            iteration: 7,
            scenarios: 20,
            ub_mean: 210.0,
            ub_std: 3.5,
            elapsed_ms: 55,
        };
        let TrainingEvent::ForwardPassComplete {
            iteration,
            scenarios,
            ub_mean,
            ub_std,
            elapsed_ms,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 7);
        assert_eq!(scenarios, 20);
        assert!((ub_mean - 210.0).abs() < f64::EPSILON);
        assert!((ub_std - 3.5).abs() < f64::EPSILON);
        assert_eq!(elapsed_ms, 55);
    }

    #[test]
    fn convergence_update_rules_evaluated_field() {
        let rules = vec![
            StoppingRuleResult {
                rule_name: "gap_tolerance",
                triggered: true,
                detail: Cow::Borrowed("gap 0.42% <= 1.00%"),
            },
            StoppingRuleResult {
                rule_name: "iteration_limit",
                triggered: false,
                detail: Cow::Borrowed("iteration 10/100"),
            },
        ];
        let event = TrainingEvent::ConvergenceUpdate {
            iteration: 10,
            lower_bound: 99.0,
            upper_bound: 100.0,
            upper_bound_std: 0.5,
            gap: 0.0042,
            rules_evaluated: rules.clone(),
        };
        let TrainingEvent::ConvergenceUpdate {
            rules_evaluated, ..
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(rules_evaluated.len(), 2);
        assert_eq!(rules_evaluated[0].rule_name, "gap_tolerance");
        assert!(rules_evaluated[0].triggered);
        assert_eq!(rules_evaluated[1].rule_name, "iteration_limit");
        assert!(!rules_evaluated[1].triggered);
    }

    #[test]
    fn stopping_rule_result_fields_accessible() {
        let r = StoppingRuleResult {
            rule_name: "bound_stalling",
            triggered: false,
            detail: Cow::Borrowed("LB stable for 8/10 iterations"),
        };
        let cloned = r.clone();
        assert_eq!(cloned.rule_name, "bound_stalling");
        assert!(!cloned.triggered);
        assert_eq!(cloned.detail, "LB stable for 8/10 iterations");
    }

    #[test]
    fn stopping_rule_result_debug_non_empty() {
        let r = StoppingRuleResult {
            rule_name: "time_limit",
            triggered: true,
            detail: Cow::Borrowed("elapsed 3602s > 3600s limit"),
        };
        let debug = format!("{r:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("time_limit"));
    }

    #[test]
    fn policy_selection_complete_fields_accessible() {
        let event = TrainingEvent::PolicySelectionComplete {
            iteration: 10,
            rows_deactivated: 30,
            stages_processed: 12,
            selection_time_ms: 25,
            allgatherv_time_ms: 2,
            per_stage: vec![],
        };
        let TrainingEvent::PolicySelectionComplete {
            iteration,
            rows_deactivated,
            stages_processed,
            selection_time_ms,
            allgatherv_time_ms,
            per_stage,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 10);
        assert_eq!(rows_deactivated, 30);
        assert_eq!(stages_processed, 12);
        assert_eq!(selection_time_ms, 25);
        assert_eq!(allgatherv_time_ms, 2);
        assert!(per_stage.is_empty());
    }

    #[test]
    fn training_started_timestamp_field() {
        let event = TrainingEvent::TrainingStarted {
            case_name: "hydro_sys".to_string(),
            stages: 120,
            hydros: 10,
            thermals: 20,
            ranks: 8,
            threads_per_rank: 4,
            timestamp: "2026-03-01T08:00:00Z".to_string(),
        };
        let TrainingEvent::TrainingStarted { timestamp, .. } = event else {
            panic!("wrong variant")
        };
        assert_eq!(timestamp, "2026-03-01T08:00:00Z");
    }

    #[test]
    fn simulation_progress_scenario_cost_field_accessible() {
        let event = TrainingEvent::SimulationProgress {
            scenarios_complete: 100,
            scenarios_total: 500,
            elapsed_ms: 10_000,
            scenario_cost: 45_230.0,
            solve_time_ms: 0.0,
            lp_solves: 0,
        };
        let TrainingEvent::SimulationProgress {
            scenarios_complete,
            scenarios_total,
            elapsed_ms,
            scenario_cost,
            ..
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(scenarios_complete, 100);
        assert_eq!(scenarios_total, 500);
        assert_eq!(elapsed_ms, 10_000);
        assert!((scenario_cost - 45_230.0).abs() < f64::EPSILON);
    }

    #[test]
    fn simulation_progress_first_scenario_cost_carried() {
        // The first scenario's cost is emitted directly — no aggregation needed.
        let event = TrainingEvent::SimulationProgress {
            scenarios_complete: 1,
            scenarios_total: 200,
            elapsed_ms: 100,
            scenario_cost: 50_000.0,
            solve_time_ms: 0.0,
            lp_solves: 0,
        };
        let TrainingEvent::SimulationProgress { scenario_cost, .. } = event else {
            panic!("wrong variant")
        };
        assert!((scenario_cost - 50_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn policy_budget_enforcement_complete_fields_accessible() {
        let event = TrainingEvent::PolicyBudgetEnforcementComplete {
            iteration: 7,
            rows_evicted: 5,
            stages_processed: 12,
            enforcement_time_ms: 3,
        };
        let TrainingEvent::PolicyBudgetEnforcementComplete {
            iteration,
            rows_evicted,
            stages_processed,
            enforcement_time_ms,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 7);
        assert_eq!(rows_evicted, 5);
        assert_eq!(stages_processed, 12);
        assert_eq!(enforcement_time_ms, 3);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn worker_timing_fields_accessible() {
        let mut timings = [0.0_f64; WORKER_TIMING_SLOT_COUNT];
        for (i, slot) in timings.iter_mut().enumerate() {
            *slot = (i as f64) * 1.5 + 0.25;
        }
        let event = TrainingEvent::WorkerTiming {
            rank: 2,
            worker_id: 3,
            iteration: 7,
            phase: WorkerTimingPhase::Forward,
            timings,
        };
        let TrainingEvent::WorkerTiming {
            rank,
            worker_id,
            iteration,
            phase,
            timings: t,
        } = event
        else {
            panic!("wrong variant");
        };
        assert_eq!(rank, 2);
        assert_eq!(worker_id, 3);
        assert_eq!(iteration, 7);
        assert!(
            !format!("{phase:?}").is_empty(),
            "WorkerTimingPhase::Forward debug must be non-empty"
        );
        for (i, &v) in t.iter().enumerate() {
            let expected = (i as f64) * 1.5 + 0.25;
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "slot {i}: expected {expected}, got {v}"
            );
        }
        // Verify Backward variant debug is also non-empty.
        let bwd = WorkerTimingPhase::Backward;
        assert!(
            !format!("{bwd:?}").is_empty(),
            "WorkerTimingPhase::Backward debug must be non-empty"
        );
    }

    #[test]
    fn policy_template_bake_complete_fields_accessible() {
        let event = TrainingEvent::PolicyTemplateBakeComplete {
            iteration: 5,
            stages_processed: 12,
            total_rows_baked: 96,
            bake_time_ms: 3,
        };
        let TrainingEvent::PolicyTemplateBakeComplete {
            iteration,
            stages_processed,
            total_rows_baked,
            bake_time_ms,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 5);
        assert_eq!(stages_processed, 12);
        assert_eq!(total_rows_baked, 96);
        assert_eq!(bake_time_ms, 3);
    }

    #[test]
    fn stage_row_selection_record_fields_accessible() {
        let record = StageRowSelectionRecord {
            stage: 3,
            rows_populated: 100,
            rows_active_before: 80,
            rows_deactivated: 10,
            rows_active_after: 70,
            selection_time_ms: 1.5,
            budget_evicted: Some(5),
            active_after_budget: Some(65),
        };
        assert_eq!(record.stage, 3);
        assert_eq!(record.rows_populated, 100);
        assert_eq!(record.rows_active_before, 80);
        assert_eq!(record.rows_deactivated, 10);
        assert_eq!(record.rows_active_after, 70);
        assert!((record.selection_time_ms - 1.5).abs() < f64::EPSILON);
        assert_eq!(record.budget_evicted, Some(5));
        assert_eq!(record.active_after_budget, Some(65));
    }
}
