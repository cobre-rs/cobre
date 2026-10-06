//! Typed event system for iterative optimization training loops and simulation runners.
//!
//! This module defines the [`TrainingEvent`] enum. Events are emitted at each step
//! of the iterative optimization lifecycle
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

/// Phase discriminant for [`TrainingEvent::WorkerTiming`].
#[derive(Clone, Debug)]
pub enum WorkerTimingPhase {
    /// Forward-pass parallel region.
    Forward,
    /// Backward-pass parallel region.
    Backward,
}

/// Slot count of the `cobre_io::WorkerTimingRecord` `[u64; 16]` writer record.
///
/// The `WORKER_TIMING_SLOT_*` constants are the canonical bridge between the
/// named [`WorkerPhaseTimings`] fields and the writer-record slot positions.
pub const WORKER_TIMING_SLOT_COUNT: usize = 16;

/// Writer-record slot index for `forward_wall_ms`.
pub const WORKER_TIMING_SLOT_FWD_WALL: usize = 0;

/// Writer-record slot index for `backward_wall_ms`.
pub const WORKER_TIMING_SLOT_BWD_WALL: usize = 1;

/// Writer-record slot index for `bwd_setup_ms`.
pub const WORKER_TIMING_SLOT_BWD_SETUP: usize = 8;

/// Writer-record slot index for `fwd_setup_ms`.
pub const WORKER_TIMING_SLOT_FWD_SETUP: usize = 11;

/// Writer-record slot index for `scoring_ms`.
pub const WORKER_TIMING_SLOT_SCORING: usize = 15;

/// Per-worker timing payload for [`TrainingEvent::WorkerTiming`].
///
/// Each field is populated only on the phase it belongs to and is 0 on the
/// other; `scoring_ms` is populated on whichever phase performed lazy candidate
/// scoring and 0 when no scoring ran.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorkerPhaseTimings {
    /// Forward-pass wall time, ms.
    pub forward_wall_ms: f64,
    /// Backward-pass wall time, ms.
    pub backward_wall_ms: f64,
    /// Forward setup time, ms.
    pub fwd_setup_ms: f64,
    /// Backward setup time, ms.
    pub bwd_setup_ms: f64,
    /// Lazy candidate-scoring time, ms.
    pub scoring_ms: f64,
}

/// Per-stage row-selection statistics for one iteration.
///
/// The two `Option` fields are `None` when budget enforcement is disabled.
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
    /// Rows reactivated at this stage during this iteration.
    pub rows_reactivated: u32,
    /// Active rows after selection.
    pub rows_active_after: u32,
    /// Wall-clock time for selection at this stage, in milliseconds.
    pub selection_time_ms: f64,
    /// Rows evicted by budget enforcement at this stage.
    pub budget_evicted: Option<u32>,
    /// Active cuts after budget enforcement.
    pub active_after_budget: Option<u32>,
    /// Total rows in the LP at this stage at the end of the iteration.
    pub rows_in_lp: u32,
}

/// Typed events emitted by an iterative optimization training loop and
/// simulation runner.
///
/// The enum has 16 variants: 11 per-iteration events (one per lifecycle step)
/// and 5 lifecycle events (emitted once per training or simulation run). Every
/// per-iteration variant's `iteration` field is 1-based.
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
/// | 4c   | [`Self::PolicyTemplateFreezeComplete`] | Per-stage frozen template rebuild done (every iteration) |
/// | 5    | [`Self::ConvergenceUpdate`]    | Stopping rules evaluated                               |
/// | 6    | [`Self::IterationSummary`]     | End-of-iteration aggregated summary                    |
/// | 7    | [`Self::CheckpointComplete`]   | Periodic checkpoint committed (scheduled iterations)   |
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
    // ── Per-iteration events ─────────────────────────────────────────────────
    /// Forward pass completed for this iteration on the local rank.
    ForwardPassComplete {
        /// Iteration number.
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

    /// Forward synchronization (allreduce) completed: the global reduction of
    /// local bound estimates across all participating ranks.
    ForwardSyncComplete {
        /// Iteration number.
        iteration: u64,
        /// Global upper bound mean after allreduce.
        global_ub_mean: f64,
        /// Global upper bound standard deviation after allreduce.
        global_ub_std: f64,
        /// Wall-clock time for the synchronization, in milliseconds.
        sync_time_ms: u64,
    },

    /// Backward pass completed: the full backward sweep that generates new rows
    /// for each stage.
    BackwardPassComplete {
        /// Iteration number.
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
        /// Aggregate non-solve work inside the parallel region, summed across all
        /// stages and workers, in milliseconds. Summing across workers means it
        /// can exceed wall time — an aggregate cost metric, not a wall-time slice.
        setup_time_ms: u64,
        /// Estimated load imbalance across worker threads (wall minus ideal
        /// parallel work), in milliseconds.
        load_imbalance_ms: u64,
        /// Scheduling and synchronisation overhead not attributable to solve
        /// work or load imbalance, in milliseconds.
        scheduling_overhead_ms: u64,
    },

    /// Policy row synchronization (allgatherv) completed: new rows from all ranks
    /// gathered and distributed to every rank.
    PolicySyncComplete {
        /// Iteration number.
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

    /// Policy row selection completed.
    ///
    /// Only emitted when `should_run(iteration)` is `true`; skipped entirely on
    /// non-selection iterations.
    PolicySelectionComplete {
        /// Iteration number.
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

    /// Active-row budget enforcement completed.
    ///
    /// Emitted every iteration when `budget` is set in `TrainingConfig`, never
    /// when it is `None`. Not gated by `check_frequency` — the budget is a hard
    /// cap maintained at all times.
    PolicyBudgetEnforcementComplete {
        /// Iteration number.
        iteration: u64,
        /// Total number of rows evicted across all stages in this iteration.
        rows_evicted: u32,
        /// Number of stages processed during budget enforcement.
        stages_processed: u32,
        /// Wall-clock time for the budget enforcement pass, in milliseconds.
        enforcement_time_ms: u64,
    },

    /// Template freeze completed: per-stage frozen templates rebuilt from the
    /// current active row set, consumed by the *next* iteration's passes.
    PolicyTemplateFreezeComplete {
        /// Iteration number.
        iteration: u64,
        /// Number of stages for which frozen templates were rebuilt.
        stages_processed: u32,
        /// Total number of rows frozen, summed across all stages.
        total_rows_frozen: u64,
        /// Wall-clock time for the freeze pass across all stages, in milliseconds.
        freeze_time_ms: u64,
    },

    /// Convergence check completed: all configured stopping rules evaluated for
    /// the current iteration.
    ConvergenceUpdate {
        /// Iteration number.
        iteration: u64,
        /// Current lower bound (non-decreasing across iterations).
        lower_bound: f64,
        /// Current upper bound (statistical estimate from forward costs).
        upper_bound: f64,
        /// Standard deviation of the upper bound estimate.
        upper_bound_std: f64,
        /// Relative optimality gap: `(upper_bound - lower_bound) / |upper_bound|`.
        gap: f64,
    },

    /// Emitted by the writing rank after each periodic checkpoint is committed.
    CheckpointComplete {
        /// Iteration number.
        iteration: u64,
        /// Filesystem path where the checkpoint was written.
        checkpoint_path: String,
        /// Wall-clock time for the checkpoint write, in milliseconds.
        elapsed_ms: u64,
    },

    /// Full iteration summary with aggregated timings; within an iteration only
    /// [`Self::CheckpointComplete`] follows it.
    IterationSummary {
        /// Iteration number.
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
        /// Aggregate non-solve work inside the forward-pass parallel region,
        /// summed across all workers, in milliseconds. Summing across workers
        /// means it can exceed `forward_ms` — a cost metric, not a wall-time slice.
        fwd_setup_time_ms: u64,
        /// Estimated load imbalance across worker threads in the forward pass,
        /// in milliseconds.
        fwd_load_imbalance_ms: u64,
        /// Scheduling and synchronisation overhead in the forward pass not
        /// attributable to solve work or load imbalance, in milliseconds.
        fwd_scheduling_overhead_ms: u64,
        /// Sum, over every lazy-selection LP solve this iteration (all ranks), of
        /// the resident row count loaded into that solve. `sum / count` is the
        /// per-iteration mean. Zero when no lazy selection ran.
        rows_in_lp_sum: u64,
        /// Number of lazy-selection LP solves this iteration (all ranks); the
        /// mean denominator. Zero when no lazy selection ran.
        rows_in_lp_count: u64,
        /// Largest resident row count over any lazy-selection LP solve up to and
        /// including this iteration (all ranks). A *running* (cumulative) peak, so
        /// a `max`-fold over iterations recovers the run-level peak with no
        /// finalize reduce. Zero when no lazy selection ran.
        rows_in_lp_max: u64,
    },

    // ── Lifecycle events ─────────────────────────────────────────────────────
    /// Emitted once when the training loop begins.
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
    /// scenario runs.
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

    /// Emitted periodically during policy simulation (not during training);
    /// each event carries the most recently completed scenario's cost.
    SimulationProgress {
        /// Scenarios completed so far, as a global estimate.
        ///
        /// Only rank 0 emits; it knows only its local count, so this is
        /// `local_completed × ranks` clamped to `scenarios_total` — exact only
        /// under the scheduler's balanced-workload invariant, an estimate otherwise.
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
    /// Emitted `n_workers_local` times per `(iteration, phase)` pair after the
    /// parallel region completes; the [`WorkerPhaseTimings`] payload is moved by
    /// value, so no heap allocation occurs per event.
    ///
    /// ## Recovery invariant
    ///
    /// `SUM(field) GROUP BY (iteration)` over the per-worker `WorkerTiming`
    /// events equals the corresponding field on the rank-level
    /// `BackwardPassComplete` / `IterationSummary` event for the same iteration.
    WorkerTiming {
        /// MPI rank that owns this worker.
        rank: i32,
        /// Rayon worker index within this rank's pool (`0..n_workers_local`).
        worker_id: i32,
        /// Training iteration (1-based), matching the rank-level events.
        iteration: u64,
        /// Which of the two per-iteration emissions this is.
        phase: WorkerTimingPhase,
        /// Per-worker timing payload.
        timings: WorkerPhaseTimings,
    },
}

#[cfg(test)]
mod tests {
    use super::{StageRowSelectionRecord, TrainingEvent, WorkerPhaseTimings, WorkerTimingPhase};

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
            TrainingEvent::PolicyTemplateFreezeComplete {
                iteration: 10,
                stages_processed: 12,
                total_rows_frozen: 48,
                freeze_time_ms: 2,
            },
            TrainingEvent::ConvergenceUpdate {
                iteration: 1,
                lower_bound: 100.0,
                upper_bound: 110.0,
                upper_bound_std: 5.0,
                gap: 0.0909,
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
                rows_in_lp_sum: 720,
                rows_in_lp_count: 240,
                rows_in_lp_max: 24,
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
            TrainingEvent::SimulationStarted {
                case_name: "test_case".to_string(),
                n_scenarios: 200,
                n_stages: 60,
                ranks: 4,
                threads_per_rank: 8,
                timestamp: "2026-01-01T00:00:00Z".to_string(),
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
                timings: WorkerPhaseTimings::default(),
            },
        ]
    }

    #[test]
    fn all_variants_construct() {
        let variants = make_all_variants();
        assert_eq!(
            variants.len(),
            16,
            "expected exactly 16 TrainingEvent variants"
        );
    }

    #[test]
    fn all_variants_clone() {
        for variant in make_all_variants() {
            let cloned = variant.clone();
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
    fn worker_timing_fields_accessible() {
        let timings = WorkerPhaseTimings {
            forward_wall_ms: 10.0,
            backward_wall_ms: 0.0,
            fwd_setup_ms: 2.5,
            bwd_setup_ms: 0.0,
            scoring_ms: 1.25,
        };
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
        assert!((t.forward_wall_ms - 10.0).abs() < f64::EPSILON);
        assert!((t.fwd_setup_ms - 2.5).abs() < f64::EPSILON);
        assert!((t.scoring_ms - 1.25).abs() < f64::EPSILON);
        assert_eq!(t.backward_wall_ms, 0.0);
        assert_eq!(t.bwd_setup_ms, 0.0);
        let bwd = WorkerTimingPhase::Backward;
        assert!(
            !format!("{bwd:?}").is_empty(),
            "WorkerTimingPhase::Backward debug must be non-empty"
        );
    }

    #[test]
    fn policy_template_freeze_complete_fields_accessible() {
        let event = TrainingEvent::PolicyTemplateFreezeComplete {
            iteration: 5,
            stages_processed: 12,
            total_rows_frozen: 96,
            freeze_time_ms: 3,
        };
        let TrainingEvent::PolicyTemplateFreezeComplete {
            iteration,
            stages_processed,
            total_rows_frozen,
            freeze_time_ms,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 5);
        assert_eq!(stages_processed, 12);
        assert_eq!(total_rows_frozen, 96);
        assert_eq!(freeze_time_ms, 3);
    }

    #[test]
    fn stage_row_selection_record_fields_accessible() {
        let record = StageRowSelectionRecord {
            stage: 3,
            rows_populated: 100,
            rows_active_before: 80,
            rows_deactivated: 10,
            rows_reactivated: 2,
            rows_active_after: 70,
            selection_time_ms: 1.5,
            budget_evicted: Some(5),
            active_after_budget: Some(65),
            rows_in_lp: 100,
        };
        assert_eq!(record.stage, 3);
        assert_eq!(record.rows_populated, 100);
        assert_eq!(record.rows_active_before, 80);
        assert_eq!(record.rows_deactivated, 10);
        assert_eq!(record.rows_reactivated, 2);
        assert_eq!(record.rows_active_after, 70);
        assert!((record.selection_time_ms - 1.5).abs() < f64::EPSILON);
        assert_eq!(record.budget_evicted, Some(5));
        assert_eq!(record.active_after_budget, Some(65));
        assert_eq!(record.rows_in_lp, 100);
    }
}
