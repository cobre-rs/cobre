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
    pub rule_name: String,
    /// Whether this rule's condition is satisfied at the current iteration.
    pub triggered: bool,
    /// Human-readable description of the rule's current state
    /// (e.g., `"gap 0.42% <= 1.00%"`, `"LB stable for 12/10 iterations"`).
    pub detail: String,
}

/// Per-stage cut selection statistics for one iteration.
///
/// Each instance describes the cut lifecycle at a single stage after a
/// selection step: how many cuts existed, how many were active before
/// selection, how many were deactivated, and how many remain active.
///
/// The three optional fields capture the multi-step pipeline state:
/// `active_after_angular` is set after Step 4b (angular dominance pruning),
/// `budget_evicted` and `active_after_budget` are set after Step 4c (budget
/// enforcement). All three are `None` when the corresponding step is disabled.
#[derive(Debug, Clone)]
pub struct StageSelectionRecord {
    /// 0-based stage index.
    pub stage: u32,
    /// Total cuts ever generated at this stage (high-water mark).
    pub cuts_populated: u32,
    /// Active cuts before selection ran.
    pub cuts_active_before: u32,
    /// Cuts deactivated by selection at this stage.
    pub cuts_deactivated: u32,
    /// Active cuts after selection.
    pub cuts_active_after: u32,
    /// Wall-clock time for selection at this stage, in milliseconds.
    pub selection_time_ms: f64,
    /// Active cuts after angular dominance pruning (Step 4b).
    ///
    /// `None` when angular pruning is disabled or has not run yet.
    pub active_after_angular: Option<u32>,
    /// Cuts evicted by budget enforcement (Step 4c) at this stage.
    ///
    /// `None` when budget enforcement is disabled.
    pub budget_evicted: Option<u32>,
    /// Active cuts after budget enforcement (Step 4c).
    ///
    /// `None` when budget enforcement is disabled.
    pub active_after_budget: Option<u32>,
}

/// Typed events emitted by an iterative optimization training loop and
/// simulation runner.
///
/// The enum has 14 variants: 10 per-iteration events (one per lifecycle step)
/// and 4 lifecycle events (emitted once per training or simulation run).
///
/// ## Per-iteration events (steps 1–7 + 4a + 4b + 4c)
///
/// | Step | Variant                  | When emitted                                           |
/// |------|--------------------------|--------------------------------------------------------|
/// | 1    | [`Self::ForwardPassComplete`]  | Local forward pass done                                |
/// | 2    | [`Self::ForwardSyncComplete`]  | Global allreduce of bounds done                        |
/// | 3    | [`Self::BackwardPassComplete`] | Backward sweep done                                    |
/// | 4    | [`Self::CutSyncComplete`]      | Cut allgatherv done                                    |
/// | 4a   | [`Self::CutSelectionComplete`] | Cut selection done (conditional on `should_run`)       |
/// | 4b   | [`Self::AngularPruningComplete`] | Angular dominance pruning done (conditional on `should_run`) |
/// | 4c   | [`Self::BudgetEnforcementComplete`] | Budget cap enforcement done (every iteration when budget is set) |
/// | 5    | [`Self::ConvergenceUpdate`]    | Stopping rules evaluated                               |
/// | 6    | [`Self::CheckpointComplete`]   | Checkpoint written (conditional on checkpoint interval)|
/// | 7    | [`Self::IterationSummary`]     | End-of-iteration aggregated summary                    |
///
/// ## Lifecycle events
///
/// | Variant                      | When emitted                        |
/// |------------------------------|-------------------------------------|
/// | [`Self::TrainingStarted`]    | Training loop entry                 |
/// | [`Self::TrainingFinished`]   | Training loop exit                  |
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
    /// Emitted after the full backward sweep that generates new cuts for each
    /// stage.
    BackwardPassComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of new cuts generated across all stages.
        cuts_generated: u32,
        /// Number of stages processed in the backward sweep.
        stages_processed: u32,
        /// Wall-clock time for the backward pass, in milliseconds.
        elapsed_ms: u64,
        /// Wall-clock time for state exchange (`allgatherv`) across all stages,
        /// in milliseconds.
        state_exchange_time_ms: u64,
        /// Wall-clock time for cut batch assembly (`build_cut_row_batch_into`)
        /// across all stages, in milliseconds.
        cut_batch_build_time_ms: u64,
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

    /// Step 4: Cut synchronization (allgatherv) completed.
    ///
    /// Emitted after new cuts from all ranks have been gathered and distributed
    /// to every rank via allgatherv.
    CutSyncComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of cuts distributed to all ranks via allgatherv.
        cuts_distributed: u32,
        /// Total number of active cuts in the approximation after synchronization.
        cuts_active: u32,
        /// Number of cuts removed during synchronization.
        cuts_removed: u32,
        /// Wall-clock time for the synchronization, in milliseconds.
        sync_time_ms: u64,
    },

    /// Step 4a: Cut selection completed.
    ///
    /// Only emitted on iterations where cut selection runs (i.e., when
    /// `should_run(iteration)` returns `true`). On non-selection iterations
    /// this variant is skipped entirely.
    CutSelectionComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Number of cuts deactivated across all stages.
        cuts_deactivated: u32,
        /// Number of stages processed during cut selection.
        stages_processed: u32,
        /// Wall-clock time for the local cut selection phase, in milliseconds.
        selection_time_ms: u64,
        /// Wall-clock time for the allgatherv deactivation-set exchange, in
        /// milliseconds.
        allgatherv_time_ms: u64,
        /// Per-stage breakdown of selection results.
        per_stage: Vec<StageSelectionRecord>,
    },

    /// Step 4b: Angular diversity pruning completed.
    ///
    /// Only emitted on iterations where angular pruning runs (i.e., when
    /// `should_run(iteration)` returns `true`). On non-pruning iterations
    /// this variant is skipped entirely. Always emitted after
    /// [`Self::CutSelectionComplete`] when both are enabled on the same
    /// iteration.
    AngularPruningComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Total number of cuts deactivated across all stages.
        cuts_deactivated: u32,
        /// Total number of angular clusters formed across all stages.
        clusters_formed: u64,
        /// Total number of within-cluster dominance checks performed across all
        /// stages.
        dominance_checks: u64,
        /// Number of stages processed (stages 1..num_stages-1; stage 0 is
        /// exempt).
        stages_processed: u32,
        /// Wall-clock time for the angular pruning phase, in milliseconds.
        pruning_time_ms: u64,
    },

    /// Step 4c: Active-cut budget enforcement completed.
    ///
    /// Emitted every iteration when `budget` is set in `TrainingConfig`.
    /// When `budget` is `None`, this variant is never emitted. Unlike Steps
    /// 4a and 4b, budget enforcement is not gated by `check_frequency`
    /// because the budget is a hard cap that must be maintained at all times.
    BudgetEnforcementComplete {
        /// Iteration number (1-based).
        iteration: u64,
        /// Total number of cuts evicted across all stages in this iteration.
        cuts_evicted: u32,
        /// Number of stages processed during budget enforcement.
        stages_processed: u32,
        /// Wall-clock time for the budget enforcement pass, in milliseconds.
        enforcement_time_ms: u64,
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
        /// Total number of cuts in the approximation at termination.
        total_cuts: u64,
    },

    /// Emitted periodically during policy simulation (not during training).
    ///
    /// Consumers can use this to display a progress indicator during the
    /// simulation phase. Each event carries the cost of the most recently
    /// completed scenario; the progress thread accumulates statistics across
    /// events (see ticket-007).
    SimulationProgress {
        /// Number of simulation scenarios completed so far.
        scenarios_complete: u32,
        /// Total number of simulation scenarios to run.
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
}

#[cfg(test)]
mod tests {
    use super::{StoppingRuleResult, TrainingEvent};

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
                cuts_generated: 48,
                stages_processed: 12,
                elapsed_ms: 87,
                state_exchange_time_ms: 0,
                cut_batch_build_time_ms: 0,
                setup_time_ms: 0,
                load_imbalance_ms: 0,
                scheduling_overhead_ms: 0,
            },
            TrainingEvent::CutSyncComplete {
                iteration: 1,
                cuts_distributed: 48,
                cuts_active: 200,
                cuts_removed: 0,
                sync_time_ms: 2,
            },
            TrainingEvent::CutSelectionComplete {
                iteration: 10,
                cuts_deactivated: 15,
                stages_processed: 12,
                selection_time_ms: 20,
                allgatherv_time_ms: 1,
                per_stage: vec![],
            },
            TrainingEvent::AngularPruningComplete {
                iteration: 10,
                cuts_deactivated: 3,
                clusters_formed: 5,
                dominance_checks: 12,
                stages_processed: 11,
                pruning_time_ms: 8,
            },
            TrainingEvent::BudgetEnforcementComplete {
                iteration: 10,
                cuts_evicted: 2,
                stages_processed: 12,
                enforcement_time_ms: 1,
            },
            TrainingEvent::ConvergenceUpdate {
                iteration: 1,
                lower_bound: 100.0,
                upper_bound: 110.0,
                upper_bound_std: 5.0,
                gap: 0.0909,
                rules_evaluated: vec![StoppingRuleResult {
                    rule_name: "gap_tolerance".to_string(),
                    triggered: false,
                    detail: "gap 9.09% > 1.00%".to_string(),
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
                total_cuts: 2400,
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
        ]
    }

    #[test]
    fn all_fourteen_variants_construct() {
        let variants = make_all_variants();
        assert_eq!(
            variants.len(),
            14,
            "expected exactly 14 TrainingEvent variants"
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
                rule_name: "gap_tolerance".to_string(),
                triggered: true,
                detail: "gap 0.42% <= 1.00%".to_string(),
            },
            StoppingRuleResult {
                rule_name: "iteration_limit".to_string(),
                triggered: false,
                detail: "iteration 10/100".to_string(),
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
            rule_name: "bound_stalling".to_string(),
            triggered: false,
            detail: "LB stable for 8/10 iterations".to_string(),
        };
        let cloned = r.clone();
        assert_eq!(cloned.rule_name, "bound_stalling");
        assert!(!cloned.triggered);
        assert_eq!(cloned.detail, "LB stable for 8/10 iterations");
    }

    #[test]
    fn stopping_rule_result_debug_non_empty() {
        let r = StoppingRuleResult {
            rule_name: "time_limit".to_string(),
            triggered: true,
            detail: "elapsed 3602s > 3600s limit".to_string(),
        };
        let debug = format!("{r:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("time_limit"));
    }

    #[test]
    fn cut_selection_complete_fields_accessible() {
        let event = TrainingEvent::CutSelectionComplete {
            iteration: 10,
            cuts_deactivated: 30,
            stages_processed: 12,
            selection_time_ms: 25,
            allgatherv_time_ms: 2,
            per_stage: vec![],
        };
        let TrainingEvent::CutSelectionComplete {
            iteration,
            cuts_deactivated,
            stages_processed,
            selection_time_ms,
            allgatherv_time_ms,
            per_stage,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 10);
        assert_eq!(cuts_deactivated, 30);
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
    fn angular_pruning_complete_fields_accessible() {
        let event = TrainingEvent::AngularPruningComplete {
            iteration: 15,
            cuts_deactivated: 7,
            clusters_formed: 4,
            dominance_checks: 20,
            stages_processed: 11,
            pruning_time_ms: 12,
        };
        let TrainingEvent::AngularPruningComplete {
            iteration,
            cuts_deactivated,
            clusters_formed,
            dominance_checks,
            stages_processed,
            pruning_time_ms,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 15);
        assert_eq!(cuts_deactivated, 7);
        assert_eq!(clusters_formed, 4);
        assert_eq!(dominance_checks, 20);
        assert_eq!(stages_processed, 11);
        assert_eq!(pruning_time_ms, 12);
    }

    #[test]
    fn budget_enforcement_complete_fields_accessible() {
        let event = TrainingEvent::BudgetEnforcementComplete {
            iteration: 7,
            cuts_evicted: 5,
            stages_processed: 12,
            enforcement_time_ms: 3,
        };
        let TrainingEvent::BudgetEnforcementComplete {
            iteration,
            cuts_evicted,
            stages_processed,
            enforcement_time_ms,
        } = event
        else {
            panic!("wrong variant")
        };
        assert_eq!(iteration, 7);
        assert_eq!(cuts_evicted, 5);
        assert_eq!(stages_processed, 12);
        assert_eq!(enforcement_time_ms, 3);
    }
}
