//! Output writers for simulation results and policy files.
//!
//! This module provides Hive-partitioned Parquet writers for simulation pipeline
//! output and `FlatBuffers` policy writers.
//!
//! The top-level entry point is [`write_results`]: it writes the training
//! result tables, the training dictionaries, and the training/simulation
//! completion metadata. It does not write the simulation scenario Parquet
//! data, the policy checkpoint, provenance, hydro-model exports, stochastic
//! echoes, solver-stats sidecars, or the `_SUCCESS` phase markers — each
//! caller writes those directly through the individual writer modules in this
//! crate, and the CLI and the Python bindings must stay in parity on the full
//! artifact set.

use std::path::{Path, PathBuf};

use chrono::{Datelike, NaiveDate};

pub(crate) mod atomic;
pub mod dictionary;
pub mod error;
pub(crate) mod file_registry;
pub mod fixed_delivery;
pub mod generic_constraints_echo;
pub mod hydro_models;
pub mod manifest;
pub(crate) mod parquet_config;
pub mod policy;
pub mod provenance;
pub mod results_writer;
pub mod scaling_report;
pub(crate) mod schemas;
pub mod simulation_writer;
pub mod software;
pub mod solver_stats_writer;
pub mod stochastic;
pub mod training_writer;

pub use dictionary::write_dictionaries;
pub use error::OutputError;
pub use fixed_delivery::{FixedDeliveryRow, write_fixed_delivery};
pub use generic_constraints_echo::{
    GENERIC_CONSTRAINT_ECHO_FILE, GenericConstraintEchoRow, write_generic_constraint_echo,
};
pub use hydro_models::{
    EVAPORATION_MODELS_FILE, FPHA_DEVIATION_POINTS_FILE, FPHA_HYPERPLANES_FILE,
    write_evaporation_models, write_fpha_deviation_points, write_fpha_hyperplanes,
    write_hydro_model_summary,
};
pub use manifest::{
    DeviationSummary, DeviationWorstEntry, DistributionInfo, HostLayout, MetadataBounds,
    MetadataConfiguration, MetadataConvergence, MetadataCost, MetadataIterations,
    MetadataProblemDimensions, MetadataRowPool, MetadataScenarios, MetadataSimulationSolveStats,
    MetadataTrainingSolveStats, OutputContext, RunStatus, SetupTimings, SimulationMetadata,
    TrainingMetadata, get_hostname, now_iso8601, read_simulation_metadata, read_training_metadata,
    write_simulation_metadata, write_training_metadata,
};
pub use provenance::write_provenance_report;
pub use results_writer::{
    remove_success_marker, write_results, write_simulation_results,
    write_skipped_simulation_results, write_success_marker, write_training_results,
};
pub use scaling_report::write_scaling_report;
pub use simulation_writer::{
    SimulationParquetWriter, remove_simulation_outputs, simulation_family_subpaths,
};
pub use software::{SOFTWARE_NAME, SOFTWARE_VERSION, SoftwareIdentity, policy_checkpoint_remedy};
pub use solver_stats_writer::{SolverStatsRow, write_simulation_solver_stats, write_solver_stats};
pub use stochastic::{
    FittingReductionEntry, FittingReport, HydroFittingEntry, write_correlation_json,
    write_fitting_report, write_inflow_annual_component, write_inflow_ar_coefficients,
    write_inflow_seasonal_stats, write_load_seasonal_stats, write_noise_openings,
};
pub use training_writer::{
    TrainingParquetWriter, remove_conditional_training_outputs, write_row_selection_records,
};

use fixed_delivery::FIXED_DELIVERIES_FILE;
use solver_stats_writer::{SIMULATION_SOLVER_DIR, TRAINING_SOLVER_DIR};
use training_writer::CUT_SELECTION_FILE;

/// How a run clears an output directory before writing its outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Clearing {
    /// The run removes the directory and everything inside it.
    WholeTree,
    /// The run removes only the files it writes there, and the directory only
    /// if that leaves it empty.
    NamedFiles,
}

/// The output subdirectories, relative to the output directory, that a run
/// clears before writing its outputs, and how.
pub(crate) fn cleared_output_dirs() -> Vec<(PathBuf, Clearing)> {
    let family_trees = simulation_family_subpaths()
        .map(|subpath| (Path::new("simulation").join(subpath), Clearing::WholeTree));
    let solver_dirs = [SIMULATION_SOLVER_DIR, TRAINING_SOLVER_DIR]
        .into_iter()
        .map(|dir| (PathBuf::from(dir), Clearing::NamedFiles));
    let named_file_dirs = [
        CUT_SELECTION_FILE,
        FIXED_DELIVERIES_FILE,
        FPHA_HYPERPLANES_FILE,
        EVAPORATION_MODELS_FILE,
        FPHA_DEVIATION_POINTS_FILE,
        GENERIC_CONSTRAINT_ECHO_FILE,
    ]
    .into_iter()
    .filter_map(|file| Path::new(file).parent())
    .map(|dir| (dir.to_path_buf(), Clearing::NamedFiles));

    let mut cleared: Vec<(PathBuf, Clearing)> = Vec::new();
    for (dir, clearing) in family_trees.chain(solver_dirs).chain(named_file_dirs) {
        if cleared.iter().all(|(listed, _)| *listed != dir) {
            cleared.push((dir, clearing));
        }
    }
    cleared
}

/// Arrow `Date32`'s native representation (days since the Unix epoch,
/// 1970-01-01) for one calendar date.
pub(crate) fn date32_days(date: NaiveDate) -> i32 {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).map_or(0, |e| e.num_days_from_ce());
    date.num_days_from_ce() - epoch
}

/// One row of convergence data for a single training iteration, written to
/// `training/convergence.parquet`.
///
/// `time_*` fields whose doc names a column map to that column in
/// `training/timing/iterations.parquet`; those tagged a sub-component of a pass
/// nest under that pass's wall-clock total rather than adding to the top level.
#[derive(Debug, Clone)]
pub struct IterationRecord {
    /// Sequential iteration number (1-based).
    pub iteration: u32,

    /// Lower bound on the optimal value at the end of this iteration.
    pub lower_bound: f64,

    /// Upper bound estimate for this iteration: the sample mean under a sampled
    /// forward, the exact probability-weighted bound under an enumerated forward.
    pub upper_bound: f64,

    /// Standard deviation of the upper bound estimate across scenarios. Written
    /// as NULL to `training/convergence.parquet` under an exact bound.
    pub upper_bound_std: f64,

    /// Relative gap between upper and lower bounds as a percentage, if defined.
    ///
    /// `None` when the lower bound is zero or negative (gap is ill-defined).
    pub gap_percent: Option<f64>,

    /// Rows added to the pool.
    pub cuts_added: u32,

    /// Rows removed from the pool.
    pub cuts_removed: u32,

    /// Total number of active rows in the pool after this iteration.
    pub cuts_active: u32,

    /// Forward pass wall-clock time (ms).
    pub time_forward_ms: u64,

    /// Backward pass wall-clock time (ms).
    pub time_backward_ms: u64,

    /// Total wall-clock time (ms).
    pub time_total_ms: u64,

    /// Forward pass wall-clock time (ms) → `forward_wall_ms`.
    pub time_forward_wall_ms: u64,

    /// Backward pass wall-clock time (ms) → `backward_wall_ms`.
    pub time_backward_wall_ms: u64,

    /// Row-selection phase time (ms) → `cut_selection_ms`.
    pub time_cut_selection_ms: u64,

    /// MPI allreduce (forward bound synchronization) time (ms) → `mpi_allreduce_ms`.
    pub time_mpi_allreduce_ms: u64,

    /// Per-stage row-sync allgatherv time (ms) → `cut_sync_ms`. Backward sub-component.
    pub time_cut_sync_ms: u64,

    /// Lower bound evaluation time (ms) → `lower_bound_ms`.
    pub time_lower_bound_ms: u64,

    /// State-exchange (`allgatherv`) time (ms) → `state_exchange_ms`. Backward sub-component.
    pub time_state_exchange_ms: u64,

    /// Row-batch assembly time (ms) → `cut_batch_build_ms`. Backward sub-component.
    pub time_cut_batch_build_ms: u64,

    /// Estimated backward worker load imbalance (ms) → `bwd_load_imbalance_ms`. Backward sub-component.
    pub time_bwd_load_imbalance_ms: u64,

    /// Backward scheduling/sync overhead (ms) → `bwd_scheduling_overhead_ms`. Backward sub-component.
    pub time_bwd_scheduling_overhead_ms: u64,

    /// Estimated forward worker load imbalance (ms) → `fwd_load_imbalance_ms`. Forward sub-component.
    pub time_fwd_load_imbalance_ms: u64,

    /// Forward scheduling/sync overhead (ms) → `fwd_scheduling_overhead_ms`. Forward sub-component.
    pub time_fwd_scheduling_overhead_ms: u64,

    /// Residual time not attributed to any phase (ms) → `overhead_ms`. Computed as
    /// `time_total_ms - (forward + backward + cut_selection + mpi_allreduce + lower_bound)`.
    pub time_overhead_ms: u64,

    /// Forward-pass scenarios solved.
    pub forward_passes: u32,

    /// Total number of LP solves (across all stages and passes) in this iteration.
    pub lp_solves: u32,

    /// Cumulative LP solve wall-clock time for this iteration, in milliseconds.
    pub solve_time_ms: f64,

    /// Mean resident row count loaded per lazy-selection LP solve during this
    /// iteration (reduced across ranks). `0.0` when no lazy selection ran. Maps
    /// to `mean_rows_in_lp` in `training/convergence.parquet`; it reflects the
    /// per-solve LP size the lazy selector actually carried, which (unlike the
    /// pool-level active count) shrinks well below the generated total.
    pub mean_rows_in_lp: f64,
}

/// Summary statistics for the row pool at the end of a training run.
///
/// Carried inside [`TrainingOutput`] and written to `training/timing/cut_stats.parquet`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RowPoolStatistics {
    /// Total number of rows generated over the entire training run.
    pub total_generated: u64,

    /// Number of rows still active in the pool at the end of training.
    pub total_active: u64,

    /// Highest number of active rows observed at any point during training.
    pub peak_active: u64,

    /// Total rows currently active in the LP.
    pub cuts_active: u64,

    /// Sum, over every lazy-selection LP solve in the run, of the resident row
    /// count loaded into that solve (reduced across ranks). With
    /// [`Self::rows_in_lp_solve_count`] this gives the mean rows-in-LP per solve.
    /// Zero when no lazy selection ran (the resident-subset solve path was never
    /// taken), letting consumers distinguish "not applicable" from "zero rows".
    pub rows_in_lp_total: u64,

    /// Number of lazy-selection LP solves in the run (reduced across ranks); the
    /// denominator for the mean rows-in-LP. Zero when no lazy selection ran.
    pub rows_in_lp_solve_count: u64,

    /// Largest resident row count loaded into any single lazy-selection LP solve
    /// over the run (reduced across ranks). Zero when no lazy selection ran.
    pub rows_in_lp_max: u64,

    /// Rows loaded from a boundary policy rather than generated by this run;
    /// a subset of [`Self::total_generated`]. Zero when no boundary policy loaded.
    pub total_loaded: u64,
}

/// One row in `training/cut_selection/iterations.parquet`.
///
/// Represents per-stage row-selection statistics for a single iteration.
/// Only populated when row selection is enabled.
#[derive(Debug, Clone)]
pub struct RowSelectionRecord {
    /// Iteration number (1-based).
    pub iteration: u32,
    /// 0-based stage index.
    pub stage: u32,
    /// Total cuts ever generated at this stage.
    pub cuts_populated: u32,
    /// Active cuts before selection ran.
    pub cuts_active_before: u32,
    /// Cuts deactivated by selection at this stage.
    pub cuts_deactivated: u32,
    /// Number of cuts reactivated this iteration.
    pub cuts_reactivated: u32,
    /// Active cuts after selection.
    pub cuts_active_after: u32,
    /// Wall-clock time for selection at this stage, in milliseconds.
    pub selection_time_ms: f64,
    /// Cuts evicted by budget enforcement at this stage.
    ///
    /// `None` when budget enforcement is disabled (`max_active_per_stage` is absent).
    pub budget_evicted: Option<u32>,
    /// Active cuts after budget enforcement.
    ///
    /// `None` when budget enforcement is disabled.
    pub active_after_budget: Option<u32>,
}

/// One row in `training/timing/iterations.parquet`.
///
/// The timing parquet stores multiple rows per iteration:
///
/// - One **rank-aggregated** row per `(iteration, rank)` carrying rank-only
///   timing columns (`worker_id = None`). Per-worker slots are `0` on this row.
/// - One **per-worker** row per `(iteration, rank, worker_id)` carrying
///   per-worker slots (`forward_wall_ms`, `backward_wall_ms`, `bwd_setup_ms`,
///   `fwd_setup_ms`, `lazy_scoring_ms`). Rank-only slots are `0` on these rows.
///
/// `SUM(col) GROUP BY iteration` across all rows recovers the
/// single-row-per-iteration value for each of the 16 timing columns.
#[derive(Debug, Clone)]
pub struct WorkerTimingRecord {
    /// Training iteration (1-based).
    pub iteration: u32,
    /// MPI rank that produced this row.
    pub rank: i32,
    /// Rayon worker index within the rank's pool, or `None` for rank-aggregated rows.
    pub worker_id: Option<i32>,
    /// Fixed-size timing payload matching the 16 timing columns of
    /// `iteration_timing_schema()` (positions 3–18, after `iteration`, `rank`,
    /// `worker_id`). Slot indices correspond to the `WORKER_TIMING_SLOT_*`
    /// constants defined in `cobre-core`.
    pub timings: [u64; 16],
}

/// Aggregate type carrying all training data needed for output writing.
///
/// Constructed by the solver after training completes and passed to
/// [`write_results`]. All convergence records and summary statistics are
/// held here so the writer can read them without contacting the solver.
#[derive(Debug, Clone)]
pub struct TrainingOutput {
    /// Ordered convergence records — one entry per completed iteration.
    pub convergence_records: Vec<IterationRecord>,

    /// Lower bound value reported after the final iteration.
    pub final_lower_bound: f64,

    /// Upper bound value reported after the final iteration, if available.
    ///
    /// `None` when no upper-bound evaluation was performed.
    pub final_upper_bound: Option<f64>,

    /// Relative gap between final upper and lower bounds as a percentage.
    ///
    /// `None` when the lower bound is zero/negative or `final_upper_bound` is `None`.
    pub final_gap_percent: Option<f64>,

    /// Standard deviation of the final upper-bound estimate, if available.
    ///
    /// `None` when no upper-bound evaluation was performed or the bound is exact.
    /// The value is carried separately in
    /// [`final_upper_bound`](Self::final_upper_bound).
    pub final_upper_bound_std: Option<f64>,

    /// Upper-bound regime for the whole run: `"statistical"` (sampled forward) or
    /// `"exact"` (enumerated forward). Mirrored into `training/convergence.parquet`
    /// and `training/metadata.json`.
    pub final_upper_bound_kind: String,

    /// Number of iterations completed before the stopping condition was triggered.
    pub iterations_completed: u32,

    /// `true` when training converged within the configured tolerance.
    pub converged: bool,

    /// Human-readable description of the rule that terminated training.
    pub termination_reason: String,

    /// The phase status written to `training/metadata.json`.
    pub status: RunStatus,

    /// Total elapsed wall-clock time for the entire training run (ms).
    pub total_time_ms: u64,

    /// Summary row pool statistics for the run.
    pub cut_stats: RowPoolStatistics,

    /// Per-stage row-selection records for Parquet output.
    ///
    /// Empty when row selection is disabled. When non-empty, written to
    /// `training/cut_selection/iterations.parquet`.
    pub cut_selection_records: Vec<RowSelectionRecord>,

    /// Per-worker timing records for `training/timing/iterations.parquet`.
    ///
    /// Each entry is either a rank-aggregated row
    /// (`worker_id = None`) or a per-worker row (`worker_id = Some(w)`).
    /// Empty when timing data was not collected (e.g. single-threaded runs
    /// without the instrumentation wired). Written in iteration-major order:
    /// rank-aggregated row first, then per-worker rows sorted by
    /// `(rank, worker_id)`.
    pub worker_timing_records: Vec<WorkerTimingRecord>,

    /// Aggregate solve statistics for the training run.
    ///
    /// Default-constructed (all fields `None`) by producers that do not yet
    /// record solve statistics; populated downstream and persisted into
    /// `training/metadata.json` by the metadata writer.
    pub training_solve_stats: MetadataTrainingSolveStats,
}

/// Aggregate type carrying simulation completion data for output writing.
///
/// Constructed by the simulation pipeline after it completes and optionally
/// passed to [`write_results`]. When `None` is supplied, the simulation
/// output directory is still created (ready for future use), but no
/// simulation artifacts are written.
#[derive(Debug, Clone)]
pub struct SimulationOutput {
    /// Total number of scenarios dispatched for simulation.
    pub n_scenarios: u32,

    /// Number of scenarios that completed without error.
    pub completed: u32,

    /// Number of scenarios that failed during simulation.
    pub failed: u32,

    /// Total elapsed wall-clock time for the simulation run (ms).
    pub total_time_ms: u64,

    /// Aggregate cost statistics for the simulated scenarios.
    ///
    /// `None` until a producer supplies it. When several per-rank outputs are
    /// combined via [`merge`](Self::merge), the first present value wins; the
    /// producer is responsible for supplying the authoritative aggregate (which
    /// the distributed pipeline computes on rank 0) first.
    pub cost: Option<MetadataCost>,

    /// Aggregate solve statistics for the simulation run.
    ///
    /// Default-constructed (all fields `None`) by producers that do not yet
    /// record solve statistics; populated downstream and persisted into
    /// `simulation/metadata.json` by the metadata writer.
    pub solve_stats: MetadataSimulationSolveStats,
}

impl SimulationOutput {
    /// Combine multiple per-rank simulation outputs into a single aggregate.
    ///
    /// Merge rules:
    /// - `n_scenarios`: sum across all outputs.
    /// - `completed`: sum across all outputs.
    /// - `failed`: sum across all outputs.
    /// - `total_time_ms`: max across all outputs (wall-clock = slowest rank).
    /// - `cost`: first present value in slice order. The producer must supply
    ///   the authoritative aggregate first (the distributed pipeline computes it
    ///   on rank 0), so the merged cost is the rank-0 aggregate rather than a
    ///   per-rank partial. `None` only when no input carried a cost.
    /// - `solve_stats`: each count field is summed treating `None` as `0`, with
    ///   the result `Some` when any input recorded it (and `None` only when no
    ///   input did); `solve_seconds` is summed with the same convention;
    ///   `parallelism` takes the maximum across inputs (`None`-safe). Sums and
    ///   max are order-invariant, so the merge is declaration-order invariant.
    ///
    /// Returns a zeroed [`SimulationOutput`] (no cost, default solve stats) when the
    /// input slice is empty.
    #[must_use]
    pub fn merge(outputs: &[Self]) -> Self {
        if outputs.is_empty() {
            return Self {
                n_scenarios: 0,
                completed: 0,
                failed: 0,
                total_time_ms: 0,
                cost: None,
                solve_stats: MetadataSimulationSolveStats::default(),
            };
        }

        let n_scenarios = outputs.iter().map(|o| o.n_scenarios).sum();
        let completed = outputs.iter().map(|o| o.completed).sum();
        let failed = outputs.iter().map(|o| o.failed).sum();
        let total_time_ms = outputs.iter().map(|o| o.total_time_ms).max().unwrap_or(0);

        let cost = outputs.iter().find_map(|o| o.cost.clone());

        let solve_stats = merge_simulation_solve_stats(outputs);

        Self {
            n_scenarios,
            completed,
            failed,
            total_time_ms,
            cost,
            solve_stats,
        }
    }
}

/// Order-invariant sum of an optional `u64` field across simulation outputs.
///
/// Returns `Some(sum)` when at least one input carried the field (treating
/// `None` as `0`), and `None` when no input recorded it. Addition is
/// commutative, so the result is independent of slice order.
fn sum_optional_u64(
    outputs: &[SimulationOutput],
    field: impl Fn(&MetadataSimulationSolveStats) -> Option<u64>,
) -> Option<u64> {
    let mut any = false;
    let mut total: u64 = 0;
    for output in outputs {
        if let Some(value) = field(&output.solve_stats) {
            any = true;
            total = total.saturating_add(value);
        }
    }
    any.then_some(total)
}

/// Combine per-rank simulation solve statistics into a single aggregate.
///
/// Count fields and `solve_seconds` are summed (treating `None` as `0`, result
/// `Some` if any input was `Some`); `parallelism` takes the maximum. All
/// operations are order-invariant.
fn merge_simulation_solve_stats(outputs: &[SimulationOutput]) -> MetadataSimulationSolveStats {
    let mut solve_seconds_any = false;
    let mut solve_seconds_total: f64 = 0.0;
    for output in outputs {
        if let Some(value) = output.solve_stats.solve_seconds {
            solve_seconds_any = true;
            solve_seconds_total += value;
        }
    }

    let parallelism = outputs
        .iter()
        .filter_map(|o| o.solve_stats.parallelism)
        .max();

    MetadataSimulationSolveStats {
        total_lp_solves: sum_optional_u64(outputs, |s| s.total_lp_solves),
        first_try: sum_optional_u64(outputs, |s| s.first_try),
        retried: sum_optional_u64(outputs, |s| s.retried),
        failed: sum_optional_u64(outputs, |s| s.failed),
        solve_seconds: solve_seconds_any.then_some(solve_seconds_total),
        parallelism,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    #[test]
    fn cleared_output_dirs_lists_each_directory_once() {
        let cleared = cleared_output_dirs();

        let mut paths: Vec<&Path> = cleared.iter().map(|(dir, _)| dir.as_path()).collect();
        paths.sort_unstable();
        paths.dedup();
        assert_eq!(
            paths.len(),
            cleared.len(),
            "a path is listed twice: {cleared:?}"
        );

        let whole_trees: Vec<&Path> = cleared
            .iter()
            .filter(|(_, clearing)| *clearing == Clearing::WholeTree)
            .map(|(dir, _)| dir.as_path())
            .collect();
        let family_trees: Vec<PathBuf> = simulation_family_subpaths()
            .map(|subpath| Path::new("simulation").join(subpath))
            .collect();
        assert_eq!(whole_trees, family_trees);

        for dir in [
            "simulation/solver",
            "training/solver",
            "training/cut_selection",
            "anticipated",
            "hydro_models",
            "generic_constraints",
        ] {
            assert!(
                cleared.contains(&(PathBuf::from(dir), Clearing::NamedFiles)),
                "{dir} must be cleared file by file: {cleared:?}"
            );
        }
        for dir in ["simulation", "training", "stochastic"] {
            assert!(
                cleared.iter().all(|(listed, _)| listed != Path::new(dir)),
                "{dir} must not be a cleared directory: {cleared:?}"
            );
        }
    }

    #[test]
    fn training_output_construction_and_field_access() {
        let records: Vec<IterationRecord> = (1..=5)
            .map(|i| IterationRecord {
                iteration: i,
                lower_bound: 1.0,
                upper_bound: 2.0,
                upper_bound_std: 0.1,
                gap_percent: Some(50.0),
                cuts_added: 10,
                cuts_removed: 2,
                cuts_active: 8,
                time_forward_ms: 100,
                time_backward_ms: 200,
                time_total_ms: 300,
                forward_passes: 4,
                lp_solves: 40,
                time_forward_wall_ms: 100,
                time_backward_wall_ms: 200,
                time_cut_selection_ms: 0,
                time_mpi_allreduce_ms: 0,
                time_cut_sync_ms: 0,
                time_lower_bound_ms: 0,
                time_state_exchange_ms: 0,
                time_cut_batch_build_ms: 0,
                time_bwd_load_imbalance_ms: 0,
                time_bwd_scheduling_overhead_ms: 0,
                time_fwd_load_imbalance_ms: 0,
                time_fwd_scheduling_overhead_ms: 0,
                time_overhead_ms: 0,
                solve_time_ms: 0.0,
                mean_rows_in_lp: 0.0,
            })
            .collect();
        let output = TrainingOutput {
            convergence_records: records,
            final_lower_bound: 50.0,
            final_upper_bound: Some(52.0),
            final_gap_percent: Some(3.85),
            final_upper_bound_std: Some(0.5),
            final_upper_bound_kind: "statistical".to_string(),
            iterations_completed: 5,
            converged: true,
            termination_reason: "relative gap < 1%".to_string(),
            status: RunStatus::Complete,
            total_time_ms: 12_000,
            cut_stats: RowPoolStatistics {
                total_generated: 300,
                total_active: 120,
                peak_active: 150,
                cuts_active: 120,
                rows_in_lp_total: 0,
                rows_in_lp_solve_count: 0,
                rows_in_lp_max: 0,
                total_loaded: 0,
            },
            cut_selection_records: vec![],
            worker_timing_records: vec![],
            training_solve_stats: MetadataTrainingSolveStats::default(),
        };

        assert_eq!(output.convergence_records.len(), 5);
        assert_eq!(output.final_lower_bound, 50.0);
        assert_eq!(output.final_upper_bound, Some(52.0));
        assert_eq!(output.final_gap_percent, Some(3.85));
        assert_eq!(output.final_upper_bound_std, Some(0.5));
        assert_eq!(output.iterations_completed, 5);
        assert!(output.converged);
        assert_eq!(output.termination_reason, "relative gap < 1%");
        assert_eq!(output.total_time_ms, 12_000);
        assert_eq!(output.cut_stats.total_generated, 300);
        assert_eq!(output.cut_stats.total_active, 120);
        assert_eq!(output.cut_stats.peak_active, 150);
    }

    #[test]
    fn iteration_record_construction_and_field_access() {
        let record = IterationRecord {
            iteration: 7,
            lower_bound: 10.5,
            upper_bound: 11.0,
            upper_bound_std: 0.25,
            gap_percent: Some(4.55),
            cuts_added: 15,
            cuts_removed: 3,
            cuts_active: 42,
            time_forward_ms: 150,
            time_backward_ms: 250,
            time_total_ms: 400,
            forward_passes: 8,
            lp_solves: 80,
            time_forward_wall_ms: 150,
            time_backward_wall_ms: 250,
            time_cut_selection_ms: 5,
            time_mpi_allreduce_ms: 3,
            time_cut_sync_ms: 2,
            time_lower_bound_ms: 4,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_bwd_load_imbalance_ms: 0,
            time_bwd_scheduling_overhead_ms: 0,
            time_fwd_load_imbalance_ms: 0,
            time_fwd_scheduling_overhead_ms: 0,
            time_overhead_ms: 400u64.saturating_sub(150 + 250 + 5 + 3 + 4),
            solve_time_ms: 0.0,
            mean_rows_in_lp: 0.0,
        };

        assert_eq!(record.iteration, 7);
        assert_eq!(record.lower_bound, 10.5);
        assert_eq!(record.upper_bound, 11.0);
        assert_eq!(record.upper_bound_std, 0.25);
        assert_eq!(record.gap_percent, Some(4.55));
        assert_eq!(record.cuts_added, 15);
        assert_eq!(record.cuts_removed, 3);
        assert_eq!(record.cuts_active, 42);
        assert_eq!(record.time_forward_ms, 150);
        assert_eq!(record.time_backward_ms, 250);
        assert_eq!(record.time_total_ms, 400);
        assert_eq!(record.forward_passes, 8);
        assert_eq!(record.lp_solves, 80);
        assert_eq!(record.time_forward_wall_ms, 150);
        assert_eq!(record.time_backward_wall_ms, 250);
        assert_eq!(record.time_cut_selection_ms, 5);
        assert_eq!(record.time_mpi_allreduce_ms, 3);
        assert_eq!(record.time_cut_sync_ms, 2);
        assert_eq!(record.time_lower_bound_ms, 4);
    }

    #[test]
    fn simulation_output_construction_and_field_access() {
        let output = SimulationOutput {
            n_scenarios: 100,
            completed: 100,
            failed: 0,
            total_time_ms: 3_200,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        };

        assert_eq!(output.n_scenarios, 100);
        assert_eq!(output.completed, 100);
        assert_eq!(output.failed, 0);
        assert_eq!(output.total_time_ms, 3_200);
    }

    #[test]
    fn row_pool_statistics_construction() {
        let stats = RowPoolStatistics {
            total_generated: 500,
            total_active: 200,
            peak_active: 250,
            cuts_active: 200,
            rows_in_lp_total: 0,
            rows_in_lp_solve_count: 0,
            rows_in_lp_max: 0,
            total_loaded: 0,
        };

        assert_eq!(stats.total_generated, 500);
        assert_eq!(stats.total_active, 200);
        assert_eq!(stats.peak_active, 250);
        assert_eq!(stats.cuts_active, 200);
    }

    #[test]
    fn row_pool_statistics_serializes_with_new_fields() {
        let stats = RowPoolStatistics {
            total_generated: 10,
            total_active: 7,
            peak_active: 9,
            cuts_active: 7,
            rows_in_lp_total: 30,
            rows_in_lp_solve_count: 6,
            rows_in_lp_max: 8,
            total_loaded: 3,
        };
        let json = serde_json::to_string(&stats).expect("serialization must succeed");
        assert!(
            !json.contains("\"cuts_in_lp\""),
            "JSON must not contain cuts_in_lp key"
        );
        assert!(
            json.contains("\"cuts_active\""),
            "JSON must contain cuts_active key"
        );
        for key in [
            "\"rows_in_lp_total\"",
            "\"rows_in_lp_solve_count\"",
            "\"rows_in_lp_max\"",
            "\"total_loaded\"",
        ] {
            assert!(json.contains(key), "JSON must contain {key}");
        }
    }

    #[test]
    fn test_merge_empty_slice() {
        let merged = SimulationOutput::merge(&[]);
        assert_eq!(merged.n_scenarios, 0);
        assert_eq!(merged.completed, 0);
        assert_eq!(merged.failed, 0);
        assert_eq!(merged.total_time_ms, 0);
    }

    #[test]
    fn test_merge_single_output() {
        let output = SimulationOutput {
            n_scenarios: 5,
            completed: 4,
            failed: 1,
            total_time_ms: 1000,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        };
        let merged = SimulationOutput::merge(std::slice::from_ref(&output));
        assert_eq!(merged.n_scenarios, 5);
        assert_eq!(merged.completed, 4);
        assert_eq!(merged.failed, 1);
        assert_eq!(merged.total_time_ms, 1000);
    }

    #[test]
    fn test_merge_two_outputs() {
        let a = SimulationOutput {
            n_scenarios: 3,
            completed: 3,
            failed: 0,
            total_time_ms: 500,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        };
        let b = SimulationOutput {
            n_scenarios: 2,
            completed: 1,
            failed: 1,
            total_time_ms: 800,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        };
        let merged = SimulationOutput::merge(&[a, b]);
        assert_eq!(merged.n_scenarios, 5);
        assert_eq!(merged.completed, 4);
        assert_eq!(merged.failed, 1);
        // total_time_ms uses max, not sum
        assert_eq!(merged.total_time_ms, 800);
    }

    #[test]
    fn simulation_output_merge_combines_solve_stats_order_invariant() {
        let a = SimulationOutput {
            n_scenarios: 2,
            completed: 2,
            failed: 0,
            total_time_ms: 500,
            cost: Some(MetadataCost {
                mean_cost: 100.0,
                std_cost: 10.0,
            }),
            solve_stats: MetadataSimulationSolveStats {
                total_lp_solves: Some(40),
                first_try: Some(35),
                retried: Some(5),
                failed: Some(0),
                solve_seconds: Some(1.5),
                parallelism: Some(4),
            },
        };
        let b = SimulationOutput {
            n_scenarios: 3,
            completed: 3,
            failed: 0,
            total_time_ms: 800,
            cost: Some(MetadataCost {
                mean_cost: 200.0,
                std_cost: 20.0,
            }),
            solve_stats: MetadataSimulationSolveStats {
                total_lp_solves: Some(60),
                first_try: Some(50),
                retried: Some(8),
                failed: Some(2),
                solve_seconds: Some(2.5),
                parallelism: Some(8),
            },
        };

        let merged_ab = SimulationOutput::merge(&[a.clone(), b.clone()]);
        let merged_ba = SimulationOutput::merge(&[b, a]);

        assert_eq!(
            merged_ab.solve_stats.total_lp_solves,
            merged_ba.solve_stats.total_lp_solves
        );
        assert_eq!(merged_ab.solve_stats.total_lp_solves, Some(100));
        assert_eq!(merged_ab.solve_stats.first_try, Some(85));
        assert_eq!(
            merged_ab.solve_stats.first_try,
            merged_ba.solve_stats.first_try
        );
        assert_eq!(merged_ab.solve_stats.retried, Some(13));
        assert_eq!(merged_ab.solve_stats.retried, merged_ba.solve_stats.retried);
        assert_eq!(merged_ab.solve_stats.failed, Some(2));
        assert_eq!(merged_ab.solve_stats.failed, merged_ba.solve_stats.failed);

        assert_eq!(merged_ab.solve_stats.solve_seconds, Some(4.0));
        assert_eq!(
            merged_ab.solve_stats.solve_seconds,
            merged_ba.solve_stats.solve_seconds
        );

        assert_eq!(merged_ab.solve_stats.parallelism, Some(8));
        assert_eq!(
            merged_ab.solve_stats.parallelism,
            merged_ba.solve_stats.parallelism
        );

        assert_eq!(
            merged_ab.cost.as_ref().map(|c| c.mean_cost),
            Some(100.0),
            "first-present cost wins in [a, b] order"
        );
        assert_eq!(
            merged_ba.cost.as_ref().map(|c| c.mean_cost),
            Some(200.0),
            "first-present cost wins in [b, a] order"
        );
    }

    #[test]
    fn simulation_output_merge_solve_stats_none_when_no_input_records() {
        let a = SimulationOutput {
            n_scenarios: 1,
            completed: 1,
            failed: 0,
            total_time_ms: 100,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        };
        let merged = SimulationOutput::merge(std::slice::from_ref(&a));
        assert_eq!(merged.solve_stats.total_lp_solves, None);
        assert_eq!(merged.solve_stats.solve_seconds, None);
        assert_eq!(merged.solve_stats.parallelism, None);
        assert!(merged.cost.is_none());
    }
}
