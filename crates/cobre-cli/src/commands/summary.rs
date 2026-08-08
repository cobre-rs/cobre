//! `cobre summary <OUTPUT_DIR>` subcommand.
//!
//! Reprints the human-readable end-block of a completed run, reusing the live
//! `print_*` renderers so the output matches a live run by construction.
//!
//! Per-file contract (drives error vs. skip):
//!
//! - `training/metadata.json` — required; missing file returns [`CliError::Io`].
//! - `training/convergence.parquet` — optional; missing file falls back to
//!   zero-valued bounds (`lp_solves` and timing reported as 0).
//! - `training/hydro_models.json`, `training/model_provenance.json`,
//!   `simulation/metadata.json` — optional; a missing file skips its section.
//!
//! A *present but malformed* optional sidecar is a real error, not a skip.
//!
//! All output goes to stderr; stdout is reserved for `cobre report`.

use std::path::PathBuf;

use clap::Args;
use console::Term;

use cobre_comm::{BackendKind, ExecutionTopology, HostInfo, MpiRuntimeInfo, SlurmJobInfo};
use cobre_io::{
    ConvergenceSummary, DistributionInfo, OutputError, SimulationMetadata, TrainingMetadata,
    output::read_initial_gap_percent, read_convergence_summary, read_hydro_model_summary,
    read_provenance_report, read_simulation_metadata, read_training_metadata,
};
use cobre_sddp::{HydroModelSummary, ModelProvenanceReport};

use crate::{
    error::CliError,
    summary::{
        SimulationSummary, TrainingSummary, print_execution_topology, print_hydro_model_summary,
        print_provenance_summary, print_simulation_summary, print_training_summary,
    },
};

// ── Arguments ────────────────────────────────────────────────────────────────

/// Arguments for the `cobre summary` subcommand.
#[derive(Debug, Args)]
#[command(about = "Display the post-run summary from a completed output directory")]
pub struct SummaryArgs {
    /// Path to the output directory produced by `cobre run`.
    pub output_dir: PathBuf,
}

// ── Execute ──────────────────────────────────────────────────────────────────

/// Execute the `summary` subcommand.
///
/// Prints the training (and optionally simulation) summary for `args.output_dir`
/// to stderr, matching what `cobre run` prints at the end of a study.
///
/// # Errors
///
/// - [`CliError::Io`] when the output directory does not exist or
///   `training/metadata.json` cannot be read.
/// - [`CliError::Internal`] when a metadata file contains malformed JSON.
pub fn execute(args: SummaryArgs) -> Result<(), CliError> {
    let output_dir = args.output_dir;

    if !output_dir.try_exists().map_err(|e| CliError::Io {
        source: e,
        context: "output directory".to_string(),
    })? {
        return Err(CliError::Io {
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "output directory not found"),
            context: output_dir.display().to_string(),
        });
    }

    let training_metadata_path = output_dir.join("training/metadata.json");
    let metadata: TrainingMetadata =
        read_training_metadata(&training_metadata_path).map_err(CliError::from)?;

    // convergence.parquet optional: any read error falls back to a zero-valued summary.
    let convergence_path = output_dir.join("training/convergence.parquet");
    let convergence = read_convergence_summary(&convergence_path)
        .unwrap_or_else(|_| convergence_fallback(&metadata));
    let initial_gap_percent = read_initial_gap_percent(&convergence_path);

    let hydro_models_path = output_dir.join("training/hydro_models.json");
    let hydro_models: Option<HydroModelSummary> =
        read_optional_sidecar(read_hydro_model_summary(&hydro_models_path))?;

    let provenance_path = output_dir.join("training/model_provenance.json");
    let provenance: Option<ModelProvenanceReport> =
        read_optional_sidecar(read_provenance_report(&provenance_path))?;

    let simulation_metadata_path = output_dir.join("simulation/metadata.json");
    let simulation: Option<SimulationMetadata> =
        read_optional_sidecar(read_simulation_metadata(&simulation_metadata_path))?;

    let training_summary = build_training_summary(&metadata, &convergence, initial_gap_percent);
    let stderr = Term::stderr();

    // Sections are blank-line separated; an absent optional section drops its separator.
    let topology = reconstruct_topology(&metadata.distribution);
    print_execution_topology(
        &stderr,
        &topology,
        metadata.distribution.threads_per_rank as usize,
        &metadata.solver,
        metadata.solver_version.as_deref(),
        metadata
            .distribution
            .rank_affinity
            .iter()
            .find(|affinity| affinity.rank == 0),
    );

    if let Some(hydro) = &hydro_models {
        let _ = stderr.write_line("");
        print_hydro_model_summary(&stderr, hydro);
    }

    if let Some(report) = &provenance {
        let _ = stderr.write_line("");
        print_provenance_summary(&stderr, report);
    }

    let _ = stderr.write_line("");
    print_training_summary(&stderr, &training_summary);

    if let Some(sim) = simulation {
        let simulation_summary = build_simulation_summary(&sim);
        let _ = stderr.write_line("");
        print_simulation_summary(&stderr, &simulation_summary);
    }

    Ok(())
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Zero-valued [`ConvergenceSummary`] used when `convergence.parquet` is missing
/// or unreadable; carries over the metadata's `final_gap_percent`.
fn convergence_fallback(metadata: &TrainingMetadata) -> ConvergenceSummary {
    ConvergenceSummary {
        total_lp_solves: 0,
        total_time_ms: 0,
        final_lower_bound: 0.0,
        final_upper_bound_mean: 0.0,
        final_upper_bound_std: 0.0,
        final_gap_percent: metadata.convergence.final_gap_percent,
    }
}

fn build_training_summary(
    metadata: &TrainingMetadata,
    convergence: &ConvergenceSummary,
    initial_gap_percent: Option<f64>,
) -> TrainingSummary {
    TrainingSummary {
        iterations: u64::from(metadata.iterations.completed),
        converged: metadata.convergence.achieved,
        converged_at: metadata.iterations.converged_at.map(u64::from),
        reason: metadata.convergence.termination_reason.clone(),
        lower_bound: if metadata.bounds.final_lower_bound.abs() > f64::EPSILON {
            metadata.bounds.final_lower_bound
        } else {
            convergence.final_lower_bound
        },
        upper_bound: metadata
            .bounds
            .final_upper_bound
            .unwrap_or(convergence.final_upper_bound_mean),
        upper_bound_std: metadata
            .bounds
            .final_upper_bound_std
            .unwrap_or(convergence.final_upper_bound_std),
        gap_percent: convergence.final_gap_percent.unwrap_or(0.0),
        total_rows_active: metadata.row_pool.total_active,
        total_rows_generated: metadata.row_pool.total_generated,
        rows_in_lp_total: metadata.row_pool.rows_in_lp_total,
        rows_in_lp_solve_count: metadata.row_pool.rows_in_lp_solve_count,
        rows_in_lp_max: metadata.row_pool.rows_in_lp_max,
        num_stages: metadata.problem_dimensions.num_stages,
        total_lp_solves: convergence.total_lp_solves,
        total_time_ms: convergence.total_time_ms,
        total_first_try: metadata.solve_stats.first_try,
        total_retried: metadata.solve_stats.retried,
        total_failed: metadata.solve_stats.failed,
        total_forward_solve_seconds: metadata.solve_stats.forward_solve_seconds,
        total_backward_solve_seconds: metadata.solve_stats.backward_solve_seconds,
        parallelism: metadata.solve_stats.parallelism,
        initial_gap_percent,
        // Per-iteration timing is not persisted to metadata.json; unavailable
        // when a summary is reconstructed from a completed output directory.
        forward_phase_wall_seconds: None,
        backward_phase_wall_seconds: None,
        forward_wait_seconds: None,
        backward_wait_seconds: None,
        serial_lower_bound_seconds: None,
        serial_cut_selection_seconds: None,
        serial_cut_sync_seconds: None,
        serial_allreduce_seconds: None,
        serial_scheduling_seconds: None,
    }
}

fn build_simulation_summary(metadata: &SimulationMetadata) -> SimulationSummary {
    // duration_seconds is non-negative wall-clock, so the ms round-and-truncate
    // cannot lose sign or overflow.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total_time_ms = (metadata.duration_seconds * 1000.0).round() as u64;

    SimulationSummary {
        n_scenarios: metadata.scenarios.total,
        completed: metadata.scenarios.completed,
        failed: metadata.scenarios.failed,
        total_time_ms,
        mean_cost: metadata.cost.as_ref().map(|c| c.mean_cost),
        std_cost: metadata.cost.as_ref().map(|c| c.std_cost),
        total_lp_solves: metadata.solve_stats.total_lp_solves,
        total_first_try: metadata.solve_stats.first_try,
        total_retried: metadata.solve_stats.retried,
        total_failed_solves: metadata.solve_stats.failed,
        total_solve_time_seconds: metadata.solve_stats.solve_seconds,
        parallelism: metadata.solve_stats.parallelism,
    }
}

/// A `NotFound` [`OutputError::IoError`] maps to `Ok(None)` (caller skips the
/// section); every other error — including a malformed file — propagates.
fn read_optional_sidecar<T>(result: Result<T, OutputError>) -> Result<Option<T>, CliError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(OutputError::IoError { source, .. })
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            Ok(None)
        }
        Err(e) => Err(CliError::from(e)),
    }
}

/// Rebuild an [`ExecutionTopology`] from the persisted [`DistributionInfo`] so
/// `summary` can reuse the live [`print_execution_topology`] renderer.
///
/// Fields absent from `DistributionInfo` reconstruct as `None`: a SLURM job keeps
/// only `job_id`; `mpi` is `Some` only when all three MPI strings are present.
fn reconstruct_topology(dist: &DistributionInfo) -> ExecutionTopology {
    let backend = match dist.backend.as_str() {
        "mpi" => BackendKind::Mpi,
        "local" => BackendKind::Local,
        _ => BackendKind::Auto,
    };

    // u32 -> usize is a lossless widening on all supported targets.
    let hosts = dist
        .hosts
        .iter()
        .map(|host| HostInfo {
            hostname: host.hostname.clone(),
            ranks: host.ranks.iter().map(|&r| r as usize).collect(),
        })
        .collect();

    let mpi = match (
        dist.mpi_library.as_ref(),
        dist.mpi_standard.as_ref(),
        dist.thread_level.as_ref(),
    ) {
        (Some(library), Some(standard), Some(thread_level)) => Some(MpiRuntimeInfo {
            library_version: library.clone(),
            standard_version: standard.clone(),
            thread_level: thread_level.clone(),
        }),
        _ => None,
    };

    let slurm = dist.slurm_job_id.clone().map(|job_id| SlurmJobInfo {
        job_id,
        node_list: None,
        cpus_per_task: None,
    });

    ExecutionTopology {
        backend,
        world_size: dist.world_size as usize,
        hosts,
        mpi,
        slurm,
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use cobre_comm::BackendKind;
    use cobre_io::{
        ConvergenceSummary, DistributionInfo, HostLayout, MetadataBounds, MetadataConfiguration,
        MetadataConvergence, MetadataCost, MetadataIterations, MetadataProblemDimensions,
        MetadataRowPool, MetadataScenarios, MetadataSimulationSolveStats,
        MetadataTrainingSolveStats, OutputError, SimulationMetadata, TrainingMetadata,
    };

    use super::{
        CliError, SummaryArgs, build_simulation_summary, build_training_summary,
        convergence_fallback, read_optional_sidecar, reconstruct_topology,
    };

    fn make_training_metadata() -> TrainingMetadata {
        TrainingMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            solver_version: None,
            started_at: "2026-01-17T08:00:00Z".to_string(),
            completed_at: "2026-01-17T12:30:00Z".to_string(),
            duration_seconds: 16_200.0,
            status: "complete".to_string(),
            configuration: MetadataConfiguration {
                seed: Some(42),
                max_iterations: Some(100),
                forward_passes: Some(192),
                stopping_mode: "any".to_string(),
                policy_mode: "fresh".to_string(),
            },
            problem_dimensions: MetadataProblemDimensions {
                num_stages: 12,
                num_hydros: 160,
                num_thermals: 200,
                num_buses: 5,
                num_lines: 8,
            },
            iterations: MetadataIterations {
                completed: 42,
                converged_at: Some(42),
            },
            convergence: MetadataConvergence {
                achieved: true,
                final_gap_percent: Some(0.45),
                termination_reason: "gap_tolerance".to_string(),
            },
            row_pool: MetadataRowPool {
                total_generated: 1_250_000,
                total_active: 980_000,
                peak_active: 1_100_000,
                cuts_active: 980_000,
                rows_in_lp_total: 0,
                rows_in_lp_solve_count: 0,
                rows_in_lp_max: 0,
            },
            bounds: MetadataBounds {
                final_lower_bound: 48_500.0,
                final_upper_bound: Some(49_000.0),
                final_upper_bound_std: Some(250.0),
            },
            solve_stats: MetadataTrainingSolveStats {
                total_lp_solves: Some(84_000),
                first_try: Some(80_000),
                retried: Some(3_800),
                failed: Some(200),
                forward_solve_seconds: Some(123.5),
                backward_solve_seconds: Some(456.75),
                parallelism: Some(8),
            },
            setup: None,
            production_fit_deviation: None,
            distribution: DistributionInfo {
                backend: "local".to_string(),
                world_size: 1,
                ranks_participated: 1,
                num_nodes: 1,
                threads_per_rank: 1,
                mpi_library: None,
                mpi_standard: None,
                thread_level: None,
                slurm_job_id: None,
                hosts: Vec::new(),
                rank_affinity: Vec::new(),
            },
        }
    }

    fn make_convergence_summary() -> ConvergenceSummary {
        ConvergenceSummary {
            // Distinct from metadata.solve_stats.total_lp_solves (84_000) so the
            // training-summary test proves the count comes from the parquet.
            total_lp_solves: 70_000,
            total_time_ms: 12_345,
            final_lower_bound: 48_500.0,
            final_upper_bound_mean: 49_000.0,
            final_upper_bound_std: 250.0,
            final_gap_percent: Some(1.03),
        }
    }

    fn make_simulation_metadata() -> SimulationMetadata {
        SimulationMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            solver_version: None,
            started_at: "2026-01-17T12:30:00Z".to_string(),
            completed_at: "2026-01-17T12:30:12Z".to_string(),
            duration_seconds: 12.5,
            status: "complete".to_string(),
            scenarios: MetadataScenarios {
                total: 192,
                completed: 192,
                failed: 0,
            },
            cost: Some(MetadataCost {
                mean_cost: 1.0e6,
                std_cost: 2.0e4,
                cvar: 1.2e6,
                cvar_alpha: 0.95,
            }),
            solve_stats: MetadataSimulationSolveStats {
                total_lp_solves: Some(5_000),
                first_try: Some(4_900),
                retried: Some(90),
                failed: Some(10),
                solve_seconds: Some(3.0),
                parallelism: Some(4),
            },
            distribution: DistributionInfo {
                backend: "local".to_string(),
                world_size: 1,
                ranks_participated: 1,
                num_nodes: 1,
                threads_per_rank: 1,
                mpi_library: None,
                mpi_standard: None,
                thread_level: None,
                slurm_job_id: None,
                hosts: Vec::new(),
                rank_affinity: Vec::new(),
            },
        }
    }

    #[test]
    fn summary_args_parses_output_dir() {
        let args = SummaryArgs {
            output_dir: PathBuf::from("/tmp/out"),
        };
        assert_eq!(args.output_dir, PathBuf::from("/tmp/out"));
    }

    #[test]
    fn construct_training_summary_from_metadata() {
        let metadata = make_training_metadata();
        let convergence = make_convergence_summary();

        let summary = build_training_summary(&metadata, &convergence, None);

        assert_eq!(summary.iterations, 42);
        assert!(summary.converged);
        assert_eq!(summary.converged_at, Some(42));
        assert_eq!(summary.reason, "gap_tolerance");
        assert!((summary.lower_bound - 48_500.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound - 49_000.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound_std - 250.0).abs() < f64::EPSILON);
        assert!((summary.gap_percent - 1.03).abs() < 1e-9);
        assert_eq!(summary.total_rows_active, 980_000);
        assert_eq!(summary.total_rows_generated, 1_250_000);
        // From the parquet (70_000), not metadata.solve_stats (84_000).
        assert_eq!(summary.total_lp_solves, 70_000);
        assert_eq!(summary.total_time_ms, 12_345);
        // Solve-stats are sourced from metadata, not the parquet.
        assert_eq!(summary.total_first_try, Some(80_000));
        assert_eq!(summary.total_retried, Some(3_800));
        assert_eq!(summary.total_failed, Some(200));
        assert_eq!(summary.total_forward_solve_seconds, Some(123.5));
        assert_eq!(summary.total_backward_solve_seconds, Some(456.75));
        assert_eq!(summary.parallelism, Some(8));
        // Per-iteration timing is not persisted to metadata.json.
        assert_eq!(summary.forward_phase_wall_seconds, None);
        assert_eq!(summary.backward_phase_wall_seconds, None);
        assert_eq!(summary.forward_wait_seconds, None);
        assert_eq!(summary.backward_wait_seconds, None);
        assert_eq!(summary.serial_lower_bound_seconds, None);
        assert_eq!(summary.serial_cut_selection_seconds, None);
        assert_eq!(summary.serial_cut_sync_seconds, None);
        assert_eq!(summary.serial_allreduce_seconds, None);
        assert_eq!(summary.serial_scheduling_seconds, None);
    }

    #[test]
    fn build_training_summary_bounds_prefer_metadata_when_present() {
        let mut metadata = make_training_metadata();
        metadata.bounds = MetadataBounds {
            final_lower_bound: 48_750.0,
            final_upper_bound: Some(49_500.0),
            final_upper_bound_std: Some(310.0),
        };
        // Convergence carries different values to prove metadata wins.
        let convergence = ConvergenceSummary {
            final_lower_bound: 40_000.0,
            final_upper_bound_mean: 49_000.0,
            final_upper_bound_std: 250.0,
            ..make_convergence_summary()
        };

        let summary = build_training_summary(&metadata, &convergence, None);

        assert!((summary.lower_bound - 48_750.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound - 49_500.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound_std - 310.0).abs() < f64::EPSILON);
    }

    #[test]
    fn build_training_summary_bounds_fall_back_to_parquet_when_metadata_default() {
        let mut metadata = make_training_metadata();
        metadata.bounds = MetadataBounds {
            final_lower_bound: 0.0,
            final_upper_bound: None,
            final_upper_bound_std: None,
        };
        let convergence = ConvergenceSummary {
            final_lower_bound: 48_500.0,
            final_upper_bound_mean: 49_000.0,
            final_upper_bound_std: 250.0,
            ..make_convergence_summary()
        };

        let summary = build_training_summary(&metadata, &convergence, None);

        // A zero lower bound in metadata is treated as "default" and recovers the
        // parquet value, matching the upper-bound fallback for legacy dirs.
        assert!((summary.lower_bound - 48_500.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound - 49_000.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound_std - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn build_simulation_summary_maps_duration_cost_and_solve_stats() {
        let metadata = make_simulation_metadata();

        let summary = build_simulation_summary(&metadata);

        assert_eq!(summary.n_scenarios, 192);
        assert_eq!(summary.completed, 192);
        assert_eq!(summary.failed, 0);
        // duration_seconds = 12.5 -> 12_500 ms.
        assert_eq!(summary.total_time_ms, 12_500);
        assert_eq!(summary.mean_cost, Some(1.0e6));
        assert_eq!(summary.std_cost, Some(2.0e4));
        assert_eq!(summary.total_lp_solves, Some(5_000));
        assert_eq!(summary.total_first_try, Some(4_900));
        assert_eq!(summary.total_retried, Some(90));
        assert_eq!(summary.total_failed_solves, Some(10));
        assert_eq!(summary.total_solve_time_seconds, Some(3.0));
        assert_eq!(summary.parallelism, Some(4));
    }

    #[test]
    fn build_simulation_summary_cost_none_yields_none_mean_std() {
        let mut metadata = make_simulation_metadata();
        metadata.cost = None;

        let summary = build_simulation_summary(&metadata);

        assert!(summary.mean_cost.is_none());
        assert!(summary.std_cost.is_none());
    }

    #[test]
    fn convergence_fallback_uses_metadata_gap_percent() {
        let metadata = make_training_metadata();
        let fallback = convergence_fallback(&metadata);

        assert_eq!(fallback.total_lp_solves, 0);
        assert_eq!(fallback.total_time_ms, 0);
        assert_eq!(fallback.final_gap_percent, Some(0.45));
    }

    #[test]
    fn convergence_fallback_gap_none_when_metadata_has_no_gap() {
        let mut metadata = make_training_metadata();
        metadata.convergence.final_gap_percent = None;

        let fallback = convergence_fallback(&metadata);

        assert!(fallback.final_gap_percent.is_none());
    }

    #[test]
    fn build_training_summary_gap_defaults_to_zero_when_none() {
        let metadata = make_training_metadata();
        let convergence = ConvergenceSummary {
            final_gap_percent: None,
            ..make_convergence_summary()
        };

        let summary = build_training_summary(&metadata, &convergence, None);

        assert!(summary.gap_percent.abs() < f64::EPSILON);
    }

    #[test]
    fn build_training_summary_converged_at_none_when_metadata_has_none() {
        let mut metadata = make_training_metadata();
        metadata.iterations.converged_at = None;
        metadata.convergence.achieved = false;

        let convergence = make_convergence_summary();
        let summary = build_training_summary(&metadata, &convergence, None);

        assert!(summary.converged_at.is_none());
        assert!(!summary.converged);
    }

    // ── reconstruct_topology ──────────────────────────────────────────────────

    #[test]
    fn reconstruct_topology_local_single_host_has_no_mpi_or_slurm() {
        let dist = DistributionInfo {
            backend: "local".to_string(),
            world_size: 1,
            ranks_participated: 1,
            num_nodes: 1,
            threads_per_rank: 1,
            mpi_library: None,
            mpi_standard: None,
            thread_level: None,
            slurm_job_id: None,
            hosts: vec![HostLayout {
                hostname: "h".to_string(),
                ranks: vec![0],
            }],
            rank_affinity: Vec::new(),
        };

        let topology = reconstruct_topology(&dist);

        assert_eq!(topology.backend, BackendKind::Local);
        assert_eq!(topology.world_size, 1);
        assert_eq!(topology.num_hosts(), 1);
        assert_eq!(topology.hosts[0].ranks, vec![0_usize]);
        assert!(topology.mpi.is_none());
        assert!(topology.slurm.is_none());
    }

    #[test]
    fn reconstruct_topology_multi_host_mpi_with_slurm() {
        let dist = DistributionInfo {
            backend: "mpi".to_string(),
            world_size: 8,
            ranks_participated: 8,
            num_nodes: 2,
            threads_per_rank: 4,
            mpi_library: Some("Open MPI v4.1.6".to_string()),
            mpi_standard: Some("MPI 4.0".to_string()),
            thread_level: Some("Funneled".to_string()),
            slurm_job_id: Some("123".to_string()),
            hosts: vec![
                HostLayout {
                    hostname: "node-a".to_string(),
                    ranks: vec![0, 1, 2, 3],
                },
                HostLayout {
                    hostname: "node-b".to_string(),
                    ranks: vec![4, 5, 6, 7],
                },
            ],
            rank_affinity: Vec::new(),
        };

        let topology = reconstruct_topology(&dist);

        assert_eq!(topology.backend, BackendKind::Mpi);
        assert_eq!(topology.num_hosts(), 2);

        let mpi = topology.mpi.expect("mpi metadata should be present");
        assert_eq!(mpi.library_version, "Open MPI v4.1.6");
        assert_eq!(mpi.standard_version, "MPI 4.0");
        assert_eq!(mpi.thread_level, "Funneled");

        let slurm = topology.slurm.expect("slurm metadata should be present");
        assert_eq!(slurm.job_id, "123");
        assert!(slurm.node_list.is_none());
        assert!(slurm.cpus_per_task.is_none());

        assert_eq!(topology.hosts[0].hostname, "node-a");
        assert_eq!(topology.hosts[0].ranks, vec![0_usize, 1, 2, 3]);
        assert_eq!(topology.hosts[1].hostname, "node-b");
        assert_eq!(topology.hosts[1].ranks, vec![4_usize, 5, 6, 7]);
    }

    #[test]
    fn reconstruct_topology_partial_mpi_strings_yield_no_mpi() {
        let mut dist = DistributionInfo {
            backend: "mpi".to_string(),
            world_size: 2,
            ranks_participated: 2,
            num_nodes: 1,
            threads_per_rank: 1,
            mpi_library: Some("Open MPI v4.1.6".to_string()),
            mpi_standard: Some("MPI 4.0".to_string()),
            thread_level: None,
            slurm_job_id: None,
            hosts: Vec::new(),
            rank_affinity: Vec::new(),
        };

        // Missing thread_level alone leaves mpi as None.
        assert!(reconstruct_topology(&dist).mpi.is_none());

        // Restoring all three flips it to Some.
        dist.thread_level = Some("Funneled".to_string());
        assert!(reconstruct_topology(&dist).mpi.is_some());
    }

    #[test]
    fn reconstruct_topology_backend_string_mapping() {
        let base = DistributionInfo {
            backend: "local".to_string(),
            world_size: 1,
            ranks_participated: 1,
            num_nodes: 1,
            threads_per_rank: 1,
            mpi_library: None,
            mpi_standard: None,
            thread_level: None,
            slurm_job_id: None,
            hosts: Vec::new(),
            rank_affinity: Vec::new(),
        };

        let mut mpi = base.clone();
        mpi.backend = "mpi".to_string();
        assert_eq!(reconstruct_topology(&mpi).backend, BackendKind::Mpi);

        assert_eq!(reconstruct_topology(&base).backend, BackendKind::Local);

        let mut other = base;
        other.backend = "something-else".to_string();
        assert_eq!(reconstruct_topology(&other).backend, BackendKind::Auto);
    }

    // ── read_optional_sidecar ─────────────────────────────────────────────────

    #[test]
    fn read_optional_sidecar_not_found_yields_none() {
        let not_found: Result<u32, OutputError> = Err(OutputError::IoError {
            path: PathBuf::from("missing.json"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        });

        let mapped = read_optional_sidecar(not_found).expect("NotFound must degrade gracefully");
        assert!(mapped.is_none());
    }

    #[test]
    fn read_optional_sidecar_present_yields_some() {
        let ok: Result<u32, OutputError> = Ok(7);
        let mapped = read_optional_sidecar(ok).expect("Ok must pass through");
        assert_eq!(mapped, Some(7));
    }

    #[test]
    fn read_optional_sidecar_malformed_propagates_error() {
        // A present-but-corrupt file surfaces as ManifestError, which must NOT be
        // swallowed as a skipped section.
        let malformed: Result<u32, OutputError> = Err(OutputError::ManifestError {
            manifest_type: "model_provenance".to_string(),
            message: "expected value at line 1".to_string(),
        });

        let result = read_optional_sidecar(malformed);
        assert!(matches!(result, Err(CliError::Internal { .. })));
    }

    #[test]
    fn read_optional_sidecar_non_not_found_io_propagates_error() {
        // A non-NotFound IoError (e.g. permission denied) is a real error.
        let denied: Result<u32, OutputError> = Err(OutputError::IoError {
            path: PathBuf::from("locked.json"),
            source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        });

        let result = read_optional_sidecar(denied);
        assert!(matches!(result, Err(CliError::Io { .. })));
    }
}
