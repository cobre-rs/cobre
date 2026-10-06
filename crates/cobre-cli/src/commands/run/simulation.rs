//! Simulation phase for `cobre run`.

use std::path::Path;
use std::sync::mpsc;

use console::Term;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::{System, TrainingEvent};
use cobre_io::MetadataCost;
use cobre_io::MetadataSimulationSolveStats;
use cobre_io::OutputContext;
use cobre_io::SimulationOutput;
use cobre_io::now_iso8601;
use cobre_io::output::simulation_writer::ScenarioWritePayload;
use cobre_io::output::simulation_writer::SimulationParquetWriter;
use cobre_io::output::simulation_writer::SimulationPathRecord;
use cobre_io::write_skipped_simulation_results;
use cobre_io::write_success_marker;
use cobre_sddp::SOLVER_STATS_DELTA_SCALAR_FIELDS;
use cobre_sddp::SimulationWeighting;
use cobre_sddp::SolverStatsDelta;
use cobre_sddp::StudySetup;
use cobre_sddp::TrainingResult;
use cobre_sddp::aggregate_simulation;
use cobre_sddp::pack_delta_scalars;
use cobre_sddp::pack_scenario_stats;
use cobre_sddp::reconcile_global_ok;
use cobre_sddp::unpack_delta_scalars;
use cobre_sddp::unpack_scenario_stats;
use cobre_solver::ActiveSolver;
use cobre_solver::active_solver_metadata_id;

use crate::error::CliError;
use crate::summary::SimulationSummary;

use super::outputs::{WriteSimulationArgs, write_simulation_outputs};
use super::{RunContext, build_distribution_info, check_stats_overflow};
use crate::progress::run_progress_thread;
use crate::summary::print_simulation_summary;

/// Run the simulation phase: workspace pool, Parquet writing, and output.
pub(super) fn run_simulation_phase(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    training_result: &TrainingResult,
    hostname: &str,
) -> Result<(), CliError> {
    let solver_factory = ActiveSolver::new;
    let n_scenarios = setup.simulation_config.n_scenarios;
    let sim_config = setup.simulation_config();

    let mut sim_pool = setup
        .create_workspace_pool(&ctx.comm, ctx.n_threads, solver_factory)
        .map_err(|e| CliError::Solver {
            message: format!(
                "{} initialisation failed for simulation pool: {e}",
                cobre_solver::active_solver_name()
            ),
        })?;

    let (sim_event_tx, sim_event_rx) = mpsc::channel::<TrainingEvent>();
    let sim_progress_handle = if ctx.quiet {
        drop(sim_event_rx);
        None
    } else {
        Some(run_progress_thread(
            sim_event_rx,
            ctx.render_mode,
            u64::from(n_scenarios),
            ctx.term_width,
        ))
    };

    let io_capacity = sim_config.io_channel_capacity;
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

    let mut sim_writer = SimulationParquetWriter::new(&ctx.output_dir, system)?;

    // Drain straight to Parquet rather than collecting into a Vec and gathering
    // on rank 0 via MPI, which overflows i32 on large cases.
    let drain_handle = std::thread::spawn(move || {
        let mut failed: u32 = 0;
        for scenario_result in result_rx {
            let payload = ScenarioWritePayload::from(scenario_result);
            if let Err(e) = sim_writer.write_scenario(payload) {
                tracing::error!("simulation write error: {e}");
                failed += 1;
            }
        }
        (sim_writer, failed)
    });

    let sim_started_at = now_iso8601();
    let sim_start = std::time::Instant::now();

    let sim_result = setup
        .simulate(
            &mut sim_pool.workspaces,
            &ctx.comm,
            &result_tx,
            Some(sim_event_tx),
            training_result.frozen_templates.as_deref(),
            &training_result.basis_cache,
        )
        .map_err(CliError::from);
    if let Some(handle) = sim_progress_handle {
        let _ = handle.join();
    }

    drop(result_tx);

    let drain_join = drain_handle.join();

    // Reconcile the per-rank simulation outcome BEFORE the first post-sim
    // collective (`merge_simulation_metadata`'s allreduce): `simulate()` is
    // collective-free, so a failure on a strict subset of ranks would otherwise
    // strand every healthy rank in that allreduce while the failing ranks skip it.
    let mut reconcile_scratch = [0_i32];
    let local_ok = drain_join.is_ok() && sim_result.is_ok();
    let global_ok =
        reconcile_global_ok(local_ok, &ctx.comm, &mut reconcile_scratch).map_err(|e| {
            CliError::Internal {
                message: format!("simulation outcome reconcile error: {e}"),
            }
        })?;

    let (sim_writer, write_failures) = drain_join.map_err(|_| CliError::Internal {
        message: "simulation drain thread panicked".to_string(),
    })?;
    let sim_run_result = sim_result?;
    if !global_ok {
        return Err(CliError::Internal {
            message: "a peer rank failed simulation; failing on every rank in lockstep".to_string(),
        });
    }

    #[allow(clippy::cast_possible_truncation)]
    let sim_time_ms = sim_start.elapsed().as_millis() as u64;

    // Grab the node-path rows before `finalize` consumes the writer; they are
    // gathered across ranks below (paths.parquet is one unpartitioned run-level
    // file, so it needs every rank's scenarios).
    let local_path_rows: Vec<SimulationPathRecord> = sim_writer.path_rows().to_vec();

    let mut local_sim_output = sim_writer.finalize(sim_time_ms);
    local_sim_output.failed = write_failures;

    let mut merged_sim_output = merge_simulation_metadata(&ctx.comm, &local_sim_output)?;

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-simulation barrier error: {e}"),
    })?;

    let (global_agg, global_scenario_stats) =
        aggregate_simulation_solver_stats(&ctx.comm, &sim_run_result.solver_stats)?;

    let global_path_rows = aggregate_simulation_paths(&ctx.comm, &local_path_rows)?;

    // Aggregate across all ranks so the printed mean/std/CI95 reflect every
    // scenario, not just rank 0's.
    let weighting = match sim_run_result.census_weights.as_deref() {
        Some(weights) => SimulationWeighting::Census { weights },
        None => SimulationWeighting::Uniform,
    };
    let (cost_summary, gathered_scenario_costs) =
        aggregate_simulation(&sim_run_result.costs, sim_config, &ctx.comm, weighting).map_err(
            |e| CliError::Internal {
                message: format!("simulation cost aggregation error: {e}"),
            },
        )?;

    let parallelism = super::compute_parallelism(ctx.n_threads, ctx.comm.size());

    merged_sim_output.cost = Some(MetadataCost {
        mean_cost: cost_summary.mean_cost,
        std_cost: cost_summary.std_cost,
    });
    merged_sim_output.solve_stats = MetadataSimulationSolveStats {
        total_lp_solves: Some(global_agg.lp_solves),
        first_try: Some(global_agg.first_try_successes),
        retried: Some(
            global_agg
                .lp_successes
                .saturating_sub(global_agg.first_try_successes),
        ),
        failed: Some(global_agg.lp_failures),
        solve_seconds: Some(global_agg.solve_time_ms / 1000.0),
        parallelism: Some(parallelism),
    };

    if !ctx.quiet && ctx.is_root {
        print_sim_summary(
            &ctx.stderr,
            n_scenarios,
            sim_time_ms,
            &global_agg,
            &cost_summary,
            parallelism,
        );
    }

    if ctx.is_root {
        write_sim_outputs_on_root(
            ctx,
            hostname,
            sim_started_at,
            &merged_sim_output,
            &global_scenario_stats,
            &global_path_rows,
            &gathered_scenario_costs,
        )?;
    }

    Ok(())
}

/// Write simulation output files on rank 0.
fn write_sim_outputs_on_root(
    ctx: &RunContext<impl Communicator>,
    hostname: &str,
    sim_started_at: String,
    merged_sim_output: &SimulationOutput,
    global_scenario_stats: &[(u32, SolverStatsDelta)],
    global_path_rows: &[SimulationPathRecord],
    gathered_scenario_costs: &[(u32, f64, Option<f64>)],
) -> Result<(), CliError> {
    write_simulation_outputs(&WriteSimulationArgs {
        output_dir: &ctx.output_dir,
        sim_output: merged_sim_output,
        sim_solver_stats: global_scenario_stats,
        sim_path_rows: global_path_rows,
        sim_scenario_costs: gathered_scenario_costs,
        output_ctx: &simulation_output_context(ctx, hostname, sim_started_at),
        quiet: ctx.quiet,
        stderr: &ctx.stderr,
    })
}

fn simulation_output_context(
    ctx: &RunContext<impl Communicator>,
    hostname: &str,
    started_at: String,
) -> OutputContext {
    let mpi_world_size = u32::try_from(ctx.topology.world_size).unwrap_or(u32::MAX);
    OutputContext {
        hostname: hostname.to_string(),
        solver: active_solver_metadata_id().to_string(),
        solver_version: Some(ctx.solver_version.clone()),
        started_at,
        completed_at: now_iso8601(),
        distribution: build_distribution_info(&ctx.topology, ctx.n_threads, mpi_world_size),
        setup: None,
        production_fit_deviation: None,
    }
}

/// Write the skipped simulation's outputs on rank 0, in place of
/// [`run_simulation_phase`]; no rank runs a scenario.
pub(super) fn skip_simulation_phase(
    ctx: &RunContext<impl Communicator>,
    hostname: &str,
    n_scenarios: u32,
) -> Result<(), CliError> {
    let sim_ctx = simulation_output_context(ctx, hostname, now_iso8601());
    write_skipped_simulation_outputs(&ctx.output_dir, n_scenarios, &sim_ctx)
}

fn write_skipped_simulation_outputs(
    output_dir: &Path,
    n_scenarios: u32,
    output_ctx: &OutputContext,
) -> Result<(), CliError> {
    write_skipped_simulation_results(output_dir, n_scenarios, output_ctx)
        .map_err(CliError::from)?;
    write_success_marker(&output_dir.join("simulation")).map_err(CliError::from)
}

/// Print the simulation summary from aggregated solver stats and cost statistics.
fn print_sim_summary(
    stderr: &Term,
    n_scenarios: u32,
    sim_time_ms: u64,
    agg: &SolverStatsDelta,
    cost_summary: &cobre_sddp::SimulationSummary,
    parallelism: u32,
) {
    print_simulation_summary(
        stderr,
        &SimulationSummary {
            n_scenarios,
            completed: n_scenarios,
            failed: 0,
            total_time_ms: sim_time_ms,
            mean_cost: Some(cost_summary.mean_cost),
            std_cost: Some(cost_summary.std_cost),
            total_lp_solves: agg.lp_solves,
            total_first_try: agg.first_try_successes,
            total_retried: agg.lp_successes.saturating_sub(agg.first_try_successes),
            total_failed_solves: agg.lp_failures,
            total_solve_time_seconds: agg.solve_time_ms / 1000.0,
            parallelism,
        },
    );
}

/// Merge each rank's local [`SimulationOutput`](cobre_io::SimulationOutput) via
/// MPI collectives.
fn merge_simulation_metadata<C: Communicator>(
    comm: &C,
    local: &SimulationOutput,
) -> Result<SimulationOutput, CliError> {
    let send_counts = [local.n_scenarios, local.completed, local.failed];
    let mut merged_counts = [0u32; 3];
    comm.allreduce(&send_counts, &mut merged_counts, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("simulation metadata count allreduce error: {e}"),
        })?;

    // Max, not Sum: wall-clock is the slowest rank's time, not the total.
    let send_time = [local.total_time_ms];
    let mut merged_time = [0u64; 1];
    comm.allreduce(&send_time, &mut merged_time, ReduceOp::Max)
        .map_err(|e| CliError::Internal {
            message: format!("simulation metadata time allreduce error: {e}"),
        })?;

    Ok(SimulationOutput {
        n_scenarios: merged_counts[0],
        completed: merged_counts[1],
        failed: merged_counts[2],
        total_time_ms: merged_time[0],
        cost: None,
        solve_stats: MetadataSimulationSolveStats::default(),
    })
}

// Rationale: per-rank buffer lengths travel as u64 on the wire and are far below usize::MAX on
// every supported target, so the u64 -> usize narrowing is exact.
#[allow(clippy::cast_possible_truncation)]
fn exchange_gather_plan<C: Communicator>(
    comm: &C,
    local_len: usize,
    context: &str,
) -> Result<(Vec<usize>, Vec<usize>), CliError> {
    let n_ranks = comm.size();
    let send_len = [local_len as u64];
    let mut all_lens = vec![0u64; n_ranks];
    let len_counts: Vec<usize> = vec![1; n_ranks];
    let len_displs: Vec<usize> = (0..n_ranks).collect();
    comm.allgatherv(&send_len, &mut all_lens, &len_counts, &len_displs)
        .map_err(|e| CliError::Internal {
            message: format!("{context} length exchange error: {e}"),
        })?;

    let recv_counts: Vec<usize> = all_lens.iter().map(|&l| l as usize).collect();
    let recv_displs: Vec<usize> = recv_counts
        .iter()
        .scan(0usize, |acc, &c| {
            let d = *acc;
            *acc += c;
            Some(d)
        })
        .collect();
    Ok((recv_counts, recv_displs))
}

/// Gather every rank's `(scenario_id, stage_id, node_id)` path rows for the
/// run-level, unpartitioned `paths.parquet`. Order is irrelevant: `write_paths`
/// fixes the canonical `(scenario_id, stage_id)` order (rank-invariance contract).
fn aggregate_simulation_paths<C: Communicator>(
    comm: &C,
    local: &[SimulationPathRecord],
) -> Result<Vec<SimulationPathRecord>, CliError> {
    let mut local_buf: Vec<i32> = Vec::with_capacity(local.len() * 3);
    for r in local {
        local_buf.push(r.scenario_id);
        local_buf.push(r.stage_id);
        local_buf.push(r.node_id);
    }

    let (recv_counts, recv_displs) =
        exchange_gather_plan(comm, local_buf.len(), "simulation path")?;
    let total: usize = recv_counts.iter().sum();
    let mut all_buf = vec![0i32; total];
    comm.allgatherv(&local_buf, &mut all_buf, &recv_counts, &recv_displs)
        .map_err(|e| CliError::Internal {
            message: format!("simulation path gather error: {e}"),
        })?;

    Ok(all_buf
        .chunks_exact(3)
        .map(|c| SimulationPathRecord {
            scenario_id: c[0],
            stage_id: c[1],
            node_id: c[2],
        })
        .collect())
}

/// Aggregate simulation solver statistics across all MPI ranks.
///
/// Returns the global [`cobre_sddp::SolverStatsDelta`] (sum over all ranks, for
/// the root summary) and a per-global-scenario `Vec`, sorted by scenario ID for
/// deterministic Parquet output.
fn aggregate_simulation_solver_stats<C: Communicator>(
    comm: &C,
    local_stats: &[(u32, i32, SolverStatsDelta)],
) -> Result<(SolverStatsDelta, Vec<(u32, SolverStatsDelta)>), CliError> {
    let local_agg = SolverStatsDelta::aggregate(local_stats.iter().map(|(_, _, d)| d));
    check_stats_overflow(&local_agg)?;
    let send_scalars = pack_delta_scalars(&local_agg);
    let mut recv_scalars = [0.0_f64; SOLVER_STATS_DELTA_SCALAR_FIELDS];
    comm.allreduce(&send_scalars, &mut recv_scalars, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("simulation solver stats allreduce error: {e}"),
        })?;
    let global_agg = unpack_delta_scalars(&recv_scalars);

    // Strip the opening field (always -1 here): the MPI wire format omits it.
    let local_stats_stripped: Vec<(u32, SolverStatsDelta)> = local_stats
        .iter()
        .map(|(id, _opening, delta)| (*id, delta.clone()))
        .collect();
    let local_buf = pack_scenario_stats(&local_stats_stripped);

    let (recv_counts, recv_displs) =
        exchange_gather_plan(comm, local_buf.len(), "simulation solver stats")?;
    let total_floats: usize = recv_counts.iter().sum();
    let mut all_buf = vec![0.0_f64; total_floats];
    comm.allgatherv(&local_buf, &mut all_buf, &recv_counts, &recv_displs)
        .map_err(|e| CliError::Internal {
            message: format!("simulation solver stats gather error: {e}"),
        })?;

    let mut global_scenario_stats = unpack_scenario_stats(&all_buf);
    global_scenario_stats.sort_by_key(|(id, _)| *id);

    Ok((global_agg, global_scenario_stats))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::any::Any;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use console::Term;
    use tempfile::TempDir;

    use cobre_comm::{
        BackendKind, CommData, CommError, Communicator, ExecutionTopology, HostInfo, LocalBackend,
        ReduceOp,
    };
    use cobre_io::{
        DistributionInfo, HostLayout, OutputContext, RunStatus, read_simulation_metadata,
    };
    use cobre_sddp::SimulationWeighting;
    use cobre_sddp::setup::{
        NodeGraph, NodeId, NodeOpenings, NodeRuntime, OpeningSource, StageIdx, Traversal,
    };

    use super::{run_simulation_phase, write_skipped_simulation_outputs};
    use crate::commands::run::setup::broadcast_and_build_setup;
    use crate::commands::run::training::run_training_phase;
    use crate::commands::run::{CommBackendArg, RunArgs, RunContext};
    use crate::error::CliError;
    use crate::progress::RenderMode;

    #[derive(Default)]
    struct PeerFailedComm {
        collective_calls: AtomicUsize,
    }

    impl PeerFailedComm {
        fn collective_calls(&self) -> usize {
            self.collective_calls.load(Ordering::SeqCst)
        }

        fn refuse(&self, operation: &'static str) -> CommError {
            self.collective_calls.fetch_add(1, Ordering::SeqCst);
            CommError::CollectiveFailed {
                operation,
                mpi_error_code: 0,
                message: "only the reconcile flag is answered".to_string(),
            }
        }
    }

    impl Communicator for PeerFailedComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            Err(self.refuse("allgatherv"))
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            if op == ReduceOp::Max
                && send.len() == 1
                && let Some(flag) = recv
                    .first_mut()
                    .and_then(|r| (r as &mut dyn Any).downcast_mut::<i32>())
            {
                self.collective_calls.fetch_add(1, Ordering::SeqCst);
                *flag = 1;
                return Ok(());
            }
            Err(self.refuse("allreduce"))
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Err(self.refuse("broadcast"))
        }

        fn barrier(&self) -> Result<(), CommError> {
            Err(self.refuse("barrier"))
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            panic!("PeerFailedComm::abort({error_code})")
        }
    }

    fn test_run_context<C: Communicator>(
        comm: C,
        world_size: usize,
        case_dir: &Path,
        output_dir: &Path,
    ) -> RunContext<C> {
        RunContext {
            comm,
            is_root: true,
            quiet: true,
            n_threads: 1,
            output_dir: output_dir.to_path_buf(),
            case_dir: case_dir.to_path_buf(),
            term_width: 80,
            stderr: Term::stderr(),
            render_mode: RenderMode::auto(),
            topology: ExecutionTopology {
                backend: BackendKind::Local,
                world_size,
                hosts: vec![HostInfo {
                    hostname: "localhost".to_string(),
                    ranks: (0..world_size).collect(),
                }],
                mpi: None,
                slurm: None,
            },
            solver_version: String::new(),
        }
    }

    #[test]
    fn simulation_writes_no_marker_when_a_peer_rank_fails() {
        let case_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy");
        let output = TempDir::new().expect("output tempdir must be creatable");
        let output_dir = output.path().to_path_buf();
        let args = RunArgs {
            case_dir: case_dir.clone(),
            output: Some(output_dir.clone()),
            quiet: true,
            threads: Some(1),
            comm_backend: CommBackendArg::Local,
        };

        let local = test_run_context(LocalBackend, 1, &case_dir, &output_dir);
        let mut loaded = broadcast_and_build_setup(&local, &args)
            .expect("1dtoy must load and build its study setup");
        let training =
            run_training_phase(&local, &mut loaded.setup, &Arc::new(AtomicUsize::new(0)))
                .expect("1dtoy must train under the local backend");
        assert!(
            training.error.is_none(),
            "1dtoy training must finish without a mid-iteration error: {:?}",
            training.error
        );

        let peer = test_run_context(PeerFailedComm::default(), 2, &case_dir, &output_dir);
        let outcome = run_simulation_phase(
            &peer,
            &loaded.system,
            &mut loaded.setup,
            &training.result,
            "localhost",
        );

        match outcome {
            Err(CliError::Internal { message }) => assert!(
                message.contains("a peer rank failed simulation"),
                "expected the peer-failure lockstep error, got: {message}"
            ),
            other => panic!("expected the peer-failure lockstep error, got {other:?}"),
        }
        assert_eq!(
            peer.comm.collective_calls(),
            1,
            "the reconcile must be the only collective entered"
        );

        let sim_dir = output_dir.join("simulation");
        assert!(
            !sim_dir.join("_SUCCESS").exists(),
            "a peer failure must leave no simulation/_SUCCESS"
        );
        assert!(
            !sim_dir.join("metadata.json").exists(),
            "a peer failure must leave no simulation/metadata.json"
        );
        assert!(
            sim_dir
                .join("costs/scenario_id=0000/data.parquet")
                .is_file(),
            "rank 0 must have written its own partitions before the reconcile"
        );
    }

    #[test]
    fn skipped_simulation_outputs_write_partial_metadata_then_the_marker() {
        let output = TempDir::new().expect("output tempdir must be creatable");
        let output_ctx = OutputContext {
            hostname: "localhost".to_string(),
            solver: "highs".to_string(),
            solver_version: None,
            started_at: "2026-01-01T00:00:00Z".to_string(),
            completed_at: "2026-01-01T00:00:00Z".to_string(),
            distribution: DistributionInfo {
                backend: "local".to_string(),
                world_size: 1,
                ranks_participated: 1,
                num_hosts: 1,
                threads_per_rank: 1,
                mpi_library: None,
                mpi_standard: None,
                thread_level: None,
                slurm_job_id: None,
                hosts: vec![HostLayout {
                    hostname: "localhost".to_string(),
                    ranks: vec![0],
                }],
            },
            setup: None,
            production_fit_deviation: None,
        };

        write_skipped_simulation_outputs(output.path(), 100, &output_ctx)
            .expect("the skipped-simulation writer must succeed");

        let sim_dir = output.path().join("simulation");
        let metadata = read_simulation_metadata(&sim_dir.join("metadata.json"))
            .expect("simulation/metadata.json must decode");
        assert_eq!(metadata.status, RunStatus::Partial);
        assert_eq!(metadata.scenarios.total, 100);
        assert_eq!(metadata.scenarios.completed, 0);
        assert!(sim_dir.join("_SUCCESS").is_file());
    }

    /// A single-node, single-leaf graph — enough to resolve a `Traversal` in
    /// either axis without a full `StudySetup`.
    fn one_node_graph() -> NodeGraph {
        NodeGraph {
            node_ids: vec![NodeId(0)].into(),
            nodes: vec![NodeRuntime {
                stage: StageIdx(0),
                pool_id: 0,
                openings: NodeOpenings {
                    source: OpeningSource::Generated,
                    offset: 0,
                    len: 1,
                    q: 1.0,
                },
            }]
            .into(),
            successors: vec![Vec::new()].into(),
            n_pools: 1,
            pool_stage: vec![StageIdx(0)],
        }
    }

    /// The CLI derives its simulation weighting from the resolved `Traversal`,
    /// exactly as `run_simulation_phase` does — under a sampled selection this
    /// can only ever resolve to `Uniform`, never `Census`, regardless of what a
    /// caller might otherwise assemble beside it.
    #[test]
    fn simulation_weighting_census_underivable_from_sampled_traversal() {
        let ng = one_node_graph();

        let sampled = Traversal::resolve(&ng, false, 10);
        assert!(matches!(
            sampled.simulation_weighting(),
            SimulationWeighting::Uniform
        ));

        let enumerated = Traversal::resolve(&ng, true, 1);
        assert!(matches!(
            enumerated.simulation_weighting(),
            SimulationWeighting::Census { .. }
        ));
    }
}
