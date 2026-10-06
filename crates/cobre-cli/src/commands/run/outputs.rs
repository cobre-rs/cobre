//! Output-writing phase for `cobre run`.
//!
//! Python-parity contract: every file written here must also be written by
//! `cobre-python`'s `run.rs`; the two writers must stay mirrored.

use std::path::Path;

use console::Term;

use cobre_core::System;
use cobre_io::Config;
use cobre_io::EVAPORATION_MODELS_FILE;
use cobre_io::FPHA_DEVIATION_POINTS_FILE;
use cobre_io::FPHA_HYPERPLANES_FILE;
use cobre_io::GENERIC_CONSTRAINT_ECHO_FILE;
use cobre_io::OutputContext;
use cobre_io::SimulationOutput;
use cobre_io::SolverStatsRow;
use cobre_io::TrainingOutput;
use cobre_io::output::simulation_writer::{
    SimulationPathRecord, write_paths, write_scenario_summary,
};
use cobre_io::write_evaporation_models;
use cobre_io::write_fpha_deviation_points;
use cobre_io::write_fpha_hyperplanes;
use cobre_io::write_row_selection_records;
use cobre_io::write_simulation_results;
use cobre_io::write_simulation_solver_stats;
use cobre_io::write_solver_stats;
use cobre_io::write_success_marker;
use cobre_io::write_training_results;
use cobre_sddp::SolverStatsDelta;
use cobre_sddp::StudySetup;
use cobre_sddp::TrainingResult;
use cobre_sddp::build_evaporation_model_rows;
use cobre_sddp::build_fixed_delivery_rows;
use cobre_sddp::build_generic_constraint_echo_rows;
use cobre_sddp::delta_to_stats_row;
use cobre_sddp::policy::orchestration::CheckpointParams;
use cobre_sddp::policy::orchestration::write_checkpoint;
use cobre_sddp::solver_stats_log_to_rows;

use crate::error::CliError;
use crate::summary::print_output_path;

/// Arguments for [`write_training_outputs`].
pub(super) struct WriteTrainingArgs<'a> {
    pub(super) output_dir: &'a Path,
    pub(super) system: &'a System,
    pub(super) config: &'a Config,
    pub(super) training_output: &'a TrainingOutput,
    pub(super) setup: &'a StudySetup,
    pub(super) training_result: &'a TrainingResult,
    pub(super) output_ctx: &'a OutputContext,
    pub(super) quiet: bool,
    pub(super) stderr: &'a Term,
}

/// Write the training artifacts (policy checkpoint, results, sidecars, stats).
pub(super) fn write_training_outputs(args: &WriteTrainingArgs<'_>) -> Result<(), CliError> {
    if !args.quiet {
        use std::io::Write;
        let _ = args.stderr.write_line("Writing training outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    let policy_dir = args.output_dir.join(&args.setup.policy_path);
    write_checkpoint(
        &policy_dir,
        args.setup,
        args.system,
        args.training_result,
        &CheckpointParams {
            max_iterations: args.setup.loop_params.max_iterations,
            forward_passes: args.setup.loop_params.forward_passes,
            seed: args.setup.loop_params.seed,
            export_states: args.config.exports.states,
        },
    )
    .map_err(CliError::from)?;

    write_training_results(
        args.output_dir,
        args.training_output,
        args.system,
        args.config,
        args.output_ctx,
    )
    .map_err(CliError::from)?;

    if !args.setup.hydro_models.fpha_export_rows.is_empty() {
        let fpha_path = args.output_dir.join(FPHA_HYPERPLANES_FILE);
        write_fpha_hyperplanes(&fpha_path, &args.setup.hydro_models.fpha_export_rows)
            .map_err(CliError::from)?;
    }

    // No evaporation-modeled hydro writes no file (FPHA "if-any" behavior).
    let evaporation_rows = build_evaporation_model_rows(&args.setup.hydro_models, args.system);
    if !evaporation_rows.is_empty() {
        let evaporation_path = args.output_dir.join(EVAPORATION_MODELS_FILE);
        write_evaporation_models(&evaporation_path, &evaporation_rows).map_err(CliError::from)?;
    }

    // Off by default, so a default run writes no file and stays byte-identical.
    let deviation_point_rows = args.setup.hydro_models.fpha_deviation_point_rows.as_slice();
    if args.config.exports.fpha_deviation_points && !deviation_point_rows.is_empty() {
        let deviation_points_path = args.output_dir.join(FPHA_DEVIATION_POINTS_FILE);
        write_fpha_deviation_points(&deviation_points_path, deviation_point_rows)
            .map_err(CliError::from)?;
    }

    // No generic constraint writes no file, so a default run stays byte-identical.
    // The writer is called fully qualified, not imported, so the Python-parity
    // checker's `cobre_io::write_*` match sees it.
    if !args.system.generic_constraints().is_empty() {
        let rows = build_generic_constraint_echo_rows(args.setup, args.system);
        let echo_path = args.output_dir.join(GENERIC_CONSTRAINT_ECHO_FILE);
        cobre_io::write_generic_constraint_echo(&echo_path, &rows).map_err(CliError::from)?;
    }

    let fixed_rows = build_fixed_delivery_rows(args.setup, args.system);
    cobre_io::write_fixed_delivery(args.output_dir, &fixed_rows).map_err(CliError::from)?;

    if !args.training_result.solver_stats_log.is_empty() {
        let rows = solver_stats_log_to_rows(&args.training_result.solver_stats_log);
        write_solver_stats(args.output_dir, &rows).map_err(CliError::from)?;
    }

    if !args.training_output.cut_selection_records.is_empty() {
        write_row_selection_records(args.output_dir, &args.training_output.cut_selection_records)
            .map_err(CliError::from)?;
    }

    write_success_marker(&args.output_dir.join("training")).map_err(CliError::from)?;

    if !args.quiet {
        let write_secs = write_start.elapsed().as_secs_f64();
        print_output_path(args.stderr, args.output_dir, write_secs);
    }

    Ok(())
}

/// Arguments for [`write_simulation_outputs`].
pub(super) struct WriteSimulationArgs<'a> {
    pub(super) output_dir: &'a Path,
    pub(super) sim_output: &'a SimulationOutput,
    pub(super) sim_solver_stats: &'a [(u32, SolverStatsDelta)],
    pub(super) sim_path_rows: &'a [SimulationPathRecord],
    /// Gathered per-scenario `(scenario_id, discounted_immediate_cost, probability)`
    /// in canonical scenario-id order; `probability` is `Some` only under a census.
    pub(super) sim_scenario_costs: &'a [(u32, f64, Option<f64>)],
    pub(super) output_ctx: &'a OutputContext,
    pub(super) quiet: bool,
    pub(super) stderr: &'a Term,
}

/// Write the simulation artifacts (results manifest and solver stats).
pub(super) fn write_simulation_outputs(args: &WriteSimulationArgs<'_>) -> Result<(), CliError> {
    if !args.quiet {
        use std::io::Write;
        let _ = args.stderr.write_line("Writing simulation outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    write_simulation_results(args.output_dir, args.sim_output, args.output_ctx)
        .map_err(CliError::from)?;

    // Simulation fills scenario_id (not iteration).
    if !args.sim_solver_stats.is_empty() {
        let rows: Vec<SolverStatsRow> = args
            .sim_solver_stats
            .iter()
            .map(|(scenario_id, delta)| {
                #[allow(clippy::cast_possible_wrap)]
                delta_to_stats_row(
                    None,
                    Some(*scenario_id as i32),
                    "simulation",
                    None,
                    None,
                    None,
                    None,
                    delta,
                )
            })
            .collect();
        write_simulation_solver_stats(args.output_dir, &rows).map_err(CliError::from)?;
    }

    write_paths(args.output_dir, args.sim_path_rows.to_vec()).map_err(CliError::from)?;

    let scenario_summary_rows: Vec<(u32, Option<f64>, f64)> = args
        .sim_scenario_costs
        .iter()
        .map(|&(scenario_id, discounted_immediate_cost, probability)| {
            (scenario_id, probability, discounted_immediate_cost)
        })
        .collect();
    write_scenario_summary(args.output_dir, &scenario_summary_rows).map_err(CliError::from)?;

    write_success_marker(&args.output_dir.join("simulation")).map_err(CliError::from)?;

    if !args.quiet {
        let write_secs = write_start.elapsed().as_secs_f64();
        print_output_path(args.stderr, args.output_dir, write_secs);
    }

    Ok(())
}
