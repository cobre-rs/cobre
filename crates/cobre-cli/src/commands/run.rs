//! `cobre run <CASE_DIR>` subcommand.
//!
//! Executes the full solve lifecycle: load → validate → train → simulate → write.
//! The actual implementation is in Epic 02; this module is the argument-parsing
//! skeleton and a stub entry point.

use std::path::PathBuf;

use clap::Args;

use crate::error::CliError;

/// Arguments for the `cobre run` subcommand.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Args)]
#[command(about = "Load a case directory, train an SDDP policy, and run simulation")]
pub struct RunArgs {
    /// Path to the case directory containing the input data files.
    pub case_dir: PathBuf,

    /// Output directory for results (defaults to `<CASE_DIR>/output/`).
    #[arg(long, value_name = "DIR")]
    pub output: Option<PathBuf>,

    /// Train only; skip the simulation phase.
    #[arg(long)]
    pub skip_simulation: bool,

    /// Suppress the banner and progress bars. Errors still go to stderr.
    #[arg(long)]
    pub quiet: bool,

    /// Suppress the banner but keep progress bars.
    #[arg(long)]
    pub no_banner: bool,

    /// Increase the tracing log level (debug for `cobre_cli`, info for library crates).
    #[arg(long)]
    pub verbose: bool,
}

/// Execute the `run` subcommand.
///
/// # Errors
///
/// Returns [`CliError`] when loading, training, simulation, or I/O fails.
/// The exit code indicates the category of failure.
// These lints are suppressed because the stub signature mirrors the final API:
// args will be consumed in the real implementation, and Result is required for
// consistent dispatch in main.
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
pub fn execute(args: RunArgs) -> Result<(), CliError> {
    eprintln!(
        "run: not yet implemented (case_dir={})",
        args.case_dir.display()
    );
    Ok(())
}
