//! `cobre report <RESULTS_DIR>` subcommand.
//!
//! Queries a completed result set and emits machine-readable JSON to stdout.
//! The actual implementation is in Epic 02; this module is the stub.

use std::path::PathBuf;

use clap::Args;

use crate::error::CliError;

/// Arguments for the `cobre report` subcommand.
#[derive(Debug, Args)]
#[command(about = "Query results from a completed run and print them to stdout")]
pub struct ReportArgs {
    /// Path to the results directory produced by `cobre run`.
    pub results_dir: PathBuf,
}

/// Execute the `report` subcommand.
///
/// # Errors
///
/// Returns [`CliError::Io`] when the results directory cannot be read, or
/// [`CliError::Internal`] on unexpected parsing failures.
// Stubs: args will be consumed and Ok(()) will be fallible in the real impl.
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
pub fn execute(args: ReportArgs) -> Result<(), CliError> {
    eprintln!(
        "report: not yet implemented (results_dir={})",
        args.results_dir.display()
    );
    Ok(())
}
