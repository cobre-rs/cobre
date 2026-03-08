//! `cobre validate <CASE_DIR>` subcommand.
//!
//! Runs the 5-layer validation pipeline and prints a structured diagnostic
//! report. No banner or progress bar — the output is the deliverable.
//! The actual implementation is in Epic 02; this module is the stub.

use std::path::PathBuf;

use clap::Args;

use crate::error::CliError;

/// Arguments for the `cobre validate` subcommand.
#[derive(Debug, Args)]
#[command(about = "Validate a case directory and print a structured diagnostic report")]
pub struct ValidateArgs {
    /// Path to the case directory to validate.
    pub case_dir: PathBuf,
}

/// Execute the `validate` subcommand.
///
/// # Errors
///
/// Returns [`CliError::Validation`] when the case directory fails the
/// validation pipeline, or [`CliError::Io`] on filesystem errors.
// Stubs: args will be consumed and Ok(()) will be fallible in the real impl.
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
pub fn execute(args: ValidateArgs) -> Result<(), CliError> {
    eprintln!(
        "validate: not yet implemented (case_dir={})",
        args.case_dir.display()
    );
    Ok(())
}
