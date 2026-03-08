//! `cobre version` subcommand.
//!
//! Prints the binary version, active solver backend, and build profile.
//! Useful for bug reports and HPC environment debugging.

use crate::error::CliError;

/// Print version, solver, and build information to stdout.
///
/// # Errors
///
/// This function is currently infallible; the `Result` return type is kept
/// consistent with all other subcommand execute functions for uniform dispatch.
#[allow(clippy::unnecessary_wraps)]
pub fn execute() -> Result<(), CliError> {
    let version = env!("CARGO_PKG_VERSION");
    let build = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    println!("cobre {version}");
    println!("solver: HiGHS");
    println!("build:  {build}");

    Ok(())
}
