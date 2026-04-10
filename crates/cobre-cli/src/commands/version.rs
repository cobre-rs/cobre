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
    println!("cobre   v{version}");
    println!("solver: HiGHS {}", cobre_solver::highs_version());
    if cfg!(feature = "mpi") {
        println!("comm:   mpi");
    } else {
        println!("comm:   local");
    }
    println!("zstd:   enabled");
    println!(
        "arch:   {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    if cfg!(debug_assertions) {
        println!("build:  debug");
    } else {
        println!("build:  release (lto=thin)");
    }

    Ok(())
}
