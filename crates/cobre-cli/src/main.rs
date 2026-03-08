//! # cobre
//!
//! Command-line interface for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! Provides commands for running SDDP studies, validating input data,
//! and inspecting results from the terminal.
//!
//! ## Subcommands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `cobre run <CASE_DIR>` | Load, train, simulate, and write results |
//! | `cobre validate <CASE_DIR>` | Validate a case directory |
//! | `cobre report <RESULTS_DIR>` | Query results and print to stdout |
//! | `cobre version` | Print version and build information |

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod commands;
mod error;
mod logging;

use clap::{Parser, Subcommand};

use commands::{
    report::{self, ReportArgs},
    run::{self, RunArgs},
    validate::{self, ValidateArgs},
    version,
};

/// Open infrastructure for power system computation.
#[derive(Debug, Parser)]
#[command(
    name = "cobre",
    about = "Open infrastructure for power system computation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Top-level subcommands for the `cobre` binary.
#[derive(Debug, Subcommand)]
enum Command {
    /// Load a case directory, train an SDDP policy, and run simulation.
    Run(RunArgs),
    /// Validate a case directory and print a structured diagnostic report.
    Validate(ValidateArgs),
    /// Query results from a completed run and print them to stdout.
    Report(ReportArgs),
    /// Print version, solver backend, and build information.
    Version,
}

fn main() {
    let cli = Cli::parse();

    // Extract the verbose flag before dispatching so logging is initialized
    // as early as possible. Only `run` exposes `--verbose`; all other
    // subcommands default to non-verbose logging.
    let verbose = matches!(&cli.command, Command::Run(args) if args.verbose);
    logging::init_logging(verbose);

    let result = match cli.command {
        Command::Run(args) => run::execute(args),
        Command::Validate(args) => validate::execute(args),
        Command::Report(args) => report::execute(args),
        Command::Version => version::execute(),
    };

    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            e.format_error(&console::Term::stderr());
            std::process::exit(e.exit_code());
        }
    }
}
