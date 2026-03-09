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

mod banner;
mod commands;
mod error;
mod logging;
mod policy_io;
mod progress;
mod simulation_io;
mod summary;
mod templates;

use clap::{Parser, Subcommand};

use commands::{
    init::{self, InitArgs},
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
    /// Scaffold a new case directory from an embedded template.
    Init(InitArgs),
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
    let verbose = matches!(&cli.command, Command::Run(args) if args.verbose);
    logging::init_logging(verbose);

    let result = match cli.command {
        Command::Init(args) => init::execute(args),
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
