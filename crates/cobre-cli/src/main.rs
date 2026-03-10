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

use clap::{Parser, Subcommand, ValueEnum};

use commands::{
    init::{self, InitArgs},
    report::{self, ReportArgs},
    run::{self, RunArgs},
    validate::{self, ValidateArgs},
    version,
};

/// Controls when ANSI color/style escapes are emitted on stderr.
///
/// Follows the same env-var override pattern as `--threads` / `COBRE_THREADS`.
/// Resolution order (highest to lowest priority):
///
/// 1. `--color <WHEN>` CLI flag
/// 2. `COBRE_COLOR` environment variable (`always` | `never`; invalid values ignored)
/// 3. `FORCE_COLOR=1` environment variable (forces color on; see <https://force-color.org>)
/// 4. Console auto-detection (the `console` crate checks whether stderr is a TTY)
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ColorWhen {
    /// Enable color when stderr is connected to a TTY (default).
    Auto,
    /// Always emit ANSI color escapes, even when stderr is not a TTY.
    Always,
    /// Never emit ANSI color escapes.
    Never,
}

/// Apply the resolved color setting to the `console` crate's global stderr flag.
///
/// Must be called before any output is written to stderr so that the banner,
/// progress bars, and error messages all honour the chosen setting.
pub(crate) fn resolve_color(cli_color: ColorWhen) {
    match cli_color {
        ColorWhen::Always => console::set_colors_enabled_stderr(true),
        ColorWhen::Never => console::set_colors_enabled_stderr(false),
        ColorWhen::Auto => {
            if let Ok(val) = std::env::var("COBRE_COLOR") {
                match val.to_ascii_lowercase().as_str() {
                    "always" => console::set_colors_enabled_stderr(true),
                    "never" => console::set_colors_enabled_stderr(false),
                    _ => {} // Invalid values are silently ignored, auto-detection applies.
                }
            } else if std::env::var_os("FORCE_COLOR").is_some() {
                console::set_colors_enabled_stderr(true);
            }
        }
    }
}

/// Open infrastructure for power system computation.
#[derive(Debug, Parser)]
#[command(
    name = "cobre",
    about = "Open infrastructure for power system computation"
)]
struct Cli {
    /// Control ANSI color output on stderr.
    ///
    /// `always` forces color on (useful under `mpiexec` which pipes stderr through
    /// a non-TTY). `never` disables all color. `auto` lets the terminal detection
    /// decide. Also honoured via the `COBRE_COLOR` env var (flag takes precedence).
    #[arg(long, global = true, default_value = "auto")]
    color: ColorWhen,

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

    // Apply color setting before any output is written so that the banner,
    // progress bars, and error messages all honour the chosen setting.
    resolve_color(cli.color);

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

#[cfg(test)]
mod tests {
    use super::{ColorWhen, resolve_color};

    // Unit tests cover only `Always` and `Never` variants (safe without env var mutation).
    // Environment variable tests are in the integration suite (cli_color.rs).

    /// `ColorWhen::Always` forces color on regardless of TTY state.
    #[test]
    fn test_resolve_color_always_enables_color() {
        resolve_color(ColorWhen::Always);
        assert!(
            console::colors_enabled_stderr(),
            "Always must set colors_enabled_stderr to true"
        );
    }

    /// `ColorWhen::Never` forces color off regardless of TTY state.
    #[test]
    fn test_resolve_color_never_disables_color() {
        resolve_color(ColorWhen::Never);
        assert!(
            !console::colors_enabled_stderr(),
            "Never must set colors_enabled_stderr to false"
        );
    }
}
