//! `cobre validate <CASE_DIR>` subcommand.
//!
//! Runs the six-layer validation pipeline followed by the pre-solver
//! preparation phases and prints a structured diagnostic report to stdout —
//! or, with `--json`, a single machine-readable JSON object: the boundary
//! reconciliation outcome on success, or an `error` object naming the first
//! failing phase. Stdout under `--json` is always one such object or empty,
//! never human report text. No banner or progress bar — the output is the
//! deliverable.
//!
//! ## Validation contract
//!
//! If `cobre validate <CASE_DIR>` exits 0, then `cobre run <CASE_DIR>` will not
//! fail in any phase before the solver begins iterating. After the case loads and
//! `config.json` parses, every pre-solver check runs through
//! [`cobre_sddp::validate_phases::validate_study`], the pipeline `cobre.io.validate`
//! shares: it builds the study with the constructor `cobre run` uses, checks the
//! warm-start, resume or simulation-only policy the run would load, and, when
//! `config.policy.boundary` is configured, reconciles the boundary policy against
//! its terminal manifest, without solving. A failure exits by its
//! [`cobre_sddp::ErrorClass`].

use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use clap::Args;
use cobre_io::{LoadError, ReportEntry, validate_case_with_artifacts};
use cobre_sddp::policy::full_fcf_load::FullFcfLoadKind;
use cobre_sddp::validate_phases::{
    PolicyLoadSummary, ValidateFailure, ValidateRequest, validate_study,
};
use cobre_sddp::{BoundaryReconciliation, BoundaryReconciliationReport, ErrorClass};
use console::{Term, style};
use serde::Serialize;

use crate::commands::resolve_output_dir;
use crate::error::CliError;

/// Arguments for the `cobre validate` subcommand.
#[derive(Debug, Args)]
#[command(about = "Validate a case directory and print a structured diagnostic report")]
pub struct ValidateArgs {
    /// Path to the case directory to validate.
    pub case_dir: PathBuf,

    /// Emit the boundary reconciliation outcome as a single JSON object to
    /// stdout instead of the human-readable report.
    #[arg(long)]
    pub json: bool,

    /// Output directory whose policy a configured warm-start, resume or
    /// simulation-only load is checked against (defaults to `<CASE_DIR>/output/`,
    /// as for `cobre run`).
    #[arg(long, value_name = "DIR")]
    pub output: Option<PathBuf>,
}

/// Success outcome (`configured`/`boundary_date`/`report` populated, `error` None) and error outcome (`configured`/`boundary_date`/`report` None, `error` populated) never overlap.
#[derive(Debug, Serialize)]
struct ValidateBoundaryOutput {
    /// Whether `policy.boundary` is configured in this case's `config.json`.
    configured: Option<bool>,
    /// The date the boundary pool was selected against, when `configured`
    /// is `Some(true)`.
    boundary_date: Option<NaiveDate>,
    /// The reconciliation report when `configured` is `Some(true)`.
    report: Option<BoundaryReconciliationReport>,
    /// The policy load `cobre run` would apply, present only when one is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_load: Option<PolicyLoadOutput>,
    /// The failing phase and message, populated only on an early abort.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ValidateErrorOutput>,
}

#[derive(Debug, Serialize)]
struct PolicyLoadOutput {
    mode: &'static str,
    unused_stored_bases: usize,
}

/// Early-abort failure. `phase` is the stable kind string from [`cobre_sddp::validate_phases::PhaseFailure::kind`], [`cobre_sddp::validate_phases::PolicyLoadFailure::kind`] or [`LoadError::kind`] (same string programmatic callers filter on).
#[derive(Debug, Serialize)]
struct ValidateErrorOutput {
    phase: String,
    message: String,
}

impl ValidateBoundaryOutput {
    fn success(
        boundary: Option<&BoundaryReconciliation>,
        policy_load: Option<PolicyLoadSummary>,
    ) -> Self {
        Self {
            configured: Some(boundary.is_some()),
            boundary_date: boundary.map(|b| b.boundary_date),
            report: boundary.map(|b| b.cuts.report().clone()),
            policy_load: policy_load.map(|summary| PolicyLoadOutput {
                mode: match summary.load {
                    FullFcfLoadKind::WarmStart => "warm_start",
                    FullFcfLoadKind::Resume => "resume",
                    FullFcfLoadKind::SimulationOnly => "simulation_only",
                },
                unused_stored_bases: summary.unused_stored_bases,
            }),
            error: None,
        }
    }

    fn error(phase: &str, message: &str) -> Self {
        Self {
            configured: None,
            boundary_date: None,
            report: None,
            policy_load: None,
            error: Some(ValidateErrorOutput {
                phase: phase.to_string(),
                message: message.to_string(),
            }),
        }
    }
}

fn format_constraint_description(
    term: &Term,
    description: &str,
    warning_count: usize,
    path: &Path,
) {
    let error_lines: Vec<&str> = description.lines().collect();
    let _ = term.write_line(&format!(
        "Validation: {} errors, {} warnings in {}",
        error_lines.len(),
        warning_count,
        path.display()
    ));
    for line in error_lines {
        let _ = term.write_line(&format!("{} {line}", style("error:").red().bold()));
    }
}

/// Formats validation warnings as report lines (empty when no warnings). The error count is always
/// zero here — [`validate_case_with_artifacts`] returns `Err` on any error.
fn report_lines(warnings: &[ReportEntry], case_dir: &Path) -> Vec<String> {
    if warnings.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![format!(
        "Validation: 0 errors, {} warnings in {}",
        warnings.len(),
        case_dir.display()
    )];
    for entry in warnings {
        let location = if let Some(entity) = &entry.entity {
            format!("{} ({})", entry.file, entity)
        } else {
            entry.file.clone()
        };
        lines.push(format!(
            "{} {location}: {}",
            style("warning:").yellow().bold(),
            entry.message
        ));
    }
    lines
}

fn print_prep_error(term: &Term, report: &str, case_dir: &Path) {
    let _ = term.write_line(&format!(
        "Validation: 1 errors, 0 warnings in {}",
        case_dir.display()
    ));
    let _ = term.write_line(&format!("{} {report}", style("error:").red().bold()));
}

/// The exit a failure takes, from its [`ErrorClass`]. `report` has already been
/// rendered, so a refusal of the case carries `already_rendered: true`; every
/// other class exits with `other`.
fn failure_cli_error(class: ErrorClass, report: String, other: CliError) -> CliError {
    match class {
        ErrorClass::InvalidInput | ErrorClass::IncompatiblePolicy => CliError::Validation {
            report,
            already_rendered: true,
        },
        ErrorClass::Io | ErrorClass::Solver | ErrorClass::Internal => other,
    }
}

/// Render a pre-solver failure (human, or the `--json` error object).
/// `stdout_sink` is `None` under `--json`, where the error object replaces the
/// human report.
fn render_failure(
    stdout_sink: Option<&Term>,
    json: bool,
    kind: &str,
    report: &str,
    case_dir: &Path,
) -> Result<(), CliError> {
    if let Some(term) = stdout_sink {
        print_prep_error(term, report, case_dir);
    }
    if json {
        emit_validate_json(&ValidateBoundaryOutput::error(kind, report))?;
    }
    Ok(())
}

/// Serialize `output` as `cobre validate --json`'s single stdout JSON object.
/// Stdout carries exactly one JSON object and no human-readable text.
fn emit_validate_json(output: &ValidateBoundaryOutput) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(output).map_err(|e| CliError::Internal {
        message: format!("failed to serialize validate output: {e}"),
    })?;
    println!("{json}");
    Ok(())
}

/// Emit `--json`'s error object ahead of the caller's own `CliError` return.
/// No-op when `json` is false.
fn emit_json_error(json: bool, kind: &str, message: &str) -> Result<(), CliError> {
    if json {
        emit_validate_json(&ValidateBoundaryOutput::error(kind, message))?;
    }
    Ok(())
}

/// Execute the `validate` subcommand, printing a structured diagnostic report
/// (with any pipeline warnings) to stdout. Honors the module's validation contract:
/// exit 0 implies `cobre run` will not fail before the solver begins iterating.
///
/// # Errors
///
/// Returns [`CliError::Validation`] when the case directory fails validation,
/// [`CliError::Io`] on filesystem errors, or [`CliError::Internal`] for
/// unexpected parse or schema failures.
pub fn execute(args: &ValidateArgs) -> Result<(), CliError> {
    let stdout = Term::stdout();
    let stdout_sink = (!args.json).then_some(&stdout);

    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("case directory not found: {}", args.case_dir.display()),
            ),
            context: args.case_dir.display().to_string(),
        });
    }

    // Reuses the pre-parsed CaseArtifacts to avoid re-reading disk.
    let (loaded, report) = match validate_case_with_artifacts(&args.case_dir) {
        Ok(result) => result,
        Err(err) => {
            let kind = err.kind();
            let message = err.to_string();
            match err {
                LoadError::IoError { path, source } => {
                    emit_json_error(args.json, kind, &message)?;
                    return Err(CliError::Io {
                        source,
                        context: path.display().to_string(),
                    });
                }
                LoadError::ConstraintError { description } => {
                    // Warnings are not available when errors abort the pipeline, so report 0.
                    if let Some(term) = stdout_sink {
                        format_constraint_description(term, &description, 0, &args.case_dir);
                    }
                    emit_json_error(args.json, kind, &description)?;
                    return Err(CliError::Validation {
                        report: description,
                        already_rendered: true,
                    });
                }
                _ => {
                    emit_json_error(args.json, kind, &message)?;
                    return Err(CliError::Internal { message });
                }
            }
        }
    };

    let config_path = args.case_dir.join("config.json");
    let config = match cobre_io::parse_config(&config_path) {
        Ok(config) => config,
        Err(err) => {
            emit_json_error(args.json, err.kind(), &err.to_string())?;
            return Err(CliError::from(err));
        }
    };

    let validated = match validate_study(ValidateRequest {
        case_dir: &args.case_dir,
        config: &config,
        system: loaded.system,
        artifacts: loaded.artifacts,
        output_dir: &resolve_output_dir(&args.case_dir, args.output.as_deref()),
    }) {
        Ok(validated) => validated,
        Err(ValidateFailure::ConfigLoad(err)) => {
            let report = err.to_string();
            render_failure(stdout_sink, args.json, err.kind(), &report, &args.case_dir)?;
            return Err(match err {
                LoadError::IoError { .. } => CliError::from(err),
                _ => CliError::Validation {
                    report,
                    already_rendered: true,
                },
            });
        }
        Err(ValidateFailure::Phase(failure)) => {
            let report = failure.report();
            render_failure(
                stdout_sink,
                args.json,
                failure.kind(),
                &report,
                &args.case_dir,
            )?;
            return Err(failure_cli_error(
                failure.error.class(),
                report,
                CliError::from(failure.error),
            ));
        }
        Err(ValidateFailure::PolicyLoad(failure)) => {
            let report = failure.report();
            render_failure(
                stdout_sink,
                args.json,
                &failure.kind(),
                &report,
                &args.case_dir,
            )?;
            return Err(failure_cli_error(
                failure.error.class(),
                report,
                CliError::from(*failure.error),
            ));
        }
    };

    if args.json {
        emit_validate_json(&ValidateBoundaryOutput::success(
            validated.boundary.as_ref(),
            validated.policy_load,
        ))?;
        return Ok(());
    }

    let system = &validated.system;
    let _ = stdout.write_line(&format!(
        "Valid case: {} buses, {} hydros, {} thermals, {} lines",
        system.n_buses(),
        system.n_hydros(),
        system.n_thermals(),
        system.n_lines(),
    ));
    let warnings: Vec<ReportEntry> = report
        .warnings
        .into_iter()
        .chain(validated.warnings)
        .collect();
    for line in report_lines(&warnings, &args.case_dir) {
        let _ = stdout.write_line(&line);
    }
    if let Some(boundary) = &validated.boundary {
        let _ = stdout.write_line(&format!(
            "boundary policy priced at {}",
            boundary.boundary_date
        ));
        let _ = stdout.write_line(&boundary.cuts.report().summary_line());
        for line in boundary.cuts.report().detail_lines() {
            tracing::debug!("{line}");
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use cobre_io::{ReportEntry, ValidationReport};
    use cobre_sddp::SddpError;

    use super::*;

    fn make_report() -> ValidationReport {
        ValidationReport {
            error_count: 0,
            warning_count: 1,
            errors: Vec::new(),
            warnings: vec![ReportEntry {
                kind: "UnusedEntity".to_string(),
                file: "system/thermals.json".to_string(),
                entity: None,
                message: "thermal has zero capacity".to_string(),
            }],
        }
    }

    #[test]
    fn phase_failure_exit_code_follows_the_error_class() {
        let stochastic =
            SddpError::Stochastic(cobre_stochastic::StochasticError::InsufficientData {
                context: "no valid historical windows found".to_string(),
            });
        let class = stochastic.class();
        let refusal = failure_cli_error(
            class,
            "scenarios/: refused".to_string(),
            CliError::from(stochastic),
        );
        assert!(matches!(
            refusal,
            CliError::Validation {
                already_rendered: true,
                ..
            }
        ));
        assert_eq!(refusal.exit_code(), 1);

        let unreadable = SddpError::Io(LoadError::IoError {
            path: PathBuf::from("scenarios/inflow_history.parquet"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        });
        let class = unreadable.class();
        let io = failure_cli_error(
            class,
            "scenarios/: not found".to_string(),
            CliError::from(unreadable),
        );
        assert_eq!(io.exit_code(), 2);
    }

    #[test]
    fn format_report_contains_warning_label() {
        let path = PathBuf::from("/case/dir");
        let output = report_lines(&make_report().warnings, &path).join("\n");
        assert!(
            output.contains("warning:"),
            "expected 'warning:' in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_file_path() {
        let path = PathBuf::from("/case/dir");
        let output = report_lines(&make_report().warnings, &path).join("\n");
        assert!(
            output.contains("system/thermals.json"),
            "expected file path in output, got: {output}"
        );
    }

    #[test]
    fn format_report_summary_header_present() {
        let path = PathBuf::from("/case/dir");
        let output = report_lines(&make_report().warnings, &path).join("\n");
        assert!(
            output.contains("0 errors") && output.contains("1 warnings"),
            "expected summary header with counts, got: {output}"
        );
    }

    #[test]
    fn report_lines_entity_present_renders_file_and_entity() {
        let report = ValidationReport {
            error_count: 0,
            warning_count: 1,
            errors: Vec::new(),
            warnings: vec![ReportEntry {
                kind: "UnusedEntity".to_string(),
                file: "system/buses.json".to_string(),
                entity: Some("bus_01".to_string()),
                message: "bus is unreferenced".to_string(),
            }],
        };
        let output = report_lines(&report.warnings, &PathBuf::from("/case/dir")).join("\n");
        assert!(
            output.contains("system/buses.json (bus_01)"),
            "entity-present location must render 'file (entity)', got: {output}"
        );
    }
}
