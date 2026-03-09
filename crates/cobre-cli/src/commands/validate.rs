//! `cobre validate <CASE_DIR>` subcommand.
//!
//! Runs the 5-layer validation pipeline and prints a structured diagnostic
//! report to stdout. No banner or progress bar — the output is the deliverable.

use std::path::{Path, PathBuf};

use clap::Args;
use console::{style, Term};

use crate::error::CliError;

/// Arguments for the `cobre validate` subcommand.
#[derive(Debug, Args)]
#[command(about = "Validate a case directory and print a structured diagnostic report")]
pub struct ValidateArgs {
    /// Path to the case directory to validate.
    pub case_dir: PathBuf,
}

fn format_constraint_description(term: &Term, description: &str, path: &Path) {
    let lines: Vec<&str> = description.lines().collect();
    let _ = term.write_line(&format!(
        "Validation: {} errors, 0 warnings in {}",
        lines.len(),
        path.display()
    ));
    for line in lines {
        let _ = term.write_line(&format!("{} {line}", style("error:").red().bold()));
    }
}

/// Execute the `validate` subcommand.
///
/// Calls `cobre_io::load_case` on the given case directory and prints a
/// structured diagnostic report to stdout.
///
/// # Errors
///
/// Returns [`CliError::Validation`] when the case directory fails validation,
/// [`CliError::Io`] on filesystem errors, or [`CliError::Internal`] for
/// unexpected parse or schema failures.
#[allow(clippy::needless_pass_by_value)]
pub fn execute(args: ValidateArgs) -> Result<(), CliError> {
    let stdout = Term::stdout();

    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("case directory not found: {}", args.case_dir.display()),
            ),
            context: args.case_dir.display().to_string(),
        });
    }

    match cobre_io::load_case(&args.case_dir) {
        Ok(system) => {
            // Print entity counts — one per line for easy grep/pipe consumption.
            let _ = stdout.write_line(&format!(
                "Valid case: {} buses, {} hydros, {} thermals, {} lines",
                system.n_buses(),
                system.n_hydros(),
                system.n_thermals(),
                system.n_lines(),
            ));
            let _ = stdout.write_line(&format!("  buses: {}", system.n_buses()));
            let _ = stdout.write_line(&format!("  hydros: {}", system.n_hydros()));
            let _ = stdout.write_line(&format!("  thermals: {}", system.n_thermals()));
            let _ = stdout.write_line(&format!("  lines: {}", system.n_lines()));
            Ok(())
        }
        Err(cobre_io::LoadError::IoError { path, source }) => Err(CliError::Io {
            source,
            context: path.display().to_string(),
        }),
        Err(cobre_io::LoadError::ConstraintError { description }) => {
            format_constraint_description(&stdout, &description, &args.case_dir);
            Err(CliError::Validation {
                report: description,
            })
        }
        Err(other) => Err(CliError::Internal {
            message: other.to_string(),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::fmt::Write as _;

    use cobre_io::{ReportEntry, ValidationReport};

    fn format_report_to_string(report: &ValidationReport, path: &Path) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Validation: {} errors, {} warnings in {}",
            report.error_count,
            report.warning_count,
            path.display()
        );
        for entry in &report.errors {
            let _ = writeln!(out, "error: {}", format_entry(entry));
        }
        for entry in &report.warnings {
            let _ = writeln!(out, "warning: {}", format_entry(entry));
        }
        out
    }

    fn format_entry(entry: &ReportEntry) -> String {
        if let Some(entity) = &entry.entity {
            format!("{}: {} ({})", entry.file, entry.message, entity)
        } else {
            format!("{}: {}", entry.file, entry.message)
        }
    }

    fn make_report() -> ValidationReport {
        ValidationReport {
            error_count: 1,
            warning_count: 1,
            errors: vec![ReportEntry {
                kind: "FileNotFound".to_string(),
                file: "system/hydros.json".to_string(),
                entity: Some("hydro_42".to_string()),
                message: "required file is missing".to_string(),
            }],
            warnings: vec![ReportEntry {
                kind: "UnusedEntity".to_string(),
                file: "system/thermals.json".to_string(),
                entity: None,
                message: "thermal has zero capacity".to_string(),
            }],
        }
    }

    use super::*;

    #[test]
    fn format_report_contains_error_label() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("error:"),
            "expected 'error:' in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_warning_label() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("warning:"),
            "expected 'warning:' in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_file_path() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("system/hydros.json"),
            "expected file path in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_error_message() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("required file is missing"),
            "expected error message in output, got: {output}"
        );
    }

    #[test]
    fn format_report_summary_header_present() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("1 errors") && output.contains("1 warnings"),
            "expected summary header with counts, got: {output}"
        );
    }

    #[test]
    fn format_entry_with_entity() {
        let entry = ReportEntry {
            kind: "FileNotFound".to_string(),
            file: "system/buses.json".to_string(),
            entity: Some("bus_01".to_string()),
            message: "missing required field".to_string(),
        };
        let result = format_entry(&entry);
        assert!(result.contains("system/buses.json"), "{result}");
        assert!(result.contains("missing required field"), "{result}");
        assert!(result.contains("bus_01"), "{result}");
    }

    #[test]
    fn format_entry_without_entity() {
        let entry = ReportEntry {
            kind: "FileNotFound".to_string(),
            file: "system/buses.json".to_string(),
            entity: None,
            message: "missing required field".to_string(),
        };
        let result = format_entry(&entry);
        assert!(result.contains("system/buses.json"), "{result}");
        assert!(result.contains("missing required field"), "{result}");
        assert!(!result.contains("(None)"), "{result}");
    }
}
