//! CLI error type with exit code mapping and terminal formatting.
//!
//! [`CliError`] is the single error type returned by all subcommand functions.
//!
//! # Exit codes
//!
//! | Code | Variant | Cause |
//! |------|---------|-------|
//! | 1 | [`CliError::Validation`] | Case directory failed validation |
//! | 2 | [`CliError::Io`] | File missing, permission denied, disk full |
//! | 3 | [`CliError::Solver`] | LP infeasible or solver error during training/simulation |
//! | 4 | [`CliError::Internal`] | Communication failure, unexpected state, channel crash |

use console::Term;

use cobre_comm::BackendError;
use cobre_io::LoadError;
use cobre_io::OutputError;
use cobre_sddp::ErrorClass;
use cobre_sddp::SddpError;
use cobre_sddp::SddpError::BasisShapeMismatch;
use cobre_sddp::SddpError::CheckpointWrite;
use cobre_sddp::SddpError::Communication;
use cobre_sddp::SddpError::Infeasible;
use cobre_sddp::SddpError::Io;
use cobre_sddp::SddpError::PolicySoftwareMismatch;
use cobre_sddp::SddpError::Simulation;
use cobre_sddp::SddpError::Solver;
use cobre_sddp::SddpError::Stochastic;
use cobre_sddp::SddpError::Validation;
use cobre_sddp::SddpError::WireVersionMismatch;
use cobre_sddp::SimulationError;
use cobre_sddp::SimulationError::LpInfeasible;
use cobre_sddp::SimulationError::SolverError;
use cobre_sddp::policy::full_fcf_load::FullFcfLoadError;

use std::io::Error;

/// Errors that can occur during CLI command execution.
///
/// Each variant maps to a specific exit code (see the module-level table).
///
/// # Examples
///
/// ```ignore
/// use cobre_cli::error::CliError;
///
/// let err = CliError::Validation {
///     report: "constraint violation: hydro cascade contains a cycle".to_string(),
///     already_rendered: false,
/// };
/// assert_eq!(err.exit_code(), 1);
/// assert!(!err.to_string().is_empty());
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// Case directory failed the validation pipeline (exit code 1).
    #[error("validation error: {report}")]
    Validation {
        /// Validation failure summary.
        report: String,
        /// `true` when the originating subcommand already printed `report` to
        /// stdout (the `validate` subcommand): [`CliError::format_error`] then
        /// suppresses the stderr re-print and the hint so the report surfaces
        /// once and the hint never points back at the command just run. `false`
        /// elsewhere, where the stderr render is the only surfacing.
        already_rendered: bool,
    },

    /// Filesystem I/O error (exit code 2).
    #[error("I/O error in {context}: {source}")]
    Io {
        /// Underlying I/O error.
        source: Error,
        /// Path or operation description that provides context for the failure.
        context: String,
    },

    /// LP solver error during training or simulation (exit code 3).
    #[error("solver error: {message}")]
    Solver {
        /// Solver failure description.
        message: String,
    },

    /// Internal or communication error (exit code 4).
    ///
    /// A software or environment problem rather than a user error.
    #[error("internal error: {message}")]
    Internal {
        /// Internal failure description.
        message: String,
    },
}

impl CliError {
    /// Return the process exit code for this error.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cobre_cli::error::CliError;
    ///
    /// assert_eq!(
    ///     CliError::Validation { report: "bad".to_string(), already_rendered: false }.exit_code(),
    ///     1
    /// );
    /// assert_eq!(CliError::Solver { message: "infeasible".to_string() }.exit_code(), 3);
    /// ```
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Validation { .. } => 1,
            Self::Io { .. } => 2,
            Self::Solver { .. } => 3,
            Self::Internal { .. } => 4,
        }
    }

    /// The error a rank returns for a failure code agreed from rank 0's writes;
    /// the inverse of [`CliError::exit_code`], so every rank exits with that code.
    pub(crate) fn for_peer_failure(exit_code: i32) -> Self {
        const MESSAGE: &str =
            "rank 0 failed to write the run outputs; failing on every rank in lockstep";
        match exit_code {
            1 => Self::Validation {
                report: MESSAGE.to_string(),
                already_rendered: false,
            },
            2 => Self::Io {
                source: Error::other(MESSAGE),
                context: "rank 0".into(),
            },
            3 => Self::Solver {
                message: MESSAGE.to_string(),
            },
            _ => Self::Internal {
                message: MESSAGE.to_string(),
            },
        }
    }

    fn prefixed(self, prefix: &str) -> Self {
        match self {
            Self::Validation {
                report,
                already_rendered,
            } => Self::Validation {
                report: format!("{prefix}: {report}"),
                already_rendered,
            },
            Self::Io { source, context } => Self::Io {
                source,
                context: format!("{prefix}: {context}"),
            },
            Self::Solver { message } => Self::Solver {
                message: format!("{prefix}: {message}"),
            },
            Self::Internal { message } => Self::Internal {
                message: format!("{prefix}: {message}"),
            },
        }
    }

    fn validation_lines(report: &str, already_rendered: bool) -> Vec<String> {
        if already_rendered {
            return Vec::new();
        }
        let label = console::style("error:").red().bold();
        let hint_arrow = console::style("->").yellow();
        vec![
            format!("{label} {report}"),
            format!("  {hint_arrow} run `cobre validate <CASE_DIR>` for a full diagnostic report"),
        ]
    }

    /// Print a structured, colored diagnostic to `stderr`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cobre_cli::error::CliError;
    /// use console::Term;
    ///
    /// let err = CliError::Solver { message: "LP infeasible at stage 12".to_string() };
    /// let stderr = Term::stderr();
    /// err.format_error(&stderr);
    /// ```
    pub fn format_error(&self, stderr: &Term) {
        let label = console::style("error:").red().bold();
        let hint_arrow = console::style("->").yellow();

        match self {
            Self::Validation {
                report,
                already_rendered,
            } => {
                for line in Self::validation_lines(report, *already_rendered) {
                    let _ = stderr.write_line(&line);
                }
            }
            Self::Io { source, context } => {
                let _ = stderr.write_line(&format!("{label} I/O error in {context}: {source}"));
                let _ = stderr.write_line(&format!(
                    "  {hint_arrow} check that the path exists and you have read/write permissions"
                ));
            }
            Self::Solver { message } => {
                let _ = stderr.write_line(&format!("{label} {message}"));
                let _ = stderr.write_line(&format!(
                    "  {hint_arrow} check constraint bounds (hydros may have conflicting min/max storage)"
                ));
                let _ = stderr.write_line(&format!(
                    "  {hint_arrow} run `cobre validate <CASE_DIR>` for a full diagnostic report"
                ));
            }
            Self::Internal { message } => {
                let _ = stderr.write_line(&format!("{label} {message}"));
                let _ = stderr.write_line(&format!(
                    "  {hint_arrow} this may indicate a software or environment problem"
                ));
                let _ = stderr.write_line(&format!(
                    "  {hint_arrow} report this at https://github.com/cobre-rs/cobre/issues"
                ));
            }
        }
    }
}

impl From<cobre_io::LoadError> for CliError {
    fn from(err: LoadError) -> Self {
        match err {
            LoadError::IoError { path, source } => Self::Io {
                source,
                context: path.display().to_string(),
            },
            other => Self::Validation {
                report: other.to_string(),
                already_rendered: false,
            },
        }
    }
}

impl From<cobre_io::OutputError> for CliError {
    fn from(err: OutputError) -> Self {
        match err {
            OutputError::IoError { path, source } => Self::Io {
                source,
                context: path.display().to_string(),
            },
            refusal @ OutputError::ForeignEntry { .. } => Self::Validation {
                report: refusal.to_string(),
                already_rendered: false,
            },
            other => Self::Internal {
                message: other.to_string(),
            },
        }
    }
}

impl From<cobre_comm::BackendError> for CliError {
    fn from(err: BackendError) -> Self {
        Self::Internal {
            message: format!("communication backend error: {err}"),
        }
    }
}

impl From<cobre_sddp::SddpError> for CliError {
    fn from(err: SddpError) -> Self {
        let class = err.class();
        let message = match err {
            Io(load_err) => return Self::from(load_err),
            CheckpointWrite { iteration, source } => {
                return Self::from(source)
                    .prefixed(&format!("checkpoint write at iteration {iteration}"));
            }
            Infeasible {
                stage,
                iteration,
                scenario,
            } => format!(
                "LP infeasible at stage {stage}, iteration {iteration}, scenario {scenario}"
            ),
            Solver(solver_err) => solver_err.to_string(),
            Communication(comm_err) => comm_err.to_string(),
            Validation(msg) | Simulation(msg) => msg,
            WireVersionMismatch { encoded, expected } => format!(
                "wire format version mismatch: encoded={encoded}, expected={expected}; \
                 restart all ranks with the same binary"
            ),
            other @ (Stochastic(_) | BasisShapeMismatch { .. } | PolicySoftwareMismatch { .. }) => {
                other.to_string()
            }
        };
        match class {
            ErrorClass::InvalidInput | ErrorClass::IncompatiblePolicy => Self::Validation {
                report: message,
                already_rendered: false,
            },
            ErrorClass::Solver => Self::Solver { message },
            ErrorClass::Io | ErrorClass::Internal => Self::Internal { message },
        }
    }
}

impl From<FullFcfLoadError> for CliError {
    fn from(err: FullFcfLoadError) -> Self {
        let class = err.class();
        let message = match err {
            FullFcfLoadError::Refused(inner)
            | FullFcfLoadError::FcfConstruction { source: inner, .. } => return Self::from(inner),
            FullFcfLoadError::Read { source } if class == ErrorClass::Io => {
                return Self::from(source);
            }
            other @ (FullFcfLoadError::MissingPolicyDirectory { .. }
            | FullFcfLoadError::Read { .. }) => other.to_string(),
        };
        match class {
            ErrorClass::InvalidInput | ErrorClass::IncompatiblePolicy => Self::Validation {
                report: message,
                already_rendered: false,
            },
            ErrorClass::Solver => Self::Solver { message },
            ErrorClass::Io | ErrorClass::Internal => Self::Internal { message },
        }
    }
}

impl From<cobre_sddp::SimulationError> for CliError {
    fn from(err: SimulationError) -> Self {
        match err {
            LpInfeasible {
                scenario_id,
                stage_id,
                solver_message,
            } => Self::Solver {
                message: format!(
                    "LP infeasible at scenario {scenario_id}, stage {stage_id}: {solver_message}"
                ),
            },
            SolverError {
                scenario_id,
                stage_id,
                solver_message,
            } => Self::Solver {
                message: format!(
                    "solver error at scenario {scenario_id}, stage {stage_id}: {solver_message}"
                ),
            },
            other => Self::Internal {
                message: other.to_string(),
            },
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use cobre_comm::CommError;
    use cobre_stochastic::StochasticError;

    fn expected_exit_code(class: ErrorClass) -> i32 {
        match class {
            ErrorClass::InvalidInput | ErrorClass::IncompatiblePolicy => 1,
            ErrorClass::Io => 2,
            ErrorClass::Solver => 3,
            ErrorClass::Internal => 4,
        }
    }

    #[test]
    fn validation_exit_code_is_1() {
        let err = CliError::Validation {
            report: "constraint violation".to_string(),
            already_rendered: false,
        };
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn validation_exit_code_is_1_when_already_rendered() {
        let err = CliError::Validation {
            report: "constraint violation".to_string(),
            already_rendered: true,
        };
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn validation_lines_skipped_when_already_rendered() {
        let lines = CliError::validation_lines("constraint violation", true);
        assert!(
            lines.is_empty(),
            "expected no stderr lines when already_rendered=true, got: {lines:?}"
        );
    }

    #[test]
    fn validation_lines_emitted_when_not_rendered() {
        console::set_colors_enabled_stderr(false);
        let lines = CliError::validation_lines("hydro cascade contains a cycle", false);
        let joined = lines.join("\n");
        assert!(
            joined.contains("hydro cascade contains a cycle"),
            "expected the report in the rendered lines, got: {joined}"
        );
        assert!(
            joined.contains("run `cobre validate <CASE_DIR>`"),
            "expected the validate hint in the rendered lines, got: {joined}"
        );
    }

    #[test]
    fn io_exit_code_is_2() {
        let err = CliError::Io {
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
            context: "system/hydros.json".to_string(),
        };
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn solver_exit_code_is_3() {
        let err = CliError::Solver {
            message: "LP infeasible at stage 5".to_string(),
        };
        assert_eq!(err.exit_code(), 3);
    }

    #[test]
    fn internal_exit_code_is_4() {
        let err = CliError::Internal {
            message: "channel closed unexpectedly".to_string(),
        };
        assert_eq!(err.exit_code(), 4);
    }

    #[test]
    fn peer_failure_keeps_the_agreed_exit_code() {
        for code in 1..=4 {
            assert_eq!(CliError::for_peer_failure(code).exit_code(), code);
        }
    }

    #[test]
    fn display_validation_non_empty() {
        let err = CliError::Validation {
            report: "hydro cascade contains a cycle".to_string(),
            already_rendered: false,
        };
        let s = err.to_string();
        assert!(!s.is_empty());
        assert!(s.contains("hydro cascade contains a cycle"), "{s}");
    }

    #[test]
    fn display_io_non_empty() {
        let err = CliError::Io {
            source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "permission denied"),
            context: "output/results".to_string(),
        };
        let s = err.to_string();
        assert!(!s.is_empty());
        assert!(s.contains("output/results"), "{s}");
        assert!(s.contains("permission denied"), "{s}");
    }

    #[test]
    fn display_solver_non_empty() {
        let err = CliError::Solver {
            message: "LP infeasible at stage 12, iteration 3, scenario 47".to_string(),
        };
        let s = err.to_string();
        assert!(!s.is_empty());
        assert!(s.contains("stage 12"), "{s}");
    }

    #[test]
    fn display_internal_non_empty() {
        let err = CliError::Internal {
            message: "allgatherv timed out".to_string(),
        };
        let s = err.to_string();
        assert!(!s.is_empty());
        assert!(s.contains("allgatherv timed out"), "{s}");
    }

    #[test]
    fn from_load_error_io_maps_to_cli_io() {
        use std::path::PathBuf;

        let load_err = LoadError::IoError {
            path: PathBuf::from("system/hydros.json"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        };
        let cli_err = CliError::from(load_err);
        assert!(
            matches!(cli_err, CliError::Io { .. }),
            "LoadError::IoError must map to CliError::Io, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 2);
    }

    #[test]
    fn from_load_error_constraint_maps_to_validation() {
        let load_err = LoadError::ConstraintError {
            description: "hydro cascade contains a cycle".to_string(),
        };
        let cli_err = CliError::from(load_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "LoadError::ConstraintError must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
    }

    #[test]
    fn from_load_error_schema_maps_to_validation() {
        use std::path::PathBuf;

        let load_err = LoadError::SchemaError {
            path: PathBuf::from("system/buses.json"),
            field: "voltage".to_string(),
            message: "must be positive".to_string(),
        };
        let cli_err = CliError::from(load_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "LoadError::SchemaError must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
    }

    #[test]
    fn from_load_error_parse_maps_to_validation() {
        let load_err = LoadError::parse("stages.json", "unexpected end of input");
        let cli_err = CliError::from(load_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "LoadError::ParseError must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
    }

    #[test]
    fn from_sddp_error_infeasible_maps_to_solver() {
        let sddp_err = Infeasible {
            stage: 5,
            iteration: 42,
            scenario: 3,
        };
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Solver { .. }),
            "SddpError::Infeasible must map to CliError::Solver, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 3);
    }

    #[test]
    fn from_sddp_error_solver_maps_to_solver() {
        let sddp_err = Solver(cobre_solver::SolverError::Infeasible);
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Solver { .. }),
            "SddpError::Solver must map to CliError::Solver, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 3);
    }

    #[test]
    fn from_sddp_error_io_maps_to_cli_io_or_validation() {
        use std::path::PathBuf;

        let load_io = LoadError::IoError {
            path: PathBuf::from("system/hydros.json"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        let sddp_err = Io(load_io);
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Io { .. }),
            "SddpError::Io(LoadError::IoError) must map to CliError::Io, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 2);
    }

    #[test]
    fn from_sddp_error_validation_maps_to_validation() {
        let sddp_err = Validation("forward_passes must be > 0".to_string());
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "SddpError::Validation must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
    }

    #[test]
    fn from_sddp_error_policy_software_mismatch_maps_to_validation() {
        let sddp_err = PolicySoftwareMismatch {
            policy_software: Some("cobre".to_string()),
            policy_version: "0.0.1".to_string(),
        };
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "SddpError::PolicySoftwareMismatch must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
        let CliError::Validation { report, .. } = cli_err else {
            unreachable!("checked above")
        };
        assert!(report.contains("0.0.1"), "{report}");
        assert!(report.contains(cobre_io::SOFTWARE_VERSION), "{report}");
    }

    #[test]
    fn from_sddp_error_communication_maps_to_internal() {
        let sddp_err = Communication(CommError::CollectiveFailed {
            operation: "allgatherv",
            mpi_error_code: 1,
            message: "timed out".to_string(),
        });
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SddpError::Communication must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_sddp_error_stochastic_maps_to_validation() {
        let stoch_err = StochasticError::InsufficientData {
            context: "hydro 7 has only 2 observations".to_string(),
        };
        let sddp_err = Stochastic(stoch_err);
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "SddpError::Stochastic must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
        let CliError::Validation { report, .. } = cli_err else {
            unreachable!("checked above")
        };
        assert!(
            report.contains("stochastic error: insufficient data: hydro 7 has only 2 observations"),
            "{report}"
        );
    }

    #[test]
    fn cli_exit_code_follows_the_error_class() {
        use cobre_sddp::policy::full_fcf_load::FullFcfLoadKind;
        use std::path::PathBuf;

        let errors = vec![
            Stochastic(StochasticError::InsufficientData {
                context: "no data".to_string(),
            }),
            Validation("bad config".to_string()),
            Io(LoadError::parse("config.json", "unexpected end of input")),
            Io(LoadError::SchemaError {
                path: PathBuf::from("system/buses.json"),
                field: "voltage".to_string(),
                message: "must be positive".to_string(),
            }),
            Io(LoadError::ConstraintError {
                description: "cycle".to_string(),
            }),
            PolicySoftwareMismatch {
                policy_software: None,
                policy_version: "0.0.1".to_string(),
            },
            Io(LoadError::IoError {
                path: PathBuf::from("system/hydros.json"),
                source: std::io::Error::other("permission denied"),
            }),
            Infeasible {
                stage: 0,
                iteration: 1,
                scenario: 0,
            },
            Solver(cobre_solver::SolverError::Infeasible),
            Communication(CommError::InvalidCommunicator),
            Simulation("output channel closed".to_string()),
            WireVersionMismatch {
                encoded: 0,
                expected: 1,
            },
            BasisShapeMismatch {
                num_row: 10,
                total_basic: 9,
                col_basic: 4,
                row_basic: 5,
            },
            CheckpointWrite {
                iteration: 2,
                source: OutputError::IoError {
                    path: PathBuf::from("out/policy.staging"),
                    source: std::io::Error::other("no space left on device"),
                },
            },
            CheckpointWrite {
                iteration: 2,
                source: OutputError::ForeignEntry {
                    dir: PathBuf::from("out/policy"),
                    entry: PathBuf::from("out/policy/notes.txt"),
                },
            },
            CheckpointWrite {
                iteration: 2,
                source: OutputError::SerializationError {
                    entity: "stage_cuts".to_string(),
                    message: "buffer too large".to_string(),
                },
            },
        ];
        for err in errors {
            let expected = expected_exit_code(err.class());
            let description = err.to_string();
            assert_eq!(CliError::from(err).exit_code(), expected, "{description}");
        }

        let mut policy_load_errors = vec![
            FullFcfLoadError::Refused(Validation("state_dimension mismatch".to_string())),
            FullFcfLoadError::Refused(PolicySoftwareMismatch {
                policy_software: None,
                policy_version: "0.0.1".to_string(),
            }),
            FullFcfLoadError::Read {
                source: OutputError::IoError {
                    path: PathBuf::from("policy/manifest.bin"),
                    source: std::io::Error::from(std::io::ErrorKind::NotFound),
                },
            },
            FullFcfLoadError::Read {
                source: OutputError::IoError {
                    path: PathBuf::from("policy/manifest.bin"),
                    source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                },
            },
            FullFcfLoadError::Read {
                source: OutputError::SerializationError {
                    entity: "checkpoint_manifest".to_string(),
                    message: "missing file identifier".to_string(),
                },
            },
            FullFcfLoadError::Read {
                source: OutputError::SchemaError {
                    file: "cuts/0.bin".to_string(),
                    column: "coefficients".to_string(),
                    message: "unexpected type".to_string(),
                },
            },
            FullFcfLoadError::Read {
                source: OutputError::ManifestError {
                    manifest_type: "checkpoint".to_string(),
                    message: "inconsistent dates".to_string(),
                },
            },
            FullFcfLoadError::Read {
                source: OutputError::ForeignEntry {
                    dir: PathBuf::from("out/policy"),
                    entry: PathBuf::from("out/policy/notes.txt"),
                },
            },
        ];
        for kind in [
            FullFcfLoadKind::WarmStart,
            FullFcfLoadKind::Resume,
            FullFcfLoadKind::SimulationOnly,
        ] {
            policy_load_errors.push(FullFcfLoadError::MissingPolicyDirectory {
                kind,
                path: PathBuf::from("out/policy"),
            });
            policy_load_errors.push(FullFcfLoadError::FcfConstruction {
                kind,
                source: Validation("stage_results is empty".to_string()),
            });
        }
        for err in policy_load_errors {
            let expected = expected_exit_code(err.class());
            let description = err.to_string();
            assert_eq!(CliError::from(err).exit_code(), expected, "{description}");
        }
    }

    #[test]
    fn checkpoint_write_failure_maps_like_its_output_error() {
        let err = CheckpointWrite {
            iteration: 2,
            source: OutputError::IoError {
                path: std::path::PathBuf::from("/proc/cobre-unwritable-policy.staging"),
                source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            },
        };
        let cli_err = CliError::from(err);
        assert!(matches!(cli_err, CliError::Io { .. }), "{cli_err:?}");
        assert_eq!(cli_err.exit_code(), 2);
        let CliError::Io { context, .. } = cli_err else {
            unreachable!("checked above")
        };
        assert_eq!(
            context,
            "checkpoint write at iteration 2: /proc/cobre-unwritable-policy.staging"
        );

        let refusal = OutputError::ForeignEntry {
            dir: std::path::PathBuf::from("out/policy"),
            entry: std::path::PathBuf::from("out/policy/notes.txt"),
        };
        let display = refusal.to_string();
        let cli_err = CliError::from(CheckpointWrite {
            iteration: 2,
            source: refusal,
        });
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "{cli_err:?}"
        );
        let CliError::Validation { report, .. } = cli_err else {
            unreachable!("checked above")
        };
        assert_eq!(
            report,
            format!("checkpoint write at iteration 2: {display}")
        );
    }

    #[test]
    fn foreign_entry_output_error_maps_to_validation_exit_1() {
        let err = OutputError::ForeignEntry {
            dir: std::path::PathBuf::from("out/policy"),
            entry: std::path::PathBuf::from("out/policy/notes.txt"),
        };
        let display = err.to_string();
        let cli_err = CliError::from(err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "OutputError::ForeignEntry must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
        let CliError::Validation { report, .. } = cli_err else {
            unreachable!("checked above")
        };
        assert_eq!(report, display);
    }

    #[test]
    fn full_fcf_load_refusals_keep_the_run_message() {
        let err =
            FullFcfLoadError::Refused(Validation("entity manifest length mismatch: x".to_string()));
        let cli_err = CliError::from(err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "a refused checkpoint must map to CliError::Validation, got: {cli_err:?}"
        );
        let CliError::Validation { report, .. } = cli_err else {
            unreachable!("checked above")
        };
        assert_eq!(report, "entity manifest length mismatch: x");
    }

    #[test]
    fn from_sddp_error_simulation_maps_to_internal() {
        let sddp_err = Simulation("output channel closed".to_string());
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SddpError::Simulation must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_simulation_error_lp_infeasible_maps_to_solver() {
        let sim_err = LpInfeasible {
            scenario_id: 5,
            stage_id: 3,
            solver_message: "primal infeasible".to_string(),
        };
        let cli_err = CliError::from(sim_err);
        assert!(
            matches!(cli_err, CliError::Solver { .. }),
            "SimulationError::LpInfeasible must map to CliError::Solver, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 3);
    }

    #[test]
    fn from_simulation_error_solver_error_maps_to_solver() {
        let sim_err = SolverError {
            scenario_id: 10,
            stage_id: 7,
            solver_message: "numerical difficulties".to_string(),
        };
        let cli_err = CliError::from(sim_err);
        assert!(
            matches!(cli_err, CliError::Solver { .. }),
            "SimulationError::SolverError must map to CliError::Solver, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 3);
    }

    #[test]
    fn from_simulation_error_io_maps_to_internal() {
        let sim_err = SimulationError::IoError {
            message: "disk full".to_string(),
        };
        let cli_err = CliError::from(sim_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SimulationError::IoError must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_simulation_error_channel_closed_maps_to_internal() {
        let sim_err = SimulationError::ChannelClosed;
        let cli_err = CliError::from(sim_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SimulationError::ChannelClosed must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_backend_error_maps_to_internal() {
        let err = BackendError::InvalidBackend {
            requested: "foobar".to_string(),
            available: vec!["local".to_string()],
        };
        let cli_err = CliError::from(err);
        assert!(matches!(cli_err, CliError::Internal { .. }));
        assert_eq!(cli_err.exit_code(), 4);
    }
}
