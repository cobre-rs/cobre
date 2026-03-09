//! CLI error type with exit code mapping and terminal formatting.
//!
//! [`CliError`] is the single error type returned by all subcommand functions.
//! It maps library errors to structured CLI exit codes and produces
//! human-readable diagnostic output with actionable hints.
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

/// Errors that can occur during CLI command execution.
///
/// Each variant maps to a specific exit code and carries enough context
/// for [`CliError::format_error`] to print an actionable diagnostic to stderr.
///
/// # Examples
///
/// ```
/// use cobre_cli::error::CliError;
///
/// let err = CliError::Validation {
///     report: "constraint violation: hydro cascade contains a cycle".to_string(),
/// };
/// assert_eq!(err.exit_code(), 1);
/// assert!(!err.to_string().is_empty());
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// Case directory failed the validation pipeline (exit code 1).
    ///
    /// Covers schema errors, cross-reference errors, semantic constraint
    /// violations, and policy compatibility mismatches detected during
    /// `cobre_io::load_case`.
    #[error("validation error: {report}")]
    Validation {
        /// Human-readable summary of the validation failure.
        report: String,
    },

    /// Filesystem I/O error (exit code 2).
    ///
    /// Covers file-not-found, permission denied, disk full, and write
    /// failures in both the loading pipeline and the output writers.
    #[error("I/O error in {context}: {source}")]
    Io {
        /// Underlying I/O error.
        source: std::io::Error,
        /// Path or operation description that provides context for the failure.
        context: String,
    },

    /// LP solver error during training or simulation (exit code 3).
    ///
    /// Covers infeasible subproblems and numerical solver failures. The
    /// training loop performs a hard stop when this error is returned.
    #[error("solver error: {message}")]
    Solver {
        /// Human-readable description of the solver failure.
        message: String,
    },

    /// Internal or communication error (exit code 4).
    ///
    /// Covers distributed communication failures, stochastic model errors,
    /// unexpected channel closures, and other conditions that indicate a
    /// software or environment problem rather than a user error.
    #[error("internal error: {message}")]
    Internal {
        /// Human-readable description of the internal failure.
        message: String,
    },
}

impl CliError {
    /// Return the process exit code for this error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_cli::error::CliError;
    ///
    /// assert_eq!(CliError::Validation { report: "bad".to_string() }.exit_code(), 1);
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

    /// Print a structured, colored diagnostic to `stderr`.
    ///
    /// Uses `console::style` for colored labels: `error:` in bold red,
    /// hint lines prefixed with `->` in yellow. Colors are suppressed
    /// automatically when the terminal is not interactive or when
    /// `NO_COLOR` is set (handled by the `console` crate).
    ///
    /// # Examples
    ///
    /// ```
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
            Self::Validation { report } => {
                let _ = stderr.write_line(&format!("{label} {report}"));
                let _ = stderr.write_line(&format!(
                    "  {hint_arrow} run `cobre validate <CASE_DIR>` for a full diagnostic report"
                ));
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
    fn from(err: cobre_io::LoadError) -> Self {
        match err {
            cobre_io::LoadError::IoError { path, source } => Self::Io {
                source,
                context: path.display().to_string(),
            },
            other => Self::Validation {
                report: other.to_string(),
            },
        }
    }
}

impl From<cobre_io::OutputError> for CliError {
    fn from(err: cobre_io::OutputError) -> Self {
        match err {
            cobre_io::OutputError::IoError { path, source } => Self::Io {
                source,
                context: path.display().to_string(),
            },
            other => Self::Internal {
                message: other.to_string(),
            },
        }
    }
}

impl From<cobre_sddp::SddpError> for CliError {
    fn from(err: cobre_sddp::SddpError) -> Self {
        match err {
            cobre_sddp::SddpError::Infeasible {
                stage,
                iteration,
                scenario,
            } => Self::Solver {
                message: format!(
                    "LP infeasible at stage {stage}, iteration {iteration}, scenario {scenario}"
                ),
            },
            cobre_sddp::SddpError::Solver(solver_err) => Self::Solver {
                message: solver_err.to_string(),
            },
            cobre_sddp::SddpError::Io(load_err) => Self::from(load_err),
            cobre_sddp::SddpError::Validation(msg) => Self::Validation { report: msg },
            cobre_sddp::SddpError::Communication(msg) | cobre_sddp::SddpError::Simulation(msg) => {
                Self::Internal { message: msg }
            }
            cobre_sddp::SddpError::Stochastic(stoch_err) => Self::Internal {
                message: stoch_err.to_string(),
            },
        }
    }
}

impl From<cobre_sddp::SimulationError> for CliError {
    fn from(err: cobre_sddp::SimulationError) -> Self {
        match err {
            cobre_sddp::SimulationError::LpInfeasible {
                scenario_id,
                stage_id,
                solver_message,
            } => Self::Solver {
                message: format!(
                    "LP infeasible at scenario {scenario_id}, stage {stage_id}: {solver_message}"
                ),
            },
            cobre_sddp::SimulationError::SolverError {
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

    #[test]
    fn validation_exit_code_is_1() {
        let err = CliError::Validation {
            report: "constraint violation".to_string(),
        };
        assert_eq!(err.exit_code(), 1);
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
    fn display_validation_non_empty() {
        let err = CliError::Validation {
            report: "hydro cascade contains a cycle".to_string(),
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

        let load_err = cobre_io::LoadError::IoError {
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
        let load_err = cobre_io::LoadError::ConstraintError {
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

        let load_err = cobre_io::LoadError::SchemaError {
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
    fn from_load_error_cross_reference_maps_to_validation() {
        use std::path::PathBuf;

        let load_err = cobre_io::LoadError::CrossReferenceError {
            source_file: PathBuf::from("system/hydros.json"),
            source_entity: "Hydro 'H1'".to_string(),
            target_collection: "bus registry".to_string(),
            target_entity: "BUS_99".to_string(),
        };
        let cli_err = CliError::from(load_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "LoadError::CrossReferenceError must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
    }

    #[test]
    fn from_sddp_error_infeasible_maps_to_solver() {
        let sddp_err = cobre_sddp::SddpError::Infeasible {
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
        let sddp_err = cobre_sddp::SddpError::Solver(cobre_solver::SolverError::Infeasible);
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

        // LoadError::IoError inside SddpError::Io -> CliError::Io
        let load_io = cobre_io::LoadError::IoError {
            path: PathBuf::from("system/hydros.json"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        let sddp_err = cobre_sddp::SddpError::Io(load_io);
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Io { .. }),
            "SddpError::Io(LoadError::IoError) must map to CliError::Io, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 2);
    }

    #[test]
    fn from_sddp_error_validation_maps_to_validation() {
        let sddp_err = cobre_sddp::SddpError::Validation("forward_passes must be > 0".to_string());
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Validation { .. }),
            "SddpError::Validation must map to CliError::Validation, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 1);
    }

    #[test]
    fn from_sddp_error_communication_maps_to_internal() {
        let sddp_err = cobre_sddp::SddpError::Communication("allgatherv timed out".to_string());
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SddpError::Communication must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_sddp_error_stochastic_maps_to_internal() {
        let stoch_err = cobre_stochastic::StochasticError::InsufficientData {
            context: "hydro 7 has only 2 observations".to_string(),
        };
        let sddp_err = cobre_sddp::SddpError::Stochastic(stoch_err);
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SddpError::Stochastic must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_sddp_error_simulation_maps_to_internal() {
        let sddp_err = cobre_sddp::SddpError::Simulation("output channel closed".to_string());
        let cli_err = CliError::from(sddp_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SddpError::Simulation must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_simulation_error_lp_infeasible_maps_to_solver() {
        let sim_err = cobre_sddp::SimulationError::LpInfeasible {
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
        let sim_err = cobre_sddp::SimulationError::SolverError {
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
        let sim_err = cobre_sddp::SimulationError::IoError {
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
    fn from_simulation_error_policy_incompatible_maps_to_internal() {
        let sim_err = cobre_sddp::SimulationError::PolicyIncompatible {
            message: "hydro count mismatch".to_string(),
        };
        let cli_err = CliError::from(sim_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SimulationError::PolicyIncompatible must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }

    #[test]
    fn from_simulation_error_channel_closed_maps_to_internal() {
        let sim_err = cobre_sddp::SimulationError::ChannelClosed;
        let cli_err = CliError::from(sim_err);
        assert!(
            matches!(cli_err, CliError::Internal { .. }),
            "SimulationError::ChannelClosed must map to CliError::Internal, got: {cli_err:?}"
        );
        assert_eq!(cli_err.exit_code(), 4);
    }
}
