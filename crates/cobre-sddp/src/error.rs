//! Error types for the `cobre-sddp` crate.

use cobre_comm::CommError;
use cobre_io::scenarios::estimation::EstimationError;
use cobre_io::{LoadError, OutputError, SOFTWARE_NAME, SOFTWARE_VERSION, policy_checkpoint_remedy};
use cobre_solver::SolverError;
use cobre_stochastic::StochasticError;

use crate::fpha_fitting::FphaFittingError;

/// Unified error type for SDDP algorithm operations.
///
/// All fallible methods in `cobre-sddp` return `Result<T, SddpError>`.
/// The type is `Send + Sync + 'static` so it can be propagated across
/// thread boundaries and wrapped by `anyhow` or `Box<dyn Error>` in
/// application-level code.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::SddpError;
///
/// fn assert_send_sync_static<E: std::error::Error + Send + Sync + 'static>() {}
/// assert_send_sync_static::<SddpError>();
/// ```
#[derive(Debug, thiserror::Error)]
pub enum SddpError {
    /// An LP subproblem solve failed in the forward or backward pass.
    ///
    /// Wraps a [`cobre_solver::SolverError`] that persisted through all retries.
    #[error("solver error: {0}")]
    Solver(#[from] SolverError),

    /// A distributed communication operation failed.
    #[error("communication error: {0}")]
    Communication(#[from] CommError),

    /// Stochastic model construction or scenario generation failed.
    #[error("stochastic error: {0}")]
    Stochastic(#[from] StochasticError),

    /// Case directory loading or validation failed.
    #[error("I/O error: {0}")]
    Io(#[from] LoadError),

    /// SDDP configuration is invalid (semantic errors not caught by the
    /// upstream loading pipeline).
    #[error("configuration validation error: {0}")]
    Validation(String),

    /// An LP subproblem was provably infeasible after all recourse actions —
    /// distinct from [`SddpError::Solver`] (numerical/timeout failure). A hard stop.
    #[error("infeasible subproblem at stage {stage}, iteration {iteration}, scenario {scenario}")]
    Infeasible {
        /// Stage index (0-based) at which infeasibility was detected.
        stage: usize,
        /// Iteration number (1-based) at which infeasibility was detected.
        iteration: u64,
        /// Scenario index (0-based) in the forward pass that triggered infeasibility.
        scenario: usize,
    },

    /// A simulation phase operation failed; the detailed type is
    /// [`SimulationError`](crate::SimulationError), stringified here.
    #[error("simulation error: {0}")]
    Simulation(String),

    /// A reconstructed warm-start basis has fewer basic variables than the LP
    /// has rows, proving the stored basis was captured against a different LP
    /// shape. See
    /// [`enforce_basic_count_invariant`](crate::basis_reconstruct::enforce_basic_count_invariant).
    #[error(
        "stored basis was captured against a different LP shape: num_row={num_row} but \
         total_basic={total_basic} (col_basic={col_basic}, row_basic={row_basic}); a basic-count \
         deficit is unreachable for a stored basis matching this LP's column count and \
         base row count"
    )]
    BasisShapeMismatch {
        /// Row count of the LP the basis is being applied to.
        num_row: usize,
        /// `col_basic + row_basic` in the reconstructed basis.
        total_basic: usize,
        /// Basic columns in the reconstructed basis.
        col_basic: usize,
        /// Basic rows in the reconstructed basis.
        row_basic: usize,
    },

    /// A postcard-encoded payload's wire `version` does not match the current binary.
    #[error(
        "wire format version mismatch: encoded={encoded}, expected={expected}; \
         restart all ranks with the same binary"
    )]
    WireVersionMismatch {
        /// The version number found in the encoded payload.
        encoded: u32,
        /// The version number expected by the current binary.
        expected: u32,
    },

    /// A policy checkpoint was not written by this build: only checkpoints from
    /// the same software at the same version load.
    #[error(
        "policy was written by {writer}, but this is {SOFTWARE_NAME} {SOFTWARE_VERSION}; a \
         policy loads only in the software and version that wrote it: {remedy}",
        writer = describe_writer(.policy_software.as_deref(), .policy_version),
        remedy = policy_checkpoint_remedy()
    )]
    PolicySoftwareMismatch {
        /// The `software` the checkpoint's manifest records, if any.
        policy_software: Option<String>,
        /// The `software_version` the checkpoint's manifest records.
        policy_version: String,
    },

    /// A checkpoint written during training could not be committed.
    #[error("checkpoint write at iteration {iteration} failed: {source}")]
    CheckpointWrite {
        /// The iteration whose checkpoint failed.
        iteration: u64,
        /// The underlying write failure.
        #[source]
        source: OutputError,
    },
}

fn describe_writer(software: Option<&str>, version: &str) -> String {
    match (software.filter(|name| !name.is_empty()), version) {
        (Some(name), "") => format!("{name}, which recorded no version"),
        (Some(name), version) => format!("{name} {version}"),
        (None, "") => "software that recorded no name or version".to_string(),
        (None, version) => format!("software that recorded no name, version {version}"),
    }
}

/// What a failure means to the person running the study.
///
/// Front ends derive their exit codes and exception classes from
/// [`SddpError::class`], so they cannot keep separate lists of which failures
/// refuse the user's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The case data or configuration is refused.
    InvalidInput,
    /// A stored policy cannot be used by this build or with this case.
    IncompatiblePolicy,
    /// The operating system refused a read or write.
    Io,
    /// An LP solve failed.
    Solver,
    /// A software or environment fault.
    Internal,
}

impl SddpError {
    /// The [`ErrorClass`] of this error.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Stochastic(
                StochasticError::InvalidParParameters { .. }
                | StochasticError::InvalidCorrelation { .. }
                | StochasticError::InsufficientData { .. }
                | StochasticError::UnsupportedNoiseMethod { .. }
                | StochasticError::DimensionExceedsCapacity { .. }
                | StochasticError::MissingScenarioSource { .. },
            )
            | Self::Validation(_)
            | Self::Io(
                LoadError::ParseError { .. }
                | LoadError::SchemaError { .. }
                | LoadError::ConstraintError { .. },
            )
            | Self::CheckpointWrite {
                source: OutputError::ForeignEntry { .. },
                ..
            } => ErrorClass::InvalidInput,
            Self::PolicySoftwareMismatch { .. } => ErrorClass::IncompatiblePolicy,
            Self::Io(LoadError::IoError { .. })
            | Self::CheckpointWrite {
                source: OutputError::IoError { .. },
                ..
            } => ErrorClass::Io,
            Self::Infeasible { .. } | Self::Solver(_) => ErrorClass::Solver,
            Self::Communication(_)
            | Self::Simulation(_)
            | Self::WireVersionMismatch { .. }
            | Self::BasisShapeMismatch { .. }
            | Self::CheckpointWrite {
                source:
                    OutputError::SerializationError { .. }
                    | OutputError::SchemaError { .. }
                    | OutputError::ManifestError { .. },
                ..
            } => ErrorClass::Internal,
        }
    }
}

impl From<EstimationError> for SddpError {
    fn from(err: EstimationError) -> Self {
        match err {
            EstimationError::Load(load_err) => Self::Io(load_err),
            EstimationError::Stochastic(stoch_err) => Self::Stochastic(stoch_err),
            EstimationError::Validation(validation_err) => {
                Self::Validation(validation_err.to_string())
            }
        }
    }
}

impl From<FphaFittingError> for SddpError {
    fn from(err: FphaFittingError) -> Self {
        Self::Validation(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorClass, SddpError};
    use cobre_comm::CommError;
    use cobre_io::{LoadError, OutputError, SOFTWARE_NAME, SOFTWARE_VERSION};
    use cobre_solver::SolverError;
    use cobre_stochastic::StochasticError;
    use std::path::PathBuf;

    use crate::fpha_fitting::FphaFittingError;

    fn assert_send_sync_static<E: std::error::Error + Send + Sync + 'static>() {}

    #[test]
    fn sddp_error_is_send_sync_static() {
        assert_send_sync_static::<SddpError>();
    }

    #[test]
    fn display_solver_variant_contains_solver_and_underlying_message() {
        let inner = SolverError::Infeasible;
        let err = SddpError::Solver(inner);
        let msg = err.to_string();
        assert!(msg.contains("solver"), "{msg}");
        assert!(msg.contains("infeasible"), "{msg}");
    }

    #[test]
    fn display_communication_variant_contains_message() {
        let err = SddpError::Communication(CommError::CollectiveFailed {
            operation: "allgatherv",
            mpi_error_code: 1,
            message: "timed out".to_string(),
        });
        let msg = err.to_string();
        assert!(msg.contains("communication"), "{msg}");
        assert!(msg.contains("allgatherv"), "{msg}");
    }

    #[test]
    fn display_stochastic_variant_contains_stochastic_and_underlying_message() {
        let inner = StochasticError::InsufficientData {
            context: "hydro 7 has only 2 observations".to_string(),
        };
        let err = SddpError::Stochastic(inner);
        let msg = err.to_string();
        assert!(msg.contains("stochastic"), "{msg}");
        assert!(msg.contains("insufficient data"), "{msg}");
    }

    #[test]
    fn display_io_variant_contains_io_and_underlying_message() {
        let inner = LoadError::ConstraintError {
            description: "hydro cascade contains a cycle".to_string(),
        };
        let err = SddpError::Io(inner);
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("i/o") || msg.to_lowercase().contains("io"),
            "{msg}"
        );
        assert!(msg.contains("hydro cascade contains a cycle"), "{msg}");
    }

    #[test]
    fn display_validation_variant_contains_message() {
        let err = SddpError::Validation("forward_passes must be greater than zero".to_string());
        let msg = err.to_string();
        assert!(msg.contains("validation"), "{msg}");
        assert!(
            msg.contains("forward_passes must be greater than zero"),
            "{msg}"
        );
    }

    #[test]
    fn display_infeasible_variant_contains_stage_iteration_scenario() {
        let err = SddpError::Infeasible {
            stage: 5,
            iteration: 42,
            scenario: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains('5'), "{msg}");
        assert!(msg.contains("42"), "{msg}");
        assert!(msg.contains('3'), "{msg}");
    }

    #[test]
    fn display_policy_software_mismatch_names_both_writers() {
        let err = SddpError::PolicySoftwareMismatch {
            policy_software: Some("another-program".to_string()),
            policy_version: "0.0.1".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("another-program 0.0.1"), "{msg}");
        assert!(
            msg.contains(&format!("{SOFTWARE_NAME} {SOFTWARE_VERSION}")),
            "{msg}"
        );
    }

    #[test]
    fn display_policy_software_mismatch_without_a_recorded_name() {
        let err = SddpError::PolicySoftwareMismatch {
            policy_software: None,
            policy_version: "0.0.1".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("recorded no name, version 0.0.1"), "{msg}");
    }

    #[test]
    fn display_policy_software_mismatch_tells_the_user_to_rerun_the_producing_program() {
        let err = SddpError::PolicySoftwareMismatch {
            policy_software: Some("another-program".to_string()),
            policy_version: "0.0.1".to_string(),
        };
        assert_eq!(
            err.to_string(),
            format!(
                "policy was written by another-program 0.0.1, but this is {SOFTWARE_NAME} \
                 {SOFTWARE_VERSION}; a policy loads only in the software and version that \
                 wrote it: re-run the program that produced it with {SOFTWARE_NAME} \
                 {SOFTWARE_VERSION}; for a converted boundary policy, convert it again"
            )
        );
    }

    #[test]
    fn display_policy_software_mismatch_names_an_unrecorded_name_or_version() {
        let cases = [
            (
                Some("cobre"),
                "",
                "policy was written by cobre, which recorded no version, but this is ",
            ),
            (
                None,
                "",
                "policy was written by software that recorded no name or version, but this is ",
            ),
            (
                Some(""),
                "0.0.1",
                "policy was written by software that recorded no name, version 0.0.1, but this is ",
            ),
        ];
        for (software, version, expected_prefix) in cases {
            let msg = SddpError::PolicySoftwareMismatch {
                policy_software: software.map(str::to_string),
                policy_version: version.to_string(),
            }
            .to_string();
            assert!(msg.starts_with(expected_prefix), "{msg}");
            assert!(!msg.contains(" ,") && !msg.contains("by  "), "{msg}");
        }
    }

    #[test]
    fn from_solver_error() {
        let inner = SolverError::InternalError {
            message: "test".to_string(),
            error_code: Some(99),
        };
        let err: SddpError = inner.into();
        assert!(matches!(err, SddpError::Solver(_)));
    }

    #[test]
    fn from_stochastic_error() {
        let inner = StochasticError::InsufficientData {
            context: "hydro 7 has only 2 observations".to_string(),
        };
        let err: SddpError = inner.into();
        assert!(matches!(err, SddpError::Stochastic(_)));
    }

    #[test]
    fn from_load_error() {
        let inner = LoadError::SchemaError {
            path: PathBuf::from("system/buses.json"),
            field: "voltage".to_string(),
            message: "must be positive".to_string(),
        };
        let err: SddpError = inner.into();
        assert!(matches!(err, SddpError::Io(_)));
    }

    #[test]
    fn from_comm_error_wraps_directly() {
        let inner = CommError::InvalidCommunicator;
        let err: SddpError = inner.into();
        assert!(matches!(
            err,
            SddpError::Communication(CommError::InvalidCommunicator)
        ));
        let msg = err.to_string();
        assert!(msg.contains("MPI"), "{msg}");
    }

    #[test]
    fn from_fpha_fitting_error_wraps_as_validation() {
        let inner = FphaFittingError::InsufficientPoints {
            hydro_name: "Itaipu".to_string(),
            count: 1,
        };
        let display_msg = inner.to_string();
        let err: SddpError = inner.into();
        assert!(
            matches!(err, SddpError::Validation(ref msg) if *msg == display_msg),
            "expected Validation wrapping the FphaFittingError display output, got {err:?}"
        );
    }

    #[test]
    fn sddp_error_satisfies_std_error_trait() {
        let variants: Vec<SddpError> = vec![
            SddpError::Solver(SolverError::Infeasible),
            SddpError::Communication(CommError::InvalidCommunicator),
            SddpError::Stochastic(StochasticError::InsufficientData {
                context: "no data".to_string(),
            }),
            SddpError::Io(LoadError::ConstraintError {
                description: "cycle".to_string(),
            }),
            SddpError::Validation("bad config".to_string()),
            SddpError::Infeasible {
                stage: 0,
                iteration: 1,
                scenario: 0,
            },
            SddpError::Simulation("simulation phase failed".to_string()),
            SddpError::WireVersionMismatch {
                encoded: 0,
                expected: 1,
            },
            SddpError::PolicySoftwareMismatch {
                policy_software: None,
                policy_version: "0.0.1".to_string(),
            },
        ];
        for err in &variants {
            let _: &dyn std::error::Error = err;
        }
    }

    #[test]
    fn all_variants_debug_non_empty() {
        let variants: Vec<SddpError> = vec![
            SddpError::Solver(SolverError::Unbounded),
            SddpError::Communication(CommError::InvalidCommunicator),
            SddpError::Stochastic(StochasticError::InvalidCorrelation {
                profile_name: "test".to_string(),
                reason: "bad value".to_string(),
            }),
            SddpError::Io(LoadError::ConstraintError {
                description: "test".to_string(),
            }),
            SddpError::Validation("test validation".to_string()),
            SddpError::Infeasible {
                stage: 1,
                iteration: 2,
                scenario: 3,
            },
            SddpError::Simulation("test simulation error".to_string()),
            SddpError::WireVersionMismatch {
                encoded: 0,
                expected: 1,
            },
            SddpError::PolicySoftwareMismatch {
                policy_software: None,
                policy_version: "0.0.1".to_string(),
            },
        ];
        for err in &variants {
            assert!(!format!("{err:?}").is_empty());
        }
    }

    #[test]
    fn stochastic_refusals_classify_as_invalid_input() {
        let refusals = [
            StochasticError::InvalidParParameters {
                hydro_id: 1,
                stage_id: 2,
                reason: "bad order".to_string(),
            },
            StochasticError::InvalidCorrelation {
                profile_name: "test".to_string(),
                reason: "bad value".to_string(),
            },
            StochasticError::InsufficientData {
                context: "no data".to_string(),
            },
            StochasticError::UnsupportedNoiseMethod {
                method: "sobol".to_string(),
                stage_id: 0,
                reason: "unsupported".to_string(),
            },
            StochasticError::DimensionExceedsCapacity {
                dim: 10,
                max_dim: 4,
                method: "sobol".to_string(),
            },
            StochasticError::MissingScenarioSource {
                scheme: "historical".to_string(),
                reason: "no history".to_string(),
            },
        ];
        for refusal in refusals {
            let err = SddpError::Stochastic(refusal);
            assert_eq!(err.class(), ErrorClass::InvalidInput, "{err}");
        }
    }

    #[test]
    fn every_error_variant_has_its_class() {
        let table = [
            (
                SddpError::Stochastic(StochasticError::InsufficientData {
                    context: "no data".to_string(),
                }),
                ErrorClass::InvalidInput,
            ),
            (
                SddpError::Validation("bad config".to_string()),
                ErrorClass::InvalidInput,
            ),
            (
                SddpError::Io(LoadError::ParseError {
                    path: PathBuf::from("config.json"),
                    message: "unexpected end of input".to_string(),
                }),
                ErrorClass::InvalidInput,
            ),
            (
                SddpError::Io(LoadError::SchemaError {
                    path: PathBuf::from("system/buses.json"),
                    field: "voltage".to_string(),
                    message: "must be positive".to_string(),
                }),
                ErrorClass::InvalidInput,
            ),
            (
                SddpError::Io(LoadError::ConstraintError {
                    description: "cycle".to_string(),
                }),
                ErrorClass::InvalidInput,
            ),
            (
                SddpError::PolicySoftwareMismatch {
                    policy_software: None,
                    policy_version: "0.0.1".to_string(),
                },
                ErrorClass::IncompatiblePolicy,
            ),
            (
                SddpError::Io(LoadError::IoError {
                    path: PathBuf::from("system/hydros.json"),
                    source: std::io::Error::other("permission denied"),
                }),
                ErrorClass::Io,
            ),
            (
                SddpError::Infeasible {
                    stage: 0,
                    iteration: 1,
                    scenario: 0,
                },
                ErrorClass::Solver,
            ),
            (
                SddpError::Solver(SolverError::Infeasible),
                ErrorClass::Solver,
            ),
            (
                SddpError::Communication(CommError::InvalidCommunicator),
                ErrorClass::Internal,
            ),
            (
                SddpError::Simulation("output channel closed".to_string()),
                ErrorClass::Internal,
            ),
            (
                SddpError::WireVersionMismatch {
                    encoded: 0,
                    expected: 1,
                },
                ErrorClass::Internal,
            ),
            (
                SddpError::BasisShapeMismatch {
                    num_row: 10,
                    total_basic: 9,
                    col_basic: 4,
                    row_basic: 5,
                },
                ErrorClass::Internal,
            ),
            (
                SddpError::CheckpointWrite {
                    iteration: 4,
                    source: OutputError::IoError {
                        path: PathBuf::from("out/policy.staging"),
                        source: std::io::Error::other("no space left on device"),
                    },
                },
                ErrorClass::Io,
            ),
            (
                SddpError::CheckpointWrite {
                    iteration: 4,
                    source: OutputError::ForeignEntry {
                        dir: PathBuf::from("out/policy"),
                        entry: PathBuf::from("out/policy/notes.txt"),
                    },
                },
                ErrorClass::InvalidInput,
            ),
            (
                SddpError::CheckpointWrite {
                    iteration: 4,
                    source: OutputError::SerializationError {
                        entity: "stage_cuts".to_string(),
                        message: "buffer too large".to_string(),
                    },
                },
                ErrorClass::Internal,
            ),
        ];
        for (err, class) in table {
            assert_eq!(err.class(), class, "{err}");
        }
    }

    #[test]
    fn display_checkpoint_write_names_the_iteration_and_the_write_failure() {
        let err = SddpError::CheckpointWrite {
            iteration: 7,
            source: OutputError::IoError {
                path: PathBuf::from("out/policy.staging"),
                source: std::io::Error::other("no space left on device"),
            },
        };
        assert_eq!(
            err.to_string(),
            "checkpoint write at iteration 7 failed: I/O error accessing out/policy.staging: \
             no space left on device"
        );
    }
}
