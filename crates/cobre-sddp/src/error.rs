//! Error types for the `cobre-sddp` crate.

use cobre_comm::CommError;
use cobre_io::LoadError;
use cobre_io::scenarios::estimation::EstimationError;
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

    /// A policy checkpoint was written by a different cobre version than the
    /// running one; only same-version policies load.
    #[error(
        "policy was written by cobre {policy_version}, but this is cobre {running}; a policy \
         loads only in the cobre version that wrote it: retrain it, or re-export it, with \
         cobre {running}",
        running = crate::POLICY_COBRE_VERSION
    )]
    PolicyVersionMismatch {
        /// The `cobre_version` the checkpoint's manifest records.
        policy_version: String,
    },

    /// A stored basis in a policy checkpoint does not match the dimensions of the
    /// LP it would warm-start.
    #[error(
        "stored basis for node {node_id} does not match its LP: the LP has {expected_cols} \
         columns and {expected_template_rows} template rows, the stored basis has {found_cols} \
         columns and {found_rows} rows with {found_cut_rows} recorded cut rows; retrain the policy"
    )]
    StoredBasisDimensionMismatch {
        /// The node the stored basis was captured at.
        node_id: i32,
        /// Column count of the node's current LP template.
        expected_cols: usize,
        /// Column count the stored basis carries.
        found_cols: usize,
        /// Row count of the node's current LP template, before any cut rows.
        expected_template_rows: usize,
        /// Row count the stored basis carries.
        found_rows: usize,
        /// Cut-row count the checkpoint recorded for this basis.
        found_cut_rows: usize,
    },
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
    use super::SddpError;
    use cobre_comm::CommError;
    use cobre_io::LoadError;
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
    fn display_policy_version_mismatch_names_both_versions() {
        let err = SddpError::PolicyVersionMismatch {
            policy_version: "0.0.1".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("0.0.1"), "{msg}");
        assert!(msg.contains(crate::POLICY_COBRE_VERSION), "{msg}");
    }

    #[test]
    fn display_stored_basis_dimension_mismatch_names_node_and_all_dimensions() {
        let err = SddpError::StoredBasisDimensionMismatch {
            node_id: 3,
            expected_cols: 4,
            found_cols: 5,
            expected_template_rows: 3,
            found_rows: 6,
            found_cut_rows: 2,
        };
        let msg = err.to_string();
        for needle in ["3", "4", "5", "6", "2"] {
            assert!(msg.contains(needle), "{msg}");
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
            SddpError::PolicyVersionMismatch {
                policy_version: "0.0.1".to_string(),
            },
            SddpError::StoredBasisDimensionMismatch {
                node_id: 0,
                expected_cols: 100,
                found_cols: 90,
                expected_template_rows: 50,
                found_rows: 45,
                found_cut_rows: 10,
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
            SddpError::PolicyVersionMismatch {
                policy_version: "0.0.1".to_string(),
            },
            SddpError::StoredBasisDimensionMismatch {
                node_id: 0,
                expected_cols: 100,
                found_cols: 90,
                expected_template_rows: 50,
                found_rows: 45,
                found_cut_rows: 10,
            },
        ];
        for err in &variants {
            assert!(!format!("{err:?}").is_empty());
        }
    }
}
