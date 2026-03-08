//! Error types for the `cobre-sddp` crate.
//!
//! [`SddpError`] is the single error type returned by all fallible SDDP
//! operations. It aggregates errors from dependency crates into a unified
//! type that is `Send + Sync + 'static`, making it safe to propagate across
//! threads and store in `Box<dyn Error>` contexts.
//!
//! ## Variant overview
//!
//! | Variant           | Origin                              |
//! |-------------------|-------------------------------------|
//! | `Solver`          | LP solve failure (`cobre-solver`)   |
//! | `Communication`   | MPI/comm failure (`cobre-comm`)     |
//! | `Stochastic`      | Scenario generation (`cobre-stochastic`) |
//! | `Io`              | Case loading failure (`cobre-io`)   |
//! | `Validation`      | SDDP configuration error            |
//! | `Infeasible`      | LP infeasibility after recourse     |

use cobre_io::LoadError;
use cobre_solver::SolverError;
use cobre_stochastic::StochasticError;

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
    /// Wraps a [`cobre_solver::SolverError`] that persisted through all
    /// retry attempts. The calling code should treat this as a hard stop
    /// unless the variant carries a usable partial solution.
    #[error("solver error: {0}")]
    Solver(#[from] SolverError),

    /// A distributed communication operation failed.
    ///
    /// The underlying [`cobre_comm::CommError`] is serialised to a `String`
    /// so that `SddpError` remains `Send + Sync` regardless of the backend's
    /// internal state.
    #[error("communication error: {0}")]
    Communication(String),

    /// Stochastic model construction or scenario generation failed.
    ///
    /// Wraps a [`cobre_stochastic::StochasticError`] from PAR model
    /// validation, Cholesky decomposition, or seed derivation.
    #[error("stochastic error: {0}")]
    Stochastic(#[from] StochasticError),

    /// Case directory loading or validation failed.
    ///
    /// Wraps a [`cobre_io::LoadError`] from any layer of the five-stage
    /// validation pipeline.
    #[error("I/O error: {0}")]
    Io(#[from] LoadError),

    /// SDDP configuration is invalid.
    ///
    /// Covers semantic errors detected at algorithm startup that are not
    /// already caught by the upstream loading pipeline (e.g., `forward_passes`
    /// is zero, `max_iterations` overflows the cut pool, or required fields
    /// are inconsistent with the loaded system).
    #[error("configuration validation error: {0}")]
    Validation(String),

    /// An LP subproblem was infeasible after all recourse actions were applied.
    ///
    /// This differs from [`SddpError::Solver`] (which covers numerical and
    /// timeout failures) in that the subproblem has provably no feasible
    /// solution. The training loop must perform a hard stop when it receives
    /// this variant.
    #[error("infeasible subproblem at stage {stage}, iteration {iteration}, scenario {scenario}")]
    Infeasible {
        /// Stage index (0-based) at which infeasibility was detected.
        stage: usize,
        /// Iteration number (1-based) at which infeasibility was detected.
        iteration: u64,
        /// Scenario index (0-based) in the forward pass that triggered infeasibility.
        scenario: usize,
    },
}

impl From<cobre_comm::CommError> for SddpError {
    fn from(err: cobre_comm::CommError) -> Self {
        Self::Communication(err.to_string())
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
        let err = SddpError::Communication("allgatherv timed out".to_string());
        let msg = err.to_string();
        assert!(msg.contains("communication"), "{msg}");
        assert!(msg.contains("allgatherv timed out"), "{msg}");
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
        let inner = StochasticError::SeedDerivationError {
            reason: "hash overflow".to_string(),
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
    fn from_comm_error_wraps_as_string() {
        let inner = CommError::InvalidCommunicator;
        let err: SddpError = inner.into();
        assert!(matches!(err, SddpError::Communication(_)));
        let msg = err.to_string();
        assert!(msg.contains("MPI"), "{msg}");
    }

    #[test]
    fn sddp_error_satisfies_std_error_trait() {
        let variants: Vec<SddpError> = vec![
            SddpError::Solver(SolverError::Infeasible),
            SddpError::Communication("network partition".to_string()),
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
        ];
        for err in &variants {
            let _: &dyn std::error::Error = err;
        }
    }

    #[test]
    fn all_variants_debug_non_empty() {
        let variants: Vec<SddpError> = vec![
            SddpError::Solver(SolverError::Unbounded),
            SddpError::Communication("test comm error".to_string()),
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
        ];
        for err in &variants {
            assert!(!format!("{err:?}").is_empty());
        }
    }
}
