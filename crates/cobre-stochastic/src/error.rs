//! Error types for the `cobre-stochastic` crate.
//!
//! All public APIs that can fail return [`StochasticError`] or a `Result`
//! wrapping it. The error variants cover the four failure domains of the
//! stochastic layer:
//!
//! - PAR model parameter validation
//! - Cholesky decomposition of correlation matrices
//! - Correlation profile validation
//! - Seed derivation for deterministic noise generation

/// Errors that can occur during stochastic model construction or scenario generation.
///
/// This type is returned by all fallible methods in `cobre-stochastic`.
/// It implements [`std::error::Error`] via the `thiserror` derive macro and
/// is both `Send` and `Sync`, making it safe to propagate across threads.
#[derive(Debug, thiserror::Error)]
pub enum StochasticError {
    /// A PAR model's parameters did not pass validation for a given hydro plant
    /// and stage combination.
    ///
    /// For example, the autoregressive order may exceed the number of available
    /// historical observations, or a coefficient matrix may be ill-conditioned.
    #[error("invalid PAR parameters for hydro {hydro_id} at stage {stage_id}: {reason}")]
    InvalidParParameters {
        /// Identifier of the hydro plant whose PAR parameters failed validation.
        hydro_id: i32,
        /// Stage index (1-based) at which the validation failure occurred.
        stage_id: i32,
        /// Human-readable description of why the parameters are invalid.
        reason: String,
    },

    /// The Cholesky decomposition of a correlation matrix failed.
    ///
    /// This typically indicates that the matrix is not positive-definite, which
    /// can result from numerical near-singularity or an invalid correlation profile.
    #[error("Cholesky decomposition failed for profile '{profile_name}': {reason}")]
    CholeskyDecompositionFailed {
        /// Name of the correlation profile whose matrix could not be decomposed.
        profile_name: String,
        /// Human-readable description of the decomposition failure.
        reason: String,
    },

    /// A correlation profile specification is invalid.
    ///
    /// For example, the profile may reference hydro plants that do not exist,
    /// contain entries outside `[-1.0, 1.0]`, or lack a unit diagonal.
    #[error("invalid correlation profile '{profile_name}': {reason}")]
    InvalidCorrelation {
        /// Name of the correlation profile that failed validation.
        profile_name: String,
        /// Human-readable description of why the profile is invalid.
        reason: String,
    },

    /// Required input data is missing or has insufficient observations.
    ///
    /// This error is raised when a computation cannot proceed because the
    /// data necessary to fit or apply a stochastic model is absent or too
    /// sparse (e.g., fewer historical inflow records than the PAR order).
    #[error("insufficient data: {context}")]
    InsufficientData {
        /// Description of what data is missing and where it was expected.
        context: String,
    },

    /// Deterministic seed derivation for noise generation failed.
    ///
    /// Seed derivation uses SipHash-1-3 to produce reproducible per-scenario
    /// seeds from a global base seed, scenario index, and stage index. This
    /// error is raised if the hash computation produces an invalid result.
    #[error("seed derivation failed: {reason}")]
    SeedDerivationError {
        /// Human-readable description of why seed derivation failed.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::StochasticError;

    fn assert_std_error<E: std::error::Error + Send + Sync + 'static>(_: &E) {}

    #[test]
    fn test_invalid_par_parameters_implements_std_error() {
        let err = StochasticError::InvalidParParameters {
            hydro_id: 1,
            stage_id: 12,
            reason: "coefficient matrix is singular".into(),
        };
        assert_std_error(&err);
        let display = format!("{err}");
        assert!(display.contains('1'));
        assert!(display.contains("12"));
        assert!(display.contains("singular"));
    }

    #[test]
    fn test_cholesky_decomposition_failed_implements_std_error() {
        let err = StochasticError::CholeskyDecompositionFailed {
            profile_name: "southeast_inflows".into(),
            reason: "matrix is not positive-definite".into(),
        };
        assert_std_error(&err);
        let display = format!("{err}");
        assert!(display.contains("southeast_inflows"));
        assert!(display.contains("positive-definite"));
    }

    #[test]
    fn test_invalid_correlation_implements_std_error() {
        let err = StochasticError::InvalidCorrelation {
            profile_name: "se_ne_correlation".into(),
            reason: "off-diagonal entry 1.5 is outside [-1.0, 1.0]".into(),
        };
        assert_std_error(&err);
        let display = format!("{err}");
        assert!(display.contains("se_ne_correlation"));
        assert!(display.contains("1.5"));
    }

    #[test]
    fn test_insufficient_data_implements_std_error() {
        let err = StochasticError::InsufficientData {
            context: "hydro 42 has 3 historical observations but PAR order is 4".into(),
        };
        assert_std_error(&err);
        let display = format!("{err}");
        assert!(display.contains("42"));
        assert!(display.contains("PAR order is 4"));
    }

    #[test]
    fn test_seed_derivation_error_implements_std_error() {
        let err = StochasticError::SeedDerivationError {
            reason: "hash output overflowed u64 accumulator".into(),
        };
        assert_std_error(&err);
        let display = format!("{err}");
        assert!(display.contains("overflowed"));
    }

    #[test]
    fn test_all_variants_debug() {
        let variants = [
            StochasticError::InvalidParParameters {
                hydro_id: 0,
                stage_id: 0,
                reason: String::new(),
            },
            StochasticError::CholeskyDecompositionFailed {
                profile_name: String::new(),
                reason: String::new(),
            },
            StochasticError::InvalidCorrelation {
                profile_name: String::new(),
                reason: String::new(),
            },
            StochasticError::InsufficientData {
                context: String::new(),
            },
            StochasticError::SeedDerivationError {
                reason: String::new(),
            },
        ];
        for v in &variants {
            let _ = format!("{v:?}");
        }
    }
}
