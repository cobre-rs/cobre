//! Error types for the `cobre-stochastic` crate.
//!
//! All public APIs that can fail return [`StochasticError`] or a `Result`
//! wrapping it. The error variants cover the six failure domains of the
//! stochastic layer:
//!
//! - PAR model parameter validation
//! - Cholesky decomposition of correlation matrices
//! - Correlation profile validation
//! - Seed derivation for deterministic noise generation
//! - Noise method dispatch (tree generation)
//! - Sampling scheme dispatch (forward sampler factory)

/// Errors that can occur during stochastic model construction or scenario generation.
///
/// This type is returned by all fallible methods in `cobre-stochastic`.
/// It implements [`std::error::Error`] via the `thiserror` derive macro and
/// is both `Send` and `Sync`, making it safe to propagate across threads.
#[derive(Debug, thiserror::Error)]
pub enum StochasticError {
    /// PAR model parameters did not pass validation (e.g., AR order exceeds
    /// available observations or coefficient matrix is ill-conditioned).
    #[error("invalid PAR parameters for hydro {hydro_id} at stage {stage_id}: {reason}")]
    InvalidParParameters {
        /// Identifier of the hydro plant.
        hydro_id: i32,
        /// Stage index where validation failed.
        stage_id: i32,
        /// Error description.
        reason: String,
    },

    /// Cholesky decomposition of a correlation matrix failed (not positive-definite).
    #[error("Cholesky decomposition failed for profile '{profile_name}': {reason}")]
    CholeskyDecompositionFailed {
        /// Name of the correlation profile.
        profile_name: String,
        /// Error description.
        reason: String,
    },

    /// Correlation profile specification is invalid (e.g., entries outside [-1.0, 1.0]).
    #[error("invalid correlation profile '{profile_name}': {reason}")]
    InvalidCorrelation {
        /// Name of the correlation profile.
        profile_name: String,
        /// Error description.
        reason: String,
    },

    /// Required input data is missing or insufficient.
    #[error("insufficient data: {context}")]
    InsufficientData {
        /// Description of missing data.
        context: String,
    },

    /// Deterministic seed derivation for noise generation failed.
    #[error("seed derivation failed: {reason}")]
    SeedDerivationError {
        /// Error description.
        reason: String,
    },

    /// Noise method requested for a given stage is not supported.
    #[error("unsupported noise method '{method}' at stage {stage_id}: {reason}")]
    UnsupportedNoiseMethod {
        /// Name of the noise method.
        method: String,
        /// Stage ID where method was requested.
        stage_id: i32,
        /// Error description.
        reason: String,
    },

    /// Noise dimension exceeds maximum supported by the method (e.g., Sobol limits).
    #[error("noise dimension {dim} exceeds maximum supported dimension {max_dim} for {method}")]
    DimensionExceedsCapacity {
        /// Requested noise dimension.
        dim: usize,
        /// Maximum supported dimension.
        max_dim: usize,
        /// Name of the noise method.
        method: String,
    },

    /// Sampling scheme requested from the forward sampler factory is not supported.
    #[error("unsupported sampling scheme '{scheme}': {reason}")]
    UnsupportedSamplingScheme {
        /// Name of the sampling scheme.
        scheme: String,
        /// Error description.
        reason: String,
    },

    /// Required scenario source is absent for the requested sampling scheme.
    #[error("missing scenario source for scheme '{scheme}': {reason}")]
    MissingScenarioSource {
        /// Name of the sampling scheme.
        scheme: String,
        /// Error description.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::StochasticError;

    fn assert_std_error<E: std::error::Error + Send + Sync + 'static>(_: &E) {}

    fn assert_all_variants_debug(err: &StochasticError) {
        match err {
            StochasticError::InvalidParParameters { .. }
            | StochasticError::CholeskyDecompositionFailed { .. }
            | StochasticError::InvalidCorrelation { .. }
            | StochasticError::InsufficientData { .. }
            | StochasticError::SeedDerivationError { .. }
            | StochasticError::UnsupportedNoiseMethod { .. }
            | StochasticError::DimensionExceedsCapacity { .. }
            | StochasticError::UnsupportedSamplingScheme { .. }
            | StochasticError::MissingScenarioSource { .. } => {}
        }
        let _ = format!("{err:?}");
    }

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
            StochasticError::UnsupportedNoiseMethod {
                method: String::new(),
                stage_id: 0,
                reason: String::new(),
            },
            StochasticError::DimensionExceedsCapacity {
                dim: 0,
                max_dim: 0,
                method: String::new(),
            },
            StochasticError::UnsupportedSamplingScheme {
                scheme: String::new(),
                reason: String::new(),
            },
            StochasticError::MissingScenarioSource {
                scheme: String::new(),
                reason: String::new(),
            },
        ];
        for v in &variants {
            assert_all_variants_debug(v);
        }
    }

    #[test]
    fn test_unsupported_noise_method_display() {
        let err = StochasticError::UnsupportedNoiseMethod {
            method: "selective".into(),
            stage_id: 5,
            reason: "not implemented".into(),
        };
        let display = format!("{err}");
        assert!(display.contains("selective"));
        assert!(display.contains('5'));
        assert!(display.contains("not implemented"));
    }

    #[test]
    fn test_dimension_exceeds_capacity_display() {
        let err = StochasticError::DimensionExceedsCapacity {
            dim: 25000,
            max_dim: 21201,
            method: "sobol".into(),
        };
        let display = format!("{err}");
        assert!(display.contains("25000"));
        assert!(display.contains("21201"));
        assert!(display.contains("sobol"));
    }

    #[test]
    fn test_unsupported_sampling_scheme_display() {
        let err = StochasticError::UnsupportedSamplingScheme {
            scheme: "historical".into(),
            reason: "not yet implemented".into(),
        };
        let display = format!("{err}");
        assert!(display.contains("historical"));
        assert!(display.contains("not yet implemented"));
    }

    #[test]
    fn test_missing_scenario_source_display() {
        let err = StochasticError::MissingScenarioSource {
            scheme: "external".into(),
            reason: "no library loaded".into(),
        };
        let display = format!("{err}");
        assert!(display.contains("external"));
        assert!(display.contains("no library loaded"));
    }
}
