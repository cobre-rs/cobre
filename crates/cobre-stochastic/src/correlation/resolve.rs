//! Resolution of correlation profiles from case input data.
//!
//! Maps correlation profile specifications from the loaded case into
//! dense matrices indexed by the internal hydro plant registry order.
//! Validates that all referenced hydro plants exist in the registry and
//! that correlation entries are in the valid range `[-1.0, 1.0]`.
//!
//! Returns [`StochasticError::InvalidCorrelation`] when validation fails.
//!
//! [`StochasticError::InvalidCorrelation`]: crate::StochasticError::InvalidCorrelation
