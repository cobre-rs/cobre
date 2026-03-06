//! Cholesky decomposition of correlation matrices.
//!
//! Computes the lower-triangular Cholesky factor `L` of a symmetric
//! positive-definite correlation matrix `C` such that `C = L Lᵀ`.
//! The factor is used to transform vectors of independent standard-normal
//! noise into spatially correlated samples.
//!
//! Returns [`StochasticError::CholeskyDecompositionFailed`] when the
//! input matrix is not positive-definite.
//!
//! [`StochasticError::CholeskyDecompositionFailed`]: crate::StochasticError::CholeskyDecompositionFailed
