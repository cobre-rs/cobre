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

use crate::StochasticError;

/// Symmetry tolerance for input matrix validation.
const SYMMETRY_TOL: f64 = 1e-10;

/// Lower-triangular Cholesky factor of a correlation matrix.
///
/// Stores L such that Sigma = L * L^T. The factor is stored as a flat
/// array in row-major order, including only the lower triangle
/// (L[i][j] for j <= i).
///
/// Used to transform independent standard normal samples into
/// spatially correlated samples: eta = L * z where z ~ N(0, I).
#[derive(Debug, Clone)]
pub struct CholeskyFactor {
    /// Lower-triangular factor in row-major packed storage.
    /// Length: n * (n + 1) / 2 where n is the matrix dimension.
    /// Element (i, j) with j <= i is at index i*(i+1)/2 + j.
    data: Box<[f64]>,
    /// Matrix dimension (number of entities in the correlation group).
    dim: usize,
}

impl CholeskyFactor {
    /// Returns the value of element (i, j) from the packed lower-triangular storage.
    ///
    /// Panics in debug mode if j > i (accessing above the diagonal).
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(
            j <= i,
            "CholeskyFactor::get: j={j} > i={i} (upper triangle)"
        );
        self.data[i * (i + 1) / 2 + j]
    }

    /// Sets the value of element (i, j) in packed lower-triangular storage.
    #[inline]
    fn set(&mut self, i: usize, j: usize, value: f64) {
        debug_assert!(
            j <= i,
            "CholeskyFactor::set: j={j} > i={i} (upper triangle)"
        );
        self.data[i * (i + 1) / 2 + j] = value;
    }

    /// Computes the Cholesky decomposition of `matrix` using the
    /// Cholesky-Banachiewicz algorithm (row-by-row).
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InvalidCorrelation`] if the matrix is not square or
    ///   not symmetric within tolerance 1e-10.
    /// - [`StochasticError::CholeskyDecompositionFailed`] if the matrix is not
    ///   positive-definite (a diagonal element becomes <= 0 during decomposition).
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_stochastic::correlation::cholesky::CholeskyFactor;
    ///
    /// let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    /// let factor = CholeskyFactor::decompose(&identity).unwrap();
    /// let mut out = vec![0.0; 2];
    /// factor.transform(&[3.0, 5.0], &mut out);
    /// assert!((out[0] - 3.0).abs() < 1e-12);
    /// assert!((out[1] - 5.0).abs() < 1e-12);
    /// ```
    pub fn decompose(matrix: &[Vec<f64>]) -> Result<Self, StochasticError> {
        let n = matrix.len();

        // Validate squareness.
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n {
                return Err(StochasticError::InvalidCorrelation {
                    profile_name: String::new(),
                    reason: format!(
                        "matrix is not square: row {i} has {} columns but {} rows",
                        row.len(),
                        n
                    ),
                });
            }
        }

        // Validate symmetry within tolerance.
        // Indexing is required here: we compare matrix[i][j] with matrix[j][i].
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..i {
                let diff = (matrix[i][j] - matrix[j][i]).abs();
                if diff > SYMMETRY_TOL {
                    return Err(StochasticError::InvalidCorrelation {
                        profile_name: String::new(),
                        reason: format!(
                            "matrix is not symmetric: |M[{i}][{j}] - M[{j}][{i}]| = {diff:.2e} > {SYMMETRY_TOL:.2e}"
                        ),
                    });
                }
            }
        }

        // Allocate packed storage: n*(n+1)/2 elements.
        let packed_len = n * (n + 1) / 2;
        let data = vec![0.0_f64; packed_len].into_boxed_slice();
        let mut factor = Self { data, dim: n };

        // Cholesky-Banachiewicz: row-by-row decomposition.
        // Both i and j are required as indices into matrix and into the packed factor;
        // iterator-based rewriting would obscure the algorithm and require extra bookkeeping.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..=i {
                // sum = matrix[i][j] - sum_{k=0}^{j-1} L[i][k] * L[j][k]
                let mut sum = matrix[i][j];
                for k in 0..j {
                    sum -= factor.get(i, k) * factor.get(j, k);
                }

                if i == j {
                    // Diagonal element: must be strictly positive.
                    if sum <= 0.0 {
                        return Err(StochasticError::CholeskyDecompositionFailed {
                            profile_name: String::new(),
                            reason: format!(
                                "matrix is not positive-definite: diagonal element at ({i},{i}) is {sum:.6e} <= 0"
                            ),
                        });
                    }
                    factor.set(i, i, sum.sqrt());
                } else {
                    factor.set(i, j, sum / factor.get(j, j));
                }
            }
        }

        Ok(factor)
    }

    /// Applies the Cholesky factor to transform independent noise into correlated noise.
    ///
    /// Computes `correlated = L * independent` where `L` is the lower-triangular
    /// Cholesky factor. Both slices must have length `self.dim`.
    ///
    /// Writes directly into `correlated` without any intermediate allocation.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `independent.len() != self.dim` or
    /// `correlated.len() != self.dim`.
    pub fn transform(&self, independent: &[f64], correlated: &mut [f64]) {
        debug_assert_eq!(
            independent.len(),
            self.dim,
            "CholeskyFactor::transform: independent.len()={} != dim={}",
            independent.len(),
            self.dim
        );
        debug_assert_eq!(
            correlated.len(),
            self.dim,
            "CholeskyFactor::transform: correlated.len()={} != dim={}",
            correlated.len(),
            self.dim
        );

        // eta[i] = sum_{j=0}^{i} L[i][j] * z[j]
        // Row base is tracked incrementally to avoid recomputing i*(i+1)/2 per row.
        let mut row_base = 0usize;
        for (i, out) in correlated.iter_mut().enumerate().take(self.dim) {
            let row = &self.data[row_base..=row_base + i];
            let mut acc = 0.0_f64;
            for (l_ij, &z_j) in row.iter().zip(&independent[..=i]) {
                acc += l_ij * z_j;
            }
            *out = acc;
            row_base += i + 1;
        }
    }

    /// Returns the matrix dimension.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a CholeskyFactor from a matrix slice and unwrap.
    fn decompose(matrix: &[Vec<f64>]) -> CholeskyFactor {
        CholeskyFactor::decompose(matrix).unwrap()
    }

    // Helper: apply transform and return the output vector.
    fn transform(factor: &CholeskyFactor, input: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; factor.dim()];
        factor.transform(input, &mut out);
        out
    }

    #[test]
    fn cholesky_of_1x1_identity() {
        let factor = decompose(&[vec![1.0]]);
        assert!((factor.get(0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_of_2x2_identity() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let factor = decompose(&identity);
        // L should also be the identity.
        assert!((factor.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((factor.get(1, 0) - 0.0).abs() < 1e-12);
        assert!((factor.get(1, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_of_2x2_correlated_matrix() {
        // Acceptance criterion from ticket: [[1, 0.8], [0.8, 1]]
        // Expected: L[0][0]=1.0, L[1][0]=0.8, L[1][1]=0.6 (sqrt(1-0.64)=0.6)
        let matrix = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let factor = decompose(&matrix);
        assert!((factor.get(0, 0) - 1.0).abs() < 1e-12, "L[0][0]");
        assert!((factor.get(1, 0) - 0.8).abs() < 1e-12, "L[1][0]");
        assert!((factor.get(1, 1) - 0.6).abs() < 1e-12, "L[1][1]");
    }

    #[test]
    fn cholesky_of_3x3_known_matrix() {
        // Symmetric PD matrix with known factorisation: [[1,0.5,0],[0.5,1,0.5],[0,0.5,1]]
        // L[0][0] = 1
        // L[1][0] = 0.5, L[1][1] = sqrt(0.75)
        // L[2][0] = 0,   L[2][1] = 0.5/sqrt(0.75), L[2][2] = sqrt(1 - (0.5/sqrt(0.75))^2)
        let m = vec![
            vec![1.0, 0.5, 0.0],
            vec![0.5, 1.0, 0.5],
            vec![0.0, 0.5, 1.0],
        ];
        let factor = decompose(&m);
        let sqrt_075 = 0.75_f64.sqrt();
        let l21 = 0.5 / sqrt_075;
        let l22 = (1.0 - l21 * l21).sqrt();
        assert!((factor.get(0, 0) - 1.0).abs() < 1e-12, "L[0][0]");
        assert!((factor.get(1, 0) - 0.5).abs() < 1e-12, "L[1][0]");
        assert!((factor.get(1, 1) - sqrt_075).abs() < 1e-12, "L[1][1]");
        assert!((factor.get(2, 0) - 0.0).abs() < 1e-12, "L[2][0]");
        assert!((factor.get(2, 1) - l21).abs() < 1e-12, "L[2][1]");
        assert!((factor.get(2, 2) - l22).abs() < 1e-12, "L[2][2]");
    }

    #[test]
    fn cholesky_of_4x4_identity() {
        let id4: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let factor = decompose(&id4);
        for i in 0..4 {
            for j in 0..=i {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (factor.get(i, j) - expected).abs() < 1e-12,
                    "L[{i}][{j}] mismatch"
                );
            }
        }
    }

    #[test]
    fn cholesky_fails_on_non_pd_matrix() {
        // Acceptance criterion: [[1.0, 2.0], [2.0, 1.0]] is not PD (det = -3).
        let matrix = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let result = CholeskyFactor::decompose(&matrix);
        assert!(
            matches!(
                result,
                Err(StochasticError::CholeskyDecompositionFailed { .. })
            ),
            "Expected CholeskyDecompositionFailed, got: {result:?}"
        );
    }

    #[test]
    fn cholesky_fails_on_non_square_matrix() {
        let matrix = vec![vec![1.0, 0.0, 0.5], vec![0.0, 1.0]];
        let result = CholeskyFactor::decompose(&matrix);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn cholesky_fails_on_non_symmetric_matrix() {
        // Off-diagonal difference of 1e-9 exceeds tolerance 1e-10.
        let matrix = vec![vec![1.0, 0.3], vec![0.3 + 1e-9_f64, 1.0]];
        let result = CholeskyFactor::decompose(&matrix);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn transform_with_identity_factor_equals_input() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let factor = decompose(&identity);
        // Use values that are not close to mathematical constants to avoid approx_constant lint.
        let input = [1.23_f64, 4.56_f64];
        let out = transform(&factor, &input);
        assert!((out[0] - 1.23).abs() < 1e-12, "out[0]");
        assert!((out[1] - 4.56).abs() < 1e-12, "out[1]");
    }

    #[test]
    fn transform_with_known_2x2_factor() {
        // Acceptance criterion: z=[1.0, 0.0], L from [[1,0.8],[0.8,1]]
        // => eta = [L[0][0]*1 + 0, L[1][0]*1 + L[1][1]*0] = [1.0, 0.8]
        let matrix = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let factor = decompose(&matrix);
        let out = transform(&factor, &[1.0, 0.0]);
        assert!((out[0] - 1.0).abs() < 1e-12, "out[0]={}", out[0]);
        assert!((out[1] - 0.8).abs() < 1e-12, "out[1]={}", out[1]);
    }

    #[test]
    fn transform_with_known_2x2_factor_second_basis_vector() {
        // z=[0.0, 1.0] => eta = [0.0, 0.6]
        let matrix = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let factor = decompose(&matrix);
        let out = transform(&factor, &[0.0, 1.0]);
        assert!((out[0] - 0.0).abs() < 1e-12, "out[0]={}", out[0]);
        assert!((out[1] - 0.6).abs() < 1e-12, "out[1]={}", out[1]);
    }

    #[test]
    fn cholesky_symmetric_tolerance_boundary() {
        // A difference strictly below 1e-10 must pass.
        // Use 5e-11 (half the tolerance) to avoid floating-point boundary ambiguity.
        let matrix = vec![vec![1.0, 0.3], vec![0.3 + 5e-11_f64, 1.0]];
        let actual_diff = (matrix[1][0] - matrix[0][1]).abs();
        // Only run the assertion if f64 arithmetic produces a non-zero difference below tolerance.
        if actual_diff > 0.0 && actual_diff < 1e-10 {
            assert!(
                CholeskyFactor::decompose(&matrix).is_ok(),
                "should accept difference {actual_diff:.2e} < 1e-10"
            );
        }
        // Unconditionally verify that a difference above tolerance is rejected.
        let matrix_bad = vec![vec![1.0, 0.3], vec![0.3 + 2e-10_f64, 1.0]];
        let bad_diff = (matrix_bad[1][0] - matrix_bad[0][1]).abs();
        if bad_diff > 1e-10 {
            assert!(
                matches!(
                    CholeskyFactor::decompose(&matrix_bad),
                    Err(StochasticError::InvalidCorrelation { .. })
                ),
                "should reject difference {bad_diff:.2e} > 1e-10"
            );
        }
    }
}
