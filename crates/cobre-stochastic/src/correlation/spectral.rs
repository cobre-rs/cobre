//! Spectral decomposition of correlation matrices.
//!
//! Computes the symmetric matrix square root `D` of a symmetric correlation
//! matrix `C` such that `C = D * Dᵀ`. Unlike Cholesky factorisation, which
//! requires the input to be strictly positive-definite, the spectral approach
//! eigendecomposes `C = V * diag(λ) * Vᵀ`, clips any negative eigenvalues to
//! 0.0, and computes `D = V * diag(√λ) * Vᵀ`. This handles non-positive-
//! definite and rank-deficient matrices naturally.
//!
//! The NEWAVE-style transform `b = D * z` where `z ~ N(0, I)` produces
//! spatially correlated noise `b` with covariance `D * Dᵀ = C` (or the nearest
//! positive-semidefinite approximation to `C` in the spectral sense).

use crate::StochasticError;

/// Symmetry tolerance for input matrix validation.
const SYMMETRY_TOL: f64 = 1e-10;

/// Symmetric matrix square root of a correlation matrix (spectral factor).
///
/// Stores `D` such that `Sigma ≈ D * Dᵀ`, where `D = V * diag(√λ) * Vᵀ` is
/// computed from the eigendecomposition `Sigma = V * diag(λ) * Vᵀ`. Negative
/// eigenvalues are clipped to 0.0 before taking the square root, so the factor
/// represents the nearest positive-semidefinite matrix in the spectral sense.
///
/// The factor is stored as a dense `n × n` matrix in row-major order.
///
/// Used to transform independent standard-normal samples into spatially
/// correlated samples: `b = D * z` where `z ~ N(0, I)`.
#[derive(Debug, Clone)]
pub struct SpectralFactor {
    /// Dense n × n transform matrix D in row-major order.
    /// Length: n * n where n is the matrix dimension.
    /// Element (i, j) is at index i * n + j.
    data: Box<[f64]>,
    /// Matrix dimension (number of entities in the correlation group).
    dim: usize,
}

impl SpectralFactor {
    /// Computes the spectral factor of `matrix` using cyclic Jacobi
    /// eigendecomposition.
    ///
    /// Eigendecomposes the input as `C = V * diag(λ) * Vᵀ`, clips negative
    /// eigenvalues to 0.0 (logging a warning if any are clipped), and returns
    /// `D = V * diag(√λ) * Vᵀ` such that `D * Dᵀ ≈ C`.
    ///
    /// Unlike Cholesky decomposition, this method succeeds for non-positive-
    /// definite and rank-deficient matrices.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InvalidCorrelation`] if the matrix is not square or
    ///   not symmetric within tolerance 1e-10.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_stochastic::correlation::spectral::SpectralFactor;
    ///
    /// let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    /// let factor = SpectralFactor::decompose(&identity).unwrap();
    /// let mut out = vec![0.0; 2];
    /// factor.transform(&[3.0, 5.0], &mut out);
    /// assert!((out[0] - 3.0).abs() < 1e-10);
    /// assert!((out[1] - 5.0).abs() < 1e-10);
    /// ```
    pub fn decompose(matrix: &[Vec<f64>]) -> Result<Self, StochasticError> {
        let n = matrix.len();

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

        let mut work = vec![0.0_f64; n * n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                work[i * n + j] = matrix[i][j];
            }
        }

        let (mut lambdas, v) = jacobi_eigen(&mut work, n);

        let mut clipped_count = 0_usize;
        let mut largest_magnitude = 0.0_f64;
        for lambda in &mut lambdas {
            if *lambda < 0.0 {
                clipped_count += 1;
                let mag = lambda.abs();
                if mag > largest_magnitude {
                    largest_magnitude = mag;
                }
                *lambda = 0.0;
            }
        }
        if clipped_count > 0 {
            tracing::warn!(
                clipped_eigenvalues = clipped_count,
                largest_negative_magnitude = largest_magnitude,
                dim = n,
                "spectral decomposition clipped negative eigenvalues to 0.0; \
                 correlation matrix was not positive-semidefinite \
                 (D*D^T is the nearest PSD approximation)"
            );
        }

        let sqrt_lambdas: Vec<f64> = lambdas.iter().map(|&l| l.sqrt()).collect();
        let mut d = vec![0.0_f64; n * n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for k in 0..n {
                    acc += v[i * n + k] * sqrt_lambdas[k] * v[j * n + k];
                }
                d[i * n + j] = acc;
            }
        }

        Ok(Self {
            data: d.into_boxed_slice(),
            dim: n,
        })
    }

    /// Applies the spectral factor to transform independent noise into correlated noise.
    ///
    /// Computes `correlated = D * independent` where `D` is the symmetric
    /// matrix square root. Both slices must have length `self.dim`.
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
            "SpectralFactor::transform: independent.len()={} != dim={}",
            independent.len(),
            self.dim
        );
        debug_assert_eq!(
            correlated.len(),
            self.dim,
            "SpectralFactor::transform: correlated.len()={} != dim={}",
            correlated.len(),
            self.dim
        );

        let n = self.dim;
        for (i, out) in correlated.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            let row = &self.data[i * n..(i + 1) * n];
            for (d_ij, &z_j) in row.iter().zip(independent) {
                acc += d_ij * z_j;
            }
            *out = acc;
        }
    }

    /// Returns the matrix dimension.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Cyclic Jacobi eigendecomposition for a symmetric `n × n` matrix.
///
/// Diagonalises the symmetric matrix `a` (given as a flat row-major slice of
/// length `n * n`) using the classical cyclic Jacobi algorithm. The input
/// matrix is modified in-place; on return it is approximately diagonal.
///
/// Returns `(eigenvalues, eigenvectors)` where:
/// - `eigenvalues` is a `Vec<f64>` of length `n` (the diagonal of the
///   diagonalised matrix).
/// - `eigenvectors` is a flat row-major `Vec<f64>` of length `n * n`; the
///   `k`-th eigenvector occupies column `k`, i.e. `V[i][k] = eigenvectors[i * n + k]`.
///
/// Convergence criterion: the off-diagonal Frobenius norm drops below 1e-12.
/// Maximum iterations: `100 * n * n` sweeps. If the maximum is reached without
/// convergence, a warning is logged and the current (approximately diagonal)
/// state is returned.
///
/// Variable names use single characters (a, v, t, c, s, tau, a_pq, etc.) to
/// match the mathematical specification; renaming would obscure the derivation.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn jacobi_eigen(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Initialise the eigenvector matrix as identity.
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 100 * n * n;

    for sweep in 0..max_sweeps {
        let mut off_norm_sq = 0.0_f64;
        for i in 0..n {
            for j in 0..i {
                let val = a[i * n + j];
                off_norm_sq += 2.0 * val * val;
            }
        }

        if off_norm_sq < 1e-24 {
            break;
        }

        if sweep == max_sweeps - 1 {
            tracing::warn!(
                dim = n,
                max_sweeps,
                off_diagonal_norm = off_norm_sq.sqrt(),
                "Jacobi eigendecomposition did not converge within the maximum \
                 number of sweeps; returning approximate result"
            );
            break;
        }

        for p in 0..n {
            for q in p + 1..n {
                let a_pq = a[p * n + q];
                if a_pq.abs() < f64::EPSILON {
                    continue;
                }

                let a_pp = a[p * n + p];
                let a_qq = a[q * n + q];
                let tau = (a_qq - a_pp) / (2.0 * a_pq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * n + p] = a_pp - t * a_pq;
                a[q * n + q] = a_qq + t * a_pq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let a_rp = a[r * n + p];
                    let a_rq = a[r * n + q];
                    let new_rp = c * a_rp - s * a_rq;
                    let new_rq = s * a_rp + c * a_rq;
                    a[r * n + p] = new_rp;
                    a[p * n + r] = new_rp;
                    a[r * n + q] = new_rq;
                    a[q * n + r] = new_rq;
                }

                for r in 0..n {
                    let v_rp = v[r * n + p];
                    let v_rq = v[r * n + q];
                    v[r * n + p] = c * v_rp - s * v_rq;
                    v[r * n + q] = s * v_rp + c * v_rq;
                }
            }
        }
    }

    let eigenvalues = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn factor_gram(factor: &SpectralFactor) -> Vec<f64> {
        let n = factor.dim();
        let d = &factor.data;
        let mut result = vec![0.0_f64; n * n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for k in 0..n {
                    acc += d[i * n + k] * d[j * n + k];
                }
                result[i * n + j] = acc;
            }
        }
        result
    }

    fn assert_gram_equals(factor: &SpectralFactor, matrix: &[Vec<f64>], tol: f64) {
        let n = factor.dim();
        let gram = factor_gram(factor);
        for i in 0..n {
            for j in 0..n {
                let expected = matrix[i][j];
                let actual = gram[i * n + j];
                assert!(
                    (actual - expected).abs() <= tol,
                    "D*D^T[{i}][{j}] = {actual:.6e}, expected {expected:.6e}, diff = {:.6e}",
                    (actual - expected).abs()
                );
            }
        }
    }

    fn decompose(matrix: &[Vec<f64>]) -> SpectralFactor {
        SpectralFactor::decompose(matrix).unwrap()
    }

    fn transform(factor: &SpectralFactor, input: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; factor.dim()];
        factor.transform(input, &mut out);
        out
    }

    #[test]
    fn spectral_of_1x1_identity() {
        let factor = decompose(&[vec![1.0]]);
        assert_eq!(factor.dim(), 1);
        assert!(
            (factor.data[0] - 1.0).abs() < 1e-10,
            "D[0][0] = {}, expected 1.0",
            factor.data[0]
        );
    }

    #[test]
    fn spectral_of_2x2_identity() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let factor = decompose(&identity);
        assert_eq!(factor.dim(), 2);
        // D should equal I (within numerical tolerance).
        assert_gram_equals(&factor, &identity, 1e-10);
    }

    #[test]
    fn spectral_of_2x2_correlated_matrix() {
        let matrix = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let factor = decompose(&matrix);
        assert_eq!(factor.dim(), 2);
        assert_gram_equals(&factor, &matrix, 1e-10);
    }

    #[test]
    fn spectral_of_3x3_known_matrix() {
        let matrix = vec![
            vec![1.0, 0.5, 0.0],
            vec![0.5, 1.0, 0.5],
            vec![0.0, 0.5, 1.0],
        ];
        let factor = decompose(&matrix);
        assert_eq!(factor.dim(), 3);
        assert_gram_equals(&factor, &matrix, 1e-10);
    }

    #[test]
    fn spectral_of_4x4_identity() {
        let id4: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let factor = decompose(&id4);
        assert_eq!(factor.dim(), 4);
        assert_gram_equals(&factor, &id4, 1e-10);
    }

    #[test]
    fn spectral_handles_non_pd_matrix() {
        // [[1, 2], [2, 1]] has eigenvalues 3 and -1; not PD.
        let matrix = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let result = SpectralFactor::decompose(&matrix);
        assert!(result.is_ok(), "Expected Ok, got: {result:?}");
        assert_eq!(result.unwrap().dim(), 2);
    }

    #[test]
    fn spectral_fails_on_non_square_matrix() {
        let matrix = vec![vec![1.0, 0.0, 0.5], vec![0.0, 1.0]];
        let result = SpectralFactor::decompose(&matrix);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn spectral_fails_on_non_symmetric_matrix() {
        // Off-diagonal difference of 1e-9 exceeds tolerance 1e-10.
        let matrix = vec![vec![1.0, 0.3], vec![0.3 + 1e-9_f64, 1.0]];
        let result = SpectralFactor::decompose(&matrix);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn transform_with_identity_factor_equals_input() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let factor = decompose(&identity);
        let input = [1.23_f64, 4.56_f64];
        let out = transform(&factor, &input);
        assert!((out[0] - 1.23).abs() < 1e-10, "out[0] = {}", out[0]);
        assert!((out[1] - 4.56).abs() < 1e-10, "out[1] = {}", out[1]);
    }

    #[test]
    fn transform_with_known_2x2_factor() {
        // For a PD matrix [[1, 0.8], [0.8, 1]], the spectral factor D satisfies
        // D * D^T = matrix. We verify that D * e_1 and D * e_2 reconstruct the
        // columns of D (which are the images of the basis vectors).
        let matrix = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let factor = decompose(&matrix);

        // D * e_1 = first column of D.
        let col0 = transform(&factor, &[1.0, 0.0]);
        // D * e_2 = second column of D.
        let col1 = transform(&factor, &[0.0, 1.0]);

        // Verify D * D^T = matrix by checking each entry:
        // (D * D^T)[i][j] = col_j[i] but since D is symmetric D = D^T, so
        // D[i][0] = col0[i] and D[i][1] = col1[i].
        let d00 = col0[0];
        let d10 = col0[1];
        let d01 = col1[0];
        let d11 = col1[1];

        assert!(
            (d00 * d00 + d01 * d01 - 1.0).abs() < 1e-10,
            "D*D^T[0][0] off"
        );
        assert!(
            (d00 * d10 + d01 * d11 - 0.8).abs() < 1e-10,
            "D*D^T[0][1] off"
        );
        assert!(
            (d10 * d10 + d11 * d11 - 1.0).abs() < 1e-10,
            "D*D^T[1][1] off"
        );
    }

    #[test]
    fn spectral_symmetric_tolerance_boundary() {
        let matrix = vec![vec![1.0, 0.3], vec![0.3 + 5e-11_f64, 1.0]];
        assert!(SpectralFactor::decompose(&matrix).is_ok());

        let matrix_bad = vec![vec![1.0, 0.3], vec![0.3 + 2e-10_f64, 1.0]];
        assert!(matches!(
            SpectralFactor::decompose(&matrix_bad),
            Err(StochasticError::InvalidCorrelation { .. })
        ));
    }

    #[test]
    fn spectral_of_rank_deficient_matrix() {
        // 3x3 all-ones matrix: rank 1, eigenvalues = (3, 0, 0).
        let matrix = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let result = SpectralFactor::decompose(&matrix);
        assert!(result.is_ok(), "Expected Ok for rank-deficient matrix");
        let factor = result.unwrap();
        assert_eq!(factor.dim(), 3);

        // D * D^T should equal the input (nearest PSD = the input itself since
        // the input is already PSD with eigenvalues >= 0).
        assert_gram_equals(&factor, &matrix, 1e-10);
    }

    #[test]
    fn spectral_of_non_pd_clips_negative_eigenvalue() {
        // [[1, 2], [2, 1]] has eigenvalues 3 and -1.
        // After clipping: eigenvalues become (3, 0).
        // The nearest PSD matrix in the spectral sense has D * D^T = V * diag(3, 0) * V^T.
        let matrix = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let factor = decompose(&matrix);
        assert_eq!(factor.dim(), 2);

        // D * D^T should be the nearest PSD matrix, not the original.
        // For [[1,2],[2,1]], eigenvectors are [1,1]/sqrt(2) and [1,-1]/sqrt(2).
        // The PSD approximation = 3 * [1,1]/sqrt(2) * [1,1]^T/sqrt(2)
        //                        = 3/2 * [[1,1],[1,1]] = [[1.5, 1.5],[1.5, 1.5]].
        let gram = factor_gram(&factor);
        let n = 2_usize;
        let expected_psd = [[1.5, 1.5], [1.5, 1.5]];
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (gram[i * n + j] - expected_psd[i][j]).abs() < 1e-10,
                    "D*D^T[{i}][{j}] = {:.6e}, expected {:.6e}",
                    gram[i * n + j],
                    expected_psd[i][j]
                );
            }
        }
    }

    #[test]
    fn spectral_of_20x20_identity() {
        let n = 20_usize;
        let id: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let factor = decompose(&id);
        assert_eq!(factor.dim(), n);

        // Transform should be a no-op.
        let input: Vec<f64> = (1..=n).map(|x| x as f64 * 0.1).collect();
        let out = transform(&factor, &input);
        for (i, (&actual, &expected)) in out.iter().zip(input.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-9,
                "transform[{i}] = {actual:.6e}, expected {expected:.6e}"
            );
        }
    }

    #[test]
    fn spectral_matches_cholesky_for_pd_matrix() {
        // For a PD correlation matrix, D*D^T must equal the original within 1e-8.
        // This is the same property that the former Cholesky factor satisfied
        // (L*L^T = Sigma), so the spectral factor is a drop-in replacement.
        let matrix = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.4],
            vec![0.2, 0.4, 1.0],
        ];

        let spectral = decompose(&matrix);
        let n = matrix.len();
        let gram = factor_gram(&spectral);

        for i in 0..n {
            for j in 0..n {
                let original = matrix[i][j];
                let spectral_val = gram[i * n + j];
                assert!(
                    (spectral_val - original).abs() < 1e-8,
                    "spectral D*D^T[{i}][{j}] = {spectral_val:.6e}, original = {original:.6e}"
                );
            }
        }
    }
}
