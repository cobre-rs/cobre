//! Sparse representation of Benders cut coefficient vectors.
//!
//! [`SparseCut`] stores only the non-zero entries of a cut's coefficient
//! vector as `(dimension_index, value)` pairs. This reduces memory for
//! high-dimensional systems where structural sparsity is significant
//! (e.g., PAR lag padding produces 29%+ exact zeros for the Convertido system).
//!
//! ## Design decisions
//!
//! - **Per-cut sparse vectors** (not pool-level CSR): Each cut stores its own
//!   `(indices, values)` pair. This is simpler for the incremental
//!   add/deactivate path and matches the per-slot structure of [`CutPool`].
//! - **Exact-zero threshold**: Only `c == 0.0` is dropped. This preserves
//!   bit-for-bit reproducibility with the dense representation. A
//!   configurable near-zero threshold is deferred.
//! - **Sorted indices**: Indices are stored in ascending order so that the
//!   sparse dot product visits coefficients in the same order as the dense
//!   loop, ensuring identical floating-point results.
//!
//! [`CutPool`]: super::pool::CutPool

/// Sparse representation of a cut's coefficient vector.
///
/// Stores only the non-zero entries as `(dimension_index, value)` pairs,
/// sorted by dimension index in ascending order. The original dense
/// dimension is stored in `state_dimension` for reconstruction and
/// validation.
///
/// Only exact zeros (0.0) are dropped. Near-zero values are preserved to
/// maintain bit-for-bit reproducibility with the dense representation.
///
/// # Example
///
/// ```rust
/// use cobre_sddp::cut::sparse::SparseCut;
///
/// let dense = vec![1.0, 0.0, 0.0, 3.0, 0.0];
/// let sparse = SparseCut::from_dense(&dense);
/// assert_eq!(sparse.nnz(), 2);
/// assert_eq!(sparse.indices(), &[0, 3]);
/// assert_eq!(sparse.values(), &[1.0, 3.0]);
/// assert_eq!(sparse.to_dense(), dense);
/// ```
#[derive(Debug, Clone)]
pub struct SparseCut {
    /// Non-zero coefficient indices, sorted ascending.
    indices: Vec<usize>,
    /// Non-zero coefficient values, parallel to `indices`.
    values: Vec<f64>,
    /// Original dense dimension (`n_state`). Used for validation and
    /// dense reconstruction.
    state_dimension: usize,
}

impl SparseCut {
    /// Create a sparse cut from a dense coefficient slice.
    ///
    /// Drops exact zeros only (`c == 0.0`); near-zero values are preserved.
    /// Indices are naturally sorted because the input is iterated in order.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::sparse::SparseCut;
    ///
    /// let sparse = SparseCut::from_dense(&[0.0, 1.0, 0.0, 2.0]);
    /// assert_eq!(sparse.nnz(), 2);
    /// assert_eq!(sparse.state_dimension(), 4);
    /// ```
    #[must_use]
    pub fn from_dense(dense: &[f64]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (j, &c) in dense.iter().enumerate() {
            if c != 0.0 {
                indices.push(j);
                values.push(c);
            }
        }

        Self {
            indices,
            values,
            state_dimension: dense.len(),
        }
    }

    /// Create an empty sparse cut with the given state dimension.
    ///
    /// All coefficients are zero (`nnz() == 0`).
    #[must_use]
    pub fn empty(state_dimension: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            state_dimension,
        }
    }

    /// Reconstruct the dense coefficient vector.
    ///
    /// Returns a `Vec<f64>` of length `state_dimension` with zeros in
    /// positions not present in `indices`.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.state_dimension];
        for (&j, &v) in self.indices.iter().zip(self.values.iter()) {
            dense[j] = v;
        }
        dense
    }

    /// Number of non-zero entries.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Return the original dense dimension.
    #[must_use]
    #[inline]
    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }

    /// Return a reference to the non-zero indices.
    #[must_use]
    #[inline]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Return a reference to the non-zero values.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Compute the dot product with a dense state vector.
    ///
    /// Iterates only over non-zero entries for O(nnz) instead of
    /// O(`state_dimension`). The summation visits coefficients in ascending
    /// index order, matching the dense iteration order and ensuring
    /// bit-identical results.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `state.len() != self.state_dimension`.
    #[must_use]
    pub fn dot(&self, state: &[f64]) -> f64 {
        debug_assert_eq!(
            state.len(),
            self.state_dimension,
            "SparseCut::dot: state length {} != state_dimension {}",
            state.len(),
            self.state_dimension
        );

        let mut sum = 0.0_f64;
        for (&j, &v) in self.indices.iter().zip(self.values.iter()) {
            sum += v * state[j];
        }
        sum
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::SparseCut;

    #[test]
    fn from_dense_drops_exact_zeros() {
        let sparse = SparseCut::from_dense(&[1.0, 0.0, 0.0, 3.0, 0.0]);
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.indices(), &[0, 3]);
        assert_eq!(sparse.values(), &[1.0, 3.0]);
        assert_eq!(sparse.state_dimension(), 5);
    }

    #[test]
    fn from_dense_all_zero() {
        let sparse = SparseCut::from_dense(&[0.0, 0.0, 0.0]);
        assert_eq!(sparse.nnz(), 0);
        assert!(sparse.indices().is_empty());
        assert!(sparse.values().is_empty());
        assert_eq!(sparse.state_dimension(), 3);
    }

    #[test]
    fn from_dense_all_nonzero() {
        let sparse = SparseCut::from_dense(&[1.0, 2.0, 3.0]);
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.indices(), &[0, 1, 2]);
        assert_eq!(sparse.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_dense_empty() {
        let sparse = SparseCut::from_dense(&[]);
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.state_dimension(), 0);
    }

    #[test]
    fn from_dense_single_nonzero() {
        let sparse = SparseCut::from_dense(&[0.0, 5.0, 0.0]);
        assert_eq!(sparse.nnz(), 1);
        assert_eq!(sparse.indices(), &[1]);
        assert_eq!(sparse.values(), &[5.0]);
    }

    #[test]
    fn from_dense_preserves_near_zero() {
        // Near-zero values are NOT dropped -- only exact 0.0 is dropped.
        let sparse = SparseCut::from_dense(&[1e-300, 0.0, -1e-300]);
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.indices(), &[0, 2]);
        assert_eq!(sparse.values(), &[1e-300, -1e-300]);
    }

    #[test]
    fn to_dense_round_trip() {
        let original = vec![1.0, 0.0, 3.0, 0.0, 5.0];
        let sparse = SparseCut::from_dense(&original);
        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn to_dense_round_trip_all_zero() {
        let original = vec![0.0, 0.0, 0.0];
        let sparse = SparseCut::from_dense(&original);
        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn to_dense_round_trip_all_nonzero() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let sparse = SparseCut::from_dense(&original);
        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn empty_creates_zero_nnz() {
        let sparse = SparseCut::empty(5);
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.state_dimension(), 5);
        assert_eq!(sparse.to_dense(), vec![0.0; 5]);
    }

    #[test]
    fn dot_matches_dense() {
        let dense = vec![1.0, 0.0, 3.0, 0.0, 5.0];
        let state = vec![2.0, 7.0, 4.0, 9.0, 6.0];
        let sparse = SparseCut::from_dense(&dense);

        let dense_dot: f64 = dense.iter().zip(state.iter()).map(|(a, b)| a * b).sum();
        let sparse_dot = sparse.dot(&state);

        assert_eq!(sparse_dot, dense_dot);
    }

    #[test]
    fn dot_all_zero_is_zero() {
        let sparse = SparseCut::from_dense(&[0.0, 0.0, 0.0]);
        let state = vec![1.0, 2.0, 3.0];
        assert_eq!(sparse.dot(&state), 0.0);
    }

    #[test]
    fn dot_empty_is_zero() {
        let sparse = SparseCut::from_dense(&[]);
        assert_eq!(sparse.dot(&[]), 0.0);
    }

    #[test]
    fn dot_single_element() {
        let sparse = SparseCut::from_dense(&[0.0, 3.0, 0.0]);
        let state = vec![10.0, 5.0, 20.0];
        assert_eq!(sparse.dot(&state), 15.0); // 3.0 * 5.0
    }

    #[test]
    fn dot_bit_identical_to_dense_with_multiple_patterns() {
        // Test several sparsity patterns to ensure bit-identical results.
        let patterns: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 2.0, 0.0, 3.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 7.0],
            vec![9.0, 0.0, 0.0, 0.0, 0.0],
            vec![-1.0, 2.0, -3.0, 4.0, -5.0],
            vec![1e-100, 0.0, 1e100, 0.0, -1e-100],
        ];
        let state = vec![0.5, 1.5, 2.5, 3.5, 4.5];

        for dense in &patterns {
            let sparse = SparseCut::from_dense(dense);
            let dense_dot: f64 = dense.iter().zip(state.iter()).map(|(a, b)| a * b).sum();
            let sparse_dot = sparse.dot(&state);
            assert_eq!(
                sparse_dot, dense_dot,
                "mismatch for pattern {dense:?}: sparse={sparse_dot}, dense={dense_dot}"
            );
        }
    }

    #[test]
    fn clone_produces_independent_copy() {
        let sparse = SparseCut::from_dense(&[1.0, 0.0, 3.0]);
        let cloned = sparse.clone();
        assert_eq!(cloned.nnz(), sparse.nnz());
        assert_eq!(cloned.indices(), sparse.indices());
        assert_eq!(cloned.values(), sparse.values());
        assert_eq!(cloned.state_dimension(), sparse.state_dimension());
    }
}
