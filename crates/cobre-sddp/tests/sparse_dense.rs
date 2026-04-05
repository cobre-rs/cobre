//! Sparse vs dense cut injection equivalence test.
//!
//! Verifies that `build_cut_row_batch_into` produces identical `RowBatch`
//! output when using the sparse path (nonzero mask covers all state indices)
//! versus the dense path (empty mask). This guards against the two code paths
//! silently diverging.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown
)]

use cobre_sddp::cut::fcf::FutureCostFunction;
use cobre_sddp::forward::build_cut_row_batch_into;
use cobre_sddp::StageIndexer;
use cobre_solver::RowBatch;

/// Build a `RowBatch` with cuts from `fcf` at `stage` using the given `indexer`.
fn build_batch(
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
) -> RowBatch {
    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    build_cut_row_batch_into(&mut batch, fcf, stage, indexer, col_scale);
    batch
}

/// When all hydros have full PAR order (mask covers all n_state indices),
/// the sparse path should produce byte-identical output to the dense path.
#[test]
fn sparse_full_mask_equals_dense() {
    // 2 hydros, max_par_order = 2 → n_state = 2 * (1 + 2) = 6
    let n_hydro = 2;
    let max_par_order = 2;
    let n_state = n_hydro * (1 + max_par_order); // 6

    // Dense indexer: empty nonzero_state_indices
    let dense_indexer = StageIndexer::new(n_hydro, max_par_order);
    assert!(
        dense_indexer.nonzero_state_indices.is_empty(),
        "dense indexer must have empty mask"
    );

    // Sparse indexer: mask covers all indices (both hydros at full AR order)
    let mut sparse_indexer = StageIndexer::new(n_hydro, max_par_order);
    sparse_indexer.set_nonzero_mask(&[max_par_order, max_par_order]);
    assert_eq!(
        sparse_indexer.nonzero_state_indices.len(),
        n_state,
        "full-order mask must cover all state indices"
    );

    // Create FCF: new(num_stages, state_dim, fwd_passes, max_iter, warm_start).
    // capacity = warm_start + max_iter * fwd_passes = 0 + 3 * 3 = 9 cuts/pool.
    let mut fcf = FutureCostFunction::new(2, n_state, 3, 3, 0);

    // Add several cuts at stage 0 with known coefficients.
    // Slot formula: warm_start + iteration * fwd_passes + fwd_idx
    let coeffs_a = vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.0];
    let coeffs_b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 7.5];
    let coeffs_c = vec![-1.5, 2.5, -3.5, 4.5, -5.5, 6.5];

    fcf.add_cut(0, 0, 0, 100.0, &coeffs_a); // slot 0
    fcf.add_cut(0, 0, 1, 200.0, &coeffs_b); // slot 1
    fcf.add_cut(0, 1, 0, 300.0, &coeffs_c); // slot 3

    let col_scale: Vec<f64> = Vec::new(); // no scaling

    // Build batches with both paths.
    let dense_batch = build_batch(&fcf, 0, &dense_indexer, &col_scale);
    let sparse_batch = build_batch(&fcf, 0, &sparse_indexer, &col_scale);

    // Assert structural equality.
    assert_eq!(
        dense_batch.num_rows, sparse_batch.num_rows,
        "num_rows mismatch"
    );
    assert_eq!(
        dense_batch.row_starts, sparse_batch.row_starts,
        "row_starts mismatch"
    );
    assert_eq!(
        dense_batch.col_indices, sparse_batch.col_indices,
        "col_indices mismatch"
    );
    assert_eq!(
        dense_batch.row_lower, sparse_batch.row_lower,
        "row_lower mismatch"
    );
    assert_eq!(
        dense_batch.row_upper.len(),
        sparse_batch.row_upper.len(),
        "row_upper length mismatch"
    );

    // Bit-for-bit comparison of floating-point values.
    for (i, (d, s)) in dense_batch
        .values
        .iter()
        .zip(sparse_batch.values.iter())
        .enumerate()
    {
        assert_eq!(
            d.to_bits(),
            s.to_bits(),
            "values[{i}] differ: dense={d}, sparse={s}"
        );
    }
    for (i, (d, s)) in dense_batch
        .row_lower
        .iter()
        .zip(sparse_batch.row_lower.iter())
        .enumerate()
    {
        assert_eq!(
            d.to_bits(),
            s.to_bits(),
            "row_lower[{i}] differ: dense={d}, sparse={s}"
        );
    }
}

/// When the sparse mask is a proper subset (mixed AR orders), the output
/// differs from dense (fewer nonzeros per row). This test verifies the
/// sparse path produces correct output for a mixed-order case.
#[test]
fn sparse_partial_mask_produces_correct_subset() {
    // 3 hydros, max_par_order = 2 → n_state = 3 * (1 + 2) = 9
    let n_hydro = 3;
    let max_par_order = 2;
    let n_state = n_hydro * (1 + max_par_order); // 9

    // Mixed AR orders: [0, 1, 2] → some lag slots are zero
    let mut indexer = StageIndexer::new(n_hydro, max_par_order);
    indexer.set_nonzero_mask(&[0, 1, 2]);
    // Expected mask: storage [0,1,2] + lag0 for h1,h2 [4,5] + lag1 for h2 [8]
    // = [0, 1, 2, 4, 5, 8]
    let mask = &indexer.nonzero_state_indices;
    assert_eq!(
        mask.len(),
        6,
        "mixed-order mask has 3 + 0 + 1 + 2 = 6 entries"
    );

    // Create FCF: new(num_stages, state_dim, fwd_passes, max_iter, warm_start).
    let mut fcf = FutureCostFunction::new(2, n_state, 1, 1, 0);
    let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    fcf.add_cut(0, 0, 0, 50.0, &coeffs);

    let col_scale: Vec<f64> = Vec::new();

    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    build_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &col_scale);

    // The sparse path should only emit entries for mask indices + theta.
    // NNZ per cut = mask.len() + 1 (theta) = 7
    assert_eq!(batch.num_rows, 1);
    // col_indices should contain the remapped LP columns plus theta_col.
    // state_to_lp_column maps outgoing-state indices to LP columns:
    // storage → identity; lag 0 → z_inflow; lag l≥1 → incoming lag l−1.
    let theta_col = indexer.theta;
    let expected_cols: Vec<i32> = mask
        .iter()
        .map(|&j| indexer.state_to_lp_column(j) as i32)
        .chain(std::iter::once(theta_col as i32))
        .collect();
    assert_eq!(
        batch.col_indices, expected_cols,
        "sparse col_indices mismatch"
    );

    // Values should be -coefficients[j] for each mask index, plus +1.0 for theta.
    let expected_values: Vec<f64> = mask
        .iter()
        .map(|&j| -coeffs[j])
        .chain(std::iter::once(1.0))
        .collect();
    for (i, (actual, expected)) in batch.values.iter().zip(expected_values.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "values[{i}] differ: actual={actual}, expected={expected}"
        );
    }
}

/// Verify scaling is applied identically in both paths.
#[test]
fn sparse_dense_with_scaling() {
    let n_hydro = 2;
    let max_par_order = 1;
    let n_state = n_hydro * (1 + max_par_order); // 4

    let dense_indexer = StageIndexer::new(n_hydro, max_par_order);
    let mut sparse_indexer = StageIndexer::new(n_hydro, max_par_order);
    sparse_indexer.set_nonzero_mask(&[max_par_order, max_par_order]);

    let mut fcf = FutureCostFunction::new(2, n_state, 1, 1, 0);
    let coeffs = vec![10.0, -20.0, 30.0, -40.0];
    fcf.add_cut(0, 0, 0, 500.0, &coeffs);

    // Column scaling: one scale factor per column (n_state + equipment columns).
    // theta is at indexer.theta position.
    let theta_col = dense_indexer.theta;
    let mut col_scale = vec![1.0; theta_col + 1];
    col_scale[0] = 2.0;
    col_scale[1] = 0.5;
    col_scale[2] = 3.0;
    col_scale[3] = 0.25;
    col_scale[theta_col] = 1.5;

    let dense_batch = build_batch(&fcf, 0, &dense_indexer, &col_scale);
    let sparse_batch = build_batch(&fcf, 0, &sparse_indexer, &col_scale);

    assert_eq!(
        dense_batch.col_indices, sparse_batch.col_indices,
        "col_indices mismatch with scaling"
    );
    for (i, (d, s)) in dense_batch
        .values
        .iter()
        .zip(sparse_batch.values.iter())
        .enumerate()
    {
        assert_eq!(
            d.to_bits(),
            s.to_bits(),
            "values[{i}] differ with scaling: dense={d}, sparse={s}"
        );
    }
}
