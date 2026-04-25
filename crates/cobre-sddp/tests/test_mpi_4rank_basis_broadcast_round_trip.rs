//! Integration test (slow-gated): 4-rank simulated basis broadcast round-trip.
//!
//! Verifies that `to_broadcast_payload` / `try_from_broadcast_payload` produce
//! bit-identical `CapturedBasis` values across four ranks reading from the same
//! shared buffers. Gated behind `slow-tests` feature.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

use cobre_sddp::workspace::{BASIS_BROADCAST_WIRE_VERSION, CapturedBasis};

// ---------------------------------------------------------------------------
// Helper: build a CapturedBasis with deterministic data
// ---------------------------------------------------------------------------

/// Construct a `CapturedBasis` with deterministic values derived from `seed`.
///
/// Uses `seed` to differentiate multiple stages/scenarios so that equality
/// assertions are meaningful.
fn make_captured_basis(seed: u32) -> CapturedBasis {
    let num_cols = 4_usize;
    let num_rows = 6_usize;
    let base_row_count = 2_usize;
    let cut_slot_capacity = 4_usize;
    let n_state = 3_usize;

    let mut cb = CapturedBasis::new(
        num_cols,
        num_rows,
        base_row_count,
        cut_slot_capacity,
        n_state,
    );

    // Populate col_status with deterministic values.
    for i in 0..num_cols {
        cb.basis.col_status.push(seed as i32 + i as i32);
    }
    // Populate row_status with deterministic values.
    for i in 0..num_rows {
        cb.basis.row_status.push(seed as i32 * 2 + i as i32);
    }
    // Populate cut_row_slots.
    for i in 0..cut_slot_capacity {
        cb.cut_row_slots.push(seed + i as u32);
    }
    // Populate state_at_capture (small indices, no precision loss in practice).
    for i in 0..n_state {
        cb.state_at_capture.push(f64::from(seed) + (i as f64) * 0.5);
    }

    cb
}

// ---------------------------------------------------------------------------
// Helper: verify two CapturedBasis values are bit-identical
// ---------------------------------------------------------------------------

fn assert_captured_basis_eq(a: &CapturedBasis, b: &CapturedBasis, label: &str) {
    assert_eq!(
        a.basis.col_status, b.basis.col_status,
        "{label}: col_status mismatch"
    );
    assert_eq!(
        a.basis.row_status, b.basis.row_status,
        "{label}: row_status mismatch"
    );
    assert_eq!(
        a.base_row_count, b.base_row_count,
        "{label}: base_row_count mismatch"
    );
    assert_eq!(
        a.cut_row_slots, b.cut_row_slots,
        "{label}: cut_row_slots mismatch"
    );
    assert_eq!(
        a.state_at_capture, b.state_at_capture,
        "{label}: state_at_capture mismatch"
    );
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Simulates a 4-rank broadcast of 3-stage basis data.
///
/// Pack side (rank 0): serializes 3 stages — stage 0 is `Some`, stage 1 is
/// `None`, stage 2 is `Some` — into shared `i32_buf` / `f64_buf`.
///
/// Unpack side (ranks 0-3): each rank independently reads from the same
/// shared buffers and reconstructs the same `Option<CapturedBasis>` values.
///
/// Asserts bit-equality across all four unpack results.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn four_rank_basis_broadcast_round_trip() {
    const NUM_RANKS: usize = 4;
    const NUM_STAGES: usize = 3;

    // -- Pack side (rank 0) --------------------------------------------------

    let stage0_basis = make_captured_basis(10);
    let stage2_basis = make_captured_basis(20);

    let mut i32_buf: Vec<i32> = Vec::new();
    let mut f64_buf: Vec<f64> = Vec::new();

    // Stage 0: Some.
    stage0_basis.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

    // Stage 1: None — write the 0_i32 sentinel directly.
    i32_buf.push(0_i32);

    // Stage 2: Some.
    stage2_basis.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

    // Verify the wire-version bytes are present at the expected positions.
    // Layout for a Some stage: [1 (sentinel), VERSION, col_len, row_len,
    //                           base_row_count, cut_slot_count, state_len, ...]
    assert_eq!(i32_buf[0], 1, "stage 0 sentinel must be 1");
    assert_eq!(
        i32_buf[1], BASIS_BROADCAST_WIRE_VERSION,
        "stage 0 version must equal BASIS_BROADCAST_WIRE_VERSION"
    );

    // -- Unpack side (all 4 ranks) -------------------------------------------
    // Simulate each rank independently reading from the same buffers.

    let unpack_all_stages = |rank: usize| -> Vec<Option<CapturedBasis>> {
        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let mut stages = Vec::with_capacity(NUM_STAGES);
        for stage in 0..NUM_STAGES {
            let result = CapturedBasis::try_from_broadcast_payload(
                stage,
                &i32_buf,
                &mut i32_cursor,
                &f64_buf,
                &mut f64_cursor,
            )
            .unwrap_or_else(|e| {
                panic!("rank {rank}: try_from_broadcast_payload must succeed at stage {stage}: {e}")
            });
            stages.push(result);
        }
        stages
    };

    let results: Vec<Vec<Option<CapturedBasis>>> = (0..NUM_RANKS).map(unpack_all_stages).collect();

    // All ranks must produce identical results.
    let ref_stage0 = results[0][0].as_ref().expect("rank 0 stage 0 must be Some");
    let ref_stage2 = results[0][2].as_ref().expect("rank 0 stage 2 must be Some");
    assert!(results[0][1].is_none(), "rank 0 stage 1 must be None");

    for (rank, rank_results) in results.iter().enumerate().skip(1) {
        // Stage 0: Some.
        let other_stage0 = rank_results[0]
            .as_ref()
            .unwrap_or_else(|| panic!("rank {rank} stage 0 must be Some"));
        assert_captured_basis_eq(ref_stage0, other_stage0, &format!("rank {rank} stage 0"));

        // Stage 1: None.
        assert!(
            rank_results[1].is_none(),
            "rank {rank} stage 1 must be None"
        );

        // Stage 2: Some.
        let other_stage2 = rank_results[2]
            .as_ref()
            .unwrap_or_else(|| panic!("rank {rank} stage 2 must be Some"));
        assert_captured_basis_eq(ref_stage2, other_stage2, &format!("rank {rank} stage 2"));
    }

    // Also verify the unpacked data matches the original pack-side data.
    let stage0_unpacked = results[0][0].as_ref().expect("rank 0 stage 0 must be Some");
    assert_captured_basis_eq(&stage0_basis, stage0_unpacked, "pack/unpack parity stage 0");

    let stage2_unpacked = results[0][2].as_ref().expect("rank 0 stage 2 must be Some");
    assert_captured_basis_eq(&stage2_basis, stage2_unpacked, "pack/unpack parity stage 2");
}
