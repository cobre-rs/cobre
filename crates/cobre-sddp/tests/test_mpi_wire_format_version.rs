//! Integration test: MPI wire-format version guards reject stale payloads.
//!
//! `CapturedBasis::try_from_broadcast_payload` and `deserialize_cut` must both
//! return `SddpError::Validation` when the wire-format version byte/field is
//! corrupted. These tests exercise the public API surface without spawning MPI processes.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use cobre_sddp::{
    CapturedBasis, SddpError,
    cut::wire::{cut_wire_size, deserialize_cut, serialize_cut},
    workspace::BASIS_BROADCAST_WIRE_VERSION,
};

// ---------------------------------------------------------------------------
// Ticket-004: basis wire-format version guard
// ---------------------------------------------------------------------------

/// Construct a valid `CapturedBasis`, serialize it via `to_broadcast_payload`,
/// corrupt the version field (`i32_buf[1]`), then verify that
/// `try_from_broadcast_payload` returns `SddpError::Validation` containing
/// `"unsupported wire version 2"`.
#[test]
fn basis_try_from_broadcast_payload_rejects_wrong_version() {
    let num_cols = 2;
    let num_rows = 3;
    let base_row_count = 1;
    let cut_slot_capacity = 2;
    let n_state = 1;

    let mut original = CapturedBasis::new(
        num_cols,
        num_rows,
        base_row_count,
        cut_slot_capacity,
        n_state,
    );
    // Populate with minimal valid data so the Some-path sentinel is written.
    original.basis.col_status.extend_from_slice(&[0_i32, 1_i32]);
    original
        .basis
        .row_status
        .extend_from_slice(&[0_i32, 1_i32, 0_i32]);
    original.cut_row_slots.push(0_u32);
    original.cut_row_slots.push(1_u32);
    original.state_at_capture.push(42.0_f64);

    let mut i32_buf: Vec<i32> = Vec::new();
    let mut f64_buf: Vec<f64> = Vec::new();
    original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

    // Layout: [sentinel=1, version=BASIS_BROADCAST_WIRE_VERSION, col_len, row_len,
    //          base_row_count, cut_slot_count, state_len, col_status..., row_status...,
    //          cut_row_slots...]
    // The version is at index 1.
    assert_eq!(i32_buf[0], 1, "sentinel must be 1 (Some path)");
    assert_eq!(
        i32_buf[1], BASIS_BROADCAST_WIRE_VERSION,
        "version field must equal BASIS_BROADCAST_WIRE_VERSION before corruption"
    );

    // Corrupt the version field to 2.
    i32_buf[1] = 2;

    let mut i32_cursor = 0_usize;
    let mut f64_cursor = 0_usize;
    let result = CapturedBasis::try_from_broadcast_payload(
        0,
        &i32_buf,
        &mut i32_cursor,
        &f64_buf,
        &mut f64_cursor,
    );

    match result {
        Err(SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("unsupported wire version 2"),
                "error must contain 'unsupported wire version 2'; got: {msg}"
            );
        }
        other => panic!(
            "expected Err(SddpError::Validation(_)) containing 'unsupported wire version 2', \
             got: {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Ticket-005: cut wire-format version guard
// ---------------------------------------------------------------------------

/// Serialize a single cut via `serialize_cut`, corrupt the version byte at
/// offset 0 to `2_u8`, then verify that `deserialize_cut` returns
/// `SddpError::Validation` containing `"unsupported cut wire version 2"`.
#[test]
fn deserialize_cut_rejects_wrong_version() {
    let n_state = 2;
    let record_size = cut_wire_size(n_state);
    let mut buf = vec![0u8; record_size];

    serialize_cut(
        &mut buf,
        /* slot_index */ 0,
        /* iteration */ 1,
        /* forward_pass_index */ 0,
        /* intercept */ 99.0,
        /* coefficients */ &[1.0, 2.0],
    );

    // Verify the correct version was written before corruption.
    assert_eq!(buf[0], 1, "version byte must be 1 before corruption");

    // Corrupt the version byte to 2.
    buf[0] = 2_u8;

    let result = deserialize_cut(&buf, n_state);

    match result {
        Err(SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("unsupported cut wire version 2"),
                "error must contain 'unsupported cut wire version 2'; got: {msg}"
            );
        }
        other => panic!(
            "expected Err(SddpError::Validation(_)) containing 'unsupported cut wire version 2', \
             got: {other:?}"
        ),
    }
}
