//! Integration test: oversized MPI broadcast buffer is rejected.
//!
//! `CommError::InvalidBufferSize` must correctly round-trip through
//! `SddpError::Communication` with both actual (oversized) and expected
//! (`i32::MAX`) length fields preserved, enabling diagnostic messages.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use cobre_comm::CommError;
use cobre_sddp::SddpError;

/// Verify `CommError::InvalidBufferSize` round-trips correctly through
/// `SddpError::Communication`, carrying the actual oversized length.
///
/// This mirrors the error that `checked_broadcast_len` would produce for
/// `len = (i32::MAX as usize) + 1` — the smallest value that cannot be
/// represented as an MPI count.
#[test]
fn comm_error_invalid_buffer_size_roundtrip() {
    let actual = (i32::MAX as usize) + 1;
    let expected = i32::MAX as usize;
    let operation = "broadcast_basis_cache_i32";

    // Construct the same error that checked_broadcast_len would return.
    let comm_err = CommError::InvalidBufferSize {
        operation,
        expected,
        actual,
    };
    let sddp_err = SddpError::Communication(comm_err);

    match sddp_err {
        SddpError::Communication(CommError::InvalidBufferSize {
            operation: op,
            expected: exp,
            actual: act,
        }) => {
            assert_eq!(
                act,
                (i32::MAX as usize) + 1,
                "actual must equal (i32::MAX as usize) + 1"
            );
            assert_eq!(
                exp,
                i32::MAX as usize,
                "expected must equal i32::MAX as usize"
            );
            assert_eq!(
                op, "broadcast_basis_cache_i32",
                "operation string must match"
            );
        }
        other => panic!(
            "expected SddpError::Communication(CommError::InvalidBufferSize {{ .. }}), got: \
             {other:?}"
        ),
    }
}

/// Verify the `Display` output of `CommError::InvalidBufferSize` contains the
/// two key quantities (actual and expected count), so diagnostic messages are
/// actionable.
#[test]
fn comm_error_invalid_buffer_size_display_contains_counts() {
    let actual = (i32::MAX as usize) + 1;
    let expected = i32::MAX as usize;

    let err = CommError::InvalidBufferSize {
        operation: "broadcast_basis_cache_i32",
        expected,
        actual,
    };
    let msg = err.to_string();

    assert!(
        msg.contains(&actual.to_string()),
        "Display must contain actual count {actual}; got: {msg}"
    );
    assert!(
        msg.contains(&expected.to_string()),
        "Display must contain expected count {expected}; got: {msg}"
    );
}
