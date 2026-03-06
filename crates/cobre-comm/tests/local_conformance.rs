//! Conformance tests for [`LocalBackend`] through the public `cobre_comm` API.
//!
//! These integration tests verify that `LocalBackend` satisfies the `Communicator`,
//! `SharedMemoryProvider`, and `LocalCommunicator` trait contracts as specified in
//! `backend-testing.md` SS1.1-SS1.8, adapted for the identity semantics of a
//! single-rank (size=1) backend.
//!
//! # Test organisation
//!
//! Tests are grouped by spec section via inline comments:
//! - SS1.1 `allgatherv` — identity copy semantics, displacement offset
//! - SS1.2 `allreduce` — Sum/Min/Max identity, single-element buffer
//! - SS1.3 `broadcast` — root=0 no-op, data preservation
//! - SS1.4 `barrier` — repeated barrier calls
//! - SS1.5 rank/size — postcondition checks
//! - SS1.6 compound sequencing — four-operation sequence without interference
//! - SS1.7 `SharedMemoryProvider` lifecycle — allocate -> write -> fence -> read,
//!   `is_leader`, `split_local` rank/size
//! - SS1.8 error cases — `InvalidBufferSize` and `InvalidRoot` precondition violations
//!
//! # Float comparisons
//!
//! `LocalBackend` uses identity copy semantics, so exact float equality is correct.
//! The `clippy::float_cmp` lint is suppressed at module level for this file.
#![allow(clippy::float_cmp)]
// Allow expect/unwrap in test code — these guard test setup paths that must not fail.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use cobre_comm::{
    CommError, Communicator, LocalBackend, ReduceOp, SharedMemoryProvider, SharedRegion,
};

// ── SS1.1 allgatherv ─────────────────────────────────────────────────────────

/// SS1.1: Identity copy with `counts=[3]`, `displs=[0]`.
#[test]
fn test_local_allgatherv_identity_size1() {
    let comm = LocalBackend;
    let send = [10.0_f64, 20.0, 30.0];
    let mut recv = [0.0_f64; 3];

    comm.allgatherv(&send, &mut recv, &[3], &[0])
        .expect("allgatherv must succeed");

    assert_eq!(recv, [10.0, 20.0, 30.0]);
}

/// SS1.1: Identity copy with non-zero displacement.
#[test]
fn test_local_allgatherv_with_displacement() {
    let comm = LocalBackend;
    let send = [7.0_f64, 8.0];
    let mut recv = [0.0_f64; 5];

    comm.allgatherv(&send, &mut recv, &[2], &[2])
        .expect("allgatherv must succeed");

    assert_eq!(recv, [0.0, 0.0, 7.0, 8.0, 0.0]);
}

// ── SS1.2 allreduce ───────────────────────────────────────────────────────────

/// SS1.2: `ReduceOp::Sum` is the identity for a single rank.
#[test]
fn test_local_allreduce_identity_sum() {
    let comm = LocalBackend;
    let send = [42.0_f64, 99.0];
    let mut recv = [0.0_f64; 2];

    comm.allreduce(&send, &mut recv, ReduceOp::Sum)
        .expect("allreduce must succeed");

    assert_eq!(recv, send);
}

/// SS1.2: `ReduceOp::Min` is the identity for a single rank.
#[test]
fn test_local_allreduce_identity_min() {
    let comm = LocalBackend;
    let send = [5.0_f64, 3.0, 7.0];
    let mut recv = [0.0_f64; 3];

    comm.allreduce(&send, &mut recv, ReduceOp::Min)
        .expect("allreduce must succeed");

    assert_eq!(recv, send);
}

/// SS1.2: `ReduceOp::Max` is the identity for a single rank.
#[test]
fn test_local_allreduce_identity_max() {
    let comm = LocalBackend;
    let send = [10.0_f64, 20.0];
    let mut recv = [0.0_f64; 2];

    comm.allreduce(&send, &mut recv, ReduceOp::Max)
        .expect("allreduce must succeed");

    assert_eq!(recv, send);
}

/// SS1.2: Single-element buffer reduction.
#[test]
fn test_local_allreduce_single_element() {
    let comm = LocalBackend;
    let send = [1.5_f64];
    let mut recv = [0.0_f64; 1];

    comm.allreduce(&send, &mut recv, ReduceOp::Sum)
        .expect("allreduce must succeed");

    assert_eq!(recv, send);
}

// ── SS1.3 broadcast ───────────────────────────────────────────────────────────

/// SS1.3: `root=0` is valid and is a no-op that preserves the buffer contents.
#[test]
fn test_local_broadcast_root0_noop() {
    let comm = LocalBackend;
    let mut buf = [1.0_f64, 2.0, 3.0];
    let original = buf;

    comm.broadcast(&mut buf, 0).expect("broadcast must succeed");

    assert_eq!(buf, original);
}

// ── SS1.4 barrier ─────────────────────────────────────────────────────────────

/// SS1.4: Three consecutive barriers must all complete without error.
#[test]
fn test_local_barrier_repeated() {
    let comm = LocalBackend;

    Communicator::barrier(&comm).expect("barrier 1 must succeed");
    Communicator::barrier(&comm).expect("barrier 2 must succeed");
    Communicator::barrier(&comm).expect("barrier 3 must succeed");
}

// ── SS1.5 rank/size ───────────────────────────────────────────────────────────

/// SS1.5: `rank()` returns 0 and `size()` returns 1 through the `Communicator` trait.
#[test]
fn test_local_rank_size() {
    let comm = LocalBackend;

    assert_eq!(Communicator::rank(&comm), 0);
    assert_eq!(Communicator::size(&comm), 1);
}

// ── SS1.6 compound sequencing ─────────────────────────────────────────────────

/// SS1.6: Four-operation sequence — `allgatherv` -> `allreduce(Sum)` -> `barrier`
/// -> `allgatherv` — with no stale state between operations.
#[test]
fn test_local_collective_sequence() {
    let comm = LocalBackend;

    let send1 = [1.0_f64, 2.0];
    let mut recv1 = [0.0_f64; 2];
    comm.allgatherv(&send1, &mut recv1, &[2], &[0])
        .expect("first allgatherv must succeed");
    assert_eq!(recv1, send1);

    let send2 = [10.0_f64, 20.0, 30.0];
    let mut recv2 = [0.0_f64; 3];
    comm.allreduce(&send2, &mut recv2, ReduceOp::Sum)
        .expect("allreduce must succeed");
    assert_eq!(recv2, send2);

    Communicator::barrier(&comm).expect("barrier must succeed");

    let send3 = [100.0_f64, 200.0, 300.0];
    let mut recv3 = [0.0_f64; 3];
    comm.allgatherv(&send3, &mut recv3, &[3], &[0])
        .expect("second allgatherv must succeed");
    assert_eq!(recv3, send3);
}

// ── SS1.7 SharedMemoryProvider lifecycle ──────────────────────────────────────

/// SS1.7: Full lifecycle — allocate -> write -> fence -> read.
#[test]
fn test_local_shared_region_lifecycle() {
    let backend = LocalBackend;

    let mut region = backend
        .create_shared_region::<f64>(100)
        .expect("create_shared_region must succeed");

    let slice = region.as_mut_slice();
    for (i, elem) in (0_u32..).zip(slice.iter_mut()) {
        *elem = f64::from(i);
    }

    region.fence().expect("fence must succeed");

    let read_slice = region.as_slice();
    assert_eq!(read_slice.len(), 100);
    for (i, &val) in (0_u32..).zip(read_slice.iter()) {
        assert_eq!(val, f64::from(i));
    }
}

/// SS1.7: `is_leader()` returns `true` for `LocalBackend`.
#[test]
fn test_local_shared_region_is_leader() {
    let backend = LocalBackend;
    assert!(backend.is_leader());
}

/// SS1.7: `split_local()` returns a communicator with `rank() == 0` and `size() == 1`.
#[test]
fn test_local_split_local_rank_size() {
    let backend = LocalBackend;

    let local_comm = backend.split_local().expect("split_local must succeed");

    assert_eq!(local_comm.rank(), 0);
    assert_eq!(local_comm.size(), 1);
}

// ── SS1.8 error cases ─────────────────────────────────────────────────────────

/// SS1.8.1: `allreduce` with mismatched send/recv buffer lengths.
#[test]
fn test_local_allreduce_buffer_mismatch() {
    let comm = LocalBackend;
    let send = [1.0_f64, 2.0, 3.0];
    let mut recv = [0.0_f64; 2];

    let err = comm
        .allreduce(&send, &mut recv, ReduceOp::Sum)
        .expect_err("must fail");

    assert!(
        matches!(
            err,
            CommError::InvalidBufferSize {
                operation: "allreduce",
                expected: 3,
                actual: 2,
            }
        ),
        "got {err:?}"
    );
}

/// SS1.8.1: `allreduce` with empty buffers.
#[test]
fn test_local_allreduce_empty_buffer() {
    let comm = LocalBackend;
    let send: &[f64] = &[];
    let mut recv: Vec<f64> = Vec::new();

    let err = comm
        .allreduce(send, &mut recv, ReduceOp::Sum)
        .expect_err("must fail");

    assert!(
        matches!(
            err,
            CommError::InvalidBufferSize {
                operation: "allreduce",
                expected: 1,
                actual: 0,
            }
        ),
        "got {err:?}"
    );
}

/// SS1.8.1: `allgatherv` with a receive buffer that is too small.
#[test]
fn test_local_allgatherv_recv_too_small() {
    let comm = LocalBackend;
    let send = [1.0_f64, 2.0, 3.0];
    let mut recv = [0.0_f64; 4];

    let err = comm
        .allgatherv(&send, &mut recv, &[3], &[2])
        .expect_err("must fail");

    assert!(
        matches!(
            err,
            CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: 5,
                actual: 4,
            }
        ),
        "got {err:?}"
    );
}

/// SS1.8.1: `allgatherv` with `counts.len() != size()`.
#[test]
fn test_local_allgatherv_counts_mismatch() {
    let comm = LocalBackend;
    let send = [1.0_f64];
    let mut recv = [0.0_f64; 2];

    let err = comm
        .allgatherv(&send, &mut recv, &[1, 1], &[0])
        .expect_err("must fail");

    assert!(
        matches!(
            err,
            CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: 1,
                actual: 2,
            }
        ),
        "got {err:?}"
    );
}

/// SS1.8.2: `broadcast` with `root >= size()`.
#[test]
fn test_local_broadcast_invalid_root() {
    let comm = LocalBackend;
    let mut buf = [1.0_f64, 2.0];

    let err = comm.broadcast(&mut buf, 1).expect_err("must fail");

    assert!(
        matches!(err, CommError::InvalidRoot { root: 1, size: 1 }),
        "got {err:?}"
    );
}
