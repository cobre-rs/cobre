//! Integration test: `sync_cuts` rejects a mismatched local cut count.
//!
//! `CutSyncBuffers::sync_cuts` validates that the number of cuts this rank
//! produces equals the expected per-rank count derived from the total forward
//! passes distributed across all ranks. When a rank produces fewer cuts than
//! expected, `sync_cuts` must return `SddpError::Validation`.
//!
//! The `n_workers_local` uniformity handshake in the backward pass is a
//! separate invariant covered by the unit test
//! `handshake_rejects_nonuniform_workers` in `crates/cobre-sddp/src/backward.rs`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::{FutureCostFunction, SddpError, cut_sync::CutSyncBuffers};

// ---------------------------------------------------------------------------
// Stub communicator: simulates 2-rank topology (rank 0 perspective).
// ---------------------------------------------------------------------------

/// Stub that simulates a 2-rank cluster from rank 0's perspective.
/// `allgatherv` fills recv by concatenating send twice (rank 0 + rank 1 copy).
/// `allreduce` copies send to recv regardless of `ReduceOp` (single-rank
/// semantics are sufficient for the `sync_cuts` invariant check exercised here).
struct StubComm2Rank;

impl Communicator for StubComm2Rank {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        // Fill each rank slot from send.
        for (&count, &displ) in counts.iter().zip(displs.iter()) {
            let src_len = count.min(send.len());
            recv[displ..displ + src_len].clone_from_slice(&send[..src_len]);
        }
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        2
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

/// Verifies that `sync_cuts` returns `SddpError::Validation` when the local
/// rank produces fewer cuts than expected.
///
/// The `with_distribution` constructor sets `per_rank_cuts[0] = 3` (uniform
/// distribution of 6 total passes across 2 ranks).  Passing only 2 local cuts
/// violates the invariant and must return a `Validation` error whose message
/// contains `"sync_cuts invariant violated"` and `"rank 0 produced 2 cuts"`.
#[test]
fn sync_cuts_rejects_mismatched_local_cut_count() {
    let n_state = 2;
    let num_ranks = 2;
    let total_forward_passes = 6;
    let max_cuts_per_rank = 3;

    // per_rank_cuts = [3, 3] (6 passes / 2 ranks).
    let mut bufs = CutSyncBuffers::with_distribution(
        n_state,
        max_cuts_per_rank,
        num_ranks,
        total_forward_passes,
    );

    let forward_passes = u32::try_from(max_cuts_per_rank).expect("max_cuts_per_rank fits in u32");
    let mut fcf = FutureCostFunction::new(1, n_state, forward_passes, 10, &[0; 1]);
    let comm = StubComm2Rank;

    // Only provide 2 cuts, but per_rank_cuts[rank=0] == 3.
    let coeffs_a = [1.0_f64, 2.0_f64];
    let coeffs_b = [3.0_f64, 4.0_f64];
    let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
        &[(0, 1, 0, 10.0, &coeffs_a), (0, 1, 1, 20.0, &coeffs_b)];

    let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm);

    match result {
        Err(SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("sync_cuts invariant violated"),
                "error must contain 'sync_cuts invariant violated'; got: {msg}"
            );
            assert!(
                msg.contains("rank 0 produced 2 cuts"),
                "error must mention 'rank 0 produced 2 cuts'; got: {msg}"
            );
            assert!(
                msg.contains("expected 3"),
                "error must mention 'expected 3'; got: {msg}"
            );
        }
        other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
    }
}
