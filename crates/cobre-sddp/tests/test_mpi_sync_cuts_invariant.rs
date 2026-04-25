//! Integration test: `sync_cuts` invariant rejects mismatched local cut counts.
//!
//! `CutSyncBuffers::sync_cuts` must return `SddpError::Validation` when
//! `local_cuts.len()` does not equal `per_rank_cuts[my_rank]`, with an error
//! message containing both the invariant violation and the specific mismatch.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::{FutureCostFunction, SddpError, cut_sync::CutSyncBuffers};

// ---------------------------------------------------------------------------
// Stub communicator: 2-rank, rank=0 perspective.
// ---------------------------------------------------------------------------

/// Stub communicator that simulates a 2-rank cluster from rank 0's perspective.
///
/// `allgatherv` fills each rank slot in recv from the send buffer.
/// `allreduce` copies send to recv (no aggregation needed for this test
/// since `sync_cuts` validates before calling `allgatherv`).
struct StubComm2Rank;

impl Communicator for StubComm2Rank {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
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

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Verify that `sync_cuts` returns `SddpError::Validation` when rank 0
/// provides 2 local cuts but `per_rank_cuts[0] == 3`.
///
/// The error message must contain both:
/// - `"sync_cuts invariant violated"` (exact prefix from the implementation)
/// - `"rank 0 produced 2 cuts, expected 3"` (the specific mismatch)
#[test]
fn sync_cuts_invariant_rejected_when_cut_count_mismatches() {
    let n_state = 2;
    let num_ranks = 2;
    let max_cuts_per_rank = 3;
    let total_forward_passes = 6;

    // with_distribution: base = 6/2 = 3, remainder = 0
    // per_rank_cuts = [3, 3]
    let mut bufs = CutSyncBuffers::with_distribution(
        n_state,
        max_cuts_per_rank,
        num_ranks,
        total_forward_passes,
    );

    // FCF sized for 1 stage, n_state=2, 3 cuts per rank.
    let forward_passes = u32::try_from(max_cuts_per_rank).expect("max_cuts_per_rank fits in u32");
    let mut fcf = FutureCostFunction::new(1, n_state, forward_passes, 10, &[0; 1]);
    let comm = StubComm2Rank;

    // Provide only 2 local cuts — one fewer than per_rank_cuts[0] == 3.
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
                msg.contains("rank 0 produced 2 cuts, expected 3"),
                "error must contain 'rank 0 produced 2 cuts, expected 3'; got: {msg}"
            );
        }
        other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
    }
}
