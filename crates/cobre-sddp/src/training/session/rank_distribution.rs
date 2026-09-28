//! Rank-distribution constants for one training run.

use cobre_comm::Communicator;

/// Base/remainder arithmetic that divides `total_forward_passes` across MPI
/// ranks. All fields are set once in `RankDistribution::new` and read-only for
/// the run.
#[derive(Copy, Clone, Debug)]
pub(crate) struct RankDistribution {
    pub num_ranks: usize,
    // Rationale: read only by unit tests via `per_rank[rd.my_rank]`; MPI call
    // sites use the `i32` `fwd_rank` instead.
    #[allow(dead_code)]
    pub my_rank: usize,
    pub my_actual_fwd: usize,
    pub my_fwd_offset: usize,
    pub max_local_fwd: usize,
    pub num_total_forward_passes: usize,
    pub fwd_rank: i32,
}

impl RankDistribution {
    /// Derive all rank-distribution constants from the communicator and config.
    ///
    /// The first `remainder_fwd` ranks each receive `base_fwd + 1` forward
    /// passes; the rest receive `base_fwd`.
    // Rationale: MPI rank integers fit in `i32`, so the `expect` cannot fire.
    #[allow(clippy::expect_used)]
    pub(crate) fn new<C: Communicator>(comm: &C, total_forward_passes: usize) -> Self {
        let num_ranks = comm.size();
        let my_rank = comm.rank();
        let base_fwd = total_forward_passes / num_ranks;
        let remainder_fwd = total_forward_passes % num_ranks;
        let my_actual_fwd = base_fwd + usize::from(my_rank < remainder_fwd);
        let my_fwd_offset = base_fwd * my_rank + my_rank.min(remainder_fwd);
        let max_local_fwd = base_fwd + usize::from(remainder_fwd > 0);
        let fwd_rank = i32::try_from(my_rank).expect("MPI rank fits in i32");
        Self {
            num_ranks,
            my_rank,
            my_actual_fwd,
            my_fwd_offset,
            max_local_fwd,
            num_total_forward_passes: total_forward_passes,
            fwd_rank,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp, per_rank_counts};

    use super::RankDistribution;

    /// Minimal communicator stub with configurable rank and size for unit tests.
    struct StubCommN {
        rank: usize,
        size: usize,
    }

    impl Communicator for StubCommN {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].clone_from_slice(send);
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
            self.rank
        }

        fn size(&self) -> usize {
            self.size
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    #[test]
    fn rank_distribution_new_3_ranks_8_forward_passes() {
        // 8 / 3 = base=2, remainder=2
        // rank 0: my_actual_fwd = 3, my_fwd_offset = 0
        // rank 1: my_actual_fwd = 3, my_fwd_offset = 3
        // rank 2: my_actual_fwd = 2, my_fwd_offset = 6
        // max_local_fwd = base + 1 = 3 (remainder > 0)
        let expected_actual = [3, 3, 2];
        let expected_offset = [0, 3, 6];

        for rank in 0..3 {
            let comm = StubCommN { rank, size: 3 };
            let rd = RankDistribution::new(&comm, 8);

            assert_eq!(rd.num_ranks, 3, "rank {rank}: num_ranks");
            assert_eq!(
                rd.my_actual_fwd, expected_actual[rank],
                "rank {rank}: my_actual_fwd"
            );
            assert_eq!(
                rd.my_fwd_offset, expected_offset[rank],
                "rank {rank}: my_fwd_offset"
            );
            assert_eq!(rd.max_local_fwd, 3, "rank {rank}: max_local_fwd");
            assert_eq!(
                rd.num_total_forward_passes, 8,
                "rank {rank}: num_total_forward_passes"
            );
        }
    }

    /// `max_local_fwd * num_stages` stays rank-uniform and non-zero even where
    /// `my_actual_fwd` is 0 — the premise the backward pass's record slice rests
    /// on. Pins the arithmetic only; that the pass actually hands `exchange` a
    /// full-length slice is pinned by `state_exchange`'s zero-work tests.
    #[test]
    fn zero_work_ranks_keep_a_rank_uniform_backward_record_length() {
        let num_stages = 3;
        let (total_forward_passes, num_ranks) = (1, 4);

        let backward_record_lens: Vec<usize> = (0..num_ranks)
            .map(|rank| {
                let comm = StubCommN {
                    rank,
                    size: num_ranks,
                };
                let rd = RankDistribution::new(&comm, total_forward_passes);
                rd.max_local_fwd * num_stages
            })
            .collect();

        assert_eq!(
            backward_record_lens,
            vec![num_stages; num_ranks],
            "every rank's backward record slice must be the same non-zero length"
        );

        let zero_work_ranks = (0..num_ranks)
            .filter(|&rank| {
                let comm = StubCommN {
                    rank,
                    size: num_ranks,
                };
                RankDistribution::new(&comm, total_forward_passes).my_actual_fwd == 0
            })
            .count();
        assert_eq!(
            zero_work_ranks, 3,
            "fixture must actually exercise ranks that draw zero forward passes"
        );
    }

    #[test]
    fn rank_distribution_actual_per_rank_is_consistent_with_my_actual_fwd() {
        for rank in 0..3 {
            let comm = StubCommN { rank, size: 3 };
            let rd = RankDistribution::new(&comm, 8);

            let per_rank = per_rank_counts(8, rd.num_ranks);
            assert_eq!(per_rank, vec![3, 3, 2], "rank {rank}: per_rank vec");
            assert_eq!(
                per_rank[rd.my_rank], rd.my_actual_fwd,
                "rank {rank}: per_rank[my_rank] == my_actual_fwd"
            );
        }
    }
}
