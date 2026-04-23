//! Rank-distribution constants for one training run.

use cobre_comm::Communicator;

/// Constants derived from the MPI communicator and training configuration.
///
/// All fields are set once in `RankDistribution::new` at the start of a
/// training run and are read-only for its duration. This struct owns the
/// base/remainder distribution arithmetic that divides `total_forward_passes`
/// across MPI ranks.
///
/// Fields are derived from `(comm.size(), comm.rank(), total_forward_passes,
/// n_state, num_stages)` and remain constant for the lifetime of the session.
#[derive(Copy, Clone, Debug)]
pub(crate) struct RankDistribution {
    pub num_stages: usize,
    pub num_ranks: usize,
    // Read by downstream tickets (ticket-002, run_lower_bound refactor).
    #[allow(dead_code)]
    pub my_rank: usize,
    pub my_actual_fwd: usize,
    pub my_fwd_offset: usize,
    pub max_local_fwd: usize,
    pub n_state: usize,
    pub fwd_rank: i32,
}

impl RankDistribution {
    /// Derive all rank-distribution constants from the communicator and training
    /// configuration.
    ///
    /// Performs the base/remainder arithmetic that distributes
    /// `total_forward_passes` across `comm.size()` ranks: the first
    /// `remainder_fwd` ranks each receive `base_fwd + 1` forward passes; the
    /// remaining ranks receive `base_fwd`.
    ///
    /// The single `expect` call is preserved verbatim from the pre-refactor
    /// inline code; MPI rank integers must fit in `i32`.
    #[allow(clippy::expect_used)]
    pub(crate) fn new<C: Communicator>(
        comm: &C,
        num_stages: usize,
        total_forward_passes: usize,
        n_state: usize,
    ) -> Self {
        let num_ranks = comm.size();
        let my_rank = comm.rank();
        let base_fwd = total_forward_passes / num_ranks;
        let remainder_fwd = total_forward_passes % num_ranks;
        let my_actual_fwd = base_fwd + usize::from(my_rank < remainder_fwd);
        let my_fwd_offset = base_fwd * my_rank + my_rank.min(remainder_fwd);
        let max_local_fwd = base_fwd + usize::from(remainder_fwd > 0);
        let fwd_rank = i32::try_from(my_rank).expect("MPI rank fits in i32");
        Self {
            num_stages,
            num_ranks,
            my_rank,
            my_actual_fwd,
            my_fwd_offset,
            max_local_fwd,
            n_state,
            fwd_rank,
        }
    }

    /// Return a vector of length `num_ranks` where index `r` holds the number
    /// of forward passes assigned to rank `r`.
    ///
    /// The per-rank value is `base_fwd + usize::from(r < remainder_fwd)`, which
    /// is identical to the `my_actual_fwd` derivation in `new` applied for every
    /// rank index. Calling `actual_per_rank(total)[self.my_rank]` equals
    /// `self.my_actual_fwd`.
    pub(crate) fn actual_per_rank(&self, total_forward_passes: usize) -> Vec<usize> {
        let base_fwd = total_forward_passes / self.num_ranks;
        let remainder_fwd = total_forward_passes % self.num_ranks;
        (0..self.num_ranks)
            .map(|r| base_fwd + usize::from(r < remainder_fwd))
            .collect()
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
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};

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
            let rd = RankDistribution::new(&comm, 5, 8, 10);

            assert_eq!(rd.num_ranks, 3, "rank {rank}: num_ranks");
            assert_eq!(rd.num_stages, 5, "rank {rank}: num_stages");
            assert_eq!(rd.n_state, 10, "rank {rank}: n_state");
            assert_eq!(
                rd.my_actual_fwd, expected_actual[rank],
                "rank {rank}: my_actual_fwd"
            );
            assert_eq!(
                rd.my_fwd_offset, expected_offset[rank],
                "rank {rank}: my_fwd_offset"
            );
            assert_eq!(rd.max_local_fwd, 3, "rank {rank}: max_local_fwd");
        }
    }

    #[test]
    fn rank_distribution_actual_per_rank_is_consistent_with_my_actual_fwd() {
        for rank in 0..3 {
            let comm = StubCommN { rank, size: 3 };
            let rd = RankDistribution::new(&comm, 5, 8, 10);

            let per_rank = rd.actual_per_rank(8);
            assert_eq!(per_rank, vec![3, 3, 2], "rank {rank}: per_rank vec");
            assert_eq!(
                per_rank[rd.my_rank], rd.my_actual_fwd,
                "rank {rank}: per_rank[my_rank] == my_actual_fwd"
            );
        }
    }
}
