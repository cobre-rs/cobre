//! State vector exchange between MPI ranks after the forward pass.
//!
//! After the forward pass completes, each MPI rank holds `local_count`
//! trajectory records per stage. The backward pass requires evaluating the
//! cost-to-go at **all** trial points from **all** ranks. Before the backward
//! pass begins, the visited state vectors must be gathered across ranks using
//! [`Communicator::allgatherv`].
//!
//! [`ExchangeBuffers`] pre-allocates send and receive buffers once at
//! construction time and reuses them across all stages and iterations, keeping
//! the per-stage exchange on the hot path allocation-free.
//!
//! ## Memory layout
//!
//! After a successful [`ExchangeBuffers::exchange`] call, the receive buffer
//! contains all ranks' state vectors for the given stage in rank-major order:
//!
//! ```text
//! recv_buf[r * local_count * n_state + m * n_state .. + n_state]
//! ```
//!
//! holds rank `r`'s local scenario `m` state vector.
//!
//! ## Single-rank mode
//!
//! When `num_ranks == 1`, [`LocalBackend`](cobre_comm::LocalBackend)'s
//! `allgatherv` performs an identity copy. No special-casing is needed in this
//! module.
//!
//! [`Communicator::allgatherv`]: cobre_comm::Communicator::allgatherv

use cobre_comm::Communicator;

use crate::{SddpError, TrajectoryRecord};

/// Pre-allocated buffers for gathering state vectors across all MPI ranks.
///
/// Holds the send buffer, receive buffer, and the static `counts` and
/// `displs` arrays needed for [`Communicator::allgatherv`]. All allocations
/// happen once in [`ExchangeBuffers::new`] and are reused across stages and
/// iterations, keeping the per-stage exchange allocation-free.
///
/// # Buffer layout
///
/// | Buffer      | Length                           | Description                                 |
/// |-------------|----------------------------------|---------------------------------------------|
/// | `send_buf`  | `local_count * n_state`          | This rank's state vectors, packed contiguously |
/// | `recv_buf`  | `local_count * num_ranks * n_state` | All ranks' state vectors in rank-major order |
/// | `counts`    | `num_ranks`                      | Each entry = `local_count * n_state`        |
/// | `displs`    | `num_ranks`                      | Entry `r` = `r * local_count * n_state`     |
///
/// # Examples
///
/// ```rust
/// use cobre_comm::LocalBackend;
/// use cobre_sddp::{ExchangeBuffers, TrajectoryRecord};
///
/// // Three scenarios, two-element state vectors, single rank.
/// let mut bufs = ExchangeBuffers::new(2, 3, 1);
///
/// let records: Vec<TrajectoryRecord> = vec![
///     TrajectoryRecord { primal: vec![], dual: vec![], stage_cost: 0.0, state: vec![1.0, 2.0] },
///     TrajectoryRecord { primal: vec![], dual: vec![], stage_cost: 0.0, state: vec![3.0, 4.0] },
///     TrajectoryRecord { primal: vec![], dual: vec![], stage_cost: 0.0, state: vec![5.0, 6.0] },
/// ];
///
/// let comm = LocalBackend;
/// bufs.exchange(&records, 0, 1, &comm).unwrap();
///
/// assert_eq!(bufs.gathered_states(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(bufs.total_scenarios(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct ExchangeBuffers {
    /// Send buffer: this rank's state vectors packed contiguously.
    ///
    /// Length `local_count * n_state`. Entry `m * n_state .. (m+1) * n_state`
    /// is local scenario `m`'s state vector at the current stage.
    send_buf: Vec<f64>,

    /// Receive buffer: all ranks' state vectors in rank-major order.
    ///
    /// Length `local_count * num_ranks * n_state`. After a successful
    /// `exchange`, the slice `recv_buf[r * local_count * n_state + m * n_state..]`
    /// (first `n_state` elements) holds rank `r`'s scenario `m` state vector.
    recv_buf: Vec<f64>,

    /// Per-rank element count for `allgatherv`.
    ///
    /// Length `num_ranks`. All entries equal `local_count * n_state`.
    counts: Vec<usize>,

    /// Per-rank displacement for `allgatherv`.
    ///
    /// Length `num_ranks`. Entry `r` = `r * local_count * n_state`.
    displs: Vec<usize>,

    /// Length of each state vector.
    n_state: usize,

    /// Number of local scenarios (forward passes per rank).
    local_count: usize,

    /// Total number of MPI ranks.
    num_ranks: usize,
}

impl ExchangeBuffers {
    /// Construct pre-allocated exchange buffers for the given topology.
    ///
    /// All allocations occur here. Subsequent calls to [`exchange`](Self::exchange)
    /// reuse these buffers without allocating.
    ///
    /// # Arguments
    ///
    /// - `n_state` — length of each state vector (`N * (1 + L)` from the
    ///   stage indexer).
    /// - `local_count` — number of local scenarios (`config.forward_passes`).
    /// - `num_ranks` — total number of MPI ranks (`comm.size()`).
    #[must_use]
    pub fn new(n_state: usize, local_count: usize, num_ranks: usize) -> Self {
        let per_rank = local_count * n_state;
        let total = per_rank * num_ranks;

        let counts: Vec<usize> = vec![per_rank; num_ranks];
        let displs: Vec<usize> = (0..num_ranks).map(|r| r * per_rank).collect();

        Self {
            send_buf: vec![0.0_f64; per_rank],
            recv_buf: vec![0.0_f64; total],
            counts,
            displs,
            n_state,
            local_count,
            num_ranks,
        }
    }

    /// Gather all ranks' state vectors for `stage` into the receive buffer.
    ///
    /// Packs the send buffer with this rank's state vectors for the requested
    /// stage (one per local scenario), then calls `allgatherv` to distribute
    /// them across all ranks. After this returns `Ok(())`, the receive buffer
    /// contains all `local_count * num_ranks` state vectors in rank-major order.
    ///
    /// # Arguments
    ///
    /// - `records` — flat trajectory record slice from the forward pass, indexed
    ///   as `records[scenario * num_stages + stage]`.
    /// - `stage` — 0-based stage index for which to exchange states.
    /// - `num_stages` — total number of stages (for record index arithmetic).
    /// - `comm` — the communicator for the `allgatherv` call.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Communication(_))` if the underlying
    /// `allgatherv` call fails. The receive buffer contents are unspecified
    /// on error.
    ///
    /// # Panics (debug builds only)
    ///
    /// Asserts that `records` is long enough to contain all
    /// `local_count * num_stages` records.
    pub fn exchange<C: Communicator>(
        &mut self,
        records: &[TrajectoryRecord],
        stage: usize,
        num_stages: usize,
        comm: &C,
    ) -> Result<(), SddpError> {
        debug_assert!(
            records.len() >= self.local_count * num_stages,
            "records.len() {} < local_count {} * num_stages {}",
            records.len(),
            self.local_count,
            num_stages,
        );

        for m in 0..self.local_count {
            let record_idx = m * num_stages + stage;
            debug_assert_eq!(
                records[record_idx].state.len(),
                self.n_state,
                "records[{record_idx}].state.len() {} != n_state {}",
                records[record_idx].state.len(),
                self.n_state,
            );
            self.send_buf[m * self.n_state..(m + 1) * self.n_state]
                .copy_from_slice(&records[record_idx].state);
        }

        comm.allgatherv(
            &self.send_buf,
            &mut self.recv_buf,
            &self.counts,
            &self.displs,
        )?;

        Ok(())
    }

    /// Return a view of the full receive buffer after a successful [`exchange`](Self::exchange).
    ///
    /// The buffer is in rank-major order: rank `r`'s scenario `m` state vector
    /// occupies `gathered_states()[r * local_count * n_state + m * n_state..][..n_state]`.
    #[must_use]
    pub fn gathered_states(&self) -> &[f64] {
        &self.recv_buf
    }

    /// Return the state vector for a specific rank and local scenario.
    ///
    /// # Arguments
    ///
    /// - `rank` — the MPI rank index (`0..num_ranks`).
    /// - `scenario` — the local scenario index (`0..local_count`).
    ///
    /// Returns a slice of length `n_state`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Asserts that `rank < num_ranks` and `scenario < local_count`.
    #[must_use]
    pub fn state_at(&self, rank: usize, scenario: usize) -> &[f64] {
        debug_assert!(
            rank < self.num_ranks,
            "rank {rank} >= num_ranks {}",
            self.num_ranks
        );
        debug_assert!(
            scenario < self.local_count,
            "scenario {scenario} >= local_count {}",
            self.local_count
        );
        let base = rank * self.local_count * self.n_state + scenario * self.n_state;
        &self.recv_buf[base..base + self.n_state]
    }

    /// Return the number of local scenarios per rank.
    ///
    /// Equal to `config.forward_passes` at construction time. Used by the
    /// backward pass to decompose a global scenario index `m` into
    /// `(rank, local_scenario)` via `(m / local_count(), m % local_count())`.
    #[must_use]
    pub fn local_count(&self) -> usize {
        self.local_count
    }

    /// Return the total number of scenarios gathered from all ranks.
    ///
    /// Equal to `local_count * num_ranks`.
    #[must_use]
    pub fn total_scenarios(&self) -> usize {
        self.local_count * self.num_ranks
    }
}

#[cfg(test)]
mod tests {
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};

    use super::ExchangeBuffers;
    use crate::TrajectoryRecord;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn make_record(state: Vec<f64>) -> TrajectoryRecord {
        TrajectoryRecord {
            primal: vec![],
            dual: vec![],
            stage_cost: 0.0,
            state,
        }
    }

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn new_allocates_correct_send_buf_length() {
        let bufs = ExchangeBuffers::new(3, 4, 2);
        // send_buf: local_count * n_state = 4 * 3 = 12
        assert_eq!(bufs.send_buf.len(), 12);
    }

    #[test]
    fn new_allocates_correct_recv_buf_length() {
        let bufs = ExchangeBuffers::new(3, 4, 2);
        // recv_buf: local_count * num_ranks * n_state = 4 * 2 * 3 = 24
        assert_eq!(bufs.recv_buf.len(), 24);
    }

    #[test]
    fn new_allocates_correct_counts_length_and_values() {
        let bufs = ExchangeBuffers::new(3, 4, 2);
        // counts: [local_count * n_state; num_ranks] = [12, 12]
        assert_eq!(bufs.counts.len(), 2);
        assert_eq!(bufs.counts[0], 12);
        assert_eq!(bufs.counts[1], 12);
    }

    #[test]
    fn new_allocates_correct_displs_length_and_values() {
        let bufs = ExchangeBuffers::new(3, 4, 2);
        // displs: [0, 12]
        assert_eq!(bufs.displs.len(), 2);
        assert_eq!(bufs.displs[0], 0);
        assert_eq!(bufs.displs[1], 12);
    }

    #[test]
    fn new_single_rank_counts_is_one_element() {
        // Acceptance criterion: counts = [local_count * n_state], displs = [0]
        let bufs = ExchangeBuffers::new(2, 3, 1);
        assert_eq!(bufs.counts, vec![6]); // 3 * 2
        assert_eq!(bufs.displs, vec![0]);
    }

    #[test]
    fn total_scenarios_returns_local_count_times_num_ranks() {
        let bufs = ExchangeBuffers::new(2, 5, 4);
        assert_eq!(bufs.total_scenarios(), 20); // 5 * 4
    }

    #[test]
    fn total_scenarios_single_rank() {
        let bufs = ExchangeBuffers::new(2, 3, 1);
        assert_eq!(bufs.total_scenarios(), 3);
    }

    #[test]
    fn state_at_indexing_arithmetic() {
        // Manually verify state_at for n_state=3, local_count=2, num_ranks=3.
        // recv_buf layout: [r0s0 r0s1 r1s0 r1s1 r2s0 r2s1] (each entry is 3 f64s)
        let n_state = 3;
        let local_count = 2;
        let num_ranks = 3;
        let mut bufs = ExchangeBuffers::new(n_state, local_count, num_ranks);

        // Manually fill recv_buf with identifiable values.
        // Indices are small (r < 3, s < 2, i < 3): sum <= 212, exact in f64.
        #[allow(clippy::cast_precision_loss)]
        for r in 0..num_ranks {
            for s in 0..local_count {
                let base = r * local_count * n_state + s * n_state;
                for i in 0..n_state {
                    bufs.recv_buf[base + i] = (r * 100 + s * 10 + i) as f64;
                }
            }
        }

        // Verify state_at(1, 0) = rank 1, scenario 0 → starts at index 6.
        let slice = bufs.state_at(1, 0);
        assert_eq!(slice.len(), n_state);
        assert_eq!(slice[0], 100.0); // r=1, s=0, i=0
        assert_eq!(slice[1], 101.0); // r=1, s=0, i=1
        assert_eq!(slice[2], 102.0); // r=1, s=0, i=2

        // Verify state_at(2, 1) = rank 2, scenario 1 → starts at index 15.
        let slice = bufs.state_at(2, 1);
        assert_eq!(slice[0], 210.0); // r=2, s=1, i=0
        assert_eq!(slice[1], 211.0); // r=2, s=1, i=1
        assert_eq!(slice[2], 212.0); // r=2, s=1, i=2
    }

    // ── Integration tests (round-trip with LocalBackend) ───────────────────────

    #[test]
    fn exchange_single_rank_three_scenarios_two_state() {
        // Acceptance criterion AC1: n_state=2, local_count=3, num_ranks=1,
        // stage 0 of a 1-stage system. gathered_states = [1,2,3,4,5,6].
        use cobre_comm::LocalBackend;

        let mut bufs = ExchangeBuffers::new(2, 3, 1);
        let records = vec![
            make_record(vec![1.0, 2.0]),
            make_record(vec![3.0, 4.0]),
            make_record(vec![5.0, 6.0]),
        ];

        let comm = LocalBackend;
        bufs.exchange(&records, 0, 1, &comm).unwrap();

        assert_eq!(bufs.gathered_states(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(bufs.total_scenarios(), 3);
    }

    #[test]
    fn exchange_selects_correct_stage_in_multi_stage_records() {
        // Acceptance criterion AC2: verify that stage 1 is selected from a
        // 3-stage system (records indexed at scenario * 3 + stage).
        use cobre_comm::LocalBackend;

        // 2 scenarios, 3 stages, n_state=2
        // records[m * 3 + stage]:
        //   stage 0: [10, 11], [20, 21]
        //   stage 1: [30, 31], [40, 41]  ← target
        //   stage 2: [50, 51], [60, 61]
        let records = vec![
            make_record(vec![10.0, 11.0]), // m=0, stage=0
            make_record(vec![30.0, 31.0]), // m=0, stage=1
            make_record(vec![50.0, 51.0]), // m=0, stage=2
            make_record(vec![20.0, 21.0]), // m=1, stage=0
            make_record(vec![40.0, 41.0]), // m=1, stage=1
            make_record(vec![60.0, 61.0]), // m=1, stage=2
        ];

        let mut bufs = ExchangeBuffers::new(2, 2, 1);
        let comm = LocalBackend;
        bufs.exchange(&records, 1, 3, &comm).unwrap();

        // Stage 1 data: scenario 0 → [30, 31], scenario 1 → [40, 41]
        assert_eq!(bufs.gathered_states(), &[30.0, 31.0, 40.0, 41.0]);

        // Verify adjacent stages are not mixed in.
        assert_ne!(bufs.gathered_states()[0], 10.0, "stage 0 must not appear");
        assert_ne!(bufs.gathered_states()[0], 50.0, "stage 2 must not appear");
    }

    #[test]
    fn state_at_matches_record_state_after_exchange() {
        // Acceptance criterion AC3: state_at(0, 1) matches records[1 * num_stages + stage].state
        use cobre_comm::LocalBackend;

        // 2 scenarios, 1 stage, n_state=3
        let records = vec![
            make_record(vec![1.0, 2.0, 3.0]), // m=0, stage=0
            make_record(vec![4.0, 5.0, 6.0]), // m=1, stage=0
        ];

        let mut bufs = ExchangeBuffers::new(3, 2, 1);
        let comm = LocalBackend;
        bufs.exchange(&records, 0, 1, &comm).unwrap();

        // state_at(rank=0, scenario=1) must match records[1 * 1 + 0].state = [4, 5, 6]
        let state = bufs.state_at(0, 1);
        assert_eq!(state, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn exchange_error_maps_to_sddp_communication_error() {
        // Acceptance criterion AC5: a failing communicator returns
        // SddpError::Communication(_).
        use crate::SddpError;

        struct FailingComm;

        impl Communicator for FailingComm {
            fn allgatherv<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                Err(CommError::CollectiveFailed {
                    operation: "allgatherv",
                    mpi_error_code: 42,
                    message: "simulated failure".to_string(),
                })
            }

            fn allreduce<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _op: ReduceOp,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn barrier(&self) -> Result<(), CommError> {
                unreachable!()
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                1
            }
        }

        let mut bufs = ExchangeBuffers::new(2, 1, 1);
        let records = vec![make_record(vec![1.0, 2.0])];

        let result = bufs.exchange(&records, 0, 1, &FailingComm);
        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "expected SddpError::Communication, got: {result:?}",
        );
    }
}
