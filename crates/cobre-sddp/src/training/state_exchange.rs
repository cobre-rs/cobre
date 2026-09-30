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

use crate::{
    error::SddpError, lp::indexer::StateSpace, setup::node_graph::StageIdx,
    trajectory::TrajectoryRecord,
};

/// Pre-allocated buffers for gathering state vectors across all MPI ranks.
///
/// Holds the send buffer, receive buffer, and the static `counts` and
/// `displs` arrays needed for [`Communicator::allgatherv`]. All allocations
/// happen once in [`ExchangeBuffers::new`] and are reused across stages and
/// iterations, keeping the per-stage exchange allocation-free.
///
/// ## Padded layout and real counts
///
/// When `total_forward_passes` does not divide evenly among ranks, some ranks
/// hold fewer actual forward passes than `local_count` (the max). The receive
/// buffer is always sized uniformly to `local_count * num_ranks * n_state` so
/// that `allgatherv` uses identical counts and displacements per rank. Ranks
/// with fewer actual forward passes zero-pad their trailing send-buffer slots.
///
/// The true total forward pass count is tracked in `real_total` and exposed
/// via [`real_total_scenarios`](Self::real_total_scenarios). Use
/// [`pack_real_states_into`](Self::pack_real_states_into) to extract only the
/// non-padded state vectors from the receive buffer.
///
/// # Buffer layout
///
/// | Buffer         | Length                              | Description                                  |
/// |----------------|-------------------------------------|----------------------------------------------|
/// | `send_buf`     | `local_count * n_state`             | This rank's state vectors, packed contiguously |
/// | `recv_buf`     | `local_count * num_ranks * n_state` | All ranks' state vectors in rank-major order  |
/// | `counts`       | `num_ranks`                         | Each entry = `local_count * n_state`         |
/// | `displs`       | `num_ranks`                         | Entry `r` = `r * local_count * n_state`      |
/// | `actual_counts`| `num_ranks`                         | Actual forward passes per rank (≤ `local_count`) |
///
/// # Examples
///
/// ```rust
/// use cobre_comm::LocalBackend;
/// use cobre_sddp::ExchangeBuffers;
/// use cobre_sddp::TrajectoryRecord;
/// use cobre_sddp::indexer::StateSpace;
/// use cobre_sddp::lead_time::AnticipatedResolution;
/// use cobre_sddp::setup::NodeId;
/// use cobre_sddp::setup::StageIdx;
///
/// // Three scenarios, two-element state vectors, single rank.
/// let state = StateSpace::new(
///     2,
///     0,
///     Vec::new(),
///     Vec::new(),
///     AnticipatedResolution::default(),
///     &[0, 0],
/// );
/// let mut bufs = ExchangeBuffers::new(&state, 3, 1);
///
/// let records: Vec<TrajectoryRecord> = vec![
///     TrajectoryRecord { primal: vec![], dual: vec![], stage_cost: 0.0, node_id: NodeId(0), state: vec![1.0, 2.0] },
///     TrajectoryRecord { primal: vec![], dual: vec![], stage_cost: 0.0, node_id: NodeId(0), state: vec![3.0, 4.0] },
///     TrajectoryRecord { primal: vec![], dual: vec![], stage_cost: 0.0, node_id: NodeId(0), state: vec![5.0, 6.0] },
/// ];
///
/// let comm = LocalBackend;
/// bufs.exchange(&records, StageIdx(0), 1, &comm).unwrap();
///
/// assert_eq!(bufs.gathered_states(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(bufs.total_scenarios(), 3);
/// assert_eq!(bufs.real_total_scenarios(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct ExchangeBuffers {
    // Buffer shapes and roles (send_buf, recv_buf, counts, displs, n_state,
    // local_count, num_ranks) are the "Buffer layout" table above — not
    // repeated per field.
    send_buf: Vec<f64>,
    recv_buf: Vec<f64>,
    counts: Vec<usize>,
    displs: Vec<usize>,
    n_state: usize,
    local_count: usize,
    num_ranks: usize,

    /// Actual forward pass count per rank (may be less than `local_count` for
    /// ranks that received fewer passes in a non-divisible distribution).
    ///
    /// Length `num_ranks`. Sum equals `real_total`.
    actual_counts: Vec<usize>,

    /// True total forward passes across all ranks (sum of `actual_counts`).
    ///
    /// Equal to `local_count * num_ranks` when the distribution is even.
    real_total: usize,
}

// Several constructors and accessors on `ExchangeBuffers` (`new`,
// `gathered_states`, `local_count`, `total_scenarios`) are part of the
// stable shape callers may want for inspection or for the uniform-rank
// distribution path. Production drivers go through `with_actual_counts`
// and pack helpers; these are retained for symmetry and tests.
#[allow(dead_code)]
impl ExchangeBuffers {
    /// Construct pre-allocated exchange buffers for the given topology.
    ///
    /// All allocations occur here. Subsequent calls to [`exchange`](Self::exchange)
    /// reuse these buffers without allocating.
    ///
    /// When `total_forward_passes` divides evenly among ranks, this is
    /// equivalent to calling [`with_actual_counts`](Self::with_actual_counts)
    /// with uniform `actual_per_rank`. For uneven distributions, prefer
    /// `with_actual_counts` so that [`real_total_scenarios`](Self::real_total_scenarios)
    /// and [`pack_real_states_into`](Self::pack_real_states_into) return
    /// correct results.
    ///
    /// # Arguments
    ///
    /// - `state` — state-vector layout owner; each state vector has length
    ///   `state.n_state`.
    /// - `local_count` — number of local scenarios per rank (the maximum across
    ///   all ranks when the distribution is uneven).
    /// - `num_ranks` — total number of MPI ranks (`comm.size()`).
    #[must_use]
    pub fn new(state: &StateSpace, local_count: usize, num_ranks: usize) -> Self {
        let actual_per_rank = vec![local_count; num_ranks];
        Self::with_actual_counts(state, local_count, num_ranks, &actual_per_rank)
    }

    /// Construct pre-allocated exchange buffers with per-rank actual forward
    /// pass counts.
    ///
    /// Use this constructor when the total number of forward passes does not
    /// divide evenly among ranks so that
    /// [`real_total_scenarios`](Self::real_total_scenarios) and
    /// [`pack_real_states_into`](Self::pack_real_states_into) return correct
    /// results.
    ///
    /// The `allgatherv` buffer sizing still uses `max_local_count` uniformly
    /// for all ranks (the padded layout). Only the archival helpers use
    /// `actual_per_rank`.
    ///
    /// # Arguments
    ///
    /// - `state` — state-vector layout owner; each state vector has length
    ///   `state.n_state`.
    /// - `max_local_count` — maximum forward passes on any single rank (used
    ///   for uniform buffer sizing).
    /// - `num_ranks` — total number of MPI ranks.
    /// - `actual_per_rank` — slice of length `num_ranks` with the actual
    ///   forward pass count for each rank. All entries must be
    ///   `≤ max_local_count`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Asserts that `actual_per_rank.len() == num_ranks` and that each entry
    /// is `≤ max_local_count`.
    #[must_use]
    pub fn with_actual_counts(
        state: &StateSpace,
        max_local_count: usize,
        num_ranks: usize,
        actual_per_rank: &[usize],
    ) -> Self {
        let n_state = state.n_state;
        debug_assert_eq!(
            actual_per_rank.len(),
            num_ranks,
            "actual_per_rank.len() {} != num_ranks {}",
            actual_per_rank.len(),
            num_ranks,
        );
        debug_assert!(
            actual_per_rank.iter().all(|&c| c <= max_local_count),
            "all actual_per_rank entries must be <= max_local_count {max_local_count}",
        );

        let per_rank = max_local_count * n_state;
        let total = per_rank * num_ranks;
        let real_total: usize = actual_per_rank.iter().sum();

        let counts: Vec<usize> = vec![per_rank; num_ranks];
        let displs: Vec<usize> = (0..num_ranks).map(|r| r * per_rank).collect();

        Self {
            send_buf: vec![0.0_f64; per_rank],
            recv_buf: vec![0.0_f64; total],
            counts,
            displs,
            n_state,
            local_count: max_local_count,
            num_ranks,
            actual_counts: actual_per_rank.to_vec(),
            real_total,
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
        stage: StageIdx,
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
            let record_idx = m * num_stages + stage.0;
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
    /// Equal to `local_count * num_ranks`. This is the padded buffer size and
    /// may exceed the true forward pass count when the distribution is uneven.
    /// Use [`real_total_scenarios`](Self::real_total_scenarios) to obtain the
    /// true count.
    #[must_use]
    pub fn total_scenarios(&self) -> usize {
        self.local_count * self.num_ranks
    }

    /// Return the true total number of forward passes across all ranks.
    ///
    /// When the total forward passes distribute evenly across ranks this equals
    /// [`total_scenarios`](Self::total_scenarios). When some ranks have fewer
    /// actual forward passes (non-divisible distribution), this returns the
    /// sum of actual counts, which is less than `total_scenarios`.
    #[must_use]
    pub fn real_total_scenarios(&self) -> usize {
        self.real_total
    }

    /// Copy only the real (non-padded) state vectors from the receive buffer
    /// into `buf`.
    ///
    /// After a successful [`exchange`](Self::exchange), the receive buffer
    /// contains `local_count * num_ranks` state slots in rank-major order.
    /// Ranks with fewer actual forward passes have zero-padded trailing slots.
    /// This method copies only the `actual_counts[r]` real slots for each rank
    /// `r` into `buf`, producing a compact slice of
    /// [`real_total_scenarios`](Self::real_total_scenarios) state vectors.
    ///
    /// The output length equals `real_total_scenarios() * n_state`.
    ///
    /// # Arguments
    ///
    /// - `buf` — destination buffer. It is cleared and then filled with the
    ///   real state data. Capacity is preserved between calls.
    pub fn pack_real_states_into(&self, buf: &mut Vec<f64>) {
        buf.clear();
        for r in 0..self.num_ranks {
            let base = r * self.local_count * self.n_state;
            let real_len = self.actual_counts[r] * self.n_state;
            buf.extend_from_slice(&self.recv_buf[base..base + real_len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use cobre_comm::{CommData, CommError, Communicator, LocalBackend, ReduceOp};

    use super::ExchangeBuffers;
    use crate::lp::indexer::StateSpace;
    use crate::setup::NodeId;
    use crate::setup::node_graph::StageIdx;
    use crate::test_support;
    use crate::trajectory::TrajectoryRecord;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn make_record(state: Vec<f64>) -> TrajectoryRecord {
        TrajectoryRecord {
            primal: vec![],
            dual: vec![],
            stage_cost: 0.0,
            node_id: NodeId(0),
            state,
        }
    }

    fn state_of(n_state: usize) -> StateSpace {
        let state = test_support::state_layout(n_state, 0);
        assert_eq!(state.n_state, n_state);
        state
    }

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn new_allocates_correct_send_buf_length() {
        let bufs = ExchangeBuffers::new(&state_of(3), 4, 2);
        // send_buf: local_count * n_state = 4 * 3 = 12
        assert_eq!(bufs.send_buf.len(), 12);
    }

    #[test]
    fn new_allocates_correct_recv_buf_length() {
        let bufs = ExchangeBuffers::new(&state_of(3), 4, 2);
        // recv_buf: local_count * num_ranks * n_state = 4 * 2 * 3 = 24
        assert_eq!(bufs.recv_buf.len(), 24);
    }

    #[test]
    fn new_allocates_correct_counts_length_and_values() {
        let bufs = ExchangeBuffers::new(&state_of(3), 4, 2);
        // counts: [local_count * n_state; num_ranks] = [12, 12]
        assert_eq!(bufs.counts.len(), 2);
        assert_eq!(bufs.counts[0], 12);
        assert_eq!(bufs.counts[1], 12);
    }

    #[test]
    fn new_allocates_correct_displs_length_and_values() {
        let bufs = ExchangeBuffers::new(&state_of(3), 4, 2);
        // displs: [0, 12]
        assert_eq!(bufs.displs.len(), 2);
        assert_eq!(bufs.displs[0], 0);
        assert_eq!(bufs.displs[1], 12);
    }

    #[test]
    fn new_single_rank_counts_is_one_element() {
        // Acceptance criterion: counts = [local_count * n_state], displs = [0]
        let bufs = ExchangeBuffers::new(&state_of(2), 3, 1);
        assert_eq!(bufs.counts, vec![6]); // 3 * 2
        assert_eq!(bufs.displs, vec![0]);
    }

    #[test]
    fn total_scenarios_returns_local_count_times_num_ranks() {
        let bufs = ExchangeBuffers::new(&state_of(2), 5, 4);
        assert_eq!(bufs.total_scenarios(), 20); // 5 * 4
    }

    #[test]
    fn total_scenarios_single_rank() {
        let bufs = ExchangeBuffers::new(&state_of(2), 3, 1);
        assert_eq!(bufs.total_scenarios(), 3);
    }

    #[test]
    fn state_at_indexing_arithmetic() {
        // Manually verify state_at for n_state=3, local_count=2, num_ranks=3.
        // recv_buf layout: [r0s0 r0s1 r1s0 r1s1 r2s0 r2s1] (each entry is 3 f64s)
        let n_state = 3;
        let local_count = 2;
        let num_ranks = 3;
        let mut bufs = ExchangeBuffers::new(&state_of(n_state), local_count, num_ranks);

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
        let mut bufs = ExchangeBuffers::new(&state_of(2), 3, 1);
        let records = vec![
            make_record(vec![1.0, 2.0]),
            make_record(vec![3.0, 4.0]),
            make_record(vec![5.0, 6.0]),
        ];

        let comm = LocalBackend;
        bufs.exchange(&records, StageIdx(0), 1, &comm).unwrap();

        assert_eq!(bufs.gathered_states(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(bufs.total_scenarios(), 3);
    }

    #[test]
    fn exchange_selects_correct_stage_in_multi_stage_records() {
        // Acceptance criterion AC2: verify that stage 1 is selected from a
        // 3-stage system (records indexed at scenario * 3 + stage).
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

        let mut bufs = ExchangeBuffers::new(&state_of(2), 2, 1);
        let comm = LocalBackend;
        bufs.exchange(&records, StageIdx(1), 3, &comm).unwrap();

        // Stage 1 data: scenario 0 → [30, 31], scenario 1 → [40, 41]
        assert_eq!(bufs.gathered_states(), &[30.0, 31.0, 40.0, 41.0]);

        // Verify adjacent stages are not mixed in.
        assert_ne!(bufs.gathered_states()[0], 10.0, "stage 0 must not appear");
        assert_ne!(bufs.gathered_states()[0], 50.0, "stage 2 must not appear");
    }

    #[test]
    fn state_at_matches_record_state_after_exchange() {
        // Acceptance criterion AC3: state_at(0, 1) matches records[1 * num_stages + stage].state
        // 2 scenarios, 1 stage, n_state=3
        let records = vec![
            make_record(vec![1.0, 2.0, 3.0]), // m=0, stage=0
            make_record(vec![4.0, 5.0, 6.0]), // m=1, stage=0
        ];

        let mut bufs = ExchangeBuffers::new(&state_of(3), 2, 1);
        let comm = LocalBackend;
        bufs.exchange(&records, StageIdx(0), 1, &comm).unwrap();

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

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let mut bufs = ExchangeBuffers::new(&state_of(2), 1, 1);
        let records = vec![make_record(vec![1.0, 2.0])];

        let result = bufs.exchange(&records, StageIdx(0), 1, &FailingComm);
        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "expected SddpError::Communication, got: {result:?}",
        );
    }

    // ── with_actual_counts / real_total_scenarios / pack_real_states_into ─────

    #[test]
    fn real_total_scenarios_uneven_distribution() {
        // AC: total_forward_passes=5, num_ranks=2, actual_per_rank=[3,2].
        // real_total_scenarios() must return 5, not 6.
        let bufs = ExchangeBuffers::with_actual_counts(&state_of(2), 3, 2, &[3, 2]);
        assert_eq!(bufs.real_total_scenarios(), 5);
        assert_eq!(bufs.total_scenarios(), 6); // padded buffer size unchanged
    }

    #[test]
    fn real_total_scenarios_even_distribution() {
        // AC: even split, real_total_scenarios() == total_scenarios().
        let bufs = ExchangeBuffers::with_actual_counts(&state_of(2), 3, 2, &[3, 3]);
        assert_eq!(bufs.real_total_scenarios(), 6);
        assert_eq!(bufs.total_scenarios(), 6);
    }

    #[test]
    fn real_total_scenarios_single_rank() {
        // AC: num_ranks==1, real_total_scenarios() == total_scenarios().
        let bufs = ExchangeBuffers::with_actual_counts(&state_of(2), 3, 1, &[3]);
        assert_eq!(bufs.real_total_scenarios(), 3);
        assert_eq!(bufs.total_scenarios(), 3);
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn pack_real_states_into_excludes_padding() {
        // AC: max_local_count=3, num_ranks=2, actual_per_rank=[3,2], n_state=2.
        // Rank 0 has 3 real states, rank 1 has 2 real states + 1 zero-padded slot.
        // pack_real_states_into must return exactly 5 state vectors (10 f64 values).
        let n_state = 2;
        let max_local = 3;
        let num_ranks = 2;
        let mut bufs =
            ExchangeBuffers::with_actual_counts(&state_of(n_state), max_local, num_ranks, &[3, 2]);

        // Manually fill recv_buf:
        // rank 0: slots [10,11], [20,21], [30,31]   (3 real)
        // rank 1: slots [40,41], [50,51], [0, 0]    (2 real + 1 padding)
        let rank0_base = 0 * max_local * n_state; // 0
        let rank1_base = 1 * max_local * n_state; // 6
        bufs.recv_buf[rank0_base..rank0_base + 2].copy_from_slice(&[10.0, 11.0]);
        bufs.recv_buf[rank0_base + 2..rank0_base + 4].copy_from_slice(&[20.0, 21.0]);
        bufs.recv_buf[rank0_base + 4..rank0_base + 6].copy_from_slice(&[30.0, 31.0]);
        bufs.recv_buf[rank1_base..rank1_base + 2].copy_from_slice(&[40.0, 41.0]);
        bufs.recv_buf[rank1_base + 2..rank1_base + 4].copy_from_slice(&[50.0, 51.0]);
        // padding slot stays zero: recv_buf[rank1_base+4..rank1_base+6] = [0.0, 0.0]

        let mut out = Vec::new();
        bufs.pack_real_states_into(&mut out);

        // Expect 5 vectors × 2 elements = 10 values, zero-padded slot excluded.
        assert_eq!(
            out.len(),
            10,
            "expected 10 f64 values (5 state vectors × 2)"
        );
        assert_eq!(
            out,
            vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0]
        );
    }

    #[test]
    fn pack_real_states_into_even_distribution_matches_gathered_states() {
        // When distribution is even, pack_real_states_into must return the
        // same data as gathered_states (in the same order).
        let mut bufs = ExchangeBuffers::new(&state_of(2), 3, 1);
        let records = vec![
            make_record(vec![1.0, 2.0]),
            make_record(vec![3.0, 4.0]),
            make_record(vec![5.0, 6.0]),
        ];
        let comm = LocalBackend;
        bufs.exchange(&records, StageIdx(0), 1, &comm).unwrap();

        let mut packed = Vec::new();
        bufs.pack_real_states_into(&mut packed);
        assert_eq!(packed, bufs.gathered_states());
    }

    #[test]
    fn pack_real_states_into_reuses_buffer_capacity() {
        // Calling pack_real_states_into twice must clear and refill the buffer
        // without changing its capacity (capacity is preserved after clear).
        let mut bufs = ExchangeBuffers::with_actual_counts(&state_of(2), 3, 2, &[3, 2]);
        // Fill all recv_buf slots with 1.0 for simplicity.
        bufs.recv_buf.fill(1.0);

        let mut out = Vec::with_capacity(20);
        bufs.pack_real_states_into(&mut out);
        assert_eq!(out.len(), 10); // 5 real states × 2

        bufs.pack_real_states_into(&mut out);
        assert_eq!(out.len(), 10); // still 10 after second call (clear + refill)
    }

    // ── zero-work rank ────────────────────────────────────────────────────────

    /// Rank 1 of a 2-rank world whose `allgatherv` honours `displs`, so a test
    /// can tell whether this rank contributed its own slot rather than reading
    /// a peer's.
    struct Rank1Of2;

    impl Communicator for Rank1Of2 {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            displs: &[usize],
        ) -> Result<(), CommError> {
            let start = displs[1];
            recv[start..start + send.len()].clone_from_slice(send);
            Ok(())
        }
        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("exchange only calls allgatherv")
        }
        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("exchange only calls allgatherv")
        }
        fn barrier(&self) -> Result<(), CommError> {
            unreachable!("exchange only calls allgatherv")
        }
        fn rank(&self) -> usize {
            1
        }
        fn size(&self) -> usize {
            2
        }
        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    /// Deadlock-freedom for the per-stage exchange: `total_forward_passes=1`
    /// across 2 ranks leaves rank 1 with zero real trial points, yet it must
    /// still pack and contribute a full-length slot. Every rank's send is
    /// `local_count` (the rank-uniform maximum) states wide, so no rank can have
    /// a reason to sit the `allgatherv` out — a rank that skipped it would
    /// strand its peers.
    #[test]
    fn rank_with_zero_actual_forward_passes_still_contributes_its_gather_slot() {
        let n_state = 1;
        let max_local_fwd = 1;
        let mut bufs =
            ExchangeBuffers::with_actual_counts(&state_of(n_state), max_local_fwd, 2, &[1, 0]);

        // Sized by `max_local_fwd`, as the backward pass receives it: rank 1's
        // slot is padding the forward pass never wrote.
        let records = vec![make_record(vec![10.0]), make_record(vec![20.0])];

        bufs.exchange(&records, StageIdx(0), 2, &Rank1Of2).unwrap();

        assert_eq!(
            bufs.gathered_states()[max_local_fwd * n_state],
            10.0,
            "the zero-work rank must contribute its own padded slot"
        );
    }

    /// The zero-work rank's padding must never reach the archive that drives cut
    /// selection: `real_total_scenarios` and `pack_real_states_into` are keyed on
    /// `actual_counts`, which is rank-uniform, so every rank archives the same
    /// states regardless of which rank drew them.
    #[test]
    fn zero_work_rank_padding_is_excluded_from_the_packed_states() {
        let mut bufs = ExchangeBuffers::with_actual_counts(&state_of(1), 1, 2, &[1, 0]);
        let records = vec![make_record(vec![10.0]), make_record(vec![20.0])];
        bufs.exchange(&records, StageIdx(0), 2, &Rank1Of2).unwrap();

        assert_eq!(
            bufs.real_total_scenarios(),
            1,
            "only rank 0 drew a trial point"
        );

        let mut packed = Vec::new();
        bufs.pack_real_states_into(&mut packed);
        assert_eq!(
            packed.len(),
            1,
            "rank 1's padding slot must not reach the archive"
        );
    }
}
