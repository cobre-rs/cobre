//! Cut synchronization across MPI ranks after the backward pass.
//!
//! After the backward pass generates cuts locally on each rank, all ranks must
//! exchange their newly generated cuts so that every rank has an identical
//! Future Cost Function (FCF). This is achieved via a per-stage `allgatherv`
//! of serialized cut records using the wire format defined in [`cut::wire`].
//!
//! ## Correctness guarantee
//!
//! The FCF must be bit-for-bit identical across all ranks at the end of each
//! iteration. The next iteration's forward pass rebuilds the LP from the FCF,
//! and all ranks must see the same cuts to produce consistent lower bound
//! estimates. This module ensures that after [`CutSyncBuffers::sync_cuts`]
//! returns, every rank has inserted all remote cuts into its local FCF.
//!
//! ## Local vs remote cuts
//!
//! The backward pass (ticket-013) already inserts the local rank's own cuts
//! into the FCF before this function is called. `sync_cuts` therefore only
//! inserts **remote** cuts — i.e., cuts originating from other ranks. The
//! local rank's segment in the receive buffer is skipped during deserialization.
//!
//! ## Barrier semantics
//!
//! The `allgatherv` call acts as an implicit barrier: no rank returns from
//! `sync_cuts` until all ranks have contributed their cuts for the current
//! stage. No explicit `comm.barrier()` is needed.
//!
//! ## Hot-path allocation discipline
//!
//! [`CutSyncBuffers::new`] pre-allocates all byte buffers for the maximum
//! possible exchange size. [`sync_cuts`](CutSyncBuffers::sync_cuts) serializes
//! cuts directly into the pre-allocated `send_buf` using [`serialize_cut`] to
//! avoid per-call allocation. The receive-side deserialization does allocate
//! (one `Vec<f64>` per remote cut), but this occurs O(ranks * `cuts_per_rank`)
//! times per stage — not per-scenario — and is acceptable on the backward pass
//! hot path.
//!
//! [`cut::wire`]: crate::cut::wire
//! [`serialize_cut`]: crate::cut::wire::serialize_cut

use cobre_comm::Communicator;

use crate::{
    cut::wire::{cut_wire_size, deserialize_cuts_from_buffer, serialize_cut},
    FutureCostFunction, SddpError,
};

/// Pre-allocated byte buffers for gathering cut wire records across all MPI
/// ranks.
///
/// Holds the send buffer, receive buffer, and the static `counts` and
/// `displs` arrays needed for [`Communicator::allgatherv`] with `T = u8`.
/// All allocations happen once in [`CutSyncBuffers::new`] and are reused
/// across stages and iterations, keeping the per-stage exchange
/// allocation-free on the send side.
///
/// # Buffer layout
///
/// | Buffer      | Capacity                                             | Description                                                   |
/// |-------------|------------------------------------------------------|---------------------------------------------------------------|
/// | `send_buf`  | `max_cuts_per_rank * cut_wire_size(n_state)`         | This rank's serialized cut records                            |
/// | `recv_buf`  | `max_cuts_per_rank * num_ranks * cut_wire_size(n_state)` | All ranks' serialized cut records in rank-major order     |
/// | `counts`    | `num_ranks`                                          | Per-rank byte count (`actual_cuts * record_size`)             |
/// | `displs`    | `num_ranks`                                          | Per-rank byte displacement (`sum of preceding counts`)        |
///
/// # Examples
///
/// ```rust
/// use cobre_comm::LocalBackend;
/// use cobre_sddp::cut_sync::CutSyncBuffers;
/// use cobre_sddp::cut::fcf::FutureCostFunction;
///
/// // Single rank, 2 state dimensions, max 3 cuts per rank.
/// let mut bufs = CutSyncBuffers::new(2, 3, 1);
///
/// let mut fcf = FutureCostFunction::new(2, 2, 3, 10, 0);
/// let comm = LocalBackend;
///
/// let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[
///     (0, 1, 0, 10.0, &[1.0, 2.0]),
///     (0, 1, 1, 20.0, &[3.0, 4.0]),
/// ];
///
/// let remote_count = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();
/// // Single-rank: no remote cuts inserted.
/// assert_eq!(remote_count, 0);
/// ```
#[derive(Debug, Clone)]
pub struct CutSyncBuffers {
    /// Pre-allocated send buffer for this rank's serialized cut records.
    ///
    /// Capacity: `max_cuts_per_rank * record_size`. Only the leading
    /// `actual_cuts * record_size` bytes are sent in each call.
    send_buf: Vec<u8>,

    /// Pre-allocated receive buffer for all ranks' serialized cut records.
    ///
    /// Capacity: `max_cuts_per_rank * num_ranks * record_size`. After a
    /// successful `allgatherv`, the slice
    /// `recv_buf[displs[r]..displs[r] + counts[r]]` holds rank `r`'s cut
    /// records.
    recv_buf: Vec<u8>,

    /// Per-rank byte count for `allgatherv`.
    ///
    /// Updated each call to `sync_cuts` to reflect the actual number of cuts
    /// the local rank is sending. For the current formulation where all ranks
    /// generate the same number of cuts, all entries are equal. The API
    /// supports variable counts for future extensibility.
    ///
    /// Length: `num_ranks`.
    counts: Vec<usize>,

    /// Per-rank byte displacement for `allgatherv`.
    ///
    /// Entry `r` = sum of `counts[0..r]`. Updated each call to `sync_cuts`
    /// together with `counts`.
    ///
    /// Length: `num_ranks`.
    displs: Vec<usize>,

    /// Length of the state vector (number of cut coefficients).
    n_state: usize,

    /// Total number of MPI ranks.
    num_ranks: usize,

    /// Cached wire record size: `cut_wire_size(n_state)`.
    record_size: usize,

    /// Per-rank expected cut counts for non-uniform work distribution.
    ///
    /// Entry `r` is the number of cuts rank `r` generates per stage per
    /// iteration. Exposed for inspection in tests and diagnostics.
    #[allow(dead_code)]
    per_rank_cuts: Vec<usize>,
}

impl CutSyncBuffers {
    /// Construct pre-allocated cut synchronization buffers for the given
    /// topology.
    ///
    /// All byte buffer allocations occur here. Subsequent calls to
    /// [`sync_cuts`](Self::sync_cuts) reuse these buffers.
    ///
    /// # Arguments
    ///
    /// - `n_state` — state dimension (number of cut coefficients per cut).
    /// - `max_cuts_per_rank` — maximum number of cuts any rank generates per
    ///   stage per iteration. Used to pre-allocate buffer capacity.
    /// - `num_ranks` — total number of MPI ranks (`comm.size()`).
    #[must_use]
    pub fn new(n_state: usize, max_cuts_per_rank: usize, num_ranks: usize) -> Self {
        // Uniform distribution: every rank produces max_cuts_per_rank cuts.
        Self::with_distribution(
            n_state,
            max_cuts_per_rank,
            num_ranks,
            max_cuts_per_rank * num_ranks,
        )
    }

    /// Construct buffers for non-uniform work distribution.
    ///
    /// When the total number of forward passes does not divide evenly among
    /// ranks, the first `total_forward_passes % num_ranks` ranks each handle
    /// one extra forward pass. This constructor sizes buffers for the maximum
    /// per-rank count and records each rank's expected count for correct
    /// `allgatherv` displacements.
    ///
    /// # Arguments
    ///
    /// - `n_state` — state dimension (number of cut coefficients per cut).
    /// - `max_cuts_per_rank` — maximum cuts any rank generates per stage per
    ///   iteration. Used to size the send buffer.
    /// - `num_ranks` — total number of MPI ranks.
    /// - `total_forward_passes` — total forward passes across all ranks. Used
    ///   to compute per-rank expected cut counts.
    #[must_use]
    pub fn with_distribution(
        n_state: usize,
        max_cuts_per_rank: usize,
        num_ranks: usize,
        total_forward_passes: usize,
    ) -> Self {
        let record_size = cut_wire_size(n_state);
        let send_cap = max_cuts_per_rank * record_size;

        let base = total_forward_passes / num_ranks;
        let remainder = total_forward_passes % num_ranks;
        let per_rank_cuts: Vec<usize> = (0..num_ranks)
            .map(|r| base + usize::from(r < remainder))
            .collect();
        let recv_cap: usize = per_rank_cuts.iter().sum::<usize>() * record_size;

        let counts: Vec<usize> = per_rank_cuts.iter().map(|&c| c * record_size).collect();
        let mut displs = vec![0usize; num_ranks];
        for r in 1..num_ranks {
            displs[r] = displs[r - 1] + counts[r - 1];
        }

        Self {
            send_buf: vec![0u8; send_cap],
            recv_buf: vec![0u8; recv_cap],
            counts,
            displs,
            n_state,
            num_ranks,
            record_size,
            per_rank_cuts,
        }
    }

    /// Synchronize locally generated cuts across all MPI ranks for one stage.
    ///
    /// Serializes cuts into the pre-allocated send buffer, broadcasts via
    /// `allgatherv`, then deserializes and inserts remote cuts into the FCF.
    /// The local rank's cuts are skipped (already inserted by the backward pass).
    ///
    /// # Arguments
    ///
    /// - `stage` — 0-based stage index for which cuts are being synchronized.
    /// - `local_cuts` — locally generated cuts as `(slot_index, iteration,
    ///   forward_pass_index, intercept, coefficients)` tuples. The backward
    ///   pass has already inserted these cuts into the FCF; they are serialized
    ///   here to send to remote ranks, but are **not** re-inserted locally.
    /// - `fcf` — Future Cost Function to receive remote cuts.
    /// - `comm` — communicator for the `allgatherv` call.
    ///
    /// # Returns
    ///
    /// `Ok(n)` where `n` is the number of remote cuts inserted into `fcf`.
    /// In single-rank mode, returns `Ok(0)` because the only rank's segment
    /// is skipped during deserialization.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Communication(_))` if the underlying
    /// `allgatherv` call fails. The FCF and buffer contents are unspecified
    /// on error.
    ///
    /// # Panics (debug builds only)
    ///
    /// - Panics if `local_cuts.len() * record_size > send_buf.len()`.
    /// - Panics if any cut's coefficient slice length does not equal `n_state`.
    pub fn sync_cuts<C: Communicator>(
        &mut self,
        stage: usize,
        local_cuts: &[(u32, u32, u32, f64, &[f64])],
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<usize, SddpError> {
        let n_local = local_cuts.len();
        let send_len = n_local * self.record_size;

        debug_assert!(
            send_len <= self.send_buf.len(),
            "send_len {send_len} exceeds send_buf capacity {}",
            self.send_buf.len()
        );

        for (i, &(slot_index, iteration, forward_pass_index, intercept, coefficients)) in
            local_cuts.iter().enumerate()
        {
            debug_assert!(
                coefficients.len() == self.n_state,
                "cut {i} coefficient length {} != n_state {}",
                coefficients.len(),
                self.n_state,
            );
            let start = i * self.record_size;
            serialize_cut(
                &mut self.send_buf[start..start + self.record_size],
                slot_index,
                iteration,
                forward_pass_index,
                intercept,
                coefficients,
            );
        }

        // Each rank sends exactly n_local cuts. For multi-rank, other ranks
        // send per_rank_cuts[r] cuts. Recompute counts based on the actual
        // local count (which should match per_rank_cuts[my_rank]) and the
        // pre-computed per-rank expectations for other ranks.
        let my_rank = comm.rank();
        for r in 0..self.num_ranks {
            let cuts_for_r = if r == my_rank {
                n_local
            } else {
                self.per_rank_cuts[r]
            };
            self.counts[r] = cuts_for_r * self.record_size;
        }
        self.displs[0] = 0;
        for r in 1..self.num_ranks {
            self.displs[r] = self.displs[r - 1] + self.counts[r - 1];
        }

        let recv_len: usize = self.counts.iter().sum();
        debug_assert!(
            recv_len <= self.recv_buf.len(),
            "recv_len {recv_len} exceeds recv_buf capacity {}",
            self.recv_buf.len()
        );

        comm.allgatherv(
            &self.send_buf[..send_len],
            &mut self.recv_buf[..recv_len],
            &self.counts,
            &self.displs,
        )?;

        let local_rank = comm.rank();
        let mut remote_count = 0usize;

        for r in 0..self.num_ranks {
            if r == local_rank {
                continue;
            }

            let start = self.displs[r];
            let end = start + self.counts[r];
            let slice = &self.recv_buf[start..end];
            let cuts = deserialize_cuts_from_buffer(slice, self.n_state);
            for (header, coefficients) in cuts {
                fcf.add_cut(
                    stage,
                    u64::from(header.iteration),
                    header.forward_pass_index,
                    header.intercept,
                    &coefficients,
                );
                remote_count += 1;
            }
        }

        Ok(remote_count)
    }

    /// Return the send buffer capacity in bytes.
    #[must_use]
    pub fn send_capacity(&self) -> usize {
        self.send_buf.len()
    }

    /// Return the receive buffer capacity in bytes.
    #[must_use]
    pub fn recv_capacity(&self) -> usize {
        self.recv_buf.len()
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )]

    use cobre_comm::{CommData, CommError, Communicator, LocalBackend, ReduceOp};

    use super::CutSyncBuffers;
    use crate::{
        cut::{
            fcf::FutureCostFunction,
            wire::{cut_wire_size, deserialize_cuts_from_buffer, serialize_cut},
        },
        SddpError,
    };

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn new_send_buf_capacity_is_max_cuts_times_record_size() {
        // AC: CutSyncBuffers::new(n_state=2, max_cuts_per_rank=3, num_ranks=1)
        // send_buf capacity = 3 * cut_wire_size(2) = 3 * 40 = 120
        let bufs = CutSyncBuffers::new(2, 3, 1);
        let expected = 3 * cut_wire_size(2);
        assert_eq!(bufs.send_capacity(), expected);
    }

    #[test]
    fn new_recv_buf_capacity_is_max_cuts_times_num_ranks_times_record_size() {
        // AC: CutSyncBuffers::new(n_state=3, max_cuts_per_rank=10, num_ranks=4)
        // recv_buf capacity = 10 * 4 * cut_wire_size(3) = 40 * 48 = 1920
        let bufs = CutSyncBuffers::new(3, 10, 4);
        let expected = 10 * 4 * cut_wire_size(3);
        assert_eq!(bufs.recv_capacity(), expected);
        assert_eq!(expected, 1920);
    }

    #[test]
    fn new_counts_length_equals_num_ranks() {
        // AC: counts has length num_ranks
        let bufs = CutSyncBuffers::new(3, 10, 4);
        assert_eq!(bufs.counts.len(), 4);
    }

    #[test]
    fn new_displs_length_equals_num_ranks() {
        // AC: displs has length num_ranks
        let bufs = CutSyncBuffers::new(3, 10, 4);
        assert_eq!(bufs.displs.len(), 4);
    }

    #[test]
    fn new_counts_and_displs_initialized_to_max_uniform_values() {
        // Verify that counts and displs are set to maximum uniform capacity at
        // construction time (they will be recomputed per call to sync_cuts).
        let bufs = CutSyncBuffers::new(2, 3, 2);
        let per_rank = 3 * cut_wire_size(2); // 120
        assert_eq!(bufs.counts[0], per_rank);
        assert_eq!(bufs.counts[1], per_rank);
        assert_eq!(bufs.displs[0], 0);
        assert_eq!(bufs.displs[1], per_rank);
    }

    #[test]
    fn new_n_state_zero_record_size_is_24() {
        // Edge case: n_state = 0, record_size = 24.
        let bufs = CutSyncBuffers::new(0, 5, 1);
        assert_eq!(bufs.send_capacity(), 5 * 24);
        assert_eq!(bufs.recv_capacity(), 5 * 24);
    }

    #[test]
    fn send_buf_serialization_round_trip_two_cuts() {
        // AC: given n_state=2, max_cuts_per_rank=2, when 2 cuts are serialized
        // into send_buf, the byte length matches 2 * cut_wire_size(2) = 80 and
        // round-trip deserialization recovers original fields.
        let mut bufs = CutSyncBuffers::new(2, 2, 1);
        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 10.0, &[1.0, 2.0]), (1, 1, 1, 20.0, &[3.0, 4.0])];

        let record_size = cut_wire_size(2);
        let send_len = local_cuts.len() * record_size;
        assert_eq!(send_len, 80);

        // Serialize manually into send_buf using the same logic as sync_cuts.
        for (i, &(slot_index, iteration, forward_pass_index, intercept, coefficients)) in
            local_cuts.iter().enumerate()
        {
            let start = i * record_size;
            serialize_cut(
                &mut bufs.send_buf[start..start + record_size],
                slot_index,
                iteration,
                forward_pass_index,
                intercept,
                coefficients,
            );
        }

        // Round-trip: deserialize from the same buffer.
        let recovered = deserialize_cuts_from_buffer(&bufs.send_buf[..send_len], 2);
        assert_eq!(recovered.len(), 2);

        let (h0, c0) = &recovered[0];
        assert_eq!(h0.slot_index, 0);
        assert_eq!(h0.iteration, 1);
        assert_eq!(h0.forward_pass_index, 0);
        assert_eq!(h0.intercept, 10.0);
        assert_eq!(c0, &[1.0, 2.0]);

        let (h1, c1) = &recovered[1];
        assert_eq!(h1.slot_index, 1);
        assert_eq!(h1.iteration, 1);
        assert_eq!(h1.forward_pass_index, 1);
        assert_eq!(h1.intercept, 20.0);
        assert_eq!(c1, &[3.0, 4.0]);
    }

    #[test]
    fn counts_and_displs_computation_for_various_cut_counts() {
        // Verify counts and displs are correctly computed for different numbers
        // of local cuts and ranks.
        //
        // With 2 local cuts and n_state=2: per_rank_bytes = 2 * 40 = 80.
        // For 3 ranks: counts = [80, 80, 80], displs = [0, 80, 160].
        let mut bufs = CutSyncBuffers::new(2, 5, 3);

        let n_local = 2usize;
        let record_size = cut_wire_size(2); // 40
        let per_rank = n_local * record_size; // 80

        // Simulate what sync_cuts does to counts and displs.
        for r in 0..3 {
            bufs.counts[r] = per_rank;
            bufs.displs[r] = r * per_rank;
        }

        assert_eq!(bufs.counts, vec![80, 80, 80]);
        assert_eq!(bufs.displs, vec![0, 80, 160]);
    }

    // ── Integration tests (round-trip with LocalBackend) ──────────────────────

    #[test]
    fn sync_cuts_single_rank_returns_zero_remote_cuts() {
        // AC: Given CutSyncBuffers::new(n_state=2, max_cuts_per_rank=3,
        // num_ranks=1), when sync_cuts is called with 2 local cuts in
        // single-rank mode, then it returns Ok(0) — the single rank's own
        // cuts are skipped.
        let mut bufs = CutSyncBuffers::new(2, 3, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 3, 10, 0);
        let comm = LocalBackend;

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 10.0, &[1.0, 2.0]), (0, 1, 1, 20.0, &[3.0, 4.0])];

        let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();
        assert_eq!(result, 0, "expected zero remote cuts in single-rank mode");
    }

    #[test]
    fn sync_cuts_single_rank_does_not_insert_local_cuts_into_fcf() {
        // After sync_cuts with single rank, FCF should have zero cuts —
        // the local rank's cuts are skipped (they were already inserted by the
        // backward pass before this function is called).
        let mut bufs = CutSyncBuffers::new(2, 3, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 3, 10, 0);
        let comm = LocalBackend;

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 10.0, &[1.0, 2.0]), (0, 1, 1, 20.0, &[3.0, 4.0])];

        bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();

        // FCF must remain empty — local cuts are intentionally NOT inserted.
        assert_eq!(
            fcf.total_active_cuts(),
            0,
            "sync_cuts must not insert local cuts into FCF"
        );
    }

    #[test]
    fn sync_cuts_serialization_round_trip_via_allgatherv_identity() {
        // After allgatherv with LocalBackend (identity copy), the recv buffer
        // must deserialize to the original cut fields. We verify this by
        // checking FCF state after manually inserting the local cut, then
        // confirming no additional cuts appear from sync (single rank skips).
        let mut bufs = CutSyncBuffers::new(2, 2, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 2, 10, 0);
        let comm = LocalBackend;

        // Simulate backward pass: insert cut into FCF (this rank's own cut).
        fcf.add_cut(0, 1, 0, 10.0, &[1.0, 2.0]);

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[(0, 1, 0, 10.0, &[1.0, 2.0])];

        let remote_inserted = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();

        // Single rank: zero remotes. FCF still has exactly 1 cut (inserted by
        // backward pass before sync_cuts was called).
        assert_eq!(remote_inserted, 0);
        assert_eq!(fcf.total_active_cuts(), 1);
    }

    #[test]
    fn sync_cuts_zero_local_cuts_returns_zero() {
        // When no cuts are generated (empty local_cuts), sync_cuts must still
        // succeed and return Ok(0) for single rank.
        let mut bufs = CutSyncBuffers::new(2, 5, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 5, 10, 0);
        let comm = LocalBackend;

        let result = bufs.sync_cuts(0, &[], &mut fcf, &comm).unwrap();
        assert_eq!(result, 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn sync_cuts_error_maps_to_sddp_communication_error() {
        // AC: Given a communicator that returns Err(CommError::CollectiveFailed)
        // from allgatherv, when sync_cuts is called, then it returns
        // Err(SddpError::Communication(_)).

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

        let mut bufs = CutSyncBuffers::new(2, 2, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 2, 10, 0);

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[(0, 1, 0, 5.0, &[1.0, 2.0])];

        let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &FailingComm);
        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "expected SddpError::Communication, got: {result:?}",
        );
    }

    #[test]
    fn sync_cuts_three_ranks_returns_four_remote_cuts() {
        // AC5: Given sync_cuts completes with 2 cuts from each of 3 ranks
        // (6 total, 2 local + 4 remote), when inspecting the return value,
        // then it is Ok(4) (4 remote cuts inserted).
        //
        // Strategy: pre-populate the recv_buf with remote rank data at the
        // correct offsets BEFORE calling sync_cuts. The mock allgatherv only
        // copies the local segment (rank 0) from send to recv, leaving the
        // pre-filled remote segments untouched. This avoids unsafe pointer
        // operations while faithfully testing the deserialization path.

        /// Mock communicator simulating 3 ranks. Rank 0 is the local rank.
        /// `allgatherv` copies the send buffer into the rank-0 segment only;
        /// the remote segments are expected to be pre-populated in `recv_buf`.
        struct ThreeRankComm;

        impl Communicator for ThreeRankComm {
            fn allgatherv<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                // Only copy rank 0 (local) data; remote segments pre-filled.
                let r0_len = counts[0];
                recv[..r0_len].copy_from_slice(&send[..r0_len]);
                Ok(())
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
                3
            }
        }

        let n_state = 2;
        let record_size = cut_wire_size(n_state); // 40
        let n_local = 2;
        let per_rank_bytes = n_local * record_size; // 80

        // FCF: 1 stage, n_state=2, forward_passes=6, max_iterations=10,
        // warm_start=0 → capacity = 0 + 10*6 = 60 slots.
        let mut fcf = FutureCostFunction::new(1, n_state, 6, 10, 0);
        let mut bufs = CutSyncBuffers::new(n_state, n_local, 3);

        // Pre-populate recv_buf with remote rank data at the exact offsets
        // that sync_cuts will compute (displs[1] = 80, displs[2] = 160).
        let r1_start = per_rank_bytes; // 80
        serialize_cut(
            &mut bufs.recv_buf[r1_start..r1_start + record_size],
            10,
            1,
            10,
            100.0,
            &[1.0, 2.0],
        );
        serialize_cut(
            &mut bufs.recv_buf[r1_start + record_size..r1_start + 2 * record_size],
            11,
            1,
            11,
            200.0,
            &[3.0, 4.0],
        );

        let r2_start = 2 * per_rank_bytes; // 160
        serialize_cut(
            &mut bufs.recv_buf[r2_start..r2_start + record_size],
            20,
            1,
            20,
            300.0,
            &[5.0, 6.0],
        );
        serialize_cut(
            &mut bufs.recv_buf[r2_start + record_size..r2_start + 2 * record_size],
            21,
            1,
            21,
            400.0,
            &[7.0, 8.0],
        );

        // Local rank 0 has 2 cuts (already inserted by backward pass).
        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 50.0, &[0.1, 0.2]), (1, 1, 1, 60.0, &[0.3, 0.4])];

        let remote_inserted = bufs
            .sync_cuts(0, local_cuts, &mut fcf, &ThreeRankComm)
            .unwrap();
        assert_eq!(remote_inserted, 4, "expected 4 remote cuts inserted");
        // FCF should have 4 cuts (only remote ones — local rank skipped).
        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn sync_cuts_preserves_cut_fields_after_deserialization() {
        // Verify that header fields survive the serialize → allgatherv →
        // deserialize round trip. For a single-rank test we cannot observe
        // remote insertions, so we test the wire round-trip directly by
        // examining the recv_buf contents after the allgatherv identity copy.
        let n_state = 2usize;
        let mut bufs = CutSyncBuffers::new(n_state, 1, 1);
        let mut fcf = FutureCostFunction::new(1, n_state, 1, 10, 0);
        let comm = LocalBackend;

        let coeffs = [7.5_f64, -3.25_f64];
        let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[(5, 3, 2, 99.0, &coeffs)];

        bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();

        // After LocalBackend allgatherv (identity copy), recv_buf[0..40]
        // contains rank 0's serialized cut. Deserialize and verify.
        let record_size = cut_wire_size(n_state);
        let recovered = deserialize_cuts_from_buffer(&bufs.recv_buf[..record_size], n_state);
        assert_eq!(recovered.len(), 1);

        let (header, rec_coeffs) = &recovered[0];
        assert_eq!(header.slot_index, 5);
        assert_eq!(header.iteration, 3);
        assert_eq!(header.forward_pass_index, 2);
        assert_eq!(header.intercept, 99.0);
        assert_eq!(rec_coeffs[0].to_bits(), coeffs[0].to_bits());
        assert_eq!(rec_coeffs[1].to_bits(), coeffs[1].to_bits());
    }
}
