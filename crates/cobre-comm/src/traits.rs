//! Trait definitions for the cobre-comm abstraction layer, decoupling distributed
//! computations from specific communication technologies: [`CommData`],
//! [`Communicator`], and the `shared-memory`-gated [`LocalCommunicator`],
//! [`SharedRegion`], and [`SharedMemoryProvider`].
//!
//! ## Shared-memory subsystem (`shared-memory` feature)
//!
//! Gated behind the feature because it has zero production consumers and its only
//! implementation (`HeapRegion`) is not true shared memory — each rank holds its
//! own `Vec<T>`.
//!
//! The real implementation should build on ferrompi's `SharedWindow<T>`
//! (`MPI_Win_allocate_shared` / `MPI_Win_shared_query`). Open tension:
//! `SharedWindow<T>` is `!Send + !Sync` (ferrompi v0.4.x) while these traits
//! require `Send + Sync` — needs a single-threaded-access wrapper or a ferrompi
//! API change. Deferred until a downstream consumer exists.

use crate::CommError;
use crate::ReduceOp;
#[cfg(all(feature = "mpi", feature = "shared-memory"))]
use crate::ferrompi::FerrompiLocalComm;
#[cfg(feature = "shared-memory")]
use crate::local::LocalBackend;
use crate::topology::ExecutionTopology;
/// Marker trait for types that can be transmitted through collective operations.
///
/// Requires `Send + Sync`, `Copy`, `Default` (zero-fill regions in
/// `SharedMemoryProvider::create_shared_region` without unsafe code), and
/// `'static`.
///
/// With the `mpi` feature it also requires [`ferrompi::MpiDatatype`], the sealed
/// FFI marker for the primitives MPI transmits directly (`f32`, `f64`, `i32`,
/// `i64`, `u8`, `u32`, `u64`) — exactly the set the data plane ever sends, so
/// the `Communicator` impl can delegate to ferrompi's generic FFI without a
/// per-method bound.
///
/// All types satisfying the bounds are `CommData` via a blanket impl:
///
/// ```rust
/// # use cobre_comm::CommData;
/// fn requires_comm_data<T: CommData>() {}
/// requires_comm_data::<f64>();
/// requires_comm_data::<u8>();
/// requires_comm_data::<i32>();
/// requires_comm_data::<u64>();
/// ```
#[cfg(feature = "mpi")]
pub trait CommData: Send + Sync + Copy + Default + 'static + ferrompi::MpiDatatype {}

#[cfg(feature = "mpi")]
impl<T: Send + Sync + Copy + Default + 'static + ferrompi::MpiDatatype> CommData for T {}

/// Marker trait for types that can be transmitted through collective operations.
///
/// Without the `mpi` feature, `CommData` accepts all `Copy + Send + Sync +
/// Default + 'static` types, including `bool` and tuple types used in tests.
#[cfg(not(feature = "mpi"))]
pub trait CommData: Send + Sync + Copy + Default + 'static {}

#[cfg(not(feature = "mpi"))]
impl<T: Send + Sync + Copy + Default + 'static> CommData for T {}

/// Backend abstraction for collective communication operations: four collectives
/// (`allgatherv`, `allreduce`, `broadcast`, `barrier`) and two infallible
/// accessors (`rank`, `size`).
///
/// # Design: compile-time static dispatch
///
/// Three methods carry a `T: CommData` parameter, so the trait is
/// **intentionally not object-safe** — `Box<dyn Communicator>` does not compile.
/// Callers use a generic bound `C: Communicator`; with one backend per binary,
/// monomorphization cost is negligible and the hot-path no-op `LocalBackend`
/// inlines to zero instructions.
///
/// # Thread safety
///
/// Requires `Send + Sync` for hybrid MPI+OpenMP execution; all methods take
/// `&self`. Callers must serialize concurrent collective calls — never invoke the
/// same collective from two threads on one communicator instance simultaneously.
///
/// # Example
///
/// ```rust
/// use cobre_comm::{Communicator, CommError};
///
/// fn print_topology<C: Communicator>(comm: &C) {
///     println!("rank {} of {}", comm.rank(), comm.size());
/// }
/// ```
pub trait Communicator: Send + Sync {
    /// Gather variable-length data from all ranks into all ranks.
    ///
    /// Each rank contributes its `send` buffer; afterwards `recv` is populated in
    /// rank order, rank `r`'s data at `recv[displs[r]..displs[r]+counts[r]]`.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | `counts.len() == self.size()` | One element-count entry per rank |
    /// | `displs.len() == self.size()` | One displacement entry per rank |
    /// | `send.len() == counts[self.rank()]` | Send buffer length matches this rank's declared count |
    /// | `recv.len() >= displs[R-1] + counts[R-1]` | Receive buffer large enough for all ranks' data |
    /// | Non-overlapping regions | `recv[displs[r]..displs[r]+counts[r]]` regions must not overlap for distinct `r` |
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Rank-ordered receive | `recv[displs[r]..displs[r]+counts[r]]` contains rank `r`'s sent data, for all `r` in `0..size()` |
    /// | Identical across ranks | All ranks have identical `recv` contents after the call returns |
    /// | Implicit barrier | No rank returns until all ranks have contributed their data |
    ///
    /// # Errors
    ///
    /// Returns [`CommError::InvalidBufferSize`] if any buffer-size precondition
    /// is violated. Returns [`CommError::CollectiveFailed`] if the underlying
    /// MPI operation fails. On error, the contents of `recv` are unspecified.
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError>;

    /// Reduce data from all ranks element-wise (`recv[i] = op(send_0[i], ...,
    /// send_{R-1}[i])`), with the result available identically on all ranks.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | `send.len() == recv.len()` | Send and receive buffers have equal length |
    /// | `send.len() > 0` | At least one element to reduce |
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Element-wise reduction | `recv[i] = op(send_0[i], ..., send_{R-1}[i])` for all `i` |
    /// | Identical across ranks | All ranks have identical `recv` contents after the call returns |
    ///
    /// # Floating-point note
    ///
    /// [`ReduceOp::Sum`] may produce results that vary across runs with different
    /// rank counts or MPI implementations, because floating-point addition is
    /// non-associative and the reduction tree shape is implementation-defined.
    /// [`ReduceOp::Min`] is exact (comparison-based, no arithmetic).
    ///
    /// # Errors
    ///
    /// Returns [`CommError::InvalidBufferSize`] if `send.len() != recv.len()`.
    /// Returns [`CommError::CollectiveFailed`] if the underlying MPI operation
    /// fails. On error, the contents of `recv` are unspecified.
    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError>;

    /// Broadcast `buf` from `root` to all other ranks, which overwrite their
    /// `buf` with it; afterwards all ranks hold identical contents, uniquely
    /// determined by the root's input regardless of backend.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | `root < self.size()` | Root rank index is valid |
    /// | `buf.len()` identical on all ranks | All ranks provide the same buffer length |
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Data from root | `buf` on every rank contains the data that was in `buf` on the root rank before the call |
    /// | Identical across ranks | All ranks have identical `buf` contents after the call returns |
    ///
    /// # Errors
    ///
    /// Returns [`CommError::InvalidRoot`] if `root >= self.size()`. Returns
    /// [`CommError::CollectiveFailed`] if the underlying MPI operation fails.
    /// On error, `buf` on non-root ranks may be partially overwritten; the root
    /// rank's `buf` is unchanged.
    fn broadcast<T: CommData>(&self, buf: &mut [T], root: usize) -> Result<(), CommError>;

    /// Block until all ranks have called barrier. Exchanges no user data.
    ///
    /// **Every rank in the communicator must call `barrier`** or it deadlocks.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | All ranks call | Every rank must enter the barrier; missing ranks cause deadlock |
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Global synchronization | No rank returns from `barrier` until all ranks have entered |
    /// | No data exchange | `barrier` does not transmit or modify any user data |
    ///
    /// # Errors
    ///
    /// Returns [`CommError::CollectiveFailed`] if the underlying MPI barrier
    /// fails. In practice the only failure mode is a process crash, detected
    /// as a communication timeout by the MPI runtime.
    fn barrier(&self) -> Result<(), CommError>;

    /// Return the rank index of the calling process, in `0..size()`.
    ///
    /// Implementations must cache this at construction — it is called frequently
    /// on the hot path. The value never changes after initialization and is
    /// unique per rank, so concurrent reads are safe.
    fn rank(&self) -> usize;

    /// Return the total number of ranks (at least 1; identical on every rank).
    ///
    /// Implementations must cache this at construction — it is called frequently
    /// on the hot path. The value never changes after initialization, so
    /// concurrent reads are safe.
    fn size(&self) -> usize;

    /// Terminate every process in the communicator with `error_code` (`MPI_Abort`,
    /// or [`std::process::exit`] on the local backend). **Never returns.**
    ///
    /// Use only for a non-recoverable error that cannot be coordinated through
    /// normal collectives (e.g. an asymmetric failure across ranks).
    fn abort(&self, error_code: i32) -> !;
}

/// Object-safe sub-trait of [`Communicator`] for intra-node initialization
/// coordination.
///
/// `Communicator` is not object-safe (its generic methods need static dispatch),
/// but `split_local` returns an init-only intra-node communicator where dynamic
/// dispatch is fine. `LocalCommunicator` exposes only the three non-generic
/// methods (`rank`, `size`, `barrier`) that lets `split_local` return a trait
/// object without compromising the hot-path trait (spec communicator-trait.md §4.5).
///
/// # Example
///
/// ```rust
/// use cobre_comm::LocalCommunicator;
///
/// fn determine_leader(local_comm: &dyn LocalCommunicator) -> bool {
///     local_comm.rank() == 0
/// }
/// ```
#[cfg(feature = "shared-memory")]
pub trait LocalCommunicator: Send + Sync {
    /// Local rank within the intra-node communicator, in `0..size()`; the shared
    /// memory leader is always local rank 0.
    fn rank(&self) -> usize;

    /// Number of ranks sharing this node (`1` for `HeapFallback`).
    fn size(&self) -> usize;

    /// Block until all intra-node ranks have called barrier, or it deadlocks.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::CollectiveFailed`] if the underlying
    /// barrier operation fails.
    fn barrier(&self) -> Result<(), CommError>;
}

/// Enum dispatch over the closed set of intra-node communicator backends,
/// returned by [`SharedMemoryProvider::split_local`]. A new variant requires
/// updating all `match` arms in the `LocalCommunicator` impl below.
#[cfg(feature = "shared-memory")]
pub enum LocalCommKind {
    /// Single-process local backend (always available).
    Local(LocalBackend),
    /// Ferrompi intra-node communicator wrapping `MPI_Comm_split_type SHARED`.
    #[cfg(feature = "mpi")]
    Ferrompi(FerrompiLocalComm),
}

#[cfg(feature = "shared-memory")]
impl LocalCommunicator for LocalCommKind {
    fn rank(&self) -> usize {
        match self {
            Self::Local(b) => LocalCommunicator::rank(b),
            #[cfg(feature = "mpi")]
            Self::Ferrompi(c) => LocalCommunicator::rank(c),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Local(b) => LocalCommunicator::size(b),
            #[cfg(feature = "mpi")]
            Self::Ferrompi(c) => LocalCommunicator::size(c),
        }
    }

    fn barrier(&self) -> Result<(), CommError> {
        match self {
            Self::Local(b) => LocalCommunicator::barrier(b),
            #[cfg(feature = "mpi")]
            Self::Ferrompi(c) => LocalCommunicator::barrier(c),
        }
    }
}

/// Handle to a shared memory region holding `count` elements of type `T`.
///
/// Three-phase lifecycle: the leader (local rank 0) allocates and writes via
/// [`as_mut_slice`](SharedRegion::as_mut_slice), all ranks call
/// [`fence`](SharedRegion::fence) to publish, then all read via
/// [`as_slice`](SharedRegion::as_slice). **Followers must not read until
/// `fence()` returns `Ok`.** On `HeapFallback` every rank is its own leader with
/// a private `Vec<T>`. Drop releases the backing resource (ferrompi:
/// `MPI_Win_free`), so all handles must drop before MPI finalization.
#[cfg(feature = "shared-memory")]
pub trait SharedRegion<T: CommData>: Send + Sync {
    /// Return the region contents as a contiguous slice of length `count`,
    /// zero-copy on true shared memory backends.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Fence completed | `fence()` must have returned `Ok` before any follower reads |
    /// | No concurrent writes | The region must be in its read-only phase |
    fn as_slice(&self) -> &[T];

    /// Return the region contents as a mutable slice for the leader to populate
    /// before `fence()`. Length `count` on the leader, 0 on followers (true
    /// shared memory backends); writes are invisible to other ranks until
    /// `fence()`.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Leader only | The caller must be the leader (`SharedMemoryProvider::is_leader()` returns `true`), or the backend must be `HeapFallback` |
    /// | Population phase | No concurrent reads may be in progress on any rank |
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Publish leader writes to all co-located ranks (`MPI_Win_fence`; no-op for
    /// `HeapFallback`). Collective — all ranks sharing the region must call it,
    /// and followers may read via `as_slice()` only after it returns `Ok`.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::CollectiveFailed`] if a rank has crashed or
    /// the underlying fence operation fails. On error, the visibility of prior
    /// writes is unspecified.
    fn fence(&self) -> Result<(), CommError>;
}

/// Backend abstraction for intra-node shared memory region management.
///
/// A **companion to [`Communicator`]**, not a supertrait — a backend implements
/// each independently, so callers needing only collectives bound `C: Communicator`
/// and those needing shared memory bound `C: Communicator + SharedMemoryProvider`:
///
/// ```rust
/// # use cobre_comm::{Communicator, SharedMemoryProvider};
/// fn train<C: Communicator + SharedMemoryProvider>(comm: &C) {
/// }
/// ```
///
/// `Send + Sync`; `create_shared_region` and `split_local` are initialization-only,
/// called before any parallel region.
#[cfg(feature = "shared-memory")]
pub trait SharedMemoryProvider: Send + Sync {
    /// The shared memory region handle type, one concrete type per backend
    /// (ferrompi wraps `SharedWindow<T>`, `HeapFallback` wraps `Vec<T>`).
    type Region<T: CommData>: SharedRegion<T>;

    /// Create a shared memory region for `count` elements of `T`, at startup only.
    ///
    /// On true shared memory, the leader allocates `count` elements and followers
    /// allocate size 0; on `HeapFallback` every rank allocates its own `Vec<T>`.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::AllocationFailed`] if the OS rejects the
    /// shared memory allocation (size too large, insufficient permissions, or
    /// system shared memory limits exceeded). For `HeapFallback`, returns
    /// `AllocationFailed` only on heap exhaustion (which on most platforms causes
    /// process abort before returning `Err`).
    fn create_shared_region<T: CommData>(&self, count: usize)
    -> Result<Self::Region<T>, CommError>;

    /// Create an intra-node communicator of the ranks sharing the calling rank's
    /// node, at startup only (ferrompi `comm.split_shared()`; spec
    /// communicator-trait.md §4.3). Local rank 0 is the shared memory leader;
    /// `HeapFallback` returns `size() == 1`, `rank() == 0`.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::CollectiveFailed`] if the underlying
    /// communicator split operation fails (e.g., MPI communicator split failure).
    fn split_local(&self) -> Result<LocalCommKind, CommError>;

    /// Return whether the calling rank is its node's shared memory leader (local
    /// rank 0 from `split_local()`); only the leader may write via
    /// `as_mut_slice()`. Always `true` on `HeapFallback` (every rank owns its copy).
    fn is_leader(&self) -> bool;
}

/// Trait for backends that can report their execution topology, separate from
/// [`Communicator`] because not all backends expose one.
///
/// The [`ExecutionTopology`] is gathered at
/// initialization and queried non-collectively, allocation-free.
pub trait TopologyProvider: Send + Sync {
    /// Return the cached execution topology (non-collective, any thread).
    #[must_use]
    fn topology(&self) -> &ExecutionTopology;
}

#[cfg(test)]
mod tests {
    use super::{CommData, Communicator};
    use crate::{CommError, ReduceOp};

    #[cfg(feature = "shared-memory")]
    use super::{LocalCommunicator, SharedMemoryProvider, SharedRegion};
    #[cfg(feature = "shared-memory")]
    use crate::LocalBackend;

    /// Compile-time assertion that a type satisfies `CommData`.
    ///
    /// If this function compiles for a given `T`, then `T` implements `CommData`.
    fn assert_comm_data<T: CommData>() {}

    #[test]
    fn test_commdata_blanket_impl() {
        // f64, u8, i32, u64 implement MpiDatatype and are CommData under both
        // feature configurations.
        assert_comm_data::<f64>();
        assert_comm_data::<u8>();
        assert_comm_data::<i32>();
        assert_comm_data::<u64>();
        // bool and (f64, f64) do not implement MpiDatatype, so they are only
        // CommData when the `mpi` feature is NOT enabled.
        #[cfg(not(feature = "mpi"))]
        assert_comm_data::<bool>();
        #[cfg(not(feature = "mpi"))]
        assert_comm_data::<(f64, f64)>();
    }

    fn use_communicator_generic<C: Communicator>(comm: &C) {
        let _ = comm.rank();
        let _ = comm.size();
    }

    fn assert_communicator_requires_send_sync<C: Communicator>(comm: &C) {
        fn needs_send_sync<T: Send + Sync>(_v: &T) {}
        needs_send_sync(comm);
    }

    #[test]
    fn test_communicator_generic_and_send_sync_compile() {
        let comm = NeverImpl;
        use_communicator_generic(&comm);
        assert_communicator_requires_send_sync(&comm);
    }

    struct NeverImpl;

    impl Communicator for NeverImpl {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("NeverImpl is only used for compile-time trait shape tests")
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("NeverImpl is only used for compile-time trait shape tests")
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("NeverImpl is only used for compile-time trait shape tests")
        }

        fn barrier(&self) -> Result<(), CommError> {
            unreachable!("NeverImpl is only used for compile-time trait shape tests")
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            1
        }

        fn abort(&self, _error_code: i32) -> ! {
            unreachable!("NeverImpl is only used for compile-time trait shape tests")
        }
    }

    // ── shared-memory feature tests ───────────────────────────────────────────

    #[cfg(feature = "shared-memory")]
    fn use_local_communicator_generic<L: super::LocalCommunicator>(lc: &L) {
        let _ = lc.rank();
        let _ = lc.size();
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_communicator_object_safe_and_generic_compile() {
        struct LocalImpl;

        impl super::LocalCommunicator for LocalImpl {
            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                1
            }

            fn barrier(&self) -> Result<(), CommError> {
                Ok(())
            }
        }

        let lc = LocalImpl;
        use_local_communicator_generic(&lc);
        assert_eq!(lc.rank(), 0);
        assert_eq!(lc.size(), 1);
        assert!(lc.barrier().is_ok());
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_communicator_requires_send_sync() {
        fn needs_send_sync<T: Send + Sync>(_v: &T) {}

        struct SendSyncLocalImpl;

        impl super::LocalCommunicator for SendSyncLocalImpl {
            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                1
            }

            fn barrier(&self) -> Result<(), CommError> {
                Ok(())
            }
        }

        let lc = SendSyncLocalImpl;
        needs_send_sync(&lc);
    }

    #[cfg(feature = "shared-memory")]
    fn use_shared_region_as_bound<R: super::SharedRegion<f64>>(region: &mut R) {
        let _slice: &[f64] = region.as_slice();
        let _mut_slice: &mut [f64] = region.as_mut_slice();
        let _fence: Result<(), CommError> = region.fence();
    }

    #[cfg(feature = "shared-memory")]
    fn assert_shared_region_requires_send_sync<R: super::SharedRegion<f64>>(region: &R) {
        fn needs_send_sync<T: Send + Sync>(_v: &T) {}
        needs_send_sync(region);
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_shared_region_trait_bounds() {
        struct HeapRegionStub(Vec<f64>);

        impl super::SharedRegion<f64> for HeapRegionStub {
            fn as_slice(&self) -> &[f64] {
                &self.0
            }

            fn as_mut_slice(&mut self) -> &mut [f64] {
                &mut self.0
            }

            fn fence(&self) -> Result<(), CommError> {
                Ok(())
            }
        }

        let mut region = HeapRegionStub(vec![1.0, 2.0, 3.0]);
        use_shared_region_as_bound(&mut region);
        assert_shared_region_requires_send_sync(&region);
        assert_eq!(region.as_slice(), &[1.0, 2.0, 3.0]);
        assert!(region.fence().is_ok());
    }

    #[cfg(feature = "shared-memory")]
    fn use_shared_memory_provider_as_bound<P: super::SharedMemoryProvider>(provider: &P) {
        let _region_result: Result<P::Region<f64>, CommError> =
            provider.create_shared_region::<f64>(100);
        let _local_comm_result: Result<super::LocalCommKind, CommError> = provider.split_local();
        let _leader: bool = provider.is_leader();
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_shared_memory_provider_gat() {
        struct HeapRegionStub<T>(Vec<T>);

        impl<T: CommData> super::SharedRegion<T> for HeapRegionStub<T> {
            fn as_slice(&self) -> &[T] {
                &self.0
            }

            fn as_mut_slice(&mut self) -> &mut [T] {
                &mut self.0
            }

            fn fence(&self) -> Result<(), CommError> {
                Ok(())
            }
        }

        struct HeapProviderStub;

        impl super::SharedMemoryProvider for HeapProviderStub {
            type Region<T: super::CommData> = HeapRegionStub<T>;

            fn create_shared_region<T: super::CommData>(
                &self,
                count: usize,
            ) -> Result<Self::Region<T>, CommError> {
                Ok(HeapRegionStub::<T>(Vec::with_capacity(count)))
            }

            fn split_local(&self) -> Result<super::LocalCommKind, CommError> {
                Ok(super::LocalCommKind::Local(LocalBackend))
            }

            fn is_leader(&self) -> bool {
                true
            }
        }

        let provider = HeapProviderStub;
        use_shared_memory_provider_as_bound(&provider);
        let region = provider.create_shared_region::<f64>(10).unwrap();
        assert_eq!(region.as_slice().len(), 0);
        let local_comm = provider.split_local().unwrap();
        assert_eq!(local_comm.rank(), 0);
        assert_eq!(local_comm.size(), 1);
        assert!(local_comm.barrier().is_ok());
        assert!(provider.is_leader());
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_shared_memory_provider_requires_send_sync() {
        fn needs_send_sync<T: Send + Sync>(_v: &T) {}

        struct HeapRegionStub<T>(Vec<T>);

        impl<T: CommData> super::SharedRegion<T> for HeapRegionStub<T> {
            fn as_slice(&self) -> &[T] {
                &self.0
            }

            fn as_mut_slice(&mut self) -> &mut [T] {
                &mut self.0
            }

            fn fence(&self) -> Result<(), CommError> {
                Ok(())
            }
        }

        struct HeapProviderStub;

        impl super::SharedMemoryProvider for HeapProviderStub {
            type Region<T: super::CommData> = HeapRegionStub<T>;

            fn create_shared_region<T: super::CommData>(
                &self,
                _count: usize,
            ) -> Result<Self::Region<T>, CommError> {
                Ok(HeapRegionStub(vec![]))
            }

            fn split_local(&self) -> Result<super::LocalCommKind, CommError> {
                Ok(super::LocalCommKind::Local(LocalBackend))
            }

            fn is_leader(&self) -> bool {
                true
            }
        }

        let provider = HeapProviderStub;
        needs_send_sync(&provider);
    }
}
