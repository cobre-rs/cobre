//! Trait definitions for the cobre-comm abstraction layer.
//!
//! This module defines the core traits that decouple distributed computations from
//! specific communication technologies:
//!
//! - [`CommData`] — marker trait for types that can be transmitted through a
//!   `Communicator`.
//! - [`Communicator`] — backend abstraction for collective communication
//!   operations (`allgatherv`, `allreduce`, `broadcast`, `barrier`, `rank`, `size`).
//! - [`LocalCommunicator`] — object-safe sub-trait exposing only the non-generic
//!   methods of `Communicator` (`rank`, `size`, `barrier`). Used as the return type
//!   of `SharedMemoryProvider::split_local` to enable dynamic dispatch for
//!   initialization-only intra-node coordination (see the design note on
//!   [`SharedMemoryProvider::split_local`]).
//! - [`SharedRegion`] — handle to a shared memory region with read, write, and
//!   fence-based synchronization methods.
//! - [`SharedMemoryProvider`] — companion trait to `Communicator` for shared memory
//!   region management. Combined with `Communicator` via `C: Communicator +
//!   SharedMemoryProvider` bounds on functions that require both.

/// Marker trait for types that can be transmitted through collective operations.
///
/// Requires `Send + Sync` (safe to share across threads and processes),
/// `Copy` (bitwise copyable — no heap indirection), `Default`
/// (zero-initializable — required by `SharedMemoryProvider::create_shared_region`
/// to produce zero-filled regions without unsafe code), and `'static`
/// (no borrowed data).
///
/// When the `mpi` feature is enabled, `CommData` additionally requires
/// [`ferrompi::MpiDatatype`], which is the FFI marker trait sealing the seven
/// primitive types that MPI can transmit directly (`f32`, `f64`, `i32`, `i64`,
/// `u8`, `u32`, `u64`). This narrows the set of valid `CommData` types to those
/// that ferrompi can pass to the MPI C library without a custom datatype commit.
/// This restriction has no practical effect: the data plane only transmits
/// these seven primitive types in practice.
///
/// # Future extensions: non-MpiDatatype payloads
///
/// If a future feature requires communicating types that are not
/// `MpiDatatype` (e.g., `bool`, tuples, or fixed-size arrays), those
/// types cannot be transmitted directly through MPI. The caller must
/// serialize them into an MPI-compatible representation first — for
/// example, mapping `bool` slices to `u8` bitmaps or packing struct
/// fields into `f64` / `i32` arrays. This is a deliberate constraint:
/// MPI collective operations are typed at the C level, and ferrompi's
/// sealed `MpiDatatype` trait reflects that.
///
/// # Blanket implementation
///
/// All types that satisfy the bounds are automatically `CommData` without
/// any explicit impl:
///
/// ```rust
/// # use cobre_comm::CommData;
/// fn requires_comm_data<T: CommData>() {}
///
/// requires_comm_data::<f64>();  // f64 is CommData
/// requires_comm_data::<u8>();   // u8 is CommData
/// requires_comm_data::<i32>();  // i32 is CommData
/// requires_comm_data::<u64>();  // u64 is CommData
/// ```
// When the `mpi` feature is active, CommData also requires MpiDatatype so that
// the FerrompiBackend Communicator impl can delegate directly to ferrompi's
// generic FFI methods without an extra bound on each method signature.
// The intersection of CommData types (7 primitives that are MpiDatatype) is
// exactly the set of types the communication data plane ever transmits.
#[cfg(feature = "mpi")]
pub trait CommData: Send + Sync + Copy + Default + 'static + ferrompi::MpiDatatype {}

/// Blanket implementation (mpi): any type satisfying the bounds is `CommData`.
#[cfg(feature = "mpi")]
impl<T: Send + Sync + Copy + Default + 'static + ferrompi::MpiDatatype> CommData for T {}

/// Marker trait for types that can be transmitted through collective operations.
///
/// Without the `mpi` feature, `CommData` accepts all `Copy + Send + Sync +
/// Default + 'static` types, including `bool` and tuple types used in tests.
#[cfg(not(feature = "mpi"))]
pub trait CommData: Send + Sync + Copy + Default + 'static {}

/// Blanket implementation (no mpi): any type satisfying the bounds is `CommData`.
///
/// This covers all payload types used in distributed computation (f64 arrays for
/// cuts and trial points, scalar statistics) without requiring explicit impls.
#[cfg(not(feature = "mpi"))]
impl<T: Send + Sync + Copy + Default + 'static> CommData for T {}

/// Backend abstraction for collective communication operations.
///
/// The trait provides the six operations used during distributed execution:
/// four collective operations (`allgatherv`, `allreduce`, `broadcast`, `barrier`)
/// and two infallible accessors (`rank`, `size`).
///
/// # Design: compile-time static dispatch
///
/// The trait is generic — three of its six methods carry a `T: CommData` type
/// parameter. This means the trait is **intentionally not object-safe**: writing
/// `Box<dyn Communicator>` does not compile. All callers use static dispatch via
/// a generic bound `C: Communicator`, consistent with the solver abstraction
/// pattern used throughout Cobre (see `SolverInterface`).
///
/// Since Cobre builds with exactly one communicator backend per binary (ferrompi
/// for distributed execution, `LocalComm` for single-process mode), the binary
/// size cost of monomorphization is negligible — only one instantiation exists
/// per generic call site. The performance benefit is significant: `allgatherv`
/// and `allreduce` are on the hot path, and the `LocalComm` no-op implementation
/// compiles to zero instructions after inlining.
///
/// # Thread safety
///
/// The trait requires `Send + Sync` to support hybrid MPI+OpenMP execution where
/// the communicator handle is shared across threads within a rank. All collective
/// operations take `&self` (shared reference). Callers are responsible for
/// serializing concurrent calls — the calling algorithm serializes collective
/// operations so that multiple threads never invoke the same collective
/// simultaneously on the same communicator instance.
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
    /// Each rank contributes its local `send` buffer. After the call, the
    /// receive buffer `recv` is populated in rank order: rank 0's contribution
    /// occupies `recv[displs[0]..displs[0]+counts[0]]`, rank 1's contribution
    /// occupies `recv[displs[1]..displs[1]+counts[1]]`, and so on.
    ///
    /// This is the most performance-critical method in the trait. It is called
    /// twice per iteration: once after the forward pass to distribute trial
    /// points (~206 MB at production scale) and once per backward stage to
    /// synchronize new cuts (~3.2 MB per stage, ~381 MB across 119 stages).
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
    /// Returns [`crate::CommError::InvalidBufferSize`] if any buffer-size precondition
    /// is violated. Returns [`crate::CommError::CollectiveFailed`] if the underlying
    /// MPI operation fails. On error, the contents of `recv` are unspecified.
    ///
    /// # Thread safety
    ///
    /// Takes `&self`. Collective operations must not be called concurrently on
    /// the same communicator from multiple threads; the calling algorithm serializes
    /// all collective calls.
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), crate::CommError>;

    /// Reduce data from all ranks using the specified operation, with the
    /// result available on all ranks.
    ///
    /// The reduction is applied element-wise: `recv[i]` receives
    /// `op(send_0[i], send_1[i], ..., send_{R-1}[i])` for all `i`. After the
    /// call, all ranks have identical `recv` contents.
    ///
    /// This operation is used for scalar reductions: convergence bound
    /// statistics during training and min/max cost aggregation during
    /// simulation. It is NOT used for forward-pass scenario exchange
    /// (which uses [`allgatherv`](Communicator::allgatherv)).
    ///
    /// During training, two calls are issued per iteration: one with
    /// [`crate::ReduceOp::Min`] for the lower bound and one with
    /// [`crate::ReduceOp::Sum`] for the remaining UB statistics. The payload
    /// is minimal (four `f64` scalars = 32 bytes) but is on the critical
    /// path for convergence checking.
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
    /// [`crate::ReduceOp::Sum`] may produce results that vary across runs with different
    /// rank counts or MPI implementations, because floating-point addition is
    /// non-associative and the reduction tree shape is implementation-defined.
    /// This is acceptable: the upper bound is a statistical estimate and small
    /// variations do not affect convergence. [`crate::ReduceOp::Min`] is exact
    /// (comparison-based, no arithmetic).
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::InvalidBufferSize`] if `send.len() != recv.len()`.
    /// Returns [`crate::CommError::CollectiveFailed`] if the underlying MPI operation
    /// fails. On error, the contents of `recv` are unspecified.
    ///
    /// # Thread safety
    ///
    /// Same as [`allgatherv`](Communicator::allgatherv).
    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: crate::ReduceOp,
    ) -> Result<(), crate::CommError>;

    /// Broadcast data from `root` rank to all other ranks.
    ///
    /// On the root rank, `buf` contains the data to broadcast. On all other
    /// ranks, `buf` is overwritten with the data from the root rank. After the
    /// call, all ranks have identical contents in `buf`.
    ///
    /// Broadcast is used only during initialization (configuration data, case
    /// data serialized via `postcard`) and is not on the per-iteration hot path.
    /// It is guaranteed to produce identical results on all ranks regardless of
    /// backend — the output is uniquely determined by the root rank's input buffer.
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
    /// Returns [`crate::CommError::InvalidRoot`] if `root >= self.size()`. Returns
    /// [`crate::CommError::CollectiveFailed`] if the underlying MPI operation fails.
    /// On error, `buf` on non-root ranks may be partially overwritten; the root
    /// rank's `buf` is unchanged.
    ///
    /// # Thread safety
    ///
    /// Same as [`allgatherv`](Communicator::allgatherv). Broadcast is called
    /// during single-threaded initialization before any parallel regions are entered.
    fn broadcast<T: CommData>(&self, buf: &mut [T], root: usize) -> Result<(), crate::CommError>;

    /// Block until all ranks have called barrier.
    ///
    /// Used only for checkpoint synchronization — not during normal iteration
    /// execution. Per-stage synchronization in the backward pass is achieved
    /// implicitly through the `allgatherv` calls.
    ///
    /// **Every rank in the communicator must call `barrier`.** Failure to do so
    /// results in a deadlock. The operation exchanges no user data: it is a
    /// pure synchronization point.
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
    /// Returns [`crate::CommError::CollectiveFailed`] if the underlying MPI barrier
    /// fails. In practice the only failure mode is a process crash, detected
    /// as a communication timeout by the MPI runtime.
    ///
    /// # Thread safety
    ///
    /// Same as [`allgatherv`](Communicator::allgatherv).
    fn barrier(&self) -> Result<(), crate::CommError>;

    /// Return the rank index of the calling process.
    ///
    /// Ranks are numbered `0..size()`. This value is fixed at communicator
    /// initialization and never changes. Implementations must cache the value
    /// at construction time — `rank()` is called frequently per iteration for
    /// distribution arithmetic, logging, and row-slot computation.
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | In range | The returned value is in `0..self.size()` |
    /// | Unique | No two ranks in the same communicator return the same value |
    /// | Constant | The value never changes after initialization |
    ///
    /// This method is infallible — it returns `usize` directly, not `Result`.
    ///
    /// # Thread safety
    ///
    /// Safe to call concurrently from multiple threads. The cached value is
    /// read-only after initialization.
    fn rank(&self) -> usize;

    /// Return the total number of ranks in the communicator.
    ///
    /// This value is fixed at communicator initialization and never changes.
    /// Implementations must cache the value at construction time — `size()` is
    /// called frequently per iteration for distribution arithmetic, displacement
    /// computation, and statistics aggregation.
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Positive | The returned value is at least 1 (single-process mode) |
    /// | Consistent | All ranks in the communicator return the same value |
    /// | Constant | The value never changes after initialization |
    ///
    /// This method is infallible — it returns `usize` directly, not `Result`.
    ///
    /// # Thread safety
    ///
    /// Safe to call concurrently from multiple threads. The cached value is
    /// read-only after initialization.
    fn size(&self) -> usize;

    /// Abort all ranks in the communicator.
    ///
    /// Terminates every process in the communicator with the given
    /// `error_code`. On MPI backends this calls `MPI_Abort`; on the local
    /// backend it calls [`std::process::exit`].
    ///
    /// This method **never returns**. It should be called only when a
    /// non-recoverable error makes continued execution impossible, and the
    /// error cannot be coordinated through normal collective operations
    /// (e.g., an asymmetric failure where some ranks succeed and others fail).
    fn abort(&self, error_code: i32) -> !;
}

/// Object-safe sub-trait of [`Communicator`] for intra-node initialization
/// coordination.
///
/// # Design rationale
///
/// [`Communicator`] is **intentionally not object-safe** — it carries generic
/// methods (`allgatherv<T>`, `allreduce<T>`, `broadcast<T>`) that require static
/// dispatch for hot-path performance. This prevents `Box<dyn Communicator>`.
///
/// `SharedMemoryProvider::split_local` returns an intra-node communicator that
/// is used only during initialization (leader/follower role assignment, distributed
/// region population counting). This is an inherently dynamic context: the caller
/// does not know the concrete backend type at compile time, and the operation is
/// far off the hot path. Dynamic dispatch is the correct trade-off here.
///
/// `LocalCommunicator` solves the tension by exposing only the three non-generic
/// methods that intra-node setup code actually needs:
/// - [`rank`](LocalCommunicator::rank) — local rank within the intra-node group
/// - [`size`](LocalCommunicator::size) — number of ranks sharing this node
/// - [`barrier`](LocalCommunicator::barrier) — synchronization fence for
///   region population phases
///
/// The separation is intentional per spec (communicator-trait.md §4.5):
/// `SharedMemoryProvider` is orthogonal to `Communicator` and backends implement
/// them independently. `LocalCommunicator` is the bridge that lets
/// `split_local` return a trait object without compromising the zero-cost
/// static dispatch of the hot-path `Communicator` trait.
///
/// # Thread safety
///
/// The trait requires `Send + Sync`, matching the `Communicator` supertrait bounds.
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
pub trait LocalCommunicator: Send + Sync {
    /// Return the rank index of the calling process within the intra-node
    /// communicator.
    ///
    /// Lifecycle phase: **allocation** — called once during startup to determine
    /// leader/follower roles before any shared regions are created.
    ///
    /// Ranks are numbered `0..size()`. The leader for shared memory operations
    /// is always local rank 0.
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | In range | The returned value is in `0..self.size()` |
    /// | Constant | The value never changes after initialization |
    ///
    /// This method is infallible — it returns `usize` directly, not `Result`.
    fn rank(&self) -> usize;

    /// Return the total number of ranks in the intra-node communicator.
    ///
    /// Lifecycle phase: **allocation** — used to partition population work
    /// across co-located ranks before shared regions are created.
    ///
    /// For a `HeapFallback`, this always returns `1`. For a true shared memory
    /// backend (MPI, POSIX shm), this returns the number of MPI ranks on the
    /// same physical node.
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Positive | The returned value is at least 1 |
    /// | Constant | The value never changes after initialization |
    ///
    /// This method is infallible — it returns `usize` directly, not `Result`.
    fn size(&self) -> usize;

    /// Block until all intra-node ranks have called barrier.
    ///
    /// Lifecycle phase: **synchronization** — called after the leader writes
    /// shared region data and calls `SharedRegion::fence()`, to ensure all
    /// co-located ranks have completed their population step before any rank
    /// transitions to **read-only access**.
    ///
    /// Every rank in the intra-node communicator must call `barrier`.
    /// Failure to do so results in a deadlock.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::CollectiveFailed`] if the underlying
    /// barrier operation fails.
    fn barrier(&self) -> Result<(), crate::CommError>;
}

/// Handle to a shared memory region holding `count` elements of type `T`.
///
/// The region follows a strict **three-phase lifecycle**:
///
/// ## Step 1: Allocation
///
/// Created via [`SharedMemoryProvider::create_shared_region`] during the
/// solver startup phase. The leader rank (local rank 0 in the intra-node
/// communicator) allocates the full backing region. Follower ranks (local
/// rank > 0) allocate size 0 and receive a handle into the leader's memory.
///
/// For `HeapFallback` backends, every rank allocates its own `Vec<T>` of
/// `count` elements — no memory is shared, but the API is identical.
///
/// ## Step 2: Population (write + fence)
///
/// The leader writes data via [`as_mut_slice`](SharedRegion::as_mut_slice),
/// then all ranks call [`fence`](SharedRegion::fence) to publish writes and
/// acquire visibility. Followers must not read until `fence()` returns `Ok`.
///
/// ## Step 3: Read-only access
///
/// After the fence, all ranks read via [`as_slice`](SharedRegion::as_slice).
/// No further writes occur during the training loop.
///
/// ## Deallocation (RAII)
///
/// When the handle is dropped:
///
/// | Backend      | Drop behavior |
/// |--------------|---------------|
/// | ferrompi     | Calls `MPI_Win_free`, releasing the MPI window |
/// | `HeapFallback` | Drops the inner `Vec<T>` |
///
/// All ranks sharing a region must drop their handles before MPI finalization.
///
/// # Safety model
///
/// Trait methods use safe Rust signatures. Any `unsafe` for raw pointer
/// dereference into MPI shared windows is encapsulated within backend
/// implementations, not exposed here.
pub trait SharedRegion<T: CommData>: Send + Sync {
    /// Return a shared reference to the region contents as a contiguous slice.
    ///
    /// Lifecycle phase: **read-only access** — safe to call after `fence()`
    /// has returned `Ok` on all ranks.
    ///
    /// Both the leader and followers can call this method. For true shared
    /// memory backends, the slice points directly into the shared region
    /// (zero-copy across ranks). For `HeapFallback`, it points into the
    /// local `Vec<T>`.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Fence completed | `fence()` must have returned `Ok` before any follower reads |
    /// | No concurrent writes | The region must be in its read-only phase |
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Length | The returned slice has length equal to `count` from `create_shared_region` |
    /// | Zero-copy | For true shared memory backends, points directly into the shared region |
    fn as_slice(&self) -> &[T];

    /// Return a mutable reference to the region contents as a contiguous slice.
    ///
    /// Lifecycle phase: **population** — called by the leader to write data
    /// before calling `fence()`.
    ///
    /// Only the leader rank (local rank 0) should call this method. On follower
    /// ranks with a true shared memory backend, the returned slice has length 0
    /// because followers allocated size 0. On `HeapFallback`, every rank is a
    /// leader and the full mutable slice is returned.
    ///
    /// # Preconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Leader only | The caller must be the leader (`SharedMemoryProvider::is_leader()` returns `true`), or the backend must be `HeapFallback` |
    /// | Population phase | No concurrent reads may be in progress on any rank |
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Length | Returns a slice of `count` elements on the leader; length 0 on followers (true shared memory backends) |
    /// | Visibility | Writes are not visible to other ranks until `fence()` is called |
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Execute a memory fence to publish leader writes to all co-located ranks.
    ///
    /// Lifecycle phase: **synchronization** — must be called by all ranks
    /// after the leader finishes writing, before any follower calls `as_slice()`.
    ///
    /// This is a collective operation: all ranks sharing the region must call
    /// `fence()`. After `fence()` returns `Ok`, followers can safely read via
    /// `as_slice()`.
    ///
    /// | Backend      | Behavior |
    /// |--------------|----------|
    /// | ferrompi     | Maps to `MPI_Win_fence` on the underlying MPI window |
    /// | `HeapFallback` | No-op — returns `Ok(())` immediately |
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::CollectiveFailed`] if a rank has crashed or
    /// the underlying fence operation fails. On error, the visibility of prior
    /// writes is unspecified.
    fn fence(&self) -> Result<(), crate::CommError>;
}

/// Backend abstraction for intra-node shared memory region management.
///
/// `SharedMemoryProvider` is a **companion to [`Communicator`]**, not a
/// supertrait of it. A backend type may implement both traits independently.
/// When both traits are needed, use combined bounds:
///
/// ```rust
/// # use cobre_comm::{Communicator, SharedMemoryProvider};
/// fn train<C: Communicator + SharedMemoryProvider>(comm: &C) {
///     // comm.allgatherv(...) — collective communication
///     // comm.create_shared_region::<f64>(n) — shared memory allocation
/// }
/// ```
///
/// # Minimal viable simplification
///
/// For the minimal viable implementation, the training entry point uses
/// `C: Communicator` only — the `SharedMemoryProvider` bound is **not**
/// part of the `train()` constraint. `HeapFallback` semantics apply
/// uniformly: each rank holds its own per-process `Vec<T>` copy. The
/// shared memory path is activated post-profiling when memory pressure
/// warrants it.
///
/// # Separate trait rationale
///
/// Not all backends support true shared memory. Keeping `SharedMemoryProvider`
/// separate from `Communicator` preserves flexibility:
/// - `LocalComm` / `TcpComm`: implement with `HeapFallback` (no true sharing)
/// - `FerrompiComm` / `ShmComm`: implement with MPI windows or POSIX shm
///
/// Functions that only need collective communication use `C: Communicator`;
/// functions that additionally need shared memory use
/// `C: Communicator + SharedMemoryProvider`. No runtime feature detection needed.
///
/// # Thread safety
///
/// The trait requires `Send + Sync`. `create_shared_region` and `split_local`
/// are initialization-only operations called before any parallel regions are
/// entered.
pub trait SharedMemoryProvider: Send + Sync {
    /// The shared memory region handle type.
    ///
    /// Each backend provides its own concrete region type:
    ///
    /// | Backend      | Region type |
    /// |--------------|-------------|
    /// | ferrompi     | Wraps `SharedWindow<T>` (MPI window) |
    /// | `HeapFallback` | Wraps `Vec<T>` (regular heap allocation) |
    ///
    /// The type must implement [`SharedRegion<T>`] for all `T: CommData`.
    type Region<T: CommData>: SharedRegion<T>;

    /// Create a shared memory region capable of holding `count` elements of type `T`.
    ///
    /// Lifecycle phase: **allocation** — must be called during solver startup,
    /// never during the training hot path.
    ///
    /// On a backend with true shared memory, the leader rank allocates the full
    /// backing region (`count` elements) and follower ranks allocate size 0,
    /// receiving a handle into the leader's memory. On `HeapFallback`, every rank
    /// allocates its own `Vec<T>` of `count` elements.
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Ready for population | The returned region is in the allocation step; the leader may call `as_mut_slice()` |
    /// | RAII | Dropping the returned region releases the backing resource |
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::AllocationFailed`] if the OS rejects the
    /// shared memory allocation (size too large, insufficient permissions, or
    /// system shared memory limits exceeded). For `HeapFallback`, returns
    /// `AllocationFailed` only on heap exhaustion (which on most platforms causes
    /// process abort before returning `Err`).
    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, crate::CommError>;

    /// Create an intra-node communicator containing only the ranks that share
    /// the same physical node as the calling rank.
    ///
    /// Lifecycle phase: **allocation** — called once during startup to determine
    /// leader/follower roles and partition region population work across co-located
    /// ranks. Corresponds to `comm.split_shared()` in the ferrompi API
    /// (spec: communicator-trait.md §4.3, hybrid-parallelism.md §6 Step 3).
    ///
    /// # Object-safety design note
    ///
    /// This method returns `Box<dyn LocalCommunicator>` rather than
    /// `Box<dyn Communicator>`. [`Communicator`] is intentionally not
    /// object-safe (it carries generic methods for hot-path collective operations).
    /// [`LocalCommunicator`] is a purpose-built object-safe sub-trait that exposes
    /// only the three methods needed for intra-node initialization coordination:
    /// `rank()`, `size()`, and `barrier()`. This is the correct trade-off because
    /// `split_local` is an initialization-only operation where dynamic dispatch
    /// imposes negligible overhead, and the intra-node communicator is never used
    /// for hot-path generic collectives.
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Leader is rank 0 | The rank with `local_comm.rank() == 0` is the leader for shared memory allocation |
    /// | `HeapFallback` | Returns a communicator with `size() == 1` and `rank() == 0` |
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::CollectiveFailed`] if the underlying
    /// communicator split operation fails (e.g., MPI communicator split failure).
    fn split_local(&self) -> Result<Box<dyn LocalCommunicator>, crate::CommError>;

    /// Return whether the calling rank is the leader for shared memory operations
    /// on its node.
    ///
    /// Lifecycle phase: **allocation and population** — checked before
    /// `create_shared_region` to determine allocation size, and before
    /// `as_mut_slice()` to determine write permission.
    ///
    /// The leader is the rank with local rank 0 within the communicator returned
    /// by `split_local()`. The leader:
    /// - Allocates the full shared region (`count` elements)
    /// - Writes data via `as_mut_slice()`
    /// - Calls `fence()` to publish writes
    ///
    /// Followers (local rank > 0) allocate size 0, must not write, and read
    /// via `as_slice()` after `fence()` returns.
    ///
    /// On `HeapFallback`, this always returns `true` because every rank is its
    /// own leader (no memory is shared, every rank populates its own copy).
    ///
    /// # Postconditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Constant | The value never changes after initialization |
    ///
    /// This method is infallible — it returns `bool` directly, not `Result`.
    fn is_leader(&self) -> bool;
}

/// Trait for backends that can report their execution topology.
///
/// Implementors provide access to an [`ExecutionTopology`](crate::topology::ExecutionTopology)
/// gathered at communicator initialization. The topology is queried
/// non-collectively (no MPI calls) and is allocation-free after construction.
///
/// This trait is separate from [`Communicator`] because not all backends expose
/// topology information. It is implemented by backends that have the data
/// (`FerrompiBackend`, `CommBackend`) and by `LocalBackend` with a single-host fallback.
pub trait TopologyProvider: Send + Sync {
    /// Return a reference to the cached execution topology.
    ///
    /// The topology is gathered once during backend initialization.
    /// This method is non-collective and may be called from any thread.
    #[must_use]
    fn topology(&self) -> &crate::topology::ExecutionTopology;
}

#[cfg(test)]
mod tests {
    use super::{CommData, Communicator, LocalCommunicator, SharedMemoryProvider, SharedRegion};
    use crate::{CommError, ReduceOp};

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

    fn assert_local_communicator_is_object_safe(comm: &dyn super::LocalCommunicator) {
        let _ = comm.rank();
        let _ = comm.size();
    }

    fn use_local_communicator_generic<L: super::LocalCommunicator>(lc: &L) {
        let _ = lc.rank();
        let _ = lc.size();
    }

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
        assert_local_communicator_is_object_safe(&lc);
        use_local_communicator_generic(&lc);
        assert_eq!(lc.rank(), 0);
        assert_eq!(lc.size(), 1);
        assert!(lc.barrier().is_ok());
    }

    #[test]
    fn test_local_communicator_requires_send_sync() {
        fn needs_send_sync<T: Send + Sync + ?Sized>(_v: &T) {}

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
        let boxed: Box<dyn super::LocalCommunicator> = Box::new(SendSyncLocalImpl);
        needs_send_sync(boxed.as_ref());
    }

    fn use_shared_region_as_bound<R: super::SharedRegion<f64>>(region: &mut R) {
        let _slice: &[f64] = region.as_slice();
        let _mut_slice: &mut [f64] = region.as_mut_slice();
        let _fence: Result<(), CommError> = region.fence();
    }

    fn assert_shared_region_requires_send_sync<R: super::SharedRegion<f64>>(region: &R) {
        fn needs_send_sync<T: Send + Sync>(_v: &T) {}
        needs_send_sync(region);
    }

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

    fn use_shared_memory_provider_as_bound<P: super::SharedMemoryProvider>(provider: &P) {
        let _region_result: Result<P::Region<f64>, CommError> =
            provider.create_shared_region::<f64>(100);
        let _local_comm_result: Result<Box<dyn super::LocalCommunicator>, CommError> =
            provider.split_local();
        let _leader: bool = provider.is_leader();
    }

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

        struct LocalCommStub;

        impl super::LocalCommunicator for LocalCommStub {
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

        struct HeapProviderStub;

        impl super::SharedMemoryProvider for HeapProviderStub {
            type Region<T: super::CommData> = HeapRegionStub<T>;

            fn create_shared_region<T: super::CommData>(
                &self,
                count: usize,
            ) -> Result<Self::Region<T>, CommError> {
                Ok(HeapRegionStub::<T>(Vec::with_capacity(count)))
            }

            fn split_local(&self) -> Result<Box<dyn super::LocalCommunicator>, CommError> {
                Ok(Box::new(LocalCommStub))
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

        struct LocalCommStub;

        impl super::LocalCommunicator for LocalCommStub {
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

        struct HeapProviderStub;

        impl super::SharedMemoryProvider for HeapProviderStub {
            type Region<T: super::CommData> = HeapRegionStub<T>;

            fn create_shared_region<T: super::CommData>(
                &self,
                _count: usize,
            ) -> Result<Self::Region<T>, CommError> {
                Ok(HeapRegionStub(vec![]))
            }

            fn split_local(&self) -> Result<Box<dyn super::LocalCommunicator>, CommError> {
                Ok(Box::new(LocalCommStub))
            }

            fn is_leader(&self) -> bool {
                true
            }
        }

        let provider = HeapProviderStub;
        needs_send_sync(&provider);
    }
}
