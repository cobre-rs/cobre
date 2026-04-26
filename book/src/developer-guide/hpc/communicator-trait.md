# Communicator Trait

## Purpose

This spec defines the `Communicator` trait -- the backend abstraction through which distributed optimization algorithms perform collective communication operations. The trait provides a pluggable interface for MPI communication backends while preserving determinism invariants required by iterative decomposition algorithms (rank-ordered receives, identical distributed state across all ranks) and achieving zero-cost abstraction via static dispatch. The concrete communicator type is a generic parameter resolved at compile time, following the same pattern as LP solver selection in [Solver Abstraction §10](../architecture/solver-abstraction.md). In single-process mode ([Hybrid Parallelism §1.0a](./hybrid-parallelism.md)), a no-op implementation satisfies the trait with zero runtime cost.

> **Convention: Rust traits as specification guidelines.** The Rust trait definitions, method signatures, and struct declarations throughout this specification corpus serve as _guidelines for implementation_, not as absolute source-of-truth contracts that must be reproduced verbatim. Their purpose is twofold: (1) to express behavioral contracts, preconditions, postconditions, and type-level invariants more precisely than prose alone, and (2) to anchor conformance test suites that verify backend interchangeability (see [Backend Testing §1](./backend-testing.md)). Implementation may diverge in naming, parameter ordering, error representation, or internal organization when practical considerations demand it -- provided the behavioral contracts and conformance tests continue to pass. When a trait signature and a prose description conflict, the prose description (which captures the domain intent) takes precedence; the conflict should be resolved by updating the trait signature. This convention applies to all trait-bearing specification documents in `src/specs/`.

## 1. Trait Definition

### 1.1 Core Trait

```rust
/// Backend abstraction for SDDP collective communication operations.
///
/// Implementations provide the four collective operations used during SDDP
/// training (allgatherv, allreduce, broadcast, barrier) plus rank/size queries.
/// The trait requires `Send + Sync` to support hybrid MPI+OpenMP execution
/// where the communicator handle is shared across threads within a rank.
pub trait Communicator: Send + Sync {
    /// Gather variable-length data from all ranks into all ranks.
    ///
    /// Each rank contributes `send.len()` elements. The receive buffer `recv`
    /// is populated in rank order: rank 0's data at `displs[0]`, rank 1's data
    /// at `displs[1]`, etc. The caller must ensure `recv` is large enough to
    /// hold `sum(counts)` elements.
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError>;

    /// Reduce data from all ranks using the specified operation, with the
    /// result available on all ranks.
    ///
    /// `send` and `recv` must have the same length. The reduction operation
    /// is applied element-wise across all ranks.
    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError>;

    /// Broadcast data from `root` rank to all other ranks.
    ///
    /// On the root rank, `buf` contains the data to broadcast. On all other
    /// ranks, `buf` is overwritten with the data from root. After the call,
    /// all ranks have identical contents in `buf`.
    fn broadcast<T: CommData>(
        &self,
        buf: &mut [T],
        root: usize,
    ) -> Result<(), CommError>;

    /// Block until all ranks have called barrier.
    ///
    /// Used only for checkpoint synchronization -- not on the per-iteration
    /// hot path. The `MPI_Allgatherv` calls provide implicit barriers at
    /// stage boundaries during the backward pass.
    fn barrier(&self) -> Result<(), CommError>;

    /// Return the rank index of the calling process.
    ///
    /// Ranks are numbered `0..size()`. This value is constant after
    /// communicator initialization and the call is infallible.
    fn rank(&self) -> usize;

    /// Return the total number of ranks in the communicator.
    ///
    /// This value is constant after communicator initialization and
    /// the call is infallible.
    fn size(&self) -> usize;
}
```

### 1.2 Data Trait

```rust
/// Marker trait for types that can be transmitted through collective operations.
///
/// Requires `Send + Sync` (safe to share across threads and processes),
/// `Copy` (bitwise copyable -- no heap indirection), `Default`
/// (zero-initializable -- required by `SharedMemoryProvider::create_shared_region`
/// to produce zero-filled regions via `vec![T::default(); count]` without
/// unsafe code), and `'static` (no borrowed data).
pub trait CommData: Send + Sync + Copy + Default + 'static {}

/// When the `mpi` feature is enabled, CommData additionally requires
/// `MpiDatatype` so that values can be passed to MPI collective operations.
#[cfg(feature = "mpi")]
pub trait CommData: Send + Sync + Copy + Default + 'static + MpiDatatype {}

#[cfg(not(feature = "mpi"))]
pub trait CommData: Send + Sync + Copy + Default + 'static {}

/// Blanket implementation: any type satisfying the bounds is CommData.
#[cfg(feature = "mpi")]
impl<T: Send + Sync + Copy + Default + 'static + MpiDatatype> CommData for T {}

#[cfg(not(feature = "mpi"))]
impl<T: Send + Sync + Copy + Default + 'static> CommData for T {}
```

The `Default` bound was added because `SharedMemoryProvider::create_shared_region` produces zero-filled regions via `vec![T::default(); count]`. Without `Default`, the region allocation would require `unsafe` code (e.g., `MaybeUninit`) or an additional initialization step. All seven MPI-transmissible primitive types (`f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`) implement `Default` with a zero value, so this bound has no practical narrowing effect on the type set.

The blanket implementation ensures that all `Copy + Send + Sync + Default + 'static` types automatically satisfy `CommData`, covering all payload types in [Communication Patterns §2](./communication-patterns.md) without explicit trait implementations.

### 1.3 Reduction Operations

```rust
/// Element-wise reduction operations for `allreduce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise summation. Used for upper bound statistics
    /// (total cost sum, sum of squares, trajectory count).
    Sum,

    /// Element-wise minimum. Used for lower bound aggregation
    /// (first-stage LP objective across ranks).
    Min,

    /// Element-wise maximum. Reserved for future use
    /// (e.g., maximum per-rank solve time for load balance diagnostics).
    Max,
}
```

The `Sum` and `Min` variants map to `MPI_SUM` and `MPI_MIN` respectively. During SDDP training, the convergence statistics (upper bound) use `allgatherv` of the full per-scenario cost vector followed by canonical-order local summation (see [Communication Patterns §2.3](./communication-patterns.md)), while the lower bound uses `broadcast` from rank 0. `allreduce` with `ReduceOp::Sum` and `ReduceOp::Min` is used in simulation mode for global min/max cost aggregation. `ReduceOp::Max` is reserved for future diagnostics (e.g., maximum per-rank solve time for load balance analysis).

### 1.4 Error Type

```rust
/// Errors that can occur during collective communication operations.
#[derive(Debug)]
pub enum CommError {
    /// An MPI collective operation failed at the library level.
    /// Contains the MPI error code and a human-readable description.
    CollectiveFailed {
        operation: &'static str,
        mpi_error_code: i32,
        message: String,
    },

    /// The buffer sizes provided to a collective operation are inconsistent.
    /// For example, `recv.len() < sum(counts)` in `allgatherv`, or
    /// `send.len() != recv.len()` in `allreduce`.
    InvalidBufferSize {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },

    /// The `root` rank argument is out of range (`root >= size()`).
    InvalidRoot {
        root: usize,
        size: usize,
    },

    /// The communicator has been finalized or is in an invalid state.
    /// This typically occurs if MPI_Finalize has been called.
    InvalidCommunicator,

    /// A buffer allocation required for a collective operation failed.
    /// This may occur when assembling large gather/scatter buffers.
    AllocationFailed {
        operation: &'static str,
        bytes_requested: usize,
    },
}
```

## 2. Method Contracts

### 2.1 allgatherv

`allgatherv` is the most performance-critical method in the trait. It is called twice per iteration during SDDP training: once to distribute visited trial points after the forward pass (~206 MB) and once per backward stage to synchronize new cuts (~3.2 MB per stage, ~381 MB across 119 stages). The combined per-iteration payload is approximately 587 MB at worst-case scale ($T = 120$ stages; the DEC-009 production baseline of $T = 60$ stages halves these figures). See [Communication Patterns §3.1](./communication-patterns.md).

**Preconditions:**

| Condition                                 | Description                                                                      |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| `counts.len() == self.size()`             | One count entry per rank                                                         |
| `displs.len() == self.size()`             | One displacement entry per rank                                                  |
| `send.len() == counts[self.rank()]`       | Send buffer matches this rank's declared count                                   |
| `recv.len() >= displs[R-1] + counts[R-1]` | Receive buffer large enough for all ranks' data                                  |
| Non-overlapping regions                   | `recv[displs[r]..displs[r]+counts[r]]` regions must not overlap for distinct `r` |

**Postconditions:**

| Condition              | Description                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Rank-ordered receive   | `recv[displs[r]..displs[r]+counts[r]]` contains the data sent by rank `r`, for all `r` in `0..size()` |
| Identical across ranks | After the call returns, all ranks have identical contents in `recv`                                   |
| Implicit barrier       | No rank returns from `allgatherv` until all ranks have contributed their data                         |

**Determinism guarantee:** The rank-ordered receive semantics are critical for SDDP correctness. The receive buffer is populated in rank order (rank 0, rank 1, ..., rank $R-1$), which ensures that all ranks construct identical cut pools and trial point sets after synchronization. This is the foundation of the reproducibility invariant in [Communication Patterns §6.1](./communication-patterns.md).

**Error semantics:** Returns `CommError::InvalidBufferSize` if any precondition on buffer sizes is violated. Returns `CommError::CollectiveFailed` if the underlying MPI operation fails. On error, the contents of `recv` are unspecified.

**Thread safety:** The method takes `&self`. Collective operations are serialized by the training loop -- multiple threads must not call `allgatherv` concurrently on the same communicator. The `Send + Sync` bound on the trait allows the handle to be shared across threads.

### 2.2 allreduce

`allreduce` performs element-wise reductions across all ranks. During SDDP training, the post-forward convergence statistics use `allgatherv` (not `allreduce`) for deterministic canonical-order summation. `allreduce` is used in simulation mode for global min/max cost aggregation and for any future scalar reduction needs. The payload is typically minimal (1--4 `f64` scalars).

**Preconditions:**

| Condition                  | Description                                |
| -------------------------- | ------------------------------------------ |
| `send.len() == recv.len()` | Send and receive buffers have equal length |
| `send.len() > 0`           | At least one element to reduce             |

**Postconditions:**

| Condition              | Description                                                          |
| ---------------------- | -------------------------------------------------------------------- |
| Element-wise reduction | `recv[i] = op(send_0[i], send_1[i], ..., send_{R-1}[i])` for all `i` |
| Identical across ranks | After the call returns, all ranks have identical contents in `recv`  |

**Reduction operations:**

- `ReduceOp::Sum` -- Element-wise summation across all ranks. Used in simulation mode for global cost aggregation.
- `ReduceOp::Min` -- Element-wise minimum across all ranks. Used in simulation mode for global minimum cost.
- `ReduceOp::Max` -- Element-wise maximum across all ranks. Used in simulation mode for global maximum cost.

> **Note on SDDP training.** During SDDP training, the upper bound (UB) statistics use `allgatherv` of the full per-scenario cost vector followed by canonical-order local summation (see [Communication Patterns §2.3](./communication-patterns.md)), not `allreduce`. The lower bound (LB) uses `broadcast` from rank 0. This design eliminates floating-point non-associativity as a source of non-determinism.

**Floating-point non-determinism:** `allreduce` with `ReduceOp::Sum` may produce results that vary across runs with different rank counts or MPI implementations, because floating-point addition is non-associative and the reduction tree shape is implementation-defined. For simulation-mode aggregation, this variance is acceptable. `ReduceOp::Min` and `ReduceOp::Max` are exact (comparison-based, no arithmetic). See [Communication Patterns §6.2](./communication-patterns.md).

**Error semantics:** Returns `CommError::InvalidBufferSize` if `send.len() != recv.len()`. Returns `CommError::CollectiveFailed` if the underlying MPI operation fails. On error, the contents of `recv` are unspecified.

**Thread safety:** Same as `allgatherv`.

### 2.3 broadcast

`broadcast` distributes data from a designated root rank to all other ranks. It is used during initialization (configuration data, case data) and once per training iteration to distribute the lower bound from rank 0 to all ranks after the backward pass (see [Communication Patterns §2.3](./communication-patterns.md)).

**Preconditions:**

| Condition                          | Description                              |
| ---------------------------------- | ---------------------------------------- |
| `root < self.size()`               | Root rank is valid                       |
| `buf.len()` identical on all ranks | All ranks provide the same buffer length |

**Postconditions:**

| Condition              | Description                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| Data from root         | After the call, `buf` on every rank contains the data that was in `buf` on the root rank before the call |
| Identical across ranks | All ranks have identical contents in `buf`                                                               |

**Determinism guarantee:** Broadcast is deterministic -- all ranks receive an identical copy of the root rank's data, regardless of backend. The output is uniquely determined by the root rank's input buffer.

**Error semantics:** Returns `CommError::InvalidRoot` if `root >= self.size()`. Returns `CommError::CollectiveFailed` if the underlying MPI operation fails. On error, the contents of `buf` on non-root ranks are unspecified; the root rank's `buf` is unchanged.

**Thread safety:** Same as `allgatherv`. Broadcast is called during single-threaded initialization before OpenMP parallel regions are entered.

### 2.4 barrier

`barrier` blocks the calling rank until all ranks have entered the barrier. It is used only for checkpoint synchronization -- not during normal iteration execution. The per-stage synchronization in the backward pass is achieved implicitly through the `allgatherv` calls ([Communication Patterns §1.2](./communication-patterns.md)).

**Preconditions:**

| Condition      | Description                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------- |
| All ranks call | Every rank in the communicator must call `barrier`. Failure to do so results in a deadlock. |

**Postconditions:**

| Condition              | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| Global synchronization | No rank returns from `barrier` until all ranks have entered it |
| No data exchange       | `barrier` does not transmit or modify any user data            |

**Determinism guarantee:** Barrier is deterministic -- it is a pure synchronization point that exchanges no data. All backends must implement the same all-ranks-must-enter-before-any-returns semantics.

**Error semantics:** Returns `CommError::CollectiveFailed` if the underlying MPI barrier fails. In practice, the only failure mode is a process crash (detected as a communication timeout by the MPI runtime).

**Thread safety:** Same as `allgatherv`.

### 2.5 rank and size

`rank()` and `size()` are accessor methods that return the calling process's rank index and the total number of ranks in the communicator, respectively.

**Preconditions:** None. These methods can be called at any time after communicator initialization.

**Postconditions:**

| Condition  | Description                                                                 |
| ---------- | --------------------------------------------------------------------------- |
| Constant   | The return values are fixed at communicator initialization and never change |
| Consistent | `rank()` returns a value in `0..size()` on every rank                       |
| Unique     | No two ranks in the same communicator return the same value from `rank()`   |

**Error semantics:** Infallible -- these methods return `usize` directly, not `Result`.

**Thread safety:** Safe to call concurrently from multiple OpenMP threads. Implementations must cache the values at construction time, as these methods are called frequently (per-iteration distribution arithmetic, logging, cut slot computation).

## 3. Generic Parameterization

The SDDP training loop and all functions that perform communication are generic over `C: Communicator`, enabling compile-time monomorphization consistent with [Solver Abstraction §10](../architecture/solver-abstraction.md).

```rust
/// Train the SDDP policy using the provided communicator backend.
///
/// The generic parameter `C` is resolved at compile time -- no trait
/// object indirection, no vtable lookup, no dynamic dispatch.
pub fn train<C: Communicator>(
    comm: &C,
    config: &TrainingConfig,
    stages: &[StageTemplate],
    // ... other parameters
) -> TrainingResult {
    let rank = comm.rank();
    let size = comm.size();

    // Distribution arithmetic uses rank/size
    let (my_start, my_count) = contiguous_block(config.forward_passes, rank, size);

    // Forward pass -- no communication
    let local_stats = forward_pass(comm, stages, my_start, my_count);

    // Aggregate convergence statistics
    let mut global_stats = [0.0f64; 4];
    comm.allreduce(&local_stats.sum_array(), &mut global_stats, ReduceOp::Sum)?;

    // ... backward pass with allgatherv for cuts ...
}
```

**Rationale for static dispatch over dynamic dispatch:**

| Aspect           | Static (`C: Communicator`)            | Dynamic (`&dyn Communicator`)     |
| ---------------- | ------------------------------------- | --------------------------------- |
| Dispatch cost    | Zero -- inlined at compile time       | Vtable lookup per call            |
| Monomorphization | Full -- optimizer sees concrete types | None -- opaque trait object       |
| Binary size      | One specialized copy per backend      | Single generic copy               |
| Consistency      | Matches solver pattern (§10)          | Would diverge from solver pattern |

Since Cobre builds with exactly one communicator backend per binary (ferrompi for MPI execution, no-op for single-process mode), the binary size cost of monomorphization is negligible -- only one instantiation exists. The performance benefit is significant: `allgatherv` and `allreduce` are called hundreds of times per training run, and the no-op backend compiles to zero instructions after inlining.

## 4. SharedMemoryProvider Trait

The `SharedMemoryProvider` trait is a companion to `Communicator` for shared memory region management. It is defined separately because not all backends support shared memory: the ferrompi backend uses MPI windows ([Communication Patterns §5.1](./communication-patterns.md)), the shm backend uses OS shared memory primitives, while the local and tcp backends provide fallback implementations using regular heap allocation.

### 4.1 Trait Definition

```rust
/// Backend abstraction for intra-node shared memory region management.
///
/// Implementations provide creation and management of shared memory regions
/// that allow ranks on the same physical node to access common data without
/// replication. The trait follows the leader/follower allocation pattern
/// established in [Shared Memory Aggregation §1.1]: one rank (the leader)
/// allocates the region and writes data; other ranks (followers) read via
/// direct pointer access.
pub trait SharedMemoryProvider: Send + Sync {
    /// The shared memory region handle type.
    ///
    /// Each backend provides its own concrete region type:
    /// - ferrompi: wraps `SharedWindow<T>` (MPI window)
    /// - shm: wraps OS shared memory (POSIX shm_open / mmap)
    /// - HeapFallback: wraps `Vec<T>` (regular heap allocation)
    type Region<T: CommData>: SharedRegion<T>;

    /// Create a shared memory region capable of holding `count` elements
    /// of type `T`.
    ///
    /// On a backend with true shared memory, the leader rank allocates the
    /// full region and follower ranks allocate size 0 (receiving a handle
    /// to the leader's memory). On the `HeapFallback`, every rank allocates
    /// its own `Vec<T>` of `count` elements.
    ///
    /// This is an initialization-only operation -- it is never called during
    /// the SDDP training hot path. All shared regions are created during the
    /// startup phase (see [Hybrid Parallelism §6 Step 3]).
    ///
    /// # Errors
    ///
    /// Returns `CommError::AllocationFailed` if the OS rejects the shared
    /// memory allocation (size too large, insufficient permissions, or
    /// system shared memory limits exceeded).
    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError>;

    /// Create a sub-communicator containing only the ranks that share the
    /// same physical node as the calling rank.
    ///
    /// The returned communicator is used for coordinating shared memory
    /// operations (determining leader/follower roles, distributing generation
    /// work within a node). This corresponds to `comm.split_shared()`
    /// in the ferrompi API ([Hybrid Parallelism §6 Step 3]).
    ///
    /// The returned communicator uses dynamic dispatch (`Box<dyn LocalCommunicator>`)
    /// because this is an initialization-only operation -- the local
    /// communicator is used for setup coordination (rank/size queries and
    /// barrier), not hot-path generic collectives. `LocalCommunicator` is
    /// an object-safe sub-trait exposing only `rank()`, `size()`, and
    /// `barrier()`.
    ///
    /// # Errors
    ///
    /// Returns `CommError::CollectiveFailed` if the underlying split operation
    /// fails (e.g., MPI communicator split failure).
    fn split_local(&self) -> Result<Box<dyn LocalCommunicator>, CommError>;

    /// Return whether the calling rank is the leader for shared memory
    /// operations on its node.
    ///
    /// The leader is the rank with local rank 0 within the intra-node
    /// communicator (the communicator returned by `split_local()`). The
    /// leader is responsible for allocating shared regions (full size) and
    /// writing shared data; followers allocate size 0 and read only.
    ///
    /// On the `HeapFallback`, this always returns `true` because every rank
    /// is its own leader (no memory is shared).
    ///
    /// This value is constant after initialization and the call is infallible.
    fn is_leader(&self) -> bool;
}
```

### 4.2 SharedRegion Type

The `SharedRegion<T>` trait defines the handle to a shared memory region. It provides read access (for all ranks), write access (for the leader only), and fence-based synchronization to ensure write visibility across ranks. The region is freed when the handle is dropped (RAII semantics).

```rust
/// Handle to a shared memory region holding `count` elements of type `T`.
///
/// The region has three lifecycle phases:
/// 1. **Allocation**: Created via `SharedMemoryProvider::create_shared_region`.
///    The leader's region contains the backing memory; followers hold a
///    handle to the leader's memory.
/// 2. **Population**: The leader writes data via `as_mut_slice()`, then
///    calls `fence()` to ensure writes are visible to all ranks on the node.
///    Followers must not read until the fence completes.
/// 3. **Read-only access**: After the fence, all ranks read via `as_slice()`.
///    No further writes occur during SDDP training. The data is read-only
///    for the duration of the training loop.
///
/// RAII: When the `SharedRegion` is dropped, the underlying shared memory
/// resource is released. For ferrompi, this calls `MPI_Win_free`; for
/// the HeapFallback, this drops the `Vec<T>`.
///
/// # Safety Model
///
/// The trait methods use safe Rust signatures. The `unsafe` boundary for
/// raw pointer dereference into MPI shared windows is encapsulated within
/// the ferrompi backend implementation, not exposed through the trait.
pub trait SharedRegion<T: CommData>: Send + Sync {
    /// Return a shared reference to the region contents as a contiguous slice.
    ///
    /// All ranks (leader and followers) can call this method. The returned
    /// slice provides zero-copy read access to the shared data.
    ///
    /// # Preconditions
    ///
    /// - The region must have been populated by the leader and a `fence()`
    ///   must have completed before any follower calls `as_slice()`.
    /// - No concurrent writes may be in progress (the region is in its
    ///   read-only phase).
    ///
    /// # Postconditions
    ///
    /// - The returned slice has length equal to the `count` argument passed
    ///   to `create_shared_region`.
    /// - For true shared memory backends, the slice points directly into
    ///   the shared region (zero-copy). For `HeapFallback`, it points into
    ///   the local `Vec<T>`.
    fn as_slice(&self) -> &[T];

    /// Return a mutable reference to the region contents as a contiguous slice.
    ///
    /// Only the leader rank may call this method. On follower ranks, the
    /// behavior is backend-defined: the ferrompi backend returns a zero-length
    /// slice (followers allocated size 0); the `HeapFallback` returns the full
    /// mutable slice (every rank is a leader).
    ///
    /// # Preconditions
    ///
    /// - The caller must be the leader (`SharedMemoryProvider::is_leader()`
    ///   returns `true`) or the backend must be `HeapFallback`.
    /// - No concurrent reads via `as_slice()` may be in progress on any
    ///   rank (the region is in its population phase).
    ///
    /// # Postconditions
    ///
    /// - The returned slice has length equal to `count` on the leader,
    ///   or length 0 on a follower (for true shared memory backends).
    /// - Writes to the returned slice are not visible to other ranks until
    ///   `fence()` is called.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Execute a memory fence that ensures all preceding writes by the
    /// leader are visible to all ranks that share this region.
    ///
    /// This is a collective operation -- all ranks sharing the region must
    /// call `fence()` for the operation to complete. After `fence()` returns,
    /// follower ranks can safely read the data written by the leader.
    ///
    /// For the ferrompi backend, this maps to `MPI_Win_fence` on the
    /// underlying MPI window. For `HeapFallback`, this is a no-op (there
    /// are no remote ranks to synchronize with).
    ///
    /// # Errors
    ///
    /// Returns `CommError::CollectiveFailed` if a rank has crashed or the
    /// underlying fence operation fails. On error, the visibility of prior
    /// writes is unspecified.
    fn fence(&self) -> Result<(), CommError>;
}
```

**Lifetime and ownership semantics:** The `SharedRegion<T>` handle owns the shared memory resource. When the handle is dropped:

| Backend      | Drop Behavior                                                               |
| ------------ | --------------------------------------------------------------------------- |
| ferrompi     | Calls `MPI_Win_free`, releasing the MPI window and underlying shared memory |
| shm          | Calls `shm_unlink` / `munmap`, releasing the POSIX shared memory segment    |
| HeapFallback | Drops the inner `Vec<T>`, returning heap memory to the allocator            |

All ranks sharing a region must drop their handles before MPI finalization. The training loop ensures this by dropping shared regions during the shutdown phase, before `MPI_Finalize` is called.

### 4.3 Leader/Follower Pattern

The leader/follower allocation pattern is central to shared memory usage. It mirrors the model established in [Shared Memory Aggregation §1.1](./shared-memory-aggregation.md):

| Phase               | Leader (local rank 0)                    | Follower (local rank > 0)                                 |
| ------------------- | ---------------------------------------- | --------------------------------------------------------- |
| **Allocation**      | Allocates full region (`count` elements) | Allocates size 0 (receives handle to leader's memory)     |
| **Population**      | Writes data via `as_mut_slice()`         | Waits (must not read until fence)                         |
| **Synchronization** | Calls `fence()` to publish writes        | Calls `fence()` to acquire visibility                     |
| **Training**        | Reads via `as_slice()` (zero-copy)       | Reads via `as_slice()` (zero-copy into leader's memory)   |
| **Deallocation**    | Drop releases the backing memory         | Drop releases the handle (memory freed when leader drops) |

**Leader determination:** The leader is the rank with local rank 0 in the intra-node communicator returned by `split_local()`. This is consistent with the `split_shared()` convention in ferrompi ([Hybrid Parallelism §6 Step 3](./hybrid-parallelism.md)).

**Distributed population:** For large shared regions (e.g., the opening tree), population work can be distributed across all ranks on the node, not just the leader. Each rank computes its assigned portion and writes directly to the shared region at the correct offset. The generation protocol from [Shared Memory Aggregation §1.2](./shared-memory-aggregation.md):

1. Leader allocates `SharedRegion<f64>` with total opening tree size
2. All ranks on the node compute their assigned portion (contiguous block assignment within the intra-node communicator)
3. Each rank writes its portion to the shared region at its assigned offset (the leader provides the base pointer; followers write via the shared memory mapping)
4. All ranks call `fence()` to ensure mutual visibility
5. All ranks read any element via `as_slice()` for the remainder of training

**Shared data candidates and their population patterns:**

| Data                | Population Strategy                  | Size (production scale) | Reference                                                          |
| ------------------- | ------------------------------------ | ----------------------- | ------------------------------------------------------------------ |
| Opening tree        | Distributed across intra-node ranks  | ~0.8 MB                 | [Scenario Generation §2.3](../architecture/scenario-generation.md) |
| Input case data     | Leader loads, followers read         | ~20 MB                  | [Shared Memory Aggregation §1.3](./shared-memory-aggregation.md)   |
| Cut pool (optional) | Leader integrates after `Allgatherv` | ~250 MB at capacity     | [Shared Memory Aggregation §1.4](./shared-memory-aggregation.md)   |

### 4.4 Fallback Strategy

Backends that do not support true shared memory (local, tcp) use a `HeapFallback` implementation that provides the same API with per-process heap allocation. The training loop code works identically -- it calls `create_shared_region`, `as_slice`, `as_mut_slice`, and `fence` without knowing whether memory is truly shared or locally allocated. The difference is purely in memory footprint: true sharing saves memory proportional to the number of ranks per node, while the fallback replicates data in each process.

**HeapFallback semantics:**

| Method                 | HeapFallback Behavior                                       | True Shared Memory Behavior                                |
| ---------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| `create_shared_region` | Allocates `Vec<T>` with `count` elements (per-process copy) | Leader allocates shared region; followers get handle to it |
| `is_leader`            | Always returns `true` (every rank is its own leader)        | Returns `true` only for local rank 0                       |
| `split_local`          | Returns a single-rank communicator (rank 0 of size 1)       | Returns intra-node communicator with co-located ranks      |
| `as_slice`             | Returns `&self.vec[..]` (local heap memory)                 | Returns slice into shared region (zero-copy across ranks)  |
| `as_mut_slice`         | Returns `&mut self.vec[..]` (local heap memory)             | Returns mutable slice (leader only; followers get empty)   |
| `fence`                | No-op (returns `Ok(())`)                                    | `MPI_Win_fence` or equivalent synchronization primitive    |
| `Drop`                 | Drops inner `Vec<T>`                                        | Releases shared memory resource (MPI window, shm segment)  |

**Why `is_leader` returns `true` on HeapFallback:** When `is_leader()` returns `true`, the calling code proceeds to write data into the region via `as_mut_slice()`. On the HeapFallback, every rank must populate its own copy because there is no shared memory -- so every rank must be a leader. The `split_local()` method returns a single-rank communicator (size 1), ensuring that distributed population logic (which divides work by the local communicator's size) assigns all work to the single rank.

**Memory footprint comparison:**

| Configuration          | Opening Tree | Case Data | Total Shareable | Savings vs. Replicated |
| ---------------------- | -----------: | --------: | --------------: | ---------------------: |
| HeapFallback (4 ranks) |       3.2 MB |     80 MB |         83.2 MB |                   none |
| True shared (4 ranks)  |       0.8 MB |     20 MB |         20.8 MB |               ~62.4 MB |

The savings figures align with [Memory Architecture §2.2](./memory-architecture.md). For deployments where memory is not constrained, the HeapFallback is functionally correct and avoids the complexity of shared memory window management.

### 4.5 Optional Implementation and Generic Bounds

The `SharedMemoryProvider` trait is intentionally separate from `Communicator`. A backend type may implement both traits, one trait, or neither. The training loop uses generic bounds to express its requirements:

```rust
/// Train the SDDP policy using the provided communicator and shared
/// memory backends.
///
/// The generic parameter `C` implements both `Communicator` (for collective
/// operations) and `SharedMemoryProvider` (for shared memory regions). Both
/// traits are resolved at compile time -- no dynamic dispatch on the hot path.
pub fn train<C: Communicator + SharedMemoryProvider>(
    comm: &C,
    config: &TrainingConfig,
    stages: &[StageTemplate],
    // ... other parameters
) -> TrainingResult {
    // Shared memory region creation (initialization phase)
    let opening_tree = comm.create_shared_region::<f64>(opening_tree_size)?;
    let case_data = comm.create_shared_region::<u8>(case_data_size)?;

    // Leader populates shared regions
    if comm.is_leader() {
        let tree_slice = opening_tree.as_mut_slice();
        generate_opening_tree(tree_slice, config);
    }
    opening_tree.fence()?;

    // All ranks read shared data during training (zero-copy)
    let tree = opening_tree.as_slice();

    // ... training loop using comm.allgatherv, comm.allreduce, etc. ...
}
```

**Backend types and their trait implementations:**

| Backend Type   | `Communicator` | `SharedMemoryProvider` | Shared Memory Mechanism      |
| -------------- | :------------: | :--------------------: | ---------------------------- |
| `FerrompiComm` |      Yes       |          Yes           | MPI windows (`SharedWindow`) |
| `ShmComm`      |      Yes       |          Yes           | POSIX shared memory          |
| `LocalComm`    |      Yes       |   Yes (HeapFallback)   | Per-process `Vec<T>`         |
| `TcpComm`      |      Yes       |   Yes (HeapFallback)   | Per-process `Vec<T>`         |

All four backend types implement both traits, so the `C: Communicator + SharedMemoryProvider` bound is always satisfiable -- callers never need to check for shared memory support at runtime.

**Alternative: conditional shared memory via separate trait bounds.** If a future backend needs to implement `Communicator` without `SharedMemoryProvider` (e.g., a minimal testing stub), functions that require shared memory can use the combined bound while functions that do not can use `C: Communicator` alone:

```rust
/// Initialization requires shared memory for opening tree setup.
fn initialize<C: Communicator + SharedMemoryProvider>(comm: &C) -> SharedState { ... }

/// The backward pass only needs collective communication.
fn backward_pass<C: Communicator>(comm: &C, cuts: &CutPool) { ... }
```

This separation preserves flexibility without introducing runtime checks or feature flags for shared memory support.

### 4.6 Error Variant Extension

The `CommError` enum (§1.4) is extended with one additional variant to cover shared memory allocation failures:

```rust
/// Errors that can occur during collective communication or shared memory
/// operations.
#[derive(Debug)]
pub enum CommError {
    // ... existing variants from §1.4 ...

    /// A shared memory allocation request was rejected by the OS.
    /// This can occur if the requested size exceeds system shared memory
    /// limits (`/proc/sys/kernel/shmmax` on Linux), if the process lacks
    /// permissions for shared memory operations, or if the system is out
    /// of shared memory resources.
    AllocationFailed {
        requested_bytes: usize,
        message: String,
    },
}
```

For the `HeapFallback`, `create_shared_region` delegates to standard heap allocation, which follows Rust's standard allocation failure semantics (abort on OOM by default). The `AllocationFailed` variant is relevant only for true shared memory backends where the OS may reject the allocation for reasons other than total memory exhaustion (e.g., shared memory segment limits, permission errors).

### 4.7 Minimal Viable Simplification

> **Minimal viable note (Phase 5).** For the minimal viable implementation, the training entry point uses `C: Communicator` only -- the `SharedMemoryProvider` bound is **not** part of the `train()` generic constraint. Each MPI rank operates with isolated per-rank heap memory: the `System` struct, opening tree, and cut pool are replicated in each rank's heap. No MPI shared memory windows (`SharedWindow<T>`) are created or used. The `HeapFallback` semantics described in §4.4 apply uniformly to all backends in Phase 5, including the ferrompi backend.

The `SharedMemoryProvider` trait definition (§4.1), the `SharedRegion<T>` type (§4.2), the leader/follower pattern (§4.3), the combined generic bound shown in §4.5, and the `AllocationFailed` error variant (§4.6) remain in this spec as the **target architecture** for a post-profiling optimization pass. They are not dead spec content -- they represent the design that will be activated when memory pressure warrants it.

**Constraint preventing early implementation.** True shared memory requires MPI window management (`MPI_Win_create`, `MPI_Win_fence`, `MPI_Win_free`) with correct leader/follower lifecycle coordination across ranks on the same node. Implementing this before the training loop, cut pool, and scenario generation are stable would couple the shared memory lifecycle to components still under active development. The `HeapFallback` path eliminates this coupling while remaining functionally correct.

**Trigger conditions for re-introducing shared memory.** The optimization will be activated when production-scale profiling (60 stages, 160 hydro plants, 15,000 cuts per stage) demonstrates any of the following:

| Trigger | Condition                                                              | Measurement                                                                                                      |
| ------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| T1      | Per-rank memory exceeds 80% of NUMA domain memory                      | `peak_rss` from `/proc/self/status` per rank vs. NUMA domain capacity from `libnuma`                             |
| T2      | Cut pool replication causes swap activity or OOM on standard HPC nodes | `vmstat` swap-in/swap-out counters during training; OOM killer invocations in `dmesg`                            |
| T3      | Significant time spent in data replication during initialization       | Wall-clock time for `System` struct broadcast + per-rank deserialization exceeds 5% of total initialization time |

Until at least one trigger fires on a representative production workload, per-rank heap replication is the correct trade-off: it is simpler, avoids shared memory lifecycle complexity, and imposes no correctness risk.

## Cross-References

- [Communication Patterns §1.1](./communication-patterns.md) -- Three collective operations per iteration, ferrompi API signatures, operation frequency
- [Communication Patterns §2](./communication-patterns.md) -- Data payloads for trial points (§2.1), cuts (§2.2), and convergence statistics (§2.3) that flow through the trait methods
- [Communication Patterns §5](./communication-patterns.md) -- `SharedWindow<T>` capabilities: window creation, intra-node grouping, read access, write synchronization
- [Communication Patterns §6](./communication-patterns.md) -- Deterministic communication invariant (§6.1 rank-ordered receives) and floating-point reduction tolerance (§6.2)
- [Shared Memory Aggregation §1.1](./shared-memory-aggregation.md) -- Leader allocation pattern, SharedWindow usage model (leader allocates, followers read)
- [Shared Memory Aggregation §1.2](./shared-memory-aggregation.md) -- Opening tree generation protocol: distributed population, fence synchronization
- [Shared Memory Aggregation §1.3](./shared-memory-aggregation.md) -- Shared input case data candidates and sizes
- [Shared Memory Aggregation §1.4](./shared-memory-aggregation.md) -- Cut pool as shared memory optimization candidate
- [Hybrid Parallelism §1.2](./hybrid-parallelism.md) -- ferrompi capabilities table: `Communicator` is `Send + Sync`, `SharedWindow<T>`, collectives API signatures
- [Hybrid Parallelism §1.3](./hybrid-parallelism.md) -- Shared memory layout: scenario storage and cut pool regions shared within a node
- [Hybrid Parallelism §1.0a](./hybrid-parallelism.md) -- Single-process mode: `SharedWindow<T>` not used, shared data allocated as regular heap memory
- [Hybrid Parallelism §6 Step 3](./hybrid-parallelism.md) -- Shared memory communicator creation via `split_shared()`
- [Memory Architecture §2.2](./memory-architecture.md) -- SharedWindow savings: per-node memory reduction from sharing opening tree and case data
- [Work Distribution §3.2](./work-distribution.md) -- MPI collective parameters: `sendcount`, `recvcounts`, `displs` computation from contiguous block assignment
- [Design Principles §5](../overview/design-principles.md) -- Rust implementation strategy, zero-cost abstractions, `unsafe` FFI boundary for MPI (ferrompi)
- [Synchronization §1.1](./synchronization.md) -- Three MPI sync points per iteration where allgatherv, allreduce, and barrier are invoked
- [Solver Abstraction §10](../architecture/solver-abstraction.md) -- Compile-time selection pattern via generic parameters and Cargo feature flags; the communicator trait follows the same design
