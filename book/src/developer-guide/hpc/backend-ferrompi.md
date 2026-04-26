# Ferrompi Backend

## Purpose

The ferrompi backend is the primary production communication backend for Cobre. It wraps the ferrompi MPI bindings behind the `Communicator` and `SharedMemoryProvider` traits defined in [Communicator Trait §1](./communicator-trait.md) and [Communicator Trait §4](./communicator-trait.md), providing zero-cost abstraction over MPI collectives and shared memory windows for HPC cluster deployments. As the reference backend, all other backends must match its observable behavior -- the ferrompi backend defines the canonical semantics that the local, TCP, and shared memory backends emulate. The architecture follows the compile-time selection pattern established in [Solver Abstraction §10](../architecture/solver-abstraction.md) and gated by the `mpi` feature flag as specified in [Backend Registration and Selection §1.2](./backend-selection.md).

## 1. Struct and Trait Implementation

### 1.1 Struct Definition

The `FerrompiBackend` struct holds the MPI communicator handles obtained during initialization. The world communicator is used for all inter-rank collective operations during training; the intra-node communicator is used for shared memory window creation and leader/follower determination.

```rust
/// Primary production communication backend wrapping ferrompi MPI bindings.
///
/// Delegates all `Communicator` trait methods directly to the corresponding
/// ferrompi API calls, achieving zero-cost abstraction via monomorphization
/// (see §4.1).
#[cfg(feature = "mpi")]
pub struct FerrompiBackend {
    // SAFETY: `mpi` must be declared before `world` and `shared` to ensure
    // Rust's reverse-declaration drop order drops the communicators before
    // the MPI guard. Dropping `mpi` calls `MPI_Finalize`, which must happen
    // only after all communicators and windows have been freed.
    /// MPI RAII guard. Its `Drop` calls `MPI_Finalize`.
    /// Must outlive all communicators and shared memory windows.
    mpi: ferrompi::Mpi,

    /// World communicator for all collective operations and rank/size queries.
    world: ferrompi::Communicator,

    /// Intra-node communicator created via `split_shared()`. Groups
    /// ranks on the same physical node for shared memory window operations.
    /// `None` only if initialization is incomplete (see §2).
    shared: Option<ferrompi::Communicator>,
}
```

### 1.2 Communicator Trait Implementation

The `impl Communicator for FerrompiBackend` delegates each trait method to the corresponding ferrompi API call. The method mapping table below summarizes the conversions.

```rust
#[cfg(feature = "mpi")]
impl Communicator for FerrompiBackend {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        // counts/displs are &[usize] in Cobre's trait but &[i32] in ferrompi;
        // conversion is performed by a helper (omitted for clarity).
        let (i32_counts, i32_displs) = to_i32_vecs(counts, displs);
        self.world
            .allgatherv(send, recv, &i32_counts, &i32_displs)
            .map_err(|e| map_ferrompi_error(e, "allgatherv"))
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError> {
        let mpi_op = match op {
            ReduceOp::Sum => ferrompi::ReduceOp::Sum,
            ReduceOp::Min => ferrompi::ReduceOp::Min,
            ReduceOp::Max => ferrompi::ReduceOp::Max,
        };
        self.world
            .allreduce(send, recv, mpi_op)
            .map_err(|e| map_ferrompi_error(e, "allreduce"))
    }

    fn broadcast<T: CommData>(
        &self,
        buf: &mut [T],
        root: usize,
    ) -> Result<(), CommError> {
        self.world
            .broadcast(buf, root as i32)
            .map_err(|e| map_ferrompi_error(e, "broadcast"))
    }

    fn barrier(&self) -> Result<(), CommError> {
        self.world
            .barrier()
            .map_err(|e| map_ferrompi_error(e, "barrier"))
    }

    fn rank(&self) -> usize {
        self.world.rank() as usize
    }

    fn size(&self) -> usize {
        self.world.size() as usize
    }
}
```

**Method mapping summary:**

| Trait Method | ferrompi API Call                                             | Type Conversion                                      |
| ------------ | ------------------------------------------------------------- | ---------------------------------------------------- |
| `allgatherv` | `self.world.allgatherv(send, recv, &i32_counts, &i32_displs)` | `counts`/`displs`: `&[usize]` to `&[i32]`            |
| `allreduce`  | `self.world.allreduce(send, recv, mpi_op)`                    | `ReduceOp` to `ferrompi::ReduceOp` (zero-cost match) |
| `broadcast`  | `self.world.broadcast(buf, root)`                             | `root: usize` to `i32`                               |
| `barrier`    | `self.world.barrier()`                                        | None                                                 |
| `rank`       | `self.world.rank()`                                           | `i32` to `usize`                                     |
| `size`       | `self.world.size()`                                           | `i32` to `usize`                                     |

**Error delegation:** All fallible methods convert `ferrompi::Error` to `CommError` via the `map_ferrompi_error` helper (see SS5.2). The helper extracts the MPI error code from the `Error::Mpi` variant and classifies errors into the most specific `CommError` variant. No new `CommError` variants are introduced -- all errors map to the variants defined in [Communicator Trait §1.4](./communicator-trait.md) and [Communicator Trait §4.6](./communicator-trait.md). See SS5 for the complete error mapping table.

**Precondition validation:** Buffer size preconditions ([Communicator Trait §2.1](./communicator-trait.md) -- §2.5) are enforced by the ferrompi layer, which validates arguments before issuing the MPI call. The `FerrompiBackend` does not duplicate these checks.

## 2. Initialization

The `FerrompiBackend::new()` constructor wraps the MPI initialization sequence defined in [Hybrid Parallelism §6](./hybrid-parallelism.md) Steps 1-3. It is called once during the Startup phase, before any collective operations or shared memory allocations.

### 2.1 Initialization Sequence

```rust
#[cfg(feature = "mpi")]
impl FerrompiBackend {
    /// Initialize the ferrompi backend (MPI startup sequence, Steps 1-3 of
    /// [Hybrid Parallelism §6](./hybrid-parallelism.md)).
    ///
    /// # Errors
    ///
    /// Returns `BackendError::InitializationFailed` if `MPI_Init_thread` fails
    /// or returns a threading level below `MPI_THREAD_FUNNELED`, or if the
    /// shared-memory communicator split fails.
    ///
    /// # Panics
    ///
    /// Panics if called more than once (MPI does not support multiple
    /// initializations).
    pub fn new() -> Result<Self, BackendError> {
        // Step 1 — MPI initialization (Hybrid Parallelism §6 Step 1)
        // MPI_THREAD_FUNNELED: only the main thread makes MPI calls.
        // The Cobre training loop serializes all MPI collective calls on
        // the main thread; Rayon worker threads never call MPI directly.
        // ferrompi::Mpi::init_thread returns the Mpi guard whose Drop
        // calls MPI_Finalize. The guard is stored in the backend struct
        // (see note below on lifetime management).
        let mpi = ferrompi::Mpi::init_thread(
            ferrompi::ThreadLevel::Funneled,
        ).map_err(|e| BackendError::InitializationFailed {
            backend: "mpi".to_string(),
            source: Box::new(e),
        })?;

        let world = mpi.world();

        // Step 2 — Topology detection (Hybrid Parallelism §6 Step 2)
        let _rank = world.rank();
        let _size = world.size();

        // Step 3 — Shared memory communicator (Hybrid Parallelism §6 Step 3)
        // split_shared() is a convenience wrapper for
        // MPI_Comm_split_type(MPI_COMM_TYPE_SHARED).
        let shared = world
            .split_shared()
            .map_err(|e| BackendError::InitializationFailed {
                backend: "mpi".to_string(),
                source: Box::new(e),
            })?;

        Ok(FerrompiBackend {
            mpi,
            world,
            shared: Some(shared),
        })
    }
}
```

### 2.2 Shutdown and Drop

The `FerrompiBackend` implements `Drop` to call `MPI_Finalize` when the backend is destroyed. All shared memory regions ([SharedRegion §3](#3-sharedwindow-implementation)) must be dropped before the backend is dropped, because `MPI_Win_free` must precede `MPI_Finalize`.

```rust
#[cfg(feature = "mpi")]
impl Drop for FerrompiBackend {
    fn drop(&mut self) {
        // Drop the intra-node communicator before MPI_Finalize.
        self.shared.take();
        // MPI_Finalize is called by ferrompi's Mpi drop.
    }
}
```

**Ordering constraint:** The training loop ensures the following drop order during shutdown:

1. All `SharedRegion<T>` handles (calls `MPI_Win_free` on each window)
2. The `FerrompiBackend` (calls `MPI_Finalize` via ferrompi's `Mpi` drop)

This ordering is consistent with the lifetime contract in [Communicator Trait §4.2](./communicator-trait.md).

## 3. SharedWindow Implementation

### 3.1 SharedMemoryProvider Trait Implementation

> **Status: Not Yet Implemented -- using HeapRegion fallback.** Per spec SS4.7 (Communicator Trait §4.7), true MPI shared windows (`MPI_Win_allocate_shared`) are deferred to post-profiling. The current implementation uses `HeapRegion<T>` (heap fallback) as the `Region` associated type for all backends, including `FerrompiBackend`. Each rank holds its own `Vec<T>` copy with no memory shared across ranks. The `FerrompiRegion<T>` / `SharedWindow<T>` design below is the aspirational target architecture that will be activated when production-scale profiling demonstrates memory pressure (see trigger conditions in Communicator Trait §4.7).

The target architecture has `FerrompiBackend` implementing `SharedMemoryProvider` by delegating to ferrompi's `SharedWindow<T>` for true intra-node shared memory. The `Region` associated type would wrap `ferrompi::SharedWindow<T>` in a `FerrompiRegion<T>` newtype that implements the `SharedRegion<T>` trait.

```rust
#[cfg(feature = "mpi")]
impl SharedMemoryProvider for FerrompiBackend {
    type Region<T: CommData> = FerrompiRegion<T>;

    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError> {
        let shared_comm = self.shared.as_ref().ok_or(
            CommError::InvalidCommunicator,
        )?;

        // The leader (local rank 0) allocates `count` elements.
        // Followers allocate size 0 and receive a handle to the
        // leader's memory via MPI window attachment.
        let alloc_count = if shared_comm.rank() == 0 {
            count
        } else {
            0
        };

        let window = ferrompi::SharedWindow::allocate(shared_comm, alloc_count)
            .map_err(|e| CommError::AllocationFailed {
                requested_bytes: count * std::mem::size_of::<T>(),
                message: e.to_string(),
            })?;

        Ok(FerrompiRegion {
            window,
            count,
        })
    }

    fn split_local(&self) -> Result<Box<dyn LocalCommunicator>, CommError> {
        // Issues MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) to obtain an
        // intra-node communicator, then wraps it in FerrompiLocalComm.
        // FerrompiLocalComm implements LocalCommunicator (rank, size, barrier)
        // but not the full generic Communicator trait.
        self.world
            .split_shared()
            .map(|c| Box::new(FerrompiLocalComm(c)) as Box<dyn LocalCommunicator>)
            .map_err(|e| map_ferrompi_error(&e, "split_local"))
    }

    fn is_leader(&self) -> bool {
        self.shared
            .as_ref()
            .map(|c| c.rank() == 0)
            .unwrap_or(true)
    }
}
```

**Leader determination:** The leader is the rank with local rank 0 within the intra-node communicator, consistent with the convention in [Hybrid Parallelism §6 Step 3](./hybrid-parallelism.md) and the leader/follower pattern in [Communicator Trait §4.3](./communicator-trait.md).

### 3.2 SharedRegion Wrapper (Aspirational -- Not Yet Implemented)

The `FerrompiRegion<T>` newtype would wrap `ferrompi::SharedWindow<T>` and implement the `SharedRegion<T>` trait. The wrapper would encapsulate the `unsafe` boundary for raw pointer dereference into MPI shared windows, presenting a safe Rust interface to the training loop. In the current codebase, `FerrompiBackend::Region<T>` is `HeapRegion<T>` and `create_shared_region` allocates via `vec![T::default(); count]`.

```rust
/// Shared memory region backed by an MPI window (`ferrompi::SharedWindow<T>`).
///
/// Lifecycle follows [Communicator Trait §4.2](./communicator-trait.md).
/// RAII: dropping calls `MPI_Win_free` on the underlying window.
#[cfg(feature = "mpi")]
pub struct FerrompiRegion<T: CommData> {
    /// The underlying MPI shared memory window.
    window: ferrompi::SharedWindow<T>,

    /// Logical element count (the `count` passed to `create_shared_region`;
    /// followers allocate 0 locally but reference the leader's memory).
    count: usize,
}
```

```rust
#[cfg(feature = "mpi")]
impl<T: CommData> SharedRegion<T> for FerrompiRegion<T> {
    fn as_slice(&self) -> &[T] {
        // All ranks (leader and followers) access the leader's (rank 0)
        // memory via remote_slice(0). This works uniformly: rank 0 gets
        // a view into its own allocation, other ranks get a view into the
        // shared window mapped from rank 0's region.
        // Caller must call fence() before reading to ensure no data races.
        self.window
            .remote_slice(0)
            .expect("rank 0 window region always valid")
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // Returns the caller's local allocation. For the leader (rank 0),
        // this is the full shared region; for followers, the local
        // allocation is size 0, so local_slice_mut() returns an empty slice.
        // Safety: &mut self prevents concurrent access at compile time.
        self.window.local_slice_mut()
    }

    fn fence(&self) -> Result<(), CommError> {
        // MPI_Win_fence: collective -- all ranks must call fence().
        self.window
            .fence()
            .map_err(|e| map_ferrompi_error(e, "fence"))
    }
}
```

```rust
#[cfg(feature = "mpi")]
impl<T: CommData> Drop for FerrompiRegion<T> {
    fn drop(&mut self) {
        // MPI_Win_free called via ferrompi::SharedWindow::drop().
        // Must complete before MPI_Finalize (see §2.2).
    }
}
```

## 4. Performance

### 4.1 Zero-Cost Abstraction via Monomorphization

When the training loop is generic over `C: Communicator` and the binary is built with only the `mpi` feature enabled, the compiler resolves `C` to `FerrompiBackend` at compile time. This enables:

1. **Full inlining** -- Each trait method call compiles to a direct call to the corresponding ferrompi function. The `match` on `ReduceOp` in `allreduce` is eliminated when the variant is known at the call site (constant propagation).

2. **No vtable, no indirection** -- Static dispatch means there is no trait object, no vtable lookup, and no dynamic dispatch overhead. The generated assembly for `train::<FerrompiBackend>(comm, ...)` is identical to code that calls `comm.world.allgatherv(...)` directly.

3. **Dead code elimination** -- In a single-feature build (`--features mpi` only), all `cfg`-gated code paths for other backends are removed from the binary. The `CommBackend` enum ([Backend Registration and Selection §4.2](./backend-selection.md)) is never instantiated.

This achieves the same zero-cost property as the solver abstraction pattern in [Solver Abstraction §10](../architecture/solver-abstraction.md): the abstraction layer has no runtime cost in production builds.

### 4.2 Persistent Collectives as Internal Optimization

MPI 4.0 persistent collectives ([Communication Patterns §4](./communication-patterns.md)) can be used as an internal optimization within the `FerrompiBackend` without affecting the `Communicator` trait interface. The trait methods (`allgatherv`, `allreduce`, `broadcast`, `barrier`) define the external API; the backend is free to implement them using either standard or persistent MPI collectives.

**Persistent collective strategy:**

| Trait Method | Persistent Candidate | Implementation Strategy                                                                                                               |
| ------------ | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `allgatherv` | Conditional          | Pre-initialize if buffer sizes are known at construction (fixed $M$); fall back to standard collective if counts change between calls |
| `allreduce`  | Yes                  | Pre-initialize with fixed buffer at construction for simulation-mode min/max; reuse every iteration                                   |
| `broadcast`  | No                   | Called only during initialization and LB distribution with varying buffer sizes; standard collective is sufficient                    |
| `barrier`    | No                   | Called only at checkpoints; standard collective is sufficient                                                                         |

**Internal state for persistent collectives:**

```rust
#[cfg(feature = "mpi")]
pub struct FerrompiBackend {
    world: ferrompi::Communicator,
    shared: Option<ferrompi::Communicator>,

    /// Pre-initialized persistent allreduce request for convergence
    /// statistics (4 x f64, fixed size). `None` if the MPI implementation
    /// does not support MPI 4.0 persistent collectives.
    persistent_allreduce: Option<ferrompi::PersistentRequest>,
}
```

The persistent collective initialization occurs during `FerrompiBackend::new()` if the MPI implementation supports it (detected via `MPI_T_get_info` or a runtime version check). If persistent collectives are unavailable, the backend falls back to standard collectives transparently -- the trait interface is unchanged.

**Key property:** Persistent collectives are an implementation detail of the ferrompi backend. They do not appear in the `Communicator` trait definition, do not affect the trait's method signatures, and do not alter the observable semantics of any collective operation. Other backends are unaffected by this optimization.

## 5. Error Mapping

The ferrompi backend maps MPI error codes to `CommError` variants defined in [Communicator Trait §1.4](./communicator-trait.md) and [Communicator Trait §4.6](./communicator-trait.md). No new `CommError` variants are introduced by this backend.

### 5.1 MPI Error Code to CommError Mapping

| MPI Error Code   | Typical Cause                                           | CommError Variant                                                    |
| ---------------- | ------------------------------------------------------- | -------------------------------------------------------------------- |
| `MPI_ERR_COMM`   | Invalid or finalized communicator handle                | `CommError::InvalidCommunicator`                                     |
| `MPI_ERR_BUFFER` | Invalid buffer pointer (null, misaligned)               | `CommError::InvalidBufferSize { operation, expected, actual }`       |
| `MPI_ERR_COUNT`  | Negative count or count/displacement mismatch           | `CommError::InvalidBufferSize { operation, expected, actual }`       |
| `MPI_ERR_TYPE`   | Unsupported or uncommitted MPI datatype                 | `CommError::CollectiveFailed { operation, mpi_error_code, message }` |
| `MPI_ERR_ROOT`   | Root rank out of range in broadcast                     | `CommError::InvalidRoot { root, size }`                              |
| `MPI_ERR_WIN`    | Invalid MPI window in shared memory operation           | `CommError::CollectiveFailed { operation, mpi_error_code, message }` |
| `MPI_ERR_NO_MEM` | Shared memory allocation rejected by OS/MPI             | `CommError::AllocationFailed { requested_bytes, message }`           |
| `MPI_ERR_OTHER`  | Unclassified MPI error (process crash, network failure) | `CommError::CollectiveFailed { operation, mpi_error_code, message }` |

### 5.2 Error Conversion Implementation

The `map_ferrompi_error` helper converts a `ferrompi::Error` to the most specific `CommError` variant. The real `ferrompi::Error` is a `thiserror`-derived enum (not a struct), so the conversion pattern-matches on the variant. Only the `Error::Mpi` variant carries an MPI error class and code; the other variants (`AlreadyInitialized`, `InvalidBuffer`, `NotSupported`, `Internal`) map to fixed `CommError` variants.

```rust
/// Convert a ferrompi::Error to the most specific CommError variant.
/// Used by all Communicator and SharedRegion trait method implementations.
#[cfg(feature = "mpi")]
fn map_ferrompi_error(e: ferrompi::Error, operation: &'static str) -> CommError {
    match e {
        ferrompi::Error::Mpi { class, code, ref message } => match class {
            ferrompi::MpiErrorClass::Comm => CommError::InvalidCommunicator,
            ferrompi::MpiErrorClass::Root => CommError::InvalidRoot {
                // ferrompi::Error does not carry root/size context;
                // use sentinel values. The message string provides details.
                root: 0,
                size: 0,
            },
            ferrompi::MpiErrorClass::Buffer | ferrompi::MpiErrorClass::Count => {
                CommError::InvalidBufferSize {
                    operation,
                    // ferrompi::Error does not carry expected/actual counts;
                    // use sentinel values. The message string provides details.
                    expected: 0,
                    actual: 0,
                }
            }
            // MPI_ERR_NO_MEM maps to AllocationFailed for shared memory ops.
            // The MpiErrorClass enum does not have a NoMem variant; this
            // condition surfaces as MpiErrorClass::Other or MpiErrorClass::Raw(_).
            // In practice, window allocation failures are rare and the fallback
            // to CollectiveFailed is acceptable.
            _ => CommError::CollectiveFailed {
                operation,
                mpi_error_code: code,
                message: message.clone(),
            },
        },
        ferrompi::Error::InvalidBuffer => CommError::InvalidBufferSize {
            operation,
            expected: 0,
            actual: 0,
        },
        ferrompi::Error::AlreadyInitialized => CommError::InvalidCommunicator,
        _ => CommError::CollectiveFailed {
            operation,
            mpi_error_code: -1,
            message: e.to_string(),
        },
    }
}
```

**Design note:** The real `ferrompi::Error` enum does not carry structured context fields (root rank, buffer sizes, requested bytes) that the speculative API assumed. The `CommError` fields that require these values (`InvalidRoot.root`, `InvalidBufferSize.expected`, etc.) are populated with sentinel values (0); the human-readable message from ferrompi provides the diagnostic detail. This is an acceptable trade-off because these error paths are not performance-critical and the message string is always available for logging.

### 5.3 MPI_Init_thread Failure

If `ferrompi::Mpi::init_thread(ThreadLevel::Funneled)` fails -- either because `MPI_Init_thread` returns an error or because the provided threading level is below `MPI_THREAD_FUNNELED` -- the error is reported as `BackendError::InitializationFailed` (defined in [Backend Registration and Selection §6.2](./backend-selection.md)), not as a `CommError`. This is because initialization failure is a factory-level concern that occurs before any `Communicator` trait method can be called.

## 6. Feature Gating

### 6.1 Conditional Compilation

The entire `FerrompiBackend` implementation is gated behind the `mpi` Cargo feature flag, consistent with the feature flag matrix in [Backend Registration and Selection §1.2](./backend-selection.md).

```rust
#[cfg(feature = "mpi")]
mod ferrompi_backend {
    use crate::comm::{
        CommData, CommError, Communicator, ReduceOp,
        SharedMemoryProvider, SharedRegion,
    };

    pub struct FerrompiBackend { /* ... */ }
    pub struct FerrompiRegion<T: CommData> { /* ... */ }

    impl Communicator for FerrompiBackend { /* ... */ }
    impl SharedMemoryProvider for FerrompiBackend { /* ... */ }
    impl<T: CommData> SharedRegion<T> for FerrompiRegion<T> { /* ... */ }
}

#[cfg(feature = "mpi")]
pub use ferrompi_backend::{FerrompiBackend, FerrompiRegion};
```

### 6.2 Cargo.toml Feature Declaration

```toml
[features]
default = []
mpi = ["dep:ferrompi"]

[dependencies]
ferrompi = { version = "0.2.2", optional = true }
```

When `mpi` is not enabled:

- The `ferrompi` crate is not compiled or linked.
- No MPI development libraries (OpenMPI, MPICH, Intel MPI) are required at build time.
- The `FerrompiBackend` and `FerrompiRegion` types do not exist -- any attempt to reference them produces a Rust compilation error.
- The `create_communicator()` factory function ([Backend Registration and Selection §4.1](./backend-selection.md)) does not include an MPI branch.

When `mpi` is enabled:

- The `ferrompi` crate is compiled and linked against the system MPI library (`libmpi.so` or equivalent).
- All `#[cfg(feature = "mpi")]` items are included in the binary.
- The MPI backend is available for selection via the factory function or `COBRE_COMM_BACKEND=mpi`.

### 6.3 Build Profile Integration

The ferrompi backend is included in the following build profiles from [Backend Registration and Selection §1.3](./backend-selection.md):

| Build Profile | Includes `mpi`? | Rationale                                                        |
| ------------- | :-------------: | ---------------------------------------------------------------- |
| CLI / HPC     |       Yes       | MPI is the production communication layer for cluster deployment |
| Python wheel  |       No        | No MPI dependency on user machines                               |
| Test / CI     |       No        | Only `local` backend; no external dependencies                   |
| Development   |       Yes       | All backends compiled for testing                                |

## 7. ferrompi API Reference

This section documents the public API surface of the **ferrompi** crate (version 0.2.2, [github.com/rjmalves/ferrompi](https://github.com/rjmalves/ferrompi)) -- a safe Rust wrapper around the MPI C library. ferrompi encapsulates all `unsafe` FFI calls internally; no public method requires the Rust `unsafe` keyword at the call site. The API documented here is the contract that `FerrompiBackend` (SS1--SS6) implements against.

> **Note.** ferrompi is a thin wrapper around the MPI C functions (`MPI_Init_thread`, `MPI_Comm_rank`, `MPI_Allgatherv`, `MPI_Win_allocate_shared`, etc.). It adds Rust type safety, RAII resource management, and `Result`-based error handling, but does not alter MPI semantics. All MPI behavioral guarantees (ordering, collective synchronization, shared memory coherence) are inherited from the underlying MPI implementation.

### 7.1 `Mpi` Struct and Initialization

The `Mpi` struct is the RAII guard for the MPI runtime lifetime. Creating an `Mpi` instance calls `MPI_Init_thread`; dropping it calls `MPI_Finalize`. Exactly one `Mpi` instance may exist per process.

#### 7.1.1 `Mpi::init`

Initializes the MPI runtime with the default threading level (`MPI_THREAD_SINGLE`).

```rust
impl Mpi {
    pub fn init() -> Result<Self>
}
```

**Preconditions:**

| Condition           | Description                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| First and only call | Must not have been called previously in this process. MPI does not support multiple initializations.    |
| No prior `MPI_Init` | No other code (including C libraries) may have called `MPI_Init` or `MPI_Init_thread` before this call. |

**Postconditions (on `Ok`):**

| Condition                         | Description                                                |
| --------------------------------- | ---------------------------------------------------------- |
| MPI runtime is initialized        | `MPI_Init_thread` has been called.                         |
| Returned `Mpi` holds MPI lifetime | The MPI runtime remains active until the `Mpi` is dropped. |

**Error type:** `ferrompi::Error::AlreadyInitialized` if called more than once. `ferrompi::Error::Mpi { .. }` if `MPI_Init_thread` fails.

**Unsafe:** No. The internal FFI call to `MPI_Init_thread` is encapsulated.

#### 7.1.2 `Mpi::init_thread`

Initializes the MPI runtime with the requested threading support level. This is the initialization function used by Cobre (see SS2.1).

```rust
impl Mpi {
    pub fn init_thread(required: ThreadLevel) -> Result<Self>
}
```

**Preconditions:**

| Condition           | Description                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| First and only call | Must not have been called previously in this process. MPI does not support multiple initializations.    |
| No prior `MPI_Init` | No other code (including C libraries) may have called `MPI_Init` or `MPI_Init_thread` before this call. |

**Postconditions (on `Ok`):**

| Condition                         | Description                                                                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| MPI runtime is initialized        | `MPI_Init_thread` has been called with the requested level.                                                                                           |
| Returned `Mpi` holds MPI lifetime | The MPI runtime remains active until the `Mpi` is dropped.                                                                                            |
| Provided level is satisfied       | The MPI implementation supports at least the requested `ThreadLevel`. If the provided level is below the requested level, the function returns `Err`. |

**Error type:** `ferrompi::Error::AlreadyInitialized` if called more than once. `ferrompi::Error::Mpi { .. }` if `MPI_Init_thread` fails or returns a level below the requested one.

**Unsafe:** No. The internal FFI call to `MPI_Init_thread` is encapsulated.

Cobre requests `ThreadLevel::Funneled` (see SS2.1) because only the main thread makes MPI calls; Rayon worker threads perform LP solves but never invoke MPI collectives directly.

#### 7.1.3 `Mpi::world`

Returns the world communicator (`MPI_COMM_WORLD`).

```rust
impl Mpi {
    pub fn world(&self) -> Communicator
}
```

**Preconditions:** None beyond holding a valid `Mpi` (guaranteed by successful `init` or `init_thread`).

**Postconditions:**

| Condition                                       | Description                                               |
| ----------------------------------------------- | --------------------------------------------------------- |
| Returned `Communicator` wraps `MPI_COMM_WORLD`  | Rank and size reflect the full set of MPI processes.      |
| Communicator is valid for the lifetime of `Mpi` | The communicator must not be used after `Mpi` is dropped. |

**Error type:** Infallible (does not return `Result`).

**Unsafe:** No.

#### 7.1.4 Other `Mpi` Methods

| Method           | Signature                               | Description                                                     |
| ---------------- | --------------------------------------- | --------------------------------------------------------------- |
| `thread_level`   | `fn thread_level(&self) -> ThreadLevel` | Returns the threading level provided by the MPI implementation. |
| `wtime`          | `fn wtime() -> f64`                     | Returns wall-clock time in seconds (wraps `MPI_Wtime`). Static. |
| `version`        | `fn version() -> Result<String>`        | Returns the MPI library version string.                         |
| `is_initialized` | `fn is_initialized() -> bool`           | Returns `true` if MPI has been initialized. Static.             |
| `is_finalized`   | `fn is_finalized() -> bool`             | Returns `true` if MPI has been finalized. Static.               |

**Drop:** Dropping `Mpi` calls `MPI_Finalize`. All `SharedWindow` instances and derived communicators must be dropped before the `Mpi` guard (see SS2.2).

### 7.2 Communicator

The `Communicator` type is a safe handle to an MPI communicator. It implements `Send + Sync`, enabling shared use across threads. Under `MPI_THREAD_FUNNELED`, only the main thread makes MPI calls; `Send + Sync` on the handle is still sound because it allows safe passing between threads without implying concurrent MPI invocation.

```rust
pub struct Communicator { /* opaque: wraps MPI_Comm handle */ }

// Thread safety: Communicator is Send + Sync.
// Under MPI_THREAD_FUNNELED, only the main thread issues MPI calls.
// Send + Sync on the handle is sound because the handle is an integer
// into a C-side table; ferrompi::Communicator carries no thread-local state.
```

The following subsections document in detail the methods used by the Cobre backend (SS1--SS3). Section 7.2.8 provides a summary table of the remaining methods.

#### 7.2.1 `rank`

Returns the rank of the calling process within this communicator.

```rust
impl Communicator {
    pub fn rank(&self) -> i32
}
```

**Preconditions:** None.

**Postconditions:** Returns a value in `[0, self.size())`.

**Error type:** Infallible. Wraps `MPI_Comm_rank`, which does not fail on a valid communicator.

**Unsafe:** No.

#### 7.2.2 `size`

Returns the number of processes in this communicator.

```rust
impl Communicator {
    pub fn size(&self) -> i32
}
```

**Preconditions:** None.

**Postconditions:** Returns a value `>= 1`.

**Error type:** Infallible. Wraps `MPI_Comm_size`.

**Unsafe:** No.

#### 7.2.3 `allgatherv`

Gathers variable-length data from all ranks and distributes the concatenated result to all ranks.

```rust
impl Communicator {
    pub fn allgatherv<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        recvcounts: &[i32],
        displs: &[i32],
    ) -> Result<()>
}
```

**Preconditions:**

| Condition                                             | Description                                                                                                                  |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `recvcounts.len() == self.size()`                     | One count per rank.                                                                                                          |
| `displs.len() == self.size()`                         | One displacement per rank.                                                                                                   |
| `send.len() == recvcounts[self.rank()]`               | Send buffer length matches this rank's declared count.                                                                       |
| `recv.len() >= displs[i] + recvcounts[i]` for all `i` | Receive buffer is large enough for all incoming data.                                                                        |
| All ranks call collectively                           | `allgatherv` is a collective operation; all ranks in the communicator must call it with consistent counts and displacements. |

**Postconditions (on `Ok`):**

| Condition                     | Description                                                             |
| ----------------------------- | ----------------------------------------------------------------------- |
| `recv` contains gathered data | `recv[displs[i]..displs[i]+recvcounts[i]]` holds data sent by rank `i`. |

**Error type:** `ferrompi::Error`. Common: `Error::Mpi` with class `Buffer`, `Count`, or `Comm`.

**Unsafe:** No.

#### 7.2.4 `allreduce`

Performs an element-wise reduction across all ranks and distributes the result to all ranks.

```rust
impl Communicator {
    pub fn allreduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<()>
}
```

**Preconditions:**

| Condition                   | Description                                                        |
| --------------------------- | ------------------------------------------------------------------ |
| `send.len() == recv.len()`  | Send and receive buffers have the same length.                     |
| All ranks call collectively | All ranks must call with the same `op` and the same element count. |

**Postconditions (on `Ok`):**

| Condition                                                | Description                                                             |
| -------------------------------------------------------- | ----------------------------------------------------------------------- |
| `recv[i]` is the reduction of `send[i]` across all ranks | The reduction operation is determined by `op` (Sum, Min, Max, or Prod). |

**Error type:** `ferrompi::Error`. Common: `Error::Mpi` with class `Buffer`, `Count`, `Comm`, or `Type`.

**Unsafe:** No.

#### 7.2.5 `broadcast`

Broadcasts data from the root rank to all other ranks.

```rust
impl Communicator {
    pub fn broadcast<T: MpiDatatype>(
        &self,
        data: &mut [T],
        root: i32,
    ) -> Result<()>
}
```

**Preconditions:**

| Condition                                      | Description                                                          |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `0 <= root < self.size()`                      | Root rank is valid within this communicator.                         |
| All ranks call collectively                    | All ranks must call with the same `root` and the same buffer length. |
| On root: `data` contains the data to broadcast | On non-root ranks: `data` content is overwritten.                    |

**Postconditions (on `Ok`):**

| Condition                            | Description                                                                                        |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| All ranks hold root's data in `data` | After return, `data` on every rank contains the data that was in `data` on the root rank at entry. |

**Error type:** `ferrompi::Error`. Common: `Error::Mpi` with class `Root`, `Buffer`, `Count`, `Comm`.

**Unsafe:** No.

#### 7.2.6 `barrier`

Synchronizes all ranks in the communicator. No rank returns until all ranks have entered the barrier.

```rust
impl Communicator {
    pub fn barrier(&self) -> Result<()>
}
```

**Preconditions:**

| Condition                   | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| All ranks call collectively | Deadlock occurs if any rank does not call `barrier`. |

**Postconditions (on `Ok`):**

| Condition                          | Description                   |
| ---------------------------------- | ----------------------------- |
| All ranks have reached the barrier | Global synchronization point. |

**Error type:** `ferrompi::Error`. Common: `Error::Mpi` with class `Comm`.

**Unsafe:** No.

#### 7.2.7 `split_shared`

Creates a sub-communicator grouping ranks that share a physical node (i.e., can use shared memory). Convenience wrapper for `MPI_Comm_split_type` with `MPI_COMM_TYPE_SHARED`.

```rust
impl Communicator {
    pub fn split_shared(&self) -> Result<Communicator>
}
```

**Preconditions:**

| Condition                   | Description                                     |
| --------------------------- | ----------------------------------------------- |
| All ranks call collectively | All ranks in the communicator must participate. |

**Postconditions (on `Ok`):**

| Condition                                         | Description                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Returned communicator groups co-located ranks     | All ranks on the same physical node are in the same sub-communicator.                 |
| Rank 0 in the sub-communicator is the node leader | By MPI convention, rank ordering within the split follows the original rank ordering. |

**Error type:** `ferrompi::Error`. Common: `Error::Mpi` with class `Comm`.

**Unsafe:** No.

**Note:** The more general `split_type` method is also available, accepting a `SplitType` enum parameter. `split_shared()` is equivalent to `split_type(SplitType::Shared, 0)`.

#### 7.2.8 `allreduce_init` (Persistent Collective)

Pre-initializes an allreduce operation for repeated execution with the same parameters (MPI 4.0+). Used by the persistent collective optimization in SS4.2.

```rust
impl Communicator {
    pub fn allreduce_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest>
}
```

**Preconditions:** Same as `allreduce` (SS7.2.4), plus the MPI implementation must support MPI 4.0 persistent collectives.

**Postconditions (on `Ok`):** Returns a `PersistentRequest` that can be started repeatedly via `start()` and completed via `wait()`. Each start/wait cycle performs one allreduce with the pre-bound parameters.

**Error type:** `ferrompi::Error`. `Error::NotSupported` if persistent collectives are unavailable.

**Unsafe:** No.

#### 7.2.9 Additional Communicator Methods (Summary)

The following methods are available on `Communicator` but are not directly used by the Cobre backend. They are listed for completeness.

**Management:**

| Method           | Signature                                                                               | Description                                        |
| ---------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `duplicate`      | `fn duplicate(&self) -> Result<Self>`                                                   | Duplicates the communicator.                       |
| `split`          | `fn split(&self, color: i32, key: i32) -> Result<Option<Communicator>>`                 | Splits by color/key.                               |
| `split_type`     | `fn split_type(&self, split_type: SplitType, key: i32) -> Result<Option<Communicator>>` | Splits by type (generalization of `split_shared`). |
| `processor_name` | `fn processor_name(&self) -> Result<String>`                                            | Returns the MPI processor name.                    |
| `raw_handle`     | `fn raw_handle(&self) -> i32`                                                           | Returns the raw `MPI_Comm` handle.                 |

**Point-to-Point (blocking):**

| Method | Signature                                                                                          | Description                                     |
| ------ | -------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `send` | `fn send<T: MpiDatatype>(&self, data: &[T], dest: i32, tag: i32) -> Result<()>`                    | Blocking send.                                  |
| `recv` | `fn recv<T: MpiDatatype>(&self, data: &mut [T], source: i32, tag: i32) -> Result<(i32, i32, i64)>` | Blocking receive; returns (source, tag, count). |

**Point-to-Point (nonblocking):**

| Method  | Signature                                                                                   | Description                          |
| ------- | ------------------------------------------------------------------------------------------- | ------------------------------------ |
| `isend` | `fn isend<T: MpiDatatype>(&self, data: &[T], dest: i32, tag: i32) -> Result<Request>`       | Nonblocking send; returns `Request`. |
| `irecv` | `fn irecv<T: MpiDatatype>(&self, data: &mut [T], source: i32, tag: i32) -> Result<Request>` | Nonblocking receive.                 |

**Blocking Collectives (beyond Cobre-used):**

| Method                   | Description                                   |
| ------------------------ | --------------------------------------------- |
| `allgather`              | Fixed-count gather-to-all.                    |
| `reduce`                 | Reduce to root rank only.                     |
| `gather` / `gatherv`     | Gather to root (fixed / variable count).      |
| `scatter` / `scatterv`   | Scatter from root (fixed / variable count).   |
| `alltoall` / `alltoallv` | All-to-all exchange (fixed / variable count). |
| `scan` / `exscan`        | Inclusive / exclusive prefix scan.            |
| `reduce_scatter_block`   | Reduce-scatter with equal block sizes.        |

**Scalar Variants:** `allreduce_scalar`, `reduce_scalar`, `scan_scalar`, `exscan_scalar` -- convenience wrappers for single-element operations.

**In-Place Variants:** `allreduce_inplace`, `reduce_inplace` -- use `MPI_IN_PLACE` to avoid separate send/recv buffers.

**Nonblocking Collectives:** `iallreduce`, `iallgather`, `iallgatherv`, `ibroadcast`, `ibarrier`, etc. -- all blocking collectives have `i`-prefixed nonblocking variants returning `Request`.

**Persistent Collectives (MPI 4.0+):** `allreduce_init`, `broadcast_init`, `allgather_init`, `allgatherv_init`, etc. -- all blocking collectives have `_init`-suffixed persistent variants returning `PersistentRequest`.

### 7.3 `SharedWindow<T>` and RMA Types

`SharedWindow<T>` provides access to MPI-3 shared memory windows (`MPI_Win_allocate_shared`). Requires the `rma` Cargo feature. All ranks in the communicator share a contiguous memory region; each rank specifies a local allocation count.

```rust
pub struct SharedWindow<T: MpiDatatype> {
    // Opaque: wraps MPI_Win handle and local/remote memory pointers.
    // The type parameter T determines the element type and alignment.
}
```

#### 7.3.1 `SharedWindow::allocate`

Allocates a shared memory window. Each rank specifies its local element count. For the Cobre use case, rank 0 allocates the full region and other ranks allocate 0.

```rust
impl<T: MpiDatatype> SharedWindow<T> {
    pub fn allocate(comm: &Communicator, local_count: usize) -> Result<Self>
}
```

**Preconditions:**

| Condition                                        | Description                                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `comm` is a shared-memory communicator           | Typically obtained from `split_shared()`. Using a non-shared communicator is undefined behavior in MPI. |
| All ranks call collectively                      | All ranks must call `allocate` (each with its own `local_count`).                                       |
| `local_count * size_of::<T>()` does not overflow | Total byte size per rank fits in `MPI_Aint`.                                                            |

**Postconditions (on `Ok`):**

| Condition                          | Description                                                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Shared memory region is allocated  | Each rank owns `local_count * size_of::<T>()` bytes. Rank 0 typically owns the full region.                    |
| All ranks can access remote memory | Via `remote_slice(rank)` (subject to synchronization).                                                         |
| Memory is uninitialized            | Caller must write initial values (via leader's `local_slice_mut`) and call `fence()` before readers access it. |

**Error type:** `ferrompi::Error`. `Error::Mpi` with class `Win` or allocation-related classes.

**Unsafe:** No. The internal `MPI_Win_allocate_shared` FFI call and raw pointer management are encapsulated.

#### 7.3.2 `local_slice` / `local_slice_mut`

Access the calling rank's own portion of the shared memory window.

```rust
impl<T: MpiDatatype> SharedWindow<T> {
    pub fn local_slice(&self) -> &[T]
    pub fn local_slice_mut(&mut self) -> &mut [T]
}
```

**Preconditions:**

| Condition                                  | Description                                                                  |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| For `local_slice_mut`: no concurrent reads | Rust's `&mut self` borrow checker enforces exclusive access at compile time. |

**Postconditions:**

| Condition                                          | Description                                                                                                 |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Returns slice over the caller's local allocation   | For the leader (who allocated `count` elements): full region. For followers (who allocated 0): empty slice. |
| Exclusive access via `&mut self` (mutable variant) | Rust's borrow checker prevents concurrent `local_slice` and `local_slice_mut` calls.                        |

**Error type:** Infallible.

**Unsafe:** No. The `&mut self` receiver enforces exclusive access at compile time.

#### 7.3.3 `remote_slice`

Returns a shared reference to the memory region owned by the specified rank.

```rust
impl<T: MpiDatatype> SharedWindow<T> {
    pub fn remote_slice(&self, rank: i32) -> Result<&[T]>
}
```

**Preconditions:**

| Condition                                              | Description                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------ |
| `0 <= rank < self.comm_size()`                         | Target rank is valid within the window's communicator.             |
| No rank is concurrently writing to the target's region | Caller must ensure a `fence()` has completed since the last write. |

**Postconditions:**

| Condition                                  | Description                                                       |
| ------------------------------------------ | ----------------------------------------------------------------- |
| Returns `&[T]` pointing to `rank`'s memory | Length equals the `local_count` that `rank` passed to `allocate`. |

This is the primary read method for the Cobre shared memory pattern: all ranks call `remote_slice(0)` to access the leader's (rank 0) memory region.

**Logical safety contract:** Same as SS7.3.4 below -- the caller must ensure that a `fence()` has been called after the last write and before any `remote_slice` call. Violating this produces **unspecified values** but not undefined behavior in the Rust sense.

**Error type:** `ferrompi::Error` if `rank` is out of range.

**Unsafe:** No (see logical safety contract above).

#### 7.3.4 `fence`

Collective synchronization on the shared memory window. All ranks in the window's communicator must call `fence()`. Completes all pending RMA operations and establishes a memory barrier.

```rust
impl<T: MpiDatatype> SharedWindow<T> {
    pub fn fence(&self) -> Result<()>
}
```

**Preconditions:**

| Condition                   | Description                                        |
| --------------------------- | -------------------------------------------------- |
| All ranks call collectively | Deadlock occurs if any rank does not call `fence`. |

**Postconditions (on `Ok`):**

| Condition                                 | Description                                                        |
| ----------------------------------------- | ------------------------------------------------------------------ |
| All prior writes are visible to all ranks | Memory consistency is established across the shared memory region. |
| Safe to read via `remote_slice`           | Until the next write + `fence` cycle.                              |

**Error type:** `ferrompi::Error`. Common: `Error::Mpi` with class `Win`.

**Unsafe:** No.

#### 7.3.5 `lock` / `lock_all`

Passive-target synchronization for fine-grained access control, as an alternative to `fence()`.

```rust
impl<T: MpiDatatype> SharedWindow<T> {
    pub fn lock(&self, lock_type: LockType, rank: i32) -> Result<LockGuard<'_, T>>
    pub fn lock_all(&self) -> Result<LockAllGuard<'_, T>>
}
```

`LockGuard` and `LockAllGuard` are RAII guards that call `MPI_Win_unlock` / `MPI_Win_unlock_all` on drop. `LockType` is `Exclusive` or `Shared`. These methods are not used by the Cobre backend (which uses the simpler `fence()` pattern) but are available for advanced use cases.

#### 7.3.6 Other `SharedWindow` Methods

| Method       | Signature                     | Description                                    |
| ------------ | ----------------------------- | ---------------------------------------------- |
| `raw_handle` | `fn raw_handle(&self) -> i32` | Returns the raw `MPI_Win` handle.              |
| `comm_size`  | `fn comm_size(&self) -> i32`  | Returns the size of the window's communicator. |

#### 7.3.7 Drop

`SharedWindow<T>` implements `Drop` to call `MPI_Win_free` on the underlying MPI window handle.

```rust
impl<T: MpiDatatype> Drop for SharedWindow<T> {
    fn drop(&mut self) {
        // Calls MPI_Win_free. Must complete before MPI_Finalize (Mpi drop).
    }
}
```

**Ordering constraint:** All `SharedWindow` instances must be dropped before the `Mpi` guard is dropped. Dropping a `SharedWindow` after `MPI_Finalize` is undefined behavior in MPI. The Cobre training loop enforces this ordering by dropping shared regions before the `FerrompiBackend` (see SS2.2).

### 7.4 Supporting Types

#### 7.4.1 `ThreadLevel`

Requested MPI threading support level, passed to `Mpi::init_thread`. Maps directly to the MPI constants with explicit `#[repr(i32)]` discriminants.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum ThreadLevel {
    /// `MPI_THREAD_SINGLE` -- Only one thread will execute.
    Single = 0,

    /// `MPI_THREAD_FUNNELED` -- Only the main thread will make MPI calls.
    Funneled = 1,

    /// `MPI_THREAD_SERIALIZED` -- Only one thread at a time will make MPI calls.
    Serialized = 2,

    /// `MPI_THREAD_MULTIPLE` -- Any thread may make MPI calls at any time.
    Multiple = 3,
}
```

The variants are ordered by increasing capability: `Single < Funneled < Serialized < Multiple`. The `Ord` implementation reflects this ordering, so `Mpi::init_thread` can compare the requested level against the provided level using standard comparison operators.

Cobre requests `ThreadLevel::Funneled` (see SS2.1) because only the main thread makes MPI calls; Rayon worker threads perform LP solves but never invoke MPI collectives directly.

#### 7.4.2 `ReduceOp`

Reduction operation for `allreduce` and other reduction collectives. Maps directly to MPI predefined operations.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    /// `MPI_SUM` -- Element-wise sum.
    Sum = 0,

    /// `MPI_MIN` -- Element-wise minimum.
    Min = 2,

    /// `MPI_MAX` -- Element-wise maximum.
    Max = 1,

    /// `MPI_PROD` -- Element-wise product.
    Prod = 3,
}
```

Cobre uses `Sum` and `Min` (mapped from `cobre_comm::ReduceOp`; see SS1.2).

#### 7.4.3 `MpiDatatype` (Sealed Trait)

Sealed marker trait for types that can be transmitted via MPI. A type implementing `MpiDatatype` has a corresponding MPI datatype tag and a fixed in-memory representation suitable for direct byte transmission. The trait cannot be implemented by downstream crates (sealed).

```rust
/// Sealed marker trait for MPI-transmissible types.
///
/// The trait is sealed: only types with implementations in ferrompi
/// can satisfy the bound. This prevents unsound transmissions of
/// types whose layout does not match an MPI datatype.
pub trait MpiDatatype: sealed::Sealed + Copy + Send + 'static {
    /// Returns the datatype tag identifying the MPI type.
    const TAG: DatatypeTag;
}
```

**Built-in implementations:**

| Rust Type | `DatatypeTag` Variant | MPI Datatype             |
| --------- | --------------------- | ------------------------ |
| `f32`     | `DatatypeTag::F32`    | `MPI_FLOAT`              |
| `f64`     | `DatatypeTag::F64`    | `MPI_DOUBLE`             |
| `i32`     | `DatatypeTag::I32`    | `MPI_INT`                |
| `i64`     | `DatatypeTag::I64`    | `MPI_LONG_LONG`          |
| `u8`      | `DatatypeTag::U8`     | `MPI_UNSIGNED_CHAR`      |
| `u32`     | `DatatypeTag::U32`    | `MPI_UNSIGNED`           |
| `u64`     | `DatatypeTag::U64`    | `MPI_UNSIGNED_LONG_LONG` |

The `MpiDatatype` bound on all generic ferrompi methods (`allgatherv`, `allreduce`, `broadcast`, `SharedWindow::allocate`) ensures that only MPI-compatible types can be transmitted. The `Copy` supertrait guarantees that the type has no drop glue and can be safely `memcpy`'d, which matches MPI's byte-oriented transmission model.

**Relationship to Cobre's `CommData`:** Cobre's `CommData` trait ([Communicator Trait §1.2](./communicator-trait.md)) is a blanket `Copy + Send + Sync + 'static` trait. The `FerrompiBackend` constrains `T: CommData` at the trait level, but the ferrompi calls require `T: MpiDatatype`. Since all `MpiDatatype` implementors satisfy `CommData` (they are all `Copy + Send + Sync + 'static`), the backend's generic bounds are compatible. The Cobre training loop only transmits `f64`, `u8`, and `u32` -- all of which implement `MpiDatatype`.

#### 7.4.4 `DatatypeTag`

Discriminant enum identifying MPI datatypes. Used as the associated constant in `MpiDatatype::TAG`.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DatatypeTag {
    F32 = 0, F64 = 1, I32 = 2, I64 = 3, U8 = 4, U32 = 5, U64 = 6,
}
```

#### 7.4.5 `Error`

Error type returned by all fallible ferrompi operations. A `thiserror`-derived enum with variants for different failure modes.

```rust
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// MPI has already been initialized (double init).
    #[error("MPI has already been initialized")]
    AlreadyInitialized,

    /// An MPI library-level error with classified error class.
    #[error("MPI error: {message} (class={class}, code={code})")]
    Mpi {
        class: MpiErrorClass,
        code: i32,
        message: String,
    },

    /// Invalid buffer argument (null, misaligned, or wrong size).
    #[error("Invalid buffer")]
    InvalidBuffer,

    /// The requested operation is not supported by this MPI implementation.
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// An internal error in ferrompi (should not occur in normal use).
    #[error("Internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Construct an Error from a raw MPI error code.
    pub fn from_code(code: i32) -> Self;

    /// Check a raw MPI error code; return Ok(()) for MPI_SUCCESS.
    pub fn check(code: i32) -> Result<()>;
}

/// Convenience type alias used throughout ferrompi.
pub type Result<T> = std::result::Result<T, Error>;
```

The `Error::Mpi` variant carries the classified `MpiErrorClass`, the raw integer code, and a human-readable message obtained from `MPI_Error_string`. The `from_code` and `check` constructors are used internally by ferrompi to convert raw MPI return codes.

#### 7.4.6 `MpiErrorClass`

Classification of MPI error codes. Maps MPI error classes to Rust enum variants for pattern matching in error conversion logic (see SS5.2). Contains 24+ variants covering the standard MPI error classes plus a `Raw(i32)` fallback for unrecognized classes.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpiErrorClass {
    Success,    // MPI_SUCCESS
    Buffer,     // MPI_ERR_BUFFER
    Count,      // MPI_ERR_COUNT
    Type,       // MPI_ERR_TYPE
    Tag,        // MPI_ERR_TAG
    Comm,       // MPI_ERR_COMM
    Rank,       // MPI_ERR_RANK
    Request,    // MPI_ERR_REQUEST
    Root,       // MPI_ERR_ROOT
    Group,      // MPI_ERR_GROUP
    Op,         // MPI_ERR_OP
    Topology,   // MPI_ERR_TOPOLOGY
    Dims,       // MPI_ERR_DIMS
    Arg,        // MPI_ERR_ARG
    Unknown,    // MPI_ERR_UNKNOWN
    Truncate,   // MPI_ERR_TRUNCATE
    Other,      // MPI_ERR_OTHER
    Intern,     // MPI_ERR_INTERN
    InStatus,   // MPI_ERR_IN_STATUS
    Pending,    // MPI_ERR_PENDING
    Win,        // MPI_ERR_WIN
    Info,       // MPI_ERR_INFO
    File,       // MPI_ERR_FILE
    Raw(i32),   // Any unrecognized error class
}
```

**Note:** The real `MpiErrorClass` does not have a dedicated `NoMem` variant (unlike the speculative API). Memory allocation failures from `MPI_ERR_NO_MEM` surface as `MpiErrorClass::Other` or `MpiErrorClass::Raw(MPI_ERR_NO_MEM)`. The error conversion in SS5.2 handles this by falling through to `CommError::CollectiveFailed`, with the human-readable message providing the diagnostic detail.

#### 7.4.7 Other Supporting Types

| Type                  | Description                                                                             |
| --------------------- | --------------------------------------------------------------------------------------- |
| `SplitType`           | Enum with `Shared = 0` variant for `split_type`. `split_shared()` uses this internally. |
| `Request`             | Handle for nonblocking MPI operations. Methods: `wait()`, `test()`, `cancel()`.         |
| `PersistentRequest`   | Handle for MPI 4.0+ persistent operations. Methods: `start()`, `wait()`, `test()`.      |
| `Status`              | Message metadata from receive operations: source rank, tag, element count.              |
| `Info`                | RAII wrapper for `MPI_Info` objects. RAII Drop calls `MPI_Info_free`.                   |
| `LockType`            | Enum: `Exclusive`, `Shared`. Used with `SharedWindow::lock()`.                          |
| `LockGuard<'a, T>`    | RAII guard from `SharedWindow::lock()`. Drop calls `MPI_Win_unlock`.                    |
| `LockAllGuard<'a, T>` | RAII guard from `SharedWindow::lock_all()`. Drop calls `MPI_Win_unlock_all`.            |

### 7.5 Unsafe Boundary Summary

No public ferrompi method or function uses the Rust `unsafe` keyword. All `unsafe` code is internal to the ferrompi crate, concentrated in:

1. **FFI calls** -- Every MPI C function call (`MPI_Init_thread`, `MPI_Comm_rank`, `MPI_Allgatherv`, `MPI_Win_allocate_shared`, `MPI_Win_free`, etc.) is wrapped in an `unsafe` block within the ferrompi implementation.
2. **`Send + Sync` impls** -- `Communicator` implements `Send + Sync` via `unsafe impl`, justified by the fact that the handle wraps an integer into a C-side table with no thread-local state. Under `MPI_THREAD_FUNNELED`, only the main thread issues MPI calls; `Send + Sync` permits safe sharing of the handle without implying concurrent MPI invocation.
3. **Raw pointer dereference** -- `SharedWindow::local_slice`, `SharedWindow::local_slice_mut`, and `SharedWindow::remote_slice` dereference the raw pointer obtained from `MPI_Win_allocate_shared`. The pointer is guaranteed valid by MPI for the lifetime of the window.

The public API exposes safe Rust types and borrows. Callers are not required to write `unsafe` code to use ferrompi. The `SharedWindow::remote_slice` method has a **logical safety contract** (SS7.3.3) but not a Rust `unsafe` contract -- violating the contract produces unspecified values, not undefined behavior.

## Cross-References

- [Communicator Trait §1](./communicator-trait.md) -- `Communicator` trait definition, `CommData`, `ReduceOp`, `CommError` type definitions implemented by this backend
- [Communicator Trait §2](./communicator-trait.md) -- Method contracts (preconditions, postconditions, determinism guarantees) that this backend preserves by delegation to ferrompi
- [Communicator Trait §3](./communicator-trait.md) -- Generic parameterization pattern (`train<C: Communicator>`) enabling zero-cost monomorphization of this backend
- [Communicator Trait §4](./communicator-trait.md) -- `SharedMemoryProvider` trait, `SharedRegion<T>` lifecycle phases, leader/follower pattern, drop behavior table
- [Communicator Trait §4.6](./communicator-trait.md) -- `CommError::AllocationFailed` variant used for shared memory allocation failure mapping
- [Communication Patterns §1.1](./communication-patterns.md) -- ferrompi API signatures (`comm.allgatherv`, `comm.allreduce`, `comm.broadcast`, `comm.barrier`) wrapped by this backend
- [Communication Patterns §4](./communication-patterns.md) -- Persistent collectives (`allreduce_init`, `allgatherv_init`) used as internal optimization (SS4.2)
- [Communication Patterns §5](./communication-patterns.md) -- `SharedWindow<T>` capabilities (window creation, intra-node grouping, read access, write synchronization) wrapped by `FerrompiRegion<T>`
- [Hybrid Parallelism §1.2](./hybrid-parallelism.md) -- ferrompi capabilities table: `Communicator` is `Send + Sync`, `SharedWindow<T>`, collectives API, threading level
- [Hybrid Parallelism §6](./hybrid-parallelism.md) -- MPI initialization sequence (Steps 1-3) implemented by `FerrompiBackend::new()`
- [Backend Registration and Selection §1.2](./backend-selection.md) -- Feature flag matrix; `mpi` feature gates the ferrompi crate dependency
- [Backend Registration and Selection §4](./backend-selection.md) -- Factory pattern returning concrete `FerrompiBackend` type in single-feature builds
- [Solver Abstraction §10](../architecture/solver-abstraction.md) -- Compile-time selection pattern via generic parameters and Cargo feature flags; the architectural precedent for this backend's zero-cost design
