# Local Backend

## Purpose

The local backend is the single-process communication backend for Cobre. It provides a `Communicator` and `SharedMemoryProvider` implementation where all collective operations are identity operations (copy input to output) or no-ops (do nothing), formalizing the single-process mode previously described as prose exceptions in [Hybrid Parallelism §1.0a](./hybrid-parallelism.md). The local backend is always available -- it requires no feature flag, no external dependencies, and no runtime configuration. It is the default fallback when no other backend is configured, as specified in the priority chain of [Backend Registration and Selection §2.2](./backend-selection.md). In single-feature builds with no communication features enabled, all collectives compile to zero instructions after inlining ([Backend Registration and Selection §1.4](./backend-selection.md)).

## 1. Struct and Trait Implementation

### 1.1 Struct Definition

The `LocalBackend` struct is a zero-sized type (ZST). It holds no state because there is exactly one rank, no MPI communicator handles, no intra-node communicator, and no connection state. The ZST property means that `LocalBackend` occupies zero bytes at runtime and has no construction cost.

```rust
/// Single-process communication backend with identity collective semantics.
///
/// Zero-sized type with no runtime state. All collective operations are
/// identity copies or no-ops, compiling to zero instructions after inlining
/// in single-feature builds (see §1.2).
pub struct LocalBackend;
```

### 1.2 Communicator Trait Implementation

The `impl Communicator for LocalBackend` provides trivial implementations for all six trait methods. Each method satisfies the contracts defined in [Communicator Trait §2](./communicator-trait.md) for the degenerate case of a single rank.

```rust
impl Communicator for LocalBackend {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        // Validate: single rank requires exactly one count/displ entry.
        if counts.len() != 1 { return Err(CommError::InvalidBufferSize { .. }); }
        if displs.len() != 1 { return Err(CommError::InvalidBufferSize { .. }); }
        if send.len() != counts[0] { return Err(CommError::InvalidBufferSize { .. }); }
        if recv.len() < displs[0] + counts[0] { return Err(CommError::InvalidBufferSize { .. }); }
        // Identity copy: with one rank, displs=[0] and counts=[send.len()].
        recv[displs[0]..displs[0] + counts[0]].copy_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        // Validate: buffer lengths must match and be non-empty.
        if send.len() != recv.len() { return Err(CommError::InvalidBufferSize { .. }); }
        if send.is_empty() { return Err(CommError::InvalidBufferSize { .. }); }
        // Identity copy: Sum(x) = Min(x) = Max(x) = x for a single operand.
        recv.copy_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(
        &self,
        _buf: &mut [T],
        root: usize,
    ) -> Result<(), CommError> {
        // Validate: only root 0 is valid for a single-rank communicator.
        if root >= 1 { return Err(CommError::InvalidRoot { root, size: 1 }); }
        // No-op: the single rank is both sender and receiver.
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        // No-op: nothing to synchronize.
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }
}
```

**Precondition validation:** Unlike the ferrompi backend, the local backend validates buffer size preconditions and root rank arguments directly, returning appropriate errors:

- `allgatherv` returns `CommError::InvalidBufferSize` if `counts.len() != 1`, `displs.len() != 1`, `send.len() != counts[0]`, or `recv.len() < displs[0] + counts[0]`.
- `allreduce` returns `CommError::InvalidBufferSize` if `send.len() != recv.len()` or `send.is_empty()`.
- `broadcast` returns `CommError::InvalidRoot` if `root >= 1` (the only valid root for a single-rank communicator is 0).
- `barrier` returns `Ok(())` unconditionally (no-op).

The local backend cannot produce `CommError::CollectiveFailed` (no MPI calls) or `CommError::InvalidCommunicator` (no communicator state to invalidate).

**Inlining and codegen:** Because `LocalBackend` is a ZST with trivial method bodies, the compiler inlines all trait methods at call sites when the concrete type is known. In a single-feature build (no `mpi`, `tcp`, or `shm` features), the generic parameter `C: Communicator` resolves to `LocalBackend`, and:

- `allgatherv` compiles to a single `memcpy` (or equivalent loop for non-`Copy` codegen).
- `allreduce` compiles to a single `memcpy`.
- `broadcast` compiles to zero instructions.
- `barrier` compiles to zero instructions.
- `rank` compiles to the constant `0`.
- `size` compiles to the constant `1`.

This achieves the zero-cost abstraction guarantee stated in [Backend Registration and Selection §1.4](./backend-selection.md).

## 2. Identity Semantics

The local backend implements each `Communicator` method as either an **identity operation** (input copied to output) or a **no-op** (no action taken). The distinction is important: identity operations perform a memory copy that must not be elided, while no-ops can be entirely eliminated by the compiler.

### 2.1 Behavior Comparison Table

| Method       | Multi-Rank Behavior                                                | Local Backend Behavior                                      | Classification |
| ------------ | ------------------------------------------------------------------ | ----------------------------------------------------------- | -------------- |
| `allgatherv` | Gathers variable-length data from all ranks, ordered by rank index | Copies `send` to `recv[displs[0]..displs[0]+counts[0]]`     | Identity copy  |
| `allreduce`  | Element-wise reduction (Sum, Min, Max) across all ranks            | Copies `send` to `recv` (reduction of one value = identity) | Identity copy  |
| `broadcast`  | Sends data from root rank to all other ranks                       | No-op (data is already in the buffer on the only rank)      | No-op          |
| `barrier`    | Blocks until all ranks have entered the barrier                    | No-op (single rank, nothing to wait for)                    | No-op          |
| `rank()`     | Returns the calling rank's index in `0..size()`                    | Returns `0`                                                 | Constant       |
| `size()`     | Returns the total number of ranks in the communicator              | Returns `1`                                                 | Constant       |

### 2.2 Postcondition Verification

Each method's postconditions from [Communicator Trait §2](./communicator-trait.md) are satisfied by the local backend:

| Postcondition                                  | Method       | Local Backend Satisfaction                                                                    |
| ---------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------- |
| Rank-ordered receive                           | `allgatherv` | Rank 0's data at `displs[0]` -- the only rank contributes, and its data appears at position 0 |
| Identical across ranks                         | `allgatherv` | Trivially satisfied (only one rank)                                                           |
| Implicit barrier                               | `allgatherv` | Trivially satisfied (only one rank)                                                           |
| Element-wise reduction                         | `allreduce`  | `op(x) = x` for any single operand -- the identity copy is the correct reduction              |
| Identical across ranks                         | `allreduce`  | Trivially satisfied (only one rank)                                                           |
| Data from root                                 | `broadcast`  | Rank 0 is both root and sole receiver; data is already in place                               |
| Identical across ranks                         | `broadcast`  | Trivially satisfied (only one rank)                                                           |
| Global synchronization (all ranks enter first) | `barrier`    | Trivially satisfied (only one rank)                                                           |
| `rank()` in `0..size()`                        | `rank`       | `0` is in `0..1`                                                                              |
| Constant after initialization                  | `rank/size`  | ZST with hardcoded values -- always constant                                                  |

## 3. Shared Memory Fallback

The local backend implements `SharedMemoryProvider` using the `HeapFallback` strategy defined in [Communicator Trait §4.4](./communicator-trait.md). Shared memory regions are regular heap-allocated `Vec<T>` instances. The semantics are fully specified in [Communicator Trait §4.4](./communicator-trait.md); this section documents the local backend's concrete realization of that specification.

### 3.1 SharedMemoryProvider Implementation

```rust
impl SharedMemoryProvider for LocalBackend {
    type Region<T: CommData> = HeapRegion<T>;

    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError> {
        Ok(HeapRegion {
            data: vec![T::default(); count],
        })
    }

    fn split_local(&self) -> Result<Box<dyn LocalCommunicator>, CommError> {
        // A single process is its own node; the intra-node communicator
        // is identical to the world communicator.
        Ok(Box::new(LocalBackend))
    }

    fn is_leader(&self) -> bool {
        // Always true: the single rank is its own leader.
        true
    }
}
```

### 3.2 HeapRegion

The `HeapRegion<T>` type wraps a `Vec<T>` and implements `SharedRegion<T>` with trivial semantics:

```rust
/// Shared memory region backed by a heap-allocated `Vec<T>`.
///
/// Used by backends without true intra-node shared memory (local, tcp).
/// Lifecycle phases from [Communicator Trait §4.2](./communicator-trait.md)
/// degenerate to simple `Vec` operations.
pub struct HeapRegion<T: CommData> {
    data: Vec<T>,
}

impl<T: CommData> SharedRegion<T> for HeapRegion<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn fence(&self) -> Result<(), CommError> {
        // No-op: all access is within a single process.
        Ok(())
    }
}
```

### 3.3 HeapFallback Behavior Summary

The local backend's `HeapFallback` realization maps to the canonical behavior table in [Communicator Trait §4.4](./communicator-trait.md):

| Method                 | HeapFallback Behavior (from §4.4)                           | Local Backend Realization                                 |
| ---------------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| `create_shared_region` | Allocates `Vec<T>` with `count` elements (per-process copy) | `HeapRegion { data: vec![T::default(); count] }`          |
| `is_leader`            | Always returns `true` (every rank is its own leader)        | Returns `true` (single rank is the sole leader)           |
| `split_local`          | Returns a single-rank communicator (rank 0 of size 1)       | Returns `Box::new(LocalBackend)` (rank 0, size 1)         |
| `as_slice`             | Returns `&self.vec[..]` (local heap memory)                 | Returns `&self.data[..]`                                  |
| `as_mut_slice`         | Returns `&mut self.vec[..]` (local heap memory)             | Returns `&mut self.data[..]`                              |
| `fence`                | No-op (returns `Ok(())`)                                    | Returns `Ok(())` (no remote ranks to synchronize)         |
| `Drop`                 | Drops inner `Vec<T>`                                        | `HeapRegion` drops inner `data: Vec<T>` via standard Drop |

**Memory footprint:** The `HeapFallback` replicates data per-process. With a single process, there is no replication overhead -- the memory footprint equals the data size. The memory savings from true shared memory backends are irrelevant when there is only one process.

## 4. Use Cases

The local backend is used whenever Cobre operates without inter-process communication. It is the communication backend for all non-MPI execution modes:

### 4.1 Python Bindings

Python bindings ([Python Bindings §1.2](../interfaces/python-bindings.md)) operate in single-process mode because the GIL is incompatible with MPI launchers ([Hybrid Parallelism §1.0a](./hybrid-parallelism.md)). The local backend is constructed directly by the binding layer before releasing the GIL and entering the Rust training function:

```rust
let comm = LocalBackend;
let result = train(&comm, &config)?;
```

### 4.2 MCP Server

The MCP server ([MCP Server §1.1](../interfaces/mcp-server.md)) is a long-lived single-process server incompatible with MPI launcher lifecycle management. It constructs the local backend once at server startup and reuses it for all training invocations. See [Backend Registration and Selection §5.4](./backend-selection.md).

### 4.3 TUI (Terminal User Interface)

The TUI operates as an interactive single-process mode where the user monitors training progress in real time. The local backend provides the communication layer without requiring MPI infrastructure on the user's workstation.

### 4.4 Testing and CI

When no communication features are enabled (the Test / CI build profile from [Backend Registration and Selection §1.3](./backend-selection.md)), the local backend is the only available backend. It provides deterministic, dependency-free execution for unit tests, integration tests, and CI pipelines. The `create_communicator()` factory ([Backend Registration and Selection §4.1](./backend-selection.md)) returns `LocalBackend` directly, with full monomorphization and zero dispatch overhead.

### 4.5 Always-Available Guarantee

The local backend requires no Cargo feature flag, no external libraries, no MPI runtime, no TCP coordinator, and no shared memory segments. It is unconditionally compiled into every Cobre binary, as specified in the feature flag matrix ([Backend Registration and Selection §1.2](./backend-selection.md)). This makes it the guaranteed fallback at the bottom of the auto-detection priority chain ([Backend Registration and Selection §2.2](./backend-selection.md)).

**No-feature build:** In a build with no communication features (`cargo test`, `cargo build` with no `--features`), the local backend is the only communication backend. The factory function returns `LocalBackend` directly:

```rust
#[cfg(not(any(feature = "mpi", feature = "tcp", feature = "shm")))]
pub fn create_communicator() -> Result<LocalBackend, BackendError> {
    Ok(LocalBackend)
}
```

**No runtime configuration:** The local backend requires no environment variables ([Backend Registration and Selection §3.1](./backend-selection.md)). No `COBRE_` prefixed variables, no launcher-injected variables, no configuration of any kind.

## 5. Determinism

### 5.1 Communication Determinism

With a single rank, there is no inter-process communication and therefore no source of communication non-determinism. The reproducibility guarantees from [Shared Memory Aggregation §3.1](./shared-memory-aggregation.md) are trivially satisfied:

| Reproducibility Requirement              | Multi-Rank Mechanism                                                          | Local Backend                                                      |
| ---------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Independent of number of MPI ranks       | Deterministic seeding, contiguous block distribution, deterministic cut slots | N/A -- always 1 rank                                               |
| Independent of number of OpenMP threads  | Thread-local accumulation, fixed merge order                                  | Same mechanism -- OpenMP parallelism remains fully active          |
| Independent of execution timing/ordering | Identity-based seeding, deterministic `MPI_Allgatherv` rank ordering          | No communication timing; local operations are sequentially ordered |

### 5.2 Floating-Point Determinism

The floating-point non-determinism described in [Communicator Trait §2.2](./communicator-trait.md) (reduction tree shape varies with rank count and MPI implementation) does not apply to the local backend. With a single rank:

- `allreduce` with `ReduceOp::Sum` performs an identity copy -- no floating-point arithmetic, no reduction tree.
- The upper bound statistics are computed from the single rank's local trajectories using the thread-local accumulation pattern ([Shared Memory Aggregation §3.3](./shared-memory-aggregation.md)), which produces deterministic results regardless of thread count.

**Guarantee:** Given the same inputs and random seed, the local backend produces **bit-for-bit identical** results regardless of the number of OpenMP threads, matching the determinism invariant from [Shared Memory Aggregation §3.1](./shared-memory-aggregation.md).

## Cross-References

- [Communicator Trait §1](./communicator-trait.md) -- `Communicator` trait definition, `CommData`, `ReduceOp`, `CommError` type definitions implemented by this backend
- [Communicator Trait §2](./communicator-trait.md) -- Method contracts (preconditions, postconditions, determinism guarantees) satisfied trivially by single-rank identity semantics
- [Communicator Trait §3](./communicator-trait.md) -- Generic parameterization pattern (`train<C: Communicator>`) enabling zero-cost monomorphization of this backend
- [Communicator Trait §4](./communicator-trait.md) -- `SharedMemoryProvider` trait, `SharedRegion<T>` lifecycle, leader/follower pattern, and HeapFallback semantics (§4.4)
- [Hybrid Parallelism §1.0a](./hybrid-parallelism.md) -- Single-process mode definition: no MPI, no `SharedWindow<T>`, OpenMP parallelism remains active
- [Hybrid Parallelism §6a](./hybrid-parallelism.md) -- Alternative initialization sequence for single-process mode (Steps 1-3 skipped)
- [Training Loop §4.3a](../architecture/training-loop.md) -- Single-rank forward pass variant: all scenarios assigned to the single rank, `MPI_Allreduce` becomes local computation
- [Training Loop §6.3a](../architecture/training-loop.md) -- Single-rank backward pass variant: `MPI_Allgatherv` for cut synchronization becomes identity copy, per-stage barrier reduces to OpenMP barrier only
- [Backend Registration and Selection §1.2](./backend-selection.md) -- Feature flag matrix: local backend is unconditional, always compiled, ~1 KB binary impact (inlined away)
- [Backend Registration and Selection §1.4](./backend-selection.md) -- Monomorphization guarantee: local backend's no-op collectives compile to zero instructions after inlining
- [Backend Registration and Selection §2.2](./backend-selection.md) -- Auto-detection priority chain: local is the lowest-priority fallback, always available
- [Backend Registration and Selection §4.1](./backend-selection.md) -- Factory function for no-feature builds returns `LocalBackend` directly
- [Backend Registration and Selection §5.3](./backend-selection.md) -- Python bindings integration using `LocalBackend` as the default backend
- [Backend Registration and Selection §5.4](./backend-selection.md) -- MCP server integration using `LocalBackend` for all training invocations
- [Shared Memory Aggregation §3.1](./shared-memory-aggregation.md) -- Reproducibility requirement: bit-for-bit identical results independent of rank count, thread count, and execution timing
- [Shared Memory Aggregation §3.3](./shared-memory-aggregation.md) -- Floating-point determinism: thread-local accumulation with fixed merge order
- [Solver Abstraction §10](../architecture/solver-abstraction.md) -- Compile-time selection pattern via generic parameters and Cargo feature flags; the architectural precedent for this backend's zero-cost design
- [Python Bindings §1.2](../interfaces/python-bindings.md) -- Single-process execution mode for Python (GIL/MPI incompatibility)
- [MCP Server §1.1](../interfaces/mcp-server.md) -- Single-process execution mode for the MCP server
