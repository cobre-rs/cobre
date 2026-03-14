# cobre-comm

<span class="status-alpha">alpha</span>

`cobre-comm` is the pluggable communication backend abstraction for the Cobre
ecosystem. It defines the `Communicator` and `SharedMemoryProvider` traits that
decouple distributed computations from specific communication technologies,
allowing solver crates to run unchanged in single-process, MPI-distributed, and
future TCP or shared-memory configurations.

The crate currently provides two concrete backends:

- **`local`** — single-process backend, always available, zero overhead, zero
  external dependencies.
- **`mpi`** — MPI backend via [ferrompi](https://github.com/cobre-rs/ferrompi),
  feature-gated behind `features = ["mpi"]`.

Two additional backend slots are deferred for future implementation:

- **`tcp`** — TCP/IP coordinator pattern (no MPI required).
- **`shm`** — POSIX shared memory for single-node multi-process execution.

The factory function [`create_communicator`](#factory-function-create_communicator)
selects the backend at startup based on Cargo feature flags and an optional
environment variable override. Downstream solver crates depend on the
`Communicator` trait through a generic type parameter — never on a concrete
backend type.

## Module overview

| Module     | Purpose                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------- |
| `traits`   | Core trait definitions: `Communicator`, `SharedMemoryProvider`, `SharedRegion`, `CommData`, `LocalCommunicator` |
| `types`    | Shared types: `ReduceOp`, `CommError`, `BackendError`                                                           |
| `local`    | `LocalBackend` (single-process) and `HeapRegion` (heap-backed shared region)                                    |
| `ferrompi` | `FerrompiBackend` — MPI backend (only compiled with `features = ["mpi"]`)                                       |
| `factory`  | `create_communicator`, `BackendKind`, `CommBackend`, `available_backends`                                       |

## `Communicator` trait

```rust
pub trait Communicator: Send + Sync { ... }
```

The trait provides the six operations used during distributed computations:
four collective operations and two infallible accessor methods. The trait is
intentionally **not object-safe** — it carries generic methods
(`allgatherv<T>`, `allreduce<T>`, `broadcast<T>`) that require static dispatch.
This is the same monomorphization pattern used by `SolverInterface` in
[`cobre-solver`](./solver.md#architecture): callers parameterize a generic
function once and the compiler generates one concrete instantiation per backend.

Since a Cobre binary uses exactly one communicator backend (MPI for distributed
execution, `LocalBackend` for single-process mode), the binary contains only
one instantiation per generic call site. The performance benefit is meaningful:
`LocalBackend`'s no-op implementations compile to zero instructions after
inlining.

### Method summary

| Method       | Signature                                                      | Returns                 | Description                                                |
| ------------ | -------------------------------------------------------------- | ----------------------- | ---------------------------------------------------------- |
| `allgatherv` | `(&self, send, recv, counts, displs) -> Result<(), CommError>` | `Result<(), CommError>` | Gather variable-length data from all ranks into all ranks  |
| `allreduce`  | `(&self, send, recv, op: ReduceOp) -> Result<(), CommError>`   | `Result<(), CommError>` | Element-wise reduction (sum, min, or max) across all ranks |
| `broadcast`  | `(&self, buf, root: usize) -> Result<(), CommError>`           | `Result<(), CommError>` | Copy data from the root rank to all other ranks            |
| `barrier`    | `(&self) -> Result<(), CommError>`                             | `Result<(), CommError>` | Block until all ranks have entered; pure synchronization   |
| `rank`       | `(&self) -> usize`                                             | `usize`                 | Return this rank's index (0..size); infallible             |
| `size`       | `(&self) -> usize`                                             | `usize`                 | Return total number of ranks; infallible                   |

### Design: compile-time static dispatch

Writing `Box<dyn Communicator>` does not compile — the trait is intentionally
not object-safe. All callers use a generic type parameter:

```rust
use cobre_comm::{Communicator, CommError};

fn print_topology<C: Communicator>(comm: &C) {
    println!("rank {} of {}", comm.rank(), comm.size());
}
```

This is the mandated enum dispatch pattern for closed variant sets in Cobre. The
dispatch overhead for `CommBackend` is a single branch-predictor-friendly
integer comparison, negligible compared to the cost of the MPI collective
operation or LP solve it wraps.

### Thread safety

`Communicator` requires `Send + Sync`. All collective methods take `&self`
(shared reference). Callers are responsible for serializing concurrent calls —
the training loop ensures that multiple threads never invoke the same collective
simultaneously on the same communicator instance. `rank()` and `size()` are
safe to call concurrently: their values are cached at construction time and
never change.

## `SharedMemoryProvider` trait

```rust
pub trait SharedMemoryProvider: Send + Sync { ... }
```

`SharedMemoryProvider` is a companion trait to `Communicator` for managing
intra-node shared memory regions. It is a **separate trait** rather than a
supertrait of `Communicator`, which preserves flexibility: not all backends
support true shared memory. Functions that only need collective communication
use `C: Communicator`; functions that additionally need shared memory use
`C: Communicator + SharedMemoryProvider`.

### `HeapRegion` — the minimal viable region type

For the minimal viable implementation, all backends use `HeapRegion<T>` as
their `SharedMemoryProvider::Region<T>` type. `HeapRegion<T>` is a thin
wrapper around `Vec<T>`: each rank holds its own private heap allocation with
no actual memory sharing between processes. The three-phase lifecycle
(allocation, population, read-only) degenerates to simple `Vec` operations,
with `fence()` a no-op.

True shared memory via MPI windows or POSIX shared memory segments is
planned for a future optimization phase.

### `LocalCommunicator` — object-safe intra-node coordination

`LocalCommunicator` is a purpose-built object-safe sub-trait that exposes
only the three non-generic methods needed for intra-node initialization
coordination:

```rust
use cobre_comm::LocalCommunicator;

fn determine_leader(local_comm: &dyn LocalCommunicator) -> bool {
    local_comm.rank() == 0
}
```

`SharedMemoryProvider::split_local` returns `Box<dyn LocalCommunicator>` — an
intra-node communicator used only during initialization (leader/follower role
assignment). Because this is an initialization-only operation far off the hot
path, dynamic dispatch is the correct trade-off, and `LocalCommunicator` is the
bridge that makes it possible without compromising the zero-cost static dispatch
of the hot-path `Communicator` trait.

## `LocalBackend`

```rust
pub struct LocalBackend;
```

`LocalBackend` is a zero-sized type (ZST) with no runtime state and no
external dependencies. All collective operations use identity-copy or no-op
semantics:

- `rank()` always returns `0`.
- `size()` always returns `1`.
- `allgatherv` copies `send` into `recv` at the specified displacement
  (identity copy — with one rank, gather is trivial).
- `allreduce` copies `send` to `recv` unchanged (reduction of a single operand
  is the identity).
- `broadcast` is a no-op (data is already at the only rank).
- `barrier` is a no-op (nothing to synchronize).

Because `LocalBackend` is a ZST, it occupies zero bytes at runtime and has no
construction cost. Its collective method implementations compile to zero
instructions after inlining in single-feature builds.

### Example

```rust
use cobre_comm::{LocalBackend, Communicator, ReduceOp};

let comm = LocalBackend;
assert_eq!(comm.rank(), 0);
assert_eq!(comm.size(), 1);

// allreduce with one rank: identity copy regardless of op.
let send = vec![1.0_f64, 2.0, 3.0];
let mut recv = vec![0.0_f64; 3];
comm.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
assert_eq!(recv, send);
```

`LocalBackend` also implements `SharedMemoryProvider` with `HeapRegion<T>` as
the region type, and `LocalCommunicator` for use in intra-node initialization
code.

## `FerrompiBackend`

`FerrompiBackend` is the MPI backend, powered by the
[ferrompi](https://github.com/cobre-rs/ferrompi) crate. It is only compiled
when `features = ["mpi"]` is specified:

```toml
# Cargo.toml
cobre-comm = { version = "0.1", features = ["mpi"] }
```

`FerrompiBackend` wraps a `ferrompi::Mpi` environment handle and an
`MPI_COMM_WORLD` communicator. Construction calls `MPI_Init_thread` with
`ThreadLevel::Funneled`, matching the Cobre execution model where only the main
thread issues MPI calls. When `FerrompiBackend` is dropped, the RAII guard
calls `MPI_Finalize` automatically.

`FerrompiBackend` requires an MPI runtime to be installed on the system. If no
MPI runtime is found, `FerrompiBackend::new()` returns
`Err(BackendError::InitializationFailed)`.

The `unsafe impl Send + Sync` on `FerrompiBackend` reflects the fact that
`ferrompi::Mpi` is `!Send + !Sync` by default (using a `PhantomData<*const ()>`
marker), but the Cobre RAII pattern guarantees that construction and
finalization happen on the same thread, making the impl sound.

## Factory function: `create_communicator`

```rust,no_run
pub fn create_communicator() -> Result<impl Communicator, BackendError>
```

`create_communicator` is the single entry point for constructing a communicator
at startup. It selects the backend according to:

1. The `COBRE_COMM_BACKEND` environment variable (runtime override).
2. The Cargo features compiled into the binary (auto-detection).
3. A fallback to `LocalBackend` when no distributed backend is available or
   detected.

### `BackendKind` enum

`BackendKind` is provided for library-mode callers (such as `cobre-python` or
`cobre-mcp`) that need to select a backend programmatically rather than through
environment variables:

| Variant              | Behavior                                                           |
| -------------------- | ------------------------------------------------------------------ |
| `BackendKind::Auto`  | Let the factory choose the best available backend (default)        |
| `BackendKind::Mpi`   | Request the MPI backend; fails if `mpi` feature is not compiled in |
| `BackendKind::Local` | Always use `LocalBackend`, even when MPI is available              |

### `COBRE_COMM_BACKEND` environment variable

| Value     | Behavior                                                                        |
| --------- | ------------------------------------------------------------------------------- |
| (unset)   | Auto-detect: MPI if MPI launcher env vars are present, otherwise `LocalBackend` |
| `"auto"`  | Same as unset                                                                   |
| `"mpi"`   | Use `FerrompiBackend`; fails if `mpi` feature is not compiled in                |
| `"local"` | Always use `LocalBackend`                                                       |
| `"tcp"`   | Deferred; returns `BackendNotAvailable` (no implementation yet)                 |
| `"shm"`   | Deferred; returns `BackendNotAvailable` (no implementation yet)                 |

Auto-detection checks for the presence of MPI launcher environment variables
(`PMI_RANK`, `PMI_SIZE`, `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE`,
`MPI_LOCALRANKID`, `SLURM_PROCID`). If any of these is set, the factory
attempts to initialize the MPI backend.

### Example

```rust,no_run
use cobre_comm::{create_communicator, Communicator};

// With COBRE_COMM_BACKEND unset (auto-detect):
// - returns FerrompiBackend if launched via mpirun/mpiexec
// - returns LocalBackend otherwise
let comm = create_communicator().expect("backend selection failed");
println!("rank {} of {}", comm.rank(), comm.size());
```

When distributed features are compiled in, `create_communicator` returns a
`CommBackend` enum that delegates each method call to the active concrete
backend via a `match`. When no distributed features are compiled in, it returns
`LocalBackend` directly.

### `CommBackend` enum

`CommBackend` is the enum-dispatched communicator wrapper present in builds
where at least one distributed backend feature (`mpi`, `tcp`, or `shm`) is
compiled in. It implements both `Communicator` and `SharedMemoryProvider` by
delegating each method to the active inner backend:

```rust,no_run
use cobre_comm::{create_communicator, Communicator};

// With COBRE_COMM_BACKEND=local, the factory returns CommBackend::Local.
let comm = create_communicator().expect("backend selection failed");
let send = [42.0_f64];
let mut recv = [0.0_f64];
comm.allgatherv(&send, &mut recv, &[1], &[0]).unwrap();
assert_eq!(recv[0], 42.0);
```

## Error types

### `CommError`

Returned by all fallible methods on `Communicator` and `SharedMemoryProvider`.

| Variant               | When it occurs                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CollectiveFailed`    | An MPI collective operation failed at the library level (carries MPI error code and description)                                                        |
| `InvalidBufferSize`   | Buffer sizes provided to a collective are inconsistent (e.g., `recv.len() < sum(counts)` in `allgatherv`, or `send.len() != recv.len()` in `allreduce`) |
| `InvalidRoot`         | The `root` rank argument is out of range (`root >= size()`)                                                                                             |
| `InvalidCommunicator` | The communicator is in an invalid state (e.g., MPI has been finalized)                                                                                  |
| `AllocationFailed`    | A shared memory allocation request was rejected by the OS (size too large, insufficient permissions, or system limits exceeded)                         |

### `BackendError`

Returned by `create_communicator` when the backend cannot be selected or
initialized.

| Variant                | When it occurs                                                                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `BackendNotAvailable`  | The requested backend is not compiled into this binary (e.g., `COBRE_COMM_BACKEND=mpi` without the `mpi` feature) |
| `InvalidBackend`       | The `COBRE_COMM_BACKEND` value does not match any known backend name                                              |
| `InitializationFailed` | The backend was correctly selected but failed to initialize (e.g., MPI runtime not installed)                     |
| `MissingConfiguration` | Required environment variables for the selected backend are not set (relevant for future `tcp`/`shm` backends)    |

## Deferred features

The following features are planned but not yet implemented:

- **TCP backend** (`"tcp"` feature): a TCP/IP coordinator pattern for
  distributed execution without requiring an MPI installation. Will follow the
  same `Communicator` trait interface.
- **Shared memory backend** (`"shm"` feature): POSIX shared memory for
  single-node multi-process execution with zero inter-process copy overhead.
  Will implement `SharedMemoryProvider` using POSIX shared memory segments or
  MPI shared windows rather than the current `HeapFallback` semantics.

## Feature flags

| Feature | Default | Description                                                    |
| ------- | ------- | -------------------------------------------------------------- |
| `mpi`   | no      | Enables `FerrompiBackend` and the `ferrompi` dependency        |
| `tcp`   | no      | Deferred: future TCP backend (no implementation yet)           |
| `shm`   | no      | Deferred: future shared memory backend (no implementation yet) |

Without any feature flags, only `LocalBackend`, the trait definitions, and
the type definitions are compiled. `create_communicator` returns `LocalBackend`
directly (not wrapped in `CommBackend`).

## Testing

### Running the test suite

```
cargo test -p cobre-comm
```

This runs all unit, integration, and doc-tests for the default (no-feature)
configuration. No MPI installation is required.

To run the full test suite including the MPI backend:

```
cargo test -p cobre-comm --features mpi
```

This requires an MPI runtime (`libmpich-dev` on Debian/Ubuntu, `mpich` on
Fedora or macOS Homebrew). CI runs tests without the `mpi` feature by default;
the MPI feature tests require a manual setup with an MPI installation.

### Conformance suite (`tests/conformance.rs`)

The integration test file `tests/conformance.rs` implements the
backend-agnostic conformance contract. It verifies the `Communicator` contract
using only the public API against the `LocalBackend` concrete type. The
conformance suite covers:

- `rank()` returns `0` and `size()` returns `1` for single-process mode.
- `allgatherv` copies `send` into `recv` at the correct displacement.
- `allreduce` copies `send` to `recv` unchanged (identity for a single rank),
  for all three `ReduceOp` variants.
- `broadcast` is a no-op for `root == 0`.
- `barrier` returns `Ok(())`.
- Buffer precondition violations return the correct `CommError` variants.
- `HeapRegion` lifecycle: allocation, write via `as_mut_slice`, `fence`,
  and read via `as_slice`.
- `CommBackend::Local` delegates all `Communicator` and `SharedMemoryProvider`
  methods correctly.

## Design notes

### Enum dispatch

`CommBackend` uses enum dispatch rather than `Box<dyn Communicator>`. The
`Communicator` trait carries generic methods that make it intentionally not
object-safe. Enum dispatch is the mandated pattern for closed variant sets
in Cobre: a single `match` arm delegates each method to the inner
concrete type. The overhead is a single branch-predictor-friendly integer
comparison per call, which is negligible compared to the cost of the
underlying MPI collective or LP solve.

### `CommData` conditional supertrait

The `CommData` marker trait — required for all types transmitted through
collective operations — has a conditional supertrait:

- **With `mpi` feature**: `CommData` additionally requires
  `ferrompi::MpiDatatype`, narrowing the set of valid types to the seven
  primitives that MPI can transmit directly (`f32`, `f64`, `i32`, `i64`,
  `u8`, `u32`, `u64`).
- **Without `mpi` feature**: `CommData` accepts all `Copy + Send + Sync +
Default + 'static` types, including `bool` and tuples used in tests.

This design avoids an extra bound on every method signature: `FerrompiBackend`
can delegate directly to ferrompi's generic FFI methods because the
`MpiDatatype` constraint is already satisfied by `CommData`.

### cfg-gate strategy

Backend modules and types are compiled only when their feature is enabled. The
`CommBackend` enum is only present when at least one distributed feature
(`mpi`, `tcp`, or `shm`) is compiled in — builds without distributed features
use `LocalBackend` directly. This ensures that single-process builds have no
code-size cost from unused backends.
