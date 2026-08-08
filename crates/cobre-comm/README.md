# cobre-comm

Pluggable communication backend abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
distributed solver ecosystem.

Defines the `Communicator` and `SharedMemoryProvider` traits that decouple
distributed computations from specific communication technologies. Two backends
are implemented: `LocalBackend`, a single-process no-op that is always
available and carries zero overhead, and `FerrompiBackend`, an MPI 4.x
implementation built on the [ferrompi](https://github.com/cobre-rs/ferrompi)
crate and enabled via the `mpi` Cargo feature. Available backends are
determined at **build time** by Cargo feature flags; the **active** backend is
then chosen at **runtime** by the explicit `BackendKind` the caller passes to
`create_communicator` — there is no environment-variable configuration
channel. All dispatch over the `Communicator` trait is static, so there is no
dynamic dispatch overhead on the hot path.

## When to Use

Depend on `cobre-comm` directly when you are writing a distributed algorithm
that needs collective communication (broadcast, reduce, scatter/gather) and
you want to test it locally without an MPI installation. Algorithm crates such
as `cobre-sddp` depend on this crate and accept a generic `Communicator`
parameter; you only need to depend here when adding a new algorithm crate or a
new backend.

## Key Types

- **`Communicator`** — the core trait for collective communication operations
  (broadcast, reduce, barrier) implemented by every backend
- **`LocalBackend`** — single-process no-op backend; always available, no
  external dependencies
- **`FerrompiBackend`** — MPI 4.x backend built on ferrompi; enabled with the
  `mpi` feature flag
- **`BackendKind`** — enum listing the available backends detected at compile
  time, used to select a backend via `create_communicator`
- **`CpuTopology`** — physical-core, processing-unit, and NUMA resources visible
  inside the process or scheduler cpuset
- **`WorkerAffinity`** — deterministic worker mapping and Linux binding hook for
  Rayon pools
- **`SharedMemoryProvider`** — trait for intra-node memory-region allocation
  used by collocated processes

## Feature flags

| Feature         | Default | Description                                                                                                                                          |
| --------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mpi`           | off     | Enables `FerrompiBackend` and the `ferrompi` dependency (gates `dep:ferrompi`)                                                                       |
| `affinity`      | off     | Enables Linux CPU/NUMA discovery and current-thread CPU binding; other platforms retain an explicit unsupported fallback                            |
| `numa`          | off     | Extends MPI topology through ferrompi and enables worker affinity (implies `mpi` and `affinity`)                                                     |
| `shared-memory` | off     | Experimental intra-node shared-memory region API (`SharedMemoryProvider`, `SharedRegion`, `LocalCommunicator`, `HeapRegion`). Composable with `mpi`. |

Without any feature flags, only `LocalBackend`, the trait definitions, and the
type definitions are compiled; `create_communicator` returns `LocalBackend`
directly (not wrapped in an enum). The `shared-memory` feature is off by
default because its only implementation, `HeapRegion<T>`, is a heap-backed
placeholder — each rank allocates its own private `Vec<T>` with no memory
actually shared across processes; a true implementation on ferrompi's
`SharedWindow<T>` is deferred until a downstream consumer exists.

## Worker affinity

`CpuTopology::discover()` intersects Linux sysfs topology with the CPU set
inherited from the operating system, MPI launcher, container, or scheduler. It
also reports the allowed memory nodes and current memory policy when the kernel
permits the query. `WorkerAffinity` then resolves one of three policies:

- `none` — observe topology without changing scheduling;
- `core` — fill physical cores before SMT siblings;
- `numa` — round-robin physical cores across visible NUMA nodes before SMT.

The CLI and Python binding use the same planner and current-thread binding API.
MPI rank placement remains the launcher's responsibility; Cobre never expands a
rank beyond its inherited CPU set.

## Backend selection

`create_communicator(kind: BackendKind) -> Result<impl Communicator, BackendError>`
is the single runtime entry point for constructing the active communicator.
The `cobre` CLI maps `--comm-backend <auto|local|mpi>` (default `auto`) onto
this argument.

| `BackendKind` variant | Behavior                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Auto`                | Detect from the launch environment: MPI when an MPI launcher is present, else local                           |
| `Mpi`                 | Force the MPI backend; fails with `BackendError::BackendNotAvailable` if the `mpi` feature is not compiled in |
| `Local`               | Always use `LocalBackend`, even when MPI is available                                                         |

`Auto` checks for the presence of MPI launcher environment variables —
`PMI_RANK`, `PMI_SIZE`, `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE`,
`MPI_LOCALRANKID`, `SLURM_PROCID` — runtime facts set by
`mpiexec`/`mpirun`/`srun`, not a configuration channel. If any is set (and the
`mpi` feature is compiled in), the MPI backend is initialized; otherwise the
local backend is used. A run started under a launcher therefore distributes
without an explicit `--comm-backend mpi`.

When the `mpi` feature is compiled in, `create_communicator` returns the
`CommBackend` enum (`CommBackend::Mpi` / `CommBackend::Local`), which
implements `Communicator` (and, with `shared-memory`, `SharedMemoryProvider`)
by delegating each call to the active inner backend via a `match` — the
mandated enum-dispatch pattern for closed variant sets in Cobre, used because
`Communicator`'s generic methods make it non-object-safe (`Box<dyn
Communicator>` does not compile). Without the `mpi` feature,
`create_communicator` returns `LocalBackend` directly, with no wrapper enum.

## `FerrompiBackend`

`FerrompiBackend` wraps an [ferrompi](https://github.com/cobre-rs/ferrompi)
`Mpi` environment handle and an `MPI_COMM_WORLD` communicator. ferrompi is a
separate upstream repository providing safe MPI 4.x bindings for Rust —
type-safe wrappers around the collective operations (`allgatherv`,
`allreduce`, `broadcast`, `barrier`) with an RAII-managed
`MPI_Init_thread`/`MPI_Finalize` lifecycle.

Construction (`FerrompiBackend::new`) calls `MPI_Init_thread` with
`ThreadLevel::Funneled`, matching the Cobre execution model where only the
main thread issues MPI calls; rank/size/topology are cached at construction
so the hot path never re-queries them. When `FerrompiBackend` is dropped, the
RAII guard calls `MPI_Finalize` automatically — field declaration order is
load-bearing (Rust drops fields in reverse declaration order), so the `Mpi`
guard is declared first in the struct to be dropped _last_, after all derived
communicator handles.

`ferrompi::Mpi` is `!Send + !Sync` by design (a `PhantomData<*const ()>`
marker forces `MPI_Init`/`MPI_Finalize` onto the same thread). `FerrompiBackend`
carries `unsafe impl Send + Sync` on top of it: sound because the backend is
the sole owner of `Mpi` until drop (single ownership bars any other thread
from finalizing it), and the training loop's `ThreadLevel::Funneled` model
means every MPI call already originates from the same (main) thread that
constructed the backend. All collective communication otherwise goes through
`ferrompi::Communicator`, an integer handle into a C-side table that is
already `Send + Sync`.

`FerrompiBackend::new()` requires an MPI runtime to be installed; if none is
found it returns `Err(BackendError::InitializationFailed)`.

## `Communicator` trait

Six methods: four fallible collectives, two infallible accessors, and one
non-returning abort.

| Method       | Signature                                                      | Description                                                                                     |
| ------------ | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `allgatherv` | `(&self, send, recv, counts, displs) -> Result<(), CommError>` | Gather variable-length data from all ranks into all ranks                                       |
| `allreduce`  | `(&self, send, recv, op: ReduceOp) -> Result<(), CommError>`   | Element-wise reduction (sum, min, max, or bitwise-or) across all ranks                          |
| `broadcast`  | `(&self, buf, root: usize) -> Result<(), CommError>`           | Copy data from the root rank to all other ranks                                                 |
| `barrier`    | `(&self) -> Result<(), CommError>`                             | Block until all ranks have entered; pure synchronization                                        |
| `rank`       | `(&self) -> usize`                                             | This rank's index (`0..size()`); infallible, cached at construction                             |
| `size`       | `(&self) -> usize`                                             | Total number of ranks; infallible, cached at construction                                       |
| `abort`      | `(&self, error_code: i32) -> !`                                | Terminate every process (`MPI_Abort`, or `std::process::exit` on `LocalBackend`); never returns |

`Communicator` requires `Send + Sync`, all methods take `&self`, and callers
are responsible for serializing concurrent calls — the training loop ensures
that multiple threads never invoke the same collective simultaneously on the
same communicator instance. `rank()` and `size()` are safe to call
concurrently: their values are cached at construction time and never change.

This is the same monomorphization pattern used by `SolverInterface` in
[cobre-solver](../cobre-solver/README.md): callers parameterize a generic
function once and the compiler generates one concrete instantiation per
backend, so `LocalBackend`'s no-op implementations compile to zero
instructions after inlining.

## Error types

`CommError` is returned by all fallible `Communicator` and
`SharedMemoryProvider` methods:

| Variant               | When it occurs                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| `CollectiveFailed`    | An MPI collective operation failed at the library level (carries the MPI error code and a message) |
| `InvalidBufferSize`   | Buffer sizes provided to a collective are inconsistent                                             |
| `InvalidRoot`         | The `root` rank argument is out of range (`root >= size()`)                                        |
| `InvalidCommunicator` | The communicator is in an invalid state (e.g., MPI has been finalized)                             |
| `AllocationFailed`    | A shared-memory allocation request was rejected by the OS                                          |

`BackendError` is returned by `create_communicator` when the backend cannot be
selected or initialized:

| Variant                | When it occurs                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `BackendNotAvailable`  | The requested backend is not compiled into this binary (e.g. `BackendKind::Mpi` without the `mpi` feature)    |
| `InvalidBackend`       | The requested backend name matches no known backend                                                           |
| `InitializationFailed` | The backend was selected but failed to initialize (e.g. MPI runtime not installed)                            |
| `MissingConfiguration` | Required environment variables for the selected backend are not set (reserved for the deferred `tcp` backend) |

## Deferred backends

Two additional backend slots are reserved but not yet implemented: **`tcp`** —
a TCP/IP coordinator pattern for distributed execution without an MPI
installation — and **`shm`** — a true POSIX/MPI-window shared-memory backend
to replace the current `HeapRegion` placeholder. Both would follow the same
`Communicator` / `SharedMemoryProvider` trait interfaces.

## Testing

```
cargo test -p cobre-comm
```

Runs all unit, integration, and doc-tests for the default (no-feature)
configuration — no MPI installation required. To include the MPI backend:

```
cargo test -p cobre-comm --features mpi
```

This requires an MPI runtime (`libmpich-dev` on Debian/Ubuntu, `mpich` on
Fedora or macOS Homebrew). CI runs tests without the `mpi` feature by default.

Integration coverage lives in `tests/local_conformance.rs` (the
backend-agnostic `Communicator` / `SharedMemoryProvider` / `LocalCommunicator`
contract, verified against `LocalBackend` through the public API only) and
`tests/factory_tests.rs` (`create_communicator` / `BackendKind` /
`available_backends` behavior across feature configurations).

## Links

| Resource   | URL                                                        |
| ---------- | ---------------------------------------------------------- |
| Docs site  | <https://docs.cobre-rs.dev/>                               |
| API Docs   | <https://docs.rs/cobre-comm/latest/cobre_comm/>            |
| Repository | <https://github.com/cobre-rs/cobre>                        |
| CHANGELOG  | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable.

## License

Apache-2.0
