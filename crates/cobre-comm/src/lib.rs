//! # cobre-comm
//!
//! Pluggable communication backend abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
//! ecosystem.
//!
//! This crate defines the [`Communicator`] trait that decouples distributed
//! computations from specific communication technologies. Backend implementations
//! are selected by Cargo feature flags:
//!
//! - **`local`** — single-process backend, always available without any feature
//!   flag. [`LocalBackend`] is a zero-sized type with identity collective
//!   semantics and zero overhead.
//! - **`mpi`** — MPI collectives via [ferrompi](https://github.com/cobre-rs/ferrompi)
//!   (Cargo feature). Provides the [`FerrompiBackend`] and the [`CommBackend`]
//!   enum-dispatched wrapper.
//! - **`affinity`** — discovers the CPU/NUMA resources visible to a Linux
//!   process and supports opt-in worker-to-CPU binding. The public API remains
//!   available without the feature and returns an explicit unsupported error.
//! - **`numa`** — extends `mpi` with NUMA-aware topology information from the
//!   ferrompi NUMA extension and enables `affinity` (Cargo feature; implies
//!   `mpi`).
//! - **`shared-memory`** — experimental intra-node shared-memory region API
//!   (Cargo feature, off by default). Exposes [`SharedMemoryProvider`],
//!   [`SharedRegion`], [`LocalCommunicator`], and [`HeapRegion`]. The current
//!   implementation is a **heap-backed placeholder**: each rank allocates its
//!   own [`Vec<T>`] with no memory shared across processes. The real
//!   implementation should be built on ferrompi's `SharedWindow<T>`
//!   (`MPI_Win_allocate_shared` / `MPI_Win_shared_query`); see the tracking
//!   note in `traits.rs` for open design tensions.
//!
//! Available backends are determined at build time via Cargo feature flags; the
//! active backend is chosen at runtime by the explicit [`BackendKind`] passed to
//! [`create_communicator`].
//!
//! ## Design principles
//!
//! - **Zero-cost static dispatch**: generics over [`Communicator`] eliminate
//!   dynamic dispatch overhead on the hot path.
//! - **Additive features**: multiple backends can coexist in one binary.
//! - **Orthogonal to the solver**: algorithm crates depend on this crate for
//!   communication; it does not depend back.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the current status.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod affinity;
mod factory;
mod local;
pub mod topology;
mod traits;
mod types;

#[cfg(feature = "mpi")]
mod ferrompi;

pub use affinity::{
    AffinityError, AffinityPolicy, AffinityReport, CpuTopology, MemoryBinding, NumaNode,
    ProcessingUnit, WorkerAffinity, current_thread_cpu_set,
};
pub use factory::{BackendKind, available_backends, create_communicator};
pub use local::LocalBackend;
pub use topology::{ExecutionTopology, HostInfo, MpiRuntimeInfo, SlurmJobInfo};
pub use traits::{CommData, Communicator, TopologyProvider};
pub use types::{BackendError, CommError, ReduceOp};

#[cfg(feature = "shared-memory")]
pub use local::HeapRegion;

#[cfg(feature = "shared-memory")]
pub use traits::{LocalCommKind, LocalCommunicator, SharedMemoryProvider, SharedRegion};

#[cfg(feature = "mpi")]
pub use ferrompi::FerrompiBackend;

#[cfg(all(feature = "mpi", feature = "shared-memory"))]
pub use ferrompi::FerrompiLocalComm;

#[cfg(feature = "mpi")]
pub use factory::CommBackend;
