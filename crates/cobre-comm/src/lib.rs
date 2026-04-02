//! # cobre-comm
//!
//! Pluggable communication backend abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
//! ecosystem.
//!
//! This crate defines the `Communicator` and `SharedMemoryProvider` traits
//! that decouple distributed computations from specific communication technologies.
//! Backend implementations are feature-gated:
//!
//! - `local` — single-process no-op (always available, zero overhead)
//! - `mpi` — MPI collectives via [ferrompi](https://github.com/cobre-rs/ferrompi)
//! - `tcp` — TCP/IP coordinator pattern (no MPI required)
//! - `shm` — POSIX shared memory for single-node multi-process execution
//!
//! The backend is selected at build time via Cargo feature flags, with optional
//! runtime override through the `COBRE_COMM_BACKEND` environment variable.
//!
//! ## Design principles
//!
//! - **Zero-cost static dispatch**: generics over `Communicator` eliminate
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

// Allow unwrap/expect in tests — these lints guard library code but are normal in test contexts.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod factory;
mod local;
mod traits;
mod types;

#[cfg(feature = "mpi")]
mod ferrompi;

pub use factory::{available_backends, create_communicator, BackendKind};
pub use local::{HeapRegion, LocalBackend};
pub use traits::{CommData, Communicator, LocalCommunicator, SharedMemoryProvider, SharedRegion};
pub use types::{BackendError, CommError, ReduceOp};

#[cfg(feature = "mpi")]
pub use ferrompi::FerrompiBackend;

#[cfg(any(feature = "mpi", feature = "tcp", feature = "shm"))]
pub use factory::CommBackend;
