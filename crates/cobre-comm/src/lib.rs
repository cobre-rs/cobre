//! # cobre-comm
//!
//! Pluggable communication backend abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
//! distributed SDDP solver.
//!
//! This crate defines the `Communicator` and `SharedMemoryProvider` traits
//! that decouple the SDDP training loop from specific communication technologies.
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
//! - **Orthogonal to SDDP**: `cobre-sddp` depends on this crate for
//!   communication; it does not depend back.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.
