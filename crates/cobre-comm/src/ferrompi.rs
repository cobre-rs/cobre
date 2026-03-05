//! MPI communication backend powered by [ferrompi](https://github.com/cobre-rs/ferrompi).
//!
//! `FerrompiBackend` implements [`Communicator`](crate::Communicator) using
//! MPI 4.x collective operations via the `ferrompi` crate. It is only available
//! when the `mpi` Cargo feature is enabled.
//!
//! Key characteristics:
//!
//! - Wraps `ferrompi::World` for rank/size queries and collective operations.
//! - Supports persistent collectives (MPI 4.x) for iterative algorithms such as
//!   the SDDP forward/backward pass, reducing per-call setup overhead by 10–30 %.
//! - Nonblocking collectives allow overlap of communication with local LP solves.
//!
//! # Feature gate
//!
//! This module is compiled only when `features = ["mpi"]` is specified in
//! `Cargo.toml` (which activates the `ferrompi` optional dependency).
