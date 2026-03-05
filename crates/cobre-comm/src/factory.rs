//! Factory function for creating the active communication backend.
//!
//! [`create_communicator`] is the single entry point for constructing a
//! [`Communicator`](crate::Communicator) instance at runtime. It selects the
//! backend according to:
//!
//! 1. The `COBRE_COMM_BACKEND` environment variable (runtime override).
//! 2. The Cargo feature flags compiled into the binary (`mpi`, `tcp`, `shm`).
//! 3. A fallback to [`LocalBackend`](crate::local) when no distributed backend
//!    is available.
//!
//! This indirection keeps `cobre-sddp` decoupled from any specific backend: the
//! training loop only sees the `Communicator` trait, never a concrete type.
