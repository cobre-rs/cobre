//! Local (single-process) communication backend.
//!
//! `LocalBackend` is always available without any feature flags. It implements
//! [`Communicator`](crate::Communicator) as a no-op suitable for single-process
//! execution and unit testing:
//!
//! - `rank()` always returns `0`.
//! - `size()` always returns `1`.
//! - `allreduce` returns the local value unchanged (identity for single-process).
//! - `broadcast` is a no-op (data is already at the only rank).
//! - `barrier` is a no-op.
//!
//! This backend imposes zero overhead and zero external dependencies.
