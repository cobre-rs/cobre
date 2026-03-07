//! Cut management data structures for the SDDP Future Cost Function (FCF).
//!
//! This module provides the per-stage cut pool, the all-stages FCF container,
//! the wire format for MPI exchange, and supporting types used to store,
//! query, and prune Benders cuts during the SDDP training loop.
//!
//! ## Contents
//!
//! - [`pool`] — [`CutPool`]: pre-allocated per-stage cut storage with
//!   deterministic slot assignment, activity tracking, and state evaluation.
//! - [`fcf`] — [`FutureCostFunction`]: all-stages container wrapping one
//!   [`CutPool`] per stage; provides the high-level API for the training loop.
//! - [`wire`] — [`CutWireHeader`] and serialization functions for the MPI
//!   cut-exchange wire format (24-byte header + variable coefficient tail).

pub mod fcf;
pub mod pool;
pub mod wire;

pub use fcf::FutureCostFunction;
pub use pool::CutPool;
pub use wire::CutWireHeader;
