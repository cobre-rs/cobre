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
//!
//! ## Sentinel value
//!
//! [`WARM_START_ITERATION`] is the sentinel stored in
//! [`CutMetadata::iteration_generated`](crate::cut_selection::CutMetadata::iteration_generated) for every cut loaded from a policy
//! checkpoint.  Cut selection strategies may inspect this sentinel to apply
//! warm-start-specific pruning policies (e.g., exempt warm-start cuts from
//! LML1 deactivation).

pub mod fcf;
pub mod pool;
pub mod row_map;
pub mod sparse;
pub mod wire;

pub use fcf::FutureCostFunction;
pub use pool::{CutPool, SparsityReport};
pub use row_map::CutRowMap;
pub use sparse::SparseCut;
pub use wire::CutWireHeader;

/// Sentinel value stored in [`crate::cut_selection::CutMetadata::iteration_generated`]
/// for warm-start cuts loaded from a policy checkpoint.
///
/// Set to [`u64::MAX`] so that `WARM_START_ITERATION != current_iteration` is
/// always true for any valid training iteration, allowing cut selection
/// strategies to distinguish warm-start cuts from training-generated cuts
/// without special casing.
pub const WARM_START_ITERATION: u64 = u64::MAX;
