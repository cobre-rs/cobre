//! Per-thread solver workspaces and the immutable hot-path context structs.
//!
//! This directory clusters the two read/write halves of what a stage solve
//! threads through its functions:
//!
//! - [`workspace`] — the mutable per-thread resources: [`SolverWorkspace`]
//!   bundles one thread's LP-solve scratch, [`WorkspacePool`] hands out one per
//!   worker, [`BasisStore`] holds per-scenario warm-start bases, and
//!   [`CapturedBasis`] is the slot-tracked basis whose
//!   [`to_broadcast_payload`](workspace::CapturedBasis::to_broadcast_payload) /
//!   [`try_from_broadcast_payload`](workspace::CapturedBasis::try_from_broadcast_payload)
//!   are the sole owners of the MPI basis-cache wire format.
//! - [`context`] — the immutable per-stage / per-training parameter bundles
//!   ([`StageContext`], [`TrainingContext`])
//!   passed by reference to keep hot-path argument counts down.

pub mod context;
// Rationale: renaming this submodule off its parent's name would break the
// `workspace::workspace::{...}` re-export path for no behavioural gain.
#[allow(clippy::module_inception)]
pub mod workspace;

// Rationale: `BackwardAccumulators` is reached only from in-crate `#[cfg(test)]`
// modules, so the re-export reads as unused in the non-test build; scoping the
// allow to `cfg(not(test))` keeps the warning live should a non-test caller land.
pub use context::{StageContext, TrainingContext};
pub use workspace::{
    BASIS_BROADCAST_FORMAT_TAG, BasisStore, BasisStoreSliceMut, CapturedBasis, NoisePreallocation,
    ScratchBuffers, SolverWorkspace, WorkspacePool, WorkspaceSizing,
};
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use workspace::{BackwardAccumulators, ByNodeScratch};
