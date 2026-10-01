//! LP-construction cluster: the column/row index map and the stage-template
//! builder that together turn a loaded `System` into the structural stage
//! LPs every pass solves.
//!
//! This directory module groups the two pieces that own the LP's structure,
//! kept together because each depends on the column layout the next encodes:
//!
//! - [`indexer`] — [`StateSpace`](indexer::StateSpace) owns the state-vector
//!   column layout and [`StudyDimensions`](indexer::StudyDimensions) the
//!   non-state study shape. The LP has no state-fixing row range: state is
//!   pinned via [`crate::indexer::StateSpace::state_to_lp_incoming_column`]
//!   column bounds, never a fixing row. The per-stage equipment geometry lives
//!   on [`StageGeometry`].
//! - [`builder`] — [`build_stage_templates`](builder::build_stage_templates) assembles the CSC structural LP,
//!   bounds, and objective for each stage once at startup; its crate-private
//!   `generic_constraints` submodule lowers user-declared generic constraints
//!   onto the indexed column layout. The FPHA generation constraint carries
//!   the `−γᵥ/2` coefficient on **both** storage columns (the FPHA
//!   average-storage contract — see [`builder`]).

#![deny(clippy::allow_attributes, clippy::allow_attributes_without_reason)]

pub mod builder;
pub mod indexer;

pub use builder::{StageGeometry, StageTemplates};
