//! Resolution functions that apply the three-tier penalty and bound cascades.
//!
//! Each module takes parsed entity data (tier-2 already resolved) and sparse
//! stage-varying override rows (tier-3 from Parquet) and produces fully
//! pre-resolved [`cobre_core::resolved`] tables ready for O(1) solver lookup.

pub mod bounds;
pub mod generic_bounds;
pub mod penalties;

pub use bounds::resolve_bounds;
pub use generic_bounds::resolve_generic_constraint_bounds;
pub use penalties::resolve_penalties;
