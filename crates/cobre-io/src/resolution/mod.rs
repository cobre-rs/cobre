//! Resolution functions that apply the three-tier penalty and bound cascades.
//!
//! Each module takes parsed entity data (tier-2 already resolved) and sparse
//! stage-varying override rows (tier-3 from Parquet) and produces fully
//! pre-resolved [`cobre_core::resolved`] tables ready for O(1) solver lookup.

pub mod bounds;
pub mod exchange_factors;
pub mod generic_bounds;
pub mod load_factors;
pub mod ncs_bounds;
pub mod ncs_factors;
pub mod penalties;

pub use bounds::{resolve_bounds, BoundsEntitySlices, BoundsOverrides};
pub use exchange_factors::resolve_exchange_factors;
pub use generic_bounds::resolve_generic_constraint_bounds;
pub use load_factors::resolve_load_factors;
pub use ncs_bounds::resolve_ncs_bounds;
pub use ncs_factors::resolve_ncs_factors;
pub use penalties::{resolve_penalties, PenaltiesEntitySlices, PenaltiesOverrides};
