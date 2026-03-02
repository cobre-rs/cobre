//! Top-level system struct and builder.
//!
//! The `System` struct is the top-level in-memory representation of a fully loaded,
//! validated, and resolved case. It is produced by `cobre-io::load_case()` and consumed
//! by `cobre-sddp::train()`, `cobre-sddp::simulate()`, and `cobre-stochastic` scenario
//! generation.
//!
//! All entity collections in `System` are stored in canonical ID-sorted order to ensure
//! declaration-order invariance: results are bit-for-bit identical regardless of input
//! entity ordering. See the design principles spec for details.
