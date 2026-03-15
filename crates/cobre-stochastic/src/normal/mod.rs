//! Normal (i.i.d. Gaussian) noise model building blocks.
//!
//! This module provides precomputed data structures for entities following a
//! simple normal noise model: `x = μ + σ·ε`, where `ε ~ N(0, 1)`.
//!
//! ## Submodules
//!
//! - [`precompute`] — derives LP-ready mean, standard deviation, and block
//!   factor arrays from raw model parameters, indexed for O(1) access during
//!   iterative optimization.

pub mod precompute;

pub use precompute::{BlockFactorPair, EntityFactorEntry, PrecomputedNormal};
