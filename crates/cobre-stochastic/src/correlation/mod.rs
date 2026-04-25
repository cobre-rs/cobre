//! Correlation matrix construction and spectral factorisation.
//!
//! This module handles the two steps needed to apply spatial correlation
//! between stochastic series during scenario generation. Three entity classes
//! are supported: inflow (hydro plants), load (buses), and non-controllable
//! sources (NCS). Each class carries its own correlation group, resolved
//! independently to a dense matrix indexed by internal entity IDs.
//!
//! 1. **Resolution** ([`resolve`]) — maps per-profile correlation data
//!    from the input case to dense matrices indexed by internal entity IDs
//!    (hydro, bus, or NCS depending on the entity class).
//! 2. **Spectral factorisation** ([`spectral`]) — eigendecomposes a resolved
//!    correlation matrix to produce the symmetric matrix square root,
//!    which is used to transform independent standard-normal draws into
//!    spatially correlated noise vectors.

pub mod resolve;
pub mod spectral;

pub use resolve::{DecomposedCorrelation, GroupFactor};
pub use spectral::SpectralFactor;
