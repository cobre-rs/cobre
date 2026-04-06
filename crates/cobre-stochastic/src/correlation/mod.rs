//! Correlation matrix construction and spectral factorisation.
//!
//! This module handles the two steps needed to apply spatial correlation
//! between inflow series during scenario generation:
//!
//! 1. **Resolution** ([`resolve`]) — maps per-profile correlation data
//!    from the input case to dense matrices indexed by internal hydro IDs.
//! 2. **Spectral factorisation** ([`spectral`]) — eigendecomposes a resolved
//!    correlation matrix to produce the symmetric matrix square root,
//!    which is used to transform independent standard-normal draws into
//!    spatially correlated noise vectors.

pub mod resolve;
pub mod spectral;

pub use resolve::{DecomposedCorrelation, GroupFactor};
pub use spectral::SpectralFactor;
