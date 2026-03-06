//! Correlation matrix construction and factorisation.
//!
//! This module handles the two steps needed to apply spatial correlation
//! between inflow series during scenario generation:
//!
//! 1. **Resolution** ([`resolve`]) — maps per-profile correlation data
//!    from the input case to dense matrices indexed by internal hydro IDs.
//! 2. **Cholesky factorisation** ([`cholesky`]) — decomposes a resolved
//!    correlation matrix into its lower-triangular Cholesky factor, which
//!    is used to transform independent standard-normal draws into
//!    spatially correlated noise vectors.

pub mod cholesky;
pub mod resolve;
