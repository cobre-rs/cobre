//! Deterministic seed derivation via SipHash-1-3.
//!
//! Derives a unique `u64` seed for a given (base seed, scenario index,
//! stage index) triple using SipHash-1-3. The derivation is fully
//! deterministic and requires no inter-process communication, enabling
//! each compute node to generate its assigned subset of scenarios
//! independently (DEC-017).
//!
//! Returns [`StochasticError::SeedDerivationError`] if the hash
//! computation cannot produce a valid seed.
//!
//! [`StochasticError::SeedDerivationError`]: crate::StochasticError::SeedDerivationError
