//! Deterministic noise generation for scenario construction.
//!
//! This module produces reproducible, communication-free noise sequences
//! using a combination of deterministic seed derivation and a fast PRNG.
//! Each scenario and stage receives a unique seed derived from a global
//! base seed via SipHash-1-3, enabling communication-free parallel noise
//! generation: any subset of scenarios can be generated independently on
//! any compute node without inter-process coordination.
//!
//! ## Submodules
//!
//! - [`seed`] — derives per-scenario, per-stage seeds from a base seed
//!   using SipHash-1-3
//! - [`rng`] — wraps the derived seeds in a `Pcg64` generator and samples
//!   standard-normal noise vectors

pub mod quantile;
pub mod rng;
pub mod seed;
