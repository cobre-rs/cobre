//! # cobre-core
//!
//! Shared data model for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate defines the fundamental types used across all Cobre tools:
//! buses, branches, generators (hydro, thermal, renewable), loads, and
//! network topology. A power system defined with `cobre-core` types can be
//! used for SDDP optimization, power flow analysis, dynamic simulation,
//! and any future solver in the ecosystem.
//!
//! ## Design principles
//!
//! - **Solver-agnostic**: no solver or algorithm dependencies.
//! - **Validate at construction**: invalid states are caught when building
//!   the system, not at solve time.
//! - **Shared types**: a `HydroPlant` is the same struct whether used in
//!   stochastic dispatch or steady-state analysis.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.
