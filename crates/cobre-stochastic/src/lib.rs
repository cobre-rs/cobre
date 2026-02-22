//! # cobre-stochastic
//!
//! Stochastic process models for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate provides the probabilistic building blocks used in stochastic
//! optimization of power systems:
//!
//! - **PAR(p)**: Periodic Autoregressive models for inflow time series,
//!   following the methodology used in the Brazilian power sector.
//! - **Scenario generation**: correlated multi-variate sampling for
//!   forward simulation in SDDP.
//! - **Monte Carlo**: simulation infrastructure for policy evaluation.
//!
//! Designed to be solver-agnostic — the stochastic models can feed into
//! SDDP, robust optimization, or any scenario-based method.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.
