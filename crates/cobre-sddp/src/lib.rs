//! # cobre-sddp
//!
//! Stochastic Dual Dynamic Programming for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate implements the SDDP algorithm (Pereira & Pinto, 1991) for
//! long-term hydrothermal dispatch and energy planning:
//!
//! - **Forward pass**: scenario-based simulation with policy evaluation.
//! - **Backward pass**: Benders cut generation for the cost-to-go function.
//! - **Cut management**: single-cut and multi-cut strategies, cut selection,
//!   and dominance pruning.
//! - **Risk measures**: expected value, CVaR, and convex combinations.
//! - **Convergence**: statistical stopping criteria and bound gap monitoring.
//! - **Parallelism**: designed for hybrid MPI + thread-level parallelism
//!   via [ferrompi](https://github.com/cobre-rs/ferrompi).
//!
//! Built on `cobre-core` for system data, `cobre-stochastic` for inflow
//! modeling, and `cobre-solver` for LP subproblems.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.
