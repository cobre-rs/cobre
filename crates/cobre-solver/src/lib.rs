//! # cobre-solver
//!
//! LP/MIP solver abstraction for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate defines a backend-agnostic interface for mathematical programming
//! solvers, with a default [HiGHS](https://highs.dev) backend:
//!
//! - **Solver trait**: unified API for LP and MIP problem construction, solving,
//!   and dual/basis extraction.
//! - **`HiGHS` backend**: production-grade open-source solver, well-suited for
//!   iterative LP solving in power system optimization.
//! - **Basis management**: warm-starting support for iterative algorithms
//!   that solve sequences of related LPs.
//!
//! Additional solver backends (e.g., Clp, CPLEX, Gurobi) can be added
//! behind feature flags.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.

// Relax strict production lints for test builds. These lints (unwrap_used,
// expect_used, etc.) guard library code but are normal in tests.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::panic,
        clippy::too_many_lines
    )
)]

#[cfg(feature = "highs")]
mod ffi;

pub mod trait_def;
pub use trait_def::SolverInterface;

pub mod types;
pub use types::{
    Basis, BasisStatus, LpSolution, RowBatch, SolverError, SolverStatistics, StageTemplate,
};

#[cfg(feature = "highs")]
pub mod highs;
#[cfg(feature = "highs")]
pub use highs::HighsSolver;
