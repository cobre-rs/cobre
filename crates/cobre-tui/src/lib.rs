//! # cobre-tui
//!
//! Interactive terminal UI for the [Cobre](https://github.com/cobre-rs/cobre) power systems solver.
//!
//! Provides real-time training monitoring, convergence visualization, cut
//! inspection, and simulation progress tracking using `ratatui` and `crossterm`.
//! Consumed by `cobre-cli` as a library.
//!
//! ## Consumption modes
//!
//! - **Co-hosted** — in-process broadcast channel subscription within the
//!   `cobre` binary, activated by `cobre run --tui`. Renders on rank 0
//!   alongside the training loop.
//! - **Standalone pipe** — reads JSON-lines from stdin, e.g.:
//!   `mpiexec cobre run ... --output-format json-lines | cobre-tui`
//!   Enables monitoring of remote or already-running jobs.
//!
//! Both modes consume the same event types defined in `cobre-core`.
//!
//! ## Design principles
//!
//! - **Depends only on `cobre-core`** for event type definitions. No solver,
//!   IO, or stochastic dependencies.
//! - **Iteration-boundary safety**: all interactive operations (pause, inspect,
//!   adjust stopping rules) operate at iteration boundaries only.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.
