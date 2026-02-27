//! # cobre-python
//!
//! Python bindings for the [Cobre](https://github.com/cobre-rs/cobre) power systems solver.
//!
//! Exposes the Cobre solver as a Python extension module (`import cobre`),
//! providing programmatic access to case loading, validation, training,
//! simulation, and result inspection from Python scripts, Jupyter notebooks,
//! and orchestration frameworks.
//!
//! ## Constraints
//!
//! - **Single-process only** — this crate MUST NOT initialize MPI or depend
//!   on `ferrompi`. The GIL/MPI incompatibility makes it unsafe to combine
//!   MPI initialization with Python embedding. For distributed execution,
//!   launch `mpiexec cobre` as a subprocess.
//! - **GIL released during computation** — all Rust computation runs with
//!   the GIL released via `py.allow_threads()`, allowing OpenMP threads
//!   within `cobre-sddp` to run at full parallelism.
//! - **No Python callbacks in the hot loop** — all customization is
//!   via configuration structs, not Python callables.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.
