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
//!   the GIL released via `py.detach()`, allowing worker threads
//!   within `cobre-sddp` to run at full parallelism.
//! - **No Python callbacks in the hot loop** — all customization is
//!   via configuration structs, not Python callables.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.

use pyo3::prelude::*;

mod io;
mod model;
mod results;
mod run;

/// Sub-module containing data model types for the Cobre power systems solver.
#[pymodule]
#[pyo3(name = "model")]
fn model_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Data model types for the Cobre power systems solver.",
    )?;
    m.add_class::<model::PySystem>()?;
    m.add_class::<model::PyBus>()?;
    m.add_class::<model::PyLine>()?;
    m.add_class::<model::PyThermal>()?;
    m.add_class::<model::PyHydro>()?;
    m.add_class::<model::PyEnergyContract>()?;
    m.add_class::<model::PyPumpingStation>()?;
    m.add_class::<model::PyNonControllableSource>()?;
    Ok(())
}

/// Sub-module containing I/O helpers for loading Cobre case directories.
#[pymodule]
#[pyo3(name = "io")]
fn io_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "I/O helpers for loading Cobre case directories.")?;
    m.add_function(wrap_pyfunction!(io::load_case, m)?)?;
    m.add_function(wrap_pyfunction!(io::validate, m)?)?;
    Ok(())
}

/// Sub-module containing solver execution entry points.
#[pymodule]
#[pyo3(name = "run")]
fn run_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Solver execution entry points for training and simulation.",
    )?;
    m.add_function(wrap_pyfunction!(run::run, m)?)?;
    Ok(())
}

/// Sub-module containing result loading and inspection functions.
#[pymodule]
#[pyo3(name = "results")]
fn results_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Result loading and inspection functions for Cobre output artifacts.",
    )?;
    m.add_function(wrap_pyfunction!(results::load_results, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_convergence, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_convergence_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_simulation_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_policy, m)?)?;
    Ok(())
}

/// Register a submodule in `sys.modules` so that `import cobre.foo` works.
///
/// PyO3's `add_submodule` attaches a module as an attribute but does not
/// register it in `sys.modules`. Without this, `import cobre.foo` fails.
fn register_submodule<'py>(
    parent: &Bound<'py, PyModule>,
    child: &Bound<'py, PyModule>,
    parent_package: &str,
) -> PyResult<()> {
    let child_name = child.name()?;
    let full_name = format!("{parent_package}.{child_name}");
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item(full_name, child)?;
    parent.add_submodule(child)
}

/// Python bindings for the Cobre power systems solver. Single-process only -- for distributed execution, launch `mpiexec cobre` as a subprocess.
#[pymodule]
fn cobre(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    register_submodule(m, pyo3::wrap_pymodule!(model_module)(py).bind(py), "cobre")?;
    register_submodule(m, pyo3::wrap_pymodule!(io_module)(py).bind(py), "cobre")?;
    register_submodule(m, pyo3::wrap_pymodule!(run_module)(py).bind(py), "cobre")?;
    register_submodule(
        m,
        pyo3::wrap_pymodule!(results_module)(py).bind(py),
        "cobre",
    )?;

    Ok(())
}
