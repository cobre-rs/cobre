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
    Ok(())
}

/// Register a submodule and make it importable via `sys.modules`.
///
/// `PyO3`'s `add_submodule` attaches a module as an attribute but does not
/// register it in `sys.modules`. Without this registration, `import cobre.foo`
/// fails with `ModuleNotFoundError` even though `cobre.foo` is accessible as
/// an attribute. See <https://pyo3.rs/v0.23.0/module#python-submodules>.
///
/// The `parent_package` parameter is the top-level package name as Python sees
/// it (e.g. `"cobre"`), not the value of `parent.name()` which may differ when
/// the entry-point function shares the crate's lib name.
fn register_submodule<'py>(
    parent: &Bound<'py, PyModule>,
    child: &Bound<'py, PyModule>,
    parent_package: &str,
) -> PyResult<()> {
    let py = parent.py();
    parent.add_submodule(child)?;
    let child_name = child.name()?;
    let full_name = format!("{parent_package}.{child_name}");
    py.import("sys")?
        .getattr("modules")?
        .set_item(full_name, child)?;
    Ok(())
}

/// Python bindings for the Cobre power systems solver. Single-process only -- for distributed execution, launch `mpiexec cobre` as a subprocess.
#[pymodule]
fn cobre(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let model_mod = pyo3::wrap_pymodule!(model_module)(py);
    register_submodule(m, model_mod.bind(py), "cobre")?;

    let io_mod = pyo3::wrap_pymodule!(io_module)(py);
    register_submodule(m, io_mod.bind(py), "cobre")?;

    let run_mod = pyo3::wrap_pymodule!(run_module)(py);
    register_submodule(m, run_mod.bind(py), "cobre")?;

    let results_mod = pyo3::wrap_pymodule!(results_module)(py);
    register_submodule(m, results_mod.bind(py), "cobre")?;

    Ok(())
}
