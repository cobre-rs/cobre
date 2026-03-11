//! I/O helpers for loading Cobre case directories from Python.
//!
//! Exposes [`load_case`] and [`validate`] in the `cobre.io` sub-module.
//! These are the primary entry points for Python scripts and Jupyter notebooks
//! that need to read and inspect Cobre power-system cases.
//!
//! ## Error mapping
//!
//! [`cobre_io::LoadError`] variants are converted to Python exceptions as follows:
//!
//! | Rust variant                        | Python exception |
//! |-------------------------------------|-----------------|
//! | `LoadError::IoError`                | `OSError`       |
//! | `LoadError::ParseError`             | `ValueError`    |
//! | `LoadError::SchemaError`            | `ValueError`    |
//! | `LoadError::CrossReferenceError`    | `ValueError`    |
//! | `LoadError::ConstraintError`        | `ValueError`    |
//! | `LoadError::PolicyIncompatible`     | `ValueError`    |
//!
//! The [`validate`] function never raises — errors are returned as data in a
//! Python dict so that callers see all problems at once.

use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cobre_io::LoadError;

use crate::model::PySystem;

// ── Error conversion ──────────────────────────────────────────────────────────

/// Convert a [`LoadError`] to the appropriate Python exception.
///
/// The mapping preserves as much context as possible in the exception message
/// without exposing Rust-internal type names in the Python API.
fn convert_load_error(err: &LoadError) -> PyErr {
    match err {
        LoadError::IoError { .. } => PyOSError::new_err(err.to_string()),
        LoadError::ParseError { .. }
        | LoadError::SchemaError { .. }
        | LoadError::CrossReferenceError { .. }
        | LoadError::ConstraintError { .. }
        | LoadError::PolicyIncompatible { .. } => PyValueError::new_err(err.to_string()),
    }
}

// ── load_case ────────────────────────────────────────────────────────────────

/// Load a Cobre case directory and return a validated `System`.
///
/// Executes the five-layer validation pipeline (structural, schema, referential
/// integrity, dimensional consistency, and semantic). Returns a fully-validated
/// `cobre.model.System` on success or raises a Python exception on failure.
///
/// # Arguments
///
/// * `path` — path to the case directory, as a `str` or `pathlib.Path`.
///   Relative paths are resolved from the process working directory.
///
/// # Raises
///
/// * `OSError` — a required file is missing or cannot be read.
/// * `ValueError` — the case data fails schema, referential integrity,
///   dimensional consistency, or semantic validation.
///
/// # Examples
///
/// ```python
/// import cobre.io
/// system = cobre.io.load_case("examples/1dtoy")
/// print(system.n_buses)
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn load_case(path: PathBuf) -> PyResult<PySystem> {
    let system = cobre_io::load_case(&path).map_err(|e| convert_load_error(&e))?;
    Ok(PySystem::from_rust(system))
}

// ── validate ─────────────────────────────────────────────────────────────────

/// Validate a Cobre case directory and return a structured report dict.
///
/// Unlike [`load_case`], this function **never raises** — all errors are
/// returned as data in the result dict. This is intentional: Jupyter workflows
/// need to see all validation problems at once rather than stopping at the
/// first failure.
///
/// # Arguments
///
/// * `path` — path to the case directory, as a `str` or `pathlib.Path`.
///
/// # Returns
///
/// A dict with the following keys:
///
/// * `"valid"` (`bool`) — `True` when the case loaded without errors.
/// * `"errors"` (`list[dict]`) — list of error dicts, each with `"kind"` and
///   `"message"` string fields. Empty when `valid` is `True`.
/// * `"warnings"` (`list[dict]`) — list of warning dicts in the same format.
///   Warnings do not affect the `valid` flag.
///
/// # Examples
///
/// ```python
/// import cobre.io
/// result = cobre.io.validate("examples/1dtoy")
/// assert result["valid"] is True
/// assert result["errors"] == []
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn validate(py: Python<'_>, path: PathBuf) -> PyResult<PyObject> {
    let result = cobre_io::load_case(&path);

    let dict = PyDict::new(py);

    match result {
        Ok(_) => {
            dict.set_item("valid", true)?;
            dict.set_item("errors", PyList::empty(py))?;
            dict.set_item("warnings", PyList::empty(py))?;
        }
        Err(err) => {
            dict.set_item("valid", false)?;

            let error_entry = PyDict::new(py);
            let kind = match &err {
                LoadError::IoError { .. } => "IoError",
                LoadError::ParseError { .. } => "ParseError",
                LoadError::SchemaError { .. } => "SchemaError",
                LoadError::CrossReferenceError { .. } => "CrossReferenceError",
                LoadError::ConstraintError { .. } => "ConstraintError",
                LoadError::PolicyIncompatible { .. } => "PolicyIncompatible",
            };
            error_entry.set_item("kind", kind)?;
            error_entry.set_item("message", err.to_string())?;

            let errors = PyList::new(py, [error_entry.as_any()])?;
            dict.set_item("errors", errors)?;
            dict.set_item("warnings", PyList::empty(py))?;
        }
    }

    Ok(dict.into())
}
