//! I/O helpers for loading Cobre case directories from Python.
//!
//! Exposes [`load_case`] and [`validate`] in the `cobre.io` sub-module.
//! These are the primary entry points for Python scripts and Jupyter notebooks
//! that need to read and inspect Cobre power-system cases.
//!
//! ## Error mapping
//!
//! [`cobre_io::LoadError`] variants are routed through the single
//! [`crate::errors::convert_error`] mapping site to the `cobre.errors` hierarchy
//! (each leaf subclasses the matching builtin, so `except OSError` /
//! `except ValueError` keeps catching):
//!
//! | Rust variant                        | Python exception (subclass of) |
//! |-------------------------------------|--------------------------------|
//! | `LoadError::IoError`                | `CaseIoError` (`OSError`)       |
//! | `LoadError::ParseError`             | `ValidationError` (`ValueError`) |
//! | `LoadError::SchemaError`            | `ValidationError` (`ValueError`) |
//! | `LoadError::ConstraintError`        | `ValidationError` (`ValueError`) |
//!
//! [`validate`] returns case-validation failures as data; a malformed
//! `config_overrides` dict raises `ValueError` at call time.

use std::path::PathBuf;

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cobre_io::Config;
use cobre_io::LoadError;
use cobre_io::ReportEntry;
use cobre_io::parse_config;
use cobre_io::validate_case_with_artifacts;
use cobre_sddp::validate_phases::{self, PolicyLoadFailure, ValidateRequest, validate_study};

use crate::convert::pydict_to_json_map;
use crate::errors::ErrorSource::Load;
use crate::errors::convert_error;
use crate::model::PySystem;
use crate::study::resolve_output_dir;

// ── Error conversion ──────────────────────────────────────────────────────────

/// Loads and validates config, deep-merging `overrides` when present
/// (validated identically to an edited `config.json`).
fn load_validate_config(
    config_path: &std::path::Path,
    overrides: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<cobre_io::Config, LoadError> {
    match overrides {
        Some(map) if !map.is_empty() => {
            let raw =
                std::fs::read_to_string(config_path).map_err(|e| LoadError::io(config_path, e))?;
            let base: serde_json::Value = serde_json::from_str(&raw)
                .map_err(|e| LoadError::parse(config_path, e.to_string()))?;
            Config::with_overrides(&base, map)
        }
        _ => parse_config(config_path),
    }
}

fn convert_load_error(err: &LoadError) -> PyErr {
    convert_error(Load(err))
}

fn build_warnings_list<'py>(
    py: Python<'py>,
    warnings: &[ReportEntry],
) -> PyResult<Bound<'py, PyList>> {
    let warnings_list = PyList::empty(py);
    for entry in warnings {
        let w = PyDict::new(py);
        w.set_item("kind", &entry.kind)?;
        w.set_item("message", &entry.message)?;
        w.set_item("file", &entry.file)?;
        w.set_item("entity", entry.entity.as_deref())?;
        warnings_list.append(w)?;
    }
    Ok(warnings_list)
}

// ── load_case ────────────────────────────────────────────────────────────────

/// Load a Cobre case directory and return a validated `System`.
///
/// Executes the six-layer validation pipeline (structural, schema, referential
/// integrity, dimensional consistency, semantic, and cross-file resolution).
/// Returns a fully-validated `cobre.model.System` on success or raises a Python
/// exception on failure.
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
pub fn load_case(py: Python<'_>, path: PathBuf) -> PyResult<PySystem> {
    if !path.exists() {
        return Err(PyOSError::new_err(format!(
            "case directory does not exist: {}",
            path.display()
        )));
    }
    let system = py
        .detach(|| cobre_io::load_case(&path))
        .map_err(|e| convert_load_error(&e))?;
    Ok(PySystem::from_rust(system))
}

// ── validate ─────────────────────────────────────────────────────────────────

/// Validate a Cobre case directory and return a structured report dict.
///
/// Case-validation failures are returned as data in the result dict so that
/// callers see all problems at once. A malformed `config_overrides` dict
/// raises `ValueError` at call time.
///
/// Executes the full validation pipeline (case structure, schema, configuration,
/// stochastic preparation, hydro models, generic constraints, study construction,
/// the configured warm-start, resume or simulation-only policy read from
/// `<output_dir>/<policy.path>`, and boundary reconciliation when configured),
/// short-circuiting on the first failure. Nothing is written, and an absent
/// `output_dir` is not created.
///
/// # Arguments
///
/// * `path` — path to the case directory, as a `str` or `pathlib.Path`.
/// * `output_dir` — keyword-only; the directory the configured policy is read
///   from, as `cobre.run.run`'s `output_dir` names it. Defaults to
///   `<path>/output`. A relative path resolves from the process working
///   directory.
///
/// # Returns
///
/// A dict with the following keys:
///
/// * `"valid"` (`bool`) — `True` when every phase completed without errors.
/// * `"errors"` (`list[dict]`) — list of error dicts, each with `"kind"` and
///   `"message"` string fields. Empty when `valid` is `True`.
/// * `"warnings"` (`list[dict]`) — list of warning dicts, each with `"kind"`,
///   `"message"`, `"file"`, and `"entity"` string fields.
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
#[pyo3(signature = (path, config_overrides=None, *, output_dir=None))]
pub fn validate(
    py: Python<'_>,
    path: PathBuf,
    config_overrides: Option<Bound<'_, PyDict>>,
    output_dir: Option<PathBuf>,
) -> PyResult<Py<PyAny>> {
    let overrides = config_overrides
        .map(|d| pydict_to_json_map(&d))
        .transpose()?;
    let output_dir = resolve_output_dir(&path, output_dir);

    let outcome = py.detach(|| run_validate_pipeline(&path, overrides.as_ref(), &output_dir));

    Ok(build_validation_report(py, outcome)?.into())
}

/// A refusal rendered as one `{"kind", "message"}` error entry.
pub(crate) struct ValidateFailure {
    kind: String,
    message: String,
}

impl From<PolicyLoadFailure> for ValidateFailure {
    fn from(failure: PolicyLoadFailure) -> Self {
        Self {
            kind: failure.kind(),
            message: failure.report(),
        }
    }
}

/// Renders a validation outcome as the `{"valid", "errors", "warnings"}` dict shared by [`validate`] and `Study::validate`.
pub(crate) fn build_validation_report(
    py: Python<'_>,
    outcome: Result<Vec<ReportEntry>, ValidateFailure>,
) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    match outcome {
        Ok(warnings) => {
            dict.set_item("valid", true)?;
            dict.set_item("errors", PyList::empty(py))?;
            dict.set_item("warnings", build_warnings_list(py, &warnings)?)?;
        }
        Err(ValidateFailure { kind, message }) => {
            dict.set_item("valid", false)?;
            let entry = PyDict::new(py);
            entry.set_item("kind", kind)?;
            entry.set_item("message", message)?;
            let errors = PyList::new(py, [entry.as_any()])?;
            dict.set_item("errors", errors)?;
            dict.set_item("warnings", PyList::empty(py))?;
        }
    }
    Ok(dict)
}

/// GIL-free validation pipeline, short-circuiting on the first failure.
fn run_validate_pipeline(
    path: &std::path::Path,
    overrides: Option<&serde_json::Map<String, serde_json::Value>>,
    output_dir: &std::path::Path,
) -> Result<Vec<ReportEntry>, ValidateFailure> {
    if !path.exists() {
        return Err(ValidateFailure {
            kind: "IoError".to_owned(),
            message: format!("case directory does not exist: {}", path.display()),
        });
    }

    let (loaded, report) = validate_case_with_artifacts(path).map_err(|err| ValidateFailure {
        kind: err.kind().to_owned(),
        message: err.to_string(),
    })?;

    let config = load_validate_config(&path.join("config.json"), overrides).map_err(|err| {
        ValidateFailure {
            kind: err.kind().to_owned(),
            message: err.to_string(),
        }
    })?;

    let validated = validate_study(ValidateRequest {
        case_dir: path,
        config: &config,
        system: loaded.system,
        artifacts: loaded.artifacts,
        output_dir,
    })
    .map_err(|failure| match failure {
        validate_phases::ValidateFailure::ConfigLoad(err) => ValidateFailure {
            kind: err.kind().to_owned(),
            message: err.to_string(),
        },
        validate_phases::ValidateFailure::Phase(failure) => ValidateFailure {
            kind: failure.kind().to_owned(),
            message: failure.report(),
        },
        validate_phases::ValidateFailure::PolicyLoad(failure) => failure.into(),
    })?;

    let mut warnings = report.warnings;
    warnings.extend(validated.warnings);
    Ok(warnings)
}
