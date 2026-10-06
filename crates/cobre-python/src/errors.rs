//! The `cobre.errors` exception hierarchy and the single error-mapping site.
//!
//! Every leaf class subclasses BOTH `CobreError` and the matching builtin
//! (`OSError`, `ValueError`, `RuntimeError`), so existing `except OSError` /
//! `except ValueError` / `except RuntimeError` code keeps catching while new code
//! can catch the typed class or the common `CobreError` base. The qualified name
//! of every class is `cobre.errors.<Name>` so tracebacks read
//! `cobre.errors.SolverError`.
//!
//! [`convert_error`] is the ONLY place Rust errors become Python exceptions for
//! the raising paths. It accepts a concrete [`ErrorSource`] enum (never a
//! `Box<dyn Trait>`) so the match is exhaustive: a newly added `LoadError` /
//! `OutputError` variant fails the build, surfacing the need to map it. The
//! `SddpError` match keeps explicit `Infeasible`/`Simulation` arms and a
//! `CheckpointWrite` arm raising its `OutputError`'s class. Every other
//! `SddpError`, and every `FullFcfLoadError`, takes its class from
//! [`cobre_sddp::ErrorClass`] through `exception_for_class`, the classification
//! `cobre run` derives its exit code from.
//!
//! The `cobre.io.validate` data-report `kind` field is intentionally NOT routed
//! here — it is a stable data contract decoupled from these class names.

use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyTuple, PyType};

use cobre_io::{LoadError, OutputError};
use cobre_sddp::policy::full_fcf_load::{FullFcfLoadError, FullFcfLoadKind};
use cobre_sddp::{ErrorClass, SddpError};

pyo3::create_exception!(
    errors,
    CobreError,
    PyException,
    "Base class for every Cobre exception (subclasses `Exception`)."
);

/// A leaf exception class in the `cobre.errors` hierarchy.
///
/// Each leaf subclasses BOTH [`CobreError`] and a builtin, so the dual-base
/// type object is built once via Python's `type(name, bases, dict)` and cached
/// for the life of the interpreter.
struct LeafClass {
    /// Unqualified class name (e.g. `"ValidationError"`).
    name: &'static str,
    /// The builtin co-base alongside [`CobreError`] (e.g. `PyValueError`).
    builtin_base: BuiltinBase,
    /// Docstring exposed as `__doc__`.
    doc: &'static str,
    /// The lazily built, interpreter-lifetime type object.
    cell: PyOnceLock<Py<PyType>>,
}

/// The builtin co-base for a [`LeafClass`].
#[derive(Clone, Copy)]
enum BuiltinBase {
    Value,
    Os,
    Runtime,
}

impl LeafClass {
    const fn new(name: &'static str, builtin_base: BuiltinBase, doc: &'static str) -> Self {
        Self {
            name,
            builtin_base,
            doc,
            cell: PyOnceLock::new(),
        }
    }

    /// Resolve (building once) the dual-base type object under the GIL.
    fn get(&self, py: Python<'_>) -> PyResult<&Py<PyType>> {
        self.cell.get_or_try_init(py, || self.build(py))
    }

    /// Build the dual-base class object with `__module__ = "cobre.errors"`, so the
    /// qualified name reads `cobre.errors.<Name>`.
    fn build(&self, py: Python<'_>) -> PyResult<Py<PyType>> {
        let cobre_base = py.get_type::<CobreError>();
        let builtin: Bound<'_, PyType> = match self.builtin_base {
            BuiltinBase::Value => py.get_type::<PyValueError>(),
            BuiltinBase::Os => py.get_type::<PyOSError>(),
            BuiltinBase::Runtime => py.get_type::<PyRuntimeError>(),
        };
        let bases = PyTuple::new(py, [cobre_base, builtin])?;
        let namespace = pyo3::types::PyDict::new(py);
        namespace.set_item("__module__", "cobre.errors")?;
        namespace.set_item("__doc__", self.doc)?;

        let type_builtin = py.get_type::<PyType>();
        let class = type_builtin.call1((self.name, bases, namespace))?;
        let class_type = class.cast_into::<PyType>().map_err(PyErr::from)?;
        Ok(class_type.unbind())
    }
}

/// `ValidationError(CobreError, ValueError)` — schema / parse / constraint /
/// config-override load failures, and study-setup configuration-validation
/// failures.
static VALIDATION_ERROR: LeafClass = LeafClass::new(
    "ValidationError",
    BuiltinBase::Value,
    "Raised when case data or configuration fails validation (subclasses ValueError).",
);

/// `PolicyIncompatibleError(CobreError, ValueError)` — warm-start / resume
/// policy incompatibility.
static POLICY_INCOMPATIBLE_ERROR: LeafClass = LeafClass::new(
    "PolicyIncompatibleError",
    BuiltinBase::Value,
    "Raised when a warm-start policy is incompatible with the current system \
     (subclasses ValueError).",
);

/// `CaseIoError(CobreError, OSError)` — filesystem read/write failures.
static CASE_IO_ERROR: LeafClass = LeafClass::new(
    "CaseIoError",
    BuiltinBase::Os,
    "Raised on a filesystem read or write failure while loading or writing a case \
     (subclasses OSError).",
);

/// `OutputError(CobreError, OSError)` — output serialization / schema / manifest
/// failures on the write path.
static OUTPUT_ERROR: LeafClass = LeafClass::new(
    "OutputError",
    BuiltinBase::Os,
    "Raised on an output serialization, schema, or manifest failure while writing \
     results (subclasses OSError).",
);

/// `SolverError(CobreError, RuntimeError)` — training/solver failures. Carries
/// `stage`/`iteration`/`scenario` int attributes for an infeasible subproblem,
/// `None` otherwise.
static SOLVER_ERROR: LeafClass = LeafClass::new(
    "SolverError",
    BuiltinBase::Runtime,
    "Raised on a training or solver failure (subclasses RuntimeError). For an \
     infeasible subproblem, exposes integer `stage`/`iteration`/`scenario` \
     attributes; otherwise they are None.",
);

/// `SimulationError(CobreError, RuntimeError)` — simulation-phase failures.
static SIMULATION_ERROR: LeafClass = LeafClass::new(
    "SimulationError",
    BuiltinBase::Runtime,
    "Raised on a simulation-phase failure (subclasses RuntimeError).",
);

/// `InternalError(CobreError, RuntimeError)` — software or environment faults.
static INTERNAL_ERROR: LeafClass = LeafClass::new(
    "InternalError",
    BuiltinBase::Runtime,
    "Raised on an internal software or environment fault (subclasses RuntimeError).",
);

/// The owned/borrowed source of an error to be mapped to a Python exception.
///
/// A concrete enum (NOT a `Box<dyn Trait>`, per the hard rules) so every call
/// site funnels through [`convert_error`] with an exhaustive match.
pub(crate) enum ErrorSource<'a> {
    /// A case-load failure from `cobre-io`.
    Load(&'a LoadError),
    /// An output-write / metadata-read failure from `cobre-io`.
    Output(&'a OutputError),
    /// A typed SDDP error carried verbatim from a phase helper, paired with the
    /// descriptive message that today's string path would have produced.
    ///
    /// `message` preserves the exact text (so `match=` assertions pass); the
    /// [`SddpError`] supplies the class and the structured fields (e.g.
    /// `Infeasible`).
    Sddp {
        /// The typed SDDP error (borrowed; `Send + Sync + 'static`).
        error: &'a SddpError,
        /// The verbatim descriptive message.
        message: String,
    },
    /// A full-FCF policy-load failure (warm-start, resume or simulation-only).
    PolicyLoad(&'a FullFcfLoadError),
    /// A string-prefixed message from run/study with no typed source.
    Message(String),
}

fn case_io_error(py: Python<'_>, message: &str) -> PyErr {
    new_leaf_err(py, &CASE_IO_ERROR, message)
}

fn validation_error(py: Python<'_>, message: &str) -> PyErr {
    new_leaf_err(py, &VALIDATION_ERROR, message)
}

/// Build a `PyErr` for the given leaf class with the supplied message.
///
/// If the dual-base class fails to build (it never should, the bases are
/// builtins), the returned `PyErr` is that build failure rather than a
/// mis-typed exception — it cannot be silently swallowed.
fn new_leaf_err(py: Python<'_>, leaf: &LeafClass, message: &str) -> PyErr {
    let class = match leaf.get(py) {
        Ok(class) => class,
        Err(err) => return err,
    };
    match class.bind(py).call1((message,)) {
        Ok(instance) => PyErr::from_value(instance),
        Err(err) => err,
    }
}

/// Map an [`ErrorSource`] to the appropriate `cobre.errors` Python exception —
/// the single mapping site for every raising path.
///
/// Requires a GIL token internally (it constructs Python exception instances and
/// `setattr`s structured fields), but its signature takes only Rust enums and
/// strings, so it is callable from a GIL-bound Rust `#[cfg(test)]` boundary test
/// as well as from the binding entry points.
pub(crate) fn convert_error(source: ErrorSource<'_>) -> PyErr {
    Python::attach(|py| convert_error_with(py, source))
}

/// GIL-bound body of [`convert_error`]. Split out so a test (already holding the
/// GIL) can call it directly with its own `py` token.
fn convert_error_with(py: Python<'_>, source: ErrorSource<'_>) -> PyErr {
    match source {
        ErrorSource::Load(err) => match err {
            LoadError::IoError { .. } => case_io_error(py, &err.to_string()),
            LoadError::ParseError { .. }
            | LoadError::SchemaError { .. }
            | LoadError::ConstraintError { .. } => validation_error(py, &err.to_string()),
        },
        ErrorSource::Output(err) => output_error(py, err, &err.to_string()),
        ErrorSource::Sddp { error, message } => match error {
            SddpError::Infeasible {
                stage,
                iteration,
                scenario,
            } => solver_error_infeasible(py, &message, *stage, *iteration, *scenario),
            SddpError::Simulation(_) => new_leaf_err(py, &SIMULATION_ERROR, &message),
            SddpError::CheckpointWrite { source, .. } => {
                output_error(py, source, &error.to_string())
            }
            other => exception_for_class(py, other.class(), &message),
        },
        ErrorSource::PolicyLoad(err) => {
            let message = match err {
                FullFcfLoadError::MissingPolicyDirectory { .. } | FullFcfLoadError::Read { .. } => {
                    err.to_string()
                }
                FullFcfLoadError::Refused(source) => {
                    format!("{POLICY_VALIDATION_ERROR_PREFIX}: {source}")
                }
                FullFcfLoadError::FcfConstruction { kind, source } => {
                    let label = match kind {
                        FullFcfLoadKind::WarmStart => "warm-start FCF construction error",
                        FullFcfLoadKind::Resume => "resume FCF construction error",
                        FullFcfLoadKind::SimulationOnly => "FCF reconstruction error",
                    };
                    format!("{label}: {source}")
                }
            };
            exception_for_class(py, err.class(), &message)
        }
        ErrorSource::Message(msg) => message_prefix_to_pyerr(py, &msg),
    }
}

fn exception_for_class(py: Python<'_>, class: ErrorClass, message: &str) -> PyErr {
    match class {
        ErrorClass::InvalidInput => validation_error(py, message),
        ErrorClass::IncompatiblePolicy => new_leaf_err(py, &POLICY_INCOMPATIBLE_ERROR, message),
        ErrorClass::Io => case_io_error(py, message),
        // Internal keeps SolverError: InternalError would change a public class.
        ErrorClass::Solver | ErrorClass::Internal => solver_error_plain(py, message),
    }
}

/// The class an [`OutputError`] raises as, carrying `message`.
fn output_error(py: Python<'_>, err: &OutputError, message: &str) -> PyErr {
    match err {
        // A NotFound I/O error maps to the builtin FileNotFoundError (no typed
        // class for it).
        OutputError::IoError { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(message.to_string())
        }
        OutputError::IoError { .. } => case_io_error(py, message),
        OutputError::SerializationError { .. } | OutputError::SchemaError { .. } => {
            new_leaf_err(py, &OUTPUT_ERROR, message)
        }
        OutputError::ManifestError { .. } | OutputError::ForeignEntry { .. } => {
            validation_error(py, message)
        }
    }
}

/// Prefix minted for output-serialization/write failures (classified as `CaseIoError`).
pub(crate) const OUTPUT_WRITE_ERROR_PREFIX: &str = "output write error";

/// Prefix minted for policy-checkpoint write failures (classified as `CaseIoError`).
pub(crate) const POLICY_CHECKPOINT_ERROR_PREFIX: &str = "policy checkpoint error";

/// Prefix minted for config-override merge failures (classified as `ValidationError`).
pub(crate) const CONFIG_OVERRIDE_ERROR_PREFIX: &str = "config override error";

/// Prefix minted for config JSON parse failures (classified as `ValidationError`).
pub(crate) const CONFIG_PARSE_ERROR_PREFIX: &str = "config parse error";

/// Prefix minted for config file read failures (classified as `ValidationError`).
pub(crate) const CONFIG_READ_ERROR_PREFIX: &str = "config read error";

/// Prefix minted for study-setup configuration-validation failures. Message
/// text only: the class comes from the typed error.
pub(crate) const SETUP_VALIDATION_ERROR_PREFIX: &str = "setup validation error";

/// Prefix minted for warm-start/resume policy validation failures (classified
/// as `PolicyIncompatibleError`).
pub(crate) const POLICY_VALIDATION_ERROR_PREFIX: &str = "policy validation error";

/// Prefix minted for simulation-phase failures (classified as `SimulationError`).
pub(crate) const SIMULATION_ERROR_PREFIX: &str = "simulation error";

/// Prefix minted for internal software/environment faults (classified as `InternalError`).
pub(crate) const INTERNAL_ERROR_PREFIX: &str = "internal error";

/// Prefix minted for training-phase failures, named here so the run-path minter
/// shares one owning constant with the other prefixes. Message text only: the
/// classifier below does not recognise it, and a typed error takes its class
/// from [`ErrorClass`]. The remaining run-path phase prefixes below are named on
/// the same rationale.
pub(crate) const TRAINING_ERROR_PREFIX: &str = "training error";

/// Prefix minted for simulation-writer initialisation failures.
pub(crate) const SIMULATION_WRITER_INIT_ERROR_PREFIX: &str =
    "simulation writer initialisation error";

/// Prefix minted for scenario-source construction failures.
pub(crate) const SCENARIO_SOURCE_ERROR_PREFIX: &str = "scenario source error";

/// Prefix minted for stochastic-preprocessing failures.
pub(crate) const STOCHASTIC_PREPROCESSING_ERROR_PREFIX: &str = "stochastic preprocessing error";

/// Prefix minted for hydro-model-preprocessing failures.
pub(crate) const HYDRO_MODEL_PREPROCESSING_ERROR_PREFIX: &str = "hydro model preprocessing error";

/// Prefix minted for boundary-cut load failures.
pub(crate) const BOUNDARY_CUT_ERROR_PREFIX: &str = "boundary cut error";

/// Map a string-prefixed run/study message to its typed class.
fn message_prefix_to_pyerr(py: Python<'_>, msg: &str) -> PyErr {
    if msg.starts_with(OUTPUT_WRITE_ERROR_PREFIX) || msg.starts_with(POLICY_CHECKPOINT_ERROR_PREFIX)
    {
        case_io_error(py, msg)
    } else if msg.starts_with(CONFIG_OVERRIDE_ERROR_PREFIX)
        || msg.starts_with(CONFIG_PARSE_ERROR_PREFIX)
        || msg.starts_with(CONFIG_READ_ERROR_PREFIX)
    {
        validation_error(py, msg)
    } else if msg.starts_with(POLICY_VALIDATION_ERROR_PREFIX) {
        new_leaf_err(py, &POLICY_INCOMPATIBLE_ERROR, msg)
    } else if msg.starts_with(SIMULATION_ERROR_PREFIX) {
        new_leaf_err(py, &SIMULATION_ERROR, msg)
    } else if msg.starts_with(INTERNAL_ERROR_PREFIX) {
        new_leaf_err(py, &INTERNAL_ERROR, msg)
    } else {
        // Unrecognized prefix (e.g. "scenario source error") falls through to
        // SolverError.
        solver_error_plain(py, msg)
    }
}

/// Build a [`SolverError`] for a non-infeasible failure: all three structured
/// attributes are `None`.
fn solver_error_plain(py: Python<'_>, message: &str) -> PyErr {
    build_solver_error(py, message, None)
}

/// Build a [`SolverError`] for an infeasible subproblem, attaching the
/// `stage`/`iteration`/`scenario` int attributes.
fn solver_error_infeasible(
    py: Python<'_>,
    message: &str,
    stage: usize,
    iteration: u64,
    scenario: usize,
) -> PyErr {
    build_solver_error(py, message, Some((stage, iteration, scenario)))
}

/// Construct a `SolverError` instance, setting `stage`/`iteration`/`scenario`
/// either to the supplied ints (infeasible) or to `None` (every other failure)
/// so the three attributes are always present on the instance.
///
/// A `setattr` failure propagates as the returned `PyErr` — it is never
/// swallowed.
fn build_solver_error(
    py: Python<'_>,
    message: &str,
    infeasible: Option<(usize, u64, usize)>,
) -> PyErr {
    let class = match SOLVER_ERROR.get(py) {
        Ok(class) => class,
        Err(err) => return err,
    };
    let instance = match class.bind(py).call1((message,)) {
        Ok(instance) => instance,
        Err(err) => return err,
    };

    let set_fields = || -> PyResult<()> {
        if let Some((stage, iteration, scenario)) = infeasible {
            instance.setattr("stage", stage)?;
            instance.setattr("iteration", iteration)?;
            instance.setattr("scenario", scenario)?;
        } else {
            instance.setattr("stage", py.None())?;
            instance.setattr("iteration", py.None())?;
            instance.setattr("scenario", py.None())?;
        }
        Ok(())
    };

    if let Err(err) = set_fields() {
        return err;
    }
    PyErr::from_value(instance)
}

/// Register the exception classes (`CobreError` plus its dual-base leaves) into
/// the `errors` submodule so `from cobre.errors import …` resolves them all.
pub(crate) fn register_errors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "__doc__",
        "Structured exception hierarchy for Cobre errors.",
    )?;
    m.add("CobreError", py.get_type::<CobreError>())?;
    for leaf in [
        &VALIDATION_ERROR,
        &POLICY_INCOMPATIBLE_ERROR,
        &CASE_IO_ERROR,
        &OUTPUT_ERROR,
        &SOLVER_ERROR,
        &SIMULATION_ERROR,
        &INTERNAL_ERROR,
    ] {
        let class = leaf.get(py)?;
        m.add(leaf.name, class.clone_ref(py))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CASE_IO_ERROR, ErrorSource, INTERNAL_ERROR, LeafClass, OUTPUT_ERROR,
        POLICY_INCOMPATIBLE_ERROR, POLICY_VALIDATION_ERROR_PREFIX, SIMULATION_ERROR, SOLVER_ERROR,
        VALIDATION_ERROR, convert_error_with,
    };
    use cobre_comm::CommError;
    use cobre_io::{LoadError, OutputError};
    use cobre_sddp::SddpError;
    use cobre_sddp::policy::full_fcf_load::{FullFcfLoadError, FullFcfLoadKind};
    use cobre_solver::SolverError;
    use cobre_stochastic::StochasticError;
    use pyo3::prelude::*;

    /// Assert the bound `PyErr` value is an instance of the supplied leaf class
    /// and that its qualified name reads `cobre.errors.<Name>`.
    ///
    /// Resolves the type object directly from the leaf static (no `sys.modules`
    /// or `import cobre.errors`, which would require the parent `cobre` package
    /// to exist in the standalone test binary).
    fn assert_leaf(py: Python<'_>, err: &PyErr, leaf: &LeafClass) {
        let class = leaf.get(py).expect("leaf class builds").bind(py);
        let value = err.value(py);
        assert!(
            value.is_instance(class).expect("isinstance check"),
            "expected cobre.errors.{}, got {value:?}",
            leaf.name
        );
        let qualname: String = class
            .getattr("__module__")
            .expect("__module__")
            .extract()
            .expect("str");
        assert_eq!(qualname, "cobre.errors", "{} __module__", leaf.name);
    }

    /// An `Infeasible` SDDP error maps to `SolverError` with the three int
    /// attributes set and the verbatim message preserved.
    #[test]
    fn convert_error_infeasible() {
        Python::initialize();
        Python::attach(|py| {
            let message = "infeasible subproblem at stage 5, iteration 42, scenario 3".to_string();
            let err = convert_error_with(
                py,
                ErrorSource::Sddp {
                    error: &SddpError::Infeasible {
                        stage: 5,
                        iteration: 42,
                        scenario: 3,
                    },
                    message: message.clone(),
                },
            );

            assert_leaf(py, &err, &SOLVER_ERROR);

            let value = err.value(py);
            let stage: usize = value.getattr("stage").unwrap().extract().unwrap();
            let iteration: u64 = value.getattr("iteration").unwrap().extract().unwrap();
            let scenario: usize = value.getattr("scenario").unwrap().extract().unwrap();
            assert_eq!(stage, 5);
            assert_eq!(iteration, 42);
            assert_eq!(scenario, 3);

            let rendered: String = value.str().unwrap().extract().unwrap();
            assert_eq!(rendered, message);
        });
    }

    /// Sibling: the non-`Infeasible` SDDP arms. `Simulation` maps to
    /// `SimulationError`; a generic `Solver` maps to `SolverError` with the three
    /// attributes `None`. Messages are preserved verbatim.
    #[test]
    fn convert_error_non_infeasible_arms() {
        Python::initialize();
        Python::attach(|py| {
            let sim_msg = "simulation error: x".to_string();
            let sim_err = convert_error_with(
                py,
                ErrorSource::Sddp {
                    error: &SddpError::Simulation("x".to_string()),
                    message: sim_msg.clone(),
                },
            );
            assert_leaf(py, &sim_err, &SIMULATION_ERROR);
            let sim_rendered: String = sim_err.value(py).str().unwrap().extract().unwrap();
            assert_eq!(sim_rendered, sim_msg);

            let solver_msg = "training error: solver error: numerical failure".to_string();
            let solver_err = convert_error_with(
                py,
                ErrorSource::Sddp {
                    error: &SddpError::Solver(SolverError::Unbounded),
                    message: solver_msg.clone(),
                },
            );
            assert_leaf(py, &solver_err, &SOLVER_ERROR);
            let value = solver_err.value(py);
            assert!(value.getattr("stage").unwrap().is_none());
            assert!(value.getattr("iteration").unwrap().is_none());
            assert!(value.getattr("scenario").unwrap().is_none());
            let solver_rendered: String = value.str().unwrap().extract().unwrap();
            assert_eq!(solver_rendered, solver_msg);
        });
    }

    /// A failed checkpoint write raises its `OutputError`'s class, with the SDDP
    /// error's own text as the message.
    #[test]
    fn convert_error_checkpoint_write_raises_its_output_error_class() {
        Python::initialize();
        Python::attach(|py| {
            for (source, leaf) in [
                (
                    OutputError::IoError {
                        path: std::path::PathBuf::from("out/policy.staging"),
                        source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                    },
                    &CASE_IO_ERROR,
                ),
                (
                    OutputError::SerializationError {
                        entity: "stage_cuts".to_string(),
                        message: "buffer too large".to_string(),
                    },
                    &OUTPUT_ERROR,
                ),
            ] {
                let error = SddpError::CheckpointWrite {
                    iteration: 2,
                    source,
                };
                let err = convert_error_with(
                    py,
                    ErrorSource::Sddp {
                        error: &error,
                        message: format!("training failed after 2 iterations: {error}"),
                    },
                );
                assert_leaf(py, &err, leaf);
                let rendered: String = err.value(py).str().unwrap().extract().unwrap();
                assert_eq!(rendered, error.to_string());
            }
        });
    }

    /// An "internal error: " prefixed message maps to `InternalError`.
    #[test]
    fn convert_error_internal_error_prefix() {
        Python::initialize();
        Python::attach(|py| {
            let msg = "internal error: drain thread panicked".to_string();
            let err = convert_error_with(py, ErrorSource::Message(msg.clone()));
            assert_leaf(py, &err, &INTERNAL_ERROR);
            let rendered: String = err.value(py).str().unwrap().extract().unwrap();
            assert_eq!(rendered, msg);
        });
    }

    /// Every `SddpError` other than the ones with a dedicated arm raises the
    /// class of its `ErrorClass`, whatever prefix its message carries.
    #[test]
    fn convert_error_sddp_follows_the_error_class() {
        let cases = [
            (
                SddpError::Stochastic(StochasticError::InsufficientData {
                    context: "no valid historical windows found".to_string(),
                }),
                "stochastic preprocessing error",
                &VALIDATION_ERROR,
            ),
            (
                SddpError::Validation("x".to_string()),
                "setup validation error",
                &VALIDATION_ERROR,
            ),
            (
                SddpError::Validation("x".to_string()),
                "training error",
                &VALIDATION_ERROR,
            ),
            (
                SddpError::Io(LoadError::IoError {
                    path: "case/config.json".into(),
                    source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                }),
                "hydro model preprocessing error",
                &CASE_IO_ERROR,
            ),
            (
                SddpError::Io(LoadError::ParseError {
                    path: "case/config.json".into(),
                    message: "unexpected token".to_string(),
                }),
                "hydro model preprocessing error",
                &VALIDATION_ERROR,
            ),
            (
                SddpError::PolicySoftwareMismatch {
                    policy_software: Some("another-program".to_string()),
                    policy_version: "0.0.1".to_string(),
                },
                "policy validation error",
                &POLICY_INCOMPATIBLE_ERROR,
            ),
            (
                SddpError::Communication(CommError::InvalidCommunicator),
                "training error",
                &SOLVER_ERROR,
            ),
            (
                SddpError::WireVersionMismatch {
                    encoded: 1,
                    expected: 2,
                },
                "training error",
                &SOLVER_ERROR,
            ),
        ];
        Python::initialize();
        Python::attach(|py| {
            for (error, prefix, leaf) in cases {
                let message = format!("{prefix}: {error}");
                let err = convert_error_with(
                    py,
                    ErrorSource::Sddp {
                        error: &error,
                        message: message.clone(),
                    },
                );
                assert_leaf(py, &err, leaf);
                let value = err.value(py);
                let rendered: String = value.str().unwrap().extract().unwrap();
                assert_eq!(rendered, message);
                if leaf.name == SOLVER_ERROR.name {
                    assert!(value.getattr("stage").unwrap().is_none());
                    assert!(value.getattr("iteration").unwrap().is_none());
                    assert!(value.getattr("scenario").unwrap().is_none());
                }
            }
        });
    }

    /// A phase prefix with no recognised wrapper falls through to `SolverError`.
    #[test]
    fn unrecognised_message_prefix_falls_through_to_solver_error() {
        Python::initialize();
        Python::attach(|py| {
            let msg = "hydro model preprocessing error: solver error: x".to_string();
            let err = convert_error_with(py, ErrorSource::Message(msg.clone()));
            assert_leaf(py, &err, &SOLVER_ERROR);
            let rendered: String = err.value(py).str().unwrap().extract().unwrap();
            assert_eq!(rendered, msg);
        });
    }

    /// Every policy-load failure raises the class of its `ErrorClass`, with the
    /// inner error's text read from its own `Display`.
    #[test]
    fn convert_error_policy_load_follows_each_failure_class() {
        let kinds = [
            (
                FullFcfLoadKind::WarmStart,
                "warm-start FCF construction error",
            ),
            (FullFcfLoadKind::Resume, "resume FCF construction error"),
            (FullFcfLoadKind::SimulationOnly, "FCF reconstruction error"),
        ];
        Python::initialize();
        Python::attach(|py| {
            let check = |err: &FullFcfLoadError, leaf: &LeafClass, expected: &str| {
                let converted = convert_error_with(py, ErrorSource::PolicyLoad(err));
                assert_leaf(py, &converted, leaf);
                let rendered: String = converted.value(py).str().unwrap().extract().unwrap();
                assert_eq!(rendered, expected);
            };

            for (kind, _) in kinds {
                let err = FullFcfLoadError::MissingPolicyDirectory {
                    kind,
                    path: "/study/policy".into(),
                };
                let expected = err.to_string();
                assert!(expected.starts_with("Policy directory not found: /study/policy. "));
                check(&err, &VALIDATION_ERROR, &expected);
            }

            let empty = tempfile::tempdir().expect("temp dir");
            let read_failure = cobre_io::read_policy_checkpoint(empty.path())
                .expect_err("an empty directory holds no checkpoint");
            let expected = format!("failed to read policy checkpoint: {read_failure}");
            check(
                &FullFcfLoadError::Read {
                    source: read_failure,
                },
                &POLICY_INCOMPATIBLE_ERROR,
                &expected,
            );

            let unreadable = OutputError::IoError {
                path: "policy/manifest.bin".into(),
                source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            };
            let expected = format!("failed to read policy checkpoint: {unreadable}");
            check(
                &FullFcfLoadError::Read { source: unreadable },
                &CASE_IO_ERROR,
                &expected,
            );

            let mismatch = SddpError::PolicySoftwareMismatch {
                policy_software: Some("another-program".to_string()),
                policy_version: "0.0.1".to_string(),
            };
            let expected = format!("{POLICY_VALIDATION_ERROR_PREFIX}: {mismatch}");
            check(
                &FullFcfLoadError::Refused(mismatch),
                &POLICY_INCOMPATIBLE_ERROR,
                &expected,
            );

            for (kind, label) in kinds {
                let source = SddpError::Validation("malformed cuts".to_string());
                let expected = format!("{label}: {source}");
                check(
                    &FullFcfLoadError::FcfConstruction { kind, source },
                    &POLICY_INCOMPATIBLE_ERROR,
                    &expected,
                );
            }
        });
    }
}
