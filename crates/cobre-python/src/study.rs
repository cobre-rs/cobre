//! The `cobre.Study` pyclass — a live, in-memory study loaded once from a case
//! directory and reused across the solve lifecycle.
//!
//! `Study.__new__` runs the front half of the solve lifecycle (load →
//! stochastic preprocessing → hydro models → `StudySetup` construction →
//! provenance/summary build + sidecar writes) exactly once via the shared
//! [`crate::run::build_study_setup`] helper, then stores the live [`StudySetup`]
//! and the adjacent immutable state so later `train`/`simulate` methods need no
//! reload. It also exposes [`Study::validate`], which replays the validation
//! warnings captured during construction without re-reading disk.
//!
//! ## Single-process only
//!
//! Like [`crate::run`], this module uses [`cobre_comm::LocalBackend`] exclusively
//! and never initializes MPI.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cobre_comm::AffinityPolicy;
use cobre_sddp::{
    FutureCostFunction, HydroModelSummary, ModelProvenanceReport, StochasticSummary, StudySetup,
    TrainingResult,
};

use crate::convert::pydict_to_json_map;
use crate::errors::{ErrorSource, convert_error};
use crate::io::build_warnings_list;
use crate::model::PySystem;
use crate::run::{
    LoadedStudy, PhaseError, SimSummary, TrainingPhaseResult, apply_training_policy_mode,
    build_study_setup, reconstruct_policy_from_checkpoint, run_in_scoped_pool,
    run_simulation_phase_py, run_training_phase_py, run_training_phase_py_streaming,
    write_evaporation_models_if_any, write_fpha_deviation_points_if_any,
    write_fpha_hyperplanes_if_any, write_training_artifacts,
};

/// Map a [`PhaseError`] to a Python exception through the single
/// [`convert_error`] mapping site, so `run_via_study` and the `Study` methods map
/// identically. The `Sddp` arm preserves the typed error's structured fields (e.g.
/// `Infeasible`'s stage/iteration/scenario) as `SolverError` attributes.
fn phase_error_to_pyerr(err: PhaseError) -> PyErr {
    match err {
        PhaseError::Message(msg) => convert_error(ErrorSource::Message(msg)),
        PhaseError::Sddp { error, message } => convert_error(ErrorSource::Sddp {
            error: &error,
            message,
        }),
    }
}

/// A loaded study: the live [`StudySetup`] plus the immutable state produced by
/// the front half of the solve lifecycle.
///
/// Constructed once via [`Study::__new__`] (which runs
/// [`crate::run::build_study_setup`]); `train`/`simulate` reuse the stored
/// state, and [`Study::validate`] replays the captured warnings.
/// The `output_dir` is fixed at construction time so every artifact this study
/// writes lands in the same directory (the always-writes contract).
// The `dead_code` fields below are captured once at construction and read only by
// `simulate`, so it need not re-load the case (the always-load-once contract).
#[pyclass(name = "Study")]
pub struct Study {
    /// The live, fully prepared study setup; the only field `train`/`simulate`
    /// mutate.
    setup: StudySetup,
    /// The system after stochastic preprocessing. Shared (not copied) with the
    /// `cobre.model.System` view returned by the `system` getter; `train`/
    /// `simulate` borrow `&*self.system`.
    system: Arc<cobre_core::System>,
    /// The effective (post-override) configuration.
    config: cobre_io::Config,
    /// The resolved tree seed.
    seed: u64,
    /// The model-provenance report, read by `simulate` for its output report.
    #[allow(dead_code)]
    provenance: ModelProvenanceReport,
    /// The structural stochastic summary, read by `simulate` for its output report.
    #[allow(dead_code)]
    stochastic_summary: StochasticSummary,
    /// The structural hydro-model summary, read by `simulate` for its output report.
    #[allow(dead_code)]
    hydro_models_summary: HydroModelSummary,
    /// Validation-pipeline warnings captured during the case load, replayed by
    /// [`Study::validate`].
    warnings: Vec<cobre_io::ReportEntry>,
    /// The output directory fixed at construction time.
    output_dir: PathBuf,
    /// The requested thread count, stored for later `train`/`simulate` calls.
    threads: Option<u32>,
    /// Worker CPU-binding policy reused by every scoped pool.
    cpu_bind: AffinityPolicy,
}

/// The output of [`Study::train`]: an in-memory trained-policy handle.
///
/// `Policy` carries enough state to drive `simulate` directly without reloading
/// a checkpoint from disk: the [`TrainingResult`] (which owns the per-stage
/// basis cache and the frozen stage templates) and a clone of the trained
/// [`FutureCostFunction`] (the cut pool). A `Policy.load`-ed handle and a
/// trained one therefore expose the same shape.
///
/// The read-only getters surface the headline convergence figures
/// (`iterations`, `final_lower_bound`, `final_upper_bound`).
#[pyclass(name = "Policy")]
pub struct Policy {
    /// The training result (basis cache + frozen templates) `simulate` warm-starts
    /// from.
    training_result: TrainingResult,
    /// The trained (or loaded) study FCF (the cut pool). `Study::simulate`
    /// `replace_fcf`s it into the study before simulating.
    fcf: FutureCostFunction,
}

#[pymethods]
impl Policy {
    /// Number of completed training iterations.
    #[getter]
    fn iterations(&self) -> u64 {
        self.training_result.iterations
    }

    /// Final lower bound at termination.
    #[getter]
    fn final_lower_bound(&self) -> f64 {
        self.training_result.final_lb
    }

    /// Final (mean) upper bound at termination.
    #[getter]
    fn final_upper_bound(&self) -> f64 {
        self.training_result.final_ub
    }

    /// Evaluate the future-cost function at `state` for the given 0-based `stage`.
    ///
    /// Returns `max_k(intercept_k + coeffs_k · state)` over the stage's active
    /// Benders cuts — the FCF lower-bound value at that state. The coefficients
    /// are the stored cut gradients (the raw `HiGHS` duals; see [`Policy::cut_matrix`]),
    /// so the cut is read as `θ ≥ intercept + coeffs · state`. A stage with no
    /// active cuts returns `float('-inf')` (NOT an error).
    ///
    /// `stage` uses the FCF's 0-based stage indexing (stage `t - 1` for the
    /// 1-based SDDP stage `t`).
    ///
    /// # Errors
    ///
    /// - `IndexError` if `stage` is out of range.
    /// - `ValueError` if `state` does not have the policy's state dimension.
    // `state: Vec<f64>` is required so PyO3 can extract from an arbitrary Python
    // sequence; only `&state` is read, hence the `needless_pass_by_value` allow.
    // `stage`/`state` are the natural API names, hence the `similar_names` allow.
    #[allow(clippy::needless_pass_by_value, clippy::similar_names)]
    fn evaluate(&self, stage: usize, state: Vec<f64>) -> PyResult<f64> {
        let n_stages = self.fcf.pools.len();
        if stage >= n_stages {
            return Err(PyIndexError::new_err(format!(
                "stage {stage} out of range (policy has {n_stages} stages)"
            )));
        }
        let dim = self.fcf.state_dimension;
        if state.len() != dim {
            return Err(PyValueError::new_err(format!(
                "state has length {}, expected {dim} (policy state dimension)",
                state.len()
            )));
        }
        Ok(self.fcf.evaluate_at_state(stage, &state))
    }

    /// Return the stage's active Benders cuts as two `NumPy` arrays.
    ///
    /// The result is the 2-tuple `(intercepts, coeffs)` where `intercepts` has
    /// shape `(n_cuts,)` and `coeffs` has shape `(n_cuts, dim)`, both `float64`.
    /// Row `k` of `coeffs` is the gradient of cut `k`, and `intercepts[k]` its
    /// constant term; the active cuts are emitted in the FCF's native ascending
    /// slot order (the deterministic pool order). `dim` is the policy state
    /// dimension and is the column count even when `n_cuts == 0` (shapes `(0,)`
    /// and `(0, dim)`).
    ///
    /// `stage` uses the FCF's 0-based stage indexing (stage `t - 1` for the
    /// 1-based SDDP stage `t`).
    ///
    /// ## Sign convention
    ///
    /// Coefficients are returned **exactly as stored** — the raw `HiGHS` dual of
    /// the state-fixing rows, used directly as the FCF gradient. They are **NOT**
    /// negated. A downstream consumer reconstructs each cut as
    /// `θ ≥ intercept + coeffs · state`, consistent with [`Policy::evaluate`].
    /// (The LP-assembly negation in `build_cut_row_batch` is an internal detail
    /// of solving and does not affect the values surfaced here.)
    ///
    /// # Errors
    ///
    /// - `IndexError` if `stage` is out of range.
    /// - `ImportError` if `NumPy` is not installed (propagated verbatim from the
    ///   lazy `import numpy`; `NumPy` is a soft, lazily imported dependency).
    fn cut_matrix(&self, py: Python<'_>, stage: usize) -> PyResult<Py<PyAny>> {
        let n_stages = self.fcf.pools.len();
        if stage >= n_stages {
            return Err(PyIndexError::new_err(format!(
                "stage {stage} out of range (policy has {n_stages} stages)"
            )));
        }
        let dim = self.fcf.state_dimension;

        let mut intercepts: Vec<f64> = Vec::new();
        let mut coeffs_flat: Vec<f64> = Vec::new();
        for (_slot, intercept, coeffs) in self.fcf.active_cuts(stage) {
            intercepts.push(intercept);
            coeffs_flat.extend_from_slice(coeffs);
        }
        let n_cuts = intercepts.len();

        let np = py.import("numpy")?;
        let intercepts_arr = np.call_method1("asarray", (intercepts,))?;
        // `dim` fixes the column count even when `n_cuts == 0`, giving shape `(0, dim)`.
        let coeffs_1d = np.call_method1("asarray", (coeffs_flat,))?;
        let coeffs_arr = coeffs_1d.call_method1("reshape", ((n_cuts, dim),))?;

        Ok((intercepts_arr, coeffs_arr)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }
}

#[pymethods]
impl Study {
    /// Load a case directory into a live, reusable [`Study`].
    ///
    /// Runs the front half of the solve lifecycle once: loads the case, resolves
    /// the effective config (deep-merging `config_overrides` when present), runs
    /// stochastic and hydro-model preprocessing, builds the [`StudySetup`], and
    /// writes the front-half sidecars (`training/scaling_report.json`,
    /// `training/model_provenance.json`, `training/hydro_models.json`, and the
    /// stochastic exports when enabled) to `output_dir`.
    ///
    /// `output_dir` defaults to `case_dir/output` when `None`. `threads` and
    /// `cpu_bind` are stored for later `train`/`simulate` calls and are not used
    /// during load.
    /// `config_overrides` is a flat dotted-key mapping (e.g.
    /// `{"training.tree_seed": 7}`) converted under the GIL before the load runs
    /// with the GIL released.
    ///
    /// # Errors
    ///
    /// - Raises `OSError` if `case_dir` does not exist (before any work).
    /// - Raises `ValueError` on a malformed override dict (non-str key or
    ///   unsupported value type), or on a config override/parse/read failure.
    /// - Raises `OSError` on a sidecar write failure, and `RuntimeError` on any
    ///   other load/preprocessing/construction failure.
    #[new]
    #[pyo3(signature = (case_dir, output_dir=None, threads=None, config_overrides=None, cpu_bind=None))]
    // PyO3 `#[new]` extracts owned argument types; the path values are then used
    // only by reference here, so clippy's pass-by-value lint does not apply.
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        py: Python<'_>,
        case_dir: PathBuf,
        output_dir: Option<PathBuf>,
        threads: Option<u32>,
        config_overrides: Option<Bound<'_, PyDict>>,
        cpu_bind: Option<String>,
    ) -> PyResult<Self> {
        if !case_dir.exists() {
            return Err(PyOSError::new_err(format!(
                "case directory does not exist: {}",
                case_dir.display()
            )));
        }

        let resolved_output = output_dir.unwrap_or_else(|| case_dir.join("output"));
        let cpu_bind = cpu_bind
            .as_deref()
            .unwrap_or("none")
            .parse::<AffinityPolicy>()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        // Convert UNDER THE GIL, before py.detach releases it; the resulting owned
        // `serde_json::Map` is Send and crosses the py.detach boundary.
        let overrides = config_overrides
            .map(|dict| pydict_to_json_map(&dict))
            .transpose()?;

        // Release the GIL for the slow PAR estimation. No rayon work runs here, so
        // no scoped pool (built per-call in `train`/`simulate`).
        let loaded: Result<LoadedStudy, String> =
            py.detach(|| build_study_setup(&case_dir, &resolved_output, overrides.as_ref()));

        let LoadedStudy {
            setup,
            system,
            config,
            seed,
            provenance,
            stochastic_summary,
            hydro_models_summary,
            warnings,
        } = loaded.map_err(|msg| convert_error(ErrorSource::Message(msg)))?;

        Ok(Study {
            setup,
            system: Arc::new(system),
            config,
            seed,
            provenance,
            stochastic_summary,
            hydro_models_summary,
            warnings,
            output_dir: resolved_output,
            threads,
            cpu_bind,
        })
    }

    /// The resolved output directory as a string.
    #[getter]
    fn output_dir(&self) -> String {
        self.output_dir.to_string_lossy().into_owned()
    }

    /// The loaded [`cobre_core::System`] (as `cobre.model.System`).
    ///
    /// Lets callers introspect the loaded study without a reload. Returned via a
    /// cheap [`Arc`] refcount bump — the underlying `System` is shared, not
    /// copied.
    #[getter]
    fn system(&self) -> PySystem {
        PySystem::from_arc(Arc::clone(&self.system))
    }

    /// Validate the loaded study, returning the same report dict shape as
    /// `cobre.io.validate`: keys `"valid"` (bool), `"errors"` (`list[dict]`),
    /// `"warnings"` (`list[dict]`).
    ///
    /// Because `__new__` already ran every `cobre.io.validate` phase (a failure
    /// there would have raised), the study is known valid here: this returns
    /// `{"valid": True, "errors": [], "warnings": [...]}`, where the warnings are
    /// the `cobre-io` pipeline warnings captured during construction. It never
    /// re-reads disk and never raises.
    fn validate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("valid", true)?;
        dict.set_item("errors", PyList::empty(py))?;
        dict.set_item("warnings", build_warnings_list(py, &self.warnings)?)?;

        Ok(dict)
    }

    /// Train an SDDP policy against this study's in-memory [`StudySetup`],
    /// writing the training artifacts and returning a [`Policy`] handle.
    ///
    /// Runs the SDDP training loop against the live setup built once in
    /// `__new__`, honoring `config.policy.mode` (warm-start / resume /
    /// boundary cuts) before training. The whole Rust computation runs with the
    /// GIL released; when an `on_iteration` callback is provided it is invoked
    /// once per training-iteration boundary in a dedicated drain thread that
    /// reacquires the GIL only at those boundaries (never in the solver's hot
    /// loop).
    ///
    /// Writes `training/policy/`, `training/solver_stats.parquet` (when
    /// non-empty), `training/cut_selection.parquet` (when non-empty),
    /// `training/metadata.json`, `training/_SUCCESS`, and
    /// `hydro_models/fpha_hyperplanes.parquet` (when non-empty) — bit-identical
    /// to `cobre.run.run`. Artifacts are always written before any captured
    /// callback exception is propagated, so a stopped or raising run still
    /// persists what it completed.
    ///
    /// When `config.training.enabled` is `false`, this is a no-op that returns
    /// a [`Policy`] whose `TrainingResult` is a synthetic zero-iteration result
    /// carrying a fresh zero-cut FCF. Such a policy cannot be simulated: pass it
    /// to [`Study::simulate`] and it raises, because simulating with no Benders
    /// cuts would silently produce a wrong result. When training is disabled, use
    /// [`Study::load_policy`] to load a previously trained policy from disk before
    /// calling [`Study::simulate`].
    ///
    /// # Arguments
    ///
    /// - `on_iteration` — optional Python callable invoked once per training
    ///   iteration boundary with a dict (`"kind"`, `"iteration"`,
    ///   `"lower_bound"`, `"upper_bound"`, `"gap"`, `"wall_time_ms"`); `gap` is
    ///   the raw relative gap (NOT scaled by 100). A truthy return requests a
    ///   cooperative stop at a later iteration boundary (asynchronous); a
    ///   raising callback propagates as this method's exception after artifacts
    ///   are written.
    ///
    /// # Errors
    ///
    /// - `RuntimeError` / `OSError` on `HiGHS` init failure, a training error, a
    ///   drain-thread panic, or a policy-mode failure (e.g. a missing prior
    ///   policy directory under `WarmStart`/`Resume`), mapped from the
    ///   descriptive message.
    /// - The original exception raised by a callback (or `KeyboardInterrupt`)
    ///   re-raised verbatim AFTER the training artifacts are written.
    #[pyo3(signature = (on_iteration=None))]
    fn train(&mut self, py: Python<'_>, on_iteration: Option<Py<PyAny>>) -> PyResult<Policy> {
        // Mirrors the training-disabled branch of `run_via_study`.
        if !self.config.training.enabled {
            let synthetic = TrainingResult::new(
                0.0,
                f64::INFINITY,
                0.0,
                0.0,
                0,
                "training disabled".to_string(),
                0,
                Vec::new(),
                Vec::new(),
                None,
                None,
            );
            return Ok(Policy {
                training_result: synthetic,
                fcf: self.setup.fcf.clone(),
            });
        }

        let seed = self.seed;
        let output_dir = self.output_dir.clone();
        let threads = self.threads;
        let cpu_bind = self.cpu_bind;
        let setup = &mut self.setup;
        let system = self.system.as_ref();
        let config = &self.config;

        // `on_iteration` (`Py<PyAny>`), `PyErr`, and the returned tuple are all
        // Send, so crossing the `py.detach` boundary into the drain thread is sound.
        let phase_result: Result<(TrainingPhaseResult, Option<PyErr>), PhaseError> =
            py.detach(|| {
                // Outer `?` surfaces pool-construction failure (a `String`);
                // the inner `Result<_, PhaseError>` is the training/artifact outcome.
                run_in_scoped_pool(threads, cpu_bind, |n, rank_affinity| {
                    // FCF replacement BEFORE training — shared verbatim with
                    // `run_via_study`.
                    apply_training_policy_mode(setup, system, config, &output_dir)?;

                    // `None` keeps the collect-after-return path bit-identical (the
                    // no-callback golden parity anchor); `Some` uses the drain thread.
                    let (training, callback_error) = match on_iteration {
                        Some(callback) => run_training_phase_py_streaming(setup, n, callback)?,
                        None => (run_training_phase_py(setup, n)?, None),
                    };

                    // Write ALL artifacts BEFORE surfacing a captured callback error,
                    // so a stopped/raising run still persists its partial artifacts.
                    write_training_artifacts(
                        &output_dir,
                        system,
                        config,
                        setup,
                        &training,
                        seed,
                        n,
                        rank_affinity,
                    )?;
                    write_fpha_hyperplanes_if_any(&output_dir, setup)?;
                    write_evaporation_models_if_any(&output_dir, setup, system)?;
                    write_fpha_deviation_points_if_any(&output_dir, setup, config)?;

                    Ok::<_, PhaseError>((training, callback_error))
                })?
            });

        let (mut training, callback_error) = phase_result.map_err(phase_error_to_pyerr)?;

        if let Some(err) = callback_error {
            return Err(err);
        }

        // Move the typed error out of the carrier so its structured fields (e.g.
        // `Infeasible`) survive to `convert_error`.
        if let Some(error) = training.error.take() {
            let iterations = training.result.iterations;
            let message = format!("training failed after {iterations} iterations: {error}");
            return Err(convert_error(ErrorSource::Sddp {
                error: &error,
                message,
            }));
        }

        Ok(Policy {
            training_result: training.result,
            fcf: setup.fcf.clone(),
        })
    }

    /// Reconstruct a [`Policy`] from an on-disk policy checkpoint so a loaded
    /// policy and a trained one converge on the IDENTICAL [`Study::simulate`]
    /// entry point.
    ///
    /// Reads `<output_dir>/<policy_path>/` (`output_dir` defaults to this study's
    /// construction-time `output_dir`), reconstructs the
    /// [`FutureCostFunction`] and a synthetic [`TrainingResult`] via the shared
    /// [`reconstruct_policy_from_checkpoint`] helper, and packages them into a
    /// [`Policy`]. The returned policy carries `frozen_templates = None`;
    /// [`Study::simulate`] re-freezes the stage templates from the FCF at startup,
    /// exactly as the monolithic simulation-only path does.
    ///
    /// Validation is unconditional: the checkpoint is always checked against
    /// this study's state dimension, stage count, and terminal entity manifest
    /// via [`cobre_sddp::validate_policy_load`].
    ///
    /// # Errors
    ///
    /// - `RuntimeError` when the policy directory does not exist (message
    ///   containing `"Policy directory not found"`), when the checkpoint cannot
    ///   be read, or when FCF reconstruction fails.
    /// - `ValueError` when policy validation fails (incompatible state dimension,
    ///   stage count, or entity manifest), mapped from the descriptive message.
    #[pyo3(signature = (output_dir=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn load_policy(&self, py: Python<'_>, output_dir: Option<PathBuf>) -> PyResult<Policy> {
        let out_dir = output_dir.unwrap_or_else(|| self.output_dir.clone());
        let policy_dir = out_dir.join(&self.setup.policy_path);

        let setup = &self.setup;
        let system = self.system.as_ref();

        // The reconstruction reads parquet/JSON from disk; release the GIL.
        let reconstructed: Result<(FutureCostFunction, TrainingResult), String> =
            py.detach(|| reconstruct_policy_from_checkpoint(setup, system, &policy_dir));

        let (fcf, training_result) =
            reconstructed.map_err(|msg| convert_error(ErrorSource::Message(msg)))?;
        Ok(Policy {
            training_result,
            fcf,
        })
    }

    /// Run the simulation phase against this study's in-memory [`StudySetup`]
    /// using the supplied [`Policy`], writing the `simulation/` artifacts and
    /// returning a `{"n_scenarios", "completed"}` dict.
    ///
    /// The policy's FCF is installed into the study via
    /// [`StudySetup::replace_fcf`] before simulating, so a trained `Policy` (from
    /// [`Study::train`]) and a loaded `Policy` (from [`Study::load_policy`]) feed
    /// the IDENTICAL simulate path: the unchanged [`run_simulation_phase_py`]
    /// reads the policy's `frozen_templates` and `basis_cache`. A trained policy
    /// carries `frozen_templates = Some(...)`; a loaded one carries `None` and the
    /// study re-freezes the stage templates from the FCF at startup — exactly the
    /// monolithic behavior.
    ///
    /// `output_dir` defaults to this study's construction-time `output_dir`.
    /// Each call writes a fresh `simulation/` output set; the method may be called
    /// repeatedly against one [`Policy`] with no reload between calls. Each call
    /// installs the supplied policy's FCF into the study via
    /// [`StudySetup::replace_fcf`] before simulating, so repeated calls remain
    /// deterministic (each one re-installs the same cut pool).
    ///
    /// Writes `simulation/metadata.json`, the per-scenario parquet,
    /// `simulation/_SUCCESS`, and `simulation/solver_stats.parquet` (when the
    /// solver log is non-empty) to `output_dir` — bit-identical to
    /// `cobre.run.run`'s simulation phase.
    ///
    /// # Errors
    ///
    /// - `RuntimeError` when `policy` carries no Benders cuts (zero active cuts).
    ///   A cut-less policy — e.g. the synthetic handle [`Study::train`] returns
    ///   when `config.training.enabled` is `false` — would silently simulate a
    ///   wrong result, so this guard rejects it up front and asks the caller to
    ///   load a trained policy via [`Study::load_policy`] first.
    /// - `OSError` on a simulation workspace-pool (`HiGHS`) init failure.
    /// - `RuntimeError` on a simulation error (infeasibility) or a writer/output
    ///   failure, mapped from the descriptive message.
    #[pyo3(signature = (policy, output_dir=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn simulate(
        &mut self,
        py: Python<'_>,
        policy: PyRef<'_, Policy>,
        output_dir: Option<PathBuf>,
    ) -> PyResult<Py<PyAny>> {
        let out_dir = output_dir.unwrap_or_else(|| self.output_dir.clone());

        // A zero-cut policy would let `StudySetup::simulate` run with no future-cost
        // approximation and silently emit a wrong result, so reject it up front.
        if policy.fcf.total_active_cuts() == 0 {
            // No recognized prefix, so the message falls through to `SolverError`
            // (a `RuntimeError` subclass) with the text preserved verbatim.
            return Err(convert_error(ErrorSource::Message(
                "Policy has no cuts to simulate; when training is disabled, call \
                 Study.load_policy() to load a trained policy before simulate()"
                    .to_string(),
            )));
        }

        self.setup.replace_fcf(policy.fcf.clone());

        let threads = self.threads;
        let cpu_bind = self.cpu_bind;
        let setup = &mut self.setup;
        let system = self.system.as_ref();
        let training_result = &policy.training_result;

        // Release the GIL for the scenario sweep; the rayon pool is built per call
        // so sequential `simulate` invocations each honor their own thread count.
        let summary: Result<SimSummary, PhaseError> = py.detach(|| {
            // Outer `?` surfaces pool-construction failure (a `String`); the inner
            // `Result<SimSummary, PhaseError>` is the simulation outcome.
            run_in_scoped_pool(threads, cpu_bind, |n, rank_affinity| {
                run_simulation_phase_py(setup, &out_dir, system, training_result, n, rank_affinity)
            })?
        });

        let summary = summary.map_err(phase_error_to_pyerr)?;

        let dict = PyDict::new(py);
        dict.set_item("n_scenarios", summary.n_scenarios)?;
        dict.set_item("completed", summary.completed)?;
        Ok(dict.into())
    }
}
