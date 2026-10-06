//! The pre-solver validation pipeline behind `cobre validate` and
//! `cobre.io.validate`, and the metadata that names the phase a failure came from.
//!
//! [`validate_study`] is the single pipeline both front ends present: it builds the
//! study the way `cobre run` does, so every refusal that the run reaches while
//! constructing the study is reached here too. [`PrepPhase`] and
//! [`prep_phase_metadata`] derive the human-readable file label and the structured
//! error-kind string from a [`SddpError`].
//!
//! # Why this lives in `cobre-sddp`
//!
//! The pipeline and the mapping touch [`SddpError`] variants and the study
//! constructors, which are defined here. `cobre-io` cannot import from `cobre-sddp`
//! (that would create a cycle), so the shared logic must live in `cobre-sddp` or
//! above it in the dependency graph.

use std::path::Path;

use cobre_core::System;
use cobre_io::{CaseArtifacts, Config, LoadError, ReportEntry};

use crate::hydro_models::prepare_hydro_models_from_artifacts;
use crate::{
    BoundaryReconciliation, SddpError, StudyParams, StudySetup, prepare_stochastic,
    reconcile_boundary_policy, resolve_boundary_state_requirements,
    validate_generic_constraint_parameters,
};

// ── PrepPhase ─────────────────────────────────────────────────────────────────

/// Which pre-solver preparation phase produced an error.
///
/// Each variant corresponds to the SDDP preparation steps that
/// follow the six-layer cobre-io loading pipeline:
///
/// | Phase                | Function called                            | Typical trigger file                  |
/// |----------------------|--------------------------------------------|---------------------------------------|
/// | [`Config`]           | `StudyParams::from_config`                 | `config.json`                         |
/// | [`Stochastic`]       | `prepare_stochastic`                       | `scenarios/inflow_history.parquet`    |
/// | [`HydroModels`]      | `prepare_hydro_models_from_artifacts`      | `system/hydro_production_models.json` |
/// | [`GenericConstraints`] | `validate_generic_constraint_parameters` | `constraints/`                        |
/// | [`StudySetup`]       | `StudySetup::new_with_boundary_requirements` | `scenarios/`                        |
/// | [`Boundary`]         | `resolve_boundary_state_requirements`, `StudySetup::new_with_boundary_requirements`, `reconcile_boundary_policy` | `policy.boundary` |
///
/// [`Config`]: PrepPhase::Config
/// [`Stochastic`]: PrepPhase::Stochastic
/// [`HydroModels`]: PrepPhase::HydroModels
/// [`GenericConstraints`]: PrepPhase::GenericConstraints
/// [`StudySetup`]: PrepPhase::StudySetup
/// [`Boundary`]: PrepPhase::Boundary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrepPhase {
    /// `StudyParams::from_config` (config.json parsing and semantic validation).
    Config,
    /// `prepare_stochastic` (PAR estimation, opening trees, stochastic context).
    Stochastic,
    /// `prepare_hydro_models_from_artifacts` (production/evaporation models).
    HydroModels,
    /// Scalar-parameter resolution and constraint validation (no-boundary decks only; boundary decks run this inside `StudySetup::new`).
    GenericConstraints,
    /// `StudySetup::new_with_boundary_requirements` on a deck without a boundary
    /// policy, which builds the scenario libraries and runs the forward-scheme
    /// historical checks. A deck with a boundary policy reports the same failure as
    /// [`PrepPhase::Boundary`], so the kinds that deck reports do not change.
    StudySetup,
    /// The boundary-policy steps of a deck with `config.policy.boundary` set: the
    /// requirements read (`resolve_boundary_state_requirements`), study construction
    /// (`StudySetup::new_with_boundary_requirements`) and `reconcile_boundary_policy`
    /// (checkpoint reconciliation against the terminal entity manifest).
    Boundary,
}

// ── validate_study ────────────────────────────────────────────────────────────

/// The loaded case and the parsed configuration [`validate_study`] checks.
#[derive(Debug)]
pub struct ValidateRequest<'a> {
    /// The case directory, for the files `prepare_stochastic` and the boundary
    /// checkpoint read from disk.
    pub case_dir: &'a Path,
    /// The parsed `config.json` (with any front-end overrides applied).
    pub config: &'a Config,
    /// The loaded system.
    pub system: System,
    /// The pre-parsed case artifacts.
    pub artifacts: CaseArtifacts,
}

/// What [`validate_study`] hands back when every check passes.
#[derive(Debug)]
pub struct ValidatedStudy {
    /// The system after `prepare_stochastic` (estimated PAR models included).
    pub system: System,
    /// The boundary reconciliation, when `config.policy.boundary` is set.
    pub boundary: Option<BoundaryReconciliation>,
    /// Warnings raised by the checks, for the front ends to render.
    pub warnings: Vec<ReportEntry>,
}

/// A failed [`PrepPhase`] and the error it raised.
#[derive(Debug)]
pub struct PhaseFailure {
    /// The phase that failed.
    pub phase: PrepPhase,
    /// The error the phase raised.
    pub error: SddpError,
}

impl PhaseFailure {
    /// The stable error-kind string programmatic callers filter on.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        prep_phase_metadata(self.phase, &self.error).0
    }

    /// The `"<file label>: <message>"` line both front ends present.
    #[must_use]
    pub fn report(&self) -> String {
        format!(
            "{}: {}",
            prep_phase_metadata(self.phase, &self.error).1,
            self.error
        )
    }
}

/// Why [`validate_study`] stopped.
#[derive(Debug)]
pub enum ValidateFailure {
    /// `config.json` names a scenario source the study cannot resolve.
    ScenarioSource(LoadError),
    /// A preparation phase failed.
    Phase(PhaseFailure),
}

fn at(phase: PrepPhase) -> impl FnOnce(SddpError) -> ValidateFailure {
    move |error| ValidateFailure::Phase(PhaseFailure { phase, error })
}

/// Run every pre-solver check `cobre run` reaches before it starts solving, in
/// the order the run reaches them, stopping at the first failure.
///
/// A deck without a boundary policy runs the generic-constraint parameter guard
/// before it builds the study, so a scalar-parameter gap keeps
/// [`PrepPhase::GenericConstraints`]; its study-construction failures report
/// [`PrepPhase::StudySetup`]. A deck with a boundary policy skips the standalone
/// guard (construction runs it) and reports every construction failure as
/// [`PrepPhase::Boundary`].
///
/// # Errors
///
/// Returns [`ValidateFailure::ScenarioSource`] when the training scenario source
/// cannot be resolved, and [`ValidateFailure::Phase`] with the failing
/// [`PrepPhase`] for every other check.
pub fn validate_study(request: ValidateRequest<'_>) -> Result<ValidatedStudy, ValidateFailure> {
    let ValidateRequest {
        case_dir,
        config,
        system,
        artifacts,
    } = request;

    let params = StudyParams::from_config(config, Vec::new()).map_err(at(PrepPhase::Config))?;
    let training_source = config
        .training_scenario_source(&case_dir.join("config.json"))
        .map_err(ValidateFailure::ScenarioSource)?;
    let requirements =
        resolve_boundary_state_requirements(case_dir, config).map_err(at(PrepPhase::Boundary))?;
    let prepared = prepare_stochastic(
        system,
        case_dir,
        config,
        params.seed,
        &training_source,
        requirements.inflow_lag_depth(),
    )
    .map_err(at(PrepPhase::Stochastic))?;
    let hydro_models =
        prepare_hydro_models_from_artifacts(&prepared.system, &artifacts, false, None)
            .map_err(at(PrepPhase::HydroModels))?;

    let boundary_policy = config.policy.boundary.as_ref();
    let setup_phase = if boundary_policy.is_some() {
        PrepPhase::Boundary
    } else {
        validate_generic_constraint_parameters(
            &prepared.system,
            &hydro_models,
            &artifacts.scalar_parameters,
            params.cost_scale_factor,
        )
        .map_err(at(PrepPhase::GenericConstraints))?;
        PrepPhase::StudySetup
    };
    let setup = StudySetup::new_with_boundary_requirements(
        &prepared.system,
        config,
        prepared.stochastic,
        hydro_models,
        requirements,
        artifacts.scalar_parameters,
    )
    .map_err(at(setup_phase))?;

    let boundary = boundary_policy
        .map(|bp| reconcile_boundary_policy(&setup, &prepared.system, bp, case_dir))
        .transpose()
        .map_err(at(PrepPhase::Boundary))?;

    Ok(ValidatedStudy {
        system: prepared.system,
        boundary,
        warnings: Vec::new(),
    })
}

// ── prep_phase_metadata ───────────────────────────────────────────────────────

/// Return the structured error kind and best-effort file label for a
/// preparation-phase error.
///
/// Both the CLI and the Python binding call this to derive:
///
/// * `kind` — a stable, camel-cased string suitable for programmatic filtering
///   (e.g. `"ConfigValidationError"`, `"StochasticPreparationError"`).
/// * `file_label` — a short relative path pointing to the file most likely
///   responsible for the error, suitable for user-facing diagnostics.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::validate_phases::{PrepPhase, prep_phase_metadata};
/// use cobre_sddp::SddpError;
///
/// let err = SddpError::Validation("unsupported stopping rule".to_string());
/// let (kind, file) = prep_phase_metadata(PrepPhase::Config, &err);
/// assert_eq!(kind, "ConfigValidationError");
/// assert_eq!(file, "config.json");
/// ```
#[must_use]
pub fn prep_phase_metadata(phase: PrepPhase, err: &SddpError) -> (&'static str, &'static str) {
    let kind = match phase {
        PrepPhase::Config => "ConfigValidationError",
        PrepPhase::Stochastic => "StochasticPreparationError",
        PrepPhase::HydroModels => "HydroModelsPreparationError",
        PrepPhase::GenericConstraints => "GenericConstraintValidationError",
        PrepPhase::StudySetup => "StudySetupError",
        PrepPhase::Boundary => "BoundaryReconciliationError",
    };

    let file_label = match (phase, err) {
        (PrepPhase::Stochastic, SddpError::Stochastic(_)) => "scenarios/inflow_history.parquet",
        (PrepPhase::Stochastic, _) | (PrepPhase::StudySetup, SddpError::Stochastic(_)) => {
            "scenarios/"
        }
        (PrepPhase::Config | PrepPhase::StudySetup, _) => "config.json",
        (PrepPhase::HydroModels, _) => "system/hydro_production_models.json",
        (PrepPhase::GenericConstraints, _) => "constraints/",
        (PrepPhase::Boundary, _) => "policy.boundary",
    };

    (kind, file_label)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::fs;

    use cobre_stochastic::StochasticError;
    use serde_json::{Value, json};
    use tempfile::TempDir;

    use super::*;

    fn copy_dir(src: &Path, dst: &Path) {
        fs::create_dir_all(dst).unwrap();
        for entry in fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let target = dst.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    fn edit_json(path: &Path, edit: impl FnOnce(&mut Value)) {
        let mut value: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        edit(&mut value);
        fs::write(path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
    }

    fn validate_toy_case(mutate: impl FnOnce(&Path)) -> Result<ValidatedStudy, ValidateFailure> {
        let case = TempDir::new().unwrap();
        copy_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy"),
            case.path(),
        );
        mutate(case.path());
        let (loaded, _) = cobre_io::validate_case_with_artifacts(case.path()).unwrap();
        let config = cobre_io::parse_config(&case.path().join("config.json")).unwrap();
        validate_study(ValidateRequest {
            case_dir: case.path(),
            config: &config,
            system: loaded.system,
            artifacts: loaded.artifacts,
        })
    }

    fn phase_failure(result: Result<ValidatedStudy, ValidateFailure>) -> PhaseFailure {
        match result {
            Err(ValidateFailure::Phase(failure)) => failure,
            other => panic!("expected a phase failure, got {other:?}"),
        }
    }

    #[test]
    fn validate_study_accepts_a_valid_case_with_no_stage_warnings() {
        let validated = validate_toy_case(|_| {}).unwrap();
        assert!(validated.boundary.is_none());
        assert!(validated.warnings.is_empty());
    }

    #[test]
    fn validate_study_refuses_a_forward_historical_case_at_study_setup() {
        let failure = phase_failure(validate_toy_case(|case| {
            edit_json(&case.join("config.json"), |config| {
                config["training"]["scenario_source"]["inflow"] = json!({"scheme": "historical"});
            });
        }));
        assert_eq!(failure.phase, PrepPhase::StudySetup);
        assert_eq!(failure.kind(), "StudySetupError");
        assert!(
            failure.report().starts_with(
                "scenarios/: stochastic error: insufficient data: no valid historical windows found"
            ),
            "got: {}",
            failure.report()
        );
    }

    #[test]
    fn validate_study_runs_the_generic_constraint_guard_before_study_setup_without_a_boundary() {
        let failure = phase_failure(validate_toy_case(|case| {
            fs::create_dir_all(case.join("constraints")).unwrap();
            fs::write(
                case.join("constraints/generic_parameters.json"),
                json!({"scalar_parameters": [
                    {"id": 1, "name": "p_gap", "kind": "seasonal", "values": [[5, 1.0]]}
                ]})
                .to_string(),
            )
            .unwrap();
        }));
        assert_eq!(failure.phase, PrepPhase::GenericConstraints);
        assert_eq!(
            failure.report(),
            "constraints/: configuration validation error: parameter 'p_gap': \
             no seasonal value for season_id=0 (needed by stage 0)"
        );
    }

    #[test]
    fn study_setup_phase_labels_stochastic_errors_with_the_scenarios_directory() {
        let stochastic = SddpError::Stochastic(StochasticError::InsufficientData {
            context: "no valid historical windows found".to_string(),
        });
        assert_eq!(
            prep_phase_metadata(PrepPhase::StudySetup, &stochastic),
            ("StudySetupError", "scenarios/")
        );
        let validation = SddpError::Validation("bad window".to_string());
        assert_eq!(
            prep_phase_metadata(PrepPhase::StudySetup, &validation),
            ("StudySetupError", "config.json")
        );
    }

    #[test]
    fn config_phase_always_returns_config_json() {
        let err = SddpError::Validation("bad window".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::Config, &err);
        assert_eq!(kind, "ConfigValidationError");
        assert_eq!(file, "config.json");
    }

    #[test]
    fn stochastic_phase_with_stochastic_error_returns_inflow_history() {
        let err = SddpError::Stochastic(StochasticError::InsufficientData {
            context: "only 2 years".to_string(),
        });
        let (kind, file) = prep_phase_metadata(PrepPhase::Stochastic, &err);
        assert_eq!(kind, "StochasticPreparationError");
        assert_eq!(file, "scenarios/inflow_history.parquet");
    }

    #[test]
    fn stochastic_phase_with_non_stochastic_error_returns_scenarios_dir() {
        let err = SddpError::Validation("something else".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::Stochastic, &err);
        assert_eq!(kind, "StochasticPreparationError");
        assert_eq!(file, "scenarios/");
    }

    #[test]
    fn hydro_models_phase_returns_production_models_file() {
        let err = SddpError::Validation("model error".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::HydroModels, &err);
        assert_eq!(kind, "HydroModelsPreparationError");
        assert_eq!(file, "system/hydro_production_models.json");
    }

    #[test]
    fn boundary_phase_returns_boundary_reconciliation_error() {
        let err = SddpError::Validation("mismatched hydro set".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::Boundary, &err);
        assert_eq!(kind, "BoundaryReconciliationError");
        assert_eq!(file, "policy.boundary");
    }

    #[test]
    fn generic_constraints_phase_returns_constraints_dir() {
        let err = SddpError::Validation("unresolved scalar parameter".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::GenericConstraints, &err);
        assert_eq!(kind, "GenericConstraintValidationError");
        assert_eq!(file, "constraints/");
    }

    #[test]
    fn all_phases_produce_non_empty_kind_and_file() {
        let err = SddpError::Validation("test".to_string());
        for phase in [
            PrepPhase::Config,
            PrepPhase::Stochastic,
            PrepPhase::HydroModels,
            PrepPhase::GenericConstraints,
            PrepPhase::StudySetup,
            PrepPhase::Boundary,
        ] {
            let (kind, file) = prep_phase_metadata(phase, &err);
            assert!(!kind.is_empty(), "kind must not be empty for {phase:?}");
            assert!(
                !file.is_empty(),
                "file_label must not be empty for {phase:?}"
            );
        }
    }
}
