//! `cobre validate <CASE_DIR>` subcommand.
//!
//! Runs the six-layer validation pipeline followed by the pre-solver
//! preparation phases and prints a structured diagnostic report to stdout —
//! or, with `--json`, a single machine-readable JSON object: the boundary
//! reconciliation outcome on success, or an `error` object naming the first
//! failing phase. Stdout under `--json` is always one such object or empty,
//! never human report text. No banner or progress bar — the output is the
//! deliverable.
//!
//! ## Validation contract
//!
//! If `cobre validate <CASE_DIR>` exits 0, then `cobre run <CASE_DIR>` will not
//! fail in any phase before the solver begins iterating. The pre-solver
//! phases exercised here are:
//!
//! 1. [`cobre_sddp::StudyParams::from_config`] — validates `config.json` fields
//!    that are only checked at algorithm startup and surfaces deprecation
//!    warnings for fields that are scheduled for removal.
//! 2. [`cobre_sddp::prepare_stochastic`] — runs PAR estimation from inflow
//!    history, loads user opening trees, and builds the stochastic context.
//! 3. [`cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts`] — resolves
//!    production and evaporation models from the pre-parsed artifact bundle.
//! 4. [`cobre_sddp::validate_generic_constraint_parameters`] — builds the resolved
//!    scalar-parameter table and rejects a generic constraint that references an
//!    unresolved id. Run for a deck with no boundary policy; a boundary deck
//!    runs the same guard inside [`cobre_sddp::StudySetup::new`] (phase 5).
//! 5. When `config.policy.boundary` is configured, [`cobre_sddp::StudySetup::new`]
//!    plus [`cobre_sddp::load_boundary_cuts`] — builds the study and reconciles
//!    the boundary policy against its terminal manifest, without solving.

use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use clap::Args;
use cobre_core::{ScalarParameter, System};
use cobre_io::{BoundaryPolicy, Config, LoadError, ValidationReport, validate_case_with_artifacts};
use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;
use cobre_sddp::policy::orchestration;
use cobre_sddp::validate_phases::{PrepPhase, prep_phase_metadata};
use cobre_sddp::{
    BoundaryLoadRequest, BoundaryReconciliationReport, PrepareHydroModelsResult, SddpError,
    StudyParams, StudySetup, load_boundary_cuts, prepare_stochastic,
    resolve_boundary_state_requirements, study_horizon_end, validate_generic_constraint_parameters,
};
use cobre_stochastic::StochasticContext;
use console::{Term, style};
use serde::Serialize;

use crate::error::CliError;

/// Arguments for the `cobre validate` subcommand.
#[derive(Debug, Args)]
#[command(about = "Validate a case directory and print a structured diagnostic report")]
pub struct ValidateArgs {
    /// Path to the case directory to validate.
    pub case_dir: PathBuf,

    /// Emit the boundary reconciliation outcome as a single JSON object to
    /// stdout instead of the human-readable report.
    #[arg(long)]
    pub json: bool,
}

/// Success outcome (`configured`/`boundary_date`/`report` populated, `error` None) and error outcome (`configured`/`boundary_date`/`report` None, `error` populated) never overlap.
#[derive(Debug, Serialize)]
struct ValidateBoundaryOutput {
    /// Whether `policy.boundary` is configured in this case's `config.json`.
    configured: Option<bool>,
    /// The date the boundary pool was selected against, when `configured`
    /// is `Some(true)`.
    boundary_date: Option<NaiveDate>,
    /// The reconciliation report when `configured` is `Some(true)`.
    report: Option<BoundaryReconciliationReport>,
    /// The failing phase and message, populated only on an early abort.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ValidateErrorOutput>,
}

/// Computed once and carried to both `--json` and human-mode outputs.
#[derive(Debug)]
struct BoundaryOutcome {
    boundary_date: NaiveDate,
    report: BoundaryReconciliationReport,
}

/// Early-abort failure. `phase` is the stable kind string from [`prep_phase_metadata`] or [`LoadError::kind`] (same string programmatic callers filter on).
#[derive(Debug, Serialize)]
struct ValidateErrorOutput {
    phase: String,
    message: String,
}

impl ValidateBoundaryOutput {
    fn success(outcome: Option<BoundaryOutcome>) -> Self {
        Self {
            configured: Some(outcome.is_some()),
            boundary_date: outcome.as_ref().map(|o| o.boundary_date),
            report: outcome.map(|o| o.report),
            error: None,
        }
    }

    fn error(phase: &str, message: &str) -> Self {
        Self {
            configured: None,
            boundary_date: None,
            report: None,
            error: Some(ValidateErrorOutput {
                phase: phase.to_string(),
                message: message.to_string(),
            }),
        }
    }
}

fn format_constraint_description(
    term: &Term,
    description: &str,
    warning_count: usize,
    path: &Path,
) {
    let error_lines: Vec<&str> = description.lines().collect();
    let _ = term.write_line(&format!(
        "Validation: {} errors, {} warnings in {}",
        error_lines.len(),
        warning_count,
        path.display()
    ));
    for line in error_lines {
        let _ = term.write_line(&format!("{} {line}", style("error:").red().bold()));
    }
}

/// Formats validation warnings as report lines (empty when no warnings). `report.error_count` is always
/// zero here — [`validate_case_with_artifacts`] returns `Err` on any error.
fn report_lines(report: &ValidationReport, case_dir: &Path) -> Vec<String> {
    if report.warning_count == 0 {
        return Vec::new();
    }
    let mut lines = vec![format!(
        "Validation: 0 errors, {} warnings in {}",
        report.warning_count,
        case_dir.display()
    )];
    for entry in &report.warnings {
        let location = if let Some(entity) = &entry.entity {
            format!("{} ({})", entry.file, entity)
        } else {
            entry.file.clone()
        };
        lines.push(format!(
            "{} {location}: {}",
            style("warning:").yellow().bold(),
            entry.message
        ));
    }
    lines
}

/// Computes the stable phase kind and `"file_label: message"` report string, shared by both human and `--json` outputs to prevent drift.
fn describe_prep_error(phase: PrepPhase, err: &SddpError) -> (&'static str, String) {
    let (kind, file_label) = prep_phase_metadata(phase, err);
    (kind, format!("{file_label}: {err}"))
}

fn print_prep_error(term: &Term, report: &str, case_dir: &Path) {
    let _ = term.write_line(&format!(
        "Validation: 1 errors, 0 warnings in {}",
        case_dir.display()
    ));
    let _ = term.write_line(&format!("{} {report}", style("error:").red().bold()));
}

/// Handle a pre-solver preparation-phase failure. `stdout_sink` is `None`
/// under `--json`, where the error object replaces the human report.
fn prep_error_to_cli_error(
    stdout_sink: Option<&Term>,
    json: bool,
    phase: PrepPhase,
    err: &SddpError,
    case_dir: &Path,
) -> Result<CliError, CliError> {
    let (kind, report) = describe_prep_error(phase, err);
    if let Some(term) = stdout_sink {
        print_prep_error(term, &report, case_dir);
    }
    if json {
        emit_validate_json(&ValidateBoundaryOutput::error(kind, &report))?;
    }
    Ok(CliError::Validation {
        report,
        already_rendered: true,
    })
}

/// Run a pre-solver preparation phase; an `Err` is reported (human and
/// `--json`) before it is returned.
fn run_prep_phase<T>(
    result: Result<T, SddpError>,
    stdout_sink: Option<&Term>,
    json: bool,
    phase: PrepPhase,
    case_dir: &Path,
) -> Result<T, CliError> {
    match result {
        Ok(value) => Ok(value),
        Err(ref err) => Err(prep_error_to_cli_error(
            stdout_sink,
            json,
            phase,
            err,
            case_dir,
        )?),
    }
}

/// Build a `StudySetup` from the parsed config and reconcile
/// `config.policy.boundary` against its terminal manifest, without solving.
fn reconcile_boundary(
    case_dir: &Path,
    config: &Config,
    bp: &BoundaryPolicy,
    system: &System,
    stochastic: StochasticContext,
    hydro_models: PrepareHydroModelsResult,
    scalar_parameters: Vec<ScalarParameter>,
) -> Result<BoundaryOutcome, SddpError> {
    let boundary_path = bp.checkpoint_path(case_dir);

    // Resolve before building the layout so validate mirrors the run path.
    let boundary_requirements = resolve_boundary_state_requirements(case_dir, config)?;

    let setup = StudySetup::new_with_boundary_requirements(
        system,
        config,
        stochastic,
        hydro_models,
        boundary_requirements,
        scalar_parameters,
    )?;

    // Rationale: the cast cannot truncate — `state_dimension` counts FCF
    // state variables (one per reservoir/lag), bounded by the validated study
    // dimensions and far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let state_dim = setup.fcf.state_dimension as u32;
    let current_manifest = setup.build_terminal_entity_manifest(system);
    let fixed_windows = setup.build_terminal_fixed_post_horizon_windows(system);

    let Some(boundary_date) = study_horizon_end(system) else {
        return Err(SddpError::Validation(format!(
            "case {}: the study declares no non-negative stage, so it has no boundary date to \
             load a boundary policy against",
            case_dir.display()
        )));
    };

    let study_seasons = orchestration::build_season_manifest(system);
    let boundary_cuts = load_boundary_cuts(
        &BoundaryLoadRequest::new(
            &boundary_path,
            boundary_date,
            state_dim,
            &current_manifest,
            setup.inputs.stage_data.stage_templates.cost_scale_factor,
        )
        .with_fixed_windows(&fixed_windows)
        .with_inflow_lag_depth(setup.boundary_requirements().inflow_lag_depth())
        .with_study_seasons(&study_seasons)
        .with_strict(bp.strict),
    )?;

    Ok(BoundaryOutcome {
        boundary_date,
        report: boundary_cuts.report().clone(),
    })
}

/// Reconciles `config.policy.boundary` when configured, mapping a reject to [`CliError::Validation`]. Returns `Ok(None)` when no boundary is configured (no `StudySetup` work runs).
fn run_boundary_check(
    case_dir: &Path,
    config: &Config,
    system: &System,
    stochastic: StochasticContext,
    hydro_models: PrepareHydroModelsResult,
    scalar_parameters: Vec<ScalarParameter>,
    stdout: Option<&Term>,
    json: bool,
) -> Result<Option<BoundaryOutcome>, CliError> {
    let Some(bp) = config.policy.boundary.as_ref() else {
        return Ok(None);
    };

    match reconcile_boundary(
        case_dir,
        config,
        bp,
        system,
        stochastic,
        hydro_models,
        scalar_parameters,
    ) {
        Ok(outcome) => Ok(Some(outcome)),
        Err(ref err) => Err(prep_error_to_cli_error(
            stdout,
            json,
            PrepPhase::Boundary,
            err,
            case_dir,
        )?),
    }
}

/// Runs the scalar-parameter presence guard for decks without a boundary policy (boundary decks run the guard inside [`StudySetup::new`], so this is a no-op there). Rejects map to [`CliError::Validation`].
fn run_generic_constraint_parameter_check(
    case_dir: &Path,
    config: &Config,
    system: &System,
    hydro_models: &PrepareHydroModelsResult,
    scalar_parameters: &[ScalarParameter],
    cost_scale_factor: f64,
    stdout: Option<&Term>,
    json: bool,
) -> Result<(), CliError> {
    if config.policy.boundary.is_some() {
        return Ok(());
    }
    run_prep_phase(
        validate_generic_constraint_parameters(
            system,
            hydro_models,
            scalar_parameters,
            cost_scale_factor,
        ),
        stdout,
        json,
        PrepPhase::GenericConstraints,
        case_dir,
    )
}

/// Serialize `output` as `cobre validate --json`'s single stdout JSON object.
/// Stdout carries exactly one JSON object and no human-readable text.
fn emit_validate_json(output: &ValidateBoundaryOutput) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(output).map_err(|e| CliError::Internal {
        message: format!("failed to serialize validate output: {e}"),
    })?;
    println!("{json}");
    Ok(())
}

/// Emit `--json`'s error object ahead of the caller's own `CliError` return.
/// No-op when `json` is false.
fn emit_json_error(json: bool, kind: &str, message: &str) -> Result<(), CliError> {
    if json {
        emit_validate_json(&ValidateBoundaryOutput::error(kind, message))?;
    }
    Ok(())
}

/// Execute the `validate` subcommand, printing a structured diagnostic report
/// (with any pipeline warnings) to stdout. Honors the module's validation contract:
/// exit 0 implies `cobre run` will not fail before the solver begins iterating.
///
/// # Errors
///
/// Returns [`CliError::Validation`] when the case directory fails validation,
/// [`CliError::Io`] on filesystem errors, or [`CliError::Internal`] for
/// unexpected parse or schema failures.
pub fn execute(args: &ValidateArgs) -> Result<(), CliError> {
    let stdout = Term::stdout();
    let stdout_sink = (!args.json).then_some(&stdout);

    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("case directory not found: {}", args.case_dir.display()),
            ),
            context: args.case_dir.display().to_string(),
        });
    }

    // Reuses the pre-parsed CaseArtifacts to avoid re-reading disk.
    let (loaded, report) = match validate_case_with_artifacts(&args.case_dir) {
        Ok(result) => result,
        Err(err) => {
            let kind = err.kind();
            let message = err.to_string();
            match err {
                LoadError::IoError { path, source } => {
                    emit_json_error(args.json, kind, &message)?;
                    return Err(CliError::Io {
                        source,
                        context: path.display().to_string(),
                    });
                }
                LoadError::ConstraintError { description } => {
                    // Warnings are not available when errors abort the pipeline, so report 0.
                    if let Some(term) = stdout_sink {
                        format_constraint_description(term, &description, 0, &args.case_dir);
                    }
                    emit_json_error(args.json, kind, &description)?;
                    return Err(CliError::Validation {
                        report: description,
                        already_rendered: true,
                    });
                }
                _ => {
                    emit_json_error(args.json, kind, &message)?;
                    return Err(CliError::Internal { message });
                }
            }
        }
    };

    let system = loaded.system;
    let artifacts = loaded.artifacts;

    let config_path = args.case_dir.join("config.json");
    let config = match cobre_io::parse_config(&config_path) {
        Ok(config) => config,
        Err(err) => {
            emit_json_error(args.json, err.kind(), &err.to_string())?;
            return Err(CliError::from(err));
        }
    };

    let study_params = run_prep_phase(
        StudyParams::from_config(&config, Vec::new()),
        stdout_sink,
        args.json,
        PrepPhase::Config,
        &args.case_dir,
    )?;

    let seed = study_params.seed;

    // config_path is used only for historical-years look-up and error messages, not file operations.
    let training_source = match config.training_scenario_source(&config_path) {
        Ok(source) => source,
        Err(err) => {
            emit_json_error(args.json, err.kind(), &err.to_string())?;
            return Err(CliError::from(err));
        }
    };

    // Runs the expensive PAR estimation/opening-trees step to guarantee that exit-0 means full parity with `run`.
    let boundary_requirements = resolve_boundary_state_requirements(&args.case_dir, &config)?;
    let prepared = run_prep_phase(
        prepare_stochastic(
            system,
            &args.case_dir,
            &config,
            seed,
            &training_source,
            boundary_requirements.inflow_lag_depth(),
        ),
        stdout_sink,
        args.json,
        PrepPhase::Stochastic,
        &args.case_dir,
    )?;

    // Reuses the parsed bundle instead of re-reading disk.
    let hydro_models = run_prep_phase(
        prepare_hydro_models_from_artifacts(&prepared.system, &artifacts, false, None),
        stdout_sink,
        args.json,
        PrepPhase::HydroModels,
        &args.case_dir,
    )?;

    if !args.json {
        let _ = stdout.write_line(&format!(
            "Valid case: {} buses, {} hydros, {} thermals, {} lines",
            prepared.system.n_buses(),
            prepared.system.n_hydros(),
            prepared.system.n_thermals(),
            prepared.system.n_lines(),
        ));
        for line in report_lines(&report, &args.case_dir) {
            let _ = stdout.write_line(&line);
        }
    }

    run_generic_constraint_parameter_check(
        &args.case_dir,
        &config,
        &prepared.system,
        &hydro_models,
        &artifacts.scalar_parameters,
        study_params.cost_scale_factor,
        stdout_sink,
        args.json,
    )?;

    let boundary_outcome = run_boundary_check(
        &args.case_dir,
        &config,
        &prepared.system,
        prepared.stochastic,
        hydro_models,
        artifacts.scalar_parameters,
        stdout_sink,
        args.json,
    )?;

    if args.json {
        emit_validate_json(&ValidateBoundaryOutput::success(boundary_outcome))?;
    } else if let Some(outcome) = &boundary_outcome {
        let _ = stdout.write_line(&format!(
            "boundary policy priced at {}",
            outcome.boundary_date
        ));
        let _ = stdout.write_line(&outcome.report.summary_line());
        for line in outcome.report.detail_lines() {
            tracing::debug!("{line}");
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use cobre_io::{ReportEntry, ValidationReport};

    fn make_report() -> ValidationReport {
        ValidationReport {
            error_count: 0,
            warning_count: 1,
            errors: Vec::new(),
            warnings: vec![ReportEntry {
                kind: "UnusedEntity".to_string(),
                file: "system/thermals.json".to_string(),
                entity: None,
                message: "thermal has zero capacity".to_string(),
            }],
        }
    }

    use super::*;

    #[test]
    fn format_report_contains_warning_label() {
        let path = PathBuf::from("/case/dir");
        let output = report_lines(&make_report(), &path).join("\n");
        assert!(
            output.contains("warning:"),
            "expected 'warning:' in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_file_path() {
        let path = PathBuf::from("/case/dir");
        let output = report_lines(&make_report(), &path).join("\n");
        assert!(
            output.contains("system/thermals.json"),
            "expected file path in output, got: {output}"
        );
    }

    #[test]
    fn format_report_summary_header_present() {
        let path = PathBuf::from("/case/dir");
        let output = report_lines(&make_report(), &path).join("\n");
        assert!(
            output.contains("0 errors") && output.contains("1 warnings"),
            "expected summary header with counts, got: {output}"
        );
    }

    #[test]
    fn report_lines_entity_present_renders_file_and_entity() {
        let report = ValidationReport {
            error_count: 0,
            warning_count: 1,
            errors: Vec::new(),
            warnings: vec![ReportEntry {
                kind: "UnusedEntity".to_string(),
                file: "system/buses.json".to_string(),
                entity: Some("bus_01".to_string()),
                message: "bus is unreferenced".to_string(),
            }],
        };
        let output = report_lines(&report, &PathBuf::from("/case/dir")).join("\n");
        assert!(
            output.contains("system/buses.json (bus_01)"),
            "entity-present location must render 'file (entity)', got: {output}"
        );
    }
}
