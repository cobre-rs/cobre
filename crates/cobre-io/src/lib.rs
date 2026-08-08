//! # cobre-io
//!
//! Case directory loading, validation, and result writing for the
//! [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate provides two top-level entry points:
//!
//! - [`load_case`] — reads a case directory and produces a fully-validated
//!   [`cobre_core::System`] ready for use by the solver.
//! - [`write_results`] — accepts aggregate result types and writes all output
//!   artifacts to a specified root directory.
//!
//! ## Loading pipeline
//!
//! [`load_case`] executes a six-layer validation pipeline:
//!
//! 1. **Structural validation** — checks that required files exist on disk and records
//!    which optional files are present ([`validation::structural`]).
//! 2. **Schema validation** — verifies required fields, types, and value ranges.
//! 3. **Referential integrity** — checks all entity ID cross-references are resolvable.
//! 4. **Dimensional consistency** — cross-file coverage checks (e.g., inflow params
//!    cover all hydros).
//! 5. **Semantic validation** — domain business rules (acyclic cascade, penalty ordering,
//!    PAR stationarity, etc.).
//! 6. **Cross-file resolution and cross-validation** — multi-file consistency checks that
//!    span the parsed data assembled by earlier layers (productivity source conflict
//!    detection, scalar-parameter hydro-ID existence, per-stage length checks).
//!
//! All validation diagnostics are collected by [`validation::ValidationContext`] before
//! failing, so users see every problem in a single report.  Final errors are reported
//! via [`LoadError`], which carries enough context for diagnostic messages without
//! re-reading input files.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the current status.

#[cfg(feature = "schema")]
pub mod schema;

pub mod broadcast;
pub mod config;
pub mod constraints;
pub mod error;
pub mod extensions;
pub mod initial_conditions;
pub mod output;
pub(crate) mod parquet_helpers;
pub mod penalties;
pub(crate) mod pipeline;
pub mod report;
pub mod resolution;
pub mod scenarios;
pub mod stages;
pub mod system;
pub mod validation;

pub use broadcast::{
    BroadcastComputedParameter, BroadcastParameterKind, BroadcastScalarParameter,
    deserialize_parameters, deserialize_system, serialize_parameters, serialize_system,
};
pub use config::{
    BoundaryPolicy, Config, EstimationConfig, OrderSelectionMethod, PolicyMode, parse_config,
};
pub use constraints::{
    BusPenaltyOverrideRow, ContractBoundsRow, GenericConstraintBoundsRow, HydroBoundsRow,
    HydroPenaltyOverrideRow, LineBoundsRow, LinePenaltyOverrideRow, NcsPenaltyOverrideRow,
    PumpingBoundsRow, ThermalBoundsRow, load_contract_bounds, load_generic_constraint_bounds,
    load_generic_constraints, load_hydro_bounds, load_line_bounds, load_penalty_overrides_bus,
    load_penalty_overrides_hydro, load_penalty_overrides_line, load_penalty_overrides_ncs,
    load_pumping_bounds, load_thermal_bounds, parse_contract_bounds,
    parse_generic_constraint_bounds, parse_generic_constraints, parse_hydro_bounds,
    parse_line_bounds, parse_penalty_overrides_bus, parse_penalty_overrides_hydro,
    parse_penalty_overrides_line, parse_penalty_overrides_ncs, parse_pumping_bounds,
    parse_thermal_bounds,
};
pub use error::LoadError;
pub use extensions::{
    EvaporationModelRow, FittingWindow, FphaColumnLayout, FphaDeviationPointRow, FphaHyperplaneRow,
    HydroEnergyProductivityRow, HydroGeometryRow, HydroReferenceVolumeFractions,
    PlaneReductionConfig, ProductionModelConfig, ProductionModelFile, SeasonConfig, SelectionMode,
    StageRange, build_hydro_reference_volumes_resolved, load_fpha_hyperplanes,
    load_hydro_energy_productivity, load_hydro_geometry, load_production_models,
    load_scalar_parameters_json, parse_evaporation_models, parse_fpha_deviation_points,
    parse_fpha_hyperplanes, parse_hydro_energy_productivity, parse_hydro_geometry,
    parse_production_models, parse_scalar_parameters_json,
};
pub use initial_conditions::parse_initial_conditions;
pub use output::policy::{
    ENTITY_SLOT_DELIVERY_ANCHOR_SENTINEL, EntitySlot, OwnedPolicyBasisRecord, OwnedPolicyCutRecord,
    PolicyBasisRecord, PolicyCheckpoint, PolicyCheckpointMetadata, PolicyCutRecord,
    StageCutsPayload, StageCutsReadResult, StageStatesPayload, StageStatesReadResult,
    deserialize_stage_basis, deserialize_stage_cuts, deserialize_stage_states,
    read_policy_checkpoint, serialize_stage_basis, serialize_stage_cuts, serialize_stage_states,
    write_policy_checkpoint,
};
pub use output::{
    ConvergenceSummary, DeviationSummary, DeviationWorstEntry, DistributionInfo, HostLayout,
    IterationRecord, MetadataBounds, MetadataConfiguration, MetadataConvergence, MetadataCost,
    MetadataIterations, MetadataProblemDimensions, MetadataRowPool, MetadataScenarios,
    MetadataSimulationSolveStats, MetadataTrainingSolveStats, OutputContext, OutputError,
    ParquetWriterConfig, RankAffinity, RowPoolStatistics, RowSelectionRecord, SetupTimings,
    SimulationMetadata, SimulationOutput, SolverStatsRow, TrainingMetadata, TrainingOutput,
    TrainingParquetWriter, WorkerTimingRecord, get_hostname, now_iso8601, read_convergence_summary,
    read_hydro_model_summary, read_provenance_report, read_simulation_metadata,
    read_training_metadata, write_dictionaries, write_evaporation_models,
    write_fpha_deviation_points, write_fpha_hyperplanes, write_hydro_model_summary,
    write_provenance_report, write_results, write_row_selection_records, write_scaling_report,
    write_simulation_metadata, write_simulation_results, write_simulation_solver_stats,
    write_solver_stats, write_training_metadata, write_training_results,
};
pub use penalties::parse_penalties;
pub use report::{ReportEntry, ValidationReport, generate_report};
pub use resolution::{resolve_bounds, resolve_penalties};
pub use scenarios::{
    BlockFactor, ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, InflowArCoefficientRow,
    InflowHistoryRow, InflowSeasonalStatsRow, LoadFactorEntry, LoadSeasonalStatsRow,
    NoiseOpeningRow, ScenarioData, assemble_inflow_models, assemble_load_models, load_correlation,
    load_external_inflow_scenarios, load_external_load_scenarios, load_external_ncs_scenarios,
    load_inflow_ar_coefficients, load_inflow_history, load_inflow_seasonal_stats,
    load_load_factors, load_load_seasonal_stats, load_noise_openings, load_scenarios,
    parse_correlation, parse_external_inflow_scenarios, parse_external_load_scenarios,
    parse_external_ncs_scenarios, parse_inflow_ar_coefficients, parse_inflow_history,
    parse_inflow_seasonal_stats, parse_load_factors, parse_load_seasonal_stats,
};
pub use stages::{StagesData, build_season_stage_map, parse_stages};
pub use system::{
    load_energy_contracts, load_non_controllable_sources, load_pumping_stations, parse_buses,
    parse_energy_contracts, parse_hydros, parse_lines, parse_non_controllable_sources,
    parse_pumping_stations, parse_thermals,
};
pub use validation::scalar_parameters::validate_scalar_parameters;
pub use validation::structural::{FileManifest, validate_structure};
pub use validation::{ErrorKind, Severity, ValidationContext, ValidationEntry};

use cobre_core::{ScalarParameter, System};
use std::path::Path;

/// Auxiliary rows produced by the load pipeline alongside [`System`].
///
/// `CaseArtifacts` is the single-source delivery of the already-parsed-and-validated
/// parquet/JSON rows, so downstream solver crates do not re-open the same files
/// from disk after [`load_case`] returns.
///
/// Fields are owned `Vec`s in deterministic (canonical) order. Empty vectors
/// indicate the optional file was absent on disk.
#[derive(Debug, Clone, Default)]
pub struct CaseArtifacts {
    /// File-presence manifest produced by Layer 1 (structural). Lets
    /// downstream code avoid re-running `validate_structure` to check
    /// optional-file presence.
    pub file_manifest: FileManifest,

    /// Rows from `system/hydro_geometry.parquet`.
    pub hydro_geometry: Vec<extensions::HydroGeometryRow>,

    /// Entries from `system/hydro_production_models.json`.
    pub production_models: Vec<extensions::ProductionModelConfig>,

    /// File-level FPHA plane-reduction block from
    /// `system/hydro_production_models.json`. `None` when the file is absent or
    /// carries no `fpha_plane_reduction` key. Carried for the post-fit
    /// plane-reduction pass; no behavior depends on it yet.
    pub plane_reduction: Option<extensions::PlaneReductionConfig>,

    /// Rows from `system/hydro_energy_productivity.parquet`.
    pub hydro_energy_productivity: Vec<extensions::HydroEnergyProductivityRow>,

    /// Rows from `system/fpha_hyperplanes.parquet`.
    pub fpha_hyperplanes: Vec<extensions::FphaHyperplaneRow>,

    /// Assembled scalar parameters from `system/scalar_parameters.json`.
    pub scalar_parameters: Vec<ScalarParameter>,

    /// Rows from `system/tailrace_curves.parquet`.
    pub tailrace_curves: Vec<extensions::TailraceCurveRow>,
}

/// Fully-loaded case bundle: the validated [`System`] plus the auxiliary
/// row sets that downstream consumers need without re-reading the case
/// directory.
#[derive(Debug)]
pub struct LoadedCase {
    /// Validated, ready-to-solve system.
    pub system: System,
    /// Auxiliary rows (parsed and validated by the load pipeline).
    pub artifacts: CaseArtifacts,
}

/// Load a case directory and return a fully-validated [`System`].
///
/// `path` must point to the root case directory containing `config.json` and the
/// standard subdirectories (`system/`, `scenarios/`, `constraints/`, `policy/`).
///
/// The function executes a six-layer validation pipeline; see the
/// [crate-level docs](crate) for the layer-by-layer breakdown.
///
/// After all layers pass, three-tier penalty/bound resolution and scenario assembly
/// are performed before constructing the [`System`].
///
/// Warnings collected during validation are silently discarded. Use [`validate_case`]
/// when you need to inspect or display warnings alongside the loaded [`System`].
///
/// Prefer [`load_case_with_artifacts`] when the downstream consumer also needs the
/// auxiliary parquet/JSON rows this pipeline already parsed: it returns them as a
/// [`CaseArtifacts`] bundle so downstream code can skip the duplicate disk reads.
///
/// # Errors
///
/// - [`LoadError::IoError`] — a required file is missing or cannot be read.
/// - [`LoadError::ParseError`] — a file contains malformed JSON or invalid Parquet.
/// - [`LoadError::SchemaError`] — a domain constraint violation detected
///   post-deserialization (e.g., AR coefficient count mismatch).
/// - [`LoadError::ConstraintError`] — one or more validation errors collected
///   across Layers 1-5, or `SystemBuilder` rejected the assembled data.
pub fn load_case(path: &Path) -> Result<System, LoadError> {
    pipeline::run_pipeline(path)
}

/// Load a case directory and return the validated [`System`] together with
/// the [`CaseArtifacts`] bundle of pre-parsed auxiliary rows.
///
/// This is the preferred entry point for solver pipelines that need the
/// production-model / hydro-geometry / FPHA hyperplane / scalar-parameter
/// rows: returning them here avoids the duplicate disk re-reads and parallel
/// validation paths in downstream crates.
///
/// The function runs the six-layer validation pipeline described in [`load_case`].
///
/// # Errors
///
/// Same error conditions as [`load_case`].
pub fn load_case_with_artifacts(path: &Path) -> Result<LoadedCase, LoadError> {
    pipeline::run_pipeline_with_artifacts(path).map(|(loaded, _report)| loaded)
}

/// Load a case directory and return both the fully-validated [`System`] and a
/// [`ValidationReport`] containing all warnings collected during the pipeline.
///
/// This function runs the same six-layer validation pipeline as [`load_case`] but
/// preserves warnings so that callers can display them to the user. Errors still
/// cause the function to return `Err`; warnings never block loading.
///
/// # Errors
///
/// Same error conditions as [`load_case`].
pub fn validate_case(path: &Path) -> Result<(System, ValidationReport), LoadError> {
    pipeline::run_pipeline_with_report(path)
}

/// Load a case directory and return the validated [`LoadedCase`] together with a
/// [`ValidationReport`] containing all warnings collected during the pipeline.
///
/// This is the preferred entry point for callers that need both the auxiliary
/// [`CaseArtifacts`] bundle (for downstream prep phases such as
/// `prepare_hydro_models_from_artifacts`) **and** the warning report.
///
/// The function runs the same six-layer validation pipeline as [`load_case`]. Errors
/// still cause the function to return `Err`; warnings never block loading.
///
/// # Errors
///
/// Same error conditions as [`load_case`].
pub fn validate_case_with_artifacts(
    path: &Path,
) -> Result<(LoadedCase, ValidationReport), LoadError> {
    pipeline::run_pipeline_with_artifacts(path)
}

#[cfg(test)]
mod tests {
    use crate::CaseArtifacts;

    #[test]
    fn case_artifacts_plane_reduction_defaults_to_none() {
        let artifacts = CaseArtifacts::default();
        assert!(artifacts.plane_reduction.is_none());
    }
}
