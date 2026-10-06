//! Layer 2 — Schema validation.
//!
//! Runs all individual file parsers and collects every parse/schema error across
//! the entire case directory before failing. This layer does **not** check
//! cross-references, dimensional consistency, or semantic constraints — those are
//! handled by later layers.
//!
//! The primary entry point is `validate_schema`, which returns a `ParsedData`
//! bundle holding every parsed output when all files parse cleanly, or `None`
//! when any parse or schema error is encountered.

use std::path::Path;

use cobre_core::{
    EntityId, GenericConstraint, PostStudyStages, ScalarParameter,
    entities::{Bus, EnergyContract, Hydro, Line, NonControllableSource, PumpingStation, Thermal},
    initial_conditions::InitialConditions,
    penalty::GlobalPenaltyDefaults,
    scenario::{CorrelationModel, NcsModel},
};

use crate::{
    LoadError,
    config::{Config, parse_config},
    constraints::{
        BusPenaltyOverrideRow, ContractBoundsRow, GenericConstraintBoundsRow, HydroBoundsRow,
        HydroPenaltyOverrideRow, HydroUnitGroupBoundsRow, LineBoundsRow, LinePenaltyOverrideRow,
        NcsBoundsRow, NcsPenaltyOverrideRow, PumpingBoundsRow, ThermalBoundsRow,
        build_line_bus_pair_index, load_contract_bounds, load_generic_constraint_bounds,
        load_generic_constraints, load_hydro_bounds, load_hydro_unit_group_bounds,
        load_line_bounds, load_ncs_bounds, load_penalty_overrides_bus,
        load_penalty_overrides_hydro, load_penalty_overrides_line, load_penalty_overrides_ncs,
        load_pumping_bounds, load_thermal_bounds,
    },
    extensions::{
        FphaHyperplaneRow, HydroEnergyProductivityRow, HydroGeometryRow, PlaneReductionConfig,
        ProductionModelConfig, ProductionModelFile, load_fpha_hyperplanes,
        load_hydro_energy_productivity, load_production_models, parse_hydro_geometry,
        parse_scalar_parameters_json,
    },
    initial_conditions::parse_initial_conditions,
    penalties::parse_penalties,
    post_study_stages::parse_post_study_stages,
    scenarios::{
        ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, InflowAnnualComponentRow,
        InflowArCoefficientRow, InflowHistoryRow, InflowSeasonalStatsRow, LoadFactorEntry,
        LoadSeasonalStatsRow, NcsFactorEntry, load_correlation, load_external_inflow_scenarios,
        load_external_load_scenarios, load_external_ncs_scenarios, load_inflow_annual_component,
        load_inflow_ar_coefficients, load_inflow_history, load_inflow_seasonal_stats,
        load_load_factors, load_load_seasonal_stats, load_ncs_stats, load_non_controllable_factors,
    },
    stages::{StagesData, parse_stages},
    system::{
        load_energy_contracts, load_non_controllable_sources, load_pumping_stations, parse_buses,
        parse_hydros, parse_lines, parse_thermals,
    },
    validation::{
        ValidationContext, rules,
        structural::{FileManifest, InputFile},
    },
};

// ── ParsedData ────────────────────────────────────────────────────────────────

/// All parsed file outputs produced by Layer 2 schema validation.
///
/// One field per input file (the field doc names the file). Required files use
/// direct types; optional files use `Vec<T>` (Parquet rows) or `Option<T>`
/// (structured JSON) and are empty or `None` when the file is absent.
pub(crate) struct ParsedData {
    /// `config.json`.
    pub(crate) config: Config,
    /// `stages.json`.
    pub(crate) stages: StagesData,
    /// `initial_conditions.json`.
    pub(crate) initial_conditions: InitialConditions,
    /// `post_study_stages.json`. `None` when absent.
    pub(crate) post_study_stages: Option<PostStudyStages>,

    /// `system/buses.json`.
    pub(crate) buses: Vec<Bus>,
    /// `system/thermals.json`.
    pub(crate) thermals: Vec<Thermal>,
    /// `system/hydros.json`.
    pub(crate) hydros: Vec<Hydro>,
    /// `system/lines.json`.
    pub(crate) lines: Vec<Line>,

    /// `system/non_controllable_sources.json`.
    pub(crate) non_controllable_sources: Vec<NonControllableSource>,
    /// `system/pumping_stations.json`.
    pub(crate) pumping_stations: Vec<PumpingStation>,
    /// `system/energy_contracts.json`.
    pub(crate) energy_contracts: Vec<EnergyContract>,
    /// `system/hydro_geometry.parquet`.
    pub(crate) hydro_geometry: Vec<HydroGeometryRow>,
    /// `system/hydro_production_models.json`.
    pub(crate) production_models: Vec<ProductionModelConfig>,
    /// File-level FPHA plane-reduction block from
    /// `system/hydro_production_models.json`. `None` when absent or unset.
    pub(crate) plane_reduction: Option<PlaneReductionConfig>,
    /// `system/hydro_energy_productivity.parquet`.
    pub(crate) hydro_energy_productivity_rows: Vec<HydroEnergyProductivityRow>,
    /// `system/fpha_hyperplanes.parquet`.
    pub(crate) fpha_hyperplanes: Vec<FphaHyperplaneRow>,
    /// `constraints/generic_parameters.json`.
    pub(crate) scalar_parameters: Vec<ScalarParameter>,

    /// `scenarios/inflow_history.parquet`.
    pub(crate) inflow_history: Vec<InflowHistoryRow>,
    /// `scenarios/inflow_seasonal_stats.parquet`.
    pub(crate) inflow_seasonal_stats: Vec<InflowSeasonalStatsRow>,
    /// `scenarios/inflow_ar_coefficients.parquet`.
    pub(crate) inflow_ar_coefficients: Vec<InflowArCoefficientRow>,
    /// `scenarios/inflow_annual_component.parquet`.
    pub(crate) inflow_annual_components: Vec<InflowAnnualComponentRow>,
    /// `scenarios/external_inflow_scenarios.parquet`.
    pub(crate) external_scenarios: Vec<ExternalScenarioRow>,
    /// `scenarios/external_load_scenarios.parquet`.
    pub(crate) external_load_scenarios: Vec<ExternalLoadRow>,
    /// `scenarios/external_ncs_scenarios.parquet`.
    pub(crate) external_ncs_scenarios: Vec<ExternalNcsRow>,
    /// `scenarios/load_seasonal_stats.parquet`.
    pub(crate) load_seasonal_stats: Vec<LoadSeasonalStatsRow>,
    /// `scenarios/load_factors.json`.
    pub(crate) load_factors: Vec<LoadFactorEntry>,
    /// `scenarios/correlation.json`. `None` when absent.
    pub(crate) correlation: Option<CorrelationModel>,
    /// `scenarios/non_controllable_factors.json`.
    pub(crate) non_controllable_factors: Vec<NcsFactorEntry>,
    /// `scenarios/non_controllable_stats.parquet`.
    pub(crate) ncs_models: Vec<NcsModel>,

    /// `constraints/thermal_bounds.parquet`.
    pub(crate) thermal_bounds: Vec<ThermalBoundsRow>,
    /// `constraints/hydro_bounds.parquet`.
    pub(crate) hydro_bounds: Vec<HydroBoundsRow>,
    /// `constraints/line_bounds.parquet`.
    pub(crate) line_bounds: Vec<LineBoundsRow>,
    /// `constraints/pumping_bounds.parquet`.
    pub(crate) pumping_bounds: Vec<PumpingBoundsRow>,
    /// `constraints/contract_bounds.parquet`.
    pub(crate) contract_bounds: Vec<ContractBoundsRow>,
    /// `constraints/generic_constraints.json`.
    pub(crate) generic_constraints: Vec<GenericConstraint>,
    /// `constraints/generic_constraint_bounds.parquet`.
    pub(crate) generic_constraint_bounds: Vec<GenericConstraintBoundsRow>,
    /// `constraints/penalty_overrides_bus.parquet`.
    pub(crate) penalty_overrides_bus: Vec<BusPenaltyOverrideRow>,
    /// `constraints/penalty_overrides_line.parquet`.
    pub(crate) penalty_overrides_line: Vec<LinePenaltyOverrideRow>,
    /// `constraints/penalty_overrides_hydro.parquet`.
    pub(crate) penalty_overrides_hydro: Vec<HydroPenaltyOverrideRow>,
    /// `constraints/penalty_overrides_ncs.parquet`.
    pub(crate) penalty_overrides_ncs: Vec<NcsPenaltyOverrideRow>,
    /// `constraints/ncs_bounds.parquet`.
    pub(crate) ncs_bounds: Vec<NcsBoundsRow>,
    /// `constraints/hydro_unit_group_bounds.parquet`.
    pub(crate) hydro_unit_group_bounds: Vec<HydroUnitGroupBoundsRow>,
}

// ── Error mapping helper ──────────────────────────────────────────────────────

/// Maps a [`LoadError`] from a parse call into a [`ValidationContext`] entry.
///
/// The entry `message` carries only the path-free detail of each variant: the
/// `ValidationEntry.file` field already holds `relative_path`, and the report
/// renderer prefixes that path. Embedding the variant's full `Display` (which
/// repeats the file path) would surface the same path twice in the combined
/// message — relative as the prefix, absolute inside the `Display`.
fn map_load_error(err: &LoadError, relative_path: &str, ctx: &mut ValidationContext) {
    match err {
        LoadError::IoError { source, .. } => {
            ctx.emit(
                &rules::SCHEMA_FILE_UNREADABLE,
                relative_path,
                None::<&str>,
                source.to_string(),
            );
        }
        LoadError::ParseError { message, .. } => {
            ctx.emit(
                &rules::SCHEMA_FILE_UNPARSABLE,
                relative_path,
                None::<&str>,
                format!("parse error: {message}"),
            );
        }
        LoadError::SchemaError { field, message, .. } => {
            ctx.emit(
                &rules::SCHEMA_FILE_NONCONFORMING,
                relative_path,
                None::<&str>,
                format!("field {field}: {message}"),
            );
        }
        LoadError::ConstraintError { .. } => {
            // Layer-2 parsers should not produce this variant; map conservatively.
            // Unlike the arms above, it does not embed `relative_path`, so its full
            // Display is safe (no path duplication).
            ctx.emit(
                &rules::SCHEMA_FILE_NONCONFORMING,
                relative_path,
                None::<&str>,
                err.to_string(),
            );
        }
    }
}

// ── validate_schema ───────────────────────────────────────────────────────────

/// Performs Layer 2 schema validation on the case directory at `case_root`.
///
/// Errors are collected across all files — the function never short-circuits on
/// the first failure. Returns `Some(ParsedData)` only when no parse/schema errors
/// were added to `ctx` during this call; optional files absent from `manifest`
/// produce empty `Vec`s or `None` fields without adding any error.
// Rationale: every manifest file is attempted in one sequential pass so that all
// errors are collected before returning; splitting into helpers would thread `ctx`
// through many boundaries or short-circuit, violating the all-errors-collected contract.
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn validate_schema(
    case_root: &Path,
    manifest: &FileManifest,
    ctx: &mut ValidationContext,
) -> Option<ParsedData> {
    let error_count_before = ctx.error_count();

    let config = parse_or_error(
        parse_config(&case_root.join("config.json")),
        "config.json",
        ctx,
    );

    let penalties = parse_or_error(
        parse_penalties(&case_root.join("penalties.json")),
        "penalties.json",
        ctx,
    );

    let stages = parse_or_error(
        parse_stages(&case_root.join("stages.json")),
        "stages.json",
        ctx,
    );

    let initial_conditions = parse_or_error(
        parse_initial_conditions(&case_root.join("initial_conditions.json")),
        "initial_conditions.json",
        ctx,
    );

    // `Option` (not the aggregate): callers distinguish "no file" from an empty deck.
    let post_study_stages: Option<PostStudyStages> = optional_or_error(
        manifest.present(InputFile::PostStudyStagesJson),
        || parse_post_study_stages(&case_root.join("post_study_stages.json")).map(Some),
        || None,
        "post_study_stages.json",
        ctx,
    );

    // Fall back to a sentinel when penalties failed to parse, so penalty-dependent
    // parsers still run and their own errors are collected in this pass.
    let sentinel = penalties.unwrap_or_else(sentinel_penalties);

    let buses = parse_or_error(
        parse_buses(&case_root.join("system/buses.json"), &sentinel),
        "system/buses.json",
        ctx,
    );

    let lines = parse_or_error(
        parse_lines(&case_root.join("system/lines.json"), &sentinel),
        "system/lines.json",
        ctx,
    );

    let hydros = parse_or_error(
        parse_hydros(&case_root.join("system/hydros.json"), &sentinel),
        "system/hydros.json",
        ctx,
    );

    let thermals = parse_or_error(
        parse_thermals(&case_root.join("system/thermals.json")),
        "system/thermals.json",
        ctx,
    );

    let non_controllable_sources = optional_or_error(
        manifest.present(InputFile::SystemNonControllableSourcesJson),
        || {
            load_non_controllable_sources(
                Some(&case_root.join("system/non_controllable_sources.json")),
                &sentinel,
            )
        },
        Vec::new,
        "system/non_controllable_sources.json",
        ctx,
    );

    let pumping_stations = optional_or_error(
        manifest.present(InputFile::SystemPumpingStationsJson),
        || load_pumping_stations(Some(&case_root.join("system/pumping_stations.json"))),
        Vec::new,
        "system/pumping_stations.json",
        ctx,
    );

    let energy_contracts = optional_or_error(
        manifest.present(InputFile::SystemEnergyContractsJson),
        || load_energy_contracts(Some(&case_root.join("system/energy_contracts.json"))),
        Vec::new,
        "system/energy_contracts.json",
        ctx,
    );

    let hydro_geometry = optional_or_error(
        manifest.present(InputFile::SystemHydroGeometryParquet),
        || parse_hydro_geometry(&case_root.join("system/hydro_geometry.parquet")),
        Vec::new,
        "system/hydro_geometry.parquet",
        ctx,
    );

    let ProductionModelFile {
        configs: production_models,
        plane_reduction,
    } = optional_or_error(
        manifest.present(InputFile::SystemHydroProductionModelsJson),
        || load_production_models(Some(&case_root.join("system/hydro_production_models.json"))),
        ProductionModelFile::default,
        "system/hydro_production_models.json",
        ctx,
    );

    let hydro_energy_productivity_rows = optional_or_error(
        manifest.present(InputFile::SystemHydroEnergyProductivityParquet),
        || {
            load_hydro_energy_productivity(Some(
                &case_root.join("system/hydro_energy_productivity.parquet"),
            ))
        },
        Vec::new,
        "system/hydro_energy_productivity.parquet",
        ctx,
    );

    let fpha_hyperplanes = optional_or_error(
        manifest.present(InputFile::SystemFphaHyperplanesParquet),
        || load_fpha_hyperplanes(Some(&case_root.join("system/fpha_hyperplanes.parquet"))),
        Vec::new,
        "system/fpha_hyperplanes.parquet",
        ctx,
    );

    let inflow_history = optional_or_error(
        manifest.present(InputFile::ScenariosInflowHistoryParquet),
        || load_inflow_history(Some(&case_root.join("scenarios/inflow_history.parquet"))),
        Vec::new,
        "scenarios/inflow_history.parquet",
        ctx,
    );

    let inflow_seasonal_stats = optional_or_error(
        manifest.present(InputFile::ScenariosInflowSeasonalStatsParquet),
        || {
            load_inflow_seasonal_stats(Some(
                &case_root.join("scenarios/inflow_seasonal_stats.parquet"),
            ))
        },
        Vec::new,
        "scenarios/inflow_seasonal_stats.parquet",
        ctx,
    );

    let inflow_ar_coefficients = optional_or_error(
        manifest.present(InputFile::ScenariosInflowArCoefficientsParquet),
        || {
            load_inflow_ar_coefficients(Some(
                &case_root.join("scenarios/inflow_ar_coefficients.parquet"),
            ))
        },
        Vec::new,
        "scenarios/inflow_ar_coefficients.parquet",
        ctx,
    );

    let inflow_annual_components = optional_or_error(
        manifest.present(InputFile::ScenariosInflowAnnualComponentParquet),
        || {
            load_inflow_annual_component(Some(
                &case_root.join("scenarios/inflow_annual_component.parquet"),
            ))
        },
        Vec::new,
        "scenarios/inflow_annual_component.parquet",
        ctx,
    );

    let external_scenarios = optional_or_error(
        manifest.present(InputFile::ScenariosExternalInflowScenariosParquet),
        || {
            load_external_inflow_scenarios(Some(
                &case_root.join("scenarios/external_inflow_scenarios.parquet"),
            ))
        },
        Vec::new,
        "scenarios/external_inflow_scenarios.parquet",
        ctx,
    );

    let external_load_scenarios = optional_or_error(
        manifest.present(InputFile::ScenariosExternalLoadScenariosParquet),
        || {
            load_external_load_scenarios(Some(
                &case_root.join("scenarios/external_load_scenarios.parquet"),
            ))
        },
        Vec::new,
        "scenarios/external_load_scenarios.parquet",
        ctx,
    );

    let external_ncs_scenarios = optional_or_error(
        manifest.present(InputFile::ScenariosExternalNcsScenariosParquet),
        || {
            load_external_ncs_scenarios(Some(
                &case_root.join("scenarios/external_ncs_scenarios.parquet"),
            ))
        },
        Vec::new,
        "scenarios/external_ncs_scenarios.parquet",
        ctx,
    );

    let load_seasonal_stats = optional_or_error(
        manifest.present(InputFile::ScenariosLoadSeasonalStatsParquet),
        || {
            load_load_seasonal_stats(Some(
                &case_root.join("scenarios/load_seasonal_stats.parquet"),
            ))
        },
        Vec::new,
        "scenarios/load_seasonal_stats.parquet",
        ctx,
    );

    let load_factors = optional_or_error(
        manifest.present(InputFile::ScenariosLoadFactorsJson),
        || load_load_factors(Some(&case_root.join("scenarios/load_factors.json"))),
        Vec::new,
        "scenarios/load_factors.json",
        ctx,
    );

    // `Option` (not `Vec`) so callers distinguish "no file" from "empty model".
    let correlation: Option<CorrelationModel> = optional_or_error(
        manifest.present(InputFile::ScenariosCorrelationJson),
        || load_correlation(Some(&case_root.join("scenarios/correlation.json"))).map(Some),
        || None,
        "scenarios/correlation.json",
        ctx,
    );

    let non_controllable_factors = optional_or_error(
        manifest.present(InputFile::ScenariosNonControllableFactorsJson),
        || {
            load_non_controllable_factors(Some(
                &case_root.join("scenarios/non_controllable_factors.json"),
            ))
        },
        Vec::new,
        "scenarios/non_controllable_factors.json",
        ctx,
    );

    let ncs_models = optional_or_error(
        manifest.present(InputFile::ScenariosNonControllableStatsParquet),
        || {
            load_ncs_stats(Some(
                &case_root.join("scenarios/non_controllable_stats.parquet"),
            ))
        },
        Vec::new,
        "scenarios/non_controllable_stats.parquet",
        ctx,
    );

    let thermal_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsThermalBoundsParquet),
        || load_thermal_bounds(Some(&case_root.join("constraints/thermal_bounds.parquet"))),
        Vec::new,
        "constraints/thermal_bounds.parquet",
        ctx,
    );

    let hydro_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsHydroBoundsParquet),
        || load_hydro_bounds(Some(&case_root.join("constraints/hydro_bounds.parquet"))),
        Vec::new,
        "constraints/hydro_bounds.parquet",
        ctx,
    );

    let line_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsLineBoundsParquet),
        || load_line_bounds(Some(&case_root.join("constraints/line_bounds.parquet"))),
        Vec::new,
        "constraints/line_bounds.parquet",
        ctx,
    );

    let pumping_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsPumpingBoundsParquet),
        || load_pumping_bounds(Some(&case_root.join("constraints/pumping_bounds.parquet"))),
        Vec::new,
        "constraints/pumping_bounds.parquet",
        ctx,
    );

    let contract_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsContractBoundsParquet),
        || load_contract_bounds(Some(&case_root.join("constraints/contract_bounds.parquet"))),
        Vec::new,
        "constraints/contract_bounds.parquet",
        ctx,
    );

    // Must load BEFORE generic_constraints below: it resolves the `@name` sigils
    // in constraint expressions. Reordering breaks that resolution.
    let scalar_parameters: Vec<ScalarParameter> = optional_or_error(
        manifest.present(InputFile::ConstraintsGenericParametersJson),
        || parse_scalar_parameters_json(&case_root.join("constraints/generic_parameters.json")),
        Vec::new,
        "constraints/generic_parameters.json",
        ctx,
    );

    // An empty map (file absent or failed to parse) is fine: any `@name` reference
    // then produces a descriptive schema error downstream.
    let scalar_name_to_id: std::collections::HashMap<String, EntityId> = scalar_parameters
        .iter()
        .map(|p| (p.name.clone(), p.id))
        .collect();

    let generic_constraints = optional_or_error(
        manifest.present(InputFile::ConstraintsGenericConstraintsJson),
        || {
            let line_pair_index = build_line_bus_pair_index(lines.as_deref().unwrap_or(&[]))?;
            load_generic_constraints(
                Some(&case_root.join("constraints/generic_constraints.json")),
                &scalar_name_to_id,
                &line_pair_index,
            )
        },
        Vec::new,
        "constraints/generic_constraints.json",
        ctx,
    );

    let generic_constraint_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsGenericConstraintBoundsParquet),
        || {
            load_generic_constraint_bounds(Some(
                &case_root.join("constraints/generic_constraint_bounds.parquet"),
            ))
        },
        Vec::new,
        "constraints/generic_constraint_bounds.parquet",
        ctx,
    );

    let penalty_overrides_bus = optional_or_error(
        manifest.present(InputFile::ConstraintsPenaltyOverridesBusParquet),
        || {
            load_penalty_overrides_bus(Some(
                &case_root.join("constraints/penalty_overrides_bus.parquet"),
            ))
        },
        Vec::new,
        "constraints/penalty_overrides_bus.parquet",
        ctx,
    );

    let penalty_overrides_line = optional_or_error(
        manifest.present(InputFile::ConstraintsPenaltyOverridesLineParquet),
        || {
            load_penalty_overrides_line(Some(
                &case_root.join("constraints/penalty_overrides_line.parquet"),
            ))
        },
        Vec::new,
        "constraints/penalty_overrides_line.parquet",
        ctx,
    );

    let penalty_overrides_hydro = optional_or_error(
        manifest.present(InputFile::ConstraintsPenaltyOverridesHydroParquet),
        || {
            load_penalty_overrides_hydro(Some(
                &case_root.join("constraints/penalty_overrides_hydro.parquet"),
            ))
        },
        Vec::new,
        "constraints/penalty_overrides_hydro.parquet",
        ctx,
    );

    let penalty_overrides_ncs = optional_or_error(
        manifest.present(InputFile::ConstraintsPenaltyOverridesNcsParquet),
        || {
            load_penalty_overrides_ncs(Some(
                &case_root.join("constraints/penalty_overrides_ncs.parquet"),
            ))
        },
        Vec::new,
        "constraints/penalty_overrides_ncs.parquet",
        ctx,
    );

    let ncs_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsNcsBoundsParquet),
        || load_ncs_bounds(Some(&case_root.join("constraints/ncs_bounds.parquet"))),
        Vec::new,
        "constraints/ncs_bounds.parquet",
        ctx,
    );

    let hydro_unit_group_bounds = optional_or_error(
        manifest.present(InputFile::ConstraintsHydroUnitGroupBoundsParquet),
        || {
            load_hydro_unit_group_bounds(Some(
                &case_root.join("constraints/hydro_unit_group_bounds.parquet"),
            ))
        },
        Vec::new,
        "constraints/hydro_unit_group_bounds.parquet",
        ctx,
    );

    if ctx.error_count() > error_count_before {
        return None;
    }

    // The guard above already ensures every required file parsed; these `?`s only
    // narrow the Options to satisfy the type checker.
    let config = config?;
    let stages = stages?;
    let initial_conditions = initial_conditions?;
    let buses = buses?;
    let lines = lines?;
    let hydros = hydros?;
    let thermals = thermals?;

    Some(ParsedData {
        config,
        stages,
        initial_conditions,
        post_study_stages,
        buses,
        thermals,
        hydros,
        lines,
        non_controllable_sources,
        pumping_stations,
        energy_contracts,
        hydro_geometry,
        production_models,
        plane_reduction,
        hydro_energy_productivity_rows,
        fpha_hyperplanes,
        scalar_parameters,
        inflow_history,
        inflow_seasonal_stats,
        inflow_ar_coefficients,
        inflow_annual_components,
        external_scenarios,
        external_load_scenarios,
        external_ncs_scenarios,
        load_seasonal_stats,
        load_factors,
        correlation,
        non_controllable_factors,
        ncs_models,
        thermal_bounds,
        hydro_bounds,
        line_bounds,
        pumping_bounds,
        contract_bounds,
        generic_constraints,
        generic_constraint_bounds,
        penalty_overrides_bus,
        penalty_overrides_line,
        penalty_overrides_hydro,
        penalty_overrides_ncs,
        ncs_bounds,
        hydro_unit_group_bounds,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn parse_or_error<T>(
    result: Result<T, LoadError>,
    relative_path: &str,
    ctx: &mut ValidationContext,
) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(ref err) => {
            map_load_error(err, relative_path, ctx);
            None
        }
    }
}

/// Like `parse_or_error`, but skips the parser (returning `default_fn()`) when
/// `present` is `false`; also returns the default on parse failure.
fn optional_or_error<T, F, D>(
    present: bool,
    parse_fn: F,
    default_fn: D,
    relative_path: &str,
    ctx: &mut ValidationContext,
) -> T
where
    F: FnOnce() -> Result<T, LoadError>,
    D: FnOnce() -> T,
{
    if present {
        match parse_fn() {
            Ok(value) => value,
            Err(ref err) => {
                map_load_error(err, relative_path, ctx);
                default_fn()
            }
        }
    } else {
        default_fn()
    }
}

/// Build a minimal valid [`GlobalPenaltyDefaults`] sentinel used when
/// `penalties.json` failed to parse.
fn sentinel_penalties() -> GlobalPenaltyDefaults {
    use cobre_core::entities::{DeficitSegment, HydroPenalties};
    GlobalPenaltyDefaults {
        bus_deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1.0,
        }],
        bus_excess_cost: 1.0,
        line_exchange_cost: 1.0,
        hydro: HydroPenalties {
            spillage_cost: 1.0,
            turbined_cost: 1.0,
            diversion_cost: 1.0,
            storage_violation_below_cost: 1.0,
            filling_target_violation_cost: 1.0,
            turbined_violation_below_cost: 1.0,
            outflow_violation_below_cost: 1.0,
            outflow_violation_above_cost: 1.0,
            generation_violation_below_cost: 1.0,
            evaporation_violation_cost: 1.0,
            water_withdrawal_violation_cost: 1.0,
            water_withdrawal_violation_pos_cost: 1.0,
            water_withdrawal_violation_neg_cost: 1.0,
            evaporation_violation_pos_cost: 1.0,
            evaporation_violation_neg_cost: 1.0,
            inflow_nonnegativity_cost: 1000.0,
        },
        ncs_curtailment_cost: 1.0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::expect_used
)]
mod tests {
    use super::*;
    use crate::test_support::{make_minimal_case, write_file};
    use crate::validation::{ErrorKind, ValidationContext, structural::validate_structure};
    use std::fs;
    use tempfile::TempDir;

    /// Given a case directory with all 8 required files containing valid JSON,
    /// `validate_schema` returns `Some(ParsedData)` with all required fields
    /// populated and `ctx.has_errors()` is `false`.
    #[test]
    fn test_valid_case_returns_some_and_no_errors() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");

        let data = validate_schema(dir.path(), &manifest, &mut ctx);

        assert!(
            data.is_some(),
            "validate_schema should return Some(ParsedData) for valid case"
        );
        assert!(
            !ctx.has_errors(),
            "ctx should have no errors for a valid case, got: {:?}",
            ctx.errors()
        );

        let data = data.unwrap();
        assert_eq!(
            data.buses.len(),
            1,
            "expected 1 bus parsed from valid buses.json"
        );
        assert!(
            data.non_controllable_sources.is_empty(),
            "non_controllable_sources should be empty when file absent"
        );
        assert!(
            data.correlation.is_none(),
            "correlation should be None when file absent"
        );
    }

    /// Given a case directory where `system/hydros.json` contains invalid JSON
    /// syntax, `validate_schema` returns `None` and `ctx` contains at least one
    /// `ParseError` entry with `file` containing `"system/hydros.json"`.
    #[test]
    fn test_invalid_json_returns_none_and_parse_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        write_file(dir.path(), "system/hydros.json", "{ invalid json !!!");

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");

        let data = validate_schema(dir.path(), &manifest, &mut ctx);

        assert!(
            data.is_none(),
            "validate_schema should return None when a file has invalid JSON"
        );
        assert!(ctx.has_errors(), "ctx should have at least one error");

        let parse_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::ParseError)
            .collect();
        assert!(
            !parse_errors.is_empty(),
            "ctx should have at least one ParseError entry"
        );
        assert!(
            parse_errors
                .iter()
                .any(|e| e.file.to_string_lossy().contains("system/hydros.json")),
            "ParseError entry should reference system/hydros.json, got: {:?}",
            parse_errors
                .iter()
                .map(|e| e.file.display().to_string())
                .collect::<Vec<_>>()
        );
    }

    /// Given a case where `system/buses.json` has a schema violation (duplicate
    /// bus IDs) AND `penalties.json` has a parse error (invalid JSON),
    /// `validate_schema` returns `None` and `ctx` contains at least 2 error
    /// entries — both errors are collected, not just the first.
    #[test]
    fn test_two_invalid_files_both_errors_collected() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);

        // buses.json: duplicate bus IDs — produces a SchemaError
        write_file(
            dir.path(),
            "system/buses.json",
            r#"{ "buses": [{ "id": 1, "name": "A", "operational_start_date": "2024-01-01" }, { "id": 1, "name": "B", "operational_start_date": "2024-01-01" }] }"#,
        );

        // penalties.json: invalid JSON syntax — produces a parse error
        write_file(dir.path(), "penalties.json", "{ not valid json");

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");

        let data = validate_schema(dir.path(), &manifest, &mut ctx);

        assert!(data.is_none(), "validate_schema should return None");
        assert!(
            ctx.errors().len() >= 2,
            "ctx should have at least 2 errors (one per invalid file), got {} errors: {:?}",
            ctx.errors().len(),
            ctx.errors()
                .iter()
                .map(|e| format!("{:?} @ {}", e.kind, e.file.display()))
                .collect::<Vec<_>>()
        );

        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.file.to_string_lossy().contains("penalties.json")),
            "expected an error referencing penalties.json"
        );
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.file.to_string_lossy().contains("system/buses.json")),
            "expected an error referencing system/buses.json"
        );
    }

    /// Given a manifest where `scenarios_correlation_json` is `false`,
    /// `validate_schema` returns `Some(ParsedData)` with `correlation == None`
    /// and no error is added for the missing file.
    #[test]
    fn test_absent_optional_file_yields_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        // Do NOT write scenarios/correlation.json.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");
        assert!(
            !manifest.present(InputFile::ScenariosCorrelationJson),
            "manifest should show correlation.json absent"
        );

        let data = validate_schema(dir.path(), &manifest, &mut ctx);

        assert!(
            data.is_some(),
            "validate_schema should return Some(ParsedData)"
        );
        assert!(
            !ctx.has_errors(),
            "no error should be added for an absent optional file"
        );
        assert!(
            data.unwrap().correlation.is_none(),
            "correlation should be None when file absent"
        );
    }

    /// `map_load_error` maps `IoError` to `ErrorKind::FileNotFound`.
    #[test]
    fn test_map_load_error_io_error() {
        let mut ctx = ValidationContext::new();
        let err = LoadError::io(
            "system/hydros.json",
            std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        );
        map_load_error(&err, "system/hydros.json", &mut ctx);
        assert_eq!(ctx.errors().len(), 1);
        assert_eq!(ctx.errors()[0].kind, ErrorKind::FileNotFound);
        assert!(
            ctx.errors()[0]
                .file
                .to_string_lossy()
                .contains("system/hydros.json")
        );
    }

    /// `map_load_error` maps `ParseError` to `ErrorKind::ParseError`.
    #[test]
    fn test_map_load_error_parse_error() {
        let mut ctx = ValidationContext::new();
        let err = LoadError::parse("stages.json", "unexpected token");
        map_load_error(&err, "stages.json", &mut ctx);
        assert_eq!(ctx.errors().len(), 1);
        assert_eq!(ctx.errors()[0].kind, ErrorKind::ParseError);
        assert!(
            ctx.errors()[0]
                .file
                .to_string_lossy()
                .contains("stages.json")
        );
    }

    /// `map_load_error` maps `SchemaError` to `ErrorKind::SchemaViolation`.
    #[test]
    fn test_map_load_error_schema_error() {
        let mut ctx = ValidationContext::new();
        let err = LoadError::SchemaError {
            path: std::path::PathBuf::from("system/buses.json"),
            field: "id".to_string(),
            message: "duplicate id".to_string(),
        };
        map_load_error(&err, "system/buses.json", &mut ctx);
        assert_eq!(ctx.errors().len(), 1);
        assert_eq!(ctx.errors()[0].kind, ErrorKind::SchemaViolation);
        assert!(
            ctx.errors()[0]
                .file
                .to_string_lossy()
                .contains("system/buses.json")
        );
    }

    // ── hydro_energy_productivity.parquet in Layer 2 ──────────────────────────

    fn write_parquet_batch(dest: &Path, batch: &arrow::record_batch::RecordBatch) {
        use parquet::arrow::ArrowWriter;

        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(dest)
            .expect("create parquet file");
        let mut writer =
            ArrowWriter::try_new(file, batch.schema(), None).expect("ArrowWriter::try_new");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
    }

    /// Write a minimal valid `system/hydro_energy_productivity.parquet` with the
    /// given rows to a path inside `root`.
    fn write_hydro_energy_productivity_parquet(
        root: &Path,
        rows: &[(i32, Option<i32>, Option<f64>)],
    ) {
        use std::sync::Arc;

        use arrow::array::{Float64Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_volume_hm3", DataType::Float64, true),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]));

        let hydro_ids: Vec<i32> = rows.iter().map(|(h, _, _)| *h).collect();
        let stage_ids: Vec<Option<i32>> = rows.iter().map(|(_, s, _)| *s).collect();
        let rho_eqs: Vec<Option<f64>> = rows.iter().map(|(_, _, r)| *r).collect();
        let nulls: Vec<Option<f64>> = rows.iter().map(|_| None).collect();

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(hydro_ids)),
                Arc::new(Int32Array::from(stage_ids)),
                Arc::new(Float64Array::from(rho_eqs)),
                Arc::new(Float64Array::from(nulls.clone())),
                Arc::new(Float64Array::from(nulls.clone())),
                Arc::new(Float64Array::from(nulls)),
            ],
        )
        .expect("valid batch");

        let dest = root.join("system/hydro_energy_productivity.parquet");
        write_parquet_batch(&dest, &batch);
    }

    /// Write a `system/hydro_energy_productivity.parquet` with a NULL `hydro_id`
    /// column to trigger a `SchemaError` during parsing.
    fn write_hydro_energy_productivity_null_hydro_id(root: &Path) {
        use std::sync::Arc;

        use arrow::array::{Float64Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;

        // Use a nullable hydro_id column so we can insert a NULL value.
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, true),
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_volume_hm3", DataType::Float64, true),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
            ],
        )
        .expect("valid batch");

        let dest = root.join("system/hydro_energy_productivity.parquet");
        write_parquet_batch(&dest, &batch);
    }

    /// Given a case directory with a valid 2-row `system/hydro_energy_productivity.parquet`,
    /// `validate_schema` returns `Some(ParsedData)` whose
    /// `hydro_energy_productivity_rows` has length 2 and both rows carry the
    /// parsed values.
    #[test]
    fn test_validate_schema_loads_hydro_energy_productivity_when_present() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);

        // Write a valid 2-row parquet: hydro_id=0 at stage_id=0 and stage_id=1.
        write_hydro_energy_productivity_parquet(
            dir.path(),
            &[(0, Some(0), Some(3.6)), (0, Some(1), Some(4.0))],
        );

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");
        assert!(
            manifest.present(InputFile::SystemHydroEnergyProductivityParquet),
            "manifest should detect the parquet"
        );

        let data = validate_schema(dir.path(), &manifest, &mut ctx);

        assert!(
            data.is_some(),
            "validate_schema should return Some(ParsedData) for a valid case"
        );
        assert!(
            !ctx.has_errors(),
            "ctx should have no errors, got: {:?}",
            ctx.errors()
        );

        let rows = &data.unwrap().hydro_energy_productivity_rows;
        assert_eq!(rows.len(), 2, "expected 2 rows parsed from the parquet");
        assert!(
            rows.iter().all(|r| r.hydro_id.0 == 0),
            "both rows should have hydro_id=0"
        );
    }

    /// Given a case directory without `system/hydro_energy_productivity.parquet`,
    /// `validate_schema` returns `Some(ParsedData)` with
    /// `hydro_energy_productivity_rows` equal to `Vec::new()` and no error is
    /// added for the absent file.
    #[test]
    fn test_validate_schema_hydro_energy_productivity_absent_is_empty() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        // Do NOT write system/hydro_energy_productivity.parquet.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");
        assert!(
            !manifest.present(InputFile::SystemHydroEnergyProductivityParquet),
            "manifest should show parquet absent"
        );

        let data = validate_schema(dir.path(), &manifest, &mut ctx);

        assert!(
            data.is_some(),
            "validate_schema should return Some(ParsedData)"
        );
        assert!(
            !ctx.has_errors(),
            "no error should be added for an absent optional file"
        );
        assert!(
            data.unwrap().hydro_energy_productivity_rows.is_empty(),
            "hydro_energy_productivity_rows should be empty when the file is absent"
        );
    }

    /// Given a case directory whose `system/hydro_energy_productivity.parquet`
    /// contains a row with a NULL `hydro_id`, `validate_schema` returns `None`
    /// and the `ValidationContext` contains a `SchemaViolation` error whose
    /// file path references the parquet.
    #[test]
    fn test_validate_schema_malformed_hydro_energy_productivity_collects_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        write_hydro_energy_productivity_null_hydro_id(dir.path());

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");
        assert!(
            manifest.present(InputFile::SystemHydroEnergyProductivityParquet),
            "manifest should detect the parquet"
        );

        let data = validate_schema(dir.path(), &manifest, &mut ctx);

        assert!(
            data.is_none(),
            "validate_schema should return None when the parquet has a schema violation"
        );
        assert!(ctx.has_errors(), "ctx should have at least one error");

        let schema_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::SchemaViolation)
            .collect();
        assert!(
            !schema_errors.is_empty(),
            "ctx should have at least one SchemaViolation entry"
        );
        assert!(
            schema_errors.iter().any(|e| e
                .file
                .to_string_lossy()
                .contains("hydro_energy_productivity")),
            "SchemaViolation should reference the parquet path, got: {:?}",
            schema_errors
                .iter()
                .map(|e| e.file.display().to_string())
                .collect::<Vec<_>>()
        );
    }
}
