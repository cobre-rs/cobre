//! Layer 2 — Schema validation.
//!
//! Runs all individual file parsers and collects every parse/schema error across
//! the entire case directory before failing. This layer does **not** check
//! cross-references, dimensional consistency, or semantic constraints — those are
//! handled by later layers (tickets 030–033).
//!
//! The primary entry point is `validate_schema`, which returns a `ParsedData`
//! bundle holding every parsed output when all files parse cleanly, or `None`
//! when any parse or schema error is encountered.

use std::path::Path;

use cobre_core::{
    GenericConstraint,
    entities::{Bus, EnergyContract, Hydro, Line, NonControllableSource, PumpingStation, Thermal},
    initial_conditions::InitialConditions,
    penalty::GlobalPenaltyDefaults,
    scenario::{CorrelationModel, NcsModel},
};

use crate::{
    LoadError,
    config::{Config, parse_config},
    constraints::{
        BusPenaltyOverrideRow, ContractBoundsRow, ExchangeFactorEntry, GenericConstraintBoundsRow,
        HydroBoundsRow, HydroPenaltyOverrideRow, LineBoundsRow, LinePenaltyOverrideRow,
        NcsBoundsRow, NcsPenaltyOverrideRow, PumpingBoundsRow, ThermalBoundsRow,
        load_contract_bounds, load_exchange_factors, load_generic_constraint_bounds,
        load_generic_constraints, load_hydro_bounds, load_line_bounds, load_ncs_bounds,
        load_penalty_overrides_bus, load_penalty_overrides_hydro, load_penalty_overrides_line,
        load_penalty_overrides_ncs, load_pumping_bounds, load_thermal_bounds,
    },
    extensions::{
        FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig, load_fpha_hyperplanes,
        load_production_models, parse_hydro_geometry,
    },
    initial_conditions::parse_initial_conditions,
    penalties::parse_penalties,
    scenarios::{
        ExternalScenarioRow, InflowArCoefficientRow, InflowHistoryRow, InflowSeasonalStatsRow,
        LoadFactorEntry, LoadSeasonalStatsRow, NcsFactorEntry, load_correlation,
        load_external_scenarios, load_inflow_ar_coefficients, load_inflow_history,
        load_inflow_seasonal_stats, load_load_factors, load_load_seasonal_stats, load_ncs_models,
        load_non_controllable_factors,
    },
    stages::StagesData,
    system::{
        load_energy_contracts, load_non_controllable_sources, load_pumping_stations, parse_buses,
        parse_hydros, parse_lines, parse_thermals,
    },
    validation::{ErrorKind, ValidationContext, structural::FileManifest},
};

// ── ParsedData ────────────────────────────────────────────────────────────────

/// All parsed file outputs produced by Layer 2 schema validation.
///
/// Fields mirror the 33 input files in the canonical case directory layout.
/// Required files use direct types; optional files use `Vec<T>` (Parquet row
/// types) or `Option<T>` (structured JSON types). When an optional file is
/// absent from the manifest the corresponding field is empty or `None`.
///
/// This type is `pub(crate)` — it is only used within the validation pipeline
/// and is never exposed to downstream crates.
pub(crate) struct ParsedData {
    // ── Required root-level files ─────────────────────────────────────────────
    /// Parsed `config.json`.
    ///
    /// Populated for completeness and future use; not yet forwarded to `System`.
    #[allow(dead_code)]
    pub(crate) config: Config,
    /// Parsed `penalties.json`.
    ///
    /// Global defaults are already embedded in entity structs by the parsers;
    /// this field is retained for Layer 5 penalty-ordering checks.
    #[allow(dead_code)]
    pub(crate) penalties: GlobalPenaltyDefaults,
    /// Parsed `stages.json`.
    pub(crate) stages: StagesData,
    /// Parsed `initial_conditions.json`.
    pub(crate) initial_conditions: InitialConditions,

    // ── Required system/ files ────────────────────────────────────────────────
    /// Parsed `system/buses.json`.
    pub(crate) buses: Vec<Bus>,
    /// Parsed `system/thermals.json`.
    pub(crate) thermals: Vec<Thermal>,
    /// Parsed `system/hydros.json`.
    pub(crate) hydros: Vec<Hydro>,
    /// Parsed `system/lines.json`.
    pub(crate) lines: Vec<Line>,

    // ── Optional system/ files ────────────────────────────────────────────────
    /// Parsed `system/non_controllable_sources.json`. Empty when absent.
    pub(crate) non_controllable_sources: Vec<NonControllableSource>,
    /// Parsed `system/pumping_stations.json`. Empty when absent.
    pub(crate) pumping_stations: Vec<PumpingStation>,
    /// Parsed `system/energy_contracts.json`. Empty when absent.
    pub(crate) energy_contracts: Vec<EnergyContract>,
    /// Parsed `system/hydro_geometry.parquet`. Empty when absent.
    pub(crate) hydro_geometry: Vec<HydroGeometryRow>,
    /// Parsed `system/hydro_production_models.json`. Empty when absent.
    pub(crate) production_models: Vec<ProductionModelConfig>,
    /// Parsed `system/fpha_hyperplanes.parquet`. Empty when absent.
    pub(crate) fpha_hyperplanes: Vec<FphaHyperplaneRow>,

    // ── Optional scenarios/ files ─────────────────────────────────────────────
    /// Parsed `scenarios/inflow_history.parquet`. Empty when absent.
    pub(crate) inflow_history: Vec<InflowHistoryRow>,
    /// Parsed `scenarios/inflow_seasonal_stats.parquet`. Empty when absent.
    pub(crate) inflow_seasonal_stats: Vec<InflowSeasonalStatsRow>,
    /// Parsed `scenarios/inflow_ar_coefficients.parquet`. Empty when absent.
    pub(crate) inflow_ar_coefficients: Vec<InflowArCoefficientRow>,
    /// Parsed `scenarios/external_scenarios.parquet`. Empty when absent.
    pub(crate) external_scenarios: Vec<ExternalScenarioRow>,
    /// Parsed `scenarios/load_seasonal_stats.parquet`. Empty when absent.
    pub(crate) load_seasonal_stats: Vec<LoadSeasonalStatsRow>,
    /// Parsed `scenarios/load_factors.json`. Empty when absent.
    ///
    /// Parsed for schema validation; not yet forwarded to `System`.
    pub(crate) load_factors: Vec<LoadFactorEntry>,
    /// Parsed `scenarios/correlation.json`. `None` when absent.
    pub(crate) correlation: Option<CorrelationModel>,
    /// Parsed `scenarios/non_controllable_factors.json`. Empty when absent.
    pub(crate) non_controllable_factors: Vec<NcsFactorEntry>,
    /// Parsed `scenarios/non_controllable_models.parquet`. Empty when absent.
    pub(crate) ncs_models: Vec<NcsModel>,

    // ── Optional constraints/ files ───────────────────────────────────────────
    /// Parsed `constraints/thermal_bounds.parquet`. Empty when absent.
    pub(crate) thermal_bounds: Vec<ThermalBoundsRow>,
    /// Parsed `constraints/hydro_bounds.parquet`. Empty when absent.
    pub(crate) hydro_bounds: Vec<HydroBoundsRow>,
    /// Parsed `constraints/line_bounds.parquet`. Empty when absent.
    pub(crate) line_bounds: Vec<LineBoundsRow>,
    /// Parsed `constraints/pumping_bounds.parquet`. Empty when absent.
    pub(crate) pumping_bounds: Vec<PumpingBoundsRow>,
    /// Parsed `constraints/contract_bounds.parquet`. Empty when absent.
    pub(crate) contract_bounds: Vec<ContractBoundsRow>,
    /// Parsed `constraints/exchange_factors.json`. Empty when absent.
    ///
    /// Parsed for schema validation; not yet forwarded to `System`.
    #[allow(dead_code)]
    pub(crate) exchange_factors: Vec<ExchangeFactorEntry>,
    /// Parsed `constraints/generic_constraints.json`. Empty when absent.
    pub(crate) generic_constraints: Vec<GenericConstraint>,
    /// Parsed `constraints/generic_constraint_bounds.parquet`. Empty when absent.
    pub(crate) generic_constraint_bounds: Vec<GenericConstraintBoundsRow>,
    /// Parsed `constraints/penalty_overrides_bus.parquet`. Empty when absent.
    pub(crate) penalty_overrides_bus: Vec<BusPenaltyOverrideRow>,
    /// Parsed `constraints/penalty_overrides_line.parquet`. Empty when absent.
    pub(crate) penalty_overrides_line: Vec<LinePenaltyOverrideRow>,
    /// Parsed `constraints/penalty_overrides_hydro.parquet`. Empty when absent.
    pub(crate) penalty_overrides_hydro: Vec<HydroPenaltyOverrideRow>,
    /// Parsed `constraints/penalty_overrides_ncs.parquet`. Empty when absent.
    pub(crate) penalty_overrides_ncs: Vec<NcsPenaltyOverrideRow>,
    /// Parsed `constraints/ncs_bounds.parquet`. Empty when absent.
    pub(crate) ncs_bounds: Vec<NcsBoundsRow>,
}

// ── Error mapping helper ──────────────────────────────────────────────────────

/// Maps a [`LoadError`] from a parse call into a [`ValidationContext`] entry.
///
/// The `relative_path` argument is the path of the file relative to
/// `case_root` (e.g., `"system/hydros.json"`), used as the `file` field on
/// the [`ValidationEntry`].
///
/// - [`LoadError::IoError`] maps to [`ErrorKind::FileNotFound`].
/// - [`LoadError::ParseError`] maps to [`ErrorKind::ParseError`].
/// - [`LoadError::SchemaError`] maps to [`ErrorKind::SchemaViolation`].
/// - All other variants map to [`ErrorKind::SchemaViolation`] as a safe
///   fallback (they should not occur in Layer 2).
fn map_load_error(err: &LoadError, relative_path: &str, ctx: &mut ValidationContext) {
    match err {
        LoadError::IoError { .. } => {
            ctx.add_error(
                ErrorKind::FileNotFound,
                relative_path,
                None::<&str>,
                err.to_string(),
            );
        }
        LoadError::ParseError { .. } => {
            ctx.add_error(
                ErrorKind::ParseError,
                relative_path,
                None::<&str>,
                err.to_string(),
            );
        }
        LoadError::SchemaError { .. } => {
            ctx.add_error(
                ErrorKind::SchemaViolation,
                relative_path,
                None::<&str>,
                err.to_string(),
            );
        }
        _ => {
            // CrossReferenceError, ConstraintError, PolicyIncompatible should
            // not arise from individual file parsers in Layer 2, but map
            // conservatively to SchemaViolation.
            ctx.add_error(
                ErrorKind::SchemaViolation,
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
/// For each file marked present in `manifest`, calls the corresponding
/// `parse_*` or `load_*` function.  Parse and schema errors are mapped to
/// [`ErrorKind::ParseError`] and [`ErrorKind::SchemaViolation`] entries in
/// `ctx`.  I/O errors that occur during parsing are mapped to
/// [`ErrorKind::FileNotFound`].
///
/// All errors are collected across all files — the function never
/// short-circuits on the first failure.
///
/// Returns `Some(ParsedData)` only when no parse/schema errors were added to
/// `ctx` during this call.  Optional files that are absent from `manifest`
/// produce empty `Vec`s or `None` fields in [`ParsedData`] without adding any
/// error.
///
/// # Arguments
///
/// * `case_root` — path to the case directory root.
/// * `manifest`  — output from [`crate::validation::structural::validate_structure`].
/// * `ctx`        — mutable validation context that accumulates diagnostics.
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn validate_schema(
    case_root: &Path,
    manifest: &FileManifest,
    ctx: &mut ValidationContext,
) -> Option<ParsedData> {
    // Track whether any error is added during this call.
    let error_count_before = ctx.error_count();

    // ── Required root-level files ─────────────────────────────────────────────

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
        crate::stages::parse_stages(&case_root.join("stages.json")),
        "stages.json",
        ctx,
    );

    let initial_conditions = parse_or_error(
        parse_initial_conditions(&case_root.join("initial_conditions.json")),
        "initial_conditions.json",
        ctx,
    );

    // ── Required system/ files ────────────────────────────────────────────────
    //
    // parse_buses, parse_hydros, parse_lines, and parse_non_controllable_sources
    // require a `&GlobalPenaltyDefaults`. We use the parsed penalties when
    // available, or a sentinel when penalties failed to parse (so that remaining
    // files are still attempted and their individual errors collected).

    let sentinel = penalties.clone().unwrap_or_else(sentinel_penalties);

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

    // ── Optional system/ files ────────────────────────────────────────────────

    let non_controllable_sources = optional_or_error(
        manifest.system_non_controllable_sources_json,
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
        manifest.system_pumping_stations_json,
        || load_pumping_stations(Some(&case_root.join("system/pumping_stations.json"))),
        Vec::new,
        "system/pumping_stations.json",
        ctx,
    );

    let energy_contracts = optional_or_error(
        manifest.system_energy_contracts_json,
        || load_energy_contracts(Some(&case_root.join("system/energy_contracts.json"))),
        Vec::new,
        "system/energy_contracts.json",
        ctx,
    );

    let hydro_geometry = optional_or_error(
        manifest.system_hydro_geometry_parquet,
        || parse_hydro_geometry(&case_root.join("system/hydro_geometry.parquet")),
        Vec::new,
        "system/hydro_geometry.parquet",
        ctx,
    );

    let production_models = optional_or_error(
        manifest.system_hydro_production_models_json,
        || load_production_models(Some(&case_root.join("system/hydro_production_models.json"))),
        Vec::new,
        "system/hydro_production_models.json",
        ctx,
    );

    let fpha_hyperplanes = optional_or_error(
        manifest.system_fpha_hyperplanes_parquet,
        || load_fpha_hyperplanes(Some(&case_root.join("system/fpha_hyperplanes.parquet"))),
        Vec::new,
        "system/fpha_hyperplanes.parquet",
        ctx,
    );

    // ── Optional scenarios/ files ─────────────────────────────────────────────

    let inflow_history = optional_or_error(
        manifest.scenarios_inflow_history_parquet,
        || load_inflow_history(Some(&case_root.join("scenarios/inflow_history.parquet"))),
        Vec::new,
        "scenarios/inflow_history.parquet",
        ctx,
    );

    let inflow_seasonal_stats = optional_or_error(
        manifest.scenarios_inflow_seasonal_stats_parquet,
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
        manifest.scenarios_inflow_ar_coefficients_parquet,
        || {
            load_inflow_ar_coefficients(Some(
                &case_root.join("scenarios/inflow_ar_coefficients.parquet"),
            ))
        },
        Vec::new,
        "scenarios/inflow_ar_coefficients.parquet",
        ctx,
    );

    let external_scenarios = optional_or_error(
        manifest.scenarios_external_scenarios_parquet,
        || {
            load_external_scenarios(Some(
                &case_root.join("scenarios/external_scenarios.parquet"),
            ))
        },
        Vec::new,
        "scenarios/external_scenarios.parquet",
        ctx,
    );

    let load_seasonal_stats = optional_or_error(
        manifest.scenarios_load_seasonal_stats_parquet,
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
        manifest.scenarios_load_factors_json,
        || load_load_factors(Some(&case_root.join("scenarios/load_factors.json"))),
        Vec::new,
        "scenarios/load_factors.json",
        ctx,
    );

    // correlation.json uses `Option<CorrelationModel>` — `None` when the file
    // is absent so that callers can distinguish "no file" from "empty model".
    let correlation: Option<CorrelationModel> = if manifest.scenarios_correlation_json {
        match load_correlation(Some(&case_root.join("scenarios/correlation.json"))) {
            Ok(model) => Some(model),
            Err(ref err) => {
                map_load_error(err, "scenarios/correlation.json", ctx);
                None
            }
        }
    } else {
        None
    };

    let non_controllable_factors = optional_or_error(
        manifest.scenarios_non_controllable_factors_json,
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
        manifest.scenarios_non_controllable_models_parquet,
        || {
            load_ncs_models(Some(
                &case_root.join("scenarios/non_controllable_models.parquet"),
            ))
        },
        Vec::new,
        "scenarios/non_controllable_models.parquet",
        ctx,
    );

    // ── Optional constraints/ files ───────────────────────────────────────────

    let thermal_bounds = optional_or_error(
        manifest.constraints_thermal_bounds_parquet,
        || load_thermal_bounds(Some(&case_root.join("constraints/thermal_bounds.parquet"))),
        Vec::new,
        "constraints/thermal_bounds.parquet",
        ctx,
    );

    let hydro_bounds = optional_or_error(
        manifest.constraints_hydro_bounds_parquet,
        || load_hydro_bounds(Some(&case_root.join("constraints/hydro_bounds.parquet"))),
        Vec::new,
        "constraints/hydro_bounds.parquet",
        ctx,
    );

    let line_bounds = optional_or_error(
        manifest.constraints_line_bounds_parquet,
        || load_line_bounds(Some(&case_root.join("constraints/line_bounds.parquet"))),
        Vec::new,
        "constraints/line_bounds.parquet",
        ctx,
    );

    let pumping_bounds = optional_or_error(
        manifest.constraints_pumping_bounds_parquet,
        || load_pumping_bounds(Some(&case_root.join("constraints/pumping_bounds.parquet"))),
        Vec::new,
        "constraints/pumping_bounds.parquet",
        ctx,
    );

    let contract_bounds = optional_or_error(
        manifest.constraints_contract_bounds_parquet,
        || load_contract_bounds(Some(&case_root.join("constraints/contract_bounds.parquet"))),
        Vec::new,
        "constraints/contract_bounds.parquet",
        ctx,
    );

    let exchange_factors = optional_or_error(
        manifest.constraints_exchange_factors_json,
        || load_exchange_factors(Some(&case_root.join("constraints/exchange_factors.json"))),
        Vec::new,
        "constraints/exchange_factors.json",
        ctx,
    );

    let generic_constraints = optional_or_error(
        manifest.constraints_generic_constraints_json,
        || {
            load_generic_constraints(Some(
                &case_root.join("constraints/generic_constraints.json"),
            ))
        },
        Vec::new,
        "constraints/generic_constraints.json",
        ctx,
    );

    let generic_constraint_bounds = optional_or_error(
        manifest.constraints_generic_constraint_bounds_parquet,
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
        manifest.constraints_penalty_overrides_bus_parquet,
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
        manifest.constraints_penalty_overrides_line_parquet,
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
        manifest.constraints_penalty_overrides_hydro_parquet,
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
        manifest.constraints_penalty_overrides_ncs_parquet,
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
        manifest.constraints_ncs_bounds_parquet,
        || load_ncs_bounds(Some(&case_root.join("constraints/ncs_bounds.parquet"))),
        Vec::new,
        "constraints/ncs_bounds.parquet",
        ctx,
    );

    // ── Assemble result ───────────────────────────────────────────────────────

    // Only return Some(ParsedData) when no new errors were added during this call.
    if ctx.error_count() > error_count_before {
        return None;
    }

    // All required files must have parsed successfully.  The error count guard
    // above ensures this, but we destructure to satisfy the type checker.
    let (
        Some(config),
        Some(penalties),
        Some(stages),
        Some(initial_conditions),
        Some(buses),
        Some(lines),
        Some(hydros),
        Some(thermals),
    ) = (
        config,
        penalties,
        stages,
        initial_conditions,
        buses,
        lines,
        hydros,
        thermals,
    )
    else {
        return None;
    };

    Some(ParsedData {
        config,
        penalties,
        stages,
        initial_conditions,
        buses,
        thermals,
        hydros,
        lines,
        non_controllable_sources,
        pumping_stations,
        energy_contracts,
        hydro_geometry,
        production_models,
        fpha_hyperplanes,
        inflow_history,
        inflow_seasonal_stats,
        inflow_ar_coefficients,
        external_scenarios,
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
        exchange_factors,
        generic_constraints,
        generic_constraint_bounds,
        penalty_overrides_bus,
        penalty_overrides_line,
        penalty_overrides_hydro,
        penalty_overrides_ncs,
        ncs_bounds,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Call a parser, map any error into `ctx`, and return `Some(value)` on success
/// or `None` on failure.
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

/// Call a parser only when `present` is `true`, otherwise return the `default`
/// value.  Maps any error into `ctx` and returns the default on failure.
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
///
/// The sentinel allows penalty-dependent parsers (`parse_buses`, `parse_hydros`,
/// `parse_lines`, `parse_non_controllable_sources`) to be attempted so that
/// their own errors are collected in the same pass.
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
            fpha_turbined_cost: 1.0,
            diversion_cost: 1.0,
            storage_violation_below_cost: 1.0,
            filling_target_violation_cost: 1.0,
            turbined_violation_below_cost: 1.0,
            outflow_violation_below_cost: 1.0,
            outflow_violation_above_cost: 1.0,
            generation_violation_below_cost: 1.0,
            evaporation_violation_cost: 1.0,
            water_withdrawal_violation_cost: 1.0,
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
    use crate::validation::{ErrorKind, ValidationContext, structural::validate_structure};
    use std::fs;
    use tempfile::TempDir;

    // ── Minimal valid JSON fragments for each required file ───────────────────

    // config.json: `training.stopping_rules[].type` = "iteration_limit",
    // field name is "limit" (not "max_iterations").
    const VALID_CONFIG_JSON: &str = r#"{
        "training": {
            "forward_passes": 10,
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        }
    }"#;

    // penalties.json: top-level keys are "bus", "line", "hydro",
    // "non_controllable_source". Under "bus": "deficit_segments" (not
    // "segments") and "excess_cost". Under each segment: "cost" (not
    // "cost_per_mwh"). Under "line": "exchange_cost". Under "hydro": plain
    // field names without unit suffixes. Under "non_controllable_source":
    // "curtailment_cost". See src/penalties.rs for the raw serde types.
    const VALID_PENALTIES_JSON: &str = r#"{
        "bus": {
            "deficit_segments": [
                { "depth_mw": 500.0, "cost": 1000.0 },
                { "depth_mw": null,  "cost": 5000.0 }
            ],
            "excess_cost": 100.0
        },
        "line": { "exchange_cost": 2.0 },
        "hydro": {
            "spillage_cost": 0.01,
            "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1,
            "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
        },
        "non_controllable_source": { "curtailment_cost": 0.005 }
    }"#;

    // stages.json: `target_id` in transitions must be an integer (not null).
    // For a single-stage finite horizon we omit transitions entirely.
    // Only mandatory per-stage fields: id, start_date, end_date, blocks,
    // num_scenarios. season_id, block_mode, state_variables, risk_measure,
    // sampling_method all have serde defaults and are optional.
    const VALID_STAGES_JSON: &str = r#"{
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.06,
            "transitions": []
        },
        "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 },
        "stages": [
            {
                "id": 0,
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
                "num_scenarios": 50
            }
        ]
    }"#;

    const VALID_INITIAL_CONDITIONS_JSON: &str = r#"{
        "storage": [],
        "filling_storage": []
    }"#;

    // buses.json: mandatory fields are "id" and "name" only.
    // "base_kv" does not exist in the actual Bus raw type.
    const VALID_BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;

    const VALID_LINES_JSON: &str = r#"{ "lines": [] }"#;

    const VALID_HYDROS_JSON: &str = r#"{ "hydros": [] }"#;

    const VALID_THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

    /// Write a string to a path relative to `root`, creating all intermediate
    /// directories.
    fn write_file(root: &Path, relative: &str, content: &str) {
        let full = root.join(relative);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, content).unwrap();
    }

    /// Populate a `TempDir` with all 8 required files using valid JSON content.
    fn make_valid_case(dir: &TempDir) {
        let root = dir.path();
        write_file(root, "config.json", VALID_CONFIG_JSON);
        write_file(root, "penalties.json", VALID_PENALTIES_JSON);
        write_file(root, "stages.json", VALID_STAGES_JSON);
        write_file(
            root,
            "initial_conditions.json",
            VALID_INITIAL_CONDITIONS_JSON,
        );
        write_file(root, "system/buses.json", VALID_BUSES_JSON);
        write_file(root, "system/lines.json", VALID_LINES_JSON);
        write_file(root, "system/hydros.json", VALID_HYDROS_JSON);
        write_file(root, "system/thermals.json", VALID_THERMALS_JSON);
    }

    // ── AC 1: fully valid case returns Some(ParsedData), no errors ────────────

    /// Given a case directory with all 8 required files containing valid JSON,
    /// `validate_schema` returns `Some(ParsedData)` with all required fields
    /// populated and `ctx.has_errors()` is `false`.
    #[test]
    fn test_valid_case_returns_some_and_no_errors() {
        let dir = TempDir::new().unwrap();
        make_valid_case(&dir);

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
        // Required fields are populated.
        assert_eq!(
            data.buses.len(),
            1,
            "expected 1 bus parsed from valid buses.json"
        );
        // Optional fields are empty when absent.
        assert!(
            data.non_controllable_sources.is_empty(),
            "non_controllable_sources should be empty when file absent"
        );
        assert!(
            data.correlation.is_none(),
            "correlation should be None when file absent"
        );
    }

    // ── AC 2: one invalid required file produces ParseError, returns None ─────

    /// Given a case directory where `system/hydros.json` contains invalid JSON
    /// syntax, `validate_schema` returns `None` and `ctx` contains at least one
    /// `ParseError` entry with `file` containing `"system/hydros.json"`.
    #[test]
    fn test_invalid_json_returns_none_and_parse_error() {
        let dir = TempDir::new().unwrap();
        make_valid_case(&dir);
        // Overwrite hydros.json with malformed JSON.
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

    // ── AC 3: two invalid files produce two errors — both collected ───────────

    /// Given a case where `system/buses.json` has a schema violation (duplicate
    /// bus IDs) AND `penalties.json` has a parse error (invalid JSON),
    /// `validate_schema` returns `None` and `ctx` contains at least 2 error
    /// entries — both errors are collected, not just the first.
    #[test]
    fn test_two_invalid_files_both_errors_collected() {
        let dir = TempDir::new().unwrap();
        make_valid_case(&dir);

        // buses.json: duplicate bus IDs — produces a SchemaError
        write_file(
            dir.path(),
            "system/buses.json",
            r#"{ "buses": [{ "id": 1, "name": "A" }, { "id": 1, "name": "B" }] }"#,
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

        // Verify one error references penalties.json
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.file.to_string_lossy().contains("penalties.json")),
            "expected an error referencing penalties.json"
        );
        // Verify one error references buses.json
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.file.to_string_lossy().contains("system/buses.json")),
            "expected an error referencing system/buses.json"
        );
    }

    // ── AC 4: absent optional file yields None field, no error ────────────────

    /// Given a manifest where `scenarios_correlation_json` is `false`,
    /// `validate_schema` returns `Some(ParsedData)` with `correlation == None`
    /// and no error is added for the missing file.
    #[test]
    fn test_absent_optional_file_yields_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_valid_case(&dir);
        // Do NOT write scenarios/correlation.json.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(!ctx.has_errors(), "structural validation should pass");
        assert!(
            !manifest.scenarios_correlation_json,
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

    // ── AC 5: map_load_error maps each LoadError variant correctly ────────────

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
}
