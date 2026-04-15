//! Pipeline orchestrator for the five-layer validation and `System` construction pipeline.
//!
//! [`run_pipeline`] and [`run_pipeline_with_report`] wire together all five validation
//! layers, the resolution step, the scenario assembly step, and `SystemBuilder::build`
//! into single callables that either return a fully-validated [`cobre_core::System`] or a
//! [`LoadError`] explaining what went wrong.
//!
//! Use [`run_pipeline_with_report`] when the caller needs access to warnings collected
//! during validation (e.g., the `validate` CLI subcommand).

use std::collections::HashMap;

use cobre_core::{scenario::CorrelationModel, SystemBuilder};

use crate::{
    report::{generate_report, ValidationReport},
    resolution::{
        resolve_bounds, resolve_exchange_factors, resolve_generic_constraint_bounds,
        resolve_load_factors, resolve_ncs_bounds, resolve_ncs_factors, resolve_penalties,
        BoundsEntitySlices, BoundsOverrides, PenaltiesEntitySlices, PenaltiesOverrides,
    },
    scenarios::assembly::{assemble_inflow_models, assemble_load_models},
    validation::{
        dimensional::validate_dimensional_consistency,
        referential::validate_referential_integrity,
        schema::validate_schema,
        semantic::{validate_semantic_hydro_thermal, validate_semantic_stages_penalties_scenarios},
        structural::validate_structure,
        ValidationContext,
    },
    LoadError,
};

use cobre_core::System;
use std::path::Path;

/// Run the complete loading pipeline for a case directory.
///
/// Executes all five validation layers, the three-tier resolution step, scenario
/// assembly, and `SystemBuilder::build`. Returns `Ok(System)` when every layer
/// succeeds, or the first `Err(LoadError)` encountered. Warnings collected during
/// validation are silently discarded; use [`run_pipeline_with_report`] to retrieve them.
///
/// # Errors
///
/// - [`LoadError::IoError`] / [`LoadError::ParseError`] — file read or JSON/Parquet
///   parse failure in Layer 2.
/// - [`LoadError::ConstraintError`] — one or more validation errors collected by
///   Layers 1-5, or `SystemBuilder::build` rejection.
/// - [`LoadError::SchemaError`] — AR coefficient count mismatch in scenario assembly.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_pipeline(path: &Path) -> Result<System, LoadError> {
    run_pipeline_with_report(path).map(|(system, _report)| system)
}

/// Run the complete loading pipeline and return both the [`System`] and a
/// [`ValidationReport`] containing any warnings collected during validation.
///
/// This is the same pipeline as [`run_pipeline`] but preserves warnings so that
/// callers (e.g., the `validate` CLI subcommand) can display them to the user.
///
/// # Errors
///
/// Same error conditions as [`run_pipeline`].
#[allow(clippy::too_many_lines)]
pub(crate) fn run_pipeline_with_report(
    path: &Path,
) -> Result<(System, ValidationReport), LoadError> {
    let mut ctx = ValidationContext::new();

    // Layer 1 — structural validation (required files present on disk).
    let manifest = validate_structure(path, &mut ctx);

    // Layer 2 — schema validation (parse all files, collect parse/schema errors).
    let data = validate_schema(path, &manifest, &mut ctx);

    let Some(data) = data else {
        // validate_schema returns None when it collected parse/schema errors.
        // into_result() should always return Err here, but if it doesn't we
        // return a descriptive error rather than panicking.
        return ctx.into_result().and(Err(LoadError::ConstraintError {
            description: "schema validation failed but no errors were collected".to_string(),
        }));
    };

    // Layers 3-5 — referential integrity, dimensional consistency, semantic rules.
    validate_referential_integrity(&data, &mut ctx);
    validate_dimensional_consistency(&data, &mut ctx);
    validate_semantic_hydro_thermal(&data, &mut ctx);
    validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

    // Capture warnings before consuming the context. Errors cause early return.
    let report = generate_report(&ctx);
    ctx.into_result()?;

    // ── Resolution step ───────────────────────────────────────────────────────

    // Count study stages only (pre-study stages have negative IDs) and build
    // a mapping from domain-level stage_id to positional 0-based index.
    let study_stages: Vec<_> = data.stages.stages.iter().filter(|s| s.id >= 0).collect();
    let n_stages = study_stages.len();
    let stage_index: HashMap<i32, usize> = study_stages
        .iter()
        .enumerate()
        .map(|(idx, s)| (s.id, idx))
        .collect();

    let penalties = resolve_penalties(
        &PenaltiesEntitySlices {
            hydros: &data.hydros,
            buses: &data.buses,
            lines: &data.lines,
            ncs_sources: &data.non_controllable_sources,
        },
        n_stages,
        &stage_index,
        &PenaltiesOverrides {
            hydro: &data.penalty_overrides_hydro,
            bus: &data.penalty_overrides_bus,
            line: &data.penalty_overrides_line,
            ncs: &data.penalty_overrides_ncs,
        },
    );

    let bounds = resolve_bounds(
        &BoundsEntitySlices {
            hydros: &data.hydros,
            thermals: &data.thermals,
            lines: &data.lines,
            pumping_stations: &data.pumping_stations,
            contracts: &data.energy_contracts,
        },
        n_stages,
        &stage_index,
        &BoundsOverrides {
            hydro: &data.hydro_bounds,
            thermal: &data.thermal_bounds,
            line: &data.line_bounds,
            pumping: &data.pumping_bounds,
            contract: &data.contract_bounds,
        },
    );

    let resolved_generic_bounds = resolve_generic_constraint_bounds(
        &data.generic_constraints,
        &data.generic_constraint_bounds,
    );

    let resolved_load_factors =
        resolve_load_factors(&data.load_factors, &data.buses, &data.stages.stages);
    let resolved_exchange_factors =
        resolve_exchange_factors(&data.exchange_factors, &data.lines, &data.stages.stages);

    let resolved_ncs_bounds = resolve_ncs_bounds(
        &data.ncs_bounds,
        &data.non_controllable_sources,
        n_stages,
        &stage_index,
    );

    let resolved_ncs_factors = resolve_ncs_factors(
        &data.non_controllable_factors,
        &data.non_controllable_sources,
        &data.stages.stages,
    );

    // ── Scenario assembly ─────────────────────────────────────────────────────

    let inflow_models =
        assemble_inflow_models(data.inflow_seasonal_stats, data.inflow_ar_coefficients)?;
    let load_models = assemble_load_models(data.load_seasonal_stats);

    // ── System construction ───────────────────────────────────────────────────

    let system = SystemBuilder::new()
        .buses(data.buses)
        .lines(data.lines)
        .hydros(data.hydros)
        .thermals(data.thermals)
        .pumping_stations(data.pumping_stations)
        .contracts(data.energy_contracts)
        .non_controllable_sources(data.non_controllable_sources)
        .stages(data.stages.stages)
        .policy_graph(data.stages.policy_graph)
        .penalties(penalties)
        .bounds(bounds)
        .resolved_generic_bounds(resolved_generic_bounds)
        .resolved_load_factors(resolved_load_factors)
        .resolved_exchange_factors(resolved_exchange_factors)
        .resolved_ncs_bounds(resolved_ncs_bounds)
        .resolved_ncs_factors(resolved_ncs_factors)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(data.ncs_models)
        .correlation(data.correlation.unwrap_or_else(CorrelationModel::default))
        .initial_conditions(data.initial_conditions)
        .generic_constraints(data.generic_constraints)
        .inflow_history(data.inflow_history)
        .external_scenarios(data.external_scenarios)
        .external_load_scenarios(data.external_load_scenarios)
        .external_ncs_scenarios(data.external_ncs_scenarios)
        .build()
        .map_err(|errs| LoadError::ConstraintError {
            description: errs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        })?;

    Ok((system, report))
}
