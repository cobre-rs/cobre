//! Pipeline orchestrator for the five-layer validation and `System` construction pipeline.
//!
//! [`run_pipeline`] wires together all five validation layers, the resolution step,
//! the scenario assembly step, and `SystemBuilder::build` into a single callable
//! that either returns a fully-validated [`cobre_core::System`] or a
//! [`LoadError`] explaining what went wrong.

use std::collections::HashMap;

use cobre_core::{SystemBuilder, scenario::CorrelationModel};

use crate::{
    LoadError,
    resolution::{
        resolve_bounds, resolve_exchange_factors, resolve_generic_constraint_bounds,
        resolve_load_factors, resolve_ncs_bounds, resolve_ncs_factors, resolve_penalties,
    },
    scenarios::assembly::{assemble_inflow_models, assemble_load_models},
    validation::{
        ValidationContext,
        dimensional::validate_dimensional_consistency,
        referential::validate_referential_integrity,
        schema::validate_schema,
        semantic::{validate_semantic_hydro_thermal, validate_semantic_stages_penalties_scenarios},
        structural::validate_structure,
    },
};

use cobre_core::System;
use std::path::Path;

/// Run the complete loading pipeline for a case directory.
///
/// Executes all five validation layers, the three-tier resolution step, scenario
/// assembly, and `SystemBuilder::build`. Returns `Ok(System)` when every layer
/// succeeds, or the first `Err(LoadError)` encountered.
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

    // Convert all collected errors to a single LoadError. Warnings are silently
    // discarded here — they were already emitted to `ctx` and callers that want
    // them can inspect `ctx` before calling `into_result`, but the public API
    // only surfaces errors.
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
        &data.hydros,
        &data.buses,
        &data.lines,
        &data.non_controllable_sources,
        n_stages,
        &stage_index,
        &data.penalty_overrides_hydro,
        &data.penalty_overrides_bus,
        &data.penalty_overrides_line,
        &data.penalty_overrides_ncs,
    );

    let bounds = resolve_bounds(
        &data.hydros,
        &data.thermals,
        &data.lines,
        &data.pumping_stations,
        &data.energy_contracts,
        n_stages,
        &stage_index,
        &data.hydro_bounds,
        &data.thermal_bounds,
        &data.line_bounds,
        &data.pumping_bounds,
        &data.contract_bounds,
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
        .scenario_source(data.stages.scenario_source)
        .build()
        .map_err(|errs| LoadError::ConstraintError {
            description: errs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        })?;

    Ok(system)
}
