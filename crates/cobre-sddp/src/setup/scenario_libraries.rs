//! Scenario library builders for historical and external sampling schemes.
//!
//! Each builder constructs, validates, standardizes, and pads a library.
//! Builders are not factored generically because external types have different
//! standardization semantics.

use cobre_core::{
    EntityId, HydroPastInflows, InflowHistoryRow, Stage,
    scenario::{
        ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, HistoricalYears, LoadModel, NcsModel,
    },
    temporal::{SeasonMap, StageLagTransition},
};
use cobre_stochastic::{
    ExternalScenarioLibrary, HistoricalScenarioLibrary, PrecomputedPar,
    discover_historical_windows, pad_library_to_uniform, standardize_external_inflow,
    standardize_external_load, standardize_external_ncs, standardize_historical_windows,
    validate_external_library, validate_historical_library,
};

use crate::SddpError;

/// Build and validate a [`HistoricalScenarioLibrary`] for inflow.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on window discovery or validation failure.
pub(crate) fn build_historical_inflow_library(
    inflow_history: &[InflowHistoryRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    season_map: Option<&SeasonMap>,
    user_pool: Option<&HistoricalYears>,
    forward_passes: u32,
) -> Result<HistoricalScenarioLibrary, SddpError> {
    let max_order = par.max_order();
    let window_years = discover_historical_windows(
        inflow_history,
        hydro_ids,
        stages,
        max_order,
        user_pool,
        season_map,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    let mut library = HistoricalScenarioLibrary::new(
        window_years.len(),
        stages.len(),
        hydro_ids.len(),
        max_order,
        window_years.clone(),
    );
    standardize_historical_windows(
        &mut library,
        inflow_history,
        hydro_ids,
        stages,
        par,
        &window_years,
        season_map,
    );
    validate_historical_library(
        &library,
        inflow_history,
        hydro_ids,
        stages,
        max_order,
        user_pool,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    Ok(library)
}

/// Build and validate an [`ExternalScenarioLibrary`] for inflow.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_inflow_library(
    external_rows: &[ExternalScenarioRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    past_inflows: &[HydroPastInflows],
    stage_lag_transitions: &[StageLagTransition],
    forward_passes: u32,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let n_stages = stages.len();
    let n_hydros = hydro_ids.len();
    let row_entity_ids: std::collections::HashSet<EntityId> =
        external_rows.iter().map(|r| r.hydro_id).collect();
    let mut rows_per_stage = vec![0usize; n_stages];
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let s = row.stage_id as usize;
        if s < n_stages {
            rows_per_stage[s] += 1;
        }
    }
    let per_stage_scenarios: Vec<usize> = if n_hydros > 0 {
        rows_per_stage.iter().map(|&r| r / n_hydros).collect()
    } else {
        vec![0usize; n_stages]
    };
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library = ExternalScenarioLibrary::new(
        n_stages,
        n_scenarios_ext,
        n_hydros,
        "inflow",
        per_stage_scenarios,
    );
    validate_external_library(
        &library,
        hydro_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    standardize_external_inflow(
        &mut library,
        external_rows,
        hydro_ids,
        stages,
        par,
        past_inflows,
        stage_lag_transitions,
    );
    pad_library_to_uniform(&mut library);
    Ok(library)
}

/// Build and validate an [`ExternalScenarioLibrary`] for load.
///
/// Uses canonical bus ID list from `load_models` (buses with `std_mw > 0.0`).
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_load_library(
    external_rows: &[ExternalLoadRow],
    load_models: &[LoadModel],
    stages: &[Stage],
    forward_passes: u32,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let n_stages = stages.len();
    let mut bus_ids: Vec<EntityId> = load_models
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    bus_ids.sort_unstable_by_key(|id| id.0);
    bus_ids.dedup();
    let n_buses = bus_ids.len();
    let row_entity_ids: std::collections::HashSet<EntityId> =
        external_rows.iter().map(|r| r.bus_id).collect();
    let mut rows_per_stage = vec![0usize; n_stages];
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let s = row.stage_id as usize;
        if s < n_stages {
            rows_per_stage[s] += 1;
        }
    }
    let per_stage_scenarios: Vec<usize> = if n_buses > 0 {
        rows_per_stage.iter().map(|&r| r / n_buses).collect()
    } else {
        vec![0usize; n_stages]
    };
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library = ExternalScenarioLibrary::new(
        n_stages,
        n_scenarios_ext,
        n_buses,
        "load",
        per_stage_scenarios,
    );
    validate_external_library(
        &library,
        &bus_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    standardize_external_load(&mut library, external_rows, &bus_ids, load_models, n_stages);
    pad_library_to_uniform(&mut library);
    Ok(library)
}

/// Build and validate an [`ExternalScenarioLibrary`] for NCS.
///
/// Uses canonical NCS ID list from `ncs_models` (all NCS entities, sorted and deduped).
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_ncs_library(
    external_rows: &[ExternalNcsRow],
    ncs_models: &[NcsModel],
    stages: &[Stage],
    forward_passes: u32,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let n_stages = stages.len();
    let mut ncs_ids: Vec<EntityId> = ncs_models.iter().map(|m| m.ncs_id).collect();
    ncs_ids.sort_unstable_by_key(|id| id.0);
    ncs_ids.dedup();
    let n_ncs = ncs_ids.len();
    let row_entity_ids: std::collections::HashSet<EntityId> =
        external_rows.iter().map(|r| r.ncs_id).collect();
    let mut rows_per_stage = vec![0usize; n_stages];
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let s = row.stage_id as usize;
        if s < n_stages {
            rows_per_stage[s] += 1;
        }
    }
    let per_stage_scenarios: Vec<usize> = if n_ncs > 0 {
        rows_per_stage.iter().map(|&r| r / n_ncs).collect()
    } else {
        vec![0usize; n_stages]
    };
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library =
        ExternalScenarioLibrary::new(n_stages, n_scenarios_ext, n_ncs, "ncs", per_stage_scenarios);
    validate_external_library(
        &library,
        &ncs_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    standardize_external_ncs(&mut library, external_rows, &ncs_ids, ncs_models, n_stages);
    pad_library_to_uniform(&mut library);
    Ok(library)
}
