//! Scenario distribution and result extraction for the SDDP simulation phase.
//!
//! This module provides three pure functions consumed by the simulation forward
//! pass:
//!
//! - [`assign_scenarios`] — static scenario-to-rank distribution.
//! - [`extract_stage_result`] — LP solution → typed [`SimulationStageResult`].
//! - [`accumulate_category_costs`] — running per-category cost totals.
//!
//! ## Column layout
//!
//! The LP column layout is defined by [`StageIndexer`]:
//!
//! ```text
//! [0, N)             storage      — outgoing storage volumes
//! [N, N*(1+L))       inflow_lags  — AR lag variables (hydro-major order)
//! [N*(1+L), N*(2+L)) storage_in   — incoming storage volumes (fixed vars)
//! N*(2+L)            theta        — future cost variable
//! [theta+1, ...)     equipment    — turbine, spillage, thermal, lines, deficit, excess
//! ```
//!
//! Equipment columns (thermals, buses, lines) are read when the indexer was
//! constructed via [`StageIndexer::with_equipment`]. Stub entity types
//! (pumping stations, contracts, non-controllables) that contribute zero LP
//! variables remain as zero-valued placeholders.

use std::ops::Range;

use cobre_core::ConstraintSense;

use crate::StageIndexer;
use crate::lp_builder::GenericConstraintRowEntry;
use crate::simulation::types::{
    ScenarioCategoryCosts, SimulationBusResult, SimulationContractResult, SimulationCostResult,
    SimulationExchangeResult, SimulationGenericViolationResult, SimulationHydroResult,
    SimulationInflowLagResult, SimulationNonControllableResult, SimulationPumpingResult,
    SimulationStageResult, SimulationThermalResult,
};

/// System entity counts needed to populate per-entity result [`Vec`]s.
///
/// All counts are the number of entities that participate in the LP at runtime
/// (i.e., entities in their active operative state). Stub entity types
/// (contracts, pumping stations, non-controllables) that contribute zero LP
/// variables still need counts so the caller can allocate zero-length or
/// pre-allocated [`Vec`]s as appropriate.
///
/// # Entity IDs
///
/// The entity ID at position `h` in the hydro result vec is taken from
/// `hydro_ids[h]`. For equipment types whose column layout is not yet
/// exposed by [`StageIndexer`], the corresponding ID slices are still
/// iterated to produce stub zero-valued entries that preserve entity
/// ordering for the output writer.
#[derive(Debug, Clone)]
pub struct EntityCounts {
    /// Entity IDs for operating hydro plants, in canonical ID-sorted order.
    pub hydro_ids: Vec<i32>,
    /// Entity IDs for thermal units, in canonical ID-sorted order.
    pub thermal_ids: Vec<i32>,
    /// Entity IDs for transmission lines, in canonical ID-sorted order.
    pub line_ids: Vec<i32>,
    /// Entity IDs for buses, in canonical ID-sorted order.
    pub bus_ids: Vec<i32>,
    /// Productivity (MW per m³/s) for each hydro plant, same order as `hydro_ids`.
    ///
    /// Used to compute `generation_mw = turbined_m3s * productivity`.
    pub hydro_productivities: Vec<f64>,
    /// Entity IDs for pumping stations (may be empty if none exist).
    pub pumping_station_ids: Vec<i32>,
    /// Entity IDs for contracts (may be empty if none exist).
    pub contract_ids: Vec<i32>,
    /// Entity IDs for non-controllable sources (may be empty if none exist).
    pub non_controllable_ids: Vec<i32>,
}

/// Return the 0-based scenario ID range assigned to `rank` out of `world_size` ranks.
///
/// Uses a two-level distribution: the first `n_scenarios % world_size` ranks
/// receive one extra scenario (the "fat" group), and the remaining ranks receive
/// the floor. This matches the distribution strategy from
/// simulation-architecture.md SS3.1.
///
/// The sum of all ranks' range lengths equals `n_scenarios`.
///
/// # Panics
///
/// Panics in debug builds when `world_size == 0`.
///
/// # Examples
///
/// ```
/// use cobre_sddp::simulation::extraction::assign_scenarios;
///
/// // 10 scenarios, 3 ranks:
/// //   10 % 3 = 1  → rank 0 gets ceil(10/3) = 4 scenarios
/// //   ranks 1-2 get floor(10/3) = 3 scenarios
/// assert_eq!(assign_scenarios(10, 0, 3), 0..4);
/// assert_eq!(assign_scenarios(10, 1, 3), 4..7);
/// assert_eq!(assign_scenarios(10, 2, 3), 7..10);
///
/// // Single rank: all scenarios assigned to rank 0.
/// assert_eq!(assign_scenarios(7, 0, 1), 0..7);
/// ```
#[must_use]
pub fn assign_scenarios(n_scenarios: u32, rank: usize, world_size: usize) -> Range<u32> {
    debug_assert!(world_size > 0, "world_size must be > 0");

    let n = n_scenarios as usize;
    let r = world_size;

    // Number of "fat" ranks (each gets one extra scenario).
    let fat_count = n % r;
    let fat_size = n / r + 1; // ceil(n / r)
    let lean_size = n / r; // floor(n / r)

    // Compute the start offset for this rank.
    let start: usize = if rank < fat_count {
        // Rank is in the fat group.
        rank * fat_size
    } else {
        // Rank is in the lean group: fat group's total + lean offset.
        fat_count * fat_size + (rank - fat_count) * lean_size
    };

    let size = if rank < fat_count {
        fat_size
    } else {
        lean_size
    };
    let end = start + size;

    #[allow(clippy::cast_possible_truncation)]
    {
        (start as u32)..(end as u32)
    }
}

/// LP solution view passed to result extraction helpers.
///
/// Bundles the five LP output arrays so that extraction helpers each receive
/// a single `&SolutionView` instead of five separate slice parameters.
pub struct SolutionView<'a> {
    /// Primal variable values from the LP solve.
    pub primal: &'a [f64],
    /// Dual variable values (shadow prices) from the LP solve.
    pub dual: &'a [f64],
    /// LP objective value.
    pub objective: f64,
    /// Objective coefficient vector from the stage template.
    pub objective_coeffs: &'a [f64],
    /// Row lower bounds from the stage template (may be patched for load noise).
    pub row_lower: &'a [f64],
}

/// Extraction parameters bundled for a single stage.
///
/// Bundles the static configuration used by all per-entity extraction helpers
/// so that `extract_stage_result` needs only three parameters.
pub struct StageExtractionSpec<'a> {
    /// Stage indexer providing column/row layout.
    pub indexer: &'a StageIndexer,
    /// Entity ID lists and productivities needed to build result records.
    pub entity_counts: &'a EntityCounts,
    /// Volumetric inflow per hydro (m³/s), one entry per hydro plant.
    pub inflow_m3s_per_hydro: &'a [f64],
    /// Block hours per dispatch block, used to convert duals to spot prices.
    pub block_hours: &'a [f64],
    /// Per-row metadata for active generic constraint rows at this stage.
    ///
    /// Empty when no generic constraints are active.  Used by the extraction
    /// pipeline to map LP row/column indices back to constraint identity and
    /// block, and to read slack/dual values from the solution vectors.
    pub generic_constraint_entries: &'a [GenericConstraintRowEntry],
}

/// Extract hydro results from a raw LP solution view.
/// Extract one hydro result for the no-turbine (stage-level aggregate) branch.
fn extract_hydro_no_turbine(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    h: usize,
    hydro_id: i32,
    stage_id: u32,
) -> SimulationHydroResult {
    let indexer = spec.indexer;
    let incremental_inflow = if h < spec.inflow_m3s_per_hydro.len() {
        spec.inflow_m3s_per_hydro[h]
    } else if indexer.max_par_order > 0 {
        view.primal[indexer.inflow_lags.start + h * indexer.max_par_order]
    } else {
        0.0
    };
    let inflow_slack = if indexer.has_inflow_penalty {
        view.primal[indexer.inflow_slack.start + h]
    } else {
        0.0
    };
    let withdrawal_violation = if indexer.has_withdrawal {
        view.primal[indexer.withdrawal_slack.start + h]
    } else {
        0.0
    };
    let water_value = view.dual.get(indexer.n_state + h).copied().unwrap_or(0.0);

    // Determine if hydro `h` is FPHA. FPHA identification comes from
    // StageIndexer, not from EntityCounts.hydro_productivities.
    let is_fpha = indexer.fpha_hydro_indices.contains(&h);
    let productivity_mw_per_m3s = if is_fpha {
        None
    } else {
        Some(spec.entity_counts.hydro_productivities[h])
    };

    // Evaporation: read from LP columns when present; fall back to 0.0.
    let (evaporation_m3s, evaporation_violation_m3s) =
        if let Some(local_evap_idx) = indexer.evap_hydro_indices.iter().position(|&e| e == h) {
            let ei = &indexer.evap_indices[local_evap_idx];
            let q_ev = view.primal[ei.q_ev_col];
            let violation = view.primal[ei.f_evap_plus_col] + view.primal[ei.f_evap_minus_col];
            (Some(q_ev), violation)
        } else {
            (Some(0.0), 0.0)
        };

    SimulationHydroResult {
        stage_id,
        block_id: None,
        hydro_id,
        turbined_m3s: 0.0,
        spillage_m3s: 0.0,
        evaporation_m3s,
        diverted_inflow_m3s: Some(0.0),
        diverted_outflow_m3s: Some(0.0),
        incremental_inflow_m3s: incremental_inflow,
        inflow_m3s: incremental_inflow,
        storage_initial_hm3: view.primal[indexer.storage_in.start + h],
        storage_final_hm3: view.primal[indexer.storage.start + h],
        generation_mw: 0.0,
        productivity_mw_per_m3s,
        spillage_cost: 0.0,
        water_value_per_hm3: water_value,
        storage_binding_code: 0,
        operative_state_code: 1,
        turbined_slack_m3s: 0.0,
        outflow_slack_below_m3s: 0.0,
        outflow_slack_above_m3s: 0.0,
        generation_slack_mw: 0.0,
        storage_violation_below_hm3: 0.0,
        filling_target_violation_hm3: 0.0,
        evaporation_violation_m3s,
        inflow_nonnegativity_slack_m3s: inflow_slack,
        water_withdrawal_violation_m3s: withdrawal_violation,
    }
}

/// Extract per-block hydro results for one hydro plant (turbined/spillage branch).
fn extract_hydro_per_block<'a>(
    view: &'a SolutionView<'a>,
    spec: &'a StageExtractionSpec<'a>,
    h: usize,
    hydro_id: i32,
    stage_id: u32,
) -> impl Iterator<Item = SimulationHydroResult> + 'a {
    let indexer = spec.indexer;
    let n_blks = indexer.n_blks;
    let storage_final = view.primal[indexer.storage.start + h];
    let storage_initial = view.primal[indexer.storage_in.start + h];
    let incremental_inflow = if h < spec.inflow_m3s_per_hydro.len() {
        spec.inflow_m3s_per_hydro[h]
    } else if indexer.max_par_order > 0 {
        view.primal[indexer.inflow_lags.start + h * indexer.max_par_order]
    } else {
        0.0
    };
    let inflow_slack = if indexer.has_inflow_penalty {
        view.primal[indexer.inflow_slack.start + h]
    } else {
        0.0
    };
    let withdrawal_violation = if indexer.has_withdrawal {
        view.primal[indexer.withdrawal_slack.start + h]
    } else {
        0.0
    };
    let water_value = view.dual.get(indexer.n_state + h).copied().unwrap_or(0.0);

    // Determine if hydro `h` is FPHA. If so, record its local FPHA index so we
    // can read generation from the LP `g_{h,k}` column rather than computing
    // turbined * productivity. productivity_mw_per_m3s is None for FPHA hydros
    // because they use a piecewise function, not a scalar constant.
    let fpha_local: Option<usize> = indexer.fpha_hydro_indices.iter().position(|&e| e == h);
    let productivity_mw_per_m3s = if fpha_local.is_some() {
        None
    } else {
        Some(spec.entity_counts.hydro_productivities[h])
    };

    // Evaporation: stage-level (one column per hydro, same for all blocks).
    let local_evap: Option<usize> = indexer.evap_hydro_indices.iter().position(|&e| e == h);
    let (evaporation_m3s, evaporation_violation_m3s) = if let Some(lei) = local_evap {
        let ei = &indexer.evap_indices[lei];
        let q_ev = view.primal[ei.q_ev_col];
        let violation = view.primal[ei.f_evap_plus_col] + view.primal[ei.f_evap_minus_col];
        (Some(q_ev), violation)
    } else {
        (Some(0.0), 0.0)
    };

    (0..n_blks).map(move |b| {
        let t_col = indexer.turbine.start + h * n_blks + b;
        let s_col = indexer.spillage.start + h * n_blks + b;
        let turbined = view.primal[t_col];
        let spillage = view.primal[s_col];

        // For FPHA hydros, read generation from the LP `g_{h,k}` column.
        // For constant-productivity hydros, compute generation as turbined * productivity.
        let generation_mw = if let Some(local_fpha_idx) = fpha_local {
            view.primal[indexer.generation.start + local_fpha_idx * n_blks + b]
        } else {
            turbined * spec.entity_counts.hydro_productivities[h]
        };

        #[allow(clippy::cast_possible_truncation)]
        SimulationHydroResult {
            stage_id,
            block_id: Some(b as u32),
            hydro_id,
            turbined_m3s: turbined,
            spillage_m3s: spillage,
            evaporation_m3s,
            diverted_inflow_m3s: Some(0.0),
            diverted_outflow_m3s: Some(0.0),
            incremental_inflow_m3s: incremental_inflow,
            inflow_m3s: incremental_inflow,
            storage_initial_hm3: storage_initial,
            storage_final_hm3: storage_final,
            generation_mw,
            productivity_mw_per_m3s,
            spillage_cost: spillage * view.objective_coeffs[s_col],
            water_value_per_hm3: water_value,
            storage_binding_code: 0,
            operative_state_code: 1,
            turbined_slack_m3s: 0.0,
            outflow_slack_below_m3s: 0.0,
            outflow_slack_above_m3s: 0.0,
            generation_slack_mw: 0.0,
            storage_violation_below_hm3: 0.0,
            filling_target_violation_hm3: 0.0,
            evaporation_violation_m3s,
            inflow_nonnegativity_slack_m3s: inflow_slack,
            water_withdrawal_violation_m3s: withdrawal_violation,
        }
    })
}

fn extract_hydros(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationHydroResult> {
    let indexer = spec.indexer;
    if indexer.turbine.is_empty() || indexer.n_blks == 0 {
        spec.entity_counts
            .hydro_ids
            .iter()
            .enumerate()
            .map(|(h, &hydro_id)| extract_hydro_no_turbine(view, spec, h, hydro_id, stage_id))
            .collect()
    } else {
        spec.entity_counts
            .hydro_ids
            .iter()
            .enumerate()
            .flat_map(|(h, &hydro_id)| extract_hydro_per_block(view, spec, h, hydro_id, stage_id))
            .collect()
    }
}

/// Extract thermal results from a raw LP solution view.
fn extract_thermals(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationThermalResult> {
    let indexer = spec.indexer;
    let n_blks = indexer.n_blks;
    if indexer.thermal.is_empty() || n_blks == 0 {
        spec.entity_counts
            .thermal_ids
            .iter()
            .map(|&thermal_id| SimulationThermalResult {
                stage_id,
                block_id: None,
                thermal_id,
                generation_mw: 0.0,
                generation_cost: 0.0,
                is_gnl: false,
                gnl_committed_mw: None,
                gnl_decision_mw: None,
                operative_state_code: 1,
            })
            .collect()
    } else {
        spec.entity_counts
            .thermal_ids
            .iter()
            .enumerate()
            .flat_map(|(t, &thermal_id)| {
                (0..n_blks).map(move |b| {
                    let col = indexer.thermal.start + t * n_blks + b;
                    let gen_mw = view.primal[col];
                    #[allow(clippy::cast_possible_truncation)]
                    SimulationThermalResult {
                        stage_id,
                        block_id: Some(b as u32),
                        thermal_id,
                        generation_mw: gen_mw,
                        generation_cost: gen_mw * view.objective_coeffs[col],
                        is_gnl: false,
                        gnl_committed_mw: None,
                        gnl_decision_mw: None,
                        operative_state_code: 1,
                    }
                })
            })
            .collect()
    }
}

/// Extract exchange (line flow) results from a raw LP solution view.
fn extract_exchanges(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationExchangeResult> {
    let indexer = spec.indexer;
    let n_blks = indexer.n_blks;
    if indexer.line_fwd.is_empty() || n_blks == 0 {
        spec.entity_counts
            .line_ids
            .iter()
            .map(|&line_id| SimulationExchangeResult {
                stage_id,
                block_id: None,
                line_id,
                direct_flow_mw: 0.0,
                reverse_flow_mw: 0.0,
                exchange_cost: 0.0,
                operative_state_code: 1,
            })
            .collect()
    } else {
        spec.entity_counts
            .line_ids
            .iter()
            .enumerate()
            .flat_map(|(l, &line_id)| {
                (0..n_blks).map(move |b| {
                    let fwd_col = indexer.line_fwd.start + l * n_blks + b;
                    let rev_col = indexer.line_rev.start + l * n_blks + b;
                    let fwd = view.primal[fwd_col];
                    let rev = view.primal[rev_col];
                    #[allow(clippy::cast_possible_truncation)]
                    SimulationExchangeResult {
                        stage_id,
                        block_id: Some(b as u32),
                        line_id,
                        direct_flow_mw: fwd,
                        reverse_flow_mw: rev,
                        exchange_cost: fwd * view.objective_coeffs[fwd_col]
                            + rev * view.objective_coeffs[rev_col],
                        operative_state_code: 2,
                    }
                })
            })
            .collect()
    }
}

/// Extract bus results from a raw LP solution view.
fn extract_buses(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationBusResult> {
    let indexer = spec.indexer;
    let n_blks = indexer.n_blks;
    if indexer.deficit.is_empty() || n_blks == 0 {
        spec.entity_counts
            .bus_ids
            .iter()
            .map(|&bus_id| SimulationBusResult {
                stage_id,
                block_id: None,
                bus_id,
                load_mw: 0.0,
                deficit_mw: 0.0,
                excess_mw: 0.0,
                spot_price: 0.0,
            })
            .collect()
    } else {
        let max_segs = indexer.max_deficit_segments;
        spec.entity_counts
            .bus_ids
            .iter()
            .enumerate()
            .flat_map(move |(bus_idx, &bus_id)| {
                (0..n_blks).map(move |b| {
                    // Sum all deficit segment columns for this bus/block.
                    let deficit_mw: f64 = (0..max_segs)
                        .map(|s| {
                            let col = indexer.deficit.start
                                + bus_idx * max_segs * n_blks
                                + s * n_blks
                                + b;
                            view.primal[col]
                        })
                        .sum();
                    let excess_col = indexer.excess.start + bus_idx * n_blks + b;
                    let load_row = indexer.load_balance.start + bus_idx * n_blks + b;
                    let raw_dual = view.dual.get(load_row).copied().unwrap_or(0.0);
                    let hrs = spec.block_hours.get(b).copied().unwrap_or(0.0);
                    #[allow(clippy::cast_possible_truncation)]
                    SimulationBusResult {
                        stage_id,
                        block_id: Some(b as u32),
                        bus_id,
                        load_mw: view.row_lower[load_row],
                        deficit_mw,
                        excess_mw: view.primal[excess_col],
                        spot_price: if hrs > 0.0 { raw_dual / hrs } else { 0.0 },
                    }
                })
            })
            .collect()
    }
}

/// Extract a [`SimulationStageResult`] from a raw LP solution at one stage.
///
/// Reads equipment column values from `view.primal` using the ranges stored in
/// `spec.indexer`. When the indexer was constructed via
/// [`StageIndexer::with_equipment`] equipment columns are read directly from the
/// primal solution vector. When constructed via [`StageIndexer::new`] equipment
/// ranges are empty and results default to zero (backward-compatible behaviour).
///
/// The LP objective is split into `future_cost = primal[indexer.theta]` and
/// `stage_cost = objective - future_cost`, following the same convention as the
/// training forward pass.
///
/// # Preconditions
///
/// - `view.primal.len() >= spec.indexer.theta + 1`
/// - `spec.entity_counts.hydro_ids.len() == spec.indexer.hydro_count`
/// - `spec.entity_counts.hydro_productivities.len() == spec.indexer.hydro_count`
/// - `view.objective_coeffs.len() >= view.primal.len()` when equipment ranges are non-empty
/// - `view.row_lower.len() >= spec.indexer.load_balance.end` when `load_balance` is non-empty
/// - `stage_id` is 0-based
///
/// Violations are caught by `debug_assert!` in debug builds.
#[must_use]
pub fn extract_stage_result(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> SimulationStageResult {
    let indexer = spec.indexer;
    debug_assert!(
        view.primal.len() > indexer.theta,
        "primal vector too short: len={}, need > theta={}",
        view.primal.len(),
        indexer.theta
    );
    debug_assert!(
        spec.entity_counts.hydro_ids.len() == indexer.hydro_count,
        "hydro_ids length {} does not match indexer.hydro_count {}",
        spec.entity_counts.hydro_ids.len(),
        indexer.hydro_count
    );
    debug_assert!(
        indexer.excess.is_empty() || view.objective_coeffs.len() >= indexer.excess.end,
        "objective_coeffs too short: len={}, need >= excess.end={}",
        view.objective_coeffs.len(),
        indexer.excess.end
    );
    debug_assert!(
        spec.entity_counts.hydro_productivities.len() == indexer.hydro_count,
        "hydro_productivities length {} does not match indexer.hydro_count {}",
        spec.entity_counts.hydro_productivities.len(),
        indexer.hydro_count
    );
    debug_assert!(
        indexer.load_balance.is_empty() || view.row_lower.len() >= indexer.load_balance.end,
        "row_lower too short: len={}, need >= load_balance.end={}",
        view.row_lower.len(),
        indexer.load_balance.end
    );

    let (generic_violations, generic_violation_cost) =
        extract_generic_violations(view, spec, stage_id);
    let costs = vec![compute_cost_result(
        view,
        spec.indexer,
        generic_violation_cost,
        stage_id,
    )];
    let (inflow_lags, pumping_stations, contracts, non_controllables) =
        extract_stub_collections(view, spec, stage_id);

    SimulationStageResult {
        stage_id,
        costs,
        hydros: extract_hydros(view, spec, stage_id),
        thermals: extract_thermals(view, spec, stage_id),
        exchanges: extract_exchanges(view, spec, stage_id),
        buses: extract_buses(view, spec, stage_id),
        pumping_stations,
        contracts,
        non_controllables,
        inflow_lags,
        generic_violations,
    }
}

/// Compute the single-stage cost breakdown from an LP solution view.
fn compute_cost_result(
    view: &SolutionView<'_>,
    indexer: &StageIndexer,
    generic_violation_cost: f64,
    stage_id: u32,
) -> SimulationCostResult {
    let col_cost = |col: usize| view.primal[col] * view.objective_coeffs[col];
    let range_sum = |r: std::ops::Range<usize>| -> f64 { r.map(col_cost).sum() };

    let future_cost = view.primal[indexer.theta];
    let immediate_cost = view.objective - future_cost;
    let thermal_cost = if indexer.thermal.is_empty() {
        0.0
    } else {
        range_sum(indexer.thermal.clone())
    };
    let spillage_cost = if indexer.spillage.is_empty() {
        0.0
    } else {
        range_sum(indexer.spillage.clone())
    };
    let exchange_cost = if indexer.line_fwd.is_empty() {
        0.0
    } else {
        indexer
            .line_fwd
            .clone()
            .chain(indexer.line_rev.clone())
            .map(col_cost)
            .sum()
    };
    let deficit_cost = if indexer.deficit.is_empty() {
        0.0
    } else {
        range_sum(indexer.deficit.clone())
    };
    let excess_cost = if indexer.excess.is_empty() {
        0.0
    } else {
        range_sum(indexer.excess.clone())
    };

    // FPHA turbined cost: sum primal[col] * objective[col] over all FPHA turbine
    // columns. Non-FPHA turbine columns have zero objective coefficient, but we
    // restrict the sum explicitly to FPHA hydros for clarity.
    let fpha_turbined_cost = if indexer.generation.is_empty() {
        0.0
    } else {
        let n_blks = indexer.n_blks;
        indexer
            .fpha_hydro_indices
            .iter()
            .enumerate()
            .flat_map(|(local_fpha_idx, _sys_h)| {
                (0..n_blks).map(move |b| indexer.generation.start + local_fpha_idx * n_blks + b)
            })
            .map(col_cost)
            .sum()
    };

    SimulationCostResult {
        stage_id,
        block_id: None,
        total_cost: view.objective,
        immediate_cost,
        future_cost,
        discount_factor: 1.0,
        thermal_cost,
        contract_cost: 0.0,
        deficit_cost,
        excess_cost,
        storage_violation_cost: 0.0,
        filling_target_cost: 0.0,
        hydro_violation_cost: 0.0,
        inflow_penalty_cost: 0.0,
        generic_violation_cost,
        spillage_cost,
        fpha_turbined_cost,
        curtailment_cost: 0.0,
        exchange_cost,
        pumping_cost: 0.0,
    }
}

/// Extract generic constraint violation results from a solved LP.
///
/// For each active generic constraint row, reads the slack value from the
/// primal vector (for constraints with `slack.enabled`) and the dual value
/// from the dual vector.  Returns the violation records and the total
/// violation cost (sum of `slack_value * penalty * block_hours` across all
/// active constraint rows).
///
/// For `==` sense constraints with two slack columns (positive and negative),
/// the reported `slack_value` is the net violation: `s_plus - s_minus`.
fn extract_generic_violations(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (Vec<SimulationGenericViolationResult>, f64) {
    let entries = spec.generic_constraint_entries;
    if entries.is_empty() {
        return (Vec::new(), 0.0);
    }

    let indexer = spec.indexer;
    let gc_row_start = indexer.generic_constraint_rows.start;
    let mut results = Vec::with_capacity(entries.len());
    let mut total_cost = 0.0;

    for (entry_idx, entry) in entries.iter().enumerate() {
        let row_idx = gc_row_start + entry_idx;

        // Dual value from the LP row (unused for now but kept for future use).
        let _dual_value = if row_idx < view.dual.len() {
            view.dual[row_idx]
        } else {
            0.0
        };

        // Slack value from the LP column(s).
        let block_hours = spec
            .block_hours
            .get(entry.block_idx)
            .copied()
            .unwrap_or(0.0);
        let (slack_value, slack_cost) = if entry.slack_enabled {
            match entry.sense {
                ConstraintSense::Equal => {
                    // Two slack columns: s_plus and s_minus.
                    let s_plus = entry.slack_plus_col.map_or(0.0, |col| view.primal[col]);
                    let s_minus = entry.slack_minus_col.map_or(0.0, |col| view.primal[col]);
                    let net = s_plus - s_minus;
                    // Cost is for both slack variables individually.
                    let cost = (s_plus + s_minus) * entry.slack_penalty * block_hours;
                    (net, cost)
                }
                ConstraintSense::LessEqual | ConstraintSense::GreaterEqual => {
                    let s = entry.slack_plus_col.map_or(0.0, |col| view.primal[col]);
                    let cost = s * entry.slack_penalty * block_hours;
                    (s, cost)
                }
            }
        } else {
            (0.0, 0.0)
        };

        total_cost += slack_cost;

        results.push(SimulationGenericViolationResult {
            stage_id,
            block_id: Some(entry.block_idx as u32),
            constraint_id: entry.entity_id,
            slack_value,
            slack_cost,
        });
    }

    (results, total_cost)
}

/// Extract stub (zero-value) result collections for currently-unmodeled entity types.
#[allow(clippy::type_complexity)]
fn extract_stub_collections(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (
    Vec<SimulationInflowLagResult>,
    Vec<SimulationPumpingResult>,
    Vec<SimulationContractResult>,
    Vec<SimulationNonControllableResult>,
) {
    let indexer = spec.indexer;
    let inflow_lags: Vec<SimulationInflowLagResult> = spec
        .entity_counts
        .hydro_ids
        .iter()
        .enumerate()
        .flat_map(|(h, &hydro_id)| {
            (0..indexer.max_par_order).map(move |l| {
                #[allow(clippy::cast_possible_truncation)]
                SimulationInflowLagResult {
                    stage_id,
                    hydro_id,
                    lag_index: l as u32,
                    inflow_m3s: view.primal
                        [indexer.inflow_lags.start + h * indexer.max_par_order + l],
                }
            })
        })
        .collect();
    let pumping_stations: Vec<SimulationPumpingResult> = spec
        .entity_counts
        .pumping_station_ids
        .iter()
        .map(|&pumping_station_id| SimulationPumpingResult {
            stage_id,
            block_id: None,
            pumping_station_id,
            pumped_flow_m3s: 0.0,
            power_consumption_mw: 0.0,
            pumping_cost: 0.0,
            operative_state_code: 1,
        })
        .collect();
    let contracts: Vec<SimulationContractResult> = spec
        .entity_counts
        .contract_ids
        .iter()
        .map(|&contract_id| SimulationContractResult {
            stage_id,
            block_id: None,
            contract_id,
            power_mw: 0.0,
            price_per_mwh: 0.0,
            total_cost: 0.0,
            operative_state_code: 1,
        })
        .collect();
    let non_controllables: Vec<SimulationNonControllableResult> = spec
        .entity_counts
        .non_controllable_ids
        .iter()
        .map(|&non_controllable_id| SimulationNonControllableResult {
            stage_id,
            block_id: None,
            non_controllable_id,
            generation_mw: 0.0,
            available_mw: 0.0,
            curtailment_mw: 0.0,
            curtailment_cost: 0.0,
            operative_state_code: 1,
        })
        .collect();
    (inflow_lags, pumping_stations, contracts, non_controllables)
}

/// Add one stage's cost breakdown into a running per-category accumulator.
///
/// The five categories follow the breakdown in `ScenarioCategoryCosts`:
///
/// | Field              | Sum expression                                        |
/// |--------------------|-------------------------------------------------------|
/// | `resource_cost`    | `thermal_cost + contract_cost`                        |
/// | `recourse_cost`    | `deficit_cost + excess_cost`                          |
/// | `violation_cost`   | `storage_violation_cost + filling_target_cost`        |
/// |                    | `+ hydro_violation_cost + inflow_penalty_cost`        |
/// |                    | `+ generic_violation_cost`                            |
/// | `regularization_cost` | `spillage_cost + fpha_turbined_cost`               |
/// |                    | `+ curtailment_cost + exchange_cost`                  |
/// | `imputed_cost`     | `pumping_cost`                                        |
///
/// # Examples
///
/// ```
/// use cobre_sddp::simulation::types::{ScenarioCategoryCosts, SimulationCostResult};
/// use cobre_sddp::simulation::extraction::accumulate_category_costs;
///
/// let cost = SimulationCostResult {
///     stage_id: 0,
///     block_id: None,
///     total_cost: 1000.0,
///     immediate_cost: 800.0,
///     future_cost: 200.0,
///     discount_factor: 1.0,
///     thermal_cost: 400.0,
///     contract_cost: 100.0,
///     deficit_cost: 50.0,
///     excess_cost: 10.0,
///     storage_violation_cost: 20.0,
///     filling_target_cost: 30.0,
///     hydro_violation_cost: 5.0,
///     inflow_penalty_cost: 3.0,
///     generic_violation_cost: 2.0,
///     spillage_cost: 1.0,
///     fpha_turbined_cost: 4.0,
///     curtailment_cost: 7.0,
///     exchange_cost: 8.0,
///     pumping_cost: 60.0,
/// };
///
/// let mut accum = ScenarioCategoryCosts {
///     resource_cost: 0.0,
///     recourse_cost: 0.0,
///     violation_cost: 0.0,
///     regularization_cost: 0.0,
///     imputed_cost: 0.0,
/// };
///
/// accumulate_category_costs(&cost, &mut accum);
/// assert_eq!(accum.resource_cost, 500.0);       // 400 + 100
/// assert_eq!(accum.recourse_cost, 60.0);         // 50 + 10
/// assert_eq!(accum.violation_cost, 60.0);        // 20 + 30 + 5 + 3 + 2
/// assert_eq!(accum.regularization_cost, 20.0);   // 1 + 4 + 7 + 8
/// assert_eq!(accum.imputed_cost, 60.0);          // 60
/// ```
pub fn accumulate_category_costs(cost: &SimulationCostResult, accum: &mut ScenarioCategoryCosts) {
    accum.resource_cost += cost.thermal_cost + cost.contract_cost;
    accum.recourse_cost += cost.deficit_cost + cost.excess_cost;
    accum.violation_cost += cost.storage_violation_cost
        + cost.filling_target_cost
        + cost.hydro_violation_cost
        + cost.inflow_penalty_cost
        + cost.generic_violation_cost;
    accum.regularization_cost +=
        cost.spillage_cost + cost.fpha_turbined_cost + cost.curtailment_cost + cost.exchange_cost;
    accum.imputed_cost += cost.pumping_cost;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::{
        EntityCounts, SolutionView, StageExtractionSpec, accumulate_category_costs,
        assign_scenarios, extract_stage_result,
    };
    use crate::StageIndexer;
    use crate::simulation::types::{ScenarioCategoryCosts, SimulationCostResult};

    // -------------------------------------------------------------------------
    // assign_scenarios
    // -------------------------------------------------------------------------

    #[test]
    fn assign_scenarios_uneven_rank0() {
        // Acceptance criterion: n=10, rank=0, world=3 → 0..4
        // 10 % 3 = 1 fat rank → rank 0 gets ceil(10/3) = 4 scenarios
        assert_eq!(assign_scenarios(10, 0, 3), 0..4);
    }

    #[test]
    fn assign_scenarios_uneven_rank2() {
        // Acceptance criterion: n=10, rank=2, world=3 → 7..10
        assert_eq!(assign_scenarios(10, 2, 3), 7..10);
    }

    #[test]
    fn assign_scenarios_single_rank() {
        // Acceptance criterion: n=7, rank=0, world=1 → 0..7
        assert_eq!(assign_scenarios(7, 0, 1), 0..7);
    }

    #[test]
    fn assign_scenarios_uneven_rank1() {
        // Derived from acceptance criteria: rank 1 is a lean rank with 3 scenarios
        // starting at offset 4 (end of rank 0's fat slice).
        assert_eq!(assign_scenarios(10, 1, 3), 4..7);
    }

    #[test]
    fn assign_scenarios_exact_division() {
        // n=9, world=3: 9 % 3 = 0 fat ranks → all lean (3 each)
        assert_eq!(assign_scenarios(9, 0, 3), 0..3);
        assert_eq!(assign_scenarios(9, 1, 3), 3..6);
        assert_eq!(assign_scenarios(9, 2, 3), 6..9);
    }

    #[test]
    fn assign_scenarios_zero_scenarios() {
        // Every rank gets an empty range.
        assert_eq!(assign_scenarios(0, 0, 1), 0..0);
        assert_eq!(assign_scenarios(0, 0, 4), 0..0);
        assert_eq!(assign_scenarios(0, 3, 4), 0..0);
    }

    #[test]
    fn assign_scenarios_more_ranks_than_scenarios() {
        // n=2, world=5: ranks 0-1 get 1 scenario each; ranks 2-4 get empty.
        assert_eq!(assign_scenarios(2, 0, 5), 0..1);
        assert_eq!(assign_scenarios(2, 1, 5), 1..2);
        assert_eq!(assign_scenarios(2, 2, 5), 2..2);
        assert_eq!(assign_scenarios(2, 3, 5), 2..2);
        assert_eq!(assign_scenarios(2, 4, 5), 2..2);
    }

    #[test]
    fn assign_scenarios_sum_equals_n_scenarios() {
        // Property test: for various (n, world_size) pairs, total assigned = n.
        for (n, world_size) in [(0_u32, 1_usize), (1, 1), (10, 3), (9, 3), (2, 5), (100, 7)] {
            let total: u32 = (0..world_size)
                .map(|rank| {
                    let r = assign_scenarios(n, rank, world_size);
                    r.end - r.start
                })
                .sum();
            assert_eq!(
                total, n,
                "total assigned {total} != n_scenarios {n} for world_size={world_size}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // extract_stage_result
    // -------------------------------------------------------------------------

    fn make_entity_counts_2_hydros() -> EntityCounts {
        EntityCounts {
            hydro_ids: vec![10, 20],
            hydro_productivities: vec![1.0, 1.0],
            thermal_ids: vec![1],
            line_ids: vec![5],
            bus_ids: vec![100],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        }
    }

    /// Build a primal vector for `StageIndexer` with `hydro_count=2`, `max_par_order=1`.
    ///
    /// Layout: storage\[0..2\], `inflow_lags`\[2..4\], `storage_in`\[4..6\], theta=6
    fn make_primal_2_1(
        storage: [f64; 2],
        lags: [f64; 2],
        storage_in: [f64; 2],
        theta: f64,
    ) -> Vec<f64> {
        vec![
            storage[0],
            storage[1],
            lags[0],
            lags[1],
            storage_in[0],
            storage_in[1],
            theta,
        ]
    }

    #[test]
    fn extract_costs_has_one_entry_matching_stage_id() {
        // Acceptance criterion: costs contains exactly one entry whose stage_id
        // matches the input stage and whose future_cost == primal[indexer.theta].
        let indexer = StageIndexer::new(2, 1);
        let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
        let dual = vec![0.0; 4];

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 1500.0,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &make_entity_counts_2_hydros(),
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            3,
        );

        assert_eq!(result.costs.len(), 1);
        assert_eq!(result.costs[0].stage_id, 3);
        assert_eq!(result.costs[0].future_cost, 999.5); // primal[theta]
    }

    #[test]
    fn extract_cost_splits_objective_correctly() {
        // objective = immediate_cost + future_cost
        let indexer = StageIndexer::new(2, 1);
        let theta_val = 300.0;
        let objective = 800.0;
        let primal = make_primal_2_1([0.0; 2], [0.0; 2], [0.0; 2], theta_val);
        let dual = vec![0.0; 4];

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &make_entity_counts_2_hydros(),
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        let cost = &result.costs[0];
        assert_eq!(cost.future_cost, theta_val);
        assert_eq!(cost.immediate_cost, objective - theta_val);
        assert_eq!(cost.total_cost, objective);
    }

    #[test]
    fn extract_hydro_storage_values_from_primal() {
        // Hydro h=0: storage[0]=100, storage_in[4]=90
        // Hydro h=1: storage[1]=200, storage_in[5]=180
        let indexer = StageIndexer::new(2, 1);
        let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
        let dual = vec![0.0; 4];

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 1500.0,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &make_entity_counts_2_hydros(),
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        assert_eq!(result.hydros.len(), 2);
        assert_eq!(result.hydros[0].hydro_id, 10);
        assert_eq!(result.hydros[0].storage_initial_hm3, 90.0);
        assert_eq!(result.hydros[0].storage_final_hm3, 100.0);

        assert_eq!(result.hydros[1].hydro_id, 20);
        assert_eq!(result.hydros[1].storage_initial_hm3, 180.0);
        assert_eq!(result.hydros[1].storage_final_hm3, 200.0);
    }

    #[test]
    fn extract_inflow_lag_values_from_primal() {
        // inflow_lags[2]=50.0 for hydro 0 lag 0, [3]=60.0 for hydro 1 lag 0
        let indexer = StageIndexer::new(2, 1);
        let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
        let dual = vec![0.0; 4];

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 1500.0,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &make_entity_counts_2_hydros(),
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        assert_eq!(result.inflow_lags.len(), 2); // 2 hydros × 1 lag each
        // Hydro 10, lag 0 → primal[2] = 50.0
        assert_eq!(result.inflow_lags[0].hydro_id, 10);
        assert_eq!(result.inflow_lags[0].lag_index, 0);
        assert_eq!(result.inflow_lags[0].inflow_m3s, 50.0);
        // Hydro 20, lag 0 → primal[3] = 60.0
        assert_eq!(result.inflow_lags[1].hydro_id, 20);
        assert_eq!(result.inflow_lags[1].lag_index, 0);
        assert_eq!(result.inflow_lags[1].inflow_m3s, 60.0);
    }

    #[test]
    fn extract_no_lags_when_max_par_order_zero() {
        // StageIndexer(N=2, L=0): no inflow_lag columns → empty inflow_lags vec.
        let indexer = StageIndexer::new(2, 0);
        // Layout: storage[0..2], storage_in[2..4], theta=4
        let primal = vec![100.0, 200.0, 90.0, 180.0, 500.0];
        let dual = vec![];
        let counts = EntityCounts {
            hydro_ids: vec![10, 20],
            hydro_productivities: vec![1.0, 1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 600.0,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            2,
        );

        assert!(result.inflow_lags.is_empty());
        assert_eq!(result.hydros[0].incremental_inflow_m3s, 0.0);
    }

    #[test]
    fn extract_stage_id_propagates_to_all_results() {
        let indexer = StageIndexer::new(2, 1);
        let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 10.0);
        let dual = vec![0.0; 4];
        let stage_id = 7_u32;

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 110.0,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &make_entity_counts_2_hydros(),
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            stage_id,
        );

        assert_eq!(result.stage_id, stage_id);
        assert_eq!(result.costs[0].stage_id, stage_id);
        assert!(result.hydros.iter().all(|h| h.stage_id == stage_id));
        assert!(result.thermals.iter().all(|t| t.stage_id == stage_id));
        assert!(result.exchanges.iter().all(|e| e.stage_id == stage_id));
        assert!(result.buses.iter().all(|b| b.stage_id == stage_id));
        assert!(result.inflow_lags.iter().all(|l| l.stage_id == stage_id));
    }

    #[test]
    fn extract_equipment_zero_when_indexer_has_no_equipment_ranges() {
        // When StageIndexer is built via `new`, equipment ranges are empty and
        // all equipment result fields default to zero — backward-compatible behaviour.
        let indexer = StageIndexer::new(2, 1);
        let primal = make_primal_2_1([0.0; 2], [0.0; 2], [0.0; 2], 0.0);
        let dual = vec![0.0; 4];

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 0.0,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &make_entity_counts_2_hydros(),
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        // Thermal — one entry per thermal entity, all zero.
        assert_eq!(result.thermals.len(), 1);
        assert_eq!(result.thermals[0].generation_mw, 0.0);
        assert_eq!(result.thermals[0].generation_cost, 0.0);
        assert_eq!(result.thermals[0].block_id, None);

        // Exchange — one entry per line entity, all zero.
        assert_eq!(result.exchanges.len(), 1);
        assert_eq!(result.exchanges[0].direct_flow_mw, 0.0);
        assert_eq!(result.exchanges[0].block_id, None);

        // Bus — one entry per bus entity, all zero.
        assert_eq!(result.buses.len(), 1);
        assert_eq!(result.buses[0].deficit_mw, 0.0);
        assert_eq!(result.buses[0].spot_price, 0.0);
        assert_eq!(result.buses[0].block_id, None);
    }

    /// Verify that equipment columns are read from the primal vector when the
    /// indexer was built via `with_equipment`.
    ///
    /// Column layout for N=2 hydros, L=1 lag, T=1 thermal, Ln=1 line, B=1 bus, K=1 block:
    ///
    /// ```text
    /// theta = N*(2+L) = 2*(2+1) = 6
    /// turbine:  [7, 9)    h0→7, h1→8
    /// spillage: [9, 11)   h0→9, h1→10
    /// thermal: [11, 12)   t0→11
    /// line_fwd:[12, 13)   l0→12
    /// line_rev:[13, 14)   l0→13
    /// deficit: [14, 15)   b0→14
    /// excess:  [15, 16)   b0→15
    /// ```
    #[test]
    fn extract_equipment_reads_primal_when_with_equipment() {
        // N=2, L=1, T=1, Ln=1, B=1, K=1
        let indexer = StageIndexer::with_equipment(2, 1, 1, 1, 1, 1, false, vec![], &[]);
        // theta = 6, equipment starts at 7
        assert_eq!(indexer.theta, 6);
        assert_eq!(indexer.turbine, 7..9);
        assert_eq!(indexer.spillage, 9..11);
        assert_eq!(indexer.thermal, 11..12);
        assert_eq!(indexer.line_fwd, 12..13);
        assert_eq!(indexer.line_rev, 13..14);
        assert_eq!(indexer.deficit, 14..15);
        assert_eq!(indexer.excess, 15..16);

        // Build a primal vector sized to include withdrawal_slack columns.
        // storage[0..2]=100,200  inflow_lags[2..4]=50,60  storage_in[4..6]=90,180  theta[6]=500
        // turbine[7..9]=30.0,40.0   spillage[9..11]=5.0,0.0
        // thermal[11]=80.0   line_fwd[12]=15.0   line_rev[13]=0.0
        // deficit[14]=10.0   excess[15]=2.0   withdrawal_slack[16..18]=0.0,0.0
        let n_cols = indexer.withdrawal_slack.end;
        let mut primal = vec![0.0_f64; n_cols];
        primal[0] = 100.0; // storage h0
        primal[1] = 200.0; // storage h1
        primal[2] = 50.0; // lag h0
        primal[3] = 60.0; // lag h1
        primal[4] = 90.0; // storage_in h0
        primal[5] = 180.0; // storage_in h1
        primal[6] = 500.0; // theta
        primal[7] = 30.0; // turbine h0 b0
        primal[8] = 40.0; // turbine h1 b0
        primal[9] = 5.0; // spillage h0 b0
        primal[10] = 0.0; // spillage h1 b0
        primal[11] = 80.0; // thermal t0 b0
        primal[12] = 15.0; // line_fwd l0 b0
        primal[13] = 0.0; // line_rev l0 b0
        primal[14] = 10.0; // deficit b0 b0
        primal[15] = 2.0; // excess b0 b0

        // Objective coefficients: thermal cost=50/MWh, spillage cost=0.1, deficit=1000, excess=50
        let mut obj = vec![0.0_f64; n_cols];
        obj[6] = 1.0; // theta (objective = 1)
        obj[9] = 0.1; // spillage h0 penalty
        obj[11] = 50.0; // thermal cost per MW
        obj[12] = 5.0; // line_fwd cost per MW
        obj[14] = 1000.0; // deficit cost per MW
        obj[15] = 50.0; // excess cost per MW

        let counts = EntityCounts {
            hydro_ids: vec![10, 20],
            hydro_productivities: vec![1.0, 1.0],
            thermal_ids: vec![1],
            line_ids: vec![5],
            bus_ids: vec![100],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };
        // Dual vector: indices 0..4 = storage/lag fixing, 4..6 = water balance, 6 = load balance.
        // water_value for h0 = dual[4], h1 = dual[5]; spot_price for b0 = dual[6].
        let mut dual = vec![0.0_f64; 7];
        dual[4] = -120.0; // water value h0 ($/hm³)
        dual[5] = -95.0; // water value h1 ($/hm³)
        dual[6] = 108_000.0; // raw load balance dual ($/MW); 150 $/MWh × 720 h

        // Build row_lower for the load balance row. N=2, L=1 → n_state=4, water_bal_start=4,
        // load_bal_start=4+2=6. K=1, B=1 → one load balance row at index 6.
        let mut row_lower = vec![0.0_f64; 7]; // must be >= load_balance.end = 7
        row_lower[6] = 75.0; // load = 75 MW for bus 100
        let block_hours = [720.0_f64]; // one block, 30-day month
        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 600.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &block_hours,
                generic_constraint_entries: &[],
            },
            0,
        );

        // Hydro: one entry per (hydro, block), block_id = Some(0)
        assert_eq!(result.hydros.len(), 2); // 2 hydros × 1 block
        assert_eq!(result.hydros[0].block_id, Some(0));
        assert_eq!(result.hydros[0].turbined_m3s, 30.0);
        assert_eq!(result.hydros[0].spillage_m3s, 5.0);
        assert!((result.hydros[0].spillage_cost - 0.5).abs() < 1e-12); // 5.0 * 0.1

        // Hydro generation = turbined * productivity (1.0)
        assert_eq!(result.hydros[0].generation_mw, 30.0); // 30 * 1.0
        assert_eq!(result.hydros[1].generation_mw, 40.0); // 40 * 1.0

        // Hydro h=1 (no spillage)
        assert_eq!(result.hydros[1].block_id, Some(0));
        assert_eq!(result.hydros[1].turbined_m3s, 40.0);
        assert_eq!(result.hydros[1].spillage_m3s, 0.0);

        // Thermal: one entry per (thermal, block), block_id = Some(0)
        assert_eq!(result.thermals.len(), 1);
        assert_eq!(result.thermals[0].generation_mw, 80.0);
        assert!((result.thermals[0].generation_cost - 4000.0).abs() < 1e-12); // 80 * 50
        assert_eq!(result.thermals[0].block_id, Some(0));

        // Exchange: one entry per (line, block)
        assert_eq!(result.exchanges.len(), 1);
        assert_eq!(result.exchanges[0].direct_flow_mw, 15.0);
        assert_eq!(result.exchanges[0].reverse_flow_mw, 0.0);
        assert!((result.exchanges[0].exchange_cost - 75.0).abs() < 1e-12); // 15 * 5
        assert_eq!(result.exchanges[0].block_id, Some(0));

        // Bus: one entry per (bus, block)
        assert_eq!(result.buses.len(), 1);
        assert_eq!(result.buses[0].load_mw, 75.0); // from row_lower
        assert_eq!(result.buses[0].deficit_mw, 10.0);
        assert_eq!(result.buses[0].excess_mw, 2.0);
        assert_eq!(result.buses[0].block_id, Some(0));
        assert!((result.buses[0].spot_price - 150.0).abs() < 1e-12); // 108_000 / 720 = 150 $/MWh

        // Water value from dual of water balance rows
        assert!((result.hydros[0].water_value_per_hm3 - (-120.0)).abs() < 1e-12);
        assert!((result.hydros[1].water_value_per_hm3 - (-95.0)).abs() < 1e-12);

        // Cost breakdown
        let cost = &result.costs[0];
        assert!((cost.thermal_cost - 4000.0).abs() < 1e-12); // 80 * 50
        assert!((cost.spillage_cost - 0.5).abs() < 1e-12); // 5 * 0.1
        assert!((cost.deficit_cost - 10_000.0).abs() < 1e-12); // 10 * 1000
        assert!((cost.excess_cost - 100.0).abs() < 1e-12); // 2 * 50
        assert!((cost.exchange_cost - 75.0).abs() < 1e-12); // 15 * 5
    }

    #[test]
    fn extract_optional_entity_types_are_empty_when_absent() {
        let indexer = StageIndexer::new(1, 0);
        let primal = vec![50.0, 40.0, 200.0]; // storage, storage_in, theta
        let dual = vec![];
        let counts = EntityCounts {
            hydro_ids: vec![1],
            hydro_productivities: vec![1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 250.0,
                objective_coeffs: &[],
                row_lower: &[],
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        assert!(result.pumping_stations.is_empty());
        assert!(result.contracts.is_empty());
        assert!(result.non_controllables.is_empty());
        assert!(result.generic_violations.is_empty());
    }

    // -------------------------------------------------------------------------
    // accumulate_category_costs
    // -------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn make_cost(
        thermal: f64,
        contract: f64,
        deficit: f64,
        excess: f64,
        storage_violation: f64,
        filling: f64,
        hydro_violation: f64,
        inflow_penalty: f64,
        generic_violation: f64,
        spillage: f64,
        fpha: f64,
        curtailment: f64,
        exchange: f64,
        pumping: f64,
    ) -> SimulationCostResult {
        SimulationCostResult {
            stage_id: 0,
            block_id: None,
            total_cost: 0.0,
            immediate_cost: 0.0,
            future_cost: 0.0,
            discount_factor: 1.0,
            thermal_cost: thermal,
            contract_cost: contract,
            deficit_cost: deficit,
            excess_cost: excess,
            storage_violation_cost: storage_violation,
            filling_target_cost: filling,
            hydro_violation_cost: hydro_violation,
            inflow_penalty_cost: inflow_penalty,
            generic_violation_cost: generic_violation,
            spillage_cost: spillage,
            fpha_turbined_cost: fpha,
            curtailment_cost: curtailment,
            exchange_cost: exchange,
            pumping_cost: pumping,
        }
    }

    fn zero_accum() -> ScenarioCategoryCosts {
        ScenarioCategoryCosts {
            resource_cost: 0.0,
            recourse_cost: 0.0,
            violation_cost: 0.0,
            regularization_cost: 0.0,
            imputed_cost: 0.0,
        }
    }

    #[test]
    fn accumulate_single_stage_all_categories() {
        let cost = make_cost(
            400.0, 100.0, // resource: 500
            50.0, 10.0, // recourse: 60
            20.0, 30.0, 5.0, 3.0, 2.0, // violation: 60
            1.0, 4.0, 7.0, 8.0,  // regularization: 20
            60.0, // imputed: 60
        );
        let mut accum = zero_accum();
        accumulate_category_costs(&cost, &mut accum);

        assert_eq!(accum.resource_cost, 500.0);
        assert_eq!(accum.recourse_cost, 60.0);
        assert_eq!(accum.violation_cost, 60.0);
        assert_eq!(accum.regularization_cost, 20.0);
        assert_eq!(accum.imputed_cost, 60.0);
    }

    #[test]
    fn accumulate_two_consecutive_stages_sums_correctly() {
        let cost1 = make_cost(
            100.0, 0.0, // resource
            10.0, 0.0, // recourse
            0.0, 0.0, 0.0, 0.0, 0.0, // violation
            0.0, 0.0, 0.0, 0.0, // regularization
            5.0, // imputed
        );
        let cost2 = make_cost(
            200.0, 50.0, // resource
            20.0, 5.0, // recourse
            0.0, 0.0, 0.0, 0.0, 0.0, // violation
            0.0, 0.0, 0.0, 0.0,  // regularization
            10.0, // imputed
        );
        let mut accum = zero_accum();
        accumulate_category_costs(&cost1, &mut accum);
        accumulate_category_costs(&cost2, &mut accum);

        assert_eq!(accum.resource_cost, 100.0 + 200.0 + 50.0);
        assert_eq!(accum.recourse_cost, 10.0 + 20.0 + 5.0);
        assert_eq!(accum.violation_cost, 0.0);
        assert_eq!(accum.regularization_cost, 0.0);
        assert_eq!(accum.imputed_cost, 5.0 + 10.0);
    }

    #[test]
    fn accumulate_all_zeros_leaves_accum_unchanged() {
        let cost = make_cost(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        );
        let mut accum = ScenarioCategoryCosts {
            resource_cost: 1.0,
            recourse_cost: 2.0,
            violation_cost: 3.0,
            regularization_cost: 4.0,
            imputed_cost: 5.0,
        };
        accumulate_category_costs(&cost, &mut accum);

        assert_eq!(accum.resource_cost, 1.0);
        assert_eq!(accum.recourse_cost, 2.0);
        assert_eq!(accum.violation_cost, 3.0);
        assert_eq!(accum.regularization_cost, 4.0);
        assert_eq!(accum.imputed_cost, 5.0);
    }

    #[test]
    fn accumulate_violation_all_five_components() {
        let cost = make_cost(
            0.0, 0.0, // resource
            0.0, 0.0, // recourse
            1.0, 2.0, 3.0, 4.0, 5.0, // violation: 15
            0.0, 0.0, 0.0, 0.0, // regularization
            0.0, // imputed
        );
        let mut accum = zero_accum();
        accumulate_category_costs(&cost, &mut accum);

        assert_eq!(accum.violation_cost, 15.0);
    }

    #[test]
    fn accumulate_regularization_all_four_components() {
        let cost = make_cost(
            0.0, 0.0, // resource
            0.0, 0.0, // recourse
            0.0, 0.0, 0.0, 0.0, 0.0, // violation
            2.0, 3.0, 4.0, 5.0, // regularization: 14
            0.0, // imputed
        );
        let mut accum = zero_accum();
        accumulate_category_costs(&cost, &mut accum);

        assert_eq!(accum.regularization_cost, 14.0);
    }

    // ── test_slack_extraction_in_simulation ──────────────────────────────────

    /// Verify that `inflow_nonnegativity_slack_m3s` is read from the primal
    /// solution when `has_inflow_penalty == true`.
    ///
    /// Column layout for N=2 hydros, L=1 lag, T=1 thermal, Ln=1 line, B=1 bus,
    /// K=1 block, with penalty method active:
    ///
    /// theta=6, `turbine`=[7,9), `spillage`=[9,11), `thermal`=[11,12),
    /// `line_fwd`=[12,13), `line_rev`=[13,14), `deficit`=[14,15), `excess`=[15,16),
    /// `inflow_slack`=[16,18)
    #[test]
    fn test_slack_extraction_with_penalty_active() {
        // N=2, L=1, T=1, Ln=1, B=1, K=1, has_inflow_penalty=true
        let indexer = StageIndexer::with_equipment(2, 1, 1, 1, 1, 1, true, vec![], &[]);

        assert!(
            indexer.has_inflow_penalty,
            "has_inflow_penalty must be true"
        );
        assert!(
            !indexer.inflow_slack.is_empty(),
            "inflow_slack must be non-empty"
        );

        // Primal vector: base columns + inflow slack + withdrawal slack columns
        let n_cols = indexer.withdrawal_slack.end;
        let mut primal = vec![0.0_f64; n_cols];

        // Fill base values
        primal[0] = 100.0; // storage h0
        primal[1] = 200.0; // storage h1
        primal[2] = 50.0; // lag h0
        primal[3] = 60.0; // lag h1
        primal[4] = 90.0; // storage_in h0
        primal[5] = 180.0; // storage_in h1
        primal[6] = 500.0; // theta

        // Inflow slack values: hydro 0 has slack 7.5, hydro 1 has slack 0.0
        primal[indexer.inflow_slack.start] = 7.5; // slack h0
        primal[indexer.inflow_slack.start + 1] = 0.0; // slack h1

        let obj = vec![0.0_f64; n_cols];
        let dual = vec![0.0_f64; 4];
        let row_lower = vec![0.0_f64; indexer.load_balance.end.max(1)];

        let counts = EntityCounts {
            hydro_ids: vec![10, 20],
            hydro_productivities: vec![1.0, 1.0],
            thermal_ids: vec![1],
            line_ids: vec![5],
            bus_ids: vec![100],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 500.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        // turbine is non-empty → per-(hydro, block) results, so 2 hydros × 1 block = 2 entries
        assert_eq!(result.hydros.len(), 2);

        // Slack for hydro 0 must equal the primal slack column value
        assert!(
            (result.hydros[0].inflow_nonnegativity_slack_m3s - 7.5).abs() < 1e-12,
            "hydro 0 slack should be 7.5, got {}",
            result.hydros[0].inflow_nonnegativity_slack_m3s
        );

        // Slack for hydro 1 must be 0.0
        assert_eq!(
            result.hydros[1].inflow_nonnegativity_slack_m3s, 0.0,
            "hydro 1 slack should be 0.0"
        );
    }

    /// Verify that `inflow_nonnegativity_slack_m3s` is zero when the penalty
    /// method is inactive (`has_inflow_penalty == false`).
    #[test]
    fn test_slack_extraction_without_penalty_is_zero() {
        // N=2, L=1, T=1, Ln=1, B=1, K=1, has_inflow_penalty=false
        let indexer = StageIndexer::with_equipment(2, 1, 1, 1, 1, 1, false, vec![], &[]);
        assert!(
            !indexer.has_inflow_penalty,
            "has_inflow_penalty must be false"
        );

        let n_cols = indexer.withdrawal_slack.end; // includes withdrawal_slack columns
        let primal = vec![1.0_f64; n_cols]; // all ones
        let obj = vec![0.0_f64; n_cols];
        let dual = vec![0.0_f64; 4];
        let row_lower = vec![0.0_f64; indexer.load_balance.end.max(1)];

        let counts = EntityCounts {
            hydro_ids: vec![10, 20],
            hydro_productivities: vec![1.0, 1.0],
            thermal_ids: vec![1],
            line_ids: vec![5],
            bus_ids: vec![100],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 1.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        for (h, hr) in result.hydros.iter().enumerate() {
            assert_eq!(
                hr.inflow_nonnegativity_slack_m3s, 0.0,
                "hydro {h} slack must be 0.0 when penalty is inactive"
            );
        }
    }

    /// Verify that the fallback path (no equipment ranges) also reads slack
    /// when `has_inflow_penalty == true`.
    #[test]
    fn test_slack_extraction_fallback_path_with_penalty() {
        // Use StageIndexer::new (no equipment) but manually set has_inflow_penalty
        // by using with_equipment with zero blocks — turbine.is_empty() triggers fallback.
        // N=2, L=1, T=0, Ln=0, B=0, K=0, has_inflow_penalty=true
        let indexer = StageIndexer::with_equipment(2, 1, 0, 0, 0, 0, true, vec![], &[]);

        // turbine is empty (n_blks=0) → fallback path
        assert!(
            indexer.turbine.is_empty(),
            "turbine must be empty to trigger fallback"
        );
        assert!(
            indexer.has_inflow_penalty,
            "has_inflow_penalty must be true"
        );

        // Layout: storage[0..2], lags[2..4], storage_in[4..6], theta=6,
        //         inflow_slack=[7..9), withdrawal_slack=[9..11)
        let n_cols = indexer.withdrawal_slack.end;
        let mut primal = vec![0.0_f64; n_cols];
        primal[0] = 150.0; // storage h0
        primal[1] = 250.0; // storage h1
        primal[2] = 55.0; // lag h0
        primal[3] = 65.0; // lag h1
        primal[4] = 140.0; // storage_in h0
        primal[5] = 240.0; // storage_in h1
        primal[6] = 0.0; // theta
        primal[indexer.inflow_slack.start] = 3.0; // slack h0
        primal[indexer.inflow_slack.start + 1] = 0.0; // slack h1

        let obj = vec![0.0_f64; n_cols];
        let dual = vec![0.0_f64; 4];
        let row_lower = vec![0.0_f64; 1];

        let counts = EntityCounts {
            hydro_ids: vec![10, 20],
            hydro_productivities: vec![1.0, 1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 0.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        // Fallback: one entry per hydro (block_id = None)
        assert_eq!(result.hydros.len(), 2);
        assert!(
            (result.hydros[0].inflow_nonnegativity_slack_m3s - 3.0).abs() < 1e-12,
            "hydro 0 fallback slack should be 3.0, got {}",
            result.hydros[0].inflow_nonnegativity_slack_m3s
        );
        assert_eq!(result.hydros[1].inflow_nonnegativity_slack_m3s, 0.0);
    }

    // ── FPHA and Evaporation extraction tests ────────────────────────────────

    /// Build a `StageIndexer` with 2 hydros (h0 = FPHA, h1 = constant-productivity),
    /// 1 block, no thermals/lines/buses.
    ///
    /// Column layout:
    /// ```text
    /// N=2, L=0, T=0, Ln=0, B=0, K=1, penalty=false, fpha=[0], planes=[2]
    /// theta = N*(2+L) = 2*(2+0) = 4
    /// turbine:   [5, 7)   h0→5, h1→6
    /// spillage:  [7, 9)   h0→7, h1→8
    /// generation:[9, 10)  fpha h0 b0 → 9
    /// ```
    fn make_indexer_2h_1fpha_1blk() -> StageIndexer {
        // h0 is FPHA (system index 0), h1 is constant-productivity (system index 1)
        StageIndexer::with_equipment(2, 0, 0, 0, 0, 1, false, vec![0], &[2])
    }

    /// Acceptance criterion: FPHA hydro's `generation_mw` equals the LP generation
    /// variable (not turbined * productivity = 0).
    #[test]
    fn fpha_generation_read_from_lp_column() {
        let indexer = make_indexer_2h_1fpha_1blk();
        // generation.start should be at turbine(5..7) + spillage(7..9) end = 9
        // generation[0] = generation.start + 0 * 1 + 0 = 9
        assert_eq!(indexer.generation.start, 9, "generation starts at 9");
        assert_eq!(indexer.fpha_hydro_indices, vec![0]);

        let n_cols = indexer.withdrawal_slack.end;
        let mut primal = vec![0.0_f64; n_cols];
        primal[0] = 50.0; // storage h0
        primal[1] = 80.0; // storage h1
        primal[2] = 45.0; // storage_in h0
        primal[3] = 75.0; // storage_in h1
        primal[4] = 0.0; // theta
        primal[5] = 20.0; // turbine h0 b0 (not used for FPHA gen)
        primal[6] = 30.0; // turbine h1 b0
        primal[7] = 0.0; // spillage h0 b0
        primal[8] = 0.0; // spillage h1 b0
        primal[9] = 75.0; // FPHA generation h0 b0 — acceptance criterion value

        let obj = vec![0.0_f64; n_cols];
        let dual = vec![0.0_f64; 2];
        let row_lower = vec![0.0_f64; 1];

        let counts = EntityCounts {
            hydro_ids: vec![1, 2],
            hydro_productivities: vec![0.0, 1.5], // FPHA has 0.0, constant has 1.5
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 0.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        // 2 hydros × 1 block = 2 entries
        assert_eq!(result.hydros.len(), 2);

        // FPHA hydro (h0, block 0): generation from LP column 9 = 75.0
        assert!(
            (result.hydros[0].generation_mw - 75.0).abs() < 1e-12,
            "FPHA generation_mw should be 75.0, got {}",
            result.hydros[0].generation_mw
        );

        // Constant-productivity hydro (h1, block 0): generation = turbined * productivity
        // turbine h1 b0 = primal[6] = 30.0, productivity = 1.5 → 45.0
        assert!(
            (result.hydros[1].generation_mw - 45.0).abs() < 1e-12,
            "constant-productivity generation_mw should be 45.0, got {}",
            result.hydros[1].generation_mw
        );
    }

    /// Acceptance criterion: FPHA hydro has `productivity_mw_per_m3s == None`;
    /// constant-productivity hydro has `Some(rho)`.
    #[test]
    fn fpha_productivity_is_none() {
        let indexer = make_indexer_2h_1fpha_1blk();
        let n_cols = indexer.withdrawal_slack.end;
        let primal = vec![0.0_f64; n_cols];
        let obj = vec![0.0_f64; n_cols];
        let dual = vec![0.0_f64; 2];
        let row_lower = vec![0.0_f64; 1];

        let counts = EntityCounts {
            hydro_ids: vec![1, 2],
            hydro_productivities: vec![0.0, 1.5],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 0.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        assert_eq!(
            result.hydros[0].productivity_mw_per_m3s, None,
            "FPHA hydro must have productivity_mw_per_m3s == None"
        );
        assert_eq!(
            result.hydros[1].productivity_mw_per_m3s,
            Some(1.5),
            "constant-productivity hydro must have Some(1.5)"
        );
    }

    /// Build a `StageIndexer` with 1 hydro that has evaporation, 1 block.
    ///
    /// Column layout:
    /// ```text
    /// N=1, L=0, T=0, Ln=0, B=0, K=1, penalty=false, fpha=[], evap=[0]
    /// theta = 1*(2+0) = 2
    /// turbine:  [3, 4)   h0→3
    /// spillage: [4, 5)   h0→4
    /// evap:     [5, 8)   Q_ev→5, f_plus→6, f_minus→7
    /// ```
    fn make_indexer_1h_evap_1blk() -> StageIndexer {
        StageIndexer::with_equipment_and_evaporation(
            1,
            0,
            0,
            0,
            0,
            1,
            false,
            vec![],
            &[],
            vec![0],
            1,
        )
    }

    /// Acceptance criterion: `evaporation_m3s` equals the LP `Q_ev` variable value.
    #[test]
    fn evaporation_read_from_lp_column() {
        let indexer = make_indexer_1h_evap_1blk();
        assert_eq!(indexer.evap_hydro_indices, vec![0]);
        let ei = &indexer.evap_indices[0];
        assert_eq!(ei.q_ev_col, 5);
        assert_eq!(ei.f_evap_plus_col, 6);
        assert_eq!(ei.f_evap_minus_col, 7);

        let n_cols = indexer.withdrawal_slack.end;
        let mut primal = vec![0.0_f64; n_cols];
        primal[0] = 200.0; // storage h0
        primal[1] = 190.0; // storage_in h0
        primal[2] = 0.0; // theta
        primal[3] = 10.0; // turbine h0 b0
        primal[4] = 0.0; // spillage h0 b0
        primal[5] = 3.5; // Q_ev — acceptance criterion value

        let obj = vec![0.0_f64; n_cols];
        let dual = vec![0.0_f64; 1];
        let row_lower = vec![0.0_f64; 1];

        let counts = EntityCounts {
            hydro_ids: vec![1],
            hydro_productivities: vec![1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 0.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        assert_eq!(result.hydros.len(), 1);
        assert_eq!(
            result.hydros[0].evaporation_m3s,
            Some(3.5),
            "evaporation_m3s should be Some(3.5)"
        );
        assert!(
            result.hydros[0].evaporation_violation_m3s.abs() < 1e-12,
            "evaporation_violation_m3s should be 0.0"
        );
    }

    /// Acceptance criterion: `evaporation_violation_m3s` equals the sum of the two slack columns.
    #[test]
    fn evaporation_violation_is_sum_of_slacks() {
        let indexer = make_indexer_1h_evap_1blk();
        let n_cols = indexer.withdrawal_slack.end;
        let mut primal = vec![0.0_f64; n_cols];
        primal[0] = 200.0;
        primal[1] = 190.0;
        // primal[2] = theta = 0
        primal[5] = 2.0; // Q_ev
        primal[6] = 0.5; // f_evap_plus — acceptance criterion value
        primal[7] = 0.0; // f_evap_minus

        let obj = vec![0.0_f64; n_cols];
        let dual = vec![0.0_f64; 1];
        let row_lower = vec![0.0_f64; 1];

        let counts = EntityCounts {
            hydro_ids: vec![1],
            hydro_productivities: vec![1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 0.0,
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        assert!(
            (result.hydros[0].evaporation_violation_m3s - 0.5).abs() < 1e-12,
            "evaporation_violation_m3s should be 0.5, got {}",
            result.hydros[0].evaporation_violation_m3s
        );
    }

    /// Acceptance criterion: `fpha_turbined_cost` equals the sum of primal * obj\_coeff
    /// over FPHA generation columns.
    ///
    /// Setup: 1 FPHA hydro (h0), 1 constant-productivity hydro (h1), 1 block.
    /// FPHA generation column: primal=30.0, `objective_coeff`=0.01 → cost=0.3
    #[test]
    fn fpha_turbined_cost_in_compute_cost_result() {
        let indexer = make_indexer_2h_1fpha_1blk();
        // generation.start = 9 (fpha h0 b0)
        let n_cols = indexer.withdrawal_slack.end;
        let mut primal = vec![0.0_f64; n_cols];
        primal[4] = 500.0; // theta

        // FPHA generation column 9: primal=30.0
        primal[9] = 30.0;

        let mut obj = vec![0.0_f64; n_cols];
        // FPHA generation column 9: objective_coeff=0.01
        obj[9] = 0.01;

        let dual = vec![0.0_f64; 2];
        let row_lower = vec![0.0_f64; 1];

        let counts = EntityCounts {
            hydro_ids: vec![1, 2],
            hydro_productivities: vec![0.0, 1.5],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };

        let result = extract_stage_result(
            &SolutionView {
                primal: &primal,
                dual: &dual,
                objective: 500.3, // theta + fpha cost
                objective_coeffs: &obj,
                row_lower: &row_lower,
            },
            &StageExtractionSpec {
                indexer: &indexer,
                entity_counts: &counts,
                inflow_m3s_per_hydro: &[],
                block_hours: &[],
                generic_constraint_entries: &[],
            },
            0,
        );

        let cost = &result.costs[0];
        assert!(
            (cost.fpha_turbined_cost - 0.3).abs() < 1e-12,
            "fpha_turbined_cost should be 0.3 (30.0 * 0.01), got {}",
            cost.fpha_turbined_cost
        );
    }
}
