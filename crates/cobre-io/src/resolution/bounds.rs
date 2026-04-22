//! Bound resolution from base entity values plus stage-varying overrides.
//!
//! [`resolve_bounds`] pre-computes per-(entity, stage) bound values by:
//!
//! 1. Deriving base `*StageBounds` values from each entity's fields (loaded from
//!    entity JSON files).
//! 2. Filling the [`ResolvedBounds`] table with these base values for all stages.
//! 3. Applying sparse stage-varying overrides from the parsed Parquet rows.
//!
//! The result is a [`ResolvedBounds`] container ready for O(1) lookup by
//! `(entity_index, stage_index)` during LP construction.

use std::collections::HashMap;

use cobre_core::{
    EntityId,
    entities::{EnergyContract, Hydro, Line, PumpingStation, Thermal},
    resolved::{
        ContractStageBounds, HydroStageBounds, LineStageBounds, PumpingStageBounds, ResolvedBounds,
        ThermalStageBounds,
    },
};

use crate::constraints::{
    ContractBoundsRow, HydroBoundsRow, LineBoundsRow, PumpingBoundsRow, ThermalBoundsRow,
};

/// Entity slices needed for bounds resolution.
pub struct BoundsEntitySlices<'a> {
    /// Hydro plants sorted by ID.
    pub hydros: &'a [Hydro],
    /// Thermal units sorted by ID.
    pub thermals: &'a [Thermal],
    /// Transmission lines sorted by ID.
    pub lines: &'a [Line],
    /// Pumping stations sorted by ID.
    pub pumping_stations: &'a [PumpingStation],
    /// Energy contracts sorted by ID.
    pub contracts: &'a [EnergyContract],
}

/// Per-entity-type override rows for bounds resolution.
pub struct BoundsOverrides<'a> {
    /// Stage-varying overrides for hydro bounds.
    pub hydro: &'a [HydroBoundsRow],
    /// Stage-varying overrides for thermal bounds.
    pub thermal: &'a [ThermalBoundsRow],
    /// Stage-varying overrides for line bounds.
    pub line: &'a [LineBoundsRow],
    /// Stage-varying overrides for pumping bounds.
    pub pumping: &'a [PumpingBoundsRow],
    /// Stage-varying overrides for contract bounds.
    pub contract: &'a [ContractBoundsRow],
}

/// Pre-compute the full bound table from entity base values and stage-varying overrides.
///
/// Entity slices must already be sorted by ID (declaration-order invariance). This
/// function uses positional index mapping: the position of an entity in its sorted
/// slice becomes its `entity_index` in the [`ResolvedBounds`] flat array.
///
/// Override rows referencing unknown entity IDs or out-of-range stage IDs are silently
/// skipped — referential integrity is a Layer 3 concern validated in Epic 06.
///
/// # Arguments
///
/// * `entities` — entity slices grouped into [`BoundsEntitySlices`]
/// * `n_stages` — total number of study stages
/// * `stage_index` — mapping from domain-level `stage_id` to positional 0-based index
/// * `overrides` — per-entity-type override rows grouped into [`BoundsOverrides`]
///
/// # Examples
///
/// ```
/// use cobre_core::EntityId;
/// use cobre_core::entities::{
///     EnergyContract, ContractType, Hydro, HydroGenerationModel, HydroPenalties,
///     Line, PumpingStation, Thermal,
/// };
/// use cobre_io::constraints::HydroBoundsRow;
/// use cobre_io::resolution::{resolve_bounds, BoundsEntitySlices, BoundsOverrides};
///
/// let penalties = HydroPenalties {
///     spillage_cost: 0.01,
///     diversion_cost: 0.02,
///     fpha_turbined_cost: 0.03,
///     storage_violation_below_cost: 1000.0,
///     filling_target_violation_cost: 5000.0,
///     turbined_violation_below_cost: 500.0,
///     outflow_violation_below_cost: 500.0,
///     outflow_violation_above_cost: 500.0,
///     generation_violation_below_cost: 500.0,
///     evaporation_violation_cost: 500.0,
///     water_withdrawal_violation_cost: 500.0,
///     water_withdrawal_violation_pos_cost: 500.0,
///     water_withdrawal_violation_neg_cost: 500.0,
///     evaporation_violation_pos_cost: 500.0,
///     evaporation_violation_neg_cost: 500.0,
///     inflow_nonnegativity_cost: 1000.0,
/// };
///
/// let hydro = Hydro {
///     id: EntityId::from(0),
///     name: "H0".to_string(),
///     bus_id: EntityId::from(1),
///     downstream_id: None,
///     entry_stage_id: None,
///     exit_stage_id: None,
///     min_storage_hm3: 10.0,
///     max_storage_hm3: 200.0,
///     min_outflow_m3s: 0.0,
///     max_outflow_m3s: None,
///     generation_model: HydroGenerationModel::ConstantProductivity { productivity_mw_per_m3s: 1.0 },
///     min_turbined_m3s: 0.0,
///     max_turbined_m3s: 50.0,
///     min_generation_mw: 0.0,
///     max_generation_mw: 50.0,
///     tailrace: None,
///     hydraulic_losses: None,
///     efficiency: None,
///     evaporation_coefficients_mm: None,
///     evaporation_reference_volumes_hm3: None,
///     diversion: None,
///     filling: None,
///     penalties,
/// };
///
/// let override_row = HydroBoundsRow {
///     hydro_id: EntityId::from(0),
///     stage_id: 1,
///     min_storage_hm3: Some(20.0),
///     max_storage_hm3: None,
///     min_turbined_m3s: None,
///     max_turbined_m3s: None,
///     min_outflow_m3s: None,
///     max_outflow_m3s: None,
///     min_generation_mw: None,
///     max_generation_mw: None,
///     max_diversion_m3s: None,
///     filling_inflow_m3s: None,
///     water_withdrawal_m3s: None,
/// };
///
/// let stage_index: std::collections::HashMap<i32, usize> =
///     [(0, 0), (1, 1), (2, 2)].into_iter().collect();
/// let result = resolve_bounds(
///     &BoundsEntitySlices {
///         hydros: &[hydro],
///         thermals: &[],
///         lines: &[],
///         pumping_stations: &[],
///         contracts: &[],
///     },
///     3,
///     &stage_index,
///     &BoundsOverrides {
///         hydro: &[override_row],
///         thermal: &[],
///         line: &[],
///         pumping: &[],
///         contract: &[],
///     },
/// );
///
/// // Stage 0: base value.
/// assert!((result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// // Stage 1: overridden.
/// assert!((result.hydro_bounds(0, 1).min_storage_hm3 - 20.0).abs() < f64::EPSILON);
/// // Stage 2: base value.
/// assert!((result.hydro_bounds(0, 2).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// // max_storage_hm3 is unchanged across all stages.
/// assert!((result.hydro_bounds(0, 0).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
/// assert!((result.hydro_bounds(0, 1).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
/// assert!((result.hydro_bounds(0, 2).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
/// ```
#[must_use]
#[allow(clippy::too_many_lines, clippy::implicit_hasher)]
pub fn resolve_bounds(
    entities: &BoundsEntitySlices<'_>,
    n_stages: usize,
    stage_index: &HashMap<i32, usize>,
    overrides: &BoundsOverrides<'_>,
) -> ResolvedBounds {
    let hydro_overrides = overrides.hydro;
    let thermal_overrides = overrides.thermal;
    let line_overrides = overrides.line;
    let pumping_overrides = overrides.pumping;
    let contract_overrides = overrides.contract;

    let hydro_index: HashMap<EntityId, usize> = entities
        .hydros
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.id, idx))
        .collect();
    let thermal_index: HashMap<EntityId, usize> = entities
        .thermals
        .iter()
        .enumerate()
        .map(|(idx, t)| (t.id, idx))
        .collect();
    let line_index: HashMap<EntityId, usize> = entities
        .lines
        .iter()
        .enumerate()
        .map(|(idx, l)| (l.id, idx))
        .collect();
    let pumping_index: HashMap<EntityId, usize> = entities
        .pumping_stations
        .iter()
        .enumerate()
        .map(|(idx, p)| (p.id, idx))
        .collect();
    let contract_index: HashMap<EntityId, usize> = entities
        .contracts
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.id, idx))
        .collect();

    // ── Step 2: Choose representative defaults for allocation ─────────────────
    //
    // ResolvedBounds::new fills the entire flat Vec with a single repeated value.
    // Since entities can have different bound values, we use the first entity's values
    // (or a sentinel zero struct for empty slices) to satisfy the allocation API.
    // Step 3 unconditionally overwrites all cells with entity-specific base values.
    let hydro_default = entities
        .hydros
        .first()
        .map_or(zero_hydro_stage_bounds(), hydro_base_bounds);
    let thermal_default = entities.thermals.first().map_or(
        ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            cost_per_mwh: 0.0,
        },
        |t| ThermalStageBounds {
            min_generation_mw: t.min_generation_mw,
            max_generation_mw: t.max_generation_mw,
            cost_per_mwh: t.cost_per_mwh,
        },
    );
    let line_default = entities.lines.first().map_or(
        LineStageBounds {
            direct_mw: 0.0,
            reverse_mw: 0.0,
        },
        |l| LineStageBounds {
            direct_mw: l.direct_capacity_mw,
            reverse_mw: l.reverse_capacity_mw,
        },
    );
    let pumping_default = entities.pumping_stations.first().map_or(
        PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        },
        |p| PumpingStageBounds {
            min_flow_m3s: p.min_flow_m3s,
            max_flow_m3s: p.max_flow_m3s,
        },
    );
    let contract_default = entities.contracts.first().map_or(
        ContractStageBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        },
        |c| ContractStageBounds {
            min_mw: c.min_mw,
            max_mw: c.max_mw,
            price_per_mwh: c.price_per_mwh,
        },
    );

    // Handle the n_stages == 0 edge case: allocate with 1 but return immediately
    // without filling (the flat Vecs are empty when n_stages == 0 is handled below).
    let alloc_stages = if n_stages == 0 { 1 } else { n_stages };

    let mut table = ResolvedBounds::new(
        &cobre_core::BoundsCountsSpec {
            n_hydros: entities.hydros.len(),
            n_thermals: entities.thermals.len(),
            n_lines: entities.lines.len(),
            n_pumping: entities.pumping_stations.len(),
            n_contracts: entities.contracts.len(),
            n_stages: alloc_stages,
        },
        &cobre_core::BoundsDefaults {
            hydro: hydro_default,
            thermal: thermal_default,
            line: line_default,
            pumping: pumping_default,
            contract: contract_default,
        },
    );

    if n_stages == 0 {
        return table;
    }

    // ── Step 3: Fill all cells with entity base values ─────────────────────────
    //
    // Each entity may have different bound values. We cannot rely on the uniform
    // default filled by new() — iterate every entity and write its values to every
    // stage cell.

    for (entity_idx, hydro) in entities.hydros.iter().enumerate() {
        let base = hydro_base_bounds(hydro);
        for stage_idx in 0..n_stages {
            *table.hydro_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    for (entity_idx, thermal) in entities.thermals.iter().enumerate() {
        let base = ThermalStageBounds {
            min_generation_mw: thermal.min_generation_mw,
            max_generation_mw: thermal.max_generation_mw,
            cost_per_mwh: thermal.cost_per_mwh,
        };
        for stage_idx in 0..n_stages {
            *table.thermal_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    for (entity_idx, line) in entities.lines.iter().enumerate() {
        let base = LineStageBounds {
            direct_mw: line.direct_capacity_mw,
            reverse_mw: line.reverse_capacity_mw,
        };
        for stage_idx in 0..n_stages {
            *table.line_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    for (entity_idx, pumping) in entities.pumping_stations.iter().enumerate() {
        let base = PumpingStageBounds {
            min_flow_m3s: pumping.min_flow_m3s,
            max_flow_m3s: pumping.max_flow_m3s,
        };
        for stage_idx in 0..n_stages {
            *table.pumping_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    for (entity_idx, contract) in entities.contracts.iter().enumerate() {
        let base = ContractStageBounds {
            min_mw: contract.min_mw,
            max_mw: contract.max_mw,
            price_per_mwh: contract.price_per_mwh,
        };
        for stage_idx in 0..n_stages {
            *table.contract_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    // ── Step 4: Apply stage-varying overrides ──────────────────────────────────
    //
    // Override rows are sparse: only (entity_id, stage_id) pairs that differ from
    // the base value need rows. Unknown entity IDs and out-of-range stage IDs are
    // silently skipped (Layer 3 validation concern, Epic 06).

    for row in hydro_overrides {
        let Some(&entity_idx) = hydro_index.get(&row.hydro_id) else {
            continue; // Unknown entity ID — silently skip.
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue; // Unknown or out-of-range stage_id — silently skip.
        };
        let cell = table.hydro_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_storage_hm3 {
            cell.min_storage_hm3 = v;
        }
        if let Some(v) = row.max_storage_hm3 {
            cell.max_storage_hm3 = v;
        }
        if let Some(v) = row.min_turbined_m3s {
            cell.min_turbined_m3s = v;
        }
        if let Some(v) = row.max_turbined_m3s {
            cell.max_turbined_m3s = v;
        }
        if let Some(v) = row.min_outflow_m3s {
            cell.min_outflow_m3s = v;
        }
        if let Some(v) = row.max_outflow_m3s {
            cell.max_outflow_m3s = Some(v);
        }
        if let Some(v) = row.min_generation_mw {
            cell.min_generation_mw = v;
        }
        if let Some(v) = row.max_generation_mw {
            cell.max_generation_mw = v;
        }
        if let Some(v) = row.max_diversion_m3s {
            cell.max_diversion_m3s = Some(v);
        }
        if let Some(v) = row.filling_inflow_m3s {
            cell.filling_inflow_m3s = v;
        }
        if let Some(v) = row.water_withdrawal_m3s {
            cell.water_withdrawal_m3s = v;
        }
    }

    for row in thermal_overrides {
        // Rows with non-null block_id are reserved for future per-block cost support
        // (DECOMP). They are parsed but silently ignored during bounds resolution.
        if row.block_id.is_some() {
            continue;
        }
        let Some(&entity_idx) = thermal_index.get(&row.thermal_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.thermal_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_generation_mw {
            cell.min_generation_mw = v;
        }
        if let Some(v) = row.max_generation_mw {
            cell.max_generation_mw = v;
        }
        if let Some(v) = row.cost_per_mwh {
            cell.cost_per_mwh = v;
        }
    }

    for row in line_overrides {
        let Some(&entity_idx) = line_index.get(&row.line_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.line_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.direct_mw {
            cell.direct_mw = v;
        }
        if let Some(v) = row.reverse_mw {
            cell.reverse_mw = v;
        }
    }

    for row in pumping_overrides {
        let Some(&entity_idx) = pumping_index.get(&row.station_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.pumping_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_m3s {
            cell.min_flow_m3s = v;
        }
        if let Some(v) = row.max_m3s {
            cell.max_flow_m3s = v;
        }
    }

    for row in contract_overrides {
        let Some(&entity_idx) = contract_index.get(&row.contract_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.contract_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_mw {
            cell.min_mw = v;
        }
        if let Some(v) = row.max_mw {
            cell.max_mw = v;
        }
        if let Some(v) = row.price_per_mwh {
            cell.price_per_mwh = v;
        }
    }

    table
}

/// Derive the base [`HydroStageBounds`] from a `Hydro` entity's fields.
///
/// The 11 fields are mapped as follows:
/// - `min_storage_hm3` / `max_storage_hm3` — direct field copy
/// - `min_turbined_m3s` / `max_turbined_m3s` — direct field copy
/// - `min_outflow_m3s` / `max_outflow_m3s` — direct field copy (`Option<f64>` preserved)
/// - `min_generation_mw` / `max_generation_mw` — direct field copy
/// - `max_diversion_m3s` — `hydro.diversion.as_ref().map(|d| d.max_flow_m3s)`;
///   `None` when no diversion channel is configured
/// - `filling_inflow_m3s` — `hydro.filling.map_or(0.0, |f| f.filling_inflow_m3s)`;
///   defaults to `0.0` when no filling configuration is present
/// - `water_withdrawal_m3s` — always `0.0` (no per-entity default; overrides only)
#[inline]
fn hydro_base_bounds(hydro: &Hydro) -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: hydro.min_storage_hm3,
        max_storage_hm3: hydro.max_storage_hm3,
        min_turbined_m3s: hydro.min_turbined_m3s,
        max_turbined_m3s: hydro.max_turbined_m3s,
        min_outflow_m3s: hydro.min_outflow_m3s,
        max_outflow_m3s: hydro.max_outflow_m3s,
        min_generation_mw: hydro.min_generation_mw,
        max_generation_mw: hydro.max_generation_mw,
        max_diversion_m3s: hydro.diversion.as_ref().map(|d| d.max_flow_m3s),
        filling_inflow_m3s: hydro.filling.map_or(0.0, |f| f.filling_inflow_m3s),
        water_withdrawal_m3s: 0.0,
    }
}

/// Return a zero-valued [`HydroStageBounds`] sentinel used when no hydro entities exist.
#[inline]
fn zero_hydro_stage_bounds() -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 0.0,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 0.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 0.0,
        max_diversion_m3s: None,
        filling_inflow_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use cobre_core::entities::{
        ContractType, DiversionChannel, FillingConfig, HydroGenerationModel, HydroPenalties,
    };

    /// Build a 0-based consecutive stage_index map: {0→0, 1→1, …, (n-1)→(n-1)}.
    fn si(n: usize) -> HashMap<i32, usize> {
        (0..n).map(|i| (i32::try_from(i).unwrap(), i)).collect()
    }

    /// Test wrapper that passes a 0-based consecutive `stage_index` to
    /// [`super::resolve_bounds`]. Shadows the import so existing test calls
    /// work without modification.
    #[allow(clippy::too_many_arguments)]
    fn resolve_bounds(
        hydros: &[Hydro],
        thermals: &[Thermal],
        lines: &[Line],
        pumping_stations: &[PumpingStation],
        contracts: &[EnergyContract],
        n_stages: usize,
        hydro_overrides: &[HydroBoundsRow],
        thermal_overrides: &[ThermalBoundsRow],
        line_overrides: &[LineBoundsRow],
        pumping_overrides: &[PumpingBoundsRow],
        contract_overrides: &[ContractBoundsRow],
    ) -> ResolvedBounds {
        super::resolve_bounds(
            &super::BoundsEntitySlices {
                hydros,
                thermals,
                lines,
                pumping_stations,
                contracts,
            },
            n_stages,
            &si(n_stages),
            &super::BoundsOverrides {
                hydro: hydro_overrides,
                thermal: thermal_overrides,
                line: line_overrides,
                pumping: pumping_overrides,
                contract: contract_overrides,
            },
        )
    }

    // ── Entity construction helpers ───────────────────────────────────────────

    fn make_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.02,
            fpha_turbined_cost: 0.03,
            storage_violation_below_cost: 1000.0,
            filling_target_violation_cost: 5000.0,
            turbined_violation_below_cost: 500.0,
            outflow_violation_below_cost: 500.0,
            outflow_violation_above_cost: 500.0,
            generation_violation_below_cost: 500.0,
            evaporation_violation_cost: 500.0,
            water_withdrawal_violation_cost: 500.0,
            water_withdrawal_violation_pos_cost: 500.0,
            water_withdrawal_violation_neg_cost: 500.0,
            evaporation_violation_pos_cost: 500.0,
            evaporation_violation_neg_cost: 500.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(
        id: i32,
        min_storage_hm3: f64,
        max_storage_hm3: f64,
        max_outflow_m3s: Option<f64>,
        diversion: Option<DiversionChannel>,
        filling: Option<FillingConfig>,
    ) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId::from(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3,
            max_storage_hm3,
            min_outflow_m3s: 0.0,
            max_outflow_m3s,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion,
            filling,
            penalties: make_penalties(),
        }
    }

    fn make_thermal(id: i32, min_generation_mw: f64, max_generation_mw: f64) -> Thermal {
        Thermal {
            id: EntityId::from(id),
            name: format!("Thermal {id}"),
            bus_id: EntityId::from(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw,
            max_generation_mw,
            gnl_config: None,
        }
    }

    fn make_line(id: i32, direct_mw: f64, reverse_mw: f64) -> Line {
        Line {
            id: EntityId::from(id),
            name: format!("Line {id}"),
            source_bus_id: EntityId::from(1),
            target_bus_id: EntityId::from(2),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: direct_mw,
            reverse_capacity_mw: reverse_mw,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn make_pumping(id: i32, min_flow_m3s: f64, max_flow_m3s: f64) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pumping {id}"),
            bus_id: EntityId::from(1),
            source_hydro_id: EntityId::from(1),
            destination_hydro_id: EntityId::from(2),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s,
            max_flow_m3s,
        }
    }

    fn make_contract(id: i32, min_mw: f64, max_mw: f64, price_per_mwh: f64) -> EnergyContract {
        EnergyContract {
            id: EntityId::from(id),
            name: format!("Contract {id}"),
            bus_id: EntityId::from(1),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh,
            min_mw,
            max_mw,
        }
    }

    fn all_none_hydro_row(hydro_id: i32, stage_id: i32) -> HydroBoundsRow {
        HydroBoundsRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_storage_hm3: None,
            max_storage_hm3: None,
            min_outflow_m3s: None,
            max_outflow_m3s: None,
            min_generation_mw: None,
            max_generation_mw: None,
            max_diversion_m3s: None,
            filling_inflow_m3s: None,
            water_withdrawal_m3s: None,
        }
    }

    // ── Test: base values only (no overrides) ──────────────────────────────────

    /// Given 2 hydros, 2 thermals, 1 line, 1 pumping, 1 contract, 3 stages, and no
    /// overrides, all cells equal each entity's base values.
    #[test]
    fn test_base_values_no_overrides() {
        let hydros = vec![
            make_hydro(0, 10.0, 100.0, None, None, None),
            make_hydro(1, 20.0, 200.0, None, None, None),
        ];
        let thermals = vec![make_thermal(0, 0.0, 400.0), make_thermal(1, 50.0, 600.0)];
        let lines = vec![make_line(0, 1000.0, 800.0)];
        let pumpings = vec![make_pumping(0, 0.0, 50.0)];
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];

        let result = resolve_bounds(
            &hydros,
            &thermals,
            &lines,
            &pumpings,
            &contracts,
            3,
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        for stage in 0..3 {
            assert!(
                (result.hydro_bounds(0, stage).min_storage_hm3 - 10.0).abs() < f64::EPSILON,
                "hydro 0 stage {stage}: expected min_storage_hm3=10.0"
            );
            assert!(
                (result.hydro_bounds(0, stage).max_storage_hm3 - 100.0).abs() < f64::EPSILON,
                "hydro 0 stage {stage}: expected max_storage_hm3=100.0"
            );
            assert!(
                (result.hydro_bounds(1, stage).min_storage_hm3 - 20.0).abs() < f64::EPSILON,
                "hydro 1 stage {stage}: expected min_storage_hm3=20.0"
            );
            assert!(
                (result.hydro_bounds(1, stage).max_storage_hm3 - 200.0).abs() < f64::EPSILON,
                "hydro 1 stage {stage}: expected max_storage_hm3=200.0"
            );
            assert!(
                (result.thermal_bounds(0, stage).max_generation_mw - 400.0).abs() < f64::EPSILON,
                "thermal 0 stage {stage}: expected max_generation_mw=400.0"
            );
            assert!(
                (result.thermal_bounds(1, stage).min_generation_mw - 50.0).abs() < f64::EPSILON,
                "thermal 1 stage {stage}: expected min_generation_mw=50.0"
            );
            assert!(
                (result.thermal_bounds(1, stage).max_generation_mw - 600.0).abs() < f64::EPSILON,
                "thermal 1 stage {stage}: expected max_generation_mw=600.0"
            );
            assert!(
                (result.line_bounds(0, stage).direct_mw - 1000.0).abs() < f64::EPSILON,
                "line 0 stage {stage}: expected direct_mw=1000.0"
            );
            assert!(
                (result.line_bounds(0, stage).reverse_mw - 800.0).abs() < f64::EPSILON,
                "line 0 stage {stage}: expected reverse_mw=800.0"
            );
            assert!(
                (result.pumping_bounds(0, stage).max_flow_m3s - 50.0).abs() < f64::EPSILON,
                "pumping 0 stage {stage}: expected max_flow_m3s=50.0"
            );
            assert!(
                (result.contract_bounds(0, stage).max_mw - 200.0).abs() < f64::EPSILON,
                "contract 0 stage {stage}: expected max_mw=200.0"
            );
            assert!(
                (result.contract_bounds(0, stage).price_per_mwh - 80.0).abs() < f64::EPSILON,
                "contract 0 stage {stage}: expected price_per_mwh=80.0"
            );
        }
    }

    // ── Test: single hydro override ────────────────────────────────────────────

    /// Given 1 hydro, 3 stages, override min_storage_hm3 at stage 1 only.
    /// Only that cell changes; others retain base value.
    #[test]
    fn test_single_hydro_override() {
        let hydros = vec![make_hydro(0, 10.0, 200.0, None, None, None)];
        let override_row = HydroBoundsRow {
            min_storage_hm3: Some(20.0),
            ..all_none_hydro_row(0, 1)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            3,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        // Stage 0: base value.
        assert!((result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        // Stage 1: overridden.
        assert!((result.hydro_bounds(0, 1).min_storage_hm3 - 20.0).abs() < f64::EPSILON);
        // Stage 2: base value.
        assert!((result.hydro_bounds(0, 2).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        // max_storage_hm3 unchanged for all stages.
        for stage in 0..3 {
            assert!((result.hydro_bounds(0, stage).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: hydro with diversion and filling ─────────────────────────────────

    /// Given a hydro with a diversion channel and filling config, base bounds correctly
    /// derive max_diversion_m3s and filling_inflow_m3s.
    #[test]
    fn test_hydro_with_diversion_and_filling() {
        let diversion = DiversionChannel {
            downstream_id: EntityId::from(2),
            max_flow_m3s: 50.0,
        };
        let filling = FillingConfig {
            start_stage_id: 0,
            filling_inflow_m3s: 30.0,
        };
        let hydros = vec![make_hydro(
            0,
            10.0,
            100.0,
            None,
            Some(diversion),
            Some(filling),
        )];

        let result = resolve_bounds(&hydros, &[], &[], &[], &[], 2, &[], &[], &[], &[], &[]);

        let b = result.hydro_bounds(0, 0);
        assert_eq!(b.max_diversion_m3s, Some(50.0));
        assert!((b.filling_inflow_m3s - 30.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Test: hydro without diversion ─────────────────────────────────────────

    /// Given a hydro with no diversion channel and no filling config, base bounds
    /// have max_diversion_m3s = None and filling_inflow_m3s = 0.0.
    #[test]
    fn test_hydro_without_diversion() {
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, None, None)];

        let result = resolve_bounds(&hydros, &[], &[], &[], &[], 2, &[], &[], &[], &[], &[]);

        let b = result.hydro_bounds(0, 0);
        assert!(b.max_diversion_m3s.is_none());
        assert!((b.filling_inflow_m3s - 0.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Test: thermal override ─────────────────────────────────────────────────

    /// Given 2 thermals, 2 stages, override max_generation_mw for thermal_id=1 at stage=0.
    /// Only that cell changes; thermal_id=0 at all stages and thermal_id=1 at stage=1
    /// retain base values.
    #[test]
    fn test_thermal_override() {
        let thermals = vec![make_thermal(0, 0.0, 400.0), make_thermal(1, 50.0, 600.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(1),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(500.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
        );

        assert!((result.thermal_bounds(1, 0).max_generation_mw - 500.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(0, 0).max_generation_mw - 400.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(1, 1).max_generation_mw - 600.0).abs() < f64::EPSILON);
    }

    // ── Test: line override ────────────────────────────────────────────────────

    /// Given 1 line, 3 stages, override direct_mw at stage=1.
    #[test]
    fn test_line_override() {
        let lines = vec![make_line(0, 1000.0, 800.0)];
        let override_row = LineBoundsRow {
            line_id: EntityId::from(0),
            stage_id: 1,
            direct_mw: Some(750.0),
            reverse_mw: None,
        };

        let result = resolve_bounds(
            &[],
            &[],
            &lines,
            &[],
            &[],
            3,
            &[],
            &[],
            &[override_row],
            &[],
            &[],
        );

        assert!((result.line_bounds(0, 0).direct_mw - 1000.0).abs() < f64::EPSILON);
        assert!((result.line_bounds(0, 1).direct_mw - 750.0).abs() < f64::EPSILON);
        assert!((result.line_bounds(0, 2).direct_mw - 1000.0).abs() < f64::EPSILON);
        // reverse_mw unchanged for all stages.
        for stage in 0..3 {
            assert!((result.line_bounds(0, stage).reverse_mw - 800.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: pumping override ─────────────────────────────────────────────────

    /// Given 1 pumping station, 2 stages, override max_m3s (→ max_flow_m3s) at stage=0.
    /// Note: PumpingBoundsRow uses min_m3s / max_m3s column names which map to
    /// PumpingStageBounds.min_flow_m3s / max_flow_m3s respectively.
    #[test]
    fn test_pumping_override() {
        let pumpings = vec![make_pumping(0, 0.0, 50.0)];
        let override_row = PumpingBoundsRow {
            station_id: EntityId::from(0),
            stage_id: 0,
            min_m3s: None,
            max_m3s: Some(100.0),
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &pumpings,
            &[],
            2,
            &[],
            &[],
            &[],
            &[override_row],
            &[],
        );

        assert!((result.pumping_bounds(0, 0).max_flow_m3s - 100.0).abs() < f64::EPSILON);
        assert!((result.pumping_bounds(0, 1).max_flow_m3s - 50.0).abs() < f64::EPSILON);
        // min_flow_m3s unchanged.
        for stage in 0..2 {
            assert!((result.pumping_bounds(0, stage).min_flow_m3s - 0.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: contract override with price ─────────────────────────────────────

    /// Given 1 contract, 3 stages, override price_per_mwh at stage=1.
    #[test]
    fn test_contract_override_with_price() {
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];
        let override_row = ContractBoundsRow {
            contract_id: EntityId::from(0),
            stage_id: 1,
            min_mw: None,
            max_mw: None,
            price_per_mwh: Some(90.0),
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &[],
            &contracts,
            3,
            &[],
            &[],
            &[],
            &[],
            &[override_row],
        );

        assert!((result.contract_bounds(0, 0).price_per_mwh - 80.0).abs() < f64::EPSILON);
        assert!((result.contract_bounds(0, 1).price_per_mwh - 90.0).abs() < f64::EPSILON);
        assert!((result.contract_bounds(0, 2).price_per_mwh - 80.0).abs() < f64::EPSILON);
        // min_mw and max_mw unchanged.
        for stage in 0..3 {
            assert!((result.contract_bounds(0, stage).min_mw - 0.0).abs() < f64::EPSILON);
            assert!((result.contract_bounds(0, stage).max_mw - 200.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: unknown entity_id silently skipped ───────────────────────────────

    /// Given an override row with entity_id not in any registry, no panic and
    /// the result is unchanged for all known entities.
    #[test]
    fn test_unknown_entity_id_skipped() {
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, None, None)];
        let override_row = HydroBoundsRow {
            hydro_id: EntityId::from(999), // Not in registry.
            min_storage_hm3: Some(9999.0),
            ..all_none_hydro_row(999, 0)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            2,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        // Hydro 0 is unchanged.
        assert!(
            (result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON,
            "unknown entity ID must not affect known entities"
        );
    }

    // ── Test: empty entities with non-empty overrides ─────────────────────────

    /// Given 0 thermals and non-empty thermal overrides, no panic occurs and the
    /// result has an empty thermal dimension.
    #[test]
    fn test_empty_entities_no_panic() {
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: Some(10.0),
            max_generation_mw: Some(200.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &[],
            &[],
            3,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
        );

        assert_eq!(result.n_stages(), 3);
        // No thermal dimension to query — verifying no panic above is sufficient.
    }

    // ── Test: acceptance criterion — AC1: hydro min_storage_hm3 override ───────

    /// AC1: 1 hydro, min_storage_hm3=10.0, max_storage_hm3=200.0, 3 stages,
    /// override at stage_id=1 with min_storage_hm3=Some(20.0), all other fields None.
    #[test]
    fn test_ac1_hydro_storage_override() {
        let hydros = vec![make_hydro(0, 10.0, 200.0, None, None, None)];
        let override_row = HydroBoundsRow {
            min_storage_hm3: Some(20.0),
            ..all_none_hydro_row(0, 1)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            3,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        assert!((result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 1).min_storage_hm3 - 20.0).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 2).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        for stage in 0..3 {
            assert!((result.hydro_bounds(0, stage).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: acceptance criterion — AC2: thermal max_generation_mw override ───

    /// AC2: 2 thermals (max=400.0 and 600.0), 2 stages,
    /// override thermal_id=1 at stage_id=0, max_generation_mw=Some(500.0).
    #[test]
    fn test_ac2_thermal_max_generation_override() {
        let thermals = vec![make_thermal(0, 0.0, 400.0), make_thermal(1, 0.0, 600.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(1),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(500.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
        );

        assert!((result.thermal_bounds(1, 0).max_generation_mw - 500.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(0, 0).max_generation_mw - 400.0).abs() < f64::EPSILON);
    }

    // ── Test: acceptance criterion — AC3: hydro with diversion no filling ──────

    /// AC3: hydro with diversion channel (max_flow_m3s=50.0) and no filling config.
    #[test]
    fn test_ac3_hydro_diversion_no_filling() {
        let diversion = DiversionChannel {
            downstream_id: EntityId::from(2),
            max_flow_m3s: 50.0,
        };
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, Some(diversion), None)];

        let result = resolve_bounds(&hydros, &[], &[], &[], &[], 1, &[], &[], &[], &[], &[]);

        let b = result.hydro_bounds(0, 0);
        assert_eq!(b.max_diversion_m3s, Some(50.0));
        assert!((b.filling_inflow_m3s - 0.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Test: acceptance criterion — AC4: empty overrides, base values ─────────

    /// AC4: all five override slices empty, every cell equals entity base value.
    #[test]
    fn test_ac4_empty_overrides_base_values() {
        let hydros = vec![make_hydro(0, 5.0, 50.0, None, None, None)];
        let thermals = vec![make_thermal(0, 10.0, 200.0)];
        let lines = vec![make_line(0, 500.0, 300.0)];
        let pumpings = vec![make_pumping(0, 0.0, 75.0)];
        let contracts = vec![make_contract(0, 0.0, 100.0, 60.0)];

        let result = resolve_bounds(
            &hydros,
            &thermals,
            &lines,
            &pumpings,
            &contracts,
            2,
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        for stage in 0..2 {
            assert!((result.hydro_bounds(0, stage).min_storage_hm3 - 5.0).abs() < f64::EPSILON);
            assert!((result.hydro_bounds(0, stage).max_storage_hm3 - 50.0).abs() < f64::EPSILON);
            assert!(
                (result.thermal_bounds(0, stage).min_generation_mw - 10.0).abs() < f64::EPSILON
            );
            assert!(
                (result.thermal_bounds(0, stage).max_generation_mw - 200.0).abs() < f64::EPSILON
            );
            assert!((result.line_bounds(0, stage).direct_mw - 500.0).abs() < f64::EPSILON);
            assert!((result.line_bounds(0, stage).reverse_mw - 300.0).abs() < f64::EPSILON);
            assert!((result.pumping_bounds(0, stage).max_flow_m3s - 75.0).abs() < f64::EPSILON);
            assert!((result.contract_bounds(0, stage).price_per_mwh - 60.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: acceptance criterion — AC5: contract price_per_mwh override ──────

    /// AC5: contract override with price_per_mwh=Some(90.0) at stage 1.
    #[test]
    fn test_ac5_contract_price_override() {
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];
        let override_row = ContractBoundsRow {
            contract_id: EntityId::from(0),
            stage_id: 1,
            min_mw: None,
            max_mw: None,
            price_per_mwh: Some(90.0),
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &[],
            &contracts,
            3,
            &[],
            &[],
            &[],
            &[],
            &[override_row],
        );

        assert!((result.contract_bounds(0, 1).price_per_mwh - 90.0).abs() < f64::EPSILON);
        assert!((result.contract_bounds(0, 0).price_per_mwh - 80.0).abs() < f64::EPSILON);
    }

    // ── Test: water_withdrawal_m3s override can be negative ───────────────────

    /// Given a water_withdrawal override with a negative value (water addition),
    /// the resolver accepts it without validation.
    #[test]
    fn test_water_withdrawal_negative_accepted() {
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, None, None)];
        let override_row = HydroBoundsRow {
            water_withdrawal_m3s: Some(-5.0),
            ..all_none_hydro_row(0, 0)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            2,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        assert!((result.hydro_bounds(0, 0).water_withdrawal_m3s - (-5.0)).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 1).water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Tests: ticket-002 acceptance criteria ────────────────────────────────

    /// AC1 (ticket-002): cost_per_mwh override with null block_id — applied.
    /// thermal T1, stages 0 and 1, cost overrides 50.0 and 100.0.
    #[test]
    fn test_thermal_cost_override_null_block_id() {
        let thermals = vec![make_thermal(0, 0.0, 400.0)];
        let overrides = vec![
            ThermalBoundsRow {
                thermal_id: EntityId::from(0),
                stage_id: 0,
                min_generation_mw: None,
                max_generation_mw: None,
                cost_per_mwh: Some(50.0),
                block_id: None,
            },
            ThermalBoundsRow {
                thermal_id: EntityId::from(0),
                stage_id: 1,
                min_generation_mw: None,
                max_generation_mw: None,
                cost_per_mwh: Some(100.0),
                block_id: None,
            },
        ];

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            &[],
            &overrides,
            &[],
            &[],
            &[],
        );

        assert!(
            (result.thermal_bounds(0, 0).cost_per_mwh - 50.0).abs() < f64::EPSILON,
            "expected cost_per_mwh=50.0 at stage 0, got {}",
            result.thermal_bounds(0, 0).cost_per_mwh
        );
        assert!(
            (result.thermal_bounds(0, 1).cost_per_mwh - 100.0).abs() < f64::EPSILON,
            "expected cost_per_mwh=100.0 at stage 1, got {}",
            result.thermal_bounds(0, 1).cost_per_mwh
        );
    }

    /// AC2 (ticket-002): no parquet cost override — falls back to base Thermal.cost_per_mwh.
    #[test]
    fn test_thermal_cost_fallback_to_base() {
        // make_thermal uses cost_per_mwh: 50.0 in the helper.
        let thermals = vec![make_thermal(0, 0.0, 400.0)];

        let result = resolve_bounds(&[], &thermals, &[], &[], &[], 3, &[], &[], &[], &[], &[]);

        for stage in 0..3 {
            assert!(
                (result.thermal_bounds(0, stage).cost_per_mwh - 50.0).abs() < f64::EPSILON,
                "expected base cost_per_mwh=50.0 at stage {stage}"
            );
        }
    }

    /// AC3 (ticket-002): row with non-null block_id is silently ignored — no effect on bounds.
    #[test]
    fn test_thermal_cost_block_id_row_ignored() {
        // make_thermal uses cost_per_mwh: 50.0; override with block_id=1 must be ignored.
        let thermals = vec![make_thermal(0, 0.0, 400.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: None,
            cost_per_mwh: Some(999.0),
            block_id: Some(1),
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
        );

        // Cost must remain at the entity base value (50.0), not 999.0.
        assert!(
            (result.thermal_bounds(0, 0).cost_per_mwh - 50.0).abs() < f64::EPSILON,
            "row with non-null block_id must be ignored; expected 50.0, got {}",
            result.thermal_bounds(0, 0).cost_per_mwh
        );
    }

    /// AC4 (ticket-002): thermal with cost_per_mwh == 0.0, no override — resolves to 0.0.
    #[test]
    fn test_thermal_zero_base_cost_no_override() {
        let thermals = vec![Thermal {
            id: EntityId::from(0),
            name: "ZeroCost".to_string(),
            bus_id: EntityId::from(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 0.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            gnl_config: None,
        }];

        let result = resolve_bounds(&[], &thermals, &[], &[], &[], 2, &[], &[], &[], &[], &[]);

        for stage in 0..2 {
            assert!(
                result.thermal_bounds(0, stage).cost_per_mwh.abs() < f64::EPSILON,
                "expected cost_per_mwh=0.0 at stage {stage}"
            );
        }
    }
}
