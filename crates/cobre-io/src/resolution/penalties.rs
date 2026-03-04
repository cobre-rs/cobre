//! Three-tier penalty cascade resolution.
//!
//! [`resolve_penalties`] pre-computes per-(entity, stage) penalty values by applying
//! the three-tier cascade (DEC-006):
//!
//! 1. **Tier-1 (global defaults)** — already merged into entity fields during loading.
//! 2. **Tier-2 (entity overrides)** — already merged into entity fields during loading.
//! 3. **Tier-3 (stage-varying overrides)** — applied here from the parsed Parquet rows.
//!
//! By the time this function is called, tier-1 and tier-2 resolution are complete:
//! `Hydro.penalties` holds the entity-level resolved value (the result of applying the
//! entity JSON override on top of the global default). This function only needs to read
//! those already-resolved values and apply sparse stage-level overrides from Parquet.
//!
//! The result is a [`ResolvedPenalties`] table that supports O(1) lookup for any
//! `(entity_index, stage_index)` pair, ready for LP construction.

use std::collections::HashMap;

use cobre_core::{
    EntityId,
    entities::{Bus, Hydro, Line, NonControllableSource},
    resolved::{
        BusStagePenalties, HydroStagePenalties, LineStagePenalties, NcsStagePenalties,
        ResolvedPenalties,
    },
};

use crate::constraints::{
    BusPenaltyOverrideRow, HydroPenaltyOverrideRow, LinePenaltyOverrideRow, NcsPenaltyOverrideRow,
};

/// Pre-compute the full penalty table by applying the three-tier cascade.
///
/// Entity slices must already be sorted by ID (declaration-order invariance). This
/// function uses positional index mapping: the position of an entity in its sorted
/// slice becomes its `entity_index` in the [`ResolvedPenalties`] flat array.
///
/// Tier-1 (global) and tier-2 (entity) resolution are already embedded in the entity
/// struct fields by the entity parsers. This function reads those fields to initialise
/// the table and then overlays the sparse tier-3 stage overrides.
///
/// Override rows referencing unknown entity IDs or out-of-range stage IDs are silently
/// skipped — referential integrity is a Layer 3 concern validated in Epic 06.
///
/// # Arguments
///
/// * `hydros` — hydro plants sorted by ID
/// * `buses` — buses sorted by ID
/// * `lines` — transmission lines sorted by ID
/// * `ncs_sources` — non-controllable sources sorted by ID
/// * `n_stages` — total number of study stages
/// * `hydro_overrides` — tier-3 rows from `penalty_overrides_hydro.parquet`
/// * `bus_overrides` — tier-3 rows from `penalty_overrides_bus.parquet`
/// * `line_overrides` — tier-3 rows from `penalty_overrides_line.parquet`
/// * `ncs_overrides` — tier-3 rows from `penalty_overrides_ncs.parquet`
///
/// # Examples
///
/// ```
/// use cobre_core::EntityId;
/// use cobre_core::entities::{Bus, DeficitSegment, Hydro, HydroPenalties, HydroGenerationModel, Line, NonControllableSource};
/// use cobre_io::constraints::HydroPenaltyOverrideRow;
/// use cobre_io::resolution::resolve_penalties;
///
/// // Two hydros with the same entity-level spillage_cost.
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
/// };
///
/// let make_hydro = |id: i32| Hydro {
///     id: EntityId::from(id),
///     name: format!("Hydro {id}"),
///     bus_id: EntityId::from(1),
///     downstream_id: None,
///     entry_stage_id: None,
///     exit_stage_id: None,
///     min_storage_hm3: 0.0,
///     max_storage_hm3: 100.0,
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
///     diversion: None,
///     filling: None,
///     penalties,
/// };
///
/// let hydros = vec![make_hydro(0), make_hydro(1)];
/// let override_row = HydroPenaltyOverrideRow {
///     hydro_id: EntityId::from(0),
///     stage_id: 1,
///     spillage_cost: Some(0.05),
///     fpha_turbined_cost: None,
///     diversion_cost: None,
///     storage_violation_below_cost: None,
///     filling_target_violation_cost: None,
///     turbined_violation_below_cost: None,
///     outflow_violation_below_cost: None,
///     outflow_violation_above_cost: None,
///     generation_violation_below_cost: None,
///     evaporation_violation_cost: None,
///     water_withdrawal_violation_cost: None,
/// };
///
/// let result = resolve_penalties(&hydros, &[], &[], &[], 3, &[override_row], &[], &[], &[]);
///
/// // Stage 0: unchanged — entity-level value.
/// assert!((result.hydro_penalties(0, 0).spillage_cost - 0.01).abs() < f64::EPSILON);
/// // Stage 1: overridden.
/// assert!((result.hydro_penalties(0, 1).spillage_cost - 0.05).abs() < f64::EPSILON);
/// // Stage 2: unchanged — entity-level value.
/// assert!((result.hydro_penalties(0, 2).spillage_cost - 0.01).abs() < f64::EPSILON);
/// // Hydro index 1: never overridden.
/// assert!((result.hydro_penalties(1, 1).spillage_cost - 0.01).abs() < f64::EPSILON);
/// ```
#[must_use]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn resolve_penalties(
    hydros: &[Hydro],
    buses: &[Bus],
    lines: &[Line],
    ncs_sources: &[NonControllableSource],
    n_stages: usize,
    hydro_overrides: &[HydroPenaltyOverrideRow],
    bus_overrides: &[BusPenaltyOverrideRow],
    line_overrides: &[LinePenaltyOverrideRow],
    ncs_overrides: &[NcsPenaltyOverrideRow],
) -> ResolvedPenalties {
    let hydro_index: HashMap<EntityId, usize> = hydros
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.id, idx))
        .collect();
    let bus_index: HashMap<EntityId, usize> = buses
        .iter()
        .enumerate()
        .map(|(idx, b)| (b.id, idx))
        .collect();
    let line_index: HashMap<EntityId, usize> = lines
        .iter()
        .enumerate()
        .map(|(idx, l)| (l.id, idx))
        .collect();
    let ncs_index: HashMap<EntityId, usize> = ncs_sources
        .iter()
        .enumerate()
        .map(|(idx, n)| (n.id, idx))
        .collect();

    // ResolvedPenalties::new fills the entire flat Vec with a single repeated value.
    // Since entities can have different penalty values, we use arbitrary representative
    // defaults here and immediately overwrite every cell in step 3.
    //
    // We choose the first entity's values (or a sentinel zero struct for empty slices)
    // to satisfy the allocation API. Step 3 unconditionally overwrites all cells.
    let hydro_default = hydros.first().map_or(
        HydroStagePenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        },
        hydro_stage_penalties,
    );
    let bus_default = buses
        .first()
        .map_or(BusStagePenalties { excess_cost: 0.0 }, |b| {
            BusStagePenalties {
                excess_cost: b.excess_cost,
            }
        });
    let line_default = lines
        .first()
        .map_or(LineStagePenalties { exchange_cost: 0.0 }, |l| {
            LineStagePenalties {
                exchange_cost: l.exchange_cost,
            }
        });
    let ncs_default = ncs_sources.first().map_or(
        NcsStagePenalties {
            curtailment_cost: 0.0,
        },
        |n| NcsStagePenalties {
            curtailment_cost: n.curtailment_cost,
        },
    );

    // Handle the n_stages == 0 edge case: use 1 as the allocation size but
    // immediately return without filling (the flat Vecs are empty).
    let alloc_stages = if n_stages == 0 { 1 } else { n_stages };

    let mut table = ResolvedPenalties::new(
        hydros.len(),
        buses.len(),
        lines.len(),
        ncs_sources.len(),
        alloc_stages,
        hydro_default,
        bus_default,
        line_default,
        ncs_default,
    );

    if n_stages == 0 {
        // Nothing to fill — return the empty table.
        return table;
    }

    // Each entity may have different penalty values. We cannot rely on the uniform
    // default filled by new() — iterate every entity and write its values to every
    // stage cell.

    for (entity_idx, hydro) in hydros.iter().enumerate() {
        let hp = hydro_stage_penalties(hydro);
        for stage_idx in 0..n_stages {
            *table.hydro_penalties_mut(entity_idx, stage_idx) = hp;
        }
    }

    for (entity_idx, bus) in buses.iter().enumerate() {
        let bp = BusStagePenalties {
            excess_cost: bus.excess_cost,
        };
        for stage_idx in 0..n_stages {
            *table.bus_penalties_mut(entity_idx, stage_idx) = bp;
        }
    }

    for (entity_idx, line) in lines.iter().enumerate() {
        let lp = LineStagePenalties {
            exchange_cost: line.exchange_cost,
        };
        for stage_idx in 0..n_stages {
            *table.line_penalties_mut(entity_idx, stage_idx) = lp;
        }
    }

    for (entity_idx, ncs) in ncs_sources.iter().enumerate() {
        let np = NcsStagePenalties {
            curtailment_cost: ncs.curtailment_cost,
        };
        for stage_idx in 0..n_stages {
            *table.ncs_penalties_mut(entity_idx, stage_idx) = np;
        }
    }

    // Override rows are sparse: only (entity_id, stage_id) pairs that differ from
    // the entity-level value need rows. Unknown entity IDs and out-of-range stage IDs
    // are silently skipped (Layer 3 validation concern, Epic 06).

    for row in hydro_overrides {
        let Some(&entity_idx) = hydro_index.get(&row.hydro_id) else {
            continue; // Unknown entity ID — silently skip.
        };
        let Ok(stage_idx) = usize::try_from(row.stage_id) else {
            continue; // Negative stage_id — silently skip (out-of-range, Layer 3 concern).
        };
        if stage_idx >= n_stages {
            continue; // Out-of-range stage — silently skip.
        }
        let cell = table.hydro_penalties_mut(entity_idx, stage_idx);
        if let Some(v) = row.spillage_cost {
            cell.spillage_cost = v;
        }
        if let Some(v) = row.diversion_cost {
            cell.diversion_cost = v;
        }
        if let Some(v) = row.fpha_turbined_cost {
            cell.fpha_turbined_cost = v;
        }
        if let Some(v) = row.storage_violation_below_cost {
            cell.storage_violation_below_cost = v;
        }
        if let Some(v) = row.filling_target_violation_cost {
            cell.filling_target_violation_cost = v;
        }
        if let Some(v) = row.turbined_violation_below_cost {
            cell.turbined_violation_below_cost = v;
        }
        if let Some(v) = row.outflow_violation_below_cost {
            cell.outflow_violation_below_cost = v;
        }
        if let Some(v) = row.outflow_violation_above_cost {
            cell.outflow_violation_above_cost = v;
        }
        if let Some(v) = row.generation_violation_below_cost {
            cell.generation_violation_below_cost = v;
        }
        if let Some(v) = row.evaporation_violation_cost {
            cell.evaporation_violation_cost = v;
        }
        if let Some(v) = row.water_withdrawal_violation_cost {
            cell.water_withdrawal_violation_cost = v;
        }
    }

    for row in bus_overrides {
        let Some(&entity_idx) = bus_index.get(&row.bus_id) else {
            continue;
        };
        let Ok(stage_idx) = usize::try_from(row.stage_id) else {
            continue;
        };
        if stage_idx >= n_stages {
            continue;
        }
        let cell = table.bus_penalties_mut(entity_idx, stage_idx);
        if let Some(v) = row.excess_cost {
            cell.excess_cost = v;
        }
    }

    for row in line_overrides {
        let Some(&entity_idx) = line_index.get(&row.line_id) else {
            continue;
        };
        let Ok(stage_idx) = usize::try_from(row.stage_id) else {
            continue;
        };
        if stage_idx >= n_stages {
            continue;
        }
        let cell = table.line_penalties_mut(entity_idx, stage_idx);
        if let Some(v) = row.exchange_cost {
            cell.exchange_cost = v;
        }
    }

    for row in ncs_overrides {
        let Some(&entity_idx) = ncs_index.get(&row.source_id) else {
            continue;
        };
        let Ok(stage_idx) = usize::try_from(row.stage_id) else {
            continue;
        };
        if stage_idx >= n_stages {
            continue;
        }
        let cell = table.ncs_penalties_mut(entity_idx, stage_idx);
        if let Some(v) = row.curtailment_cost {
            cell.curtailment_cost = v;
        }
    }

    table
}

/// Convert a `Hydro`'s entity-level `HydroPenalties` to the `HydroStagePenalties` type.
///
/// The two types carry identical fields but are distinct types: `HydroPenalties` lives
/// on the entity struct (in `cobre-core::entities`); `HydroStagePenalties` is the
/// per-(entity, stage) cell type in the `ResolvedPenalties` table.
#[inline]
fn hydro_stage_penalties(hydro: &Hydro) -> HydroStagePenalties {
    let p = &hydro.penalties;
    HydroStagePenalties {
        spillage_cost: p.spillage_cost,
        diversion_cost: p.diversion_cost,
        fpha_turbined_cost: p.fpha_turbined_cost,
        storage_violation_below_cost: p.storage_violation_below_cost,
        filling_target_violation_cost: p.filling_target_violation_cost,
        turbined_violation_below_cost: p.turbined_violation_below_cost,
        outflow_violation_below_cost: p.outflow_violation_below_cost,
        outflow_violation_above_cost: p.outflow_violation_above_cost,
        generation_violation_below_cost: p.generation_violation_below_cost,
        evaporation_violation_cost: p.evaporation_violation_cost,
        water_withdrawal_violation_cost: p.water_withdrawal_violation_cost,
    }
}

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
        Bus, DeficitSegment, HydroGenerationModel, HydroPenalties, Line, NonControllableSource,
    };

    /// Build a `Hydro` with all 11 penalty fields set to the given value.
    fn make_hydro(id: i32, penalty_value: f64) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId::from(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 50.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: penalty_value,
                diversion_cost: penalty_value,
                fpha_turbined_cost: penalty_value,
                storage_violation_below_cost: penalty_value,
                filling_target_violation_cost: penalty_value,
                turbined_violation_below_cost: penalty_value,
                outflow_violation_below_cost: penalty_value,
                outflow_violation_above_cost: penalty_value,
                generation_violation_below_cost: penalty_value,
                evaporation_violation_cost: penalty_value,
                water_withdrawal_violation_cost: penalty_value,
            },
        }
    }

    /// Build a `Hydro` with distinct penalty values for each field.
    fn make_hydro_distinct_penalties(id: i32) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId::from(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 50.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.02,
                fpha_turbined_cost: 0.03,
                storage_violation_below_cost: 1000.0,
                filling_target_violation_cost: 5000.0,
                turbined_violation_below_cost: 500.0,
                outflow_violation_below_cost: 400.0,
                outflow_violation_above_cost: 300.0,
                generation_violation_below_cost: 200.0,
                evaporation_violation_cost: 150.0,
                water_withdrawal_violation_cost: 100.0,
            },
        }
    }

    fn make_bus(id: i32, excess_cost: f64) -> Bus {
        Bus {
            id: EntityId::from(id),
            name: format!("Bus {id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost,
        }
    }

    fn make_line(id: i32, exchange_cost: f64) -> Line {
        Line {
            id: EntityId::from(id),
            name: format!("Line {id}"),
            source_bus_id: EntityId::from(1),
            target_bus_id: EntityId::from(2),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost,
        }
    }

    fn make_ncs(id: i32, curtailment_cost: f64) -> NonControllableSource {
        NonControllableSource {
            id: EntityId::from(id),
            name: format!("NCS {id}"),
            bus_id: EntityId::from(1),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 100.0,
            curtailment_cost,
        }
    }

    fn all_none_hydro_override(hydro_id: i32, stage_id: i32) -> HydroPenaltyOverrideRow {
        HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            spillage_cost: None,
            fpha_turbined_cost: None,
            diversion_cost: None,
            storage_violation_below_cost: None,
            filling_target_violation_cost: None,
            turbined_violation_below_cost: None,
            outflow_violation_below_cost: None,
            outflow_violation_above_cost: None,
            generation_violation_below_cost: None,
            evaporation_violation_cost: None,
            water_withdrawal_violation_cost: None,
        }
    }

    // ── Test: tier-2 only, no overrides ───────────────────────────────────────

    /// Given 2 hydros with different penalties and 3 stages, with no overrides,
    /// every cell in the result equals the entity-level resolved value.
    #[test]
    fn test_tier2_only_no_overrides() {
        let hydros = vec![make_hydro(0, 0.01), make_hydro(1, 0.99)];
        let result = resolve_penalties(&hydros, &[], &[], &[], 3, &[], &[], &[], &[]);

        for stage in 0..3 {
            assert!(
                (result.hydro_penalties(0, stage).spillage_cost - 0.01).abs() < f64::EPSILON,
                "hydro 0 stage {stage}: expected spillage_cost=0.01"
            );
            assert!(
                (result.hydro_penalties(1, stage).spillage_cost - 0.99).abs() < f64::EPSILON,
                "hydro 1 stage {stage}: expected spillage_cost=0.99"
            );
        }
    }

    // ── Test: single field override ────────────────────────────────────────────

    /// Given 1 hydro and a stage override for spillage_cost at stage 1 only,
    /// only that cell is updated; all others retain the entity-level value.
    #[test]
    fn test_single_field_override() {
        let hydros = vec![make_hydro(0, 0.01)];
        let override_row = HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(0),
            stage_id: 1,
            spillage_cost: Some(0.05),
            ..all_none_hydro_override(0, 1)
        };
        let result = resolve_penalties(&hydros, &[], &[], &[], 3, &[override_row], &[], &[], &[]);

        // Stage 0: unchanged.
        assert!(
            (result.hydro_penalties(0, 0).spillage_cost - 0.01).abs() < f64::EPSILON,
            "stage 0 spillage_cost should be entity-level 0.01"
        );
        // Stage 1: overridden.
        assert!(
            (result.hydro_penalties(0, 1).spillage_cost - 0.05).abs() < f64::EPSILON,
            "stage 1 spillage_cost should be overridden to 0.05"
        );
        // Stage 2: unchanged.
        assert!(
            (result.hydro_penalties(0, 2).spillage_cost - 0.01).abs() < f64::EPSILON,
            "stage 2 spillage_cost should be entity-level 0.01"
        );
        // All other fields at stage 1 still at entity-level value.
        assert!(
            (result.hydro_penalties(0, 1).diversion_cost - 0.01).abs() < f64::EPSILON,
            "other fields should remain at entity-level"
        );
    }

    // ── Test: full 11-field hydro override ────────────────────────────────────

    /// Given a hydro override with all 11 fields Some, all 11 fields are updated.
    #[test]
    fn test_full_11_field_hydro_override() {
        let hydros = vec![make_hydro_distinct_penalties(0)];
        let override_row = HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(0),
            stage_id: 0,
            spillage_cost: Some(11.0),
            fpha_turbined_cost: Some(22.0),
            diversion_cost: Some(33.0),
            storage_violation_below_cost: Some(44.0),
            filling_target_violation_cost: Some(55.0),
            turbined_violation_below_cost: Some(66.0),
            outflow_violation_below_cost: Some(77.0),
            outflow_violation_above_cost: Some(88.0),
            generation_violation_below_cost: Some(99.0),
            evaporation_violation_cost: Some(110.0),
            water_withdrawal_violation_cost: Some(120.0),
        };

        let result = resolve_penalties(&hydros, &[], &[], &[], 2, &[override_row], &[], &[], &[]);

        let cell = result.hydro_penalties(0, 0);
        assert!((cell.spillage_cost - 11.0).abs() < f64::EPSILON);
        assert!((cell.fpha_turbined_cost - 22.0).abs() < f64::EPSILON);
        assert!((cell.diversion_cost - 33.0).abs() < f64::EPSILON);
        assert!((cell.storage_violation_below_cost - 44.0).abs() < f64::EPSILON);
        assert!((cell.filling_target_violation_cost - 55.0).abs() < f64::EPSILON);
        assert!((cell.turbined_violation_below_cost - 66.0).abs() < f64::EPSILON);
        assert!((cell.outflow_violation_below_cost - 77.0).abs() < f64::EPSILON);
        assert!((cell.outflow_violation_above_cost - 88.0).abs() < f64::EPSILON);
        assert!((cell.generation_violation_below_cost - 99.0).abs() < f64::EPSILON);
        assert!((cell.evaporation_violation_cost - 110.0).abs() < f64::EPSILON);
        assert!((cell.water_withdrawal_violation_cost - 120.0).abs() < f64::EPSILON);

        // Stage 1 is unchanged.
        let cell1 = result.hydro_penalties(0, 1);
        assert!((cell1.spillage_cost - 0.01).abs() < f64::EPSILON);
        assert!((cell1.filling_target_violation_cost - 5000.0).abs() < f64::EPSILON);
    }

    // ── Test: partial override (mix of Some/None) ─────────────────────────────

    /// Given a hydro override with 3 of 11 fields Some and 8 None,
    /// only the 3 fields are updated; the other 8 retain their entity-level values.
    #[test]
    fn test_partial_override_mix_some_none() {
        let hydros = vec![make_hydro_distinct_penalties(0)];
        let override_row = HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(0),
            stage_id: 2,
            spillage_cost: Some(9.0),
            storage_violation_below_cost: Some(9999.0),
            filling_target_violation_cost: Some(99999.0),
            ..all_none_hydro_override(0, 2)
        };

        let result = resolve_penalties(&hydros, &[], &[], &[], 3, &[override_row], &[], &[], &[]);

        let cell = result.hydro_penalties(0, 2);
        // Updated fields.
        assert!((cell.spillage_cost - 9.0).abs() < f64::EPSILON);
        assert!((cell.storage_violation_below_cost - 9999.0).abs() < f64::EPSILON);
        assert!((cell.filling_target_violation_cost - 99999.0).abs() < f64::EPSILON);

        // Unchanged fields retain entity-level values from make_hydro_distinct_penalties.
        assert!((cell.diversion_cost - 0.02).abs() < f64::EPSILON);
        assert!((cell.fpha_turbined_cost - 0.03).abs() < f64::EPSILON);
        assert!((cell.turbined_violation_below_cost - 500.0).abs() < f64::EPSILON);
        assert!((cell.outflow_violation_below_cost - 400.0).abs() < f64::EPSILON);
        assert!((cell.outflow_violation_above_cost - 300.0).abs() < f64::EPSILON);
        assert!((cell.generation_violation_below_cost - 200.0).abs() < f64::EPSILON);
        assert!((cell.evaporation_violation_cost - 150.0).abs() < f64::EPSILON);
        assert!((cell.water_withdrawal_violation_cost - 100.0).abs() < f64::EPSILON);
    }

    // ── Test: bus override ─────────────────────────────────────────────────────

    /// Given 1 bus with excess_cost=100.0 and an override at stage 0 to 250.0,
    /// stage 0 is updated and stage 1 retains 100.0.
    #[test]
    fn test_bus_override() {
        let buses = vec![make_bus(0, 100.0)];
        let override_row = BusPenaltyOverrideRow {
            bus_id: EntityId::from(0),
            stage_id: 0,
            excess_cost: Some(250.0),
        };

        let result = resolve_penalties(&[], &buses, &[], &[], 2, &[], &[override_row], &[], &[]);

        assert!(
            (result.bus_penalties(0, 0).excess_cost - 250.0).abs() < f64::EPSILON,
            "stage 0 should be overridden to 250.0"
        );
        assert!(
            (result.bus_penalties(0, 1).excess_cost - 100.0).abs() < f64::EPSILON,
            "stage 1 should retain entity-level 100.0"
        );
    }

    // ── Test: line override ────────────────────────────────────────────────────

    /// Given 1 line with exchange_cost=5.0 and an override at stage 1 to 50.0,
    /// stage 1 is updated and stage 0 retains 5.0.
    #[test]
    fn test_line_override() {
        let lines = vec![make_line(0, 5.0)];
        let override_row = LinePenaltyOverrideRow {
            line_id: EntityId::from(0),
            stage_id: 1,
            exchange_cost: Some(50.0),
        };

        let result = resolve_penalties(&[], &[], &lines, &[], 3, &[], &[], &[override_row], &[]);

        assert!(
            (result.line_penalties(0, 0).exchange_cost - 5.0).abs() < f64::EPSILON,
            "stage 0 should retain entity-level 5.0"
        );
        assert!(
            (result.line_penalties(0, 1).exchange_cost - 50.0).abs() < f64::EPSILON,
            "stage 1 should be overridden to 50.0"
        );
        assert!(
            (result.line_penalties(0, 2).exchange_cost - 5.0).abs() < f64::EPSILON,
            "stage 2 should retain entity-level 5.0"
        );
    }

    // ── Test: NCS override ─────────────────────────────────────────────────────

    /// Given 1 NCS with curtailment_cost=10.0 and an override at stage 2 to 999.0.
    #[test]
    fn test_ncs_override() {
        let ncs_sources = vec![make_ncs(0, 10.0)];
        let override_row = NcsPenaltyOverrideRow {
            source_id: EntityId::from(0),
            stage_id: 2,
            curtailment_cost: Some(999.0),
        };

        let result = resolve_penalties(
            &[],
            &[],
            &[],
            &ncs_sources,
            4,
            &[],
            &[],
            &[],
            &[override_row],
        );

        assert!(
            (result.ncs_penalties(0, 0).curtailment_cost - 10.0).abs() < f64::EPSILON,
            "stage 0 should retain entity-level 10.0"
        );
        assert!(
            (result.ncs_penalties(0, 2).curtailment_cost - 999.0).abs() < f64::EPSILON,
            "stage 2 should be overridden to 999.0"
        );
        assert!(
            (result.ncs_penalties(0, 3).curtailment_cost - 10.0).abs() < f64::EPSILON,
            "stage 3 should retain entity-level 10.0"
        );
    }

    // ── Test: unknown entity_id silently skipped ───────────────────────────────

    /// Given an override row with entity_id not in any registry, no panic and
    /// the result is unchanged for all known entities.
    #[test]
    fn test_unknown_entity_id_skipped() {
        let hydros = vec![make_hydro(0, 0.01)];
        let override_row = HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(999), // Not in registry.
            stage_id: 0,
            spillage_cost: Some(9999.0),
            ..all_none_hydro_override(999, 0)
        };

        // Should not panic.
        let result = resolve_penalties(&hydros, &[], &[], &[], 2, &[override_row], &[], &[], &[]);

        // Hydro 0 is unchanged.
        assert!(
            (result.hydro_penalties(0, 0).spillage_cost - 0.01).abs() < f64::EPSILON,
            "unknown entity ID must not affect known entities"
        );
    }

    // ── Test: multiple entities, multiple stages, multiple overrides ───────────

    /// Given 3 hydros, 5 stages, and 4 override rows targeting 4 distinct
    /// (hydro, stage) cells, verify correct cells are updated.
    #[test]
    fn test_multiple_entities_multiple_stages() {
        let hydros = vec![
            make_hydro(0, 0.10),
            make_hydro(1, 0.20),
            make_hydro(2, 0.30),
        ];

        let overrides = vec![
            HydroPenaltyOverrideRow {
                hydro_id: EntityId::from(0),
                stage_id: 0,
                spillage_cost: Some(1.0),
                ..all_none_hydro_override(0, 0)
            },
            HydroPenaltyOverrideRow {
                hydro_id: EntityId::from(1),
                stage_id: 2,
                spillage_cost: Some(2.0),
                ..all_none_hydro_override(1, 2)
            },
            HydroPenaltyOverrideRow {
                hydro_id: EntityId::from(2),
                stage_id: 4,
                spillage_cost: Some(3.0),
                ..all_none_hydro_override(2, 4)
            },
            HydroPenaltyOverrideRow {
                hydro_id: EntityId::from(0),
                stage_id: 3,
                spillage_cost: Some(4.0),
                ..all_none_hydro_override(0, 3)
            },
        ];

        let result = resolve_penalties(&hydros, &[], &[], &[], 5, &overrides, &[], &[], &[]);

        // Overridden cells.
        assert!((result.hydro_penalties(0, 0).spillage_cost - 1.0).abs() < f64::EPSILON);
        assert!((result.hydro_penalties(1, 2).spillage_cost - 2.0).abs() < f64::EPSILON);
        assert!((result.hydro_penalties(2, 4).spillage_cost - 3.0).abs() < f64::EPSILON);
        assert!((result.hydro_penalties(0, 3).spillage_cost - 4.0).abs() < f64::EPSILON);

        // Unchanged cells.
        assert!((result.hydro_penalties(0, 1).spillage_cost - 0.10).abs() < f64::EPSILON);
        assert!((result.hydro_penalties(1, 0).spillage_cost - 0.20).abs() < f64::EPSILON);
        assert!((result.hydro_penalties(2, 0).spillage_cost - 0.30).abs() < f64::EPSILON);
    }

    // ── Test: empty entity slices ──────────────────────────────────────────────

    /// Given 0 hydros and non-empty hydro overrides, no panic occurs and the
    /// result has an empty hydro dimension.
    #[test]
    fn test_empty_entities_no_panic() {
        let overrides = vec![HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(0),
            stage_id: 0,
            spillage_cost: Some(1.0),
            ..all_none_hydro_override(0, 0)
        }];

        // Should not panic.
        let result = resolve_penalties(&[], &[], &[], &[], 3, &overrides, &[], &[], &[]);
        assert_eq!(result.n_stages(), 3);
        // No hydro dimension to query — just verifying no panic above is sufficient.
    }

    // ── Test: empty override collections ──────────────────────────────────────

    /// Given 2 hydros, 3 stages, and all four override slices empty, every cell
    /// equals the entity-level resolved value.
    #[test]
    fn test_empty_overrides_entity_level_values() {
        let hydros = vec![make_hydro(0, 0.01), make_hydro(1, 0.99)];

        let result = resolve_penalties(&hydros, &[], &[], &[], 3, &[], &[], &[], &[]);

        for stage in 0..3 {
            assert!((result.hydro_penalties(0, stage).spillage_cost - 0.01).abs() < f64::EPSILON);
            assert!((result.hydro_penalties(1, stage).spillage_cost - 0.99).abs() < f64::EPSILON);
        }
    }

    // ── Test: acceptance criterion — hydro spillage_cost ─────────────────────

    /// Acceptance criterion from ticket: 2 hydros, 3 stages, override row for
    /// hydro_id=0, stage_id=1, spillage_cost=Some(0.05). Verifies exact cells.
    #[test]
    fn test_ac_hydro_spillage_cost_override() {
        let hydros = vec![make_hydro(0, 0.01), make_hydro(1, 0.01)];
        let override_row = HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(0),
            stage_id: 1,
            spillage_cost: Some(0.05),
            ..all_none_hydro_override(0, 1)
        };

        let result = resolve_penalties(&hydros, &[], &[], &[], 3, &[override_row], &[], &[], &[]);

        // hydro_id=0, stage=0: entity-level.
        assert!((result.hydro_penalties(0, 0).spillage_cost - 0.01).abs() < f64::EPSILON);
        // hydro_id=0, stage=1: overridden.
        assert!((result.hydro_penalties(0, 1).spillage_cost - 0.05).abs() < f64::EPSILON);
        // hydro_id=0, stage=2: entity-level.
        assert!((result.hydro_penalties(0, 2).spillage_cost - 0.01).abs() < f64::EPSILON);
        // hydro_id=1, stage=1: entity-level (never overridden).
        assert!((result.hydro_penalties(1, 1).spillage_cost - 0.01).abs() < f64::EPSILON);
    }

    // ── Test: acceptance criterion — bus excess_cost ───────────────────────────

    /// Acceptance criterion from ticket: 1 bus with excess_cost=100.0, 2 stages,
    /// override at stage 0 to 250.0. Verifies exact cells.
    #[test]
    fn test_ac_bus_excess_cost_override() {
        let buses = vec![make_bus(0, 100.0)];
        let override_row = BusPenaltyOverrideRow {
            bus_id: EntityId::from(0),
            stage_id: 0,
            excess_cost: Some(250.0),
        };

        let result = resolve_penalties(&[], &buses, &[], &[], 2, &[], &[override_row], &[], &[]);

        assert!((result.bus_penalties(0, 0).excess_cost - 250.0).abs() < f64::EPSILON);
        assert!((result.bus_penalties(0, 1).excess_cost - 100.0).abs() < f64::EPSILON);
    }

    // ── Test: acceptance criterion — filling_target_violation_cost only ────────

    /// Acceptance criterion from ticket: a hydro override where only
    /// filling_target_violation_cost is Some(99999.0) and all other 10 fields
    /// are None. Verifies only that field is updated.
    #[test]
    fn test_ac_single_field_filling_target_violation_cost() {
        let hydros = vec![make_hydro_distinct_penalties(0)];
        let override_row = HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(0),
            stage_id: 0,
            filling_target_violation_cost: Some(99999.0),
            ..all_none_hydro_override(0, 0)
        };

        let result = resolve_penalties(&hydros, &[], &[], &[], 2, &[override_row], &[], &[], &[]);

        let cell = result.hydro_penalties(0, 0);
        // Only filling_target_violation_cost updated.
        assert!((cell.filling_target_violation_cost - 99999.0).abs() < f64::EPSILON);
        // All other 10 fields retain entity-level values from make_hydro_distinct_penalties.
        assert!((cell.spillage_cost - 0.01).abs() < f64::EPSILON);
        assert!((cell.diversion_cost - 0.02).abs() < f64::EPSILON);
        assert!((cell.fpha_turbined_cost - 0.03).abs() < f64::EPSILON);
        assert!((cell.storage_violation_below_cost - 1000.0).abs() < f64::EPSILON);
        assert!((cell.turbined_violation_below_cost - 500.0).abs() < f64::EPSILON);
        assert!((cell.outflow_violation_below_cost - 400.0).abs() < f64::EPSILON);
        assert!((cell.outflow_violation_above_cost - 300.0).abs() < f64::EPSILON);
        assert!((cell.generation_violation_below_cost - 200.0).abs() < f64::EPSILON);
        assert!((cell.evaporation_violation_cost - 150.0).abs() < f64::EPSILON);
        assert!((cell.water_withdrawal_violation_cost - 100.0).abs() < f64::EPSILON);
    }
}
