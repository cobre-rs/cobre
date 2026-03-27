//! Variable reference to LP column index mapping for generic constraints.
//!
//! This module provides [`resolve_variable_ref`], which maps a [`VariableRef`]
//! and block index to a list of `(column_index, coefficient_multiplier)` pairs.
//! The LP builder (ticket-004) calls this function for each [`cobre_core::LinearTerm`] in a
//! generic constraint expression to produce the CSC matrix entries.
//!
//! ## Column index arithmetic
//!
//! All column offsets follow the layout defined in [`StageIndexer`]:
//!
//! - Block-level variables (turbine, spillage, thermal, line, deficit, excess)
//!   use `col_start + entity_pos * n_blks + block_idx`.
//! - Stage-level variables (storage, evaporation, withdrawal) use `col_start + entity_pos`.
//! - FPHA generation uses `generation.start + fpha_local_idx * n_blks + block_idx`.
//!
//! ## Block expansion
//!
//! When `block_id = None` for a block-level variable, the function returns the
//! column for the *current* `block_idx` rather than expanding to all blocks.
//! The caller iterates over blocks and calls this function once per block, so
//! the per-block expansion happens in the caller loop, not here.
//!
//! ## Stub entities
//!
//! Variables that reference entity types with no LP columns (pumping stations,
//! contracts, non-controllable sources) return an empty vec. This is consistent
//! with the convention that the constraint term has no LP effect for those
//! entity types.

use std::collections::HashMap;
use std::hash::BuildHasher;

use cobre_core::{EntityId, VariableRef};

use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
use crate::indexer::StageIndexer;

/// Position maps for entity types, mapping entity IDs to their index in
/// the system's entity arrays.
///
/// Used by [`resolve_variable_ref`] to translate `VariableRef` entity IDs
/// into LP column offsets.
pub(crate) struct EntityPositionMaps<'a, S: BuildHasher = std::hash::RandomState> {
    /// Hydro plant ID to position index.
    pub hydro: &'a HashMap<EntityId, usize, S>,
    /// Thermal unit ID to position index.
    pub thermal: &'a HashMap<EntityId, usize, S>,
    /// Bus ID to position index.
    pub bus: &'a HashMap<EntityId, usize, S>,
    /// Line ID to position index.
    pub line: &'a HashMap<EntityId, usize, S>,
}

/// Map a [`VariableRef`] and block index to LP column indices with multipliers.
///
/// Returns a `Vec<(column_index, coefficient_multiplier)>`. The caller scales
/// each entry by the `LinearTerm::coefficient` to get the final CSC value.
///
/// # Arguments
///
/// - `var_ref` — the LP variable being referenced.
/// - `block_idx` — the block being built (0-indexed). For stage-level variables
///   this is ignored; for block-level variables with `block_id = Some(b)` the
///   function returns the column for block `b` regardless of `block_idx`.
/// - `n_blks` — number of operating blocks in this stage.
/// - `stage_idx` — stage index used to look up per-stage production models.
/// - `indexer` — column layout for the current stage LP.
/// - `production_models` — resolved production model set, used to distinguish
///   FPHA hydros from constant-productivity hydros for `HydroGeneration`.
/// - `positions` — entity position maps grouped into [`EntityPositionMaps`].
///
/// # Returns
///
/// An empty vec when:
/// - The entity ID is not found in the relevant position map (should have been
///   caught by referential validation, but this is defense-in-depth).
/// - The variable type references a stub entity with no LP columns (pumping
///   stations, contracts, non-controllable sources, diversion, withdrawal).
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn resolve_variable_ref<S: BuildHasher>(
    var_ref: &VariableRef,
    block_idx: usize,
    n_blks: usize,
    stage_idx: usize,
    indexer: &StageIndexer,
    production_models: &ProductionModelSet,
    positions: &EntityPositionMaps<'_, S>,
) -> Vec<(usize, f64)> {
    let hydro_pos = positions.hydro;
    let thermal_pos = positions.thermal;
    let bus_pos = positions.bus;
    let line_pos = positions.line;
    match var_ref {
        // ── Stage-level hydro variables ────────────────────────────────────
        VariableRef::HydroStorage { hydro_id } => {
            // Outgoing storage column (stage-level, not per-block).
            // indexer.storage[h] = storage.start + h
            if let Some(&pos) = hydro_pos.get(hydro_id) {
                vec![(indexer.storage.start + pos, 1.0)]
            } else {
                vec![]
            }
        }

        VariableRef::HydroEvaporation { hydro_id } => {
            // Maps to the Q_ev column for the matching evaporation hydro.
            // The evaporation hydro list uses a local index; we must find
            // the local index by matching the system-level hydro position.
            if let Some(&sys_pos) = hydro_pos.get(hydro_id) {
                if let Some(local_idx) = indexer
                    .evap_hydro_indices
                    .iter()
                    .position(|&p| p == sys_pos)
                {
                    let q_ev_col = indexer.evap_indices[local_idx].q_ev_col;
                    vec![(q_ev_col, 1.0)]
                } else {
                    // Hydro exists but has no linearized evaporation at this stage.
                    vec![]
                }
            } else {
                vec![]
            }
        }

        // ── Block-level hydro variables ────────────────────────────────────
        VariableRef::HydroTurbined { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.turbine.start,
            hydro_pos,
            1.0,
        ),

        VariableRef::HydroSpillage { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.spillage.start,
            hydro_pos,
            1.0,
        ),

        VariableRef::HydroOutflow { hydro_id, block_id } => {
            // HydroOutflow expands to turbine + spillage (both with coefficient 1.0).
            let mut result = Vec::with_capacity(2);
            result.extend(resolve_block_variable(
                *hydro_id,
                *block_id,
                block_idx,
                n_blks,
                indexer.turbine.start,
                hydro_pos,
                1.0,
            ));
            result.extend(resolve_block_variable(
                *hydro_id,
                *block_id,
                block_idx,
                n_blks,
                indexer.spillage.start,
                hydro_pos,
                1.0,
            ));
            result
        }

        VariableRef::HydroGeneration { hydro_id, block_id } => {
            // Dispatch depends on the production model for this hydro at this stage.
            // - FPHA hydros: maps to the generation column at fpha_local_idx * n_blks + blk.
            // - Constant-productivity hydros: generation = productivity * turbined,
            //   so map to turbine column with the productivity as the multiplier.
            if let Some(&sys_pos) = hydro_pos.get(hydro_id) {
                match production_models.model(sys_pos, stage_idx) {
                    ResolvedProductionModel::Fpha { .. } => {
                        // Find this hydro's FPHA local index.
                        if let Some(fpha_local_idx) = indexer
                            .fpha_hydro_indices
                            .iter()
                            .position(|&p| p == sys_pos)
                        {
                            let effective_blk = block_id.unwrap_or(block_idx);
                            let col =
                                indexer.generation.start + fpha_local_idx * n_blks + effective_blk;
                            vec![(col, 1.0)]
                        } else {
                            // Should not happen if indexer and production_models are consistent.
                            vec![]
                        }
                    }
                    ResolvedProductionModel::ConstantProductivity { productivity } => {
                        // generation = productivity * turbined → map to turbine column.
                        resolve_block_variable(
                            *hydro_id,
                            *block_id,
                            block_idx,
                            n_blks,
                            indexer.turbine.start,
                            hydro_pos,
                            *productivity,
                        )
                    }
                }
            } else {
                vec![]
            }
        }

        // ── Thermal ────────────────────────────────────────────────────────
        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => resolve_block_variable(
            *thermal_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.thermal.start,
            thermal_pos,
            1.0,
        ),

        // ── Transmission lines ─────────────────────────────────────────────
        VariableRef::LineDirect { line_id, block_id } => resolve_block_variable(
            *line_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.line_fwd.start,
            line_pos,
            1.0,
        ),

        VariableRef::LineReverse { line_id, block_id } => resolve_block_variable(
            *line_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.line_rev.start,
            line_pos,
            1.0,
        ),

        VariableRef::LineExchange { line_id, block_id } => {
            // Net exchange = forward - reverse: return two entries with opposite signs.
            if let Some(&pos) = line_pos.get(line_id) {
                let effective_blk = block_id.unwrap_or(block_idx);
                let fwd_col = indexer.line_fwd.start + pos * n_blks + effective_blk;
                let rev_col = indexer.line_rev.start + pos * n_blks + effective_blk;
                vec![(fwd_col, 1.0), (rev_col, -1.0)]
            } else {
                vec![]
            }
        }

        // ── Bus deficit / excess ───────────────────────────────────────────
        VariableRef::BusDeficit { bus_id, block_id } => {
            // Deficit expands over all S segments for the bus.
            // Column layout: deficit.start + b_pos * S * n_blks + seg * n_blks + blk
            if let Some(&b_pos) = bus_pos.get(bus_id) {
                let effective_blk = block_id.unwrap_or(block_idx);
                let s = indexer.max_deficit_segments;
                let base = indexer.deficit.start + b_pos * s * n_blks + effective_blk;
                // Return one entry per segment (each with coefficient 1.0).
                (0..s).map(|seg| (base + seg * n_blks, 1.0)).collect()
            } else {
                vec![]
            }
        }

        VariableRef::BusExcess { bus_id, block_id } => resolve_block_variable(
            *bus_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.excess.start,
            bus_pos,
            1.0,
        ),

        VariableRef::HydroDiversion { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.diversion.start,
            hydro_pos,
            1.0,
        ),

        // ── Stub entities with no LP columns ──────────────────────────────
        // The following entity types are registered in the data model but do not
        // contribute LP decision variables in this implementation:
        // - HydroWithdrawal: withdrawal is a schedule fixed by bounds, not a
        //   decision variable.
        // - PumpingFlow, PumpingPower: pumping stations are NO-OP stubs.
        // - ContractImport, ContractExport: contracts are NO-OP stubs.
        // - NonControllableGeneration, NonControllableCurtailment: non-controllable
        //   sources are NO-OP stubs.
        VariableRef::HydroWithdrawal { .. }
        | VariableRef::PumpingFlow { .. }
        | VariableRef::PumpingPower { .. }
        | VariableRef::ContractImport { .. }
        | VariableRef::ContractExport { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. } => vec![],
    }
}

/// Resolve a block-level LP variable to a `(column_index, multiplier)` pair.
///
/// Computes `col_start + entity_pos * n_blks + effective_block_idx` where
/// `effective_block_idx` is `block_idx` when `ref_block_id` is `None`, or
/// `b` when `ref_block_id` is `Some(b)`.
///
/// Returns an empty vec if the entity ID is not found in `pos_map`.
fn resolve_block_variable<S: BuildHasher>(
    entity_id: EntityId,
    ref_block_id: Option<usize>,
    current_block_idx: usize,
    n_blks: usize,
    col_start: usize,
    pos_map: &HashMap<EntityId, usize, S>,
    multiplier: f64,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = pos_map.get(&entity_id) {
        let effective_blk = ref_block_id.unwrap_or(current_block_idx);
        vec![(col_start + pos * n_blks + effective_blk, multiplier)]
    } else {
        vec![]
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::identity_op,
    clippy::erasing_op
)]
mod tests {
    use std::collections::HashMap;

    use cobre_core::{EntityId, VariableRef};

    use super::resolve_variable_ref;
    use crate::hydro_models::{FphaPlane, ProductionModelSet, ResolvedProductionModel};
    use crate::indexer::StageIndexer;

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Build a `StageIndexer` with equipment for tests.
    ///
    /// N=4 hydros (2 FPHA at positions 0, 2), L=0, T=2 thermals, Ln=1 line, B=2 buses, K=3 blocks.
    /// S=2 max deficit segments.
    ///
    /// Column layout (NEW: z_inflow at [N*(1+L), N*(2+L)) shifts storage_in and theta by +N):
    ///   storage:   [0, 4)         = 0..4
    ///   lags:      [4, 4*(1+0))   = 4..4   (L=0, empty)
    ///   z_inflow:  [4*(1+0), 4*(2+0)) = 4..8
    ///   storage_in:[4*(2+0), 4*(3+0)) = 8..12
    ///   theta = N*(3+L) = 4*(3+0) = 12
    ///   decision_start = 13
    ///   turbine:    [13, 13+4*3) = 13..25   (4 hydros * 3 blocks)
    ///   spillage:   [25, 25+4*3) = 25..37
    ///   diversion:  [37, 37+4*3) = 37..49  (4 hydros * 3 blocks)
    ///   thermal:    [49, 49+2*3) = 49..55  (2 thermals * 3 blocks)
    ///   line_fwd:   [55, 55+1*3) = 55..58  (1 line * 3 blocks)
    ///   line_rev:   [58, 58+1*3) = 58..61
    ///   deficit:    [61, 61+2*2*3) = 61..73 (2 buses * 2 segs * 3 blocks)
    ///   excess:     [73, 73+2*3) = 73..79  (2 buses * 3 blocks)
    ///   generation: [79, 79+2*3) = 79..85  (2 FPHA hydros * 3 blocks)
    ///   evap: none
    ///   withdrawal_slack: [85, 89) (4 hydros, since hydro_count > 0)
    ///
    /// Storage: 0..4
    fn make_indexer() -> StageIndexer {
        // N=4, L=0, T=2, Ln=1, B=2, K=3, no penalty, 2 FPHA hydros at positions 0 and 2
        // (local FPHA indices 0 and 1), each with 3 planes.
        StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: 4,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 1,
                n_buses: 2,
                n_blks: 3,
                has_inflow_penalty: false,
                max_deficit_segments: 2,
            },
            &crate::indexer::FphaConfig {
                hydro_indices: vec![0, 2],
                planes_per_hydro: vec![3, 3],
            },
            &crate::indexer::EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// Build a `ProductionModelSet` for 4 hydros and 2 stages.
    ///
    /// - Hydro 0: FPHA at all stages
    /// - Hydro 1: ConstantProductivity(2.5) at all stages
    /// - Hydro 2: FPHA at all stages
    /// - Hydro 3: ConstantProductivity(1.0) at all stages
    fn make_production_models() -> ProductionModelSet {
        let fpha_plane = FphaPlane {
            intercept: 0.0,
            gamma_v: 0.1,
            gamma_q: 0.5,
            gamma_s: 0.0,
        };
        let fpha_model = || ResolvedProductionModel::Fpha {
            planes: vec![fpha_plane],
            turbined_cost: 0.0,
        };
        let models: Vec<Vec<ResolvedProductionModel>> = vec![
            vec![fpha_model(), fpha_model()], // hydro 0 — FPHA
            vec![
                ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
                ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
            ], // hydro 1 — constant
            vec![fpha_model(), fpha_model()], // hydro 2 — FPHA
            vec![
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ], // hydro 3 — constant
        ];
        ProductionModelSet::new(models, 4, 2)
    }

    fn make_hydro_pos() -> HashMap<EntityId, usize> {
        // Hydros with EntityId 10, 20, 30, 40 at system positions 0, 1, 2, 3
        [
            (EntityId(10), 0),
            (EntityId(20), 1),
            (EntityId(30), 2),
            (EntityId(40), 3),
        ]
        .into_iter()
        .collect()
    }

    fn make_thermal_pos() -> HashMap<EntityId, usize> {
        // Thermals with EntityId 5 and 6 at positions 0 and 1
        [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect()
    }

    fn make_bus_pos() -> HashMap<EntityId, usize> {
        // Buses with EntityId 100, 200 at positions 0, 1
        [(EntityId(100), 0), (EntityId(200), 1)]
            .into_iter()
            .collect()
    }

    fn make_line_pos() -> HashMap<EntityId, usize> {
        // Line with EntityId 50 at position 0
        [(EntityId(50), 0)].into_iter().collect()
    }

    fn call(
        var_ref: VariableRef,
        block_idx: usize,
        indexer: &StageIndexer,
        production_models: &ProductionModelSet,
        hydro_pos: &HashMap<EntityId, usize>,
        thermal_pos: &HashMap<EntityId, usize>,
        bus_pos: &HashMap<EntityId, usize>,
        line_pos: &HashMap<EntityId, usize>,
    ) -> Vec<(usize, f64)> {
        let positions = super::EntityPositionMaps {
            hydro: hydro_pos,
            thermal: thermal_pos,
            bus: bus_pos,
            line: line_pos,
        };
        resolve_variable_ref(
            &var_ref,
            block_idx,
            indexer.n_blks,
            0, // stage_idx = 0
            indexer,
            production_models,
            &positions,
        )
    }

    // ── ThermalGeneration tests ───────────────────────────────────────────────

    /// AC from ticket: ThermalGeneration block_id=None at block 1 of 3.
    ///
    /// thermal.start = 49, thermal_pos[5] = 0, n_blks = 3, block_idx = 1
    /// Expected column = 49 + 0 * 3 + 1 = 50
    #[test]
    fn thermal_generation_block_id_none_at_block_1() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            },
            1, // block_idx
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // thermal.start = 49, pos_5 = 0, n_blks = 3, block = 1
        assert_eq!(result, vec![(49 + 0 * 3 + 1, 1.0)]);
    }

    /// ThermalGeneration with block_id=Some(2) at block 2: should use the explicit block.
    ///
    /// thermal.start = 49, thermal_pos[5] = 0, n_blks = 3, block_id = Some(2)
    /// Expected column = 49 + 0 * 3 + 2 = 51
    #[test]
    fn thermal_generation_block_id_some_at_block_2() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: Some(2),
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(49 + 0 * 3 + 2, 1.0)]);
    }

    /// ThermalGeneration for thermal at position 1.
    ///
    /// thermal.start = 49, thermal_pos[6] = 1, n_blks = 3, block = 0
    /// Expected column = 49 + 1 * 3 + 0 = 52
    #[test]
    fn thermal_generation_second_thermal() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(6),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(49 + 1 * 3 + 0, 1.0)]);
    }

    // ── HydroStorage tests ────────────────────────────────────────────────────

    /// AC from ticket: HydroStorage returns stage-level storage column.
    ///
    /// storage.start = 0, hydro_pos[EntityId(10)] = 0
    /// Expected column = 0 + 0 = 0, regardless of block_idx.
    #[test]
    fn hydro_storage_stage_level_ignores_block() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        for block_idx in [0, 1, 2] {
            let result = call(
                VariableRef::HydroStorage {
                    hydro_id: EntityId(10),
                },
                block_idx,
                &indexer,
                &prod,
                &hpos,
                &tpos,
                &bpos,
                &lpos,
            );
            // storage.start = 0, pos = 0 → column 0
            assert_eq!(result, vec![(0, 1.0)], "block_idx={block_idx}");
        }

        // Hydro at position 2 (EntityId 30)
        let result2 = call(
            VariableRef::HydroStorage {
                hydro_id: EntityId(30),
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );
        // storage.start = 0, pos = 2 → column 2
        assert_eq!(result2, vec![(2, 1.0)]);
    }

    // ── HydroOutflow tests ────────────────────────────────────────────────────

    /// AC from ticket: HydroOutflow returns 2 entries (turbine + spillage).
    ///
    /// hydro_pos[EntityId(40)] = 3 (position 3), block_id=None, block_idx=0
    /// turbine.start = 13, spillage.start = 25, n_blks = 3
    /// Expected: [(13 + 3*3 + 0, 1.0), (25 + 3*3 + 0, 1.0)] = [(22, 1.0), (34, 1.0)]
    #[test]
    fn hydro_outflow_expands_to_turbine_and_spillage() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroOutflow {
                hydro_id: EntityId(40),
                block_id: None,
            },
            0, // block_idx
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        let turbine_col = 13 + 3 * 3 + 0; // 22
        let spillage_col = 25 + 3 * 3 + 0; // 34
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (turbine_col, 1.0));
        assert_eq!(result[1], (spillage_col, 1.0));
    }

    /// HydroOutflow with block_id=Some(1) at block_idx=0: should use the explicit block.
    #[test]
    fn hydro_outflow_block_id_some_uses_explicit_block() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroOutflow {
                hydro_id: EntityId(10),
                block_id: Some(1),
            },
            0, // block_idx is irrelevant when block_id = Some
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // hydro pos=0, turbine.start=13, spillage.start=25, block=1, n_blks=3
        assert_eq!(result, vec![(13 + 0 * 3 + 1, 1.0), (25 + 0 * 3 + 1, 1.0)]);
    }

    // ── HydroGeneration tests ─────────────────────────────────────────────────

    /// AC from ticket: HydroGeneration for constant-productivity hydro returns
    /// turbine column with productivity multiplier.
    ///
    /// hydro_pos[EntityId(20)] = 1 → constant productivity 2.5
    /// turbine.start = 13, n_blks = 3, block_idx = 0
    /// Expected: [(13 + 1*3 + 0, 2.5)] = [(16, 2.5)]
    #[test]
    fn hydro_generation_constant_productivity_maps_to_turbine() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroGeneration {
                hydro_id: EntityId(20),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // hydro pos=1, turbine.start=13, n_blks=3, block=0, productivity=2.5
        assert_eq!(result, vec![(13 + 1 * 3 + 0, 2.5)]);
    }

    /// AC from ticket: HydroGeneration for FPHA hydro returns generation column.
    ///
    /// hydro_pos[EntityId(10)] = 0 → FPHA (local FPHA index = 0)
    /// generation.start = 79, n_blks = 3, block_idx = 0
    /// Expected: [(79 + 0*3 + 0, 1.0)] = [(79, 1.0)]
    #[test]
    fn hydro_generation_fpha_maps_to_generation_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // FPHA local index 0, generation.start=67, n_blks=3, block=0
        assert_eq!(result, vec![(79 + 0 * 3 + 0, 1.0)]);
    }

    /// HydroGeneration for FPHA hydro at position 2 (second FPHA hydro, local index 1).
    ///
    /// hydro_pos[EntityId(30)] = 2 → FPHA (local FPHA index = 1)
    /// generation.start = 79, n_blks = 3, block_idx = 2
    /// Expected: [(79 + 1*3 + 2, 1.0)] = [(84, 1.0)]
    #[test]
    fn hydro_generation_fpha_second_hydro_block_2() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroGeneration {
                hydro_id: EntityId(30),
                block_id: None,
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // FPHA local index 1, generation.start=67, n_blks=3, block=2
        assert_eq!(result, vec![(79 + 1 * 3 + 2, 1.0)]);
    }

    // ── HydroEvaporation tests ────────────────────────────────────────────────

    /// HydroEvaporation maps to the Q_ev column for the matching evaporation hydro.
    ///
    /// Use a dedicated indexer with evaporation hydros to test this path.
    ///
    /// N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, evap hydro at pos 0.
    /// theta = 2*(3+0) = 6
    /// turbine:    [7, 9)
    /// spillage:   [9, 11)
    /// diversion: [11, 13)
    /// deficit:   [13, 14)
    /// excess:    [14, 15)
    /// evap cols: [15, 18)  → Q_ev=15, f_evap_plus=16, f_evap_minus=17
    #[test]
    fn hydro_evaporation_maps_to_q_ev_col() {
        let evap_indexer = StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: 2,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
            },
            &crate::indexer::FphaConfig {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &crate::indexer::EvapConfig {
                hydro_indices: vec![0],
            },
        );

        let prod_models = ProductionModelSet::new(
            vec![
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            ],
            2,
            1,
        );

        let hpos: HashMap<EntityId, usize> =
            [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
        let tpos: HashMap<EntityId, usize> = HashMap::new();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

        let positions = super::EntityPositionMaps {
            hydro: &hpos,
            thermal: &tpos,
            bus: &bpos,
            line: &lpos,
        };
        let result = resolve_variable_ref(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(10),
            },
            0,
            1, // n_blks
            0, // stage_idx
            &evap_indexer,
            &prod_models,
            &positions,
        );

        assert_eq!(result, vec![(15, 1.0)]);
    }

    /// HydroEvaporation for hydro that has no evaporation model returns empty vec.
    #[test]
    fn hydro_evaporation_no_evap_model_returns_empty() {
        let evap_indexer = StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: 2,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
            },
            &crate::indexer::FphaConfig {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &crate::indexer::EvapConfig {
                hydro_indices: vec![0],
            },
        );

        let prod_models = ProductionModelSet::new(
            vec![
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            ],
            2,
            1,
        );

        let hpos: HashMap<EntityId, usize> =
            [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
        let tpos: HashMap<EntityId, usize> = HashMap::new();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

        // Hydro 20 (pos=1) has no evaporation in evap_hydro_indices=[0]
        let positions = super::EntityPositionMaps {
            hydro: &hpos,
            thermal: &tpos,
            bus: &bpos,
            line: &lpos,
        };
        let result = resolve_variable_ref(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(20),
            },
            0,
            1,
            0,
            &evap_indexer,
            &prod_models,
            &positions,
        );

        assert!(result.is_empty());
    }

    // ── Stub entity tests ─────────────────────────────────────────────────────

    /// AC from ticket: PumpingFlow returns empty vec.
    #[test]
    fn pumping_flow_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::PumpingFlow {
                station_id: EntityId(1),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    /// PumpingPower returns empty vec.
    #[test]
    fn pumping_power_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::PumpingPower {
                station_id: EntityId(1),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    /// ContractImport returns empty vec.
    #[test]
    fn contract_import_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ContractImport {
                contract_id: EntityId(99),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    /// NonControllableGeneration returns empty vec.
    #[test]
    fn non_controllable_generation_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::NonControllableGeneration {
                source_id: EntityId(7),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    // ── Missing entity ID test ─────────────────────────────────────────────────

    /// AC from ticket: missing entity ID returns empty vec (defense-in-depth).
    #[test]
    fn missing_entity_id_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        // EntityId(999) is not in thermal_pos
        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(999),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    // ── BusDeficit tests ──────────────────────────────────────────────────────

    /// AC from ticket: BusDeficit with S=2 deficit segments returns 2 column entries.
    ///
    /// bus_pos[EntityId(100)] = 0, deficit.start = 61, max_deficit_segments = 2,
    /// n_blks = 3, block_idx = 0
    /// Expected: [(61 + 0*2*3 + 0*3 + 0, 1.0), (61 + 0*2*3 + 1*3 + 0, 1.0)]
    ///         = [(61, 1.0), (64, 1.0)]
    #[test]
    fn bus_deficit_returns_one_entry_per_segment() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::BusDeficit {
                bus_id: EntityId(100),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // deficit.start=61, b_pos=0, S=2, n_blks=3, blk=0
        // seg0: 61 + 0*2*3 + 0*3 + 0 = 61
        // seg1: 61 + 0*2*3 + 1*3 + 0 = 64
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (61, 1.0));
        assert_eq!(result[1], (64, 1.0));
    }

    /// BusDeficit for second bus (position 1) at block 1.
    ///
    /// bus_pos[EntityId(200)] = 1, deficit.start = 61, S = 2, n_blks = 3, blk = 1
    /// seg0: 61 + 1*2*3 + 0*3 + 1 = 61 + 6 + 0 + 1 = 68
    /// seg1: 61 + 1*2*3 + 1*3 + 1 = 61 + 6 + 3 + 1 = 71
    #[test]
    fn bus_deficit_second_bus_block_1() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::BusDeficit {
                bus_id: EntityId(200),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (68, 1.0));
        assert_eq!(result[1], (71, 1.0));
    }

    // ── BusExcess tests ───────────────────────────────────────────────────────

    /// BusExcess maps to the excess column for the bus.
    ///
    /// bus_pos[EntityId(100)] = 0, excess.start = 73, n_blks = 3, block = 2
    /// Expected: [(73 + 0*3 + 2, 1.0)] = [(75, 1.0)]
    #[test]
    fn bus_excess_maps_to_excess_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::BusExcess {
                bus_id: EntityId(100),
                block_id: None,
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(73 + 0 * 3 + 2, 1.0)]);
    }

    // ── LineDirect / LineReverse tests ────────────────────────────────────────

    /// LineDirect maps to line_fwd column.
    ///
    /// line_pos[EntityId(50)] = 0, line_fwd.start = 55, n_blks = 3, block = 1
    /// Expected: [(55 + 0*3 + 1, 1.0)] = [(56, 1.0)]
    #[test]
    fn line_direct_maps_to_fwd_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineDirect {
                line_id: EntityId(50),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(55 + 0 * 3 + 1, 1.0)]);
    }

    /// LineReverse maps to line_rev column.
    ///
    /// line_pos[EntityId(50)] = 0, line_rev.start = 58, n_blks = 3, block = 0
    /// Expected: [(58 + 0*3 + 0, 1.0)] = [(58, 1.0)]
    #[test]
    fn line_reverse_maps_to_rev_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineReverse {
                line_id: EntityId(50),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(58, 1.0)]);
    }

    // ── LineExchange tests ──────────────────────────────────────────────────────

    /// LineExchange maps to both line_fwd and line_rev columns with opposite signs.
    ///
    /// line_pos[EntityId(50)] = 0, line_fwd.start = 55, line_rev.start = 58,
    /// n_blks = 3, block = 1
    /// Expected: [(55 + 0*3 + 1, 1.0), (58 + 0*3 + 1, -1.0)] = [(56, 1.0), (59, -1.0)]
    #[test]
    fn line_exchange_maps_to_fwd_and_rev_columns() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineExchange {
                line_id: EntityId(50),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(56, 1.0), (59, -1.0)]);
    }

    /// LineExchange with explicit block_id overrides current block_idx.
    ///
    /// block_idx = 2 but block_id = Some(0)
    /// Expected: [(55, 1.0), (58, -1.0)]
    #[test]
    fn line_exchange_with_explicit_block() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineExchange {
                line_id: EntityId(50),
                block_id: Some(0),
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(55, 1.0), (58, -1.0)]);
    }

    /// LineExchange with unknown line ID returns empty vec.
    #[test]
    fn line_exchange_unknown_id_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineExchange {
                line_id: EntityId(999),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    // ── HydroTurbined / HydroSpillage tests ───────────────────────────────────

    /// HydroTurbined maps to turbine column.
    #[test]
    fn hydro_turbined_maps_to_turbine_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        // hydro pos=1 (EntityId 20), turbine.start=13, n_blks=3, block=2
        let result = call(
            VariableRef::HydroTurbined {
                hydro_id: EntityId(20),
                block_id: None,
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(13 + 1 * 3 + 2, 1.0)]);
    }

    /// HydroSpillage maps to spillage column.
    #[test]
    fn hydro_spillage_maps_to_spillage_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        // hydro pos=3 (EntityId 40), spillage.start=25, n_blks=3, block=1
        let result = call(
            VariableRef::HydroSpillage {
                hydro_id: EntityId(40),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(25 + 3 * 3 + 1, 1.0)]);
    }
}
