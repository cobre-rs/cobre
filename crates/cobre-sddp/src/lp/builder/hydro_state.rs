//! The per-hydro stage state that the builder's fills share.

use cobre_core::commissioning::{Phase, filling_phase};
use cobre_core::{
    CascadeTopology, Hydro, HydroBlockBounds, HydroUnitGroup, ResolvedHydroUnitGroupBounds,
};

use crate::hydro_models::ResolvedProductionModel;
use crate::indexer::EntityPositions;

pub(super) fn hydro_phase(hydro: &Hydro, stage_id: i32) -> Phase {
    filling_phase(
        hydro.filling.as_ref(),
        hydro.entry_stage_id,
        hydro.exit_stage_id,
        stage_id,
    )
}

/// Resolve the cascade target an absent `PreFilling` hydro `h_idx` routes its water
/// onto: the FIRST downstream hydro NOT `PreFilling` at this stage. `None` (SINK) when
/// the chain reaches a terminal, an unresolved id, or stays `PreFilling` all the way
/// down — then `h`'s water exits the system.
///
/// The target MUST be non-`PreFilling`: a `PreFilling` row is the frozen identity
/// `v_d − v_d_in = 0`, and routing any term onto it corrupts that constraint. Routing
/// to the immediate `downstream(h)` unconditionally is the wrong-but-compiling
/// alternative — it corrupts that frozen row when the immediate downstream is itself
/// `PreFilling` (see `fill_prefilling_shortcircuit`).
///
/// The `hydros.len()`-bounded loop is defense-in-depth: `check_cascade_acyclic` already
/// proves the walk terminates. `None` also when `h` is not `PreFilling`, because its
/// water stays on its own row.
pub(super) fn resolve_shortcircuit_target(
    hydros: &[Hydro],
    cascade: &CascadeTopology,
    positions: &EntityPositions,
    stage_id: i32,
    h_idx: usize,
) -> Option<usize> {
    if !matches!(hydro_phase(&hydros[h_idx], stage_id), Phase::PreFilling) {
        return None;
    }
    let mut current_id = hydros[h_idx].id;
    for _ in 0..hydros.len() {
        let down_id = cascade.downstream(current_id)?;
        let d_idx = positions.hydro(down_id)?;
        if !matches!(hydro_phase(&hydros[d_idx], stage_id), Phase::PreFilling) {
            return Some(d_idx);
        }
        current_id = down_id;
    }
    None
}

/// Bundles the resolved group-bounds table with the three indices that are
/// constant across a cell's member groups, so `cell_max_turbined`/
/// `cell_max_generation` take one bundled parameter instead of four loose
/// ones that would cross `clippy::too_many_arguments`. `pub(super)` with its
/// `new` constructor, so `columns` and `rows` build the same per-block lookup.
#[derive(Clone, Copy)]
pub(super) struct GroupBoundLookup<'a> {
    table: &'a ResolvedHydroUnitGroupBounds,
    hydro_idx: usize,
    stage_idx: usize,
    block_idx: usize,
}

impl<'a> GroupBoundLookup<'a> {
    pub(super) fn new(
        table: &'a ResolvedHydroUnitGroupBounds,
        hydro_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Self {
        Self {
            table,
            hydro_idx,
            stage_idx,
            block_idx,
        }
    }
}

/// Methods return the resolved per-block value: the override when the study
/// supplies one, the declaration otherwise.
impl GroupBoundLookup<'_> {
    /// Group `group_pos`'s resolved turbined-flow maximum.
    fn max_turbined(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .max_turbined_m3s
            .unwrap_or(group.max_turbined_m3s)
    }

    /// Group `group_pos`'s resolved generation maximum.
    fn max_generation(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .max_generation_mw
            .unwrap_or(group.max_generation_mw)
    }

    /// Group `group_pos`'s resolved turbined-flow minimum.
    fn min_turbined(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .min_turbined_m3s
            .unwrap_or(group.min_turbined_m3s)
    }

    /// Group `group_pos`'s resolved generation minimum.
    fn min_generation(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .min_generation_mw
            .unwrap_or(group.min_generation_mw)
    }
}

/// Cell `c`'s turbined-flow upper bound. A `ConstantProductivity` model folds
/// EACH member group's own MW cap into its own flow cap first, then sums —
/// summing the raw group boxes and folding the total instead overstates the
/// cell, since `min` does not distribute over a sum whose terms bind on
/// different sides (`test_same_bus_groups_sum_into_one_cell_box`). Any other
/// model (FPHA; a non-positive productivity) sums each group's flow cap
/// unfolded, exact because FPHA's turbine and generation columns are
/// independent.
///
/// Both terms of the closing `sum.min(fold(hb...))` are load-bearing, not a
/// group term guarded by an inert plant-side cap. Drop the plant term and a
/// lowering `hydro_bounds` override — the no-raising rule's own prescribed
/// remedy for a mid-horizon capacity cut — is silently discarded. Drop the
/// group term and a multi-cell plant can turbine past its declared capacity:
/// this helper and `cell_max_generation` are the ONLY readers of
/// `hb.max_turbined_m3s`/`hb.max_generation_mw` in the hydro LP path, so
/// nothing else would catch it. The plant term is a no-op only for a plant
/// with no declared groups (never a same-bus plant with several) — inert on
/// today's fixtures, not provably inert, since both admission rules allow an
/// envelope tolerance no shipped fixture exercises.
///
/// Each member group's own cap fed into the fold is its RESOLVED per-block
/// value — the override when the study supplies one, the declaration
/// otherwise (`test_cell_bound_takes_the_resolved_group_override`).
pub(super) fn cell_max_turbined(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    model: &ResolvedProductionModel,
    hb: HydroBlockBounds,
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    let fold = |turbined: f64, generation: f64| match model {
        ResolvedProductionModel::ConstantProductivity { productivity } if *productivity > 0.0 => {
            turbined.min(generation / productivity)
        }
        _ => turbined,
    };
    let sum: f64 = positions
        .iter()
        .map(|&pos| {
            fold(
                lookup.max_turbined(pos, &groups[pos]),
                lookup.max_generation(pos, &groups[pos]),
            )
        })
        .sum();
    sum.min(fold(hb.max_turbined_m3s, hb.max_generation_mw))
}

/// Cell `c`'s min-turbine soft-floor RHS: the PLAIN SUM of the cell's own
/// member groups' resolved `min_turbined_m3s`, never a fold and never clamped
/// against the plant's declared minimum — see the min-floor contract. A floor
/// on a sum of variables (the cell's member groups all feed the same
/// aggregate turbine column) adds; it does not fold or clamp the way the
/// closing `MAX` bound does.
pub(super) fn cell_min_turbined(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    positions
        .iter()
        .map(|&pos| lookup.min_turbined(pos, &groups[pos]))
        .sum()
}

/// Cell `c`'s FPHA generation-column upper bound. FPHA's turbine and generation
/// columns are independent (no productivity fold couples them), so summing
/// `max_generation_mw` over the cell's member groups directly is exact.
///
/// Both terms of `sum.min(hb.max_generation_mw)` are load-bearing — the same
/// two-term contract `cell_max_turbined` states in full. Drop the plant term
/// and a lowering `hydro_bounds` override is silently discarded; drop the
/// group term and a multi-cell plant can generate past its declared capacity,
/// since this helper is the ONLY reader of `hb.max_generation_mw` in the
/// hydro LP path.
///
/// Each member group's own cap fed into the sum is its RESOLVED per-block
/// value — the override when the study supplies one, the declaration
/// otherwise (`test_generation_cell_bound_takes_the_resolved_group_override`).
pub(super) fn cell_max_generation(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    hb: HydroBlockBounds,
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    let sum: f64 = positions
        .iter()
        .map(|&pos| lookup.max_generation(pos, &groups[pos]))
        .sum();
    sum.min(hb.max_generation_mw)
}

/// Cell `c`'s min-generation soft-floor RHS: the PLAIN SUM of the cell's own
/// member groups' resolved `min_generation_mw` — never folded through a
/// productivity, never clamped against the plant's declared minimum. See
/// [`cell_min_turbined`] and the min-floor contract.
pub(super) fn cell_min_generation(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    positions
        .iter()
        .map(|&pos| lookup.min_generation(pos, &groups[pos]))
        .sum()
}

#[cfg(test)]
mod tests {
    use cobre_core::{CascadeTopology, EntityId, Hydro, HydroGenerationModel, HydroPenalties};

    use crate::indexer::EntityPositions;

    use super::resolve_shortcircuit_target;

    const STAGE_ID: i32 = 0;
    const FUTURE_ENTRY: i32 = 1;

    /// A cascade hydro with a caller-chosen `downstream_id` and `entry_stage_id`,
    /// no `FillingConfig` (a `PreFilling`/`Operating` chain needs none).
    fn chain_hydro(id: i32, downstream: Option<i32>, entry_stage_id: Option<i32>) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: downstream.map(EntityId),
            travel_time_hours: None,
            entry_stage_id,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            unit_groups: Vec::new(),
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties::uniform(0.0),
        }
    }

    fn positions_of(hydros: &[Hydro]) -> EntityPositions {
        EntityPositions::from_slices(hydros.iter().map(|h| h.id), [], [], [], [], [])
    }

    #[test]
    fn resolve_shortcircuit_target_walks_prefilling_chains_and_skips_other_phases() {
        // A(0) -> B(1) -> C(2): A and B PreFilling (entry lies past STAGE_ID), C
        // Operating (no entry window at all).
        let hydros = vec![
            chain_hydro(1, Some(2), Some(FUTURE_ENTRY)),
            chain_hydro(2, Some(3), Some(FUTURE_ENTRY)),
            chain_hydro(3, None, None),
        ];
        let cascade = CascadeTopology::build(&hydros);
        let positions = positions_of(&hydros);

        assert_eq!(
            resolve_shortcircuit_target(&hydros, &cascade, &positions, STAGE_ID, 0),
            Some(2),
            "A's chain walk skips PreFilling B and lands on Operating C"
        );
        assert_eq!(
            resolve_shortcircuit_target(&hydros, &cascade, &positions, STAGE_ID, 1),
            Some(2),
            "B routes directly onto Operating C"
        );
        assert_eq!(
            resolve_shortcircuit_target(&hydros, &cascade, &positions, STAGE_ID, 2),
            None,
            "C is not PreFilling: the entry guard returns None before any walk"
        );

        // Same chain, but C is PreFilling too and has no downstream: the walk
        // reaches the terminal without ever finding a non-PreFilling target.
        let sink_hydros = vec![
            chain_hydro(1, Some(2), Some(FUTURE_ENTRY)),
            chain_hydro(2, Some(3), Some(FUTURE_ENTRY)),
            chain_hydro(3, None, Some(FUTURE_ENTRY)),
        ];
        let sink_cascade = CascadeTopology::build(&sink_hydros);
        let sink_positions = positions_of(&sink_hydros);

        assert_eq!(
            resolve_shortcircuit_target(&sink_hydros, &sink_cascade, &sink_positions, STAGE_ID, 0),
            None,
            "C is a PreFilling sink: the chain never reaches a non-PreFilling target"
        );
    }
}
