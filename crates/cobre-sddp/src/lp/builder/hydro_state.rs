//! The per-hydro stage state that the builder's fills share.

use cobre_core::{HydroBlockBounds, HydroUnitGroup, ResolvedHydroUnitGroupBounds};

use crate::hydro_models::ResolvedProductionModel;

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
