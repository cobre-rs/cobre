//! The per-hydro stage state that the builder's fills share.

use std::ops::Range;

use cobre_core::commissioning::{Phase, filling_phase};
use cobre_core::{
    CascadeTopology, Hydro, HydroBlockBounds, HydroPenalties, HydroUnitGroup, ResolvedBounds,
    ResolvedHydroUnitGroupBounds, Stage,
};
use cobre_io::validation::semantic::envelope_tolerance;

use crate::hydro_models::ResolvedProductionModel;
use crate::indexer::{EntityPositions, HydroCell, HydroCellIndex, HydroSys};

pub(super) fn hydro_phase(hydro: &Hydro, stage_id: i32) -> Phase {
    filling_phase(
        hydro.filling.as_ref(),
        hydro.entry_stage_id,
        hydro.exit_stage_id,
        stage_id,
    )
}

/// Whether `hydro`'s `min_storage` floor is lifted to `0` at `stage_id`.
///
/// CONTRACT: `min_storage` is a HARD lower bound for every hydro EXCEPT (a) a
/// filling one (`hydro.filling.is_some()`), whose floor relaxes to `0` in ALL
/// phases and is re-imposed as a SOFT floor by
/// `fill_filled_min_storage_floor_columns`, and (b) a non-filling hydro while
/// `PreFilling` (commissioning-dormant), whose frozen-identity row pins `v_h`
/// to the inert IC storage — a hard floor above that IC value would reject the
/// pin and make the LP infeasible. FORBIDDEN — globalizing the relax to all
/// Operating hydros (makes dead volume soft system-wide); keeping it hard
/// through a dormant non-filling stage (rejects the IC pin). The dormant relax
/// disappears at `Operating`, restoring the hard floor.
pub(crate) fn storage_floor_relaxed(hydro: &Hydro, stage_id: i32) -> bool {
    hydro.filling.is_some() || matches!(hydro_phase(hydro, stage_id), Phase::PreFilling)
}

/// Whether `hydro` is commissioning (`PreFilling`/`Filling`) at `stage_id`, its
/// turbine columns pinned at zero.
pub(super) fn is_suspended(hydro: &Hydro, stage_id: i32) -> bool {
    matches!(
        hydro_phase(hydro, stage_id),
        Phase::PreFilling | Phase::Filling
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
/// remedy for a mid-horizon capacity cut — is silently discarded on a
/// single-cell plant, which has no cap row ([`plant_cap_rhs`]). Drop the group
/// term and a cell can turbine past its own groups' capacity: the cap row bounds
/// only the cells' total. The plant term is a no-op only for a plant
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

/// Cell `c`'s own min-turbine floor: the PLAIN SUM of its member groups'
/// resolved `min_turbined_m3s`, never a fold and never a `.min(plant)` clamp —
/// a floor on a sum of variables adds, unlike the closing `MAX` bound.
/// [`PlantFloor`] decides where the plant minimum binds on top of it.
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
/// two-term contract `cell_max_turbined` states in full.
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

/// Cell `c`'s own min-generation floor: the PLAIN SUM of its member groups'
/// resolved `min_generation_mw`, never folded through a productivity. See
/// [`cell_min_turbined`].
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

/// The two soft minima declared both on a plant and on its unit groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MinFloor {
    /// `min_turbined_m3s`.
    Turbined,
    /// `min_generation_mw`.
    Generation,
}

impl MinFloor {
    pub(super) const ALL: [Self; 2] = [Self::Turbined, Self::Generation];

    fn of_cell(
        self,
        groups: &[HydroUnitGroup],
        positions: &[usize],
        lookup: GroupBoundLookup<'_>,
    ) -> f64 {
        match self {
            Self::Turbined => cell_min_turbined(groups, positions, lookup),
            Self::Generation => cell_min_generation(groups, positions, lookup),
        }
    }

    fn of_plant(self, hb: HydroBlockBounds) -> f64 {
        match self {
            Self::Turbined => hb.min_turbined_m3s,
            Self::Generation => hb.min_generation_mw,
        }
    }

    /// Penalty per unit of shortfall, shared by the cell and plant slacks.
    pub(super) fn below_cost(self, hp: HydroPenalties) -> f64 {
        match self {
            Self::Turbined => hp.turbined_violation_below_cost,
            Self::Generation => hp.generation_violation_below_cost,
        }
    }

    fn declared(self, hydro: &Hydro) -> f64 {
        match self {
            Self::Turbined => hydro.min_turbined_m3s,
            Self::Generation => hydro.min_generation_mw,
        }
    }

    fn field(self) -> &'static str {
        match self {
            Self::Turbined => "min_turbined_m3s",
            Self::Generation => "min_generation_mw",
        }
    }
}

/// One plant's `MinFloor` at one block, split into what its cell rows and its
/// residual row enforce. The tighter floor wins: the plant's resolved minimum
/// binds only where it exceeds the cells' summed floor by more than
/// [`envelope_tolerance`] — the tolerance rule 44 admits a declared minimum
/// with, so a declared minimum never changes the LP. A single-cell plant folds
/// the binding minimum into its one cell row; a multi-cell plant gets a
/// residual row on top of its cell rows. Either way the shortfall is charged
/// once, as `max(plant shortfall, Σ cell shortfalls)`.
pub(super) struct PlantFloor<'a> {
    kind: MinFloor,
    groups: &'a [HydroUnitGroup],
    cells: &'a HydroCellIndex,
    plant_cells: Range<usize>,
    lookup: GroupBoundLookup<'a>,
    plant_min: f64,
    cell_sum: f64,
}

impl<'a> PlantFloor<'a> {
    pub(super) fn resolve(
        kind: MinFloor,
        hydro: &'a Hydro,
        cells: &'a HydroCellIndex,
        h: HydroSys,
        lookup: GroupBoundLookup<'a>,
        hb: HydroBlockBounds,
    ) -> Self {
        let plant_cells = cells.cells_of(h);
        let cell_sum = plant_cells
            .clone()
            .map(|c| {
                kind.of_cell(
                    &hydro.unit_groups,
                    cells.groups_of(HydroCell::new(c)),
                    lookup,
                )
            })
            .sum();
        Self {
            kind,
            groups: &hydro.unit_groups,
            cells,
            plant_cells,
            lookup,
            plant_min: kind.of_plant(hb),
            cell_sum,
        }
    }

    fn uncovered(&self) -> Option<f64> {
        (self.plant_min > self.cell_sum + envelope_tolerance(self.plant_min))
            .then_some(self.plant_min)
    }

    /// RHS of cell `c`'s floor row.
    pub(super) fn cell_rhs(&self, c: HydroCell) -> f64 {
        match self.uncovered() {
            Some(plant_min) if self.plant_cells.len() == 1 => plant_min,
            _ => self
                .kind
                .of_cell(self.groups, self.cells.groups_of(c), self.lookup),
        }
    }

    /// RHS of the plant's residual row; `None` when no residual binds.
    pub(super) fn residual_rhs(&self) -> Option<f64> {
        self.uncovered().filter(|_| self.plant_cells.len() > 1)
    }

    fn is_below_cells(&self) -> bool {
        self.plant_min + envelope_tolerance(self.plant_min) < self.cell_sum
    }
}

/// The two hard maxima declared both on a plant and on its unit groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MaxCap {
    /// `max_turbined_m3s`, on the turbine columns.
    Turbined,
    /// `max_generation_mw`, on the generation terms.
    Generation,
}

impl MaxCap {
    pub(super) const ALL: [Self; 2] = [Self::Turbined, Self::Generation];

    fn of_plant(self, hb: HydroBlockBounds) -> f64 {
        match self {
            Self::Turbined => hb.max_turbined_m3s,
            Self::Generation => hb.max_generation_mw,
        }
    }

    /// The largest value the cell's term on this kind's row can take under the
    /// column bounds `cell_max_turbined`/`cell_max_generation` set.
    fn of_cell(
        self,
        groups: &[HydroUnitGroup],
        positions: &[usize],
        model: &ResolvedProductionModel,
        hb: HydroBlockBounds,
        lookup: GroupBoundLookup<'_>,
    ) -> f64 {
        match (self, model) {
            (Self::Turbined, _) => cell_max_turbined(groups, positions, model, hb, lookup),
            (Self::Generation, ResolvedProductionModel::ConstantProductivity { productivity }) => {
                productivity * cell_max_turbined(groups, positions, model, hb, lookup)
            }
            (Self::Generation, ResolvedProductionModel::Fpha { .. }) => {
                cell_max_generation(groups, positions, hb, lookup)
            }
        }
    }
}

/// RHS of plant `h`'s `kind` cap row: the plant's resolved maximum where it is
/// below its cells' summed caps by more than [`envelope_tolerance`] — the
/// tolerance rule 41 admits declared maxima with, so a declared maximum never
/// changes the LP. `None` for a single-cell plant, whose column bound already
/// clamps to the plant maximum.
pub(super) fn plant_cap_rhs(
    kind: MaxCap,
    hydro: &Hydro,
    cells: &HydroCellIndex,
    h: HydroSys,
    model: &ResolvedProductionModel,
    lookup: GroupBoundLookup<'_>,
    hb: HydroBlockBounds,
) -> Option<f64> {
    let plant_cells = cells.cells_of(h);
    if plant_cells.len() < 2 {
        return None;
    }
    let plant_max = kind.of_plant(hb);
    let cell_sum: f64 = plant_cells
        .map(|c| {
            kind.of_cell(
                &hydro.unit_groups,
                cells.groups_of(HydroCell::new(c)),
                model,
                hb,
                lookup,
            )
        })
        .sum();
    (plant_max + envelope_tolerance(plant_max) < cell_sum).then_some(plant_max)
}

/// Emit one setup-time advisory per plant and `MinFloor` whose `hydro_bounds`
/// override lowers the plant minimum below its unit groups' summed minimum:
/// the groups' floors still bind (see [`PlantFloor`]), so the override has no
/// effect and the group minima are what must be lowered.
pub(crate) fn warn_on_inert_plant_floor_overrides(
    hydros: &[Hydro],
    stages: &[Stage],
    bounds: &ResolvedBounds,
    cells: &HydroCellIndex,
) {
    let study_stages: Vec<_> = stages.iter().filter(|s| s.id >= 0).collect();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        for kind in MinFloor::ALL {
            let declared = kind.declared(hydro).to_bits();
            let mut inert_stage_ids = study_stages
                .iter()
                .enumerate()
                .filter(|(stage_idx, stage)| {
                    (0..stage.blocks.len()).any(|blk| {
                        let floor = PlantFloor::resolve(
                            kind,
                            hydro,
                            cells,
                            HydroSys::new(h_idx),
                            GroupBoundLookup::new(bounds.group_overlay(), h_idx, *stage_idx, blk),
                            bounds.hydro_bounds_at_block(h_idx, *stage_idx, blk),
                        );
                        floor.plant_min.to_bits() != declared && floor.is_below_cells()
                    })
                })
                .map(|(_, stage)| stage.id);
            if let Some(first_stage_id) = inert_stage_ids.next() {
                let n_stages = 1 + inert_stage_ids.count();
                tracing::warn!(
                    "hydro {} ({}): hydro_bounds lowers {field} below the sum of its unit \
                     groups' {field} at {n_stages} stage(s), first at stage {first_stage_id}; \
                     the unit groups' minima still bind, so the override has no effect — lower \
                     them in hydro_unit_group_bounds instead",
                    hydro.id,
                    hydro.name,
                    field = kind.field(),
                );
            }
        }
    }
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
