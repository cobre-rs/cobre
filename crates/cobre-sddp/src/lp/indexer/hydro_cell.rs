//! [`HydroCellIndex`]: the study-scope partition of each hydro plant's unit
//! groups into `bus_id`-equivalence-class cells.
//!
//! Same-bus groups collapse into one cell; a plant with groups on several
//! buses splits into one cell per distinct bus. For every study whose plants
//! each occupy a single bus — every study shipping today — the partition is
//! the identity: `n_cells() == n_hydros` and plant `h`'s only cell is cell
//! `h`. That identity is the safety mechanism: nothing but this index changes
//! for a single-bus study, so every consumer built on top of it stays
//! byte-neutral until a multi-bus plant actually exists.
//!
//! Storage is CSR-shaped: [`HydroCellIndex::plant_cell_start`] and
//! [`HydroCellIndex::cell_group_start`]
//! are offset vectors of length `n + 1`, following the same `start[i]..start[i +
//! 1]` convention as every other CSR range in this crate. The plant→cell map is
//! a *range*, not the single-valued `Option` a filtering local index (e.g.
//! `fpha_local_index`) would use — a plant's groups only ever expand into more
//! cells, never fewer, so `hydro → range of cells` is the correct shape.
//!
//! A plant with no groups cannot reach `build` — both `parse_hydros` and
//! `SystemBuilder::build` reject an empty `unit_groups` before a system is
//! ever constructed — and `build` would contribute zero cells for one
//! regardless (`test_build_yields_no_cells_for_an_unnormalized_plant`).

use std::ops::Range;

use cobre_core::{EntityId, Hydro};

use super::{HydroCell, HydroSys};

/// Study-scope partition of each hydro plant's unit groups into `bus_id`
/// cells, built once from `System::hydros` and stage-invariant.
#[derive(Debug, Clone)]
pub struct HydroCellIndex {
    /// `plant_cell_start[h]..plant_cell_start[h + 1]` is plant `h`'s half-open
    /// cell range. Length `n_hydros + 1`.
    plant_cell_start: Vec<usize>,
    /// Owning plant of each cell. Length `n_cells()`.
    cell_plant: Vec<HydroSys>,
    /// `bus_id` of each cell. Length `n_cells()`.
    cell_bus: Vec<EntityId>,
    /// CSR offsets into `cell_group_pos`. Length `n_cells() + 1`.
    cell_group_start: Vec<usize>,
    /// Member group positions within the owning plant's `unit_groups` slice,
    /// ascending. Length `Σ groups`.
    cell_group_pos: Vec<usize>,
    /// Cell `c`'s share of its plant's declared turbine capacity:
    /// `(Σ_{g∈c} max_turbined_m3s) / (Σ_{g∈plant} max_turbined_m3s)`, `0.0` when
    /// the plant total is `0.0` (never `NaN` from a `0.0 / 0.0`). Length
    /// `n_cells()`. A study-time constant derived from DECLARED capacity, never
    /// recomputed per stage or block, and never sourced from a resolved
    /// per-stage bound override on a group, should one later exist: an
    /// availability derate is already enforced on the flow path by each
    /// cell's own column bounds, so tracking one here would double-count it
    /// and can reintroduce the zero-denominator case on a fully-derated
    /// stage. A permanent capacity change belongs on the group's declared
    /// value, which this constant already picks up at the next study build.
    cell_share: Vec<f64>,
}

impl HydroCellIndex {
    /// Partition every plant's `unit_groups` by `bus_id`. Cells within a plant
    /// are ordered by ascending `bus_id` — the cell's own defining attribute,
    /// so renumbering a plant's group ids or reordering their declaration
    /// cannot reorder its cells.
    ///
    /// The counting pass sizes every `Vec` at its exact capacity. The sort is
    /// stable and runs over the already `id`-ascending `unit_groups`, so
    /// `cell_group_pos` stays ascending within a cell. Never a
    /// `HashMap`/`HashSet` grouping — declaration-order invariance requires a
    /// sort or a positional scan.
    #[must_use]
    pub fn build(hydros: &[Hydro]) -> Self {
        let n_hydros = hydros.len();

        let mut buses: Vec<EntityId> = Vec::new();
        let mut total_cells = 0_usize;
        let mut total_groups = 0_usize;
        for hydro in hydros {
            buses.clear();
            buses.extend(hydro.unit_groups.iter().map(|g| g.bus_id));
            total_groups += buses.len();
            buses.sort_unstable();
            buses.dedup();
            total_cells += buses.len();
        }

        let mut plant_cell_start = Vec::with_capacity(n_hydros + 1);
        let mut cell_plant = Vec::with_capacity(total_cells);
        let mut cell_bus = Vec::with_capacity(total_cells);
        let mut cell_group_start = Vec::with_capacity(total_cells + 1);
        let mut cell_group_pos = Vec::with_capacity(total_groups);
        let mut cell_share = Vec::with_capacity(total_cells);

        plant_cell_start.push(0);
        cell_group_start.push(0);

        let mut order: Vec<(EntityId, usize)> = Vec::new();
        for (h, hydro) in hydros.iter().enumerate() {
            order.clear();
            order.extend(
                hydro
                    .unit_groups
                    .iter()
                    .enumerate()
                    .map(|(pos, g)| (g.bus_id, pos)),
            );
            order.sort_by_key(|&(bus_id, _)| bus_id);

            // Guard against 0.0 / 0.0: `fill_turbine_columns` divides by productivity
            // under the same `> 0.0` idiom (`lp/builder/columns.rs`).
            let plant_total: f64 = hydro.unit_groups.iter().map(|g| g.max_turbined_m3s).sum();

            let mut i = 0;
            while i < order.len() {
                let bus_id = order[i].0;
                cell_plant.push(HydroSys::new(h));
                cell_bus.push(bus_id);
                let mut cell_total = 0.0_f64;
                while i < order.len() && order[i].0 == bus_id {
                    let pos = order[i].1;
                    cell_group_pos.push(pos);
                    cell_total += hydro.unit_groups[pos].max_turbined_m3s;
                    i += 1;
                }
                cell_group_start.push(cell_group_pos.len());
                cell_share.push(if plant_total > 0.0 {
                    cell_total / plant_total
                } else {
                    0.0
                });
            }
            plant_cell_start.push(cell_plant.len());
        }

        Self {
            plant_cell_start,
            cell_plant,
            cell_bus,
            cell_group_start,
            cell_group_pos,
            cell_share,
        }
    }

    /// Total cell count across every plant.
    #[inline]
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.cell_plant.len()
    }

    /// Plant `h`'s half-open cell range.
    #[inline]
    #[must_use]
    pub fn cells_of(&self, h: HydroSys) -> Range<usize> {
        self.plant_cell_start[h.get()]..self.plant_cell_start[h.get() + 1]
    }

    /// Plant `h`'s first cell — exact only while `h` has exactly one cell; a
    /// consumer resolving a whole-plant read to this single cell is
    /// deliberately provisional and must be replaced with the correct
    /// per-cell or summed form once a downstream consumer gives per-cell
    /// values their own meaning.
    #[inline]
    #[must_use]
    pub fn first_cell_of(&self, h: HydroSys) -> HydroCell {
        HydroCell::new(self.cells_of(h).start)
    }

    /// The `bus_id` of cell `c`.
    #[inline]
    #[must_use]
    pub fn bus_of(&self, c: HydroCell) -> EntityId {
        self.cell_bus[c.get()]
    }

    /// Cell `c`'s share of its plant's declared turbine capacity:
    /// `(Σ_{g∈c} max_turbined_m3s) / (Σ_{g∈plant} max_turbined_m3s)`, `0.0` when
    /// the plant total is `0.0`.
    #[inline]
    #[must_use]
    pub fn share_of(&self, c: HydroCell) -> f64 {
        self.cell_share[c.get()]
    }

    /// Member group positions (within the owning plant's `unit_groups` slice)
    /// of cell `c`, ascending.
    #[inline]
    #[must_use]
    pub fn groups_of(&self, c: HydroCell) -> &[usize] {
        &self.cell_group_pos[self.cell_group_start[c.get()]..self.cell_group_start[c.get() + 1]]
    }

    /// Plant `h`'s cell whose unit groups sit on bus `bus_id`, or `None` when no
    /// group of `h` carries it — a scan over `cells_of(h)` (already ascending by
    /// `bus_id`), never a `HashMap`.
    #[must_use]
    pub fn cell_of_bus(&self, h: HydroSys, bus_id: EntityId) -> Option<HydroCell> {
        self.cells_of(h)
            .map(HydroCell::new)
            .find(|&c| self.bus_of(c) == bus_id)
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{EntityId, Hydro, HydroGenerationModel, HydroPenalties, HydroUnitGroup};

    use super::{HydroCell, HydroCellIndex, HydroSys};
    use crate::test_support::{geometry_hydro, make_unit_group};

    fn zero_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    /// A minimal constant-productivity hydro at `id`/`unit_groups` the caller
    /// chooses; every other field is an inert default. `unit_groups` passes
    /// through unmodified — a caller needing a mirror group or canonical id
    /// order calls `Hydro::declare_mirror_unit_group`/`Hydro::sort_unit_groups`
    /// itself.
    fn base_hydro(id: i32, unit_groups: Vec<HydroUnitGroup>) -> Hydro {
        Hydro {
            unit_groups,
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: NaiveDate::default(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        }
    }

    /// Three plants: plant 0 declares no groups, so the mirror helper gives it a
    /// single group on bus 100; plant 1 declares three groups all on bus 5, with
    /// deliberately differing bounds (pinning that the partition key is
    /// `bus_id` alone — same-bus groups merge regardless of how their bounds
    /// differ, the precondition a later per-cell bound-summing pass rests on);
    /// plant 2 declares two groups, on bus 9 and bus 3, whose declaration order and
    /// group ids the `swap_plant2` axis reverses together. Its group ids (20/21)
    /// are deliberately neither 2 nor 3 (the plant/cell/group/bus
    /// index-coincidence check) — every fixture axis varies independently.
    fn three_plant_fixture(swap_plant2: bool) -> Vec<Hydro> {
        let mut plant0 = base_hydro(0, Vec::new());
        plant0.declare_mirror_unit_group(EntityId(100));

        let plant1 = base_hydro(
            1,
            vec![
                make_unit_group(EntityId(10), EntityId(5), 0.0, 10.0, 0.0, 10.0),
                make_unit_group(EntityId(11), EntityId(5), 2.0, 15.0, 1.0, 12.0),
                make_unit_group(EntityId(12), EntityId(5), 4.0, 20.0, 3.0, 14.0),
            ],
        );

        let (bus9_id, bus3_id) = if swap_plant2 {
            (EntityId(21), EntityId(20))
        } else {
            (EntityId(20), EntityId(21))
        };
        let bus9_group = make_unit_group(bus9_id, EntityId(9), 0.0, 10.0, 0.0, 10.0);
        let bus3_group = make_unit_group(bus3_id, EntityId(3), 0.0, 10.0, 0.0, 10.0);
        let plant2_groups = if swap_plant2 {
            vec![bus3_group, bus9_group]
        } else {
            vec![bus9_group, bus3_group]
        };
        let plant2 = base_hydro(2, plant2_groups);

        vec![plant0, plant1, plant2]
    }

    #[test]
    fn test_cell_index_is_identity_for_single_bus_plants() {
        let hydros: Vec<Hydro> = (0..4).map(geometry_hydro).collect();
        let index = HydroCellIndex::build(&hydros);

        assert_eq!(index.n_cells(), 4);
        for (i, hydro) in hydros.iter().enumerate() {
            assert_eq!(index.cells_of(HydroSys::new(i)), i..i + 1);
            // Pins geometry_hydro's declare_mirror_unit_group(id) call passing the
            // SAME id as the plant's own `id` field — a mismatched argument
            // would mirror the wrong bus.
            assert_eq!(hydro.unit_groups.len(), 1);
            assert_eq!(
                hydro.unit_groups[0].bus_id,
                EntityId(i32::try_from(i).unwrap())
            );
        }
    }

    #[test]
    fn test_multi_bus_plant_splits_into_bus_ordered_cells() {
        let hydros = three_plant_fixture(false);
        let index = HydroCellIndex::build(&hydros);

        assert_eq!(index.n_cells(), 4);
        assert_eq!(index.cells_of(HydroSys::new(0)), 0..1);
        assert_eq!(index.cells_of(HydroSys::new(1)), 1..2);
        assert_eq!(index.cells_of(HydroSys::new(2)), 2..4);
        assert_eq!(index.groups_of(HydroCell::new(1)).len(), 3);
        // Plant 2 declares bus 9 before bus 3, so this is what pins ascending
        // `bus_id` order against a first-appearance implementation.
        assert_eq!(index.bus_of(HydroCell::new(2)), EntityId(3));
        assert_eq!(index.bus_of(HydroCell::new(3)), EntityId(9));
    }

    /// `cell_of_bus` resolves each of plant 2's own two buses to that bus's
    /// cell, and misses on a bus no group of plant 2 carries — including bus 5,
    /// which plant 1's groups DO carry, pinning that the scan is confined to
    /// `cells_of(h)` and never the whole index.
    #[test]
    fn cell_of_bus_resolves_each_bus_and_misses_a_foreign_one() {
        let hydros = three_plant_fixture(false);
        let index = HydroCellIndex::build(&hydros);
        let plant2 = HydroSys::new(2);

        assert_eq!(
            index.cell_of_bus(plant2, EntityId(3)),
            Some(HydroCell::new(2))
        );
        assert_eq!(
            index.cell_of_bus(plant2, EntityId(9)),
            Some(HydroCell::new(3))
        );
        assert_eq!(index.cell_of_bus(plant2, EntityId(5)), None);
        assert_eq!(index.cell_of_bus(plant2, EntityId(12345)), None);
    }

    #[test]
    fn test_cell_index_is_invariant_to_group_declaration_order_and_ids() {
        let a = HydroCellIndex::build(&three_plant_fixture(false));
        let b = HydroCellIndex::build(&three_plant_fixture(true));

        assert_eq!(a.plant_cell_start, b.plant_cell_start);
        assert_eq!(a.cell_plant, b.cell_plant);
        assert_eq!(a.cell_bus, b.cell_bus);
        assert_eq!(a.cell_group_start, b.cell_group_start);

        for c in 0..a.n_cells() {
            let cell = HydroCell::new(c);
            assert_eq!(a.bus_of(cell), b.bus_of(cell));
            assert_eq!(a.groups_of(cell).len(), b.groups_of(cell).len());
        }
    }

    /// `build`'s chosen semantics: a plant with an empty `unit_groups`
    /// contributes zero cells rather than a defensively derived implicit one.
    /// Unreachable in production — `SystemBuilder::build` and `cobre-io`'s
    /// parse path both reject an empty `unit_groups` before a system is ever
    /// constructed.
    #[test]
    fn test_build_yields_no_cells_for_an_unnormalized_plant() {
        let unnormalized = base_hydro(0, Vec::new());
        let index = HydroCellIndex::build(&[unnormalized]);

        assert!(index.cells_of(HydroSys::new(0)).is_empty());
        assert_eq!(index.n_cells(), 0);
    }

    /// Plant A's two groups declare `max_turbined_m3s` 9000.0 (bus 5) and
    /// 3000.0 (bus 6) — the same unequal split the production-row apportionment
    /// tests use — so `share_of` must read `[0.75, 0.25]` in ascending-bus-id
    /// cell order, summing to `1.0`. Plant B has a single group with
    /// `max_turbined_m3s = 0.0`, pinning the zero-denominator guard:
    /// `!is_nan()` is asserted explicitly because `NaN != NaN` would make an
    /// equality assertion against the wrong value pass vacuously.
    #[test]
    fn test_cell_share_sums_to_one_and_is_zero_for_a_zero_capacity_plant() {
        let plant_a = base_hydro(
            0,
            vec![
                make_unit_group(EntityId(10), EntityId(5), 0.0, 100.0, 0.0, 9000.0),
                make_unit_group(EntityId(11), EntityId(6), 0.0, 100.0, 0.0, 3000.0),
            ],
        );
        let plant_b = base_hydro(
            1,
            vec![make_unit_group(
                EntityId(20),
                EntityId(7),
                0.0,
                100.0,
                0.0,
                0.0,
            )],
        );
        let hydros = vec![plant_a, plant_b];
        let index = HydroCellIndex::build(&hydros);

        assert_eq!(index.n_cells(), 3);
        let cells_a: Vec<HydroCell> = index
            .cells_of(HydroSys::new(0))
            .map(HydroCell::new)
            .collect();
        assert_eq!(cells_a.len(), 2);
        let share_0 = index.share_of(cells_a[0]);
        let share_1 = index.share_of(cells_a[1]);
        assert!(
            (share_0 - 0.75).abs() < 1e-12,
            "cell on bus 5 (9000 of 12000) must carry share 0.75, got {share_0}"
        );
        assert!(
            (share_1 - 0.25).abs() < 1e-12,
            "cell on bus 6 (3000 of 12000) must carry share 0.25, got {share_1}"
        );
        assert!(
            (share_0 + share_1 - 1.0).abs() < 1e-12,
            "plant A's cell shares must sum to 1.0, got {}",
            share_0 + share_1
        );

        let cell_b = index.first_cell_of(HydroSys::new(1));
        let share_b = index.share_of(cell_b);
        assert!(
            !share_b.is_nan(),
            "a zero-capacity plant's cell share must be 0.0, never NaN"
        );
        assert_eq!(share_b, 0.0);
    }
}
