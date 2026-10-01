//! Template-family structural decoders: inverses of the address owners
//! (`StageGeometry`, `StateSpace`, `DeliveryRing`), read by
//! `tests/template_family_structure.rs`. The owners they invert are
//! `pub(crate)`, so the decoders live here rather than in an integration-test
//! crate, which could only re-derive their arithmetic.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use cobre_core::{BlockMode, ContractType, System};
use cobre_solver::StageTemplate;

use crate::indexer::{
    AnticipatedLocal, BlockGrid, BlockIdx, Boundary, BusSys, FphaCellLocal, HydroCell, HydroSys,
    LineSys, NcsSys, PumpingSys, ThermalSys,
};
use crate::lp::builder::{
    DeliveryRing, StageGeometry, contract_family_slot, evaporation_slot_count,
};
use crate::lp::indexer::{HydroCellIndex, StateSpace};

/// Column- and row-major unscaled view of one stage's structural LP.
/// Duplicate `(row, col)` entries are kept exactly as `assemble_csc` wrote
/// them.
pub struct UnscaledMatrix {
    rows: Vec<Vec<(usize, f64)>>,
    cols: Vec<Vec<(usize, f64)>>,
}

impl UnscaledMatrix {
    /// Unscales every structural entry of `t`: `value / (row_scale[r] *
    /// col_scale[c])`, treating an empty scale vector as all-`1.0`.
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        reason = "CSC col_starts/row_indices are non-negative by construction"
    )]
    pub fn of(t: &StageTemplate) -> Self {
        let mut rows = vec![Vec::new(); t.num_rows];
        let mut cols = vec![Vec::new(); t.num_cols];
        let col_bounds = t.col_starts.iter().zip(t.col_starts.iter().skip(1));
        for (c, (&start, &end)) in col_bounds.enumerate() {
            let col_scale = t.col_scale.get(c).copied().unwrap_or(1.0);
            for k in (start as usize)..(end as usize) {
                let r = t.row_indices[k] as usize;
                let row_scale = t.row_scale.get(r).copied().unwrap_or(1.0);
                let v = t.values[k] / (row_scale * col_scale);
                rows[r].push((c, v));
                cols[c].push((r, v));
            }
        }
        Self { rows, cols }
    }

    /// Row `r`'s unscaled `(column, value)` entries, column-sorted.
    #[must_use]
    pub fn row(&self, r: usize) -> &[(usize, f64)] {
        &self.rows[r]
    }

    /// Column `c`'s unscaled `(row, value)` entries.
    #[must_use]
    pub fn col(&self, c: usize) -> &[(usize, f64)] {
        &self.cols[c]
    }
}

/// Inverts [`StageGeometry::block_storage_col`] for every hydro at every
/// boundary the stage's block mode reaches: `(Incoming, Outgoing)` on a
/// parallel stage, every `Boundary::from_index(0..=n_blks)` on a
/// chronological one.
#[must_use]
pub fn storage_column_owners(
    geom: &StageGeometry,
    state: &StateSpace,
) -> HashMap<usize, (HydroSys, Boundary)> {
    let mut owners = HashMap::new();
    let boundaries: Vec<Boundary> = match geom.block_mode {
        BlockMode::Parallel => vec![Boundary::Incoming, Boundary::Outgoing],
        BlockMode::Chronological => (0..=geom.n_blks)
            .map(|k| Boundary::from_index(k, geom.n_blks))
            .collect(),
    };
    for h in 0..state.hydro_count {
        let h = HydroSys::new(h);
        for &b in &boundaries {
            owners.insert(geom.block_storage_col(state, h, b), (h, b));
        }
    }
    owners
}

/// Inverts [`StageGeometry::generation_col`] by walking [`FphaCellLocal`]
/// indices until the block-0 address leaves `geom.generation` — the sole
/// family whose cell count is not one of [`StateSpace`]'s or
/// [`StageGeometry`]'s own counts. Probes the candidate address through
/// [`BlockGrid`] (the same primitive `generation_col` wraps) rather than the
/// accessor itself, since the accessor's own debug assertion would panic on
/// the very out-of-range probe that ends the walk.
#[must_use]
pub fn generation_column_owners(geom: &StageGeometry) -> HashMap<usize, (FphaCellLocal, BlockIdx)> {
    let mut owners = HashMap::new();
    if geom.n_blks == 0 {
        return owners;
    }
    let grid = BlockGrid::new(geom.n_blks, 0);
    let mut c = 0;
    loop {
        let probe = grid.flat(geom.generation.start, c, BlockIdx::new(0));
        if !geom.generation.contains(&probe) {
            break;
        }
        let local = FphaCellLocal::new(c);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(geom.generation_col(local, blk), (local, blk));
        }
        c += 1;
    }
    owners
}

/// Inverts [`StageGeometry::water_balance_row`] over every hydro and every
/// block position the family's own `rows_per_entity` reaches.
#[must_use]
pub fn water_row_owners(
    geom: &StageGeometry,
    n_hydros: usize,
) -> HashMap<usize, (HydroSys, BlockIdx)> {
    let mut owners = HashMap::new();
    let rows_per_entity = geom.water_balance.rows_per_entity(geom.n_blks);
    for h in 0..n_hydros {
        let h = HydroSys::new(h);
        for blk in 0..rows_per_entity {
            let blk = BlockIdx::new(blk);
            owners.insert(geom.water_balance_row(h, blk), (h, blk));
        }
    }
    owners
}

/// Every row family [`StageGeometry`] holds, plus the z-inflow rows
/// [`StateSpace::z_inflow_rows`] leads every stage with. One entry per
/// family: `water_balance`, `load_balance`, `fpha`, `filling_target`,
/// `filled_min_storage_floor`, `evaporation` (every
/// [`EvaporationIndices`](crate::lp::indexer::EvaporationIndices)'s
/// `evap_row`), and `z_inflow`.
#[must_use]
pub fn geometry_row_families(
    geom: &StageGeometry,
    state: &StateSpace,
) -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("water_balance", geom.water_balance.range().collect()),
        ("load_balance", geom.load_balance.range().collect()),
        ("fpha", geom.fpha.clone().collect()),
        ("filling_target", geom.filling_target.clone().collect()),
        (
            "filled_min_storage_floor",
            geom.filled_min_storage_floor.clone().collect(),
        ),
        (
            "evaporation",
            geom.evap_indices.iter().map(|e| e.evap_row).collect(),
        ),
        ("z_inflow", state.z_inflow_rows().collect()),
    ]
}

/// One [`DeliveryRing`] lane's out/in column runs, decoded by [`ring_lanes`].
pub struct RingLane {
    /// This lane's ring identity.
    pub kind: RingLaneKind,
    /// Outgoing-block columns, slot order.
    pub out_cols: Vec<usize>,
    /// Incoming-block columns, slot order.
    pub in_cols: Vec<usize>,
    /// The anticipated lane's own decision column; `None` for a water lane.
    pub decision_col: Option<usize>,
}

/// [`RingLane`]'s ring identity: an anticipated-decision lane, by lane index,
/// or a water-transit-bucket lane, by plant.
pub enum RingLaneKind {
    /// An anticipated-decision ring lane.
    Anticipated {
        /// The lane's index in `0..state.n_anticipated`.
        lane: usize,
    },
    /// A water-transit-bucket ring lane.
    Water {
        /// The plant the bucket ring belongs to.
        plant: HydroSys,
    },
}

/// Decodes every [`DeliveryRing`] lane at this stage: one [`RingLane`] per
/// [`DeliveryRing::anticipated`] lane, and one per
/// [`DeliveryRing::transit_buckets`] plant. Every column comes from the
/// ring's own [`DeliveryRing::out_col`]/[`DeliveryRing::in_col`].
#[must_use]
pub fn ring_lanes(state: &StateSpace, geom: &StageGeometry) -> Vec<RingLane> {
    let mut lanes = Vec::new();

    let anticipated = DeliveryRing::anticipated(state);
    for lane in 0..state.n_anticipated {
        lanes.push(RingLane {
            kind: RingLaneKind::Anticipated { lane },
            out_cols: (0..state.k_max)
                .map(|slot| anticipated.out_col(slot, lane))
                .collect(),
            in_cols: (0..state.k_max)
                .map(|slot| anticipated.in_col(slot, lane))
                .collect(),
            decision_col: Some(geom.anticipated_decision_col(AnticipatedLocal::new(lane))),
        });
    }

    for bucket in DeliveryRing::transit_buckets(state) {
        let depth = bucket.local.len();
        lanes.push(RingLane {
            kind: RingLaneKind::Water {
                plant: bucket.plant,
            },
            out_cols: (0..depth)
                .map(|slot| bucket.ring.out_col(slot, 0))
                .collect(),
            in_cols: (0..depth).map(|slot| bucket.ring.in_col(slot, 0)).collect(),
            decision_col: None,
        });
    }

    lanes
}

/// `hours * M3S_TO_HM3`, so no test declares the conversion constant again.
#[must_use]
pub fn hours_to_hm3(hours: f64) -> f64 {
    hours * crate::block_clock::M3S_TO_HM3
}

/// Decoded owner of a water-balance, load-balance or z-inflow row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowOwner {
    /// A water-balance row: hydro `hydro`'s row for block `blk`.
    Water {
        /// The row's owning hydro.
        hydro: HydroSys,
        /// The row's block.
        blk: BlockIdx,
    },
    /// A load-balance row: bus `bus`'s row for block `blk`.
    Load {
        /// The row's owning bus.
        bus: BusSys,
        /// The row's block.
        blk: BlockIdx,
    },
    /// Hydro `hydro`'s z-inflow definition row.
    ZInflow {
        /// The row's owning hydro.
        hydro: HydroSys,
    },
}

/// Inverts [`StageGeometry::water_balance_row`], [`StageGeometry::load_balance_row`]
/// and [`StateSpace::z_inflow_row`] over the `System`'s own entity counts and
/// `0..family.rows_per_entity(geom.n_blks)`.
#[must_use]
pub fn row_owners(
    system: &System,
    geom: &StageGeometry,
    state: &StateSpace,
) -> HashMap<usize, RowOwner> {
    let mut owners = HashMap::new();
    for h in 0..system.hydros().len() {
        let hydro = HydroSys::new(h);
        for blk in 0..geom.water_balance.rows_per_entity(geom.n_blks) {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.water_balance_row(hydro, blk),
                RowOwner::Water { hydro, blk },
            );
        }
    }
    for b in 0..system.buses().len() {
        let bus = BusSys::new(b);
        for blk in 0..geom.load_balance.rows_per_entity(geom.n_blks) {
            let blk = BlockIdx::new(blk);
            owners.insert(geom.load_balance_row(bus, blk), RowOwner::Load { bus, blk });
        }
    }
    for h in 0..system.hydros().len() {
        let hydro = HydroSys::new(h);
        owners.insert(state.z_inflow_row(hydro), RowOwner::ZInflow { hydro });
    }
    owners
}

/// Pass-through to [`StateSpace::z_inflow_col`], so no decoder re-derives
/// the z-inflow column's address itself.
#[must_use]
pub fn z_inflow_column(state: &StateSpace, h: HydroSys) -> usize {
    state.z_inflow_col(h).get()
}

/// Decoded owner of a column that can enter a water, load or z-inflow row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColOwner {
    /// A storage-boundary column for hydro `hydro` at `boundary`.
    Storage {
        /// The column's owning hydro.
        hydro: HydroSys,
        /// The storage boundary this column addresses.
        boundary: Boundary,
    },
    /// Hydro `hydro`'s realized-inflow column.
    ZInflow {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// An AR inflow-lag column for hydro `hydro` (any lag depth).
    InflowLag {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// A water-transit-bucket ring column for plant `plant`, ring slot `slot`.
    Bucket {
        /// The bucket ring's owning plant.
        plant: HydroSys,
        /// The slot within the plant's own bucket-ring run.
        slot: usize,
        /// `true` for the outgoing column, `false` for the incoming one.
        outgoing: bool,
    },
    /// A turbine-flow column for hydro-cell `cell` (owned by `hydro`), block `blk`.
    Turbine {
        /// The cell's owning plant.
        hydro: HydroSys,
        /// The turbine column's cell.
        cell: HydroCell,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A spillage column for hydro `hydro`, block `blk`.
    Spillage {
        /// The column's owning hydro.
        hydro: HydroSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A diversion-flow column for hydro `hydro`, block `blk`.
    Diversion {
        /// The column's owning hydro.
        hydro: HydroSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A pumping-flow column for station `station`, block `blk`.
    Pumping {
        /// The column's owning pumping station.
        station: PumpingSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// An evaporation-flow column for hydro `hydro`, evaporation slot `slot`.
    Evaporation {
        /// The column's owning hydro.
        hydro: HydroSys,
        /// The evaporation slot: stage-level on a parallel stage, per-block
        /// on a chronological one ([`evaporation_slot_count`]).
        slot: usize,
    },
    /// Hydro `hydro`'s inflow non-negativity slack column.
    InflowSlack {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// Hydro `hydro`'s below-withdrawal-target slack column.
    WithdrawalNeg {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// Hydro `hydro`'s above-withdrawal-target slack column.
    WithdrawalPos {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// A thermal-generation column for thermal `thermal`, block `blk`.
    Thermal {
        /// The column's owning thermal.
        thermal: ThermalSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A bus-deficit column for bus `bus`, block `blk` (any segment).
    Deficit {
        /// The column's owning bus.
        bus: BusSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A bus-excess column for bus `bus`, block `blk`.
    Excess {
        /// The column's owning bus.
        bus: BusSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A non-controllable-source generation column for `ncs`, block `blk`.
    Ncs {
        /// The column's owning non-controllable source.
        ncs: NcsSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A forward line-flow column for line `line`, block `blk`.
    LineFwd {
        /// The column's owning line.
        line: LineSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A reverse line-flow column for line `line`, block `blk`.
    LineRev {
        /// The column's owning line.
        line: LineSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// An FPHA-generation column for hydro-cell `cell`, block `blk`.
    Generation {
        /// The generation column's cell.
        cell: HydroCell,
        /// The column's block.
        blk: BlockIdx,
    },
    /// An in-study anticipated-ring incoming column: ring lane `lane`
    /// (anticipated-local index), slot `slot`.
    AnticipatedIn {
        /// The ring lane's anticipated-local index.
        lane: usize,
        /// The ring slot.
        slot: usize,
    },
    /// A contract column of direction `contract_type`, per-direction slot
    /// `family_slot`, block `blk`.
    Contract {
        /// The contract's direction.
        contract_type: ContractType,
        /// The contract's per-direction slot.
        family_slot: usize,
        /// The column's block.
        blk: BlockIdx,
    },
}

/// Inverts every column family that can enter a water, load or z-inflow row.
/// [`storage_column_owners`] and [`ring_lanes`] supply the state-ring
/// families; every other family is read through its own [`StageGeometry`]
/// accessor, never `start + i`. `HydroCellIndex` is built once, here, and
/// inverted over every hydro to recover a cell's owning plant — the study's
/// own partition, never a second one (`HydroCellIndex` has no `plant_of`).
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one insertion loop per column family, each a single accessor call; splitting would fragment the family list with no reuse benefit"
)]
pub fn column_owners(
    system: &System,
    geom: &StageGeometry,
    state: &StateSpace,
) -> HashMap<usize, ColOwner> {
    let mut owners = HashMap::new();

    let cell_index = HydroCellIndex::build(system.hydros());
    let mut plant_of = vec![HydroSys::new(0); cell_index.n_cells()];
    for h in 0..state.hydro_count {
        let hydro = HydroSys::new(h);
        for c in cell_index.cells_of(hydro) {
            plant_of[c] = hydro;
        }
    }

    for (&col, &(hydro, boundary)) in &storage_column_owners(geom, state) {
        owners.insert(col, ColOwner::Storage { hydro, boundary });
    }

    for h in 0..state.hydro_count {
        let hydro = HydroSys::new(h);
        owners.insert(z_inflow_column(state, hydro), ColOwner::ZInflow { hydro });
    }
    for lag in 0..state.max_par_order {
        for h in 0..state.hydro_count {
            let hydro = HydroSys::new(h);
            owners.insert(
                state.lag_incoming_col(lag, hydro).get(),
                ColOwner::InflowLag { hydro },
            );
        }
    }

    for lane in ring_lanes(state, geom) {
        match lane.kind {
            RingLaneKind::Water { plant } => {
                for (slot, &col) in lane.out_cols.iter().enumerate() {
                    owners.insert(
                        col,
                        ColOwner::Bucket {
                            plant,
                            slot,
                            outgoing: true,
                        },
                    );
                }
                for (slot, &col) in lane.in_cols.iter().enumerate() {
                    owners.insert(
                        col,
                        ColOwner::Bucket {
                            plant,
                            slot,
                            outgoing: false,
                        },
                    );
                }
            }
            RingLaneKind::Anticipated { lane: lane_idx } => {
                for (slot, &col) in lane.in_cols.iter().enumerate() {
                    owners.insert(
                        col,
                        ColOwner::AnticipatedIn {
                            lane: lane_idx,
                            slot,
                        },
                    );
                }
            }
        }
    }

    for (c, &hydro) in plant_of.iter().enumerate() {
        let cell = HydroCell::new(c);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.turbine_col(cell, blk),
                ColOwner::Turbine { hydro, cell, blk },
            );
        }
    }

    for h in 0..state.hydro_count {
        let hydro = HydroSys::new(h);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.spillage_col(hydro, blk),
                ColOwner::Spillage { hydro, blk },
            );
            owners.insert(
                geom.diversion_col(hydro, blk),
                ColOwner::Diversion { hydro, blk },
            );
        }
        if !geom.inflow_slack.is_empty() {
            owners.insert(
                geom.inflow_slack_col(hydro),
                ColOwner::InflowSlack { hydro },
            );
        }
        owners.insert(
            geom.withdrawal_slack_neg_col(hydro),
            ColOwner::WithdrawalNeg { hydro },
        );
        owners.insert(
            geom.withdrawal_slack_pos_col(hydro),
            ColOwner::WithdrawalPos { hydro },
        );
    }

    let n_evap_slots = evaporation_slot_count(geom.block_mode, geom.n_blks);
    for (local_idx, &hydro) in geom.evap_hydro_indices.iter().enumerate() {
        for slot in 0..n_evap_slots {
            let evap = &geom.evap_indices[local_idx * n_evap_slots + slot];
            owners.insert(
                evap.evaporation_flow_col,
                ColOwner::Evaporation { hydro, slot },
            );
        }
    }

    for t in 0..system.thermals().len() {
        let thermal = ThermalSys::new(t);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.thermal_col(thermal, blk),
                ColOwner::Thermal { thermal, blk },
            );
        }
    }

    let max_deficit_segments = system
        .buses()
        .iter()
        .map(|b| b.deficit_segments.len())
        .max()
        .unwrap_or(0);
    for b in 0..system.buses().len() {
        let bus = BusSys::new(b);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            for seg in 0..max_deficit_segments {
                owners.insert(
                    geom.deficit_col(bus, seg, blk, max_deficit_segments),
                    ColOwner::Deficit { bus, blk },
                );
            }
            owners.insert(geom.excess_col(bus, blk), ColOwner::Excess { bus, blk });
        }
    }

    for n in 0..system.non_controllable_sources().len() {
        let ncs = NcsSys::new(n);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.ncs_generation_col(ncs, blk),
                ColOwner::Ncs { ncs, blk },
            );
        }
    }

    for p in 0..system.pumping_stations().len() {
        let station = PumpingSys::new(p);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.pumping_flow_col(station, blk),
                ColOwner::Pumping { station, blk },
            );
        }
    }

    for l in 0..system.lines().len() {
        let line = LineSys::new(l);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.line_fwd_col(line, blk),
                ColOwner::LineFwd { line, blk },
            );
            owners.insert(
                geom.line_rev_col(line, blk),
                ColOwner::LineRev { line, blk },
            );
        }
    }

    for (local_idx, &hydro) in geom.fpha_hydro_indices.iter().enumerate() {
        let cell_base: usize = geom.fpha_hydro_indices[..local_idx]
            .iter()
            .map(|&h| cell_index.cells_of(h).len())
            .sum();
        for (offset, c) in cell_index.cells_of(hydro).enumerate() {
            let cell = HydroCell::new(c);
            let cell_local = FphaCellLocal::new(cell_base + offset);
            for blk in 0..geom.n_blks {
                let blk = BlockIdx::new(blk);
                owners.insert(
                    geom.generation_col(cell_local, blk),
                    ColOwner::Generation { cell, blk },
                );
            }
        }
    }

    for c_sys in 0..system.contracts().len() {
        let (contract_type, family_slot) = contract_family_slot(system.contracts(), c_sys);
        for blk in 0..geom.n_blks {
            let blk = BlockIdx::new(blk);
            owners.insert(
                geom.contract_col(contract_type, family_slot, blk),
                ColOwner::Contract {
                    contract_type,
                    family_slot,
                    blk,
                },
            );
        }
    }

    owners
}

/// [`ColOwner`] collapsed to the identity a block-mode flip preserves:
/// identical to [`ColOwner`] in every other variant, but `Evaporation {
/// hydro, slot }` collapses to `Evaporation { hydro }` because the slot
/// count differs by mode (stage-level on a parallel stage, per-block on a
/// chronological one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColKey {
    /// A storage-boundary column for hydro `hydro` at `boundary`.
    Storage {
        /// The column's owning hydro.
        hydro: HydroSys,
        /// The storage boundary this column addresses.
        boundary: Boundary,
    },
    /// Hydro `hydro`'s realized-inflow column.
    ZInflow {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// An AR inflow-lag column for hydro `hydro` (any lag depth).
    InflowLag {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// A water-transit-bucket ring column for plant `plant`, ring slot `slot`.
    Bucket {
        /// The bucket ring's owning plant.
        plant: HydroSys,
        /// The slot within the plant's own bucket-ring run.
        slot: usize,
        /// `true` for the outgoing column, `false` for the incoming one.
        outgoing: bool,
    },
    /// A turbine-flow column for hydro-cell `cell` (owned by `hydro`), block `blk`.
    Turbine {
        /// The cell's owning plant.
        hydro: HydroSys,
        /// The turbine column's cell.
        cell: HydroCell,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A spillage column for hydro `hydro`, block `blk`.
    Spillage {
        /// The column's owning hydro.
        hydro: HydroSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A diversion-flow column for hydro `hydro`, block `blk`.
    Diversion {
        /// The column's owning hydro.
        hydro: HydroSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A pumping-flow column for station `station`, block `blk`.
    Pumping {
        /// The column's owning pumping station.
        station: PumpingSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// An evaporation-flow column for hydro `hydro`, any evaporation slot.
    Evaporation {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// Hydro `hydro`'s inflow non-negativity slack column.
    InflowSlack {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// Hydro `hydro`'s below-withdrawal-target slack column.
    WithdrawalNeg {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// Hydro `hydro`'s above-withdrawal-target slack column.
    WithdrawalPos {
        /// The column's owning hydro.
        hydro: HydroSys,
    },
    /// A thermal-generation column for thermal `thermal`, block `blk`.
    Thermal {
        /// The column's owning thermal.
        thermal: ThermalSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A bus-deficit column for bus `bus`, block `blk` (any segment).
    Deficit {
        /// The column's owning bus.
        bus: BusSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A bus-excess column for bus `bus`, block `blk`.
    Excess {
        /// The column's owning bus.
        bus: BusSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A non-controllable-source generation column for `ncs`, block `blk`.
    Ncs {
        /// The column's owning non-controllable source.
        ncs: NcsSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A forward line-flow column for line `line`, block `blk`.
    LineFwd {
        /// The column's owning line.
        line: LineSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// A reverse line-flow column for line `line`, block `blk`.
    LineRev {
        /// The column's owning line.
        line: LineSys,
        /// The column's block.
        blk: BlockIdx,
    },
    /// An FPHA-generation column for hydro-cell `cell`, block `blk`.
    Generation {
        /// The generation column's cell.
        cell: HydroCell,
        /// The column's block.
        blk: BlockIdx,
    },
    /// An in-study anticipated-ring incoming column: ring lane `lane`
    /// (anticipated-local index), slot `slot`.
    AnticipatedIn {
        /// The ring lane's anticipated-local index.
        lane: usize,
        /// The ring slot.
        slot: usize,
    },
    /// A contract column of direction `contract_type`, per-direction slot
    /// `family_slot`, block `blk`.
    Contract {
        /// The contract's direction.
        contract_type: ContractType,
        /// The contract's per-direction slot.
        family_slot: usize,
        /// The column's block.
        blk: BlockIdx,
    },
}

impl From<ColOwner> for ColKey {
    fn from(owner: ColOwner) -> Self {
        match owner {
            ColOwner::Storage { hydro, boundary } => ColKey::Storage { hydro, boundary },
            ColOwner::ZInflow { hydro } => ColKey::ZInflow { hydro },
            ColOwner::InflowLag { hydro } => ColKey::InflowLag { hydro },
            ColOwner::Bucket {
                plant,
                slot,
                outgoing,
            } => ColKey::Bucket {
                plant,
                slot,
                outgoing,
            },
            ColOwner::Turbine { hydro, cell, blk } => ColKey::Turbine { hydro, cell, blk },
            ColOwner::Spillage { hydro, blk } => ColKey::Spillage { hydro, blk },
            ColOwner::Diversion { hydro, blk } => ColKey::Diversion { hydro, blk },
            ColOwner::Pumping { station, blk } => ColKey::Pumping { station, blk },
            ColOwner::Evaporation { hydro, .. } => ColKey::Evaporation { hydro },
            ColOwner::InflowSlack { hydro } => ColKey::InflowSlack { hydro },
            ColOwner::WithdrawalNeg { hydro } => ColKey::WithdrawalNeg { hydro },
            ColOwner::WithdrawalPos { hydro } => ColKey::WithdrawalPos { hydro },
            ColOwner::Thermal { thermal, blk } => ColKey::Thermal { thermal, blk },
            ColOwner::Deficit { bus, blk } => ColKey::Deficit { bus, blk },
            ColOwner::Excess { bus, blk } => ColKey::Excess { bus, blk },
            ColOwner::Ncs { ncs, blk } => ColKey::Ncs { ncs, blk },
            ColOwner::LineFwd { line, blk } => ColKey::LineFwd { line, blk },
            ColOwner::LineRev { line, blk } => ColKey::LineRev { line, blk },
            ColOwner::Generation { cell, blk } => ColKey::Generation { cell, blk },
            ColOwner::AnticipatedIn { lane, slot } => ColKey::AnticipatedIn { lane, slot },
            ColOwner::Contract {
                contract_type,
                family_slot,
                blk,
            } => ColKey::Contract {
                contract_type,
                family_slot,
                blk,
            },
        }
    }
}

impl PartialOrd for ColKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ColKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // No field type here derives `Ord`; compare the derived `Debug`
        // rendering instead — a total order with no per-field wrapper type.
        format!("{self:?}").cmp(&format!("{other:?}"))
    }
}

/// Row `r`'s unscaled `(column, value)` entries, keyed by [`ColKey`] and
/// summed: several columns one owner splits across — a chronological row's
/// several block columns, a storage boundary's two columns — collapse to
/// one coefficient.
///
/// # Panics
/// Panics if a column entry in row `r` has no owner in `cols`.
#[must_use]
#[expect(
    clippy::implicit_hasher,
    reason = "cols is always column_owners()'s RandomState HashMap; no call site needs a generic hasher"
)]
pub fn row_by_owner(
    m: &UnscaledMatrix,
    cols: &HashMap<usize, ColOwner>,
    r: usize,
) -> BTreeMap<ColKey, f64> {
    let mut keyed: BTreeMap<ColKey, f64> = BTreeMap::new();
    for &(c, v) in m.row(r) {
        #[expect(
            clippy::panic,
            reason = "an undecoded column on a water/load/z/FPHA row is a gap in column_owners's coverage, not a runtime condition to recover from"
        )]
        let owner = *cols
            .get(&c)
            .unwrap_or_else(|| panic!("row {r} has an undecoded column {c}"));
        *keyed.entry(ColKey::from(owner)).or_insert(0.0) += v;
    }
    keyed
}
