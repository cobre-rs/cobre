//! Template-family structural decoders: inverses of the address owners
//! (`StageGeometry`, `StateSpace`, `DeliveryRing`), read by
//! `tests/template_family_structure.rs`. The owners they invert are
//! `pub(crate)`, so the decoders live here rather than in an integration-test
//! crate, which could only re-derive their arithmetic.

use std::collections::HashMap;

use cobre_core::BlockMode;
use cobre_solver::StageTemplate;

use crate::indexer::{AnticipatedLocal, BlockGrid, BlockIdx, Boundary, FphaCellLocal, HydroSys};
use crate::lp::builder::{DeliveryRing, StageGeometry};
use crate::lp::indexer::StateSpace;

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
    for h in 0..state.hydro_count {
        let h = HydroSys::new(h);
        let boundaries: Vec<Boundary> = match geom.block_mode {
            BlockMode::Parallel => vec![Boundary::Incoming, Boundary::Outgoing],
            BlockMode::Chronological => (0..=geom.n_blks)
                .map(|k| Boundary::from_index(k, geom.n_blks))
                .collect(),
        };
        for b in boundaries {
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
