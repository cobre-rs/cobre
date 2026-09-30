//! Generic lagged-delivery ring primitive shared by the travel-time
//! water-bucket ring and the anticipated-thermal ring: a borrowed outgoing
//! block (identity-resolved) and a paired incoming block (pinned) advance one
//! Markov-1 slot per stage through the same interior shift-row skeleton and
//! paired row-cap/column-freeze masking. The two rings differ in how each
//! deposits into its newest slot and in what a masked terminal slot means —
//! both differences live at each ring's own call site, never a second
//! skeleton implementation.
//!
//! [`StateSpace`] remains the sole owner of the
//! out/in state-index ranges: a [`DeliveryRing`] borrows them for one
//! construction and never re-derives or persists an independent copy. The
//! block-mode-coupled per-lag deposit fill stays at each ring's own call
//! site; this module owns only the shared skeleton, plus the column-bound
//! buffers the ring's column-freeze writes — the anticipated ring's
//! delivery-axis → ring-slot map is owned by
//! [`crate::lp::indexer::for_each_ring_residue`].

use std::ops::Range;

use cobre_core::EntityId;

use crate::block_clock::BlockClock;
use crate::indexer::{BlockIdx, HydroSys, StateSpace};

use super::layout::{TemplateBuildCtx, position_table_row};

/// Mutable column-bound and objective buffers shared by all fill helpers.
pub(super) struct ColumnBufs<'a> {
    pub(super) col_lower: &'a mut [f64],
    pub(super) col_upper: &'a mut [f64],
    pub(super) objective: &'a mut [f64],
}

/// A lagged-delivery ring over one dense, slot-major/lane-minor state-column
/// grid: `n_lanes` parallel delivery lanes (plants), each `depth` slots deep.
/// Borrows its outgoing/incoming column blocks from
/// [`StateSpace`] — never a private copy of the
/// ranges.
#[derive(Debug, Clone)]
pub struct DeliveryRing {
    /// Outgoing (identity-resolved) column block, size `n_lanes * depth`.
    out_block: Range<usize>,
    /// Incoming (pinned) column block, size `n_lanes * depth`.
    in_block: Range<usize>,
    /// Delivery lanes (plants).
    n_lanes: usize,
    /// Slots per lane.
    depth: usize,
}

/// One downstream plant's water-bucket ring, its local sub-range (relative to
/// [`StateSpace::transit_buckets_out`]/[`StateSpace::transit_buckets_in`]'s
/// own start), and the plant it belongs to — [`DeliveryRing::transit_buckets`]'s
/// and [`DeliveryRing::transit_bucket`]'s item type.
pub(crate) struct PlantBucketRing {
    pub(crate) plant: HydroSys,
    pub(crate) local: Range<usize>,
    pub(crate) ring: DeliveryRing,
}

impl DeliveryRing {
    /// Constructs a ring over the borrowed out/in state blocks.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `out_block.len()` or `in_block.len()` differs from
    /// `n_lanes * depth`.
    #[must_use]
    fn new(out_block: Range<usize>, in_block: Range<usize>, n_lanes: usize, depth: usize) -> Self {
        let dense_len = n_lanes * depth;
        debug_assert_eq!(
            out_block.len(),
            dense_len,
            "out_block must be sized n_lanes * depth ({dense_len}), got {}",
            out_block.len()
        );
        debug_assert_eq!(
            in_block.len(),
            dense_len,
            "in_block must be sized n_lanes * depth ({dense_len}), got {}",
            in_block.len()
        );
        Self {
            out_block,
            in_block,
            n_lanes,
            depth,
        }
    }

    /// The in-study commitment-hold ring (`n_lanes = n_anticipated`,
    /// slot-major/plant-minor, `depth = k_max`, modular-addressed) every
    /// anticipated call site shares — the single owner of its out/in block
    /// construction. Borrows the merged [`StateSpace::commit_out`]/
    /// [`StateSpace::commit_in`] region.
    #[must_use]
    pub fn anticipated(state: &StateSpace) -> Self {
        Self::new(
            state.commit_out.clone(),
            state.commit_in.clone(),
            state.n_anticipated,
            state.k_max,
        )
    }

    /// Every downstream plant's own water-bucket ring (`n_lanes = 1`), over
    /// its contiguous local sub-range in [`StateSpace::transit_bucket_plants`]
    /// order — the single owner of the ragged-to-dense addressing every
    /// bucket call site shares.
    pub(crate) fn transit_buckets(
        state: &StateSpace,
    ) -> impl Iterator<Item = PlantBucketRing> + '_ {
        state
            .transit_bucket_plants()
            .map(|(plant, local)| PlantBucketRing {
                plant,
                ring: Self::new(
                    state.bucket_outgoing_block(local.clone()),
                    state.bucket_incoming_block(local.clone()),
                    1,
                    local.len(),
                ),
                local,
            })
    }

    /// `plant`'s own [`PlantBucketRing`], or `None` when it declares no
    /// incoming arc.
    pub(crate) fn transit_bucket(state: &StateSpace, plant: HydroSys) -> Option<PlantBucketRing> {
        Self::transit_buckets(state).find(|bucket| bucket.plant == plant)
    }

    /// Outgoing-block column for ring position `(slot, lane)` — the single
    /// owner of the ring's addressing arithmetic.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= depth` or `lane >= n_lanes`.
    #[must_use]
    pub fn out_col(&self, slot: usize, lane: usize) -> usize {
        debug_assert!(
            slot < self.depth,
            "slot {slot} must be < depth {}",
            self.depth
        );
        debug_assert!(
            lane < self.n_lanes,
            "lane {lane} must be < n_lanes {}",
            self.n_lanes
        );
        self.out_block.start + slot * self.n_lanes + lane
    }

    /// Incoming-block column for ring position `(slot, lane)`; see
    /// [`Self::out_col`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= depth` or `lane >= n_lanes`.
    #[must_use]
    pub fn in_col(&self, slot: usize, lane: usize) -> usize {
        debug_assert!(
            slot < self.depth,
            "slot {slot} must be < depth {}",
            self.depth
        );
        debug_assert!(
            lane < self.n_lanes,
            "lane {lane} must be < n_lanes {}",
            self.n_lanes
        );
        self.in_block.start + slot * self.n_lanes + lane
    }

    /// Decomposes a flat block-relative offset (`col − out_block.start` or
    /// `col − in_block.start`) into `(slot, lane)` — the exact inverse of
    /// [`Self::out_col`]/[`Self::in_col`]'s slot-major, lane-minor addressing.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `offset >= n_lanes * depth`.
    #[must_use]
    pub(crate) fn slot_lane_at(&self, offset: usize) -> (usize, usize) {
        debug_assert!(
            offset < self.n_lanes * self.depth,
            "offset {offset} must be < n_lanes*depth {}",
            self.n_lanes * self.depth
        );
        (offset / self.n_lanes, offset % self.n_lanes)
    }

    /// Emits the ring's interior shift-row CSC entries: `out[slot] +1`, and
    /// `in[slot+1] −1` when a deeper ring position exists for the same lane
    /// (`slot + 1 < depth`). `row_pos[slot * n_lanes + lane]` gives this
    /// stage's compact row position, `None` when masked — the paired column
    /// freeze for a masked position lives in [`Self::freeze_masked_columns`],
    /// never here. Returns the count of rows emitted.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `row_pos.len() != n_lanes * depth`.
    pub fn emit_shift_rows(
        &self,
        row_pos: &[Option<usize>],
        row_start: usize,
        col_entries: &mut [Vec<(usize, f64)>],
    ) -> usize {
        if self.n_lanes == 0 || self.depth == 0 {
            debug_assert!(
                row_pos.is_empty(),
                "row_pos must be empty when n_lanes or depth is 0"
            );
            return 0;
        }
        debug_assert_eq!(
            row_pos.len(),
            self.n_lanes * self.depth,
            "row_pos must be sized n_lanes * depth (dense, slot-major, lane-minor)"
        );
        let mut n_reachable = 0_usize;
        for flat in 0..row_pos.len() {
            let Some(row) = position_table_row(row_start, row_pos, flat) else {
                continue;
            };
            let (slot, lane) = self.slot_lane_at(flat);
            col_entries[self.out_col(slot, lane)].push((row, 1.0));
            if slot + 1 < self.depth {
                col_entries[self.in_col(slot + 1, lane)].push((row, -1.0));
            }
            n_reachable += 1;
        }
        n_reachable
    }

    /// Emits the ring's interior carry-row CSC entries: `out[slot] +1` and
    /// `in[slot] −1` at the SAME slot — the same-slot hold identity that pins
    /// `out − in = 0`, unlike [`Self::emit_shift_rows`]'s `slot + 1` target.
    /// `row_pos[slot * n_lanes + lane]` gives this stage's compact row
    /// position, `None` when masked — the paired column freeze lives in
    /// [`Self::freeze_masked_columns`], never here. Returns the count of rows
    /// emitted.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `row_pos.len() != n_lanes * depth`.
    pub fn emit_carry_rows(
        &self,
        row_pos: &[Option<usize>],
        row_start: usize,
        col_entries: &mut [Vec<(usize, f64)>],
    ) -> usize {
        if self.n_lanes == 0 || self.depth == 0 {
            debug_assert!(
                row_pos.is_empty(),
                "row_pos must be empty when n_lanes or depth is 0"
            );
            return 0;
        }
        debug_assert_eq!(
            row_pos.len(),
            self.n_lanes * self.depth,
            "row_pos must be sized n_lanes * depth (dense, slot-major, lane-minor)"
        );
        let mut n_reachable = 0_usize;
        for flat in 0..row_pos.len() {
            let Some(row) = position_table_row(row_start, row_pos, flat) else {
                continue;
            };
            let (slot, lane) = self.slot_lane_at(flat);
            col_entries[self.out_col(slot, lane)].push((row, 1.0));
            col_entries[self.in_col(slot, lane)].push((row, -1.0));
            n_reachable += 1;
        }
        n_reachable
    }

    /// Freezes every masked ring position (`row_pos[i] == None`) to `[0, 0]`
    /// and leaves every reachable position at `reachable_bound` — the paired
    /// column half of the masking contract [`Self::emit_shift_rows`]
    /// discharges the row half for. `reachable_bound` is the ring's open
    /// default (water's implicit `[0, inf)`, anticipated's signed
    /// `(-inf, inf)`); the masked bound is always `[0, 0]` regardless of
    /// `reachable_bound` — scale-independent, so no column bound is ever
    /// rescaled to share the freeze between rings.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `row_pos.len() != n_lanes * depth`.
    pub(super) fn freeze_masked_columns(
        &self,
        row_pos: &[Option<usize>],
        reachable_bound: (f64, f64),
        bufs: &mut ColumnBufs<'_>,
    ) {
        debug_assert_eq!(
            row_pos.len(),
            self.n_lanes * self.depth,
            "row_pos must be sized n_lanes * depth (dense, slot-major, lane-minor)"
        );
        let (reachable_lower, reachable_upper) = reachable_bound;
        for (offset, pos) in row_pos.iter().enumerate() {
            let (slot, lane) = self.slot_lane_at(offset);
            let col = self.out_col(slot, lane);
            if pos.is_some() {
                bufs.col_lower[col] = reachable_lower;
                bufs.col_upper[col] = reachable_upper;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
        }
    }

    /// Pins the ring's outgoing column at `(slot, lane)` to `decision_col`
    /// (`+1` on `out_col(slot, lane)`, `−1` on `decision_col`) at `row` — the
    /// anticipated ring's deposit. The water ring's block-mode-coupled
    /// per-lag deposit share is emitted at its own call site
    /// (`fill_arc_release_block_entries`), never through this function.
    pub(crate) fn emit_deposit(
        &self,
        slot: usize,
        lane: usize,
        row: usize,
        decision_col: usize,
        col_entries: &mut [Vec<(usize, f64)>],
    ) {
        col_entries[self.out_col(slot, lane)].push((row, 1.0));
        col_entries[decision_col].push((row, -1.0));
    }

    /// Ring slot targeted by lane `lane`'s `lag`-th deposit (`lag >= 1`;
    /// `lag == 0` is the same-stage share, never a ring slot), returned as a
    /// flat `row_pos`/column index (`(lag - 1) * n_lanes + lane`) the caller
    /// resolves against its own per-stage reachability table.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `lag == 0`, the resulting slot is `>= depth`, or
    /// `lane >= n_lanes`.
    #[must_use]
    pub(crate) fn slot_target(&self, lane: usize, lag: usize) -> usize {
        debug_assert!(
            lag >= 1,
            "lag must be >= 1; lag 0 is the same-stage share, never a ring slot"
        );
        let slot = lag - 1;
        debug_assert!(
            slot < self.depth,
            "slot {slot} must be < depth {}",
            self.depth
        );
        debug_assert!(
            lane < self.n_lanes,
            "lane {lane} must be < n_lanes {}",
            self.n_lanes
        );
        slot * self.n_lanes + lane
    }
}

/// The maturing-now bucket's incoming column (`in_col(0, 0)`) for downstream
/// `plant`, or `None` when it declares no incoming arc. The single owner of
/// [`DeliveryRing::transit_bucket`]'s `in_col(0, 0)` read both water-balance
/// fills, and the generic-constraint `hydro_inflow` resolver, share.
pub(super) fn maturing_bucket_in_col(state: &StateSpace, plant: HydroSys) -> Option<usize> {
    DeliveryRing::transit_bucket(state, plant).map(|bucket| bucket.ring.in_col(0, 0))
}

/// Resolve this stage's incoming maturing bucket `arrival_density` (fixed-delivery-density
/// contract): a lookup of the setup-precomputed per-`(arc, arrival stage)` blend
/// ([`build_arc_arrival_density`](crate::bucket_topology::build_arc_arrival_density)),
/// already resolved in this arrival stage's own frame. Falls back to duration-weighted
/// uniform only where the table holds no blend (the study's first stage) or the plant has
/// no travel-time upstream.
///
/// A non-travel-time upstream is EXCLUDED, never folded in via `uniform`: it would
/// disagree with the sole travel-time arc's non-uniform density — a false
/// heterogeneous-confluence panic in debug, a silent uniform split in release.
///
/// A heterogeneous-density confluence has no resolved policy;
/// `check_chronological_confluence_heterogeneous_travel_time` (`cobre-io`) rejects it at
/// config time, so the `debug_assert!` below is a defensive backstop, not the enforcement
/// point.
pub(super) fn resolve_bucket_arrival_density(
    ctx: &TemplateBuildCtx<'_>,
    clock: BlockClock<'_>,
    stage_idx: usize,
    downstream_id: EntityId,
    n_blks: usize,
) -> Vec<f64> {
    let uniform = || {
        (0..n_blks)
            .map(|b| clock.hours(BlockIdx::new(b)) / clock.total_hours())
            .collect::<Vec<f64>>()
    };

    let mut chosen: Option<Vec<f64>> = None;
    for &up_id in ctx.cascade.upstream(downstream_id) {
        let Some(u_idx) = ctx.positions.hydro(up_id) else {
            continue;
        };
        let Some(by_stage) = ctx.topology.arc_arrival_density.get(&u_idx) else {
            continue;
        };
        let candidate = by_stage[stage_idx].clone().map_or_else(uniform, |density| {
            debug_assert_eq!(
                density.len(),
                n_blks,
                "arc {u_idx} stage {stage_idx}: arrival_density length must equal n_blks"
            );
            density
        });
        match &chosen {
            None => chosen = Some(candidate),
            Some(existing) => {
                debug_assert!(
                    existing.len() == candidate.len()
                        && existing
                            .iter()
                            .zip(&candidate)
                            .all(|(&a, &b)| (a - b).abs() < 1e-9),
                    "confluence with heterogeneous chronological delivery densities into \
                     one downstream plant is not yet supported (arc {u_idx} disagrees at \
                     stage {stage_idx})"
                );
            }
        }
    }
    chosen.unwrap_or_else(uniform)
}

#[cfg(test)]
mod tests {
    use super::{ColumnBufs, DeliveryRing, PlantBucketRing};
    use crate::indexer::HydroSys;

    /// One ring instance per lane (`n_lanes = 1`), mirroring the water ring's
    /// per-plant contiguous addressing: lane A is 3 slots deep (every
    /// interior slot has a deeper neighbor except the last), lane B is 1 slot
    /// deep (its only slot has no deeper neighbor at all). Hand-computed
    /// against the ring's own addressing formula, never against
    /// `fill_transit_bucket_definition_entries`'s output.
    #[test]
    fn emit_shift_rows_drops_the_shift_term_past_a_lanes_own_depth() {
        let ring_a = DeliveryRing::new(100..103, 200..203, 1, 3);
        let row_pos_a = vec![Some(0), Some(1), Some(2)];
        let mut col_entries_a: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 210];
        let n_a = ring_a.emit_shift_rows(&row_pos_a, 900, &mut col_entries_a);
        assert_eq!(n_a, 3);
        assert_eq!(col_entries_a[100], vec![(900, 1.0)]);
        assert_eq!(col_entries_a[201], vec![(900, -1.0)]);
        assert_eq!(col_entries_a[101], vec![(901, 1.0)]);
        assert_eq!(col_entries_a[202], vec![(901, -1.0)]);
        assert_eq!(col_entries_a[102], vec![(902, 1.0)]);
        assert!(col_entries_a[203].is_empty());

        let ring_b = DeliveryRing::new(150..151, 250..251, 1, 1);
        let row_pos_b = vec![Some(0)];
        let mut col_entries_b: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 260];
        let n_b = ring_b.emit_shift_rows(&row_pos_b, 950, &mut col_entries_b);
        assert_eq!(n_b, 1);
        assert_eq!(col_entries_b[150], vec![(950, 1.0)]);
        assert!(col_entries_b[250].is_empty());
    }

    /// One ring shared across `n_lanes = 2` (dense, slot-major/lane-minor),
    /// mirroring the anticipated ring: lane 0 has two interior rows (slots 0
    /// and 1; slot 2 is its own deposit slot, out of scope here), lane 1 has
    /// none (its own deposit slot is slot 0, so every slot is masked in this
    /// table). Slot 1's shift term still targets slot 2's incoming column
    /// even though slot 2 has no row of its own — the incoming column is a
    /// state variable regardless of whether it gets its own definition row.
    #[test]
    fn emit_shift_rows_dense_grid_shares_one_ring_across_heterogeneous_lanes() {
        let ring = DeliveryRing::new(300..306, 400..406, 2, 3);
        // flat = slot * n_lanes + lane.
        let row_pos = vec![
            Some(0), // slot 0, lane 0
            None,    // slot 0, lane 1
            Some(1), // slot 1, lane 0
            None,    // slot 1, lane 1
            None,    // slot 2, lane 0 (deposit slot, not an interior row)
            None,    // slot 2, lane 1
        ];
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 410];
        let n = ring.emit_shift_rows(&row_pos, 500, &mut col_entries);
        assert_eq!(n, 2);
        assert_eq!(col_entries[300], vec![(500, 1.0)]);
        assert_eq!(col_entries[402], vec![(500, -1.0)]);
        assert_eq!(col_entries[302], vec![(501, 1.0)]);
        assert_eq!(col_entries[404], vec![(501, -1.0)]);
        for &col in &[301, 303, 305] {
            assert!(
                col_entries[col].is_empty(),
                "lane 1 column {col} must stay untouched"
            );
        }
    }

    /// `emit_carry_rows`' basic shape: `n_lanes = 2`, `depth = 1`, every
    /// position reachable. Each lane's row carries the same-slot hold
    /// identity `out(0,l) +1`, `in(0,l) −1`.
    #[test]
    fn emit_carry_rows_pins_the_same_slot_hold_identity() {
        let ring = DeliveryRing::new(100..102, 200..202, 2, 1);
        let row_pos = vec![Some(0), Some(1)];
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 210];
        let n = ring.emit_carry_rows(&row_pos, 0, &mut col_entries);
        assert_eq!(n, 2);
        assert_eq!(col_entries[100], vec![(0, 1.0)]);
        assert_eq!(col_entries[200], vec![(0, -1.0)]);
        assert_eq!(col_entries[101], vec![(1, 1.0)]);
        assert_eq!(col_entries[201], vec![(1, -1.0)]);
    }

    /// A masked position gets no row at all, and the returned count
    /// excludes it — the reachable positions on either side are unaffected.
    #[test]
    fn emit_carry_rows_masked_position_emits_no_row() {
        let ring = DeliveryRing::new(300..303, 400..403, 1, 3);
        let row_pos = vec![Some(0), None, Some(1)];
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 410];
        let n = ring.emit_carry_rows(&row_pos, 500, &mut col_entries);
        assert_eq!(n, 2);
        assert_eq!(col_entries[300], vec![(500, 1.0)]);
        assert_eq!(col_entries[400], vec![(500, -1.0)]);
        assert!(col_entries[301].is_empty());
        assert!(col_entries[401].is_empty());
        assert_eq!(col_entries[302], vec![(501, 1.0)]);
        assert_eq!(col_entries[402], vec![(501, -1.0)]);
    }

    /// The explicit shift-vs-hold pin: on the same ring and `row_pos`,
    /// `emit_shift_rows` writes its `-1.0` term on `in_col(slot + 1, lane)`
    /// (the next slot) while `emit_carry_rows` writes it on
    /// `in_col(slot, lane)` (the same slot) — the sole semantic difference
    /// between the two primitives.
    #[test]
    fn emit_carry_rows_targets_the_same_slot_where_emit_shift_rows_targets_the_next() {
        let ring = DeliveryRing::new(600..602, 700..702, 1, 2);
        let row_pos = vec![Some(0), Some(1)];

        let mut shift_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 710];
        let n_shift = ring.emit_shift_rows(&row_pos, 0, &mut shift_entries);

        let mut carry_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 710];
        let n_carry = ring.emit_carry_rows(&row_pos, 0, &mut carry_entries);

        assert_eq!(n_shift, 2);
        assert_eq!(n_carry, 2);

        assert_eq!(shift_entries[600], vec![(0, 1.0)]);
        assert_eq!(carry_entries[600], vec![(0, 1.0)]);
        assert_eq!(shift_entries[601], vec![(1, 1.0)]);
        assert_eq!(carry_entries[601], vec![(1, 1.0)]);

        assert!(shift_entries[700].is_empty());
        assert_eq!(shift_entries[701], vec![(0, -1.0)]);
        assert_eq!(carry_entries[700], vec![(0, -1.0)]);

        assert_eq!(carry_entries[701], vec![(1, -1.0)]);
    }

    /// A masked position freezes to `[0, 0]` with no dependence on the
    /// ring's `reachable_bound` — the shared-masking assertion, checked for
    /// both a water-like open reachable bound and an anticipated-like signed
    /// one.
    #[test]
    fn freeze_masked_columns_masks_identically_across_reachable_bound() {
        let row_pos = vec![Some(0), None, Some(1)];
        let bounds: [((f64, f64), &str); 2] = [
            ((0.0, f64::INFINITY), "water-like [0, inf)"),
            (
                (f64::NEG_INFINITY, f64::INFINITY),
                "anticipated-like (-inf, inf)",
            ),
        ];

        for (reachable_bound, label) in bounds {
            let ring = DeliveryRing::new(50..53, 150..153, 1, 3);
            let mut col_lower = vec![-9.0; 60];
            let mut col_upper = vec![9.0; 60];
            let mut objective = vec![0.0; 60];
            let mut bufs = ColumnBufs {
                col_lower: &mut col_lower,
                col_upper: &mut col_upper,
                objective: &mut objective,
            };
            ring.freeze_masked_columns(&row_pos, reachable_bound, &mut bufs);

            assert_eq!(col_lower[50], reachable_bound.0, "{label}: col 50 lower");
            assert_eq!(col_upper[50], reachable_bound.1, "{label}: col 50 upper");
            assert_eq!(col_lower[51], 0.0, "{label}: col 51 lower");
            assert_eq!(col_upper[51], 0.0, "{label}: col 51 upper");
            assert_eq!(col_lower[52], reachable_bound.0, "{label}: col 52 lower");
            assert_eq!(col_upper[52], reachable_bound.1, "{label}: col 52 upper");
        }
    }

    #[test]
    fn out_col_in_col_addressing_is_slot_major_lane_minor() {
        let ring = DeliveryRing::new(1000..1006, 2000..2006, 3, 2);
        assert_eq!(ring.out_col(0, 0), 1000);
        assert_eq!(ring.out_col(0, 1), 1001);
        assert_eq!(ring.out_col(0, 2), 1002);
        assert_eq!(ring.out_col(1, 0), 1003);
        assert_eq!(ring.out_col(1, 1), 1004);
        assert_eq!(ring.out_col(1, 2), 1005);
        assert_eq!(ring.in_col(0, 0), 2000);
        assert_eq!(ring.in_col(1, 2), 2005);
    }

    /// `slot_lane_at` is the exact inverse of `out_col`/`in_col`: recovering
    /// `(slot, lane)` from every block-relative offset in a 3-lane, 2-deep
    /// ring round-trips through `out_col` back to the same offset.
    #[test]
    fn slot_lane_at_is_the_inverse_of_out_col_in_col() {
        let ring = DeliveryRing::new(1000..1006, 2000..2006, 3, 2);
        assert_eq!(ring.slot_lane_at(0), (0, 0));
        assert_eq!(ring.slot_lane_at(1), (0, 1));
        assert_eq!(ring.slot_lane_at(2), (0, 2));
        assert_eq!(ring.slot_lane_at(3), (1, 0));
        assert_eq!(ring.slot_lane_at(4), (1, 1));
        assert_eq!(ring.slot_lane_at(5), (1, 2));

        for offset in 0..6 {
            let (slot, lane) = ring.slot_lane_at(offset);
            assert_eq!(
                ring.out_col(slot, lane) - 1000,
                offset,
                "slot_lane_at({offset}) must round-trip through out_col"
            );
            assert_eq!(
                ring.in_col(slot, lane) - 2000,
                offset,
                "slot_lane_at({offset}) must round-trip through in_col"
            );
        }
    }

    /// `emit_deposit` is the ring's only real entry emission: the `(+1, -1)`
    /// slot↔decision pair, under a `col_scale = 1.0` column context — the
    /// freeze pins `[0, 0]`, never a scaled bound, so no bound needs
    /// adjusting for the column's scale.
    #[test]
    fn equality_pin_deposit_emits_the_slot_decision_pair_at_unit_col_scale() {
        let ring = DeliveryRing::new(10..13, 20..23, 1, 3);
        let decision_col = 99;
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 100];
        let col_scale = 1.0_f64;

        ring.emit_deposit(2, 0, 777, decision_col, &mut col_entries);

        let out_col = ring.out_col(2, 0);
        assert_eq!(col_entries[out_col], vec![(777, col_scale)]);
        assert_eq!(col_entries[decision_col], vec![(777, -col_scale)]);
    }

    #[test]
    fn slot_target_maps_lag_to_the_flat_row_pos_index() {
        let ring = DeliveryRing::new(0..9, 0..9, 3, 3);
        assert_eq!(ring.slot_target(1, 1), 1);
        assert_eq!(ring.slot_target(2, 3), 8);
    }

    /// At `depth == 1`, `emit_shift_rows`'s `slot + 1 < depth` guard is always
    /// false, so every reachable lane writes only its outgoing column and no
    /// incoming-column entry — the multi-lane (`n_lanes = 2`) extension of
    /// `emit_shift_rows_drops_the_shift_term_past_a_lanes_own_depth`'s
    /// single-lane depth-1 case.
    #[test]
    fn depth_one_emit_shift_rows_never_writes_an_in_col_entry() {
        let ring = DeliveryRing::new(100..102, 200..202, 2, 1);
        let row_pos = vec![Some(0), Some(1)];
        let mut entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 210];

        let n = ring.emit_shift_rows(&row_pos, 0, &mut entries);

        assert_eq!(n, 2, "both lanes are reachable at depth 1");
        assert_eq!(entries[ring.out_col(0, 0)], vec![(0, 1.0)]);
        assert_eq!(entries[ring.out_col(0, 1)], vec![(1, 1.0)]);
        assert!(
            entries[ring.in_col(0, 0)].is_empty() && entries[ring.in_col(0, 1)].is_empty(),
            "the slot+1<depth guard is unconditionally false at depth 1"
        );
    }

    /// At `depth == 1` every column either `emit_shift_rows` or
    /// `emit_carry_rows` touches lies in `{out_col(0, lane), in_col(0, lane)}`
    /// for some lane — the latch/fish column pair every anticipated-ring
    /// caller already addresses at that depth, so switching between the two
    /// interior-transition primitives cannot introduce a third column.
    #[test]
    fn depth_one_column_footprint_coincides_between_shift_and_carry() {
        let ring = DeliveryRing::new(100..102, 200..202, 2, 1);
        let row_pos = vec![Some(0), Some(1)];

        let mut shift_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 210];
        ring.emit_shift_rows(&row_pos, 0, &mut shift_entries);

        let mut carry_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 210];
        let n_carry = ring.emit_carry_rows(&row_pos, 0, &mut carry_entries);
        assert_eq!(
            n_carry, 2,
            "unlike emit_shift_rows, emit_carry_rows fires at depth 1 over a reachable row_pos"
        );

        let latch_fish_columns = [
            ring.out_col(0, 0),
            ring.out_col(0, 1),
            ring.in_col(0, 0),
            ring.in_col(0, 1),
        ];
        for (col, (shift, carry)) in shift_entries.iter().zip(carry_entries.iter()).enumerate() {
            let touched = !shift.is_empty() || !carry.is_empty();
            assert!(
                !touched || latch_fish_columns.contains(&col),
                "column {col} lies outside the depth-1 {{latch, fish}} column set"
            );
        }
    }

    /// A two-plant state (H1 depth 2, H3 depth 3) yields one `PlantBucketRing`
    /// per plant, each addressed at its own local offset, and `transit_bucket`
    /// looks up a declared plant while missing an undeclared one.
    #[test]
    fn transit_buckets_builds_each_plant_ring_over_its_bucket_run() {
        let h1 = HydroSys::new(1);
        let h3 = HydroSys::new(3);
        let state = crate::test_support::state_layout_with_transit_buckets(
            4,
            0,
            vec![(h1, 1), (h1, 2), (h3, 1), (h3, 2), (h3, 3)],
            vec![],
        );

        let buckets: Vec<PlantBucketRing> = DeliveryRing::transit_buckets(&state).collect();
        assert_eq!(buckets.len(), 2);
        for bucket in &buckets {
            let out_block = state.bucket_outgoing_block(bucket.local.clone());
            let in_block = state.bucket_incoming_block(bucket.local.clone());
            for s in 0..bucket.local.len() {
                assert_eq!(bucket.ring.out_col(s, 0), out_block.start + s);
                assert_eq!(bucket.ring.in_col(s, 0), in_block.start + s);
            }
        }

        let h3_bucket =
            DeliveryRing::transit_bucket(&state, h3).expect("H3 must have a declared bucket");
        assert_eq!(h3_bucket.plant, buckets[1].plant);
        assert_eq!(h3_bucket.local, buckets[1].local);

        assert!(DeliveryRing::transit_bucket(&state, HydroSys::new(0)).is_none());
    }
}
