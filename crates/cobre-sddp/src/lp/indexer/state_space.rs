//! The stage-invariant state-vector layout and its LP-column resolvers.
//!
//! State pinning uses column bounds, not equality rows:
//! [`StateSpace::state_to_lp_incoming_column`] is the single authoritative
//! incoming-state column resolver — the LP column for both pinning and dual
//! extraction is always resolved through it, never by assuming a fixing-row
//! index. The companion [`StateSpace::state_to_lp_column`] maps the outgoing
//! state vector to the LP columns a forward-pass cut row references.
//!
//! Every offset here is a pure function of `N` (`hydro_count`), `L`
//! (`max_par_order`), `B` (`n_buckets`), `A` (`n_anticipated`), and `k_max` —
//! independent of `n_blks`/`n_thermals` — so a single global stage-0 layout
//! resolves onto the correct column at every stage regardless of per-stage
//! block counts.

use std::ops::Range;

use super::{HydroSys, InCol, OutCol, RangeCursor, StateDim, for_each_live_commitment_slot};
use crate::bucket_topology::TransitBucketTopology;
use crate::lead_time::AnticipatedResolution;

use cobre_core::Hydro;
use cobre_core::temporal::StageStateConfig;

/// Stage-invariant state-vector layout for one SDDP stage subproblem.
///
/// ## Column layout
///
/// ```text
/// [0, N)                                     storage               — outgoing storage volumes (N = hydro_count)
/// [N, N*(1+L))                               inflow_lags           — AR lag variables (L lags per hydro)
/// [N*(1+L), N*(1+L) + B)                     transit_buckets_out   — travel-time bucket state (outgoing, identity)
/// [N*(1+L) + B, N*(1+L) + B + S)             commit_out            — commitment-hold outgoing slots (outgoing, identity)
/// [… + S, … + N)                             z_inflow              — realized inflow (auxiliary, not state)
/// [… + N, … + 2*N)                           storage_in            — incoming storage volumes
/// [… + 2*N, … + 2*N + B)                     transit_buckets_in    — incoming travel-time bucket volumes (pinned)
/// [… + B, … + B + S)                         commit_in             — incoming commitment-hold slots (pinned)
/// … + S                                       theta                 — future cost variable (scalar)
/// ```
///
/// `S = A*k_max`: the commitment-hold region is the in-study anticipated-ring
/// slots in one contiguous out/in pair — an outgoing block
/// ([`Self::commit_out`], identity-resolved, contributing to
/// [`Self::n_state`]) and a separate incoming block ([`Self::commit_in`],
/// pinned via [`Self::state_to_lp_incoming_column`]) — never one
/// dual-purpose range shifted out-of-LP. Slots are slot-major/plant-minor,
/// keyed ring-axis-modular (delivery target `m`'s slot is `r(m) mod k_max`,
/// see [`Self::commitment_hold_in_study_offset`]). Like the bucket block,
/// the whole region is always cut-enabled,
/// ignoring [`StageStateConfig`] (see [`StateRegion::cut_enabled`]).
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// Outgoing storage volumes.
    pub storage: Range<usize>,

    /// AR lag variables, lag-major: all hydros for lag 0, then lag 1, …. Hydro
    /// `h` at lag `l` is at `inflow_lags.start + l * hydro_count + h`.
    pub inflow_lags: Range<usize>,

    /// Travel-time in-transit bucket state. Outgoing bucket state maps to its
    /// LP column by identity — the `storage` convention, not the `z_inflow`
    /// lag remap.
    pub transit_buckets_out: Range<usize>,

    /// Commitment-hold OUTGOING slots: the `n_anticipated * k_max` in-study
    /// anticipated-ring slots, slot-major/plant-minor (slot `k` for plant `i`
    /// at `commit_out.start + k * n_anticipated + i` —
    /// [`Self::commitment_hold_in_study_offset`]). Every slot is a genuine LP
    /// column resolved by identity (the `transit_buckets_out` convention) and
    /// defined by an in-LP definition row — never resolved out-of-LP. A slot
    /// `k >= k_i` for a plant whose own reachable depth is smaller is
    /// padding, frozen `[0, 0]`.
    pub commit_out: Range<usize>,

    /// Incoming storage volumes, pinned via
    /// [`StateSpace::state_to_lp_incoming_column`].
    pub storage_in: Range<usize>,

    /// Incoming travel-time bucket volumes, pinned via
    /// [`StateSpace::state_to_lp_incoming_column`].
    pub transit_buckets_in: Range<usize>,

    /// Realized-inflow variables `z_h`, one per hydro.
    pub z_inflow: Range<usize>,

    /// Commitment-hold INCOMING slots, the same layout as [`Self::commit_out`],
    /// pinned via [`StateSpace::state_to_lp_incoming_column`]. The slot
    /// maturing this stage is also read directly by [`crate::lp::builder`]'s
    /// commitment-fishing row fill.
    pub commit_in: Range<usize>,

    /// Future cost variable (theta) column.
    pub theta: usize,

    /// State-vector dimension used by cut storage and broadcast payloads —
    /// **not** a valid LP row index. No state-fixing rows exist; do not slice
    /// the LP row buffer as `[0, n_state)`. Resolve the pinning/subgradient
    /// column via [`StateSpace::state_to_lp_incoming_column`].
    pub n_state: usize,

    /// Number of operating hydro plants (N).
    pub hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L); every hydro uses this
    /// uniform lag stride.
    pub max_par_order: usize,

    /// Global travel-time bucket count `B`, the sum of every plant's own run
    /// in [`Self::transit_bucket_plants`], `0` when no arc is declared.
    pub n_buckets: usize,

    /// Number of anticipated thermals — [`super::AnticipatedPlants::len`].
    pub n_anticipated: usize,

    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    pub k_max: usize,

    /// Per-plant `lead_stages` (`K_i`), indexed by anticipated-local position;
    /// length [`Self::n_anticipated`].
    pub anticipated_lead_stages: Vec<usize>,

    /// Delivery-anchored point-commitment resolution per anticipated plant
    /// (anticipated-local order), attached at construction.
    pub(crate) anticipated_resolution: AnticipatedResolution,

    /// Canonical `(plant, lag)` pair per bucket state-vector dimension,
    /// plant named by its `HydroSys` position, in
    /// [`Self::transit_buckets_out`] order.
    pub transit_bucket_column_order: Vec<(HydroSys, usize)>,

    /// State dimensions whose cut coefficients can be nonzero (padded lag/ring
    /// slots excluded); computed by [`Self::set_nonzero_mask`].
    pub nonzero_state_indices: Vec<StateDim>,

    /// Precomputed `state_to_lp_column(j)` for every `j ∈ [0, n_state)` (cut-row
    /// hot path).
    pub state_to_lp_column_map: Vec<OutCol>,
}

/// One of the four stage-invariant state-vector regions, in the canonical walk
/// order [`REGION_ORDER`] declares.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StateRegion {
    /// Outgoing storage volumes — [`StateSpace::state_dim_storage_range`].
    Storage,
    /// AR inflow lags — [`StateSpace::state_dim_lag_range`].
    Lag,
    /// Travel-time in-transit buckets — [`StateSpace::state_dim_bucket_range`].
    Buckets,
    /// Merged in-study anticipated-ring + terminal post-horizon commitment
    /// slots — [`StateSpace::state_dim_commitment_hold_range`].
    CommitmentHold,
}

/// The single owner of the `storage → lag → buckets → commitment hold`
/// state-region walk order. [`StateSpace::state_to_lp_column`],
/// [`StateSpace::state_to_lp_incoming_column`],
/// [`StateSpace::set_nonzero_mask`], and [`super::CutStateProjection::new`]
/// all dispatch over this array via [`StateSpace::state_dim_range`]; a
/// variant added to [`StateRegion`] fails to compile at any of those sites
/// (each matches it exhaustively, no `_` catch-all) until handled.
pub(crate) const REGION_ORDER: [StateRegion; 4] = [
    StateRegion::Storage,
    StateRegion::Lag,
    StateRegion::Buckets,
    StateRegion::CommitmentHold,
];

impl StateRegion {
    /// Whether `region`'s state dimensions are cut-enabled under `config`: storage
    /// and lag are config-gated; buckets and the merged commitment-hold region
    /// are always included, regardless of `config` — every declared in-study or
    /// post-horizon commitment slot is a priced FCF dimension at every pool, the
    /// terminal pool included, so a loaded boundary cut's coefficient lands on
    /// the right column. Gating either on `config` shrinks the cut pool's
    /// state-dimension below the global trial state and misaligns the intercept
    /// dot ([`super::CutStateProjection::new`]).
    #[inline]
    #[must_use]
    pub(crate) fn cut_enabled(self, config: StageStateConfig) -> bool {
        match self {
            StateRegion::Storage => config.storage,
            StateRegion::Lag => config.inflow_lags,
            StateRegion::Buckets | StateRegion::CommitmentHold => true,
        }
    }
}

/// `0..0` when `n == 0`, never `cursor.alloc(0)`'s `pos..pos` — the sentinel
/// an empty bucket/commitment-hold block must carry byte-identically.
fn alloc_or_empty(cursor: &mut RangeCursor, n: usize) -> Range<usize> {
    if n > 0 { cursor.alloc(n) } else { 0..0 }
}

impl StateSpace {
    /// Construct the production [`StateSpace`] from the system's hydros, the
    /// resolved travel-time topology, and the anticipated leads/resolution —
    /// see [`Self::assemble`] for the shared construction contract and
    /// panics.
    #[must_use]
    pub(crate) fn build(
        hydros: &[Hydro],
        max_par_order: usize,
        effective_lag_counts: &[usize],
        topology: &TransitBucketTopology,
        anticipated_lead_stages: Vec<usize>,
        anticipated_resolution: AnticipatedResolution,
    ) -> Self {
        Self::assemble(
            hydros.len(),
            max_par_order,
            topology.column_order.clone(),
            anticipated_lead_stages,
            anticipated_resolution,
            effective_lag_counts,
        )
    }

    /// Loose constructor for benches, doctests, and tests with no
    /// [`TransitBucketTopology`] to build from — see [`Self::assemble`] for
    /// the shared construction contract and panics. Production code calls
    /// [`Self::build`].
    #[cfg(any(test, feature = "test-support"))]
    #[must_use]
    pub fn new(
        hydro_count: usize,
        max_par_order: usize,
        transit_bucket_column_order: Vec<(HydroSys, usize)>,
        anticipated_lead_stages: Vec<usize>,
        anticipated_resolution: AnticipatedResolution,
        effective_lag_count: &[usize],
    ) -> Self {
        Self::assemble(
            hydro_count,
            max_par_order,
            transit_bucket_column_order,
            anticipated_lead_stages,
            anticipated_resolution,
            effective_lag_count,
        )
    }

    /// Assemble a finalized [`StateSpace`] from the state dimensions, the
    /// per-plant anticipated resolution, and the per-hydro effective lag-slot
    /// counts — the shared body [`Self::build`] and [`Self::new`] both
    /// delegate to. `n_buckets` is `transit_bucket_column_order.len()`; `0`
    /// reproduces the pre-bucket layout byte-for-byte.
    ///
    /// `n_anticipated` is `anticipated_lead_stages.len()`; `k_max` is
    /// `anticipated_resolution.ring_size(&anticipated_lead_stages)` — the
    /// single ring-size owner. `effective_lag_count` must have length
    /// `hydro_count`; each entry is `PrecomputedPar::effective_lag_count(h)` —
    /// the count of lag slots that may carry non-zero cut coefficients (see
    /// [`Self::set_nonzero_mask`] for why `order(h)` is wrong).
    ///
    /// # Panics (debug builds only)
    ///
    /// `anticipated_resolution.per_plant.len() == anticipated_lead_stages.len()`,
    /// with every plant's `decider` sharing one per-study delivery-stage
    /// count. Inherits the [`Self::set_nonzero_mask`] and
    /// [`Self::finalize_state_column_map`] debug assertions:
    /// `effective_lag_count.len() == hydro_count`, lag bounds, and
    /// `state_to_lp_column_map.len() == n_state`.
    fn assemble(
        hydro_count: usize,
        max_par_order: usize,
        transit_bucket_column_order: Vec<(HydroSys, usize)>,
        anticipated_lead_stages: Vec<usize>,
        anticipated_resolution: AnticipatedResolution,
        effective_lag_count: &[usize],
    ) -> Self {
        let n_buckets = transit_bucket_column_order.len();
        debug_assert_eq!(
            anticipated_resolution.per_plant.len(),
            anticipated_lead_stages.len(),
            "resolution must carry one PointResolution per anticipated plant"
        );

        let n_anticipated = anticipated_lead_stages.len();
        let k_max = anticipated_resolution.ring_size(&anticipated_lead_stages);
        let n_delivery = anticipated_resolution
            .per_plant
            .first()
            .map_or(0, |plant| plant.decider.len());
        debug_assert!(
            anticipated_resolution
                .per_plant
                .iter()
                .all(|plant| plant.decider.len() == n_delivery),
            "every plant's decider must share one per-study delivery-stage count \
             (the delivery axis is per-study, not per-plant)"
        );

        let n = hydro_count;
        let l = max_par_order;
        let n_ant_state = n_anticipated * k_max;

        let mut cursor = RangeCursor::new(0);
        let storage = cursor.alloc(n);
        let inflow_lags = cursor.alloc(n * l);
        let transit_buckets_out = alloc_or_empty(&mut cursor, n_buckets);
        let commit_out = alloc_or_empty(&mut cursor, n_ant_state);
        let z_inflow = cursor.alloc(n);
        let storage_in = cursor.alloc(n);
        let transit_buckets_in = alloc_or_empty(&mut cursor, n_buckets);
        let commit_in = alloc_or_empty(&mut cursor, n_ant_state);

        let theta = cursor.pos();

        // n_ant_state counted once (commit_out/commit_in are the same dimensions), not twice.
        let n_state = n * (1 + l) + n_buckets + n_ant_state;

        debug_assert_eq!(
            commit_out.len(),
            n_ant_state,
            "commit_out must span exactly the in-study anticipated-ring slots"
        );
        debug_assert_eq!(
            commit_out.len(),
            commit_in.len(),
            "commit_out and commit_in must have equal width"
        );

        let mut layout = Self {
            storage,
            inflow_lags,
            transit_buckets_out,
            commit_out,
            storage_in,
            transit_buckets_in,
            z_inflow,
            commit_in,
            theta,
            n_state,
            hydro_count,
            max_par_order,
            n_buckets,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            anticipated_resolution,
            transit_bucket_column_order,
            nonzero_state_indices: Vec::new(),
            state_to_lp_column_map: Vec::new(),
        };

        layout.set_nonzero_mask(effective_lag_count);
        layout.finalize_state_column_map();
        layout
    }

    /// First column of the control region (`theta + 1`): the state region
    /// occupies `[0, theta]`, equipment columns follow from here.
    #[inline]
    #[must_use]
    pub fn control_region_start(&self) -> usize {
        self.theta + 1
    }

    // GLOBAL STATE INDEX space (not LP columns).

    /// State-dimension region `[0, N)` — storage.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_storage_range(&self) -> Range<usize> {
        0..self.hydro_count
    }

    /// State-dimension region `[N, N*(1+L))` — AR inflow lags.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_lag_range(&self) -> Range<usize> {
        let n = self.hydro_count;
        n..n * (1 + self.max_par_order)
    }

    /// State-dimension region `[N*(1+L), N*(1+L) + B)` — travel-time buckets.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_bucket_range(&self) -> Range<usize> {
        let start = self.state_dim_lag_range().end;
        start..start + self.n_buckets
    }

    /// State-dimension region `[N*(1+L) + B, N*(1+L) + B + A*k_max)` — the
    /// in-study anticipated-ring slots.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_commitment_hold_range(&self) -> Range<usize> {
        let start = self.state_dim_bucket_range().end;
        start..start + self.n_anticipated * self.k_max
    }

    /// Resolve `region`'s state-dimension range through the matching
    /// `state_dim_*_range` accessor above — the exhaustive dispatch every
    /// [`REGION_ORDER`]-driven call site uses instead of re-deriving region
    /// boundaries inline.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_range(&self, region: StateRegion) -> Range<usize> {
        match region {
            StateRegion::Storage => self.state_dim_storage_range(),
            StateRegion::Lag => self.state_dim_lag_range(),
            StateRegion::Buckets => self.state_dim_bucket_range(),
            StateRegion::CommitmentHold => self.state_dim_commitment_hold_range(),
        }
    }

    /// `region`'s incoming pinned-block start column — the single owner of
    /// which block [`Self::state_to_lp_incoming_column`] pins.
    #[inline]
    #[must_use]
    pub(crate) fn incoming_block_start(&self, region: StateRegion) -> usize {
        match region {
            StateRegion::Storage => self.storage_in.start,
            StateRegion::Lag => self.inflow_lags.start,
            StateRegion::Buckets => self.transit_buckets_in.start,
            StateRegion::CommitmentHold => self.commit_in.start,
        }
    }

    /// Classify `j` into its [`StateRegion`] and its offset within that
    /// region's [`Self::state_dim_range`] — the single scan both
    /// [`Self::state_to_lp_column`] and [`Self::state_to_lp_incoming_column`]
    /// consume.
    ///
    /// [`REGION_ORDER`]'s four ranges partition `[0, n_state)` contiguously
    /// (`state_dim_ranges_partition_n_state_contiguously`), so `find` only
    /// misses for an out-of-range `j`; `unwrap_or` keeps this function total
    /// instead of adding a panic path for that case.
    #[inline]
    #[must_use]
    pub(crate) fn classify(&self, j: StateDim) -> (StateRegion, usize) {
        let j = j.get();
        let region = REGION_ORDER
            .into_iter()
            .find(|&region| self.state_dim_range(region).contains(&j))
            .unwrap_or(StateRegion::CommitmentHold);
        (region, j - self.state_dim_range(region).start)
    }

    /// Map a state-vector index to the LP column it references in a cut.
    ///
    /// Classifies `j` into its `StateRegion` via `REGION_ORDER` before any
    /// lag arithmetic runs, then resolves through an exhaustive match:
    /// storage, `transit_buckets_out`, and `commit_out` map by identity. Lag
    /// indices remap to the outgoing state after `shift_lag_state`: lag 0 is
    /// realised inflow → [`Self::z_inflow_col`]; lag `l ≥ 1` is the previous
    /// stage's lag `l − 1` → [`Self::lag_incoming_col`]. Classifying
    /// first — rather than falling through an `if`/`else` chain — is what
    /// keeps buckets/commitment-hold from ever reaching the lag decode.
    ///
    /// A bare `usize` cannot skip the [`StateDim`] wrap at this resolver
    /// boundary:
    ///
    /// ```compile_fail
    /// use cobre_sddp::indexer::StateSpace;
    ///
    /// fn misuse(state: &StateSpace) {
    ///     let _col = state.state_to_lp_column(0); // bare usize handed where StateDim is required
    /// }
    /// ```
    ///
    /// Nor can an already-resolved [`OutCol`] re-enter as the unresolved
    /// dimension:
    ///
    /// ```compile_fail
    /// use cobre_sddp::indexer::{StateDim, StateSpace};
    ///
    /// fn misuse(state: &StateSpace) {
    ///     let col = state.state_to_lp_column(StateDim::new(0));
    ///     let _reentered = state.state_to_lp_column(col); // OutCol handed where StateDim is required
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn state_to_lp_column(&self, j: StateDim) -> OutCol {
        let (region, offset) = self.classify(j);

        OutCol::new(match region {
            StateRegion::Storage | StateRegion::Buckets | StateRegion::CommitmentHold => j.get(),
            StateRegion::Lag => {
                let n = self.hydro_count;
                let h = offset % n;
                let lag = offset / n;
                if lag == 0 {
                    self.z_inflow_col(HydroSys::new(h)).get()
                } else {
                    self.lag_incoming_col(lag - 1, HydroSys::new(h)).get()
                }
            }
        })
    }

    /// Fill [`Self::state_to_lp_column_map`] by calling
    /// [`Self::state_to_lp_column`] for every `j ∈ [0, n_state)` — a pure cache
    /// of the resolver, never a reimplementation of its arithmetic.
    pub fn finalize_state_column_map(&mut self) {
        self.state_to_lp_column_map.clear();
        self.state_to_lp_column_map.reserve(self.n_state);
        for j in 0..self.n_state {
            self.state_to_lp_column_map
                .push(self.state_to_lp_column(StateDim::new(j)));
        }
        debug_assert_eq!(self.state_to_lp_column_map.len(), self.n_state);
    }

    /// Read the precomputed `state_to_lp_column(j)` from
    /// [`Self::state_to_lp_column_map`], which [`Self::build`] always
    /// finalizes to `n_state` length (indexed read is in range for
    /// `j ∈ [0, n_state)`).
    #[inline]
    #[must_use]
    pub fn lp_column_for_state(&self, j: StateDim) -> OutCol {
        debug_assert_eq!(
            self.state_to_lp_column_map.len(),
            self.n_state,
            "state_to_lp_column_map must be finalized to n_state length"
        );
        self.state_to_lp_column_map[j.get()]
    }

    /// Map a state-vector index to its **incoming-state** LP column — the column
    /// pinned to `lb = ub = v` via `set_col_bounds` (never an equality/fixing
    /// row; the LP has no state-fixing row range). The returned columns are
    /// exactly those
    /// [`fill_col_state_patches`](crate::lp::builder::PatchBuffer::fill_col_state_patches)
    /// writes, in state-vector order; the backward pass reads
    /// `view.reduced_costs[col]` at them for the cut subgradient, one per
    /// component `j ∈ [0, n_state)`.
    ///
    /// The travel-time bucket range resolves through an explicit
    /// `transit_buckets_in` arm, not the trailing anticipated-state catch-all.
    ///
    /// Contrast [`state_to_lp_column`], which returns the **outgoing** column
    /// for forward-pass cut-row construction: for storage it returns `j`
    /// (outgoing), this returns `storage_in.start + j` (incoming). The two are
    /// related by the water-balance equality row — by KKT duality the incoming
    /// column's reduced cost equals the dual a row-based state-fixing
    /// formulation would produce, which is why column-bound pinning is exact.
    ///
    /// [`state_to_lp_column`]: Self::state_to_lp_column
    #[inline]
    #[must_use]
    pub fn state_to_lp_incoming_column(&self, j: StateDim) -> InCol {
        let (region, offset) = self.classify(j);
        InCol::new(self.incoming_block_start(region) + offset)
    }

    /// `region`'s incoming pinned-block column range.
    fn incoming_block_range(&self, region: StateRegion) -> Range<usize> {
        let start = self.incoming_block_start(region);
        start..start + self.state_dim_range(region).len()
    }

    /// Inverse of [`Self::state_to_lp_incoming_column`]: the region and
    /// in-region offset owning pinned column `c`. Total for the same reason
    /// as [`Self::classify`].
    #[must_use]
    pub(crate) fn classify_incoming_column(&self, c: InCol) -> (StateRegion, usize) {
        let c = c.get();
        let region = REGION_ORDER
            .into_iter()
            .find(|&region| self.incoming_block_range(region).contains(&c))
            .unwrap_or(StateRegion::CommitmentHold);
        (region, c - self.incoming_block_start(region))
    }

    /// Encode an in-study commitment-hold position: anticipated-local plant
    /// `plant`'s offset within [`Self::commit_out`]/[`Self::commit_in`] for
    /// delivery target `m`, keyed ring-axis-modular (`slot = r(m) mod
    /// k_max`, `PointResolution::ring_index` owns `r`) — `m mod k_max`
    /// alone is injective only on a contiguous run of the raw delivery axis,
    /// which the excised fixed post-horizon window breaks — slot-major/plant-minor.
    #[must_use]
    pub(crate) fn commitment_hold_in_study_offset(&self, plant: usize, m: usize) -> usize {
        debug_assert!(
            plant < self.n_anticipated,
            "plant {plant} out of range (n_anticipated = {})",
            self.n_anticipated
        );
        debug_assert!(
            self.k_max > 0,
            "commitment_hold_in_study_offset requires k_max > 0"
        );
        let r = match self.anticipated_resolution.per_plant.get(plant) {
            Some(point) => {
                let r = point.ring_index(m);
                debug_assert!(
                    r.is_some(),
                    "delivery target {m} is inside plant {plant}'s excised fixed \
                     post-horizon window — never a ring member, so addressing it \
                     is a caller bug"
                );
                r.unwrap_or(m)
            }
            None => m,
        };
        (r % self.k_max) * self.n_anticipated + plant
    }

    /// The storage state dimension for hydro `h` (`state_dim_storage_range().start + h`).
    #[inline]
    #[must_use]
    pub(crate) fn storage_state_dim(&self, h: HydroSys) -> StateDim {
        debug_assert!(h.get() < self.hydro_count);
        StateDim::new(self.state_dim_storage_range().start + h.get())
    }

    /// Incoming (stage-initial) storage column of hydro `h`.
    #[inline]
    #[must_use]
    pub(crate) fn storage_incoming_col(&self, h: HydroSys) -> InCol {
        self.state_to_lp_incoming_column(self.storage_state_dim(h))
    }

    /// Outgoing (stage-final) storage column of hydro `h`.
    #[inline]
    #[must_use]
    pub(crate) fn storage_outgoing_col(&self, h: HydroSys) -> OutCol {
        self.state_to_lp_column(self.storage_state_dim(h))
    }

    /// The lag-major state dimension for hydro `h` at `lag`
    /// (`state_dim_lag_range().start + lag * hydro_count + h`).
    #[inline]
    #[must_use]
    pub(crate) fn lag_state_dim(&self, lag: usize, h: HydroSys) -> StateDim {
        debug_assert!(lag < self.max_par_order && h.get() < self.hydro_count);
        StateDim::new(self.state_dim_lag_range().start + lag * self.hydro_count + h.get())
    }

    /// Incoming pinned lag column of hydro `h` at `lag` (lag-major block).
    #[inline]
    #[must_use]
    pub(crate) fn lag_incoming_col(&self, lag: usize, h: HydroSys) -> InCol {
        debug_assert!(lag < self.max_par_order && h.get() < self.hydro_count);
        self.state_to_lp_incoming_column(self.lag_state_dim(lag, h))
    }

    /// Incoming pinned bucket column of bucket `b`
    /// ([`Self::transit_bucket_column_order`] order).
    #[must_use]
    pub(crate) fn bucket_incoming_col(&self, b: usize) -> InCol {
        debug_assert!(b < self.n_buckets);
        self.state_to_lp_incoming_column(StateDim::new(self.state_dim_bucket_range().start + b))
    }

    /// Outgoing bucket column of bucket `b`.
    #[must_use]
    pub(crate) fn bucket_outgoing_col(&self, b: usize) -> OutCol {
        debug_assert!(b < self.n_buckets);
        self.state_to_lp_column(StateDim::new(self.state_dim_bucket_range().start + b))
    }

    /// Outgoing-bucket column block for the sub-range `local` (relative to
    /// [`Self::transit_buckets_out`]'s own start).
    #[inline]
    #[must_use]
    pub(crate) fn bucket_outgoing_block(&self, local: Range<usize>) -> Range<usize> {
        debug_assert!(local.end <= self.n_buckets);
        self.transit_buckets_out.start + local.start..self.transit_buckets_out.start + local.end
    }

    /// Incoming-bucket column block for the sub-range `local`; see
    /// [`Self::bucket_outgoing_block`].
    #[inline]
    #[must_use]
    pub(crate) fn bucket_incoming_block(&self, local: Range<usize>) -> Range<usize> {
        debug_assert!(local.end <= self.n_buckets);
        self.transit_buckets_in.start + local.start..self.transit_buckets_in.start + local.end
    }

    /// Each plant's contiguous run within [`Self::transit_bucket_column_order`],
    /// as a local sub-range relative to [`Self::transit_buckets_out`]/
    /// [`Self::transit_buckets_in`]'s own start. The run's length is that
    /// plant's own bucket depth.
    pub(crate) fn transit_bucket_plants(
        &self,
    ) -> impl Iterator<Item = (HydroSys, Range<usize>)> + '_ {
        let mut start = 0;
        self.transit_bucket_column_order
            .chunk_by(|a, b| a.0 == b.0)
            .map(move |run| {
                let local = start..start + run.len();
                start = local.end;
                (run[0].0, local)
            })
    }

    fn commitment_hold_state_dim(&self, plant: usize, m: usize) -> StateDim {
        StateDim::new(
            self.state_dim_commitment_hold_range().start
                + self.commitment_hold_in_study_offset(plant, m),
        )
    }

    /// Incoming (pinned) commitment-hold column for anticipated-local `plant`'s
    /// delivery target `m`.
    #[must_use]
    pub(crate) fn commitment_hold_incoming_col(&self, plant: usize, m: usize) -> InCol {
        self.state_to_lp_incoming_column(self.commitment_hold_state_dim(plant, m))
    }

    /// Outgoing commitment-hold column for anticipated-local `plant`'s delivery
    /// target `m`.
    #[must_use]
    pub(crate) fn commitment_hold_outgoing_col(&self, plant: usize, m: usize) -> OutCol {
        self.state_to_lp_column(self.commitment_hold_state_dim(plant, m))
    }

    /// The delivery-axis stage count: the attached resolution's own decider
    /// length (`0` with no anticipated plants), read from the first plant —
    /// every plant shares one per-study delivery axis (constructor
    /// `debug_assert`).
    #[inline]
    #[must_use]
    pub(crate) fn n_delivery(&self) -> usize {
        self.anticipated_resolution
            .per_plant
            .first()
            .map_or(0, |plant| plant.decider.len())
    }

    /// Compute and store [`Self::nonzero_state_indices`] from per-hydro
    /// lag-slot counts.
    ///
    /// `lag_counts` must have length `hydro_count`; `lag_counts[h]` is the count
    /// of lag slots that may carry non-zero cut coefficients for hydro `h`. It
    /// must be `PrecomputedPar::effective_lag_count(h)`, **not** `order(h)`: for
    /// PAR(p)-A hydros `effective_lag_count == max_par_order` so the `ψ̂/12`
    /// annual contributions on slots `order..max_par_order` reach the cut rows;
    /// `order(h)` would truncate them and produce over-estimating cuts (LB > UB
    /// at convergence). Storage `[0, N)` is always included.
    ///
    /// Every travel-time bucket slot is always included — bucket depth is
    /// already sized as the per-stage reachability union, so there is no
    /// padding to exclude. The commitment-hold region instead keeps exactly
    /// the slots [`super::for_each_live_commitment_slot`] visits over every
    /// decision stage — the union, over `0..n_decision`, of the LP's own
    /// per-stage latch set — since the resolution is already attached by the
    /// time this method runs from the constructor.
    ///
    /// The loop iterates lag-first, then commitment-hold in ascending offset
    /// order, so the emitted indices stay strictly ascending (the sortedness
    /// the `debug_assert` enforces).
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `lag_counts.len() != hydro_count` or any
    /// `lag_counts[h] > max_par_order`.
    pub fn set_nonzero_mask(&mut self, lag_counts: &[usize]) {
        debug_assert_eq!(lag_counts.len(), self.hydro_count);

        let n_lag_active: usize = lag_counts.iter().copied().sum();
        let n_ant_state = self.n_anticipated * self.k_max;
        let mut mask =
            Vec::with_capacity(self.hydro_count + n_lag_active + self.n_buckets + n_ant_state);

        // REGION_ORDER fixes the walk order; storage and buckets have no
        // padding to exclude and extend their full range, lag keeps its own
        // active-slot filter, and commitment-hold keeps its own live-slot
        // filter (padding stays excluded — see the doc comment above).
        for region in REGION_ORDER {
            match region {
                StateRegion::Storage | StateRegion::Buckets => {
                    mask.extend(self.state_dim_range(region).map(StateDim::new));
                }
                StateRegion::Lag => {
                    let start = self.state_dim_range(region).start;
                    for lag in 0..self.max_par_order {
                        for (h, &lag_count) in lag_counts.iter().enumerate() {
                            debug_assert!(lag_count <= self.max_par_order);
                            if lag < lag_count {
                                mask.push(StateDim::new(start + lag * self.hydro_count + h));
                            }
                        }
                    }
                }
                StateRegion::CommitmentHold => {
                    let start = self.state_dim_range(region).start;
                    let n_anticipated = self.n_anticipated;
                    let n_decision = self
                        .anticipated_resolution
                        .per_plant
                        .first()
                        .map_or(0, |plant| plant.decision_sets.len());
                    let mut live = vec![false; n_ant_state];
                    for stage_idx in 0..n_decision {
                        for_each_live_commitment_slot(self, stage_idx, |res, _| {
                            live[res.slot * n_anticipated + res.plant] = true;
                        });
                    }
                    mask.extend(
                        live.iter()
                            .enumerate()
                            .filter(|&(_, &is_live)| is_live)
                            .map(|(offset, _)| StateDim::new(start + offset)),
                    );
                }
            }
        }

        debug_assert!(
            mask.windows(2).all(|w| w[0] < w[1]),
            "nonzero_state_indices must be sorted and unique"
        );

        self.nonzero_state_indices = mask;
    }

    /// The z-inflow definition rows: one per hydro, leading every stage's row
    /// space.
    #[inline]
    #[must_use]
    pub fn z_inflow_rows(&self) -> Range<usize> {
        0..self.hydro_count
    }

    /// Hydro `h`'s z-inflow definition row.
    #[inline]
    #[must_use]
    pub fn z_inflow_row(&self, h: HydroSys) -> usize {
        debug_assert!(h.get() < self.hydro_count);
        self.z_inflow_rows().start + h.get()
    }

    /// Hydro `h`'s z-inflow column — the outgoing lag-0 state column.
    #[inline]
    #[must_use]
    pub(crate) fn z_inflow_col(&self, h: HydroSys) -> OutCol {
        debug_assert!(h.get() < self.hydro_count);
        OutCol::new(self.z_inflow.start + h.get())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::{
        AnticipatedResolution, HydroSys, InCol, OutCol, StateDim, StateSpace,
        for_each_live_commitment_slot,
    };
    use crate::lead_time::{DeliveryAxis, LeadTime, PointResolution};
    use crate::test_support::constant_lead_resolution;

    /// Build a [`StateSpace`] finalized the way production `resolve_state_layout`
    /// does: full `max_par_order` lag stride for every hydro (the coverage the
    /// dense path emits for test layouts without a PAR model), the layout's own
    /// `anticipated_lead_stages`, and an explicit `anticipated_resolution` —
    /// the single owner every other helper below delegates to.
    fn finalized_with_transit_buckets_and_resolution(
        hydro_count: usize,
        max_par_order: usize,
        transit_bucket_column_order: Vec<(HydroSys, usize)>,
        anticipated_lead_stages: Vec<usize>,
        anticipated_resolution: AnticipatedResolution,
    ) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        StateSpace::new(
            hydro_count,
            max_par_order,
            transit_bucket_column_order,
            anticipated_lead_stages,
            anticipated_resolution,
            &lag_counts,
        )
    }

    /// [`finalized_with_transit_buckets_and_resolution`] with no bucket block
    /// and no anticipated plants.
    fn finalized(
        hydro_count: usize,
        max_par_order: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateSpace {
        finalized_with_transit_buckets_and_resolution(
            hydro_count,
            max_par_order,
            Vec::new(),
            anticipated_lead_stages,
            AnticipatedResolution::default(),
        )
    }

    /// Same as [`finalized`] but with a declared bucket block
    /// (`transit_bucket_column_order`), for the bucket-arm resolver and mask
    /// tests.
    fn finalized_with_transit_buckets(
        hydro_count: usize,
        max_par_order: usize,
        transit_bucket_column_order: Vec<(HydroSys, usize)>,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateSpace {
        finalized_with_transit_buckets_and_resolution(
            hydro_count,
            max_par_order,
            transit_bucket_column_order,
            anticipated_lead_stages,
            AnticipatedResolution::default(),
        )
    }

    /// Like [`finalized`] but with a real, saturating
    /// [`AnticipatedResolution`] attached — the prerequisite the folded
    /// constructor requires at construction time for every `n_anticipated >
    /// 0` fixture that does not pin its own custom resolution.
    fn finalized_resolved(
        hydro_count: usize,
        max_par_order: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateSpace {
        finalized_with_transit_buckets_resolved(
            hydro_count,
            max_par_order,
            Vec::new(),
            anticipated_lead_stages,
        )
    }

    /// Like [`finalized_with_transit_buckets`] but with a real, saturating
    /// [`AnticipatedResolution`] attached (see [`finalized_resolved`]).
    fn finalized_with_transit_buckets_resolved(
        hydro_count: usize,
        max_par_order: usize,
        transit_bucket_column_order: Vec<(HydroSys, usize)>,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateSpace {
        let n_stages = anticipated_lead_stages.iter().copied().max().unwrap_or(0) + 2;
        let resolution = constant_lead_resolution(&anticipated_lead_stages, n_stages);
        finalized_with_transit_buckets_and_resolution(
            hydro_count,
            max_par_order,
            transit_bucket_column_order,
            anticipated_lead_stages,
            resolution,
        )
    }

    // ── state_to_lp_column precompute tests ─────────────────────────────────

    /// A finalized layout carrying storage + AR lags + anticipated thermals
    /// (every `state_to_lp_column` branch) must have
    /// `lp_column_for_state(j) == state_to_lp_column(j)` for every state index.
    #[test]
    fn lp_column_map_matches_resolver_with_lags_and_anticipated() {
        // hydro_count=3, max_par_order=2, n_anticipated=2 (K = [1, 2], k_max=2).
        let idx = finalized_resolved(3, 2, vec![1, 2]);

        assert_eq!(idx.state_to_lp_column_map.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(
                idx.lp_column_for_state(StateDim::new(j)),
                idx.state_to_lp_column(StateDim::new(j)),
                "finalized map must match the resolver at j={j}"
            );
        }
    }

    /// `StateSpace::new` always finalizes `state_to_lp_column_map` to `n_state`
    /// length, so `lp_column_for_state` reads the precomputed map directly with
    /// no live-resolver fallback. Cover every distinct layout shape (storage-only,
    /// storage + lags, and storage + lags + anticipated) to pin the
    /// always-finalized invariant the fallback removal relies on.
    #[test]
    fn lp_column_for_state_map_always_finalized() {
        for idx in [
            finalized(0, 0, vec![]),              // pure-thermal: n_state == 0
            finalized(3, 0, vec![]),              // storage-only
            finalized(2, 3, vec![]),              // storage + lags
            finalized_resolved(3, 2, vec![1, 2]), // storage + lags + anticipated
            finalized_with_transit_buckets_resolved(
                3,
                2,
                vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
                vec![1, 2],
            ), // storage + lags + buckets + anticipated
        ] {
            assert_eq!(
                idx.state_to_lp_column_map.len(),
                idx.n_state,
                "constructor must finalize the column map to n_state length"
            );
            for j in 0..idx.n_state {
                assert_eq!(
                    idx.lp_column_for_state(StateDim::new(j)),
                    idx.state_to_lp_column_map[j]
                );
            }
        }
    }

    /// Storage-only (`max_par_order == 0`, no anticipated): the mask is exactly
    /// `[0, n_state)` ascending and `lp_column_for_state(j) == j` — the
    /// dense→sparse bit-identity premise for the unified cut-row loop.
    #[test]
    fn lp_column_map_storage_only_mask_is_full_range() {
        let idx = finalized(3, 0, vec![]);

        assert_eq!(
            idx.nonzero_state_indices,
            vec![StateDim::new(0), StateDim::new(1), StateDim::new(2)]
        );
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(idx.lp_column_for_state(StateDim::new(j)), OutCol::new(j));
        }
    }

    // ── state_to_lp_column tests ──────────────────────────────────────────────

    /// `commit_out` indices resolve by identity when
    /// `max_par_order == 0 && n_anticipated > 0` — the ring transition
    /// (shift/deposit) is now an in-LP definition row, not a resolver-side
    /// remap. The `commit_out` branch runs before the
    /// `max_par_order == 0` lag-block guard; verify identity holds even when
    /// there are no inflow lags.
    #[test]
    fn state_to_lp_column_commit_out_is_identity_no_lag() {
        // N=1, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+0) + 1*2 = 3.
        // commit_out = [1, 3); slot 0 at j=1, slot 1 at j=2.
        let idx = finalized_resolved(1, 0, vec![2]);
        assert_eq!(idx.commit_out, 1..3);
        // Storage index: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));
        // Both ring slots: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(1)), OutCol::new(1));
        assert_eq!(idx.state_to_lp_column(StateDim::new(2)), OutCol::new(2));
    }

    /// `commit_out` indices resolve by identity when
    /// `max_par_order > 0 && n_anticipated > 0`, and the lag-remap branch for
    /// non-anticipated indices is unaffected. Fixture: N=1, L=1,
    /// `n_anticipated=1`, `k_max=2`, `anticipated_lead_stages=[2]`.
    #[test]
    fn state_to_lp_column_commit_out_is_identity_with_lag() {
        // N=1, L=1, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+1) + 1*2 = 4.
        // Layout: j=0 storage, j=1 lag-0, j=2 ant slot-0, j=3 ant slot-1.
        let idx = finalized_resolved(1, 1, vec![2]);
        assert_eq!(idx.commit_out, 2..4);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));
        // Lag block: remapped (unaffected by the anticipated-ring change).
        // j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        assert_eq!(
            idx.state_to_lp_column(StateDim::new(1)),
            OutCol::new(idx.z_inflow.start)
        );
        // Both ring slots: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(2)), OutCol::new(2));
        assert_eq!(idx.state_to_lp_column(StateDim::new(3)), OutCol::new(3));
    }

    /// Lag-remap branch is preserved when `n_anticipated == 0` and
    /// `max_par_order > 0`.  The anticipated-state guard must not fire
    /// when there are no anticipated thermals.
    #[test]
    fn state_to_lp_column_lag_remap_preserved_no_anticipated() {
        // N=1, L=1, n_anticipated=0 — classic PAR(p) case.
        // n_state = 1*(1+1) = 2. Layout: j=0 storage, j=1 lag-0.
        let idx = finalized(1, 1, vec![]);
        assert_eq!(idx.n_anticipated, 0);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));
        // Lag block j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        // For N=1, L=0 anticipated: z_inflow = N*(1+L)..N*(2+L) = 2..3.
        assert_eq!(idx.z_inflow.start, 2);
        assert_eq!(idx.state_to_lp_column(StateDim::new(1)), OutCol::new(2));
    }

    /// Every anticipated ring slot resolves by identity, regardless of which
    /// slot is a plant's own newest (delivery-deposit) slot, a plain-shift
    /// interior slot, or a stage-invariant padding slot beyond that plant's own
    /// `K_p`: the in-LP definition rows (built in `lp/builder`) resolve the
    /// ring transition, so `state_to_lp_column` never special-cases a slot
    /// index the way the deleted constant-lead shift-map did.
    #[test]
    fn state_to_lp_column_commit_out_identity_multi_plant_heterogeneous_k() {
        // Two plants: plant 0 has K_p=1 (only slot 0 is in-use), plant 1 has
        // K_p=3 (slots 0, 1, 2 all in-use). k_max=3 so plant 0 has padding
        // at slots 1 and 2.
        let idx = finalized_resolved(0, 0, vec![1, 3]);
        assert_eq!(idx.commit_out, 0..6);
        for j in idx.commit_out.clone() {
            assert_eq!(
                idx.state_to_lp_column(StateDim::new(j)),
                OutCol::new(j),
                "anticipated ring slot {j} must resolve by identity"
            );
        }
    }

    /// The anticipated-ring identity resolution lands inside the **state
    /// region**: `commit_out` sits strictly between
    /// `transit_buckets_out` and `theta`, never inside the control region.
    #[test]
    fn state_to_lp_column_commit_out_resolves_into_state_region() {
        // N=3, L=2, A=2, k_max=3, uniform K_p = 3.
        let idx = finalized_resolved(3, 2, vec![3, 3]);
        for j in idx.commit_out.clone() {
            let col = idx.state_to_lp_column(StateDim::new(j)).get();
            assert_eq!(col, j, "identity resolution");
            assert!(
                col >= idx.transit_buckets_out.end,
                "resolved column {col} must be >= transit_buckets_out.end {}",
                idx.transit_buckets_out.end
            );
            assert!(
                col < idx.theta,
                "resolved column {col} must be < theta {}",
                idx.theta
            );
        }
    }

    // ── state_to_lp_incoming_column tests ────────────────────────────────────

    /// Storage range: for a layout with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, N)` returns
    /// `storage_in.start + j`.
    #[test]
    fn state_to_lp_incoming_column_storage_range() {
        // N=3, L=2: storage_in.start = N*(2+L) = 3*4 = 12.
        let idx = finalized(3, 2, vec![]);
        assert_eq!(idx.storage_in.start, 12);
        for j in 0..3_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(StateDim::new(j)),
                InCol::new(idx.storage_in.start + j),
                "j={j}: expected storage_in.start + {j}"
            );
        }
    }

    /// AR lag range: for a layout with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [N, N*(1+L))` returns
    /// `inflow_lags.start + (j − N)`.
    #[test]
    fn state_to_lp_incoming_column_lag_range() {
        // N=3, L=2: inflow_lags = 3..9.
        let idx = finalized(3, 2, vec![]);
        assert_eq!(idx.inflow_lags.start, 3);
        for j in 3..9_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(StateDim::new(j)),
                InCol::new(idx.inflow_lags.start + (j - 3)),
                "j={j}: expected inflow_lags.start + {}",
                j - 3
            );
        }
    }

    /// Commitment-hold range: for a layout with `N=0, L=0, A=1, K=2`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, n_state)` returns
    /// `commit_in.start + j` (since `lag_end` = N*(1+L) = 0). With `N=0`
    /// every non-commitment-hold block collapses to `0..0`, so `commit_in`
    /// (the relocated incoming block, after `commit_out`/`z_inflow`/
    /// `storage_in`/`transit_buckets_in`) starts at `commit_out`'s own
    /// width (`A*K = 2`), not `0`.
    #[test]
    fn state_to_lp_incoming_column_anticipated_range() {
        // N=0, L=0, A=1, K=2: n_state = 0 + 1*2 = 2.
        let idx = finalized_resolved(0, 0, vec![2]);
        assert_eq!(idx.commit_in.start, 2);
        assert_eq!(idx.n_state, 2);
        for j in 0..2_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(StateDim::new(j)),
                InCol::new(idx.commit_in.start + j),
                "j={j}: expected commit_in.start + {j}"
            );
        }
    }

    /// Combined boundary-case test: `N=3, L=2, A=1, K=2`.
    /// Checks j = 0, 2, 3, 8, 9, 10 (the boundary points from the spec).
    #[test]
    fn state_to_lp_incoming_column_combined_layout() {
        // N=3, L=2, A=1, K=2:
        //   n_state = N*(1+L) + A*K = 3*3 + 1*2 = 11.
        //   inflow_lags.start = N = 3.
        //   storage_in.start = N*(1+L) + A*K + N = 3*3 + 1*2 + 3 = 14
        //     (storage + inflow_lags + commit_out + z_inflow).
        //   commit_in.start = storage_in.start + N = 17 (transit_buckets_in
        //     is empty; the relocated incoming block follows storage_in directly).
        //   lag_end = N*(1+L) = 9.
        let idx = finalized_resolved(3, 2, vec![2]);
        assert_eq!(idx.n_state, 11);
        // j=0: storage range → storage_in.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(0)),
            InCol::new(idx.storage_in.start),
            "j=0"
        );
        // j=2: storage range → storage_in.start + 2.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(2)),
            InCol::new(idx.storage_in.start + 2),
            "j=2"
        );
        // j=3: first lag → inflow_lags.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(3)),
            InCol::new(idx.inflow_lags.start),
            "j=3"
        );
        // j=8: last lag → inflow_lags.start + 5.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(8)),
            InCol::new(idx.inflow_lags.start + 5),
            "j=8"
        );
        // j=9: first anticipated-state → commit_in.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(9)),
            InCol::new(idx.commit_in.start),
            "j=9"
        );
        // j=10: last anticipated-state → commit_in.start + 1.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(10)),
            InCol::new(idx.commit_in.start + 1),
            "j=10"
        );
        // All returned columns must be within the LP's column range.
        for j in 0..idx.n_state {
            let col = idx.state_to_lp_incoming_column(StateDim::new(j)).get();
            assert!(
                col < idx.theta + 1,
                "j={j}: column {col} out of range (theta={})",
                idx.theta
            );
        }
    }

    /// For the lag range, `state_to_lp_incoming_column` and `state_to_lp_column`
    /// return different values (the former returns the incoming-lag column, the
    /// latter returns the `z_inflow` or lag-out column). For the storage range
    /// the results differ because `storage_in.start > 0` while
    /// `state_to_lp_column` returns `j` (outgoing, which starts at column 0).
    #[test]
    fn state_to_lp_incoming_column_differs_from_state_to_lp_column_for_lag() {
        // N=2, L=1: storage_in.start = N*(2+L) = 2*3 = 6.
        // state_to_lp_column(0) = 0 (outgoing storage).
        // state_to_lp_incoming_column(0) = storage_in.start + 0 = 6.
        let idx = finalized(2, 1, vec![]);
        // Storage range: incoming ≠ outgoing.
        assert_ne!(
            idx.state_to_lp_incoming_column(StateDim::new(0)).get(),
            idx.state_to_lp_column(StateDim::new(0)).get(),
            "storage range should differ: incoming={} outgoing={}",
            idx.state_to_lp_incoming_column(StateDim::new(0)).get(),
            idx.state_to_lp_column(StateDim::new(0)).get()
        );
        // j=0: incoming returns storage_in.start, outgoing returns 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(0)),
            InCol::new(idx.storage_in.start)
        );
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));

        // Lag range (j=2, j=3): incoming returns inflow_lags column;
        // outgoing returns z_inflow (lag=0) or lag-out (lag>=1) column.
        // j=2: lag 0, hydro 0. incoming = inflow_lags.start + 0.
        //                       outgoing = z_inflow.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(2)),
            InCol::new(idx.inflow_lags.start),
            "j=2 incoming should be inflow_lags.start"
        );
        assert_eq!(
            idx.state_to_lp_column(StateDim::new(2)),
            OutCol::new(idx.z_inflow.start),
            "j=2 outgoing should be z_inflow.start"
        );
        assert_ne!(
            idx.state_to_lp_incoming_column(StateDim::new(2)).get(),
            idx.state_to_lp_column(StateDim::new(2)).get(),
            "lag range should differ for j=2"
        );
        // j=3: lag 0, hydro 1. incoming = inflow_lags.start + 1.
        //                       outgoing = z_inflow.start + 1.
        assert_ne!(
            idx.state_to_lp_incoming_column(StateDim::new(3)).get(),
            idx.state_to_lp_column(StateDim::new(3)).get(),
            "lag range should differ for j=3"
        );
    }

    // ── Nonzero state mask tests ───────────────────────────────────────────

    #[test]
    fn nonzero_mask_mixed_ar_orders() {
        // 4 hydros (N=4), max_par_order=6 (L=6), ar_orders=[0, 1, 3, 6]
        // inflow_lags.start = N = 4
        // Lag-major layout: slot = 4 + lag * N + h
        let mut idx = finalized(4, 6, vec![]);
        idx.set_nonzero_mask(&[0, 1, 3, 6]);

        // Storage: [0, 1, 2, 3]
        // lag0: h1→4+0*4+1=5, h2→6, h3→7
        // lag1: h2→4+1*4+2=10, h3→11
        // lag2: h2→4+2*4+2=14, h3→15
        // lag3: h3→4+3*4+3=19
        // lag4: h3→4+4*4+3=23
        // lag5: h3→4+5*4+3=27
        // Total: 4 + 0 + 1 + 3 + 6 = 14
        assert_eq!(
            idx.nonzero_state_indices.len(),
            14,
            "mask length: 4 storage + 0 + 1 + 3 + 6 = 14"
        );

        assert_eq!(
            &idx.nonzero_state_indices[..4],
            &[0, 1, 2, 3].map(StateDim::new)
        );
        assert_eq!(
            &idx.nonzero_state_indices[4..],
            &[5, 6, 7, 10, 11, 14, 15, 19, 23, 27].map(StateDim::new)
        );

        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn nonzero_mask_zero_par_order() {
        // max_par_order=0: no lags, mask = storage only
        let mut idx = finalized(3, 0, vec![]);
        idx.set_nonzero_mask(&[0, 0, 0]);
        assert_eq!(idx.nonzero_state_indices.len(), 3);
        assert_eq!(&idx.nonzero_state_indices, &[0, 1, 2].map(StateDim::new));
    }

    #[test]
    fn nonzero_mask_all_full_order() {
        // All hydros at max AR order: mask covers all n_state indices
        let mut idx = finalized(2, 3, vec![]);
        idx.set_nonzero_mask(&[3, 3]);
        // n_state = 2*(1+3) = 8, mask should have 2 + 2*3 = 8
        assert_eq!(idx.nonzero_state_indices.len(), 8);
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
    }

    /// Regression test for the PAR(p)-A cut sparse-mask bug.
    ///
    /// `lag_counts` is the per-hydro count of lag-state
    /// slots that may carry non-zero cut coefficients — equal to
    /// `PrecomputedPar::effective_lag_count(h)`. When PAR(p)-A annual is active
    /// on a hydro this is `max_par_order` (= 12) even though the classical AR
    /// order is smaller, because `ψ̂/12` fills the trailing lag slots.
    ///
    /// Passing `par.order(h)` here instead of `effective_lag_count(h)` omits
    /// state coefficients on slots `order..max_par_order`, producing
    /// over-estimating cuts (LB > UB at convergence).
    #[test]
    fn nonzero_mask_par_a_includes_full_psi_stride() {
        // Two hydros: hydro 0 has classical AR(4); hydro 1 has PAR(4)-A and
        // therefore uses all 12 lag slots. max_par_order = 12 (widened by
        // PrecomputedPar when any model has an annual component).
        let mut idx = finalized(2, 12, vec![]);
        idx.set_nonzero_mask(&[4, 12]);

        // n_state = 2 * (1 + 12) = 26.
        // Mask = [storage 0..2] + [lag * 2 + h for lag in 0..lag_count[h]]
        //      = [0, 1] + [hydro 0 lags 0..4] + [hydro 1 lags 0..12]
        //      = 2 + 4 + 12 = 18 entries.
        assert_eq!(
            idx.nonzero_state_indices.len(),
            18,
            "PAR-A hydro must contribute all 12 lag slots to the cut mask; \
             omitting slots 4..12 (where ψ̂/12 lives) shifts the cut hyperplane \
             above the LP value at the visited state (over-estimating cuts)."
        );

        // Storage indices.
        assert_eq!(&idx.nonzero_state_indices[..2], &[0, 1].map(StateDim::new));

        // Hydro 0 (lag_count = 4): expect lag slots at indices
        //   inflow_lags.start + lag * hydro_count + h = 2 + lag*2 + 0 for lag in 0..4
        // → {2, 4, 6, 8}.
        // Hydro 1 (lag_count = 12): expect lag slots at indices
        //   2 + lag*2 + 1 for lag in 0..12 → {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}.
        // Mask is sorted globally. Confirm a few discriminating positions.
        assert!(
            idx.nonzero_state_indices.contains(&StateDim::new(25)),
            "lag-11 slot for hydro 1 (the trailing PAR-A annual slot) must be in the mask"
        );
        assert!(
            !idx.nonzero_state_indices.contains(&StateDim::new(10)),
            "lag-4 slot for hydro 0 (classical AR(4)) must NOT be in the mask"
        );
        // Sorted.
        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    // ── Commitment-hold in-study nonzero mask tests ────────────────────────

    /// `n_anticipated == 0` reproduces the pre-anticipated behaviour exactly.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_zero_anticipated_matches_existing() {
        let mut idx_with = finalized(4, 6, vec![]);
        idx_with.set_nonzero_mask(&[0, 1, 3, 6]);

        // Same expected mask as `nonzero_mask_mixed_ar_orders`:
        // [0,1,2,3] (storage) + [5,6,7,10,11,14,15,19,23,27] (lags).
        assert_eq!(
            idx_with.nonzero_state_indices,
            [0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 15, 19, 23, 27].map(StateDim::new)
        );
    }

    /// The extended mask is sorted ascending with no duplicates.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_sorted_ascending() {
        // Mixed configuration: 3 hydros with mixed lag_counts + 2 anticipated
        // plants with mixed K_i. The slot-major iteration over anticipated
        // must keep the global mask sorted.
        let mut idx = finalized_resolved(3, 2, vec![2, 3]);

        idx.set_nonzero_mask(&[1, 2, 0]);

        assert!(
            idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]),
            "mask must be strictly ascending with no duplicates: {:?}",
            idx.nonzero_state_indices
        );
    }

    /// The constructor's `CommitmentHold` mask tail matches an independent
    /// `for_each_live_commitment_slot` sweep over every decision stage — the
    /// same union `set_nonzero_mask`'s own `CommitmentHold` arm computes,
    /// proved against a two-plant mixed-lead resolution rather than against
    /// the code under test.
    #[test]
    fn state_space_mask_matches_live_commitment_slots() {
        let leads = vec![1, 3];
        let n_anticipated = leads.len();
        let n_stages = leads.iter().copied().max().unwrap_or(0) + 2;
        let resolution = constant_lead_resolution(&leads, n_stages);
        let state =
            finalized_with_transit_buckets_and_resolution(0, 0, Vec::new(), leads, resolution);

        let start = state.commit_out.start;
        let n_decision = state.anticipated_resolution.per_plant[0]
            .decision_sets
            .len();
        let mut live = vec![false; n_anticipated * state.k_max];
        for stage_idx in 0..n_decision {
            for_each_live_commitment_slot(&state, stage_idx, |res, _| {
                live[res.slot * n_anticipated + res.plant] = true;
            });
        }
        let expected: Vec<StateDim> = live
            .iter()
            .enumerate()
            .filter(|&(_, &is_live)| is_live)
            .map(|(offset, _)| StateDim::new(start + offset))
            .collect();

        let mask_tail: Vec<StateDim> = state
            .nonzero_state_indices
            .iter()
            .copied()
            .filter(|d| d.get() >= start)
            .collect();

        assert_eq!(mask_tail, expected);
    }

    /// A single shared lead across every plant: over `n_decision = 4` stages
    /// `(>= k_max = 2)`, each plant cycles through every residue, so the
    /// commitment-hold tail of the mask is the whole region.
    #[test]
    fn single_lead_nonzero_mask_keeps_the_whole_commitment_region() {
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2), LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &[720.0; 4],
                post_study_stage_hours: &[],
            },
        );
        let idx =
            finalized_with_transit_buckets_and_resolution(0, 0, Vec::new(), vec![2, 2], resolution);

        assert_eq!(idx.nonzero_state_indices, [0, 1, 2, 3].map(StateDim::new));
    }

    /// Accepted edge case: a single-lead study shorter than its own lead. The
    /// plant's lead (`K = 3`) exceeds the study's own delivery axis
    /// (`n_decision = n_delivery = 2`), so every window's ring-axis target
    /// `r >= n_delivery` is skipped except `r = 1` (the only in-window index
    /// below `n_delivery`, reachable only from `stage_idx = 0`, `depth = 0`).
    /// Residues 0 and 2 are never latched at any stage. Hand-derived against
    /// the ring's own addressing formula, never against
    /// `for_each_live_commitment_slot` — the oracle this test pins is
    /// independent of the code under test.
    #[test]
    fn single_lead_shorter_than_its_own_lead_mask_omits_the_never_latched_slots() {
        let point = PointResolution {
            decider: vec![None, None],
            decision_sets: vec![Vec::new(), Vec::new()],
            depth: vec![0, 0],
            occupancy: vec![1, 0],
        };
        let resolution = AnticipatedResolution {
            per_plant: vec![point],
        };
        let idx =
            finalized_with_transit_buckets_and_resolution(0, 0, Vec::new(), vec![3], resolution);

        assert_eq!(idx.nonzero_state_indices, [1].map(StateDim::new));
    }

    // ── Bucket block tests ─────────────────────────────────────────────────

    /// Bucket-arm resolution for `state_to_lp_column`: outgoing bucket state
    /// maps to its LP column by identity (the `storage` convention), not the
    /// lag remap — verified with lags present so the bucket check must
    /// correctly intercept before the modular lag decode.
    #[test]
    fn state_to_lp_column_transit_bucket_arm_is_identity() {
        // N=2, L=2 (lags present), B=3, no anticipated.
        let idx = finalized_with_transit_buckets(
            2,
            2,
            vec![
                (HydroSys::new(0), 1),
                (HydroSys::new(0), 2),
                (HydroSys::new(1), 1),
            ],
            vec![],
        );

        assert_eq!(idx.transit_buckets_out, 6..9);
        for j in idx.transit_buckets_out.clone() {
            assert_eq!(
                idx.state_to_lp_column(StateDim::new(j)),
                OutCol::new(j),
                "bucket state index {j} must map to its LP column by identity"
            );
        }
        // The last lag index (j=5, just below the bucket block) still resolves
        // via the lag remap, proving the bucket check does not swallow lag
        // indices.
        assert_eq!(idx.state_to_lp_column(StateDim::new(5)), OutCol::new(3));
    }

    /// Bucket-arm resolution for `state_to_lp_incoming_column`: bucket indices
    /// resolve to the pinned `transit_buckets_in` column via an explicit arm, not the
    /// anticipated arm — verified with anticipated state present so both arms
    /// are live and either could otherwise swallow the other's indices.
    #[test]
    fn state_to_lp_incoming_column_transit_bucket_arm_is_pinned_not_anticipated() {
        // N=2, L=1, B=2, A=1 (k_max=2, K=[2]).
        let idx = finalized_with_transit_buckets_resolved(
            2,
            1,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            vec![2],
        );

        assert_eq!(idx.transit_buckets_in, 12..14);
        assert_eq!(idx.commit_in.start, 14);

        // Bucket state indices: j=4, j=5 (lag_end = N*(1+L) = 4).
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(4)),
            InCol::new(idx.transit_buckets_in.start)
        );
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(5)),
            InCol::new(idx.transit_buckets_in.start + 1)
        );
        assert_ne!(
            idx.state_to_lp_incoming_column(StateDim::new(4)),
            InCol::new(idx.commit_in.start),
            "bucket index must not resolve to the anticipated catch-all"
        );

        // The first anticipated-state index (j=6) still resolves via its own
        // arm, immediately after the bucket range.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(6)),
            InCol::new(idx.commit_in.start)
        );
    }

    /// `state_to_lp_column_map.len() == n_state` for a bucket-only layout (no
    /// lags, no anticipated) — the map-length invariant isolated from the
    /// other blocks.
    #[test]
    fn state_to_lp_column_map_length_matches_n_state_with_transit_buckets_only() {
        let idx = finalized_with_transit_buckets(
            0,
            0,
            vec![
                (HydroSys::new(0), 1),
                (HydroSys::new(0), 2),
                (HydroSys::new(0), 3),
            ],
            vec![],
        );

        assert_eq!(idx.n_state, 3);
        assert_eq!(idx.state_to_lp_column_map.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(
                idx.lp_column_for_state(StateDim::new(j)),
                OutCol::new(j),
                "bucket-only identity map"
            );
        }
    }

    /// Mask contiguity with a masked bucket slot: a bucket block is always
    /// fully included (mirroring `storage`), coexisting with a genuinely
    /// excluded lag slot from an unrelated block — the overall mask stays
    /// sorted with the excluded lag index as the only gap.
    #[test]
    fn nonzero_mask_transit_bucket_block_full_range_with_masked_lag_slot() {
        // N=2, L=2, B=2 (single plant, depth 2), no anticipated. Hydro 0 has
        // lag_count=1 (lag slot 1 masked out); hydro 1 has lag_count=2 (full).
        let mut idx = finalized_with_transit_buckets(
            2,
            2,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            vec![],
        );

        idx.set_nonzero_mask(&[1, 2]);

        assert_eq!(idx.transit_buckets_out, 6..8);
        assert_eq!(
            idx.nonzero_state_indices,
            [0, 1, 2, 3, 5, 6, 7].map(StateDim::new),
            "hydro 0's lag-1 slot (index 4) must be excluded while the full \
             bucket block (6, 7) is included"
        );
        assert!(
            idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]),
            "mask must stay sorted and unique with a masked slot present"
        );
    }

    /// `B == 0` (no declared travel-time arc) collapses `transit_buckets_out`/
    /// `transit_buckets_in` to `0..0` and leaves every other offset the literal
    /// formula for `N=3, L=2, A=2, k_max=2`; a stray `+0` that reorders the
    /// sequential-offset chain would move one of these off its hardcoded value.
    #[test]
    fn state_layout_b_zero_is_byte_identical_to_pre_transit_bucket_layout() {
        let idx = finalized_resolved(3, 2, vec![1, 2]);

        assert_eq!(idx.n_buckets, 0);
        assert!(idx.transit_bucket_column_order.is_empty());
        assert_eq!(idx.transit_buckets_out, 0..0);
        assert_eq!(idx.transit_buckets_in, 0..0);

        assert_eq!(idx.storage, 0..3);
        assert_eq!(idx.inflow_lags, 3..9);
        assert_eq!(idx.commit_out, 9..13);
        assert_eq!(idx.z_inflow, 13..16);
        assert_eq!(idx.storage_in, 16..19);
        assert_eq!(idx.commit_in, 19..23);
        assert_eq!(idx.theta, 23);
        assert_eq!(idx.n_state, 13);
    }

    /// `A * k_max == 0` (no anticipated thermals) collapses `commit_out`/
    /// `commit_in` to `0..0` and reproduces the pre-anticipated-ring layout
    /// byte-for-byte (`N=3, L=2, B=2`).
    #[test]
    fn state_layout_a_zero_collapses_to_pre_anticipated_ring_layout() {
        let idx = finalized_with_transit_buckets(
            3,
            2,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            vec![],
        );

        assert_eq!(idx.n_anticipated, 0);
        assert_eq!(idx.k_max, 0);
        assert_eq!(idx.commit_out, 0..0);
        assert_eq!(idx.commit_in, 0..0);

        assert_eq!(idx.storage, 0..3);
        assert_eq!(idx.inflow_lags, 3..9);
        assert_eq!(idx.transit_buckets_out, 9..11);
        assert_eq!(idx.z_inflow, 11..14);
        assert_eq!(idx.storage_in, 14..17);
        assert_eq!(idx.transit_buckets_in, 17..19);
        assert_eq!(idx.theta, 19);
        assert_eq!(idx.n_state, 11);

        for j in 0..idx.n_state {
            assert_eq!(
                idx.lp_column_for_state(StateDim::new(j)),
                idx.state_to_lp_column(StateDim::new(j)),
                "A==0 collapse must not disturb the storage/lag/bucket resolvers"
            );
        }
    }

    /// `transit_bucket_plants` groups a two-plant `column_order` into each
    /// plant's own contiguous local sub-range, and an empty order yields
    /// nothing.
    #[test]
    fn transit_bucket_plants_groups_the_bucket_order_by_plant() {
        let h1 = HydroSys::new(1);
        let h3 = HydroSys::new(3);
        let idx = finalized_with_transit_buckets(
            4,
            0,
            vec![(h1, 1), (h1, 2), (h3, 1), (h3, 2), (h3, 3)],
            vec![],
        );

        let groups: Vec<(HydroSys, Range<usize>)> = idx.transit_bucket_plants().collect();
        assert_eq!(groups, vec![(h1, 0..2), (h3, 2..5)]);

        let empty = finalized(4, 0, vec![]);
        assert_eq!(empty.transit_bucket_plants().count(), 0);
    }

    // ── In-LP anticipated ring: masking + collapse ─────────────

    /// `k_max == 0` collapses the ring to empty even when `n_anticipated > 0`
    /// (defensive: `n_ant_state = n_anticipated * k_max` is the sole gate, not
    /// `n_anticipated` alone), matching the `A * k_max == 0` layout exactly.
    #[test]
    fn anticipated_ring_k_max_zero_collapses_even_with_plants_declared() {
        let zero_k_max = StateSpace::new(
            3,
            2,
            Vec::new(),
            vec![0, 0],
            constant_lead_resolution(&[0, 0], 2),
            &[2, 2, 2],
        );
        let no_plants = StateSpace::new(
            3,
            2,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &[2, 2, 2],
        );

        assert_eq!(zero_k_max.commit_out, 0..0);
        assert_eq!(zero_k_max.commit_in, 0..0);
        assert_eq!(zero_k_max.theta, no_plants.theta);
        assert_eq!(zero_k_max.n_state, no_plants.n_state);
        assert_eq!(
            zero_k_max.state_to_lp_column_map,
            no_plants.state_to_lp_column_map
        );
    }

    // ── CommitmentHold addressing helper ────────────────────────────────────

    /// `commitment_hold_in_study_offset` is delivery-target-modular: delivery
    /// targets congruent modulo `k_max` share the same plant's slot, and one
    /// full period of `m` bijects onto every `(plant, slot)` pair in the
    /// leading in-study block exactly once.
    #[test]
    fn commitment_hold_in_study_offset_is_delivery_target_modular_bijection() {
        let idx = finalized_resolved(0, 0, vec![4, 4, 4]);

        for plant in 0..idx.n_anticipated {
            for m in 0..idx.k_max {
                assert_eq!(
                    idx.commitment_hold_in_study_offset(plant, m),
                    idx.commitment_hold_in_study_offset(plant, m + idx.k_max * 3),
                    "delivery targets congruent mod k_max must share plant {plant}'s slot"
                );
            }
        }

        let mut offsets: Vec<usize> = Vec::new();
        for m in 0..idx.k_max {
            for plant in 0..idx.n_anticipated {
                offsets.push(idx.commitment_hold_in_study_offset(plant, m));
            }
        }
        offsets.sort_unstable();
        let expected: Vec<usize> = (0..idx.n_anticipated * idx.k_max).collect();
        assert_eq!(
            offsets, expected,
            "one period of m must biject onto the full leading in-study block"
        );
    }

    /// Two-plant [`AnticipatedResolution`] sibling to `single_plant_resolution`
    /// (defined below): plant 0's `decider` carries a `g`-long excised fixed
    /// post-horizon `None` run right after `n_decision` in-study entries; plant
    /// 1's post-study suffix is entirely decided (`g == 0`). The excision input
    /// is the `decider` itself, so no separate width parameter is threaded
    /// through — `decision_sets`/`depth`/`occupancy` stay structurally minimal
    /// since `commitment_hold_in_study_offset` never reads them.
    fn two_plant_resolution_with_fixed_window(
        n_decision: usize,
        g: usize,
    ) -> AnticipatedResolution {
        let decider_len = n_decision + g + 1;
        let with_window = PointResolution {
            decider: (0..decider_len)
                .map(|m| (m >= n_decision + g).then_some(0))
                .collect(),
            decision_sets: vec![Vec::new(); n_decision],
            depth: Vec::new(),
            occupancy: Vec::new(),
        };
        let without_window = PointResolution {
            decider: (0..decider_len)
                .map(|m| (m >= n_decision).then_some(0))
                .collect(),
            decision_sets: vec![Vec::new(); n_decision],
            depth: Vec::new(),
            occupancy: Vec::new(),
        };
        AnticipatedResolution {
            per_plant: vec![with_window, without_window],
        }
    }

    /// Shared fixture for the ring-axis offset tests: `n_anticipated = 2`,
    /// `k_max = 4`; plant 0 has a 3-wide excised fixed post-horizon window
    /// (`n_decision = 4`, `g = 3`); plant 1 has none.
    fn ring_axis_offset_fixture() -> StateSpace {
        finalized_with_transit_buckets_and_resolution(
            0,
            0,
            Vec::new(),
            vec![4, 4],
            two_plant_resolution_with_fixed_window(4, 3),
        )
    }

    /// `3 ≡ 7 mod 4` collide on the raw delivery axis, but plant 0's `g == 3`
    /// fixed post-horizon window shifts ring-eligible `m = 7` to ring index 4,
    /// landing on a distinct slot from `m = 3`.
    #[test]
    fn commitment_hold_offset_separates_the_colliding_raw_axis_residues() {
        let idx = ring_axis_offset_fixture();
        assert_eq!(idx.commitment_hold_in_study_offset(0, 7), 0);
        assert_eq!(idx.commitment_hold_in_study_offset(0, 3), 6);
    }

    /// Plant 1 has no fixed window (`g == 0`): its own offset is unaffected by
    /// sibling plant 0's excised width.
    #[test]
    fn commitment_hold_offset_ignores_a_sibling_plants_fixed_window() {
        let idx = ring_axis_offset_fixture();
        assert_eq!(idx.commitment_hold_in_study_offset(1, 7), 7);
    }

    /// `m = 5` sits inside plant 0's excised window `[4, 7)` — never a ring
    /// member — so the `debug_assert!` fires instead of silently falling back
    /// to the raw `m`.
    #[test]
    #[should_panic = "excised fixed"]
    fn commitment_hold_offset_rejects_an_excised_delivery_target() {
        let idx = ring_axis_offset_fixture();
        let _ = idx.commitment_hold_in_study_offset(0, 5);
    }

    // ── State-dimension region accessors ────────────────────────────────────

    /// The four `state_dim_*_range` accessors partition `[0, n_state)`
    /// contiguously with no gap and no overlap — storage, lags, buckets, and
    /// the merged commitment-hold region all present — the direct pin for the
    /// region order `CutStateProjection::new` consumes.
    #[test]
    fn state_dim_ranges_partition_n_state_contiguously() {
        // N=3, L=2, B=2, A=2, k_max=2: every region non-empty.
        let idx = finalized_with_transit_buckets_resolved(
            3,
            2,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            vec![1, 2],
        );

        let storage = idx.state_dim_storage_range();
        let lag = idx.state_dim_lag_range();
        let bucket = idx.state_dim_bucket_range();
        let commitment_hold = idx.state_dim_commitment_hold_range();

        assert!(
            !storage.is_empty()
                && !lag.is_empty()
                && !bucket.is_empty()
                && !commitment_hold.is_empty(),
            "fixture must exercise every region non-empty"
        );

        assert_eq!(storage.start, 0, "storage must start at the state origin");
        assert_eq!(
            lag.start, storage.end,
            "lag region must start exactly where storage ends (no gap/overlap)"
        );
        assert_eq!(
            bucket.start, lag.end,
            "bucket region must start exactly where lag ends (no gap/overlap)"
        );
        assert_eq!(
            commitment_hold.start, bucket.end,
            "commitment-hold region must start exactly where bucket ends (no gap/overlap)"
        );
        assert_eq!(
            commitment_hold.end, idx.n_state,
            "commitment-hold region must end exactly at n_state (no trailing gap)"
        );
    }

    /// [`StateSpace::classify_incoming_column`] must invert
    /// [`StateSpace::state_to_lp_incoming_column`] for every state dimension,
    /// with every region non-empty.
    #[test]
    fn classify_incoming_column_inverts_incoming_resolver() {
        let idx = finalized_with_transit_buckets_resolved(
            3,
            2,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            vec![1, 2],
        );
        for j in 0..idx.n_state {
            let dim = StateDim::new(j);
            let (region, offset) =
                idx.classify_incoming_column(idx.state_to_lp_incoming_column(dim));
            assert_eq!(
                idx.state_dim_range(region).start + offset,
                j,
                "incoming classification must round-trip state dim {j}"
            );
        }
    }

    /// `lag_state_dim` is lag-major, right after the storage region.
    #[test]
    fn lag_state_dim_is_lag_major_after_storage() {
        let idx = finalized(3, 2, vec![]);
        assert_eq!(idx.lag_state_dim(0, HydroSys::new(0)).get(), 3);
        assert_eq!(idx.lag_state_dim(1, HydroSys::new(2)).get(), 8);
        for l in 0..idx.max_par_order {
            for h in 0..idx.hydro_count {
                assert_eq!(idx.lag_state_dim(l, HydroSys::new(h)).get(), 3 + l * 3 + h);
            }
        }
    }

    /// The purpose-named column accessors resolve to the same columns as the
    /// raw range arithmetic they replaced at the extraction and manifest
    /// seams — the byte-neutrality pin for that migration.
    #[test]
    fn typed_state_col_accessors_match_block_layout() {
        let idx = finalized_with_transit_buckets_resolved(
            3,
            2,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            vec![1, 2],
        );
        for h in 0..idx.hydro_count {
            assert_eq!(
                idx.storage_incoming_col(HydroSys::new(h)).get(),
                idx.storage_in.start + h
            );
            assert_eq!(
                idx.storage_outgoing_col(HydroSys::new(h)).get(),
                idx.storage.start + h
            );
            for lag in 0..idx.max_par_order {
                assert_eq!(
                    idx.lag_incoming_col(lag, HydroSys::new(h)).get(),
                    idx.inflow_lags.start + lag * idx.hydro_count + h
                );
            }
        }
        for b in 0..idx.n_buckets {
            assert_eq!(
                idx.bucket_incoming_col(b).get(),
                idx.transit_buckets_in.start + b
            );
            assert_eq!(
                idx.bucket_outgoing_col(b).get(),
                idx.transit_buckets_out.start + b
            );
        }
    }

    /// Number of pinned addresses compared per family: `storage` covers both
    /// storage accessors and the z-inflow lag-0 arm, one per hydro; `lag`
    /// covers the lag ≥ 1 arm compared against `lag_incoming_col`. A
    /// degenerate fixture (`hydro_count == 0`) cannot pass the calling test's
    /// assertions vacuously.
    fn count_pinned_state_columns(idx: &StateSpace) -> (usize, usize) {
        let mut storage = 0;
        let mut lag = 0;
        for h in 0..idx.hydro_count {
            assert_eq!(idx.storage_outgoing_col(HydroSys::new(h)).get(), h);
            assert_eq!(
                idx.storage_incoming_col(HydroSys::new(h)).get(),
                idx.storage_in.start + h
            );
            assert_eq!(
                idx.state_to_lp_column(idx.lag_state_dim(0, HydroSys::new(h)))
                    .get(),
                idx.z_inflow.start + h
            );
            assert_eq!(
                idx.z_inflow_col(HydroSys::new(h)).get(),
                idx.z_inflow.start + h
            );
            storage += 1;
            for l in 1..idx.max_par_order {
                assert_eq!(
                    idx.state_to_lp_column(idx.lag_state_dim(l, HydroSys::new(h)))
                        .get(),
                    idx.lag_incoming_col(l - 1, HydroSys::new(h)).get()
                );
                lag += 1;
            }
        }
        (storage, lag)
    }

    /// The builder's exact storage/z-inflow/AR-lag column spellings — the
    /// migration pin for routing them through [`StateSpace`]'s own accessors
    /// instead of hand address arithmetic.
    #[test]
    fn builder_state_column_spellings_match_the_state_space() {
        for idx in [
            finalized(3, 2, vec![]),
            finalized_with_transit_buckets_resolved(
                3,
                2,
                vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
                vec![1, 2],
            ),
        ] {
            let (storage, lag) = count_pinned_state_columns(&idx);
            assert!(storage > 0 && lag > 0);
            assert_eq!(
                idx.inflow_lags,
                idx.inflow_lags.start..idx.inflow_lags.start + idx.max_par_order * idx.hydro_count
            );
            assert_eq!(idx.z_inflow.len(), idx.hydro_count);
        }
    }

    #[test]
    fn z_inflow_rows_lead_the_row_space_one_per_hydro() {
        let idx = finalized(5, 2, vec![]);
        assert_eq!(idx.z_inflow_rows(), 0..idx.hydro_count);
        for h in 0..idx.hydro_count {
            assert_eq!(idx.z_inflow_row(HydroSys::new(h)), h);
        }
    }

    /// The two commitment-hold column resolvers resolve to the exact columns the
    /// extraction sites recomposed by hand — `commit_in.start + offset` (incoming,
    /// pinned) and `commit_out.start + offset` (outgoing) — for every ring-member
    /// delivery target, the byte-neutrality pin for that migration.
    #[test]
    fn commitment_hold_col_accessors_match_extraction_recomposition() {
        let idx = finalized_with_transit_buckets_and_resolution(
            3,
            2,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            vec![1, 2],
            two_plant_resolution_with_fixed_window(4, 3),
        );
        for plant in 0..idx.n_anticipated {
            for m in 0..idx.k_max {
                let offset = idx.commitment_hold_in_study_offset(plant, m);
                assert_eq!(
                    idx.commitment_hold_incoming_col(plant, m).get(),
                    idx.commit_in.start + offset,
                    "plant {plant}, m {m}: incoming resolver must match commit_in.start + offset"
                );
                assert_eq!(
                    idx.commitment_hold_outgoing_col(plant, m).get(),
                    idx.commit_out.start + offset,
                    "plant {plant}, m {m}: outgoing resolver must match commit_out.start + offset"
                );
            }
        }
    }

    // ── n_delivery tests ────────────────────────────────────────

    /// Build an [`AnticipatedResolution`] whose single plant's `decider` has
    /// `decider_len` entries — the only field [`StateSpace::n_delivery`]
    /// reads.
    fn single_plant_resolution(decider_len: usize) -> AnticipatedResolution {
        AnticipatedResolution {
            per_plant: vec![PointResolution {
                decider: vec![None; decider_len],
                decision_sets: Vec::new(),
                depth: Vec::new(),
                occupancy: Vec::new(),
            }],
        }
    }

    /// An attached resolution whose single plant's `decider` extends past the
    /// study horizon: `n_delivery` returns the extended decider length.
    #[test]
    fn n_delivery_returns_the_attached_resolutions_extended_decider_length() {
        let idx = finalized_with_transit_buckets_and_resolution(
            0,
            0,
            Vec::new(),
            vec![2],
            single_plant_resolution(12),
        );
        assert_eq!(idx.n_delivery(), 12);
    }

    /// A study-only resolution: `n_delivery` returns exactly the decider
    /// length.
    #[test]
    fn n_delivery_returns_the_attached_resolutions_study_only_decider_length() {
        let idx = finalized_with_transit_buckets_and_resolution(
            0,
            0,
            Vec::new(),
            vec![2],
            single_plant_resolution(4),
        );
        assert_eq!(idx.n_delivery(), 4);
    }

    /// A zero-anticipated study (empty `per_plant`): `n_delivery` is `0`, and
    /// the per-plant decider-length-agreement `debug_assert` does not fire.
    #[test]
    fn n_delivery_is_zero_without_anticipated_plants() {
        let idx = finalized(0, 0, vec![]);
        assert_eq!(idx.n_delivery(), 0);
    }
}
