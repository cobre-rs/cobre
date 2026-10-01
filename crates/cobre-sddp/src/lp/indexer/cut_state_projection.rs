//! Storage-scoped projection of the global [`StateSpace`] onto the cut-state
//! dimensions a stage enables, for cut storage, subgradient extraction (incoming
//! columns), and cut rendering (outgoing columns).

use cobre_core::temporal::StageStateConfig;

use super::{CutSlot, InCol, OutCol, REGION_ORDER, StateDim, StateSpace};

/// A storage-scoped view of the global state vector exposing only the cut-state
/// dimensions a stage enables, plus travel-time buckets and anticipated state
/// (always included, never gated by [`StageStateConfig`]).
///
/// A projection **of** a [`StateSpace`], not a sibling layout: it delegates all
/// column arithmetic to the global layout and never drives the LP.
///
/// ## Default-identity contract
///
/// When `storage: true, inflow_lags: true` the projection is the identity:
/// [`Self::n_slots`] equals [`StateSpace::n_state`],
/// [`Self::incoming_column`] equals the global incoming resolver for
/// every `j`, and [`Self::render_pairs`] reproduces the global
/// `nonzero_state_indices` render exactly. Holds because the construction walks
/// the state index space in `REGION_ORDER`'s storage → lag → buckets →
/// anticipated order — the single owner [`StateSpace::set_nonzero_mask`] also
/// walks. The forbidden alternative is reordering dimensions (e.g. anticipated
/// before buckets), landing a cut coefficient on the wrong LP column and
/// silently corrupting every existing study.
///
/// ## Incoming vs outgoing vs render
///
/// Incoming ([`Self::incoming_column`]) and outgoing
/// ([`Self::outgoing_column`]) both span the full enabled range
/// `[0, n_slots())`, including structurally-zero AR- and anticipated-padding
/// slots, because dual extraction and DCS scoring read every enabled dimension.
/// Render ([`Self::render_pairs`]) is the **nonzero subset** (padding dropped,
/// matching the global `nonzero_state_indices`), because a cut row emits no entry
/// for a structurally-zero coefficient.
#[derive(Debug, Clone)]
pub struct CutStateProjection {
    /// Global state-vector [`StateDim`] per cut slot — the projection's own
    /// global→projected gather index (identity for an all-enabled pool).
    global_state_indices: Vec<StateDim>,

    /// LP incoming column per cut slot (extraction hot path).
    incoming_columns: Vec<InCol>,

    /// LP outgoing column per cut slot (DCS scoring); parallel to
    /// [`Self::incoming_columns`].
    outgoing_columns: Vec<OutCol>,

    /// Cut slot per render pair, nonzero-subset order; parallel to
    /// [`Self::render_columns`].
    render_coeff_indices: Vec<CutSlot>,

    /// LP outgoing column per render pair; parallel to
    /// [`Self::render_coeff_indices`].
    render_columns: Vec<OutCol>,
}

impl CutStateProjection {
    /// Project the global [`StateSpace`] onto the cut-state dimensions
    /// `state_config` enables, with travel-time buckets and anticipated state
    /// always included, walking `REGION_ORDER`'s storage → lag → buckets →
    /// anticipated order (the default-identity contract) via
    /// `StateSpace`'s `state_dim_range`, the sole owner of the boundary
    /// arithmetic.
    ///
    /// The per-slot fan-in cannot swap roles: an outgoing column pushed onto
    /// the incoming-column vector fails to compile.
    ///
    /// ```compile_fail
    /// use cobre_sddp::indexer::{InCol, StateDim, StateSpace};
    ///
    /// fn misuse(global: &StateSpace) {
    ///     let outgoing = global.lp_column_for_state(StateDim::new(0));
    ///     let mut incoming_columns: Vec<InCol> = Vec::new();
    ///     incoming_columns.push(outgoing); // fan-in swap: outgoing pushed onto the incoming-column vec
    /// }
    /// ```
    #[must_use]
    pub fn new(global: &StateSpace, state_config: StageStateConfig) -> Self {
        let mut global_state_indices = Vec::new();
        let mut incoming_columns = Vec::new();
        let mut outgoing_columns = Vec::new();
        let mut render_coeff_indices = Vec::new();
        let mut render_columns = Vec::new();

        let mut push_dim = |g: StateDim| {
            let reduced_j = incoming_columns.len();
            let outgoing = global.lp_column_for_state(g);
            global_state_indices.push(g);
            incoming_columns.push(global.state_to_lp_incoming_column(g));
            outgoing_columns.push(outgoing);
            // Drop padding slots, never zero-fill: keeps the default render
            // bit-identical to the global nonzero_state_indices render.
            if global.nonzero_state_indices.binary_search(&g).is_ok() {
                render_coeff_indices.push(CutSlot::new(reduced_j));
                render_columns.push(outgoing);
            }
        };

        for region in REGION_ORDER {
            if region.cut_enabled(state_config) {
                for g in global.state_dim_range(region) {
                    push_dim(StateDim::new(g));
                }
            }
        }

        debug_assert!(
            !(state_config.storage && state_config.inflow_lags)
                || incoming_columns.len() == global.n_state,
            "default (all-enabled) projection must reproduce the global n_state \
             ({}); got {}",
            global.n_state,
            incoming_columns.len()
        );

        Self {
            global_state_indices,
            incoming_columns,
            outgoing_columns,
            render_coeff_indices,
            render_columns,
        }
    }

    /// Count of enabled cut-state dimensions:
    /// `(storage ? N : 0) + (inflow_lags ? N*L : 0) + B + A*k_max`.
    #[inline]
    #[must_use]
    pub fn n_slots(&self) -> usize {
        self.incoming_columns.len()
    }

    #[inline]
    fn checked_slot(&self, s: CutSlot) -> usize {
        let j = s.get();
        debug_assert!(
            j < self.n_slots(),
            "cut slot {j} out of bounds (n_slots = {})",
            self.n_slots()
        );
        j
    }

    /// Map a cut slot `s ∈ [0, n_slots())` to the global [`StateDim`] it
    /// projects — the gather index for reading a `StateDim`-packed trial-state
    /// vector into the pool's projected slot space.
    ///
    /// Identity for an all-enabled pool (`s == global_state_index(s).get()`); a
    /// reduced pool selects the enabled dimensions, which are NOT a prefix (a
    /// `storage:false` pool begins at the first inflow-lag [`StateDim`]). Index a
    /// `StateDim`-packed archive through this, never [`Self::outgoing_column`],
    /// which remaps inflow-lag slots off the [`StateDim`] axis (to `z_inflow`,
    /// outside `[0, n_state)`).
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `s >= n_slots()`.
    #[inline]
    #[must_use]
    pub fn global_state_index(&self, s: CutSlot) -> StateDim {
        self.global_state_indices[self.checked_slot(s)]
    }

    /// Dot this pool's projected cut `coefficients` (length [`Self::n_slots`])
    /// with the full trial-state vector `x_hat` (length
    /// [`StateSpace::n_state`](super::StateSpace::n_state)), gathering each
    /// slot's value through [`Self::global_state_index`]. Cut-intercept
    /// construction (`Q(x̂) − β·x̂`) is its sole caller.
    ///
    /// A positional `coefficients.iter().zip(x_hat)` is the wrong-but-compiling
    /// alternative: it holds only for the all-enabled identity projection; once
    /// a region is dropped (a `storage:false`/`inflow_lags:false` pool), slot
    /// `j` past the gap projects a global dimension `> j`, so the zip pairs the
    /// coefficient with the wrong dimension's value — a silently over- or
    /// under-stated intercept, hence an invalid bound.
    #[inline]
    #[must_use]
    pub(crate) fn dot_trial_state(&self, coefficients: &[f64], x_hat: &[f64]) -> f64 {
        debug_assert_eq!(
            coefficients.len(),
            self.n_slots(),
            "coefficients length {} != n_slots {}",
            coefficients.len(),
            self.n_slots()
        );
        coefficients
            .iter()
            .enumerate()
            .map(|(j, &c)| c * x_hat[self.global_state_index(CutSlot::new(j)).get()])
            .sum()
    }

    /// Map a cut slot `s ∈ [0, n_slots())` to its LP incoming-state column,
    /// indexing the enabled subset in storage → lag → buckets → anticipated
    /// order.
    ///
    /// Neither the global [`StateDim`] nor the outgoing-column role [`OutCol`]
    /// can substitute for the [`CutSlot`] `s`: state-dimension and cut-slot-space
    /// diverge under a non-identity projection, and outgoing is this accessor's
    /// return role, not its parameter.
    ///
    /// ```compile_fail
    /// use cobre_core::temporal::StageStateConfig;
    /// use cobre_sddp::indexer::{CutStateProjection, StateDim, StateSpace};
    ///
    /// fn misuse(global: &StateSpace) {
    ///     let cut = CutStateProjection::new(
    ///         global,
    ///         StageStateConfig { storage: true, inflow_lags: true },
    ///     );
    ///     let _ = cut.incoming_column(StateDim::new(0)); // StateDim substituted for CutSlot
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use cobre_core::temporal::StageStateConfig;
    /// use cobre_sddp::indexer::{CutStateProjection, OutCol, StateSpace};
    ///
    /// fn misuse(global: &StateSpace) {
    ///     let cut = CutStateProjection::new(
    ///         global,
    ///         StageStateConfig { storage: true, inflow_lags: true },
    ///     );
    ///     let _ = cut.incoming_column(OutCol::new(0)); // OutCol substituted for CutSlot
    /// }
    /// ```
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `s >= n_slots()`.
    #[inline]
    #[must_use]
    pub fn incoming_column(&self, s: CutSlot) -> InCol {
        self.incoming_columns[self.checked_slot(s)]
    }

    /// Map a cut slot `s ∈ [0, n_slots())` to its LP **outgoing**-state
    /// column — the column its coefficient is dotted against in DCS scoring,
    /// indexing the enabled subset in storage → lag → buckets → anticipated
    /// order.
    ///
    /// The per-pool analogue of [`StateSpace::lp_column_for_state`], spanning the
    /// full enabled range (padding included); see the struct-level "Incoming vs
    /// outgoing vs render" section for the contrast with [`Self::render_pairs`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `s >= n_slots()`.
    #[inline]
    #[must_use]
    pub fn outgoing_column(&self, s: CutSlot) -> OutCol {
        self.outgoing_columns[self.checked_slot(s)]
    }

    /// Iterate the cut-row render pairs `(cut_slot, outgoing_lp_column)` for this
    /// pool, in nonzero-subset order (storage → lag → buckets → anticipated,
    /// padding slots dropped).
    ///
    /// `cut_slot` indexes a stored cut's `coefficients` slice (length
    /// [`Self::n_slots`]); `outgoing_lp_column` is where the cut-row builder places
    /// the negated, scaled coefficient — identity for storage; for a lag
    /// dimension the outgoing state (after `shift_lag_state`) holds `z_inflow`
    /// at lag 0 and the shifted incoming lags at lag 1+, so the column
    /// addresses `z_inflow` / incoming lag `l−1`. For an all-enabled study this reproduces
    /// the global `nonzero_state_indices` render — same reduced index, same column.
    #[inline]
    #[must_use]
    pub fn render_pairs(&self) -> impl ExactSizeIterator<Item = (CutSlot, OutCol)> + '_ {
        self.render_coeff_indices
            .iter()
            .copied()
            .zip(self.render_columns.iter().copied())
    }

    /// Count of cut-row render entries (nonzero-subset, padding dropped) — the
    /// per-cut non-zero state-coefficient count a builder emits before the theta
    /// entry.
    #[inline]
    #[must_use]
    pub fn render_len(&self) -> usize {
        self.render_coeff_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{CutSlot, CutStateProjection, InCol, OutCol, StageStateConfig, StateDim};
    use crate::indexer::{HydroSys, StateRegion, StateSpace};
    use crate::lead_time::AnticipatedResolution;
    use crate::test_support::constant_lead_resolution;

    fn finalized(
        hydro_count: usize,
        max_par_order: usize,
        anticipated_lead_stages: &[usize],
    ) -> StateSpace {
        finalized_with_transit_buckets(
            hydro_count,
            max_par_order,
            Vec::new(),
            anticipated_lead_stages,
        )
    }

    /// Like [`finalized`] but with a declared bucket block.
    fn finalized_with_transit_buckets(
        hydro_count: usize,
        max_par_order: usize,
        transit_bucket_column_order: Vec<(HydroSys, usize)>,
        anticipated_lead_stages: &[usize],
    ) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        let n_stages = anticipated_lead_stages.iter().copied().max().unwrap_or(0) + 2;
        let resolution = constant_lead_resolution(anticipated_lead_stages, n_stages);
        StateSpace::new(
            hydro_count,
            max_par_order,
            transit_bucket_column_order,
            anticipated_lead_stages.to_owned(),
            resolution,
            &lag_counts,
        )
    }

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };

    #[test]
    fn default_projection_is_identity() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_state, 9);
        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                global.state_to_lp_incoming_column(StateDim::new(j)),
                "default projection must match the global resolver at j={j}"
            );
        }
    }

    #[test]
    fn storage_only_projection() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3);
        for j in 0..3 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                InCol::new(global.storage_in.start + j),
                "storage-only slot {j} must map to storage_in.start + {j}"
            );
        }
    }

    #[test]
    fn global_state_index_is_identity_for_all_enabled() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(cut.global_state_index(CutSlot::new(j)), StateDim::new(j));
        }
    }

    /// `storage:false, inflow_lags:true` selects the lag dimensions, which begin
    /// at `StateDim` `N` — NOT a prefix, and where `outgoing_column` would remap
    /// off the `StateDim` axis; `global_state_index` selects the raw lag dims.
    #[test]
    fn global_state_index_selects_nonprefix_enabled_dims() {
        let global = finalized(2, 1, &[]);
        let cut = CutStateProjection::new(
            &global,
            StageStateConfig {
                storage: false,
                inflow_lags: true,
            },
        );

        assert_eq!(cut.n_slots(), global.hydro_count * global.max_par_order);
        for i in 0..cut.n_slots() {
            assert_eq!(
                cut.global_state_index(CutSlot::new(i)),
                StateDim::new(global.hydro_count + i),
                "lag slot {i} projects global StateDim N + {i}, not a prefix index"
            );
        }
    }

    /// A reduced projection must dot its coefficients against `x_hat` gathered
    /// through `global_state_index`, never positionally: the cut-intercept
    /// regression. `inflow_lags:false` drops the lag block, so the anticipated
    /// slot projects a global dim past its own slot index; a positional
    /// `zip(x_hat)` would multiply the anticipated coefficient by a lag value.
    #[test]
    fn dot_trial_state_gathers_reduced_projection_not_positional() {
        // N=2 storage, L=3 (6 lag dims), A=1/k_max=1 (1 anticipated): full state
        // is 9 dims — storage [0,2), lag [2,8), anticipated {8}.
        let global = finalized(2, 3, &[1]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3, "storage(2) + anticipated(1), lag dropped");
        assert_eq!(
            cut.global_state_index(CutSlot::new(2)),
            StateDim::new(8),
            "the anticipated slot projects global dim 8, not its positional index 2"
        );

        let x_hat = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
        ];
        let coeffs = vec![1.0, 1.0, 1.0];
        assert_eq!(
            cut.dot_trial_state(&coeffs, &x_hat),
            100.0 + 101.0 + 108.0,
            "gathered dot pairs the anticipated coeff with x_hat[8], not x_hat[2]"
        );
        assert_ne!(
            cut.dot_trial_state(&coeffs, &x_hat),
            100.0 + 101.0 + 102.0,
            "a positional zip (the bug) would pair it with a lag value"
        );
    }

    /// For an all-enabled (identity) projection `dot_trial_state` reduces to the
    /// positional `zip(x_hat)` bit-for-bit — the byte-neutrality the reduced-case
    /// gather must not disturb.
    #[test]
    fn dot_trial_state_all_enabled_matches_positional_zip() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        let x_hat = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        assert_eq!(x_hat.len(), global.n_state);
        let coeffs: Vec<f64> = vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0];
        assert_eq!(coeffs.len(), cut.n_slots());

        let positional: f64 = coeffs.iter().zip(&x_hat).map(|(c, x)| c * x).sum();
        assert_eq!(cut.dot_trial_state(&coeffs, &x_hat), positional);
    }

    /// `inflow_lags` disabled drops the lag dims but keeps anticipated:
    /// `n_slots() = N + A*k_max = 2 + 2 = 4`.
    #[test]
    fn storage_only_with_anticipated_includes_anticipated() {
        let global = finalized(2, 1, &[2]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 4);
        for j in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                InCol::new(global.storage_in.start + j)
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(2 + i)),
                InCol::new(global.commit_in.start + i),
                "anticipated slot {i} must map to commit_in.start + {i}"
            );
        }
    }

    /// `storage: false`: cut state begins at the first enabled dimension, the
    /// inflow lags (`N*L = 2` slots, no storage or anticipated).
    #[test]
    fn storage_disabled_begins_at_first_enabled_dimension() {
        let global = finalized(2, 1, &[]);
        let cut = CutStateProjection::new(
            &global,
            StageStateConfig {
                storage: false,
                inflow_lags: true,
            },
        );

        assert_eq!(cut.n_slots(), global.hydro_count * global.max_par_order);
        for i in 0..cut.n_slots() {
            assert_eq!(
                cut.incoming_column(CutSlot::new(i)),
                InCol::new(global.inflow_lags.start + i),
                "first enabled dimension is the inflow lags: slot {i}"
            );
        }
    }

    // ── Outgoing projection / render-pairs tests ──────────────────────────────

    #[test]
    fn default_render_matches_global_nonzero_mask() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.outgoing_column(CutSlot::new(j)),
                global.lp_column_for_state(StateDim::new(j)),
                "default outgoing projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "render_pairs must reproduce the global nonzero_state_indices render"
        );
        assert_eq!(cut.render_len(), global.nonzero_state_indices.len());
    }

    #[test]
    fn storage_only_render_touches_only_storage_columns() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3);
        assert_eq!(cut.render_len(), 3);

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        // Storage is identity under the outgoing resolver: state_to_lp_column(j)=j.
        assert_eq!(
            rendered,
            vec![
                (CutSlot::new(0), OutCol::new(0)),
                (CutSlot::new(1), OutCol::new(1)),
                (CutSlot::new(2), OutCol::new(2))
            ]
        );

        // No rendered column lands in the lag block [N, N*(1+L)) = [3, 9).
        for (_, col) in &rendered {
            assert!(
                !global.inflow_lags.contains(&col.get()),
                "storage-only render must touch no lag column (got {})",
                col.get()
            );
        }
    }

    /// AR-padding slots are dropped from the render but kept in the full enabled
    /// range. Per-hydro lag counts `[1, 3]` give hydro 0 padding at lags 1,2, so
    /// the render (nonzero subset) is shorter than `n_slots`.
    #[test]
    fn render_drops_ar_padding_slots() {
        let lag_counts = [1usize, 3];
        let global = StateSpace::new(
            2,
            3,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &lag_counts,
        );
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_slots(), global.n_state);
        assert_eq!(cut.render_len(), global.nonzero_state_indices.len());
        assert!(
            cut.render_len() < cut.n_slots(),
            "padding slots must be dropped, so render_len < n_slots"
        );

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(rendered, global_render);
    }

    // ── Bucket block tests ────────────────────────────────────────────────────

    /// `inflow_lags` disabled, but the bucket block stays included (never gated
    /// by `StageStateConfig`): `n_slots() = N + B + A*k_max = 2 + 2 + 2 = 6`, with
    /// bucket slots between storage and anticipated.
    #[test]
    fn bucket_block_always_included_with_storage_only() {
        let global = finalized_with_transit_buckets(
            2,
            1,
            vec![(HydroSys::new(0), 1), (HydroSys::new(1), 1)],
            &[2],
        );
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 6);

        for j in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                InCol::new(global.storage_in.start + j),
                "storage slot {j} must map to storage_in.start + {j}"
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(2 + i)),
                InCol::new(global.transit_buckets_in.start + i),
                "bucket slot {i} must map to transit_buckets_in.start + {i} despite \
                 inflow_lags disabled"
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(4 + i)),
                InCol::new(global.commit_in.start + i),
                "anticipated slot {i} must map to commit_in.start + {i}"
            );
        }
    }

    /// The ordering regression guard for the storage→lag→buckets→anticipated walk:
    /// swapping buckets and anticipated would misassign reduced coefficient
    /// indices even though every `(index, column)` pair still resolves to a valid
    /// column.
    #[test]
    fn bucket_render_pairs_sit_between_lag_and_anticipated() {
        let global = finalized_with_transit_buckets(
            2,
            1,
            vec![(HydroSys::new(0), 1), (HydroSys::new(1), 1)],
            &[2],
        );
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.transit_buckets_out, 4..6);
        assert_eq!(cut.n_slots(), global.n_state);

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "render_pairs must match the global walk order exactly, \
             storage→lag→buckets→anticipated"
        );

        let transit_bucket_positions: Vec<usize> = rendered
            .iter()
            .enumerate()
            .filter(|&(_, &(_, col))| global.transit_buckets_out.contains(&col.get()))
            .map(|(pos, _)| pos)
            .collect();
        assert_eq!(
            transit_bucket_positions,
            vec![4, 5],
            "bucket render pairs must land after the lag block (positions 0..4) \
             and before the anticipated block (positions 6..8)"
        );
    }

    /// `B == 0` regression guard for the always-included bucket block: the
    /// projection reproduces `[0, StateSpace::n_state)` and its render
    /// byte-identically to the pre-bucket walk.
    #[test]
    fn b_zero_projection_matches_pre_transit_bucket_walk() {
        let global = finalized_with_transit_buckets(3, 2, vec![], &[1, 2]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_buckets, 0);
        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                global.state_to_lp_incoming_column(StateDim::new(j)),
                "B==0 default projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "B==0 render must reproduce the global nonzero_state_indices render"
        );
    }

    /// A bucket region deeper than a terminal stage's horizon cap still projects
    /// EVERY bucket dim — the deep-lag slots `horizon_cap_active` freezes `[0, 0]`
    /// at the terminal included. `CutStateProjection::new` keys bucket inclusion on
    /// `StateRegion::Buckets` being cut-enabled and walks the whole
    /// `state_dim_range`, with no entity-type or per-stage gate, so the terminal
    /// bucket-state pricing path (`β·bucket_state`) is already wired at the
    /// projection level: un-masking a deep-lag slot adds no projection code.
    #[test]
    fn every_bucket_dim_projects_including_deep_terminal_lags() {
        // One downstream plant, depth 4: lags 1..=3 sit beyond the terminal cap
        // `n_stages − 1 − t` (which reaches 0 at the terminal, keeping only lag 0),
        // so they are frozen `[0, 0]` in the LP today; the state layout retains
        // them (sized from the global max over every anchor).
        let global = finalized_with_transit_buckets(
            1,
            1,
            vec![
                (HydroSys::new(0), 0),
                (HydroSys::new(0), 1),
                (HydroSys::new(0), 2),
                (HydroSys::new(0), 3),
            ],
            &[],
        );
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert!(
            StateRegion::Buckets.cut_enabled(ALL_ENABLED),
            "buckets are the always-included region the projection walk keys on"
        );

        let buckets = global.state_dim_range(StateRegion::Buckets);
        assert_eq!(buckets.len(), global.n_buckets);

        for g in buckets.clone() {
            let dim = StateDim::new(g);
            let offset = g - buckets.start;

            let projecting_slots: Vec<CutSlot> = (0..cut.n_slots())
                .map(CutSlot::new)
                .filter(|&s| cut.global_state_index(s) == dim)
                .collect();
            assert_eq!(
                projecting_slots.len(),
                1,
                "bucket dim {g} must project to exactly one cut slot"
            );
            let slot = projecting_slots[0];

            assert_eq!(
                cut.incoming_column(slot),
                InCol::new(global.transit_buckets_in.start + offset),
                "bucket dim {g}: incoming column is the pinned bucket column (subgradient read site)"
            );
            assert_eq!(
                cut.outgoing_column(slot),
                OutCol::new(global.transit_buckets_out.start + offset),
                "bucket dim {g}: outgoing column is the identity bucket column (cut-render site)"
            );

            let rendered: Vec<OutCol> = cut
                .render_pairs()
                .filter(|&(s, _)| s == slot)
                .map(|(_, col)| col)
                .collect();
            assert_eq!(
                rendered.len(),
                1,
                "bucket dim {g} must appear in exactly one render pair"
            );
            assert_eq!(
                rendered[0],
                OutCol::new(global.transit_buckets_out.start + offset)
            );
        }
    }

    // ── Commitment-hold ring tests ──────────────────────────────────────────

    /// The commitment-hold region — the single in-study anticipated ring that
    /// now carries every post-study target directly — joins the projection
    /// completely: `n_slots()` grows by exactly the ring's width over the
    /// pre-ring dimension, and every ring slot is present, mirroring the
    /// always-included bucket contract. No entity-type or per-stage arm gates
    /// this inclusion.
    #[test]
    fn commitment_hold_post_study_target_joins_the_projection() {
        let pre_ring = finalized(2, 1, &[]);
        let with_ring = finalized(2, 1, &[2]);

        let cut_pre = CutStateProjection::new(&pre_ring, ALL_ENABLED);
        let cut_with = CutStateProjection::new(&with_ring, ALL_ENABLED);

        assert_eq!(
            cut_with.n_slots(),
            cut_pre.n_slots() + with_ring.n_anticipated * with_ring.k_max,
            "n_slots must grow by exactly the anticipated-ring width"
        );
        assert_eq!(cut_with.n_slots(), with_ring.n_state);

        for (i, j) in with_ring.commit_in.clone().enumerate() {
            assert_eq!(
                cut_with.incoming_column(CutSlot::new(cut_pre.n_slots() + i)),
                InCol::new(j),
                "commitment-hold ring incoming slot {j} must appear in the projection"
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::{CutSlot, CutStateProjection, OutCol, StageStateConfig, StateDim};
    use crate::indexer::{HydroSys, StateSpace};
    use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };

    /// Fixed cases/seed so a failing shrink is reproducible run-to-run (the
    /// determinism discipline applied to test generation itself).
    fn fixed_config() -> ProptestConfig {
        ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        }
    }

    /// A valid [`StateSpace`] over the small parameter space (`hydro_count`,
    /// `max_par_order`, `n_buckets`, `n_anticipated` each `0..=4` or `0..=3`,
    /// per-plant leads `0..=3`, a delivery axis `n_decision <= n_delivery`
    /// both `<= 7`), with every dependent-length vector sized and bounded to
    /// satisfy `StateSpace::new`'s debug-asserts:
    /// `anticipated_lead_stages.len() == n_anticipated`, and
    /// `effective_lag_count.len() == hydro_count` (each `<= max_par_order`).
    /// `n_buckets` sizes `transit_bucket_column_order` (`StateSpace::new`
    /// derives its own bucket count from that vector's length). `k_max` is
    /// derived — `AnticipatedResolution::resolve`'s own
    /// `ring_size(&anticipated_lead_stages)` — never ranged independently of
    /// the leads: a ring deeper than its leads is covered wherever a
    /// resolution produces one.
    fn state_layout_strategy() -> impl Strategy<Value = StateSpace> {
        (0..=4usize, 0..=3usize, 0..=3usize, 0..=3usize)
            .prop_flat_map(|(hydro_count, max_par_order, n_buckets, n_anticipated)| {
                (
                    Just(hydro_count),
                    Just(max_par_order),
                    Just(n_buckets),
                    Just(n_anticipated),
                    prop::collection::vec(0..=3usize, n_anticipated),
                    0..=5usize,
                    0..=2usize,
                    prop::collection::vec((0..=4usize, 0..=3usize), n_buckets),
                    prop::collection::vec(0..=max_par_order, hydro_count),
                )
            })
            .prop_map(
                |(
                    hydro_count,
                    max_par_order,
                    _n_buckets,
                    _n_anticipated,
                    anticipated_lead_stages,
                    n_decision,
                    extra_delivery,
                    transit_bucket_column_order,
                    effective_lag_count,
                )| {
                    let leads: Vec<LeadTime> = anticipated_lead_stages
                        .iter()
                        .map(|&l| LeadTime::Stages(u32::try_from(l).unwrap_or(u32::MAX)))
                        .collect();
                    let study_stage_hours = vec![720.0; n_decision];
                    let post_study_stage_hours = vec![720.0; extra_delivery];
                    let resolution = AnticipatedResolution::resolve(
                        &leads,
                        DeliveryAxis {
                            study_stage_hours: &study_stage_hours,
                            post_study_stage_hours: &post_study_stage_hours,
                        },
                    );
                    let transit_bucket_column_order: Vec<(HydroSys, usize)> =
                        transit_bucket_column_order
                            .into_iter()
                            .map(|(p, lag)| (HydroSys::new(p), lag))
                            .collect();
                    StateSpace::new(
                        hydro_count,
                        max_par_order,
                        transit_bucket_column_order,
                        anticipated_lead_stages,
                        resolution,
                        &effective_lag_count,
                    )
                },
            )
    }

    /// A [`StateSpace`] paired with an arbitrary [`StageStateConfig`] gate
    /// combination, for the gated-projection agreement property.
    fn state_layout_and_config_strategy() -> impl Strategy<Value = (StateSpace, StageStateConfig)> {
        (state_layout_strategy(), any::<bool>(), any::<bool>()).prop_map(
            |(global, storage, inflow_lags)| {
                (
                    global,
                    StageStateConfig {
                        storage,
                        inflow_lags,
                    },
                )
            },
        )
    }

    proptest! {
        #![proptest_config(fixed_config())]

        /// Generalizes `state_dim_ranges_partition_n_state_contiguously`: the
        /// four `state_dim_*_range` regions partition `[0, n_state)` contiguously
        /// (no gap, no overlap) for every generated shape.
        #[test]
        fn partition(global in state_layout_strategy()) {
            let storage = global.state_dim_storage_range();
            let lag = global.state_dim_lag_range();
            let bucket = global.state_dim_bucket_range();
            let commitment_hold = global.state_dim_commitment_hold_range();

            prop_assert_eq!(storage.start, 0);
            prop_assert_eq!(lag.start, storage.end);
            prop_assert_eq!(bucket.start, lag.end);
            prop_assert_eq!(commitment_hold.start, bucket.end);
            prop_assert_eq!(commitment_hold.end, global.n_state);
        }

        /// Generalizes `default_projection_is_identity` and
        /// `default_render_matches_global_nonzero_mask`: an all-enabled
        /// projection reproduces the global resolvers index-for-index, and
        /// `render_pairs` reproduces the global `nonzero_state_indices` render
        /// exactly, for every generated shape.
        #[test]
        fn default_identity(global in state_layout_strategy()) {
            let cut = CutStateProjection::new(&global, ALL_ENABLED);

            prop_assert_eq!(cut.n_slots(), global.n_state);
            for j in 0..global.n_state {
                prop_assert_eq!(
                    cut.incoming_column(CutSlot::new(j)),
                    global.state_to_lp_incoming_column(StateDim::new(j))
                );
                prop_assert_eq!(
                    cut.outgoing_column(CutSlot::new(j)),
                    global.lp_column_for_state(StateDim::new(j))
                );
            }

            let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
            let global_render: Vec<(CutSlot, OutCol)> = global
                .nonzero_state_indices
                .iter()
                .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
                .collect();
            prop_assert_eq!(rendered, global_render);
        }

        /// Generalizes `bucket_render_pairs_sit_between_lag_and_anticipated` over
        /// every gate combination. The expected mapping is built from the
        /// concrete per-region block formulas (the raw range fields, with the
        /// storage → lag → buckets → anticipated order written out literally) —
        /// never from `REGION_ORDER` or the resolvers the constructor itself
        /// walks, which would shift both sides of the assertion in lockstep
        /// and turn a walk-order or resolver bug green.
        #[test]
        fn gated_projection_agreement(
            (global, state_config) in state_layout_and_config_strategy()
        ) {
            let cut = CutStateProjection::new(&global, state_config);

            let n = global.hydro_count;
            let mut expected: Vec<(usize, usize)> = Vec::new();
            if state_config.storage {
                for h in 0..n {
                    expected.push((global.storage_in.start + h, global.storage.start + h));
                }
            }
            if state_config.inflow_lags {
                for lag in 0..global.max_par_order {
                    for h in 0..n {
                        let outgoing = if lag == 0 {
                            global.z_inflow.start + h
                        } else {
                            global.inflow_lags.start + (lag - 1) * n + h
                        };
                        expected.push((global.inflow_lags.start + lag * n + h, outgoing));
                    }
                }
            }
            for b in 0..global.n_buckets {
                expected.push((
                    global.transit_buckets_in.start + b,
                    global.transit_buckets_out.start + b,
                ));
            }
            for o in 0..global.n_anticipated * global.k_max {
                expected.push((global.commit_in.start + o, global.commit_out.start + o));
            }

            prop_assert_eq!(expected.len(), cut.n_slots());
            for (slot, &(incoming, outgoing)) in expected.iter().enumerate() {
                prop_assert_eq!(cut.incoming_column(CutSlot::new(slot)).get(), incoming);
                prop_assert_eq!(cut.outgoing_column(CutSlot::new(slot)).get(), outgoing);
            }
        }
    }
}
