use super::state_box::StateBox;
use crate::lp::indexer::{BlockGrid, BlockIdx, StateDim, StateSpace};

/// Pre-allocated row-bound and column-bound patch arrays for one SDDP stage LP solve.
///
/// Reused across all iterations. The row-bound region (`indices`/`lower`/`upper`,
/// length `M*B + N`) holds only the load and z-inflow patches; state
/// fixing (storage, AR lags, travel-time buckets, anticipated state) is applied
/// exclusively via column bounds and lives in the column-bound region
/// (`col_indices`/`col_lower`/`col_upper`, length `N*(1+L) + n_buckets + A*K`).
/// `N` is the hydro count, `M` the stochastic-load-bus count, `B` the max block
/// count, `L` the max PAR order, `n_buckets` the travel-time bucket count, and
/// `A*K` the anticipated-thermal state count — this must stay equal to what
/// `StateSpace::n_state` counts, since `fill_col_state_patches` writes one
/// patch per `state_layout.n_state` index. For equality constraints each
/// entry's lower == upper.
///
/// [`fill_col_state_patches`](Self::fill_col_state_patches) is the single owner
/// of the column-bound region: it iterates
/// [`StateSpace::state_to_lp_incoming_column`] over every state-vector index,
/// so storage, AR lags, buckets, and anticipated state all resolve through one
/// call site rather than parallel hardcoded offsets.
#[derive(Debug, Clone)]
pub struct PatchBuffer {
    /// Row indices to patch.
    pub indices: Vec<usize>,

    /// New lower bounds for each patched row.
    pub lower: Vec<f64>,

    /// New upper bounds for each patched row.
    pub upper: Vec<f64>,

    /// Column indices to patch in the column-bound region.
    pub col_indices: Vec<usize>,

    /// New lower bounds for each patched column in the column-bound region.
    pub col_lower: Vec<f64>,

    /// New upper bounds for each patched column in the column-bound region.
    pub col_upper: Vec<f64>,

    /// Number of operating hydro plants (N).
    hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    max_par_order: usize,

    /// Number of buses with stochastic load noise (M).
    load_bus_count: usize,

    /// Maximum block count across all stages (B).
    max_blocks: usize,

    /// Global travel-time bucket count; `0` when no arc is declared.
    n_buckets: usize,

    /// Number of anticipated thermals (A).
    n_anticipated: usize,

    /// Maximum lead-time horizon across anticipated thermals (K).
    k_max: usize,

    /// Number of load patches from the most recent [`fill_load_patches`] call
    /// (`load_bus_count * n_blocks`); zero before any call or when
    /// `load_bus_count == 0`.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    active_load_patches: usize,

    /// Number of z-inflow patches from the most recent [`fill_z_inflow_patches`]
    /// call (`hydro_count` when active, zero otherwise).
    ///
    /// [`fill_z_inflow_patches`]: PatchBuffer::fill_z_inflow_patches
    active_z_inflow_patches: usize,
}

impl PatchBuffer {
    /// Construct a [`PatchBuffer`] sized to `M*B + N` row patches and
    /// `N*(1+L) + n_buckets + A*K` column patches, zero-initialised.
    ///
    /// The row region holds load patches at `[0, M*B)` followed by `N` z-inflow
    /// patches, populated before each LP solve via [`fill_load_patches`] /
    /// [`fill_z_inflow_patches`]; the column region is populated by
    /// `fill_col_state_patches`. Pass `0` for `n_load_buses`/`max_blocks` when
    /// there is no stochastic load, for `n_buckets` when there are no
    /// travel-time buckets, and for `n_anticipated`/`k_max` when there are no
    /// anticipated thermals.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::lp::builder::PatchBuffer;
    ///
    /// // 3-hydro AR(2) system, no stochastic load, no buckets, no anticipated thermals
    /// // Row capacity = M*B + N = 0 + 3 = 3
    /// // Col capacity = N*(1+L) + n_buckets + A*K = 3*(1+2) + 0 + 0 = 9
    /// let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
    /// assert_eq!(buf.indices.len(), 3);
    /// assert_eq!(buf.col_indices.len(), 9);
    ///
    /// // 3-hydro AR(2) system with 2 stochastic load buses, up to 3 blocks
    /// // Row capacity = M*B + N = 6 + 3 = 9
    /// let buf_load = PatchBuffer::new(3, 2, 2, 3, 0, 0, 0);
    /// assert_eq!(buf_load.indices.len(), 9);
    ///
    /// // Production scale: N = 160, L = 12, no stochastic load
    /// // Row capacity = M*B + N = 0 + 160 = 160
    /// let big = PatchBuffer::new(160, 12, 0, 0, 0, 0, 0);
    /// assert_eq!(big.indices.len(), 160);
    ///
    /// // Edge case: no lags (L = 0)
    /// // Row capacity = M*B + N = 0 + 5 = 5
    /// let no_lag = PatchBuffer::new(5, 0, 0, 0, 0, 0, 0);
    /// assert_eq!(no_lag.indices.len(), 5);
    ///
    /// // Anticipated thermals: 1 plant, K=2 — row capacity unchanged (A*K is col-only)
    /// // Row capacity = M*B + N = 0 + 3 = 3
    /// let ant = PatchBuffer::new(3, 2, 0, 0, 0, 1, 2);
    /// assert_eq!(ant.indices.len(), 3);
    ///
    /// // Travel-time buckets: n_buckets=4 — row capacity unchanged (bucket state is col-only)
    /// // Col capacity = N*(1+L) + n_buckets + A*K = 3*3 + 4 + 0 = 13
    /// let transit_buckets = PatchBuffer::new(3, 2, 0, 0, 4, 0, 0);
    /// assert_eq!(transit_buckets.col_indices.len(), 13);
    /// assert_eq!(transit_buckets.indices.len(), 3);
    /// ```
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    /// [`fill_z_inflow_patches`]: PatchBuffer::fill_z_inflow_patches
    #[must_use]
    pub fn new(
        hydro_count: usize,
        max_par_order: usize,
        n_load_buses: usize,
        max_blocks: usize,
        n_buckets: usize,
        n_anticipated: usize,
        k_max: usize,
    ) -> Self {
        let capacity = n_load_buses * max_blocks + hydro_count;
        let col_capacity = hydro_count * (1 + max_par_order) + n_buckets + n_anticipated * k_max;
        Self {
            indices: vec![0; capacity],
            lower: vec![0.0; capacity],
            upper: vec![0.0; capacity],
            col_indices: vec![0; col_capacity],
            col_lower: vec![0.0; col_capacity],
            col_upper: vec![0.0; col_capacity],
            hydro_count,
            max_par_order,
            load_bus_count: n_load_buses,
            max_blocks,
            n_buckets,
            n_anticipated,
            k_max,
            active_load_patches: 0,
            active_z_inflow_patches: 0,
        }
    }

    /// Fill `N*(1+L) + n_buckets + A*K` equality column-bound patches
    /// pinning incoming state: storage, AR lags, travel-time buckets, and
    /// anticipated state.
    ///
    /// The single owner of the column-bound region: iterates
    /// [`StateSpace::state_to_lp_incoming_column`] over every state-vector index
    /// `j ∈ [0, n_state)`, writing one patch per `j` at buffer slot `j` — storage,
    /// AR lags, buckets, and anticipated state all resolve through this one
    /// resolver call rather than parallel hardcoded offsets. A hardcoded
    /// `anticipated_start = N*(1+L)` for the anticipated block silently drops the
    /// `+ n_buckets` shift whenever a bucket block precedes it, misaligning the
    /// anticipated patches and leaving every bucket incoming column unpinned.
    ///
    /// Enforces `x == v` by setting `lb = ub = v / col_scale[col]` in the scaled LP
    /// — **divided**, contrast with the row-equality path that **multiplies** by
    /// `row_scale[row]`. `col_scale` may be `&[]` (no scaling); when non-empty it
    /// must be at least `state_layout.commit_in.end` long.
    ///
    /// Bucket incoming columns are pinned to whatever the caller supplies in
    /// `state` at the bucket state-vector indices — pin this **once per
    /// stage-visit** (the trial point / decision-driven state), never per
    /// opening, the same contract [`state_to_lp_incoming_column`] documents for
    /// storage and AR lags (contrast with NCS availability, which patches per
    /// opening).
    ///
    /// [`state_to_lp_incoming_column`]: StateSpace::state_to_lp_incoming_column
    ///
    /// The pinned incoming state is whatever the caller supplies in `state`: at
    /// every stage but the first, the previous stage's own seam-canonicalized
    /// outgoing state, so it lies inside that PRODUCING stage's own admissible
    /// box; `producer_box` is that box (`None` at the first stage, whose
    /// incoming state is the study's initial condition, validated at its own
    /// source with no producing seam). `Some` asserts every pinned dimension
    /// lies inside it, catching an entry point that bypassed the seam.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != state_layout.n_state`, if the
    /// emitted patch count does not equal [`Self::state_col_patch_count`], if
    /// any travel-time bucket state index fails to resolve into
    /// `state_layout.transit_buckets_in`, or if `producer_box` is `Some` and
    /// any pinned dimension lies outside it.
    pub fn fill_col_state_patches(
        &mut self,
        state_layout: &StateSpace,
        state: &[f64],
        col_scale: &[f64],
        producer_box: Option<&StateBox>,
    ) {
        debug_assert_eq!(
            state.len(),
            state_layout.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = state_layout.n_state,
        );

        for (j, &sv) in state.iter().enumerate() {
            if let Some(state_box) = producer_box {
                debug_assert!(
                    sv >= state_box.lower[j] && sv <= state_box.upper[j],
                    "pinned incoming state[{j}] = {sv} outside admissible box [{lower}, {upper}]",
                    lower = state_box.lower[j],
                    upper = state_box.upper[j],
                );
            }
            let col = state_layout
                .state_to_lp_incoming_column(StateDim::new(j))
                .get();
            let scaled = if col_scale.is_empty() {
                sv
            } else {
                sv / col_scale[col]
            };
            self.col_indices[j] = col;
            self.col_lower[j] = scaled;
            self.col_upper[j] = scaled;
        }

        debug_assert_eq!(
            state.len(),
            self.state_col_patch_count(),
            "emitted patch count {emitted} must equal state_col_patch_count() {expected}",
            emitted = state.len(),
            expected = self.state_col_patch_count(),
        );
        debug_assert!(
            state_layout
                .transit_buckets_out
                .clone()
                .all(|j| state_layout
                    .transit_buckets_in
                    .contains(&self.col_indices[j])),
            "every travel-time bucket state index must resolve to a transit_buckets_in column"
        );
    }

    /// Fill `n_load_buses * n_blocks` load-balance equality patches into
    /// the row buffer at offset `0`, addressed via [`BlockGrid::flat`] (the single
    /// owner of block-major strides). `load_rhs` is bus-major, block-minor matching
    /// `bus_positions` order; values are prescaled by `row_scale[row]` when
    /// `row_scale` is non-empty (pass `&[]` for no scaling).
    ///
    /// `grid` must carry this stage's block count (the value the LP template was
    /// built with), NOT a global grid: a global grid would stride by the wrong block
    /// count at any stage whose count differs.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if:
    /// - `load_rhs.len() != self.load_bus_count * grid.n_blks()`
    /// - `bus_positions.len() != self.load_bus_count`
    /// - `grid.n_blks() > self.max_blocks`
    pub fn fill_load_patches(
        &mut self,
        load_row_start: usize,
        grid: BlockGrid,
        load_rhs: &[f64],
        bus_positions: &[usize],
        row_scale: &[f64],
    ) {
        let n_blocks = grid.n_blks();
        debug_assert_eq!(
            load_rhs.len(),
            self.load_bus_count * n_blocks,
            "load_rhs length {got} != load_bus_count*n_blocks {expected}",
            got = load_rhs.len(),
            expected = self.load_bus_count * n_blocks,
        );
        debug_assert_eq!(
            bus_positions.len(),
            self.load_bus_count,
            "bus_positions length {got} != load_bus_count {expected}",
            got = bus_positions.len(),
            expected = self.load_bus_count,
        );
        debug_assert!(
            n_blocks <= self.max_blocks,
            "n_blocks {n_blocks} exceeds max_blocks {mb}",
            mb = self.max_blocks,
        );

        let mut slot = 0;

        for (i, &bus_pos) in bus_positions.iter().enumerate() {
            for blk in 0..n_blocks {
                let row = grid.flat(load_row_start, bus_pos, BlockIdx::new(blk));
                // Host-array index `i * n_blks + blk` routed through the same
                // `grid.flat` (start = 0) to keep one owner of the stride.
                let rhs = load_rhs[grid.flat(0, i, BlockIdx::new(blk))];
                let scaled = if row_scale.is_empty() {
                    rhs
                } else {
                    rhs * row_scale[row]
                };
                self.indices[slot] = row;
                self.lower[slot] = scaled;
                self.upper[slot] = scaled;
                slot += 1;
            }
        }

        self.active_load_patches = self.load_bus_count * n_blocks;
    }

    /// Fill the `N` z-inflow-definition equality patches at
    /// `z_inflow_row_start` from `z_inflow_rhs`, prescaled by `row_scale[row]` when
    /// `row_scale` is non-empty (pass `&[]` for no scaling).
    ///
    /// Must be called after [`fill_load_patches`] (whose `active_load_patches` sets
    /// this region's offset) and before `set_row_bounds`.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    pub fn fill_z_inflow_patches(
        &mut self,
        z_inflow_row_start: usize,
        z_inflow_rhs: &[f64],
        row_scale: &[f64],
    ) {
        let n = self.hydro_count;
        if n == 0 || z_inflow_rhs.is_empty() {
            self.active_z_inflow_patches = 0;
            return;
        }

        let z_inflow_start = self.active_load_patches;

        for (h, &rhs) in z_inflow_rhs.iter().enumerate().take(n) {
            let slot = z_inflow_start + h;
            let row = z_inflow_row_start + h;
            let scaled = if row_scale.is_empty() {
                rhs
            } else {
                rhs * row_scale[row]
            };
            self.indices[slot] = row;
            self.lower[slot] = scaled;
            self.upper[slot] = scaled;
        }

        self.active_z_inflow_patches = n;
    }

    /// Active row-patch count (load + z-inflow) for the full forward-pass
    /// slice passed to `set_row_bounds`.
    #[must_use]
    #[inline]
    pub fn forward_patch_count(&self) -> usize {
        self.active_load_patches + self.active_z_inflow_patches
    }

    /// Column-bound region capacity (`N*(1+L) + n_buckets + A*K` state-fixing slots).
    #[must_use]
    #[inline]
    pub fn state_col_patch_count(&self) -> usize {
        self.hydro_count * (1 + self.max_par_order)
            + self.n_buckets
            + self.n_anticipated * self.k_max
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::{PatchBuffer, StateBox};
    use crate::lp::indexer::BlockGrid;
    use crate::test_support::{state_layout, state_layout_full, state_layout_with_transit_buckets};

    /// Every dimension unbounded — the pin-time box-membership assert is vacuous.
    fn unbounded_state_box(n_state: usize) -> StateBox {
        StateBox {
            lower: vec![f64::NEG_INFINITY; n_state],
            upper: vec![f64::INFINITY; n_state],
        }
    }

    // -------------------------------------------------------------------------
    // Capacity formulas (row + column buffers) across scales
    // -------------------------------------------------------------------------

    /// Row capacity is `n_load_buses*max_blocks + N` and column capacity is
    /// `N*(1+L) + n_buckets + A*K`. All formulas are exercised at zero /
    /// unit-anticipated / bucket / combined / production scales in one table
    /// so each scale stays legible via the tuple-naming failure message.
    #[test]
    fn patch_buffer_capacity_formulas() {
        // (n, l, n_load_buses, max_blocks, n_buckets, a, k, expected_row_cap, expected_col_cap)
        let cases = [
            (
                0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
            ),
            (3, 2, 0, 0, 0, 0, 0, 3, 9),
            (0, 0, 0, 0, 0, 1, 2, 0, 2),
            (0, 0, 0, 0, 3, 0, 0, 0, 3),
            (3, 2, 0, 0, 4, 2, 3, 3, 19),
            (160, 12, 0, 0, 0, 0, 0, 160, 2080),
        ];

        for (n, l, n_load_buses, max_blocks, n_buckets, a, k, expected_row_cap, expected_col_cap) in
            cases
        {
            let buf = PatchBuffer::new(n, l, n_load_buses, max_blocks, n_buckets, a, k);

            for (label, len) in [
                ("col_indices", buf.col_indices.len()),
                ("col_lower", buf.col_lower.len()),
                ("col_upper", buf.col_upper.len()),
            ] {
                assert_eq!(
                    len, expected_col_cap,
                    "{label} col cap mismatch for (n={n}, l={l}, n_buckets={n_buckets}, a={a}, k={k})"
                );
            }

            for (label, len) in [
                ("indices", buf.indices.len()),
                ("lower", buf.lower.len()),
                ("upper", buf.upper.len()),
            ] {
                assert_eq!(
                    len, expected_row_cap,
                    "{label} row cap mismatch for (n={n}, l={l}, n_load_buses={n_load_buses}, max_blocks={max_blocks}, n_buckets={n_buckets}, a={a}, k={k})"
                );
            }
        }
    }

    /// `state_col_patch_count` returns N*(1+L) + n_buckets + A*K.
    #[test]
    fn state_col_patch_count_returns_n_times_one_plus_l() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 0, 1, 2);
        // N*(1+L) + n_buckets + A*K = 3*3 + 0 + 1*2 = 11
        assert_eq!(buf.state_col_patch_count(), 11);
    }

    /// `state_col_patch_count` includes `n_buckets` alongside storage/lag/anticipated.
    #[test]
    fn state_col_patch_count_includes_transit_bucket_count() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 4, 1, 2);
        // N*(1+L) + n_buckets + A*K = 3*3 + 4 + 1*2 = 15
        assert_eq!(buf.state_col_patch_count(), 15);
    }

    /// `state_col_patch_count` equals `N*(1+L) + n_buckets + A*K`, pinned on a
    /// fixture whose anticipated ring is nonzero.
    #[test]
    fn state_col_patch_count_matches_hydro_lag_bucket_anticipated_formula() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 4, 2, 3);
        // N*(1+L) + n_buckets + A*K = 3*3 + 4 + 2*3 = 19
        assert_eq!(buf.state_col_patch_count(), 19);
    }

    /// Column buffer is zero-initialised at construction.
    #[test]
    fn col_buffer_zero_initialised() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        assert_eq!(buf.col_indices.len(), 9);
        assert!(
            buf.col_indices.iter().all(|&v| v == 0),
            "col_indices not zero-initialised"
        );
        assert!(
            buf.col_lower.iter().all(|&v| v == 0.0),
            "col_lower not zero-initialised"
        );
        assert!(
            buf.col_upper.iter().all(|&v| v == 0.0),
            "col_upper not zero-initialised"
        );
    }

    /// `forward_patch_count` is zero before any row-patch fill call.
    #[test]
    fn forward_patch_count_zero_before_any_fill() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        assert_eq!(buf.forward_patch_count(), 0);
    }

    /// Production scale: row capacity is `M*B + N = 0 + 160 = 160`, and
    /// `forward_patch_count` counts the `N` z-inflow patches once filled.
    #[test]
    fn production_scale_forward_patch_count() {
        let n = 160;
        let mut buf = PatchBuffer::new(n, 12, 0, 0, 0, 0, 0);
        assert_eq!(buf.indices.len(), 160);

        let z_rhs = vec![0.0_f64; n];
        buf.fill_z_inflow_patches(500, &z_rhs, &[]);
        assert_eq!(buf.forward_patch_count(), 160);
    }

    #[test]
    fn clone_and_debug() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let cloned = buf.clone();
        assert_eq!(cloned.indices.len(), buf.indices.len());

        let s = format!("{buf:?}");
        assert!(s.contains("PatchBuffer"));
    }

    // -------------------------------------------------------------------------
    // Load-balance unit tests
    // -------------------------------------------------------------------------

    /// AC (capacity): `PatchBuffer::new(2, 1, 1, 3, 0, 0, 0)` → row capacity = M*B + N = 3 + 2 = 5.
    #[test]
    fn new_with_load_allocates_correct_capacity() {
        let buf = PatchBuffer::new(2, 1, 1, 3, 0, 0, 0);
        // M*B + N = 1*3 + 2 = 5
        assert_eq!(buf.indices.len(), 5);
        assert_eq!(buf.lower.len(), 5);
        assert_eq!(buf.upper.len(), 5);
    }

    /// Load-balance row indices follow `row = load_row_start + bus_positions[i] * n_blocks + blk`.
    ///
    /// With `n_load_buses=2, n_blocks=2, bus_positions=[0,1], load_row_start=100`:
    /// load patches start at slot 0 so indices[0..4] = [100, 101, 102, 103].
    #[test]
    fn fill_load_patches_correct_indices() {
        // N=0, L=0, M=2, B=2, A=0, K=0 → row capacity = M*B + N = 2*2 + 0 = 4
        let mut buf = PatchBuffer::new(0, 0, 2, 2, 0, 0, 0);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(100, BlockGrid::new(2, 1), &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.indices[0], 100); // bus 0, blk 0
        assert_eq!(buf.indices[1], 101); // bus 0, blk 1
        assert_eq!(buf.indices[2], 102); // bus 1, blk 0
        assert_eq!(buf.indices[3], 103); // bus 1, blk 1
    }

    /// Load-balance lower and upper bounds equal the corresponding `load_rhs` value.
    #[test]
    fn fill_load_patches_correct_values() {
        let mut buf = PatchBuffer::new(0, 0, 2, 2, 0, 0, 0);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(100, BlockGrid::new(2, 1), &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.lower[0], 300.0);
        assert_eq!(buf.upper[0], 300.0);
        assert_eq!(buf.lower[1], 280.0);
        assert_eq!(buf.upper[1], 280.0);
        assert_eq!(buf.lower[2], 500.0);
        assert_eq!(buf.upper[2], 500.0);
        assert_eq!(buf.lower[3], 450.0);
        assert_eq!(buf.upper[3], 450.0);
    }

    /// Every load patch must be an equality constraint: `lower[i] == upper[i]`.
    #[test]
    fn fill_load_patches_equality_constraints() {
        let mut buf = PatchBuffer::new(3, 2, 2, 3, 0, 0, 0);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, BlockGrid::new(3, 1), &load_rhs, &bus_positions, &[]);

        let count = buf.forward_patch_count();
        for i in 0..count {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    /// `forward_patch_count` includes load-balance patches after `fill_load_patches`.
    ///
    /// M=2, n_blocks=3 → forward_patch_count = M*n_blocks = 6 (no z-inflow fill).
    #[test]
    fn forward_patch_count_includes_load() {
        let mut buf = PatchBuffer::new(3, 2, 2, 3, 0, 0, 0);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, BlockGrid::new(3, 1), &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.forward_patch_count(), 6); // M*n_blocks=6
    }

    /// When `n_load_buses == 0`, `forward_patch_count` equals `N` once
    /// `fill_z_inflow_patches` has run (the sole inflow-noise site).
    #[test]
    fn zero_load_buses_forward_patch_count_is_z_inflow_only() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let z_rhs = [0.1_f64, 0.2, 0.3];
        buf.fill_z_inflow_patches(50, &z_rhs, &[]);

        assert_eq!(buf.forward_patch_count(), 3);
    }

    // -------------------------------------------------------------------------
    // fill_col_state_patches unit tests
    // -------------------------------------------------------------------------

    /// Storage col_indices[0..3] = [storage_in.start, +1, +2].
    #[test]
    fn fill_col_state_patches_storage_indices() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let state_layout = state_layout(3, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        let s = state_layout.storage_in.start;
        assert_eq!(buf.col_indices[0], s);
        assert_eq!(buf.col_indices[1], s + 1);
        assert_eq!(buf.col_indices[2], s + 2);
    }

    /// Storage col_lower[0..3] == col_upper[0..3] == [10.0, 20.0, 30.0].
    #[test]
    fn fill_col_state_patches_storage_values() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let state_layout = state_layout(3, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_lower[0], 10.0);
        assert_eq!(buf.col_upper[0], 10.0);
        assert_eq!(buf.col_lower[1], 20.0);
        assert_eq!(buf.col_upper[1], 20.0);
        assert_eq!(buf.col_lower[2], 30.0);
        assert_eq!(buf.col_upper[2], 30.0);
    }

    /// inflow_lags col_indices[3..9] matches lag column targets; col_lower matches lag values.
    ///
    /// Lag-column formula: `inflow_lags.start + lag*N + h`.
    /// - lag=0: cols `[il, il+1, il+2]`
    /// - lag=1: cols `[il+3, il+4, il+5]`
    ///
    /// State layout: lags at `state[3..9]` = [1,2,3,4,5,6].
    #[test]
    fn fill_col_state_patches_inflow_lags_indices_and_values() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let state_layout = state_layout(3, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        let il = state_layout.inflow_lags.start;
        // lag=0
        assert_eq!(buf.col_indices[3], il);
        assert_eq!(buf.col_indices[4], il + 1);
        assert_eq!(buf.col_indices[5], il + 2);
        // lag=1
        assert_eq!(buf.col_indices[6], il + 3);
        assert_eq!(buf.col_indices[7], il + 4);
        assert_eq!(buf.col_indices[8], il + 5);

        assert_eq!(buf.col_lower[3], 1.0);
        assert_eq!(buf.col_upper[3], 1.0);
        assert_eq!(buf.col_lower[6], 4.0);
        assert_eq!(buf.col_upper[6], 4.0);
        assert_eq!(buf.col_lower[8], 6.0);
        assert_eq!(buf.col_upper[8], 6.0);
    }

    /// Anticipated column-bound patches for N=0, L=0, A=1, K=2.
    ///
    /// col_indices[0..2] = [commit_in.start, +1],
    /// col_lower[0..2] == col_upper[0..2] == [7.0, 11.0] (slot-major / plant-minor).
    #[test]
    fn fill_col_state_patches_anticipated_state() {
        // N=0, A=1, K=2 anticipated-only state layout.
        let state_layout = state_layout_full(0, 0, 1, vec![2]);

        // n_state = 0 + 1*2 = 2; the state-VECTOR anticipated block is
        // `commit_out` (== 0 here), NOT the relocated incoming
        // `commit_in` (== 2 here).
        let ant_state_vec_start = state_layout.commit_out.start;
        let ant_incoming_col_start = state_layout.commit_in.start;
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[ant_state_vec_start] = 7.0;
        state[ant_state_vec_start + 1] = 11.0;

        let mut buf = PatchBuffer::new(0, 0, 0, 0, 0, 1, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_indices[0], ant_incoming_col_start);
        assert_eq!(buf.col_indices[1], ant_incoming_col_start + 1);
        assert_eq!(buf.col_lower[0], 7.0);
        assert_eq!(buf.col_upper[0], 7.0);
        assert_eq!(buf.col_lower[1], 11.0);
        assert_eq!(buf.col_upper[1], 11.0);
    }

    /// Pin-exactness: with `col_scale = 1.0` at the anticipated ring's incoming
    /// columns (D1's unscale override), `fill_col_state_patches` writes
    /// `col_lower == col_upper == v` — the raw commitment, no division — unlike a
    /// non-unit scale elsewhere in the same array, whose `v / d` round-trip is not
    /// guaranteed bit-exact.
    #[test]
    fn fill_col_state_patches_anticipated_state_unscaled_is_exact() {
        let state_layout = state_layout_full(0, 0, 1, vec![2]);
        let ant_state_vec_start = state_layout.commit_out.start;
        let ant_incoming_col_start = state_layout.commit_in.start;
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[ant_state_vec_start] = 7.0;
        state[ant_state_vec_start + 1] = 11.0;

        let ncols = state_layout.commit_in.end;
        let mut col_scale = vec![3.0_f64; ncols];
        col_scale[ant_incoming_col_start] = 1.0;
        col_scale[ant_incoming_col_start + 1] = 1.0;

        let mut buf = PatchBuffer::new(0, 0, 0, 0, 0, 1, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &col_scale,
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_lower[0], 7.0);
        assert_eq!(buf.col_upper[0], 7.0);
        assert_eq!(buf.col_lower[1], 11.0);
        assert_eq!(buf.col_upper[1], 11.0);
    }

    /// Every patch in the active col region has col_lower[i] == col_upper[i].
    #[test]
    fn fill_col_state_patches_equality_constraints() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let state_layout = state_layout(3, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        let count = buf.state_col_patch_count();
        for i in 0..count {
            assert_eq!(
                buf.col_lower[i],
                buf.col_upper[i],
                "col patch {i}: lower {lo} != upper {up}",
                lo = buf.col_lower[i],
                up = buf.col_upper[i],
            );
        }
    }

    /// col_scale divides: col_lower[h] == state[h] / col_scale[col] for storage.
    ///
    /// With col_scale[storage_in.start + h] = 2.0 for all h, and state = [10, 20, 30, ...],
    /// expected col_lower[0..3] = [5.0, 10.0, 15.0].
    #[test]
    fn fill_col_state_patches_unscaled_with_col_scale() {
        let state_layout = state_layout(3, 2);
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);

        // Build a col_scale long enough to cover commit_in.end.
        // Fill with 1.0 everywhere, then override the storage_in columns to 2.0.
        let ncols = state_layout.commit_in.end.max(state_layout.storage_in.end);
        let mut col_scale = vec![1.0_f64; ncols];
        let s = state_layout.storage_in.start;
        col_scale[s] = 2.0;
        col_scale[s + 1] = 2.0;
        col_scale[s + 2] = 2.0;

        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &col_scale,
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_lower[0], 5.0);
        assert_eq!(buf.col_upper[0], 5.0);
        assert_eq!(buf.col_lower[1], 10.0);
        assert_eq!(buf.col_upper[1], 10.0);
        assert_eq!(buf.col_lower[2], 15.0);
        assert_eq!(buf.col_upper[2], 15.0);
    }

    /// When n_anticipated == 0, the anticipated region is empty and state_col_patch_count() == N*(1+L).
    #[test]
    fn fill_col_state_patches_zero_anticipated_collapses_correctly() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let state_layout = state_layout(3, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        // N*(1+L) + A*K = 3*3 + 0 = 9
        assert_eq!(buf.state_col_patch_count(), 9);
        assert_eq!(buf.col_indices.len(), 9);
    }

    /// After fill_col_state_patches, the row buffer (indices/lower/upper) is untouched.
    ///
    /// Catches accidental cross-buffer writes; the row buffer must remain
    /// zero-initialised since no row-equality filler has been called.
    #[test]
    fn row_buffer_unchanged_after_fill_col_state_patches() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let state_layout = state_layout(3, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        assert!(
            buf.indices.iter().all(|&v| v == 0),
            "row indices modified by fill_col_state_patches"
        );
        assert!(
            buf.lower.iter().all(|&v| v == 0.0),
            "row lower modified by fill_col_state_patches"
        );
        assert!(
            buf.upper.iter().all(|&v| v == 0.0),
            "row upper modified by fill_col_state_patches"
        );
    }

    // -------------------------------------------------------------------------
    // Travel-time bucket column-bound patches (single-owner refactor)
    // -------------------------------------------------------------------------

    /// Every travel-time bucket incoming column receives a state-col patch,
    /// pinned to the value supplied in `state` — the trial point /
    /// decision-driven state, sourced once per stage-visit before any
    /// per-opening loop (contrast NCS availability, which patches per opening).
    #[test]
    fn fill_col_state_patches_every_transit_bucket_incoming_column_is_pinned() {
        let n_buckets = 2;
        let state_layout =
            state_layout_with_transit_buckets(3, 2, n_buckets, vec![(0, 0), (0, 1)], 0, vec![]);
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[state_layout.transit_buckets_out.start] = 100.0;
        state[state_layout.transit_buckets_out.start + 1] = 200.0;

        let mut buf = PatchBuffer::new(3, 2, 0, 0, n_buckets, 0, 0);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        let cp = buf.state_col_patch_count();
        assert_eq!(cp, state_layout.n_state);

        for (i, j) in state_layout.transit_buckets_out.clone().enumerate() {
            let col = state_layout.transit_buckets_in.start + i;
            assert_eq!(
                buf.col_indices[j], col,
                "bucket state index {j} must pin transit_buckets_in column {col}"
            );
            assert_eq!(buf.col_lower[j], state[j]);
            assert_eq!(buf.col_upper[j], state[j]);
        }
    }

    /// With both buckets (`B=2`) and anticipated thermals (`A=1, K=2`), the
    /// anticipated patches land on the columns shifted by `+B`: the anticipated
    /// slots occupy buffer positions `[N*(1+L)+B, N*(1+L)+B+A*K)`, not the
    /// unshifted `[N*(1+L), N*(1+L)+A*K)` a hardcoded `anticipated_start = N*(1+L)`
    /// would (incorrectly) target.
    #[test]
    fn fill_col_state_patches_anticipated_lands_at_shifted_offset() {
        let n = 3;
        let l = 2;
        let n_buckets = 2;
        let state_layout =
            state_layout_with_transit_buckets(n, l, n_buckets, vec![(0, 0), (0, 1)], 1, vec![2]);
        let unshifted_anticipated_start = n * (1 + l);
        // The state-VECTOR anticipated position is `commit_out`
        // (the `state_to_lp_column` identity domain), shifted by `n_buckets`
        // past the unshifted `N*(1+L)` a hardcoded `anticipated_start =
        // N*(1+L)` would (incorrectly) target.
        let shifted_anticipated_start = unshifted_anticipated_start + n_buckets;
        assert_eq!(state_layout.commit_out.start, shifted_anticipated_start);

        let mut state = vec![0.0_f64; state_layout.n_state];
        state[state_layout.transit_buckets_out.start] = 100.0;
        state[state_layout.transit_buckets_out.start + 1] = 200.0;
        state[state_layout.commit_out.start] = 7.0;
        state[state_layout.commit_out.start + 1] = 11.0;

        let mut buf = PatchBuffer::new(n, l, 0, 0, n_buckets, 1, 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        // Buckets occupy the unshifted slots; anticipated occupies the shifted ones.
        assert_eq!(buf.col_lower[unshifted_anticipated_start], 100.0);
        assert_eq!(buf.col_lower[unshifted_anticipated_start + 1], 200.0);
        assert_eq!(buf.col_lower[shifted_anticipated_start], 7.0);
        assert_eq!(buf.col_lower[shifted_anticipated_start + 1], 11.0);
        // The pinned LP column is the RELOCATED incoming `commit_in`
        // range, not the state-vector index used to populate `state` above.
        assert_eq!(
            buf.col_indices[shifted_anticipated_start],
            state_layout.commit_in.start
        );
        assert_eq!(
            buf.col_indices[shifted_anticipated_start + 1],
            state_layout.commit_in.start + 1
        );
        assert_eq!(buf.state_col_patch_count(), state_layout.n_state);
    }

    /// `B == 0`: `fill_col_state_patches` output is byte-identical to the
    /// pre-refactor per-family formula (storage → `storage_in`, AR
    /// lags → `inflow_lags.start + lag*N + h`, anticipated →
    /// `commit_in.start + slot*A + plant` at unshifted `anticipated_start =
    /// N*(1+L)`, which is only correct when `B == 0`).
    #[test]
    #[allow(clippy::cast_precision_loss)] // fixture: small integer indices, no precision lost
    fn fill_col_state_patches_b_zero_byte_identical_to_legacy_formula() {
        let n = 3;
        let l = 2;
        let a = 2;
        let k = 3;
        let state_layout = state_layout_full(n, l, a, vec![k; a]);
        assert_eq!(state_layout.n_buckets, 0);

        let state: Vec<f64> = (0..state_layout.n_state)
            .map(|i| (i as f64).mul_add(1.5, 1.0))
            .collect();
        let scale_len = state_layout.commit_in.end.max(state_layout.storage_in.end);
        let col_scale: Vec<f64> = (0..scale_len)
            .map(|i| (i as f64).mul_add(0.1, 1.0))
            .collect();

        let mut legacy_indices = vec![0usize; state_layout.n_state];
        let mut legacy_lower = vec![0.0_f64; state_layout.n_state];
        let mut legacy_upper = vec![0.0_f64; state_layout.n_state];

        let storage_in_start = state_layout.storage_in.start;
        for h in 0..n {
            let col = storage_in_start + h;
            let scaled = state[h] / col_scale[col];
            legacy_indices[h] = col;
            legacy_lower[h] = scaled;
            legacy_upper[h] = scaled;
        }
        let inflow_lags_start = state_layout.inflow_lags.start;
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                let col = inflow_lags_start + lag * n + h;
                let scaled = state[slot] / col_scale[col];
                legacy_indices[slot] = col;
                legacy_lower[slot] = scaled;
                legacy_upper[slot] = scaled;
            }
        }
        let anticipated_start = n * (1 + l);
        let ant_state_col_start = state_layout.commit_in.start;
        for slot in 0..k {
            for plant in 0..a {
                let off = slot * a + plant;
                let buf_slot = anticipated_start + off;
                let col = ant_state_col_start + off;
                let scaled = state[buf_slot] / col_scale[col];
                legacy_indices[buf_slot] = col;
                legacy_lower[buf_slot] = scaled;
                legacy_upper[buf_slot] = scaled;
            }
        }

        let mut buf = PatchBuffer::new(n, l, 0, 0, 0, a, k);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &col_scale,
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_indices, legacy_indices);
        assert_eq!(buf.col_lower, legacy_lower);
        assert_eq!(buf.col_upper, legacy_upper);
    }

    /// A `PatchBuffer` constructed with `n_buckets = 0` panics (index out of
    /// bounds) when filled against a bucket-aware `StateSpace` — the sizing
    /// contract between `PatchBuffer::new`'s `n_buckets` and the layout it
    /// patches must match, or the column-bound region has no room for the
    /// bucket slots.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn fill_col_state_patches_undersized_buffer_panics() {
        let state_layout =
            state_layout_with_transit_buckets(3, 2, 2, vec![(0, 0), (0, 1)], 0, vec![]);
        let state = vec![0.0_f64; state_layout.n_state];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );
    }

    /// A pinned incoming state dimension outside its admissible box panics via
    /// the pin-time debug_assert — read-back is the single canonicalization
    /// point, so an out-of-box pin here means an entry point bypassed it.
    #[test]
    #[should_panic(expected = "outside admissible box")]
    fn fill_col_state_patches_panics_on_out_of_box_pin() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0, 0);
        let state_layout = state_layout(3, 2);
        let mut state_box = unbounded_state_box(state.len());
        state_box.upper[0] = 5.0;

        buf.fill_col_state_patches(&state_layout, &state, &[], Some(&state_box));
    }
}
