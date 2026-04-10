use crate::indexer::StageIndexer;

/// Pre-allocated row-bound patch arrays for one SDDP stage LP solve.
///
/// Holds three parallel `Vec`s of equal length ready for a single
/// `SolverInterface::set_row_bounds` call.  The buffer is sized for
/// `N*(2+L) + N + M*B` patches at construction and reused across all
/// iterations, where `M` is the number of stochastic load buses and `B` is
/// the maximum block count across stages.
///
/// # Memory layout
///
/// Entries are written in category order:
///
/// | Entry range                         | Category                                | LP row indices              |
/// | ----------------------------------- | --------------------------------------- | --------------------------- |
/// | `[0, N)`                            | Storage-fixing (Category 1)             | `[0, N)`                    |
/// | `[N, N*(1+L))`                      | AR lag-fixing (Category 2)              | `[N, N*(1+L))`              |
/// | `[N*(1+L), N*(2+L))`               | AR dynamics / noise (Category 3)        | `base_rows[s]` = `N*(2+L)`  |
/// | `[N*(2+L), N*(2+L) + M*B_active)`  | Load balance row patches (Category 4)   | per-stage                   |
/// | `[N*(2+L)+M*B, N*(2+L)+M*B+N)`     | Z-inflow definition (Category 5)        | `N*(1+L)` (fixed)           |
///
/// Note: Category 3 patches LP rows at `base_rows[stage]` = `N*(2+L)` (water
/// balance rows, shifted by `+N` from the old layout where they started at
/// `N*(1+L)`). Category 5 patches LP rows at `N*(1+L)` (z-inflow definition
/// rows, now at a fixed offset between lag-fixing and water balance).
///
/// [`fill_state_patches`](PatchBuffer::fill_state_patches) writes only
/// `[0, N*(1+L))` (Categories 1 and 2).  Category 3 is left from the
/// previous iteration, which is safe because the caller passes only
/// `&self.indices[..active_len]` to `set_row_bounds`.
///
/// [`fill_load_patches`](PatchBuffer::fill_load_patches) writes Category 4
/// and records `active_load_patches` for the current stage's block count.
/// When `n_load_buses == 0`, Category 4 is empty and `forward_patch_count`
/// returns `N*(2+L)` unchanged.
#[derive(Debug, Clone)]
pub struct PatchBuffer {
    /// Row indices to patch.
    ///
    /// Length `N*(2+L) + M*max_blocks`.  Entries are `usize` to match the
    /// `set_row_bounds(&[usize], ...)` interface directly.
    pub indices: Vec<usize>,

    /// New lower bounds for each patched row.
    ///
    /// Length `N*(2+L) + M*max_blocks`.  For equality constraints, `lower[i] == upper[i]`.
    pub lower: Vec<f64>,

    /// New upper bounds for each patched row.
    ///
    /// Length `N*(2+L) + M*max_blocks`.  For equality constraints, `upper[i] == lower[i]`.
    pub upper: Vec<f64>,

    /// Number of operating hydro plants (N).
    hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    max_par_order: usize,

    /// Number of buses with stochastic load noise (M).
    load_bus_count: usize,

    /// Maximum block count across all stages.
    ///
    /// Determines the Category 4 capacity: `load_bus_count * max_blocks`.
    max_blocks: usize,

    /// Number of load patches written by the most recent [`fill_load_patches`] call.
    ///
    /// Equals `load_bus_count * n_blocks` for the stage solved most recently.
    /// Zero when `fill_load_patches` has not yet been called or when
    /// `load_bus_count == 0`.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    active_load_patches: usize,

    /// Number of z-inflow patches written by the most recent [`fill_z_inflow_patches`] call.
    ///
    /// Equals `hydro_count` when z-inflow patches are active, zero otherwise.
    ///
    /// [`fill_z_inflow_patches`]: PatchBuffer::fill_z_inflow_patches
    active_z_inflow_patches: usize,
}

impl PatchBuffer {
    /// Construct a [`PatchBuffer`] pre-allocated for `N*(2+L) + M*B + N` patches.
    ///
    /// - `hydro_count` — number of operating hydro plants (N).
    /// - `max_par_order` — maximum PAR order across all operating hydros (L).
    /// - `n_load_buses` — number of buses with stochastic load noise (M).
    ///   Pass `0` when there is no stochastic load.
    /// - `max_blocks` — maximum block count across all stages (B).
    ///   Pass `0` when there is no stochastic load.
    ///
    /// The buffer's `indices`, `lower`, and `upper` vectors are sized to
    /// `N*(2+L) + M*B` and zero-initialised.  Call [`fill_forward_patches`],
    /// [`fill_load_patches`], or [`fill_state_patches`] to populate them
    /// before each LP solve.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::PatchBuffer;
    ///
    /// // 3-hydro AR(2) system, no stochastic load
    /// // Capacity = N*(2+L) + M*B + N = 3*(2+2) + 0 + 3 = 15
    /// let buf = PatchBuffer::new(3, 2, 0, 0);
    /// assert_eq!(buf.indices.len(), 15);
    ///
    /// // 3-hydro AR(2) system with 2 stochastic load buses, up to 3 blocks
    /// // Capacity = 3*(2+2) + 2*3 + 3 = 12 + 6 + 3 = 21
    /// let buf_load = PatchBuffer::new(3, 2, 2, 3);
    /// assert_eq!(buf_load.indices.len(), 21);
    ///
    /// // Production scale: N = 160, L = 12, no stochastic load
    /// // Capacity = 160*(2+12) + 160 = 2240 + 160 = 2400
    /// let big = PatchBuffer::new(160, 12, 0, 0);
    /// assert_eq!(big.indices.len(), 2400);
    ///
    /// // Edge case: no lags (L = 0) — only storage + noise + z-inflow patches
    /// // Capacity = 5*(2+0) + 5 = 15
    /// let no_lag = PatchBuffer::new(5, 0, 0, 0);
    /// assert_eq!(no_lag.indices.len(), 15);
    /// ```
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    /// [`fill_state_patches`]: PatchBuffer::fill_state_patches
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    #[must_use]
    pub fn new(
        hydro_count: usize,
        max_par_order: usize,
        n_load_buses: usize,
        max_blocks: usize,
    ) -> Self {
        // Category 5 (z-inflow) adds N entries after Category 4 (load patches).
        let capacity = hydro_count * (2 + max_par_order) + n_load_buses * max_blocks + hydro_count;
        Self {
            indices: vec![0; capacity],
            lower: vec![0.0; capacity],
            upper: vec![0.0; capacity],
            hydro_count,
            max_par_order,
            load_bus_count: n_load_buses,
            max_blocks,
            active_load_patches: 0,
            active_z_inflow_patches: 0,
        }
    }

    /// Fill all `N*(2+L)` patches for a forward-pass solve.
    ///
    /// Populates Categories 1, 2, and 3 in sequence:
    ///
    /// - **Category 1** — `N` storage-fixing patches: row `h` ← `row_scale[h] * state[h]`
    ///   for `h ∈ [0, N)`.
    /// - **Category 2** — `N*L` AR lag-fixing patches: row `N + ℓ·N + h` ←
    ///   `row_scale[N+ℓN+h] * state[N + ℓ·N + h]` for `h ∈ [0, N)`, `ℓ ∈ [0, L)`.
    /// - **Category 3** — `N` noise-fixing patches: row
    ///   `ar_dynamics_row_offset(base_row, h)` ← `noise[h]` for `h ∈ [0, N)`.
    ///   Category 3 is NOT prescaled by `row_scale` because `noise[h]` is computed
    ///   from `template.row_lower` (already row-scaled) plus an unscaled noise term.
    ///   Prescaling would double-scale the base component.
    ///
    /// All patches are equality constraints: `lower[i] == upper[i] == value`.
    ///
    /// When `row_scale` is non-empty, Categories 1 and 2 values are multiplied by
    /// the corresponding `row_scale[row_index]` before being stored.  Pass an
    /// empty slice when no row scaling has been applied.  Category 3 is always
    /// written as-is regardless of `row_scale`.
    ///
    /// After this call, pass `&buf.indices`, `&buf.lower`, `&buf.upper` to
    /// `SolverInterface::set_row_bounds`.
    ///
    /// # Arguments
    ///
    /// - `indexer` — LP layout map for this stage (provides `hydro_count`,
    ///   `max_par_order`, and fixing-constraint row ranges).
    /// - `state` — incoming state vector of length `n_state = N*(1+L)`.
    ///   Prefix `[0, N)` is storage; `[N, N*(1+L))` is AR lags.
    /// - `noise` — stochastic noise innovations of length `N`, one per hydro.
    /// - `base_row` — first row index of the AR dynamics constraints in the
    ///   static non-dual region of the LP ([Solver Abstraction SS2.2]).
    ///   Computed during stage template construction and stored alongside
    ///   `indexer`.
    /// - `row_scale` — per-row scaling factors from the stage template.
    ///   Pass `&[]` when no scaling is active.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != indexer.n_state` or
    /// `noise.len() != indexer.hydro_count`.
    pub fn fill_forward_patches(
        &mut self,
        indexer: &StageIndexer,
        state: &[f64],
        noise: &[f64],
        base_row: usize,
        row_scale: &[f64],
    ) {
        debug_assert_eq!(
            state.len(),
            indexer.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = indexer.n_state,
        );
        debug_assert!(
            noise.len() == indexer.hydro_count || noise.is_empty(),
            "noise slice length {got} must equal hydro_count {expected} or be empty",
            got = noise.len(),
            expected = indexer.hydro_count,
        );

        let n = self.hydro_count;
        let l = self.max_par_order;

        // Category 1: storage-fixing rows [0, N)
        // patch(row = h, value = state[h] * row_scale[h])
        for (h, &sv) in state[..n].iter().enumerate() {
            self.indices[h] = h;
            let scaled = if row_scale.is_empty() {
                sv
            } else {
                sv * row_scale[h]
            };
            self.lower[h] = scaled;
            self.upper[h] = scaled;
        }

        // Category 2: AR lag-fixing rows [N, N*(1+L))
        // patch(row = N + ℓ·N + h, value = state[slot] * row_scale[slot])
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                self.indices[slot] = slot;
                let sv = state[slot];
                let scaled = if row_scale.is_empty() {
                    sv
                } else {
                    sv * row_scale[slot]
                };
                self.lower[slot] = scaled;
                self.upper[slot] = scaled;
            }
        }

        // Category 3: AR dynamics rows in the static non-dual region.
        // The noise value is computed by the caller as:
        //   noise[h] = template.row_lower[base_row + h] + noise_scale[h] * eta
        // where `template.row_lower` is already scaled (by `apply_row_scale`).
        // The `noise_scale` factor IS pre-scaled by the row scaling factor
        // during LP setup (see `setup.rs`: noise_scale[h] *= row_scale[base_row + h]),
        // so `noise[h]` is already in the correct scaled units and must be
        // written as-is without additional prescaling here.
        let cat3_start = n * (1 + l);
        for (h, &nv) in noise.iter().enumerate() {
            let slot = cat3_start + h;
            self.indices[slot] = ar_dynamics_row_offset(base_row, h);
            self.lower[slot] = nv;
            self.upper[slot] = nv;
        }
    }

    /// Fill `N*(1+L)` patches for a backward-pass (state-only) solve.
    ///
    /// Populates Categories 1 and 2 only.  Category 3 (noise innovations)
    /// is omitted because noise values for the backward pass come from the
    /// fixed opening tree and are applied separately by the caller.
    ///
    /// - **Category 1** — `N` storage-fixing patches: row `h` ← `state[h]`
    ///   for `h ∈ [0, N)`.
    /// - **Category 2** — `N*L` AR lag-fixing patches: row `N + ℓ·N + h` ←
    ///   `state[N + ℓ·N + h]` for `h ∈ [0, N)`, `ℓ ∈ [0, L)`.
    ///
    /// When `row_scale` is non-empty, each patch value is prescaled by
    /// `row_scale[row_index]` before being stored.  Pass `&[]` when no row
    /// scaling has been applied.
    ///
    /// Pass `&buf.indices[..active_len()]`, `&buf.lower[..active_len()]`, and
    /// `&buf.upper[..active_len()]` to `SolverInterface::set_row_bounds`,
    /// where `active_len` is `N*(1+L)`.  Use [`state_patch_count`] to obtain
    /// this length.
    ///
    /// # Arguments
    ///
    /// - `indexer` — LP layout map for this stage.
    /// - `state` — incoming state vector of length `n_state = N*(1+L)`.
    /// - `row_scale` — per-row scaling factors from the stage template.
    ///   Pass `&[]` when no scaling is active.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != indexer.n_state`.
    ///
    /// [`state_patch_count`]: PatchBuffer::state_patch_count
    pub fn fill_state_patches(&mut self, indexer: &StageIndexer, state: &[f64], row_scale: &[f64]) {
        debug_assert_eq!(
            state.len(),
            indexer.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = indexer.n_state,
        );

        let n = self.hydro_count;
        let l = self.max_par_order;

        // Category 1: storage-fixing rows [0, N)
        for (h, &sv) in state[..n].iter().enumerate() {
            self.indices[h] = h;
            let scaled = if row_scale.is_empty() {
                sv
            } else {
                sv * row_scale[h]
            };
            self.lower[h] = scaled;
            self.upper[h] = scaled;
        }

        // Category 2: AR lag-fixing rows [N, N*(1+L))
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                self.indices[slot] = slot;
                let sv = state[slot];
                let scaled = if row_scale.is_empty() {
                    sv
                } else {
                    sv * row_scale[slot]
                };
                self.lower[slot] = scaled;
                self.upper[slot] = scaled;
            }
        }
        // Category 3 is intentionally not written; the caller slices
        // [0..state_patch_count()] before passing to set_row_bounds.
    }

    /// Fill Category 4 load balance row patches for a forward-pass solve.
    ///
    /// Writes `n_load_buses * n_blocks` equality patches into the Category 4
    /// region starting at offset `N*(2+L)`.  Each patch targets the exact load
    /// balance row for bus `bus_positions[i]` and block `blk`:
    ///
    /// ```text
    /// row = load_row_start + bus_positions[i] * n_blocks + blk
    /// ```
    ///
    /// The `load_rhs` slice is laid out as `[bus0_blk0, bus0_blk1, …, bus1_blk0, …]`
    /// (bus-major, block-minor), matching `bus_positions` order.
    ///
    /// When `row_scale` is non-empty, each patch value is prescaled by
    /// `row_scale[row]` before being stored.  Pass `&[]` when no row scaling
    /// has been applied.
    ///
    /// After this call, [`forward_patch_count`] returns `N*(2+L) + n_load_buses * n_blocks`
    /// so that the correct slice is passed to `set_row_bounds`.
    ///
    /// # Arguments
    ///
    /// - `load_row_start` — first row index of the load-balance block in the LP.
    /// - `n_blocks` — number of time blocks for this stage.
    /// - `load_rhs` — patched RHS values; length must equal
    ///   `self.load_bus_count * n_blocks`.
    /// - `bus_positions` — LP bus position for each stochastic load bus;
    ///   length must equal `self.load_bus_count`.
    /// - `row_scale` — per-row scaling factors from the stage template.
    ///   Pass `&[]` when no scaling is active.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if:
    /// - `load_rhs.len() != self.load_bus_count * n_blocks`
    /// - `bus_positions.len() != self.load_bus_count`
    /// - `n_blocks > self.max_blocks`
    ///
    /// [`forward_patch_count`]: PatchBuffer::forward_patch_count
    pub fn fill_load_patches(
        &mut self,
        load_row_start: usize,
        n_blocks: usize,
        load_rhs: &[f64],
        bus_positions: &[usize],
        row_scale: &[f64],
    ) {
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

        let cat4_start = self.hydro_count * (2 + self.max_par_order);
        let mut slot = cat4_start;

        for (i, &bus_pos) in bus_positions.iter().enumerate() {
            for blk in 0..n_blocks {
                let row = load_row_start + bus_pos * n_blocks + blk;
                let rhs = load_rhs[i * n_blocks + blk];
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

    /// Fill Category 5 patches: z-inflow definition row RHS.
    ///
    /// Updates N rows starting at `z_inflow_row_start` with the realized-inflow
    /// RHS values from `z_inflow_rhs`. Each row is an equality constraint:
    /// `lower[i] = upper[i] = z_inflow_rhs[h]`.
    ///
    /// This method must be called after `fill_forward_patches` (which fills
    /// categories 1-3) and [`fill_load_patches`] (category 4), before
    /// `solver.set_row_bounds`.
    ///
    /// When `row_scale` is non-empty, each patch value is prescaled by
    /// `row_scale[row]`.  Pass `&[]` when no row scaling is active.
    ///
    /// # Arguments
    ///
    /// - `z_inflow_row_start` - first row index of the z-inflow definition rows.
    /// - `z_inflow_rhs` - per-hydro RHS values (length >= `hydro_count`).
    /// - `row_scale` - per-row scaling factors. Pass `&[]` when no scaling.
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

        // Place z-inflow patches immediately after active load patches
        // (not at the fixed Category 5 capacity offset) so they're included
        // in the forward_patch_count slice.
        let cat5_start = self.hydro_count * (2 + self.max_par_order) + self.active_load_patches;

        for (h, &rhs) in z_inflow_rhs.iter().enumerate().take(n) {
            let slot = cat5_start + h;
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

    /// Number of active patches after [`fill_forward_patches`], (optionally)
    /// [`fill_load_patches`], and (optionally) [`fill_z_inflow_patches`]:
    /// `N*(2+L) + active_load_patches + active_z_inflow_patches`.
    ///
    /// Use this to pass the full forward-pass buffer to `set_row_bounds`.
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    /// [`fill_z_inflow_patches`]: PatchBuffer::fill_z_inflow_patches
    #[must_use]
    #[inline]
    pub fn forward_patch_count(&self) -> usize {
        self.hydro_count * (2 + self.max_par_order)
            + self.active_load_patches
            + self.active_z_inflow_patches
    }

    /// Number of active patches after [`fill_state_patches`]: `N*(1+L)`.
    ///
    /// Slice the buffer to this length before passing to `set_row_bounds`
    /// when using the state-only (backward-pass) fill.
    ///
    /// [`fill_state_patches`]: PatchBuffer::fill_state_patches
    #[must_use]
    #[inline]
    pub fn state_patch_count(&self) -> usize {
        self.hydro_count * (1 + self.max_par_order)
    }
}

/// Compute the row index of hydro `h`'s AR dynamics constraint.
///
/// AR dynamics constraints live in the static non-dual region of the LP row
/// layout ([Solver Abstraction SS2.2]).  Their exact position depends on the
/// system's bus and block counts, which are not known to this crate.  The
/// `base_row` parameter is the first row index of the AR dynamics block,
/// computed during stage template construction and stored alongside the
/// [`StageIndexer`].
///
/// The formula is:
///
/// ```text
/// ar_dynamics_row(base_row, h) = base_row + h
/// ```
///
/// Hydros are laid out sequentially in the AR dynamics block, matching the
/// hydro-major order used throughout the column and row layout
/// ([Solver Abstraction SS2.1–SS2.2]).
///
/// # Pitfall
///
/// This row is in the **static non-dual region**, not in the fixing constraint
/// region `[0, n_state)`.  Do not confuse with the lag-fixing row
/// `N + ℓ·N + h` from Category 2.
#[must_use]
#[inline]
pub fn ar_dynamics_row_offset(base_row: usize, hydro_index: usize) -> usize {
    base_row + hydro_index
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::{PatchBuffer, ar_dynamics_row_offset};
    use crate::indexer::StageIndexer;

    /// Convenience: make an indexer without repeating N/L everywhere.
    fn idx(n: usize, l: usize) -> StageIndexer {
        StageIndexer::new(n, l)
    }

    #[test]
    fn new_3_2_sizes_to_15() {
        // N*(2+L) + N = 3*(2+2) + 3 = 15 (includes z-inflow capacity)
        let buf = PatchBuffer::new(3, 2, 0, 0);
        assert_eq!(buf.indices.len(), 15);
        assert_eq!(buf.lower.len(), 15);
        assert_eq!(buf.upper.len(), 15);
    }

    #[test]
    fn new_160_12_sizes_to_2400() {
        // N*(2+L) + N = 160*(2+12) + 160 = 2240 + 160 = 2400
        let buf = PatchBuffer::new(160, 12, 0, 0);
        assert_eq!(buf.indices.len(), 2400);
        assert_eq!(buf.lower.len(), 2400);
        assert_eq!(buf.upper.len(), 2400);
    }

    #[test]
    fn new_zero_lags_sizes_to_3n() {
        // N*(2+0) + N = 3*N patches (categories 1-3 + z-inflow)
        let buf = PatchBuffer::new(5, 0, 0, 0);
        assert_eq!(buf.indices.len(), 15); // 5*3 = 15
    }

    #[test]
    fn new_zero_hydros_sizes_to_zero() {
        let buf = PatchBuffer::new(0, 0, 0, 0);
        assert_eq!(buf.indices.len(), 0);
    }

    #[test]
    fn forward_patch_count_without_z_inflow_fill() {
        // Without calling fill_z_inflow_patches, active_z_inflow_patches=0.
        let buf = PatchBuffer::new(3, 2, 0, 0);
        // forward_patch_count = N*(2+L) + 0 + 0 = 12
        assert_eq!(buf.forward_patch_count(), 12);
    }

    #[test]
    fn state_patch_count_is_n_times_one_plus_l() {
        // N*(1+L) = 3*(1+2) = 9 for the spec worked example
        let buf = PatchBuffer::new(3, 2, 0, 0);
        assert_eq!(buf.state_patch_count(), 9);
    }

    #[test]
    fn state_patch_count_zero_lags() {
        // L = 0 → N*(1+0) = N = 4
        let buf = PatchBuffer::new(4, 0, 0, 0);
        assert_eq!(buf.state_patch_count(), 4);
    }

    #[test]
    fn ar_dynamics_row_offset_adds_base_plus_hydro() {
        assert_eq!(ar_dynamics_row_offset(100, 0), 100);
        assert_eq!(ar_dynamics_row_offset(100, 1), 101);
        assert_eq!(ar_dynamics_row_offset(100, 2), 102);
    }

    #[test]
    fn ar_dynamics_row_offset_zero_base() {
        assert_eq!(ar_dynamics_row_offset(0, 7), 7);
    }

    #[test]
    fn fill_forward_patches_category1_indices() {
        // First 3 patches correspond to storage fixing rows 0, 1, 2
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.indices[2], 2);
    }

    #[test]
    fn fill_forward_patches_category2_indices() {
        // Patches 3-8 correspond to AR lag fixing rows 3..=8
        // Row index formula: N + ℓ·N + h
        // ℓ=0: 3+0=3, 3+1=4, 3+2=5
        // ℓ=1: 6+0=6, 6+1=7, 6+2=8
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        assert_eq!(buf.indices[3], 3); // N + 0·N + 0
        assert_eq!(buf.indices[4], 4); // N + 0·N + 1
        assert_eq!(buf.indices[5], 5); // N + 0·N + 2
        assert_eq!(buf.indices[6], 6); // N + 1·N + 0
        assert_eq!(buf.indices[7], 7); // N + 1·N + 1
        assert_eq!(buf.indices[8], 8); // N + 1·N + 2
    }

    #[test]
    fn fill_forward_patches_category3_indices() {
        // Last 3 patches correspond to AR dynamics rows
        // base_row = 50 → rows 50, 51, 52
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        assert_eq!(buf.indices[9], 50); // ar_dynamics_row_offset(50, 0)
        assert_eq!(buf.indices[10], 51); // ar_dynamics_row_offset(50, 1)
        assert_eq!(buf.indices[11], 52); // ar_dynamics_row_offset(50, 2)
    }

    #[test]
    fn fill_forward_patches_category1_values() {
        // Category 1: lower == upper == state[h]
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        assert_eq!(buf.lower[0], 10.0);
        assert_eq!(buf.upper[0], 10.0);
        assert_eq!(buf.lower[1], 20.0);
        assert_eq!(buf.upper[1], 20.0);
        assert_eq!(buf.lower[2], 30.0);
        assert_eq!(buf.upper[2], 30.0);
    }

    #[test]
    fn fill_forward_patches_category2_values() {
        // Category 2: lower == upper == state[N + ℓ·N + h]
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        // ℓ=0: state[3]=1, state[4]=2, state[5]=3
        assert_eq!(buf.lower[3], 1.0);
        assert_eq!(buf.upper[3], 1.0);
        assert_eq!(buf.lower[4], 2.0);
        assert_eq!(buf.upper[4], 2.0);
        assert_eq!(buf.lower[5], 3.0);
        assert_eq!(buf.upper[5], 3.0);
        // ℓ=1: state[6]=4, state[7]=5, state[8]=6
        assert_eq!(buf.lower[6], 4.0);
        assert_eq!(buf.upper[6], 4.0);
        assert_eq!(buf.lower[7], 5.0);
        assert_eq!(buf.upper[7], 5.0);
        assert_eq!(buf.lower[8], 6.0);
        assert_eq!(buf.upper[8], 6.0);
    }

    #[test]
    fn fill_forward_patches_category3_values() {
        // Category 3: lower == upper == noise[h]
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        assert_eq!(buf.lower[9], 0.1);
        assert_eq!(buf.upper[9], 0.1);
        assert_eq!(buf.lower[10], 0.2);
        assert_eq!(buf.upper[10], 0.2);
        assert_eq!(buf.lower[11], 0.3);
        assert_eq!(buf.upper[11], 0.3);
    }

    #[test]
    fn fill_forward_patches_all_equality_constraints() {
        // Every patch must satisfy lower == upper (equality constraint)
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        for i in 0..buf.forward_patch_count() {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    #[test]
    fn fill_state_patches_count_is_n_times_one_plus_l() {
        // Backward pass: N*(1+L) patches, no noise
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state, &[]);

        // Active slice is [0, 9) = [0, N*(1+L))
        assert_eq!(buf.state_patch_count(), 9);
    }

    #[test]
    fn fill_state_patches_category1_correct() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state, &[]);

        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 10.0);
        assert_eq!(buf.upper[0], 10.0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.lower[1], 20.0);
        assert_eq!(buf.upper[1], 20.0);
        assert_eq!(buf.indices[2], 2);
        assert_eq!(buf.lower[2], 30.0);
        assert_eq!(buf.upper[2], 30.0);
    }

    #[test]
    fn fill_state_patches_category2_correct() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state, &[]);

        // Same index/value expectations as fill_forward_patches for cat 2
        assert_eq!(buf.indices[3], 3);
        assert_eq!(buf.lower[3], 1.0);
        assert_eq!(buf.upper[3], 1.0);
        assert_eq!(buf.indices[8], 8);
        assert_eq!(buf.lower[8], 6.0);
        assert_eq!(buf.upper[8], 6.0);
    }

    #[test]
    fn fill_state_patches_equality_constraints_in_active_range() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state, &[]);

        let active = buf.state_patch_count();
        for i in 0..active {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "state patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    #[test]
    fn forward_patches_zero_lags_only_storage_and_noise() {
        // L=0: N*(2+0) = 2*N patches: N storage + N noise, zero lag
        let n = 2;
        let mut buf = PatchBuffer::new(n, 0, 0, 0);
        let state = [5.0, 7.0]; // n_state = 2*(1+0) = 2
        let noise = [0.5, 0.6];
        buf.fill_forward_patches(&idx(n, 0), &state, &noise, 10, &[]);

        // 4 patches total
        assert_eq!(buf.forward_patch_count(), 4);

        // Category 1: rows 0, 1
        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 5.0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.lower[1], 7.0);

        // Category 2: empty (L=0, zero iterations)

        // Category 3 starts at slot N*(1+0) = N = 2
        assert_eq!(buf.indices[2], 10); // ar_dynamics_row_offset(10, 0)
        assert_eq!(buf.lower[2], 0.5);
        assert_eq!(buf.indices[3], 11); // ar_dynamics_row_offset(10, 1)
        assert_eq!(buf.lower[3], 0.6);
    }

    #[test]
    fn state_patches_zero_lags_only_storage() {
        // L=0: N*(1+0) = N patches (storage only)
        let n = 3;
        let mut buf = PatchBuffer::new(n, 0, 0, 0);
        let state = [1.0, 2.0, 3.0]; // n_state = 3
        buf.fill_state_patches(&idx(n, 0), &state, &[]);

        assert_eq!(buf.state_patch_count(), 3);
        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 1.0);
        assert_eq!(buf.upper[0], 1.0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.lower[1], 2.0);
        assert_eq!(buf.indices[2], 2);
        assert_eq!(buf.lower[2], 3.0);
    }

    #[test]
    fn production_scale_forward_patch_count() {
        // Without fill_z_inflow_patches, forward_patch_count = 160*(2+12) = 2240.
        // Buffer capacity = 2240 + 160 (z-inflow) = 2400.
        let buf = PatchBuffer::new(160, 12, 0, 0);
        assert_eq!(buf.forward_patch_count(), 2240);
        assert_eq!(buf.indices.len(), 2400);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // fixture: values are small integers, no precision lost
    fn production_scale_fill_forward_patches_smoke() {
        let n = 160;
        let l = 12;
        let mut buf = PatchBuffer::new(n, l, 0, 0);
        let n_state = n * (1 + l);
        let state: Vec<f64> = (0..n_state).map(|i| i as f64).collect();
        let noise: Vec<f64> = (0..n).map(|h| h as f64 * 0.01).collect();
        buf.fill_forward_patches(&StageIndexer::new(n, l), &state, &noise, 500, &[]);

        // Spot-check category 1
        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 0.0);

        // Spot-check category 3 start at slot N*(1+L) = 160*13 = 2080
        assert_eq!(buf.indices[2080], 500); // ar_dynamics_row_offset(500, 0)
        assert_eq!(buf.lower[2080], 0.0); // noise[0]
        assert_eq!(buf.indices[2239], 659); // ar_dynamics_row_offset(500, 159)
        assert_eq!(buf.lower[2239], 159.0 * 0.01);

        // All patches must be equality constraints
        for i in 0..buf.forward_patch_count() {
            assert_eq!(buf.lower[i], buf.upper[i], "patch {i} not equality");
        }
    }

    #[test]
    fn clone_and_debug() {
        let buf = PatchBuffer::new(3, 2, 0, 0);
        let cloned = buf.clone();
        assert_eq!(cloned.indices.len(), buf.indices.len());

        let s = format!("{buf:?}");
        assert!(s.contains("PatchBuffer"));
    }

    // -------------------------------------------------------------------------
    // Category 4 (load balance) unit tests
    // -------------------------------------------------------------------------

    /// AC: `PatchBuffer::new(2, 1, 1, 3)` → capacity = 2*(2+1) + 1*3 + 2 = 11.
    #[test]
    fn new_with_load_allocates_correct_capacity() {
        let buf = PatchBuffer::new(2, 1, 1, 3);
        // 2*(2+1) + 1*3 + 2 (z-inflow) = 6 + 3 + 2 = 11
        assert_eq!(buf.indices.len(), 11);
        assert_eq!(buf.lower.len(), 11);
        assert_eq!(buf.upper.len(), 11);
    }

    /// Category 4 row indices follow `row = load_row_start + bus_positions[i] * n_blocks + blk`.
    ///
    /// With `n_load_buses=2, n_blocks=2, bus_positions=[0,1], load_row_start=100`:
    /// - bus 0, blk 0 → 100 + 0*2 + 0 = 100
    /// - bus 0, blk 1 → 100 + 0*2 + 1 = 101
    /// - bus 1, blk 0 → 100 + 1*2 + 0 = 102
    /// - bus 1, blk 1 → 100 + 1*2 + 1 = 103
    #[test]
    fn fill_load_patches_correct_indices() {
        // N=0, L=0, M=2, B=2 → capacity = 0 + 2*2 = 4
        let mut buf = PatchBuffer::new(0, 0, 2, 2);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(100, 2, &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.indices[0], 100); // bus 0, blk 0
        assert_eq!(buf.indices[1], 101); // bus 0, blk 1
        assert_eq!(buf.indices[2], 102); // bus 1, blk 0
        assert_eq!(buf.indices[3], 103); // bus 1, blk 1
    }

    /// Category 4 lower and upper bounds equal the corresponding `load_rhs` value.
    #[test]
    fn fill_load_patches_correct_values() {
        let mut buf = PatchBuffer::new(0, 0, 2, 2);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(100, 2, &load_rhs, &bus_positions, &[]);

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
        let mut buf = PatchBuffer::new(3, 2, 2, 3);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, 3, &load_rhs, &bus_positions, &[]);

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

    /// `forward_patch_count` includes Category 4 after `fill_load_patches`.
    ///
    /// N=3, L=2 → base = 3*(2+2) = 12; M=2, `n_blocks=3` → load = 6; total = 18.
    #[test]
    fn forward_patch_count_includes_load() {
        let mut buf = PatchBuffer::new(3, 2, 2, 3);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, 3, &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.forward_patch_count(), 18); // 12 + 6
    }

    /// `state_patch_count` is unaffected by Category 4 — no lag structure for load.
    #[test]
    fn state_patch_count_excludes_load() {
        let mut buf = PatchBuffer::new(3, 2, 2, 3);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state, &[]);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, 3, &load_rhs, &bus_positions, &[]);

        // state_patch_count must be N*(1+L) = 3*3 = 9, not 18
        assert_eq!(buf.state_patch_count(), 9);
    }

    /// When `n_load_buses == 0`, `forward_patch_count` equals `N*(2+L)` unchanged.
    #[test]
    fn zero_load_buses_no_category4() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        // No fill_load_patches call: active_load_patches stays 0
        assert_eq!(buf.forward_patch_count(), 12); // 3*(2+2) only
    }
}
