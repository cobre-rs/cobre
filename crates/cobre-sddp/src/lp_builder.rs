//! Stage LP patch buffer and stage template builder for SDDP subproblems.
//!
//! This module provides two public facilities:
//!
//! - [`PatchBuffer`]: pre-allocates the three parallel arrays consumed by
//!   `SolverInterface::set_row_bounds` and fills them with scenario-dependent
//!   values before each LP solve.  Allocating once at training start and
//!   reusing the same buffer across all iterations and stages is critical for
//!   hot-path performance: the training loop calls `fill_forward_patches` or
//!   `fill_state_patches` millions of times.
//!
//! - [`build_stage_templates`]: constructs a `Vec<StageTemplate>` from a
//!   loaded `System` — one template per study stage.  The template encodes
//!   the full structural LP in CSC format, column/row bounds, and objective
//!   coefficients.  It is built once at startup and shared read-only across
//!   all threads.
//!
//! ## LP layout (Solver Abstraction SS2)
//!
//! ### Column layout (contiguous regions)
//!
//! ```text
//! [0,  N)            outgoing storage      (N = n_hydros)
//! [N,  N*(1+L))      AR lag variables      (N*L lags, hydro-major)
//! [N*(1+L), N*(2+L)) incoming storage      (fixed by storage-fixing rows)
//! N*(2+L)            theta                 (future cost, scalar)
//! N*(2+L)+1 ..       decision variables:
//!   hydro turbine:   N*K columns
//!   hydro spillage:  N*K columns
//!   thermal gen:     T*K columns
//!   line fwd flow:   Lines*K columns
//!   line rev flow:   Lines*K columns
//!   bus deficit:     B*K columns
//!   bus excess:      B*K columns
//! ```
//!
//! ### Row layout (contiguous regions)
//!
//! ```text
//! [0,  N)            storage-fixing constraints
//! [N,  N*(1+L))      AR lag-fixing constraints
//! N*(1+L) ..         structural constraints (non-dual region):
//!   water balance:   N rows   (one per hydro)
//!   load balance:    B*K rows (one per bus per block)
//! ```
//!
//! The AR dynamics (noise patch target) rows are the water balance constraints
//! beginning at row `n_dual_relevant` (= `N*(1+L)`).  The `base_rows` value
//! returned alongside the templates encodes this offset for each stage so
//! that [`PatchBuffer`] can update the correct RHS during forward-pass solves.
//!
//! ## Patch sequence (Training Loop SS4.2a)
//!
//! Each forward-pass solve requires exactly `N*(2+L)` row-bound patches:
//!
//! ```text
//! Category 1 — storage fixing    rows [0, N)
//!     patch row h = state[h]   for h in [0, N)
//!
//! Category 2 — AR lag fixing     rows [N, N*(1+L))
//!     patch row N + ℓ·N + h = state[N + ℓ·N + h]
//!     for h in [0, N), ℓ in [0, L)
//!
//! Category 3 — noise innovation   N rows in the static non-dual region
//!     patch ar_dynamics_row(base_row, h) = noise[h]   for h in [0, N)
//! ```
//!
//! The backward pass uses only categories 1 and 2 (`N*(1+L)` patches); noise
//! innovations are drawn from the fixed opening tree by the caller.
//!
//! ## Row index types
//!
//! All indices are stored as `usize` to match the `set_row_bounds` interface
//! signature directly; no casting is required at the call site.
//!
//! ## Worked example (SS4.2a): N = 3, L = 2
//!
//! ```text
//! Patch  Category       Row index              Value
//!     0  storage-fix    0                      state[0]  (H0)
//!     1  storage-fix    1                      state[1]  (H1)
//!     2  storage-fix    2                      state[2]  (H2)
//!     3  lag-fix        N + 0·N + 0 = 3        state[3]  (H0 lag 0)
//!     4  lag-fix        N + 0·N + 1 = 4        state[4]  (H1 lag 0)
//!     5  lag-fix        N + 0·N + 2 = 5        state[5]  (H2 lag 0)
//!     6  lag-fix        N + 1·N + 0 = 6        state[6]  (H0 lag 1)
//!     7  lag-fix        N + 1·N + 1 = 7        state[7]  (H1 lag 1)
//!     8  lag-fix        N + 1·N + 2 = 8        state[8]  (H2 lag 1)
//!     9  noise-fix      ar_dynamics_row(br, 0) noise[0]  (H0)
//!    10  noise-fix      ar_dynamics_row(br, 1) noise[1]  (H1)
//!    11  noise-fix      ar_dynamics_row(br, 2) noise[2]  (H2)
//! ```
//!
//! Total: 12 = 3*(2+2) patches.

use cobre_core::System;
use cobre_core::entities::hydro::HydroGenerationModel;
use cobre_solver::StageTemplate;

use crate::indexer::StageIndexer;

/// Pre-allocated row-bound patch arrays for one SDDP stage LP solve.
///
/// Holds three parallel `Vec`s of equal length ready for a single
/// `SolverInterface::set_row_bounds` call.  The buffer is sized for
/// `N*(2+L)` patches at construction and reused across all iterations.
///
/// # Memory layout
///
/// The `N*(2+L)` entries are written in category order:
///
/// | Entry range           | Category                           |
/// | --------------------- | ---------------------------------- |
/// | `[0, N)`              | Storage-fixing (Category 1)        |
/// | `[N, N*(1+L))`        | AR lag-fixing (Category 2)         |
/// | `[N*(1+L), N*(2+L))`  | AR dynamics / noise (Category 3)   |
///
/// [`fill_state_patches`](PatchBuffer::fill_state_patches) writes only
/// `[0, N*(1+L))` (Categories 1 and 2).  The Category 3 slice is left as
/// the previous iteration's values, which is safe because the caller passes
/// only `&self.indices[..active_len]` to `set_row_bounds`.
#[derive(Debug, Clone)]
pub struct PatchBuffer {
    /// Row indices to patch.
    ///
    /// Length `N*(2+L)`.  Entries are `usize` to match the
    /// `set_row_bounds(&[usize], ...)` interface directly.
    pub indices: Vec<usize>,

    /// New lower bounds for each patched row.
    ///
    /// Length `N*(2+L)`.  For equality constraints, `lower[i] == upper[i]`.
    pub lower: Vec<f64>,

    /// New upper bounds for each patched row.
    ///
    /// Length `N*(2+L)`.  For equality constraints, `upper[i] == lower[i]`.
    pub upper: Vec<f64>,

    /// Number of operating hydro plants (N).
    hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    max_par_order: usize,
}

impl PatchBuffer {
    /// Construct a [`PatchBuffer`] pre-allocated for `N*(2+L)` patches.
    ///
    /// `hydro_count` is the number of operating hydro plants (N).
    /// `max_par_order` is the maximum PAR order across all operating hydros (L).
    ///
    /// The buffer's `indices`, `lower`, and `upper` vectors are sized to
    /// `N*(2+L)` and zero-initialised.  Call [`fill_forward_patches`] or
    /// [`fill_state_patches`] to populate them before each LP solve.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::PatchBuffer;
    ///
    /// // 3-hydro AR(2) system from spec SS4.2a worked example
    /// let buf = PatchBuffer::new(3, 2);
    /// assert_eq!(buf.indices.len(), 12); // 3*(2+2)
    ///
    /// // Production scale: N = 160, L = 12
    /// let big = PatchBuffer::new(160, 12);
    /// assert_eq!(big.indices.len(), 2240); // 160*(2+12)
    ///
    /// // Edge case: no lags (L = 0) — only storage + noise patches
    /// let no_lag = PatchBuffer::new(5, 0);
    /// assert_eq!(no_lag.indices.len(), 10); // 5*(2+0)
    /// ```
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    /// [`fill_state_patches`]: PatchBuffer::fill_state_patches
    #[must_use]
    pub fn new(hydro_count: usize, max_par_order: usize) -> Self {
        let capacity = hydro_count * (2 + max_par_order);
        Self {
            indices: vec![0; capacity],
            lower: vec![0.0; capacity],
            upper: vec![0.0; capacity],
            hydro_count,
            max_par_order,
        }
    }

    /// Fill all `N*(2+L)` patches for a forward-pass solve.
    ///
    /// Populates Categories 1, 2, and 3 in sequence:
    ///
    /// - **Category 1** — `N` storage-fixing patches: row `h` ← `state[h]`
    ///   for `h ∈ [0, N)`.
    /// - **Category 2** — `N*L` AR lag-fixing patches: row `N + ℓ·N + h` ←
    ///   `state[N + ℓ·N + h]` for `h ∈ [0, N)`, `ℓ ∈ [0, L)`.
    /// - **Category 3** — `N` noise-fixing patches: row
    ///   `ar_dynamics_row_offset(base_row, h)` ← `noise[h]` for `h ∈ [0, N)`.
    ///
    /// All patches are equality constraints: `lower[i] == upper[i] == value`.
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
    ) {
        debug_assert_eq!(
            state.len(),
            indexer.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = indexer.n_state,
        );
        debug_assert_eq!(
            noise.len(),
            indexer.hydro_count,
            "noise slice length {got} != hydro_count {expected}",
            got = noise.len(),
            expected = indexer.hydro_count,
        );

        let n = self.hydro_count;
        let l = self.max_par_order;

        // Category 1: storage-fixing rows [0, N)
        // patch(row = h, value = state[h])
        for (h, &sv) in state[..n].iter().enumerate() {
            self.indices[h] = h;
            self.lower[h] = sv;
            self.upper[h] = sv;
        }

        // Category 2: AR lag-fixing rows [N, N*(1+L))
        // patch(row = N + ℓ·N + h, value = state[N + ℓ·N + h])
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                self.indices[slot] = slot;
                self.lower[slot] = state[slot];
                self.upper[slot] = state[slot];
            }
        }

        // Category 3: AR dynamics rows in the static non-dual region
        // patch(row = ar_dynamics_row_offset(base_row, h), value = noise[h])
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
    /// Pass `&buf.indices[..active_len()]`, `&buf.lower[..active_len()]`, and
    /// `&buf.upper[..active_len()]` to `SolverInterface::set_row_bounds`,
    /// where `active_len` is `N*(1+L)`.  Use [`state_patch_count`] to obtain
    /// this length.
    ///
    /// # Arguments
    ///
    /// - `indexer` — LP layout map for this stage.
    /// - `state` — incoming state vector of length `n_state = N*(1+L)`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != indexer.n_state`.
    ///
    /// [`state_patch_count`]: PatchBuffer::state_patch_count
    pub fn fill_state_patches(&mut self, indexer: &StageIndexer, state: &[f64]) {
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
            self.lower[h] = sv;
            self.upper[h] = sv;
        }

        // Category 2: AR lag-fixing rows [N, N*(1+L))
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                self.indices[slot] = slot;
                self.lower[slot] = state[slot];
                self.upper[slot] = state[slot];
            }
        }
        // Category 3 is intentionally not written; the caller slices
        // [0..state_patch_count()] before passing to set_row_bounds.
    }

    /// Number of active patches after [`fill_forward_patches`]: `N*(2+L)`.
    ///
    /// Use this to pass the full buffer to `set_row_bounds`.
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    #[must_use]
    #[inline]
    pub fn forward_patch_count(&self) -> usize {
        self.hydro_count * (2 + self.max_par_order)
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

// ---------------------------------------------------------------------------
// build_stage_templates
// ---------------------------------------------------------------------------

/// Outcome of [`build_stage_templates`]: one [`StageTemplate`] per study stage
/// plus the per-stage `base_rows` offsets needed by [`PatchBuffer`].
///
/// `base_rows[s]` is the row index of the first water-balance (AR dynamics)
/// constraint in stage `s`.  It equals `template.n_dual_relevant` for every
/// stage (constant when all stages share the same entity set, which is the
/// case for the minimal viable solver).  It is stored per-stage for forward
/// compatibility with stages that have different active entity sets.
#[derive(Debug, Clone)]
pub struct StageTemplates {
    /// One structural LP template per study stage, in stage order.
    pub templates: Vec<StageTemplate>,
    /// Row index of the first water-balance constraint in each stage's LP.
    ///
    /// Length equals `templates.len()`.  Used by [`PatchBuffer::fill_forward_patches`]
    /// to locate the noise-injection rows (Category 3 patches).
    pub base_rows: Vec<usize>,
}

/// Conversion factor from m³/s-per-block to hm³, assuming 30-day months.
///
/// `zeta = seconds_per_hour * duration_hours / m3_per_hm3`
/// `     = 3600 * 720 / 1_000_000 = 2.592` (30-day month, all hours)
///
/// This constant is overridden per block using the actual `Block::duration_hours`
/// from the stage definition.
const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0; // multiply by hours to get hm³

/// Build one [`StageTemplate`] per study stage from a fully loaded [`System`].
///
/// The templates encode the complete structural LP for each SDDP subproblem
/// in CSC format, ready for bulk-loading via `SolverInterface::load_model`.
/// They are constructed once at solver initialisation and shared read-only
/// across all solver threads.
///
/// ## Column and row layout
///
/// See the [module-level documentation](self) for the full LP layout.
/// Key dimensions for a stage with N hydros, T thermals, Lines lines,
/// B buses, and K blocks per stage:
///
/// - `num_cols = N*(2+L) + 1 + N*K*2 + T*K + Lines*K*2 + B*K*2`
/// - `num_rows = N*(1+L) + N + B*K`  (fixing + water balance + load balance)
/// - `n_state  = N*(1+L)`
/// - `n_transfer = N*L`  (storage + all lags except the oldest)
/// - `n_dual_relevant = N*(1+L)`  (no FPHA for v0.1.0)
///
/// ## PAR order and `max_par_order`
///
/// `max_par_order` is derived from the maximum AR coefficient count across all
/// hydro inflow models for the stage.  All hydros use the same uniform lag
/// stride `max_par_order` to enable SIMD-friendly contiguous access.
///
/// ## Objective coefficients
///
/// Costs are expressed in `$/MWh` (thermal, deficit, excess, lines) multiplied
/// by the block duration in hours so they integrate to $/block.  Storage, lag,
/// incoming-storage, theta, turbine, and spillage columns carry zero or small
/// regularization costs drawn from the resolved penalty tables.
///
/// ## Errors
///
/// Returns `None` for a system with zero stages.  All entity counts may be
/// zero (valid for degenerate test systems).
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_sddp::lp_builder::build_stage_templates;
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "B1".to_string(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 1000.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().expect("valid");
/// // No stages → empty result.
/// let result = build_stage_templates(&system);
/// assert!(result.templates.is_empty());
/// ```
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_stage_templates(system: &System) -> StageTemplates {
    // Only build templates for study stages (id >= 0), in canonical order.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    if study_stages.is_empty() {
        return StageTemplates {
            templates: Vec::new(),
            base_rows: Vec::new(),
        };
    }

    let hydros = system.hydros();
    let thermals = system.thermals();
    let lines = system.lines();
    let buses = system.buses();
    let cascade = system.cascade();
    let bounds = system.bounds();
    let penalties = system.penalties();

    let n_hydros = hydros.len();
    let n_thermals = thermals.len();
    let n_lines = lines.len();
    let n_buses = buses.len();

    // Compute max PAR order across all hydros and study stages.
    // All hydros use a uniform lag stride for SIMD alignment.
    let max_par_order: usize = system
        .inflow_models()
        .iter()
        .filter(|m| m.stage_id >= 0)
        .map(|m| m.ar_coefficients.len())
        .max()
        .unwrap_or(0);

    // Build a position map: hydro EntityId -> index in hydros slice.
    let hydro_pos: std::collections::HashMap<cobre_core::EntityId, usize> =
        hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();

    // Build a position map: bus EntityId -> index in buses slice.
    let bus_pos: std::collections::HashMap<cobre_core::EntityId, usize> =
        buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    let n_study_stages = study_stages.len();
    let mut templates = Vec::with_capacity(n_study_stages);
    let mut base_rows = Vec::with_capacity(n_study_stages);

    for (stage_idx, stage) in study_stages.iter().enumerate() {
        let n_blks = stage.blocks.len();
        let idx = StageIndexer::new(n_hydros, max_par_order);
        let decision_start = idx.theta + 1;

        let col_turbine_start = decision_start;
        let col_spillage_start = col_turbine_start + n_hydros * n_blks;
        let col_thermal_start = col_spillage_start + n_hydros * n_blks;
        let col_line_fwd_start = col_thermal_start + n_thermals * n_blks;
        let col_line_rev_start = col_line_fwd_start + n_lines * n_blks;
        let col_deficit_start = col_line_rev_start + n_lines * n_blks;
        let col_excess_start = col_deficit_start + n_buses * n_blks;
        let num_cols = col_excess_start + n_buses * n_blks;

        let n_state = idx.n_state;
        let n_dual_relevant = n_state;
        let row_water_balance_start = n_dual_relevant;
        let row_load_balance_start = row_water_balance_start + n_hydros;
        let num_rows = row_load_balance_start + n_buses * n_blks;

        let n_h = n_hydros;
        let n_b = n_buses;
        let lag_order = max_par_order;

        let stage_base_row = row_water_balance_start;

        let mut col_lower = vec![0.0_f64; num_cols];
        let mut col_upper = vec![f64::INFINITY; num_cols];
        let mut objective = vec![0.0_f64; num_cols];

        for (h_idx, hydro) in hydros.iter().enumerate() {
            let hb = bounds.hydro_bounds(h_idx, stage_idx);
            col_lower[h_idx] = hb.min_storage_hm3;
            col_upper[h_idx] = hb.max_storage_hm3;
            col_lower[idx.storage_in.start + h_idx] = f64::NEG_INFINITY;
            col_upper[idx.storage_in.start + h_idx] = f64::INFINITY;
            let _ = hydro;
        }

        for lag_col in idx.inflow_lags.clone() {
            col_lower[lag_col] = f64::NEG_INFINITY;
            col_upper[lag_col] = f64::INFINITY;
        }

        // Theta bounded below by zero: all penalties are non-negative, terminal
        // stage value is zero. This ensures iteration 1 LPs with empty cut pools
        // are bounded (return theta=0) rather than unbounded by HiGHS.
        col_lower[idx.theta] = 0.0;
        col_upper[idx.theta] = f64::INFINITY;
        objective[idx.theta] = 1.0;

        for (h_idx, hydro) in hydros.iter().enumerate() {
            let hb = bounds.hydro_bounds(h_idx, stage_idx);
            for blk in 0..n_blks {
                let col = col_turbine_start + h_idx * n_blks + blk;
                col_lower[col] = hb.min_turbined_m3s;
                col_upper[col] = hb.max_turbined_m3s;
                objective[col] = 0.0;
                let _ = hydro;
            }
        }

        for (h_idx, _hydro) in hydros.iter().enumerate() {
            let hp = penalties.hydro_penalties(h_idx, stage_idx);
            for blk in 0..n_blks {
                let col = col_spillage_start + h_idx * n_blks + blk;
                col_lower[col] = 0.0;
                col_upper[col] = f64::INFINITY;
                let block_hours = stage.blocks[blk].duration_hours;
                objective[col] = hp.spillage_cost * block_hours;
            }
        }

        for (t_idx, thermal) in thermals.iter().enumerate() {
            let tb = bounds.thermal_bounds(t_idx, stage_idx);
            let marginal_cost_per_mwh = thermal
                .cost_segments
                .first()
                .map_or(0.0, |seg| seg.cost_per_mwh);
            for blk in 0..n_blks {
                let col = col_thermal_start + t_idx * n_blks + blk;
                col_lower[col] = tb.min_generation_mw;
                col_upper[col] = tb.max_generation_mw;
                let block_hours = stage.blocks[blk].duration_hours;
                objective[col] = marginal_cost_per_mwh * block_hours;
            }
        }

        for (l_idx, line) in lines.iter().enumerate() {
            let lb = bounds.line_bounds(l_idx, stage_idx);
            let lp = penalties.line_penalties(l_idx, stage_idx);
            for blk in 0..n_blks {
                let col_fwd = col_line_fwd_start + l_idx * n_blks + blk;
                let col_rev = col_line_rev_start + l_idx * n_blks + blk;
                col_lower[col_fwd] = 0.0;
                col_upper[col_fwd] = lb.direct_mw;
                col_lower[col_rev] = 0.0;
                col_upper[col_rev] = lb.reverse_mw;
                let block_hours = stage.blocks[blk].duration_hours;
                objective[col_fwd] = lp.exchange_cost * block_hours;
                objective[col_rev] = lp.exchange_cost * block_hours;
                let _ = line;
            }
        }

        for (b_idx, bus) in buses.iter().enumerate() {
            let bp = penalties.bus_penalties(b_idx, stage_idx);
            let deficit_cost = bus
                .deficit_segments
                .last()
                .map_or(0.0, |seg| seg.cost_per_mwh);
            for blk in 0..n_blks {
                let col_def = col_deficit_start + b_idx * n_blks + blk;
                let col_exc = col_excess_start + b_idx * n_blks + blk;
                col_lower[col_def] = 0.0;
                col_upper[col_def] = f64::INFINITY;
                col_lower[col_exc] = 0.0;
                col_upper[col_exc] = f64::INFINITY;
                let block_hours = stage.blocks[blk].duration_hours;
                objective[col_def] = deficit_cost * block_hours;
                objective[col_exc] = bp.excess_cost * block_hours;
            }
        }

        let mut row_lower = vec![0.0_f64; num_rows];
        let mut row_upper = vec![0.0_f64; num_rows];

        for (b_idx, bus) in buses.iter().enumerate() {
            let load_models = system.load_models();
            let mean_mw = load_models
                .iter()
                .find(|lm| lm.bus_id == bus.id && lm.stage_id == stage.id)
                .map_or(0.0, |lm| lm.mean_mw);
            for blk in 0..n_blks {
                let row = row_load_balance_start + b_idx * n_blks + blk;
                row_lower[row] = mean_mw;
                row_upper[row] = mean_mw;
            }
        }

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_cols];

        macro_rules! add_entry {
            ($col:expr, $row:expr, $val:expr) => {
                col_entries[$col].push(($row, $val));
            };
        }

        for h in 0..n_h {
            let col = idx.storage_in.start + h;
            add_entry!(col, h, 1.0);
        }

        for lag in 0..lag_order {
            for h in 0..n_h {
                let col = idx.inflow_lags.start + lag * n_h + h;
                let row = n_h + lag * n_h + h;
                add_entry!(col, row, 1.0);
            }
        }

        for h_idx in 0..n_h {
            let hydro = &hydros[h_idx];
            let row = row_water_balance_start + h_idx;
            add_entry!(h_idx, row, 1.0);
            add_entry!(idx.storage_in.start + h_idx, row, -1.0);
            for blk in 0..n_blks {
                let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
                let col_turbine = col_turbine_start + h_idx * n_blks + blk;
                add_entry!(col_turbine, row, -tau_h);
                let col_spillage = col_spillage_start + h_idx * n_blks + blk;
                add_entry!(col_spillage, row, -tau_h);
                let upstream_ids = cascade.upstream(hydro.id);
                for &up_id in upstream_ids {
                    if let Some(&u_idx) = hydro_pos.get(&up_id) {
                        let col_upstream_turbine = col_turbine_start + u_idx * n_blks + blk;
                        add_entry!(col_upstream_turbine, row, tau_h);
                        let col_upstream_spillage = col_spillage_start + u_idx * n_blks + blk;
                        add_entry!(col_upstream_spillage, row, tau_h);
                    }
                }
            }
        }
        for (h_idx, hydro) in hydros.iter().enumerate() {
            let rho = match &hydro.generation_model {
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s,
                }
                | HydroGenerationModel::LinearizedHead {
                    productivity_mw_per_m3s,
                } => *productivity_mw_per_m3s,
                HydroGenerationModel::Fpha => 0.0, // FPHA not supported in v0.1.0
            };

            if let Some(&b_idx) = bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = row_load_balance_start + b_idx * n_blks + blk;
                    let col = col_turbine_start + h_idx * n_blks + blk;
                    add_entry!(col, row, rho);
                }
            }
        }

        // Thermal generation contribution
        for (t_idx, thermal) in thermals.iter().enumerate() {
            if let Some(&b_idx) = bus_pos.get(&thermal.bus_id) {
                for blk in 0..n_blks {
                    let row = row_load_balance_start + b_idx * n_blks + blk;
                    let col = col_thermal_start + t_idx * n_blks + blk;
                    add_entry!(col, row, 1.0);
                }
            }
        }

        // Line forward flow (source→target): +1.0 at target bus, -1.0 at source bus
        for (l_idx, line) in lines.iter().enumerate() {
            let src_idx = bus_pos.get(&line.source_bus_id).copied();
            let tgt_idx = bus_pos.get(&line.target_bus_id).copied();
            for blk in 0..n_blks {
                let col_fwd = col_line_fwd_start + l_idx * n_blks + blk;
                let col_rev = col_line_rev_start + l_idx * n_blks + blk;
                if let Some(tgt) = tgt_idx {
                    let row = row_load_balance_start + tgt * n_blks + blk;
                    add_entry!(col_fwd, row, 1.0); // fwd flow adds to target
                    add_entry!(col_rev, row, -1.0); // rev flow takes from target
                }
                if let Some(src) = src_idx {
                    let row = row_load_balance_start + src * n_blks + blk;
                    add_entry!(col_fwd, row, -1.0); // fwd flow takes from source
                    add_entry!(col_rev, row, 1.0); // rev flow adds to source
                }
            }
        }

        // Deficit and excess contributions
        for b_idx in 0..n_b {
            for blk in 0..n_blks {
                let row = row_load_balance_start + b_idx * n_blks + blk;
                let col_def = col_deficit_start + b_idx * n_blks + blk;
                let col_exc = col_excess_start + b_idx * n_blks + blk;
                add_entry!(col_def, row, 1.0); // deficit adds supply
                add_entry!(col_exc, row, -1.0); // excess reduces net supply
            }
        }

        for entries in &mut col_entries {
            entries.sort_unstable_by_key(|&(row, _)| row);
        }

        let total_nz: usize = col_entries.iter().map(Vec::len).sum();
        let mut col_starts = Vec::with_capacity(num_cols + 1);
        let mut row_indices = Vec::with_capacity(total_nz);
        let mut values = Vec::with_capacity(total_nz);

        let mut offset: i32 = 0;
        for entries in &col_entries {
            col_starts.push(offset);
            for &(row, val) in entries {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                row_indices.push(row as i32);
                values.push(val);
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                offset += entries.len() as i32;
            }
        }
        col_starts.push(offset);

        let n_transfer = n_hydros * max_par_order;

        let template = StageTemplate {
            num_cols,
            num_rows,
            num_nz: total_nz,
            col_starts,
            row_indices,
            values,
            col_lower,
            col_upper,
            objective,
            row_lower,
            row_upper,
            n_state,
            n_transfer,
            n_dual_relevant,
            n_hydro: n_h,
            max_par_order: lag_order,
        };

        templates.push(template);
        base_rows.push(stage_base_row);
    }

    StageTemplates {
        templates,
        base_rows,
    }
}

#[cfg(test)]
mod tests {
    use super::{PatchBuffer, ar_dynamics_row_offset};
    use crate::indexer::StageIndexer;

    /// Convenience: make an indexer without repeating N/L everywhere.
    fn idx(n: usize, l: usize) -> StageIndexer {
        StageIndexer::new(n, l)
    }

    #[test]
    fn new_3_2_sizes_to_12() {
        // Spec acceptance criterion: PatchBuffer::new(3, 2) → 12 = 3*(2+2)
        let buf = PatchBuffer::new(3, 2);
        assert_eq!(buf.indices.len(), 12);
        assert_eq!(buf.lower.len(), 12);
        assert_eq!(buf.upper.len(), 12);
    }

    #[test]
    fn new_160_12_sizes_to_2240() {
        // Spec acceptance criterion: PatchBuffer::new(160, 12) → 2240 = 160*(2+12)
        let buf = PatchBuffer::new(160, 12);
        assert_eq!(buf.indices.len(), 2240);
        assert_eq!(buf.lower.len(), 2240);
        assert_eq!(buf.upper.len(), 2240);
    }

    #[test]
    fn new_zero_lags_sizes_to_2n() {
        // Edge case: L = 0 → N*(2+0) = 2*N patches
        let buf = PatchBuffer::new(5, 0);
        assert_eq!(buf.indices.len(), 10); // 5*(2+0)
    }

    #[test]
    fn new_zero_hydros_sizes_to_zero() {
        let buf = PatchBuffer::new(0, 0);
        assert_eq!(buf.indices.len(), 0);
    }

    #[test]
    fn forward_patch_count_matches_buffer_len() {
        let buf = PatchBuffer::new(3, 2);
        assert_eq!(buf.forward_patch_count(), buf.indices.len());
        assert_eq!(buf.forward_patch_count(), 12);
    }

    #[test]
    fn state_patch_count_is_n_times_one_plus_l() {
        // N*(1+L) = 3*(1+2) = 9 for the spec worked example
        let buf = PatchBuffer::new(3, 2);
        assert_eq!(buf.state_patch_count(), 9);
    }

    #[test]
    fn state_patch_count_zero_lags() {
        // L = 0 → N*(1+0) = N = 4
        let buf = PatchBuffer::new(4, 0);
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
        // Spec AC: first 3 patches correspond to storage fixing rows 0, 1, 2
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.indices[2], 2);
    }

    #[test]
    fn fill_forward_patches_category2_indices() {
        // Spec AC: patches 3-8 correspond to AR lag fixing rows 3..=8
        // Row index formula: N + ℓ·N + h
        // ℓ=0: 3+0=3, 3+1=4, 3+2=5
        // ℓ=1: 6+0=6, 6+1=7, 6+2=8
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.indices[3], 3); // N + 0·N + 0
        assert_eq!(buf.indices[4], 4); // N + 0·N + 1
        assert_eq!(buf.indices[5], 5); // N + 0·N + 2
        assert_eq!(buf.indices[6], 6); // N + 1·N + 0
        assert_eq!(buf.indices[7], 7); // N + 1·N + 1
        assert_eq!(buf.indices[8], 8); // N + 1·N + 2
    }

    #[test]
    fn fill_forward_patches_category3_indices() {
        // Spec AC: last 3 patches correspond to AR dynamics rows
        // base_row = 50 → rows 50, 51, 52
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.indices[9], 50); // ar_dynamics_row_offset(50, 0)
        assert_eq!(buf.indices[10], 51); // ar_dynamics_row_offset(50, 1)
        assert_eq!(buf.indices[11], 52); // ar_dynamics_row_offset(50, 2)
    }

    #[test]
    fn fill_forward_patches_category1_values() {
        // Category 1: lower == upper == state[h]
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

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
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

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
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

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
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

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
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

        // Active slice is [0, 9) = [0, N*(1+L))
        assert_eq!(buf.state_patch_count(), 9);
    }

    #[test]
    fn fill_state_patches_category1_correct() {
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

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
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

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
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

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
        let mut buf = PatchBuffer::new(n, 0);
        let state = [5.0, 7.0]; // n_state = 2*(1+0) = 2
        let noise = [0.5, 0.6];
        buf.fill_forward_patches(&idx(n, 0), &state, &noise, 10);

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
        let mut buf = PatchBuffer::new(n, 0);
        let state = [1.0, 2.0, 3.0]; // n_state = 3
        buf.fill_state_patches(&idx(n, 0), &state);

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
        // Spec acceptance criterion: 160*(2+12) = 2240
        let buf = PatchBuffer::new(160, 12);
        assert_eq!(buf.forward_patch_count(), 2240);
        assert_eq!(buf.indices.len(), 2240);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // fixture: values are small integers, no precision lost
    fn production_scale_fill_forward_patches_smoke() {
        let n = 160;
        let l = 12;
        let mut buf = PatchBuffer::new(n, l);
        let n_state = n * (1 + l);
        let state: Vec<f64> = (0..n_state).map(|i| i as f64).collect();
        let noise: Vec<f64> = (0..n).map(|h| h as f64 * 0.01).collect();
        buf.fill_forward_patches(&StageIndexer::new(n, l), &state, &noise, 500);

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
        let buf = PatchBuffer::new(3, 2);
        let cloned = buf.clone();
        assert_eq!(cloned.indices.len(), buf.indices.len());

        let s = format!("{buf:?}");
        assert!(s.contains("PatchBuffer"));
    }

    // -------------------------------------------------------------------------
    // build_stage_templates unit tests
    // -------------------------------------------------------------------------

    use super::build_stage_templates;
    use cobre_core::{
        Bus, BusStagePenalties, ContractStageBounds, DeficitSegment, EntityId, HydroStageBounds,
        HydroStagePenalties, LineStageBounds, LineStagePenalties, NcsStagePenalties,
        PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    };

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        }
    }

    /// Build a minimal one-bus, no-entity system with `n_stages` study stages.
    /// Used as the base fixture for structural tests.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn one_bus_system(n_stages: usize) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: false,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        // Resolved bounds and penalties are required for build_stage_templates to access
        // hydro/thermal/line bounds without panicking.
        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            0,
            0,
            0,
            0,
            0,
            n_st,
            default_hydro_bounds(),
            ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        );
        let penalties = ResolvedPenalties::new(
            0,
            1,
            0,
            0,
            n_st,
            default_hydro_penalties(),
            BusStagePenalties { excess_cost: 0.0 },
            LineStagePenalties { exchange_cost: 0.0 },
            NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("one_bus_system: valid")
    }

    /// Build a system with 1 hydro, 1 bus, no thermals, no lines, K=1 block.
    /// N=1, L=`lag_order`, so we get a concrete formula to check.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn one_hydro_system(n_stages: usize, lag_order: usize) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydro = Hydro {
            id: EntityId(2),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: lag_order > 0,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let ar_coefficients: Vec<f64> = (0..lag_order).map(|_| 0.5).collect();
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(2),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: ar_coefficients.clone(),
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        // Resolved bounds and penalties are required for build_stage_templates to access
        // hydro/thermal/line bounds without panicking.
        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            1,
            0,
            0,
            0,
            0,
            n_st,
            default_hydro_bounds(),
            ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        );
        let penalties = ResolvedPenalties::new(
            1,
            1,
            0,
            0,
            n_st,
            default_hydro_penalties(),
            BusStagePenalties { excess_cost: 0.0 },
            LineStagePenalties { exchange_cost: 0.0 },
            NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("one_hydro_system: valid")
    }

    #[test]
    fn empty_stages_returns_empty() {
        // A system with no study stages returns empty StageTemplates.
        let system = one_bus_system(0);
        let result = build_stage_templates(&system);
        assert!(result.templates.is_empty());
        assert!(result.base_rows.is_empty());
    }

    #[test]
    fn one_stage_one_template() {
        // One study stage produces exactly one template and one base_row.
        let system = one_bus_system(1);
        let result = build_stage_templates(&system);
        assert_eq!(result.templates.len(), 1);
        assert_eq!(result.base_rows.len(), 1);
    }

    #[test]
    fn num_cols_formula_no_hydro_no_thermal_no_line() {
        // N=0, T=0, Lines=0, B=1, K=1, L=0
        // num_cols = N*(2+L)+1 + N*K*2 + T*K + Lines*K*2 + B*K*2
        //          = 0*2+1 + 0 + 0 + 0 + 1*1*2 = 3
        // (0 state + 0 lags + 0 storage_in + 1 theta) + (0 turb + 0 spill) + (0 thermal) + (0 lines) + (1 def + 1 exc)
        let system = one_bus_system(1);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        // theta + deficit + excess = 1 + 1 + 1 = 3
        assert_eq!(t.num_cols, 3, "num_cols mismatch for no-entity system");
    }

    #[test]
    fn num_cols_formula_one_hydro_lag_zero() {
        // N=1, L=0, T=0, Lines=0, B=1, K=1
        // State cols: N*(2+L)+1 = 1*2+1 = 3  (v_out, v_in, theta)
        // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
        // Total: 7
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(t.num_cols, 7, "num_cols mismatch for N=1 L=0");
    }

    #[test]
    fn num_cols_formula_one_hydro_lag_two() {
        // N=1, L=2, T=0, Lines=0, B=1, K=1
        // State cols: N*(2+L)+1 = 1*4+1 = 5  (v_out, lag0, lag1, v_in, theta)
        // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
        // Total: 9
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(t.num_cols, 9, "num_cols mismatch for N=1 L=2");
    }

    #[test]
    fn num_rows_formula_no_hydro() {
        // N=0, B=1, K=1, L=0 → n_state = 0*(1+0) = 0
        // fixing rows: 0, water balance: 0, load balance: 1*1 = 1
        // num_rows = 0 + 0 + 1 = 1
        let system = one_bus_system(1);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(t.num_rows, 1, "num_rows mismatch for no-hydro system");
    }

    #[test]
    fn num_rows_formula_one_hydro_lag_zero() {
        // N=1, L=0, B=1, K=1
        // n_state = 1*(1+0) = 1
        // fixing rows = 1, water balance = 1, load balance = 1
        // num_rows = 3
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(t.num_rows, 3, "num_rows mismatch for N=1 L=0");
    }

    #[test]
    fn num_rows_formula_one_hydro_lag_two() {
        // N=1, L=2, B=1, K=1
        // n_state = 1*(1+2) = 3
        // fixing rows = 3, water balance = 1, load balance = 1
        // num_rows = 5
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(t.num_rows, 5, "num_rows mismatch for N=1 L=2");
    }

    #[test]
    fn n_state_matches_indexer() {
        // n_state must equal StageIndexer::new(N, L).n_state
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        let expected = StageIndexer::new(1, 2).n_state;
        assert_eq!(t.n_state, expected, "n_state must match StageIndexer");
    }

    #[test]
    fn n_transfer_is_n_times_lag_order() {
        // n_transfer = N*L = 1*2 = 2
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(t.n_transfer, 2, "n_transfer = N*L");
    }

    #[test]
    fn n_dual_relevant_equals_n_state_for_constant_productivity() {
        // For v0.1.0 with no FPHA, n_dual_relevant = n_state.
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(
            t.n_dual_relevant, t.n_state,
            "n_dual_relevant must equal n_state for constant-productivity hydros"
        );
    }

    #[test]
    fn base_row_is_n_dual_relevant() {
        // base_rows[s] = n_dual_relevant = n_state for the no-FPHA case.
        let system = one_hydro_system(2, 2);
        let result = build_stage_templates(&system);
        for (s, (&br, t)) in result.base_rows.iter().zip(&result.templates).enumerate() {
            assert_eq!(
                br, t.n_dual_relevant,
                "base_rows[{s}] must equal n_dual_relevant"
            );
        }
    }

    #[test]
    fn csc_col_starts_monotone_nondecreasing() {
        // CSC validity: col_starts must be monotone non-decreasing.
        let system = one_hydro_system(1, 1);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        for w in t.col_starts.windows(2) {
            assert!(w[0] <= w[1], "col_starts not monotone: {} > {}", w[0], w[1]);
        }
        // Length must be num_cols + 1
        assert_eq!(t.col_starts.len(), t.num_cols + 1);
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn csc_row_indices_in_range() {
        // All row_indices must be in [0, num_rows).
        let system = one_hydro_system(1, 1);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        for &r in &t.row_indices {
            assert!(
                r >= 0 && (r as usize) < t.num_rows,
                "row index {r} out of range [0, {})",
                t.num_rows
            );
        }
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn csc_nz_count_matches_col_starts() {
        // num_nz == col_starts[num_cols]
        let system = one_hydro_system(1, 1);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        assert_eq!(
            t.num_nz,
            *t.col_starts.last().unwrap() as usize,
            "num_nz must equal col_starts[num_cols]"
        );
        assert_eq!(
            t.row_indices.len(),
            t.num_nz,
            "row_indices.len() must equal num_nz"
        );
        assert_eq!(t.values.len(), t.num_nz, "values.len() must equal num_nz");
    }

    #[test]
    fn theta_column_has_unit_objective() {
        // The theta column (index = N*(2+L)) must have objective coefficient = 1.0.
        let lag_order = 2;
        let system = one_hydro_system(1, lag_order);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        let theta_col = StageIndexer::new(1, lag_order).theta;
        assert_eq!(
            t.objective[theta_col], 1.0,
            "theta column objective must be 1.0"
        );
    }

    #[test]
    fn spillage_objective_nonzero_for_nonzero_penalty() {
        // The spillage column should carry a non-zero objective when spillage_cost > 0.
        // Hydro has spillage_cost = 0.01, block duration = 744h.
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        // spillage col for h=0, blk=0: col_spillage_start + 0 = N*(2+L)+1 + N*K
        // With N=1, L=0, K=1: theta=2, decision_start=3, turbine_start=3, spill_start=4
        let spill_col = 4;
        assert!(
            t.objective[spill_col] > 0.0,
            "spillage objective must be > 0 when spillage_cost > 0"
        );
    }

    #[test]
    fn load_balance_rhs_matches_load_model_mean_mw() {
        // The load balance row RHS must equal the mean_mw from LoadModel (100.0 in fixture).
        let system = one_bus_system(1);
        let result = build_stage_templates(&system);
        let t = &result.templates[0];
        // No hydros → n_dual_relevant=0, water_balance_rows=0, load_balance at row 0, blk 0
        let load_row = 0;
        assert_eq!(
            t.row_lower[load_row], 100.0,
            "row_lower for load balance must be mean_mw"
        );
        assert_eq!(
            t.row_upper[load_row], 100.0,
            "row_upper for load balance must be mean_mw"
        );
    }

    #[test]
    fn multiple_stages_produce_same_count_templates_and_base_rows() {
        // A 3-stage system yields 3 templates and 3 base_rows.
        let system = one_hydro_system(3, 1);
        let result = build_stage_templates(&system);
        assert_eq!(result.templates.len(), 3);
        assert_eq!(result.base_rows.len(), 3);
    }

    #[test]
    fn stage_templates_clone_and_debug() {
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(&system);
        let cloned = result.clone();
        assert_eq!(cloned.templates.len(), result.templates.len());
        let s = format!("{result:?}");
        assert!(s.contains("StageTemplates"));
    }
}
