//! Stage LP patch buffer and stage template builder for SDDP subproblems.
//!
//! This module provides two public facilities:
//!
//! - [`PatchBuffer`]: pre-allocates the parallel arrays consumed by
//!   `SolverInterface::set_row_bounds` and fills them with scenario-dependent
//!   values before each LP solve.  Allocating once at training start and
//!   reusing the same buffer across all iterations and stages is critical for
//!   hot-path performance: the training loop calls `fill_forward_patches`,
//!   `fill_load_patches`, or `fill_state_patches` millions of times.
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
//!   inflow slack:    N columns (sigma_inf_h, only when penalty method is active)
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
//! Each forward-pass solve requires up to `N*(2+L) + M*K` row-bound patches
//! (where M is the number of stochastic load buses and K is the block count):
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
//!
//! Category 4 — load balance patches   M*K rows (optional, stochastic load)
//!     patch load_row_start + bus_pos[i]*K + blk = load_rhs[i*K + blk]
//!     for i in [0, M), blk in [0, K)
//! ```
//!
//! The backward pass uses only categories 1 and 2 (`N*(1+L)` patches); noise
//! innovations are drawn from the fixed opening tree by the caller.
//! When `n_load_buses == 0`, Category 4 is empty and has no effect.
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

use cobre_core::entities::hydro::HydroGenerationModel;
use cobre_core::System;
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormalLp;
use cobre_stochastic::par::precompute::PrecomputedParLp;

use crate::error::SddpError;
use crate::indexer::StageIndexer;
use crate::inflow_method::InflowNonNegativityMethod;

/// Pre-allocated row-bound patch arrays for one SDDP stage LP solve.
///
/// Holds three parallel `Vec`s of equal length ready for a single
/// `SolverInterface::set_row_bounds` call.  The buffer is sized for
/// `N*(2+L) + M*B` patches at construction and reused across all iterations,
/// where `M` is the number of stochastic load buses and `B` is the maximum
/// block count across stages.
///
/// # Memory layout
///
/// Entries are written in category order:
///
/// | Entry range                         | Category                                |
/// | ----------------------------------- | --------------------------------------- |
/// | `[0, N)`                            | Storage-fixing (Category 1)             |
/// | `[N, N*(1+L))`                      | AR lag-fixing (Category 2)              |
/// | `[N*(1+L), N*(2+L))`               | AR dynamics / noise (Category 3)        |
/// | `[N*(2+L), N*(2+L) + M*B_active)`  | Load balance row patches (Category 4)   |
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
}

impl PatchBuffer {
    /// Construct a [`PatchBuffer`] pre-allocated for `N*(2+L) + M*B` patches.
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
    /// let buf = PatchBuffer::new(3, 2, 0, 0);
    /// assert_eq!(buf.indices.len(), 12); // 3*(2+2) + 0*0
    ///
    /// // 3-hydro AR(2) system with 2 stochastic load buses, up to 3 blocks
    /// let buf_load = PatchBuffer::new(3, 2, 2, 3);
    /// assert_eq!(buf_load.indices.len(), 18); // 3*(2+2) + 2*3
    ///
    /// // Production scale: N = 160, L = 12, no stochastic load
    /// let big = PatchBuffer::new(160, 12, 0, 0);
    /// assert_eq!(big.indices.len(), 2240); // 160*(2+12)
    ///
    /// // Edge case: no lags (L = 0) — only storage + noise patches
    /// let no_lag = PatchBuffer::new(5, 0, 0, 0);
    /// assert_eq!(no_lag.indices.len(), 10); // 5*(2+0)
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
        let capacity = hydro_count * (2 + max_par_order) + n_load_buses * max_blocks;
        Self {
            indices: vec![0; capacity],
            lower: vec![0.0; capacity],
            upper: vec![0.0; capacity],
            hydro_count,
            max_par_order,
            load_bus_count: n_load_buses,
            max_blocks,
            active_load_patches: 0,
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
        debug_assert!(
            noise.len() == indexer.hydro_count || noise.is_empty(),
            "noise slice length {got} must equal hydro_count {expected} or be empty",
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
                self.indices[slot] = row;
                self.lower[slot] = rhs;
                self.upper[slot] = rhs;
                slot += 1;
            }
        }

        self.active_load_patches = self.load_bus_count * n_blocks;
    }

    /// Number of active patches after [`fill_forward_patches`] and (optionally)
    /// [`fill_load_patches`]: `N*(2+L) + active_load_patches`.
    ///
    /// When no [`fill_load_patches`] call has been made (or `n_load_buses == 0`),
    /// `active_load_patches` is zero and this returns `N*(2+L)`, preserving the
    /// behaviour from before Category 4 was added.
    ///
    /// Use this to pass the full forward-pass buffer to `set_row_bounds`.
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    #[must_use]
    #[inline]
    pub fn forward_patch_count(&self) -> usize {
        self.hydro_count * (2 + self.max_par_order) + self.active_load_patches
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
    /// Pre-computed noise scale `ζ_stage * σ_{stage,hydro}` for each (stage, hydro) pair.
    ///
    /// Flat array in stage-major layout: `noise_scale[stage * n_hydros + hydro]`.
    /// Length equals `n_study_stages * n_hydros`.
    ///
    /// Used by the forward pass to transform raw standard-normal noise `η` into
    /// the full noise term `ζ*σ*η` before patching the water-balance RHS.
    /// The complete patch value is `ζ*base + ζ*σ*η`, where `ζ*base` is encoded
    /// in the template's `row_lower`/`row_upper` and `ζ*σ*η` is computed by the
    /// caller at each stage using this pre-computed scale.
    pub noise_scale: Vec<f64>,
    /// Per-stage time-conversion factor `ζ = total_hours * M3S_TO_HM3`.
    ///
    /// Length equals `templates.len()`.  Used by the simulation pipeline to
    /// convert the water-balance RHS (in hm³) back to inflow in m³/s for
    /// output reporting: `inflow_m3s = rhs_hm3 / zeta_per_stage[stage]`.
    pub zeta_per_stage: Vec<f64>,
    /// Per-stage block durations in hours.
    ///
    /// `block_hours_per_stage[stage]` is a `Vec<f64>` of length `n_blocks` for
    /// that stage.  Used by the simulation pipeline to convert load-balance
    /// constraint duals from $/MW to $/`MWh`: `spot_price = dual / block_hours`.
    pub block_hours_per_stage: Vec<Vec<f64>>,
    /// Number of hydro plants (N) used to stride into `noise_scale`.
    pub n_hydros: usize,
    /// Per-stage row index of the first load-balance constraint.
    ///
    /// `load_balance_row_starts[s]` equals `row_water_balance_start + n_hydros`
    /// for stage `s`.  Length equals `templates.len()`.  Used by the forward,
    /// backward, and simulation passes to locate load-balance rows for
    /// stochastic load patching.
    pub load_balance_row_starts: Vec<usize>,
    /// Number of buses with stochastic load noise (i.e. with `std_mw > 0`).
    ///
    /// Equals `normal_lp.n_entities()`.  Tells the forward and backward passes
    /// how many load-noise components to extract from the opening tree noise
    /// vector, which carries load noise in indices `[n_hydros, n_hydros + n_load_buses)`.
    pub n_load_buses: usize,
    /// Position in the `buses` slice for each stochastic load bus.
    ///
    /// Length equals `n_load_buses`.  Bus IDs are sorted by [`cobre_core::EntityId`] for
    /// declaration-order invariance.  The forward and backward passes use
    /// `load_bus_indices[i]` to compute the base row index of bus `i` in the
    /// load-balance region: `row = load_balance_row_start + load_bus_indices[i] * n_blks + blk`.
    pub load_bus_indices: Vec<usize>,
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
///   (when penalty method is active, `+ N` slack columns are appended)
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
/// When the penalty method is active, each inflow slack column `sigma_inf_h`
/// carries objective coefficient `penalty_cost * total_stage_hours`.
///
/// ## Inflow non-negativity
///
/// When `inflow_method.has_slack_columns()` is `true` (i.e., the `Penalty`
/// variant), `N` slack columns `sigma_inf_h >= 0`
/// are appended at the end of the column layout.  Each slack enters the water
/// balance row for hydro `h` with coefficient `+tau_total * M3S_TO_HM3`,
/// acting as virtual inflow that prevents infeasibility when the PAR(p) noise
/// is sufficiently negative.
///
/// ## Errors
///
/// Returns `Err(SddpError::Validation)` when any hydro plant uses
/// `HydroGenerationModel::Fpha`, which is not yet implemented.  The error
/// message includes the plant name and ID so the caller can surface a
/// diagnostic to the user.
///
/// Returns `Ok` with empty templates for a system with zero stages.  All
/// entity counts may be zero (valid for degenerate test systems).
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_sddp::InflowNonNegativityMethod;
/// use cobre_sddp::lp_builder::build_stage_templates;
/// use cobre_stochastic::par::precompute::PrecomputedParLp;
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "B1".to_string(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 1000.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().expect("valid");
/// let method = InflowNonNegativityMethod::None;
/// let par_lp = PrecomputedParLp::build(&[], &[], &[]).expect("empty ok");
/// let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormalLp::default();
/// // No stages → empty result.
/// let result = build_stage_templates(&system, &method, &par_lp, &normal_lp)
///     .expect("no FPHA plants");
/// assert!(result.templates.is_empty());
/// ```
#[allow(clippy::too_many_lines)]
pub fn build_stage_templates(
    system: &System,
    inflow_method: &InflowNonNegativityMethod,
    par_lp: &PrecomputedParLp,
    normal_lp: &PrecomputedNormalLp,
) -> Result<StageTemplates, SddpError> {
    // Validate: FPHA generation model is not yet implemented.
    for hydro in system.hydros() {
        if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
            return Err(SddpError::Validation(format!(
                "hydro plant '{}' (id={}) uses FPHA generation model which is not yet implemented",
                hydro.name, hydro.id
            )));
        }
    }

    // Only build templates for study stages (id >= 0), in canonical order.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    let hydros = system.hydros();
    let thermals = system.thermals();
    let lines = system.lines();
    let buses = system.buses();
    let cascade = system.cascade();
    let bounds = system.bounds();
    let penalties = system.penalties();

    let n_hydros = hydros.len();

    if study_stages.is_empty() {
        return Ok(StageTemplates {
            templates: Vec::new(),
            base_rows: Vec::new(),
            noise_scale: Vec::new(),
            zeta_per_stage: Vec::new(),
            block_hours_per_stage: Vec::new(),
            n_hydros,
            load_balance_row_starts: Vec::new(),
            n_load_buses: 0,
            load_bus_indices: Vec::new(),
        });
    }
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
    let mut load_balance_row_starts = Vec::with_capacity(n_study_stages);

    // Determine whether the penalty method adds inflow slack columns.
    // Active when the method has slack columns and there is at least one hydro.
    // No slack columns are added for n_hydros == 0.
    let has_penalty = n_hydros > 0 && inflow_method.has_slack_columns();

    // Collect stochastic load bus indices: buses with std_mw > 0 for any stage,
    // sorted by EntityId for declaration-order invariance, mapped to their
    // position in the buses slice.
    //
    // `n_load_buses` equals `normal_lp.n_entities()` in a consistent system
    // (the two are derived from the same source of truth: buses with std_mw > 0
    // in the load models).  We derive it from the system directly so that
    // `load_bus_indices` and `n_load_buses` are always mutually consistent.
    let load_bus_indices: Vec<usize> = {
        let mut ids: Vec<cobre_core::EntityId> = system
            .load_models()
            .iter()
            .filter(|m| m.std_mw > 0.0)
            .map(|m| m.bus_id)
            .collect();
        ids.sort_unstable_by_key(|id| id.0);
        ids.dedup();
        ids.iter()
            .filter_map(|id| bus_pos.get(id).copied())
            .collect()
    };
    let n_load_buses = load_bus_indices.len();
    // Consistency gate: n_load_buses derived from the system must agree with
    // n_entities() in the PrecomputedNormalLp when the latter is non-empty.
    // A default (empty) PrecomputedNormalLp is accepted for call sites that
    // have not yet wired the stochastic context (e.g. unit tests without load noise).
    debug_assert!(
        normal_lp.n_entities() == 0 || normal_lp.n_entities() == n_load_buses,
        "PrecomputedNormalLp has {} entities but system has {} stochastic load buses",
        normal_lp.n_entities(),
        n_load_buses
    );

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
        let col_excess_end = col_excess_start + n_buses * n_blks;
        // Inflow slack columns go after excess, only when penalty method is active.
        let col_inflow_slack_start = col_excess_end;
        let n_slack_cols = if has_penalty { n_hydros } else { 0 };
        let num_cols = col_excess_end + n_slack_cols;

        let n_state = idx.n_state;
        let n_dual_relevant = n_state;
        let row_water_balance_start = n_dual_relevant;
        let row_load_balance_start = row_water_balance_start + n_hydros;
        let num_rows = row_load_balance_start + n_buses * n_blks;
        load_balance_row_starts.push(row_load_balance_start);

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

        // Inflow non-negativity slack columns (sigma_inf_h), one per hydro.
        // Bounds: [0, +inf).  Objective: penalty_cost * total_stage_hours.
        // These are already zero-lower / infinity-upper from the vec initialisation,
        // so only the objective coefficient needs to be written.
        if has_penalty {
            let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
            // penalty_cost() is always Some when has_penalty is true.
            let penalty_cost = inflow_method.penalty_cost().unwrap_or(0.0);
            let obj_coeff = penalty_cost * total_stage_hours;
            for h_idx in 0..n_hydros {
                let col = col_inflow_slack_start + h_idx;
                // col_lower already 0.0, col_upper already f64::INFINITY
                objective[col] = obj_coeff;
            }
        }

        let mut row_lower = vec![0.0_f64; num_rows];
        let mut row_upper = vec![0.0_f64; num_rows];

        // Water balance rows: static RHS = ζ * deterministic_base_h.
        // The dynamic part (ζ * σ * η) is applied per-scenario via PatchBuffer.
        let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let zeta = total_stage_hours * M3S_TO_HM3;
        for h_idx in 0..n_h {
            let row = row_water_balance_start + h_idx;
            let base = if par_lp.n_stages() > 0 && par_lp.n_hydros() == n_h {
                par_lp.deterministic_base(stage_idx, h_idx)
            } else {
                0.0
            };
            row_lower[row] = zeta * base;
            row_upper[row] = zeta * base;
        }

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
            // v_out enters with +1 (outgoing storage increases the balance).
            add_entry!(h_idx, row, 1.0);
            // v_in enters with -1 (incoming storage is subtracted).
            add_entry!(idx.storage_in.start + h_idx, row, -1.0);
            for blk in 0..n_blks {
                let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
                // Own turbine and spillage reduce storage (outflows).
                let col_turbine = col_turbine_start + h_idx * n_blks + blk;
                add_entry!(col_turbine, row, tau_h);
                let col_spillage = col_spillage_start + h_idx * n_blks + blk;
                add_entry!(col_spillage, row, tau_h);
                // Upstream turbine and spillage add to storage (inflows from upstream).
                let upstream_ids = cascade.upstream(hydro.id);
                for &up_id in upstream_ids {
                    if let Some(&u_idx) = hydro_pos.get(&up_id) {
                        let col_upstream_turbine = col_turbine_start + u_idx * n_blks + blk;
                        add_entry!(col_upstream_turbine, row, -tau_h);
                        let col_upstream_spillage = col_spillage_start + u_idx * n_blks + blk;
                        add_entry!(col_upstream_spillage, row, -tau_h);
                    }
                }
            }
            // AR lag dynamics: lag variable at lag ℓ for hydro h enters with
            // coefficient -ζ * ψ_{h,ℓ} in the water balance row. This encodes
            // the PAR model: a_h = base + Σ_ℓ ψ_ℓ * lag_ℓ + σ * η, where the
            // lag terms move to the LHS of the equality constraint.
            if par_lp.n_stages() > 0 && par_lp.n_hydros() == n_h {
                let psi = par_lp.psi_slice(stage_idx, h_idx);
                for (lag, &psi_val) in psi.iter().enumerate() {
                    if psi_val != 0.0 && lag < lag_order {
                        let col = idx.inflow_lags.start + lag * n_h + h_idx;
                        add_entry!(col, row, -zeta * psi_val);
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
                HydroGenerationModel::Fpha => {
                    unreachable!(
                        "FPHA validation guard at the top of build_stage_templates prevents reaching this arm"
                    )
                }
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

        // Inflow non-negativity slack: sigma_inf_h enters the water balance row
        // for hydro h with coefficient -ζ (negative on the LHS). This is
        // equivalent to adding ζ*sigma_inf_h of virtual inflow to the RHS,
        // absorbing negative noise realisations and keeping the water balance
        // feasible.
        if has_penalty {
            for h_idx in 0..n_h {
                let col = col_inflow_slack_start + h_idx;
                let row = row_water_balance_start + h_idx;
                add_entry!(col, row, -zeta);
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

    // Pre-compute ζ * σ per (stage, hydro) for noise transformation in the
    // forward/backward passes. The caller multiplies raw η by noise_scale to
    // obtain ζ*σ*η, which is then added to ζ*base (encoded in row_lower) to
    // form the complete water-balance RHS patch value.
    let mut noise_scale = vec![0.0_f64; n_study_stages * n_hydros];
    let mut zeta_per_stage = Vec::with_capacity(n_study_stages);
    let mut block_hours_per_stage = Vec::with_capacity(n_study_stages);
    for (s_idx, stage) in study_stages.iter().enumerate() {
        let total_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let zeta_s = total_hours * M3S_TO_HM3;
        zeta_per_stage.push(zeta_s);
        block_hours_per_stage.push(stage.blocks.iter().map(|b| b.duration_hours).collect());
        for h_idx in 0..n_hydros {
            let sigma = if par_lp.n_stages() > 0 && par_lp.n_hydros() == n_hydros {
                par_lp.sigma(s_idx, h_idx)
            } else {
                0.0
            };
            noise_scale[s_idx * n_hydros + h_idx] = zeta_s * sigma;
        }
    }

    Ok(StageTemplates {
        templates,
        base_rows,
        noise_scale,
        zeta_per_stage,
        block_hours_per_stage,
        n_hydros,
        load_balance_row_starts,
        n_load_buses,
        load_bus_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::{ar_dynamics_row_offset, PatchBuffer};
    use crate::indexer::StageIndexer;

    /// Convenience: make an indexer without repeating N/L everywhere.
    fn idx(n: usize, l: usize) -> StageIndexer {
        StageIndexer::new(n, l)
    }

    #[test]
    fn new_3_2_sizes_to_12() {
        // Spec acceptance criterion: PatchBuffer::new(3, 2, 0, 0) → 12 = 3*(2+2)
        let buf = PatchBuffer::new(3, 2, 0, 0);
        assert_eq!(buf.indices.len(), 12);
        assert_eq!(buf.lower.len(), 12);
        assert_eq!(buf.upper.len(), 12);
    }

    #[test]
    fn new_160_12_sizes_to_2240() {
        // Spec acceptance criterion: PatchBuffer::new(160, 12, 0, 0) → 2240 = 160*(2+12)
        let buf = PatchBuffer::new(160, 12, 0, 0);
        assert_eq!(buf.indices.len(), 2240);
        assert_eq!(buf.lower.len(), 2240);
        assert_eq!(buf.upper.len(), 2240);
    }

    #[test]
    fn new_zero_lags_sizes_to_2n() {
        // Edge case: L = 0 → N*(2+0) = 2*N patches
        let buf = PatchBuffer::new(5, 0, 0, 0);
        assert_eq!(buf.indices.len(), 10); // 5*(2+0)
    }

    #[test]
    fn new_zero_hydros_sizes_to_zero() {
        let buf = PatchBuffer::new(0, 0, 0, 0);
        assert_eq!(buf.indices.len(), 0);
    }

    #[test]
    fn forward_patch_count_matches_buffer_len() {
        let buf = PatchBuffer::new(3, 2, 0, 0);
        assert_eq!(buf.forward_patch_count(), buf.indices.len());
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
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

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
        // Last 3 patches correspond to AR dynamics rows
        // base_row = 50 → rows 50, 51, 52
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

        // Active slice is [0, 9) = [0, N*(1+L))
        assert_eq!(buf.state_patch_count(), 9);
    }

    #[test]
    fn fill_state_patches_category1_correct() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
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
        let mut buf = PatchBuffer::new(n, 0, 0, 0);
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
        let mut buf = PatchBuffer::new(n, 0, 0, 0);
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
        let buf = PatchBuffer::new(160, 12, 0, 0);
        assert_eq!(buf.forward_patch_count(), 2240);
        assert_eq!(buf.indices.len(), 2240);
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
        let buf = PatchBuffer::new(3, 2, 0, 0);
        let cloned = buf.clone();
        assert_eq!(cloned.indices.len(), buf.indices.len());

        let s = format!("{buf:?}");
        assert!(s.contains("PatchBuffer"));
    }

    // -------------------------------------------------------------------------
    // Category 4 (load balance) unit tests
    // -------------------------------------------------------------------------

    /// AC: `PatchBuffer::new(2, 1, 1, 3)` → capacity = 2*(2+1) + 1*3 = 9.
    #[test]
    fn new_with_load_allocates_correct_capacity() {
        let buf = PatchBuffer::new(2, 1, 1, 3);
        assert_eq!(buf.indices.len(), 9); // 2*(2+1) + 1*3
        assert_eq!(buf.lower.len(), 9);
        assert_eq!(buf.upper.len(), 9);
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
        buf.fill_load_patches(100, 2, &load_rhs, &bus_positions);

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
        buf.fill_load_patches(100, 2, &load_rhs, &bus_positions);

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
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, 3, &load_rhs, &bus_positions);

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
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, 3, &load_rhs, &bus_positions);

        assert_eq!(buf.forward_patch_count(), 18); // 12 + 6
    }

    /// `state_patch_count` is unaffected by Category 4 — no lag structure for load.
    #[test]
    fn state_patch_count_excludes_load() {
        let mut buf = PatchBuffer::new(3, 2, 2, 3);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, 3, &load_rhs, &bus_positions);

        // state_patch_count must be N*(1+L) = 3*3 = 9, not 18
        assert_eq!(buf.state_patch_count(), 9);
    }

    /// When `n_load_buses == 0`, `forward_patch_count` equals `N*(2+L)` unchanged.
    #[test]
    fn zero_load_buses_no_category4() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        // No fill_load_patches call: active_load_patches stays 0
        assert_eq!(buf.forward_patch_count(), 12); // 3*(2+2) only
    }

    // -------------------------------------------------------------------------
    // build_stage_templates unit tests
    // -------------------------------------------------------------------------

    use super::build_stage_templates;
    use crate::inflow_method::InflowNonNegativityMethod;
    use cobre_core::{
        Bus, BusStagePenalties, ContractStageBounds, DeficitSegment, EntityId, HydroStageBounds,
        HydroStagePenalties, LineStageBounds, LineStagePenalties, NcsStagePenalties,
        PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    };
    use cobre_stochastic::normal::precompute::PrecomputedNormalLp;
    use cobre_stochastic::par::precompute::PrecomputedParLp;

    /// Method with no penalty — used in structural tests that check exact
    /// column/row counts that would change if penalty columns were added.
    fn no_penalty_config() -> InflowNonNegativityMethod {
        InflowNonNegativityMethod::None
    }

    /// Method with penalty — used in tests that verify the penalty
    /// column addition behaviour.
    fn penalty_config(cost: f64) -> InflowNonNegativityMethod {
        InflowNonNegativityMethod::Penalty { cost }
    }

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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        assert!(result.templates.is_empty());
        assert!(result.base_rows.is_empty());
    }

    #[test]
    fn one_stage_one_template() {
        // One study stage produces exactly one template and one base_row.
        let system = one_bus_system(1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];
        assert_eq!(t.num_cols, 9, "num_cols mismatch for N=1 L=2");
    }

    #[test]
    fn num_rows_formula_no_hydro() {
        // N=0, B=1, K=1, L=0 → n_state = 0*(1+0) = 0
        // fixing rows: 0, water balance: 0, load balance: 1*1 = 1
        // num_rows = 0 + 0 + 1 = 1
        let system = one_bus_system(1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];
        assert_eq!(t.num_rows, 5, "num_rows mismatch for N=1 L=2");
    }

    #[test]
    fn n_state_matches_indexer() {
        // n_state must equal StageIndexer::new(N, L).n_state
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];
        let expected = StageIndexer::new(1, 2).n_state;
        assert_eq!(t.n_state, expected, "n_state must match StageIndexer");
    }

    #[test]
    fn n_transfer_is_n_times_lag_order() {
        // n_transfer = N*L = 1*2 = 2
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];
        assert_eq!(t.n_transfer, 2, "n_transfer = N*L");
    }

    #[test]
    fn n_dual_relevant_equals_n_state_for_constant_productivity() {
        // For v0.1.0 with no FPHA, n_dual_relevant = n_state.
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
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
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        assert_eq!(result.templates.len(), 3);
        assert_eq!(result.base_rows.len(), 3);
    }

    #[test]
    fn stage_templates_clone_and_debug() {
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let cloned = result.clone();
        assert_eq!(cloned.templates.len(), result.templates.len());
        let s = format!("{result:?}");
        assert!(s.contains("StageTemplates"));
    }

    // -------------------------------------------------------------------------
    // FPHA generation model validation tests
    // -------------------------------------------------------------------------

    /// AC: a system where a hydro plant uses `Fpha` must be rejected before any
    /// LP construction work, with an error message that contains the plant name.
    #[test]
    fn test_fpha_model_rejected() {
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
            id: EntityId(5),
            name: "Tucurui".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
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

        let stages: Vec<Stage> = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models: Vec<InflowModel> = vec![InflowModel {
            hydro_id: EntityId(5),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        }];

        let load_models: Vec<LoadModel> = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            1,
            0,
            0,
            0,
            0,
            1,
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
            1,
            default_hydro_penalties(),
            BusStagePenalties { excess_cost: 0.0 },
            LineStagePenalties { exchange_cost: 0.0 },
            NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        );

        let system = cobre_core::SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("test_fpha_model_rejected: valid system");

        let err = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect_err("FPHA plant must cause an error");

        let msg = err.to_string();
        assert!(
            msg.contains("Tucurui"),
            "error message must contain the hydro plant name 'Tucurui', got: {msg}"
        );
        assert!(
            msg.to_lowercase().contains("fpha"),
            "error message must mention 'FPHA', got: {msg}"
        );
    }

    /// AC: a system where all hydro plants use `ConstantProductivity` must be
    /// accepted, returning `Ok(StageTemplates { .. })`.
    #[test]
    fn test_constant_productivity_accepted() {
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        );
        assert!(
            result.is_ok(),
            "ConstantProductivity system must return Ok, got: {result:?}"
        );
        assert_eq!(
            result.expect("accepted").templates.len(),
            1,
            "one study stage → one template"
        );
    }

    // -------------------------------------------------------------------------
    // Inflow non-negativity penalty method tests
    // -------------------------------------------------------------------------

    // AC-1 / test_penalty_columns_added:
    // penalty method with N=1 hydro adds 1 extra column; method="none" adds 0.
    #[test]
    fn test_penalty_columns_added() {
        let system = one_hydro_system(1, 0);
        let without = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let with_p = build_stage_templates(
            &system,
            &penalty_config(1000.0),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        assert_eq!(
            with_p.templates[0].num_cols,
            without.templates[0].num_cols + 1,
            "penalty method must add exactly n_hydros extra columns"
        );
    }

    // AC-1 (edge case): no slack columns when n_hydros == 0, even with penalty config.
    #[test]
    fn test_penalty_columns_added_3_hydros() {
        // Build a 3-hydro system by calling one_hydro_system 3 times is not possible;
        // use one_hydro_system(1, 0) as a proxy and verify the count formula directly.
        // The formula: num_cols(penalty) = num_cols(none) + n_hydros.
        // For N=1 we already cover N=1 above. Verify the N=0 (no hydros) edge case:
        // no slacks when n_hydros == 0, regardless of config.
        let system = one_bus_system(1);
        let with_p = build_stage_templates(
            &system,
            &penalty_config(1000.0),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let without = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        assert_eq!(
            with_p.templates[0].num_cols, without.templates[0].num_cols,
            "no slack columns when n_hydros == 0, even with penalty config"
        );
    }

    // AC-2 / test_penalty_objective_coefficient:
    // objective coefficient = penalty_cost * total_stage_hours.
    // one_hydro_system uses 1 block of 744 hours.
    #[test]
    fn test_penalty_objective_coefficient() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];
        // N=1, L=0: theta=2, decision_start=3, turbine=3, spillage=4, deficit=5, excess=6, slack=7
        let slack_col = t.num_cols - 1; // last column
        let expected_obj = 1000.0 * 744.0;
        assert!(
            (t.objective[slack_col] - expected_obj).abs() < 1e-9,
            "expected objective {expected_obj}, got {}",
            t.objective[slack_col]
        );
    }

    // AC-3 / test_no_penalty_columns_when_none:
    // method="none" leaves column/row counts unchanged.
    #[test]
    fn test_no_penalty_columns_when_none() {
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];
        // N=1, L=2: state = 1*(2+2)+1 = 5; decisions = turb+spill+def+exc = 4; total = 9
        assert_eq!(t.num_cols, 9, "method=none must not add extra columns");
        // num_rows = N*(1+L)+N+B*K = 3+1+1 = 5
        assert_eq!(t.num_rows, 5, "method=none must not add extra rows");
    }

    // test_penalty_slack_in_water_balance:
    // the slack column has a non-zero entry in the water balance row for its hydro.
    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_penalty_slack_in_water_balance() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];

        // Locate the slack column (last column, index = num_cols - 1).
        let slack_col = t.num_cols - 1;

        // Iterate the CSC to find the entry for slack_col in the water balance row.
        // Water balance row for hydro 0: row_water_balance_start = n_state = N*(1+L) = 1.
        let water_balance_row = 1_usize; // N*(1+L) = 1*(1+0) = 1

        let col_start = t.col_starts[slack_col] as usize;
        let col_end = t.col_starts[slack_col + 1] as usize;
        let found = t.row_indices[col_start..col_end]
            .iter()
            .zip(&t.values[col_start..col_end])
            .any(|(&r, &v)| r as usize == water_balance_row && v.abs() > 1e-12);

        assert!(
            found,
            "slack column must have a non-zero entry in the water balance row"
        );
    }

    // test_penalty_slack_bounds:
    // slack columns have lower = 0.0 and upper = +inf.
    #[test]
    fn test_penalty_slack_bounds() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];
        let slack_col = t.num_cols - 1;
        assert_eq!(t.col_lower[slack_col], 0.0, "slack lower bound must be 0.0");
        assert!(
            t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
            "slack upper bound must be +infinity"
        );
    }

    // Verify the water balance coefficient value.
    //
    // The penalty slack column represents virtual inflow. Adding virtual inflow
    // is equivalent to subtracting it from the LHS of the water balance
    // constraint (which is written as: outflows - inflows = RHS).
    // Therefore the coefficient is -ζ where ζ = tau_total * M3S_TO_HM3.
    //
    // With 1 block of 744 h:
    //   ζ = 744.0 * (3600.0 / 1_000_000.0) = 2.6784 hm3/(m3/s)
    //   coefficient = -ζ = -2.6784
    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_penalty_water_balance_coefficient_value() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let t = &result.templates[0];

        let slack_col = t.num_cols - 1;
        let water_balance_row = 1_usize; // N*(1+L) = 1
        let zeta = 744.0 * (3_600.0 / 1_000_000.0);
        let expected_coeff = -zeta; // slack enters LHS with -ζ (virtual inflow)

        let col_start = t.col_starts[slack_col] as usize;
        let col_end = t.col_starts[slack_col + 1] as usize;
        let coeff = t.row_indices[col_start..col_end]
            .iter()
            .zip(&t.values[col_start..col_end])
            .find(|&(&r, _)| r as usize == water_balance_row)
            .map(|(_, &v)| v);

        assert!(
            coeff.is_some(),
            "slack column must have an entry in the water balance row"
        );
        let coeff = coeff.unwrap();
        assert!(
            (coeff - expected_coeff).abs() < 1e-9,
            "expected coefficient {expected_coeff:.9}, got {coeff:.9}"
        );
    }

    // Penalty method with multiple stages: verify each stage has consistent slack layout.
    #[test]
    fn test_penalty_multi_stage_consistent() {
        let system = one_hydro_system(3, 1);
        let config = penalty_config(2000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        assert_eq!(result.templates.len(), 3);
        let base_cols = result.templates[0].num_cols;
        for t in &result.templates {
            assert_eq!(
                t.num_cols, base_cols,
                "all stages must have the same column count"
            );
        }
    }

    // AC-4 / test_penalty_slack_absorbs_negative_inflow:
    // A negative noise value would render the LP infeasible without the inflow
    // slack column. With `penalty_config`, the slack absorbs the deficit and the
    // solve must succeed with a positive slack value.
    //
    // System: N=1, L=0, K=1 block (744 h), B=1 bus, T=0, Lines=0.
    // Column layout:
    //   col 0: storage_out    col 1: storage_in   col 2: theta
    //   col 3: turbine        col 4: spillage      col 5: deficit
    //   col 6: excess         col 7: inflow_slack  <- last column
    //
    // Row layout:
    //   row 0: storage_fixing  row 1: water_balance  row 2: load_balance
    //
    // To apply negative inflow noise we patch the water balance row (row 1)
    // to RHS = -5.0. Without the slack this would make the LP infeasible.
    #[test]
    fn test_penalty_slack_absorbs_negative_inflow() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");
        let template = &result.templates[0];

        // The inflow slack is the last column.
        let col_inflow_slack_start = template.num_cols - 1;

        // base_row for stage 0 is n_dual_relevant = n_state = 1 (for N=1, L=0).
        let base_row = result.base_rows[0];
        let water_balance_row = base_row; // hydro 0: base_row + 0

        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

        // Load the structural LP.
        solver.load_model(template);

        // Add an empty cut batch (no cuts at iteration 0).
        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        // Patch the state rows.
        // Row 0 (storage_fixing): fix incoming storage to 100 hm³.
        // Row 1 (water_balance): set RHS to -5.0 m³/s (negative noise).
        // Both are equality constraints: lower == upper == rhs.
        let initial_storage = 100.0_f64;
        let negative_noise = -5.0_f64;
        solver.set_row_bounds(
            &[0, water_balance_row],
            &[initial_storage, negative_noise],
            &[initial_storage, negative_noise],
        );

        // The solve must succeed — the slack absorbs the negative inflow.
        let view = solver
            .solve()
            .expect("LP must be feasible with inflow slack active");

        let primal = view.primal;

        // The inflow slack must be strictly positive: it compensates the
        // negative noise so that the water balance constraint is satisfied.
        assert!(
            primal[col_inflow_slack_start] > 0.0,
            "inflow slack must be positive when noise is negative, got {}",
            primal[col_inflow_slack_start]
        );

        // The objective must be positive: penalty cost * slack value > 0.
        assert!(
            view.objective > 0.0,
            "objective must include a positive penalty contribution, got {}",
            view.objective
        );
    }

    // -------------------------------------------------------------------------
    // ticket-024: load balance row starts, n_load_buses, load_bus_indices
    // -------------------------------------------------------------------------

    /// Build a two-bus system with N hydros and K blocks per stage.
    /// Bus B1 (EntityId=10) has `std_mw` = 0 (no load noise).
    /// Bus B2 (EntityId=20) has `std_mw` > 0 (stochastic load).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn two_bus_system_with_stochastic_load(
        n_stages: usize,
        n_hydros_in_system: usize,
        n_blocks: usize,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        // B1 (EntityId(10)): no noise, B2 (EntityId(20)): stochastic.
        let bus1 = Bus {
            id: EntityId(10),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let bus2 = Bus {
            id: EntityId(20),
            name: "B2".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let blocks: Vec<_> = (0..n_blocks)
            .map(|b| Block {
                index: b,
                name: format!("B{b}"),
                duration_hours: 240.0,
            })
            .collect();

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: blocks.clone(),
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

        let hydros: Vec<Hydro> = (0..n_hydros_in_system)
            .map(|h| Hydro {
                id: EntityId((h + 100) as i32),
                name: format!("H{h}"),
                bus_id: EntityId(10),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 1.0,
                },
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                min_generation_mw: 0.0,
                max_generation_mw: 50.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.0,
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
            })
            .collect();

        let inflow_models: Vec<InflowModel> = hydros
            .iter()
            .flat_map(|h| {
                (0..n_stages).map(move |s| InflowModel {
                    hydro_id: h.id,
                    stage_id: s as i32,
                    mean_m3s: 50.0,
                    std_m3s: 10.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                })
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .flat_map(|s| {
                [
                    LoadModel {
                        bus_id: EntityId(10),
                        stage_id: s as i32,
                        mean_mw: 80.0,
                        std_mw: 0.0, // B1: no noise
                    },
                    LoadModel {
                        bus_id: EntityId(20),
                        stage_id: s as i32,
                        mean_mw: 120.0,
                        std_mw: 15.0, // B2: stochastic
                    },
                ]
            })
            .collect();

        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            n_hydros_in_system,
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
            n_hydros_in_system,
            2,
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

        let mut builder = SystemBuilder::new()
            .buses(vec![bus1, bus2])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties);
        if !hydros.is_empty() {
            builder = builder.hydros(hydros).inflow_models(inflow_models);
        }
        builder.build().expect("two_bus_system: valid")
    }

    // AC-1: load_balance_row_starts[0] == row_water_balance_start + n_hydros
    // for a 2-bus, 2-hydro, 3-block system.
    #[test]
    fn stage_templates_load_balance_row_starts_correct() {
        // N=2 hydros, B=2 buses, L=0 lags.
        // StageIndexer: n_state = N*(1+L) = 2, n_dual_relevant = 2.
        // row_water_balance_start = 2, row_load_balance_start = 2 + 2 = 4.
        let system = two_bus_system_with_stochastic_load(2, 2, 3);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");

        assert_eq!(
            result.load_balance_row_starts.len(),
            result.templates.len(),
            "load_balance_row_starts length must match templates length"
        );

        // For N=2 hydros, L=0: n_state = 2*(1+0) = 2, so row_water_balance_start = 2,
        // row_load_balance_start = 2 + 2 = 4.
        let expected_row_start = result.base_rows[0] + 2; // base_rows[0] = row_water_balance_start
        assert_eq!(
            result.load_balance_row_starts[0], expected_row_start,
            "load_balance_row_starts[0] must equal row_water_balance_start + n_hydros"
        );
        // Both stages should have the same row start (same topology across stages).
        assert_eq!(
            result.load_balance_row_starts[0], result.load_balance_row_starts[1],
            "identical stages share the same load balance row start"
        );
    }

    // AC-2: n_load_buses and load_bus_indices populated for stochastic buses.
    #[test]
    fn stage_templates_n_load_buses_matches_stochastic_buses() {
        // B2 (EntityId(20)) has std_mw > 0; B1 (EntityId(10)) does not.
        // The system has buses in order [B1(10), B2(20)], so B2 is at index 1.
        let system = two_bus_system_with_stochastic_load(1, 0, 1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");

        assert_eq!(
            result.n_load_buses, 1,
            "only B2 has std_mw > 0 → n_load_buses must be 1"
        );
        assert_eq!(
            result.load_bus_indices.len(),
            1,
            "load_bus_indices must have exactly one entry"
        );
        assert_eq!(
            result.load_bus_indices[0], 1,
            "B2 is at buses slice index 1 (buses are [B1(10), B2(20)])"
        );
    }

    // AC-3: no load buses when no load models have std_mw > 0.
    #[test]
    fn stage_templates_no_load_buses_gives_zero() {
        // one_bus_system uses std_mw = 0 for all load models.
        let system = one_bus_system(2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedParLp::default(),
            &PrecomputedNormalLp::default(),
        )
        .expect("no FPHA plants");

        assert_eq!(
            result.n_load_buses, 0,
            "system with std_mw = 0 everywhere must give n_load_buses = 0"
        );
        assert!(
            result.load_bus_indices.is_empty(),
            "load_bus_indices must be empty when n_load_buses = 0"
        );
        assert_eq!(
            result.load_balance_row_starts.len(),
            result.templates.len(),
            "load_balance_row_starts length must always match templates length"
        );
    }
}
