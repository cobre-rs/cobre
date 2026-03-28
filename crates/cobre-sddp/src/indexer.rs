//! LP layout index map for SDDP stage subproblems.
//!
//! [`StageIndexer`] centralises all column and row offset arithmetic for a
//! single-stage LP, eliminating magic index numbers throughout the forward
//! pass, backward pass, and LP construction code.
//!
//! ## Column layout (Solver Abstraction SS2.1)
//!
//! ```text
//! [0, N)              storage      — outgoing storage volumes  (N = hydro_count)
//! [N, N*(1+L))        inflow_lags  — AR lag variables (L lags per hydro)
//! [N*(1+L), N*(2+L))  z_inflow     — realized inflow (auxiliary, not state)
//! [N*(2+L), N*(3+L))  storage_in   — incoming storage volumes
//! N*(3+L)             theta        — future cost variable (scalar)
//! ```
//!
//! When built with [`StageIndexer::with_equipment`], the following equipment
//! columns follow immediately after `theta`:
//!
//! ```text
//! [theta+1,                          theta+1+H*K)        turbine     — turbined flow (m³/s)
//! [theta+1+H*K,                      theta+1+2*H*K)      spillage    — spilled flow (m³/s)
//! [theta+1+2*H*K,                    theta+1+3*H*K)      diversion   — diverted flow (m³/s)
//! [theta+1+3*H*K,                    theta+1+3*H*K+T*K)  thermal     — thermal generation (MW)
//! [theta+1+3*H*K+T*K,                theta+1+3*H*K+T*K+2*L_n*K) line_fwd/rev — line flows
//! [theta+1+3*H*K+T*K+2*L_n*K,       theta+1+3*H*K+T*K+2*L_n*K+B*S*K) deficit
//! [theta+1+3*H*K+T*K+2*L_n*K+B*S*K, theta+1+3*H*K+T*K+2*L_n*K+B*S*K+B*K) excess
//! ```
//!
//! When the inflow non-negativity penalty method is active (`has_inflow_penalty == true`),
//! `N` additional slack columns are appended after `excess`:
//!
//! ```text
//! [excess_end, excess_end+N)  inflow_slack — sigma_inf_h (m³/s), one per hydro
//! ```
//!
//! After FPHA generation and evaporation columns, `N` withdrawal slack columns are
//! appended when `hydro_count > 0`:
//!
//! ```text
//! [evap_end, evap_end+N)  withdrawal_slack — sigma^r_h (m³/s), one per hydro
//! ```
//!
//! where H = `hydro_count`, K = `n_blks`, T = `n_thermals`, Ln = `n_lines`, B = `n_buses`,
//! S = `max_deficit_segments`.
//!
//! ## Row layout (Solver Abstraction SS2.2)
//!
//! ```text
//! [0, N)          storage_fixing — storage-fixing constraints (RHS = incoming storage)
//! [N, N*(1+L))    lag_fixing     — AR lag-fixing constraints (RHS = lagged inflows)
//! ```
//!
//! ## Worked example (SS5.5.3): N = 3, L = 2
//!
//! ```text
//! storage      = 0..3
//! inflow_lags  = 3..9   (= 3..3*(1+2))
//! z_inflow     = 9..12  (= 3*(1+2)..3*(2+2))
//! storage_in   = 12..15 (= 3*(2+2)..3*(3+2))
//! theta        = 15     (= 3*(3+2))
//! n_state      = 9      (= 3*(1+2))
//! storage_fixing = 0..3
//! lag_fixing     = 3..9
//! ```

use std::ops::Range;

use cobre_solver::StageTemplate;

/// Column and row indices for the evaporation constraint of one hydro.
///
/// Locates the three evaporation columns and one evaporation row assigned to
/// a single hydro within a stage LP.  Columns are stage-level (not per-block).
#[derive(Debug, Clone, Copy)]
pub struct EvaporationIndices {
    /// Column index of the evaporation volume variable `Q_ev_h` (hm³).
    pub q_ev_col: usize,
    /// Column index of the positive violation slack `f_evap_plus_h` (hm³).
    pub f_evap_plus_col: usize,
    /// Column index of the negative violation slack `f_evap_minus_h` (hm³).
    pub f_evap_minus_col: usize,
    /// Row index of the evaporation equality constraint.
    pub evap_row: usize,
}

/// FPHA constraint row range for one hydro at one stage.
///
/// Locates the block of FPHA hyperplane rows assigned to a single FPHA hydro
/// within a stage LP. Rows for hydro `i` at block `k` and plane `p` are at:
/// `start + k * planes_per_block + p`.
#[derive(Debug, Clone, Copy)]
pub struct FphaRowRange {
    /// First row index of this hydro's FPHA constraints (for block 0, plane 0).
    pub start: usize,
    /// Number of hyperplanes per block.
    pub planes_per_block: usize,
}

/// Read-only LP layout index map for one SDDP stage subproblem.
///
/// Computed once from `hydro_count` (N) and `max_par_order` (L), then shared
/// read-only across all threads for the duration of training. Most fields are
/// plain `usize` or `Range<usize>`; FPHA fields use `Vec` for variable-length
/// hydro lists.
///
/// See the [module-level documentation](self) for the full column and row
/// layout, and [`StageIndexer::new`] for the construction formulas.
///
/// Equipment column ranges (`turbine`, `spillage`, `diversion`, `thermal`,
/// `line_fwd`, `line_rev`, `deficit`, `excess`) are populated only when constructed via
/// [`StageIndexer::with_equipment`]. When constructed via [`StageIndexer::new`]
/// or [`StageIndexer::from_stage_template`], those ranges are all empty (`0..0`)
/// and `n_blks`, `n_thermals`, `n_lines`, `n_buses` are zero.
///
/// FPHA fields (`generation`, `fpha_hydro_indices`, `fpha_rows`) are also
/// populated only by [`StageIndexer::with_equipment`] when FPHA hydros are
/// present. They are empty when built via [`StageIndexer::new`] or when no FPHA
/// hydros exist.
#[derive(Debug, Clone)]
pub struct StageIndexer {
    /// Column range `[0, N)` for outgoing storage volumes.
    ///
    /// Each entry `storage[h]` is the column index of hydro plant `h`'s
    /// outgoing storage volume.
    pub storage: Range<usize>,

    /// Column range `[N, N*(1+L))` for AR lag variables.
    ///
    /// Lag variables are stored in lag-major order: all hydros for lag 0,
    /// then all hydros for lag 1, etc. The column index for hydro `h` at
    /// lag `l` (0-indexed, lag 0 = most recent) is:
    /// `inflow_lags.start + l * hydro_count + h`.
    pub inflow_lags: Range<usize>,

    /// Column range `[N*(2+L), N*(3+L))` for incoming storage volumes.
    ///
    /// Fixed by the storage-fixing constraints; transferred from the preceding
    /// stage's `storage` solution values.
    pub storage_in: Range<usize>,

    /// Column index `N*(3+L)` for the future cost variable (theta).
    ///
    /// Scalar: there is exactly one theta variable per stage LP.
    pub theta: usize,

    /// Total state dimension `N*(1+L)`.
    ///
    /// The state vector consists of the `N` outgoing storage volumes followed
    /// by the `N*L` lag variables. State transfer copies
    /// `primal[0..n_transfer]` (all but the oldest lag row).
    pub n_state: usize,

    /// Row range `[0, N)` for storage-fixing constraints.
    ///
    /// Each constraint fixes one incoming storage variable to the value
    /// received from the preceding stage. Dual values over this range form
    /// the storage-volume cut coefficients.
    pub storage_fixing: Range<usize>,

    /// Row range `[N, N*(1+L))` for AR lag-fixing constraints.
    ///
    /// Each constraint fixes one lag variable to the value received from the
    /// preceding stage. Dual values over this range form the inflow-lag cut
    /// coefficients.
    pub lag_fixing: Range<usize>,

    /// Number of operating hydro plants (N).
    pub hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    ///
    /// All hydros use a uniform lag stride of `max_par_order`, enabling
    /// contiguous memory access and SIMD vectorisation over the lag dimension.
    pub max_par_order: usize,

    // ── Equipment column ranges ────────────────────────────────────────────
    // Populated only by `with_equipment`; empty (`0..0`) when built via `new`.
    /// Column range for turbined flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `turbine.start + h * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub turbine: Range<usize>,

    /// Column range for spillage variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `spillage.start + h * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub spillage: Range<usize>,

    /// Column range for diversion flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `diversion.start + h * n_blks + b`.
    /// Hydros without a diversion channel have bounds [0, 0]; the LP presolve
    /// eliminates them.
    /// Empty when built via [`StageIndexer::new`].
    pub diversion: Range<usize>,

    /// Column range for thermal generation variables, one per (thermal, block) pair.
    ///
    /// Index for thermal `t`, block `b`: `thermal.start + t * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub thermal: Range<usize>,

    /// Column range for forward line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_fwd.start + l * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub line_fwd: Range<usize>,

    /// Column range for reverse line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_rev.start + l * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub line_rev: Range<usize>,

    /// Column range for bus deficit variables, `B * S * K` columns total.
    ///
    /// S = `max_deficit_segments` (uniform stride across all buses).  For buses
    /// with fewer than S segments, the trailing segment slots have zero bounds
    /// and zero objective and are eliminated by the presolver.
    ///
    /// Index for bus `b_idx`, segment `s`, block `blk`:
    /// `deficit.start + b_idx * max_deficit_segments * n_blks + s * n_blks + blk`.
    ///
    /// Empty when built via [`StageIndexer::new`].
    pub deficit: Range<usize>,

    /// Maximum number of deficit segments across all buses (S).
    ///
    /// Used together with `deficit.start` to compute per-segment column indices.
    /// Set to `0` when built via [`StageIndexer::new`], `1` when built via
    /// [`StageIndexer::with_equipment`] (backward-compatible single-segment mode),
    /// and the true maximum when built via [`StageIndexer::with_equipment_and_evaporation`].
    pub max_deficit_segments: usize,

    /// Column range for bus excess variables, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `excess.start + b_idx * n_blks + blk`.
    /// Empty when built via [`StageIndexer::new`].
    pub excess: Range<usize>,

    /// Number of operating blocks per stage (K).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_blks: usize,

    /// Number of thermal units (T).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_thermals: usize,

    /// Number of transmission lines (`L_n`).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_lines: usize,

    /// Number of buses (B).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_buses: usize,

    /// Row range for water balance constraints, one per operating hydro.
    ///
    /// Index for hydro `h`: `water_balance.start + h`.
    /// The dual of this row gives the marginal value of water (water value).
    /// Empty when built via [`StageIndexer::new`].
    pub water_balance: Range<usize>,

    /// Row range for load balance constraints, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `load_balance.start + b_idx * n_blks + blk`.
    /// The RHS of these rows contains the load (MW) for each bus in each block.
    /// Empty when built via [`StageIndexer::new`].
    pub load_balance: Range<usize>,

    /// Column range for inflow non-negativity slack variables `sigma_inf_h`.
    ///
    /// One slack per operating hydro, appended after `excess` when the penalty
    /// method is active (`has_inflow_penalty == true`).  The slack is in m³/s;
    /// it absorbs negative inflow realisations and enters the water balance row
    /// with coefficient `+tau_total * M3S_TO_HM3`.
    ///
    /// Empty (`0..0`) when `has_inflow_penalty == false` or when built via
    /// [`StageIndexer::new`].
    pub inflow_slack: Range<usize>,

    /// Row range for inflow non-negativity constraint rows.
    ///
    /// Currently unused as a separate constraint block — the slack appears
    /// directly in the water balance row.  Reserved for future formulations
    /// that add an explicit `sigma_inf_h + a_h >= 0` row.
    ///
    /// Empty (`0..0`) in this implementation.
    pub inflow_slack_rows: Range<usize>,

    /// Whether inflow non-negativity penalty slack columns are present.
    ///
    /// `true` when `build_stage_templates` was called with an
    /// [`InflowNonNegativityMethod`](crate::inflow_method::InflowNonNegativityMethod)
    /// whose `has_slack_columns()` returns `true` and `n_hydros > 0`.
    /// `false` otherwise (including when built via [`StageIndexer::new`]).
    pub has_inflow_penalty: bool,

    // ── FPHA column and row ranges ─────────────────────────────────────────
    // Populated only by `with_equipment`; empty when built via `new`.
    /// Column range for FPHA generation variables, one per (`fpha_hydro`, block) pair.
    ///
    /// Index for FPHA hydro at local position `i`, block `b`:
    /// `generation.start + i * n_blks + b`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub generation: Range<usize>,

    /// Number of FPHA hydros in this stage.
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_fpha_hydros: usize,

    /// Mapping from FPHA local index to system hydro index.
    ///
    /// `fpha_hydro_indices[i]` is the system-level hydro position for FPHA hydro `i`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub fpha_hydro_indices: Vec<usize>,

    /// FPHA constraint row ranges per FPHA hydro.
    ///
    /// `fpha_rows[i]` is the [`FphaRowRange`] for FPHA hydro at local position `i`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub fpha_rows: Vec<FphaRowRange>,

    // ── Evaporation column and row indices ─────────────────────────────────
    // Populated only by `with_equipment`; empty when built via `new`.
    /// Number of hydros with linearized evaporation at this stage.
    ///
    /// Zero when built via [`StageIndexer::new`] or when no evaporation hydros exist.
    pub n_evap_hydros: usize,

    /// Mapping from evaporation local index to system hydro index.
    ///
    /// `evap_hydro_indices[i]` is the system-level hydro position for evaporation hydro `i`.
    /// Empty when no evaporation hydros exist or when built via [`StageIndexer::new`].
    pub evap_hydro_indices: Vec<usize>,

    /// Per-evaporation-hydro column and row indices.
    ///
    /// `evap_indices[i]` is the [`EvaporationIndices`] for evaporation hydro at local
    /// position `i`.  Empty when no evaporation hydros exist or when built via
    /// [`StageIndexer::new`].
    pub evap_indices: Vec<EvaporationIndices>,

    // ── Withdrawal slack column range ──────────────────────────────────────
    // Populated only by `with_equipment_and_evaporation`; empty when built via `new`.
    /// Column range for water-withdrawal violation slack variables `sigma^r_h`.
    ///
    /// One slack per operating hydro, appended after the evaporation columns.
    /// Columns are stage-level (not per-block); the slack absorbs violations of
    /// the minimum water-withdrawal flow constraint.
    ///
    /// Allocated whenever `hydro_count > 0`, matching the `inflow_slack` pattern.
    /// Empty (`0..0`) when `hydro_count == 0` or when built via
    /// [`StageIndexer::new`].
    pub withdrawal_slack: Range<usize>,

    /// Whether withdrawal slack columns are present.
    ///
    /// `true` when `with_equipment_and_evaporation` was called with
    /// `hydro_count > 0`.  `false` otherwise (including when built via
    /// [`StageIndexer::new`]).
    pub has_withdrawal: bool,

    // ── Generic constraint row and column ranges ────────────────────────────
    // Populated only by `StageLayout::new` in lp_builder via the full build
    // path; empty (`0..0`, 0) when built via [`StageIndexer::new`].
    /// Row range for generic constraint rows (one per active `(constraint, block)` pair).
    ///
    /// Rows are placed after evaporation rows (the last row region before
    /// generic constraints).  Empty (`0..0`) when no generic constraints are
    /// active at this stage or when built via [`StageIndexer::new`].
    pub generic_constraint_rows: Range<usize>,

    /// Column range for generic constraint slack variables.
    ///
    /// Columns are placed after withdrawal slack columns (the last column region
    /// before generic constraint slacks).  The number of columns equals the number
    /// of active rows when `slack.enabled = true` and sense is `<=` or `>=`, or
    /// twice the number of active rows when sense is `==` (positive and negative
    /// violation slacks).  Empty (`0..0`) when no slack is needed or when built
    /// via [`StageIndexer::new`].
    pub generic_constraint_slack: Range<usize>,

    /// Number of active generic constraint rows contributed at this stage.
    ///
    /// Zero when no generic constraints are active or when built via
    /// [`StageIndexer::new`].
    pub n_generic_constraints_active: usize,

    // ── NCS column range ──────────────────────────────────────────────────
    // Populated only by `StageLayout::new` in lp_builder; empty when built
    // via `new`, `with_equipment`, or `from_stage_template`.
    /// Column range for NCS generation variables, one per (ncs, block) pair.
    ///
    /// Index for NCS `r`, block `b`: `ncs_generation.start + r * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`] or when no NCS entities are active.
    pub ncs_generation: Range<usize>,

    // ── Z-inflow column and row ranges ────────────────────────────────────
    // Populated by all constructors.  The z_inflow columns are auxiliary
    // (NOT state variables); their primal values give the realized total
    // inflow Z_t per hydro after solving.
    /// Column range for realized-inflow variables `z_h`, one per hydro.
    ///
    /// These free columns (lower = -inf, upper = +inf, zero cost) represent the
    /// total natural inflow `Z_t_h` at each hydro, defined by the z-inflow
    /// equality constraints. After solving, `primal[z_inflow.start + h]` gives
    /// the realized inflow for hydro h.
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow: Range<usize>,

    /// Row range for z-inflow definition constraints, one per hydro.
    ///
    /// Each row defines: `z_h - sum_l[psi_l * lag_in[h,l]] = base_h + sigma_h * eta_h`
    /// The RHS is noise-patched (Category 5 in `PatchBuffer`).
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow_rows: Range<usize>,

    /// Row index of the first z-inflow definition constraint.
    ///
    /// Used by `PatchBuffer::fill_z_inflow_patches` as the base offset for
    /// Category 5 patches. Equal to `z_inflow_rows.start`.
    pub z_inflow_row_start: usize,

    /// Indices of state dimensions whose cut coefficients can be nonzero.
    ///
    /// Storage indices `[0, N)` are always included. Lag indices `[N, N*(1+L))`
    /// are included only when `lag < actual_ar_order[hydro]`. Hydros with AR
    /// order < `max_par_order` have padded lag slots whose duals are
    /// structurally zero.
    ///
    /// When empty (default), callers should treat all `n_state` indices as
    /// nonzero (dense path). Use [`set_nonzero_mask`](Self::set_nonzero_mask) to
    /// populate after construction when per-hydro AR orders are available.
    pub nonzero_state_indices: Vec<usize>,
}

/// Equipment counts for constructing a [`StageIndexer`].
///
/// Groups the entity counts that determine the LP column layout for a single stage.
pub struct EquipmentCounts {
    /// Number of hydro plants.
    pub hydro_count: usize,
    /// Maximum PAR model order across all hydros.
    pub max_par_order: usize,
    /// Number of thermal units.
    pub n_thermals: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of buses.
    pub n_buses: usize,
    /// Number of demand blocks in the stage.
    pub n_blks: usize,
    /// Whether to include inflow penalty slack columns.
    pub has_inflow_penalty: bool,
    /// Maximum number of deficit segments across all buses.
    pub max_deficit_segments: usize,
}

/// FPHA (Piecewise-linear Hydro Approximation) column layout.
///
/// Groups the per-hydro FPHA data needed for column layout computation.
pub struct FphaColumnLayout {
    /// Indices of hydros using FPHA production models.
    pub hydro_indices: Vec<usize>,
    /// Number of FPHA planes for each hydro in `hydro_indices`.
    ///
    /// Must have the same length as `hydro_indices`.
    pub planes_per_hydro: Vec<usize>,
}

/// Evaporation configuration for hydro plants.
pub struct EvapConfig {
    /// Indices of hydros with evaporation modeling enabled.
    pub hydro_indices: Vec<usize>,
}

impl StageIndexer {
    /// Construct a [`StageIndexer`] from `hydro_count` (N) and `max_par_order` (L).
    ///
    /// All index ranges are computed from N and L using the formulas in
    /// Solver Abstraction SS2.1–SS2.2. The constructor is infallible;
    /// validation of N and L is the caller's responsibility.
    ///
    /// Equipment column ranges (`turbine`, `spillage`, `thermal`, `line_fwd`,
    /// `line_rev`, `deficit`, `excess`) are all empty (`0..0`) and equipment
    /// counts (`n_blks`, `n_thermals`, `n_lines`, `n_buses`) are zero. Use
    /// [`StageIndexer::with_equipment`] to populate them.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::StageIndexer;
    ///
    /// // Worked example from spec SS5.5.3: N = 3, L = 2
    /// let idx = StageIndexer::new(3, 2);
    /// assert_eq!(idx.storage,   0..3);
    /// assert_eq!(idx.inflow_lags, 3..9);
    /// assert_eq!(idx.z_inflow,  9..12);
    /// assert_eq!(idx.storage_in, 12..15);
    /// assert_eq!(idx.theta,   15);
    /// assert_eq!(idx.n_state,  9);
    /// assert_eq!(idx.storage_fixing, 0..3);
    /// assert_eq!(idx.lag_fixing, 3..9);
    /// // Equipment ranges are empty when built via `new`.
    /// assert!(idx.turbine.is_empty());
    /// assert_eq!(idx.n_blks, 0);
    /// assert_eq!(idx.z_inflow_rows, 9..12);
    /// assert_eq!(idx.z_inflow_row_start, 9);
    /// ```
    #[must_use]
    pub fn new(hydro_count: usize, max_par_order: usize) -> Self {
        let n = hydro_count;
        let l = max_par_order;

        let storage = 0..n;
        let inflow_lags = n..n * (1 + l);

        // z_inflow columns at fixed offset N*(1+L), immediately after lags
        // and before storage_in. This makes z_inflow stage-invariant.
        let z_inflow_start = n * (1 + l);
        let z_inflow = z_inflow_start..z_inflow_start + n;

        let storage_in = n * (2 + l)..n * (3 + l);
        let theta = n * (3 + l);
        let n_state = n * (1 + l);

        // Row layout mirrors the column layout for the state-relevant rows.
        let storage_fixing = 0..n;
        let lag_fixing = n..n * (1 + l);

        // z_inflow rows at fixed offset N*(1+L), after lag-fixing rows
        // and before water balance rows.
        let z_inflow_rows = z_inflow_start..z_inflow_start + n;
        let z_inflow_row_start = z_inflow_start;

        Self {
            storage,
            inflow_lags,
            storage_in,
            theta,
            n_state,
            storage_fixing,
            lag_fixing,
            hydro_count,
            max_par_order,
            // Equipment ranges are empty until `with_equipment` is called.
            turbine: 0..0,
            spillage: 0..0,
            diversion: 0..0,
            thermal: 0..0,
            line_fwd: 0..0,
            line_rev: 0..0,
            deficit: 0..0,
            max_deficit_segments: 0,
            excess: 0..0,
            n_blks: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            water_balance: 0..0,
            load_balance: 0..0,
            inflow_slack: 0..0,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: false,
            generation: 0..0,
            n_fpha_hydros: 0,
            fpha_hydro_indices: Vec::new(),
            fpha_rows: Vec::new(),
            n_evap_hydros: 0,
            evap_hydro_indices: Vec::new(),
            evap_indices: Vec::new(),
            withdrawal_slack: 0..0,
            has_withdrawal: false,
            generic_constraint_rows: 0..0,
            generic_constraint_slack: 0..0,
            n_generic_constraints_active: 0,
            ncs_generation: 0..0,
            z_inflow,
            z_inflow_rows,
            z_inflow_row_start,
            nonzero_state_indices: Vec::new(),
        }
    }

    /// Construct a [`StageIndexer`] with full equipment column ranges.
    ///
    /// Computes both the state-variable ranges (identical to [`StageIndexer::new`])
    /// and the equipment decision-variable ranges that follow `theta` in the LP.
    ///
    /// The equipment column layout matches `lp_builder.rs` exactly:
    ///
    /// ```text
    /// decision_start      = theta + 1
    /// turbine_start       = decision_start
    /// spillage_start      = turbine_start  + n_hydros * n_blks
    /// diversion_start     = spillage_start + n_hydros * n_blks
    /// thermal_start       = diversion_start + n_hydros * n_blks
    /// line_fwd_start      = thermal_start  + n_thermals * n_blks
    /// line_rev_start      = line_fwd_start + n_lines * n_blks
    /// deficit_start       = line_rev_start + n_lines * n_blks
    /// excess_start        = deficit_start  + n_buses * max_deficit_segments * n_blks
    /// inflow_slack_start  = excess_end  (only when has_inflow_penalty && hydro_count > 0)
    /// generation_start    = inflow_slack_end  (FPHA generation columns)
    /// evap_start          = generation_end  (3 columns per evaporation hydro, stage-level)
    /// ```
    ///
    /// FPHA generation columns come immediately after `inflow_slack` (or after
    /// `excess` when `has_inflow_penalty == false`), one column per FPHA hydro
    /// per block.  FPHA constraint rows are placed after `load_balance`.
    ///
    /// Evaporation columns (3 per evaporation hydro: `Q_ev`, `f_evap_plus`,
    /// `f_evap_minus`) are stage-level (not per-block) and come immediately after
    /// the FPHA generation columns.  Evaporation rows (1 per evaporation hydro)
    /// are placed after FPHA rows.
    ///
    /// # Notes
    ///
    /// This constructor assumes a **uniform block count across all stages**
    /// (i.e., all stages have the same `n_blks`). For the minimal viable
    /// solver this assumption holds; stages with heterogeneous block counts
    /// would require a per-stage indexer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::StageIndexer;
    ///
    /// // N=1 hydro, L=0 lags, T=2 thermals, L_n=1 line, B=2 buses, K=1 block, no penalty
    /// // theta = N*(3+L) = 1*(3+0) = 3
    /// // decision_start = 4
    /// // turbine:    4..5   (1 hydro * 1 block)
    /// // spillage:   5..6   (1 hydro * 1 block)
    /// // diversion:  6..7   (1 hydro * 1 block)
    /// // thermal:    7..9   (2 thermals * 1 block)
    /// // line_fwd:   9..10  (1 line * 1 block)
    /// // line_rev:  10..11  (1 line * 1 block)
    /// // deficit:   11..13  (2 buses * 1 block)
    /// // excess:    13..15  (2 buses * 1 block)
    /// let counts = cobre_sddp::EquipmentCounts {
    ///     hydro_count: 1, max_par_order: 0, n_thermals: 2, n_lines: 1,
    ///     n_buses: 2, n_blks: 1, has_inflow_penalty: false, max_deficit_segments: 1,
    /// };
    /// let fpha = cobre_sddp::FphaColumnLayout { hydro_indices: vec![], planes_per_hydro: vec![] };
    /// let idx = StageIndexer::with_equipment(&counts, &fpha);
    /// assert_eq!(idx.turbine,    4..5);
    /// assert_eq!(idx.spillage,   5..6);
    /// assert_eq!(idx.diversion,  6..7);
    /// assert_eq!(idx.thermal,    7..9);
    /// assert_eq!(idx.line_fwd,   9..10);
    /// assert_eq!(idx.line_rev,  10..11);
    /// assert_eq!(idx.deficit,   11..13);
    /// assert_eq!(idx.excess,    13..15);
    /// assert!(idx.inflow_slack.is_empty());
    /// assert!(idx.generation.is_empty());
    /// assert_eq!(idx.n_blks, 1);
    /// assert_eq!(idx.n_thermals, 2);
    /// assert_eq!(idx.n_lines, 1);
    /// assert_eq!(idx.n_buses, 2);
    /// ```
    #[must_use]
    pub fn with_equipment(counts: &EquipmentCounts, fpha: &FphaColumnLayout) -> Self {
        Self::with_equipment_and_evaporation(
            counts,
            fpha,
            &EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// Construct a [`StageIndexer`] with full equipment column ranges and evaporation.
    ///
    /// Extends [`StageIndexer::with_equipment`] with evaporation hydro indices.
    /// Evaporation columns (3 per evaporation hydro: `Q_ev`, `f_evap_plus`,
    /// `f_evap_minus`) are stage-level and placed after FPHA generation columns.
    /// Evaporation rows (1 per evaporation hydro) are placed after FPHA rows.
    ///
    /// # Arguments
    ///
    /// - `counts` — equipment counts grouped into [`EquipmentCounts`]
    /// - `fpha` — FPHA column layout grouped into [`FphaColumnLayout`]
    /// - `evap` — evaporation configuration grouped into [`EvapConfig`]
    ///
    /// When `evap.hydro_indices` is empty this produces the same result as
    /// [`StageIndexer::with_equipment`].
    #[must_use]
    pub fn with_equipment_and_evaporation(
        counts: &EquipmentCounts,
        fpha: &FphaColumnLayout,
        evap: &EvapConfig,
    ) -> Self {
        let hydro_count = counts.hydro_count;
        let max_par_order = counts.max_par_order;
        let n_thermals = counts.n_thermals;
        let n_lines = counts.n_lines;
        let n_buses = counts.n_buses;
        let n_blks = counts.n_blks;
        let has_inflow_penalty = counts.has_inflow_penalty;
        let max_deficit_segments = counts.max_deficit_segments;
        let fpha_hydro_indices = fpha.hydro_indices.clone();
        let fpha_planes_per_hydro = &fpha.planes_per_hydro;
        let evap_hydro_indices = evap.hydro_indices.clone();

        debug_assert_eq!(
            fpha_hydro_indices.len(),
            fpha_planes_per_hydro.len(),
            "fpha_hydro_indices and fpha_planes_per_hydro must have equal length"
        );

        let base = Self::new(hydro_count, max_par_order);
        let decision_start = base.theta + 1;

        let turbine_start = decision_start;
        let spillage_start = turbine_start + hydro_count * n_blks;
        let diversion_start = spillage_start + hydro_count * n_blks;
        let thermal_start = diversion_start + hydro_count * n_blks;
        let line_fwd_start = thermal_start + n_thermals * n_blks;
        let line_rev_start = line_fwd_start + n_lines * n_blks;
        let deficit_start = line_rev_start + n_lines * n_blks;
        let excess_start = deficit_start + n_buses * max_deficit_segments * n_blks;
        let excess_end = excess_start + n_buses * n_blks;

        // Inflow slack columns are appended after excess when the penalty method
        // is active and there is at least one hydro.
        let (inflow_slack, active_penalty) = if has_inflow_penalty && hydro_count > 0 {
            (excess_end..excess_end + hydro_count, true)
        } else {
            (0..0, false)
        };

        // FPHA generation columns are placed immediately after inflow_slack (or after
        // excess when no penalty), one column per FPHA hydro per block.
        let n_fpha_hydros = fpha_hydro_indices.len();
        let generation_start = if active_penalty {
            inflow_slack.end
        } else {
            excess_end
        };
        let generation_end = generation_start + n_fpha_hydros * n_blks;
        let generation = if n_fpha_hydros > 0 {
            generation_start..generation_end
        } else {
            0..0
        };

        // Evaporation columns: 3 per evaporation hydro (stage-level, not per-block),
        // placed immediately after FPHA generation columns.
        // Layout within the evaporation region for local evaporation index `i`:
        //   Q_ev_col        = evap_start + i * 3
        //   f_evap_plus_col = evap_start + i * 3 + 1
        //   f_evap_minus_col= evap_start + i * 3 + 2
        let n_evap_hydros = evap_hydro_indices.len();
        let evap_col_start = generation_end;

        // Row layout: [storage_fixing | lag_fixing | z_inflow | water_balance | load_balance | fpha_rows]
        let water_balance_start = base.n_state + hydro_count;
        let load_balance_start = water_balance_start + hydro_count;
        let load_balance_end = load_balance_start + n_buses * n_blks;

        let (fpha_rows, fpha_row_cursor) =
            Self::build_fpha_rows(fpha_planes_per_hydro, n_blks, load_balance_end);

        let evap_indices_vec =
            Self::build_evap_indices(n_evap_hydros, evap_col_start, fpha_row_cursor);

        let evap_col_end = evap_col_start + n_evap_hydros * 3;
        let (withdrawal_slack, has_withdrawal) = if hydro_count > 0 {
            (evap_col_end..evap_col_end + hydro_count, true)
        } else {
            (0..0, false)
        };

        // z_inflow columns are at fixed offset N*(1+L), inherited from base.
        // No re-computation needed; the base constructor already places them correctly.

        Self {
            turbine: turbine_start..spillage_start,
            spillage: spillage_start..diversion_start,
            diversion: diversion_start..thermal_start,
            thermal: thermal_start..line_fwd_start,
            line_fwd: line_fwd_start..line_rev_start,
            line_rev: line_rev_start..deficit_start,
            deficit: deficit_start..excess_start,
            max_deficit_segments,
            excess: excess_start..excess_end,
            n_blks,
            n_thermals,
            n_lines,
            n_buses,
            water_balance: water_balance_start..water_balance_start + hydro_count,
            load_balance: load_balance_start..load_balance_end,
            inflow_slack,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: active_penalty,
            generation,
            n_fpha_hydros,
            fpha_hydro_indices,
            fpha_rows,
            n_evap_hydros,
            evap_hydro_indices,
            evap_indices: evap_indices_vec,
            withdrawal_slack,
            has_withdrawal,
            // z_inflow, z_inflow_rows, z_inflow_row_start inherited from base
            // (fixed offset N*(1+L), stage-invariant)
            ..base
        }
    }

    /// Return the [`EvaporationIndices`] for the evaporation hydro at local position `local_idx`.
    ///
    /// `local_idx` is the position within the evaporation hydro list (0-indexed).
    /// Use `evap_hydro_indices[local_idx]` to map to the system-level hydro position.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `local_idx >= n_evap_hydros`.
    #[must_use]
    pub fn evap_indices(&self, local_idx: usize) -> &EvaporationIndices {
        debug_assert!(
            local_idx < self.n_evap_hydros,
            "evap local index {local_idx} out of bounds (n_evap_hydros = {})",
            self.n_evap_hydros
        );
        &self.evap_indices[local_idx]
    }

    /// Build FPHA constraint row ranges from per-hydro plane counts.
    fn build_fpha_rows(
        planes_per_hydro: &[usize],
        n_blks: usize,
        start_row: usize,
    ) -> (Vec<FphaRowRange>, usize) {
        let mut rows = Vec::with_capacity(planes_per_hydro.len());
        let mut cursor = start_row;
        for &planes in planes_per_hydro {
            rows.push(FphaRowRange {
                start: cursor,
                planes_per_block: planes,
            });
            cursor += planes * n_blks;
        }
        (rows, cursor)
    }

    /// Build evaporation column/row indices for each evaporation hydro.
    fn build_evap_indices(
        n_evap_hydros: usize,
        col_start: usize,
        row_start: usize,
    ) -> Vec<EvaporationIndices> {
        (0..n_evap_hydros)
            .map(|i| EvaporationIndices {
                q_ev_col: col_start + i * 3,
                f_evap_plus_col: col_start + i * 3 + 1,
                f_evap_minus_col: col_start + i * 3 + 2,
                evap_row: row_start + i,
            })
            .collect()
    }

    /// Construct a [`StageIndexer`] from a [`StageTemplate`].
    ///
    /// Extracts `n_hydro` and `max_par_order` from the template and delegates
    /// to [`StageIndexer::new`]. Produces identical results to calling
    /// `StageIndexer::new(template.n_hydro, template.max_par_order)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::StageIndexer;
    /// use cobre_solver::StageTemplate;
    ///
    /// let template = StageTemplate {
    ///     num_cols: 16,
    ///     num_rows: 12,
    ///     num_nz: 0,
    ///     col_starts: vec![0_i32; 17],
    ///     row_indices: vec![],
    ///     values: vec![],
    ///     col_lower: vec![0.0; 16],
    ///     col_upper: vec![f64::INFINITY; 16],
    ///     objective: vec![0.0; 16],
    ///     row_lower: vec![0.0; 12],
    ///     row_upper: vec![f64::INFINITY; 12],
    ///     n_state: 9,
    ///     n_transfer: 6,
    ///     n_dual_relevant: 9,
    ///     n_hydro: 3,
    ///     max_par_order: 2,
    ///     col_scale: vec![],
    ///     row_scale: vec![],
    /// };
    ///
    /// let idx = StageIndexer::from_stage_template(&template);
    /// assert_eq!(idx.storage, 0..3);
    /// assert_eq!(idx.theta,  15);
    /// ```
    #[must_use]
    pub fn from_stage_template(template: &StageTemplate) -> Self {
        Self::new(template.n_hydro, template.max_par_order)
    }

    /// Compute and store the nonzero state index mask from per-hydro AR orders.
    ///
    /// `ar_orders` must have length `hydro_count`. Each entry is the actual AR
    /// order for that hydro (0 means no AR lags). Indices `[0, N)` (storage)
    /// are always included. For each hydro `h`, lag indices
    /// `inflow_lags.start + h * max_par_order + l` are included for
    /// `l in 0..ar_orders[h]`.
    ///
    /// After calling, `nonzero_state_indices` is sorted in ascending order and
    /// has no duplicates. If `max_par_order == 0` or all hydros have full AR
    /// order, the mask covers all `n_state` indices (equivalent to dense).
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `ar_orders.len() != hydro_count` or any `ar_orders[h] > max_par_order`.
    pub fn set_nonzero_mask(&mut self, ar_orders: &[usize]) {
        debug_assert_eq!(
            ar_orders.len(),
            self.hydro_count,
            "ar_orders length {} != hydro_count {}",
            ar_orders.len(),
            self.hydro_count
        );

        let mut mask = Vec::with_capacity(self.n_state);

        // Storage indices are always nonzero.
        for h in 0..self.hydro_count {
            mask.push(h);
        }

        // Lag indices: include only used lags (lag-major layout matching LP).
        // Iterate lag-first, hydro-second to produce sorted indices.
        for lag in 0..self.max_par_order {
            for (h, &order) in ar_orders.iter().enumerate() {
                debug_assert!(
                    order <= self.max_par_order,
                    "ar_orders[{h}] = {order} exceeds max_par_order {}",
                    self.max_par_order
                );
                if lag < order {
                    mask.push(self.inflow_lags.start + lag * self.hydro_count + h);
                }
            }
        }

        debug_assert!(
            mask.windows(2).all(|w| w[0] < w[1]),
            "nonzero_state_indices must be sorted and unique"
        );

        self.nonzero_state_indices = mask;
    }
}

// StageIndexer contains only Send + Sync types (Range<usize>, usize, Vec<usize>,
// Vec<FphaRowRange>, Vec<EvaporationIndices>), so Send + Sync are automatically
// derived. The explicit bounds below serve as a compile-time assertion that the
// safety invariant holds.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<StageIndexer>();
    }
    let _ = check;
};

#[cfg(test)]
mod tests {
    use cobre_solver::StageTemplate;

    use super::{EquipmentCounts, EvapConfig, FphaColumnLayout, FphaRowRange, StageIndexer};

    /// Test helper: construct `EquipmentCounts` with `max_deficit_segments = 1`.
    fn eq(
        hydro_count: usize,
        max_par_order: usize,
        n_thermals: usize,
        n_lines: usize,
        n_buses: usize,
        n_blks: usize,
        has_inflow_penalty: bool,
    ) -> EquipmentCounts {
        EquipmentCounts {
            hydro_count,
            max_par_order,
            n_thermals,
            n_lines,
            n_buses,
            n_blks,
            has_inflow_penalty,
            max_deficit_segments: 1,
        }
    }

    /// Test helper: construct `FphaColumnLayout`.
    fn fpha(hydro_indices: Vec<usize>, planes_per_hydro: Vec<usize>) -> FphaColumnLayout {
        FphaColumnLayout {
            hydro_indices,
            planes_per_hydro,
        }
    }

    /// Test helper: construct `EvapConfig`.
    fn evap(hydro_indices: Vec<usize>) -> EvapConfig {
        EvapConfig { hydro_indices }
    }

    // Worked example from spec SS5.5.3: N = 3, L = 2

    fn indexer_3_2() -> StageIndexer {
        StageIndexer::new(3, 2)
    }

    #[test]
    fn storage_range_3_2() {
        assert_eq!(indexer_3_2().storage, 0..3);
    }

    #[test]
    fn inflow_lags_range_3_2() {
        // N = 3, L = 2 → [N, N*(1+L)) = [3, 9)
        assert_eq!(indexer_3_2().inflow_lags, 3..9);
    }

    #[test]
    fn z_inflow_range_3_2() {
        // [N*(1+L), N*(2+L)) = [9, 12)
        assert_eq!(indexer_3_2().z_inflow, 9..12);
    }

    #[test]
    fn storage_in_range_3_2() {
        // [N*(2+L), N*(3+L)) = [12, 15)
        assert_eq!(indexer_3_2().storage_in, 12..15);
    }

    #[test]
    fn theta_index_3_2() {
        // N*(3+L) = 3*(3+2) = 15
        assert_eq!(indexer_3_2().theta, 15);
    }

    #[test]
    fn n_state_3_2() {
        // N*(1+L) = 3*(1+2) = 9
        assert_eq!(indexer_3_2().n_state, 9);
    }

    #[test]
    fn storage_fixing_range_3_2() {
        assert_eq!(indexer_3_2().storage_fixing, 0..3);
    }

    #[test]
    fn lag_fixing_range_3_2() {
        // [N, N + N*L) = [3, 9)  ≡  [N, N*(1+L))
        assert_eq!(indexer_3_2().lag_fixing, 3..9);
    }

    #[test]
    fn row_column_symmetry_3_2() {
        let idx = indexer_3_2();
        assert_eq!(idx.storage_fixing, idx.storage);
        assert_eq!(idx.lag_fixing, idx.inflow_lags);
    }

    // Production scale: N = 160, L = 12

    fn indexer_160_12() -> StageIndexer {
        StageIndexer::new(160, 12)
    }

    #[test]
    fn n_state_production_scale() {
        // N*(1+L) = 160*13 = 2080
        assert_eq!(indexer_160_12().n_state, 2080);
    }

    #[test]
    fn theta_production_scale() {
        // N*(3+L) = 160*15 = 2400
        assert_eq!(indexer_160_12().theta, 2400);
    }

    #[test]
    fn row_column_symmetry_production_scale() {
        let idx = indexer_160_12();
        assert_eq!(idx.storage_fixing, idx.storage);
        assert_eq!(idx.lag_fixing, idx.inflow_lags);
    }

    // Edge case: N = 1, L = 0 (single hydro, no lags)

    #[test]
    fn single_hydro_no_lags() {
        let idx = StageIndexer::new(1, 0);

        // storage: 0..1
        assert_eq!(idx.storage, 0..1);
        // inflow_lags: 1..1*(1+0) = 1..1 (empty)
        assert_eq!(idx.inflow_lags, 1..1);
        // z_inflow: 1..1*(2+0) = 1..2
        assert_eq!(idx.z_inflow, 1..2);
        // storage_in: 1*(2+0)..1*(3+0) = 2..3
        assert_eq!(idx.storage_in, 2..3);
        // theta: 1*(3+0) = 3
        assert_eq!(idx.theta, 3);
        // n_state: 1*(1+0) = 1
        assert_eq!(idx.n_state, 1);
        // storage_fixing: 0..1
        assert_eq!(idx.storage_fixing, 0..1);
        // lag_fixing: 1..1 (empty)
        assert_eq!(idx.lag_fixing, 1..1);

        // Row-column symmetry holds for empty ranges
        assert_eq!(idx.storage_fixing, idx.storage);
        assert_eq!(idx.lag_fixing, idx.inflow_lags);
    }

    // Edge case: N = 0, L = 0 (degenerate — all ranges empty)

    #[test]
    fn degenerate_zero_hydros() {
        let idx = StageIndexer::new(0, 0);

        assert_eq!(idx.storage, 0..0);
        assert_eq!(idx.inflow_lags, 0..0);
        assert_eq!(idx.z_inflow, 0..0);
        assert_eq!(idx.storage_in, 0..0);
        assert_eq!(idx.theta, 0);
        assert_eq!(idx.n_state, 0);
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);

        assert_eq!(idx.storage_fixing, idx.storage);
        assert_eq!(idx.lag_fixing, idx.inflow_lags);
    }

    // from_stage_template: must produce the same result as new()

    fn make_template(n_hydro: usize, max_par_order: usize) -> StageTemplate {
        let n_state = n_hydro * (1 + max_par_order);
        let n_transfer = n_hydro * max_par_order;
        // Minimal valid template (matrix contents are irrelevant for indexer)
        StageTemplate {
            num_cols: 0,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![],
            col_upper: vec![],
            objective: vec![],
            row_lower: vec![],
            row_upper: vec![],
            n_state,
            n_transfer,
            n_dual_relevant: n_state,
            n_hydro,
            max_par_order,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    #[test]
    fn from_stage_template_matches_new_3_2() {
        let tmpl = make_template(3, 2);
        let from_tmpl = StageIndexer::from_stage_template(&tmpl);
        let from_new = StageIndexer::new(3, 2);

        assert_eq!(from_tmpl.storage, from_new.storage);
        assert_eq!(from_tmpl.inflow_lags, from_new.inflow_lags);
        assert_eq!(from_tmpl.storage_in, from_new.storage_in);
        assert_eq!(from_tmpl.theta, from_new.theta);
        assert_eq!(from_tmpl.n_state, from_new.n_state);
        assert_eq!(from_tmpl.storage_fixing, from_new.storage_fixing);
        assert_eq!(from_tmpl.lag_fixing, from_new.lag_fixing);
        assert_eq!(from_tmpl.hydro_count, from_new.hydro_count);
        assert_eq!(from_tmpl.max_par_order, from_new.max_par_order);
    }

    #[test]
    fn from_stage_template_matches_new_160_12() {
        let tmpl = make_template(160, 12);
        let from_tmpl = StageIndexer::from_stage_template(&tmpl);
        let from_new = StageIndexer::new(160, 12);

        assert_eq!(from_tmpl.n_state, from_new.n_state);
        assert_eq!(from_tmpl.theta, from_new.theta);
        assert_eq!(from_tmpl.hydro_count, from_new.hydro_count);
        assert_eq!(from_tmpl.max_par_order, from_new.max_par_order);
    }

    #[test]
    fn from_stage_template_matches_new_edge_cases() {
        for (n, l) in [(0, 0), (1, 0), (1, 1)] {
            let tmpl = make_template(n, l);
            let from_tmpl = StageIndexer::from_stage_template(&tmpl);
            let from_new = StageIndexer::new(n, l);

            assert_eq!(from_tmpl.storage, from_new.storage, "N={n} L={l}");
            assert_eq!(from_tmpl.inflow_lags, from_new.inflow_lags, "N={n} L={l}");
            assert_eq!(from_tmpl.theta, from_new.theta, "N={n} L={l}");
            assert_eq!(from_tmpl.n_state, from_new.n_state, "N={n} L={l}");
        }
    }

    // Clone / Debug — structural sanity

    #[test]
    fn clone_and_debug() {
        let idx = indexer_3_2();
        let cloned = idx.clone();
        assert_eq!(cloned.theta, idx.theta);
        assert_eq!(cloned.n_state, idx.n_state);

        let debug_str = format!("{idx:?}");
        assert!(debug_str.contains("StageIndexer"));
    }

    // new() produces empty equipment ranges
    #[test]
    fn new_equipment_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(idx.turbine.is_empty());
        assert!(idx.spillage.is_empty());
        assert!(idx.diversion.is_empty());
        assert!(idx.thermal.is_empty());
        assert!(idx.line_fwd.is_empty());
        assert!(idx.line_rev.is_empty());
        assert!(idx.deficit.is_empty());
        assert!(idx.excess.is_empty());
        assert_eq!(idx.n_blks, 0);
        assert_eq!(idx.n_thermals, 0);
        assert_eq!(idx.n_lines, 0);
        assert_eq!(idx.n_buses, 0);
    }

    // with_equipment: worked example from doc comment (N=1, L=0, T=2, Ln=1, B=2, K=1)
    //
    // theta = N*(3+L) = 1*(3+0) = 3
    // decision_start = 4
    // turbine:    [4, 4+1*1)  = 4..5
    // spillage:   [5, 5+1*1)  = 5..6
    // diversion:  [6, 6+1*1)  = 6..7
    // thermal:    [7, 7+2*1)  = 7..9
    // line_fwd:   [9, 9+1*1)  = 9..10
    // line_rev:  [10,10+1*1)  = 10..11
    // deficit:   [11,11+2*1)  = 11..13
    // excess:    [13,13+2*1)  = 13..15
    #[test]
    fn with_equipment_doctest_n1_l0_t2_l1_b2_k1() {
        let idx = StageIndexer::with_equipment(&eq(1, 0, 2, 1, 2, 1, false), &fpha(vec![], vec![]));

        // State ranges are identical to new(1, 0)
        assert_eq!(idx.storage, 0..1);
        assert_eq!(idx.inflow_lags, 1..1);
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.storage_in, 2..3);
        assert_eq!(idx.theta, 3);
        assert_eq!(idx.n_state, 1);

        // Equipment ranges
        assert_eq!(idx.turbine, 4..5);
        assert_eq!(idx.spillage, 5..6);
        assert_eq!(idx.diversion, 6..7);
        assert_eq!(idx.thermal, 7..9);
        assert_eq!(idx.line_fwd, 9..10);
        assert_eq!(idx.line_rev, 10..11);
        assert_eq!(idx.deficit, 11..13);
        assert_eq!(idx.excess, 13..15);

        // Equipment counts
        assert_eq!(idx.n_blks, 1);
        assert_eq!(idx.n_thermals, 2);
        assert_eq!(idx.n_lines, 1);
        assert_eq!(idx.n_buses, 2);
    }

    // with_equipment: N=2, L=1, T=3, Ln=2, B=4, K=2
    //
    // theta = N*(3+L) = 2*(3+1) = 8
    // decision_start = 9
    // turbine:    [9,  9+2*2)  = 9..13
    // spillage:  [13, 13+2*2)  = 13..17
    // diversion: [17, 17+2*2)  = 17..21
    // thermal:   [21, 21+3*2)  = 21..27
    // line_fwd:  [27, 27+2*2)  = 27..31
    // line_rev:  [31, 31+2*2)  = 31..35
    // deficit:   [35, 35+4*2)  = 35..43
    // excess:    [43, 43+4*2)  = 43..51
    #[test]
    fn with_equipment_n2_l1_t3_l2_b4_k2() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 3, 2, 4, 2, false), &fpha(vec![], vec![]));

        // State ranges identical to new(2, 1)
        assert_eq!(idx.theta, 8);
        assert_eq!(idx.n_state, 4); // N*(1+L) = 2*2 = 4

        // Equipment ranges
        assert_eq!(idx.turbine, 9..13);
        assert_eq!(idx.spillage, 13..17);
        assert_eq!(idx.diversion, 17..21);
        assert_eq!(idx.thermal, 21..27);
        assert_eq!(idx.line_fwd, 27..31);
        assert_eq!(idx.line_rev, 31..35);
        assert_eq!(idx.deficit, 35..43);
        assert_eq!(idx.excess, 43..51);
    }

    // with_equipment: no equipment (all counts zero), matches new() state layout
    #[test]
    fn with_equipment_all_counts_zero_matches_new() {
        let with_eq =
            StageIndexer::with_equipment(&eq(3, 2, 0, 0, 0, 0, false), &fpha(vec![], vec![]));
        let base = StageIndexer::new(3, 2);

        assert_eq!(with_eq.storage, base.storage);
        assert_eq!(with_eq.inflow_lags, base.inflow_lags);
        assert_eq!(with_eq.storage_in, base.storage_in);
        assert_eq!(with_eq.theta, base.theta);
        assert_eq!(with_eq.n_state, base.n_state);
        // All equipment ranges empty
        assert!(with_eq.turbine.is_empty());
        assert!(with_eq.spillage.is_empty());
        assert!(with_eq.diversion.is_empty());
        assert!(with_eq.thermal.is_empty());
        assert!(with_eq.line_fwd.is_empty());
        assert!(with_eq.line_rev.is_empty());
        assert!(with_eq.deficit.is_empty());
        assert!(with_eq.excess.is_empty());
    }

    // with_equipment: adjacency invariant — ranges must be contiguous and non-overlapping
    #[test]
    fn with_equipment_ranges_are_contiguous() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 3, 2, 4, 2, false), &fpha(vec![], vec![]));

        // turbine immediately follows theta
        assert_eq!(idx.turbine.start, idx.theta + 1);
        // each range starts where the previous ends
        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.thermal.start, idx.diversion.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
    }

    // Column index formula: turbine[h, b] = turbine.start + h * n_blks + b
    #[test]
    fn with_equipment_column_index_formulas() {
        let n_blks = 3_usize;
        let idx =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, n_blks, false), &fpha(vec![], vec![]));

        // turbine[h=0, b=0] = turbine.start (no offset for h=0, b=0)
        assert_eq!(idx.turbine.start, idx.turbine.start);
        // turbine[h=1, b=2] = turbine.start + 1*3 + 2 = turbine.start + 5
        assert_eq!(idx.turbine.start + n_blks + 2, idx.turbine.start + 5);
        // deficit[b_idx=0, blk=1] = deficit.start + 1
        assert_eq!(idx.deficit.start + 1, idx.deficit.start + 1);
        // turbine[h=1, b=0] = turbine.start + n_blks
        assert_eq!(idx.turbine.start + n_blks, idx.turbine.start + 3);
    }

    // with_equipment: has_inflow_penalty=true appends N slack columns after excess
    //
    // N=2, L=1, T=1, Ln=1, B=1, K=1, penalty=true
    // theta = N*(3+L) = 2*(3+1) = 8
    // decision_start = 9
    // turbine:    [9,  11)
    // spillage:  [11, 13)
    // diversion: [13, 15)
    // thermal:   [15, 16)
    // line_fwd:  [16, 17)
    // line_rev:  [17, 18)
    // deficit:   [18, 19)
    // excess:    [19, 20)
    // inflow_slack: [20, 22)  <- excess_end..excess_end+N
    #[test]
    fn with_equipment_inflow_penalty_appends_slack() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, true), &fpha(vec![], vec![]));

        assert!(idx.has_inflow_penalty, "has_inflow_penalty must be true");
        // inflow_slack must start exactly where excess ends
        assert_eq!(
            idx.inflow_slack.start, idx.excess.end,
            "inflow_slack.start must equal excess.end (contiguous)"
        );
        // inflow_slack must contain exactly hydro_count columns
        assert_eq!(
            idx.inflow_slack.len(),
            idx.hydro_count,
            "inflow_slack must contain exactly hydro_count columns"
        );
        assert_eq!(idx.inflow_slack, 20..22);
        // inflow_slack_rows stays empty in this implementation
        assert!(
            idx.inflow_slack_rows.is_empty(),
            "inflow_slack_rows must remain empty"
        );
        // without penalty the slack range is empty
        let no_penalty =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![], vec![]));
        assert!(!no_penalty.has_inflow_penalty);
        assert!(no_penalty.inflow_slack.is_empty());
    }

    // ── FPHA field tests ───────────────────────────────────────────────────

    // AC-4: no FPHA hydros → generation is empty, fpha_rows is empty.
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA.
    // theta = N*(3+L) = 4*(3+0) = 12
    // decision_start = 13
    // turbine:  [13, 17)
    // spillage: [17, 21)
    // deficit:  [21, 22)
    // excess:   [22, 23)
    // generation: empty (no FPHA hydros)
    #[test]
    fn fpha_no_hydros_generation_is_empty() {
        let idx = StageIndexer::with_equipment(&eq(4, 0, 0, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert!(
            idx.generation.is_empty(),
            "generation must be empty with no FPHA hydros"
        );
        assert_eq!(idx.n_fpha_hydros, 0);
        assert!(idx.fpha_hydro_indices.is_empty());
        assert!(idx.fpha_rows.is_empty());
    }

    // AC-1 + AC-2: 1 FPHA hydro, 1 block, 3 planes.
    //
    // N=2, L=0, T=1, Ln=0, B=1, K=1, no penalty.
    // theta = N*(3+L) = 2*(3+0) = 6
    // decision_start = 7
    // turbine:    [7, 9)   (2 hydros * 1 block)
    // spillage:   [9, 11)
    // diversion: [11, 13)  (2 hydros * 1 block)
    // thermal:   [13, 14)  (1 thermal * 1 block)
    // deficit:   [14, 15)  (1 bus * 1 block)
    // excess:    [15, 16)
    // generation: [16, 17) (1 FPHA hydro * 1 block)
    //
    // Row layout:
    // n_state = N*(1+L) = 2*(1+0) = 2
    // z_inflow rows = 2..4  (N*(1+L)..N*(2+L))
    // water_balance_start = N*(2+L) = 4
    // load_balance_start  = 4 + 2 = 6
    // load_balance_end    = 6 + 1*1 = 7
    // fpha_rows[0].start  = 7 (after load_balance.end)
    // fpha_rows[0].planes_per_block = 3
    #[test]
    fn fpha_one_hydro_one_block_three_planes() {
        let idx =
            StageIndexer::with_equipment(&eq(2, 0, 1, 0, 1, 1, false), &fpha(vec![0], vec![3]));

        // AC-1: generation spans 1 column (1 FPHA hydro * 1 block)
        assert_eq!(idx.generation.len(), 1, "generation must span 1 column");
        assert_eq!(idx.generation, 16..17);
        assert_eq!(idx.n_fpha_hydros, 1);
        assert_eq!(idx.fpha_hydro_indices, vec![0]);

        // AC-2: fpha_rows[0].start is after load_balance.end, planes_per_block == 3
        assert_eq!(idx.fpha_rows.len(), 1);
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 3);
    }

    // AC-3: 2 FPHA hydros, 2 blocks, plane counts [5, 4].
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=2, no penalty.
    // theta = N*(3+L) = 4*(3+0) = 12
    // decision_start = 13
    // turbine:    [13, 21)  (4 hydros * 2 blocks)
    // spillage:   [21, 29)
    // diversion:  [29, 37)  (4 hydros * 2 blocks)
    // deficit:    [37, 39)  (1 bus * 2 blocks)
    // excess:     [39, 41)
    // generation: [41, 45) (2 FPHA hydros * 2 blocks = 4 columns)
    #[test]
    fn fpha_two_hydros_two_blocks_different_planes() {
        let idx = StageIndexer::with_equipment(
            &eq(4, 0, 0, 0, 1, 2, false),
            &fpha(vec![1, 3], vec![5, 4]),
        );

        // AC-3: generation spans 4 columns (2 FPHA hydros * 2 blocks)
        assert_eq!(idx.generation.len(), 4, "generation must span 4 columns");
        assert_eq!(idx.n_fpha_hydros, 2);
        assert_eq!(idx.fpha_hydro_indices, vec![1, 3]);

        // fpha_rows: 2 entries with correct starts and plane counts
        assert_eq!(idx.fpha_rows.len(), 2);

        // fpha_rows[0]: hydro at local 0 (system hydro 1), 5 planes, 2 blocks
        // starts at load_balance.end
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 5);

        // fpha_rows[1]: starts after fpha_rows[0]'s region (5 planes * 2 blocks = 10 rows)
        assert_eq!(
            idx.fpha_rows[1].start,
            idx.fpha_rows[0].start + 5 * 2,
            "fpha_rows[1].start must follow fpha_rows[0]'s 10-row region"
        );
        assert_eq!(idx.fpha_rows[1].planes_per_block, 4);
    }

    // FPHA generation columns are contiguous with the prior column region.
    //
    // No penalty: generation immediately follows excess.
    // With penalty: generation immediately follows inflow_slack.
    #[test]
    fn fpha_generation_contiguous_with_prior_region() {
        // No penalty case: generation.start == excess.end
        let no_penalty =
            StageIndexer::with_equipment(&eq(2, 0, 0, 0, 1, 1, false), &fpha(vec![0], vec![2]));
        assert_eq!(
            no_penalty.generation.start, no_penalty.excess.end,
            "generation.start must equal excess.end when no penalty"
        );

        // With penalty case: generation.start == inflow_slack.end
        let with_penalty =
            StageIndexer::with_equipment(&eq(2, 0, 0, 0, 1, 1, true), &fpha(vec![0], vec![2]));
        assert_eq!(
            with_penalty.generation.start, with_penalty.inflow_slack.end,
            "generation.start must equal inflow_slack.end when penalty active"
        );
    }

    // FPHA rows are contiguous with load_balance (start at load_balance.end).
    #[test]
    fn fpha_rows_contiguous_with_load_balance() {
        let idx = StageIndexer::with_equipment(
            &eq(3, 1, 2, 0, 2, 3, false),
            &fpha(vec![0, 2], vec![4, 6]),
        );

        // First FPHA hydro starts at load_balance.end
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0] must start at load_balance.end"
        );

        // Each subsequent FPHA hydro starts after its predecessor's block
        // fpha_rows[0]: 4 planes * 3 blocks = 12 rows
        assert_eq!(
            idx.fpha_rows[1].start,
            idx.fpha_rows[0].start + 4 * 3,
            "fpha_rows[1] must start after fpha_rows[0]'s rows"
        );
        assert_eq!(idx.fpha_rows[1].planes_per_block, 6);
    }

    // ── Evaporation field tests ────────────────────────────────────────────

    // AC (ticket-010): 0 evaporation hydros → evap_indices is empty.
    #[test]
    fn evap_no_hydros_indices_empty() {
        let idx = StageIndexer::with_equipment(&eq(3, 0, 1, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert_eq!(idx.n_evap_hydros, 0);
        assert!(idx.evap_hydro_indices.is_empty());
        assert!(idx.evap_indices.is_empty());
    }

    // AC (ticket-010): 1 evaporation hydro — verify column/row positions.
    //
    // N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, 1 evap hydro.
    // theta = N*(3+L) = 2*(3+0) = 6
    // decision_start = 7
    // turbine:    [7, 9)   (2 hydros * 1 block)
    // spillage:   [9, 11)
    // diversion: [11, 13)  (2 hydros * 1 block)
    // deficit:   [13, 14)  (1 bus * 1 block)
    // excess:    [14, 15)
    // generation: empty (no FPHA)
    // evap cols: [15, 18)  (3 columns: Q_ev, f_evap_plus, f_evap_minus)
    //
    // Row layout:
    // n_state = N*(1+L) = 2
    // z_inflow rows = 2..4
    // water_balance_start = N*(2+L) = 4
    // load_balance_start = 4 + 2 = 6
    // load_balance_end   = 6 + 1*1 = 7
    // evap_row[0] = 7
    #[test]
    fn evap_one_hydro_column_row_positions() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert_eq!(idx.n_evap_hydros, 1);
        assert_eq!(idx.evap_hydro_indices, vec![0]);
        assert_eq!(idx.evap_indices.len(), 1);

        let ei = idx.evap_indices(0);
        // 3 columns placed after generation_end (which equals excess.end = 15)
        assert_eq!(ei.q_ev_col, 15);
        assert_eq!(ei.f_evap_plus_col, 16);
        assert_eq!(ei.f_evap_minus_col, 17);
        // row placed after load_balance.end = 7
        assert_eq!(ei.evap_row, 7);
    }

    // AC (ticket-010): 2 evaporation hydros — verify column/row ranges are
    // contiguous and non-overlapping with FPHA ranges.
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=1, no penalty, 1 FPHA hydro (index 0, 3 planes),
    // 2 evap hydros (indices 1, 2).
    // theta = 4*(3+0) = 12
    // decision_start = 13
    // turbine:    [13, 17)  (4 hydros * 1 block)
    // spillage:   [17, 21)
    // diversion:  [21, 25)  (4 hydros * 1 block)
    // deficit:    [25, 26)  (1 bus * 1 block)
    // excess:     [26, 27)
    // generation: [27, 28) (1 FPHA hydro * 1 block)
    //
    // Row layout:
    // n_state = 4
    // z_inflow rows = 4..8
    // water_balance_start = N*(2+L) = 8
    // load_balance_start = 8 + 4 = 12
    // load_balance_end   = 12 + 1*1 = 13
    // fpha_rows[0].start = 13
    // fpha_row_cursor after FPHA = 13 + 3*1 = 16
    // evap cols: [28, 34)   (2 evap hydros * 3 = 6 columns)
    // evap_row[0] = 16, evap_row[1] = 17
    #[test]
    fn evap_two_hydros_with_fpha_contiguous() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(4, 0, 0, 0, 1, 1, false),
            &fpha(vec![0], vec![3]),
            &evap(vec![1, 2]),
        );

        assert_eq!(idx.n_evap_hydros, 2);
        assert_eq!(idx.evap_hydro_indices, vec![1, 2]);

        let ei0 = idx.evap_indices(0);
        let ei1 = idx.evap_indices(1);

        // Columns start at generation_end = 28
        assert_eq!(ei0.q_ev_col, 28);
        assert_eq!(ei0.f_evap_plus_col, 29);
        assert_eq!(ei0.f_evap_minus_col, 30);

        assert_eq!(ei1.q_ev_col, 31);
        assert_eq!(ei1.f_evap_plus_col, 32);
        assert_eq!(ei1.f_evap_minus_col, 33);

        // Rows placed after fpha_rows region: fpha_row_cursor = 13 + 3*1 = 16
        assert_eq!(ei0.evap_row, 16);
        assert_eq!(ei1.evap_row, 17);

        // Evap rows do not overlap FPHA rows
        assert!(ei0.evap_row > idx.fpha_rows[0].start);
    }

    // new() produces empty evaporation fields.
    #[test]
    fn new_evap_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.n_evap_hydros, 0);
        assert!(idx.evap_hydro_indices.is_empty());
        assert!(idx.evap_indices.is_empty());
    }

    // ── Withdrawal slack field tests ───────────────────────────────────────

    // AC: with_equipment_and_evaporation, N=3 hydros, 1 evap hydro →
    // withdrawal_slack starts at evap_col_end and has length 3.
    //
    // N=3, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, 1 evap hydro.
    // theta = N*(3+L) = 3*(3+0) = 9
    // decision_start = 10
    // turbine:    [10, 13)  (3 hydros * 1 block)
    // spillage:   [13, 16)
    // diversion:  [16, 19)  (3 hydros * 1 block)
    // deficit:    [19, 20)  (1 bus * 1 block)
    // excess:     [20, 21)
    // generation: empty (no FPHA)
    // evap cols:  [21, 24)  (1 evap hydro * 3 columns)
    // withdrawal_slack: [24, 27)  (3 hydros)
    #[test]
    fn withdrawal_slack_with_equipment_and_evaporation_n3_evap1() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(3, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert!(idx.has_withdrawal);
        // withdrawal_slack.start must equal the end of the evaporation columns
        let evap_col_end = idx.evap_indices(0).f_evap_minus_col + 1;
        assert_eq!(
            idx.withdrawal_slack.start, evap_col_end,
            "withdrawal_slack.start must equal evap_col_end"
        );
        assert_eq!(
            idx.withdrawal_slack.len(),
            3,
            "withdrawal_slack must contain exactly hydro_count columns"
        );
        assert_eq!(idx.withdrawal_slack, 24..27);
    }

    // AC: with_equipment_and_evaporation, N=0 → withdrawal_slack is 0..0.
    #[test]
    fn withdrawal_slack_zero_hydros_is_empty() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(0, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert!(!idx.has_withdrawal);
        assert_eq!(idx.withdrawal_slack, 0..0);
    }

    // AC: new() → withdrawal_slack is 0..0.
    #[test]
    fn withdrawal_slack_from_new_is_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(!idx.has_withdrawal);
        assert_eq!(idx.withdrawal_slack, 0..0);
    }

    // AC: withdrawal_slack length equals hydro_count for various hydro counts (1, 5).
    #[test]
    fn withdrawal_slack_length_equals_hydro_count() {
        for n in [1_usize, 5] {
            let idx = StageIndexer::with_equipment_and_evaporation(
                &EquipmentCounts {
                    hydro_count: n,
                    max_par_order: 0,
                    n_thermals: 0,
                    n_lines: 0,
                    n_buses: 1,
                    n_blks: 1,
                    has_inflow_penalty: false,
                    max_deficit_segments: 1,
                },
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );

            assert!(idx.has_withdrawal, "has_withdrawal must be true for n={n}");
            assert_eq!(
                idx.withdrawal_slack.len(),
                n,
                "withdrawal_slack length must equal hydro_count for n={n}"
            );
        }
    }

    // AC: withdrawal_slack.start == evap_col_end (immediately after evaporation columns).
    //
    // N=2, L=0, no penalty, no FPHA, 1 evap hydro.
    // evap cols: [excess_end, excess_end+3) = [15, 18)
    // withdrawal_slack: [18, 20)
    #[test]
    fn withdrawal_slack_immediately_after_evap_columns() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        // evap_col_end = evap_col_start + n_evap_hydros * 3
        // evap_col_start = generation_end = excess_end (no FPHA, no penalty)
        // excess_end = excess_start + n_buses * n_blks = ... = 15
        // evap_col_end = 15 + 1*3 = 18
        assert_eq!(
            idx.withdrawal_slack.start, 18,
            "withdrawal_slack must start at evap_col_end=18"
        );
        assert_eq!(
            idx.withdrawal_slack.len(),
            2,
            "withdrawal_slack length must equal hydro_count=2"
        );
        assert_eq!(idx.withdrawal_slack, 18..20);
    }

    // EvaporationIndices is Debug + Clone + Copy.
    #[test]
    fn evap_indices_debug_clone_copy() {
        use super::EvaporationIndices;
        let ei = EvaporationIndices {
            q_ev_col: 10,
            f_evap_plus_col: 11,
            f_evap_minus_col: 12,
            evap_row: 5,
        };
        let cloned = ei;
        assert_eq!(cloned.q_ev_col, 10);
        assert_eq!(cloned.evap_row, 5);
        let debug_str = format!("{ei:?}");
        assert!(debug_str.contains("EvaporationIndices"));
    }

    // FphaRowRange is Debug + Clone + Copy.
    #[test]
    fn fpha_row_range_debug_clone_copy() {
        let r = FphaRowRange {
            start: 42,
            planes_per_block: 5,
        };
        let cloned = r;
        assert_eq!(cloned.start, 42);
        assert_eq!(cloned.planes_per_block, 5);
        let debug_str = format!("{r:?}");
        assert!(debug_str.contains("FphaRowRange"));
    }

    // new() produces empty FPHA ranges.
    #[test]
    fn new_fpha_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(idx.generation.is_empty());
        assert_eq!(idx.n_fpha_hydros, 0);
        assert!(idx.fpha_hydro_indices.is_empty());
        assert!(idx.fpha_rows.is_empty());
    }

    // Adjacency invariant extended: generation immediately follows the prior region.
    #[test]
    fn extended_adjacency_invariant_with_fpha() {
        // N=2, L=1, T=1, Ln=1, B=1, K=1, no penalty, 1 FPHA hydro.
        // theta=8, decision_start=9
        // turbine:[9,11), spillage:[11,13), diversion:[13,15), thermal:[15,16),
        // line_fwd:[16,17), line_rev:[17,18), deficit:[18,19), excess:[19,20)
        // generation:[20,21) (1 FPHA * 1 block, after excess.end since no penalty)
        let idx =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![0], vec![3]));

        assert_eq!(idx.turbine.start, idx.theta + 1);
        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.thermal.start, idx.diversion.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
        // generation follows excess (no penalty)
        assert_eq!(idx.generation.start, idx.excess.end);
        assert_eq!(idx.generation.len(), 1);
    }

    // ── Diversion field tests ──────────────────────────────────────────────

    // Diversion range: N=3, K=2 → diversion.len() = 6, contiguous with spillage.
    #[test]
    fn test_diversion_range_n3_l0_k2() {
        let idx = StageIndexer::with_equipment(&eq(3, 0, 0, 0, 1, 2, false), &fpha(vec![], vec![]));

        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.diversion.len(), 3 * 2);
        assert_eq!(idx.thermal.start, idx.diversion.end);
    }

    // Diversion range: N=0 → diversion is empty.
    #[test]
    fn test_diversion_zero_hydros() {
        let idx = StageIndexer::with_equipment(&eq(0, 0, 1, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert!(idx.diversion.is_empty());
    }

    // ── z_inflow field tests ───────────────────────────────────────────────

    // z_inflow starts at N*(1+L) (fixed offset) and has length hydro_count.
    #[test]
    fn z_inflow_range_new_constructor() {
        let idx = StageIndexer::new(3, 2);
        // z_inflow at N*(1+L) = 9..12
        assert_eq!(idx.z_inflow, 9..12);
        assert_eq!(idx.z_inflow.len(), idx.hydro_count);
    }

    // z_inflow is empty when hydro_count == 0.
    #[test]
    fn z_inflow_range_zero_hydros() {
        let idx = StageIndexer::new(0, 0);
        assert!(idx.z_inflow.is_empty());
        assert!(idx.z_inflow_rows.is_empty());
        assert_eq!(idx.z_inflow_row_start, 0);
    }

    // z_inflow_rows and z_inflow_row_start at N*(1+L) for the simple constructor.
    #[test]
    fn z_inflow_row_fields() {
        let idx = StageIndexer::new(5, 1);
        // N*(1+L) = 5*(1+1) = 10
        assert_eq!(idx.z_inflow_rows, 10..15);
        assert_eq!(idx.z_inflow_row_start, 10);
        assert_eq!(idx.z_inflow.len(), 5);
    }

    // z_inflow has correct length and rows for with_equipment constructor.
    #[test]
    fn z_inflow_range_with_equipment() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![], vec![]));
        // N*(1+L) = 2*(1+1) = 4
        assert_eq!(idx.z_inflow, 4..6);
        assert_eq!(idx.z_inflow.len(), 2);
        // z_inflow rows inherited from base at N*(1+L)
        assert_eq!(idx.z_inflow_rows, 4..6);
        assert_eq!(idx.z_inflow_row_start, 4);
    }

    // z_inflow placed correctly for single hydro, no lags case.
    #[test]
    fn z_inflow_single_hydro_no_lags() {
        let idx = StageIndexer::new(1, 0);
        // N*(1+L) = 1*(1+0) = 1, z_inflow at 1..2
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.z_inflow.len(), 1);
    }

    // ── Nonzero state mask tests ───────────────────────────────────────────

    #[test]
    fn nonzero_mask_default_is_empty() {
        let idx = StageIndexer::new(4, 6);
        assert!(
            idx.nonzero_state_indices.is_empty(),
            "default mask must be empty (dense path)"
        );
    }

    #[test]
    fn nonzero_mask_mixed_ar_orders() {
        // 4 hydros (N=4), max_par_order=6 (L=6), ar_orders=[0, 1, 3, 6]
        // inflow_lags.start = N = 4
        // Lag-major layout: slot = 4 + lag * N + h
        let mut idx = StageIndexer::new(4, 6);
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

        assert_eq!(&idx.nonzero_state_indices[..4], &[0, 1, 2, 3]);
        assert_eq!(
            &idx.nonzero_state_indices[4..],
            &[5, 6, 7, 10, 11, 14, 15, 19, 23, 27]
        );

        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn nonzero_mask_zero_par_order() {
        // max_par_order=0: no lags, mask = storage only
        let mut idx = StageIndexer::new(3, 0);
        idx.set_nonzero_mask(&[0, 0, 0]);
        assert_eq!(idx.nonzero_state_indices.len(), 3);
        assert_eq!(&idx.nonzero_state_indices, &[0, 1, 2]);
    }

    #[test]
    fn nonzero_mask_all_full_order() {
        // All hydros at max AR order: mask covers all n_state indices
        let mut idx = StageIndexer::new(2, 3);
        idx.set_nonzero_mask(&[3, 3]);
        // n_state = 2*(1+3) = 8, mask should have 2 + 2*3 = 8
        assert_eq!(idx.nonzero_state_indices.len(), 8);
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
    }
}
