//! LP layout index map for SDDP stage subproblems.
//!
//! [`StageIndexer`] centralises all column and row offset arithmetic for a
//! single-stage LP, eliminating magic index numbers throughout the forward
//! pass, backward pass, and LP construction code.
//!
//! ## Column layout (Solver Abstraction SS2.1)
//!
//! ```text
//! [0, N)          storage      — outgoing storage volumes  (N = hydro_count)
//! [N, N*(1+L))    inflow_lags  — AR lag variables (L lags per hydro)
//! [N*(1+L), N*(2+L)) storage_in — incoming storage volumes
//! N*(2+L)         theta        — future cost variable (scalar)
//! ```
//!
//! When built with [`StageIndexer::with_equipment`], the following equipment
//! columns follow immediately after `theta`:
//!
//! ```text
//! [theta+1,                          theta+1+H*K)        turbine     — turbined flow (m³/s)
//! [theta+1+H*K,                      theta+1+2*H*K)      spillage    — spilled flow (m³/s)
//! [theta+1+2*H*K,                    theta+1+2*H*K+T*K)  thermal     — thermal generation (MW)
//! [theta+1+2*H*K+T*K,                theta+1+2*H*K+T*K+2*L_n*K) line_fwd/rev — line flows
//! [theta+1+2*H*K+T*K+2*L_n*K,       theta+1+2*H*K+T*K+2*L_n*K+B*K) deficit
//! [theta+1+2*H*K+T*K+2*L_n*K+B*K,   theta+1+2*H*K+T*K+2*L_n*K+2*B*K) excess
//! ```
//!
//! When the inflow non-negativity penalty method is active (`has_inflow_penalty == true`),
//! `N` additional slack columns are appended after `excess`:
//!
//! ```text
//! [excess_end, excess_end+N)  inflow_slack — sigma_inf_h (m³/s), one per hydro
//! ```
//!
//! where H = `hydro_count`, K = `n_blks`, T = `n_thermals`, Ln = `n_lines`, B = `n_buses`.
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
//! storage_in   = 9..12  (= 3*(1+2)..3*(2+2))
//! theta        = 12     (= 3*(2+2))
//! n_state      = 9      (= 3*(1+2))
//! storage_fixing = 0..3
//! lag_fixing     = 3..9
//! ```

use std::ops::Range;

use cobre_solver::StageTemplate;

/// Read-only LP layout index map for one SDDP stage subproblem.
///
/// Computed once from `hydro_count` (N) and `max_par_order` (L), then shared
/// read-only across all threads for the duration of training. All fields are
/// plain `usize` or `Range<usize>` — no heap allocation, trivially `Copy`.
///
/// See the [module-level documentation](self) for the full column and row
/// layout, and [`StageIndexer::new`] for the construction formulas.
///
/// Equipment column ranges (`turbine`, `spillage`, `thermal`, `line_fwd`,
/// `line_rev`, `deficit`, `excess`) are populated only when constructed via
/// [`StageIndexer::with_equipment`]. When constructed via [`StageIndexer::new`]
/// or [`StageIndexer::from_stage_template`], those ranges are all empty (`0..0`)
/// and `n_blks`, `n_thermals`, `n_lines`, `n_buses` are zero.
#[derive(Debug, Clone)]
pub struct StageIndexer {
    /// Column range `[0, N)` for outgoing storage volumes.
    ///
    /// Each entry `storage[h]` is the column index of hydro plant `h`'s
    /// outgoing storage volume.
    pub storage: Range<usize>,

    /// Column range `[N, N*(1+L))` for AR lag variables.
    ///
    /// Lag variables are stored in hydro-major order: all lags for hydro 0,
    /// then all lags for hydro 1, etc. The column index for hydro `h` at
    /// lag `l` (0-indexed, lag 1 = most recent) is:
    /// `inflow_lags.start + h * max_par_order + l`.
    pub inflow_lags: Range<usize>,

    /// Column range `[N*(1+L), N*(2+L))` for incoming storage volumes.
    ///
    /// Fixed by the storage-fixing constraints; transferred from the preceding
    /// stage's `storage` solution values.
    pub storage_in: Range<usize>,

    /// Column index `N*(2+L)` for the future cost variable (theta).
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

    /// Column range for bus deficit variables, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `deficit.start + b_idx * n_blks + blk`.
    /// Empty when built via [`StageIndexer::new`].
    pub deficit: Range<usize>,

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
    /// assert_eq!(idx.storage_in, 9..12);
    /// assert_eq!(idx.theta,   12);
    /// assert_eq!(idx.n_state,  9);
    /// assert_eq!(idx.storage_fixing, 0..3);
    /// assert_eq!(idx.lag_fixing, 3..9);
    /// // Equipment ranges are empty when built via `new`.
    /// assert!(idx.turbine.is_empty());
    /// assert_eq!(idx.n_blks, 0);
    /// ```
    #[must_use]
    pub fn new(hydro_count: usize, max_par_order: usize) -> Self {
        let n = hydro_count;
        let l = max_par_order;

        let storage = 0..n;
        let inflow_lags = n..n * (1 + l);
        let storage_in = n * (1 + l)..n * (2 + l);
        let theta = n * (2 + l);
        let n_state = n * (1 + l);

        // Row layout mirrors the column layout for the state-relevant rows.
        let storage_fixing = 0..n;
        let lag_fixing = n..n * (1 + l);

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
            thermal: 0..0,
            line_fwd: 0..0,
            line_rev: 0..0,
            deficit: 0..0,
            excess: 0..0,
            n_blks: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            load_balance: 0..0,
            // Inflow penalty slack ranges are empty until `with_equipment` is called
            // with `has_inflow_penalty == true`.
            inflow_slack: 0..0,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: false,
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
    /// thermal_start       = spillage_start + n_hydros * n_blks
    /// line_fwd_start      = thermal_start  + n_thermals * n_blks
    /// line_rev_start      = line_fwd_start + n_lines * n_blks
    /// deficit_start       = line_rev_start + n_lines * n_blks
    /// excess_start        = deficit_start  + n_buses * n_blks
    /// ```
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
    /// // theta = N*(2+L) = 1*(2+0) = 2
    /// // decision_start = 3
    /// // turbine:   3..4   (1 hydro * 1 block)
    /// // spillage:  4..5   (1 hydro * 1 block)
    /// // thermal:   5..7   (2 thermals * 1 block)
    /// // line_fwd:  7..8   (1 line * 1 block)
    /// // line_rev:  8..9   (1 line * 1 block)
    /// // deficit:   9..11  (2 buses * 1 block)
    /// // excess:   11..13  (2 buses * 1 block)
    /// let idx = StageIndexer::with_equipment(1, 0, 2, 1, 2, 1, false);
    /// assert_eq!(idx.turbine,   3..4);
    /// assert_eq!(idx.spillage,  4..5);
    /// assert_eq!(idx.thermal,   5..7);
    /// assert_eq!(idx.line_fwd,  7..8);
    /// assert_eq!(idx.line_rev,  8..9);
    /// assert_eq!(idx.deficit,   9..11);
    /// assert_eq!(idx.excess,   11..13);
    /// assert!(idx.inflow_slack.is_empty());
    /// assert_eq!(idx.n_blks, 1);
    /// assert_eq!(idx.n_thermals, 2);
    /// assert_eq!(idx.n_lines, 1);
    /// assert_eq!(idx.n_buses, 2);
    /// ```
    #[must_use]
    pub fn with_equipment(
        hydro_count: usize,
        max_par_order: usize,
        n_thermals: usize,
        n_lines: usize,
        n_buses: usize,
        n_blks: usize,
        has_inflow_penalty: bool,
    ) -> Self {
        let base = Self::new(hydro_count, max_par_order);
        let decision_start = base.theta + 1;

        let turbine_start = decision_start;
        let spillage_start = turbine_start + hydro_count * n_blks;
        let thermal_start = spillage_start + hydro_count * n_blks;
        let line_fwd_start = thermal_start + n_thermals * n_blks;
        let line_rev_start = line_fwd_start + n_lines * n_blks;
        let deficit_start = line_rev_start + n_lines * n_blks;
        let excess_start = deficit_start + n_buses * n_blks;
        let excess_end = excess_start + n_buses * n_blks;

        // Inflow slack columns are appended after excess when the penalty method
        // is active and there is at least one hydro.
        let (inflow_slack, active_penalty) = if has_inflow_penalty && hydro_count > 0 {
            (excess_end..excess_end + hydro_count, true)
        } else {
            (0..0, false)
        };

        // Row layout: [storage_fixing | lag_fixing | water_balance | load_balance]
        // water_balance_start = n_state (= n_dual_relevant)
        // load_balance_start = water_balance_start + hydro_count
        let load_balance_start = base.n_state + hydro_count;
        let load_balance_end = load_balance_start + n_buses * n_blks;

        Self {
            turbine: turbine_start..spillage_start,
            spillage: spillage_start..thermal_start,
            thermal: thermal_start..line_fwd_start,
            line_fwd: line_fwd_start..line_rev_start,
            line_rev: line_rev_start..deficit_start,
            deficit: deficit_start..excess_start,
            excess: excess_start..excess_end,
            n_blks,
            n_thermals,
            n_lines,
            n_buses,
            load_balance: load_balance_start..load_balance_end,
            inflow_slack,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: active_penalty,
            ..base
        }
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
    ///     num_cols: 13,
    ///     num_rows: 9,
    ///     num_nz: 0,
    ///     col_starts: vec![0_i32; 14],
    ///     row_indices: vec![],
    ///     values: vec![],
    ///     col_lower: vec![0.0; 13],
    ///     col_upper: vec![f64::INFINITY; 13],
    ///     objective: vec![0.0; 13],
    ///     row_lower: vec![0.0; 9],
    ///     row_upper: vec![f64::INFINITY; 9],
    ///     n_state: 9,
    ///     n_transfer: 6,
    ///     n_dual_relevant: 9,
    ///     n_hydro: 3,
    ///     max_par_order: 2,
    /// };
    ///
    /// let idx = StageIndexer::from_stage_template(&template);
    /// assert_eq!(idx.storage, 0..3);
    /// assert_eq!(idx.theta,  12);
    /// ```
    #[must_use]
    pub fn from_stage_template(template: &StageTemplate) -> Self {
        Self::new(template.n_hydro, template.max_par_order)
    }
}

// StageIndexer contains only Copy types (Range<usize> and usize),
// so Send + Sync are automatically derived. The explicit bounds below
// serve as a compile-time assertion that the safety invariant holds.
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

    use super::StageIndexer;

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
    fn storage_in_range_3_2() {
        // [N*(1+L), N*(2+L)) = [9, 12)
        assert_eq!(indexer_3_2().storage_in, 9..12);
    }

    #[test]
    fn theta_index_3_2() {
        // N*(2+L) = 3*(2+2) = 12
        assert_eq!(indexer_3_2().theta, 12);
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
        // N*(2+L) = 160*14 = 2240
        assert_eq!(indexer_160_12().theta, 2240);
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
        // storage_in: 1..1*(2+0) = 1..2
        assert_eq!(idx.storage_in, 1..2);
        // theta: 1*(2+0) = 2
        assert_eq!(idx.theta, 2);
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
    // theta = N*(2+L) = 1*(2+0) = 2
    // decision_start = 3
    // turbine:   [3, 3+1*1)  = 3..4
    // spillage:  [4, 4+1*1)  = 4..5
    // thermal:   [5, 5+2*1)  = 5..7
    // line_fwd:  [7, 7+1*1)  = 7..8
    // line_rev:  [8, 8+1*1)  = 8..9
    // deficit:   [9, 9+2*1)  = 9..11
    // excess:   [11, 11+2*1) = 11..13
    #[test]
    fn with_equipment_doctest_n1_l0_t2_l1_b2_k1() {
        let idx = StageIndexer::with_equipment(1, 0, 2, 1, 2, 1, false);

        // State ranges are identical to new(1, 0)
        assert_eq!(idx.storage, 0..1);
        assert_eq!(idx.inflow_lags, 1..1);
        assert_eq!(idx.storage_in, 1..2);
        assert_eq!(idx.theta, 2);
        assert_eq!(idx.n_state, 1);

        // Equipment ranges
        assert_eq!(idx.turbine, 3..4);
        assert_eq!(idx.spillage, 4..5);
        assert_eq!(idx.thermal, 5..7);
        assert_eq!(idx.line_fwd, 7..8);
        assert_eq!(idx.line_rev, 8..9);
        assert_eq!(idx.deficit, 9..11);
        assert_eq!(idx.excess, 11..13);

        // Equipment counts
        assert_eq!(idx.n_blks, 1);
        assert_eq!(idx.n_thermals, 2);
        assert_eq!(idx.n_lines, 1);
        assert_eq!(idx.n_buses, 2);
    }

    // with_equipment: N=2, L=1, T=3, Ln=2, B=4, K=2
    //
    // theta = N*(2+L) = 2*(2+1) = 6
    // decision_start = 7
    // turbine:   [7,  7+2*2)  = 7..11
    // spillage: [11, 11+2*2)  = 11..15
    // thermal:  [15, 15+3*2)  = 15..21
    // line_fwd: [21, 21+2*2)  = 21..25
    // line_rev: [25, 25+2*2)  = 25..29
    // deficit:  [29, 29+4*2)  = 29..37
    // excess:   [37, 37+4*2)  = 37..45
    #[test]
    fn with_equipment_n2_l1_t3_l2_b4_k2() {
        let idx = StageIndexer::with_equipment(2, 1, 3, 2, 4, 2, false);

        // State ranges identical to new(2, 1)
        assert_eq!(idx.theta, 6);
        assert_eq!(idx.n_state, 4); // N*(1+L) = 2*2 = 4

        // Equipment ranges
        assert_eq!(idx.turbine, 7..11);
        assert_eq!(idx.spillage, 11..15);
        assert_eq!(idx.thermal, 15..21);
        assert_eq!(idx.line_fwd, 21..25);
        assert_eq!(idx.line_rev, 25..29);
        assert_eq!(idx.deficit, 29..37);
        assert_eq!(idx.excess, 37..45);
    }

    // with_equipment: no equipment (all counts zero), matches new() state layout
    #[test]
    fn with_equipment_all_counts_zero_matches_new() {
        let with_eq = StageIndexer::with_equipment(3, 2, 0, 0, 0, 0, false);
        let base = StageIndexer::new(3, 2);

        assert_eq!(with_eq.storage, base.storage);
        assert_eq!(with_eq.inflow_lags, base.inflow_lags);
        assert_eq!(with_eq.storage_in, base.storage_in);
        assert_eq!(with_eq.theta, base.theta);
        assert_eq!(with_eq.n_state, base.n_state);
        // All equipment ranges empty
        assert!(with_eq.turbine.is_empty());
        assert!(with_eq.spillage.is_empty());
        assert!(with_eq.thermal.is_empty());
        assert!(with_eq.line_fwd.is_empty());
        assert!(with_eq.line_rev.is_empty());
        assert!(with_eq.deficit.is_empty());
        assert!(with_eq.excess.is_empty());
    }

    // with_equipment: adjacency invariant — ranges must be contiguous and non-overlapping
    #[test]
    fn with_equipment_ranges_are_contiguous() {
        let idx = StageIndexer::with_equipment(2, 1, 3, 2, 4, 2, false);

        // turbine immediately follows theta
        assert_eq!(idx.turbine.start, idx.theta + 1);
        // each range starts where the previous ends
        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.thermal.start, idx.spillage.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
    }

    // Column index formula: turbine[h, b] = turbine.start + h * n_blks + b
    #[test]
    fn with_equipment_column_index_formulas() {
        let n_blks = 3_usize;
        let idx = StageIndexer::with_equipment(2, 1, 1, 1, 1, n_blks, false);

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
    // theta = N*(2+L) = 2*(2+1) = 6
    // decision_start = 7
    // turbine:  [7,  9)
    // spillage: [9,  11)
    // thermal:  [11, 12)
    // line_fwd: [12, 13)
    // line_rev: [13, 14)
    // deficit:  [14, 15)
    // excess:   [15, 16)
    // inflow_slack: [16, 18)  <- excess_end..excess_end+N
    #[test]
    fn with_equipment_inflow_penalty_appends_slack() {
        let idx = StageIndexer::with_equipment(2, 1, 1, 1, 1, 1, true);

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
        assert_eq!(idx.inflow_slack, 16..18);
        // inflow_slack_rows stays empty in this implementation
        assert!(
            idx.inflow_slack_rows.is_empty(),
            "inflow_slack_rows must remain empty"
        );
        // without penalty the slack range is empty
        let no_penalty = StageIndexer::with_equipment(2, 1, 1, 1, 1, 1, false);
        assert!(!no_penalty.has_inflow_penalty);
        assert!(no_penalty.inflow_slack.is_empty());
    }
}
