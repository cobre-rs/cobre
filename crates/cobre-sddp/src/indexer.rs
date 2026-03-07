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
}

impl StageIndexer {
    /// Construct a [`StageIndexer`] from `hydro_count` (N) and `max_par_order` (L).
    ///
    /// All index ranges are computed from N and L using the formulas in
    /// Solver Abstraction SS2.1–SS2.2. The constructor is infallible;
    /// validation of N and L is the caller's responsibility.
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
    ///     col_starts: vec![0; 14],
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
            col_starts: vec![0],
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
}
