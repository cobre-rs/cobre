//! `Historical` scenario sampling scheme — library type and eta pre-standardization.
//!
//! [`HistoricalScenarioLibrary`] stores pre-standardized eta values for
//! historical inflow windows. During the forward pass, the
//! `ClassSampler::Historical` variant indexes into this library to retrieve
//! noise vectors for a given (window, stage) pair.
//!
//! The [`standardize_historical_windows`] function populates the library by
//! inverting the PAR(p) model: for every valid window and study stage it
//! computes `η = (obs - deterministic_base - Σ ψ[ℓ]·lag[ℓ]) / σ` using **raw
//! historical inflow values** as lags (BR6 — not PAR-reconstructed values).
//!
//! ## Eta storage layout
//!
//! The `eta` buffer uses **window-major** layout:
//! `eta[window * n_stages * n_hydros + stage * n_hydros + hydro]`.
//!
//! This is optimal for sequential stage iteration within a single window
//! (same cache-line access pattern as [`PrecomputedPar`]).
//!
//! ## Lag storage layout
//!
//! The `lag_values` buffer uses **window-major** layout:
//! `lag_values[window * max_order * n_hydros + lag * n_hydros + hydro]`.
//!
//! lag index 0 is the most recent pre-study observation (lag-1),
//! lag index 1 is lag-2, and so on.
//!
//! These are the raw historical inflow values at the `max_order` stages
//! immediately before the window starts (BR6 specification), used by
//! `apply_initial_state` to seed the solver state vector.
//!
//! [`PrecomputedPar`]: crate::par::precompute::PrecomputedPar

use chrono::Datelike;
use cobre_core::{
    EntityId,
    scenario::{HistoricalYears, InflowHistoryRow},
    temporal::Stage,
};

use crate::{
    StochasticError,
    par::{evaluate::solve_par_noise, precompute::PrecomputedPar},
};

// ---------------------------------------------------------------------------
// HistoricalScenarioLibrary
// ---------------------------------------------------------------------------

/// Pre-standardized eta store for historical scenario windows.
///
/// A pure data container — no sampling logic (permutation, selection) is
/// included. Population is performed by the eta-standardisation pass after
/// construction; selection is performed by the `ClassSampler::Historical`
/// variant during the forward pass.
///
/// # Construction
///
/// Use [`HistoricalScenarioLibrary::new`], which allocates zero-filled
/// buffers. Population is done by ticket-019.
///
/// # Examples
///
/// ```
/// use cobre_stochastic::HistoricalScenarioLibrary;
///
/// let mut lib = HistoricalScenarioLibrary::new(3, 12, 5, 2, vec![1990, 1995, 2000]);
/// assert_eq!(lib.n_windows(), 3);
/// assert_eq!(lib.n_stages(), 12);
/// assert_eq!(lib.n_hydros(), 5);
/// assert_eq!(lib.max_order(), 2);
/// assert_eq!(lib.window_year(0), 1990);
/// assert_eq!(lib.window_year(2), 2000);
///
/// // Write and read eta values.
/// lib.eta_slice_mut(1, 3).copy_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5]);
/// assert_eq!(lib.eta_slice(1, 3), &[0.1, 0.2, 0.3, 0.4, 0.5]);
/// ```
#[derive(Debug, Clone)]
pub struct HistoricalScenarioLibrary {
    /// Flat eta buffer in window-major layout.
    eta: Box<[f64]>,
    /// Flat pre-study lag buffer in window-major layout.
    lag_values: Box<[f64]>,
    /// Year labels for each window (for diagnostics).
    window_years: Box<[i32]>,
    /// Number of historical windows.
    n_windows: usize,
    /// Number of study stages per window.
    n_stages: usize,
    /// Number of hydro entities.
    n_hydros: usize,
    /// PAR model order — number of lag stages before each window start.
    max_order: usize,
}

impl HistoricalScenarioLibrary {
    /// Construct a new library with zero-filled buffers.
    ///
    /// # Parameters
    ///
    /// - `n_windows` — number of historical windows
    /// - `n_stages` — number of study stages per window
    /// - `n_hydros` — number of hydro entities (eta and lag vector width)
    /// - `max_order` — PAR model order; number of pre-window lag stages to store
    /// - `window_years` — starting year for each window (diagnostic label)
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `window_years.len() != n_windows`.
    #[must_use]
    pub fn new(
        n_windows: usize,
        n_stages: usize,
        n_hydros: usize,
        max_order: usize,
        window_years: Vec<i32>,
    ) -> Self {
        debug_assert_eq!(
            window_years.len(),
            n_windows,
            "window_years length ({}) must equal n_windows ({})",
            window_years.len(),
            n_windows,
        );
        Self {
            eta: vec![0.0_f64; n_windows * n_stages * n_hydros].into_boxed_slice(),
            lag_values: vec![0.0_f64; n_windows * max_order * n_hydros].into_boxed_slice(),
            window_years: window_years.into_boxed_slice(),
            n_windows,
            n_stages,
            n_hydros,
            max_order,
        }
    }

    // -----------------------------------------------------------------------
    // Dimension accessors
    // -----------------------------------------------------------------------

    /// Returns the number of historical windows.
    #[must_use]
    #[inline]
    pub fn n_windows(&self) -> usize {
        self.n_windows
    }

    /// Returns the number of study stages per window.
    #[must_use]
    #[inline]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Returns the number of hydro entities (eta vector width).
    #[must_use]
    #[inline]
    pub fn n_hydros(&self) -> usize {
        self.n_hydros
    }

    /// Returns the PAR model order (number of pre-window lag stages stored).
    #[must_use]
    #[inline]
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Returns the starting year label for window `window`.
    ///
    /// Used for diagnostic and logging purposes only.
    #[must_use]
    #[inline]
    pub fn window_year(&self, window: usize) -> i32 {
        debug_assert!(
            window < self.n_windows,
            "window ({window}) must be < n_windows ({})",
            self.n_windows
        );
        self.window_years[window]
    }

    // -----------------------------------------------------------------------
    // Eta accessors
    // -----------------------------------------------------------------------

    /// Returns the `n_hydros`-length slice of eta values for `(window, stage)`.
    ///
    /// Layout: `eta[window * n_stages * n_hydros + stage * n_hydros + hydro]`.
    ///
    /// # Panics
    ///
    /// Panics if `window >= n_windows` or `stage >= n_stages`.
    #[must_use]
    #[inline]
    pub fn eta_slice(&self, window: usize, stage: usize) -> &[f64] {
        assert!(
            window < self.n_windows,
            "window ({window}) must be < n_windows ({})",
            self.n_windows
        );
        assert!(
            stage < self.n_stages,
            "stage ({stage}) must be < n_stages ({})",
            self.n_stages
        );
        let offset = (window * self.n_stages + stage) * self.n_hydros;
        &self.eta[offset..offset + self.n_hydros]
    }

    /// Returns a mutable `n_hydros`-length slice of eta values for `(window, stage)`.
    ///
    /// Used by the eta-standardisation pass (ticket-019) to populate the library.
    ///
    /// # Panics
    ///
    /// Panics if `window >= n_windows` or `stage >= n_stages`.
    #[must_use]
    #[inline]
    pub fn eta_slice_mut(&mut self, window: usize, stage: usize) -> &mut [f64] {
        assert!(
            window < self.n_windows,
            "window ({window}) must be < n_windows ({})",
            self.n_windows
        );
        assert!(
            stage < self.n_stages,
            "stage ({stage}) must be < n_stages ({})",
            self.n_stages
        );
        let offset = (window * self.n_stages + stage) * self.n_hydros;
        &mut self.eta[offset..offset + self.n_hydros]
    }

    // -----------------------------------------------------------------------
    // Lag accessors
    // -----------------------------------------------------------------------

    /// Returns the `max_order * n_hydros`-length slice of pre-study lag values
    /// for `window`.
    ///
    /// Layout: `lag_values[window * max_order * n_hydros + lag * n_hydros + hydro]`.
    ///
    /// These are the raw historical inflow values at the `max_order` stages
    /// immediately before the window starts.
    #[must_use]
    #[inline]
    pub fn lag_slice(&self, window: usize) -> &[f64] {
        debug_assert!(
            window < self.n_windows,
            "window ({window}) must be < n_windows ({})",
            self.n_windows
        );
        let len = self.max_order * self.n_hydros;
        let offset = window * len;
        &self.lag_values[offset..offset + len]
    }

    /// Returns a mutable `max_order * n_hydros`-length slice of pre-study lag
    /// values for `window`.
    ///
    /// Used by the library-population pass (ticket-025) to write lag values.
    #[must_use]
    #[inline]
    pub fn lag_slice_mut(&mut self, window: usize) -> &mut [f64] {
        debug_assert!(
            window < self.n_windows,
            "window ({window}) must be < n_windows ({})",
            self.n_windows
        );
        let len = self.max_order * self.n_hydros;
        let offset = window * len;
        &mut self.lag_values[offset..offset + len]
    }
}

// ---------------------------------------------------------------------------
// standardize_historical_windows
// ---------------------------------------------------------------------------

/// Populate a [`HistoricalScenarioLibrary`] with pre-standardized eta values
/// and pre-study lag values derived from raw historical inflow observations.
///
/// For every valid window (indexed by position in `window_years`) and every
/// study stage, this function computes the standardized noise value
///
/// ```text
/// η = (obs - b - Σ ψ[ℓ] · raw_lag[ℓ]) / σ
/// ```
///
/// where `raw_lag[ℓ]` is the **raw historical inflow** at lag `ℓ` (BR6 —
/// not a PAR-reconstructed value). When `σ = 0` and the observation matches
/// the deterministic value exactly, `η = 0.0` is stored; when they disagree,
/// `f64::NEG_INFINITY` is stored (data quality issue, caught by V2.8
/// validation in ticket-020).
///
/// The function additionally writes the `max_order` pre-study raw inflow
/// observations into `library.lag_slice_mut(w)` in the layout
/// `[lag * n_hydros + hydro]`, where lag index 0 is the most recent
/// pre-study observation (lag-1).
///
/// # Inputs
///
/// - `library` — pre-allocated library (from [`HistoricalScenarioLibrary::new`])
/// - `inflow_history` — raw observations; keyed by date's `month0()` season
/// - `hydro_ids` — canonical-order hydro entity IDs (must match `par`)
/// - `stages` — study stages (non-negative IDs) with `season_id`
/// - `par` — precomputed PAR coefficient cache
/// - `window_years` — valid starting years from window discovery (ticket-018)
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches between `library`, `par`,
/// and `stages` are detected. Does not panic for valid inputs where all
/// windows were pre-validated by ticket-018.
#[allow(clippy::too_many_lines)]
pub fn standardize_historical_windows(
    library: &mut HistoricalScenarioLibrary,
    inflow_history: &[InflowHistoryRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    window_years: &[i32],
) {
    debug_assert_eq!(
        library.n_windows(),
        window_years.len(),
        "library.n_windows() ({}) must equal window_years.len() ({})",
        library.n_windows(),
        window_years.len(),
    );
    debug_assert_eq!(
        library.n_stages(),
        stages.len(),
        "library.n_stages() ({}) must equal stages.len() ({})",
        library.n_stages(),
        stages.len(),
    );
    debug_assert_eq!(
        library.n_hydros(),
        hydro_ids.len(),
        "library.n_hydros() ({}) must equal hydro_ids.len() ({})",
        library.n_hydros(),
        hydro_ids.len(),
    );
    debug_assert_eq!(
        library.max_order(),
        par.max_order(),
        "library.max_order() ({}) must equal par.max_order() ({})",
        library.max_order(),
        par.max_order(),
    );

    let n_hydros = library.n_hydros();
    let n_stages = library.n_stages();
    let max_order = library.max_order();

    if n_hydros == 0 || n_stages == 0 || window_years.is_empty() {
        return;
    }

    // -----------------------------------------------------------------------
    // Compute n_seasons from the stage season_ids (consistent with window.rs).
    // -----------------------------------------------------------------------
    let n_seasons = stages
        .iter()
        .filter_map(|s| s.season_id)
        .max()
        .map_or(1, |m| m + 1);

    // -----------------------------------------------------------------------
    // Build flat 3D observation table indexed by (hydro_idx, year_offset, season_id).
    //
    // Layout: table[hydro_idx * n_years * n_seasons + year_offset * n_seasons + season_id]
    //
    // Missing entries are represented by f64::NAN.  Valid inflow values are
    // always finite, so NAN is a safe sentinel for "not present".
    //
    // This replaces the previous HashMap<(EntityId,i32,usize),f64> and
    // reduces lookup cost from O(1) amortised with hash overhead to O(1)
    // with a single multiply-add index computation.
    // -----------------------------------------------------------------------

    // Build hydro_id -> hydro_idx map (used only during table construction).
    let hydro_id_to_idx: std::collections::HashMap<EntityId, usize> = hydro_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Year range from the inflow history.
    let (Some(min_year), Some(max_year)) = (
        inflow_history.iter().map(|r| r.date.year()).min(),
        inflow_history.iter().map(|r| r.date.year()).max(),
    ) else {
        // Empty history — nothing to do (early-exit guard above already
        // covers the n_hydros/n_stages/window_years checks).
        return;
    };
    #[allow(clippy::cast_sign_loss)]
    let n_years = (max_year - min_year + 1) as usize;

    // Allocate flat table initialised to NAN (= missing).
    let table_size = n_hydros * n_years * n_seasons;
    let mut obs_table = vec![f64::NAN; table_size];

    // Helper: compute flat index for (hydro_idx, year, season_id).
    let table_idx = |h: usize, year: i32, s: usize| -> Option<usize> {
        if year < min_year || year > max_year || s >= n_seasons {
            return None;
        }
        #[allow(clippy::cast_sign_loss)]
        let y = (year - min_year) as usize;
        Some(h * n_years * n_seasons + y * n_seasons + s)
    };

    // Populate the table from inflow history.
    for r in inflow_history {
        let season_id = r.date.month0() as usize;
        if let Some(&h) = hydro_id_to_idx.get(&r.hydro_id) {
            if let Some(idx) = table_idx(h, r.date.year(), season_id) {
                obs_table[idx] = r.value_m3s;
            }
        }
    }

    // Inline lookup: returns the table value for (hydro_idx, year, season_id).
    // Preserves the debug_assert contract: NAN in the table indicates a missing
    // observation that window discovery should have excluded.
    let lookup = |h: usize, year: i32, season_id: usize| -> f64 {
        table_idx(h, year, season_id).map_or(0.0, |idx| {
            let v = obs_table[idx];
            if v.is_nan() { 0.0 } else { v }
        })
    };

    // -----------------------------------------------------------------------
    // Build the full observation sequence template as (year_offset, season_id).
    //
    // Indices 0..max_order   : pre-study lag seasons (oldest first)
    // Indices max_order..end : study seasons
    // -----------------------------------------------------------------------
    let full_sequence: Vec<(i32, usize)> =
        super::build_observation_sequence(stages, max_order, n_seasons);

    // -----------------------------------------------------------------------
    // Pre-allocate a reusable lag buffer for solve_par_noise calls.
    // Size: max_order. Reused across all (window, stage, hydro) iterations.
    // -----------------------------------------------------------------------
    let mut lag_buf = vec![0.0_f64; max_order.max(1)];

    // -----------------------------------------------------------------------
    // Process each window.
    // -----------------------------------------------------------------------
    for (w, &window_year) in window_years.iter().enumerate() {
        // -------------------------------------------------------------------
        // Write pre-study lag values into library.lag_slice_mut(w).
        //
        // The lag seasons are the first max_order elements of full_sequence.
        // They are in chronological order (index 0 = oldest lag = lag-max_order).
        // The lag buffer layout is [lag * n_hydros + hydro] where lag 0 = most
        // recent (lag-1). Therefore we reverse the chronological order when
        // writing: buffer[0] = full_sequence[max_order - 1] (most recent),
        //                       buffer[max_order - 1] = full_sequence[0] (oldest).
        // -------------------------------------------------------------------
        if max_order > 0 {
            let lag_slice = library.lag_slice_mut(w);
            for h in 0..n_hydros {
                for lag_buf_idx in 0..max_order {
                    // lag_buf_idx 0 = most recent pre-study lag.
                    // Corresponds to full_sequence[max_order - 1 - lag_buf_idx].
                    let seq_idx = max_order - 1 - lag_buf_idx;
                    let (year_offset, season_id) = full_sequence[seq_idx];
                    let obs_year = window_year + year_offset;
                    debug_assert!(
                        table_idx(h, obs_year, season_id).is_some_and(|i| !obs_table[i].is_nan()),
                        "missing lag observation for hydro={}, year={obs_year}, season={season_id}; \
                         window discovery should have excluded this window",
                        hydro_ids[h].0,
                    );
                    lag_slice[lag_buf_idx * n_hydros + h] = lookup(h, obs_year, season_id);
                }
            }
        }

        // -------------------------------------------------------------------
        // Compute eta for each (stage, hydro).
        //
        // Study seasons start at index max_order in full_sequence.
        // For stage t, its season is full_sequence[max_order + t].
        // The lags for stage t are the raw observations at stages t-1, t-2, ...,
        // t-order(h). These are in full_sequence at indices
        // (max_order + t - 1), (max_order + t - 2), ..., (max_order + t - order(h)).
        // When the index < max_order we use the pre-study lag sequence.
        // -------------------------------------------------------------------
        for t in 0..n_stages {
            let eta_slice = library.eta_slice_mut(w, t);
            for h in 0..n_hydros {
                // Look up the target raw observation for this (window, stage, hydro).
                let (year_offset, season_id) = full_sequence[max_order + t];
                let obs_year = window_year + year_offset;
                debug_assert!(
                    table_idx(h, obs_year, season_id).is_some_and(|i| !obs_table[i].is_nan()),
                    "missing study observation for hydro={}, year={obs_year}, season={season_id}; \
                     window discovery should have excluded this window",
                    hydro_ids[h].0,
                );
                let target = lookup(h, obs_year, season_id);

                // Build the lag vector from raw historical observations (BR6).
                // lag_buf[0] = raw inflow at t-1, lag_buf[1] = raw at t-2, etc.
                let order_h = par.order(h);
                for (l, slot) in lag_buf.iter_mut().enumerate().take(order_h) {
                    // Sequence index for lag l+1: position (max_order + t) - (l+1).
                    // max_order + t >= l + 1 holds because l < order_h <= max_order,
                    // so the subtraction is always non-negative in usize.
                    debug_assert!(
                        max_order + t >= l + 1,
                        "lag index underflow: t={t}, l={l}, max_order={max_order}",
                    );
                    let seq_idx = max_order + t - (l + 1);
                    let (lag_year_offset, lag_season_id) = full_sequence[seq_idx];
                    let lag_year = window_year + lag_year_offset;
                    debug_assert!(
                        table_idx(h, lag_year, lag_season_id)
                            .is_some_and(|i| !obs_table[i].is_nan()),
                        "missing lag observation for hydro={}, year={lag_year}, \
                         season={lag_season_id}; window discovery should have excluded \
                         this window",
                        hydro_ids[h].0,
                    );
                    *slot = lookup(h, lag_year, lag_season_id);
                }

                let det_base = par.deterministic_base(t, h);
                let psi = par.psi_slice(t, h);
                let sigma = par.sigma(t, h);

                let eta = solve_par_noise(det_base, psi, order_h, &lag_buf, sigma, target);

                eta_slice[h] = eta;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// validate_historical_library
// ---------------------------------------------------------------------------

/// Validate a [`HistoricalScenarioLibrary`] against construction inputs.
///
/// This is the Tier 2 validation gate for the historical scenario library.
/// It runs after window discovery (ticket-018) and eta standardization
/// (ticket-019), confirming that the library is well-formed before it is
/// stored on `StudySetup`.
///
/// Validation uses **fail-fast** semantics: the first failed check immediately
/// returns `Err`. Warnings (V2.6) are emitted via `tracing::warn!` and do not
/// abort construction.
///
/// ## Checks performed
///
/// | ID  | Kind    | Description                                                   |
/// |-----|---------|---------------------------------------------------------------|
/// | V2.1 | Error  | Every study stage must have `season_id: Some(_)`.             |
/// | V2.9 | Error  | `hydro_ids.len()` must equal `library.n_hydros()`.            |
/// | V2.5 | Error  | At least one window must be discovered when `user_pool` is `None`. |
/// | V2.3 | Error  | No eta value in the library may be `f64::NEG_INFINITY`.       |
/// | V2.6 | Warning| `library.n_windows() < forward_passes` — log a warning.      |
/// | V2.2 | Assert | Window contiguity — `debug_assert` only (construction invariant). |
/// | V2.4 | Assert | User pool validity — `debug_assert` only (construction invariant). |
/// | V2.7 | Assert | Lag warmup sufficiency — `debug_assert` only (construction invariant). |
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] with a message prefixed by
/// the check ID (e.g., `"V2.1: ..."`) for the first failed error check.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowHistoryRow};
/// use cobre_core::temporal::{
///     Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
///     StageStateConfig,
/// };
/// use chrono::NaiveDate;
/// use cobre_stochastic::{HistoricalScenarioLibrary, sampling::historical::validate_historical_library};
///
/// let lib = HistoricalScenarioLibrary::new(3, 1, 2, 1, vec![1990, 1995, 2000]);
/// let stage = Stage {
///     index: 0,
///     id: 0,
///     start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "B".to_string(), duration_hours: 720.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 1, noise_method: NoiseMethod::Saa },
/// };
/// let hydro_ids = [EntityId(1), EntityId(2)];
/// let result = validate_historical_library(&lib, &[], &hydro_ids, &[stage], 1, None, 5);
/// assert!(result.is_ok());
/// ```
pub fn validate_historical_library(
    library: &HistoricalScenarioLibrary,
    _inflow_history: &[InflowHistoryRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    max_par_order: usize,
    user_pool: Option<&HistoricalYears>,
    forward_passes: u32,
) -> Result<(), StochasticError> {
    // -----------------------------------------------------------------------
    // V2.1 — Season alignment: every study stage must have season_id: Some(_).
    // -----------------------------------------------------------------------
    for stage in stages {
        if stage.season_id.is_none() {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "V2.1: stage {} (index {}) has season_id: None; \
                     all study stages must have a season_id assigned",
                    stage.id, stage.index,
                ),
            });
        }
    }

    // -----------------------------------------------------------------------
    // V2.9 — Per-class scope: hydro_ids length must equal library.n_hydros().
    // -----------------------------------------------------------------------
    if hydro_ids.len() != library.n_hydros() {
        return Err(StochasticError::InsufficientData {
            context: format!(
                "V2.9: hydro_ids slice length ({}) does not match \
                 library.n_hydros() ({})",
                hydro_ids.len(),
                library.n_hydros(),
            ),
        });
    }

    // -----------------------------------------------------------------------
    // V2.5 — Auto-discovery minimum: at least 1 window when user_pool is None.
    // -----------------------------------------------------------------------
    if user_pool.is_none() && library.n_windows() == 0 {
        return Err(StochasticError::InsufficientData {
            context: "V2.5: historical library has 0 windows after auto-discovery; \
                      at least 1 complete historical window is required"
                .to_string(),
        });
    }

    // -----------------------------------------------------------------------
    // V2.3 / V2.8 — Complete coverage: no eta value may be f64::NEG_INFINITY.
    //
    // NEG_INFINITY is the sentinel written by standardize_historical_windows
    // when sigma=0 but the historical observation does not match the
    // deterministic base (data quality issue).
    // -----------------------------------------------------------------------
    for w in 0..library.n_windows() {
        for t in 0..library.n_stages() {
            let eta = library.eta_slice(w, t);
            for (h, &value) in eta.iter().enumerate() {
                if value == f64::NEG_INFINITY || value.is_nan() {
                    return Err(StochasticError::InsufficientData {
                        context: format!(
                            "V2.3: historical library contains non-finite eta (NEG_INFINITY or NaN) \
                             at window {w}, stage {t}, hydro {h} — sigma=0 with \
                             non-matching historical observation or numerical failure",
                        ),
                    });
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // V2.2 — Contiguous windows (construction invariant — debug_assert only).
    // V2.4 — User pool validity (construction invariant — debug_assert only).
    // V2.7 — Lag warmup sufficiency (constructor enforces — debug_assert only).
    // -----------------------------------------------------------------------
    debug_assert!(
        library.max_order() == max_par_order,
        "V2.7: library.max_order() ({}) must equal max_par_order ({max_par_order})",
        library.max_order(),
    );

    // -----------------------------------------------------------------------
    // V2.6 — Pool size warning: fewer windows than forward passes.
    // -----------------------------------------------------------------------
    if library.n_windows() < forward_passes as usize {
        tracing::warn!(
            n_windows = library.n_windows(),
            forward_passes = forward_passes,
            "historical library has fewer windows ({}) than forward passes ({}); \
             windows will be reused across forward passes",
            library.n_windows(),
            forward_passes,
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use super::HistoricalScenarioLibrary;

    #[test]
    fn test_new_allocates_correct_sizes() {
        let lib = HistoricalScenarioLibrary::new(3, 12, 5, 2, vec![1990, 1995, 2000]);

        assert_eq!(lib.n_windows(), 3);
        assert_eq!(lib.n_stages(), 12);
        assert_eq!(lib.n_hydros(), 5);
        assert_eq!(lib.max_order(), 2);

        // Verify slice lengths for each accessor.
        assert_eq!(
            lib.eta_slice(0, 0).len(),
            5,
            "eta slice length must equal n_hydros"
        );
        assert_eq!(
            lib.eta_slice(2, 11).len(),
            5,
            "eta slice length must equal n_hydros"
        );
        assert_eq!(
            lib.lag_slice(0).len(),
            2 * 5,
            "lag slice length must equal max_order * n_hydros"
        );
        assert_eq!(
            lib.lag_slice(2).len(),
            2 * 5,
            "lag slice length must equal max_order * n_hydros"
        );
    }

    #[test]
    fn test_eta_roundtrip() {
        let mut lib = HistoricalScenarioLibrary::new(2, 3, 4, 1, vec![2000, 2001]);

        let values = [1.0_f64, 2.0, 3.0, 4.0];
        lib.eta_slice_mut(1, 2).copy_from_slice(&values);

        assert_eq!(
            lib.eta_slice(1, 2),
            &values,
            "eta_slice must return the values written via eta_slice_mut"
        );

        // Confirm that other (window, stage) cells were not disturbed.
        assert_eq!(
            lib.eta_slice(0, 0),
            &[0.0, 0.0, 0.0, 0.0],
            "untouched eta cells must remain zero"
        );
        assert_eq!(
            lib.eta_slice(1, 0),
            &[0.0, 0.0, 0.0, 0.0],
            "untouched eta cells must remain zero"
        );
    }

    #[test]
    fn test_lag_roundtrip() {
        let mut lib = HistoricalScenarioLibrary::new(2, 3, 3, 2, vec![1990, 1991]);

        // lag_slice length = max_order * n_hydros = 2 * 3 = 6
        let values = [10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0];
        lib.lag_slice_mut(0).copy_from_slice(&values);

        assert_eq!(
            lib.lag_slice(0),
            &values,
            "lag_slice must return the values written via lag_slice_mut"
        );
        assert_eq!(
            lib.lag_slice(0).len(),
            6,
            "lag slice length must equal max_order * n_hydros"
        );

        // Confirm window 1 was not disturbed.
        assert_eq!(
            lib.lag_slice(1),
            &[0.0; 6],
            "untouched lag cells must remain zero"
        );
    }

    #[test]
    fn test_window_years() {
        let lib = HistoricalScenarioLibrary::new(3, 1, 1, 1, vec![1990, 1995, 2000]);

        assert_eq!(lib.window_year(0), 1990);
        assert_eq!(lib.window_year(1), 1995);
        assert_eq!(lib.window_year(2), 2000);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HistoricalScenarioLibrary>();
    }

    // -----------------------------------------------------------------------
    // Helpers for standardize_historical_windows tests
    // -----------------------------------------------------------------------

    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        scenario::{InflowHistoryRow, InflowModel},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        },
    };

    use super::{Stage, standardize_historical_windows};
    use crate::par::precompute::PrecomputedPar;

    /// Build a monthly stage with the given array index and 0-based `season_id` (0=Jan..11=Dec).
    ///
    /// Uses a 12-season cycle so that wrap-around (Dec→Jan) causes a year-offset
    /// increment in `build_observation_sequence`, matching the window.rs tests.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn make_monthly_stage(index: usize, season_id: usize) -> Stage {
        let month = (season_id as u32) + 1;
        Stage {
            index,
            id: index as i32,
            start_date: NaiveDate::from_ymd_opt(2024, month, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, month, 28).unwrap(),
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
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
        }
    }

    /// Build a row keyed by `month0` (0-based month: 0=Jan, 11=Dec).
    ///
    /// The lookup in `standardize_historical_windows` uses `date.month0()` as
    /// the `season_id`. This helper makes that mapping explicit.
    fn make_row(hydro_id: EntityId, year: i32, month0: u32, value: f64) -> InflowHistoryRow {
        InflowHistoryRow {
            hydro_id,
            date: NaiveDate::from_ymd_opt(year, month0 + 1, 1).unwrap(),
            value_m3s: value,
        }
    }

    /// Build 12 monthly study stages covering a full calendar year (seasons 0-11).
    ///
    /// With `max_order=0` and these 12 stages, the year-offset increments once
    /// at the Jan→Feb transition when the season wraps from 11→0, placing study
    /// stages at year `window_year + 1`.
    fn twelve_monthly_stages() -> Vec<Stage> {
        (0..12).map(|i| make_monthly_stage(i, i)).collect()
    }

    // -----------------------------------------------------------------------
    // Test 1: AR(0) single hydro standardization
    // -----------------------------------------------------------------------

    /// Given 1 hydro with AR(0), mean=100, std=30, 12 study stages (full year),
    /// and window year 1990 with observations [120.0, 90.0, ...], the eta values
    /// at stage 0 and 1 must be (120-100)/30 and (90-100)/30.
    ///
    /// With 12 monthly stages (`season_ids` 0-11), `n_seasons`=12. `max_order`=0.
    /// Full sequence: [(0,0),(0,1),...,(0,11)] — all `year_offset`=0 because the
    /// sequence 0→1→...→11 never wraps backwards. So all study observations
    /// are in year `window_year` itself.
    #[test]
    fn test_ar0_standardization() {
        let hydro = EntityId(1);
        // Use 2 study stages with season_ids 0 and 1 (Jan and Feb).
        // n_seasons = max(0,1)+1 = 2. Full sequence with max_order=0: [(0,0),(0,1)].
        // Year offsets: 0→1 is increasing (no wrap), both stay at offset=0.
        // Observations are at year=window_year=1990.
        let stages = vec![make_monthly_stage(0, 0), make_monthly_stage(1, 1)];
        let models = vec![
            InflowModel {
                hydro_id: hydro,
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: hydro,
                stage_id: 1,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &[hydro]).unwrap();

        // Observations at window_year=1990, season 0 (Jan) and season 1 (Feb).
        let history = vec![
            make_row(hydro, 1990, 0, 120.0),
            make_row(hydro, 1990, 1, 90.0),
        ];

        let mut lib = HistoricalScenarioLibrary::new(1, 2, 1, 0, vec![1990]);
        standardize_historical_windows(&mut lib, &history, &[hydro], &stages, &par, &[1990]);

        let expected_0 = (120.0 - 100.0) / 30.0;
        let expected_1 = (90.0 - 100.0) / 30.0;

        assert!(
            (lib.eta_slice(0, 0)[0] - expected_0).abs() < 1e-10,
            "AR(0) eta stage 0: expected {expected_0}, got {}",
            lib.eta_slice(0, 0)[0]
        );
        assert!(
            (lib.eta_slice(0, 1)[0] - expected_1).abs() < 1e-10,
            "AR(0) eta stage 1: expected {expected_1}, got {}",
            lib.eta_slice(0, 1)[0]
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: AR(1) uses RAW historical lags (not reconstructed)
    // -----------------------------------------------------------------------

    /// Single hydro, AR(1), `psi_orig`=0.5 in original units, base=80, sigma=25.
    ///
    /// Uses 12 monthly stages so that the lag season (one step before Jan) is Dec.
    /// With `n_seasons`=12 and `max_order`=1:
    ///   - Lag season: (0 - 1 + 12) % 12 = 11 (Dec)
    ///   - Full sequence: [(0,11),(1,0),(1,1),...,(1,11)]
    ///     * season 11→0 wraps → `year_offset` becomes 1
    ///   - Lag observation: (`window_year` + 0, season 11) = (1990, Dec) → month0=11
    ///   - Stage 0 observation: (1990 + 1, season 0) = (1991, Jan) → month0=0
    ///   - Stage 1 observation: (1991, Feb) → month0=1
    ///
    /// For stage 0: eta = (130 - 80 - 0.5*110) / 25 = -5/25 = -0.2
    /// For stage 1: lags are RAW → lag[0] = 130.0 (Jan 1991 raw observation)
    ///              eta = (95 - 80 - 0.5*130) / 25 = -50/25 = -2.0
    ///
    /// PAR parametrisation: mean=160, std=25, `psi_star`=0.5 (when stds equal,
    /// `psi_orig` = `psi_star`). base = mean - `psi_orig`*`mean_lag` = 160 - 0.5*160 = 80.
    #[test]
    fn test_ar1_standardization_uses_raw_lags() {
        let hydro = EntityId(1);
        // 12 monthly stages. n_seasons=12.
        let stages = twelve_monthly_stages();

        // Build PAR models for study stages (stage_id 0-11) plus one pre-study
        // stage (stage_id=-1, season 11) needed for coefficient unit conversion.
        let all_stage_ids: Vec<i32> = std::iter::once(-1_i32).chain(0..12_i32).collect();
        let models: Vec<InflowModel> = all_stage_ids
            .iter()
            .map(|&sid| InflowModel {
                hydro_id: hydro,
                stage_id: sid,
                mean_m3s: 160.0,
                std_m3s: 25.0,
                ar_coefficients: vec![0.5],
                residual_std_ratio: 1.0,
            })
            .collect();
        let par = PrecomputedPar::build(&models, &stages, &[hydro]).unwrap();

        // Verify precomputed values: psi_orig = 0.5 * 25/25 = 0.5;
        // base = 160 - 0.5*160 = 80; sigma = 25.
        assert!(
            (par.deterministic_base(0, 0) - 80.0).abs() < 1e-10,
            "expected base=80, got {}",
            par.deterministic_base(0, 0)
        );
        assert!(
            (par.sigma(0, 0) - 25.0).abs() < 1e-10,
            "expected sigma=25, got {}",
            par.sigma(0, 0)
        );

        // Window year 1990, max_order=1:
        //   lag: (1990, season 11 = Dec) → 110.0
        //   stage 0: (1991, season 0 = Jan) → 130.0
        //   stage 1: (1991, season 1 = Feb) → 95.0
        //   (remaining study stages: use 100.0, not used in assertions)
        let mut history = vec![
            make_row(hydro, 1990, 11, 110.0), // Dec 1990 = pre-study lag
            make_row(hydro, 1991, 0, 130.0),  // Jan 1991 = stage 0
            make_row(hydro, 1991, 1, 95.0),   // Feb 1991 = stage 1
        ];
        for m in 2..12_u32 {
            history.push(make_row(hydro, 1991, m, 100.0));
        }

        let mut lib = HistoricalScenarioLibrary::new(1, 12, 1, 1, vec![1990]);
        standardize_historical_windows(&mut lib, &history, &[hydro], &stages, &par, &[1990]);

        // Pre-study lag buffer: lag 0 = most recent = Dec 1990 = 110.0.
        let lag_slice = lib.lag_slice(0);
        assert!(
            (lag_slice[0] - 110.0).abs() < 1e-10,
            "pre-study lag[0] expected 110.0, got {}",
            lag_slice[0]
        );

        // Stage 0: eta = (130 - 80 - 0.5*110) / 25 = -5/25 = -0.2
        let eta_0 = lib.eta_slice(0, 0)[0];
        let expected_0 = (130.0 - 80.0 - 0.5 * 110.0) / 25.0;
        assert!(
            (eta_0 - expected_0).abs() < 1e-10,
            "eta stage 0: expected {expected_0}, got {eta_0}"
        );

        // Stage 1: raw lag = 130.0 (Jan 1991, NOT reconstructed).
        // eta = (95 - 80 - 0.5*130) / 25 = (95 - 145) / 25 = -2.0
        let eta_1 = lib.eta_slice(0, 1)[0];
        let expected_1 = (95.0 - 80.0 - 0.5 * 130.0) / 25.0;
        assert!(
            (eta_1 - expected_1).abs() < 1e-10,
            "eta stage 1: expected {expected_1}, got {eta_1} (must use RAW lag=130.0)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: 2 hydros, 2 windows — all slices populated independently
    // -----------------------------------------------------------------------

    /// Two hydros, two window years, AR(0). All four (window,stage) slices must
    /// be independently populated with the correct eta values.
    ///
    /// Uses `season_ids` 0 and 1 (`n_seasons`=2). With `max_order`=0 and seasons 0→1
    /// (increasing, no wrap), all `year_offset`s are 0. Observations are at
    /// year = `window_year` itself.
    #[test]
    fn test_multi_hydro_multi_window() {
        let h1 = EntityId(1);
        let h2 = EntityId(2);
        // 2 stages, season_ids 0 and 1 (n_seasons=2, no wrap → year_offset=0).
        let stages = vec![make_monthly_stage(0, 0), make_monthly_stage(1, 1)];

        let models = vec![
            InflowModel {
                hydro_id: h1,
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: h1,
                stage_id: 1,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: h2,
                stage_id: 0,
                mean_m3s: 200.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: h2,
                stage_id: 1,
                mean_m3s: 200.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &[h1, h2]).unwrap();

        // Observations at year=window_year (year_offset=0 for both stages).
        let history = vec![
            // Window 1990: year_offset=0 → obs at 1990
            make_row(h1, 1990, 0, 110.0),
            make_row(h1, 1990, 1, 90.0),
            make_row(h2, 1990, 0, 220.0),
            make_row(h2, 1990, 1, 180.0),
            // Window 1991: obs at 1991
            make_row(h1, 1991, 0, 105.0),
            make_row(h1, 1991, 1, 95.0),
            make_row(h2, 1991, 0, 210.0),
            make_row(h2, 1991, 1, 190.0),
        ];

        let mut lib = HistoricalScenarioLibrary::new(2, 2, 2, 0, vec![1990, 1991]);
        standardize_historical_windows(&mut lib, &history, &[h1, h2], &stages, &par, &[1990, 1991]);

        // All 4 (window, stage) slices have length 2 (n_hydros).
        for w in 0..2 {
            for t in 0..2 {
                assert_eq!(
                    lib.eta_slice(w, t).len(),
                    2,
                    "eta slice (w={w}, t={t}) must have length n_hydros=2"
                );
            }
        }

        // Window 0, stage 0: h1=(110-100)/10=1.0, h2=(220-200)/20=1.0
        let e00 = lib.eta_slice(0, 0);
        assert!((e00[0] - 1.0).abs() < 1e-10, "w=0,t=0,h=0: {}", e00[0]);
        assert!((e00[1] - 1.0).abs() < 1e-10, "w=0,t=0,h=1: {}", e00[1]);

        // Window 1, stage 1: h1=(95-100)/10=-0.5, h2=(190-200)/20=-0.5
        let e11 = lib.eta_slice(1, 1);
        assert!((e11[0] - (-0.5)).abs() < 1e-10, "w=1,t=1,h=0: {}", e11[0]);
        assert!((e11[1] - (-0.5)).abs() < 1e-10, "w=1,t=1,h=1: {}", e11[1]);
    }

    // -----------------------------------------------------------------------
    // Test 4: pre-study lag buffer populated correctly
    // -----------------------------------------------------------------------

    /// Single hydro, `max_order`=2 (AR(2) dummy), full 12-stage monthly year.
    ///
    /// With `n_seasons`=12 and `max_order`=2, the lag seasons for stage 0 (Jan) are:
    ///   - lag-2 (oldest, buf index 1): season (0-2+12)%12 = 10 (Nov) at year+0
    ///   - lag-1 (most recent, buf index 0): season (0-1+12)%12 = 11 (Dec) at year+0
    ///
    /// Full sequence: [(0,10),(0,11),(1,0),(1,1),...,(1,11)]
    ///   Wrap at 11→0 increments `year_offset` to 1 for study stages.
    ///
    /// The lag buffer layout is [`lag*n_hydros + h`] where lag=0 = most recent:
    ///   `lag_slice`[0] = Dec 1990 = 66.0
    ///   `lag_slice`[1] = Nov 1990 = 55.0
    #[test]
    fn test_pre_study_lags_populated() {
        let hydro = EntityId(1);
        let stages = twelve_monthly_stages();

        // AR(2) model: need ar_coefficients of length 2 so par.max_order()=2.
        // Use psi=[0.0, 0.0] so the AR terms do not affect eta.
        let study_models: Vec<InflowModel> = stages
            .iter()
            .map(|s| InflowModel {
                hydro_id: hydro,
                stage_id: s.id,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![0.0, 0.0],
                residual_std_ratio: 1.0,
            })
            .collect();
        // Also add pre-study models for stage_id -1 (Dec) and -2 (Nov)
        // so the coefficient unit conversion can resolve lag stds.
        let pre_study_models = vec![
            InflowModel {
                hydro_id: hydro,
                stage_id: -1,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![0.0, 0.0],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: hydro,
                stage_id: -2,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![0.0, 0.0],
                residual_std_ratio: 1.0,
            },
        ];
        let mut all_models = pre_study_models;
        all_models.extend(study_models);
        let par = PrecomputedPar::build(&all_models, &stages, &[hydro]).unwrap();
        assert_eq!(par.max_order(), 2, "expected par.max_order()=2");

        // Window year 1990:
        //   lag-2 (seq_idx=0): (1990+0, season 10 = Nov) → 55.0
        //   lag-1 (seq_idx=1): (1990+0, season 11 = Dec) → 66.0
        //   study stages at year_offset=1 (1991): months 0-11 → all 100.0
        let mut history = vec![
            make_row(hydro, 1990, 10, 55.0), // Nov 1990 = lag-2
            make_row(hydro, 1990, 11, 66.0), // Dec 1990 = lag-1
        ];
        for m in 0..12_u32 {
            history.push(make_row(hydro, 1991, m, 100.0));
        }

        let mut lib = HistoricalScenarioLibrary::new(1, 12, 1, 2, vec![1990]);
        standardize_historical_windows(&mut lib, &history, &[hydro], &stages, &par, &[1990]);

        let lag = lib.lag_slice(0);
        // lag[0] = most recent (lag-1) = Dec 1990 = 66.0
        assert!(
            (lag[0] - 66.0).abs() < 1e-10,
            "lag[0] (most recent) expected 66.0, got {}",
            lag[0]
        );
        // lag[1] = lag-2 = Nov 1990 = 55.0
        assert!(
            (lag[1] - 55.0).abs() < 1e-10,
            "lag[1] (lag-2) expected 55.0, got {}",
            lag[1]
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: sigma=0 matching deterministic value stores eta=0.0
    // -----------------------------------------------------------------------

    /// Single hydro, single stage, sigma=0, observation matches deterministic value.
    ///
    /// With 1 stage (`season_id`=0, `n_seasons`=1) and `max_order`=0:
    ///   Full sequence: [(0,0)]. `year_offset`=0. Observation at year=`window_year`.
    #[test]
    fn test_sigma_zero_returns_zero_eta() {
        let hydro = EntityId(1);
        // Single stage, season_id=0.
        let stages = vec![make_monthly_stage(0, 0)];
        let models = vec![InflowModel {
            hydro_id: hydro,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        }];
        let par = PrecomputedPar::build(&models, &stages, &[hydro]).unwrap();

        // n_seasons=1, max_order=0: full_sequence = [(0,0)].
        // Observation at (window_year + 0, season 0) = (2000, month 0 = Jan 2000).
        let history = vec![make_row(hydro, 2000, 0, 50.0)];

        let mut lib = HistoricalScenarioLibrary::new(1, 1, 1, 0, vec![2000]);
        standardize_historical_windows(&mut lib, &history, &[hydro], &stages, &par, &[2000]);

        let eta = lib.eta_slice(0, 0)[0];
        assert!(
            eta == 0.0,
            "sigma=0 with obs matching deterministic value must give eta=0.0, got {eta}"
        );
    }

    // -----------------------------------------------------------------------
    // validate_historical_library tests
    // -----------------------------------------------------------------------

    use super::validate_historical_library;
    use crate::StochasticError;

    /// Build a minimal valid Stage with `season_id`: `Some(season)`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn make_validate_stage(index: usize, season_id: Option<usize>) -> Stage {
        Stage {
            index,
            id: index as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id,
            blocks: vec![Block {
                index: 0,
                name: "B".to_string(),
                duration_hours: 720.0,
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
        }
    }

    /// Given a valid library with 5 windows, 12 stages, 3 hydros, all finite
    /// eta values, `validate_historical_library` returns `Ok(())`.
    #[test]
    fn test_valid_library_passes() {
        let n_windows = 5;
        let n_stages = 12;
        let n_hydros = 3;
        let lib = HistoricalScenarioLibrary::new(
            n_windows,
            n_stages,
            n_hydros,
            1,
            (1990..1995).collect(),
        );
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| make_validate_stage(i, Some(i % 12)))
            .collect();
        let hydro_ids: Vec<EntityId> = (1..=3).map(EntityId).collect();

        let result = validate_historical_library(&lib, &[], &hydro_ids, &stages, 1, None, 5);
        assert!(result.is_ok(), "expected Ok(()), got: {result:?}");
    }

    /// Given a library where `eta_slice(2, 5)[1]` is `f64::NEG_INFINITY`,
    /// `validate_historical_library` returns `Err` with a message containing
    /// "V2.3" and `"NEG_INFINITY"`.
    #[test]
    fn test_neg_infinity_eta_fails_v2_3() {
        let n_windows = 5;
        let n_stages = 12;
        let n_hydros = 3;
        let mut lib = HistoricalScenarioLibrary::new(
            n_windows,
            n_stages,
            n_hydros,
            1,
            (1990..1995).collect(),
        );
        // Inject NEG_INFINITY at window=2, stage=5, hydro=1.
        lib.eta_slice_mut(2, 5)[1] = f64::NEG_INFINITY;

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| make_validate_stage(i, Some(i % 12)))
            .collect();
        let hydro_ids: Vec<EntityId> = (1..=3).map(EntityId).collect();

        let result = validate_historical_library(&lib, &[], &hydro_ids, &stages, 1, None, 5);
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V2.3"),
                    "expected message to contain 'V2.3', got: {context}"
                );
                assert!(
                    context.contains("NEG_INFINITY"),
                    "expected message to contain 'NEG_INFINITY', got: {context}"
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    /// Given a stage with `season_id: None`, `validate_historical_library`
    /// returns `Err` with a message containing "V2.1" and `"season_id"`.
    #[test]
    fn test_missing_season_id_fails_v2_1() {
        let lib = HistoricalScenarioLibrary::new(1, 2, 2, 0, vec![1990]);
        let stages = vec![
            make_validate_stage(0, Some(0)),
            make_validate_stage(1, None), // missing season_id
        ];
        let hydro_ids = vec![EntityId(1), EntityId(2)];

        let result = validate_historical_library(&lib, &[], &hydro_ids, &stages, 0, None, 1);
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V2.1"),
                    "expected message to contain 'V2.1', got: {context}"
                );
                assert!(
                    context.contains("season_id"),
                    "expected message to contain 'season_id', got: {context}"
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    /// Given `library.n_hydros() = 3` but `hydro_ids.len() = 4`,
    /// `validate_historical_library` returns `Err` with a message containing
    /// "V2.9".
    #[test]
    fn test_hydro_count_mismatch_fails_v2_9() {
        let lib = HistoricalScenarioLibrary::new(1, 1, 3, 0, vec![1990]);
        let stages = vec![make_validate_stage(0, Some(0))];
        // 4 hydro IDs but library has n_hydros=3.
        let hydro_ids = vec![EntityId(1), EntityId(2), EntityId(3), EntityId(4)];

        let result = validate_historical_library(&lib, &[], &hydro_ids, &stages, 0, None, 1);
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V2.9"),
                    "expected message to contain 'V2.9', got: {context}"
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    /// Given `library.n_windows() = 5` and `forward_passes = 20`,
    /// `validate_historical_library` returns `Ok(())` (warning is emitted
    /// via tracing but does not abort construction).
    #[test]
    fn test_pool_warning_path_returns_ok() {
        let lib = HistoricalScenarioLibrary::new(5, 1, 2, 0, (1990..1995).collect());
        let stages = vec![make_validate_stage(0, Some(0))];
        let hydro_ids = vec![EntityId(1), EntityId(2)];

        // 5 windows < 20 forward passes triggers warn! but must still return Ok(()).
        let result = validate_historical_library(&lib, &[], &hydro_ids, &stages, 0, None, 20);
        assert!(
            result.is_ok(),
            "warning path must return Ok(()), got: {result:?}"
        );
    }
}
