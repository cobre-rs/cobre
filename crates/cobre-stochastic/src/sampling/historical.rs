//! `Historical` scenario sampling scheme — library type and eta pre-standardization.
//!
//! [`HistoricalScenarioLibrary`] stores pre-standardized eta values for
//! historical inflow windows.
//!
//! The [`standardize_historical_windows`] function populates the library by
//! inverting the PAR(p) model with a **rolling lag chain seeded from the
//! derived lag seed** (not the window's own pre-study observations), so a
//! forward pass starting from that same seed exactly reconstructs the raw
//! historical target at every stage of the replay.
//!
//! ## Replay correctness
//!
//! With `x₀` the derived lag seed at stage 0, the η stored for window `w` at
//! stage `t` satisfies
//!
//! ```text
//! η(w, t) = (target(w, t) - b(t) - Σ ψ(t)[ℓ] · lag(w, t)[ℓ]) / σ(t)
//! ```
//!
//! where `lag(w, t)` advances `x₀` through window `w`'s raw historical targets
//! at stages `0..t` — exactly what the forward accumulator produces from `x₀`.
//! Replay is therefore exact only if the forward pass starts from the same
//! seed `x₀` verbatim; `ClassSampler::Historical::apply_initial_state` is a
//! no-op, so that is the caller's responsibility, not this module's.
//!
//! The `eta` buffer uses **window-major** layout
//! (`eta[window * n_stages * n_hydros + stage * n_hydros + hydro]`), matching
//! sequential stage iteration within a window (same access pattern as
//! [`PrecomputedPar`]).
//!
//! Initial inflow lags are **NOT** stored on the library nor injected into the
//! solver state vector. They come from the derived lag seed for every scenario
//! regardless of the window replayed; η inversion uses that seeded rolling chain.
//!
//! [`PrecomputedPar`]: crate::par::precompute::PrecomputedPar

use chrono::NaiveDate;
use cobre_core::{
    EntityId,
    scenario::{HistoricalYears, InflowHistoryRow},
    temporal::{SeasonMap, Stage, StageLagTransition},
};
use siphasher::sip::SipHasher13;
use std::hash::Hasher as _;

use crate::{StochasticError, par::precompute::PrecomputedPar, seeds::DerivedSeed};

use super::{eta_inversion::run_eta_inversion, window::history_row_key};

// ---------------------------------------------------------------------------
// HistoricalScenarioLibrary
// ---------------------------------------------------------------------------

/// Pre-standardized eta store for historical scenario windows.
///
/// A pure data container: population is the eta-standardisation pass's job,
/// selection the `ClassSampler::Historical` variant's.
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
    eta: Box<[f64]>,
    window_years: Box<[i32]>,
    n_windows: usize,
    n_stages: usize,
    n_hydros: usize,
    max_order: usize,
    /// SipHash-1-3 fingerprint of the seeding derived lag values; `0` until
    /// [`standardize_historical_windows`] writes it. A mismatch against a fresh
    /// digest means the library is stale and must be re-standardised.
    seed_digest: u64,
}

impl HistoricalScenarioLibrary {
    /// Construct a new library with zero-filled buffers.
    ///
    /// # Parameters
    ///
    /// - `max_order` — lag-state depth that η inversion seeds from the stage-0 seed
    /// - `window_years` — starting year per window (length must equal `n_windows`)
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
            window_years: window_years.into_boxed_slice(),
            n_windows,
            n_stages,
            n_hydros,
            max_order,
            seed_digest: 0,
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

    /// Returns the lag-state depth that η inversion seeds from the stage-0 seed.
    #[must_use]
    #[inline]
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Returns the SipHash-1-3 fingerprint of the seeding derived lag values,
    /// or `0` before [`standardize_historical_windows`] has run.
    #[must_use]
    #[inline]
    pub fn seed_digest(&self) -> u64 {
        self.seed_digest
    }

    /// Returns the starting year label for window `window`.
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
}

// ---------------------------------------------------------------------------
// check_historical_structure
// ---------------------------------------------------------------------------

/// Unforgeable evidence that [`check_historical_structure`] passed for the
/// `stages` and `hydro_ids` it carries.
///
/// Only [`check_historical_structure`] constructs it.
/// [`standardize_historical_windows`] requires it and reads exactly the slices
/// that were checked.
#[derive(Debug)]
pub struct HistoricalStructureProof<'a> {
    stages: &'a [Stage],
    hydro_ids: &'a [EntityId],
}

/// Check the structural preconditions of historical standardization.
///
/// ## Checks performed
///
/// | ID  | Kind    | Description                                                   |
/// |-----|---------|---------------------------------------------------------------|
/// | V2.1 | Error  | Every study stage must have `season_id: Some(_)`.             |
/// | V2.9 | Error  | `hydro_ids.len()` must equal `library.n_hydros()`.            |
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] with a message prefixed by
/// the check ID (e.g., `"V2.1: ..."`) for the first failed check.
pub fn check_historical_structure<'a>(
    library: &HistoricalScenarioLibrary,
    hydro_ids: &'a [EntityId],
    stages: &'a [Stage],
) -> Result<HistoricalStructureProof<'a>, StochasticError> {
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

    Ok(HistoricalStructureProof { stages, hydro_ids })
}

// ---------------------------------------------------------------------------
// standardize_historical_windows
// ---------------------------------------------------------------------------

/// Populate a [`HistoricalScenarioLibrary`] with pre-standardized η values whose
/// lag chain is seeded from `seed` ([`DerivedSeed`]) and advanced by the same
/// accumulate/finalize pattern as `standardize_external_inflow`.
///
/// Replay is exact only if the forward pass starts from the same derived seed;
/// see the module docs for the inductive argument. The η values depend on
/// `seed`; a change requires re-standardising, which `seed_digest` lets callers
/// detect.
///
/// The full accumulate/finalize/spillover/downstream-ring pattern is supported,
/// via the same [`advance_lag_chain`](crate::par::advance_lag_chain) kernel the
/// forward pass and `standardize_external_inflow` route through — including
/// monthly→quarterly multi-resolution grids.
///
/// # Inputs
///
/// - `structure` — proof from [`check_historical_structure`]; carries the study
///   stages and the canonical-order hydro entity IDs (must match `par`)
/// - `season_map` — observation-date → season mapping, resolved exactly as in
///   [`discover_historical_windows`](super::window::discover_historical_windows);
///   unmappable observations are skipped.
/// - `seed` — stage-0 lag/accumulator seed; see [`DerivedSeed`]. Absent lag
///   slots default to `0.0`.
/// - `stage_lag_transitions` — empty, or one per stage.
/// - `downstream_par_order` — PAR order of the downstream (coarser) resolution;
///   `0` for uniform-resolution studies. Reuse the same value the forward pass
///   was set up with — recomputing it independently here can size the sampler's
///   ring differently and desync the replay from the forward lag chain.
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches between `library`, `par`,
/// `stages`, or `stage_lag_transitions` are detected.
// Rationale: one argument per independent input; a grouping type would have
// this single consumer.
#[allow(clippy::too_many_arguments)]
pub fn standardize_historical_windows(
    structure: &HistoricalStructureProof<'_>,
    library: &mut HistoricalScenarioLibrary,
    inflow_history: &[InflowHistoryRow],
    par: &PrecomputedPar,
    window_years: &[i32],
    season_map: Option<&SeasonMap>,
    seed: DerivedSeed<'_>,
    stage_lag_transitions: &[StageLagTransition],
    downstream_par_order: usize,
) {
    let &HistoricalStructureProof { stages, hydro_ids } = structure;
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
    // `library.max_order()` may exceed `par.max_order()` when the caller widens
    // it to a declared lag-state depth beyond the fitted AR order; it must never
    // fall short, which would truncate a real PAR coefficient.
    debug_assert!(
        library.max_order() >= par.max_order(),
        "library.max_order() ({}) must be >= par.max_order() ({})",
        library.max_order(),
        par.max_order(),
    );
    debug_assert!(
        stage_lag_transitions.is_empty() || stage_lag_transitions.len() == stages.len(),
        "stage_lag_transitions.len() ({}) must be 0 or equal to stages.len() ({})",
        stage_lag_transitions.len(),
        stages.len(),
    );

    let n_hydros = library.n_hydros();
    let n_stages = library.n_stages();
    let max_order = library.max_order();

    if n_hydros == 0 || n_stages == 0 || window_years.is_empty() {
        return;
    }

    let n_seasons = stages
        .iter()
        .filter_map(|s| s.season_id)
        .max()
        .map_or(1, |m| m + 1);

    let hydro_id_to_idx: std::collections::HashMap<EntityId, usize> = hydro_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let row_keys: Vec<Option<(usize, i32)>> = inflow_history
        .iter()
        .map(|r| history_row_key(&stage_index, season_map, r.start_date))
        .collect();
    let key_years = || row_keys.iter().flatten().map(|&(_, year)| year);
    let (Some(min_year), Some(max_year)) = (key_years().min(), key_years().max()) else {
        return;
    };
    #[allow(clippy::cast_sign_loss)]
    let n_years = (max_year - min_year + 1) as usize;

    let table_size = n_hydros * n_years * n_seasons;
    let mut obs_table = vec![f64::NAN; table_size];

    let table_idx = |h: usize, year: i32, s: usize| -> Option<usize> {
        if year < min_year || year > max_year || s >= n_seasons {
            return None;
        }
        #[allow(clippy::cast_sign_loss)]
        let y = (year - min_year) as usize;
        Some(h * n_years * n_seasons + y * n_seasons + s)
    };

    for (r, &key) in inflow_history.iter().zip(&row_keys) {
        if let Some((sid, year)) = key
            && let Some(&h) = hydro_id_to_idx.get(&r.hydro_id)
            && let Some(idx) = table_idx(h, year, sid)
        {
            obs_table[idx] = r.value_m3s;
        }
    }

    let lookup = |h: usize, year: i32, season_id: usize| -> f64 {
        table_idx(h, year, season_id).map_or(0.0, |idx| {
            let v = obs_table[idx];
            if v.is_nan() { 0.0 } else { v }
        })
    };

    let full_sequence: Vec<(i32, usize)> = super::build_observation_sequence(stages, season_map);

    // Digest over little-endian f64 bytes so it is reproducible across runs.
    {
        let mut hasher = SipHasher13::new();
        for &v in seed.lag_values {
            hasher.write(&v.to_le_bytes());
        }
        library.seed_digest = hasher.finish();
    }

    run_eta_inversion(
        n_stages,
        window_years.len(),
        n_hydros,
        max_order,
        par,
        seed,
        stage_lag_transitions,
        downstream_par_order,
        |t, w, h| {
            let (year_offset, season_id) = full_sequence[t];
            let obs_year = window_years[w] + year_offset;
            debug_assert!(
                table_idx(0, obs_year, season_id).is_some(),
                "missing study observation for year={obs_year}, season={season_id}; \
                 window discovery should have excluded this window",
            );
            debug_assert!(
                max_order == 0
                    || table_idx(h, obs_year, season_id).is_some_and(|i| !obs_table[i].is_nan()),
                "missing study observation for hydro={}, year={obs_year}, \
                 season={season_id}; window discovery should have excluded this window",
                hydro_ids[h].0,
            );
            lookup(h, obs_year, season_id)
        },
        |t, w, h, eta| library.eta_slice_mut(w, t)[h] = eta,
    );
}

// ---------------------------------------------------------------------------
// validate_historical_library
// ---------------------------------------------------------------------------

/// Validate a [`HistoricalScenarioLibrary`] against construction inputs.
///
/// Runs after window discovery and eta standardization; the first failed error
/// check returns `Err`. The structural checks V2.1 and V2.9 run earlier, in
/// [`check_historical_structure`].
///
/// ## Checks performed
///
/// | ID  | Kind    | Description                                                   |
/// |-----|---------|---------------------------------------------------------------|
/// | V2.5 | Error  | At least one window must be discovered when `user_pool` is `None`. |
/// | V2.3 | Error  | No eta value may be `NEG_INFINITY` or NaN; the refusal names the window year, the stage id and the hydro id. |
/// | V2.6 | Warning| `library.n_windows() < forward_passes` — log a warning.      |
/// | V2.2 | Assert | Window contiguity — `debug_assert` only (construction invariant). |
/// | V2.4 | Assert | User pool validity — `debug_assert` only (construction invariant). |
/// | V2.7 | Assert | Library lag depth equals the PAR order — `debug_assert` only (construction invariant). |
///
/// # Inputs
///
/// - `structure` — proof from [`check_historical_structure`]; its stage and
///   hydro ids label the V2.3 refusal
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] with a message prefixed by
/// the check ID (e.g., `"V2.1: ..."`) for the first failed error check.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::EntityId;
/// use cobre_core::temporal::{
///     Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
///     StageStateConfig,
/// };
/// use cobre_stochastic::HistoricalScenarioLibrary;
/// use cobre_stochastic::sampling::historical::{
///     check_historical_structure, validate_historical_library,
/// };
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
/// let stages = [stage];
/// let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
/// let result = validate_historical_library(&structure, &lib, 1, None, 5);
/// assert!(result.is_ok());
/// ```
pub fn validate_historical_library(
    structure: &HistoricalStructureProof<'_>,
    library: &HistoricalScenarioLibrary,
    max_par_order: usize,
    user_pool: Option<&HistoricalYears>,
    forward_passes: u32,
) -> Result<(), StochasticError> {
    if user_pool.is_none() && library.n_windows() == 0 {
        return Err(StochasticError::InsufficientData {
            context: "V2.5: historical library has 0 windows after auto-discovery; \
                      at least 1 complete historical window is required"
                .to_string(),
        });
    }

    let &HistoricalStructureProof { stages, hydro_ids } = structure;
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

    // V2.3 / V2.8 — no eta may be NEG_INFINITY (the sigma=0 non-matching-observation
    // sentinel from standardize_historical_windows) or NaN.
    for w in 0..library.n_windows() {
        for (t, stage) in stages.iter().enumerate().take(library.n_stages()) {
            let eta = library.eta_slice(w, t);
            for (h, &value) in eta.iter().enumerate() {
                if value == f64::NEG_INFINITY || value.is_nan() {
                    let year = library.window_year(w);
                    let stage_id = stage.id;
                    let hydro_id = hydro_ids[h];
                    return Err(StochasticError::InsufficientData {
                        context: format!(
                            "V2.3: historical library contains non-finite eta (NEG_INFINITY or NaN) \
                             at window year {year}, stage id {stage_id}, hydro id {hydro_id} — sigma=0 with \
                             non-matching historical observation or numerical failure",
                        ),
                    });
                }
            }
        }
    }

    // V2.2/V2.4/V2.7 are construction invariants; only V2.7 is re-asserted here.
    debug_assert!(
        library.max_order() == max_par_order,
        "V2.7: library.max_order() ({}) must equal max_par_order ({max_par_order})",
        library.max_order(),
    );

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

    use chrono::{Datelike, Months, NaiveDate, TimeDelta, Weekday};
    use cobre_core::{
        EntityId, Hydro,
        scenario::{InflowHistoryRow, InflowModel},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
            StageLagTransition, StageRiskConfig, StageStateConfig,
        },
        test_support::{HydroSpec, MirrorUnitGroup, StageSpec, date, single_block},
    };

    use super::{
        DerivedSeed, SeasonMap, Stage, check_historical_structure, standardize_historical_windows,
    };
    use crate::derive_inflow_seeds;
    use crate::par::{
        DownstreamLagAccum, EntityMajor, PrimaryLagAccum, advance_lag_chain,
        evaluate::{evaluate_par, solve_par_noise},
        precompute::PrecomputedPar,
        precompute_stage_lag_transitions,
    };
    use crate::test_support::{
        MonthlyLabels, monthly_season_map, quarterly_season_map, weekly_season_map,
    };

    /// `season_id` is 0-based (0=Jan .. 11=Dec).
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

    /// `month0` is 0-based (0=Jan, 11=Dec), matching the `season_id` that
    /// `standardize_historical_windows` derives from `start_date.month0()`.
    fn make_row(hydro_id: EntityId, year: i32, month0: u32, value: f64) -> InflowHistoryRow {
        let start_date = NaiveDate::from_ymd_opt(year, month0 + 1, 1).unwrap();
        InflowHistoryRow {
            hydro_id,
            start_date,
            end_date: start_date.checked_add_months(Months::new(1)).unwrap(),
            value_m3s: value,
        }
    }

    fn twelve_monthly_stages() -> Vec<Stage> {
        (0..12).map(|i| make_monthly_stage(i, i)).collect()
    }

    #[test]
    fn test_ar0_standardization() {
        let hydro = EntityId(1);
        // Both study stages resolve to their own calendar year, 1990, so both
        // sit at year_offset 0 — window_year 1990 itself.
        let stages = vec![make_monthly_stage(0, 0), make_monthly_stage(1, 1)];
        let models = vec![
            InflowModel {
                hydro_id: hydro,
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: hydro,
                stage_id: 1,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();

        let history = vec![
            make_row(hydro, 1990, 0, 120.0),
            make_row(hydro, 1990, 1, 90.0),
        ];

        let mut lib = HistoricalScenarioLibrary::new(1, 2, 1, 0, vec![1990]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &[1990],
            None,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

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

    /// Stage 1's lag is stage 0's RAW historical observation (130.0), not a
    /// value reconstructed from the stored eta. Twelve monthly stages put the
    /// pre-study lag season one step before Jan, i.e. Dec of `window_year - 1`.
    #[test]
    fn test_ar1_standardization_uses_raw_lags() {
        let hydro = EntityId(1);
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
                annual: None,
            })
            .collect();
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();

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

        // Window year 1990, max_order=1 (study starts at window_year):
        //   lag: (1989, season 11 = Dec) → 110.0
        //   stage 0: (1990, season 0 = Jan) → 130.0
        //   stage 1: (1990, season 1 = Feb) → 95.0
        //   (remaining study stages: use 100.0, not used in assertions)
        let mut history = vec![
            make_row(hydro, 1989, 11, 110.0),
            make_row(hydro, 1990, 0, 130.0),
            make_row(hydro, 1990, 1, 95.0),
        ];
        for m in 2..12_u32 {
            history.push(make_row(hydro, 1990, m, 100.0));
        }

        // lag-1 = Dec 1989 = 110.0, so the rolling chain starts from the same
        // value as the window's own pre-study observation.
        let derived_lag_values = [110.0];

        let mut lib = HistoricalScenarioLibrary::new(1, 12, 1, 1, vec![1990]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &[1990],
            None,
            DerivedSeed {
                lag_values: &derived_lag_values,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        // Stage 0: lag_state seeded from the derived seed → lag = 110.0.
        let eta_0 = lib.eta_slice(0, 0)[0];
        let expected_0 = (130.0 - 80.0 - 0.5 * 110.0) / 25.0;
        assert!(
            (eta_0 - expected_0).abs() < 1e-10,
            "eta stage 0: expected {expected_0}, got {eta_0}"
        );

        // Stage 1: lag_state advanced by finalize after stage 0 → lag = 130.0 (Jan 1990).
        let eta_1 = lib.eta_slice(0, 1)[0];
        let expected_1 = (95.0 - 80.0 - 0.5 * 130.0) / 25.0;
        assert!(
            (eta_1 - expected_1).abs() < 1e-10,
            "eta stage 1: expected {expected_1}, got {eta_1} (rolling chain: lag=130.0)"
        );
    }

    #[test]
    fn standardize_reads_only_the_study_entry_observations() {
        let hydro = EntityId(1);
        let stages = twelve_monthly_stages();
        let models: Vec<InflowModel> = std::iter::once(-1_i32)
            .chain(0..12_i32)
            .map(|stage_id| InflowModel {
                hydro_id: hydro,
                stage_id,
                mean_m3s: 160.0,
                std_m3s: 25.0,
                ar_coefficients: vec![0.5],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();
        assert_eq!(par.max_order(), 1);

        let window_years = [1991, 1992];
        let full_history: Vec<InflowHistoryRow> = (1990..=1992)
            .flat_map(|y| {
                (0..12_u32).map(move |m| {
                    make_row(
                        hydro,
                        y,
                        m,
                        100.0 + 3.0 * f64::from(m) + 7.0 * f64::from(y - 1990),
                    )
                })
            })
            .collect();
        let study_entry_history: Vec<InflowHistoryRow> = full_history
            .iter()
            .filter(|r| window_years.contains(&r.start_date.year()))
            .cloned()
            .collect();

        let standardize = |history: &[InflowHistoryRow]| {
            let mut lib = HistoricalScenarioLibrary::new(2, 12, 1, 1, window_years.to_vec());
            let hydro_ids = [hydro];
            let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
            standardize_historical_windows(
                &structure,
                &mut lib,
                history,
                &par,
                &window_years,
                None,
                DerivedSeed {
                    lag_values: &[110.0],
                    l_state: 1,
                    accum: &[],
                    weight: &[],
                },
                &[],
                0,
            );
            lib
        };
        let lib_full = standardize(&full_history);
        let lib_study_entries = standardize(&study_entry_history);

        for w in 0..window_years.len() {
            for t in 0..stages.len() {
                let full: Vec<u64> = lib_full
                    .eta_slice(w, t)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let study: Vec<u64> = lib_study_entries
                    .eta_slice(w, t)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(full, study, "window {w}, stage {t}");
            }
        }
    }

    #[test]
    fn test_multi_hydro_multi_window() {
        let h1 = EntityId(1);
        let h2 = EntityId(2);
        // 2 stages, season_ids 0 and 1, both resolving to their own calendar
        // year (year_offset=0).
        let stages = vec![make_monthly_stage(0, 0), make_monthly_stage(1, 1)];

        let models = vec![
            InflowModel {
                hydro_id: h1,
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: h1,
                stage_id: 1,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: h2,
                stage_id: 0,
                mean_m3s: 200.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: h2,
                stage_id: 1,
                mean_m3s: 200.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &[h1, h2], None).unwrap();

        // Observations at year=window_year (year_offset=0 for both stages).
        let history = vec![
            make_row(h1, 1990, 0, 110.0),
            make_row(h1, 1990, 1, 90.0),
            make_row(h2, 1990, 0, 220.0),
            make_row(h2, 1990, 1, 180.0),
            make_row(h1, 1991, 0, 105.0),
            make_row(h1, 1991, 1, 95.0),
            make_row(h2, 1991, 0, 210.0),
            make_row(h2, 1991, 1, 190.0),
        ];

        let mut lib = HistoricalScenarioLibrary::new(2, 2, 2, 0, vec![1990, 1991]);
        let hydro_ids = [h1, h2];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &[1990, 1991],
            None,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

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

    /// sigma=0 with an observation that matches the deterministic value.
    #[test]
    fn test_sigma_zero_returns_zero_eta() {
        let hydro = EntityId(1);
        let stages = vec![make_monthly_stage(0, 0)];
        let models = vec![InflowModel {
            hydro_id: hydro,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }];
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();

        let history = vec![make_row(hydro, 2000, 0, 50.0)];

        let mut lib = HistoricalScenarioLibrary::new(1, 1, 1, 0, vec![2000]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &[2000],
            None,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

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

        let structure = check_historical_structure(&lib, &hydro_ids, &stages);
        assert!(structure.is_ok(), "expected Ok(_), got: {structure:?}");
        let result = validate_historical_library(&structure.unwrap(), &lib, 1, None, 5);
        assert!(result.is_ok(), "expected Ok(()), got: {result:?}");
    }

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
        lib.eta_slice_mut(2, 5)[1] = f64::NEG_INFINITY;
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| make_validate_stage(i, Some(i % 12)))
            .collect();
        let hydro_ids: Vec<EntityId> = (1..=3).map(EntityId).collect();
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();

        let result = validate_historical_library(&structure, &lib, 1, None, 5);
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

    #[test]
    fn non_finite_eta_refusal_names_window_year_stage_id_and_hydro_id() {
        let n_stages = 12;
        let mut lib = HistoricalScenarioLibrary::new(5, n_stages, 3, 1, (1990..1995).collect());
        lib.eta_slice_mut(2, 5)[1] = f64::NEG_INFINITY;
        let stages: Vec<Stage> = (101_i32..)
            .zip(0..n_stages)
            .map(|(id, i)| Stage {
                id,
                ..make_validate_stage(i, Some(i % 12))
            })
            .collect();
        let hydro_ids: Vec<EntityId> = (11..=13).map(EntityId).collect();
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();

        let result = validate_historical_library(&structure, &lib, 1, None, 5);
        match result {
            Err(StochasticError::InsufficientData { context }) => assert_eq!(
                context,
                "V2.3: historical library contains non-finite eta (NEG_INFINITY or NaN) \
                 at window year 1992, stage id 106, hydro id 12 — sigma=0 with \
                 non-matching historical observation or numerical failure"
            ),
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    #[test]
    fn test_missing_season_id_fails_v2_1() {
        let lib = HistoricalScenarioLibrary::new(1, 2, 2, 0, vec![1990]);
        let stages = vec![
            make_validate_stage(0, Some(0)),
            make_validate_stage(1, None),
        ];
        let hydro_ids = vec![EntityId(1), EntityId(2)];

        let result = check_historical_structure(&lib, &hydro_ids, &stages);
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

    #[test]
    fn test_hydro_count_mismatch_fails_v2_9() {
        let lib = HistoricalScenarioLibrary::new(1, 1, 3, 0, vec![1990]);
        let stages = vec![make_validate_stage(0, Some(0))];
        let hydro_ids = vec![EntityId(1), EntityId(2), EntityId(3), EntityId(4)];

        let result = check_historical_structure(&lib, &hydro_ids, &stages);
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

    #[test]
    fn test_pool_warning_path_returns_ok() {
        let lib = HistoricalScenarioLibrary::new(5, 1, 2, 0, (1990..1995).collect());
        let stages = vec![make_validate_stage(0, Some(0))];
        let hydro_ids = vec![EntityId(1), EntityId(2)];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();

        // 5 windows < 20 forward passes triggers warn! but must still return Ok(()).
        let result = validate_historical_library(&structure, &lib, 0, None, 20);
        assert!(
            result.is_ok(),
            "warning path must return Ok(()), got: {result:?}"
        );
    }

    #[test]
    fn test_standardize_monthly_season_map_identical() {
        let hydro = EntityId(1);
        let stages = vec![make_monthly_stage(0, 0), make_monthly_stage(1, 1)];
        let models = vec![
            InflowModel {
                hydro_id: hydro,
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: hydro,
                stage_id: 1,
                mean_m3s: 60.0,
                std_m3s: 15.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();

        let history = vec![
            make_row(hydro, 2000, 0, 100.0),
            make_row(hydro, 2000, 1, 45.0),
        ];

        let sm = monthly_season_map(MonthlyLabels::ZeroBased);

        let mut lib_none = HistoricalScenarioLibrary::new(1, 2, 1, 0, vec![2000]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib_none, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib_none,
            &history,
            &par,
            &[2000],
            None,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        let mut lib_sm = HistoricalScenarioLibrary::new(1, 2, 1, 0, vec![2000]);
        let structure = check_historical_structure(&lib_sm, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib_sm,
            &history,
            &par,
            &[2000],
            Some(&sm),
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        for t in 0..2 {
            assert_eq!(
                lib_none.eta_slice(0, t),
                lib_sm.eta_slice(0, t),
                "eta values at stage {t} must be identical between None and monthly SeasonMap"
            );
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn make_quarterly_stage(index: usize, season_id: usize) -> Stage {
        let month = (season_id as u32) * 3 + 1;
        Stage {
            index,
            id: index as i32,
            start_date: NaiveDate::from_ymd_opt(2024, month, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(
                2024,
                if month + 2 <= 12 { month + 2 } else { 12 },
                28,
            )
            .unwrap(),
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 2160.0,
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

    #[test]
    fn test_standardize_quarterly_season_map_correct() {
        let hydro = EntityId(1);
        let stages: Vec<Stage> = (0..4).map(|i| make_quarterly_stage(i, i)).collect();
        let models: Vec<InflowModel> = (0_i32..4)
            .map(|i| InflowModel {
                hydro_id: hydro,
                stage_id: i,
                mean_m3s: 100.0 + f64::from(i) * 10.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();

        let sm = quarterly_season_map();

        // Quarterly observations: one per quarter for year 2000.
        // Jan 1 -> Q1 (season 0), Apr 1 -> Q2 (season 1),
        // Jul 1 -> Q3 (season 2), Oct 1 -> Q4 (season 3).
        let history = vec![
            InflowHistoryRow {
                hydro_id: hydro,
                start_date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2000, 4, 1).unwrap(),
                value_m3s: 120.0,
            },
            InflowHistoryRow {
                hydro_id: hydro,
                start_date: NaiveDate::from_ymd_opt(2000, 4, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2000, 7, 1).unwrap(),
                value_m3s: 130.0,
            },
            InflowHistoryRow {
                hydro_id: hydro,
                start_date: NaiveDate::from_ymd_opt(2000, 7, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2000, 10, 1).unwrap(),
                value_m3s: 140.0,
            },
            InflowHistoryRow {
                hydro_id: hydro,
                start_date: NaiveDate::from_ymd_opt(2000, 10, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2001, 1, 1).unwrap(),
                value_m3s: 150.0,
            },
        ];

        let mut lib = HistoricalScenarioLibrary::new(1, 4, 1, 0, vec![2000]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &[2000],
            Some(&sm),
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        // eta = (obs - mean) / std for each season (AR(0)):
        // Q1: (120 - 100) / 10 = 2.0
        // Q2: (130 - 110) / 10 = 2.0
        // Q3: (140 - 120) / 10 = 2.0
        // Q4: (150 - 130) / 10 = 2.0
        for t in 0..4 {
            let eta = lib.eta_slice(0, t)[0];
            assert!(
                (eta - 2.0).abs() < 1e-10,
                "stage {t}: expected eta=2.0, got {eta}"
            );
        }
    }

    /// Same fixture and oracle as `external`'s
    /// `quarterly_ring_sampler_external_matches_oracle`: 3 monthly stages feed
    /// the downstream ring (weight 1/3 each, finalized at stage 2), stage 3
    /// rebuilds the primary lag from the ring, and stage 4's AR(1) eta reads
    /// that rebuilt lag — the first point downstream of the transition whose
    /// value depends on whether the ring fired.
    ///
    /// Oracle: ring average = (130+140+150)/3 = 140.0. The negative control
    /// hand-computes stage 4's eta under the old primary-only advance (stage
    /// 3's own raw value 500.0 shifted into the lag instead of the ring
    /// average) and asserts it differs from the kernel-routed result.
    #[test]
    fn quarterly_ring_sampler_historical_matches_oracle() {
        let hydro = EntityId(1);
        let stages: Vec<Stage> = (0..5).map(|i| make_monthly_stage(i, i)).collect();

        let mut models: Vec<InflowModel> = (0_i32..4)
            .map(|sid| InflowModel {
                hydro_id: hydro,
                stage_id: sid,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        // Stage 4: AR(1), base=80, psi=0.5, sigma=25.
        models.push(InflowModel {
            hydro_id: hydro,
            stage_id: 4,
            mean_m3s: 160.0,
            std_m3s: 25.0,
            ar_coefficients: vec![0.5],
            residual_std_ratio: 1.0,
            annual: None,
        });
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();
        assert_eq!(
            par.max_order(),
            1,
            "stage 4's AR(1) sets the global max_order"
        );

        let window_year = 1990;
        let raw_values = [130.0, 140.0, 150.0, 500.0, 200.0];
        let history: Vec<InflowHistoryRow> = raw_values
            .iter()
            .enumerate()
            .map(|(month0, &value)| {
                make_row(hydro, window_year, u32::try_from(month0).unwrap(), value)
            })
            .collect();
        let derived_lag_values = [0.0];

        let downstream_transition = |downstream_finalize: bool| StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize,
            rebuild_from_downstream: false,
        };
        let transitions = vec![
            downstream_transition(false),
            downstream_transition(false),
            downstream_transition(true),
            StageLagTransition {
                accumulate_weight: 1.0,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: true,
            },
            StageLagTransition {
                accumulate_weight: 1.0,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
        ];

        let mut lib = HistoricalScenarioLibrary::new(1, 5, 1, 1, vec![window_year]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &[window_year],
            None,
            DerivedSeed {
                lag_values: &derived_lag_values,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &transitions,
            1,
            // downstream_par_order: one completed quarter needed to rebuild,
        );

        let det_base = par.deterministic_base(4, 0);
        let psi = par.psi_slice(4, 0)[0];
        let sigma = par.sigma(4, 0);

        let ring_average = (130.0 + 140.0 + 150.0) / 3.0;
        let expected_eta_4 = (raw_values[4] - det_base - psi * ring_average) / sigma;
        let eta_4 = lib.eta_slice(0, 4)[0];
        assert!(
            (eta_4 - expected_eta_4).abs() < 1e-10,
            "eta[stage=4] = {eta_4}, expected {expected_eta_4} (ring average lag = {ring_average})"
        );

        // Negative control: the old primary-only advance would have shifted
        // stage 3's own raw value (500.0), not the ring average, into the lag.
        let naive_lag = raw_values[3];
        let naive_eta_4 = (raw_values[4] - det_base - psi * naive_lag) / sigma;
        assert!(
            (eta_4 - naive_eta_4).abs() > 1e-6,
            "eta[stage=4] must differ from the primary-only naive value; \
             got {eta_4} == naive {naive_eta_4}"
        );
    }

    /// One row per quarter per year, dated the 1st of Jan/Apr/Jul/Oct, at a flat
    /// 100.0; callers overwrite the years they need to tell apart.
    fn quarterly_history(
        hydro_id: EntityId,
        from_year: i32,
        to_year: i32,
    ) -> Vec<InflowHistoryRow> {
        let quarter_months = [1u32, 4, 7, 10];
        (from_year..=to_year)
            .flat_map(|y| {
                quarter_months.iter().map(move |&m| {
                    let start_date = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
                    InflowHistoryRow {
                        hydro_id,
                        start_date,
                        end_date: start_date.checked_add_months(Months::new(3)).unwrap(),
                        value_m3s: 100.0,
                    }
                })
            })
            .collect()
    }

    #[test]
    fn test_standardize_none_season_map_backward_compat() {
        let hydro = EntityId(1);
        let stages = vec![make_monthly_stage(0, 0)];
        let models = vec![InflowModel {
            hydro_id: hydro,
            stage_id: 0,
            mean_m3s: 90.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }];
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();

        // Observation at (window_year=1995, season 0 = Jan) via month0() = 0.
        let history = vec![make_row(hydro, 1995, 0, 110.0)];

        let mut lib = HistoricalScenarioLibrary::new(1, 1, 1, 0, vec![1995]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &[1995],
            None,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        let eta = lib.eta_slice(0, 0)[0];
        let expected = (110.0 - 90.0) / 10.0; // = 2.0
        assert!(
            (eta - expected).abs() < 1e-10,
            "None season_map backward compat: expected eta={expected}, got {eta}"
        );
    }

    /// The `season_map.season_for_date()` path must resolve lag observations
    /// outside the study stage date ranges, never the `month0()` fallback: under
    /// `month0()` the Apr/Jul/Oct rows land on 3/6/9, out of range for
    /// `n_seasons`=4, so Q2–Q4 would be silently dropped and eta would read 0.0.
    ///
    /// Expected eta (AR(0), eta = (obs - mean) / std):
    ///   h1: Q1=(90-80)/10=1.0, Q2=(110-90)/10=2.0, Q3=(115-100)/10=1.5, Q4=(120-110)/10=1.0
    ///   h2: Q1=(85-80)/10=0.5, Q2=(95-90)/10=0.5, Q3=(105-100)/10=0.5, Q4=(125-110)/10=1.5
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    #[test]
    fn test_quarterly_standardize_historical_windows() {
        let h1 = EntityId(1);
        let h2 = EntityId(2);

        let stages: Vec<Stage> = (0..4).map(|i| make_quarterly_stage(i, i)).collect();

        // Distinct mean per season (identical std=10) so quarter-alignment can
        // be confirmed independently for each season.
        let mean_per_season = [80.0_f64, 90.0, 100.0, 110.0];
        let std_per_season = [10.0_f64; 4];

        let mut models: Vec<InflowModel> = Vec::new();
        for &hydro in &[h1, h2] {
            for (i, &mean) in mean_per_season.iter().enumerate() {
                models.push(InflowModel {
                    hydro_id: hydro,
                    stage_id: i as i32,
                    mean_m3s: mean,
                    std_m3s: std_per_season[i],
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                });
            }
        }
        let par = PrecomputedPar::build(&models, &stages, &[h1, h2], None).unwrap();

        let sm = quarterly_season_map();

        let quarter_months = [1u32, 4, 7, 10];
        let h1_obs_2000 = [90.0_f64, 110.0, 115.0, 120.0];
        let h2_obs_2000 = [85.0_f64, 95.0, 105.0, 125.0];

        let mut history: Vec<InflowHistoryRow> = quarterly_history(h1, 1990, 2010);
        history.extend(quarterly_history(h2, 1990, 2010));

        for row in &mut history {
            if row.start_date.year() == 2000 {
                let q = quarter_months
                    .iter()
                    .position(|&m| m == row.start_date.month())
                    .expect("start_date month must be a quarter month");
                if row.hydro_id == h1 {
                    row.value_m3s = h1_obs_2000[q];
                } else if row.hydro_id == h2 {
                    row.value_m3s = h2_obs_2000[q];
                }
            }
        }

        let window_years = vec![2000_i32];
        let n_hydros = 2;
        let n_stages = 4;
        let mut lib =
            HistoricalScenarioLibrary::new(1, n_stages, n_hydros, 0, window_years.clone());

        let hydro_ids = [h1, h2];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &window_years,
            Some(&sm),
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        let h1_expected = [1.0_f64, 2.0, 1.5, 1.0];
        let h2_expected = [0.5_f64, 0.5, 0.5, 1.5];

        for t in 0..n_stages {
            let eta = lib.eta_slice(0, t);
            assert!(
                (eta[0] - h1_expected[t]).abs() < 1e-10,
                "h1 stage {t} (Q{}): expected eta={}, got {} — \
                 seasonal alignment via quarterly SeasonMap must be correct",
                t + 1,
                h1_expected[t],
                eta[0]
            );
            assert!(
                (eta[1] - h2_expected[t]).abs() < 1e-10,
                "h2 stage {t} (Q{}): expected eta={}, got {} — \
                 seasonal alignment via quarterly SeasonMap must be correct",
                t + 1,
                h2_expected[t],
                eta[1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Derived-seed replay: the replay seeds from the same derived lag state as
    // the forward pass, so z == v.
    // -----------------------------------------------------------------------

    fn make_hydro(id: i32) -> Hydro {
        cobre_core::test_support::make_hydro(HydroSpec {
            id,
            name: format!("H{id}"),
            max_storage_hm3: 100.0,
            max_turbined_m3s: 100.0,
            max_generation_mw: 100.0,
            operational_start_date: date(2020, 1, 1),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        })
    }

    /// With no conditioning, the derived lag seed carries exactly the same
    /// values an explicit literal seed would hold, so threading it
    /// through the per-(stage, hydro) `solve_par_noise` calls reproduces the
    /// eta a hard-coded seed of those values would have produced. Covers 2
    /// hydros and a 2-lag stride so a transposed `(hydro, lag)` index in the
    /// fill loop would be caught.
    #[test]
    fn standardize_historical_eta_matches_positional_seed() {
        let h1 = EntityId(1);
        let h2 = EntityId(2);
        let hydro_ids = vec![h1, h2];
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let season_map = monthly_season_map(MonthlyLabels::ZeroBased);

        // Two monthly study stages (Jan, Feb) so lag lookups resolve through
        // the ordinary multi-season path rather than the single-season
        // annual-cycle fallback.
        let stages = vec![make_monthly_stage(0, 0), make_monthly_stage(1, 1)];
        let first_stage = stages[0].clone();

        let models = vec![
            InflowModel {
                hydro_id: h1,
                stage_id: 0,
                mean_m3s: 300.0,
                std_m3s: 40.0,
                ar_coefficients: vec![0.3, 0.1],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: h1,
                stage_id: 1,
                mean_m3s: 300.0,
                std_m3s: 40.0,
                ar_coefficients: vec![0.3, 0.1],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: h2,
                stage_id: 0,
                mean_m3s: 500.0,
                std_m3s: 60.0,
                ar_coefficients: vec![0.2, 0.05],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: h2,
                stage_id: 1,
                mean_m3s: 500.0,
                std_m3s: 60.0,
                ar_coefficients: vec![0.2, 0.05],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();
        let l_state = par.max_order();
        assert_eq!(l_state, 2);

        let record = vec![
            InflowHistoryRow {
                hydro_id: h1,
                start_date: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                value_m3s: 110.0,
            },
            InflowHistoryRow {
                hydro_id: h1,
                start_date: NaiveDate::from_ymd_opt(2023, 11, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
                value_m3s: 120.0,
            },
            InflowHistoryRow {
                hydro_id: h2,
                start_date: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                value_m3s: 210.0,
            },
            InflowHistoryRow {
                hydro_id: h2,
                start_date: NaiveDate::from_ymd_opt(2023, 11, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
                value_m3s: 220.0,
            },
        ];
        let derived =
            derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, l_state);
        assert_eq!(derived.lag_values, vec![110.0, 120.0, 210.0, 220.0]);

        // Study-window observations for both stages/hydros; only stage 0's
        // eta is asserted below.
        let inflow_history = vec![
            make_row(h1, 2024, 0, 250.0),
            make_row(h1, 2024, 1, 999.0),
            make_row(h2, 2024, 0, 300.0),
            make_row(h2, 2024, 1, 999.0),
        ];

        let mut lib = HistoricalScenarioLibrary::new(1, 2, 2, l_state, vec![2024]);
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &inflow_history,
            &par,
            &[2024],
            Some(&season_map),
            DerivedSeed {
                lag_values: &derived.lag_values,
                l_state,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        let expected_h1 = solve_par_noise(
            par.deterministic_base(0, 0),
            par.psi_slice(0, 0),
            &[110.0, 120.0],
            par.sigma(0, 0),
            250.0,
        );
        let expected_h2 = solve_par_noise(
            par.deterministic_base(0, 1),
            par.psi_slice(0, 1),
            &[210.0, 220.0],
            par.sigma(0, 1),
            300.0,
        );

        let eta = lib.eta_slice(0, 0);
        assert_eq!(
            eta[0], expected_h1,
            "hydro 1 eta must match the positional-seed formula"
        );
        assert_eq!(
            eta[1], expected_h2,
            "hydro 2 eta must match the positional-seed formula"
        );
    }

    /// Build a stage with an arbitrary (non-1st-of-month) `start_date`.
    fn dated_stage(
        index: usize,
        id: i32,
        start: NaiveDate,
        end: NaiveDate,
        season_id: usize,
    ) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            start_date: start,
            end_date: end,
            season_id: Some(season_id),
            blocks: single_block("SINGLE", 744.0),
            ..Default::default()
        })
    }

    /// The historical-replay analogue of `external_eta_round_trip_exact_mid_coarse_period`
    /// (`external.rs`): a study whose stage 0 starts mid-coarse-period, with
    /// pre-study record coverage seeding a genuine partial `accum`/`weight`,
    /// exercises a divergence no monthly-boundary fixture above can reach.
    /// Forward generation and `standardize_historical_windows`'s replay reset
    /// both advance the lag chain from the same `derived.accum`/`derived.weight`
    /// seed, so `z == v`.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn historical_replay_round_trip_exact_mid_coarse_period() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let hydros = vec![make_hydro(1)];
        let season_map = monthly_season_map(MonthlyLabels::ZeroBased);

        // Stage 0 starts April 11: the in-progress occurrence [April 1,
        // April 11) is non-empty, and the remaining 20 of April's 30 days
        // still finalize within stage 0.
        let stages = vec![
            dated_stage(
                0,
                0,
                NaiveDate::from_ymd_opt(2026, 4, 11).unwrap(),
                NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
                3,
            ),
            dated_stage(
                1,
                1,
                NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 6, 1).unwrap(),
                4,
            ),
        ];
        let first_stage = stages[0].clone();

        let models = vec![
            InflowModel {
                hydro_id,
                stage_id: 0,
                mean_m3s: 160.0,
                std_m3s: 25.0,
                ar_coefficients: vec![0.5],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id,
                stage_id: 1,
                mean_m3s: 160.0,
                std_m3s: 25.0,
                ar_coefficients: vec![0.5],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();
        let l_state = par.max_order();
        assert_eq!(l_state, 1, "AR(1) with no annual coupling stays order 1");

        let record = vec![
            InflowHistoryRow {
                hydro_id,
                start_date: NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                value_m3s: 300.0,
            },
            InflowHistoryRow {
                hydro_id,
                start_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2026, 4, 11).unwrap(),
                value_m3s: 200.0,
            },
        ];

        let derived =
            derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, l_state);
        assert_eq!(derived.lag_values.len(), l_state);
        assert_eq!(derived.lag_values[0], 300.0);
        assert!(
            derived.weight[0] > 0.0 && derived.weight[0] < 1.0,
            "the accumulator seed must be a genuine partial-coverage fraction \
             in (0, 1), or this test is a tautology (every monthly-boundary \
             fixture misses the bug this way); got weight={}",
            derived.weight[0]
        );

        let stage_lag_transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert!(
            stage_lag_transitions[0].finalize_period && stage_lag_transitions[1].finalize_period,
            "both stages must finalize their own period for this fixture to \
             exercise a mid-period accumulate/finalize transition"
        );

        // Forward-generate the window's realized values by advancing the
        // SEEDED lag chain exactly as the training/simulation forward pass
        // does: the accumulator starts from `derived.accum`/`derived.weight`,
        // not zero.
        let eta_sequence = [0.35_f64, -0.6];
        let mut lag_state = derived.lag_values.clone();
        let mut accum = derived.accum.clone();
        let mut weight = derived.weight.clone();
        let mut incoming_scratch = vec![0.0_f64; l_state];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0_f64;
        let mut downstream_completed_lags: Vec<f64> = Vec::new();
        let mut downstream_n_completed = 0_usize;
        let mut targets = Vec::with_capacity(stages.len());
        for (t, &eta) in eta_sequence.iter().enumerate() {
            let det_base = par.deterministic_base(t, 0);
            let psi = par.psi_slice(t, 0);
            let sigma = par.sigma(t, 0);
            let value = evaluate_par(det_base, psi, &lag_state, sigma, eta);
            targets.push(value);

            incoming_scratch.copy_from_slice(&lag_state);
            let mut primary = PrimaryLagAccum {
                accumulator: &mut accum,
                weight_accum: &mut weight,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut downstream_completed_lags,
                n_completed: &mut downstream_n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                EntityMajor {
                    entity_count: 1,
                    max_order: l_state,
                },
                &mut lag_state,
                &incoming_scratch,
                &[value],
                &stage_lag_transitions[t],
                &mut primary,
                &mut downstream,
            );
        }

        // The window's realized observations: season 3 (April) and 4 (May) of
        // `window_year`, holding the forward-generated targets.
        let window_year = 2026;
        let inflow_history = vec![
            make_row(hydro_id, window_year, 3, targets[0]),
            make_row(hydro_id, window_year, 4, targets[1]),
        ];

        let mut lib =
            HistoricalScenarioLibrary::new(1, stages.len(), 1, l_state, vec![window_year]);
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &inflow_history,
            &par,
            &[window_year],
            Some(&season_map),
            DerivedSeed {
                lag_values: &derived.lag_values,
                l_state,
                accum: &derived.accum,
                weight: &derived.weight,
            },
            &stage_lag_transitions,
            0,
        );

        // Replay from the SAME seeded lag chain the production forward pass
        // would carry, inverting the stored eta; z must equal v.
        let mut replay_lag_state = derived.lag_values.clone();
        let mut replay_accum = derived.accum.clone();
        let mut replay_weight = derived.weight.clone();
        let mut replay_downstream_accumulator: Vec<f64> = Vec::new();
        let mut replay_downstream_weight_accum = 0.0_f64;
        let mut replay_downstream_completed_lags: Vec<f64> = Vec::new();
        let mut replay_downstream_n_completed = 0_usize;
        for (t, &target) in targets.iter().enumerate() {
            let eta = lib.eta_slice(0, t)[0];
            let det_base = par.deterministic_base(t, 0);
            let psi = par.psi_slice(t, 0);
            let sigma = par.sigma(t, 0);
            let reconstructed = evaluate_par(det_base, psi, &replay_lag_state, sigma, eta);
            assert!(
                (reconstructed - target).abs() < 1e-9,
                "stage {t}: mid-period replay reconstructed {reconstructed:.12} \
                 (z) != forward target {target:.12} (v)",
            );

            incoming_scratch.copy_from_slice(&replay_lag_state);
            let mut primary = PrimaryLagAccum {
                accumulator: &mut replay_accum,
                weight_accum: &mut replay_weight,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut replay_downstream_accumulator,
                weight_accum: &mut replay_downstream_weight_accum,
                completed_lags: &mut replay_downstream_completed_lags,
                n_completed: &mut replay_downstream_n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                EntityMajor {
                    entity_count: 1,
                    max_order: l_state,
                },
                &mut replay_lag_state,
                &incoming_scratch,
                &[target],
                &stage_lag_transitions[t],
                &mut primary,
                &mut downstream,
            );
        }
    }

    /// Two derived seeds differing in a single value must produce different
    /// `seed_digest`s — the cache-staleness signal a caller relies on to
    /// detect that the library needs re-standardising.
    #[test]
    fn historical_seed_digest_tracks_derived_seed() {
        let hydro = EntityId(1);
        let stages = vec![make_monthly_stage(0, 0), make_monthly_stage(1, 1)];
        let models = vec![
            InflowModel {
                hydro_id: hydro,
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.4],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: hydro,
                stage_id: 1,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.4],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();
        let history = vec![
            make_row(hydro, 2024, 0, 150.0),
            make_row(hydro, 2024, 1, 160.0),
        ];

        let seed_a = [100.0_f64];
        let seed_b = [999.0_f64];

        let mut lib_a = HistoricalScenarioLibrary::new(1, 2, 1, 1, vec![2024]);
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib_a, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib_a,
            &history,
            &par,
            &[2024],
            None,
            DerivedSeed {
                lag_values: &seed_a,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        let mut lib_b = HistoricalScenarioLibrary::new(1, 2, 1, 1, vec![2024]);
        let structure = check_historical_structure(&lib_b, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib_b,
            &history,
            &par,
            &[2024],
            None,
            DerivedSeed {
                lag_values: &seed_b,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        assert_ne!(
            lib_a.seed_digest(),
            lib_b.seed_digest(),
            "seeds differing in one lag value must fingerprint differently"
        );
        assert_ne!(
            lib_a.seed_digest(),
            0,
            "standardize_historical_windows must write a non-sentinel digest"
        );
    }

    #[test]
    fn standardize_replays_the_window_year_for_a_single_season_study() {
        let hydro = EntityId(1);
        let stages = vec![make_monthly_stage(0, 0)];
        let sm = monthly_season_map(MonthlyLabels::ZeroBased);

        let models = vec![
            InflowModel {
                hydro_id: hydro,
                stage_id: -1,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: hydro,
                stage_id: 0,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![0.0],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par = PrecomputedPar::build(&models, &stages, &[hydro], None).unwrap();

        let history: Vec<InflowHistoryRow> = (1990..=1993)
            .map(|y| make_row(hydro, y, 0, 1000.0 + f64::from(y - 1990)))
            .collect();

        let windows = crate::sampling::discover_historical_windows(
            &history,
            &[hydro],
            &stages,
            None,
            Some(&sm),
            10,
        )
        .unwrap();
        assert_eq!(windows, vec![1990, 1991, 1992, 1993]);

        let mut lib =
            HistoricalScenarioLibrary::new(windows.len(), stages.len(), 1, 1, windows.clone());
        let hydro_ids = [hydro];
        let structure = check_historical_structure(&lib, &hydro_ids, &stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            &history,
            &par,
            &windows,
            Some(&sm),
            DerivedSeed {
                lag_values: &[0.0],
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );

        for (w, &year) in windows.iter().enumerate() {
            let expected = 1000.0 + f64::from(year - 1990);
            assert_eq!(lib.eta_slice(w, 0)[0], expected);
        }
    }

    fn ar0_par(stages: &[Stage], moments: impl Fn(i32) -> (f64, f64)) -> PrecomputedPar {
        let models: Vec<InflowModel> = stages
            .iter()
            .map(|stage| {
                let (mean_m3s, std_m3s) = moments(stage.id);
                InflowModel {
                    hydro_id: EntityId(1),
                    stage_id: stage.id,
                    mean_m3s,
                    std_m3s,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                }
            })
            .collect();
        PrecomputedPar::build(&models, stages, &[EntityId(1)], None).unwrap()
    }

    /// Discovers hydro 1's windows and standardizes them, returning the windows
    /// and each window's per-stage η.
    fn discover_and_standardize(
        history: &[InflowHistoryRow],
        stages: &[Stage],
        season_map: Option<&SeasonMap>,
        par: &PrecomputedPar,
    ) -> (Vec<i32>, Vec<Vec<f64>>) {
        let hydro_ids = [EntityId(1)];
        let windows = crate::sampling::discover_historical_windows(
            history, &hydro_ids, stages, None, season_map, 1,
        )
        .unwrap();
        let mut lib =
            HistoricalScenarioLibrary::new(windows.len(), stages.len(), 1, 0, windows.clone());
        let structure = check_historical_structure(&lib, &hydro_ids, stages).unwrap();
        standardize_historical_windows(
            &structure,
            &mut lib,
            history,
            par,
            &windows,
            season_map,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &[],
            0,
        );
        let eta = (0..windows.len())
            .map(|w| (0..stages.len()).map(|t| lib.eta_slice(w, t)[0]).collect())
            .collect();
        (windows, eta)
    }

    /// ISO 2019-W01 starts on Monday 2018-12-31.
    #[test]
    fn weekly_windows_replay_their_own_iso_year() {
        let sm = weekly_season_map();
        let study_start = NaiveDate::from_isoywd_opt(2024, 1, Weekday::Mon).unwrap();
        let stages: Vec<Stage> = (0..52_usize)
            .map(|week| {
                let start = study_start + TimeDelta::weeks(i64::try_from(week).unwrap());
                let id = i32::try_from(week).unwrap();
                dated_stage(week, id, start, start + TimeDelta::weeks(1), week)
            })
            .collect();
        let last_monday = NaiveDate::from_isoywd_opt(2019, 52, Weekday::Mon).unwrap();
        let history: Vec<InflowHistoryRow> = std::iter::successors(
            NaiveDate::from_isoywd_opt(2017, 1, Weekday::Mon),
            |monday| Some(*monday + TimeDelta::weeks(1)),
        )
        .take_while(|monday| *monday <= last_monday)
        .map(|monday| {
            let iso = monday.iso_week();
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: monday,
                end_date: monday + TimeDelta::weeks(1),
                value_m3s: f64::from(iso.year() * 100) + f64::from(iso.week()),
            }
        })
        .collect();

        let par = ar0_par(&stages, |_| (0.0, 1.0));
        let (windows, eta) = discover_and_standardize(&history, &stages, Some(&sm), &par);

        assert_eq!(windows, vec![2017, 2018, 2019]);
        for (replayed, &year) in eta.iter().zip(&windows) {
            let own_weeks: Vec<f64> = (1..=52).map(|week| f64::from(year * 100 + week)).collect();
            assert_eq!(replayed, &own_weeks, "window {year}");
        }
    }

    /// The wet season runs 15 December to 14 March, so its January–March rows
    /// belong to the previous year's occurrence and no row is dated 1999.
    #[test]
    fn custom_windows_replay_the_occurrence_starting_in_their_december() {
        let season =
            |id: usize, label: &str, start: (u32, u32), end: (u32, u32)| SeasonDefinition {
                id,
                label: label.to_string(),
                month_start: start.0,
                day_start: Some(start.1),
                month_end: Some(end.0),
                day_end: Some(end.1),
            };
        let sm = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                season(0, "Wet", (12, 15), (3, 14)),
                season(1, "Dry", (3, 15), (12, 14)),
            ],
        };
        let stages = vec![
            dated_stage(0, 0, date(2030, 12, 15), date(2031, 3, 15), 0),
            dated_stage(1, 1, date(2031, 3, 15), date(2031, 12, 15), 1),
        ];
        let history: Vec<InflowHistoryRow> = (2000..=2003)
            .flat_map(|year| {
                (0..12_u32).map(move |month0| {
                    let value = if month0 < 3 {
                        1000 + year - 1
                    } else {
                        2000 + year
                    };
                    make_row(EntityId(1), year, month0, f64::from(value))
                })
            })
            .collect();

        let par = ar0_par(&stages, |_| (0.0, 1.0));
        let (windows, eta) = discover_and_standardize(&history, &stages, Some(&sm), &par);

        assert_eq!(windows, vec![1999, 2000, 2001, 2002]);
        for (replayed, &year) in eta.iter().zip(&windows) {
            let own_occurrences = vec![f64::from(1000 + year), f64::from(2000 + year + 1)];
            assert_eq!(replayed, &own_occurrences, "window {year}");
        }
    }

    #[test]
    fn no_map_windows_and_eta_keep_calendar_year_keying() {
        let stages: Vec<Stage> = (0..12_u32)
            .map(|k| {
                let start = date(2024, 7, 1) + Months::new(k);
                let index = usize::try_from(k).unwrap();
                let id = i32::try_from(k).unwrap();
                dated_stage(
                    index,
                    id,
                    start,
                    start + Months::new(1),
                    start.month0() as usize,
                )
            })
            .collect();
        let observed = |year: i32, month0: u32| {
            100.0 + 7.0 * f64::from(year - 1990) + 1.25 * f64::from(month0)
        };
        let history: Vec<InflowHistoryRow> = (1990..=1993)
            .flat_map(|year| {
                (0..12_u32)
                    .map(move |month0| make_row(EntityId(1), year, month0, observed(year, month0)))
            })
            .collect();
        let par = ar0_par(&stages, |id| {
            (40.0 + f64::from(id), 3.0 + 0.5 * f64::from(id))
        });

        let (windows, eta) = discover_and_standardize(&history, &stages, None, &par);

        assert_eq!(windows, vec![1990, 1991, 1992]);
        for (replayed, &year) in eta.iter().zip(&windows) {
            let expected: Vec<f64> = stages
                .iter()
                .enumerate()
                .map(|(t, stage)| {
                    let obs_year = year + stage.start_date.year() - 2024;
                    let target = observed(obs_year, stage.start_date.month0());
                    (target - par.deterministic_base(t, 0)) / par.sigma(t, 0)
                })
                .collect();
            assert_eq!(replayed, &expected, "window {year}");
        }
    }
}
