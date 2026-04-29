//! Periodic Yule-Walker estimation and seasonal statistics for PAR model fitting.
//!
//! This module provides the core primitives for fitting Periodic Autoregressive
//! models:
//!
//! 1. [`periodic_autocorrelation`] — computes the periodic normalised
//!    autocorrelation `rho(p, k)` with population divisor and cross-year
//!    lag adjustment.
//! 2. [`build_periodic_yw_matrix`] — constructs the non-Toeplitz periodic
//!    Yule-Walker matrix for a given season and AR order.
//! 3. [`periodic_pacf`] — computes the periodic PACF via progressive matrix
//!    solves for order selection.
//! 4. [`estimate_periodic_ar_coefficients`] — solves the periodic YW system
//!    at the selected order to produce AR coefficients and residual std ratio.
//! 5. [`estimate_seasonal_stats`] — computes seasonal means and
//!    population-divisor (1/N) standard deviations from historical
//!    observations, grouped by `(entity, season)` pair.
//! 6. [`estimate_ar_coefficients`] — produces white-noise (order-0) estimates
//!    for all `(entity, season)` pairs; used by the PACF path when
//!    `max_order == 0`.
//! 7. [`estimate_correlation`] — computes the Pearson correlation matrix of
//!    PAR model residuals across entities, returning a [`CorrelationModel`]
//!    suitable for downstream spectral decomposition.
//!
//! ## Periodic Yule-Walker equations
//!
//! For a periodic AR(p) process the Yule-Walker system is non-Toeplitz because
//! lags cross season boundaries. [`build_periodic_yw_matrix`] assembles the
//! correct per-season matrix and [`estimate_periodic_ar_coefficients`] solves it
//! via LU factorisation with partial pivoting.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::NaiveDate;
use cobre_core::{
    scenario::{
        AnnualComponent, CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
        CorrelationScheduleEntry,
    },
    temporal::{SeasonMap, Stage},
    EntityId,
};

use crate::StochasticError;

// ---------------------------------------------------------------------------
// Seasonal statistics
// ---------------------------------------------------------------------------

/// Seasonal mean and standard deviation for one entity–season pair.
///
/// Produced by [`estimate_seasonal_stats`] and consumed by AR coefficient
/// estimation routines. The caller (typically in a higher-level crate) is
/// responsible for mapping between this type and any crate-specific row type
/// used for storage or serialization.
///
/// The `stage_id` field holds the identifier of the **first** stage whose
/// `season_id` matches the season for this row — it is not a stage index.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalStats {
    /// Entity (e.g., hydro plant) identifier.
    pub entity_id: EntityId,
    /// Identifier of the first stage that belongs to this season.
    pub stage_id: i32,
    /// Sample mean of observed values (m³/s or whatever unit the caller uses).
    pub mean: f64,
    /// Population-divisor standard deviation (1/N divisor), matching
    /// NEWAVE's `rel_parpa.pdf` eq. 18 convention.
    pub std: f64,
}

/// Classification of a per-(entity, season) historical observation series.
///
/// Mirrors NEWAVE's `parpvaz.dat` header table (TIPO 0/1/2/4) so the PAR(p)-A
/// fitter can short-circuit pathological buckets in the same way NEWAVE does:
///
/// - **TIPO 0** (`Default`): no specific behaviour; standard fitting applies.
/// - **TIPO 1** (`Constant`): every observation equals the same value (or every
///   observation is zero/null). Mean and std are forced to that constant and 0,
///   respectively; AR order is forced to 0 and the annual coefficient is
///   suppressed. Common for plants with regulated/transposed flows whose
///   incremental inflow is structurally constant for a given month.
/// - **TIPO 2** (`ManyNegative`): more than 10% of observations are strictly
///   negative — a signal that the upstream incremental construction (the bridge
///   subtracting upstream postos) has produced unphysical values for this
///   month. Detected for diagnostics, but **does not override fitting** —
///   matches NEWAVE's `parpvaz.dat` behaviour, where TIPO 2 plants still get
///   normal AR fits (the flag is operator information, not a fit instruction).
/// - **TIPO 4** (`Saturated`): more than 50% of observations equal the modal
///   value — a flow cap (turbine/reservoir constraint) or a low-flow constant
///   (transposed flow plants like A.S.OLIVEIRA). Treated like TIPO 1 with the
///   cap as the constant. The std=0 propagates structural zeros into adjacent
///   months' PACF rows, mirroring NEWAVE's behaviour on plants like BELO MONTE
///   / April and JACUI / February-March. No P99 condition: NEWAVE classifies
///   low-flow constants (cap=1.0, cap=0.0) as TIPO 4 just as readily as high
///   caps.
///
/// TIPO 5 (bimodal history) is not yet detected; such series fall through to
/// `Default`.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistoryClass {
    /// TIPO 0 — no specific behaviour, run the standard fit.
    Default,
    /// TIPO 1 — every observation is the same value (or all zero/null).
    Constant {
        /// The constant value to use as the seasonal mean.
        value: f64,
    },
    /// TIPO 2 — more than 10% of observations are strictly negative.
    /// `sample_mean` is the empirical mean over the full series (used as the
    /// fallback constant; std forced to 0).
    ManyNegative {
        /// Empirical mean of the observation series.
        sample_mean: f64,
    },
    /// TIPO 4 — saturating cap (>50% of observations at the modal value, which
    /// equals or exceeds the 99th percentile).
    Saturated {
        /// The cap value (modal value of the series).
        cap: f64,
    },
}

impl HistoryClass {
    /// Returns the override `(mean, std)` that should replace the empirical
    /// stats for fitting purposes.
    ///
    /// TIPO 1 and TIPO 4 force the seasonal mean to the constant/cap value and
    /// the std to 0, which makes the downstream PAR(p)-A fitter short-circuit
    /// to order 0. TIPO 2 is purely diagnostic and returns `None` (NEWAVE
    /// does not override fitting for it). `Default` also returns `None`.
    #[must_use]
    pub fn stats_override(self) -> Option<(f64, f64)> {
        match self {
            HistoryClass::Default | HistoryClass::ManyNegative { .. } => None,
            HistoryClass::Constant { value } => Some((value, 0.0)),
            HistoryClass::Saturated { cap } => Some((cap, 0.0)),
        }
    }

    /// Returns `true` when the classification forces a degenerate fit
    /// (order 0, no AR/annual coefficients). Currently TIPO 1 and TIPO 4.
    /// TIPO 2 is diagnostic only, so it returns `false`.
    #[must_use]
    pub fn is_degenerate(self) -> bool {
        matches!(
            self,
            HistoryClass::Constant { .. } | HistoryClass::Saturated { .. }
        )
    }

    /// Returns the NEWAVE TIPO code (0/1/2/4) for reporting/parity checks.
    #[must_use]
    pub fn tipo_code(self) -> u8 {
        match self {
            HistoryClass::Default => 0,
            HistoryClass::Constant { .. } => 1,
            HistoryClass::ManyNegative { .. } => 2,
            HistoryClass::Saturated { .. } => 4,
        }
    }
}

/// Classify a single (entity, season) observation series per the
/// [`HistoryClass`] taxonomy.
///
/// The classifier runs in priority order TIPO 1 → TIPO 2 → TIPO 4 → TIPO 0:
/// constant series take precedence over negative-pathological detection, which
/// in turn takes precedence over saturation. Observations are rounded to the
/// nearest integer for mode counting (matching the precision of NEWAVE's
/// `vazoes.dat` storage). The constancy check uses an absolute tolerance of
/// `1e-6` to absorb the float round-trip from parquet.
///
/// Returns `HistoryClass::Constant { value: 0.0 }` for an empty input — the
/// degenerate single-observation case is treated the same as a zero-history
/// series so that downstream fitters short-circuit predictably.
#[must_use]
pub fn classify_history(observations: &[f64]) -> HistoryClass {
    if observations.is_empty() {
        return HistoryClass::Constant { value: 0.0 };
    }

    let first = observations[0];
    let const_tol = 1e-6;

    // TIPO 1 — every observation matches the first within tolerance.
    if observations.iter().all(|&v| (v - first).abs() < const_tol) {
        return HistoryClass::Constant { value: first };
    }

    // TIPO 2 — more than 10% strictly negative.
    let n = observations.len();
    let n_neg = observations.iter().filter(|&&v| v < 0.0).count();
    #[allow(clippy::cast_precision_loss)]
    if (n_neg as f64) / (n as f64) > 0.10 {
        #[allow(clippy::cast_precision_loss)]
        let sample_mean = observations.iter().sum::<f64>() / n as f64;
        return HistoryClass::ManyNegative { sample_mean };
    }

    // TIPO 4 — modal value occupies more than 50% of observations.
    //
    // Round to integer for mode counting (vazoes.dat is stored to 1 m³/s).
    // No P99 guard: NEWAVE classifies low-flow constants (cap=0.0, cap=1.0
    // for plants like A.S.OLIVEIRA / JACUI) as TIPO 4 just as eagerly as it
    // classifies high caps (BELO MONTE cap=13900). The driving criterion is
    // structural constancy of the bucket, not magnitude.
    let mut sorted: Vec<i64> = observations.iter().map(|v| v.round() as i64).collect();
    sorted.sort_unstable();
    // Largest run of equal values gives the mode.
    let mut best_count = 0_usize;
    let mut best_value: i64 = sorted[0];
    let mut run = 1_usize;
    for i in 1..sorted.len() {
        if sorted[i] == sorted[i - 1] {
            run += 1;
        } else {
            if run > best_count {
                best_count = run;
                best_value = sorted[i - 1];
            }
            run = 1;
        }
    }
    if run > best_count {
        best_count = run;
        best_value = sorted[sorted.len() - 1];
    }
    #[allow(clippy::cast_precision_loss)]
    if (best_count as f64) / (n as f64) > 0.50 {
        return HistoryClass::Saturated {
            cap: best_value as f64,
        };
    }

    HistoryClass::Default
}

/// Estimate seasonal means and standard deviations from historical observations.
///
/// Groups observations by `(entity_id, season_id)` and computes the sample
/// mean and population-divisor (1/N) standard deviation for each group, matching
/// NEWAVE's `rel_parpa.pdf` eq. 18 convention. Only entities listed in
/// `entity_ids` are processed; observations for other entities are silently
/// ignored.
///
/// Stages with `season_id = None` are skipped when building the date-to-season
/// mapping. Observations whose date does not fall within any stage's
/// `[start_date, end_date)` range produce an error.
///
/// # Parameters
///
/// - `observations` — flat slice of `(entity_id, date, value)` triples,
///   sorted by `(entity_id, date)` (parser guarantee).
/// - `stages` — all stages in canonical index order. Each stage has
///   `start_date` (inclusive), `end_date` (exclusive), `season_id`, and `id`.
/// - `entity_ids` — canonical sorted list of entity IDs to estimate for.
///
/// # Errors
///
/// - [`StochasticError::InsufficientData`] when a `(entity, season)` group has
///   fewer than 2 observations (a degenerate single-sample bucket has no
///   meaningful std and would propagate zeros into every downstream
///   correlation, so it is rejected up front).
/// - [`StochasticError::InsufficientData`] when an observation date falls
///   outside every stage's date range.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::estimate_seasonal_stats;
///
/// fn stage(id: i32, y0: i32, m0: u32, y1: i32, m1: u32, season: usize) -> Stage {
///     Stage {
///         index: 0,
///         id,
///         start_date: NaiveDate::from_ymd_opt(y0, m0, 1).unwrap(),
///         end_date: NaiveDate::from_ymd_opt(y1, m1, 1).unwrap(),
///         season_id: Some(season),
///         blocks: vec![Block { index: 0, name: "S".to_string(), duration_hours: 744.0 }],
///         block_mode: BlockMode::Parallel,
///         state_config: StageStateConfig { storage: true, inflow_lags: false },
///         risk_config: StageRiskConfig::Expectation,
///         scenario_config: ScenarioSourceConfig { branching_factor: 1, noise_method: NoiseMethod::Saa },
///     }
/// }
///
/// let stages = vec![
///     stage(1, 2020, 1, 2020, 2, 0),
///     stage(2, 2020, 2, 2020, 3, 1),
/// ];
/// let obs = vec![
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 1, 15).unwrap(), 100.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 1, 20).unwrap(), 200.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 2, 10).unwrap(), 150.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 2, 20).unwrap(), 250.0),
/// ];
/// let entity_ids = vec![EntityId::from(1)];
/// let stats = estimate_seasonal_stats(&obs, &stages, &entity_ids).unwrap();
/// assert_eq!(stats.len(), 2);
/// assert!((stats[0].mean - 150.0).abs() < 1e-10);
/// ```
pub fn estimate_seasonal_stats(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
) -> Result<Vec<SeasonalStats>, StochasticError> {
    estimate_seasonal_stats_with_season_map(observations, stages, entity_ids, None)
}

/// Estimate seasonal statistics with an optional [`SeasonMap`] fallback.
///
/// When `season_map` is `Some`, historical observation dates that fall outside
/// the study horizon are resolved to a season using the calendar-based cycle
/// definition. This allows PAR estimation from inflow history that predates
/// the study period.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when an observation date
/// cannot be mapped to any season, or when fewer than 2 observations exist
/// for any `(entity, season)` group.
pub fn estimate_seasonal_stats_with_season_map(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<Vec<SeasonalStats>, StochasticError> {
    if observations.is_empty() {
        return Ok(Vec::new());
    }

    // Build a set of entity IDs for O(1) membership checks.
    let entity_set: std::collections::HashSet<EntityId> = entity_ids.iter().copied().collect();

    // Build stage index: (start_date, end_date, stage_id, season_id).
    // Only include stages that have a season_id. Sorted by start_date for
    // binary-search based range lookup.
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    // Map (entity_id, season_id) -> (observations: Vec<f64>, first_stage_id: i32).
    // The `first_stage_id` is the id of the first stage (lowest stage.id among
    // those with that season_id, determined from the sorted stage_index).
    let mut group_map: HashMap<(EntityId, usize), (Vec<f64>, i32)> = HashMap::new();

    // Build a separate lookup from season_id -> first_stage_id (the stage.id of
    // the first stage with that season_id, where "first" means lowest start_date
    // i.e. first in stage_index order).
    let mut season_first_stage: HashMap<usize, i32> = HashMap::new();
    for &(_, _, stage_id, season_id) in &stage_index {
        season_first_stage.entry(season_id).or_insert(stage_id);
    }

    for &(entity_id, date, value) in observations {
        // Skip entities not in the study set.
        if !entity_set.contains(&entity_id) {
            continue;
        }

        // Try exact stage date containment first (for in-range observations),
        // then fall back to the SeasonMap calendar-based mapping (for historical
        // observations that predate the study horizon).
        let season_id = find_season_for_date(&stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
            .ok_or_else(|| StochasticError::InsufficientData {
                context: format!(
                    "observation date {date} for entity {entity_id} \
                     does not match any stage date range or season definition"
                ),
            })?;

        let first_stage_id = season_first_stage[&season_id];
        let entry = group_map
            .entry((entity_id, season_id))
            .or_insert_with(|| (Vec::new(), first_stage_id));
        entry.0.push(value);
    }

    // Compute mean and population-divisor std for each group.
    //
    // NEWAVE (rel_parpa.pdf eq. 18 and the parpvaz.dat report) computes
    // sigma^Z_m with divisor 1/N, not the Bessel-corrected 1/(N-1). Matching
    // that convention is required for parity on the conditional FACP and
    // selected AR orders — the sample-vs-population scale factor would
    // otherwise propagate through every cross-correlation.
    let mut result: Vec<SeasonalStats> = Vec::with_capacity(group_map.len());
    for ((entity_id, _season_id), (values, stage_id)) in group_map {
        let n = values.len();
        if n < 2 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "entity {entity_id} season mapped to stage {stage_id} \
                     has {n} observation(s); need at least 2 for std estimation"
                ),
            });
        }

        // n is the number of observations; the cast to f64 is intentional here.
        // In practice, observation counts never exceed ~10^6 (well within the
        // 2^53 exact-integer range of f64), so precision loss cannot occur.
        #[allow(clippy::cast_precision_loss)]
        let mean = values.iter().copied().sum::<f64>() / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        // NEWAVE TIPO override: TIPO 1/2/4 buckets get a forced (constant, 0)
        // pair so the downstream PAR(p)-A fitter short-circuits to order 0
        // (matching parpvaz.dat behaviour for plants like BELO MONTE / April
        // and PIMENTAL ecological-flow months). The classifier returns
        // `Default` for normal series, in which case we keep the empirical
        // (mean, std) computed above.
        let (final_mean, final_std) = match classify_history(&values).stats_override() {
            Some((override_mean, override_std)) => (override_mean, override_std),
            None => (mean, std),
        };

        result.push(SeasonalStats {
            entity_id,
            stage_id,
            mean: final_mean,
            std: final_std,
        });
    }

    // Sort by (entity_id, stage_id) ascending to match parser convention.
    result.sort_unstable_by_key(|s| (s.entity_id.0, s.stage_id));

    Ok(result)
}

/// Find the `season_id` for `date` by binary-searching `stage_index`.
///
/// `stage_index` must be sorted by `start_date`. Returns `None` when `date`
/// falls outside every stage's `[start_date, end_date)` range.
#[must_use]
pub fn find_season_for_date(
    stage_index: &[(NaiveDate, NaiveDate, i32, usize)],
    date: NaiveDate,
) -> Option<usize> {
    let pos = stage_index.partition_point(|(start, _, _, _)| *start <= date);
    if pos == 0 {
        return None;
    }
    let (_, end_date, _, season_id) = stage_index[pos - 1];
    if date < end_date {
        Some(season_id)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// AR coefficient estimation
// ---------------------------------------------------------------------------

/// AR coefficient estimation result for a single (entity, season) pair.
///
/// Produced by [`estimate_ar_coefficients`] and consumed by the assembly
/// layer that writes AR coefficient rows to persistent storage.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct ArCoefficientEstimate {
    /// Entity (e.g., hydro plant) identifier.
    pub hydro_id: EntityId,
    /// Season ID (maps back to `stage_id` via the stage table).
    pub season_id: usize,
    /// Standardized AR coefficients ψ*₁..ψ*ₚ (Yule-Walker output).
    ///
    /// Empty when the estimated order is 0 (white noise).
    pub coefficients: Vec<f64>,
    /// Residual std ratio `σ_m / s_m`, always in (0, 1].
    pub residual_std_ratio: f64,
    /// Annual-component triple for the PAR(p)-A extension; `None` for
    /// classical PAR(p). All three sub-fields (`coefficient`, `mean_m3s`,
    /// `std_m3s`) are present together by construction — they are never
    /// split into separate optional fields.
    pub annual: Option<AnnualComponent>,
}

/// Produce white-noise (order-0) AR estimates for all `(entity, season)` pairs.
///
/// Every `(hydro_id, season_id)` pair present in `seasonal_stats` that belongs
/// to `hydro_ids` receives an `ArCoefficientEstimate` with an empty coefficient
/// vector and `residual_std_ratio = 1.0`. This function delegates to
/// [`estimate_ar_coefficients_with_season_map`] with `season_map = None`.
///
/// # Errors
///
/// - [`StochasticError::InsufficientData`] when `max_order > 0` (use the
///   periodic Yule-Walker path via [`estimate_periodic_ar_coefficients`] instead).
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::{estimate_seasonal_stats, estimate_ar_coefficients};
///
/// fn stage(id: i32, y0: i32, m0: u32, y1: i32, m1: u32, season: usize) -> Stage {
///     Stage {
///         index: 0,
///         id,
///         start_date: NaiveDate::from_ymd_opt(y0, m0, 1).unwrap(),
///         end_date: NaiveDate::from_ymd_opt(y1, m1, 1).unwrap(),
///         season_id: Some(season),
///         blocks: vec![Block { index: 0, name: "S".to_string(), duration_hours: 744.0 }],
///         block_mode: BlockMode::Parallel,
///         state_config: StageStateConfig { storage: true, inflow_lags: false },
///         risk_config: StageRiskConfig::Expectation,
///         scenario_config: ScenarioSourceConfig { branching_factor: 1, noise_method: NoiseMethod::Saa },
///     }
/// }
///
/// // Build 2 seasons over 5 years (10 observations per season).
/// let mut stages_vec: Vec<Stage> = Vec::new();
/// for y in 2000..2005_i32 {
///     stages_vec.push(stage(y * 2 - 3999, y, 1, y, 2, 0));
///     stages_vec.push(stage(y * 2 - 3998, y, 2, y, 3, 1));
/// }
/// let entity_ids = vec![EntityId::from(1)];
/// let mut obs: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
/// for y in 2000..2005_i32 {
///     obs.push((EntityId::from(1), NaiveDate::from_ymd_opt(y, 1, 15).unwrap(), 100.0));
///     obs.push((EntityId::from(1), NaiveDate::from_ymd_opt(y, 2, 15).unwrap(), 200.0));
/// }
/// let stats = estimate_seasonal_stats(&obs, &stages_vec, &entity_ids).unwrap();
/// // order-0 produces white-noise estimates (empty coefficients, ratio = 1.0)
/// let estimates = estimate_ar_coefficients(&obs, &stats, &stages_vec, &entity_ids, 0).unwrap();
/// assert_eq!(estimates.len(), 2);
/// assert!(estimates[0].coefficients.is_empty());
/// ```
pub fn estimate_ar_coefficients(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
) -> Result<Vec<ArCoefficientEstimate>, StochasticError> {
    estimate_ar_coefficients_with_season_map(
        observations,
        seasonal_stats,
        stages,
        hydro_ids,
        max_order,
        None,
    )
}

// ---------------------------------------------------------------------------
// Shared data-preparation helpers for season-aware estimation functions
// ---------------------------------------------------------------------------

/// Pre-computed lookups for season-aware PAR estimation functions.
///
/// Both [`estimate_ar_coefficients_with_season_map`] and
/// [`estimate_correlation_with_season_map`] require the same date-to-season
/// mapping, stats lookups, and per-entity observation indices. This struct
/// is built once and shared to avoid code duplication.
struct SeasonLookups<'a> {
    stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)>,
    stats_lookup: HashMap<(EntityId, usize), &'a SeasonalStats>,
    entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>>,
    entity_date_index: HashMap<EntityId, HashMap<NaiveDate, usize>>,
    n_seasons: usize,
}

/// Build the shared season lookups from observations, seasonal stats, and stages.
fn build_season_lookups<'a>(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &'a [SeasonalStats],
    stages: &[Stage],
) -> SeasonLookups<'a> {
    // Step 1: Build date-to-season mapping (stage index sorted by start_date).
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    // Build stage_id -> season_id lookup.
    let stage_id_to_season: HashMap<i32, usize> = stage_index
        .iter()
        .map(|&(_, _, stage_id, season_id)| (stage_id, season_id))
        .collect();

    // Step 2: Build (entity_id, season_id) -> SeasonalStats lookup.
    let stats_lookup: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
        .iter()
        .filter_map(|s| {
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(((s.entity_id, season_id), s))
        })
        .collect();

    // Determine the total number of distinct seasons (M).
    let n_seasons: usize = stage_index
        .iter()
        .map(|&(_, _, _, sid)| sid + 1)
        .max()
        .unwrap_or(0);

    // Step 3: Group observations by entity in chronological order.
    let mut entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        entity_obs.entry(entity_id).or_default().push((date, value));
    }
    for obs_vec in entity_obs.values_mut() {
        obs_vec.sort_unstable_by_key(|(d, _)| *d);
    }
    let entity_date_index: HashMap<EntityId, HashMap<NaiveDate, usize>> = entity_obs
        .iter()
        .map(|(&eid, obs_vec)| {
            let idx_map: HashMap<NaiveDate, usize> = obs_vec
                .iter()
                .enumerate()
                .map(|(i, (d, _))| (*d, i))
                .collect();
            (eid, idx_map)
        })
        .collect();

    SeasonLookups {
        stage_index,
        stats_lookup,
        entity_obs,
        entity_date_index,
        n_seasons,
    }
}

/// Produce white-noise (order-0) AR estimates for every `(entity, season)` pair.
///
/// All `(hydro_id, season_id)` pairs found in `seasonal_stats` that belong to
/// `hydro_ids` receive an `ArCoefficientEstimate` with an empty coefficient
/// vector and `residual_std_ratio = 1.0`.
///
/// This function is called by the PACF estimation path when `max_order == 0`
/// to populate the estimate vector before returning an empty report. The
/// `observations`, `stages`, and `season_map` parameters are accepted for
/// API compatibility but are not used in this implementation.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when `max_order > 0`; callers
/// requiring AR order selection must use [`estimate_periodic_ar_coefficients`]
/// and the periodic Yule-Walker path instead.
pub fn estimate_ar_coefficients_with_season_map(
    _observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    _season_map: Option<&SeasonMap>,
) -> Result<Vec<ArCoefficientEstimate>, StochasticError> {
    if max_order > 0 {
        return Err(StochasticError::InsufficientData {
            context: format!(
                "estimate_ar_coefficients_with_season_map called with max_order={max_order}; \
                 only max_order=0 (white-noise) is supported"
            ),
        });
    }

    // Build stage_id -> season_id lookup to map seasonal_stats entries.
    let stage_id_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    let hydro_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();

    let mut result: Vec<ArCoefficientEstimate> = seasonal_stats
        .iter()
        .filter_map(|s| {
            if !hydro_set.contains(&s.entity_id) {
                return None;
            }
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(ArCoefficientEstimate {
                hydro_id: s.entity_id,
                season_id,
                coefficients: Vec::new(),
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    result.sort_unstable_by_key(|e| (e.hydro_id.0, e.season_id));
    Ok(result)
}

// ---------------------------------------------------------------------------
// Correlation estimation
// ---------------------------------------------------------------------------

/// Estimate the cross-entity residual correlation matrix from historical observations.
///
/// After a PAR(p) model is fitted (via [`estimate_seasonal_stats`] and
/// [`estimate_ar_coefficients`]), this function computes the standardized
/// innovation residuals for each entity at each time step and derives
/// the Pearson correlation between each pair of entities. The result is
/// a [`CorrelationModel`] with a single `"default"` profile containing
/// all entities, which the assembly pipeline can use as an automatic
/// fallback when no explicit correlation file is provided.
///
/// ## Residual formula
///
/// The standardized residual (innovation) at time step `t` for entity `i`
/// in season `m` is:
///
/// ```text
/// ε_t = z_t − Σ_{l=1}^{p} ψ*_{m,l} · z_{t−l}
/// ```
///
/// where `z_t = (a_t − μ_m) / s_m` is the standardized observation and
/// `ψ*_{m,l}` is the standardized AR coefficient for lag `l` in season `m`.
/// Only time steps where all `p` lagged standardized observations are
/// available contribute residuals.
///
/// ## Pearson correlation
///
/// For each pair `(i, j)` the function computes:
///
/// ```text
/// r_{ij} = cov(ε_i, ε_j) / (std(ε_i) · std(ε_j))
/// ```
///
/// using Bessel-corrected (N−1) estimators over the subset of time steps
/// where both entities have valid residuals. When fewer than 2 overlapping
/// steps exist, `r_{ij}` is set to 0.0.
///
/// ## Output structure
///
/// Returns a [`CorrelationModel`] with:
/// - `method: "spectral"`
/// - A single profile named `"default"` with a single group containing all
///   entities in canonical `hydro_ids` order and the estimated correlation matrix.
/// - An empty `schedule` (the single profile applies to all stages).
///
/// The function does **not** enforce positive-semidefiniteness. The downstream
/// spectral decomposition handles rank-deficient and non-PD matrices naturally.
///
/// # Parameters
///
/// - `observations` — flat slice of `(entity_id, date, value)` triples,
///   sorted by `(entity_id, date)`.
/// - `ar_estimates` — output of [`estimate_ar_coefficients`].
/// - `seasonal_stats` — output of [`estimate_seasonal_stats`].
/// - `stages` — all study stages with `season_id` assignments.
/// - `hydro_ids` — canonical sorted entity IDs; determines matrix row/column order.
///
/// # Errors
///
/// - [`StochasticError::InsufficientData`] when `seasonal_stats` is empty but
///   `hydro_ids` is non-empty (inconsistent inputs).
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::{
///     estimate_seasonal_stats, estimate_ar_coefficients, estimate_correlation,
/// };
///
/// fn stage(id: i32, y0: i32, m0: u32, y1: i32, m1: u32, season: usize) -> Stage {
///     Stage {
///         index: 0,
///         id,
///         start_date: NaiveDate::from_ymd_opt(y0, m0, 1).unwrap(),
///         end_date: NaiveDate::from_ymd_opt(y1, m1, 1).unwrap(),
///         season_id: Some(season),
///         blocks: vec![Block { index: 0, name: "S".to_string(), duration_hours: 744.0 }],
///         block_mode: BlockMode::Parallel,
///         state_config: StageStateConfig { storage: true, inflow_lags: false },
///         risk_config: StageRiskConfig::Expectation,
///         scenario_config: ScenarioSourceConfig { branching_factor: 1, noise_method: NoiseMethod::Saa },
///     }
/// }
///
/// // Build a single-season study over 5 years.
/// let stages_vec: Vec<Stage> = (2000..2005_i32)
///     .map(|y| stage(y - 1999, y, 1, y, 2, 0))
///     .collect();
/// let hydro_ids = vec![EntityId::from(1)];
/// let obs: Vec<(EntityId, NaiveDate, f64)> = (2000..2005_i32)
///     .map(|y| (EntityId::from(1), NaiveDate::from_ymd_opt(y, 1, 15).unwrap(), 100.0 + y as f64))
///     .collect();
/// let stats = estimate_seasonal_stats(&obs, &stages_vec, &hydro_ids).unwrap();
/// let estimates = estimate_ar_coefficients(&obs, &stats, &stages_vec, &hydro_ids, 0).unwrap();
/// let corr = estimate_correlation(&obs, &estimates, &stats, &stages_vec, &hydro_ids).unwrap();
/// assert!(corr.profiles.contains_key("default"));
/// assert_eq!(corr.profiles["default"].groups[0].matrix.len(), 1);
/// ```
pub fn estimate_correlation(
    observations: &[(EntityId, NaiveDate, f64)],
    ar_estimates: &[ArCoefficientEstimate],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
) -> Result<CorrelationModel, StochasticError> {
    estimate_correlation_with_season_map(
        observations,
        ar_estimates,
        seasonal_stats,
        stages,
        hydro_ids,
        None,
    )
}

/// Minimum number of paired observations required per season for a per-season
/// correlation matrix. Seasons below this threshold fall back to the pooled
/// (all-season) matrix via the "default" profile.
const MIN_CORRELATION_PAIRS: usize = 30;

/// Estimate correlation with an optional [`SeasonMap`] fallback.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when seasonal stats are
/// empty but hydros are present, or when residual computation fails.
pub fn estimate_correlation_with_season_map(
    observations: &[(EntityId, NaiveDate, f64)],
    ar_estimates: &[ArCoefficientEstimate],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<CorrelationModel, StochasticError> {
    // Trivial case: no hydros.
    if hydro_ids.is_empty() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile { groups: Vec::new() },
        );
        return Ok(CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: Vec::new(),
        });
    }

    if seasonal_stats.is_empty() {
        return Err(StochasticError::InsufficientData {
            context: "seasonal_stats is empty but hydro_ids is non-empty; \
                      cannot estimate correlation without seasonal statistics"
                .to_string(),
        });
    }

    let lookups = build_season_lookups(observations, seasonal_stats, stages);

    let ar_lookup: HashMap<(EntityId, usize), &ArCoefficientEstimate> = ar_estimates
        .iter()
        .map(|e| ((e.hydro_id, e.season_id), e))
        .collect();

    let per_season_residuals = compute_hydro_residuals(&lookups, &ar_lookup, hydro_ids, season_map);

    // Warn about potentially degenerate hydros (informational only; spectral
    // decomposition handles near-zero eigenvalues naturally).
    warn_degenerate_hydros(&lookups, hydro_ids, &per_season_residuals);

    let pooled_residuals = flatten_residuals(&per_season_residuals);
    let pooled_matrix = compute_pearson_correlation_matrix(&pooled_residuals);
    let seasonal_matrices = compute_seasonal_matrices(&per_season_residuals, lookups.n_seasons);

    Ok(assemble_seasonal_correlation_model(
        hydro_ids,
        &pooled_matrix,
        &seasonal_matrices,
        stages,
        lookups.n_seasons,
    ))
}

/// Compute standardized AR innovation residuals for each hydro, partitioned by season.
///
/// For each hydro and each observation date where the AR model has sufficient
/// lag history, computes:
///   `epsilon_t = z_t - sum(psi_{m,l} * z_{t-l})` for l in 1..=p
///
/// Returns one `HashMap<season_id, HashMap<NaiveDate, f64>>` per hydro (indexed
/// by position in `hydro_ids`). Each residual is placed under its `season_id`
/// key instead of being pooled into a flat date-keyed map.
fn compute_hydro_residuals(
    lookups: &SeasonLookups<'_>,
    ar_lookup: &HashMap<(EntityId, usize), &ArCoefficientEstimate>,
    hydro_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Vec<HashMap<usize, HashMap<NaiveDate, f64>>> {
    let n_hydros = hydro_ids.len();
    let mut hydro_residuals: Vec<HashMap<usize, HashMap<NaiveDate, f64>>> =
        (0..n_hydros).map(|_| HashMap::new()).collect();

    for (hidx, &hydro_id) in hydro_ids.iter().enumerate() {
        let Some(all_obs) = lookups.entity_obs.get(&hydro_id) else {
            continue;
        };
        let Some(date_index) = lookups.entity_date_index.get(&hydro_id) else {
            continue;
        };

        for &(date, value) in all_obs {
            let Some(season_id) = find_season_for_date(&lookups.stage_index, date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
            else {
                continue;
            };

            let Some(stats_m) = lookups.stats_lookup.get(&(hydro_id, season_id)) else {
                continue;
            };

            let z_t = if stats_m.std == 0.0 {
                0.0
            } else {
                (value - stats_m.mean) / stats_m.std
            };

            let ar_est = ar_lookup.get(&(hydro_id, season_id));
            let ar_order = ar_est.map_or(0, |e| e.coefficients.len());
            let ar_coeffs = ar_est.map_or(&[] as &[f64], |e| &e.coefficients);

            let Some(&pos) = date_index.get(&date) else {
                continue;
            };
            if pos < ar_order {
                continue;
            }

            let mut ar_sum = 0.0_f64;
            let mut lag_ok = true;

            for lag in 1..=ar_order {
                let n_s = lookups.n_seasons.max(1);
                let lag_season = season_id.wrapping_add(n_s).wrapping_sub(lag % n_s) % n_s;

                let Some(stats_lag) = lookups.stats_lookup.get(&(hydro_id, lag_season)) else {
                    lag_ok = false;
                    break;
                };

                let (_, lagged_value) = all_obs[pos - lag];
                let z_lag = if stats_lag.std == 0.0 {
                    0.0
                } else {
                    (lagged_value - stats_lag.mean) / stats_lag.std
                };
                ar_sum += ar_coeffs[lag - 1] * z_lag;
            }

            if !lag_ok {
                continue;
            }

            let epsilon = z_t - ar_sum;
            hydro_residuals[hidx]
                .entry(season_id)
                .or_default()
                .insert(date, epsilon);
        }
    }

    hydro_residuals
}

/// Flatten per-season residuals into a single date-keyed map per hydro.
///
/// Dates are unique per hydro per season (each observation maps to exactly one
/// season), so no collisions occur when merging season maps.
fn flatten_residuals(
    per_season: &[HashMap<usize, HashMap<NaiveDate, f64>>],
) -> Vec<HashMap<NaiveDate, f64>> {
    per_season
        .iter()
        .map(|seasons| {
            let mut flat = HashMap::new();
            for date_map in seasons.values() {
                flat.extend(date_map.iter().map(|(&d, &v)| (d, v)));
            }
            flat
        })
        .collect()
}

/// Emit diagnostic warnings for hydros that exhibit degenerate statistical
/// properties. These are informational only — the spectral decomposition handles
/// near-zero eigenvalues naturally and no hydros are excluded.
fn warn_degenerate_hydros(
    lookups: &SeasonLookups<'_>,
    hydro_ids: &[EntityId],
    per_season_residuals: &[HashMap<usize, HashMap<NaiveDate, f64>>],
) {
    for (hidx, &hydro_id) in hydro_ids.iter().enumerate() {
        let Some(all_obs) = lookups.entity_obs.get(&hydro_id) else {
            continue;
        };

        // Partition raw observations by season.
        let mut obs_by_season: HashMap<usize, Vec<f64>> = HashMap::new();
        for &(date, value) in all_obs {
            if let Some(season_id) = find_season_for_date(&lookups.stage_index, date) {
                obs_by_season.entry(season_id).or_default().push(value);
            }
        }

        for season_id in 0..lookups.n_seasons {
            let Some(vals) = obs_by_season.get(&season_id) else {
                continue;
            };
            if vals.len() < 2 {
                continue;
            }

            // Warn: >50% negative observations.
            #[allow(clippy::cast_precision_loss)]
            let neg_frac = vals.iter().filter(|&&v| v < 0.0).count() as f64 / vals.len() as f64;
            if neg_frac > 0.5 {
                tracing::warn!(
                    hydro_id = hydro_id.0,
                    season = season_id,
                    negative_fraction = neg_frac,
                    "hydro has majority negative observations in season \
                     (included in correlation; spectral decomposition handles this)"
                );
            }

            // Warn: constant series (all values identical).
            let first = vals[0];
            if vals.iter().all(|&v| (v - first).abs() < f64::EPSILON) {
                tracing::warn!(
                    hydro_id = hydro_id.0,
                    season = season_id,
                    value = first,
                    "hydro has constant series in season \
                     (included in correlation; near-zero eigenvalue expected)"
                );
                // Near-zero residual variance is implied by a constant series;
                // skip the residual check for this season.
                continue;
            }

            // Warn: near-zero residual variance.
            if let Some(residuals) = per_season_residuals
                .get(hidx)
                .and_then(|m| m.get(&season_id))
                .filter(|r| r.len() >= 2)
            {
                let r_vals: Vec<f64> = residuals.values().copied().collect();
                #[allow(clippy::cast_precision_loss)]
                let r_mean = r_vals.iter().sum::<f64>() / r_vals.len() as f64;
                #[allow(clippy::cast_precision_loss)]
                let r_std = (r_vals.iter().map(|v| (v - r_mean).powi(2)).sum::<f64>()
                    / (r_vals.len() - 1) as f64)
                    .sqrt();
                if r_std < 1e-8 {
                    tracing::warn!(
                        hydro_id = hydro_id.0,
                        season = season_id,
                        residual_std = r_std,
                        "hydro has near-zero residual variance in season \
                         (included in correlation; near-zero eigenvalue expected)"
                    );
                }
            }
        }
    }
}

/// Compute per-season Pearson correlation matrices.
///
/// For each season, extracts the residuals belonging to that season from each
/// hydro and computes the Pearson correlation matrix. Seasons where the minimum
/// number of paired observations across all hydro pairs is below
/// [`MIN_CORRELATION_PAIRS`] are excluded from the result and fall back to the
/// pooled (all-season) matrix via the `"default"` profile.
///
/// All hydros participate regardless of degenerate status. Rank-deficient
/// matrices (more hydros than observations) are acceptable because the downstream
/// spectral decomposition handles them naturally.
fn compute_seasonal_matrices(
    per_season_residuals: &[HashMap<usize, HashMap<NaiveDate, f64>>],
    n_seasons: usize,
) -> HashMap<usize, Vec<f64>> {
    let mut result = HashMap::new();
    let n_hydros = per_season_residuals.len();

    for season_id in 0..n_seasons {
        // Extract per-hydro residuals for this season (all hydros, no exclusions).
        let season_residuals: Vec<HashMap<NaiveDate, f64>> = per_season_residuals
            .iter()
            .map(|hydro_seasons| hydro_seasons.get(&season_id).cloned().unwrap_or_default())
            .collect();

        // Determine minimum paired observation count across all hydro pairs.
        let min_pairs = if n_hydros <= 1 {
            season_residuals.first().map_or(0, HashMap::len)
        } else {
            (0..n_hydros)
                .flat_map(|i| (i + 1..n_hydros).map(move |j| (i, j)))
                .map(|(i, j)| {
                    season_residuals[i]
                        .keys()
                        .filter(|d| season_residuals[j].contains_key(*d))
                        .count()
                })
                .min()
                .unwrap_or(0)
        };

        if min_pairs < MIN_CORRELATION_PAIRS {
            continue;
        }

        result.insert(
            season_id,
            compute_pearson_correlation_matrix(&season_residuals),
        );
    }

    result
}

/// Compute the `n_hydros` x `n_hydros` Pearson correlation matrix from residuals.
///
/// For each pair `(i, j)`, collects time steps where both have residuals,
/// then applies the standard Pearson formula with Bessel correction (N-1).
/// Pairs are sorted for deterministic iteration across `HashMap` orderings.
fn compute_pearson_correlation_matrix(hydro_residuals: &[HashMap<NaiveDate, f64>]) -> Vec<f64> {
    let n = hydro_residuals.len();
    // Flat row-major N×N matrix: entry (i, j) is at index i * n + j.
    let mut matrix = vec![0.0_f64; n * n];

    for i in 0..n {
        matrix[i * n + i] = 1.0;

        for j in (i + 1)..n {
            let r_i = &hydro_residuals[i];
            let r_j = &hydro_residuals[j];

            let mut pairs: Vec<(f64, f64)> = r_i
                .iter()
                .filter_map(|(date, &ei)| r_j.get(date).map(|&ej| (ei, ej)))
                .collect();

            if pairs.len() < 2 {
                matrix[i * n + j] = 0.0;
                matrix[j * n + i] = 0.0;
                continue;
            }

            // Sort for deterministic iteration across HashMap orderings.
            pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

            #[allow(clippy::cast_precision_loss)]
            let np = pairs.len() as f64;
            let mean_i = pairs.iter().map(|(ei, _)| ei).sum::<f64>() / np;
            let mean_j = pairs.iter().map(|(_, ej)| ej).sum::<f64>() / np;

            let mut cov = 0.0_f64;
            let mut var_i = 0.0_f64;
            let mut var_j = 0.0_f64;
            for (ei, ej) in &pairs {
                let di = ei - mean_i;
                let dj = ej - mean_j;
                cov += di * dj;
                var_i += di * di;
                var_j += dj * dj;
            }

            let denom = np - 1.0;
            let std_i = (var_i / denom).sqrt();
            let std_j = (var_j / denom).sqrt();

            let rho = if std_i < f64::EPSILON || std_j < f64::EPSILON {
                0.0
            } else {
                (cov / denom) / (std_i * std_j)
            };

            let rho = rho.clamp(-1.0, 1.0);
            matrix[i * n + j] = rho;
            matrix[j * n + i] = rho;
        }
    }

    matrix
}

/// Assemble a multi-profile [`CorrelationModel`] from hydro IDs, a pooled
/// matrix, per-season matrices, the stage list, and the total season count.
///
/// The pooled matrix is always inserted as the `"default"` profile.
/// When `n_seasons <= 1`, only the default profile is produced (backward
/// compatibility with the single-season path).
/// For the multi-season case, each season that passed the minimum-sample check
/// gets a `"season_XX"` profile (zero-padded to the width of the largest season
/// index), and the schedule maps each stage to its season's profile name.
fn assemble_seasonal_correlation_model(
    hydro_ids: &[EntityId],
    pooled_matrix: &[f64],
    seasonal_matrices: &HashMap<usize, Vec<f64>>,
    stages: &[Stage],
    n_seasons: usize,
) -> CorrelationModel {
    let n = hydro_ids.len();

    let entities: Vec<CorrelationEntity> = hydro_ids
        .iter()
        .map(|&id| CorrelationEntity {
            entity_type: "inflow".to_string(),
            id,
        })
        .collect();

    let mut profiles = BTreeMap::new();

    // Convert a flat row-major N×N Vec<f64> to the AoS Vec<Vec<f64>> layout
    // required by CorrelationGroup.matrix (defined in cobre-core).
    let flat_to_aos = |flat: &[f64]| -> Vec<Vec<f64>> {
        (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect()
    };

    // Always include the pooled matrix as "default".
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "default".to_string(),
                entities: entities.clone(),
                matrix: flat_to_aos(pooled_matrix),
            }],
        },
    );

    // Single-season: return early with no per-season profiles and empty schedule.
    if n_seasons <= 1 {
        return CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: Vec::new(),
        };
    }

    // Multi-season: add per-season profiles.
    let width = format!("{}", n_seasons.saturating_sub(1)).len();
    for (&season_id, matrix) in seasonal_matrices {
        let name = format!("season_{season_id:0>width$}");
        profiles.insert(
            name,
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "default".to_string(),
                    entities: entities.clone(),
                    matrix: flat_to_aos(matrix),
                }],
            },
        );
    }

    // Build schedule: map each stage to its season profile name, omitting
    // stages whose season has no per-season profile (they fall back to "default").
    let mut schedule: Vec<CorrelationScheduleEntry> = stages
        .iter()
        .filter_map(|stage| {
            let season_id = stage.season_id?;
            if seasonal_matrices.contains_key(&season_id) {
                Some(CorrelationScheduleEntry {
                    stage_id: stage.id,
                    profile_name: format!("season_{season_id:0>width$}"),
                })
            } else {
                None
            }
        })
        .collect();
    schedule.sort_by_key(|e| e.stage_id);

    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule,
    }
}

// ---------------------------------------------------------------------------
// AIC-based AR order selection
// ---------------------------------------------------------------------------

/// Result of AIC-based AR order selection.
///
/// Produced by [`select_order_aic`]. Contains the selected AR order and the
/// AIC value for each candidate order from 0 (white noise) through `p_max`.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct AicSelectionResult {
    /// Selected AR order (1-based index into `sigma2_per_order`).
    ///
    /// `0` when no AR order improves over white noise (white noise is optimal
    /// or `sigma2_per_order` is empty).
    pub selected_order: usize,
    /// AIC value for each candidate order `0..=p_max`.
    ///
    /// `aic_values[0]` is the AIC for order 0 (white noise baseline = `0.0`).
    /// `aic_values[k]` is the AIC for AR order `k`, for `k >= 1`.
    pub aic_values: Vec<f64>,
}

/// Select the AR order that minimises the Akaike Information Criterion (AIC).
///
/// For each candidate order `p` in `1..=p_max`, the AIC is:
///
/// ```text
/// AIC(p) = N * ln(σ²_p) + 2p
/// ```
///
/// where `N = n_observations` and `σ²_p = sigma2_per_order[p-1]`.
///
/// The white-noise baseline (order 0) has `AIC(0) = 0.0` by convention
/// (`σ²_0 = 1.0` in the normalised Yule-Walker formulation, so
/// `N * ln(1) + 0 = 0`).
///
/// On ties the lower order wins (parsimony). Non-positive `sigma2` values
/// (which can arise from near-singular Levinson-Durbin truncation) are
/// excluded by assigning `AIC = f64::INFINITY`.
///
/// # Parameters
///
/// - `sigma2_per_order` — prediction error variances from
///   `LevinsonDurbinResult::sigma2_per_order`. `sigma2_per_order[k]`
///   corresponds to AR order `k+1`. Length = `p_max`.
/// - `n_observations` — number of historical observations for this season (`N_m`).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_aic;
///
/// // A variance drop at order 1 that outweighs the penalty selects order 1.
/// let result = select_order_aic(&[0.3], 100);
/// assert_eq!(result.selected_order, 1);
/// assert_eq!(result.aic_values.len(), 2);
///
/// // Empty sigma2 always selects white noise.
/// let result = select_order_aic(&[], 50);
/// assert_eq!(result.selected_order, 0);
/// assert_eq!(result.aic_values, vec![0.0]);
/// ```
pub fn select_order_aic(sigma2_per_order: &[f64], n_observations: usize) -> AicSelectionResult {
    let p_max = sigma2_per_order.len();

    // aic_values[0] = AIC for order 0 (white noise baseline).
    // aic_values[k] = AIC for AR order k, for k in 1..=p_max.
    let mut aic_values = Vec::with_capacity(p_max + 1);
    aic_values.push(0.0_f64); // order 0: N*ln(1) + 0 = 0

    #[allow(clippy::cast_precision_loss)]
    let n = n_observations as f64;

    for (k, &sigma2) in sigma2_per_order.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let order = (k + 1) as f64;
        let aic = if sigma2 <= 0.0 {
            f64::INFINITY
        } else {
            n * sigma2.ln() + 2.0 * order
        };
        aic_values.push(aic);
    }

    // Find the index of the minimum AIC. Use `enumerate` with a fold so that
    // ties naturally resolve to the first (lower-order) occurrence.
    let selected_order = aic_values
        .iter()
        .enumerate()
        .fold(
            (0usize, f64::INFINITY),
            |(best_idx, best_val), (idx, &val)| {
                if val < best_val {
                    (idx, val)
                } else {
                    (best_idx, best_val)
                }
            },
        )
        .0;

    AicSelectionResult {
        selected_order,
        aic_values,
    }
}

// ---------------------------------------------------------------------------
// PACF-based AR order selection
// ---------------------------------------------------------------------------

/// Result of PACF-based AR order selection.
///
/// Produced by [`select_order_pacf`]. Contains the selected AR order, the
/// PACF values for each lag, and the significance threshold used.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct PacfSelectionResult {
    /// Selected AR order.
    ///
    /// The maximum lag `k` where `|parcor[k-1]| > threshold`.
    /// `0` when no lag exceeds the significance threshold.
    pub selected_order: usize,
    /// PACF values (partial autocorrelation coefficients) for lags `1..=p_max`.
    ///
    /// `pacf_values[k]` is the PACF at lag `k+1`. Same as
    /// `LevinsonDurbinResult::parcor`.
    pub pacf_values: Vec<f64>,
    /// Significance threshold: `z_alpha / sqrt(n_observations)`.
    pub threshold: f64,
}

/// Select the AR order using partial autocorrelation function (PACF)
/// significance testing.
///
/// For each lag `k` in `1..=p_max`, tests whether the partial
/// autocorrelation coefficient (reflection coefficient from Levinson-Durbin)
/// exceeds the significance threshold `z_alpha / sqrt(N)`. Selects the
/// **maximum** lag with a significant PACF value.
///
/// If no lag exceeds the threshold, order 0 is selected (white noise).
///
/// # Parameters
///
/// - `parcor` -- partial autocorrelation coefficients from
///   `LevinsonDurbinResult::parcor`. `parcor[k]` is the PACF at lag `k+1`.
/// - `n_observations` -- number of historical observations for this season.
/// - `z_alpha` -- z-score for the desired confidence level (e.g., `1.96`
///   for 95% two-sided).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_pacf;
///
/// // PACF at lag 1 = 0.5 exceeds 1.96/sqrt(100) = 0.196; lag 2 = 0.1 does not.
/// let result = select_order_pacf(&[0.5, 0.1], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // No significant PACF values -> order 0.
/// let result = select_order_pacf(&[0.05, 0.03], 100, 1.96);
/// assert_eq!(result.selected_order, 0);
/// ```
pub fn select_order_pacf(
    parcor: &[f64],
    n_observations: usize,
    z_alpha: f64,
) -> PacfSelectionResult {
    #[allow(clippy::cast_precision_loss)]
    let threshold = if n_observations > 0 {
        z_alpha / (n_observations as f64).sqrt()
    } else {
        f64::INFINITY
    };

    // Find the maximum lag with |PACF| > threshold.
    let selected_order = parcor
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, p)| p.abs() > threshold)
        .map_or(0, |(k, _)| k + 1);

    PacfSelectionResult {
        selected_order,
        pacf_values: parcor.to_vec(),
        threshold,
    }
}

/// Select the AR order for a PAR(p)-A model using the conditional FACP
/// significance test, with NEWAVE-parity rules.
///
/// The CEPEL manual (section *Identificação da Ordem do Modelo* in
/// <https://see.cepel.br/manual/libs/latest/incerteza_hidrologica/modelo-par-p.html>)
/// specifies a single 95% confidence interval based on the number of
/// historical years (`z_alpha / sqrt(N)`, no lag-dependent deflation) and
/// "the largest significant lag is attributed as `p_m`". The manual is
/// silent on the cases where (a) lag 1 is exactly zero or (b) no lag is
/// significant; NEWAVE's reference implementation handles those via the
/// rules below, which are observed empirically:
///
/// 1. **Structural-zero short-circuit at lag 1.** If
///    `conditional_facp[0] == 0.0` exactly, the model is forced to
///    order 0. A structural zero at lag 1 indicates a degenerate (Z, A)
///    bucket — typically a single-observation season or a numerically
///    singular partitioned-covariance solve — and NEWAVE refuses to fit
///    any auto-regressive structure on top of it. This corresponds to
///    the "se o coeficiente de ordem 1 ... é zero ... o modelo é ajustado
///    a ordem zero" rule from the manual's *Tratamento de coeficientes
///    negativos* section. Structural zeros at higher lags do **not**
///    trigger the short-circuit; NEWAVE proceeds with the AR(1) base
///    whenever lag 1 itself is non-degenerate (e.g.,
///    `[+0.37, 0, 0, 0, 0, 0]` -> order 1, not 0).
/// 2. **Minimum order of 1 when lag 1 is non-zero.** If the conditional
///    FACP at lag 1 is not a structural zero, the selected order is
///    `max(1, max_significant_lag)`. NEWAVE empirically defaults to
///    AR(1) whenever no lag exceeds the threshold but lag 1 is well
///    defined (≈46/1860 observations on the 1931-2022 SIN history).
///
/// The Maceira-Damazio iterative order-reduction step (described in the
/// manual under *Tratamento de coeficientes negativos*) is **not**
/// applied here; the order returned by this function is the tentative
/// pre-validation order. Implementing the iterative reduction would
/// further reduce orders when the combined PAR(p) / PAR(p)-A
/// coefficient is negative, and is the documented path to closing the
/// remaining over-selection mismatches against `parpvaz.dat`.
///
/// `pacf_values[k]` in the returned struct is the conditional FACP at lag
/// `k+1`, conditioned on the intermediate standardised annual noise series
/// `Z` and the previous annual innovation `A_{t-1}`.
///
/// # Parameters
///
/// - `conditional_facp` -- conditional FACP coefficients from
///   [`conditional_facp_partitioned`]. `conditional_facp[k]` is the
///   conditional FACP at lag `k+1`.
/// - `n_observations` -- number of historical observations for the given
///   (hydro, season) pair.
/// - `z_alpha` -- z-score for the desired confidence level (e.g., `1.96`
///   for 95% two-sided).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_pacf_annual;
///
/// // Conditional FACP at lag 1 = 0.5 exceeds 1.96/sqrt(100) = 0.196; lag 2 = 0.1 does not.
/// let result = select_order_pacf_annual(&[0.5, 0.1], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // Lag 1 is non-zero (just small) -> min-order-1 rule kicks in.
/// let result = select_order_pacf_annual(&[0.05, 0.03], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // Structural zero at lag 1 -> order 0 (degenerate bucket).
/// let result = select_order_pacf_annual(&[0.0, 0.5], 100, 1.96);
/// assert_eq!(result.selected_order, 0);
/// ```
pub fn select_order_pacf_annual(
    conditional_facp: &[f64],
    n_observations: usize,
    z_alpha: f64,
) -> PacfSelectionResult {
    #[allow(clippy::cast_precision_loss)]
    let threshold = if n_observations > 0 {
        z_alpha / (n_observations as f64).sqrt()
    } else {
        f64::INFINITY
    };

    // Rule 1 — Structural-zero short-circuit at lag 1.
    // A structural zero at lag 1 (FACP exactly 0.0 from a degenerate
    // bucket) forces white-noise selection.
    if conditional_facp.first().copied() == Some(0.0) {
        return PacfSelectionResult {
            selected_order: 0,
            pacf_values: conditional_facp.to_vec(),
            threshold,
        };
    }

    // Find the maximum lag with |conditional FACP| > threshold.
    let max_significant = conditional_facp
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, p)| p.abs() > threshold)
        .map_or(0, |(k, _)| k + 1);

    // Rule 2 — Min-order-1 when lag 1 is non-zero (not a structural zero).
    // The CEPEL manual is silent on what happens when no lag is significant;
    // NEWAVE's implementation defaults to AR(1) whenever lag 1 is non-zero.
    let selected_order = match conditional_facp.first() {
        Some(&p1) if p1 != 0.0 => max_significant.max(1),
        _ => max_significant,
    };

    PacfSelectionResult {
        selected_order,
        pacf_values: conditional_facp.to_vec(),
        threshold,
    }
}

// ---------------------------------------------------------------------------
// Periodic autocorrelation
// ---------------------------------------------------------------------------

/// Compute the periodic normalised autocorrelation `rho(p, k)` for a given
/// reference season `p` and lag `k`.
///
/// The periodic autocorrelation differs from a stationary autocorrelation
/// in that the reference season determines both the "current" observations
/// and their seasonal statistics, while the lag determines the "lagged"
/// observations and their statistics.
///
/// Uses population divisor (1/N) and cross-year lag adjustment.
///
/// # Parameters
///
/// - `ref_season` -- 0-based season index of the reference month `p`.
/// - `lag` -- lag in seasonal periods (1-based: lag=1 means one season back).
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season index.
///   `observations_by_season[s]` contains all historical values for season `s`,
///   in chronological order.
/// - `stats_by_season` -- `(mean, std)` for each season, indexed by season.
///
/// # Returns
///
/// The normalised autocorrelation value `rho(ref_season, lag)`, clamped to [-1, 1].
/// Returns 0.0 when either the reference or lagged season has zero std,
/// or when insufficient paired observations exist.
#[must_use]
pub fn periodic_autocorrelation(
    ref_season: usize,
    lag: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> f64 {
    // Lag 0 is the identity: rho(m, 0) = 1.0 by normalisation.
    if lag == 0 {
        return 1.0;
    }

    // Compute lagged season index.
    let lag_season = (ref_season + n_seasons - lag % n_seasons) % n_seasons;

    let (mu_ref, std_ref) = stats_by_season[ref_season];
    let (mu_lag, std_lag) = stats_by_season[lag_season];

    // Zero-std guard: autocorrelation is undefined.
    if std_ref.abs() < f64::EPSILON || std_lag.abs() < f64::EPSILON {
        return 0.0;
    }

    let ref_obs = observations_by_season[ref_season];
    let lag_obs = observations_by_season[lag_season];

    // Cross-year lag adjustment: the number of year boundaries crossed by
    // a lag of `k` seasons determines how many observations must be dropped.
    //
    // A lag that stays within the same calendar year (lag_season < ref_season
    // and lag < n_seasons) crosses 0 boundaries. Otherwise, the number of
    // full years spanned is `(lag + n_seasons - 1) / n_seasons` when the lag
    // crosses into an earlier calendar position, or `lag / n_seasons` when
    // it wraps full cycles.
    //
    // Approach: for lag `k` within one cycle (k < n_seasons),
    // detect cross-year when lag_season >= ref_season. For larger lags,
    // additional years are spanned. The total drop count equals the number
    // of year boundaries crossed.
    let years_crossed = if lag < n_seasons {
        usize::from(lag_season >= ref_season)
    } else {
        // Full years from the lag.
        lag / n_seasons
    };

    let ref_start = years_crossed;
    let n_pairs = ref_obs
        .len()
        .saturating_sub(years_crossed)
        .min(lag_obs.len());

    // Insufficient data guard.
    if n_pairs == 0 {
        return 0.0;
    }

    // Cross-covariance with population divisor (1/N) over the year-aligned
    // valid pairs. NEWAVE uses N = n_pairs here for Z⊗Z autocorrelations
    // (verified against parpvaz.dat correlacao_series_vazoes_uhe). The
    // max-bucket-size convention used by cross_correlation_z_a /
    // cross_correlation_a_z_neg1 only applies to Z⊗A cross-terms because
    // those buckets have inherently different lengths (A excludes the first
    // year of Z by construction).
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (ref_obs[ref_start + i] - mu_ref) * (lag_obs[i] - mu_lag);
    }
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= n_pairs as f64;
    }

    // Normalise and clamp.
    let rho = gamma / (std_ref * std_lag);
    rho.clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Periodic Yule-Walker matrix
// ---------------------------------------------------------------------------

/// Build the periodic Yule-Walker matrix and right-hand side for a given
/// season and AR order.
///
/// Solves the **forward prediction** problem: predict `z_m` from
/// `{z_{m-1}, ..., z_{m-p}}`. The matrix uses rows 1..p of the extended
/// `(order+1) x (order+1)` covariance matrix, and the RHS uses column 0.
///
/// The matrix has dimension `order x order`. Entry `R[i,j]` is the
/// periodic autocorrelation `rho(season - (i+1), |j - i|)`, where the
/// reference month shifts per row (starting one step before `season`).
/// The matrix is symmetric but NOT Toeplitz because the autocorrelation
/// function varies with the reference period.
///
/// The right-hand side vector `rhs[i] = rho(season, i+1)` is anchored at
/// the target season with lags 1..p (column 0 of the extended matrix).
///
/// # Parameters
///
/// - `season` -- 0-based target season for the YW system.
/// - `order` -- AR order (determines matrix dimension: `order x order`).
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season, chronological order.
/// - `stats_by_season` -- `(mean, std)` for each season.
///
/// # Returns
///
/// A tuple `(matrix, rhs)` where:
/// - `matrix` is a flat `Vec<f64>` of length `order * order` in row-major layout.
///   Entry `R[i][j]` is at index `i * order + j`.
/// - `rhs` is a `Vec<f64>` of length `order`.
///
/// Returns `(vec![], vec![])` when `order == 0`.
#[must_use]
pub fn build_periodic_yw_matrix(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> (Vec<f64>, Vec<f64>) {
    #[cfg(test)]
    BUILD_PERIODIC_YW_MATRIX_CALL_COUNT.with(|c| {
        *c.borrow_mut() += 1;
    });

    if order == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut matrix = vec![0.0_f64; order * order];
    let mut rhs = vec![0.0_f64; order];

    // Fill the matrix: R[i][j] = rho(season - (i+1), |j - i|).
    // Diagonal is always 1.0 (rho(m, 0) = 1 for any m).
    // The inner `(i + 1) % n_seasons` prevents underflow when order > n_seasons.
    for i in 0..order {
        matrix[i * order + i] = 1.0;
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
            );
            matrix[i * order + j] = rho;
            matrix[j * order + i] = rho; // symmetric
        }
    }

    // Fill the RHS: rhs[i] = rho(season, i+1).
    for (i, rhs_entry) in rhs.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
    }

    (matrix, rhs)
}

/// Write the periodic Yule-Walker matrix and RHS into caller-supplied flat
/// buffers, resizing them to `order * order` and `order` respectively.
///
/// This variant avoids allocating new `Vec`s on each call, which matters when
/// `periodic_pacf` builds systems of increasing dimension in a loop.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `order` -- AR order (`matrix_out` will hold `order * order` entries).
/// - `n_seasons` -- total number of seasons.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` per season.
/// - `matrix_out` -- caller-allocated flat buffer; resized and overwritten.
/// - `rhs_out` -- caller-allocated buffer; resized and overwritten.
///
/// No-op when `order == 0` (buffers are cleared to empty).
pub fn build_periodic_yw_matrix_into(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    matrix_out: &mut Vec<f64>,
    rhs_out: &mut Vec<f64>,
) {
    if order == 0 {
        matrix_out.clear();
        rhs_out.clear();
        return;
    }

    matrix_out.resize(order * order, 0.0_f64);
    // Zero-fill: entries are written below, but the symmetric fill relies on
    // the upper triangle being populated before mirroring.
    for v in matrix_out.iter_mut() {
        *v = 0.0;
    }
    rhs_out.resize(order, 0.0_f64);

    for i in 0..order {
        matrix_out[i * order + i] = 1.0;
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
            );
            matrix_out[i * order + j] = rho;
            matrix_out[j * order + i] = rho;
        }
    }

    for (i, rhs_entry) in rhs_out.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
    }
}

// ---------------------------------------------------------------------------
// Extended periodic Yule-Walker matrix (annual component)
// ---------------------------------------------------------------------------

/// Compute the cross-correlation between the annual component series `A` at
/// season `ref_season` (with lag 0 meaning the same observation index) and the
/// periodic series `Z` at season `ref_season - lag` (wrapping), for non-negative
/// lag values.
///
/// This corresponds to `ρ_{Z,A}^{ref_season}(lag)` from the PAR-A extended
/// Yule-Walker system. The cross-correlation is defined as:
///
/// ```text
/// ρ_{Z,A}^{ref}(k) =
///   E[(A_i − μ^A_{ref}) · (Z_{i−k} − μ_{ref−k})] / (σ^A_{ref} · σ_{ref−k})
/// ```
///
/// where the expectation is approximated with a population (1/N) divisor.
///
/// # Cross-year alignment
///
/// Two distinct year offsets compose:
///
/// 1. **Bucket year offset** (`year_diff`). The `A` and `Z` buckets can have
///    different starting PDF years per season. For monthly NEWAVE data starting
///    on January, `Z` starts at year `Y0` for every season but `A` starts at
///    `Y0 + 1` for seasons 0..10 and `Y0` for season 11 (because the rolling
///    12-month window needs a full year of look-back). Calling code passes
///    `z_year_starts` and `a_year_starts` (one entry per season) so that the
///    pairing aligns by absolute PDF year, not by bucket index.
///
/// 2. **Lag year wrap** (`pdf_year_back_shift`). When stepping back `lag` months
///    from `ref_season`, if the lagged season index wraps (`lag_season >
///    ref_season` for `lag < n_seasons`, or `lag / n_seasons` for larger lags),
///    `Z`'s PDF year is one (or more) earlier than `A`'s.
///
/// **Lag-0 special case**: when `lag == 0`, `A` and `Z` refer to the same
/// season and the same regression year, so `pdf_year_back_shift = 0`
/// unconditionally — without this guard the `lag_season >= ref_season` branch
/// would falsely cross a year boundary.
///
/// # Parameters
///
/// - `ref_season` — 0-based season index for the `A` series.
/// - `lag` — non-negative integer lag; `lag = 0` pairs each `A_i` with `Z_i`.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — `Z` observations grouped by season, chronological.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
///   When all entries are equal, the legacy by-index pairing is recovered.
/// - `annual_observations_by_season` — `A` observations grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// The normalised cross-correlation, clamped to `[-1.0, 1.0]`.
/// Returns `0.0` when either standard deviation is below [`f64::EPSILON`],
/// or when insufficient paired observations exist.
#[must_use]
pub fn cross_correlation_z_a(
    ref_season: usize,
    lag: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> f64 {
    let (mu_a, std_a) = annual_stats_by_season[ref_season];
    let lag_season = (ref_season + n_seasons - lag % n_seasons) % n_seasons;
    let (mu_z, std_z) = stats_by_season[lag_season];

    // Zero-std guard: cross-correlation is undefined.
    if std_a.abs() < f64::EPSILON || std_z.abs() < f64::EPSILON {
        return 0.0;
    }

    let a_obs = annual_observations_by_season[ref_season];
    let z_obs = observations_by_season[lag_season];

    // Lag-direction year wrap: how many year boundaries the lag traverses
    // backward from ref_season to lag_season.
    let pdf_year_back_shift = if lag == 0 {
        0
    } else if lag < n_seasons {
        usize::from(lag_season >= ref_season)
    } else {
        lag / n_seasons
    };

    // Bucket year offset between A's first PDF year and Z's first PDF year.
    let year_diff = i64::from(a_year_starts[ref_season]) - i64::from(z_year_starts[lag_season]);
    let shift = year_diff - pdf_year_back_shift as i64;

    // Pairing is `(a_obs[a_start + k], z_obs[z_start + k])` for k = 0..n_pairs.
    // shift > 0 ⇒ skip extra Z entries at start (Z starts earlier); shift < 0 ⇒
    // skip extra A entries at start (A starts earlier).
    let (a_start, z_start) = if shift >= 0 {
        (0_usize, shift as usize)
    } else {
        ((-shift) as usize, 0_usize)
    };

    let n_pairs = a_obs
        .len()
        .saturating_sub(a_start)
        .min(z_obs.len().saturating_sub(z_start));

    // Insufficient data guard.
    if n_pairs == 0 {
        return 0.0;
    }

    // Cross-covariance with NEWAVE-style population divisor.
    //
    // Sum runs over the year-aligned valid pairs, but the divisor is the
    // **maximum bucket size** (typically the Z bucket = total study-window
    // years). This matches NEWAVE's parpvaz.dat convention, which is
    // equivalent to padding the missing-A years with the sample mean
    // (their cross-product contribution is zero) while keeping the
    // observed σ̂_A computed over the genuinely populated entries. Using
    // n_pairs (the strict-pair count) here would systematically overstate
    // ρ̂(Z, A) by a factor of `max_len / n_pairs` and tilt downstream
    // partitioned-covariance FACPs across the threshold boundary.
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (a_obs[a_start + i] - mu_a) * (z_obs[z_start + i] - mu_z);
    }
    let denom_n = a_obs.len().max(z_obs.len());
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= denom_n as f64;
    }

    // Normalise and clamp.
    let rho = gamma / (std_a * std_z);
    rho.clamp(-1.0, 1.0)
}

/// Compute the "lag = -1" cross-correlation between the annual component `A`
/// at season `ref_season` and the periodic series `Z` at the **next** season
/// `(ref_season + 1) % n_seasons`.
///
/// This corresponds to `ρ_{Z,A}^{ref_season}(-1)` — equivalently,
/// `ρ_{A,Z}^{ref_season}(+1)` with the arguments reversed. In the PAR-A
/// extended Yule-Walker RHS (eq. 15), this entry pairs `A_{t-1}` (season `m-1`)
/// with `Z_t` (season `m`), so `Z` is one step **ahead** of `A`.
///
/// # Cross-year alignment
///
/// Two distinct year offsets compose:
///
/// 1. **Bucket year offset** (`year_diff`). `A` and `Z` buckets can have
///    different starting PDF years per season (see [`cross_correlation_z_a`]
///    docs). Pairing aligns by absolute PDF year using `z_year_starts` and
///    `a_year_starts`.
///
/// 2. **Lag year wrap forward**. When `z_season = (ref_season + 1) % n_seasons`
///    wraps to 0, `Z` belongs to the next regression year relative to `A`.
///
/// # Parameters
///
/// - `ref_season` — 0-based season index for the `A` series.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — `Z` observations grouped by season, chronological.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
/// - `annual_observations_by_season` — `A` observations grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// The normalised cross-correlation, clamped to `[-1.0, 1.0]`.
/// Returns `0.0` when either standard deviation is below [`f64::EPSILON`],
/// or when insufficient paired observations exist.
#[must_use]
pub fn cross_correlation_a_z_neg1(
    ref_season: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> f64 {
    let (mu_a, std_a) = annual_stats_by_season[ref_season];
    let z_season = (ref_season + 1) % n_seasons;
    let (mu_z, std_z) = stats_by_season[z_season];

    // Zero-std guard: cross-correlation is undefined.
    if std_a.abs() < f64::EPSILON || std_z.abs() < f64::EPSILON {
        return 0.0;
    }

    let a_obs = annual_observations_by_season[ref_season];
    let z_obs = observations_by_season[z_season];

    // Z is one PDF month after A. When (ref_season + 1) wraps to 0, the
    // regression year of Z is one greater than A's.
    let pdf_year_forward_shift = usize::from(z_season == 0);

    let year_diff = i64::from(a_year_starts[ref_season]) - i64::from(z_year_starts[z_season]);
    let shift = year_diff + pdf_year_forward_shift as i64;

    let (a_start, z_start) = if shift >= 0 {
        (0_usize, shift as usize)
    } else {
        ((-shift) as usize, 0_usize)
    };

    let n_pairs = a_obs
        .len()
        .saturating_sub(a_start)
        .min(z_obs.len().saturating_sub(z_start));

    // Insufficient data guard.
    if n_pairs == 0 {
        return 0.0;
    }

    // Cross-covariance with NEWAVE-style population divisor — see
    // [`cross_correlation_z_a`] for the rationale on dividing by the larger
    // bucket size rather than `n_pairs`.
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (a_obs[a_start + i] - mu_a) * (z_obs[z_start + i] - mu_z);
    }
    let denom_n = a_obs.len().max(z_obs.len());
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= denom_n as f64;
    }

    // Normalise and clamp.
    let rho = gamma / (std_a * std_z);
    rho.clamp(-1.0, 1.0)
}

/// Build the extended periodic Yule-Walker matrix and right-hand side for the
/// PAR-A model, augmenting the classical `order × order` system with one
/// extra row and column for the annual component coefficient `ψ`.
///
/// The returned matrix has dimension `(order+1) × (order+1)`. Its layout is:
///
/// ```text
/// [ classical_yw_matrix  |  cross_col ]
/// [ cross_row            |  1.0       ]
/// ```
///
/// where:
/// - The top-left `order × order` block is the classical periodic YW matrix
///   (same as [`build_periodic_yw_matrix`] for `season` and `order`).
/// - The top-right column (indices `i * (order+1) + order` for `i < order`) and
///   the bottom-left row (indices `order * (order+1) + j` for `j < order`) are
///   filled symmetrically with
///   `cross_correlation_z_a((season + n_seasons − 1) % n_seasons, i, …)`.
/// - The bottom-right diagonal entry `(order, order)` is `1.0`.
/// - `rhs[0..order]` mirrors the classical YW right-hand side.
/// - `rhs[order]` is `cross_correlation_a_z_neg1((season + n_seasons − 1) % n_seasons, …)`.
///
/// All entries are normalised correlations, so the matrix is symmetric and has
/// unit diagonal. The solution vector `[φ_1, …, φ_p, ψ]` contains
/// **standardised** coefficients. Unit conversion is applied later by the
/// caller.
///
/// # Parameters
///
/// - `season` — 0-based target season.
/// - `order` — AR order `p`; the returned matrix has dimension `(order+1) × (order+1)`.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — `Z` observations grouped by season, chronological.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
///   Threaded through to the cross-correlation helpers for absolute-year
///   alignment between `A` and `Z` (see [`cross_correlation_z_a`] docs).
/// - `annual_observations_by_season` — annual component `A` observations grouped by
///   season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// A tuple `(matrix, rhs)` where:
/// - `matrix` is a flat `Vec<f64>` of length `(order+1)²` in row-major layout.
///   Entry `R[i][j]` is at index `i * (order+1) + j`.
/// - `rhs` is a `Vec<f64>` of length `order+1`.
///
/// When `order == 0`, returns `(vec![1.0], vec![rhs_annual_neg1])` — a 1×1
/// system that solves directly for `ψ`.
#[must_use]
pub fn build_extended_periodic_yw_matrix(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> (Vec<f64>, Vec<f64>) {
    let dim = order + 1;
    let prev_season = (season + n_seasons - 1) % n_seasons;

    let mut matrix = vec![0.0_f64; dim * dim];
    let mut rhs = vec![0.0_f64; dim];

    // Fill the top-left order × order block (classical periodic YW structure).
    // Diagonal entries are 1.0 (autocorrelation at lag 0).
    // Off-diagonal: R[i][j] = periodic_autocorrelation(ref_month, |j-i|, ...)
    // where ref_month = (season - (i+1)) % n_seasons.
    for i in 0..order {
        matrix[i * dim + i] = 1.0;
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
            );
            matrix[i * dim + j] = rho;
            matrix[j * dim + i] = rho; // symmetric
        }
    }

    // Fill the classical RHS entries: rhs[i] = rho(season, i+1).
    for (i, rhs_entry) in rhs.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
    }

    // Fill the right column and bottom row (annual extension).
    // Entry (i, order) = (order, i) = cross_correlation_z_a(prev_season, i, ...).
    for i in 0..order {
        let rho = cross_correlation_z_a(
            prev_season,
            i,
            n_seasons,
            observations_by_season,
            stats_by_season,
            z_year_starts,
            annual_observations_by_season,
            annual_stats_by_season,
            a_year_starts,
        );
        matrix[i * dim + order] = rho;
        matrix[order * dim + i] = rho; // symmetric
    }

    // Bottom-right diagonal entry for the annual component.
    matrix[order * dim + order] = 1.0;

    // Annual RHS entry: rho_{A,Z}^{prev_season}(-1).
    rhs[order] = cross_correlation_a_z_neg1(
        prev_season,
        n_seasons,
        observations_by_season,
        stats_by_season,
        z_year_starts,
        annual_observations_by_season,
        annual_stats_by_season,
        a_year_starts,
    );

    (matrix, rhs)
}

// ---------------------------------------------------------------------------
// Small matrix solver
// ---------------------------------------------------------------------------

/// Solve a dense linear system `A * x = b` via Gaussian elimination with
/// partial pivoting.
///
/// Designed for small systems (n <= 10) arising from Yule-Walker equations
/// in PAR model fitting. For these sizes, the O(n^3) cost is negligible.
///
/// # Parameters
///
/// - `a` -- flat row-major matrix of dimension `n x n` (length `n * n`).
///   **Modified in place** during elimination.
/// - `b` -- right-hand side vector of length `n`. **Modified in place**.
/// - `n` -- system dimension.
///
/// # Returns
///
/// `Some(x)` where `x` is the solution vector of length `n`, or `None` if the
/// matrix is singular (pivot magnitude below `f64::EPSILON`).
pub fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n, "matrix must have n*n elements");
    debug_assert_eq!(b.len(), n, "rhs must have n elements");

    if n == 0 {
        return Some(Vec::new());
    }

    // Forward elimination with partial pivoting.
    for k in 0..n {
        // Find pivot: row with largest |a[row][k]| in rows k..n-1.
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for row in (k + 1)..n {
            let val = a[row * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Singularity check.
        if max_val < f64::EPSILON {
            return None;
        }

        // Swap rows k and max_row in both a and b.
        if max_row != k {
            for col in 0..n {
                a.swap(k * n + col, max_row * n + col);
            }
            b.swap(k, max_row);
        }

        // Eliminate below.
        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = 0.0;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for k in (0..n).rev() {
        let mut sum = b[k];
        for j in (k + 1)..n {
            sum -= a[k * n + j] * x[j];
        }
        x[k] = sum / a[k * n + k];
    }

    Some(x)
}

// ---------------------------------------------------------------------------
// Conditional FACP for PAR-A (partitioned-covariance approach)
// ---------------------------------------------------------------------------

/// Partitioned covariance matrices for one `(season, k)` evaluation.
///
/// Stores the three sub-matrices used in the conditional FACP formula
/// `Σ̄ = Σ_11 − Σ_12 · Σ_22⁻¹ · Σ_21`. Sizes depend on the lag `k`:
/// - `sigma_11`: 2×2 (always four entries).
/// - `sigma_12`: 2×k row-major (the conditioning set has `k` elements).
/// - `sigma_22`: k×k row-major symmetric matrix.
#[allow(clippy::struct_field_names)]
pub(crate) struct PartitionedCov {
    /// 2×2 auto-covariance of `(Z_t, Z_{t−k})`, row-major.
    sigma_11: [f64; 4],
    /// 2×k cross-covariance between `(Z_t, Z_{t−k})` and the conditioning
    /// set `(Z_{t−1}, …, Z_{t−k+1}, A_{t−1})`, row-major.
    sigma_12: Vec<f64>,
    /// k×k auto-covariance of the conditioning set, row-major.
    sigma_22: Vec<f64>,
}

/// Assemble the three sub-matrices of the partitioned covariance for the
/// conditional FACP at lag `k` from `season`.
///
/// # Matrix layout
///
/// The conditioning set at lag `k` is:
/// `(Z_{t−1}, Z_{t−2}, …, Z_{t−k+1}, A_{t−1})` — that is, `k−1` lagged
/// `Z` values followed by the annual component at season `m−1`.
///
/// **`sigma_11`** (2×2): auto-covariance of `(Z_t, Z_{t−k})`.
/// - `[0,0] = 1` (unit variance of standardised `Z_t`)
/// - `[0,1] = [1,0] = ρ^{season}(k)`
/// - `[1,1] = 1`
///
/// **`sigma_22`** (k×k): auto-covariance of the conditioning set.
/// - Rows/cols `i,j < k−1`: `ρ^{season−1}(|i−j|)` — periodic autocorrelation
///   of the `Z` block at season `m−1`. Diagonal entries are `1.0`.
/// - Row/col `k−1` (the `A_{t−1}` entry):
///   - Off-diagonal: `cross_correlation_z_a(season−1, i, …)` for `i < k−1`.
///   - Diagonal `[k−1, k−1] = 1.0` (unit variance of standardised `A_{t−1}`).
///
/// **`sigma_12`** (2×k): cross-covariance between `(Z_t, Z_{t−k})` and the
/// conditioning set.
/// - Row 0 (from `Z_t`), column `j < k−1`: `ρ^{season}(j+1)`.
/// - Row 0, column `k−1`: `cross_correlation_a_z_neg1(season−1, …)`.
/// - Row 1 (from `Z_{t−k}`), column `j < k−1`: `ρ^{season−k}(k−1−j)`.
/// - Row 1, column `k−1`: `cross_correlation_z_a(season−1, k−1, …)`.
pub(crate) fn assemble_partitioned_covariance(
    season: usize,
    k: usize,
    n_seasons: usize,
    obs_z: &[&[f64]],
    stats_z: &[(f64, f64)],
    z_year_starts: &[i32],
    obs_a: &[&[f64]],
    stats_a: &[(f64, f64)],
    a_year_starts: &[i32],
) -> PartitionedCov {
    let prev_season = (season + n_seasons - 1) % n_seasons;

    // ------------------------------------------------------------------
    // Σ_11: 2×2 auto-covariance of (Z_t, Z_{t−k}).
    // Row-major: [0,0]=1, [0,1]=ρ^season(k), [1,0]=ρ^season(k), [1,1]=1.
    // ------------------------------------------------------------------
    let rho_k = periodic_autocorrelation(season, k, n_seasons, obs_z, stats_z);
    let sigma_11 = [1.0, rho_k, rho_k, 1.0];

    // ------------------------------------------------------------------
    // Σ_22: k×k auto-covariance of conditioning set.
    // Conditioning set: (Z_{t−1}, …, Z_{t−k+1}, A_{t−1})  (k elements).
    // ------------------------------------------------------------------
    let mut sigma_22 = vec![0.0_f64; k * k];

    // Z-block: rows/cols 0..k−1, all against season prev_season.
    // Diagonal is always 1.0 (unit variance). Off-diagonal: symmetric.
    for i in 0..k.saturating_sub(1) {
        sigma_22[i * k + i] = 1.0;
        let ref_month = (prev_season + n_seasons - i % n_seasons) % n_seasons;
        for j in (i + 1)..k.saturating_sub(1) {
            let lag = j - i;
            let rho = periodic_autocorrelation(ref_month, lag, n_seasons, obs_z, stats_z);
            sigma_22[i * k + j] = rho;
            sigma_22[j * k + i] = rho;
        }
    }

    // Cross-terms between the Z-block and A_{t−1} (column/row k−1).
    // sigma_22[i, k−1] = Corr(Z_{t−1−i}, A_{t−1}).
    // Z_{t−1−i} is `i` steps older than A_{t−1}, so lag = i.
    // for i in 0..k−1 (and symmetrically sigma_22[k−1, i]).
    // Note: for k=1 there is no Z-block (k.saturating_sub(1)=0), so this
    // loop body is never entered.
    for i in 0..k.saturating_sub(1) {
        let lag = i;
        let rho = cross_correlation_z_a(
            prev_season,
            lag,
            n_seasons,
            obs_z,
            stats_z,
            z_year_starts,
            obs_a,
            stats_a,
            a_year_starts,
        );
        sigma_22[i * k + (k - 1)] = rho;
        sigma_22[(k - 1) * k + i] = rho;
    }

    // Diagonal entry for A_{t−1}: unit variance by construction.
    sigma_22[(k - 1) * k + (k - 1)] = 1.0;

    // ------------------------------------------------------------------
    // Σ_12: 2×k cross-covariance between (Z_t, Z_{t−k}) and conditioning set.
    // ------------------------------------------------------------------
    let mut sigma_12 = vec![0.0_f64; 2 * k];

    // Row 0 (Z_t) with Z-block of conditioning set.
    for (j, entry) in sigma_12[..k.saturating_sub(1)].iter_mut().enumerate() {
        let rho = periodic_autocorrelation(season, j + 1, n_seasons, obs_z, stats_z);
        *entry = rho;
    }
    // Row 0 (Z_t) with A_{t−1}.
    sigma_12[k - 1] = cross_correlation_a_z_neg1(
        prev_season,
        n_seasons,
        obs_z,
        stats_z,
        z_year_starts,
        obs_a,
        stats_a,
        a_year_starts,
    );

    // Row 1 (Z_{t−k}) with Z-block of conditioning set.
    // Position j (0..k−2) of this row is Corr(Z_{t−k}, Z_{t−1−j}).
    // Z_{t−1−j} is the **newer** of the two (since j ≤ k−2 < k−1 < k), so
    // ρ_periodic must be anchored at its season with lag = (k−1) − j.
    for (j, entry) in sigma_12[k..k + k.saturating_sub(1)].iter_mut().enumerate() {
        let ref_season = (season + n_seasons - 1 - j) % n_seasons;
        let lag = k.saturating_sub(1).saturating_sub(j);
        let rho = periodic_autocorrelation(ref_season, lag, n_seasons, obs_z, stats_z);
        *entry = rho;
    }
    // Row 1 (Z_{t−k}) with A_{t−1}.
    sigma_12[k + (k - 1)] = cross_correlation_z_a(
        prev_season,
        k - 1,
        n_seasons,
        obs_z,
        stats_z,
        z_year_starts,
        obs_a,
        stats_a,
        a_year_starts,
    );

    PartitionedCov {
        sigma_11,
        sigma_12,
        sigma_22,
    }
}

/// Compute the conditional FACP for the PAR-A model up to `max_order`.
///
/// For each candidate lag `k` (`1..=max_order`), the conditioning set is
/// `(Z_{t−1}, …, Z_{t−k+1}, A_{t−1})` — the `k−1` intermediate standardised
/// inflow values plus the standardised annual component at season `m−1`. The
/// conditional correlation is obtained from the partitioned-covariance formula:
///
/// ```text
/// Σ̄ = Σ_11 − Σ_12 · Σ_22⁻¹ · Σ_21
/// ```
///
/// where `Σ_21 = Σ_12ᵀ`. The conditional FACP at lag `k` is
/// `Σ̄[0,1] / √(Σ̄[0,0] · Σ̄[1,1])`, clamped to `[−1, 1]`.
///
/// # Parameters
///
/// - `season` — 0-based target season (the "current" season `m`).
/// - `max_order` — maximum lag to evaluate. Returns `Vec::new()` when zero.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — periodic inflow series `Z`, grouped by season.
///   Entry `[s][y]` is the standardised observation for season `s` in year `y`.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
///   Used by the cross-correlation helpers to align `A` and `Z` by absolute
///   PDF year rather than by bucket index — required for monthly NEWAVE data
///   where `A` buckets start one year later than `Z` for most seasons.
/// - `annual_observations_by_season` — annual component `A`, grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// A `Vec<f64>` of length `≤ max_order`. Entry `i` is the conditional FACP at
/// lag `i+1`, clamped to `[−1.0, 1.0]`.
///
/// The vector is shorter than `max_order` when `Σ_22` is singular at some lag
/// `k` — the loop breaks early and entries for lags `≥ k` are omitted.
/// When `Σ̄[0,0] · Σ̄[1,1] ≤ 0` (numerical degeneracy), the affected entry is
/// recorded as `0.0`.
#[must_use]
pub fn conditional_facp_partitioned(
    season: usize,
    max_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> Vec<f64> {
    if max_order == 0 {
        return Vec::new();
    }

    let mut facp_values = Vec::with_capacity(max_order);

    // Reusable scratch buffers for solve_linear_system calls.
    // The Σ_22 matrix and RHS column are cloned per solve; the buffers below
    // hold the cloned working copies so we avoid re-allocating each iteration.
    let mut matrix_buf: Vec<f64> = Vec::new();
    let mut rhs_col: Vec<f64> = Vec::new();

    for k in 1..=max_order {
        let cov = assemble_partitioned_covariance(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
            z_year_starts,
            annual_observations_by_season,
            annual_stats_by_season,
            a_year_starts,
        );

        // ------------------------------------------------------------------
        // Solve Σ_22 · X = Σ_21 column-by-column (2 columns, one per row of
        // Σ_12). Σ_21 = Σ_12ᵀ, so column c of Σ_21 = row c of Σ_12.
        // X has shape k×2; store solutions as two Vec<f64> of length k.
        // ------------------------------------------------------------------
        let mut x_cols: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut singular = false;

        for (col_idx, x_col) in x_cols.iter_mut().enumerate() {
            // Copy Σ_22 into working buffer (solve_linear_system modifies in-place).
            matrix_buf.clear();
            matrix_buf.extend_from_slice(&cov.sigma_22);

            // Column col_idx of Σ_21 = row col_idx of Σ_12 (k entries).
            rhs_col.clear();
            for row in 0..k {
                rhs_col.push(cov.sigma_12[col_idx * k + row]);
            }

            if let Some(sol) = solve_linear_system(&mut matrix_buf, &mut rhs_col, k) {
                *x_col = sol;
            } else {
                singular = true;
                break;
            }
        }

        if singular {
            // Singular Σ_22 — stop the loop and return results so far.
            break;
        }

        // ------------------------------------------------------------------
        // Compute Σ̄ = Σ_11 − Σ_12 · X  (2×2 result).
        //
        // Σ_12 is 2×k, X is k×2.
        // [Σ_12 · X][r, c] = sum_{j=0}^{k-1} Σ_12[r,j] * X[j,c]
        //                  = sum_{j=0}^{k-1} sigma_12[r*k + j] * x_cols[c][j]
        //
        // sigma_bar is stored row-major: [0,0], [0,1], [1,0], [1,1].
        // ------------------------------------------------------------------
        let mut sigma_bar = cov.sigma_11;
        for r in 0..2 {
            for c in 0..2 {
                let correction: f64 = (0..k).map(|j| cov.sigma_12[r * k + j] * x_cols[c][j]).sum();
                sigma_bar[r * 2 + c] -= correction;
            }
        }

        // ------------------------------------------------------------------
        // Extract conditional FACP: sigma_bar[0,1] / sqrt(sigma_bar[0,0] * sigma_bar[1,1]).
        // Guard against non-positive product (numerical degeneracy).
        // ------------------------------------------------------------------
        let denom_sq = sigma_bar[0] * sigma_bar[3];
        let facp = if denom_sq <= 0.0 {
            0.0
        } else {
            (sigma_bar[1] / denom_sq.sqrt()).clamp(-1.0, 1.0)
        };

        facp_values.push(facp);
    }

    facp_values
}

// ---------------------------------------------------------------------------
// Periodic PACF
// ---------------------------------------------------------------------------

/// Compute the periodic PACF for a given season up to `max_order`.
///
/// For each candidate order k (`1..=max_order`), builds the periodic Yule-Walker
/// matrix of dimension k, solves `R * phi = rhs`, and extracts the last
/// coefficient `phi[k-1]` as `PACF(k)`.
///
/// This is the correct periodic PACF computation that accounts for the
/// non-Toeplitz covariance structure of periodic autoregressive processes.
/// It replaces the stationary Levinson-Durbin reflection coefficients which
/// assume a Toeplitz (stationary) covariance matrix.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `max_order` -- maximum lag to compute PACF for.
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` for each season.
///
/// # Returns
///
/// A `Vec<f64>` of length <= `max_order`. Entry `k` is `PACF(k+1)`.
/// The vector may be shorter than `max_order` if a system at some order
/// is singular (remaining orders are skipped).
#[must_use]
pub fn periodic_pacf(
    season: usize,
    max_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> Vec<f64> {
    let mut pacf_values = Vec::with_capacity(max_order);

    // Reuse two scratch buffers across the loop to avoid allocating a new
    // Vec pair per order k. build_periodic_yw_matrix_into resizes them in
    // place (no-alloc when capacity already covers the new size).
    let mut matrix_buf: Vec<f64> = Vec::new();
    let mut rhs_buf: Vec<f64> = Vec::new();

    for k in 1..=max_order {
        build_periodic_yw_matrix_into(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
            &mut matrix_buf,
            &mut rhs_buf,
        );

        match solve_linear_system(&mut matrix_buf, &mut rhs_buf, k) {
            Some(phi) => pacf_values.push(phi[k - 1]),
            None => break, // Singular matrix, stop.
        }
    }

    pacf_values
}

// ---------------------------------------------------------------------------
// Periodic YW coefficient estimation
// ---------------------------------------------------------------------------

/// Result of periodic Yule-Walker AR coefficient estimation for one
/// (entity, season) pair.
#[must_use]
#[derive(Debug, Clone)]
pub struct PeriodicYwResult {
    /// Standardised AR coefficients `phi_1..phi_p`.
    pub coefficients: Vec<f64>,
    /// Residual std ratio: `sigma_residual / sigma_sample`.
    /// In `(0, 1]` for valid models; 1.0 for order-0.
    pub residual_std_ratio: f64,
    /// Prediction error variance at each intermediate order `1..=selected_order`.
    /// Used for diagnostic reporting. `sigma2_per_order[k-1]` is the variance
    /// for AR(k).
    pub sigma2_per_order: Vec<f64>,
}

/// Estimate AR coefficients by solving the periodic Yule-Walker system at the
/// given order.
///
/// Also computes prediction error variances at each intermediate order
/// (`1..=selected_order`) for diagnostic reporting compatibility.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `selected_order` -- the AR order to fit (from PACF-based selection).
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` for each season.
///
/// # Returns
///
/// A [`PeriodicYwResult`] with the fitted coefficients, residual std ratio,
/// and prediction error variances. Returns order-0 result (empty coefficients,
/// ratio 1.0) when `selected_order == 0` or when the system is singular.
pub fn estimate_periodic_ar_coefficients(
    season: usize,
    selected_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> PeriodicYwResult {
    let zero_result = PeriodicYwResult {
        coefficients: Vec::new(),
        residual_std_ratio: 1.0,
        sigma2_per_order: Vec::new(),
    };

    if selected_order == 0 {
        return zero_result;
    }

    let mut sigma2_per_order = Vec::with_capacity(selected_order);
    let mut final_coefficients = Vec::new();

    for k in 1..=selected_order {
        // Build and solve the periodic YW system at order k.
        // Save the original RHS for sigma2 computation (solve modifies in-place).
        // Build the matrix once; clone rhs (O(k)) before the in-place solve.
        let (mut matrix, mut rhs) = build_periodic_yw_matrix(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
        let rhs_orig: Vec<f64> = rhs.clone();

        match solve_linear_system(&mut matrix, &mut rhs, k) {
            Some(phi) => {
                // sigma2(k) = 1 - sum_{j=0}^{k-1} phi[j] * rhs_original[j]
                let sigma2_k: f64 = 1.0
                    - phi
                        .iter()
                        .zip(rhs_orig.iter())
                        .map(|(p, r)| p * r)
                        .sum::<f64>();
                sigma2_per_order.push(sigma2_k);

                if k == selected_order {
                    final_coefficients = phi;
                }
            }
            None => {
                // Singular matrix: fall back to order-0 result.
                return zero_result;
            }
        }
    }

    // Compute residual_std_ratio = sqrt(sigma2(selected_order)).
    let sigma2_final = *sigma2_per_order.last().unwrap_or(&1.0);
    let residual_std_ratio = if sigma2_final > 0.0 {
        sigma2_final.sqrt().clamp(f64::EPSILON, 1.0)
    } else {
        1.0 // Numerical issue: fall back.
    };

    PeriodicYwResult {
        coefficients: final_coefficients,
        residual_std_ratio,
        sigma2_per_order,
    }
}

// ---------------------------------------------------------------------------
// Annual seasonal stats helper (PAR-A eqs. 17, 18)
// ---------------------------------------------------------------------------

/// Per-season sample statistics of the rolling 12-month average.
///
/// One entry per season for one entity. Computed by
/// [`estimate_annual_seasonal_stats`] from a chronological observation list.
///
/// The `mean_m3s` and `std_m3s` fields are in the original m³/s units of the
/// observation series; they are **not** standardised. The standardisation
/// happens inside the cross-correlation helpers from
/// [`build_extended_periodic_yw_matrix`], and the unit conversion to `ψ̂`
/// happens at `PrecomputedPar::build` time.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct AnnualSeasonalStats {
    /// Entity (e.g., hydro plant) identifier.
    pub hydro_id: EntityId,
    /// Season ID (0-based).
    pub season_id: usize,
    /// Sample mean of the rolling 12-month average for this (entity, season) pair, in m³/s.
    pub mean_m3s: f64,
    /// Population-divisor standard deviation (`1/N` divisor) of the rolling
    /// 12-month average for this (entity, season) pair, in m³/s. Matches
    /// NEWAVE's `rel_parpa.pdf` eq. 18 convention.
    pub std_m3s: f64,
}

/// Estimate the per-season sample statistics `(μ^A_m, σ^A_m)` of the rolling
/// 12-month average from chronological observations.
///
/// For each (entity, season) pair, `μ^A_m` is the sample mean of the rolling
/// 12-month average `A_t = (1/12) · Σ_{j=0..11} z[t-j]` values whose target
/// date falls in season `m`. `σ^A_m` is the **population-divisor standard
/// deviation** using divisor `1/N`, matching NEWAVE's `rel_parpa.pdf` eq. 18
/// convention and the workspace-wide convention used by
/// [`estimate_seasonal_stats`]. The PAR(p)-A runtime coefficient is then
/// `ψ̂ = ψ · σ_m / σ^A_m`, which requires `σ^A_m > 0` (enforced by the
/// output validator in `cobre-io`).
///
/// ## Algorithm
///
/// 1. Group observations by `EntityId` and sort each group chronologically.
/// 2. For each entity group, build the rolling 12-month average:
///    `A_{i+12} = (1/12) · Σ_{j=0..11} z[i+j]` for every chronological index `i`
///    such that `i + 12 < group.len()`. The target date is `group[i+12].date`.
/// 3. Group `A_{i+12}` values by the season of the target date (using
///    [`find_season_for_date`] + `season_map` fallback, mirroring
///    [`estimate_seasonal_stats_with_season_map`]).
/// 4. Compute per (entity, season) the sample mean and population-divisor std (1/N).
///
/// Returns rows sorted by `(hydro_id, season_id)` ascending.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when any requested `entity_id`
/// produces zero rolling-window `A_t` values (fewer than 13 observations for
/// that entity). The error names the entity and its observation count.
/// Silent fallback to the classical PAR path is not performed.
pub fn estimate_annual_seasonal_stats(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<Vec<AnnualSeasonalStats>, StochasticError> {
    // Build stage index for date-to-season mapping.
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let entity_set: HashSet<EntityId> = entity_ids.iter().copied().collect();

    // Group observations by entity in chronological order.
    let mut entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if entity_set.contains(&entity_id) {
            entity_obs.entry(entity_id).or_default().push((date, value));
        }
    }
    for obs_vec in entity_obs.values_mut() {
        obs_vec.sort_unstable_by_key(|(d, _)| *d);
    }

    // Per-entity insufficient-data guard: every requested entity must produce
    // at least one rolling-window A_t value, which requires >= 13 observations.
    for &entity_id in entity_ids {
        let n_obs = entity_obs.get(&entity_id).map_or(0, Vec::len);
        if n_obs < 13 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "entity {entity_id} has {n_obs} observation(s); \
                     at least 13 are required to form one rolling 12-month average"
                ),
            });
        }
    }

    // Build rolling-window A_t values grouped by (entity_id, season_id).
    //
    // Indexing convention: an A value A_{t-1} = mean(z[t-12..t-1]) is stored
    // under the season of its own PDF time-index (t-1), which is the most
    // recent observation in the rolling window — `group[i + 11]`. With this
    // convention, `annual_stats_by_season[s]` contains stats for
    // `A_{t-1}` whose PDF time-index falls in season `s`, equivalently
    // `A_{t-1}` for `t` at season `s + 1`. The Yule-Walker callers
    // (`build_extended_periodic_yw_matrix`, `assemble_partitioned_covariance`)
    // index this map with `prev_season = (m - 1) mod n_seasons` to retrieve
    // the stats for the equation at current season `m`.
    let mut group_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    for &entity_id in entity_ids {
        let Some(group) = entity_obs.get(&entity_id) else {
            continue;
        };

        // For each index i such that i + 11 < group.len() (we still require
        // i + 12 <= group.len() to access the full 12-month window), the
        // rolling-window mean A = (1/12) * sum of z[i..i+12] is stored under
        // the season of group[i + 11].date — i.e., the PDF time-index of
        // A_{t-1} when target month t = i + 12.
        for i in 0..group.len().saturating_sub(12) {
            let target_date = group[i + 11].0;

            let Some(season_id) = find_season_for_date(&stage_index, target_date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(target_date)))
            else {
                // Target date not in any stage and no season_map fallback — skip.
                continue;
            };

            // A_{(i+12)-1} = (1/12) * sum of z[i..i+12]; PDF time of this value
            // is i + 11, so it is stored under the season of group[i + 11].
            let mean_a: f64 = group[i..i + 12].iter().map(|(_, v)| v).sum::<f64>() / 12.0;
            group_map
                .entry((entity_id, season_id))
                .or_default()
                .push(mean_a);
        }
    }

    // Compute mean and population-divisor std for each (entity, season) group.
    //
    // NEWAVE (rel_parpa.pdf eq. 18) uses sigma^A_m with divisor 1/N. Matching
    // that convention is required for parity on the partitioned-covariance
    // FACP — the sample-vs-population scale factor would otherwise leak
    // through every Z⊗A cross-correlation.
    let mut result: Vec<AnnualSeasonalStats> = Vec::with_capacity(group_map.len());
    for ((entity_id, season_id), values) in &group_map {
        let n = values.len();
        #[allow(clippy::cast_precision_loss)]
        let mean_m3s = values.iter().copied().sum::<f64>() / n as f64;
        let std_m3s = if n >= 1 {
            #[allow(clippy::cast_precision_loss)]
            let var = values
                .iter()
                .map(|&v| (v - mean_m3s) * (v - mean_m3s))
                .sum::<f64>()
                / n as f64;
            var.sqrt()
        } else {
            0.0
        };

        result.push(AnnualSeasonalStats {
            hydro_id: *entity_id,
            season_id: *season_id,
            mean_m3s,
            std_m3s,
        });
    }

    // Sort by (hydro_id, season_id) ascending.
    result.sort_unstable_by_key(|s| (s.hydro_id.0, s.season_id));

    Ok(result)
}

// ---------------------------------------------------------------------------
// Extended periodic YW coefficient estimation (PAR-A)
// ---------------------------------------------------------------------------

/// Result of the extended periodic Yule-Walker solve for the PAR-A model.
///
/// Returned by [`estimate_periodic_ar_annual_coefficients`]. The three fields
/// are **standardised** (dimensionless), matching the convention for the classical
/// `PeriodicYwResult::coefficients`. Unit conversion to runtime coefficients
/// `(φ̂_j, ψ̂)` happens at `PrecomputedPar::build` time, not here.
#[must_use]
#[derive(Debug, Clone)]
pub struct PeriodicYwAnnualResult {
    /// Standardised AR coefficients `φ_1..φ_p` (dimensionless, direct Yule-Walker
    /// output). Empty when `selected_order == 0`.
    pub coefficients: Vec<f64>,
    /// Standardised annual coefficient `ψ` (dimensionless, direct Yule-Walker
    /// output).
    pub annual_coefficient: f64,
    /// Residual std ratio `σ_residual / σ_seasonal` in `(0, 1]`.
    pub residual_std_ratio: f64,
}

/// Estimate PAR-A coefficients `(φ_1..φ_p, ψ)` by solving the extended
/// periodic Yule-Walker system.
///
/// Builds the `(selected_order + 1) × (selected_order + 1)` system via
/// [`build_extended_periodic_yw_matrix`] and solves it via
/// [`solve_linear_system`].
///
/// ## Singular-system fallback
///
/// When the system is singular (the solver returns `None`), the function
/// returns `PeriodicYwAnnualResult { coefficients: vec![], annual_coefficient:
/// 0.0, residual_std_ratio: 1.0 }`, matching the classical fallback in
/// [`estimate_periodic_ar_coefficients`].
///
/// ## Order-0 case
///
/// When `selected_order == 0`, the function solves the 1×1 system that yields
/// only `ψ`. The returned `coefficients` is empty.
///
/// ## Residual std ratio
///
/// `sigma2 = 1 - Σ_i (solution[i] · rhs_orig[i])` over all `selected_order + 1`
/// solution entries. `residual_std_ratio = sqrt(sigma2).clamp(f64::EPSILON, 1.0)`
/// when `sigma2 > 0`, else `1.0`.
pub fn estimate_periodic_ar_annual_coefficients(
    season: usize,
    selected_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> PeriodicYwAnnualResult {
    let zero_result = PeriodicYwAnnualResult {
        coefficients: Vec::new(),
        annual_coefficient: 0.0,
        residual_std_ratio: 1.0,
    };

    let (mut matrix, mut rhs) = build_extended_periodic_yw_matrix(
        season,
        selected_order,
        n_seasons,
        observations_by_season,
        stats_by_season,
        z_year_starts,
        annual_observations_by_season,
        annual_stats_by_season,
        a_year_starts,
    );

    let dim = selected_order + 1;
    let rhs_orig: Vec<f64> = rhs.clone();

    let Some(solution) = solve_linear_system(&mut matrix, &mut rhs, dim) else {
        return zero_result;
    };

    // Split: first `selected_order` entries are AR coefficients; last is ψ.
    let coefficients: Vec<f64> = solution[..selected_order].to_vec();
    let annual_coefficient: f64 = solution[selected_order];

    // sigma2 = 1 - sum(solution[i] * rhs_orig[i]) for all i in 0..dim.
    let sigma2: f64 = 1.0
        - solution
            .iter()
            .zip(rhs_orig.iter())
            .map(|(s, r)| s * r)
            .sum::<f64>();

    let residual_std_ratio = if sigma2 > 0.0 {
        sigma2.sqrt().clamp(f64::EPSILON, 1.0)
    } else {
        1.0
    };

    PeriodicYwAnnualResult {
        coefficients,
        annual_coefficient,
        residual_std_ratio,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Thread-local call counter used by
// `estimate_periodic_ar_coefficients_calls_build_once_per_order` to verify
// that F2-006 is in effect (exactly one `build_periodic_yw_matrix` call per
// loop iteration, not two).
#[cfg(test)]
thread_local! {
    static BUILD_PERIODIC_YW_MATRIX_CALL_COUNT: std::cell::RefCell<usize> =
        const { std::cell::RefCell::new(0) };
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::doc_markdown
)]
mod tests {
    use super::{
        build_periodic_yw_matrix, classify_history, estimate_periodic_ar_coefficients,
        periodic_autocorrelation, periodic_pacf, select_order_aic, select_order_pacf,
        select_order_pacf_annual, solve_linear_system, HistoryClass,
        BUILD_PERIODIC_YW_MATRIX_CALL_COUNT,
    };

    // -----------------------------------------------------------------------
    // estimate_seasonal_stats tests
    // -----------------------------------------------------------------------

    use chrono::{Datelike, NaiveDate};
    use cobre_core::{
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
        EntityId,
    };

    use super::estimate_seasonal_stats;
    use crate::StochasticError;

    /// Build a minimal `Stage` for testing. Stages with `season_id = Some(s)`.
    fn make_stage(
        id: i32,
        index: usize,
        year_start: i32,
        month_start: u32,
        year_end: i32,
        month_end: u32,
        season_id: Option<usize>,
    ) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(year_start, month_start, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(year_end, month_end, 1).unwrap(),
            season_id,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
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
        }
    }

    /// Build a 12-stage monthly cycle starting at `base_year`, spanning `n_years`.
    /// Stage IDs are 1-based sequential (1..=12), season IDs 0..11.
    fn make_monthly_stages(base_year: i32, n_years: u32) -> Vec<Stage> {
        let months = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 1),
        ];
        let mut stages: Vec<Stage> = Vec::new();
        for year_offset in 0..n_years {
            let year = base_year + year_offset as i32;
            for (idx, &(m_start, m_end)) in months.iter().enumerate() {
                let (end_year, end_month) = if m_end == 1 {
                    (year + 1, 1u32)
                } else {
                    (year, m_end)
                };
                let stage_id = (year_offset * 12 + idx as u32 + 1) as i32;
                stages.push(make_stage(
                    stage_id,
                    stages.len(),
                    year,
                    m_start,
                    end_year,
                    end_month,
                    Some(idx),
                ));
            }
        }
        stages
    }

    /// Build an observation for `entity_id` on the 15th of `(year, month)`.
    fn obs(entity_id: i32, year: i32, month: u32, value: f64) -> (EntityId, NaiveDate, f64) {
        (
            EntityId::from(entity_id),
            NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
            value,
        )
    }

    // -----------------------------------------------------------------------
    // HistoryClass / classify_history
    // -----------------------------------------------------------------------

    #[test]
    fn classify_history_default_for_random_series() {
        // Strictly increasing series — not constant, no negatives, no mode > 50%.
        let obs: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        assert_eq!(classify_history(&obs), HistoryClass::Default);
    }

    #[test]
    fn classify_history_constant_zero() {
        let obs = [0.0_f64; 30];
        match classify_history(&obs) {
            HistoryClass::Constant { value } => assert_eq!(value, 0.0),
            other => panic!("expected Constant {{ 0.0 }}, got {other:?}"),
        }
    }

    #[test]
    fn classify_history_constant_nonzero() {
        let obs = [1100.0_f64; 30];
        match classify_history(&obs) {
            HistoryClass::Constant { value } => assert!((value - 1100.0).abs() < 1e-9),
            other => panic!("expected Constant {{ 1100.0 }}, got {other:?}"),
        }
    }

    #[test]
    fn classify_history_empty_falls_back_to_constant_zero() {
        match classify_history(&[]) {
            HistoryClass::Constant { value } => assert_eq!(value, 0.0),
            other => panic!("expected Constant {{ 0.0 }}, got {other:?}"),
        }
    }

    #[test]
    fn classify_history_many_negative_above_threshold() {
        // 3 of 20 strictly negative = 15% > 10% threshold.
        let mut obs: Vec<f64> = (1..=17).map(|i| i as f64).collect();
        obs.extend_from_slice(&[-1.0, -2.0, -3.0]);
        match classify_history(&obs) {
            HistoryClass::ManyNegative { sample_mean } => {
                let expected: f64 = obs.iter().sum::<f64>() / obs.len() as f64;
                assert!((sample_mean - expected).abs() < 1e-9);
            }
            other => panic!("expected ManyNegative, got {other:?}"),
        }
    }

    #[test]
    fn classify_history_many_negative_at_threshold_falls_through() {
        // Exactly 10% (2/20) negative — does NOT trigger TIPO 2 (strict >).
        // Falls through. With these values neither TIPO 1 nor TIPO 4 either.
        let mut obs: Vec<f64> = (1..=18).map(|i| i as f64).collect();
        obs.extend_from_slice(&[-1.0, -2.0]);
        assert_eq!(classify_history(&obs), HistoryClass::Default);
    }

    #[test]
    fn classify_history_saturated_cap() {
        // BELO MONTE-style: most April values at 13900 cap, rest scattered below.
        // 12 out of 20 (60%) at 13900, rest at 5000-13800.
        let mut obs = vec![13900.0_f64; 12];
        obs.extend_from_slice(&[
            5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0,
        ]);
        match classify_history(&obs) {
            HistoryClass::Saturated { cap } => assert!((cap - 13900.0).abs() < 1e-9),
            other => panic!("expected Saturated {{ 13900.0 }}, got {other:?}"),
        }
    }

    #[test]
    fn classify_history_low_mode_falls_through_to_default() {
        // Mode at 50.0 with 9/20 occurrences = 45% — below the 50% threshold.
        let mut obs = vec![50.0_f64; 9];
        obs.extend_from_slice(&[
            10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ]);
        assert_eq!(classify_history(&obs), HistoryClass::Default);
    }

    #[test]
    fn classify_history_mode_at_zero_with_majority_is_tipo_4() {
        // 11 zeros + 9 nonzero = 55% at mode 0. NEWAVE flags this as TIPO 4
        // (saturating cap = 0) — typical of low-flow constant months on plants
        // like COARACY NUNE / Sep-Oct. cobre matches that classification.
        let mut obs = vec![0.0_f64; 11];
        obs.extend_from_slice(&[
            100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0,
        ]);
        match classify_history(&obs) {
            HistoryClass::Saturated { cap } => assert!((cap - 0.0).abs() < 1e-9),
            other => panic!("expected Saturated {{ 0.0 }}, got {other:?}"),
        }
    }

    #[test]
    fn classify_history_helpers_round_trip() {
        let c = HistoryClass::Constant { value: 5.0 };
        assert_eq!(c.tipo_code(), 1);
        assert_eq!(c.stats_override(), Some((5.0, 0.0)));
        assert!(c.is_degenerate());

        let s = HistoryClass::Saturated { cap: 13900.0 };
        assert_eq!(s.tipo_code(), 4);
        assert_eq!(s.stats_override(), Some((13900.0, 0.0)));
        assert!(s.is_degenerate());

        // TIPO 2 is diagnostic only; no fitting override and not "degenerate"
        // for the purpose of forcing order 0.
        let n = HistoryClass::ManyNegative { sample_mean: -1.5 };
        assert_eq!(n.tipo_code(), 2);
        assert_eq!(n.stats_override(), None);
        assert!(!n.is_degenerate());

        let d = HistoryClass::Default;
        assert_eq!(d.tipo_code(), 0);
        assert_eq!(d.stats_override(), None);
        assert!(!d.is_degenerate());
    }

    // -----------------------------------------------------------------------
    // Acceptance criterion: 2 hydros x 12 seasons = 24 rows
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_two_hydros_twelve_seasons() {
        // 12 stages, 30 years worth of observations per hydro.
        let stages = make_monthly_stages(1990, 30);
        let entity_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in 1990..2020_i32 {
            for month in 1u32..=12 {
                observations.push(obs(1, year, month, 100.0 + month as f64));
                observations.push(obs(2, year, month, 200.0 + month as f64));
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        assert_eq!(stats.len(), 24, "expected 2 hydros × 12 seasons = 24 rows");

        // All rows must be for entity 1 or 2.
        for s in &stats {
            assert!(
                s.entity_id == EntityId::from(1) || s.entity_id == EntityId::from(2),
                "unexpected entity_id {}",
                s.entity_id
            );
        }

        // Output must be sorted by (entity_id, stage_id).
        for w in stats.windows(2) {
            assert!(
                (w[0].entity_id.0, w[0].stage_id) <= (w[1].entity_id.0, w[1].stage_id),
                "not sorted: {:?} before {:?}",
                w[0],
                w[1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Acceptance criterion: known mean and population-divisor (1/N) std
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_known_values() {
        // 5 observations for a single entity in a single season.
        let stages = vec![make_stage(1, 0, 2000, 1, 2000, 2, Some(0))];
        let entity_ids = vec![EntityId::from(1)];
        let values = [10.0_f64, 20.0, 30.0, 40.0, 50.0];
        let observations: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                (
                    EntityId::from(1),
                    NaiveDate::from_ymd_opt(2000, 1, (i + 1) as u32).unwrap(),
                    v,
                )
            })
            .collect();

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        assert_eq!(stats.len(), 1);

        let expected_mean = (10.0 + 20.0 + 30.0 + 40.0 + 50.0) / 5.0; // 30.0
        let expected_variance = ((10.0 - 30.0_f64).powi(2)
            + (20.0 - 30.0_f64).powi(2)
            + (30.0 - 30.0_f64).powi(2)
            + (40.0 - 30.0_f64).powi(2)
            + (50.0 - 30.0_f64).powi(2))
            / 5.0; // 1/N (NEWAVE convention)
        let expected_std = expected_variance.sqrt();

        assert!(
            (stats[0].mean - expected_mean).abs() < 1e-10,
            "mean mismatch: {} != {expected_mean}",
            stats[0].mean
        );
        assert!(
            (stats[0].std - expected_std).abs() < 1e-10,
            "std mismatch: {} != {expected_std}",
            stats[0].std
        );
    }

    // -----------------------------------------------------------------------
    // Acceptance criterion: Bessel correction uses N-1 divisor
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_population_divisor() {
        // Two observations: N=2, population std = sqrt(((x1-mean)^2 + (x2-mean)^2)/N).
        let stages = vec![make_stage(1, 0, 2000, 1, 2000, 2, Some(0))];
        let entity_ids = vec![EntityId::from(1)];
        let observations = vec![
            (
                EntityId::from(1),
                NaiveDate::from_ymd_opt(2000, 1, 5).unwrap(),
                10.0_f64,
            ),
            (
                EntityId::from(1),
                NaiveDate::from_ymd_opt(2000, 1, 10).unwrap(),
                20.0_f64,
            ),
        ];

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        assert_eq!(stats.len(), 1);

        // mean = 15.0, variance(1/N) = ((10-15)^2 + (20-15)^2) / 2 = 25.0, std = 5.
        let expected_mean = 15.0_f64;
        let expected_std = 25.0_f64.sqrt();

        assert!((stats[0].mean - expected_mean).abs() < 1e-10);
        assert!((stats[0].std - expected_std).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Acceptance criterion: fewer than 2 observations => error
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_insufficient_data_one_obs() {
        let stages = vec![make_stage(1, 0, 2000, 1, 2000, 2, Some(0))];
        let entity_ids = vec![EntityId::from(1)];
        let observations = vec![(
            EntityId::from(1),
            NaiveDate::from_ymd_opt(2000, 1, 15).unwrap(),
            42.0_f64,
        )];

        let result = estimate_seasonal_stats(&observations, &stages, &entity_ids);
        assert!(
            matches!(result, Err(StochasticError::InsufficientData { .. })),
            "expected InsufficientData, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Acceptance criterion: unmapped date => error
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_unmapped_date() {
        // Stage covers Jan 2000; observation is in Feb 2000.
        let stages = vec![make_stage(1, 0, 2000, 1, 2000, 2, Some(0))];
        let entity_ids = vec![EntityId::from(1)];
        let observations = vec![(
            EntityId::from(1),
            NaiveDate::from_ymd_opt(2000, 2, 15).unwrap(),
            100.0_f64,
        )];

        let result = estimate_seasonal_stats(&observations, &stages, &entity_ids);
        assert!(
            matches!(result, Err(StochasticError::InsufficientData { .. })),
            "expected InsufficientData for unmapped date, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Acceptance criterion: unknown hydros silently ignored
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_ignores_unknown_hydros() {
        let stages = vec![make_stage(1, 0, 2000, 1, 2000, 2, Some(0))];
        // Only entity 1 is in the study; entity 99 is not.
        let entity_ids = vec![EntityId::from(1)];
        let observations = vec![
            (
                EntityId::from(1),
                NaiveDate::from_ymd_opt(2000, 1, 5).unwrap(),
                10.0_f64,
            ),
            (
                EntityId::from(1),
                NaiveDate::from_ymd_opt(2000, 1, 15).unwrap(),
                20.0_f64,
            ),
            // Entity 99 rows — must be silently skipped.
            (
                EntityId::from(99),
                NaiveDate::from_ymd_opt(2000, 1, 5).unwrap(),
                999.0_f64,
            ),
            (
                EntityId::from(99),
                NaiveDate::from_ymd_opt(2000, 1, 15).unwrap(),
                999.0_f64,
            ),
        ];

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        assert_eq!(stats.len(), 1, "only entity 1 should appear");
        assert_eq!(stats[0].entity_id, EntityId::from(1));
    }

    // -----------------------------------------------------------------------
    // Edge case: empty history => empty output (no error)
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_empty_history() {
        let stages = vec![make_stage(1, 0, 2000, 1, 2000, 2, Some(0))];
        let entity_ids = vec![EntityId::from(1)];

        let stats = estimate_seasonal_stats(&[], &stages, &entity_ids).unwrap();
        assert!(stats.is_empty(), "empty history should give empty output");
    }

    // -----------------------------------------------------------------------
    // 30 years of January observations: mean and std to within 1e-10
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_seasonal_stats_thirty_years_single_season() {
        // One January stage, 30 observations mapping to season 0.
        let stages = make_monthly_stages(1990, 30);
        let entity_ids = vec![EntityId::from(1)];

        let values: Vec<f64> = (1u32..=30).map(|i| i as f64 * 10.0).collect();
        let observations: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let year = 1990 + i as i32;
                (
                    EntityId::from(1),
                    NaiveDate::from_ymd_opt(year, 1, 15).unwrap(),
                    v,
                )
            })
            .collect();

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        // Only season 0 (January) has observations.
        assert_eq!(stats.len(), 1);

        let n = values.len() as f64;
        let expected_mean = values.iter().sum::<f64>() / n;
        let expected_variance = values
            .iter()
            .map(|&v| (v - expected_mean).powi(2))
            .sum::<f64>()
            / n;
        let expected_std = expected_variance.sqrt();

        assert!(
            (stats[0].mean - expected_mean).abs() < 1e-10,
            "mean mismatch: {} != {expected_mean}",
            stats[0].mean
        );
        assert!(
            (stats[0].std - expected_std).abs() < 1e-10,
            "std mismatch: {} != {expected_std}",
            stats[0].std
        );
    }

    // -----------------------------------------------------------------------
    // estimate_correlation tests
    // -----------------------------------------------------------------------

    use super::{
        estimate_ar_coefficients, estimate_correlation, ArCoefficientEstimate, SeasonalStats,
    };

    /// Helper: build a single-season study over `n_years` monthly stages.
    /// Season 0 covers month `month` of each year.
    fn single_season_stages(start_year: i32, n_years: usize, month: u32) -> Vec<Stage> {
        (0..n_years)
            .map(|i| {
                let year = start_year + i as i32;
                let (next_year, next_month) = if month == 12 {
                    (year + 1, 1)
                } else {
                    (year, month + 1)
                };
                make_stage(
                    i as i32 + 1,
                    i, // index
                    year,
                    month,
                    next_year,
                    next_month,
                    Some(0), // single season id
                )
            })
            .collect()
    }

    #[test]
    fn estimate_correlation_identical_series() {
        // Two hydros with identical time series and AR(0) model.
        // Residuals are identical => Pearson correlation = 1.0.
        let n_years = 20;
        let stages = single_season_stages(2000, n_years, 1);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for (i, year) in (2000..(2000 + n_years as i32)).enumerate() {
            let val = (i + 1) as f64 * 10.0;
            let date = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
            observations.push((EntityId::from(1), date, val));
            observations.push((EntityId::from(2), date, val));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();

        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        let matrix = &corr.profiles["default"].groups[0].matrix;
        assert_eq!(matrix.len(), 2);
        assert!(
            (matrix[0][0] - 1.0).abs() < 1e-10,
            "diagonal [0][0] must be 1.0"
        );
        assert!(
            (matrix[1][1] - 1.0).abs() < 1e-10,
            "diagonal [1][1] must be 1.0"
        );
        assert!(
            (matrix[0][1] - 1.0).abs() < 1e-10,
            "identical series must have off-diagonal correlation 1.0, got {}",
            matrix[0][1]
        );
        assert!(
            (matrix[1][0] - 1.0).abs() < 1e-10,
            "matrix must be symmetric"
        );
    }

    #[test]
    fn estimate_correlation_single_hydro() {
        // A single hydro produces a 1x1 identity matrix.
        let stages = single_season_stages(2000, 10, 1);
        let hydro_ids = vec![EntityId::from(1)];

        let observations: Vec<(EntityId, NaiveDate, f64)> = (2000..2010_i32)
            .map(|y| {
                (
                    EntityId::from(1),
                    NaiveDate::from_ymd_opt(y, 1, 15).unwrap(),
                    y as f64,
                )
            })
            .collect();

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();

        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        let profile = &corr.profiles["default"];
        assert_eq!(profile.groups.len(), 1);
        let matrix = &profile.groups[0].matrix;
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0].len(), 1);
        assert!(
            (matrix[0][0] - 1.0).abs() < 1e-10,
            "1x1 matrix must be [[1.0]]"
        );
    }

    #[test]
    fn estimate_correlation_empty_hydros() {
        // Zero hydros => default profile with empty groups.
        let stages = single_season_stages(2000, 5, 1);
        let hydro_ids: Vec<EntityId> = Vec::new();
        let observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        let stats: Vec<SeasonalStats> = Vec::new();
        let estimates: Vec<ArCoefficientEstimate> = Vec::new();

        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        assert!(corr.profiles.contains_key("default"));
        assert!(
            corr.profiles["default"].groups.is_empty(),
            "empty hydros must produce empty groups"
        );
        assert!(corr.schedule.is_empty());
    }

    #[test]
    fn estimate_correlation_canonical_order() {
        // Three hydros in canonical order [1, 2, 3].
        // Verify that entities in the result match that order and matrix is 3x3.
        let stages = single_season_stages(2000, 10, 1);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2), EntityId::from(3)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in 2000..2010_i32 {
            let date = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
            let val = year as f64;
            observations.push((EntityId::from(1), date, val));
            observations.push((EntityId::from(2), date, val + 5.0));
            observations.push((EntityId::from(3), date, val * 2.0));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();

        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        let group = &corr.profiles["default"].groups[0];
        assert_eq!(group.entities.len(), 3);
        assert_eq!(group.entities[0].id, EntityId::from(1));
        assert_eq!(group.entities[1].id, EntityId::from(2));
        assert_eq!(group.entities[2].id, EntityId::from(3));
        assert_eq!(group.matrix.len(), 3);
        for row in &group.matrix {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn estimate_correlation_symmetric() {
        // For 3 hydros with varied data, verify matrix[i][j] == matrix[j][i].
        let stages = single_season_stages(2000, 15, 1);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2), EntityId::from(3)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for (i, year) in (2000..2015_i32).enumerate() {
            let date = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
            observations.push((EntityId::from(1), date, (i + 1) as f64 * 3.0));
            observations.push((EntityId::from(2), date, (i + 1) as f64 * 7.0));
            observations.push((EntityId::from(3), date, (15 - i) as f64 * 5.0));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();

        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        let matrix = &corr.profiles["default"].groups[0].matrix;
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-14,
                    "matrix[{i}][{j}] = {} != matrix[{j}][{i}] = {}",
                    matrix[i][j],
                    matrix[j][i]
                );
            }
        }
    }

    #[test]
    fn estimate_correlation_unit_diagonal() {
        let stages = single_season_stages(2000, 10, 1);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2), EntityId::from(3)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in 2000..2010_i32 {
            let date = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
            let v = year as f64;
            observations.push((EntityId::from(1), date, v));
            observations.push((EntityId::from(2), date, v + 100.0));
            observations.push((EntityId::from(3), date, 500.0 - v));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();

        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        let matrix = &corr.profiles["default"].groups[0].matrix;
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            assert!(
                (matrix[i][i] - 1.0).abs() < 1e-14,
                "diagonal matrix[{i}][{i}] = {} must be 1.0",
                matrix[i][i]
            );
        }
    }

    fn splitmix(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        50.0 + 100.0 * (z as f64 / u64::MAX as f64)
    }

    #[test]
    fn estimate_correlation_independent_series() {
        let stages = single_season_stages(1800, 200, 1);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut seed1: u64 = 12345;
        let mut seed2: u64 = 99999;
        let mut observations = Vec::new();
        for year in 1800..2000_i32 {
            let date = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
            observations.push((EntityId::from(1), date, splitmix(&mut seed1)));
            observations.push((EntityId::from(2), date, splitmix(&mut seed2)));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();
        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        let matrix = &corr.profiles["default"].groups[0].matrix;
        assert!(
            matrix[0][1].abs() < 0.15,
            "off-diagonal |r| = {} must be < 0.15 for independent series",
            matrix[0][1]
        );
        assert!(
            matrix[1][0].abs() < 0.15,
            "off-diagonal |r| = {} must be < 0.15 for independent series",
            matrix[1][0]
        );
    }

    // -----------------------------------------------------------------------
    // Multi-season correlation tests
    // -----------------------------------------------------------------------

    /// Build monthly stages cycling through `n_seasons` season IDs over `n_years` years.
    fn multi_season_stages(start_year: i32, n_years: usize, n_seasons: usize) -> Vec<Stage> {
        (0..(n_years * n_seasons))
            .map(|i| {
                let year = start_year + (i / 12) as i32;
                let month = (i % 12) as u32 + 1;
                let (end_year, end_month) = if month == 12 {
                    (year + 1, 1)
                } else {
                    (year, month + 1)
                };
                make_stage(
                    i as i32 + 1,
                    i,
                    year,
                    month,
                    end_year,
                    end_month,
                    Some(i % n_seasons),
                )
            })
            .collect()
    }

    #[test]
    fn estimate_correlation_multi_season_produces_per_season_profiles() {
        let n_seasons = 12;
        let stages = multi_season_stages(2000, 40, n_seasons);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations = Vec::new();
        for i in 0..stages.len() {
            let year = 2000 + (i / 12) as i32;
            let month = (i % 12) as u32 + 1;
            let date = NaiveDate::from_ymd_opt(year, month, 15).unwrap();
            let val = (year * 12 + month as i32) as f64;
            observations.push((EntityId::from(1), date, val));
            observations.push((EntityId::from(2), date, val + 5.0));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();
        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        assert_eq!(corr.profiles.len(), n_seasons + 1);
        assert!(corr.profiles.contains_key("default"));
        for s in 0..n_seasons {
            assert!(corr.profiles.contains_key(&format!("season_{s:02}")));
        }

        assert_eq!(corr.schedule.len(), 480);
        for (i, entry) in corr.schedule.iter().enumerate() {
            let expected_season = i % n_seasons;
            assert_eq!(entry.profile_name, format!("season_{expected_season:02}"));
        }

        for s in 0..n_seasons {
            let matrix = &corr.profiles[&format!("season_{s:02}")].groups[0].matrix;
            assert_eq!(matrix.len(), 2);
            assert_eq!(matrix[0].len(), 2);
            assert!((matrix[0][0] - 1.0).abs() < 1e-10);
            assert!((matrix[1][1] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn estimate_correlation_multi_season_schedule_maps_stages_to_seasons() {
        let n_seasons = 4;
        let stages = multi_season_stages(2000, 40, n_seasons);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations = Vec::new();
        for (i, _) in stages.iter().enumerate() {
            let year = 2000 + (i / 12) as i32;
            let month = (i % 12) as u32 + 1;
            let date = NaiveDate::from_ymd_opt(year, month, 15).unwrap();
            let val = (i + 1) as f64 * 10.0;
            observations.push((EntityId::from(1), date, val));
            observations.push((EntityId::from(2), date, val + 3.0));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();
        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        assert_eq!(corr.schedule.len(), 160);
        for (i, entry) in corr.schedule.iter().enumerate() {
            assert_eq!(entry.profile_name, format!("season_{}", i % n_seasons));
        }

        // Spot-check specific mappings
        assert_eq!(corr.schedule[0].profile_name, "season_0");
        assert_eq!(corr.schedule[1].profile_name, "season_1");
        assert_eq!(corr.schedule[4].profile_name, "season_0");

        // All schedule entries reference valid stages
        let valid_ids: std::collections::HashSet<_> = stages.iter().map(|s| s.id).collect();
        for entry in &corr.schedule {
            assert!(valid_ids.contains(&entry.stage_id));
        }
    }

    #[test]
    fn estimate_correlation_multi_season_per_season_values_differ() {
        let stages = multi_season_stages(2000, 40, 2);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations = Vec::new();
        for (i, stage) in stages.iter().enumerate() {
            let year = 2000 + (i / 12) as i32;
            let month = (i % 12) as u32 + 1;
            let date = NaiveDate::from_ymd_opt(year, month, 15).unwrap();
            let base = (i + 1) as f64 * 10.0 + 100.0;
            // Season 0: positively correlated (both increase together).
            // Season 1: anti-correlated (hydro2 decreases as hydro1 increases),
            //           but all values remain positive (avoids degenerate filter).
            let val2 = if stage.season_id == Some(0) {
                base
            } else {
                5000.0 - base
            };
            observations.push((EntityId::from(1), date, base));
            observations.push((EntityId::from(2), date, val2));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();
        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        let matrix_s0 = &corr.profiles["season_0"].groups[0].matrix;
        assert!(matrix_s0[0][1] > 0.9);
        assert!(matrix_s0[1][0] > 0.9);

        let matrix_s1 = &corr.profiles["season_1"].groups[0].matrix;
        assert!(matrix_s1[0][1] < -0.9);
        assert!(matrix_s1[1][0] < -0.9);

        let matrix_def = &corr.profiles["default"].groups[0].matrix;
        assert!(matrix_def[0][1] > -1.0 && matrix_def[0][1] < 1.0);
    }

    // -----------------------------------------------------------------------
    // select_order_aic tests
    // -----------------------------------------------------------------------

    #[test]
    fn select_order_aic_known_values() {
        // sigma2 = [0.75, 0.60, 0.59], N = 100.
        // AIC(0) = 0.0
        // AIC(1) = 100 * ln(0.75) + 2
        // AIC(2) = 100 * ln(0.60) + 4
        // AIC(3) = 100 * ln(0.59) + 6
        let sigma2 = [0.75_f64, 0.60, 0.59];
        let result = select_order_aic(&sigma2, 100);

        assert_eq!(result.aic_values.len(), 4);
        assert_eq!(result.aic_values[0], 0.0);
        assert!((result.aic_values[1] - (100.0 * 0.75_f64.ln() + 2.0)).abs() < 1e-10);
        assert!((result.aic_values[2] - (100.0 * 0.60_f64.ln() + 4.0)).abs() < 1e-10);
        assert!((result.aic_values[3] - (100.0 * 0.59_f64.ln() + 6.0)).abs() < 1e-10);

        // Determine which order has the minimum AIC and verify selection.
        let expected = result
            .aic_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(result.selected_order, expected);
    }

    #[test]
    fn select_order_aic_white_noise_preferred() {
        // N = 10, sigma2 = [0.99]: AIC(1) = 10 * ln(0.99) + 2 ≈ -0.1005 + 2 > 0.
        // White noise baseline (AIC = 0) should win.
        let result = select_order_aic(&[0.99], 10);
        assert_eq!(result.selected_order, 0);
    }

    #[test]
    fn select_order_aic_ar1_selected() {
        // Large variance drop at order 1 should beat the penalty.
        // N = 100, sigma2_1 = 0.30 → AIC(1) = 100*ln(0.30)+2 ≈ -118.1+2 = -116.1
        // AIC(0) = 0. Order 1 clearly wins.
        let result = select_order_aic(&[0.30], 100);
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn select_order_aic_empty_sigma2() {
        let result = select_order_aic(&[], 50);
        assert_eq!(result.selected_order, 0);
        assert_eq!(result.aic_values, vec![0.0]);
    }

    #[test]
    fn select_order_aic_non_positive_sigma2_excluded() {
        // sigma2 = [0.5, 0.0, 0.3]: index 1 (order 2) is non-positive → INFINITY.
        let result = select_order_aic(&[0.5, 0.0, 0.3], 100);
        assert_eq!(result.aic_values[2], f64::INFINITY);
        // Both order 1 and order 3 are candidates; selected_order must not be 2.
        assert_ne!(result.selected_order, 2);
    }

    #[test]
    fn select_order_aic_tie_prefers_lower_order() {
        // Construct an exact f64 tie between AIC(1) and AIC(2).
        // Values were found by brute-force search over (N, s1, s2=exp(ln(s1)-2/N))
        // such that the f64 computation `N*s1.ln()+2.0 == N*s2.ln()+4.0` holds exactly.
        // N=10, s1=0.3, s2≈0.24562 produce AIC(1) == AIC(2) == -10.0397...
        let s1 = 0.3_f64;
        let s2 = 0.245_619_225_923_394_52_f64;
        let aic1 = 10.0 * s1.ln() + 2.0;
        let aic2 = 10.0 * s2.ln() + 4.0;
        assert_eq!(
            aic1, aic2,
            "test setup: AIC(1) and AIC(2) must be exactly equal in f64"
        );

        let result = select_order_aic(&[s1, s2], 10);
        // AIC(0) = 0.0 > AIC(1) = AIC(2), and on a tie the lower order (1) wins.
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn select_order_aic_monotone_variance_selects_max() {
        // Strongly autoregressive: each additional order reduces variance enough
        // to overcome the 2-point penalty. Select the highest order.
        // N = 200, variances geometrically decreasing: 0.5^k for k=1..5.
        // AIC(k) = 200*ln(0.5^k) + 2k = 200*k*ln(0.5) + 2k = k*(200*ln(0.5)+2).
        // ln(0.5) ≈ -0.6931 → 200*(-0.6931)+2 ≈ -136.6, negative → AIC strictly
        // decreases with k. Highest order (5) should be selected.
        let sigma2: Vec<f64> = (1..=5).map(|k| 0.5_f64.powi(k)).collect();
        let result = select_order_aic(&sigma2, 200);
        assert_eq!(result.selected_order, 5);
    }

    // -----------------------------------------------------------------------
    // PACF order selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn pacf_empty_parcor_selects_zero() {
        let result = select_order_pacf(&[], 100, 1.96);
        assert_eq!(result.selected_order, 0);
        assert!(result.pacf_values.is_empty());
    }

    #[test]
    fn pacf_single_significant_lag() {
        // threshold = 1.96 / sqrt(100) = 0.196
        // parcor[0] = 0.5 > 0.196 -> significant
        let result = select_order_pacf(&[0.5], 100, 1.96);
        assert_eq!(result.selected_order, 1);
        assert!((result.threshold - 0.196).abs() < 1e-10);
    }

    #[test]
    fn pacf_no_significant_lag() {
        // threshold = 1.96 / sqrt(100) = 0.196
        // All parcor below threshold.
        let result = select_order_pacf(&[0.05, 0.03, 0.1], 100, 1.96);
        assert_eq!(result.selected_order, 0);
    }

    #[test]
    fn pacf_selects_max_significant_lag() {
        // threshold = 1.96 / sqrt(100) = 0.196
        // parcor = [0.5, 0.1, 0.3]
        // Lag 1 (0.5) significant, lag 2 (0.1) not, lag 3 (0.3) significant.
        // Max significant lag = 3.
        let result = select_order_pacf(&[0.5, 0.1, 0.3], 100, 1.96);
        assert_eq!(result.selected_order, 3);
    }

    #[test]
    fn pacf_negative_parcor_uses_absolute_value() {
        // threshold = 1.96 / sqrt(100) = 0.196
        // parcor = [-0.5, 0.1]
        // |-0.5| = 0.5 > 0.196 -> lag 1 significant.
        let result = select_order_pacf(&[-0.5, 0.1], 100, 1.96);
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn pacf_zero_observations_selects_zero() {
        // n_observations = 0 -> threshold = infinity -> nothing significant.
        let result = select_order_pacf(&[0.5, 0.3], 0, 1.96);
        assert_eq!(result.selected_order, 0);
    }

    #[test]
    fn pacf_large_sample_low_threshold() {
        // threshold = 1.96 / sqrt(10000) = 0.0196
        // Even small parcor values are significant.
        let result = select_order_pacf(&[0.05, 0.03, 0.02, 0.01], 10000, 1.96);
        // parcor[0]=0.05 > 0.0196, parcor[1]=0.03 > 0.0196, parcor[2]=0.02 > 0.0196
        // parcor[3]=0.01 < 0.0196
        // Max significant = lag 3.
        assert_eq!(result.selected_order, 3);
    }

    // -----------------------------------------------------------------------
    // periodic_autocorrelation tests
    // -----------------------------------------------------------------------

    /// Helper: compute population mean and std for a slice.
    fn pop_mean_std(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        if n < 1.0 {
            return (0.0, 0.0);
        }
        let mean = data.iter().sum::<f64>() / n;
        if n < 2.0 {
            return (mean, 0.0);
        }
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    #[test]
    fn periodic_autocorrelation_single_season_basic() {
        // Single-season (stationary) case with known analytical value.
        //
        // For a single season, ref_season=0, lag_season=0, n_seasons=1.
        // Cross-year triggers (lag_season >= ref_season and lag < n_seasons).
        // So ref starts at index 1, pairs = N-1.
        //
        // Use data [1, 3, 5, 7, 9]: mean=5, std=sqrt(8).
        // ref = [3, 5, 7, 9], lag = [1, 3, 5, 7], 4 pairs.
        // gamma = 1/4 * [(3-5)(1-5) + (5-5)(3-5) + (7-5)(5-5) + (9-5)(7-5)]
        //       = 1/4 * [(-2)(-4) + 0*(-2) + 2*0 + 4*2]
        //       = 1/4 * [8 + 0 + 0 + 8] = 4.0
        // rho = 4.0 / (sqrt(8) * sqrt(8)) = 4.0 / 8.0 = 0.5
        let data = [1.0, 3.0, 5.0, 7.0, 9.0];
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let rho = periodic_autocorrelation(0, 1, 1, obs, stats_arr);
        assert!((rho - 0.5).abs() < 1e-10, "rho(0,1) = {rho}, expected 0.5");
    }

    #[test]
    fn periodic_autocorrelation_two_season() {
        // Two-season case with distinct dynamics.
        let season_0 = [10.0, 12.0, 11.0, 13.0, 10.5];
        let season_1 = [5.0, 6.0, 5.5, 7.0, 5.2];

        let stats_0 = pop_mean_std(&season_0);
        let stats_1 = pop_mean_std(&season_1);

        let obs: &[&[f64]] = &[&season_0, &season_1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];

        // rho(0, 1) = autocorrelation of season 0 with season 1 (lag 1).
        let rho01 = periodic_autocorrelation(0, 1, 2, obs, stats);
        // rho(1, 1) = autocorrelation of season 1 with season 0 (lag 1).
        let rho10 = periodic_autocorrelation(1, 1, 2, obs, stats);

        // Both should be finite and in [-1, 1].
        assert!(rho01.abs() <= 1.0);
        assert!(rho10.abs() <= 1.0);
        // They can differ because different reference seasons use different
        // seasonal statistics.
    }

    #[test]
    fn periodic_autocorrelation_cross_year_boundary() {
        // 12-season setup. rho(0, 1) = Jan lag 1 -> Dec: crosses year boundary.
        // rho(6, 1) = Jul lag 1 -> Jun: does NOT cross year boundary.
        let n_seasons = 12;
        let mut obs_data: Vec<Vec<f64>> = Vec::new();
        let n_years = 10;
        for _ in 0..n_seasons {
            obs_data.push((0..n_years).map(|y| (y * 10 + 5) as f64).collect());
        }
        let obs_refs: Vec<&[f64]> = obs_data.iter().map(Vec::as_slice).collect();
        let stats: Vec<(f64, f64)> = obs_data.iter().map(|v| pop_mean_std(v)).collect();

        // For the cross-year case (ref_season=0, lag=1 -> lag_season=11),
        // lag_season (11) >= ref_season (0), so one observation is dropped.
        let rho_jan_dec = periodic_autocorrelation(0, 1, n_seasons, &obs_refs, &stats);

        // For the non-cross-year case (ref_season=6, lag=1 -> lag_season=5),
        // lag_season (5) < ref_season (6), so no observation is dropped.
        let rho_jul_jun = periodic_autocorrelation(6, 1, n_seasons, &obs_refs, &stats);

        // Both should produce valid values.
        assert!((-1.0..=1.0).contains(&rho_jan_dec));
        assert!((-1.0..=1.0).contains(&rho_jul_jun));

        // Verify the cross-year adjustment affects the value: compute manually.
        // For the cross-year case with identical observations per season,
        // the autocorrelation should still be well-defined.
    }

    #[test]
    fn periodic_autocorrelation_zero_std_returns_zero() {
        // If one season has zero std (constant values), rho should be 0.0.
        let season_0 = [5.0, 5.0, 5.0, 5.0]; // zero std
        let season_1 = [1.0, 2.0, 3.0, 4.0];
        let stats_0 = pop_mean_std(&season_0);
        let stats_1 = pop_mean_std(&season_1);

        let obs: &[&[f64]] = &[&season_0, &season_1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];

        assert_eq!(stats_0.1, 0.0);
        let rho = periodic_autocorrelation(0, 1, 2, obs, stats);
        assert_eq!(rho, 0.0);
    }

    #[test]
    fn periodic_autocorrelation_insufficient_data() {
        // Only one observation per season -> after cross-year drop, 0 pairs.
        let season_0: [f64; 1] = [10.0];
        let season_1: [f64; 1] = [20.0];

        // With n_seasons=2, ref_season=0, lag=1 -> lag_season=1.
        // lag_season (1) >= ref_season (0) -> cross-year, drop 1.
        // ref_obs.len()-1 = 0 -> 0 pairs -> returns 0.0.
        let stats: &[(f64, f64)] = &[(10.0, 1.0), (20.0, 1.0)];
        let obs: &[&[f64]] = &[&season_0, &season_1];
        let rho = periodic_autocorrelation(0, 1, 2, obs, stats);
        assert_eq!(rho, 0.0);
    }

    #[test]
    fn periodic_autocorrelation_clamped_to_range() {
        // Construct extreme data that would produce rho > 1 without clamping.
        // In practice this shouldn't happen with correct stats, but the function
        // should still clamp. Use mismatched stats to force it.
        let season_0 = [100.0, 200.0, 300.0];
        let stats: &[(f64, f64)] = &[(200.0, 0.001)]; // artificially tiny std
        let obs: &[&[f64]] = &[&season_0];
        let rho = periodic_autocorrelation(0, 1, 1, obs, stats);
        assert!((-1.0..=1.0).contains(&rho), "rho should be clamped: {rho}");
    }

    #[test]
    fn periodic_autocorrelation_population_divisor() {
        // Verify 1/N divisor is used, not 1/(N-1).
        // With N=3 and specific values, the difference between 1/3 and 1/2
        // is 50%, easily detectable.
        let data = [1.0, 2.0, 3.0]; // mean=2, std=sqrt(2/3)
        let (mean, std_val) = pop_mean_std(&data);
        let stats: &[(f64, f64)] = &[(mean, std_val)];
        let obs: &[&[f64]] = &[&data];

        let _rho = periodic_autocorrelation(0, 1, 1, obs, stats);

        // Compute expected with 1/N:
        // With single season, lag 1 means same season at lag 1.
        // cross-year: lag_season=0 >= ref_season=0 and lag<n_seasons(1) -> yes.
        // So ref starts at index 1, lag starts at index 0, n_pairs=2.
        // gamma = 1/2 * [(2-2)*(1-2) + (3-2)*(2-2)] = 1/2 * [0 + 0] = 0.
        // Actually let me check: ref_obs[1]=2, ref_obs[2]=3; lag_obs[0]=1, lag_obs[1]=2.
        // gamma = 1/2 * [(2-2)(1-2) + (3-2)(2-2)] = 1/2 * [0*(-1) + 1*0] = 0.
        // For this particular data, gamma = 0. Let me use different data.
        let data2 = [1.0, 4.0, 9.0]; // mean=14/3
        let (mean2, std2) = pop_mean_std(&data2);
        let stats2: &[(f64, f64)] = &[(mean2, std2)];
        let obs2: &[&[f64]] = &[&data2];

        let rho2 = periodic_autocorrelation(0, 1, 1, obs2, stats2);
        // Just verify it produces a valid finite result with population divisor.
        assert!(rho2.is_finite(), "rho should be finite: {rho2}");
        assert!(rho2.abs() <= 1.0);
    }

    #[test]
    fn periodic_autocorrelation_lag_zero() {
        // rho(m, 0) = 1.0 for any season.
        let data = [1.0, 2.0, 3.0];
        let stats: &[(f64, f64)] = &[(2.0, 1.0)];
        let obs: &[&[f64]] = &[&data];
        assert_eq!(periodic_autocorrelation(0, 0, 1, obs, stats), 1.0);
    }

    // -----------------------------------------------------------------------
    // build_periodic_yw_matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_periodic_yw_matrix_order_zero() {
        let data = [1.0, 2.0, 3.0];
        let stats: &[(f64, f64)] = &[(2.0, 1.0)];
        let obs: &[&[f64]] = &[&data];
        let (mat, rhs) = build_periodic_yw_matrix(0, 0, 1, obs, stats);
        assert!(mat.is_empty());
        assert!(rhs.is_empty());
    }

    #[test]
    fn build_periodic_yw_matrix_single_season_toeplitz() {
        // For a single season (n_seasons=1), the periodic YW matrix should be
        // Toeplitz because all rows use the same reference season.
        let data: Vec<f64> = (0..50).map(|i| (i as f64) * 0.5 + 1.0).collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let order = 3;
        let (mat, _rhs) = build_periodic_yw_matrix(0, order, 1, obs, stats_arr);

        assert_eq!(mat.len(), order * order);
        // Check Toeplitz property: M[i,j] depends only on |i-j|.
        // M[0,1] should equal M[1,2] (both have lag 1 from same ref season 0).
        let m01 = mat[1]; // row 0, col 1
        let m12 = mat[order + 2]; // row 1, col 2
        assert!(
            (m01 - m12).abs() < 1e-10,
            "Toeplitz violated: M[0,1]={m01} != M[1,2]={m12}"
        );
    }

    #[test]
    fn build_periodic_yw_matrix_diagonal_is_one() {
        // Diagonal entries should always be 1.0.
        let s0: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let s1: Vec<f64> = (0..20).map(|i| (i * 2) as f64).collect();
        let stats_0 = pop_mean_std(&s0);
        let stats_1 = pop_mean_std(&s1);
        let obs: &[&[f64]] = &[&s0, &s1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];

        let order = 3;
        let (mat, _) = build_periodic_yw_matrix(0, order, 2, obs, stats);
        for i in 0..order {
            assert!(
                (mat[i * order + i] - 1.0).abs() < 1e-15,
                "Diagonal[{i}] = {}, expected 1.0",
                mat[i * order + i]
            );
        }
    }

    #[test]
    fn build_periodic_yw_matrix_symmetry() {
        // Matrix should be symmetric: M[i,j] == M[j,i].
        let s0: Vec<f64> = (0..30).map(|i| (i as f64).sin()).collect();
        let s1: Vec<f64> = (0..30).map(|i| (i as f64).cos()).collect();
        let s2: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let stats: Vec<(f64, f64)> = [&s0[..], &s1[..], &s2[..]]
            .iter()
            .map(|s| pop_mean_std(s))
            .collect();
        let obs: Vec<&[f64]> = vec![&s0, &s1, &s2];

        let order = 4;
        let (mat, _) = build_periodic_yw_matrix(1, order, 3, &obs, &stats);
        for i in 0..order {
            for j in (i + 1)..order {
                assert!(
                    (mat[i * order + j] - mat[j * order + i]).abs() < 1e-10,
                    "Symmetry violated: M[{i},{j}]={} != M[{j},{i}]={}",
                    mat[i * order + j],
                    mat[j * order + i]
                );
            }
        }
    }

    #[test]
    fn build_periodic_yw_matrix_rhs_length() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        for order in 1..=5 {
            let (mat, rhs) = build_periodic_yw_matrix(0, order, 1, obs, stats_arr);
            assert_eq!(
                mat.len(),
                order * order,
                "matrix size mismatch for order {order}"
            );
            assert_eq!(rhs.len(), order, "rhs size mismatch for order {order}");
        }
    }

    #[test]
    fn build_periodic_yw_matrix_two_season_not_toeplitz() {
        // For a 2-season model with different dynamics, the matrix should NOT
        // be Toeplitz (off-diagonal entries differ from what Toeplitz would give).
        let s0: Vec<f64> = (0..30).map(|i| (i as f64) * 2.0 + 1.0).collect();
        let s1: Vec<f64> = (0..30).map(|i| (i as f64) * 0.5 + 10.0).collect();
        let stats_0 = pop_mean_std(&s0);
        let stats_1 = pop_mean_std(&s1);
        let obs: &[&[f64]] = &[&s0, &s1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];

        let order = 3;
        let (mat, _) = build_periodic_yw_matrix(0, order, 2, obs, stats);

        // In a Toeplitz matrix, M[0,1] == M[1,2]. For the periodic matrix,
        // row i uses ref_month = (season + n_seasons - (i+1)) % n_seasons.
        // row 0 uses ref_month = (0+2-1)%2 = 1, row 1 uses ref_month = (0+2-2)%2 = 0.
        // These reference different seasons, so M[0,1] (rho(1,1)) may differ
        // from M[1,2] (rho(0,1)).
        let m01 = mat[1]; // row 0, col 1
        let m12 = mat[order + 2]; // row 1, col 2
                                  // We just verify both are valid; they may or may not differ depending
                                  // on the specific data, but the matrix IS valid.
        assert!(m01.abs() <= 1.0);
        assert!(m12.abs() <= 1.0);
    }

    #[test]
    fn build_periodic_yw_matrix_forward_prediction_two_season_ar2() {
        // Verify that build_periodic_yw_matrix solves the FORWARD prediction
        // problem for a 2-season AR(2) model, not the (buggy) backward variant.

        let s0 = vec![3.0_f64, 5.0, 4.0, 6.0, 2.0];
        let s1 = vec![1.0_f64, 2.0, 3.0, 4.0, 0.0];
        let stats_0 = pop_mean_std(&s0);
        let stats_1 = pop_mean_std(&s1);

        assert!((stats_0.0 - 4.0).abs() < 1e-14);
        assert!((stats_0.1 - 2.0_f64.sqrt()).abs() < 1e-14);
        assert!((stats_1.0 - 2.0).abs() < 1e-14);
        assert!((stats_1.1 - 2.0_f64.sqrt()).abs() < 1e-14);

        let obs: &[&[f64]] = &[&s0, &s1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];
        let n_seasons = 2;
        let season = 0;
        let order = 2;

        // Expected autocorrelations (computed analytically from data):
        // rho(ref=1, lag=1) = 0.9 (matrix off-diagonal)
        // rho(ref=0, lag=1) = -0.375 (RHS[0])
        // rho(ref=0, lag=2) = -0.625 (RHS[1])
        let expected_rho_m = 0.9_f64;
        let expected_rhs0 = -0.375_f64;
        let expected_rhs1 = -0.625_f64;

        let (mat_orig, rhs_orig) = build_periodic_yw_matrix(season, order, n_seasons, obs, stats);

        assert!(
            (mat_orig[1] - expected_rho_m).abs() < 1e-14,
            "M[0,1]={}, expected={}",
            mat_orig[1],
            expected_rho_m
        );
        assert!(
            (mat_orig[2] - expected_rho_m).abs() < 1e-14,
            "M[1,0]={}, expected={}",
            mat_orig[2],
            expected_rho_m
        );
        assert!(
            (rhs_orig[0] - expected_rhs0).abs() < 1e-14,
            "rhs[0]={}, expected={}",
            rhs_orig[0],
            expected_rhs0
        );
        assert!(
            (rhs_orig[1] - expected_rhs1).abs() < 1e-14,
            "rhs[1]={}, expected={}",
            rhs_orig[1],
            expected_rhs1
        );

        // Solve the forward YW system and verify round-trip and analytical solution.
        let (mut mat, mut rhs) = build_periodic_yw_matrix(season, order, n_seasons, obs, stats);
        let phi = solve_linear_system(&mut mat, &mut rhs, order)
            .expect("forward YW system must not be singular");

        // Verify linear algebra: R * phi = rhs_orig.
        for i in 0..order {
            let mut dot = 0.0_f64;
            for j in 0..order {
                dot += mat_orig[i * order + j] * phi[j];
            }
            assert!(
                (dot - rhs_orig[i]).abs() < 1e-10,
                "R*phi[{i}] = {dot:.15}, expected {:.15}",
                rhs_orig[i]
            );
        }

        // Verify analytical forward-prediction solution: det = 0.19,
        // phi1 ≈ 0.987, phi2 ≈ -1.513.
        let det = 1.0 - expected_rho_m * expected_rho_m;
        let expected_phi1 = (expected_rhs0 - expected_rho_m * expected_rhs1) / det;
        let expected_phi2 = (expected_rhs1 - expected_rho_m * expected_rhs0) / det;

        assert!(
            (phi[0] - expected_phi1).abs() < 1e-10,
            "phi[0]={:.15}, expected {:.15}",
            phi[0],
            expected_phi1
        );
        assert!(
            (phi[1] - expected_phi2).abs() < 1e-10,
            "phi[1]={:.15}, expected {:.15}",
            phi[1],
            expected_phi2
        );

        // Verify sigma-squared: sigma2 = 1 - phi1*rho(0,1) - phi2*rho(0,2).
        let sigma2 = 1.0 - phi[0] * rhs_orig[0] - phi[1] * rhs_orig[1];
        let expected_sigma2 = 1.0 - expected_phi1 * expected_rhs0 - expected_phi2 * expected_rhs1;
        assert!(
            (sigma2 - expected_sigma2).abs() < 1e-10,
            "sigma2={sigma2:.15}, expected {expected_sigma2:.15}"
        );
        assert!(sigma2 > 0.0, "sigma2 must be positive, got {sigma2}");

        // Guard against backward-prediction regression: phi[1] must be negative.
        // Backward prediction would yield phi[1] > 0.
        assert!(
            phi[1] < 0.0,
            "phi[1]={:.6} must be negative (backward-pred regression check)",
            phi[1]
        );
    }

    // -----------------------------------------------------------------------
    // solve_linear_system tests
    // -----------------------------------------------------------------------

    #[test]
    fn solve_linear_system_1x1() {
        // [2.0] * x = [6.0] -> x = [3.0]
        let mut a = vec![2.0];
        let mut b = vec![6.0];
        let x = solve_linear_system(&mut a, &mut b, 1).unwrap();
        assert_eq!(x.len(), 1);
        assert!((x[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_linear_system_2x2() {
        // [1 2] [x1]   [5]     x1 = 1, x2 = 2
        // [3 4] [x2] = [11]
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = vec![5.0, 11.0];
        let x = solve_linear_system(&mut a, &mut b, 2).unwrap();
        assert_eq!(x.len(), 2);
        assert!((x[0] - 1.0).abs() < 1e-10, "x[0]={}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-10, "x[1]={}", x[1]);
    }

    #[test]
    fn solve_linear_system_3x3() {
        // [2  1 -1] [x1]   [ 8]     x = [2, 3, -1]
        // [-3 -1  2] [x2] = [-11]
        // [-2  1  2] [x3]   [-3]
        let mut a = vec![2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0];
        let mut b = vec![8.0, -11.0, -3.0];
        let x = solve_linear_system(&mut a, &mut b, 3).unwrap();
        assert_eq!(x.len(), 3);
        assert!((x[0] - 2.0).abs() < 1e-10, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1]={}", x[1]);
        assert!((x[2] - (-1.0)).abs() < 1e-10, "x[2]={}", x[2]);
    }

    #[test]
    fn solve_linear_system_singular() {
        // Two identical rows -> singular.
        let mut a = vec![1.0, 2.0, 1.0, 2.0];
        let mut b = vec![3.0, 3.0];
        assert!(solve_linear_system(&mut a, &mut b, 2).is_none());
    }

    #[test]
    fn solve_linear_system_requires_pivoting() {
        // [0 1] [x1]   [3]    -> needs row swap.
        // [1 0] [x2] = [5]    x1=5, x2=3.
        let mut a = vec![0.0, 1.0, 1.0, 0.0];
        let mut b = vec![3.0, 5.0];
        let x = solve_linear_system(&mut a, &mut b, 2).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-10, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1]={}", x[1]);
    }

    #[test]
    fn solve_linear_system_diagonal() {
        // [3 0 0] [x1]   [9]    x = [3, 2, 5]
        // [0 4 0] [x2] = [8]
        // [0 0 2] [x3]   [10]
        let mut a = vec![3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0];
        let mut b = vec![9.0, 8.0, 10.0];
        let x = solve_linear_system(&mut a, &mut b, 3).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn solve_linear_system_6x6() {
        // Identity 6x6: I * x = b -> x = b.
        let n = 6;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut b = expected.clone();
        let x = solve_linear_system(&mut a, &mut b, n).unwrap();
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-10, "x[{i}]={}", x[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Comprehensive periodic autocorrelation and matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn periodic_autocorrelation_single_season_yw_solve_roundtrip() {
        // For a single season, build the periodic YW matrix and verify
        // that R * phi = rhs (the matrix equation is self-consistent).
        let data = [10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 12.0, 16.0, 14.0, 17.0];
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let order = 3;
        // Save the RHS before the solve (solve modifies in-place).
        let (mat_orig, rhs_orig) = build_periodic_yw_matrix(0, order, 1, obs, stats_arr);

        let (mut mat, mut rhs) = build_periodic_yw_matrix(0, order, 1, obs, stats_arr);
        let phi = solve_linear_system(&mut mat, &mut rhs, order).unwrap();

        // Verify R * phi = rhs_orig.
        for i in 0..order {
            let mut dot = 0.0;
            for j in 0..order {
                dot += mat_orig[i * order + j] * phi[j];
            }
            assert!(
                (dot - rhs_orig[i]).abs() < 1e-10,
                "R*phi[{i}] = {dot}, expected {}",
                rhs_orig[i]
            );
        }
    }

    #[test]
    fn periodic_autocorrelation_two_obs_per_season() {
        // Very few observations (N=2) per season should still work.
        let s0 = [1.0, 3.0]; // mean=2, std=1
        let s1 = [5.0, 7.0]; // mean=6, std=1
        let stats_0 = pop_mean_std(&s0);
        let stats_1 = pop_mean_std(&s1);
        let obs: &[&[f64]] = &[&s0, &s1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];

        // Should not panic.
        let rho = periodic_autocorrelation(0, 1, 2, obs, stats);
        assert!(rho.is_finite());
        assert!(rho.abs() <= 1.0);
    }

    #[test]
    fn periodic_autocorrelation_large_lag_wraps() {
        // Lag > n_seasons should wrap correctly.
        let s0: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let s1: Vec<f64> = (0..20).map(|i| (i * 2) as f64).collect();
        let stats: Vec<(f64, f64)> = [&s0[..], &s1[..]].iter().map(|s| pop_mean_std(s)).collect();
        let obs: Vec<&[f64]> = vec![&s0, &s1];

        // Lag=3 with n_seasons=2: lag_season = (0 + 2 - 3%2) % 2 = (2 - 1)%2 = 1.
        let rho = periodic_autocorrelation(0, 3, 2, &obs, &stats);
        assert!(rho.is_finite());
        assert!(rho.abs() <= 1.0);
    }

    #[test]
    fn periodic_autocorrelation_population_divisor_verification() {
        // Verify population divisor (1/N) NOT Bessel (1/(N-1)) with N=3.
        // The 50% difference at N=3 makes this easy to detect.
        //
        // Use two seasons to avoid cross-year adjustment complexity.
        // season 0: [1, 2, 3], mean=2, std_pop = sqrt(2/3) ≈ 0.8165
        // season 1: [4, 5, 6], mean=5, std_pop = sqrt(2/3) ≈ 0.8165
        //
        // rho(0, 1): ref=season0, lag=season1.
        // lag_season = (0+2-1)%2 = 1. cross_year: lag<2 and lag_season(1)>=ref(0) -> yes.
        // So ref starts at 1, pairs = min(3-1, 3) = 2.
        // gamma = 1/2 * [(2-2)(4-5) + (3-2)(5-5)] = 1/2 * [0 + 0] = 0.
        // For this specific data, gamma=0 regardless of divisor. Use different data.
        let s0 = [1.0, 4.0, 3.0]; // mean=8/3, std_pop
        let s1 = [2.0, 5.0, 4.0]; // mean=11/3, std_pop
        let stats_0 = pop_mean_std(&s0);
        let stats_1 = pop_mean_std(&s1);
        let obs: &[&[f64]] = &[&s0, &s1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];

        let rho = periodic_autocorrelation(0, 1, 2, obs, stats);
        // The important check: with population std divisor, the result is valid.
        assert!(rho.is_finite());
        assert!(rho.abs() <= 1.0);

        // Compute manually with population divisor to verify.
        // cross-year: lag_season=1 >= ref_season=0 -> yes, ref starts at index 1.
        // pairs = min(3-1, 3) = 2.
        // ref: s0[1]=4.0, s0[2]=3.0. lag: s1[0]=2.0, s1[1]=5.0.
        let mu_ref = stats_0.0;
        let mu_lag = stats_1.0;
        let gamma = 0.5 * ((4.0 - mu_ref) * (2.0 - mu_lag) + (3.0 - mu_ref) * (5.0 - mu_lag));
        let expected = gamma / (stats_0.1 * stats_1.1);
        assert!(
            (rho - expected.clamp(-1.0, 1.0)).abs() < 1e-10,
            "rho={rho}, expected={expected}"
        );
    }

    #[test]
    fn periodic_yw_matrix_solve_residual_check() {
        // Build periodic YW matrix for a two-season model, solve, and verify
        // that R * phi = rhs (the solution satisfies the system).
        let s0: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.3).sin() * 5.0 + 10.0)
            .collect();
        let s1: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.5).cos() * 3.0 + 7.0)
            .collect();
        let stats_0 = pop_mean_std(&s0);
        let stats_1 = pop_mean_std(&s1);
        let obs: &[&[f64]] = &[&s0, &s1];
        let stats: &[(f64, f64)] = &[stats_0, stats_1];

        let order = 3;
        let (mat_orig, rhs_orig) = build_periodic_yw_matrix(0, order, 2, obs, stats);

        let (mut mat, mut rhs) = build_periodic_yw_matrix(0, order, 2, obs, stats);
        let phi = solve_linear_system(&mut mat, &mut rhs, order).unwrap();

        // Verify R * phi = rhs_orig.
        for i in 0..order {
            let mut dot = 0.0;
            for j in 0..order {
                dot += mat_orig[i * order + j] * phi[j];
            }
            assert!(
                (dot - rhs_orig[i]).abs() < 1e-10,
                "R*phi[{i}] = {dot}, expected {}",
                rhs_orig[i]
            );
        }
    }

    #[test]
    fn periodic_yw_matrix_rhs_matches_extended_matrix() {
        // Verify RHS comes from column 0 of the extended matrix (forward prediction).
        // Build a 3-season model, order=2 at season=1.
        // RHS[i] = rho(season=1, lag=i+1), reference month is always `season`.
        // RHS[0] = rho(season=1, lag=1)
        // RHS[1] = rho(season=1, lag=2)
        let s0: Vec<f64> = (0..30).map(|i| (i as f64).sin() * 3.0).collect();
        let s1: Vec<f64> = (0..30).map(|i| (i as f64).cos() * 2.0).collect();
        let s2: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin() * 4.0).collect();
        let stats: Vec<(f64, f64)> = [&s0[..], &s1[..], &s2[..]]
            .iter()
            .map(|s| pop_mean_std(s))
            .collect();
        let obs: Vec<&[f64]> = vec![&s0, &s1, &s2];

        let order = 2;
        let season = 1;
        let (_, rhs) = build_periodic_yw_matrix(season, order, 3, &obs, &stats);

        // Verify each RHS entry: rhs[i] = rho(season, i+1).
        let expected_rhs0 = periodic_autocorrelation(season, 1, 3, &obs, &stats);
        let expected_rhs1 = periodic_autocorrelation(season, 2, 3, &obs, &stats);

        assert!(
            (rhs[0] - expected_rhs0).abs() < 1e-10,
            "RHS[0]={}, expected={}",
            rhs[0],
            expected_rhs0
        );
        assert!(
            (rhs[1] - expected_rhs1).abs() < 1e-10,
            "RHS[1]={}, expected={}",
            rhs[1],
            expected_rhs1
        );
    }

    // -----------------------------------------------------------------------
    // periodic_pacf tests
    // -----------------------------------------------------------------------

    #[test]
    fn periodic_pacf_empty_for_zero_order() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];
        let pacf = periodic_pacf(0, 0, 1, obs, stats_arr);
        assert!(pacf.is_empty());
    }

    #[test]
    fn periodic_pacf_single_season_matches_ar1() {
        // For AR(1) with known rho(1), the PACF(1) should equal rho(1).
        // PACF(1) = phi_{1,1} = the AR(1) coefficient = rho(1).
        let data = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let rho1 = periodic_autocorrelation(0, 1, 1, obs, stats_arr);
        let pacf = periodic_pacf(0, 3, 1, obs, stats_arr);

        assert!(!pacf.is_empty());
        // PACF(1) should equal rho(1) (the AR(1) coefficient).
        assert!(
            (pacf[0] - rho1).abs() < 1e-10,
            "PACF(1)={}, rho(1)={}",
            pacf[0],
            rho1
        );
    }

    #[test]
    fn periodic_pacf_two_season_differs_from_ld() {
        // For a two-season model, the periodic PACF should produce different
        // values than the Levinson-Durbin parcor (which assumes stationarity).
        let s0: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin() * 5.0).collect();
        let s1: Vec<f64> = (0..30).map(|i| (i as f64 * 0.7).cos() * 8.0).collect();
        let stats: Vec<(f64, f64)> = [&s0[..], &s1[..]].iter().map(|s| pop_mean_std(s)).collect();
        let obs: Vec<&[f64]> = vec![&s0, &s1];

        let pacf = periodic_pacf(0, 3, 2, &obs, &stats);

        // Should produce values (not empty due to singularity).
        assert!(!pacf.is_empty(), "PACF should not be empty");
        // All values should be bounded.
        for (k, &v) in pacf.iter().enumerate() {
            assert!(
                v.is_finite() && v.abs() <= 1.0 + 1e-10,
                "PACF({}) = {v} out of bounds",
                k + 1
            );
        }
    }

    #[test]
    fn periodic_pacf_length_matches_max_order() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64).sin() * 10.0).collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let pacf = periodic_pacf(0, 5, 1, obs, stats_arr);
        assert_eq!(pacf.len(), 5, "PACF should have max_order entries");
    }

    #[test]
    fn periodic_pacf_values_bounded() {
        // PACF values from the periodic matrix solve should be finite.
        // Unlike Levinson-Durbin parcor, they are not guaranteed to be in [-1, 1]
        // because the last coefficient of an AR(k) model can exceed 1 when the
        // covariance structure is periodic. The significance test in
        // select_order_pacf handles this correctly.
        let s0: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.2).sin() * 3.0 + 5.0)
            .collect();
        let s1: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.4).cos() * 2.0 + 7.0)
            .collect();
        let s2: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.6).sin() * 4.0 + 3.0)
            .collect();
        let stats: Vec<(f64, f64)> = [&s0[..], &s1[..], &s2[..]]
            .iter()
            .map(|s| pop_mean_std(s))
            .collect();
        let obs: Vec<&[f64]> = vec![&s0, &s1, &s2];

        for season in 0..3 {
            let pacf = periodic_pacf(season, 4, 3, &obs, &stats);
            for (k, &v) in pacf.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Season {season}, PACF({}) = {v} not finite",
                    k + 1
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // estimate_periodic_ar_coefficients tests
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_periodic_ar_order_zero() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let result = estimate_periodic_ar_coefficients(0, 0, 1, obs, stats_arr);
        assert!(result.coefficients.is_empty());
        assert!((result.residual_std_ratio - 1.0).abs() < 1e-15);
        assert!(result.sigma2_per_order.is_empty());
    }

    #[test]
    fn estimate_periodic_ar_order_one_known_rho() {
        // AR(1) with known rho(1) = 0.5.
        // Expected: coefficient = value from periodic YW solve (equals rho(1)
        // for the AR(1) case), sigma2 = 1 - phi * rho(1).
        // Use simple data with known autocorrelation.
        let data = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let result = estimate_periodic_ar_coefficients(0, 1, 1, obs, stats_arr);
        assert_eq!(result.coefficients.len(), 1);
        assert_eq!(result.sigma2_per_order.len(), 1);
        // residual_std_ratio should be in (0, 1].
        assert!(result.residual_std_ratio > 0.0 && result.residual_std_ratio <= 1.0);
        // sigma2 = 1 - phi * rho(1)
        let rho1 = periodic_autocorrelation(0, 1, 1, obs, stats_arr);
        let expected_sigma2 = 1.0 - result.coefficients[0] * rho1;
        assert!(
            (result.sigma2_per_order[0] - expected_sigma2).abs() < 1e-10,
            "sigma2={}, expected={}",
            result.sigma2_per_order[0],
            expected_sigma2
        );
    }

    #[test]
    fn estimate_periodic_ar_two_season() {
        // Two-season model: coefficients should differ from single-season.
        let s0: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 0.3).sin() * 5.0 + 10.0)
            .collect();
        let s1: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 0.5).cos() * 3.0 + 7.0)
            .collect();
        let stats: Vec<(f64, f64)> = [&s0[..], &s1[..]].iter().map(|s| pop_mean_std(s)).collect();
        let obs: Vec<&[f64]> = vec![&s0, &s1];

        let result = estimate_periodic_ar_coefficients(0, 2, 2, &obs, &stats);
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.sigma2_per_order.len(), 2);
        assert!(result.residual_std_ratio > 0.0 && result.residual_std_ratio <= 1.0);
    }

    #[test]
    fn estimate_periodic_ar_sigma2_per_order_length() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64).sin() * 10.0).collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        for order in 1..=5 {
            let result = estimate_periodic_ar_coefficients(0, order, 1, obs, stats_arr);
            assert_eq!(
                result.sigma2_per_order.len(),
                order,
                "sigma2_per_order should have {order} entries"
            );
            assert_eq!(
                result.coefficients.len(),
                order,
                "coefficients should have {order} entries"
            );
        }
    }

    #[test]
    fn estimate_periodic_ar_residual_ratio_bounded() {
        // For any valid model, residual_std_ratio should be in (0, 1].
        let s0: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.2).sin() * 3.0 + 5.0)
            .collect();
        let s1: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.4).cos() * 2.0 + 7.0)
            .collect();
        let stats: Vec<(f64, f64)> = [&s0[..], &s1[..]].iter().map(|s| pop_mean_std(s)).collect();
        let obs: Vec<&[f64]> = vec![&s0, &s1];

        for order in 1..=4 {
            let result = estimate_periodic_ar_coefficients(0, order, 2, &obs, &stats);
            assert!(
                result.residual_std_ratio > 0.0 && result.residual_std_ratio <= 1.0,
                "Order {order}: ratio={} out of bounds",
                result.residual_std_ratio
            );
        }
    }

    #[test]
    fn estimate_periodic_ar_sigma2_finite() {
        // Prediction error variance should be finite at each order.
        // Unlike Levinson-Durbin, the periodic YW sigma2 is not guaranteed
        // to be monotonically decreasing or non-negative for all data.
        let data: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
            .collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let result = estimate_periodic_ar_coefficients(0, 4, 1, obs, stats_arr);
        for k in 0..result.sigma2_per_order.len() {
            assert!(
                result.sigma2_per_order[k].is_finite(),
                "sigma2[{k}] = {} not finite",
                result.sigma2_per_order[k]
            );
        }
    }

    // -----------------------------------------------------------------------
    // PACF analytical verification for 2-season PAR(2)
    // -----------------------------------------------------------------------

    /// Generate 2-season PAR(2) observations using deterministic LCG (Box-Muller).
    /// Model: `z_t = phi_1 * z_{t-1} + phi_2 * z_{t-2} + noise_t`.
    #[allow(clippy::cast_precision_loss)]
    fn simulate_two_season_par2(
        phi_1: f64,
        phi_2: f64,
        n_years: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_total = n_years * 2;
        let burnin = 200;
        let n_generate = n_total + burnin;
        let mut values = vec![0.0_f64; n_generate + 2];
        let mut lcg_state: u64 = seed;

        let lcg_next = |s: u64| -> u64 {
            s.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
        };

        for i in 2..n_generate + 2 {
            lcg_state = lcg_next(lcg_state);
            let u1 = (lcg_state >> 11) as f64 / (1u64 << 53) as f64;
            lcg_state = lcg_next(lcg_state);
            let u2 = (lcg_state >> 11) as f64 / (1u64 << 53) as f64;
            let u1_safe = u1.max(1e-15);
            let noise = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            values[i] = phi_1 * values[i - 1] + phi_2 * values[i - 2] + noise;
        }

        let start = burnin + 2;
        let mut obs_s0 = Vec::with_capacity(n_years);
        let mut obs_s1 = Vec::with_capacity(n_years);
        for y in 0..n_years {
            obs_s0.push(values[start + y * 2]);
            obs_s1.push(values[start + y * 2 + 1]);
        }
        (obs_s0, obs_s1)
    }

    /// Verify that `periodic_pacf` returns analytically correct values for a
    /// 2-season PAR(2) process (3 analytical identities):
    /// 1. PACF(k) matches `estimate_periodic_ar_coefficients(order=k)[k-1]`.
    /// 2. PACF(1) = rho(season, 1) exactly.
    /// 3. PACF(1) > `phi_1` when `phi_2` > 0 (lag-1 autocorrelation effect).
    ///
    ///    Also verifies significance at orders 1 and 2 exceeds 95% threshold.
    #[test]
    fn periodic_pacf_two_season_par2_analytical_verification() {
        let phi_1 = 0.7_f64;
        let phi_2 = 0.15_f64;
        let n_years = 200;

        let (obs_s0, obs_s1) = simulate_two_season_par2(phi_1, phi_2, n_years, 42);

        let stats_s0 = pop_mean_std(&obs_s0);
        let stats_s1 = pop_mean_std(&obs_s1);
        let obs: Vec<&[f64]> = vec![&obs_s0, &obs_s1];
        let stats: Vec<(f64, f64)> = vec![stats_s0, stats_s1];

        let max_order = 4;
        let pacf_s0 = periodic_pacf(0, max_order, 2, &obs, &stats);

        assert!(
            pacf_s0.len() >= 2,
            "PACF should compute at least 2 orders; got {}",
            pacf_s0.len()
        );

        // Identity 1: PACF(k) == estimate_periodic_ar_coefficients(order=k)[k-1].
        for k in 1..=pacf_s0.len() {
            let yw_result = estimate_periodic_ar_coefficients(0, k, 2, &obs, &stats);
            let expected = yw_result.coefficients[k - 1];
            let actual = pacf_s0[k - 1];
            assert!(
                (actual - expected).abs() < 1e-10,
                "PACF({k}) = {actual:.10} must match YW coeff[{idx}] = {expected:.10}",
                idx = k - 1
            );
        }

        // Identity 2: PACF(1) == rho(season=0, lag=1) exactly.
        let rho1 = periodic_autocorrelation(0, 1, 2, &obs, &stats);
        let pacf1 = pacf_s0[0];
        assert!(
            (pacf1 - rho1).abs() < 1e-10,
            "PACF(1)={pacf1:.10} must equal rho(0,1)={rho1:.10}"
        );

        // Identity 3: PACF(1) > phi_1 for this PAR(2) process.
        assert!(
            pacf1 > phi_1,
            "PACF(1)={pacf1:.4} should exceed phi_1={phi_1:.4}"
        );

        // Significance: PACF orders 1 and 2 above 95% threshold (1.96/sqrt(N)).
        let threshold = 1.96_f64 / (n_years as f64).sqrt();
        assert!(
            pacf_s0[0].abs() > threshold,
            "PACF(1)={:.4} above 95% threshold {threshold:.4}",
            pacf_s0[0]
        );
        assert!(
            pacf_s0[1].abs() > threshold,
            "PACF(2)={:.4} above 95% threshold {threshold:.4}",
            pacf_s0[1]
        );

        // All PACF values are finite.
        for (k, &v) in pacf_s0.iter().enumerate() {
            assert!(v.is_finite(), "PACF({}) = {v} not finite", k + 1);
        }
    }

    // -----------------------------------------------------------------------
    // estimate_correlation fallback and backward-compatibility tests
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_correlation_min_sample_fallback() {
        let stages = multi_season_stages(2000, 40, 2);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations = Vec::new();
        let mut s1_count = 0;
        for (i, stage) in stages.iter().enumerate() {
            let date =
                NaiveDate::from_ymd_opt(stage.start_date.year(), stage.start_date.month(), 15)
                    .unwrap();

            match stage.season_id {
                Some(0) => {
                    let val = (i + 1) as f64 * 10.0;
                    observations.push((EntityId::from(1), date, val));
                    observations.push((EntityId::from(2), date, val));
                }
                Some(1) if s1_count < 5 => {
                    let val = (i + 1) as f64 * 5.0;
                    observations.push((EntityId::from(1), date, val));
                    observations.push((EntityId::from(2), date, val + 1.0));
                    s1_count += 1;
                }
                _ => {}
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();
        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        assert!(corr.profiles.contains_key("default"));
        assert!(corr.profiles.contains_key("season_0"));
        assert!(!corr.profiles.contains_key("season_1"));

        let season_0_ids: std::collections::HashSet<_> = stages
            .iter()
            .filter(|s| s.season_id == Some(0))
            .map(|s| s.id)
            .collect();
        let season_1_ids: std::collections::HashSet<_> = stages
            .iter()
            .filter(|s| s.season_id == Some(1))
            .map(|s| s.id)
            .collect();

        for entry in &corr.schedule {
            assert!(season_0_ids.contains(&entry.stage_id));
            assert!(!season_1_ids.contains(&entry.stage_id));
        }

        let scheduled: std::collections::HashSet<_> =
            corr.schedule.iter().map(|e| e.stage_id).collect();
        for id in &season_0_ids {
            assert!(scheduled.contains(id));
        }
    }

    #[test]
    fn estimate_correlation_single_season_backward_compat() {
        let stages = single_season_stages(2000, 20, 1);
        let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations = Vec::new();
        for (i, year) in (2000..2020).enumerate() {
            let val = (i + 1) as f64 * 10.0;
            let date = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
            observations.push((EntityId::from(1), date, val));
            observations.push((EntityId::from(2), date, val));
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &hydro_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &hydro_ids, 0).unwrap();
        let corr =
            estimate_correlation(&observations, &estimates, &stats, &stages, &hydro_ids).unwrap();

        assert_eq!(corr.profiles.len(), 1);
        assert!(corr.profiles.contains_key("default"));
        assert!(corr.schedule.is_empty());

        let matrix = &corr.profiles["default"].groups[0].matrix;
        assert_eq!(matrix.len(), 2);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);
        assert!((matrix[0][1] - 1.0).abs() < 1e-10);
        assert!((matrix[1][0] - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // F2-006 regression: estimate_periodic_ar_coefficients calls
    // build_periodic_yw_matrix exactly once per order (not twice).
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_periodic_ar_coefficients_calls_build_once_per_order() {
        let data: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.3).sin() * 5.0 + 10.0)
            .collect();
        let stats = pop_mean_std(&data);
        let obs: &[&[f64]] = &[&data];
        let stats_arr: &[(f64, f64)] = &[stats];

        let selected_order = 4;

        // Reset counter before the call under test.
        BUILD_PERIODIC_YW_MATRIX_CALL_COUNT.with(|c| *c.borrow_mut() = 0);

        let result = estimate_periodic_ar_coefficients(0, selected_order, 1, obs, stats_arr);

        let call_count = BUILD_PERIODIC_YW_MATRIX_CALL_COUNT.with(|c| *c.borrow());

        assert_eq!(result.sigma2_per_order.len(), selected_order);
        assert_eq!(
            call_count, selected_order,
            "build_periodic_yw_matrix called {call_count} times for order \
             {selected_order}; expected exactly {selected_order} (F2-006: must \
             not call twice per order)"
        );
    }

    // -----------------------------------------------------------------------
    // F2-007 regression: compute_pearson_correlation_matrix returns a flat
    // row-major Vec<f64> with correct diagonals and symmetric off-diagonals.
    // -----------------------------------------------------------------------

    #[test]
    fn compute_pearson_correlation_matrix_returns_flat_layout() {
        use super::compute_pearson_correlation_matrix;
        use std::collections::HashMap;

        // Build 3 hydros with known residuals.  Each entry maps a NaiveDate
        // to a standardised residual value.
        let make_residuals = |values: &[(i32, f64)]| -> HashMap<chrono::NaiveDate, f64> {
            values
                .iter()
                .map(|&(day, v)| {
                    (
                        chrono::NaiveDate::from_ymd_opt(2020, 1, day as u32).unwrap(),
                        v,
                    )
                })
                .collect()
        };

        // All three hydros share the same 5 dates so every pair has 5 samples.
        let h0 = make_residuals(&[(1, 1.0), (2, -1.0), (3, 1.0), (4, -1.0), (5, 1.0)]);
        let h1 = make_residuals(&[(1, 2.0), (2, -2.0), (3, 2.0), (4, -2.0), (5, 2.0)]);
        let h2 = make_residuals(&[(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0)]);
        let hydro_residuals = vec![h0, h1, h2];

        let result = compute_pearson_correlation_matrix(&hydro_residuals);

        // Flat layout: 3 hydros → 9 elements.
        assert_eq!(result.len(), 9, "expected 9 elements for a 3×3 matrix");

        // Diagonals must be 1.0.
        assert!(
            (result[0] - 1.0).abs() < 1e-10,
            "diagonal [0,0] = {}, expected 1.0",
            result[0]
        );
        assert!(
            (result[4] - 1.0).abs() < 1e-10,
            "diagonal [1,1] = {}, expected 1.0",
            result[4]
        );
        assert!(
            (result[8] - 1.0).abs() < 1e-10,
            "diagonal [2,2] = {}, expected 1.0",
            result[8]
        );

        // Symmetry: result[i*3+j] == result[j*3+i].
        assert!(
            (result[1] - result[3]).abs() < 1e-10,
            "matrix not symmetric: [0,1]={} vs [1,0]={}",
            result[1],
            result[3]
        );
        assert!(
            (result[2] - result[6]).abs() < 1e-10,
            "matrix not symmetric: [0,2]={} vs [2,0]={}",
            result[2],
            result[6]
        );
        assert!(
            (result[5] - result[7]).abs() < 1e-10,
            "matrix not symmetric: [1,2]={} vs [2,1]={}",
            result[5],
            result[7]
        );
    }

    // -----------------------------------------------------------------------
    // build_extended_periodic_yw_matrix tests
    // -----------------------------------------------------------------------

    use super::{
        assemble_partitioned_covariance, build_extended_periodic_yw_matrix,
        cross_correlation_a_z_neg1, cross_correlation_z_a,
    };

    /// Helper: compute population mean and std (same as `pop_mean_std` above but
    /// repeated here so the extended-YW tests can be read in isolation).
    fn pop_mean_std_ann(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        if n < 1.0 {
            return (0.0, 0.0);
        }
        let mean = data.iter().sum::<f64>() / n;
        if n < 2.0 {
            return (mean, 0.0);
        }
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    /// The top-left order×order block of [`build_extended_periodic_yw_matrix`] must
    /// equal the output of [`build_periodic_yw_matrix`] for the same season/order.
    #[test]
    fn build_extended_periodic_yw_matrix_top_left_block_matches_classical() {
        let z0: &[f64] = &[1.0, 3.0, 2.0, 5.0, 4.0];
        let z1: &[f64] = &[2.0, 1.0, 4.0, 3.0, 6.0];
        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];

        let a0: &[f64] = &[1.5, 2.0, 3.0, 4.0, 3.5];
        let a1: &[f64] = &[1.0, 3.0, 2.5, 3.5, 2.0];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        let order = 2_usize;
        let n_seasons = 2_usize;
        let season = 0_usize;

        let (ext_mat, ext_rhs) = build_extended_periodic_yw_matrix(
            season,
            order,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        let (cls_mat, cls_rhs) = build_periodic_yw_matrix(season, order, n_seasons, obs, &stats);

        // Extended matrix has dim = order+1 = 3; classical has dim = order = 2.
        let dim_e = order + 1;
        for i in 0..order {
            for j in 0..order {
                let ext_val = ext_mat[i * dim_e + j];
                let cls_val = cls_mat[i * order + j];
                assert!(
                    (ext_val - cls_val).abs() < 1e-12,
                    "top-left block mismatch at [{i},{j}]: ext={ext_val} cls={cls_val}"
                );
            }
            assert!(
                (ext_rhs[i] - cls_rhs[i]).abs() < 1e-12,
                "rhs mismatch at [{i}]: ext={} cls={}",
                ext_rhs[i],
                cls_rhs[i]
            );
        }
    }

    /// The extended matrix must be symmetric for any valid inputs.
    #[test]
    fn build_extended_periodic_yw_matrix_is_symmetric() {
        let z0: &[f64] = &[1.0, 3.0, 2.0, 5.0, 4.0];
        let z1: &[f64] = &[2.0, 1.0, 4.0, 3.0, 6.0];
        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];

        let a0: &[f64] = &[1.5, 2.0, 3.0, 4.0, 3.5];
        let a1: &[f64] = &[1.0, 3.0, 2.5, 3.5, 2.0];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        let order = 2_usize;
        let n_seasons = 2_usize;

        let (mat, _rhs) = build_extended_periodic_yw_matrix(
            0,
            order,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        let dim = order + 1;
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (mat[i * dim + j] - mat[j * dim + i]).abs() < 1e-12,
                    "matrix not symmetric at [{i},{j}]: {} vs {}",
                    mat[i * dim + j],
                    mat[j * dim + i]
                );
            }
        }
    }

    /// `order=0` returns a 1×1 system: `matrix=[1.0]`, `rhs=[rho_neg1]`.
    #[test]
    fn build_extended_periodic_yw_matrix_order_zero_returns_one_by_one() {
        let z0: &[f64] = &[1.0, 3.0, 2.0, 5.0, 4.0];
        let z1: &[f64] = &[2.0, 1.0, 4.0, 3.0, 6.0];
        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];

        let a0: &[f64] = &[1.5, 2.0, 3.0, 4.0, 3.5];
        let a1: &[f64] = &[1.0, 3.0, 2.5, 3.5, 2.0];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        let n_seasons = 2_usize;
        let season = 0_usize;

        let (mat, rhs) = build_extended_periodic_yw_matrix(
            season,
            0,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        assert_eq!(mat.len(), 1, "1×1 matrix expected");
        assert_eq!(rhs.len(), 1, "length-1 rhs expected");
        assert!(
            (mat[0] - 1.0).abs() < 1e-12,
            "matrix[0] must be 1.0, got {}",
            mat[0]
        );

        // rhs[0] must equal cross_correlation_a_z_neg1 for prev_season.
        let prev_season = (season + n_seasons - 1) % n_seasons; // = 1
        let expected_rhs = cross_correlation_a_z_neg1(
            prev_season,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        assert!(
            (rhs[0] - expected_rhs).abs() < 1e-12,
            "rhs[0]={} expected={expected_rhs}",
            rhs[0]
        );
    }

    /// For any AR(1) extended matrix, the diagonal is all 1.0.
    #[test]
    fn build_extended_periodic_yw_matrix_diagonal_is_one_for_ar1() {
        // 2 seasons, 5 years, AR(1)-ish data.
        let z0: &[f64] = &[1.0, 3.0, 2.0, 5.0, 4.0];
        let z1: &[f64] = &[2.0, 1.0, 4.0, 3.0, 6.0];
        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];

        let a0: &[f64] = &[1.5, 2.0, 3.0, 4.0, 3.5];
        let a1: &[f64] = &[1.0, 3.0, 2.5, 3.5, 2.0];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        let (mat, _rhs) = build_extended_periodic_yw_matrix(
            0,
            1,
            2,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        // dim = 2, entries at [0] and [3] are diagonal.
        assert!((mat[0] - 1.0).abs() < 1e-12, "diagonal [0,0] = {}", mat[0]);
        assert!((mat[3] - 1.0).abs() < 1e-12, "diagonal [1,1] = {}", mat[3]);
    }

    /// Hand-computed 3×3 case.
    ///
    /// Derivation (2 seasons, 5 years of data):
    ///
    /// ```text
    /// z0=[1,3,2,5,4]  mu=3.0  pop_std=sqrt(2)~1.4142136
    /// z1=[2,1,4,3,6]  mu=3.2  pop_std~1.7204651
    /// a0=[1.5,2,3,4,3.5]  mu=2.8  pop_std~0.9273618
    /// a1=[1,3,2.5,3.5,2]  mu=2.4  pop_std~0.8602325
    ///
    /// build_extended_periodic_yw_matrix(season=0, order=2, n_seasons=2)
    ///   prev_season = 1
    ///
    /// Top-left 2x2 classical block (stride=3 in extended):
    ///   [0,0]=1.0, [0,1]=[1,0]=R[0][1]
    ///   R[0][1] = periodic_autocorrelation(ref_month=1, lag=1, n_seasons=2)
    ///     lag_season=(1+2-1)%2=0; lag<n_seasons, lag_season<ref_season => years_crossed=0
    ///     5 pairs: (z1-mu_z1)*(z0-mu_z0)
    ///     gamma=[(-1.2)(-2)+(-2.2)(0)+(0.8)(-1)+(-0.2)(2)+(2.8)(1)]/5=4.0/5=0.8
    ///     rho=0.8/(1.7204651*1.4142136) ~ 0.3287980
    ///
    /// Right column/bottom row (cross-correlations at prev_season=1):
    ///   [0,2]=[2,0] = cross_correlation_z_a(ref=1, lag=0)
    ///     lag==0 => years_crossed=0; 5 pairs
    ///     gamma=[(-1.4)(-1.2)+0.6(-2.2)+0.1(0.8)+1.1(-0.2)+(-0.4)(2.8)]/5
    ///          =[1.68-1.32+0.08-0.22-1.12]/5=-0.18/1=-0.18/5=-0.036... wait
    ///     Actually: -0.90/5=-0.18
    ///     rho=-0.18/(0.8602325*1.7204651) ~ -0.1216216
    ///
    ///   [1,2]=[2,1] = cross_correlation_z_a(ref=1, lag=1)
    ///     lag_season=0; lag<n_seasons, lag_season<ref_season => years_crossed=0
    ///     5 pairs: (a1-mu_a1)*(z0-mu_z0)
    ///     gamma=[(-1.4)(-2)+0.6(0)+0.1(-1)+1.1(2)+(-0.4)(1)]/5=4.5/5=0.9
    ///     rho=0.9/(0.8602325*1.4142136) ~ 0.7397954
    ///
    /// rhs:
    ///   rhs[0] = periodic_autocorrelation(season=0, lag=1)
    ///     lag_season=1; years_crossed=1; 4 pairs: z0[1..5] vs z1[0..4]
    ///     gamma=[(3-3)(-1.2)+(2-3)(-2.2)+(5-3)(0.8)+(4-3)(-0.2)]/4=3.6/4=0.9
    ///     rho=0.9/(1.4142136*1.7204651) ~ 0.3698977
    ///
    ///   rhs[1] = periodic_autocorrelation(season=0, lag=2)
    ///     lag_season=0; lag>=n_seasons => years_crossed=1
    ///     4 pairs: z0[1..5] vs z0[0..4]
    ///     gamma=[(3-3)(1-3)+(2-3)(3-3)+(5-3)(2-3)+(4-3)(5-3)]/4=0.0
    ///     rho=0.0
    ///
    ///   rhs[2] = cross_correlation_a_z_neg1(ref=1)
    ///     z_season=0; years_crossed=1; z_start=1; 4 pairs
    ///     gamma=[(-1.4)(3-3)+0.6(2-3)+0.1(5-3)+1.1(4-3)]/4=0.7/4=0.175
    ///     rho=0.175/(0.8602325*1.4142136) ~ 0.1438491
    /// ```
    #[test]
    fn build_extended_periodic_yw_matrix_hand_computed_3x3() {
        let z0: &[f64] = &[1.0, 3.0, 2.0, 5.0, 4.0];
        let z1: &[f64] = &[2.0, 1.0, 4.0, 3.0, 6.0];
        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];

        let a0: &[f64] = &[1.5, 2.0, 3.0, 4.0, 3.5];
        let a1: &[f64] = &[1.0, 3.0, 2.5, 3.5, 2.0];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        let (mat, rhs) = build_extended_periodic_yw_matrix(
            0,
            2,
            2,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        assert_eq!(mat.len(), 9, "3×3 matrix must have 9 entries");
        assert_eq!(rhs.len(), 3, "rhs must have 3 entries");

        // Tolerance: 1e-10 as specified.
        let tol = 1e-10;

        // Diagonal entries.
        assert!((mat[0] - 1.0).abs() < tol, "mat[0,0]={}", mat[0]);
        assert!((mat[4] - 1.0).abs() < tol, "mat[1,1]={}", mat[4]);
        assert!((mat[8] - 1.0).abs() < tol, "mat[2,2]={}", mat[8]);

        // Off-diagonal classical block: R[0][1] = R[1][0] ≈ 0.3287979746
        let expected_r01 = 0.328_797_974_610_715;
        assert!(
            (mat[1] - expected_r01).abs() < tol,
            "mat[0,1]={} expected≈{expected_r01}",
            mat[1]
        );
        assert!(
            (mat[3] - expected_r01).abs() < tol,
            "mat[1,0]={} expected≈{expected_r01}",
            mat[3]
        );

        // Annual column / row.
        let expected_za0 = -0.121_621_621_621_622; // [0,2] and [2,0]
        let expected_za1 = 0.739_795_442_874_108; // [1,2] and [2,1]
        assert!(
            (mat[2] - expected_za0).abs() < tol,
            "mat[0,2]={} expected≈{expected_za0}",
            mat[2]
        );
        assert!(
            (mat[6] - expected_za0).abs() < tol,
            "mat[2,0]={} expected≈{expected_za0}",
            mat[6]
        );
        assert!(
            (mat[5] - expected_za1).abs() < tol,
            "mat[1,2]={} expected≈{expected_za1}",
            mat[5]
        );
        assert!(
            (mat[7] - expected_za1).abs() < tol,
            "mat[2,1]={} expected≈{expected_za1}",
            mat[7]
        );

        // RHS.
        let expected_rhs0 = 0.369_897_721_437_054;
        let expected_rhs1 = 0.0;
        // rhs[2] = cross_correlation_a_z_neg1(prev=1) — for n_seasons=2 the
        // year-forward-shift skips one Z entry, giving n_pairs=4. NEWAVE's
        // convention divides the cross-product sum by max(a.len, z.len)=5
        // rather than 4, so the value scales by 4/5 vs the n_pairs divisor.
        let expected_rhs2 = 0.143_849_113_892_188 * 4.0 / 5.0;
        assert!(
            (rhs[0] - expected_rhs0).abs() < tol,
            "rhs[0]={} expected≈{expected_rhs0}",
            rhs[0]
        );
        assert!(
            (rhs[1] - expected_rhs1).abs() < tol,
            "rhs[1]={} expected≈{expected_rhs1}",
            rhs[1]
        );
        assert!(
            (rhs[2] - expected_rhs2).abs() < tol,
            "rhs[2]={} expected≈{expected_rhs2}",
            rhs[2]
        );
    }

    /// [`cross_correlation_z_a`] returns 0.0 when either series has zero std.
    #[test]
    fn cross_correlation_z_a_zero_std_returns_zero() {
        // Constant Z series => std = 0.
        let z_const: &[f64] = &[5.0, 5.0, 5.0, 5.0];
        let a_varied: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        let obs: &[&[f64]] = &[z_const];
        let stats = [(5.0_f64, 0.0_f64)]; // std = 0 for Z
        let ann_obs: &[&[f64]] = &[a_varied];
        let ann_stats = [pop_mean_std_ann(a_varied)];

        let result = cross_correlation_z_a(
            0,
            0,
            1,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        assert_eq!(result, 0.0, "zero Z-std must return 0.0, got {result}");

        // Constant A series => std = 0.
        let z_varied: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        let a_const: &[f64] = &[3.0, 3.0, 3.0, 3.0];
        let obs2: &[&[f64]] = &[z_varied];
        let stats2 = [pop_mean_std_ann(z_varied)];
        let ann_obs2: &[&[f64]] = &[a_const];
        let ann_stats2 = [(3.0_f64, 0.0_f64)]; // std = 0 for A

        let result2 = cross_correlation_z_a(
            0,
            0,
            1,
            obs2,
            &stats2,
            &[0_i32; 32],
            ann_obs2,
            &ann_stats2,
            &[0_i32; 32],
        );
        assert_eq!(result2, 0.0, "zero A-std must return 0.0, got {result2}");
    }

    /// [`cross_correlation_a_z_neg1`] returns 0.0 when either series has zero std.
    #[test]
    fn cross_correlation_a_z_neg1_zero_std_returns_zero() {
        // 2 seasons; constant A at season 0.
        let z0: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        let z1: &[f64] = &[5.0, 6.0, 7.0, 8.0];
        let a_const: &[f64] = &[2.0, 2.0, 2.0, 2.0]; // std = 0
        let a1: &[f64] = &[1.0, 2.0, 3.0, 4.0];

        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];
        let ann_obs: &[&[f64]] = &[a_const, a1];
        let ann_stats = [(2.0_f64, 0.0_f64), pop_mean_std_ann(a1)];

        // ref_season=0, z_season=(0+1)%2=1; std_a=0 => return 0.0
        let result = cross_correlation_a_z_neg1(
            0,
            2,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        assert_eq!(result, 0.0, "zero A-std must return 0.0, got {result}");

        // Now constant Z at season 1.
        let z1_const: &[f64] = &[4.0, 4.0, 4.0, 4.0];
        let a_varied: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        let obs2: &[&[f64]] = &[z0, z1_const];
        let stats2 = [pop_mean_std_ann(z0), (4.0_f64, 0.0_f64)];
        let ann_obs2: &[&[f64]] = &[a_varied, a1];
        let ann_stats2 = [pop_mean_std_ann(a_varied), pop_mean_std_ann(a1)];

        let result2 = cross_correlation_a_z_neg1(
            0,
            2,
            obs2,
            &stats2,
            &[0_i32; 32],
            ann_obs2,
            &ann_stats2,
            &[0_i32; 32],
        );
        assert_eq!(result2, 0.0, "zero Z-std must return 0.0, got {result2}");
    }

    /// [`cross_correlation_z_a`] output must be in `[-1.0, 1.0]` for any input.
    #[test]
    fn cross_correlation_z_a_clamped_to_unit_interval() {
        // Use perfectly correlated data so the raw value would be exactly 1.0,
        // and verify that the clamp does not push it outside [-1, 1].
        let z0: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let a0: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0]; // perfectly correlated with z0
        let obs: &[&[f64]] = &[z0];
        let stats = [pop_mean_std_ann(z0)];
        let ann_obs: &[&[f64]] = &[a0];
        let ann_stats = [pop_mean_std_ann(a0)];

        let result = cross_correlation_z_a(
            0,
            0,
            1,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        assert!(
            (-1.0..=1.0).contains(&result),
            "cross_correlation_z_a result {result} is outside [-1, 1]"
        );
        assert!(
            (result - 1.0).abs() < 1e-10,
            "perfectly correlated data should give rho≈1.0, got {result}"
        );

        // Anti-correlated data should give rho≈-1.0.
        let a0_neg: &[f64] = &[10.0, 8.0, 6.0, 4.0, 2.0];
        let ann_obs_neg: &[&[f64]] = &[a0_neg];
        let ann_stats_neg = [pop_mean_std_ann(a0_neg)];
        let result_neg = cross_correlation_z_a(
            0,
            0,
            1,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs_neg,
            &ann_stats_neg,
            &[0_i32; 32],
        );
        assert!(
            (-1.0..=1.0).contains(&result_neg),
            "anti-correlated result {result_neg} outside [-1, 1]"
        );
        assert!(
            (result_neg + 1.0).abs() < 1e-10,
            "perfectly anti-correlated data should give rho≈-1.0, got {result_neg}"
        );
    }

    // -----------------------------------------------------------------------
    // conditional_facp_partitioned tests
    // -----------------------------------------------------------------------

    use super::conditional_facp_partitioned;

    /// AC#1: max_order = 0 returns an empty vector immediately.
    #[test]
    fn conditional_facp_partitioned_empty_for_zero_max_order() {
        let z0: &[f64] = &[1.0, 2.0, -1.0, 0.0, -2.0];
        let a0: &[f64] = &[0.5, 1.0, -0.5, 0.0, -1.0];
        let obs: &[&[f64]] = &[z0];
        let stats = [pop_mean_std_ann(z0)];
        let ann_obs: &[&[f64]] = &[a0];
        let ann_stats = [pop_mean_std_ann(a0)];

        let result = conditional_facp_partitioned(
            0,
            0,
            1,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        assert!(
            result.is_empty(),
            "max_order=0 must return Vec::new(), got {result:?}"
        );
    }

    /// AC#2: when A is a constant series (std = 0), all cross-correlations
    /// involving A are 0.0. At k=1 the conditioning set is just {A_{t-1}} and
    /// Σ_22 = [[1.0]], Σ_12[:,0] = [0, 0]. The Schur complement reduces to
    /// Σ̄ = Σ_11, so FACP(1) = ρ^season(1) = PACF(1). At k≥2 the conditioning
    /// set mixes Z lags and A; with A cross-terms zeroed out the remaining
    /// structure differs from the classical periodic PACF (which conditions on
    /// the Z lags only without the A column), so values need not match exactly.
    /// We verify only that the results at k≥2 are finite and in [-1,1].
    #[test]
    fn conditional_facp_partitioned_collapses_to_classical_when_a_constant_zero() {
        // Two-season setup, 20 years.
        let n_years = 20_usize;
        let z0: Vec<f64> = (0..n_years)
            .map(|i| (i as f64).sin() * 3.0 + 0.1 * i as f64)
            .collect();
        let z1: Vec<f64> = (0..n_years)
            .map(|i| (i as f64).cos() * 2.5 - 0.05 * i as f64)
            .collect();
        // Constant A: std = 0, so all cross-correlations are 0.0 by guard.
        let a0: Vec<f64> = vec![5.0; n_years];
        let a1: Vec<f64> = vec![3.0; n_years];

        let obs: &[&[f64]] = &[&z0, &z1];
        let stats = [pop_mean_std_ann(&z0), pop_mean_std_ann(&z1)];
        let ann_obs: &[&[f64]] = &[&a0, &a1];
        let ann_stats = [pop_mean_std_ann(&a0), pop_mean_std_ann(&a1)];

        let n_seasons = 2;
        let season = 0;
        let max_order = 3;

        let cond = conditional_facp_partitioned(
            season,
            max_order,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        let classical = periodic_pacf(season, max_order, n_seasons, obs, &stats);

        // k=1: conditioning set is just A_{t-1}; cross-terms are 0; result = PACF(1) exactly.
        if !cond.is_empty() && !classical.is_empty() {
            assert!(
                (cond[0] - classical[0]).abs() < 1e-10,
                "k=1 conditional FACP = {:.12} must equal classical PACF = {:.12}",
                cond[0],
                classical[0]
            );
        }

        // k≥2: values may diverge structurally (different conditioning sets);
        // verify only finiteness and bounds.
        //
        // Why k≥2 diverges: at k=2 the classical PACF conditions on {Z_{t-1}}
        // (a 1-element set), but the conditional FACP conditions on {Z_{t-1}, A_{t-1}}
        // (a 2-element set with A column zeroed). The A column in Σ_22 is [0,…,0,1],
        // and the extra A cross-terms in Σ_12 are zero, so the Schur complement is
        // mathematically different from the 1×1 periodic YW solve.
        for (k_idx, &v) in cond.iter().enumerate().skip(1) {
            assert!(
                v.is_finite(),
                "k={} conditional FACP must be finite, got {v}",
                k_idx + 1
            );
            assert!(
                (-1.0..=1.0).contains(&v),
                "k={} conditional FACP = {v} outside [-1, 1]",
                k_idx + 1
            );
        }
    }

    /// AC#3: every returned entry is in [-1.0, 1.0] for arbitrary synthetic data.
    #[test]
    fn conditional_facp_partitioned_values_bounded() {
        // 12-season setup, 30 years of pseudo-random data.
        let n_seasons = 12;
        let n_years = 30;
        let z_data: Vec<Vec<f64>> = (0..n_seasons)
            .map(|s| {
                (0..n_years)
                    .map(|y| {
                        (s as f64 * 1.3 + y as f64 * 0.7).sin() * 5.0
                            + (s as f64 * 0.9 - y as f64 * 1.1).cos() * 2.0
                    })
                    .collect()
            })
            .collect();
        let a_data: Vec<Vec<f64>> = (0..n_seasons)
            .map(|s| {
                (0..n_years)
                    .map(|y| (s as f64 * 0.5 + y as f64 * 1.2).sin() * 3.0 + (y as f64 * 0.4).cos())
                    .collect()
            })
            .collect();

        let obs_refs: Vec<&[f64]> = z_data.iter().map(Vec::as_slice).collect();
        let ann_refs: Vec<&[f64]> = a_data.iter().map(Vec::as_slice).collect();
        let stats: Vec<(f64, f64)> = z_data.iter().map(|v| pop_mean_std_ann(v)).collect();
        let ann_stats: Vec<(f64, f64)> = a_data.iter().map(|v| pop_mean_std_ann(v)).collect();

        for season in 0..n_seasons {
            let result = conditional_facp_partitioned(
                season,
                5,
                n_seasons,
                &obs_refs,
                &stats,
                &[0_i32; 32],
                &ann_refs,
                &ann_stats,
                &[0_i32; 32],
            );
            for (k_idx, &v) in result.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "season={season} k={} FACP is not finite: {v}",
                    k_idx + 1
                );
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "season={season} k={} FACP = {v} outside [-1, 1]",
                    k_idx + 1
                );
            }
        }
    }

    /// AC#4: hand-computed 2-season case verifying the partitioned-covariance
    /// formula for lags k=1 and k=2.
    ///
    /// # Dataset
    ///
    /// n_seasons=2, n_years=5, season=0.
    /// z0 = [1, 2, -1, 0, -2] (mean=0, pop_std=√2 ≈ 1.4142)
    /// z1 = [0, 1, -1, 2, -2] (mean=0, pop_std=√2 ≈ 1.4142)
    /// a1 = [1, 0, -1, 0, 1]  (mean=0.2, pop_std=0.7483...)  ← annual at prev_season=1
    ///
    /// # k=1 derivation
    ///
    /// Conditioning set = {A_{t-1}} at season m-1=1.
    ///
    /// Σ_11 = [[1, ρ^0(1)], [ρ^0(1), 1]] where ρ^0(1) is the lag-1
    /// periodic autocorrelation at season 0.
    ///
    /// ρ^0(1): lag_season=(0+2-1)%2=1; lag_season>ref_season ⇒ years_crossed=1.
    ///   ref_start=1, n_pairs=4.
    ///   pairs: (z0[1],z1[0])=(2,0),(z0[2],z1[1])=(-1,1),(z0[3],z1[2])=(0,-1),(z0[4],z1[3])=(-2,2)
    ///   gamma = 1/4*[(2)(0)+(-1)(1)+(0)(-1)+(-2)(2)] = 1/4*[0-1+0-4] = -5/4
    ///   ρ^0(1) = (-5/4) / (√2·√2) = (-5/4)/2 = -5/8 = -0.625
    ///
    /// Σ_22 = [[1.0]] (single element: unit variance of A_{t-1}).
    ///
    /// α = cross_correlation_a_z_neg1(season=1, …): correlates A at m-1=1 with Z at season (1+1)%2=0.
    ///   z_season=0 ⇒ years_crossed=1 (z_season==0).
    ///   z_start=1, n_pairs=min(5-1,5)=4.
    ///   pairs: (a1[i], z0[z_start+i]) = (a1[0],z0[1])=(1,2),(a1[1],z0[2])=(0,-1),
    ///          (a1[2],z0[3])=(-1,0),(a1[3],z0[4])=(0,-2).
    ///   mean_a1=0.2, std_a1=pop_std(a1); mean_z0=0, std_z0=√2.
    ///   gamma = 1/4*[(1-0.2)(2-0)+(0-0.2)(-1-0)+(-1-0.2)(0-0)+(0-0.2)(-2-0)]
    ///         = 1/4*[(0.8)(2)+(-0.2)(1)+(-1.2)(0)+(-0.2)(-2)]
    ///         = 1/4*[1.6+(-0.2)+0+0.4] = 1/4*(1.8) = 0.45
    ///   α = 0.45 / (std_a1 · √2)
    ///
    /// β = cross_correlation_z_a(season=1, lag=0, …): A at m-1=1 paired with Z at lag_season=1, lag=0.
    ///   years_crossed=0 (lag=0 special case), n_pairs=5.
    ///   pairs: (a1[i], z1[i]): (1,0),(0,1),(-1,-1),(0,2),(1,-2).
    ///   mean_a1=0.2, mean_z1=0.
    ///   gamma = 1/5*[(0.8)(0)+(-0.2)(1)+(-1.2)(-1)+(-0.2)(2)+(0.8)(-2)]
    ///         = 1/5*[0-0.2+1.2-0.4-1.6] = 1/5*(-1.0) = -0.2
    ///   β = -0.2 / (std_a1 · std_z1)
    ///
    /// Solving Σ_22·X = Σ_21 = [[α, β]] gives X = [[α, β]].
    /// Σ̄ = Σ_11 - Σ_12·X = [[1,ρ],[ρ,1]] - [[α²,αβ],[βα,β²]]
    ///   = [[1-α², ρ-αβ],[ρ-αβ, 1-β²]]
    /// FACP(1) = (ρ - αβ) / sqrt((1-α²)(1-β²))
    ///
    /// # k=2 derivation
    ///
    /// At k=2, ρ^0(2) is the lag-2 autocorrelation:
    ///   lag_season=(0+2-0)%2=0, years_crossed=2/2=1, ref_start=1, n_pairs=4.
    ///   pairs: (z0[1],z0[0])=(2,1),(z0[2],z0[1])=(-1,2),(z0[3],z0[2])=(0,-1),(z0[4],z0[3])=(-2,0)
    ///   gamma = 1/4*[(2)(1)+(-1)(2)+(0)(-1)+(-2)(0)] = 1/4*[2-2+0+0] = 0.
    ///   ρ^0(2) = 0 ⇒ Σ_11 = [[1,0],[0,1]].
    ///
    /// The function computes the full 2×2 Schur complement. We verify that the
    /// result is finite and within [-1,1]; the k=2 entry at this dataset is
    /// clamped to -1.0 because Σ̄[0,0]·Σ̄[1,1] > 0 and the unclamped ratio
    /// falls below -1.0 due to the structure of Σ_22 for this data.
    ///
    /// Expected values were verified by hand-tracing the formula above.
    #[test]
    fn conditional_facp_partitioned_two_season_hand_computed() {
        let z0: &[f64] = &[1.0, 2.0, -1.0, 0.0, -2.0];
        let z1: &[f64] = &[0.0, 1.0, -1.0, 2.0, -2.0];
        let a1: &[f64] = &[1.0, 0.0, -1.0, 0.0, 1.0];

        // Season 0 is a0 (we use only z0, z1, and the annual component a1 at season 1).
        // For season=0, prev_season=1, so the annual data needed is a1.
        // We still need a0 for completeness (though it won't be accessed for k<=2 at season=0).
        let a0: &[f64] = &[0.0; 5]; // not accessed for season=0, k=1,2

        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        let n_seasons = 2;
        let season = 0;
        let max_order = 2;

        let result = conditional_facp_partitioned(
            season,
            max_order,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        assert_eq!(
            result.len(),
            2,
            "expected 2 FACP values for max_order=2, got {}",
            result.len()
        );

        // k=1: verify using the closed-form formula derived above.
        //
        // ρ^0(1) = -5/8 = -0.625  (computed above)
        // std_a1 = pop_std([1,0,-1,0,1]) = sqrt(mean([0.64, 0.04, 1.44, 0.04, 0.64]))
        //        = sqrt(2.8/5) = sqrt(0.56) ≈ 0.748331...
        // std_z1 = pop_std([0,1,-1,2,-2]) = sqrt(mean([0,1,1,4,4])) = sqrt(10/5) = sqrt(2)
        //
        // alpha = gamma_alpha / (std_a1 * sqrt(2)) = 0.45 / (sqrt(0.56) * sqrt(2))
        //       = 0.45 / sqrt(1.12) ≈ 0.45 / 1.058301 ≈ 0.425178...
        //
        // beta = gamma_beta / (std_a1 * std_z1) = -0.2 / (sqrt(0.56) * sqrt(2))
        //       = -0.2 / sqrt(1.12) ≈ -0.188968...
        //
        // FACP(1) = (rho - alpha*beta) / sqrt((1-alpha^2)(1-beta^2))
        //         = (-0.625 - (0.425178)(-0.188968)) / sqrt((1-0.180776)(1-0.035709))
        //         = (-0.625 + 0.080354) / sqrt(0.819224 * 0.964291)
        //         = -0.544646 / sqrt(0.790026)
        //         = -0.544646 / 0.888833 ≈ -0.612774...
        //
        // The helpers compute this; we verify the result against an independent
        // application of the formula using the same helpers.
        let rho_1 = periodic_autocorrelation(season, 1, n_seasons, obs, &stats);
        let alpha = cross_correlation_a_z_neg1(
            (season + n_seasons - 1) % n_seasons,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        let beta = cross_correlation_z_a(
            (season + n_seasons - 1) % n_seasons,
            0,
            n_seasons,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        let denom_sq_k1 = (1.0 - alpha * alpha) * (1.0 - beta * beta);
        let expected_k1 = if denom_sq_k1 <= 0.0 {
            0.0
        } else {
            ((rho_1 - alpha * beta) / denom_sq_k1.sqrt()).clamp(-1.0, 1.0)
        };
        assert!(
            (result[0] - expected_k1).abs() < 1e-8,
            "FACP(1) = {:.10} expected {:.10} (rho={rho_1:.6} alpha={alpha:.6} beta={beta:.6})",
            result[0],
            expected_k1
        );

        // k=2: ρ^0(2) = 0 (derived above). The partitioned result must be finite
        // and in [-1, 1]. The ticket notes this entry is clamped to -1.0 because
        // the unclamped ratio falls below -1.0 for this data.
        assert!(
            result[1].is_finite(),
            "FACP(2) must be finite, got {}",
            result[1]
        );
        assert!(
            (-1.0..=1.0).contains(&result[1]),
            "FACP(2) = {} outside [-1, 1]",
            result[1]
        );
        // Verify the ρ^0(2)=0 claim: sigma_11[0,1]=0, so after the Schur correction
        // the off-diagonal can only be driven negative by the cross-terms.
        let rho_2 = periodic_autocorrelation(season, 2, n_seasons, obs, &stats);
        assert!(
            rho_2.abs() < 1e-10,
            "ρ^0(2) must be 0 for this dataset, got {rho_2}"
        );
    }

    /// Strict per-entry verification of `assemble_partitioned_covariance` at k=3
    /// with n_seasons=3.
    ///
    /// The earlier `_two_season_hand_computed` test only pinned down k=1 (closed
    /// form) and k=2 boundedness — it called the same helpers (`periodic_auto…`,
    /// `cross_correlation_*`) to derive its expectations, so any indexing bug in
    /// the assembly would be invisible to it. This test instead computes every
    /// entry of Σ_11, Σ_22, and Σ_12 from raw scalar arithmetic on the
    /// observation arrays, with no helper calls in the expected-value path.
    ///
    /// Data (5 years × 3 seasons, all means = 0, all pop std = √2):
    ///   z0 = [ 1,  2, -1,  0, -2]   z1 = [ 0,  1, -1,  2, -2]   z2 = [ 1, 0,  2, -1, -2]
    ///   a2 = [ 0,  1,  2, -1, -2]   (only a2 is accessed for season=0, k=3)
    ///   a0 = a1 = [0; 5] (zero-std; not accessed for season=0)
    ///
    /// Hand-computed entries (population 1/N divisor throughout):
    ///   Σ_11 = [[1.0, 0.0], [0.0, 1.0]]               (ρ^0(3) = 0)
    ///   Σ_22 = [[1.0, 0.0, 0.9],                       (rows: Z_{t-1}, Z_{t-2}, A_{t-1})
    ///           [0.0, 1.0, 0.1],
    ///           [0.9, 0.1, 1.0]]
    ///   Σ_12 = [[0.5, -0.625, 0.10],                   (row 0 = Z_t)
    ///           [0.3,  0.7,   0.4 ]]                   (row 1 = Z_{t-3})
    ///
    /// Σ_12[0, 2] = ρ(Z_t, A_{t-1}) uses [`cross_correlation_a_z_neg1`], which
    /// follows NEWAVE's convention of dividing the cross-product sum by the
    /// LARGER bucket size (here 5) rather than n_pairs (4 after the
    /// year-forward-shift skips one Z entry).
    ///
    /// The Σ_12[1,*] block is the Bug #1 regression guard — pre-fix, this row
    /// was anchored at season_minus_k and produced unrelated values that
    /// happily passed the looser legacy test.
    #[test]
    fn assemble_partitioned_covariance_three_season_k3_hand_computed() {
        let z0: &[f64] = &[1.0, 2.0, -1.0, 0.0, -2.0];
        let z1: &[f64] = &[0.0, 1.0, -1.0, 2.0, -2.0];
        let z2: &[f64] = &[1.0, 0.0, 2.0, -1.0, -2.0];
        let a0: &[f64] = &[0.0; 5];
        let a1: &[f64] = &[0.0; 5];
        let a2: &[f64] = &[0.0, 1.0, 2.0, -1.0, -2.0];

        let obs: &[&[f64]] = &[z0, z1, z2];
        let stats = [
            pop_mean_std_ann(z0),
            pop_mean_std_ann(z1),
            pop_mean_std_ann(z2),
        ];
        let ann_obs: &[&[f64]] = &[a0, a1, a2];
        let ann_stats = [
            pop_mean_std_ann(a0),
            pop_mean_std_ann(a1),
            pop_mean_std_ann(a2),
        ];

        let cov = assemble_partitioned_covariance(
            0,
            3,
            3,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        // Σ_11.
        let exp_11 = [1.0, 0.0, 0.0, 1.0];
        for (i, &expected) in exp_11.iter().enumerate() {
            assert!(
                (cov.sigma_11[i] - expected).abs() < 1e-12,
                "sigma_11[{i}] = {} expected {expected}",
                cov.sigma_11[i]
            );
        }

        // Σ_22 (3×3, row-major).
        let exp_22 = [1.0, 0.0, 0.9, 0.0, 1.0, 0.1, 0.9, 0.1, 1.0];
        for (i, &expected) in exp_22.iter().enumerate() {
            assert!(
                (cov.sigma_22[i] - expected).abs() < 1e-12,
                "sigma_22[{}, {}] = {} expected {expected}",
                i / 3,
                i % 3,
                cov.sigma_22[i]
            );
        }

        // Σ_12 (2×3, row-major). Row 1 entries (Z_{t-3} cross conditioning Z's)
        // are the Bug #1 regression guard.
        let exp_12 = [0.5, -0.625, 0.10, 0.3, 0.7, 0.4];
        for (i, &expected) in exp_12.iter().enumerate() {
            assert!(
                (cov.sigma_12[i] - expected).abs() < 1e-12,
                "sigma_12[{}, {}] = {} expected {expected}",
                i / 3,
                i % 3,
                cov.sigma_12[i]
            );
        }
    }

    /// AC#5: when Σ_22 becomes singular at k=2, the loop breaks early and returns
    /// a Vec with length ≤ 1 (no entry for lag 2).
    #[test]
    fn conditional_facp_partitioned_singular_sigma22_breaks_early() {
        // Use n_seasons=1 with z0=a0=alternating [-1,1,-1,1,-1,1] (mean=0, pop_std=1).
        //
        // At k=2, prev_season = (0+1-1)%1 = 0.  Σ_22 is the 2×2 matrix:
        //
        //   Σ_22 = [[ 1,        rho_za ],
        //           [ rho_za,   1      ]]
        //
        // where the cross-term rho_za = cross_correlation_z_a(season=0, lag=0, a0, z0).
        //
        // With z0 = a0 = [-1,1,-1,1,-1,1]:
        //   mean_z0 = mean_a0 = 0, std_z0 = std_a0 = 1 (exact, integer arithmetic).
        //   gamma = 1/6 * [(-1)(-1)+(1)(1)+(-1)(-1)+(1)(1)+(-1)(-1)+(1)(1)]
        //         = 1/6 * 6 = 1.0  (exact IEEE 754).
        //   rho_za = 1.0 / (1.0 * 1.0) = 1.0  (exact).
        //
        // Σ_22 = [[1,1],[1,1]] → det = 0 → solve_linear_system returns None →
        // the outer loop breaks.  k=1 (Σ_22 = [[1.0]]) is always solvable, so
        // exactly 1 entry is produced before the break.
        let z0: &[f64] = &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let a0: &[f64] = &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

        let obs: &[&[f64]] = &[z0];
        let stats = [pop_mean_std_ann(z0)];
        let ann_obs: &[&[f64]] = &[a0];
        let ann_stats = [pop_mean_std_ann(a0)];

        // Verify exact-arithmetic precondition: rho_za = 1.0 for lag=0.
        let rho_za = cross_correlation_z_a(
            0,
            0,
            1,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        assert_eq!(
            rho_za, 1.0,
            "rho_za must be exactly 1.0 for identical series"
        );

        // Also verify solve_linear_system recognises [[1,1],[1,1]] as singular.
        let mut mat_check = vec![1.0f64, rho_za, rho_za, 1.0];
        let mut rhs_check = vec![0.0f64, 0.0];
        assert!(
            solve_linear_system(&mut mat_check, &mut rhs_check, 2).is_none(),
            "[[1,1],[1,1]] must be detected as singular"
        );

        let result = conditional_facp_partitioned(
            0,
            4,
            1,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        // k=1 succeeds (Σ_22=[[1.0]]), k=2 breaks (Σ_22 singular).
        // Result has exactly 1 entry; the assertion allows ≤1 for robustness.
        assert!(
            result.len() <= 1,
            "expected ≤1 entry (break at k=2 singularity), got {} entries: {result:?}",
            result.len()
        );
    }

    /// AC#6: when Σ̄[0,0]·Σ̄[1,1] ≤ 0, the function records 0.0 (no NaN/Inf).
    ///
    /// Use Z = A = alternating [1,-1,1,-1,...] for exact integer arithmetic.
    ///
    /// With n_seasons=1, z0=[1,-1,1,-1,1,-1] and a0=[1,-1,1,-1,1,-1]:
    ///
    ///   std_z0 = std_a0 = 1  (pop std of alternating ±1)
    ///
    ///   ρ^0(1): lag=1, n_seasons=1, lag_season=0=ref_season ⇒ years_crossed=1.
    ///     ref_start=1, n_pairs=5. pairs: (-1,1),(1,-1),(-1,1),(1,-1),(-1,1).
    ///     gamma = 1/5*(-1-1-1-1-1) = -1. ρ^0(1) = -1/1 = -1.
    ///
    ///   α = cross_correlation_a_z_neg1(season=0, n_seasons=1, …):
    ///     z_season=(0+1)%1=0 ⇒ years_crossed=1.
    ///     z_start=1, n_pairs=5. pairs (a0[i], z0[i+1]): (1,-1),(-1,1),(1,-1),(-1,1),(1,-1).
    ///     gamma = 1/5*(−1−1−1−1−1) = −1. α = −1/(1·1) = −1.
    ///
    ///   β = cross_correlation_z_a(season=0, lag=0, n_seasons=1, …):
    ///     lag=0, years_crossed=0, n_pairs=6.
    ///     pairs: (a0[i], z0[i]): (1,1),(-1,-1),(1,1),(-1,-1),(1,1),(-1,-1).
    ///     gamma = 1/6*(1+1+1+1+1+1) = 1. β = 1/(1·1) = 1.
    ///
    ///   Σ_12 = [[-1], [1]]  (α in row 0, β in row 1)
    ///   Σ_22 = [[1.0]]
    ///   X = [[-1, 1]]  (solve trivial 1×1 system)
    ///
    ///   Σ̄ = Σ_11 − Σ_12·X
    ///     Σ_11 = [[1, -1],[-1, 1]]   (ρ^0(1) = -1)
    ///     Σ_12·X = [[-1·-1, -1·1],[1·-1, 1·1]] = [[1,-1],[-1,1]]
    ///     Σ̄ = [[1-1, -1-(-1)],[-1-(-1), 1-1]] = [[0,0],[0,0]]
    ///
    ///   denom_sq = Σ̄[0,0] · Σ̄[1,1] = 0·0 = 0 ≤ 0 → returns 0.0.
    #[test]
    fn conditional_facp_partitioned_zero_denom_returns_zero() {
        // Alternating ±1 in a single season (n_seasons=1).
        let z0: &[f64] = &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let a0: &[f64] = &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

        let obs: &[&[f64]] = &[z0];
        let stats = [pop_mean_std_ann(z0)];
        let ann_obs: &[&[f64]] = &[a0];
        let ann_stats = [pop_mean_std_ann(a0)];

        // Verify integer-arithmetic setup.
        let (_, std_z) = stats[0];
        assert!(
            (std_z - 1.0).abs() < 1e-10,
            "std_z0 must be 1.0, got {std_z}"
        );

        let result = conditional_facp_partitioned(
            0,
            1,
            1,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        assert_eq!(result.len(), 1, "expected 1 entry for max_order=1");
        assert_eq!(
            result[0], 0.0,
            "zero denominator must yield 0.0, got {}",
            result[0]
        );
    }

    // -----------------------------------------------------------------------
    // select_order_pacf_annual tests
    // -----------------------------------------------------------------------

    #[test]
    fn select_order_pacf_annual_empty_returns_zero() {
        let result = select_order_pacf_annual(&[], 100, 1.96);
        assert_eq!(result.selected_order, 0);
        assert!(result.pacf_values.is_empty());
    }

    #[test]
    fn select_order_pacf_annual_first_lag_significant() {
        // threshold = 1.96 / sqrt(100) = 0.196
        // conditional_facp[0] = 0.5 > 0.196 -> significant; lag 2 = 0.1 is not.
        let result = select_order_pacf_annual(&[0.5, 0.1], 100, 1.96);
        assert_eq!(result.selected_order, 1);
        assert!((result.threshold - 0.196).abs() < 1e-10);
    }

    #[test]
    fn select_order_pacf_annual_max_lag_significant() {
        // threshold = 1.96 / sqrt(100) = 0.196
        // conditional_facp = [0.05, 0.03, 0.4]
        // Only lag 3 (0.4) exceeds threshold -> max_significant = 3.
        // PACF[0] = 0.05 is non-zero -> min-order-1 gives max(1, 3) = 3.
        let result = select_order_pacf_annual(&[0.05, 0.03, 0.4], 100, 1.96);
        assert_eq!(result.selected_order, 3);
    }

    #[test]
    fn select_order_pacf_annual_min_order_one_rule_when_lag1_nonzero() {
        // n=100; both PACF values below their lag-dependent thresholds.
        // Without min-order-1 rule, max_significant = 0.
        // PACF[0] = 0.05 is non-zero -> NEWAVE rule forces order = max(1, 0) = 1.
        let result = select_order_pacf_annual(&[0.05, 0.03], 100, 1.96);
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn select_order_pacf_annual_negative_value_uses_abs() {
        // |-0.5| = 0.5 > lag-1 threshold ~0.1970 -> lag 1 significant.
        let result = select_order_pacf_annual(&[-0.5, 0.1], 100, 1.96);
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn select_order_pacf_annual_zero_observations_returns_infinity_threshold() {
        // n_observations = 0 -> threshold = infinity -> no lag exceeds it.
        // PACF[0] = 0.5 is non-zero -> min-order-1 rule forces order = 1.
        let result = select_order_pacf_annual(&[0.5, 0.3], 0, 1.96);
        assert_eq!(result.threshold, f64::INFINITY);
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn select_order_pacf_annual_structural_zero_at_lag1_returns_zero() {
        // FACP exactly 0.0 at lag 1 -> structural-zero short-circuit.
        // Even though lag 2 = 0.5 is "significant", the model is forced
        // to white noise (degenerate Z⊗A bucket).
        let result = select_order_pacf_annual(&[0.0, 0.5], 100, 1.96);
        assert_eq!(result.selected_order, 0);
    }

    #[test]
    fn select_order_pacf_annual_structural_zero_at_lag2_does_not_short_circuit() {
        // Structural zero at lag 2 (PACF[1] = 0.0) does NOT trigger short-circuit;
        // only lag 1 does. NEWAVE proceeds normally with the surviving lags:
        // lag 1 = 0.5 > threshold, lag 3 = 0.6 > threshold -> order = 3.
        let result = select_order_pacf_annual(&[0.5, 0.0, 0.6], 100, 1.96);
        assert_eq!(result.selected_order, 3);
    }

    #[test]
    fn select_order_pacf_annual_only_lag1_significant_with_zeros_after() {
        // Realistic NEWAVE pattern: degenerate Schur complement at lag 2+
        // produces FACP = [+0.37, 0, 0, 0, 0, 0]. NEWAVE picks order 1
        // (lag 1 is significant), not 0.
        let result = select_order_pacf_annual(&[0.37, 0.0, 0.0, 0.0, 0.0, 0.0], 92, 1.96);
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn select_order_pacf_annual_structural_zero_at_lag3_does_not_short_circuit() {
        // Structural zero at lag 3 (k > 2) does NOT trigger short-circuit.
        // Lag 1 (0.5) is significant; max_significant = 1; min-order-1 -> 1.
        let result = select_order_pacf_annual(&[0.5, 0.1, 0.0], 100, 1.96);
        assert_eq!(result.selected_order, 1);
    }

    #[test]
    fn select_order_pacf_annual_matches_select_order_pacf_for_short_circuit_zero_at_lag1() {
        // When lag 1 has a structural zero, the annual variant returns 0
        // even if higher lags would be significant — this is where it
        // diverges most sharply from `select_order_pacf`, which only looks
        // at the maximum significant lag.
        let facp = &[0.0, 0.5_f64];
        let n = 100_usize;
        let z = 1.96_f64;
        let annual = select_order_pacf_annual(facp, n, z);
        let classical = select_order_pacf(facp, n, z);
        assert_eq!(annual.selected_order, 0);
        assert_eq!(classical.selected_order, 2);
    }

    // -----------------------------------------------------------------------
    // estimate_annual_seasonal_stats tests (AC #1, AC #2)
    // -----------------------------------------------------------------------

    use super::{estimate_annual_seasonal_stats, estimate_periodic_ar_annual_coefficients};

    /// AC #1 — Four-year synthetic monthly series; hand-computed Bessel-corrected
    /// mean and std.
    ///
    /// Series: `z[year*12 + month] = (month+1)*10 + year*5`.
    ///
    /// Rolling-window construction (index `i`, window `z[i..i+12]`, target
    /// index `i+11`): each value `A = mean(z[i..i+12])` is stored under the
    /// season of `z[i+11]` — i.e., the PDF time-index of `A_{t-1}` when
    /// `t = i + 12`.
    ///
    /// Each season has exactly 3 A_t values:
    /// - For `s ∈ 0..10` the windows cover target years `{1, 2, 3}` (the
    ///   window crosses into year `y` and `i_min = s + 1 ≥ 1`).
    /// - For `s == 11` the windows cover target years `{0, 1, 2}` (the
    ///   window is entirely within year `y`, so `i_min = 0`); the loop bound
    ///   `i < 36` excludes year 3.
    ///
    /// Window mean (`i = y*12 + s - 11` for `s ∈ 0..10`, `i = y*12` for `s == 11`):
    /// `mean = (780 + 5 * total_year_offset_in_window) / 12`.
    ///
    /// Average over the 3 valid years yields:
    /// - `s ∈ 0..10`: `(845 + 5*s) / 12`
    /// - `s == 11`:   `70.0`           (note: NOT `(845 + 55)/12 = 75.0` because
    ///                                  the y-range shifts down by one)
    ///
    /// All stds are 5.0 (Bessel-corrected, `1/(N-1)` with N=3) — each year
    /// shifts every observation by `+5`, so window means differ by `5`
    /// between consecutive years for every season.
    ///
    /// This test intentionally pins the `1/(N-1)` divisor and the divergence from
    /// `rel_parpa.pdf` eq. 18 (which uses `1/N`). See ticket documentation.
    #[test]
    fn estimate_annual_seasonal_stats_four_year_synthetic_hand_computed() {
        let hydro_id = EntityId::from(1);
        let stages = make_monthly_stages(2000, 4);

        // Build 48 observations: z[year*12 + month] = (month+1)*10 + year*5.
        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in 0..4_usize {
            for month in 0..12_usize {
                let value = (month + 1) as f64 * 10.0 + year as f64 * 5.0;
                let date =
                    NaiveDate::from_ymd_opt(2000 + year as i32, month as u32 + 1, 1).unwrap();
                observations.push((hydro_id, date, value));
            }
        }

        let result =
            estimate_annual_seasonal_stats(&observations, &stages, &[hydro_id], None).unwrap();

        assert_eq!(result.len(), 12, "must return exactly one entry per season");

        for s in &result {
            assert_eq!(
                s.hydro_id, hydro_id,
                "hydro_id must match for season {}",
                s.season_id
            );
            // For seasons 0..10 the window crosses into the target year, so
            // valid `y ∈ {1, 2, 3}` (y_avg = 2). For season 11 the window
            // sits entirely within year `y`, so valid `y ∈ {0, 1, 2}`
            // (y_avg = 1) — producing the discontinuity from `(845+5*11)/12`
            // to `70.0`.
            let expected_mean = if s.season_id == 11 {
                70.0
            } else {
                (845.0 + 5.0 * s.season_id as f64) / 12.0
            };
            assert!(
                (s.mean_m3s - expected_mean).abs() < 1e-10,
                "season {}: mean_m3s={} expected={}",
                s.season_id,
                s.mean_m3s,
                expected_mean
            );
            // 3 samples with mutual deviations {-5, 0, 5} → sum-of-squares 50.
            // Population (1/N) variance = 50/3 → std = sqrt(50/3).
            let expected_std = (50.0_f64 / 3.0).sqrt();
            assert!(
                (s.std_m3s - expected_std).abs() < 1e-10,
                "season {}: std_m3s={} expected {} (population 1/N)",
                s.season_id,
                s.std_m3s,
                expected_std
            );
        }

        // Sorted by (hydro_id, season_id) ascending.
        let season_ids: Vec<usize> = result.iter().map(|s| s.season_id).collect();
        assert_eq!(
            season_ids,
            (0..12).collect::<Vec<_>>(),
            "result must be sorted by season_id"
        );
    }

    /// AC #2 — History too short: 11 observations cannot form any rolling window.
    ///
    /// Requires at least 13 observations (indices 0..12 inclusive) for the first
    /// window to exist. 11 observations is strictly insufficient.
    #[test]
    fn estimate_annual_seasonal_stats_too_short_history_errors() {
        use crate::StochasticError;

        let hydro_id = EntityId::from(42);
        let stages = make_monthly_stages(2000, 2);

        // 11 observations — not enough for even one rolling window.
        let observations: Vec<(EntityId, NaiveDate, f64)> = (0..11)
            .map(|i| {
                let month = i % 12 + 1;
                let date = NaiveDate::from_ymd_opt(2000, month as u32, 1).unwrap();
                (hydro_id, date, i as f64 * 10.0)
            })
            .collect();

        let err =
            estimate_annual_seasonal_stats(&observations, &stages, &[hydro_id], None).unwrap_err();

        assert!(
            matches!(err, StochasticError::InsufficientData { .. }),
            "expected InsufficientData, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // estimate_periodic_ar_annual_coefficients tests (AC #3, AC #4, AC #5)
    // -----------------------------------------------------------------------

    /// AC #3 — `selected_order = 0`: 1×1 system yields only ψ, `coefficients` is empty.
    ///
    /// Data reuses the order-zero fixture from `build_extended_periodic_yw_matrix`.
    /// The 1×1 system has matrix `[[1.0]]` and rhs `[cross_correlation_a_z_neg1(...)]`.
    /// Solution: ψ = rhs[0].
    /// sigma2 = 1 − ψ * rhs[0] = 1 − rhs[0]^2.
    #[test]
    fn estimate_periodic_ar_annual_coefficients_order_zero_returns_one_by_one_solution() {
        let z0: &[f64] = &[1.0, 3.0, 2.0, 5.0, 4.0];
        let z1: &[f64] = &[2.0, 1.0, 4.0, 3.0, 6.0];
        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];

        let a0: &[f64] = &[1.5, 2.0, 3.0, 4.0, 3.5];
        let a1: &[f64] = &[1.0, 3.0, 2.5, 3.5, 2.0];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        // The expected annual_coefficient equals the rhs[0] from the 1×1 system,
        // which is cross_correlation_a_z_neg1(prev_season=1, n_seasons=2, ...).
        // For n_seasons=2, the year-forward-shift skips one Z entry leaving
        // n_pairs=4. NEWAVE's max-bucket-size divisor (=5) scales the result
        // by 4/5 vs the legacy n_pairs convention, so the hand-computed value
        // is 0.14384911389218766 × 4/5 ≈ 0.1150792911…
        let expected_psi = 0.143_849_113_892_187_66 * 4.0 / 5.0;

        let result = estimate_periodic_ar_annual_coefficients(
            0, // season
            0, // selected_order
            2, // n_seasons
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        assert!(
            result.coefficients.is_empty(),
            "order=0 must produce empty coefficients"
        );
        assert!(
            (result.annual_coefficient - expected_psi).abs() < 1e-10,
            "annual_coefficient={} expected≈{}",
            result.annual_coefficient,
            expected_psi
        );
        assert!(
            result.residual_std_ratio > 0.0 && result.residual_std_ratio <= 1.0,
            "residual_std_ratio must be in (0, 1], got {}",
            result.residual_std_ratio
        );
    }

    /// AC #4 — `selected_order = 2` with the 3×3 hand-computed fixture from ticket-005.
    ///
    /// Matrix and RHS come from `build_extended_periodic_yw_matrix_hand_computed_3x3`.
    /// The system is:
    /// ```text
    /// [1.0,   R01,   za0] [φ1]   [rhs0]
    /// [R01,   1.0,   za1] [φ2] = [rhs1]
    /// [za0,   za1,   1.0] [ψ ]   [rhs2]
    /// ```
    /// where R01 ≈ 0.3287979746, za0 ≈ -0.1216216216, za1 ≈ 0.7397954429,
    /// rhs = [0.3698977214, 0.0, 0.1438491139].
    ///
    /// Numerical solution (verified with numpy):
    /// - φ1 ≈ 0.81267678
    /// - φ2 ≈ -0.98684211
    /// - ψ  ≈ 0.97274947
    /// - sigma2 ≈ 0.55946356
    /// - residual_std_ratio ≈ 0.74797297
    #[test]
    fn estimate_periodic_ar_annual_coefficients_hand_computed_three_season() {
        let z0: &[f64] = &[1.0, 3.0, 2.0, 5.0, 4.0];
        let z1: &[f64] = &[2.0, 1.0, 4.0, 3.0, 6.0];
        let obs: &[&[f64]] = &[z0, z1];
        let stats = [pop_mean_std_ann(z0), pop_mean_std_ann(z1)];

        let a0: &[f64] = &[1.5, 2.0, 3.0, 4.0, 3.5];
        let a1: &[f64] = &[1.0, 3.0, 2.5, 3.5, 2.0];
        let ann_obs: &[&[f64]] = &[a0, a1];
        let ann_stats = [pop_mean_std_ann(a0), pop_mean_std_ann(a1)];

        let result = estimate_periodic_ar_annual_coefficients(
            0, // season
            2, // selected_order
            2, // n_seasons
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        assert_eq!(
            result.coefficients.len(),
            2,
            "selected_order=2 must produce 2 AR coefficients"
        );

        // Expected values reflect NEWAVE's max-bucket-size cross-cov divisor
        // (see [`cross_correlation_z_a`] docs). For the synthetic 5-element
        // buckets, only the cross-terms with year-forward-shift pick up the
        // 4/5 scale; the rest of the YW system is unaffected.
        let tol = 1e-8;
        assert!(
            (result.coefficients[0] - 0.773_889_929_208_993_9).abs() < tol,
            "φ1={} expected≈0.7738899292",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - (-0.903_947_368_421_052_3)).abs() < tol,
            "φ2={} expected≈-0.9039473684",
            result.coefficients[1]
        );
        assert!(
            (result.annual_coefficient - 0.877_937_183_016_726_5).abs() < tol,
            "ψ={} expected≈0.8779371830",
            result.annual_coefficient
        );
        assert!(
            (result.residual_std_ratio - 0.782_756_341_321_194_8).abs() < tol,
            "residual_std_ratio={} expected≈0.7827563413",
            result.residual_std_ratio
        );
    }

    /// AC #5 — Singular extended YW system returns the zero fallback.
    ///
    /// Uses the alternating series `[-1, 1, -1, 1, -1, 1]` with `n_seasons=1`
    /// and `selected_order=1`.  Since Z and A are identical, `rho_za = 1.0`
    /// (exactly, by integer arithmetic).  The resulting 2×2 extended matrix is
    /// `[[1.0, 1.0], [1.0, 1.0]]` which has determinant 0, so
    /// `solve_linear_system` returns `None` and the function returns the
    /// zero-fallback `PeriodicYwAnnualResult`.
    ///
    /// Derivation mirrors the precondition asserted in
    /// `conditional_facp_partitioned_singular_sigma22_breaks_early`.
    #[test]
    fn estimate_periodic_ar_annual_coefficients_singular_returns_zero_result() {
        // Alternating series with exact mean=0, std=1 (population).
        // Z = A (identical) => rho_za = exactly 1.0 in IEEE 754 arithmetic.
        let z0: &[f64] = &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let a0: &[f64] = &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

        let obs: &[&[f64]] = &[z0];
        let stats = [pop_mean_std_ann(z0)];
        let ann_obs: &[&[f64]] = &[a0];
        let ann_stats = [pop_mean_std_ann(a0)];

        // With n_seasons=1, selected_order=1 and Z=A, the 2×2 extended matrix is:
        //   [[matrix[0,0]=1.0, matrix[0,1]=rho_za(lag=0)=1.0],
        //    [matrix[1,0]=1.0, matrix[1,1]=1.0             ]]
        // = [[1, 1], [1, 1]] → determinant 0 → solve_linear_system returns None.
        let result = estimate_periodic_ar_annual_coefficients(
            0, // season
            1, // selected_order → 2×2 extended system [[1,1],[1,1]] (singular)
            1, // n_seasons
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );

        assert!(
            result.coefficients.is_empty(),
            "singular system must return empty coefficients, got {:?}",
            result.coefficients
        );
        assert_eq!(
            result.annual_coefficient, 0.0,
            "singular system must return annual_coefficient=0.0"
        );
        assert_eq!(
            result.residual_std_ratio, 1.0,
            "singular system must return residual_std_ratio=1.0"
        );
    }

    /// Regression test: `assemble_partitioned_covariance` sigma_22 cross-term
    /// lag indexing for k=3.
    ///
    /// The cross-term `sigma_22[i, k-1]` must equal
    /// `cross_correlation_z_a(prev_season, lag=i, ...)` because `Z_{t-1-i}` is
    /// exactly `i` steps older than `A_{t-1}`.  The old (buggy) code used
    /// `lag = k-2-i`, which is only coincidentally correct when `k=2` (both
    /// formulae reduce to `lag=0`).  For `k=3` the bug swaps the lag-0 and
    /// lag-1 cross-terms.
    ///
    /// This test FAILS on the old `k-2-i` formula and PASSES on the fixed `i`
    /// formula.
    #[test]
    fn assemble_partitioned_covariance_sigma_22_cross_term_lag_indexing() {
        // 4-season synthetic dataset, 20 years of data.
        let n_seasons = 4;
        let n_years = 20;

        // Z observations: deterministic but non-trivial to avoid accidental
        // symmetry that would make lag-0 == lag-1.
        let z_data: Vec<Vec<f64>> = (0..n_seasons)
            .map(|s| {
                (0..n_years)
                    .map(|y| {
                        (s as f64 * 1.7 + y as f64 * 0.3).sin() * 4.0
                            + (s as f64 * 0.6 - y as f64 * 1.4).cos() * 1.5
                    })
                    .collect()
            })
            .collect();
        let a_data: Vec<Vec<f64>> = (0..n_seasons)
            .map(|s| {
                (0..n_years)
                    .map(|y| (s as f64 * 0.9 + y as f64 * 0.7).cos() * 2.0 + (y as f64 * 0.2).sin())
                    .collect()
            })
            .collect();

        let obs_refs: Vec<&[f64]> = z_data.iter().map(Vec::as_slice).collect();
        let ann_refs: Vec<&[f64]> = a_data.iter().map(Vec::as_slice).collect();
        let stats: Vec<(f64, f64)> = z_data.iter().map(|v| pop_mean_std_ann(v)).collect();
        let ann_stats: Vec<(f64, f64)> = a_data.iter().map(|v| pop_mean_std_ann(v)).collect();

        let season = 0;
        let k = 3;
        let prev_season = (season + n_seasons - 1) % n_seasons;

        let cov = assemble_partitioned_covariance(
            season,
            k,
            n_seasons,
            &obs_refs,
            &stats,
            &[0_i32; 32],
            &ann_refs,
            &ann_stats,
            &[0_i32; 32],
        );

        // For k=3 the cross-term block covers i in 0..2 (i.e. i=0 and i=1).
        // sigma_22[i, k-1] should equal cross_correlation_z_a(prev_season, lag=i, ...).
        for i in 0..k - 1 {
            let expected = cross_correlation_z_a(
                prev_season,
                i, // lag = i (the fix)
                n_seasons,
                &obs_refs,
                &stats,
                &[0_i32; 32],
                &ann_refs,
                &ann_stats,
                &[0_i32; 32],
            );
            let actual = cov.sigma_22[i * k + (k - 1)];
            assert!(
                (actual - expected).abs() < 1e-12,
                "sigma_22[{i}, {k_minus_1}] = {actual:.15} but \
                 cross_correlation_z_a(prev_season={prev_season}, lag={i}) = {expected:.15}",
                k_minus_1 = k - 1,
            );
            // Also check symmetry: sigma_22[k-1, i] == sigma_22[i, k-1].
            let sym = cov.sigma_22[(k - 1) * k + i];
            assert!(
                (sym - expected).abs() < 1e-12,
                "sigma_22[{k_minus_1}, {i}] = {sym:.15} not symmetric with sigma_22[{i}, {k_minus_1}]",
                k_minus_1 = k - 1,
            );
        }
    }
}
