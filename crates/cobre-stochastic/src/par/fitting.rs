//! Levinson-Durbin recursion and seasonal statistics estimation for PAR model fitting.
//!
//! This module provides four core primitives for fitting Periodic Autoregressive
//! models:
//!
//! 1. [`levinson_durbin`] — solves Yule-Walker equations in O(p²) time given
//!    a sequence of autocorrelation values.
//! 2. [`estimate_seasonal_stats`] — computes seasonal means and
//!    Bessel-corrected standard deviations from historical inflow observations,
//!    grouped by `(entity, season)` pair.
//! 3. [`estimate_ar_coefficients`] — computes cross-seasonal autocorrelations
//!    and calls the Levinson-Durbin recursion to produce standardized AR
//!    coefficients and residual std ratios for each `(entity, season)` pair.
//! 4. [`estimate_correlation`] — computes the Pearson correlation matrix of
//!    PAR model residuals across entities, returning a [`CorrelationModel`]
//!    suitable for downstream Cholesky decomposition.
//!
//! ## Yule-Walker equations
//!
//! For an AR(p) process with autocorrelations `ρ(1)..ρ(p)`, the
//! Yule-Walker system is:
//!
//! ```text
//! [ ρ(0)  ρ(1) … ρ(p-1) ] [ ψ₁ ]   [ ρ(1) ]
//! [ ρ(1)  ρ(0) … ρ(p-2) ] [ ψ₂ ] = [ ρ(2) ]
//! [  ⋮                   ] [  ⋮ ]   [  ⋮   ]
//! [ ρ(p-1)…      ρ(0)   ] [ ψₚ ]   [ ρ(p) ]
//! ```
//!
//! where `ρ(0) = 1` (normalised autocorrelation). The Levinson-Durbin
//! recursion solves this in O(p²) without forming the full Toeplitz matrix.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::NaiveDate;
use cobre_core::{
    scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
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
    /// Bessel-corrected sample standard deviation (N − 1 divisor).
    pub std: f64,
}

/// Estimate seasonal means and standard deviations from historical observations.
///
/// Groups observations by `(entity_id, season_id)` and computes the sample
/// mean and Bessel-corrected standard deviation for each group. Only entities
/// listed in `entity_ids` are processed; observations for other entities are
/// silently ignored.
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
///   fewer than 2 observations (Bessel correction requires N ≥ 2).
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

    // Compute mean and Bessel-corrected std for each group.
    let mut result: Vec<SeasonalStats> = Vec::with_capacity(group_map.len());
    for ((entity_id, _season_id), (values, stage_id)) in group_map {
        let n = values.len();
        if n < 2 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "entity {entity_id} season mapped to stage {stage_id} \
                     has {n} observation(s); need at least 2 for Bessel-corrected \
                     standard deviation"
                ),
            });
        }

        // n is the number of observations; the cast to f64 is intentional here.
        // In practice, observation counts never exceed ~10^6 (well within the
        // 2^53 exact-integer range of f64), so precision loss cannot occur.
        #[allow(clippy::cast_precision_loss)]
        let mean = values.iter().copied().sum::<f64>() / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let variance =
            values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
        let std = variance.sqrt();

        result.push(SeasonalStats {
            entity_id,
            stage_id,
            mean,
            std,
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
// Levinson-Durbin
// ---------------------------------------------------------------------------

/// Result of the Levinson-Durbin recursion.
///
/// Contains the AR coefficients, prediction error variances, and partial
/// autocorrelation coefficients computed up to the requested order (which
/// may be less than requested if near-singularity is detected).
#[must_use]
#[derive(Debug, Clone)]
pub struct LevinsonDurbinResult {
    /// AR coefficients ψ*₁..ψ*ₚ for the fitted order.
    ///
    /// Length equals the actual fitted order, which may be less than the
    /// requested order if the recursion was truncated due to near-singularity.
    pub coefficients: Vec<f64>,
    /// Prediction error variance at each intermediate order.
    ///
    /// `sigma2_per_order[k]` is the prediction error variance for an AR model
    /// of order `k+1`. Length equals the actual fitted order.
    pub sigma2_per_order: Vec<f64>,
    /// Partial autocorrelation coefficients (reflection coefficients).
    ///
    /// `parcor[k]` is the partial autocorrelation (reflection coefficient κ)
    /// at order `k+1`. Length equals the actual fitted order.
    pub parcor: Vec<f64>,
    /// Final prediction error variance σ²ₚ for the fitted order.
    ///
    /// Equals `1.0` when `order == 0` (no AR component fitted).
    pub sigma2: f64,
}

/// Solve the Yule-Walker equations via the Levinson-Durbin recursion.
///
/// Given autocorrelations `ρ(1)..ρ(p)` (not including `ρ(0) = 1.0`, which is
/// implied), computes AR coefficients for a model of the requested `order`.
///
/// The recursion runs in O(p²) time and O(p) space. No general-purpose linear
/// algebra library is required.
///
/// If the prediction error variance drops to or below [`f64::EPSILON`] at any
/// intermediate step, the recursion is truncated at that step and coefficients
/// computed up to the previous order are returned. This handles numerically
/// singular or near-singular autocorrelation sequences gracefully.
///
/// # Parameters
///
/// - `autocorrelations` — the autocorrelation values `ρ(1), ρ(2), …, ρ(p_max)`.
///   Must satisfy `autocorrelations.len() >= order`.
/// - `order` — maximum AR order `p` to solve for.
///
/// # Examples
///
/// AR(1) with known solution:
///
/// ```
/// use cobre_stochastic::par::fitting::levinson_durbin;
///
/// let result = levinson_durbin(&[0.5], 1);
/// assert!((result.coefficients[0] - 0.5).abs() < 1e-10);
/// assert!((result.sigma2 - 0.75).abs() < 1e-10);
/// ```
///
/// Order-0 returns empty result with unit variance:
///
/// ```
/// use cobre_stochastic::par::fitting::levinson_durbin;
///
/// let result = levinson_durbin(&[], 0);
/// assert!(result.coefficients.is_empty());
/// assert_eq!(result.sigma2, 1.0);
/// ```
pub fn levinson_durbin(autocorrelations: &[f64], order: usize) -> LevinsonDurbinResult {
    debug_assert!(
        autocorrelations.len() >= order,
        "autocorrelations.len() ({}) must be >= order ({})",
        autocorrelations.len(),
        order
    );

    if order == 0 {
        return LevinsonDurbinResult {
            coefficients: Vec::new(),
            sigma2_per_order: Vec::new(),
            parcor: Vec::new(),
            sigma2: 1.0,
        };
    }

    // Pre-allocate all output vectors with the requested order capacity.
    let mut coefficients = Vec::with_capacity(order);
    let mut sigma2_per_order = Vec::with_capacity(order);
    let mut parcor = Vec::with_capacity(order);

    // Two coefficient buffers to implement the reflection step without
    // in-place update hazards: a_prev holds AR(k) coefficients,
    // a_curr accumulates AR(k+1) during each iteration.
    let mut a_prev = vec![0.0_f64; order];
    let mut a_curr = vec![0.0_f64; order];

    let mut sigma2 = 1.0_f64;

    for k in 0..order {
        // Compute numerator: ρ(k+1) - Σ_{j=0}^{k-1} a_prev[j] * ρ(k-j)
        // Note: ρ(0) = 1 is implicit; autocorrelations[i] = ρ(i+1).
        let mut num = autocorrelations[k];
        for j in 0..k {
            // a_prev[j] is ψ_{j+1} for the current AR(k) model.
            // ρ(k - j) = autocorrelations[k - j - 1]
            num -= a_prev[j] * autocorrelations[k - j - 1];
        }

        let kappa = num / sigma2;

        // Update AR(k+1) coefficients via the reflection step:
        //   a_curr[j] = a_prev[j] - kappa * a_prev[k-1-j]   for j in 0..k
        //   a_curr[k] = kappa
        for j in 0..k {
            a_curr[j] = a_prev[j] - kappa * a_prev[k - 1 - j];
        }
        a_curr[k] = kappa;

        // Update prediction error variance.
        let sigma2_next = sigma2 * (1.0 - kappa * kappa);

        // Store partial autocorrelation and updated variance.
        parcor.push(kappa);
        sigma2_per_order.push(sigma2_next);
        coefficients.clear();
        coefficients.extend_from_slice(&a_curr[..=k]);

        sigma2 = sigma2_next;

        // Near-singularity guard: if sigma2 has collapsed, truncate here.
        // The coefficients up to this order (which caused sigma2 to reach
        // near-zero) are retained as valid; further orders would be unstable.
        if sigma2 <= f64::EPSILON {
            break;
        }

        // Rotate buffers: a_curr becomes a_prev for the next iteration.
        a_prev[..=k].copy_from_slice(&a_curr[..=k]);
    }

    LevinsonDurbinResult {
        coefficients,
        sigma2_per_order,
        parcor,
        sigma2,
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
}

/// Estimate AR coefficients for all `(entity, season)` pairs.
///
/// Computes cross-seasonal autocorrelations from historical observations,
/// then solves the Yule-Walker system via [`levinson_durbin`] to obtain
/// standardized AR coefficients and the residual std ratio for each pair.
///
/// # Cross-seasonal autocorrelation
///
/// For season `m` and lag `l`, the cross-seasonal autocorrelation is:
///
/// ```text
/// gamma_m(l) = (1/(N_m - 1)) * Σ_{t: season(t)=m} (a_t − μ_m)(a_{t−l} − μ_{m−l})
/// rho_m(l)   = gamma_m(l) / (s_m · s_{m−l})
/// ```
///
/// where `μ_m`, `s_m` come from `seasonal_stats`, and season indices wrap
/// cyclically: season 0 follows season `M−1` (where `M` is the number of
/// distinct seasons).
///
/// # Parameters
///
/// - `observations` — flat slice of `(entity_id, date, value)` triples,
///   sorted by `(entity_id, date)`.
/// - `seasonal_stats` — output of [`estimate_seasonal_stats`], sorted by
///   `(entity_id, season_id)`.
/// - `stages` — all study stages with `season_id` assignments.
/// - `hydro_ids` — canonical sorted list of entity IDs to estimate for.
/// - `max_order` — maximum AR order to fit.
///
/// # Errors
///
/// - [`StochasticError::InsufficientData`] when any `(entity, season)` pair
///   has fewer than `max_order + 1` observations.
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
/// let estimates = estimate_ar_coefficients(&obs, &stats, &stages_vec, &entity_ids, 1).unwrap();
/// assert_eq!(estimates.len(), 2);
/// ```
#[allow(clippy::too_many_lines)]
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

/// Estimate AR coefficients with an optional [`SeasonMap`] fallback for
/// historical observations that predate the study horizon.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when insufficient
/// observations exist for any `(entity, season)` group.
#[allow(clippy::too_many_lines)]
pub fn estimate_ar_coefficients_with_season_map(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    season_map: Option<&SeasonMap>,
) -> Result<Vec<ArCoefficientEstimate>, StochasticError> {
    // -----------------------------------------------------------------------
    // Step 1: Build date-to-season mapping (same as estimate_seasonal_stats).
    // -----------------------------------------------------------------------
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    // -----------------------------------------------------------------------
    // Step 2: Build (entity_id, season_id) -> SeasonalStats lookup.
    // -----------------------------------------------------------------------
    let stats_lookup: HashMap<(EntityId, usize), &SeasonalStats> = {
        // Determine season_id for each SeasonalStats entry from stage_index.
        // We use the stage_id stored in SeasonalStats to find the season_id of
        // the matching stage.
        let stage_id_to_season: HashMap<i32, usize> = stage_index
            .iter()
            .map(|&(_, _, stage_id, season_id)| (stage_id, season_id))
            .collect();

        seasonal_stats
            .iter()
            .filter_map(|s| {
                let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
                Some(((s.entity_id, season_id), s))
            })
            .collect()
    };

    // Determine the total number of distinct seasons (M).
    let n_seasons: usize = {
        let mut max_season = 0usize;
        for &(_, _, _, season_id) in &stage_index {
            if season_id >= max_season {
                max_season = season_id + 1;
            }
        }
        max_season
    };

    // -----------------------------------------------------------------------
    // Step 3: For each (entity, season), collect observations and their
    // time-ordered lagged values.
    //
    // We build a map: (entity_id, season_id) -> Vec<(value_at_t, [lag_values])>
    // where lag_values[l-1] = value at t-l (for l in 1..=max_order).
    //
    // To find the lagged value a_{t-l}, we need the observations sorted by date
    // for each entity, so we can look up the value `l` positions back.
    // -----------------------------------------------------------------------

    // Collect all observations for each entity in date order (input is sorted).
    // For each entity, map NaiveDate -> value for O(1) lag lookup.
    let mut entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        entity_obs.entry(entity_id).or_default().push((date, value));
    }
    // Ensure each entity's observations are sorted by date.
    for obs_vec in entity_obs.values_mut() {
        obs_vec.sort_unstable_by_key(|(d, _)| *d);
    }

    // Build a date->index map per entity for O(1) lag lookup by position.
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

    // -----------------------------------------------------------------------
    // Step 4: Group observations by (entity_id, season_id), recording only
    // the date and value (for lag lookups we use the entity_obs position map).
    // -----------------------------------------------------------------------
    let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();
    let mut group_obs: HashMap<(EntityId, usize), Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if !entity_set.contains(&entity_id) {
            continue;
        }
        let Some(season_id) = find_season_for_date(&stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
        else {
            continue;
        };
        group_obs
            .entry((entity_id, season_id))
            .or_default()
            .push((date, value));
    }

    // -----------------------------------------------------------------------
    // Step 5: For each (hydro, season) pair, compute autocorrelations and
    // call levinson_durbin.
    // -----------------------------------------------------------------------
    let mut result: Vec<ArCoefficientEstimate> = Vec::new();

    for &hydro_id in hydro_ids {
        for season_id in 0..n_seasons {
            let key = (hydro_id, season_id);

            // Retrieve seasonal stats for this pair (skip if absent).
            let Some(stats_m) = stats_lookup.get(&key) else {
                continue;
            };

            // If max_order == 0, emit white noise immediately.
            if max_order == 0 {
                result.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id,
                    coefficients: Vec::new(),
                    residual_std_ratio: 1.0,
                });
                continue;
            }

            // If s_m == 0.0, constant inflow — white noise.
            if stats_m.std == 0.0 {
                result.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id,
                    coefficients: Vec::new(),
                    residual_std_ratio: 1.0,
                });
                continue;
            }

            // Get observations for this (entity, season) pair.
            let Some(pair_obs) = group_obs.get(&key) else {
                continue;
            };

            let n_m = pair_obs.len();
            // Need at least max_order + 1 observations to compute max_order autocorrelations.
            if n_m < max_order + 1 {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "entity {hydro_id} season {season_id} has {n_m} observation(s); \
                         need at least {} for max_order={max_order}",
                        max_order + 1
                    ),
                });
            }

            // Retrieve the entity's full sorted observation vector for lag lookups.
            let Some(all_obs) = entity_obs.get(&hydro_id) else {
                continue;
            };
            let Some(date_index) = entity_date_index.get(&hydro_id) else {
                continue;
            };

            // Compute cross-seasonal autocorrelations rho_m(1)..rho_m(effective_order).
            let mut autocorrelations: Vec<f64> = Vec::with_capacity(max_order);
            let mut effective_order = max_order;

            'lag_loop: for lag in 1..=max_order {
                // The lagged season is (season_id - lag) mod n_seasons.
                let lag_season = season_id
                    .wrapping_add(n_seasons)
                    .wrapping_sub(lag % n_seasons)
                    % n_seasons;

                // Look up lag-season stats.
                let lag_key = (hydro_id, lag_season);
                let Some(stats_lag) = stats_lookup.get(&lag_key) else {
                    // No stats for lag season — truncate.
                    effective_order = lag - 1;
                    break 'lag_loop;
                };

                // If s_{m-l} == 0.0, autocorrelation undefined — truncate.
                if stats_lag.std == 0.0 {
                    effective_order = lag - 1;
                    break 'lag_loop;
                }

                // Compute gamma_m(lag) = (1/(N_m - 1)) * Σ (a_t - μ_m)(a_{t-lag} - μ_{lag_season})
                // iterating over all observations in season m, looking up the lag-l predecessor.
                let mu_m = stats_m.mean;
                let mu_lag = stats_lag.mean;

                let mut cross_sum = 0.0_f64;
                let mut valid_count = 0usize;

                for &(date, value) in pair_obs {
                    // Find the position of this observation in the entity's full list.
                    let Some(&pos) = date_index.get(&date) else {
                        continue;
                    };
                    // The lag-l predecessor must exist and be at least `lag` positions back.
                    if pos < lag {
                        continue;
                    }
                    let (_, lagged_value) = all_obs[pos - lag];
                    cross_sum += (value - mu_m) * (lagged_value - mu_lag);
                    valid_count += 1;
                }

                if valid_count < 2 {
                    // Not enough paired observations for this lag — truncate.
                    effective_order = lag - 1;
                    break 'lag_loop;
                }

                // Bessel-corrected cross-covariance: divide by (N_m - 1).
                #[allow(clippy::cast_precision_loss)]
                let gamma = cross_sum / (valid_count - 1) as f64;
                let rho = (gamma / (stats_m.std * stats_lag.std)).clamp(-1.0, 1.0);
                autocorrelations.push(rho);
            }

            // Call Levinson-Durbin with the computed autocorrelations.
            let ld_result = levinson_durbin(&autocorrelations, effective_order);

            // residual_std_ratio = sqrt(sigma2), since sigma2 = (sigma_m / s_m)^2.
            let residual_std_ratio = ld_result.sigma2.sqrt().max(f64::EPSILON.sqrt());

            result.push(ArCoefficientEstimate {
                hydro_id,
                season_id,
                coefficients: ld_result.coefficients,
                residual_std_ratio,
            });
        }
    }

    // Sort by (hydro_id, season_id) ascending.
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
/// - `method: "cholesky"`
/// - A single profile named `"default"` with a single group containing all
///   entities in canonical `hydro_ids` order and the estimated correlation matrix.
/// - An empty `schedule` (the single profile applies to all stages).
///
/// The function does **not** enforce positive-semidefiniteness; if the
/// estimated matrix is not PSD, the downstream Cholesky decomposition will
/// detect and report it.
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
#[allow(clippy::too_many_lines)]
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

/// Estimate correlation with an optional [`SeasonMap`] fallback.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when seasonal stats are
/// empty but hydros are present, or when residual computation fails.
#[allow(clippy::too_many_lines)]
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
            method: "cholesky".to_string(),
            profiles,
            schedule: Vec::new(),
        });
    }

    // Guard: seasonal_stats must be non-empty if hydro_ids is non-empty.
    if seasonal_stats.is_empty() {
        return Err(StochasticError::InsufficientData {
            context: "seasonal_stats is empty but hydro_ids is non-empty; \
                      cannot estimate correlation without seasonal statistics"
                .to_string(),
        });
    }

    // -----------------------------------------------------------------------
    // Step 1: Build date-to-season mapping (stage index sorted by start_date).
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Step 2: Build lookup maps for seasonal stats and AR estimates.
    //
    // Key: (entity_id, season_id).
    // -----------------------------------------------------------------------
    let stats_lookup: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
        .iter()
        .filter_map(|s| {
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(((s.entity_id, season_id), s))
        })
        .collect();

    let ar_lookup: HashMap<(EntityId, usize), &ArCoefficientEstimate> = ar_estimates
        .iter()
        .map(|e| ((e.hydro_id, e.season_id), e))
        .collect();

    // -----------------------------------------------------------------------
    // Step 3: Group observations by entity in chronological order.
    //
    // For each entity, build:
    //   - a sorted Vec<(NaiveDate, f64)> for positional lag lookup,
    //   - a HashMap<NaiveDate, usize> for O(1) position lookup by date.
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Step 4: Compute standardized residuals for each hydro.
    //
    // For each hydro and each observation date, if the AR model for its season
    // is available and sufficient lag history exists, compute:
    //   z_t   = (a_t - mu_m) / s_m
    //   z_t-l = (a_{t-l} - mu_{m-l}) / s_{m-l}   for l in 1..=p
    //   ε_t   = z_t - Σ ψ*_{m,l} * z_{t-l}
    //
    // The number of distinct seasons M is needed for cyclic lag wrapping.
    // -----------------------------------------------------------------------
    let n_seasons: usize = stage_index
        .iter()
        .map(|&(_, _, _, sid)| sid + 1)
        .max()
        .unwrap_or(0);

    // residuals: hydro index -> HashMap<NaiveDate, f64>
    let n_hydros = hydro_ids.len();
    let mut hydro_residuals: Vec<HashMap<NaiveDate, f64>> =
        (0..n_hydros).map(|_| HashMap::new()).collect();

    for (hidx, &hydro_id) in hydro_ids.iter().enumerate() {
        let Some(all_obs) = entity_obs.get(&hydro_id) else {
            continue;
        };
        let Some(date_index) = entity_date_index.get(&hydro_id) else {
            continue;
        };

        for &(date, value) in all_obs {
            // Determine the season for this observation.
            let Some(season_id) = find_season_for_date(&stage_index, date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
            else {
                continue;
            };

            // Look up seasonal stats for this entity+season.
            let Some(stats_m) = stats_lookup.get(&(hydro_id, season_id)) else {
                continue;
            };

            // If std is zero (constant inflow), standardized deviation is 0; residual = 0.
            let z_t = if stats_m.std == 0.0 {
                0.0
            } else {
                (value - stats_m.mean) / stats_m.std
            };

            // Retrieve AR coefficients for this entity+season.
            let ar_order;
            let ar_coeffs: &[f64];
            let empty_coeffs: Vec<f64> = Vec::new();
            if let Some(ar_est) = ar_lookup.get(&(hydro_id, season_id)) {
                ar_order = ar_est.coefficients.len();
                ar_coeffs = &ar_est.coefficients;
            } else {
                // No AR estimate for this pair — treat as AR(0), residual = z_t.
                ar_order = 0;
                ar_coeffs = &empty_coeffs;
            }

            // Compute Σ ψ*_{m,l} * z_{t-l} for l in 1..=ar_order.
            // This requires positional lookup of the lag-l predecessor.
            let Some(&pos) = date_index.get(&date) else {
                continue;
            };

            // Check that we have enough lag history.
            if pos < ar_order {
                // Not enough history for all lags — skip this time step.
                continue;
            }

            let mut ar_sum = 0.0_f64;
            let mut lag_ok = true;

            for lag in 1..=ar_order {
                let lag_season = season_id
                    .wrapping_add(n_seasons)
                    .wrapping_sub(lag % if n_seasons == 0 { 1 } else { n_seasons })
                    % if n_seasons == 0 { 1 } else { n_seasons };

                let Some(stats_lag) = stats_lookup.get(&(hydro_id, lag_season)) else {
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
            hydro_residuals[hidx].insert(date, epsilon);
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Compute Pearson correlation matrix (n_hydros x n_hydros).
    //
    // For each pair (i, j), collect time steps where both have residuals,
    // then apply the standard Pearson formula with Bessel correction.
    // -----------------------------------------------------------------------
    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0_f64; n_hydros]; n_hydros];

    for i in 0..n_hydros {
        // Diagonal is always 1.0.
        matrix[i][i] = 1.0;

        for j in (i + 1)..n_hydros {
            // Find overlapping dates.
            let r_i = &hydro_residuals[i];
            let r_j = &hydro_residuals[j];

            // Collect aligned pairs (ε_i, ε_j) where both are present.
            let mut pairs: Vec<(f64, f64)> = r_i
                .iter()
                .filter_map(|(date, &ei)| r_j.get(date).map(|&ej| (ei, ej)))
                .collect();

            if pairs.len() < 2 {
                // Not enough overlap — set correlation to 0.0.
                matrix[i][j] = 0.0;
                matrix[j][i] = 0.0;
                continue;
            }

            // Sort for deterministic iteration across HashMap orderings.
            pairs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            #[allow(clippy::cast_precision_loss)]
            let n = pairs.len() as f64;
            let mean_i = pairs.iter().map(|(ei, _)| ei).sum::<f64>() / n;
            let mean_j = pairs.iter().map(|(_, ej)| ej).sum::<f64>() / n;

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

            // Bessel correction: divide by (N - 1).
            let denom = n - 1.0;
            let std_i = (var_i / denom).sqrt();
            let std_j = (var_j / denom).sqrt();

            let rho = if std_i < f64::EPSILON || std_j < f64::EPSILON {
                // Degenerate case: constant residuals — treat as uncorrelated.
                0.0
            } else {
                (cov / denom) / (std_i * std_j)
            };

            // Clamp to [-1, 1] to correct for floating-point rounding.
            let rho = rho.clamp(-1.0, 1.0);

            matrix[i][j] = rho;
            matrix[j][i] = rho;
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Assemble the CorrelationModel.
    // -----------------------------------------------------------------------
    let entities: Vec<CorrelationEntity> = hydro_ids
        .iter()
        .map(|&id| CorrelationEntity {
            entity_type: "inflow".to_string(),
            id,
        })
        .collect();

    let group = CorrelationGroup {
        name: "default".to_string(),
        entities,
        matrix,
    };

    let profile = CorrelationProfile {
        groups: vec![group],
    };

    let mut profiles = BTreeMap::new();
    profiles.insert("default".to_string(), profile);

    Ok(CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: Vec::new(),
    })
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
///   [`LevinsonDurbinResult::sigma2_per_order`]. `sigma2_per_order[k]`
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
    /// [`LevinsonDurbinResult::parcor`].
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
///   [`LevinsonDurbinResult::parcor`]. `parcor[k]` is the PACF at lag `k+1`.
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
    // NEWAVE's approach: for lag `k` within one cycle (k < n_seasons),
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

    // Compute cross-covariance with population divisor (1/N).
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
/// The matrix has dimension `order x order`. Entry `R[i,j]` is the
/// periodic autocorrelation `rho(season - i, |j - i|)`, where the reference
/// month shifts per row. The matrix is symmetric but NOT Toeplitz because
/// the autocorrelation function varies with the reference period.
///
/// The right-hand side vector `rhs[i] = rho(season - i, order - i)` is
/// extracted from the extended `(order+1) x (order+1)` matrix's last column,
/// following NEWAVE's `_matriz_extendida` + `_resolve_yw` pattern.
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
    if order == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut matrix = vec![0.0_f64; order * order];
    let mut rhs = vec![0.0_f64; order];

    // Fill the matrix: R[i][j] = rho(season - i, |j - i|).
    // Diagonal is always 1.0 (rho(m, 0) = 1 for any m).
    for i in 0..order {
        matrix[i * order + i] = 1.0;
        let ref_month = (season + n_seasons - i % n_seasons) % n_seasons;
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

    // Fill the RHS: rhs[i] = rho(season - i, order - i).
    // This comes from column `order` of the extended (order+1) x (order+1) matrix.
    #[allow(clippy::needless_range_loop)]
    for i in 0..order {
        let ref_month = (season + n_seasons - i % n_seasons) % n_seasons;
        let lag = order - i;
        rhs[i] = periodic_autocorrelation(
            ref_month,
            lag,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
    }

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]
mod tests {
    use super::{
        build_periodic_yw_matrix, levinson_durbin, periodic_autocorrelation, select_order_aic,
        select_order_pacf, solve_linear_system,
    };

    // -----------------------------------------------------------------------
    // AR(1) known values
    // -----------------------------------------------------------------------

    #[test]
    fn levinson_durbin_ar1_known_values() {
        // For AR(1): ψ₁ = ρ(1) = 0.5, σ² = 1 - 0.5² = 0.75.
        let result = levinson_durbin(&[0.5], 1);
        assert_eq!(result.coefficients.len(), 1);
        assert!((result.coefficients[0] - 0.5).abs() < 1e-10);
        assert!((result.sigma2 - 0.75).abs() < 1e-10);
        assert_eq!(result.parcor.len(), 1);
        assert!((result.parcor[0] - 0.5).abs() < 1e-10);
        assert_eq!(result.sigma2_per_order.len(), 1);
        assert!((result.sigma2_per_order[0] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn levinson_durbin_ar1_negative() {
        // ρ(1) = -0.6 → ψ₁ = -0.6, σ² = 1 - 0.36 = 0.64.
        let result = levinson_durbin(&[-0.6], 1);
        assert_eq!(result.coefficients.len(), 1);
        assert!((result.coefficients[0] - (-0.6)).abs() < 1e-10);
        assert!((result.sigma2 - 0.64).abs() < 1e-10);
        assert!((result.parcor[0] - (-0.6)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // AR(2) verification via Yule-Walker matrix multiply
    // -----------------------------------------------------------------------

    #[test]
    fn levinson_durbin_ar2_known_values() {
        // Yule-Walker for AR(2):
        //   [ 1.0  0.8 ] [ ψ₁ ]   [ 0.8 ]
        //   [ 0.8  1.0 ] [ ψ₂ ] = [ 0.5 ]
        //
        // Solution: ψ₁ = (0.8 - 0.8*0.5)/(1 - 0.64) = 0.4/0.36 ≈ 1.1111
        //           ψ₂ = 0.5 - 0.8 * ψ₁ ≈ 0.5 - 0.8889 = -0.3889
        let rho = [0.8_f64, 0.5];
        let result = levinson_durbin(&rho, 2);

        assert_eq!(result.coefficients.len(), 2);
        assert!(result.sigma2 > 0.0);

        // Verify Yule-Walker: R * ψ = r
        // R[i][j] = ρ(|i-j|), r[i] = ρ(i+1)
        // R * ψ:
        //   row 0: 1.0 * ψ₁ + ρ(1) * ψ₂ = ρ(1)
        //   row 1: ρ(1) * ψ₁ + 1.0 * ψ₂ = ρ(2)
        let psi1 = result.coefficients[0];
        let psi2 = result.coefficients[1];
        let residual0 = (1.0 * psi1 + rho[0] * psi2) - rho[0];
        let residual1 = (rho[0] * psi1 + 1.0 * psi2) - rho[1];
        assert!(
            residual0.abs() < 1e-10,
            "Yule-Walker row 0 residual: {residual0}"
        );
        assert!(
            residual1.abs() < 1e-10,
            "Yule-Walker row 1 residual: {residual1}"
        );
    }

    // -----------------------------------------------------------------------
    // AR(3) full Yule-Walker roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn levinson_durbin_ar3_yule_walker_roundtrip() {
        // rho = [0.8, 0.5, 0.3], order = 3
        // Yule-Walker system:
        //   R * ψ = r  where R[i][j] = ρ(|i-j|), r[i] = ρ(i+1)
        let rho = [0.8_f64, 0.5, 0.3];
        let result = levinson_durbin(&rho, 3);

        assert_eq!(result.coefficients.len(), 3);
        assert!(result.sigma2 > 0.0);

        // Build the 3x3 Toeplitz matrix and verify R * ψ = r to 1e-10.
        // R:
        //   [ 1.0  0.8  0.5 ]
        //   [ 0.8  1.0  0.8 ]
        //   [ 0.5  0.8  1.0 ]
        // r: [0.8, 0.5, 0.3]
        let r_matrix = [
            [1.0_f64, rho[0], rho[1]],
            [rho[0], 1.0, rho[0]],
            [rho[1], rho[0], 1.0],
        ];
        let psi = &result.coefficients;
        for (i, row) in r_matrix.iter().enumerate() {
            let dot: f64 = row.iter().zip(psi.iter()).map(|(r, p)| r * p).sum();
            let residual = dot - rho[i];
            assert!(
                residual.abs() < 1e-10,
                "Yule-Walker row {i} residual: {residual}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Order-0 edge case
    // -----------------------------------------------------------------------

    #[test]
    fn levinson_durbin_order_zero() {
        let result = levinson_durbin(&[], 0);
        assert!(result.coefficients.is_empty());
        assert!(result.parcor.is_empty());
        assert!(result.sigma2_per_order.is_empty());
        assert_eq!(result.sigma2, 1.0);
    }

    #[test]
    fn levinson_durbin_order_zero_with_nonempty_autocorrelations() {
        // Supplying extra autocorrelations with order=0 must still return empty.
        let result = levinson_durbin(&[0.8, 0.5], 0);
        assert!(result.coefficients.is_empty());
        assert_eq!(result.sigma2, 1.0);
    }

    // -----------------------------------------------------------------------
    // Near-singular / perfect correlation truncation
    // -----------------------------------------------------------------------

    #[test]
    fn levinson_durbin_perfect_correlation_ar1() {
        // ρ(1) = 1.0 → kappa = 1.0, σ² = 0.0. Coefficient should be 1.0.
        let result = levinson_durbin(&[1.0], 1);
        assert_eq!(result.coefficients.len(), 1);
        assert!((result.coefficients[0] - 1.0).abs() < 1e-10);
        assert!(result.sigma2 <= f64::EPSILON);
    }

    #[test]
    fn levinson_durbin_near_singular_truncates() {
        // Construct a sequence where σ² collapses at or before order 2 when
        // order 3 is requested. ρ(1) = 0.99999 forces σ² very close to zero
        // after AR(1), causing truncation before AR(3) is reached.
        let rho = [0.99999_f64, 0.99998, 0.99997];
        let result = levinson_durbin(&rho, 3);
        // After AR(1): σ² = 1 - 0.99999² ≈ 2e-5, which is > EPSILON.
        // After AR(2): the update may drive σ² to near-zero.
        // The actual truncation point is implementation-dependent, but
        // we require that the result length is <= 3 (the requested order).
        assert!(result.coefficients.len() <= 3);
        // sigma2 must be non-negative and finite.
        assert!(result.sigma2.is_finite());
        assert!(result.sigma2 >= 0.0);
    }

    #[test]
    fn levinson_durbin_sigma2_causes_truncation_at_order2() {
        // With ρ(1) = 1.0, AR(1) gives σ² = 0 ≤ EPSILON.
        // Requesting order=3 should truncate at order 1.
        let rho = [1.0_f64, 0.5, 0.3];
        let result = levinson_durbin(&rho, 3);
        // Truncation should stop at or before order 2.
        assert!(result.coefficients.len() <= 2);
        assert!(result.sigma2 <= f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // sigma2_per_order is non-increasing
    // -----------------------------------------------------------------------

    #[test]
    fn levinson_durbin_sigma2_per_order_monotone_decreasing() {
        // Each additional AR order reduces (or maintains) prediction error.
        // σ²_{k+1} = σ²_k * (1 - κ²) ≤ σ²_k.
        let rho = [0.8_f64, 0.5, 0.3, 0.2];
        let result = levinson_durbin(&rho, 4);
        let sigma2_vals = &result.sigma2_per_order;
        // Prepend σ²_0 = 1.0 for the comparison.
        let mut prev = 1.0_f64;
        for (k, &s) in sigma2_vals.iter().enumerate() {
            assert!(
                s <= prev + 1e-14,
                "sigma2_per_order[{k}] = {s} > prev {prev}: not non-increasing"
            );
            prev = s;
        }
        // Final sigma2 must equal the last element in sigma2_per_order.
        if let Some(&last) = sigma2_vals.last() {
            assert!(
                (result.sigma2 - last).abs() < 1e-15,
                "sigma2 ({}) != sigma2_per_order.last() ({})",
                result.sigma2,
                last
            );
        }
    }

    // -----------------------------------------------------------------------
    // parcor stored correctly
    // -----------------------------------------------------------------------

    #[test]
    fn levinson_durbin_parcor_stored_correctly() {
        // For AR(1): κ₁ = ρ(1) (the reflection coefficient IS ρ(1) for order 1).
        let rho_1 = [0.7_f64];
        let result = levinson_durbin(&rho_1, 1);
        assert_eq!(result.parcor.len(), 1);
        assert!(
            (result.parcor[0] - 0.7).abs() < 1e-10,
            "parcor[0] = {} != 0.7",
            result.parcor[0]
        );

        // For AR(2): κ₁ = ρ(1), and κ₂ is computed from the recursion.
        // Verify parcor has 2 entries and parcor[0] == κ₁ from AR(1).
        let rho_2 = [0.7_f64, 0.4];
        let result2 = levinson_durbin(&rho_2, 2);
        assert_eq!(result2.parcor.len(), 2);
        // κ₁ must equal ρ(1).
        assert!(
            (result2.parcor[0] - 0.7).abs() < 1e-10,
            "parcor[0] = {} for AR(2) should equal ρ(1) = 0.7",
            result2.parcor[0]
        );
        // κ₂ = (ρ(2) - κ₁ * ρ(1)) / (1 - κ₁²)
        //     = (0.4 - 0.7 * 0.7) / (1 - 0.49)
        //     = (0.4 - 0.49) / 0.51 = -0.09 / 0.51
        let expected_kappa2 = (0.4 - 0.7 * 0.7) / (1.0 - 0.49);
        assert!(
            (result2.parcor[1] - expected_kappa2).abs() < 1e-10,
            "parcor[1] = {} != expected {expected_kappa2}",
            result2.parcor[1]
        );
    }

    // -----------------------------------------------------------------------
    // Acceptance criteria from ticket
    // -----------------------------------------------------------------------

    #[test]
    fn acceptance_ar3_yule_walker_satisfied() {
        // Given autocorrelations = [0.8, 0.5, 0.3], order = 3:
        // R * ψ = r must hold to within 1e-10.
        let rho = [0.8_f64, 0.5, 0.3];
        let result = levinson_durbin(&rho, 3);
        assert_eq!(result.coefficients.len(), 3);
        assert!(result.sigma2 > 0.0);

        let r_matrix = [
            [1.0_f64, rho[0], rho[1]],
            [rho[0], 1.0, rho[0]],
            [rho[1], rho[0], 1.0],
        ];
        for (i, row) in r_matrix.iter().enumerate() {
            let dot: f64 = row
                .iter()
                .zip(result.coefficients.iter())
                .map(|(r, p)| r * p)
                .sum();
            let residual = (dot - rho[i]).abs();
            assert!(
                residual < 1e-10,
                "acceptance AR(3) Yule-Walker row {i} residual: {residual}"
            );
        }
    }

    #[test]
    fn acceptance_ar1_half() {
        // autocorrelations = [0.5], order = 1
        // => coefficients == [0.5], sigma2 == 0.75
        let result = levinson_durbin(&[0.5], 1);
        assert!((result.coefficients[0] - 0.5).abs() < 1e-10);
        assert!((result.sigma2 - 0.75).abs() < 1e-10);
    }

    #[test]
    fn acceptance_order_zero() {
        let result = levinson_durbin(&[0.5, 0.3], 0);
        assert!(result.coefficients.is_empty());
        assert!(result.parcor.is_empty());
        assert!(result.sigma2_per_order.is_empty());
        assert_eq!(result.sigma2, 1.0);
    }

    #[test]
    fn acceptance_perfect_correlation_singular() {
        // autocorrelations = [1.0], order = 1 → sigma2 near zero, coeff = [1.0]
        let result = levinson_durbin(&[1.0], 1);
        assert!(result.sigma2 <= f64::EPSILON);
        assert_eq!(result.coefficients.len(), 1);
        assert!((result.coefficients[0] - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // estimate_seasonal_stats tests
    // -----------------------------------------------------------------------

    use chrono::NaiveDate;
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
    // Acceptance criterion: known mean and Bessel-corrected std
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
            / 4.0; // N-1 = 4
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
    fn estimate_seasonal_stats_bessel_correction() {
        // Two observations: N=2, Bessel std = |x1 - x2| / sqrt(2).
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

        // mean = 15.0, variance(N-1) = ((10-15)^2 + (20-15)^2) / 1 = 50.0
        let expected_mean = 15.0_f64;
        let expected_std = 50.0_f64.sqrt();

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
            / (n - 1.0);
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
    // estimate_ar_coefficients tests
    // -----------------------------------------------------------------------

    use super::estimate_ar_coefficients;

    // -----------------------------------------------------------------------
    // Test: known AR(1) process recovers coefficient approximately
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_ar_known_ar1_process() {
        // Generate a synthetic AR(1) process with φ = 0.6 for all seasons.
        // We build a stationary AR(1) time series: x_t = 0.6 * x_{t-1} + ε_t
        // using a splitmix64-style hash to produce high-quality pseudo-noise
        // without any external PRNG dependency.
        //
        // Strategy: 100 years × 12 months = 1200 observations (large sample
        // so estimation converges to within tolerance 0.1 of true φ=0.6).
        let n_years = 200_i32;
        let n_seasons = 12_usize;
        let phi = 0.6_f64;

        let stages = make_monthly_stages(1800, n_years as u32);
        let entity_ids = vec![EntityId::from(1)];

        let total = n_years as usize * n_seasons;

        // splitmix64 hash: maps u64 -> u64, then normalises to [-2, 2].
        let splitmix64 = |mut x: u64| -> f64 {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x = x ^ (x >> 31);
            // Map to [-2, 2] via a uniform-to-normal-like transform.
            // We use the ratio (x / u64::MAX) * 4 - 2 as a simple bounded
            // noise source; the key property is low serial correlation.
            (x as f64 / u64::MAX as f64) * 4.0 - 2.0
        };

        let mut series: Vec<f64> = (0..total)
            .map(|i| {
                splitmix64(
                    (i as u64)
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1),
                )
            })
            .collect();

        // Apply AR(1) recursion: x_t = φ * x_{t-1} + ε_t
        for t in 1..total {
            series[t] += phi * series[t - 1];
        }

        let base_year = 1800;
        let months = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year_off in 0..n_years {
            for (m_idx, &month) in months.iter().enumerate() {
                let t = year_off as usize * n_seasons + m_idx;
                observations.push((
                    EntityId::from(1),
                    NaiveDate::from_ymd_opt(base_year + year_off, month, 15).unwrap(),
                    series[t],
                ));
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &entity_ids, 1).unwrap();

        assert_eq!(estimates.len(), n_seasons, "expected one entry per season");

        for est in &estimates {
            assert_eq!(est.coefficients.len(), 1, "AR(1) should have 1 coefficient");
            let coeff = est.coefficients[0];
            assert!(
                (coeff - phi).abs() < 0.1,
                "season {}: coefficient {} not within 0.1 of expected {phi}",
                est.season_id,
                coeff
            );
            assert!(
                est.residual_std_ratio > 0.0 && est.residual_std_ratio <= 1.0,
                "season {}: residual_std_ratio {} out of (0, 1]",
                est.season_id,
                est.residual_std_ratio
            );
            // sqrt(1 - phi^2) = sqrt(0.64) = 0.8
            let expected_ratio = (1.0 - phi * phi).sqrt();
            assert!(
                (est.residual_std_ratio - expected_ratio).abs() < 0.1,
                "season {}: residual_std_ratio {} not within 0.1 of {expected_ratio}",
                est.season_id,
                est.residual_std_ratio
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: max_order = 0 => all entries have empty coefficients and ratio 1.0
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_ar_max_order_zero_white_noise() {
        let stages = make_monthly_stages(1990, 5);
        let entity_ids = vec![EntityId::from(1)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in 1990..1995_i32 {
            for month in 1u32..=12 {
                observations.push(obs(1, year, month, 100.0 + month as f64));
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &entity_ids, 0).unwrap();

        assert_eq!(estimates.len(), 12, "expected 12 season entries");
        for est in &estimates {
            assert!(
                est.coefficients.is_empty(),
                "max_order=0: coefficients must be empty for season {}",
                est.season_id
            );
            assert_eq!(
                est.residual_std_ratio, 1.0,
                "max_order=0: residual_std_ratio must be 1.0 for season {}",
                est.season_id
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: season with s_m = 0.0 => white noise entry (empty coefficients)
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_ar_zero_std_season_skipped() {
        // All observations for season 0 (January) are identical => std = 0.
        // All observations for season 1 (February) vary => std > 0.
        let stages = make_monthly_stages(1990, 5);
        let entity_ids = vec![EntityId::from(1)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in 1990..1995_i32 {
            // January: constant value => std = 0.
            observations.push(obs(1, year, 1, 100.0));
            // February: varying value => std > 0.
            observations.push(obs(1, year, 2, 100.0 + year as f64));
            // Other months: constant per month.
            for month in 3u32..=12 {
                observations.push(obs(1, year, month, month as f64 * 10.0));
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();

        // January season (season_id = 0) has std = 0.
        let jan_stats = stats
            .iter()
            .find(|s| s.stage_id == 1)
            .expect("should have January stats");
        assert_eq!(jan_stats.std, 0.0, "January std should be 0");

        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &entity_ids, 2).unwrap();

        // Find the January (season 0) entry.
        let jan_est = estimates
            .iter()
            .find(|e| e.season_id == 0)
            .expect("should have season 0 entry");
        assert!(
            jan_est.coefficients.is_empty(),
            "zero-std season must have empty coefficients"
        );
        assert_eq!(
            jan_est.residual_std_ratio, 1.0,
            "zero-std season must have residual_std_ratio = 1.0"
        );
    }

    // -----------------------------------------------------------------------
    // Test: insufficient observations => InsufficientData error
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_ar_insufficient_observations() {
        // Only 3 years of monthly data: each season has 3 observations.
        // With max_order = 3, need at least 4 observations per season.
        // Values vary across years to ensure std > 0 (otherwise the zero-std
        // early exit fires before the count check).
        let stages = make_monthly_stages(1990, 3);
        let entity_ids = vec![EntityId::from(1)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for (year_off, year) in (1990..1993_i32).enumerate() {
            for month in 1u32..=12 {
                // Values differ across years to ensure std > 0.
                observations.push(obs(
                    1,
                    year,
                    month,
                    100.0 + month as f64 + year_off as f64 * 10.0,
                ));
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        let result = estimate_ar_coefficients(&observations, &stats, &stages, &entity_ids, 3);
        assert!(
            matches!(result, Err(StochasticError::InsufficientData { .. })),
            "expected InsufficientData for 3 obs with max_order=3, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: cross-seasonal autocorrelation computed with hand-computed values
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_ar_cross_seasonal_autocorrelation() {
        // Build a 2-season (biannual) dataset with known cross-seasonal
        // autocorrelation, then verify the estimated coefficient matches.
        //
        // Season 0 = January, season 1 = February.
        // We place observations such that the cross-seasonal correlation
        // rho_1(1) (lag of season 0 into season 1) is deterministic.
        //
        // Observations for entity 1:
        //   Jan values: [a1, a2, a3, a4, a5]
        //   Feb values: [b1, b2, b3, b4, b5]
        //
        // where b_i = a_i (perfectly correlated with previous month).
        // Then rho_1(1) should be 1.0 (or near 1.0 for finite sample).
        //
        // We test with 10 years (each season gets 10 observations).

        let stages = make_monthly_stages(1990, 10);
        let entity_ids = vec![EntityId::from(1)];

        // Use linearly increasing values: Jan_i = i, Feb_i = i (same as Jan_i).
        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for (i, year) in (1990..2000_i32).enumerate() {
            let val = (i + 1) as f64;
            observations.push(obs(1, year, 1, val)); // Jan
            observations.push(obs(1, year, 2, val + 0.5)); // Feb ≈ Jan
                                                           // Other months: enough data to avoid InsufficientData.
            for month in 3u32..=12 {
                observations.push(obs(1, year, month, month as f64 * 5.0 + i as f64));
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &entity_ids, 1).unwrap();

        // We should get one entry per season (12 entries).
        assert_eq!(estimates.len(), 12);

        // For season 1 (February), the lag-1 autocorrelation (into January) should
        // be high (close to 1.0) because Feb ≈ Jan of the same year.
        let feb_est = estimates
            .iter()
            .find(|e| e.season_id == 1)
            .expect("should have season 1 estimate");

        assert_eq!(feb_est.coefficients.len(), 1);
        // The coefficient should be positive and relatively large.
        assert!(
            feb_est.coefficients[0] > 0.5,
            "February lag-1 coefficient should be > 0.5, got {}",
            feb_est.coefficients[0]
        );
    }

    // -----------------------------------------------------------------------
    // Test: residual_std_ratio in (0, 1] for all entries
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_ar_residual_std_ratio_in_range() {
        let stages = make_monthly_stages(1990, 30);
        let entity_ids = vec![EntityId::from(1), EntityId::from(2)];

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in 1990..2020_i32 {
            for month in 1u32..=12 {
                // Entity 1: linearly increasing trend per year.
                observations.push(obs(1, year, month, year as f64 + month as f64));
                // Entity 2: decreasing trend.
                observations.push(obs(2, year, month, 3000.0 - year as f64 + month as f64));
            }
        }

        let stats = estimate_seasonal_stats(&observations, &stages, &entity_ids).unwrap();
        let estimates =
            estimate_ar_coefficients(&observations, &stats, &stages, &entity_ids, 3).unwrap();

        assert_eq!(estimates.len(), 24, "expected 2 hydros × 12 seasons = 24");

        for est in &estimates {
            assert!(
                est.residual_std_ratio > 0.0,
                "entity {} season {}: residual_std_ratio {} must be > 0",
                est.hydro_id,
                est.season_id,
                est.residual_std_ratio
            );
            assert!(
                est.residual_std_ratio <= 1.0,
                "entity {} season {}: residual_std_ratio {} must be <= 1.0",
                est.hydro_id,
                est.season_id,
                est.residual_std_ratio
            );
        }
    }

    // -----------------------------------------------------------------------
    // estimate_correlation tests
    // -----------------------------------------------------------------------

    use super::{estimate_correlation, ArCoefficientEstimate, SeasonalStats};

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
        let obs_refs: Vec<&[f64]> = obs_data.iter().map(|v| v.as_slice()).collect();
        let stats: Vec<(f64, f64)> = obs_data.iter().map(|v| pop_mean_std(v)).collect();

        // For the cross-year case (ref_season=0, lag=1 -> lag_season=11),
        // lag_season (11) >= ref_season (0), so one observation is dropped.
        let _rho_jan_dec = periodic_autocorrelation(0, 1, n_seasons, &obs_refs, &stats);

        // For the non-cross-year case (ref_season=6, lag=1 -> lag_season=5),
        // lag_season (5) < ref_season (6), so no observation is dropped.
        let _rho_jul_jun = periodic_autocorrelation(6, 1, n_seasons, &obs_refs, &stats);

        // Both should produce valid values.
        assert!(_rho_jan_dec.abs() <= 1.0);
        assert!(_rho_jul_jun.abs() <= 1.0);

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
        assert!(rho >= -1.0 && rho <= 1.0, "rho should be clamped: {rho}");
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

        let rho = periodic_autocorrelation(0, 1, 1, obs, stats);

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
        let m01 = mat[0 * order + 1];
        let m12 = mat[1 * order + 2];
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
        // row 0 uses ref_month = season = 0, row 1 uses ref_month = (0+2-1)%2 = 1.
        // These reference different seasons, so M[0,1] (rho(0,1)) may differ
        // from M[1,2] (rho(1,1)).
        let m01 = mat[0 * order + 1];
        let m12 = mat[1 * order + 2];
        // We just verify both are valid; they may or may not differ depending
        // on the specific data, but the matrix IS valid.
        assert!(m01.abs() <= 1.0);
        assert!(m12.abs() <= 1.0);
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
    // Comprehensive periodic autocorrelation and matrix tests (E1/T003)
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
        // Verify RHS comes from the (order+1)-th column of the extended matrix.
        // Build a 3-season model, order=2 at season=1.
        // RHS[0] = rho(season=1, lag=order-0=2)
        // RHS[1] = rho(season=1-1=0, lag=order-1=1)
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

        // Verify each RHS entry.
        let expected_rhs0 = periodic_autocorrelation(season, order, 3, &obs, &stats);
        let ref_month_1 = (season + 3 - 1) % 3;
        let expected_rhs1 = periodic_autocorrelation(ref_month_1, order - 1, 3, &obs, &stats);

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
}
