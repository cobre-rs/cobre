//! AR coefficient estimation results and the shared season-lookup data
//! preparation used by the season-aware estimation functions.

use std::collections::{HashMap, HashSet};

use chrono::NaiveDate;

use cobre_core::{
    EntityId,
    scenario::AnnualComponent,
    temporal::{SeasonMap, Stage},
};

use super::seasonal_stats::{SeasonalStats, build_stage_index};
use crate::StochasticError;

// ---------------------------------------------------------------------------
// AR coefficient estimation
// ---------------------------------------------------------------------------

/// AR coefficient estimation result for a single (entity, season) pair.
///
/// Produced by [`estimate_ar_coefficients_with_season_map`] and consumed by the assembly
/// layer that writes AR coefficient rows to persistent storage.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct ArCoefficientEstimate {
    /// Entity identifier.
    pub hydro_id: EntityId,
    /// Season ID (maps back to `stage_id` via the stage table).
    pub season_id: usize,
    /// Standardized AR coefficients ψ*₁..ψ*ₚ (Yule-Walker output); empty at order 0.
    pub coefficients: Vec<f64>,
    /// PAR(p)-A annual-component triple; `None` for classical PAR(p). All three
    /// sub-fields are present together by construction, never split.
    pub annual: Option<AnnualComponent>,
}

// ---------------------------------------------------------------------------
// Shared data-preparation helpers for season-aware estimation functions
// ---------------------------------------------------------------------------

/// Pre-computed lookups for season-aware PAR estimation functions.
///
/// Both [`estimate_ar_coefficients_with_season_map`] and
/// [`estimate_correlation_with_season_map`](super::correlation::estimate_correlation_with_season_map)
/// require the same date-to-season mapping, stats lookups, and per-entity
/// observation indices. This struct is built once and shared to avoid code
/// duplication.
pub(super) struct SeasonLookups<'a> {
    pub(super) stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)>,
    pub(super) stats_lookup: HashMap<(EntityId, usize), &'a SeasonalStats>,
    pub(super) entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>>,
    pub(super) entity_date_index: HashMap<EntityId, HashMap<NaiveDate, usize>>,
    pub(super) n_seasons: usize,
}

/// Build the shared season lookups from observations, seasonal stats, and stages.
///
/// When present, `season_map` supplies the **true** cycle length as the
/// lag-season wrap modulus; inferring `max(season_id)+1` from in-window seasons
/// would map a partial-year study's out-of-window lag months to the wrong season
/// (a no-op for full-year studies, where the two agree).
pub(super) fn build_season_lookups<'a>(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &'a [SeasonalStats],
    stages: &[Stage],
    season_map: Option<&SeasonMap>,
) -> SeasonLookups<'a> {
    let stage_index = build_stage_index(stages);

    let stage_id_to_season: HashMap<i32, usize> = stage_index
        .iter()
        .map(|&(_, _, stage_id, season_id)| (stage_id, season_id))
        .collect();

    let stats_lookup: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
        .iter()
        .filter_map(|s| {
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(((s.entity_id, season_id), s))
        })
        .collect();

    let n_seasons: usize = season_map.map_or_else(
        || {
            stage_index
                .iter()
                .map(|&(_, _, _, sid)| sid + 1)
                .max()
                .unwrap_or(0)
        },
        |sm| sm.seasons.len(),
    );

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

/// Produce white-noise (order-0) AR estimates (empty coefficients) for every
/// `(entity, season)` pair in `seasonal_stats` that belongs to `hydro_ids`.
///
/// `observations`, `stages`, and `season_map` are accepted for signature parity
/// with the order-selecting path but unused here.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when `max_order > 0`; callers
/// requiring AR order selection must use
/// [`estimate_periodic_ar_coefficients`](super::estimate_periodic_ar_coefficients)
/// and the periodic Yule-Walker path instead.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::{estimate_seasonal_stats_with_season_map, estimate_ar_coefficients_with_season_map};
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
/// let stats = estimate_seasonal_stats_with_season_map(&obs, &stages_vec, &entity_ids, None).unwrap();
/// // order-0 produces white-noise estimates (empty coefficients, ratio = 1.0)
/// let estimates = estimate_ar_coefficients_with_season_map(&obs, &stats, &stages_vec, &entity_ids, 0, None).unwrap();
/// assert_eq!(estimates.len(), 2);
/// assert!(estimates[0].coefficients.is_empty());
/// ```
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
                annual: None,
            })
        })
        .collect();

    // Canonical (hydro_id, season_id) order: output is independent of the
    // seasonal_stats slice order, upholding declaration-order bit-determinism.
    result.sort_unstable_by_key(|e| (e.hydro_id.0, e.season_id));
    Ok(result)
}
