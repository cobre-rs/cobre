//! Cross-entity residual correlation estimation for fitted PAR models.

use std::collections::{BTreeMap, HashMap};

use chrono::NaiveDate;
use rayon::prelude::*;

use cobre_core::{
    EntityId,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
        CorrelationScheduleEntry,
    },
    temporal::{SeasonMap, Stage},
};

use super::ar_coefficients::{ArCoefficientEstimate, SeasonLookups, build_season_lookups};
use super::cycle_positions::CyclePositions;
use super::seasonal_stats::{SeasonalStats, find_season_for_date};
use crate::StochasticError;

// ---------------------------------------------------------------------------
// Correlation estimation
// ---------------------------------------------------------------------------

/// Minimum paired observations per season for a seasonal matrix; seasons below
/// this fall back to the pooled "default" profile.
const MIN_CORRELATION_PAIRS: usize = 30;

/// Estimate the cross-entity residual correlation matrix from historical observations.
///
/// After a PAR(p) model is fitted (via
/// [`estimate_seasonal_stats_with_season_map`](super::estimate_seasonal_stats_with_season_map) and
/// [`estimate_ar_coefficients_with_season_map`](super::estimate_ar_coefficients_with_season_map)), this function computes the standardized
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
/// The returned [`CorrelationModel`] carries a single `"default"` profile over
/// all entities in canonical `hydro_ids` order. The function does **not** enforce
/// positive-semidefiniteness — the downstream spectral decomposition handles
/// rank-deficient and non-PD matrices.
///
/// On a single-resolution `season_map`, a season's lags are the seasons before
/// it in calendar order, whatever their ids; profile names and the schedule
/// carry the map's own season ids.
///
/// # Parameters
///
/// - `observations` — `(entity_id, date, value)` triples sorted by `(entity_id, date)`.
/// - `hydro_ids` — canonical sorted entity IDs; determines matrix row/column order.
/// - `season_map` — fallback to map dates outside `stages` to a season.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when `seasonal_stats` is empty
/// but `hydro_ids` is non-empty (inconsistent inputs).
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::{
///     estimate_seasonal_stats_with_season_map, estimate_ar_coefficients_with_season_map,
///     estimate_correlation_with_season_map,
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
/// let stats = estimate_seasonal_stats_with_season_map(&obs, &stages_vec, &hydro_ids, None).unwrap();
/// let estimates = estimate_ar_coefficients_with_season_map(&obs, &stats, &stages_vec, &hydro_ids, 0, None).unwrap();
/// let corr = estimate_correlation_with_season_map(&obs, &estimates, &stats, &stages_vec, &hydro_ids, None).unwrap();
/// assert!(corr.profiles.contains_key("default"));
/// assert_eq!(corr.profiles["default"].groups[0].matrix.len(), 1);
/// ```
pub fn estimate_correlation_with_season_map(
    observations: &[(EntityId, NaiveDate, f64)],
    ar_estimates: &[ArCoefficientEstimate],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<CorrelationModel, StochasticError> {
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

    let residual_correlations = match CyclePositions::new(season_map) {
        None => residual_correlations_in_cycle_order(
            observations,
            ar_estimates,
            seasonal_stats,
            stages,
            hydro_ids,
            season_map,
            None,
        ),
        Some(cycle_positions) => {
            let season_map =
                season_map.map(|season_map| cycle_positions.relabel_season_map(season_map));
            let in_positions = residual_correlations_in_cycle_order(
                observations,
                &cycle_positions.estimates_to_positions(ar_estimates),
                seasonal_stats,
                &cycle_positions.relabel_stages(stages),
                hydro_ids,
                season_map.as_ref(),
                Some(cycle_positions.raw_ids()),
            );
            ResidualCorrelations {
                seasonal_matrices: cycle_positions.seasons_to_raw(in_positions.seasonal_matrices),
                ..in_positions
            }
        }
    };

    Ok(assemble_seasonal_correlation_model(
        hydro_ids,
        &residual_correlations.pooled_matrix,
        &residual_correlations.seasonal_matrices,
        stages,
        residual_correlations.n_seasons,
    ))
}

struct ResidualCorrelations {
    pooled_matrix: Vec<f64>,
    seasonal_matrices: HashMap<usize, Vec<f64>>,
    n_seasons: usize,
}

fn residual_correlations_in_cycle_order(
    observations: &[(EntityId, NaiveDate, f64)],
    ar_estimates: &[ArCoefficientEstimate],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
    season_labels: Option<&[usize]>,
) -> ResidualCorrelations {
    let lookups = build_season_lookups(observations, seasonal_stats, stages, season_map);

    let ar_lookup: HashMap<(EntityId, usize), &ArCoefficientEstimate> = ar_estimates
        .iter()
        .map(|e| ((e.hydro_id, e.season_id), e))
        .collect();

    let per_season_residuals = compute_hydro_residuals(&lookups, &ar_lookup, hydro_ids, season_map);

    let identity_labels: Vec<usize> = (0..lookups.n_seasons).collect();
    warn_degenerate_hydros(
        &lookups,
        hydro_ids,
        &per_season_residuals,
        season_labels.unwrap_or(&identity_labels),
    );

    let pooled_residuals = flatten_residuals(&per_season_residuals);
    let pooled_matrix = compute_pearson_correlation_matrix(&pooled_residuals);
    let seasonal_matrices = compute_seasonal_matrices(&per_season_residuals, lookups.n_seasons);

    ResidualCorrelations {
        pooled_matrix,
        seasonal_matrices,
        n_seasons: lookups.n_seasons,
    }
}

/// Compute standardized AR innovation residuals (see
/// `estimate_correlation_with_season_map`'s residual formula) for each hydro, keyed by
/// `season_id`; one entry per position in `hydro_ids`. Each season's vec is date-sorted.
fn compute_hydro_residuals(
    lookups: &SeasonLookups<'_>,
    ar_lookup: &HashMap<(EntityId, usize), &ArCoefficientEstimate>,
    hydro_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Vec<HashMap<usize, Vec<(NaiveDate, f64)>>> {
    // Determinism: `collect()` reassembles per-hydro maps in canonical
    // `hydro_ids` order, and the sequential `ar_sum` lag accumulation is
    // bit-identical to a single-threaded pass — thread scheduling cannot change
    // the output.
    hydro_ids
        .par_iter()
        .map(|&hydro_id| {
            let mut residuals: HashMap<usize, Vec<(NaiveDate, f64)>> = HashMap::new();

            let Some(all_obs) = lookups.entity_obs.get(&hydro_id) else {
                return residuals;
            };
            let Some(date_index) = lookups.entity_date_index.get(&hydro_id) else {
                return residuals;
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
                residuals
                    .entry(season_id)
                    .or_default()
                    .push((date, epsilon));
            }

            residuals
        })
        .collect()
}

/// Flatten per-season residuals into a single date-keyed map per hydro.
///
/// Dates are unique per hydro per season (each observation maps to exactly one
/// season), so no collisions occur when merging season maps.
fn flatten_residuals(
    per_season: &[HashMap<usize, Vec<(NaiveDate, f64)>>],
) -> Vec<HashMap<NaiveDate, f64>> {
    per_season
        .iter()
        .map(|seasons| {
            let mut flat = HashMap::new();
            for date_values in seasons.values() {
                flat.extend(date_values.iter().copied());
            }
            flat
        })
        .collect()
}

/// Emit diagnostic warnings for statistically degenerate hydros. Informational
/// only — no hydros are excluded; the spectral decomposition handles near-zero
/// eigenvalues. Scans season indices `0..season_labels.len()` and names index
/// `s` as `season_labels[s]`.
fn warn_degenerate_hydros(
    lookups: &SeasonLookups<'_>,
    hydro_ids: &[EntityId],
    per_season_residuals: &[HashMap<usize, Vec<(NaiveDate, f64)>>],
    season_labels: &[usize],
) {
    for (hidx, &hydro_id) in hydro_ids.iter().enumerate() {
        let Some(all_obs) = lookups.entity_obs.get(&hydro_id) else {
            continue;
        };

        let mut obs_by_season: HashMap<usize, Vec<f64>> = HashMap::new();
        for &(date, value) in all_obs {
            if let Some(season_id) = find_season_for_date(&lookups.stage_index, date) {
                obs_by_season.entry(season_id).or_default().push(value);
            }
        }

        for (season_id, &season_label) in season_labels.iter().enumerate() {
            let Some(vals) = obs_by_season.get(&season_id) else {
                continue;
            };
            if vals.len() < 2 {
                continue;
            }

            #[allow(clippy::cast_precision_loss)]
            let neg_frac = vals.iter().filter(|&&v| v < 0.0).count() as f64 / vals.len() as f64;
            if neg_frac > 0.5 {
                tracing::warn!(
                    hydro_id = hydro_id.0,
                    season = season_label,
                    negative_fraction = neg_frac,
                    "hydro has majority negative observations in season \
                     (included in correlation; spectral decomposition handles this)"
                );
            }

            let first = vals[0];
            if vals.iter().all(|&v| (v - first).abs() < f64::EPSILON) {
                tracing::warn!(
                    hydro_id = hydro_id.0,
                    season = season_label,
                    value = first,
                    "hydro has constant series in season \
                     (included in correlation; near-zero eigenvalue expected)"
                );
                // Constant series implies near-zero residual variance; skip the
                // redundant residual-variance warning below.
                continue;
            }

            if let Some(residuals) = per_season_residuals
                .get(hidx)
                .and_then(|m| m.get(&season_id))
                .filter(|r| r.len() >= 2)
            {
                let r_vals: Vec<f64> = residuals.iter().map(|&(_, v)| v).collect();
                #[allow(clippy::cast_precision_loss)]
                let r_mean = r_vals.iter().sum::<f64>() / r_vals.len() as f64;
                #[allow(clippy::cast_precision_loss)]
                let r_std = (r_vals.iter().map(|v| (v - r_mean).powi(2)).sum::<f64>()
                    / (r_vals.len() - 1) as f64)
                    .sqrt();
                if r_std < 1e-8 {
                    tracing::warn!(
                        hydro_id = hydro_id.0,
                        season = season_label,
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
/// Seasons whose minimum paired-observation count is below [`MIN_CORRELATION_PAIRS`]
/// are omitted and fall back to the pooled "default" profile. All hydros participate
/// regardless of degeneracy; rank-deficient matrices are acceptable (spectral
/// decomposition handles them).
fn compute_seasonal_matrices(
    per_season_residuals: &[HashMap<usize, Vec<(NaiveDate, f64)>>],
    n_seasons: usize,
) -> HashMap<usize, Vec<f64>> {
    let mut result = HashMap::new();

    for season_id in 0..n_seasons {
        let season_residuals: Vec<&[(NaiveDate, f64)]> = per_season_residuals
            .iter()
            .map(|hydro_seasons| hydro_seasons.get(&season_id).map_or(&[][..], Vec::as_slice))
            .collect();

        let (matrix, min_pairs) = compute_seasonal_pearson_matrix(&season_residuals);

        if min_pairs < MIN_CORRELATION_PAIRS {
            continue;
        }

        result.insert(season_id, matrix);
    }

    result
}

/// Pearson correlation coefficient from a pair's overlapping standardized
/// residuals, Bessel-corrected (N-1). Callers guarantee `pairs.len() >= 2` and
/// that `pairs` is already in the deterministic accumulation order.
fn pearson_rho(pairs: &[(f64, f64)]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let np = pairs.len() as f64;
    let mean_i = pairs.iter().map(|(ei, _)| ei).sum::<f64>() / np;
    let mean_j = pairs.iter().map(|(_, ej)| ej).sum::<f64>() / np;

    let mut cov = 0.0_f64;
    let mut var_i = 0.0_f64;
    let mut var_j = 0.0_f64;
    for (ei, ej) in pairs {
        let di = ei - mean_i;
        let dj = ej - mean_j;
        cov += di * dj;
        var_i += di * di;
        var_j += dj * dj;
    }

    let denom = np - 1.0;
    let std_i = (var_i / denom).sqrt();
    let std_j = (var_j / denom).sqrt();

    if std_i < f64::EPSILON || std_j < f64::EPSILON {
        0.0
    } else {
        (cov / denom / (std_i * std_j)).clamp(-1.0, 1.0)
    }
}

/// Compute the `n_hydros` x `n_hydros` Pearson correlation matrix from residuals,
/// using the standard formula with Bessel correction (N-1).
///
/// Determinism: pairs are summed in `NaiveDate` order, making every partial sum
/// independent of both `HashMap` traversal order and which series is row vs column
/// — so the result is invariant to the declaration order of the input series.
pub(super) fn compute_pearson_correlation_matrix(
    hydro_residuals: &[HashMap<NaiveDate, f64>],
) -> Vec<f64> {
    let n = hydro_residuals.len();
    let mut matrix = vec![0.0_f64; n * n];

    for i in 0..n {
        matrix[i * n + i] = 1.0;

        for j in (i + 1)..n {
            let r_i = &hydro_residuals[i];
            let r_j = &hydro_residuals[j];

            let mut pairs: Vec<(NaiveDate, f64, f64)> = r_i
                .iter()
                .filter_map(|(&date, &ei)| r_j.get(&date).map(|&ej| (date, ei, ej)))
                .collect();

            if pairs.len() < 2 {
                matrix[i * n + j] = 0.0;
                matrix[j * n + i] = 0.0;
                continue;
            }

            // Dates are unique within a series: the order is total, so the unstable sort is deterministic.
            pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            let rho = pearson_rho(
                &pairs
                    .iter()
                    .map(|&(_, ei, ej)| (ei, ej))
                    .collect::<Vec<_>>(),
            );
            matrix[i * n + j] = rho;
            matrix[j * n + i] = rho;
        }
    }

    matrix
}

/// Compute the `n_hydros` x `n_hydros` Pearson correlation matrix from
/// date-sorted residual slices, plus the minimum per-pair overlap count
/// [`compute_seasonal_matrices`] gates on.
///
/// Determinism: merge-walk visits shared dates in ascending `NaiveDate` order,
/// matching the accumulation order [`compute_pearson_correlation_matrix`]
/// produces, so results are bit-identical and invariant to input series order.
fn compute_seasonal_pearson_matrix(hydro_residuals: &[&[(NaiveDate, f64)]]) -> (Vec<f64>, usize) {
    let n = hydro_residuals.len();
    let mut matrix = vec![0.0_f64; n * n];

    if n <= 1 {
        let min_pairs = hydro_residuals.first().map_or(0, |r| r.len());
        if n == 1 {
            matrix[0] = 1.0;
        }
        return (matrix, min_pairs);
    }

    let mut min_pairs = usize::MAX;

    for i in 0..n {
        matrix[i * n + i] = 1.0;

        for j in (i + 1)..n {
            let r_i = hydro_residuals[i];
            let r_j = hydro_residuals[j];

            let mut pairs: Vec<(f64, f64)> = Vec::new();
            let (mut a, mut b) = (0usize, 0usize);
            while a < r_i.len() && b < r_j.len() {
                match r_i[a].0.cmp(&r_j[b].0) {
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                    std::cmp::Ordering::Equal => {
                        pairs.push((r_i[a].1, r_j[b].1));
                        a += 1;
                        b += 1;
                    }
                }
            }

            min_pairs = min_pairs.min(pairs.len());

            if pairs.len() < 2 {
                matrix[i * n + j] = 0.0;
                matrix[j * n + i] = 0.0;
                continue;
            }

            let rho = pearson_rho(&pairs);
            matrix[i * n + j] = rho;
            matrix[j * n + i] = rho;
        }
    }

    (matrix, min_pairs)
}

/// Assemble a multi-profile [`CorrelationModel`].
///
/// The pooled matrix is always the `"default"` profile; when `n_seasons <= 1`
/// it is the only profile. Otherwise each season passing the minimum-sample
/// check adds a `"season_XX"` profile (zero-padded to the widest season index)
/// and the schedule maps each stage to its season's profile.
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

    // CorrelationGroup.matrix (cobre-core) is AoS, not the flat row-major buffer.
    let flat_to_aos = |flat: &[f64]| -> Vec<Vec<f64>> {
        (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect()
    };

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

    if n_seasons <= 1 {
        return CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: Vec::new(),
        };
    }

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

    let mut schedule: Vec<CorrelationScheduleEntry> = stages
        .iter()
        .filter_map(|stage| {
            let season_id = stage.season_id?;
            seasonal_matrices
                .contains_key(&season_id)
                .then(|| CorrelationScheduleEntry {
                    stage_id: stage.id,
                    profile_name: format!("season_{season_id:0>width$}"),
                })
        })
        .collect();
    schedule.sort_by_key(|e| e.stage_id);

    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use chrono::{Datelike, NaiveDate};
    use cobre_core::scenario::CorrelationModel;
    use cobre_core::{EntityId, SeasonMap};
    use tracing::field::{Field, Visit};
    use tracing::{Event, Level, Metadata, Subscriber, span};

    use super::{compute_seasonal_pearson_matrix, estimate_correlation_with_season_map};
    use crate::par::fitting::cycle_positions::twin_fixtures::{
        TwinMaps, calendar_history, calendar_stages, permuted_quarterly_twins, sparse_ring_twins,
    };
    use crate::par::fitting::{
        ArEstimationConfig, estimate_ar_coefficients_with_selection,
        estimate_seasonal_stats_with_season_map,
    };
    use crate::test_support::sparse_ring_season_map;

    /// Pre-change oracle: the `HashMap`-intersection + per-pair-sort algorithm
    /// `compute_seasonal_pearson_matrix`'s date-sorted merge-walk replaces.
    fn oracle_min_pairs_and_matrix(residuals: &[HashMap<NaiveDate, f64>]) -> (usize, Vec<f64>) {
        let n = residuals.len();
        let min_pairs = if n <= 1 {
            residuals.first().map_or(0, HashMap::len)
        } else {
            (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .map(|(i, j)| {
                    residuals[i]
                        .keys()
                        .filter(|d| residuals[j].contains_key(*d))
                        .count()
                })
                .min()
                .unwrap_or(0)
        };

        let mut matrix = vec![0.0_f64; n * n];
        for i in 0..n {
            matrix[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let r_i = &residuals[i];
                let r_j = &residuals[j];
                let mut pairs: Vec<(NaiveDate, f64, f64)> = r_i
                    .iter()
                    .filter_map(|(&date, &ei)| r_j.get(&date).map(|&ej| (date, ei, ej)))
                    .collect();
                if pairs.len() < 2 {
                    continue;
                }
                pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                #[allow(clippy::cast_precision_loss)]
                let np = pairs.len() as f64;
                let mean_i = pairs.iter().map(|(_, ei, _)| ei).sum::<f64>() / np;
                let mean_j = pairs.iter().map(|(_, _, ej)| ej).sum::<f64>() / np;
                let mut cov = 0.0_f64;
                let mut var_i = 0.0_f64;
                let mut var_j = 0.0_f64;
                for (_, ei, ej) in &pairs {
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
        (min_pairs, matrix)
    }

    fn date(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2020, 1, day).unwrap()
    }

    /// On a fixture with partial date overlap across three hydros
    /// (A has all 10 days, B is missing the first and last, C holds only odd
    /// days), the date-sorted merge-walk in `compute_seasonal_pearson_matrix`
    /// must reproduce both the correlation matrix and the min-pair count the
    /// pre-change `HashMap` intersection produced, bit-for-bit.
    #[test]
    fn seasonal_pearson_matrix_merge_walk_matches_hashmap_intersection_oracle() {
        let a_vals = [1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 2.5, -2.5, 1.0, -1.0];
        let a: HashMap<NaiveDate, f64> = (1..=10)
            .map(|day| (date(day), a_vals[(day - 1) as usize]))
            .collect();

        let b_vals = [0.5, -0.5, 1.0, -1.0, 0.5, -0.5, 1.0, -1.0];
        let b: HashMap<NaiveDate, f64> = (2..=9)
            .map(|day| (date(day), b_vals[(day - 2) as usize]))
            .collect();

        let c_vals = [2.0, 1.0, -1.0, -2.0, 0.5];
        let c: HashMap<NaiveDate, f64> = [1u32, 3, 5, 7, 9]
            .into_iter()
            .zip(c_vals)
            .map(|(day, v)| (date(day), v))
            .collect();

        let (oracle_min_pairs, oracle_matrix) =
            oracle_min_pairs_and_matrix(&[a.clone(), b.clone(), c.clone()]);

        let sorted_vec = |m: &HashMap<NaiveDate, f64>| -> Vec<(NaiveDate, f64)> {
            let mut v: Vec<(NaiveDate, f64)> = m.iter().map(|(&d, &v)| (d, v)).collect();
            v.sort_unstable_by_key(|&(d, _)| d);
            v
        };
        let a_sorted = sorted_vec(&a);
        let b_sorted = sorted_vec(&b);
        let c_sorted = sorted_vec(&c);
        let slices: [&[(NaiveDate, f64)]; 3] = [&a_sorted, &b_sorted, &c_sorted];

        let (matrix, min_pairs) = compute_seasonal_pearson_matrix(&slices);

        assert_eq!(
            min_pairs, oracle_min_pairs,
            "min pair count must match the HashMap-intersection oracle"
        );
        assert_eq!(matrix.len(), oracle_matrix.len());
        for (got, want) in matrix.iter().zip(&oracle_matrix) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "matrix entry must be bit-identical to the HashMap-intersection oracle"
            );
        }
    }

    /// The single-series special case (no pairs to intersect) must still
    /// report its own residual count as `min_pairs`, matching the oracle's
    /// `n <= 1` branch.
    #[test]
    fn seasonal_pearson_matrix_single_hydro_reports_its_own_count() {
        let solo: HashMap<NaiveDate, f64> =
            (1..=5).map(|day| (date(day), f64::from(day))).collect();
        let (oracle_min_pairs, oracle_matrix) =
            oracle_min_pairs_and_matrix(std::slice::from_ref(&solo));

        let solo_sorted: Vec<(NaiveDate, f64)> = {
            let mut v: Vec<(NaiveDate, f64)> = solo.iter().map(|(&d, &v)| (d, v)).collect();
            v.sort_unstable_by_key(|&(d, _)| d);
            v
        };
        let slices: [&[(NaiveDate, f64)]; 1] = [&solo_sorted];

        let (matrix, min_pairs) = compute_seasonal_pearson_matrix(&slices);

        assert_eq!(min_pairs, oracle_min_pairs);
        assert_eq!(matrix, oracle_matrix);
    }

    const HYDROS: [EntityId; 2] = [EntityId(1), EntityId(2)];

    fn correlation_of_calendar_history(
        season_map: &SeasonMap,
        calendar_ids: &[usize],
    ) -> CorrelationModel {
        let stages = calendar_stages(season_map, calendar_ids);
        let history = calendar_history(season_map, calendar_ids, &HYDROS, 30);
        let stats =
            estimate_seasonal_stats_with_season_map(&history, &stages, &HYDROS, Some(season_map))
                .unwrap();
        let (estimates, _) = estimate_ar_coefficients_with_selection(
            &history,
            &stats,
            &stages,
            &HYDROS,
            &ArEstimationConfig {
                max_order: 2,
                max_coeff_magnitude: None,
                season_map: Some(season_map),
                use_annual_component: false,
            },
        )
        .unwrap();
        estimate_correlation_with_season_map(
            &history,
            &estimates,
            &stats,
            &stages,
            &HYDROS,
            Some(season_map),
        )
        .unwrap()
    }

    fn scheduled_matrix_bits(model: &CorrelationModel, stage_id: i32) -> Vec<Vec<u64>> {
        let profile = model
            .schedule
            .iter()
            .find(|entry| entry.stage_id == stage_id)
            .map_or("default", |entry| entry.profile_name.as_str());
        model.profiles[profile].groups[0]
            .matrix
            .iter()
            .map(|row| row.iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    fn assert_correlation_equals_twin_correlation(twins: &TwinMaps) -> CorrelationModel {
        let model = correlation_of_calendar_history(&twins.map, &twins.map_ids());
        let twin_model = correlation_of_calendar_history(&twins.twin, &twins.twin_ids());
        assert!(
            !twin_model.schedule.is_empty(),
            "the twin must schedule at least one seasonal profile"
        );
        for stage_id in (0_i32..).take(twins.ids.len()) {
            assert_eq!(
                scheduled_matrix_bits(&model, stage_id),
                scheduled_matrix_bits(&twin_model, stage_id),
                "stage {stage_id}"
            );
        }
        model
    }

    #[test]
    fn seasonal_correlation_on_a_sparse_custom_map_equals_the_one_on_its_dense_twin() {
        let model = assert_correlation_equals_twin_correlation(&sparse_ring_twins());
        for profile in ["season_12", "season_13"] {
            assert!(
                model.profiles.contains_key(profile),
                "missing {profile} in {:?}",
                model.profiles.keys().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn seasonal_correlation_on_an_out_of_calendar_order_map_equals_the_one_on_its_calendar_ordered_twin()
     {
        assert_correlation_equals_twin_correlation(&permuted_quarterly_twins());
    }

    /// The `season` field of every warning event.
    #[derive(Clone, Default)]
    struct WarnedSeasons(Arc<Mutex<Vec<u64>>>);

    impl Subscriber for WarnedSeasons {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            *metadata.level() <= Level::WARN
        }

        fn new_span(&self, _attrs: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

        fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            struct SeasonField(Option<u64>);
            impl Visit for SeasonField {
                fn record_u64(&mut self, field: &Field, value: u64) {
                    if field.name() == "season" {
                        self.0 = Some(value);
                    }
                }

                fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
            }
            if *event.metadata().level() == Level::WARN {
                let mut season = SeasonField(None);
                event.record(&mut season);
                if let Some(season) = season.0 {
                    self.0.lock().unwrap().push(season);
                }
            }
        }

        fn enter(&self, _span: &span::Id) {}

        fn exit(&self, _span: &span::Id) {}
    }

    #[test]
    fn degenerate_season_warnings_name_the_raw_season_id_on_a_relabeled_map() {
        let season_map = sparse_ring_season_map();
        let stages = calendar_stages(&season_map, &[0, 1, 2, 12, 13]);
        let hydro_ids = [EntityId(1)];
        let observations: Vec<(EntityId, NaiveDate, f64)> = stages
            .iter()
            .flat_map(|stage| {
                let values = if stage.season_id == Some(12) {
                    [50.0, 50.0]
                } else {
                    [100.0, 110.0]
                };
                let mid_season = stage.start_date.with_day(15).unwrap();
                [
                    (hydro_ids[0], stage.start_date, values[0]),
                    (hydro_ids[0], mid_season, values[1]),
                ]
            })
            .collect();
        let stats = estimate_seasonal_stats_with_season_map(
            &observations,
            &stages,
            &hydro_ids,
            Some(&season_map),
        )
        .unwrap();

        let warned = WarnedSeasons::default();
        tracing::subscriber::with_default(warned.clone(), || {
            estimate_correlation_with_season_map(
                &observations,
                &[],
                &stats,
                &stages,
                &hydro_ids,
                Some(&season_map),
            )
            .unwrap();
        });

        assert_eq!(*warned.0.lock().unwrap(), vec![12]);
    }
}
