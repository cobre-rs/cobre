//! Numeric PAR(p) parameter-fitting core: order selection, contribution-based
//! reduction, and report assembly, driving the per-season `fitting/` primitives.
//!
//! Public entry points: [`estimate_ar_coefficients_with_selection`] (classical
//! and annual dispatch) and [`build_estimation_report`]. The core fails only with
//! [`StochasticError`] and never touches case-loading or row types — that
//! orchestration lives in the I/O shell.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{Datelike, NaiveDate};
use cobre_core::EntityId;
use cobre_core::SeasonMap;
use cobre_core::Stage;
use cobre_core::scenario::AnnualComponent;
use rayon::prelude::*;

use crate::StochasticError;
use crate::par::contribution::{
    check_negative_contributions, compute_contributions, find_max_valid_order, has_negative_phi1,
};
use crate::par::fitting::cycle_positions::CyclePositions;
use crate::par::fitting::estimate_ar_coefficients_with_season_map;
use crate::par::fitting::{
    AnnualSeasonalStats, ArCoefficientEstimate, SeasonalStats, conditional_facp_partitioned,
    estimate_annual_seasonal_stats, estimate_periodic_ar_annual_coefficients,
    estimate_periodic_ar_coefficients, find_season_for_date, periodic_pacf, select_order_pacf,
    select_order_pacf_annual,
};
use crate::season_cast::observation_occurrence_year;

/// Reason a season's AR order was reduced during estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionReason {
    /// Coefficient exceeds the magnitude-bound safety threshold.
    MagnitudeBound,
    /// First AR coefficient (`phi_1`) is negative, contradicting
    /// hydrological persistence.
    Phi1Negative,
    /// Contribution analysis detected negative entries at one or more lags.
    NegativeContribution,
}

impl ReductionReason {
    /// Stable string tag for diagnostic output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MagnitudeBound => "magnitude_bound",
            Self::Phi1Negative => "phi1_negative",
            Self::NegativeContribution => "negative_contribution",
        }
    }
}

/// A single AR-order reduction event.
#[derive(Debug, Clone)]
pub struct ContributionReduction {
    /// Season where the reduction occurred.
    pub season_id: usize,
    /// Order before reduction.
    pub original_order: usize,
    /// Order after reduction (max valid order from the contribution check).
    pub reduced_order: usize,
    /// Contribution values at the original order; empty for magnitude/`phi_1` reductions.
    pub contributions: Vec<f64>,
    /// Mechanism that triggered the reduction.
    pub reason: ReductionReason,
}

/// Per-hydro diagnostic data captured during AR order selection.
#[derive(Debug, Clone)]
pub struct HydroEstimationEntry {
    /// Selected AR order: the **maximum** per-season order, which sets the
    /// output coefficient-vector length.
    pub selected_order: u32,
    /// Fitted AR lag coefficients, one inner vector per season in `season_id`
    /// order; empty for seasons where estimation was skipped.
    pub coefficients: Vec<Vec<f64>>,
    /// Order-reduction records applied during fitting; empty when none occurred.
    pub contribution_reductions: Vec<ContributionReduction>,
}

/// Computation-side summary of the AR estimation pipeline, keyed by [`EntityId`]
/// for canonical deterministic ordering.
///
/// `white_noise_fallbacks` and `std_ratio_warnings` are populated only by the
/// partial-estimation path; both are empty for other paths.
#[must_use]
#[derive(Debug, Clone)]
pub struct EstimationReport {
    /// Per-hydro diagnostic entries.
    pub entries: BTreeMap<EntityId, HydroEstimationEntry>,
    /// Order selection method (e.g., `"AIC"`, `"PACF"`, `"fixed"`).
    pub method: String,
    /// Hydros with user-provided stats but no estimated AR coefficients
    /// (white-noise fallback: empty AR).
    pub white_noise_fallbacks: Vec<EntityId>,
    /// Hydros whose consecutive-season std ratios diverge between the
    /// user-provided and history-estimated profiles.
    pub std_ratio_warnings: Vec<StdRatioDivergence>,
}

/// Advisory diagnostic for a `(hydro, season pair)` whose cross-season std ratio
/// diverges between the user-provided and history-estimated profiles.
///
/// Produced by the partial-estimation path when
/// `max(user_ratio / est_ratio, est_ratio / user_ratio) > 2.0` for any
/// consecutive season pair.
#[derive(Debug, Clone)]
pub struct StdRatioDivergence {
    /// Hydro for which the divergence was detected.
    pub hydro_id: EntityId,
    /// First season of the consecutive pair.
    pub season_a: usize,
    /// Second season of the consecutive pair (wraps around).
    pub season_b: usize,
    /// `std[season_a] / std[season_b]` from the user-provided profile.
    pub user_ratio: f64,
    /// `std[season_a] / std[season_b]` from the history-estimated profile.
    pub estimated_ratio: f64,
    /// `max(user_ratio / estimated_ratio, estimated_ratio / user_ratio)`.
    pub divergence: f64,
}

/// Result of validating an AR order via contribution analysis.
#[derive(Debug, Clone)]
pub struct ContributionValidationResult {
    /// Whether the current order passed (all contributions non-negative).
    pub valid: bool,
    /// Maximum valid order (equals `current_order` when valid, less otherwise).
    pub max_valid_order: usize,
    /// Computed contribution values for the current order.
    pub contributions: Vec<f64>,
}

/// Validate an AR order for one `(entity, season)` pair via contribution analysis,
/// returning stability and the maximum valid order.
///
/// `current_order == 0` returns `valid: true` with no contributions (an order-0
/// model has no autoregressive dependence to validate).
fn validate_order_contributions(
    season_id: usize,
    n_seasons: usize,
    current_order: usize,
    all_season_coefficients: &[Vec<f64>],
    std_by_season: &[f64],
) -> ContributionValidationResult {
    if current_order == 0 {
        return ContributionValidationResult {
            valid: true,
            max_valid_order: 0,
            contributions: Vec::new(),
        };
    }

    let coeff_refs: Vec<&[f64]> = all_season_coefficients.iter().map(Vec::as_slice).collect();
    let contributions = compute_contributions(
        season_id,
        n_seasons,
        current_order,
        &coeff_refs,
        std_by_season,
    );

    let valid = !check_negative_contributions(&contributions);
    let max_valid_order = if valid {
        current_order
    } else {
        find_max_valid_order(&contributions)
    };

    ContributionValidationResult {
        valid,
        max_valid_order,
        contributions,
    }
}

/// Configuration parameters for AR coefficient estimation.
pub struct ArEstimationConfig<'a> {
    /// Maximum AR order considered before order selection.
    pub max_order: usize,
    /// Optional per-coefficient magnitude safety bound.
    pub max_coeff_magnitude: Option<f64>,
    /// Season map for calendar-based date-to-season fallback.
    pub season_map: Option<&'a SeasonMap>,
    /// `true` selects the PAR-A path (conditional FACP + extended YW); `false`
    /// (default) the classical PACF path.
    pub use_annual_component: bool,
}

/// Estimate AR coefficients, dispatching to the classical or PAR-A path on
/// `cfg.use_annual_component`.
///
/// On a single-resolution `cfg.season_map`, a season's lags are the seasons
/// before it in calendar order, whatever their ids; the returned estimates and
/// report carry the map's own season ids.
///
/// # Errors
///
/// Propagates [`StochasticError`] from the underlying fitting primitives.
pub fn estimate_ar_coefficients_with_selection(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    cfg: &ArEstimationConfig<'_>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    let Some(cycle_positions) = CyclePositions::new(cfg.season_map) else {
        return estimate_ar_in_cycle_order(observations, seasonal_stats, stages, hydro_ids, cfg);
    };
    let stages = cycle_positions.relabel_stages(stages);
    let season_map = cfg
        .season_map
        .map(|season_map| cycle_positions.relabel_season_map(season_map));
    let cfg = ArEstimationConfig {
        season_map: season_map.as_ref(),
        ..*cfg
    };
    let (estimates, report) =
        estimate_ar_in_cycle_order(observations, seasonal_stats, &stages, hydro_ids, &cfg)?;
    Ok((
        cycle_positions.estimates_to_raw(estimates),
        cycle_positions.report_to_raw(report),
    ))
}

fn estimate_ar_in_cycle_order(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    cfg: &ArEstimationConfig<'_>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    if cfg.use_annual_component {
        estimate_ar_with_pacf_annual(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            cfg.max_order,
            cfg.season_map,
            cfg.max_coeff_magnitude,
        )
    } else {
        estimate_ar_with_pacf(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            cfg.max_order,
            cfg.season_map,
            cfg.max_coeff_magnitude,
        )
    }
}

/// Periodic-PACF AR order selection (95% CI significance test) followed by a
/// periodic Yule-Walker solve, accounting for the non-Toeplitz covariance
/// structure of periodic autoregressive processes.
fn estimate_ar_with_pacf(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    season_map: Option<&SeasonMap>,
    max_coeff_magnitude: Option<f64>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    if max_order == 0 {
        let estimates = estimate_ar_coefficients_with_season_map(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            0,
            season_map,
        )?;
        let report = EstimationReport {
            entries: BTreeMap::new(),
            method: "PACF".to_string(),
            white_noise_fallbacks: Vec::new(),
            std_ratio_warnings: Vec::new(),
        };
        return Ok((estimates, report));
    }

    let (stage_index, stats_map, n_seasons) = build_pacf_stage_lookups(stages, seasonal_stats);
    let group_obs = group_observations_by_season(observations, hydro_ids, &stage_index, season_map);
    let group_year_starts = group_year_starts(observations, hydro_ids, &stage_index, season_map);

    // 95% CI z-score for the PACF significance threshold.
    let z_alpha = 1.96_f64;

    let mut estimates = estimate_all_hydro_ar_coefficients(
        hydro_ids,
        &group_obs,
        &group_year_starts,
        &stats_map,
        n_seasons,
        max_order,
        z_alpha,
    );

    let reductions = iterative_pacf_reduction(
        &mut estimates,
        n_seasons,
        hydro_ids,
        &group_obs,
        &group_year_starts,
        &stats_map,
        &PacfReductionParams {
            initial_max_order: max_order,
            z_alpha,
            max_coeff_magnitude,
        },
    );

    let report = build_estimation_report(&estimates, n_seasons, &reductions, "PACF");
    Ok((estimates, report))
}

/// PAR-A path: extended periodic Yule-Walker with rolling 12-month annual
/// component (report `method = "PACF_ANNUAL"`).
///
/// A returned [`ArCoefficientEstimate`] carries `annual: Some(..)` when its
/// `(hydro, season)` pair has at least one rolling-window `A_t` observation;
/// seasons without one fall through to the classical PAR(p) path with
/// `annual: None`.
///
/// # Errors
///
/// Propagates `StochasticError::InsufficientData` from
/// [`estimate_annual_seasonal_stats`] when any hydro has fewer than 13
/// chronological observations (no rolling window can be formed).
// Rationale: splitting would thread intermediate look-ups as extra arguments and obscure sequential data-flow.
#[allow(clippy::too_many_lines)]
fn estimate_ar_with_pacf_annual(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    season_map: Option<&SeasonMap>,
    max_coeff_magnitude: Option<f64>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    let annual_stats: Vec<AnnualSeasonalStats> =
        estimate_annual_seasonal_stats(observations, stages, hydro_ids, season_map)?;

    let annual_stats_map: HashMap<(EntityId, usize), &AnnualSeasonalStats> = annual_stats
        .iter()
        .map(|s| ((s.hydro_id, s.season_id), s))
        .collect();

    let (stage_index, stats_map, n_seasons) = build_pacf_stage_lookups(stages, seasonal_stats);

    let group_obs = group_observations_by_season(observations, hydro_ids, &stage_index, season_map);
    let group_z_year_starts = group_year_starts(observations, hydro_ids, &stage_index, season_map);
    let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();

    // Rolling-window A_t groups must reproduce the chronological grouping of
    // `estimate_annual_seasonal_stats` so A and Z align by season.
    let mut entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if entity_set.contains(&entity_id) {
            entity_obs.entry(entity_id).or_default().push((date, value));
        }
    }
    for obs_vec in entity_obs.values_mut() {
        obs_vec.sort_unstable_by_key(|(d, _)| *d);
    }

    // Same storage convention as `estimate_annual_seasonal_stats` (A_{t-1} keyed
    // under `group[i + 11]`'s season). The per-bucket minimum year is the first
    // PDF year, which `cross_correlation_z_a` needs to align A and Z by absolute
    // year, not by bucket index.
    let mut annual_group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    let mut annual_group_year_starts: HashMap<(EntityId, usize), i32> = HashMap::new();
    for &entity_id in hydro_ids {
        let Some(group) = entity_obs.get(&entity_id) else {
            continue;
        };
        for i in 0..group.len().saturating_sub(12) {
            let target_date = group[i + 11].0;
            let Some(season_id) = find_season_for_date(&stage_index, target_date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(target_date)))
            else {
                continue;
            };
            let mean_a: f64 = group[i..i + 12].iter().map(|(_, v)| v).sum::<f64>() / 12.0;
            annual_group_obs
                .entry((entity_id, season_id))
                .or_default()
                .push(mean_a);
            let y = observation_year(season_map, season_id, target_date);
            annual_group_year_starts
                .entry((entity_id, season_id))
                .and_modify(|cur| {
                    if y < *cur {
                        *cur = y;
                    }
                })
                .or_insert(y);
        }
    }

    let z_alpha = 1.96_f64;

    // Determinism: `flat_map_iter`/`collect` reassembles the per-hydro blocks in
    // canonical `hydro_ids` order, and the inner per-season PACF → extended
    // Yule-Walker solve is bit-identical to a single-threaded pass — thread
    // scheduling cannot change the output. `flat_map_iter` (not `flat_map`):
    // each hydro's `Vec` is small (`n_seasons`), so nesting work-stealing over
    // it would gain nothing.
    let mut estimates: Vec<ArCoefficientEstimate> = hydro_ids
        .par_iter()
        .flat_map_iter(|&hydro_id| {
            let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
            let mut annual_stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
            let mut z_year_starts: Vec<i32> = vec![0; n_seasons];
            let mut a_year_starts: Vec<i32> = vec![0; n_seasons];

            for season in 0..n_seasons {
                if let Some(stats) = stats_map.get(&(hydro_id, season)) {
                    stats_by_season[season] = (stats.mean, stats.std);
                }
                if let Some(ann_stats) = annual_stats_map.get(&(hydro_id, season)) {
                    annual_stats_by_season[season] = (ann_stats.mean_m3s, ann_stats.std_m3s);
                }
                if let Some(&y) = group_z_year_starts.get(&(hydro_id, season)) {
                    z_year_starts[season] = y;
                }
                if let Some(&y) = annual_group_year_starts.get(&(hydro_id, season)) {
                    a_year_starts[season] = y;
                }
            }

            let obs_refs: Vec<&[f64]> = (0..n_seasons)
                .map(|season| {
                    group_obs
                        .get(&(hydro_id, season))
                        .map_or(&[][..], Vec::as_slice)
                })
                .collect();
            let annual_obs_refs: Vec<&[f64]> = (0..n_seasons)
                .map(|season| {
                    annual_group_obs
                        .get(&(hydro_id, season))
                        .map_or(&[][..], Vec::as_slice)
                })
                .collect();

            let mut hydro_estimates: Vec<ArCoefficientEstimate> = Vec::with_capacity(n_seasons);
            for season in 0..n_seasons {
                // YW for `season` couples Z_t with A_{t-1}, stored under `prev_season`
                // (storage convention in `estimate_annual_seasonal_stats`).
                let prev_season = (season + n_seasons - 1) % n_seasons;
                let n_obs = obs_refs[season].len();
                let n_ann_obs = annual_obs_refs[prev_season].len();
                let stats_s = stats_by_season[season];
                let annual_stats_s = annual_stats_by_season[prev_season];

                if stats_s.1 == 0.0 || n_obs < 2 || n_ann_obs == 0 || annual_stats_s.1 == 0.0 {
                    hydro_estimates.push(ArCoefficientEstimate {
                        hydro_id,
                        season_id: season,
                        coefficients: Vec::new(),
                        annual: annual_stats_map.get(&(hydro_id, prev_season)).map(|s| {
                            AnnualComponent {
                                coefficient: 0.0,
                                mean_m3s: s.mean_m3s,
                                std_m3s: s.std_m3s,
                            }
                        }),
                    });
                    continue;
                }

                let facp_values = conditional_facp_partitioned(
                    season,
                    max_order,
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &z_year_starts,
                    &annual_obs_refs,
                    &annual_stats_by_season,
                    &a_year_starts,
                );
                let pacf_result = select_order_pacf_annual(&facp_values, n_obs, z_alpha);
                let yw_result = estimate_periodic_ar_annual_coefficients(
                    season,
                    pacf_result.selected_order,
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &z_year_starts,
                    &annual_obs_refs,
                    &annual_stats_by_season,
                    &a_year_starts,
                );

                // The annual std σ^A in the runtime `psi_hat = ψ · s_m / σ^A` must be
                // the std of A_{t-1} — the entry at `prev_season`, not `season`.
                let (ann_mean, ann_std) = annual_stats_by_season[prev_season];
                hydro_estimates.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id: season,
                    coefficients: yw_result.coefficients,
                    annual: Some(AnnualComponent {
                        coefficient: yw_result.annual_coefficient,
                        mean_m3s: ann_mean,
                        std_m3s: ann_std,
                    }),
                });
            }
            hydro_estimates
        })
        .collect();

    let reductions = apply_annual_prepass_reductions(
        &mut estimates,
        n_seasons,
        hydro_ids,
        &group_obs,
        &group_z_year_starts,
        &annual_group_obs,
        &annual_group_year_starts,
        &stats_map,
        &annual_stats_map,
        max_order,
        z_alpha,
        max_coeff_magnitude,
    );

    let report = build_estimation_report(&estimates, n_seasons, &reductions, "PACF_ANNUAL");
    Ok((estimates, report))
}

/// Apply magnitude-bound, `phi_1`, and contribution pre-passes for the PAR-A path
/// (the PAR-A counterpart of [`iterative_pacf_reduction`]).
///
/// Reductions act on the AR order alone (the φ vector); the annual term ψ is a
/// separate parameter preserved across reductions and refreshed via re-solves of
/// the extended Yule-Walker system at the new ceiling.
// Rationale: bundling into a struct would displace the arity to the struct literal with no clarity gain.
#[allow(clippy::too_many_arguments)]
fn apply_annual_prepass_reductions(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_ids: &[EntityId],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    group_z_year_starts: &HashMap<(EntityId, usize), i32>,
    annual_group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    annual_group_year_starts: &HashMap<(EntityId, usize), i32>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    annual_stats_map: &HashMap<(EntityId, usize), &AnnualSeasonalStats>,
    initial_max_order: usize,
    z_alpha: f64,
    max_coeff_magnitude: Option<f64>,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    apply_prepass_reductions(estimates, max_coeff_magnitude, &mut all_reductions);

    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    for &hydro_id in hydro_ids {
        let Some(indices) = hydro_indices.get(&hydro_id) else {
            continue;
        };
        reduce_entity_orders_annual(
            estimates,
            n_seasons,
            hydro_id,
            indices,
            group_obs,
            group_z_year_starts,
            annual_group_obs,
            annual_group_year_starts,
            stats_map,
            annual_stats_map,
            initial_max_order,
            z_alpha,
            &mut all_reductions,
        );
    }

    all_reductions
}

/// Detect seasons whose AR contributions turned negative at the current order,
/// recording a `NegativeContribution` reduction for each and returning the
/// failing season ids.
///
/// Shared by both reduction loops ([`reduce_entity_orders`] and
/// [`reduce_entity_orders_annual`]); the per-season re-solve that follows is
/// path-specific and stays in each caller.
fn detect_failing_seasons(
    estimates: &[ArCoefficientEstimate],
    indices: &[usize],
    frozen: &[bool],
    n_seasons: usize,
    all_coeffs: &[Vec<f64>],
    std_by_season: &[f64],
    hydro_id: EntityId,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) -> Vec<usize> {
    let mut failing_seasons: Vec<usize> = Vec::new();
    for &idx in indices {
        let season_id = estimates[idx].season_id;
        if frozen[season_id] || estimates[idx].coefficients.is_empty() {
            continue;
        }
        let current_order = estimates[idx].coefficients.len();
        let result = validate_order_contributions(
            season_id,
            n_seasons,
            current_order,
            all_coeffs,
            std_by_season,
        );
        if !result.valid {
            all_reductions
                .entry(hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id,
                    original_order: estimates[idx].coefficients.len(),
                    reduced_order: result.max_valid_order,
                    contributions: result.contributions,
                    reason: ReductionReason::NegativeContribution,
                });
            failing_seasons.push(season_id);
        }
    }
    failing_seasons
}

/// Iterative contribution-based AR-order reduction for one entity, PAR-A path
/// (the annual counterpart of [`reduce_entity_orders`]).
///
/// Each ceiling reduction re-solves the extended Yule-Walker system so both φ
/// and ψ are refreshed. When the ceiling reaches 0 the AR coefficients are
/// dropped but ψ is retained via a final order-0 YW solve, keeping the constant
/// term consistent with the per-season annual stats.
// Rationale: independently-sourced tables with no natural context struct; the mutable slice prevents decomposition.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn reduce_entity_orders_annual(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_id: EntityId,
    indices: &[usize],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    group_z_year_starts: &HashMap<(EntityId, usize), i32>,
    annual_group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    annual_group_year_starts: &HashMap<(EntityId, usize), i32>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    annual_stats_map: &HashMap<(EntityId, usize), &AnnualSeasonalStats>,
    initial_max_order: usize,
    z_alpha: f64,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) {
    let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
    let mut annual_stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
    let mut z_year_starts: Vec<i32> = vec![0; n_seasons];
    let mut a_year_starts: Vec<i32> = vec![0; n_seasons];
    for season in 0..n_seasons {
        if let Some(s) = stats_map.get(&(hydro_id, season)) {
            stats_by_season[season] = (s.mean, s.std);
        }
        if let Some(s) = annual_stats_map.get(&(hydro_id, season)) {
            annual_stats_by_season[season] = (s.mean_m3s, s.std_m3s);
        }
        if let Some(&y) = group_z_year_starts.get(&(hydro_id, season)) {
            z_year_starts[season] = y;
        }
        if let Some(&y) = annual_group_year_starts.get(&(hydro_id, season)) {
            a_year_starts[season] = y;
        }
    }
    let std_by_season: Vec<f64> = stats_by_season.iter().map(|&(_, s)| s).collect();

    let mut max_orders: Vec<usize> = vec![initial_max_order; n_seasons];
    let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
    for &idx in indices {
        let est = &estimates[idx];
        if est.season_id < n_seasons {
            all_coeffs[est.season_id].clone_from(&est.coefficients);
        }
    }

    // `frozen` means the AR component is zero; ψ may still update via order-0 re-fits.
    let mut frozen: Vec<bool> = vec![false; n_seasons];
    for &idx in indices {
        let sid = estimates[idx].season_id;
        if estimates[idx].coefficients.is_empty() {
            frozen[sid] = true;
        }
    }

    let obs_refs: Vec<&[f64]> = (0..n_seasons)
        .map(|season| {
            group_obs
                .get(&(hydro_id, season))
                .map_or(&[][..], Vec::as_slice)
        })
        .collect();
    let annual_obs_refs: Vec<&[f64]> = (0..n_seasons)
        .map(|season| {
            annual_group_obs
                .get(&(hydro_id, season))
                .map_or(&[][..], Vec::as_slice)
        })
        .collect();

    loop {
        let failing_seasons = detect_failing_seasons(
            estimates,
            indices,
            &frozen,
            n_seasons,
            &all_coeffs,
            &std_by_season,
            hydro_id,
            all_reductions,
        );
        if failing_seasons.is_empty() {
            break;
        }

        let mut any_reselected = false;
        for &season_id in &failing_seasons {
            if max_orders[season_id] == 0 {
                continue;
            }
            max_orders[season_id] -= 1;

            // Even at ceiling 0 the 1×1 extended YW is solved so ψ is refreshed.
            let stats_s = stats_by_season[season_id];
            if stats_s.1 == 0.0
                || obs_refs[season_id].len() < 2
                || annual_obs_refs[season_id].is_empty()
                || annual_stats_by_season[season_id].1 == 0.0
            {
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
                continue;
            }

            let n_obs = obs_refs[season_id].len();
            let selected_order = if max_orders[season_id] == 0 {
                0
            } else {
                let facp = conditional_facp_partitioned(
                    season_id,
                    max_orders[season_id],
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &z_year_starts,
                    &annual_obs_refs,
                    &annual_stats_by_season,
                    &a_year_starts,
                );
                select_order_pacf_annual(&facp, n_obs, z_alpha).selected_order
            };
            let yw_result = estimate_periodic_ar_annual_coefficients(
                season_id,
                selected_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
                &z_year_starts,
                &annual_obs_refs,
                &annual_stats_by_season,
                &a_year_starts,
            );

            // A_{t-1}'s annual stats are stored at prev_season, not season_id.
            let prev_season = (season_id + n_seasons - 1) % n_seasons;
            let (ann_mean, ann_std) = annual_stats_by_season[prev_season];
            for &idx in indices {
                if estimates[idx].season_id == season_id {
                    estimates[idx]
                        .coefficients
                        .clone_from(&yw_result.coefficients);
                    estimates[idx].annual = Some(AnnualComponent {
                        coefficient: yw_result.annual_coefficient,
                        mean_m3s: ann_mean,
                        std_m3s: ann_std,
                    });
                    all_coeffs[season_id].clone_from(&yw_result.coefficients);
                }
            }

            if max_orders[season_id] == 0 || yw_result.coefficients.is_empty() {
                frozen[season_id] = true;
                continue;
            }

            // φ_1 < 0 after the re-solve drops AR; ψ is retained via the order-0
            // refit below.
            if has_negative_phi1(&all_coeffs[season_id]) {
                let original_order = all_coeffs[season_id].len();
                all_reductions
                    .entry(hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::Phi1Negative,
                    });
                let yw0 = estimate_periodic_ar_annual_coefficients(
                    season_id,
                    0,
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &z_year_starts,
                    &annual_obs_refs,
                    &annual_stats_by_season,
                    &a_year_starts,
                );
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        estimates[idx].annual = Some(AnnualComponent {
                            coefficient: yw0.annual_coefficient,
                            mean_m3s: ann_mean,
                            std_m3s: ann_std,
                        });
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
            } else {
                any_reselected = true;
            }
        }
        if !any_reselected {
            break;
        }
    }
}

/// Stage index entry: `(start_date, end_date, stage_id, season_id)`.
type StageSeasonEntry = (chrono::NaiveDate, chrono::NaiveDate, i32, usize);

/// Return type of [`build_pacf_stage_lookups`]:
/// `(stage_index, stats_map, n_seasons)`.
type PacfStageLookups<'a> = (
    Vec<StageSeasonEntry>,
    HashMap<(EntityId, usize), &'a SeasonalStats>,
    usize,
);

/// Build the PACF stage-season lookups: `stage_index` sorted by start date,
/// `stats_map` keyed by `(EntityId, season_id)`, and the season count.
fn build_pacf_stage_lookups<'a>(
    stages: &[Stage],
    seasonal_stats: &'a [SeasonalStats],
) -> PacfStageLookups<'a> {
    let mut stage_index = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect::<Vec<_>>();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let stage_id_to_season: HashMap<i32, usize> = stage_index
        .iter()
        .map(|&(_, _, stage_id, season_id)| (stage_id, season_id))
        .collect();

    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
        .iter()
        .filter_map(|s| {
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(((s.entity_id, season_id), s))
        })
        .collect();

    let n_seasons = stage_index
        .iter()
        .map(|&(_, _, _, season_id)| season_id + 1)
        .max()
        .unwrap_or(0);

    (stage_index, stats_map, n_seasons)
}

/// Group raw observations by `(EntityId, season_id)` for PACF fitting.
fn group_observations_by_season(
    observations: &[(EntityId, NaiveDate, f64)],
    hydro_ids: &[EntityId],
    stage_index: &[(chrono::NaiveDate, chrono::NaiveDate, i32, usize)],
    season_map: Option<&SeasonMap>,
) -> HashMap<(EntityId, usize), Vec<f64>> {
    let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();
    let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if !entity_set.contains(&entity_id) {
            continue;
        }
        let Some(season_id) = find_season_for_date(stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
        else {
            continue;
        };
        group_obs
            .entry((entity_id, season_id))
            .or_default()
            .push(value);
    }
    group_obs
}

/// The year that keys an observation's season occurrence: the season map's
/// occurrence year, or the calendar year without a map.
fn observation_year(season_map: Option<&SeasonMap>, season_id: usize, date: NaiveDate) -> i32 {
    season_map.map_or_else(
        || date.year(),
        |sm| observation_occurrence_year(sm, season_id, date),
    )
}

/// First [`observation_year`] of each `(EntityId, season_id)` bucket of
/// [`group_observations_by_season`].
fn group_year_starts(
    observations: &[(EntityId, NaiveDate, f64)],
    hydro_ids: &[EntityId],
    stage_index: &[(chrono::NaiveDate, chrono::NaiveDate, i32, usize)],
    season_map: Option<&SeasonMap>,
) -> HashMap<(EntityId, usize), i32> {
    let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();
    let mut starts: HashMap<(EntityId, usize), i32> = HashMap::new();
    for &(entity_id, date, _value) in observations {
        if !entity_set.contains(&entity_id) {
            continue;
        }
        let Some(season_id) = find_season_for_date(stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
        else {
            continue;
        };
        let y = observation_year(season_map, season_id, date);
        starts
            .entry((entity_id, season_id))
            .and_modify(|cur| {
                if y < *cur {
                    *cur = y;
                }
            })
            .or_insert(y);
    }
    starts
}

/// Build initial AR coefficient estimates for all hydros using periodic PACF + YW.
fn estimate_all_hydro_ar_coefficients(
    hydro_ids: &[EntityId],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    group_year_starts: &HashMap<(EntityId, usize), i32>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    n_seasons: usize,
    max_order: usize,
    z_alpha: f64,
) -> Vec<ArCoefficientEstimate> {
    // Determinism: `flat_map_iter`/`collect` reassembles the per-hydro blocks in
    // canonical `hydro_ids` order, and the inner per-season PACF → Yule-Walker
    // solve is bit-identical to a single-threaded pass — thread scheduling cannot
    // change the output. `flat_map_iter` (not `flat_map`): each hydro's `Vec` is
    // small (`n_seasons`), so nesting work-stealing over it would gain nothing.
    hydro_ids
        .par_iter()
        .flat_map_iter(|&hydro_id| {
            let stats_by_season: Vec<(f64, f64)> = (0..n_seasons)
                .map(|season| {
                    stats_map
                        .get(&(hydro_id, season))
                        .map_or((0.0, 0.0), |s| (s.mean, s.std))
                })
                .collect();
            let obs_refs: Vec<&[f64]> = (0..n_seasons)
                .map(|season| {
                    group_obs
                        .get(&(hydro_id, season))
                        .map_or(&[][..], Vec::as_slice)
                })
                .collect();
            let year_starts: Vec<i32> = (0..n_seasons)
                .map(|season| {
                    group_year_starts
                        .get(&(hydro_id, season))
                        .copied()
                        .unwrap_or(0)
                })
                .collect();
            let mut hydro_estimates: Vec<ArCoefficientEstimate> = Vec::with_capacity(n_seasons);
            for season in 0..n_seasons {
                let stats_s = stats_by_season[season];
                if stats_s.1 == 0.0 || obs_refs[season].len() < 2 {
                    hydro_estimates.push(ArCoefficientEstimate {
                        hydro_id,
                        season_id: season,
                        coefficients: Vec::new(),
                        annual: None,
                    });
                    continue;
                }
                let n_obs = obs_refs[season].len();
                let pacf_values = periodic_pacf(
                    season,
                    max_order,
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &year_starts,
                );
                let pacf_result = select_order_pacf(&pacf_values, n_obs, z_alpha);
                let yw_result = estimate_periodic_ar_coefficients(
                    season,
                    pacf_result.selected_order,
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &year_starts,
                );
                hydro_estimates.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id: season,
                    coefficients: yw_result.coefficients,
                    annual: None,
                });
            }
            hydro_estimates
        })
        .collect()
}

/// PAR-fit knobs for `iterative_pacf_reduction` and its helpers.
struct PacfReductionParams {
    initial_max_order: usize,
    z_alpha: f64,
    max_coeff_magnitude: Option<f64>,
}

/// Apply magnitude-bound and `phi_1` pre-passes, recording reductions in `all_reductions`.
fn apply_prepass_reductions(
    estimates: &mut [ArCoefficientEstimate],
    max_coeff_magnitude: Option<f64>,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) {
    if let Some(threshold) = max_coeff_magnitude {
        for est in estimates.iter_mut() {
            let has_explosive = est.coefficients.iter().any(|c| c.abs() > threshold);
            if has_explosive {
                let original_order = est.coefficients.len();
                all_reductions
                    .entry(est.hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id: est.season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::MagnitudeBound,
                    });
                est.coefficients.clear();
            }
        }
    }
    for est in estimates.iter_mut() {
        if has_negative_phi1(&est.coefficients) {
            let original_order = est.coefficients.len();
            all_reductions
                .entry(est.hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id: est.season_id,
                    original_order,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                });
            est.coefficients.clear();
        }
    }
}

/// Iterative PACF order-reduction loop for one entity: mutates `estimates` for
/// the seasons in `indices` and appends records to `all_reductions`.
fn reduce_entity_orders(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_id: EntityId,
    indices: &[usize],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    group_year_starts: &HashMap<(EntityId, usize), i32>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    params: &PacfReductionParams,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) {
    let stats_by_season: Vec<(f64, f64)> = (0..n_seasons)
        .map(|season| {
            stats_map
                .get(&(hydro_id, season))
                .map_or((0.0, 0.0), |s| (s.mean, s.std))
        })
        .collect();
    let std_by_season: Vec<f64> = stats_by_season.iter().map(|&(_, s)| s).collect();
    let mut max_orders: Vec<usize> = vec![params.initial_max_order; n_seasons];
    let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
    for &idx in indices {
        let est = &estimates[idx];
        if est.season_id < n_seasons {
            all_coeffs[est.season_id].clone_from(&est.coefficients);
        }
    }
    let mut frozen: Vec<bool> = vec![false; n_seasons];
    for &idx in indices {
        let sid = estimates[idx].season_id;
        if estimates[idx].coefficients.is_empty() {
            frozen[sid] = true;
        }
    }
    let obs_refs: Vec<&[f64]> = (0..n_seasons)
        .map(|season| {
            group_obs
                .get(&(hydro_id, season))
                .map_or(&[][..], Vec::as_slice)
        })
        .collect();
    let year_starts: Vec<i32> = (0..n_seasons)
        .map(|season| {
            group_year_starts
                .get(&(hydro_id, season))
                .copied()
                .unwrap_or(0)
        })
        .collect();
    loop {
        let failing_seasons = detect_failing_seasons(
            estimates,
            indices,
            &frozen,
            n_seasons,
            &all_coeffs,
            &std_by_season,
            hydro_id,
            all_reductions,
        );
        if failing_seasons.is_empty() {
            break;
        }
        let mut any_reselected = false;
        for &season_id in &failing_seasons {
            if max_orders[season_id] == 0 {
                continue;
            }
            max_orders[season_id] -= 1;
            if max_orders[season_id] == 0 {
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
                continue;
            }
            let stats_s = stats_by_season[season_id];
            if stats_s.1 == 0.0 || obs_refs[season_id].len() < 2 {
                frozen[season_id] = true;
                continue;
            }
            let n_obs = obs_refs[season_id].len();
            let pacf_values = periodic_pacf(
                season_id,
                max_orders[season_id],
                n_seasons,
                &obs_refs,
                &stats_by_season,
                &year_starts,
            );
            let pacf_result = select_order_pacf(&pacf_values, n_obs, params.z_alpha);
            let yw_result = estimate_periodic_ar_coefficients(
                season_id,
                pacf_result.selected_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
                &year_starts,
            );
            for &idx in indices {
                if estimates[idx].season_id == season_id {
                    estimates[idx]
                        .coefficients
                        .clone_from(&yw_result.coefficients);
                    all_coeffs[season_id].clone_from(&yw_result.coefficients);
                }
            }
            if has_negative_phi1(&all_coeffs[season_id]) {
                let original_order = all_coeffs[season_id].len();
                all_reductions
                    .entry(hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::Phi1Negative,
                    });
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
            } else {
                any_reselected = true;
            }
        }
        if !any_reselected {
            break;
        }
    }
}

/// Iteratively reduce AR orders via PACF re-selection and contribution validation.
///
/// Per entity, a failing season's ceiling drops by 1 and the full
/// PACF-selection / YW-estimation / `phi_1` / contribution cycle re-runs, until
/// every season passes or its ceiling reaches 0.
fn iterative_pacf_reduction(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_ids: &[EntityId],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    group_year_starts: &HashMap<(EntityId, usize), i32>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    params: &PacfReductionParams,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    apply_prepass_reductions(estimates, params.max_coeff_magnitude, &mut all_reductions);

    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    for &hydro_id in hydro_ids {
        let Some(indices) = hydro_indices.get(&hydro_id) else {
            continue;
        };
        reduce_entity_orders(
            estimates,
            n_seasons,
            hydro_id,
            indices,
            group_obs,
            group_year_starts,
            stats_map,
            params,
            &mut all_reductions,
        );
    }

    all_reductions
}

/// Build an [`EstimationReport`] (infallible — it only reorganises computed data).
///
/// Each hydro's selected order is the **maximum** across its seasons, matching
/// the single-order-per-hydro shape the I/O layer (`FittingReport`) expects.
#[allow(clippy::implicit_hasher)]
pub fn build_estimation_report(
    estimates: &[ArCoefficientEstimate],
    n_seasons: usize,
    contribution_reductions: &HashMap<EntityId, Vec<ContributionReduction>>,
    method: &str,
) -> EstimationReport {
    // Estimates arrive sorted by (hydro_id, season_id).
    let mut hydro_coeffs: BTreeMap<EntityId, Vec<(usize, Vec<f64>)>> = BTreeMap::new();
    for est in estimates {
        hydro_coeffs
            .entry(est.hydro_id)
            .or_default()
            .push((est.season_id, est.coefficients.clone()));
    }

    let mut entries: BTreeMap<EntityId, HydroEstimationEntry> = BTreeMap::new();

    for (hydro_id, mut season_coeffs) in hydro_coeffs {
        season_coeffs.sort_by_key(|(season_id, _)| *season_id);

        let selected_order = season_coeffs
            .iter()
            .map(|(_, coeffs)| coeffs.len())
            .max()
            .unwrap_or(0);

        let season_map: HashMap<usize, Vec<f64>> = season_coeffs.into_iter().collect();
        let coefficients: Vec<Vec<f64>> = (0..n_seasons)
            .map(|sid| season_map.get(&sid).cloned().unwrap_or_default())
            .collect();

        let reductions = contribution_reductions
            .get(&hydro_id)
            .cloned()
            .unwrap_or_default();

        #[allow(clippy::cast_possible_truncation)]
        entries.insert(
            hydro_id,
            HydroEstimationEntry {
                selected_order: selected_order as u32,
                coefficients,
                contribution_reductions: reductions,
            },
        );
    }

    EstimationReport {
        entries,
        method: method.to_string(),
        white_noise_fallbacks: Vec::new(),
        std_ratio_warnings: Vec::new(),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::useless_vec
)]
mod tests;
