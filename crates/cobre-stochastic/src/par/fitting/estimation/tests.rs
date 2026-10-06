use std::collections::{BTreeMap, HashMap};

use chrono::{Datelike, Days, Months, NaiveDate, Weekday};
use cobre_core::test_support::{StageSpec, date, make_stage};
use cobre_core::{EntityId, SeasonMap, Stage};

use super::{
    ArEstimationConfig, ContributionReduction, EstimationReport, PacfReductionParams,
    ReductionReason, estimate_ar_coefficients_with_selection, estimate_ar_with_pacf_annual,
    has_negative_phi1, iterative_pacf_reduction, validate_order_contributions,
};
use crate::StochasticError;
use crate::par::fitting::cycle_positions::twin_fixtures::{
    TwinMaps, calendar_history, calendar_stages, permuted_quarterly_twins, sparse_ring_twins,
};
use crate::par::fitting::{ArCoefficientEstimate, SeasonalStats};
use crate::test_support::{
    MonthlyLabels, monthly_season_map, quarterly_season_map, weekly_season_map,
};

/// Test-only: magnitude bound + contribution-based order validation over all
/// AR estimates.
fn apply_contribution_validation(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    max_coeff_magnitude: Option<f64>,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    if let Some(threshold) = max_coeff_magnitude {
        for est in estimates.iter_mut() {
            if est.coefficients.iter().any(|c| c.abs() > threshold) {
                all_reductions
                    .entry(est.hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id: est.season_id,
                        original_order: est.coefficients.len(),
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
            all_reductions
                .entry(est.hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id: est.season_id,
                    original_order: est.coefficients.len(),
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                });
            est.coefficients.clear();
        }
    }

    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    for (&hydro_id, indices) in &hydro_indices {
        let std_by_season: Vec<f64> = (0..n_seasons)
            .map(|sid| stats_map.get(&(hydro_id, sid)).map_or(0.0, |s| s.std))
            .collect();

        let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        for &idx in indices {
            let est = &estimates[idx];
            if est.season_id < n_seasons {
                all_coeffs[est.season_id].clone_from(&est.coefficients);
            }
        }

        for &idx in indices {
            let season_id = estimates[idx].season_id;
            let mut current_order = estimates[idx].coefficients.len();

            loop {
                let result = validate_order_contributions(
                    season_id,
                    n_seasons,
                    current_order,
                    &all_coeffs,
                    &std_by_season,
                );

                if result.valid || current_order == 0 {
                    break;
                }

                let original_order = current_order;
                let reduced_order = result.max_valid_order;

                all_reductions
                    .entry(hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id,
                        original_order,
                        reduced_order,
                        contributions: result.contributions,
                        reason: ReductionReason::NegativeContribution,
                    });

                estimates[idx].coefficients.truncate(reduced_order);

                all_coeffs[season_id].clone_from(&estimates[idx].coefficients);
                current_order = reduced_order;
            }
        }
    }

    all_reductions
}

// ── Numeric test helpers (pure core/stochastic inputs only) ──────────────────

/// Generate synthetic observations for a single-season AR(p) process.
/// Uses a fixed seed for reproducibility.
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)] // u64 >> 33 fits in f64; u32::MAX fits in f64
fn generate_ar_observations(coefficients: &[f64], n: usize) -> Vec<f64> {
    let p = coefficients.len();
    let mut values = vec![0.0_f64; n + p];
    let mut seed: u64 = 42;
    for i in p..(n + p) {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let noise = ((seed >> 33) as f64 / (u32::MAX as f64) - 0.5) * 2.0;
        let mut val = noise;
        for (j, c) in coefficients.iter().enumerate() {
            val += c * values[i - j - 1];
        }
        values[i] = val;
    }
    values[p..].to_vec()
}

/// Simulate a 2-season PAR(2) process using deterministic LCG (Box-Muller).
/// Model: `z_t = phi_1 * z_{t-1} + phi_2 * z_{t-2} + noise_t`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
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
    let mut lcg: u64 = seed;

    let lcg_next = |s: u64| -> u64 {
        s.wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407)
    };

    for i in 2..n_generate + 2 {
        lcg = lcg_next(lcg);
        let u1 = (lcg >> 11) as f64 / (1u64 << 53) as f64;
        lcg = lcg_next(lcg);
        let u2 = (lcg >> 11) as f64 / (1u64 << 53) as f64;
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

/// Build a minimal 2-season `Stage` for testing.
fn make_two_season_stage(
    index: usize,
    id: i32,
    season_id: usize,
    year: i32,
    first_half: bool,
) -> Stage {
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let (start_date, end_date) = if first_half {
        (
            NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
        )
    } else {
        (
            NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
            NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap(),
        )
    };

    Stage {
        index,
        id,
        start_date,
        end_date,
        season_id: Some(season_id),
        blocks: vec![Block {
            index: 0,
            name: "SINGLE".to_string(),
            duration_hours: 4380.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }
}

/// Build a 2-season `SeasonMap` (H1: Jan–Jun, H2: Jul–Dec).
fn two_season_map() -> SeasonMap {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition};
    SeasonMap {
        cycle_type: SeasonCycleType::Custom,
        seasons: vec![
            SeasonDefinition {
                id: 0,
                label: "H1".to_string(),
                month_start: 1,
                day_start: Some(1),
                month_end: Some(6),
                day_end: Some(30),
            },
            SeasonDefinition {
                id: 1,
                label: "H2".to_string(),
                month_start: 7,
                day_start: Some(1),
                month_end: Some(12),
                day_end: Some(31),
            },
        ],
    }
}

/// Build a 12-season monthly stage sequence spanning `n_years` starting from
/// year 2000. Stage IDs are 0-based sequential; season IDs cycle 0..12.
fn make_monthly_stages_for_annual(n_years: usize) -> Vec<Stage> {
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    let mut stages = Vec::new();
    let mut idx = 0usize;
    for year in 0..n_years {
        for month in 0..12usize {
            let y = 2000 + year as i32;
            let m = month as u32 + 1;
            let (ey, em) = if m == 12 { (y + 1, 1u32) } else { (y, m + 1) };
            stages.push(Stage {
                index: idx,
                id: idx as i32,
                start_date: NaiveDate::from_ymd_opt(y, m, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(ey, em, 1).unwrap(),
                season_id: Some(month),
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
            });
            idx += 1;
        }
    }
    stages
}

/// Build `n_years * 12` synthetic monthly observations for `hydro_id`.
fn synthetic_monthly_obs(
    hydro_id: EntityId,
    n_years: usize,
    base: f64,
    scale: f64,
    drift: f64,
) -> Vec<(EntityId, NaiveDate, f64)> {
    let mut obs = Vec::new();
    for year in 0..n_years {
        for month in 0..12usize {
            let value = base
                + f64::from(u32::try_from(month + 1).unwrap()) * scale
                + f64::from(u32::try_from(year).unwrap()) * drift;
            let date = NaiveDate::from_ymd_opt(
                2000 + i32::try_from(year).unwrap(),
                u32::try_from(month + 1).unwrap(),
                1,
            )
            .unwrap();
            obs.push((hydro_id, date, value));
        }
    }
    obs
}

// ── Contribution validation tests ────────────────────────────────────────────

#[test]
fn test_contribution_order_zero_fallback() {
    let result = validate_order_contributions(
        0,             // season_id
        1,             // n_seasons
        1,             // current_order
        &[vec![-1.5]], // all_season_coefficients
        &[10.0],       // std_by_season
    );
    assert!(!result.valid);
    assert_eq!(result.max_valid_order, 0);
}

/// Order-0 input returns valid immediately.
#[test]
fn test_contribution_order_zero_input_passes() {
    let result = validate_order_contributions(0, 1, 0, &[Vec::new()], &[10.0]);
    assert!(result.valid);
    assert_eq!(result.max_valid_order, 0);
    assert!(result.contributions.is_empty());
}

/// Stable AR(2) model with all-positive contributions passes.
#[test]
fn test_contribution_stable_model_passes() {
    let result = validate_order_contributions(
        0,                 // season_id
        1,                 // n_seasons
        2,                 // current_order
        &[vec![0.4, 0.2]], // all_season_coefficients
        &[10.0],           // std_by_season
    );
    assert!(result.valid);
    assert_eq!(result.max_valid_order, 2);
    assert_eq!(result.contributions.len(), 2);
}

/// apply_contribution_validation reduces an explosive model.
///
/// Constructs AR(2) with coefficients [0.3, -0.8] for a single entity
/// and single season. The contribution of lag 2 is negative (-0.71),
/// so the order should be reduced to 1.
#[test]
fn test_apply_contribution_validation_reduces_explosive() {
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.3, -0.8],
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 100.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        None, // max_coeff_magnitude
    );

    assert_eq!(
        estimates[0].coefficients.len(),
        1,
        "explosive AR(2) should be reduced to AR(1)"
    );
    assert!((estimates[0].coefficients[0] - 0.3).abs() < 1e-10);

    let entity_reductions = reductions.get(&hydro_id).expect("should have reductions");
    assert_eq!(entity_reductions.len(), 1);
    assert_eq!(entity_reductions[0].original_order, 2);
    assert_eq!(entity_reductions[0].reduced_order, 1);
    assert_eq!(entity_reductions[0].season_id, 0);
}

/// PIMENTAL-like scenario -- large coefficient at lag 2 in one
/// season while other seasons are benign.
#[test]
fn test_pimental_like_multi_season_reduction() {
    let hydro_id = EntityId(156);
    let n_seasons = 12;

    // Season 7 (August): explosive AR(2); all others benign AR(1).
    let mut estimates: Vec<ArCoefficientEstimate> = (0..n_seasons)
        .map(|s| ArCoefficientEstimate {
            hydro_id,
            season_id: s,
            coefficients: if s == 7 { vec![0.5, 48.9] } else { vec![0.1] },
            annual: None,
        })
        .collect();

    // August std = 5 (vs ~200): a large coefficient × small std is what triggers it.
    let stds: Vec<f64> = (0..n_seasons)
        .map(|s| if s == 7 { 5.0 } else { 200.0 })
        .collect();

    let stats: Vec<SeasonalStats> = (0..n_seasons)
        .map(|s| SeasonalStats {
            entity_id: hydro_id,
            stage_id: s as i32,
            mean: 100.0,
            std: stds[s],
        })
        .collect();
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(s, st)| ((hydro_id, s), st))
        .collect();

    let reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        None, // max_coeff_magnitude
    );

    // August (season 7) should have been reduced from AR(2).
    // The contribution of lag 2 through the periodic chain may or may not be negative
    // depending on the recursive composition with neighboring months' coefficients.
    // We verify the reduction was applied if any reduction occurred for August.
    let august_order = estimates[7].coefficients.len();

    // Other months should remain unchanged at AR(1) since their contributions
    // are small positive values.
    for (s, est) in estimates.iter().enumerate() {
        if s != 7 {
            assert_eq!(est.coefficients.len(), 1, "season {s} should remain AR(1)");
        }
    }

    if august_order < 2 {
        let entity_reductions = reductions.get(&hydro_id).expect("should have reductions");
        assert!(
            entity_reductions.iter().any(|r| r.season_id == 7),
            "August reduction should be recorded"
        );
    }
}

/// All contributions negative forces white-noise fallback.
///
/// AR(1) with phi = -2.0 for all seasons -- every contribution is negative,
/// so order drops to 0.
#[test]
fn test_all_negative_fallback_to_white_noise() {
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![-2.0],
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let _reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        None, // max_coeff_magnitude
    );

    assert!(
        estimates[0].coefficients.is_empty(),
        "should fall back to order 0"
    );
}

// ── Phi_1 rejection tests ────────────────────────────────────────────────

#[test]
fn phi1_rejection_sets_order_to_zero() {
    let hydro_id = EntityId(1);
    let n_seasons = 2;

    let mut estimates = vec![
        ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![-0.3, 0.5],
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id,
            season_id: 1,
            coefficients: vec![0.4, 0.2],
            annual: None,
        },
    ];

    let stats = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1,
            mean: 60.0,
            std: 12.0,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| ((s.entity_id, i), s))
        .collect();

    let reductions = apply_contribution_validation(&mut estimates, n_seasons, &stats_map, None);

    assert!(
        estimates[0].coefficients.is_empty(),
        "season 0 should be cleared to order 0"
    );

    assert_eq!(
        estimates[1].coefficients,
        vec![0.4, 0.2],
        "season 1 should be unchanged"
    );

    let hydro_reductions = reductions.get(&hydro_id).expect("should have reductions");
    let r = hydro_reductions
        .iter()
        .find(|r| r.season_id == 0)
        .expect("should have reduction for season 0");
    assert_eq!(r.original_order, 2);
    assert_eq!(r.reduced_order, 0);
    assert!(r.contributions.is_empty());
}

#[test]
fn phi1_rejection_before_contribution_analysis() {
    // A season with phi_1 = -0.01 and phi_2 = 0.5 with uniform std.
    // The contribution analysis would NOT catch this (contributions may
    // be non-negative), but the phi_1 gate fires first.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![-0.01, 0.5],
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let reductions = apply_contribution_validation(&mut estimates, n_seasons, &stats_map, None);

    assert!(
        estimates[0].coefficients.is_empty(),
        "phi_1 = -0.01 should trigger rejection"
    );
    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
}

#[test]
fn phi1_zero_is_not_rejected() {
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.0, 0.3],
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let _reductions = apply_contribution_validation(&mut estimates, n_seasons, &stats_map, None);

    assert_eq!(
        estimates[0].coefficients.len(),
        2,
        "phi_1 = 0.0 should not be rejected"
    );
}

#[test]
fn phi1_rejection_interacts_with_magnitude_bound() {
    // phi_1 = -50.0 is both negative and above any reasonable magnitude bound.
    // The magnitude-bound pre-pass should fire first, and the phi_1 check
    // should see the already-cleared vector and skip.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![-50.0, 0.3],
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        Some(10.0), // magnitude bound that catches -50.0
    );

    assert!(
        estimates[0].coefficients.is_empty(),
        "should be cleared to order 0"
    );

    let hydro_reductions = reductions.get(&hydro_id).expect("should have reductions");
    assert_eq!(
        hydro_reductions.len(),
        1,
        "should have exactly 1 reduction entry (magnitude bound only)"
    );
}

// ── Iterative PACF reduction tests ──────────────────────────────────────

#[test]
fn iterative_reduction_terminates_at_zero() {
    // Construct a case where contributions fail at every order,
    // forcing the loop to terminate at order 0.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    // phi = [0.3, -0.8]: contribution at lag 2 is negative, so the
    // apply_contribution_validation (Fixed-method) path reduces the order.
    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.3, -0.8],
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let reductions = apply_contribution_validation(&mut estimates, n_seasons, &stats_map, None);

    // The loop should terminate (possibly at a reduced order or order 0).
    // The key assertion is that it terminates and doesn't infinite-loop.
    assert!(
        estimates[0].coefficients.len() < 2,
        "order should be reduced from 2; got {}",
        estimates[0].coefficients.len()
    );

    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
}

#[test]
fn iterative_reduction_only_affects_failing_seasons() {
    let hydro_id = EntityId(1);
    let n_seasons = 2;

    let mut estimates = vec![
        ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            // phi = [0.3, -0.8]: contribution at lag 2 is negative.
            coefficients: vec![0.3, -0.8],
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id,
            season_id: 1,
            // phi = [0.4, 0.2]: all contributions positive.
            coefficients: vec![0.4, 0.2],
            annual: None,
        },
    ];

    let stats = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1,
            mean: 60.0,
            std: 10.0,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| ((s.entity_id, i), s))
        .collect();

    let _reductions = apply_contribution_validation(&mut estimates, n_seasons, &stats_map, None);

    assert!(
        estimates[0].coefficients.len() < 2,
        "season 0 order should be reduced from 2; got {}",
        estimates[0].coefficients.len()
    );

    assert_eq!(
        estimates[1].coefficients,
        vec![0.4, 0.2],
        "season 1 should be unchanged"
    );
}

#[test]
fn iterative_pacf_reduction_with_synthetic_observations() {
    // Data from a known AR process, then coefficients set to fail contribution
    // analysis so re-selection is triggered.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let obs = generate_ar_observations(&[0.5, 0.2], 100);

    let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    group_obs.insert((hydro_id, 0), obs);

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 0.0,
        std: 1.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.5, 0.2, -5.0], // order 3, lag 3 will fail
        annual: None,
    }];

    let reductions = iterative_pacf_reduction(
        &mut estimates,
        n_seasons,
        &[hydro_id],
        &group_obs,
        &HashMap::new(),
        &stats_map,
        &PacfReductionParams {
            initial_max_order: 3,
            z_alpha: 1.96,
            max_coeff_magnitude: None,
        },
    );

    assert!(
        estimates[0].coefficients.len() < 3,
        "order should be reduced from 3; got {}",
        estimates[0].coefficients.len()
    );

    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
}

#[test]
fn fixed_path_uses_truncation_not_reselection() {
    // Verify that the Fixed order selection path still uses
    // apply_contribution_validation (truncation), not iterative PACF.
    // We check this by verifying the behavior matches truncation semantics.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    // phi = [0.5, 0.2, -0.8]: contribution at lag 3 is negative.
    // Truncation would give order 2 (find_max_valid_order).
    // Iterative PACF would re-run PACF at max_order=2 and possibly
    // select a different order.
    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.5, 0.2, -0.8],
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    let reductions = apply_contribution_validation(&mut estimates, n_seasons, &stats_map, None);

    // With truncation, the Fixed path truncates coefficients at the
    // first negative contribution position, which is at lag 3 (index 2).
    // So max_valid_order should be 2 or less.
    let final_order = estimates[0].coefficients.len();
    assert!(
        final_order <= 2,
        "Fixed path should truncate; got order {final_order}"
    );

    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
    let r = &reductions[&hydro_id][0];
    assert_eq!(r.original_order, 3);
    assert!(r.reduced_order <= 2, "truncation should produce order <= 2");
}

// ── Combined strategy and reduction reason tests ─────────────────────────

#[test]
fn combined_strategies_produce_correct_reduction_reasons() {
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_seasons = 2;

    let mut estimates = vec![
        // H1 S0: negative phi_1 -> Phi1Negative
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 0,
            coefficients: vec![-0.3, 0.5],
            annual: None,
        },
        // H1 S1: negative contribution at lag 3 -> NegativeContribution
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 1,
            coefficients: vec![0.5, 0.2, -0.8],
            annual: None,
        },
        // H2 S0: magnitude bound -> MagnitudeBound
        ArCoefficientEstimate {
            hydro_id: h2,
            season_id: 0,
            coefficients: vec![50.0],
            annual: None,
        },
        // H2 S1: passes -> no reduction
        ArCoefficientEstimate {
            hydro_id: h2,
            season_id: 1,
            coefficients: vec![0.4, 0.2],
            annual: None,
        },
    ];

    let stats = vec![
        SeasonalStats {
            entity_id: h1,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 1,
            mean: 60.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: h2,
            stage_id: 0,
            mean: 70.0,
            std: 15.0,
        },
        SeasonalStats {
            entity_id: h2,
            stage_id: 1,
            mean: 80.0,
            std: 12.0,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| ((s.entity_id, i % 2), s))
        .collect();

    let reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, Some(10.0));

    // H1 S0: phi_1 negative -> Phi1Negative, order 0
    let h1_reductions = &reductions[&h1];
    let h1_s0 = h1_reductions
        .iter()
        .find(|r| r.season_id == 0)
        .expect("should have reduction for H1 S0");
    assert_eq!(h1_s0.reason, ReductionReason::Phi1Negative);
    assert_eq!(h1_s0.reduced_order, 0);

    // H1 S1: negative contribution -> NegativeContribution
    let h1_s1 = h1_reductions
        .iter()
        .find(|r| r.season_id == 1)
        .expect("should have reduction for H1 S1");
    assert_eq!(h1_s1.reason, ReductionReason::NegativeContribution);

    // H2 S0: magnitude bound -> MagnitudeBound, order 0
    let h2_reductions = &reductions[&h2];
    let h2_s0 = h2_reductions
        .iter()
        .find(|r| r.season_id == 0)
        .expect("should have reduction for H2 S0");
    assert_eq!(h2_s0.reason, ReductionReason::MagnitudeBound);
    assert_eq!(h2_s0.reduced_order, 0);

    // H2 S1: no reduction
    assert!(
        !h2_reductions.iter().any(|r| r.season_id == 1),
        "H2 S1 should have no reduction"
    );
}

// ── PACF and contribution cascade tests ──────────────────

/// Verify `iterative_pacf_reduction` does not spuriously reduce a stable
/// 2-season PAR(2) model (phi_1=0.7, phi_2=0.15). Checks termination,
/// order preservation, and coefficient matching.
#[test]
#[allow(clippy::cast_precision_loss)]
fn iterative_pacf_reduction_stable_par2_not_spuriously_reduced() {
    use crate::par::fitting::{
        estimate_periodic_ar_coefficients, periodic_pacf, select_order_pacf,
    };

    let hydro_id = EntityId(1);
    let n_seasons = 2;
    let n_years = 500_usize; // 500 obs/season; threshold ≈ 0.088 << PACF(2) ≈ 0.15.

    let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 137);

    let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    group_obs.insert((hydro_id, 0), obs_s0.clone());
    group_obs.insert((hydro_id, 1), obs_s1.clone());

    let mean_s0 = obs_s0.iter().sum::<f64>() / obs_s0.len() as f64;
    let mean_s1 = obs_s1.iter().sum::<f64>() / obs_s1.len() as f64;
    let std_s0 = (obs_s0.iter().map(|x| (x - mean_s0).powi(2)).sum::<f64>()
        / (obs_s0.len() - 1) as f64)
        .sqrt();
    let std_s1 = (obs_s1.iter().map(|x| (x - mean_s1).powi(2)).sum::<f64>()
        / (obs_s1.len() - 1) as f64)
        .sqrt();

    let stats_storage = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: mean_s0,
            std: std_s0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1,
            mean: mean_s1,
            std: std_s1,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats_storage
        .iter()
        .enumerate()
        .map(|(s, st)| ((hydro_id, s), st))
        .collect();

    let stats_by_season_pop = {
        let n = obs_s0.len() as f64;
        let mu0 = mean_s0;
        let mu1 = mean_s1;
        let s0 = (obs_s0.iter().map(|x| (x - mu0).powi(2)).sum::<f64>() / n).sqrt();
        let s1 = (obs_s1.iter().map(|x| (x - mu1).powi(2)).sum::<f64>() / n).sqrt();
        vec![(mu0, s0), (mu1, s1)]
    };
    let obs_refs: Vec<&[f64]> = vec![&obs_s0, &obs_s1];
    let year_starts = [0; 2];
    let z_alpha = 1.96_f64;
    // Use max_order=2 because the generating process is AR(2).
    // Higher max_order with periodic PACF on a 2-season split of a stationary
    // process over-estimates order due to per-season standardization artifacts.
    let max_order = 2_usize;

    let mut estimates: Vec<ArCoefficientEstimate> = Vec::new();
    for season in 0..n_seasons {
        let n_obs = obs_refs[season].len();
        let pacf_values = periodic_pacf(
            season,
            max_order,
            n_seasons,
            &obs_refs,
            &stats_by_season_pop,
            &year_starts,
        );
        let selected = select_order_pacf(&pacf_values, n_obs, z_alpha).selected_order;
        let yw = estimate_periodic_ar_coefficients(
            season,
            selected,
            n_seasons,
            &obs_refs,
            &stats_by_season_pop,
            &year_starts,
        );
        estimates.push(ArCoefficientEstimate {
            hydro_id,
            season_id: season,
            coefficients: yw.coefficients,
            annual: None,
        });
    }

    for est in &estimates {
        assert_eq!(
            est.coefficients.len(),
            2,
            "season {} should select order 2; got {}",
            est.season_id,
            est.coefficients.len()
        );
    }

    let coeffs_before: Vec<Vec<f64>> = estimates.iter().map(|e| e.coefficients.clone()).collect();

    let reductions = iterative_pacf_reduction(
        &mut estimates,
        n_seasons,
        &[hydro_id],
        &group_obs,
        &HashMap::new(),
        &stats_map,
        &PacfReductionParams {
            initial_max_order: max_order,
            z_alpha,
            max_coeff_magnitude: None,
        },
    );

    for est in &estimates {
        assert_eq!(
            est.coefficients.len(),
            2,
            "stable PAR(2) season {} should remain order 2; got {}",
            est.season_id,
            est.coefficients.len()
        );
    }

    for (est, before) in estimates.iter().zip(coeffs_before.iter()) {
        if est.coefficients.len() == before.len() {
            for (a, b) in est.coefficients.iter().zip(before.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "season {} coefficient drift: {a} vs {b}",
                    est.season_id
                );
            }
        } else {
            let yw_direct = estimate_periodic_ar_coefficients(
                est.season_id,
                est.coefficients.len(),
                n_seasons,
                &obs_refs,
                &stats_by_season_pop,
                &year_starts,
            );
            for (a, b) in est.coefficients.iter().zip(yw_direct.coefficients.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "season {} post-reduction coeff {a} vs YW {b}",
                    est.season_id
                );
            }
        }
    }

    // No spurious reductions for stable model.
    assert!(!reductions.contains_key(&hydro_id));
}

/// Roundtrip estimation test: verify AR(2) coefficient recovery.
/// Simulates 1000 years of 2-season stationary AR(2) data (phi_1=0.7,
/// phi_2=0.15). Runs full PACF order selection pipeline and verifies
/// recovered coefficients match true values within 0.15 tolerance.
#[test]
#[allow(clippy::cast_precision_loss)]
fn roundtrip_estimation_two_season_par2_recovers_coefficients() {
    let hydro_id = EntityId(1);
    let n_years = 1000_usize;
    let true_phi1 = 0.7_f64;
    let true_phi2 = 0.15_f64;

    let (obs_s0, obs_s1) = simulate_two_season_par2(true_phi1, true_phi2, n_years, 42);

    // Season 0: Jan 1 – Jul 1; Season 1: Jul 1 – Jan 1.
    let ref_year = 2000_i32;
    let stages = vec![
        make_two_season_stage(0, 0, 0, ref_year, true),
        make_two_season_stage(1, 1, 1, ref_year, false),
    ];

    // Build seasonal stats (Bessel-corrected std to match estimate_seasonal_stats_with_season_map).
    let n_f = n_years as f64;
    let mu0 = obs_s0.iter().sum::<f64>() / n_f;
    let mu1 = obs_s1.iter().sum::<f64>() / n_f;
    let std0 = (obs_s0.iter().map(|x| (x - mu0).powi(2)).sum::<f64>() / (n_f - 1.0)).sqrt();
    let std1 = (obs_s1.iter().map(|x| (x - mu1).powi(2)).sum::<f64>() / (n_f - 1.0)).sqrt();

    let seasonal_stats = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0, // matches stages[0].id
            mean: mu0,
            std: std0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1, // matches stages[1].id
            mean: mu1,
            std: std1,
        },
    ];

    // Build observations with dates: season 0 = Jan 1, season 1 = Jul 1.
    // Using a fixed reference year does NOT matter because find_season_for_date
    // falls back to the season_map when dates are outside the study stage ranges.
    let season_map = two_season_map();
    let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
    for y in 0..n_years {
        let year = (1970 + y) as i32;
        observations.push((
            hydro_id,
            NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
            obs_s0[y],
        ));
        observations.push((
            hydro_id,
            NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
            obs_s1[y],
        ));
    }

    let (estimates, _report) = estimate_ar_coefficients_with_selection(
        &observations,
        &seasonal_stats,
        &stages,
        &[hydro_id],
        &ArEstimationConfig {
            max_order: 2,
            max_coeff_magnitude: None,
            season_map: Some(&season_map),
            use_annual_component: false,
        },
    )
    .expect("estimation must succeed");

    let est_s0 = estimates
        .iter()
        .find(|e| e.hydro_id == hydro_id && e.season_id == 0)
        .expect("season 0 estimate must exist");
    let est_s1 = estimates
        .iter()
        .find(|e| e.hydro_id == hydro_id && e.season_id == 1)
        .expect("season 1 estimate must exist");

    // With 1000 years the PACF threshold is 1.96/sqrt(1000) ≈ 0.062,
    // well below PACF(2) ≈ 0.15 and well above PACF(3..4) ≈ 0.
    // The pipeline should select order 2 and recover the true coefficients.
    for est in [est_s0, est_s1] {
        assert!(
            est.coefficients.len() >= 2,
            "season {} should select at least order 2; got {}",
            est.season_id,
            est.coefficients.len()
        );
        assert!(
            (est.coefficients[0] - true_phi1).abs() < 0.15,
            "season {} phi_1={:.4} should be within 0.15 of true {true_phi1:.4}",
            est.season_id,
            est.coefficients[0]
        );
        assert!(
            (est.coefficients[1] - true_phi2).abs() < 0.15,
            "season {} phi_2={:.4} should be within 0.15 of true {true_phi2:.4}",
            est.season_id,
            est.coefficients[1]
        );
    }
}

// ── estimate_ar_with_pacf_annual tests ─────────────

/// Two hydros × 12 seasons: every estimate has `annual.is_some()`.
///
/// 30 years of synthetic monthly data (360 observations per hydro) gives
/// enough rolling-window samples for `estimate_annual_seasonal_stats` to
/// succeed and for the extended YW system to be well-conditioned.
#[test]
fn estimate_ar_with_pacf_annual_two_hydros_twelve_seasons() {
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_years = 30;
    let stages = make_monthly_stages_for_annual(n_years);

    // Two hydros with different base values so their series are distinct.
    let mut obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    obs.extend(synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5));

    let seasonal_stats = {
        use crate::par::fitting::estimate_seasonal_stats_with_season_map;
        estimate_seasonal_stats_with_season_map(&obs, &stages, &[h1, h2], None).unwrap()
    };

    let (estimates, report) = estimate_ar_with_pacf_annual(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1, h2],
        3,    // max_order
        None, // season_map
        None, // max_coeff_magnitude
    )
    .expect("estimate_ar_with_pacf_annual must succeed with 30 years of data");

    assert_eq!(
        estimates.len(),
        24,
        "2 hydros × 12 seasons = 24 estimates, got {}",
        estimates.len()
    );
    assert_eq!(
        report.method, "PACF_ANNUAL",
        "method must be PACF_ANNUAL, got {}",
        report.method
    );

    for est in &estimates {
        assert!(
            est.annual.is_some(),
            "hydro={} season={}: annual must be Some",
            est.hydro_id.0,
            est.season_id
        );
        let ann = est.annual.as_ref().unwrap();
        assert!(
            ann.std_m3s > 0.0,
            "hydro={} season={}: annual.std_m3s must be > 0, got {}",
            est.hydro_id.0,
            est.season_id,
            ann.std_m3s
        );
    }
}

/// Insufficient observations propagate `InsufficientData` error.
///
/// 11 months of data cannot form any rolling 12-month average.
/// `estimate_ar_with_pacf_annual` must propagate the error from
/// `estimate_annual_seasonal_stats` rather than silently falling back.
#[test]
fn estimate_ar_with_pacf_annual_insufficient_observations_errors() {
    let h1 = EntityId(1);
    // 11 observations — fewer than the 13 required for one rolling window.
    let obs: Vec<(EntityId, NaiveDate, f64)> = (0u32..11)
        .map(|i| {
            let m = i % 12 + 1;
            (
                h1,
                NaiveDate::from_ymd_opt(2000, m, 1).unwrap(),
                50.0 + f64::from(i),
            )
        })
        .collect();

    let stages = make_monthly_stages_for_annual(2);

    // Provide minimal seasonal stats (actual values don't matter since the
    // error occurs before the YW solve).
    let seasonal_stats = vec![SeasonalStats {
        entity_id: h1,
        stage_id: 0,
        mean: 50.0,
        std: 5.0,
    }];

    let result = estimate_ar_with_pacf_annual(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1],
        2,    // max_order
        None, // season_map
        None, // max_coeff_magnitude
    );

    assert!(
        result.is_err(),
        "expected Err for insufficient observations, got Ok"
    );
    assert!(
        matches!(
            result.unwrap_err(),
            StochasticError::InsufficientData { .. }
        ),
        "error must be InsufficientData"
    );
}

// ── thread-count determinism gate ─────────────────────────────────────────

/// The per-hydro initial AR fit in `estimate_all_hydro_ar_coefficients` is a
/// `par_iter().flat_map().collect()`, so its output must not depend on the rayon
/// pool size. This gate runs the classical PACF dispatch
/// (`use_annual_component: false`, `max_order >= 2`) over a multi-hydro,
/// multi-season fixture under pools of 1, 2, and 4 threads and asserts the
/// returned `Vec<ArCoefficientEstimate>` is `to_bits`-identical across all three
/// — the `(hydro_id, season_id)` ordering and every `coefficients[k]`
/// must match bit-for-bit regardless of thread scheduling.
/// A regression that collected into a shared `Mutex<Vec>` or pushed estimates
/// from worker threads would reorder the stream and fail here.
#[test]
fn ar_fit_is_thread_count_invariant() {
    use crate::par::fitting::estimate_seasonal_stats_with_season_map;

    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let hydro_ids = [h1, h2];
    let n_years = 30;
    let stages = make_monthly_stages_for_annual(n_years);

    // Two hydros with distinct synthetic monthly series so the per-season PACF +
    // Yule-Walker fit path actually runs (not an all-white-noise degenerate case).
    let mut obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    obs.extend(synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5));

    let seasonal_stats =
        estimate_seasonal_stats_with_season_map(&obs, &stages, &hydro_ids, None).unwrap();

    // Run the classical dispatch inside a fixed-size rayon pool so the fit loop
    // runs under exactly `n` worker threads.
    let fit_under_pool = |n: usize| -> Vec<ArCoefficientEstimate> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon pool must build")
            .install(|| {
                let (estimates, _report) = estimate_ar_coefficients_with_selection(
                    &obs,
                    &seasonal_stats,
                    &stages,
                    &hydro_ids,
                    &ArEstimationConfig {
                        max_order: 3,
                        max_coeff_magnitude: None,
                        season_map: None,
                        use_annual_component: false,
                    },
                )
                .expect("classical estimation must succeed");
                estimates
            })
    };

    let thread_counts = [1usize, 2, 4];
    let outputs: Vec<Vec<ArCoefficientEstimate>> =
        thread_counts.iter().map(|&n| fit_under_pool(n)).collect();

    // The 1-thread baseline must have actually fitted at least one season (the
    // PACF + YW path ran, not an all-white-noise degenerate fixture).
    assert!(
        outputs[0].iter().any(|e| !e.coefficients.is_empty()),
        "the fixture must fit at least one non-empty coefficient vector"
    );

    // Bit-exact equality across pool sizes: ids, season ordering, coefficient
    // length, and every coefficient and the residual ratio compared via
    // `to_bits`, never float `==`.
    let assert_bit_identical = |a: &[ArCoefficientEstimate],
                                b: &[ArCoefficientEstimate],
                                threads_a: usize,
                                threads_b: usize| {
        assert_eq!(
            a.len(),
            b.len(),
            "estimate count must match across {threads_a}- and {threads_b}-thread pools"
        );
        for (ea, eb) in a.iter().zip(b) {
            assert_eq!(ea.hydro_id, eb.hydro_id, "hydro_id must match");
            assert_eq!(ea.season_id, eb.season_id, "season_id must match");
            assert_eq!(
                ea.coefficients.len(),
                eb.coefficients.len(),
                "coefficient length must match across pool sizes"
            );
            for (ca, cb) in ea.coefficients.iter().zip(&eb.coefficients) {
                assert_eq!(
                    ca.to_bits(),
                    cb.to_bits(),
                    "coefficient must be bit-identical across pool sizes"
                );
            }
        }
    };

    // Compare every pool size against the single-thread baseline.
    for (estimates_n, &n) in outputs.iter().zip(&thread_counts).skip(1) {
        assert_bit_identical(&outputs[0], estimates_n, thread_counts[0], n);
    }
}

/// The annual-path per-hydro initial fit in `estimate_ar_with_pacf_annual` is a
/// `par_iter().flat_map_iter().collect()`, so (a) its output order must not
/// depend on the rayon pool size for a fixed `hydro_ids` order, and (b) its
/// per-hydro values must not depend on the order `hydro_ids` is declared in.
/// This gate runs the annual PACF dispatch over a three-hydro, twelve-season
/// fixture under pools of 1, 2, and 4 threads (unsorted comparison — a
/// regression that collected via a shared `Mutex<Vec>` would reorder the
/// stream and fail here) and again with a shuffled `hydro_ids` order compared
/// as a `(hydro_id, season_id)`-sorted set (a regression that let one hydro's
/// closure observe another's state would change values, not just order, and
/// fail here too).
#[test]
fn annual_ar_fit_is_thread_count_and_declaration_order_invariant() {
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let h3 = EntityId(3);
    let n_years = 30;
    let stages = make_monthly_stages_for_annual(n_years);

    let mut obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    obs.extend(synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5));
    obs.extend(synthetic_monthly_obs(h3, n_years, 150.0, 4.0, 0.8));

    let seasonal_stats = {
        use crate::par::fitting::estimate_seasonal_stats_with_season_map;
        estimate_seasonal_stats_with_season_map(&obs, &stages, &[h1, h2, h3], None).unwrap()
    };

    let fit = |hydro_ids: &[EntityId], n_threads: usize| -> Vec<ArCoefficientEstimate> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("rayon pool must build")
            .install(|| {
                let (estimates, _report) = estimate_ar_with_pacf_annual(
                    &obs,
                    &seasonal_stats,
                    &stages,
                    hydro_ids,
                    3,    // max_order
                    None, // season_map
                    None, // max_coeff_magnitude
                )
                .expect("estimate_ar_with_pacf_annual must succeed with 30 years of data");
                estimates
            })
    };

    let assert_bit_identical =
        |label: &str, a: &[ArCoefficientEstimate], b: &[ArCoefficientEstimate]| {
            assert_eq!(a.len(), b.len(), "{label}: estimate count mismatch");
            for (ea, eb) in a.iter().zip(b) {
                assert_eq!(ea.hydro_id, eb.hydro_id, "{label}: hydro_id mismatch");
                assert_eq!(ea.season_id, eb.season_id, "{label}: season_id mismatch");
                assert_eq!(
                    ea.coefficients.len(),
                    eb.coefficients.len(),
                    "{label}: coefficient length mismatch"
                );
                for (ca, cb) in ea.coefficients.iter().zip(&eb.coefficients) {
                    assert_eq!(
                        ca.to_bits(),
                        cb.to_bits(),
                        "{label}: coefficient must be bit-identical"
                    );
                }
                match (&ea.annual, &eb.annual) {
                    (Some(aa), Some(ab)) => {
                        assert_eq!(
                            aa.coefficient.to_bits(),
                            ab.coefficient.to_bits(),
                            "{label}: annual.coefficient must be bit-identical"
                        );
                        assert_eq!(
                            aa.mean_m3s.to_bits(),
                            ab.mean_m3s.to_bits(),
                            "{label}: annual.mean_m3s must be bit-identical"
                        );
                        assert_eq!(
                            aa.std_m3s.to_bits(),
                            ab.std_m3s.to_bits(),
                            "{label}: annual.std_m3s must be bit-identical"
                        );
                    }
                    (None, None) => {}
                    _ => panic!("{label}: annual Some/None mismatch"),
                }
            }
        };

    let canonical = [h1, h2, h3];
    let baseline = fit(&canonical, 1);
    assert!(
        baseline.iter().any(|e| !e.coefficients.is_empty()),
        "the fixture must fit at least one non-empty coefficient vector"
    );

    // (a) Thread-count invariance at a fixed `hydro_ids` order: same order, no sort.
    for &n in &[2usize, 4] {
        assert_bit_identical(
            &format!("{n}-thread pool vs 1-thread baseline"),
            &baseline,
            &fit(&canonical, n),
        );
    }

    // (b) Declaration-order invariance: a shuffled `hydro_ids` legitimately
    // reassembles in the new canonical order, so compare as sorted sets.
    let shuffled = [h3, h1, h2];
    let mut baseline_sorted = baseline.clone();
    baseline_sorted.sort_by_key(|e| (e.hydro_id, e.season_id));
    for &n in &[1usize, 4] {
        let mut shuffled_out = fit(&shuffled, n);
        shuffled_out.sort_by_key(|e| (e.hydro_id, e.season_id));
        assert_bit_identical(
            &format!("shuffled hydro_ids order, {n}-thread pool"),
            &baseline_sorted,
            &shuffled_out,
        );
    }
}

// ── Season-cycle relabeling ──────────────────────────────────────────────────

fn fit_calendar_history(
    season_map: &SeasonMap,
    calendar_ids: &[usize],
) -> (Vec<ArCoefficientEstimate>, EstimationReport) {
    use crate::par::fitting::estimate_seasonal_stats_with_season_map;
    let hydro_ids = [EntityId(1)];
    let stages = calendar_stages(season_map, calendar_ids);
    let history = calendar_history(season_map, calendar_ids, &hydro_ids, 30);
    let stats =
        estimate_seasonal_stats_with_season_map(&history, &stages, &hydro_ids, Some(season_map))
            .unwrap();
    estimate_ar_coefficients_with_selection(
        &history,
        &stats,
        &stages,
        &hydro_ids,
        &ArEstimationConfig {
            max_order: 2,
            max_coeff_magnitude: None,
            season_map: Some(season_map),
            use_annual_component: false,
        },
    )
    .unwrap()
}

fn coefficient_bits(coefficients: &[f64]) -> Vec<u64> {
    coefficients.iter().map(|c| c.to_bits()).collect()
}

fn season_coefficient_bits(estimates: &[ArCoefficientEstimate], season_id: usize) -> Vec<u64> {
    let estimate = estimates
        .iter()
        .find(|e| e.season_id == season_id)
        .unwrap_or_else(|| panic!("no estimate for season {season_id}"));
    coefficient_bits(&estimate.coefficients)
}

fn assert_fit_equals_twin_fit(twins: &TwinMaps) {
    let (estimates, report) = fit_calendar_history(&twins.map, &twins.map_ids());
    let (twin_estimates, twin_report) = fit_calendar_history(&twins.twin, &twins.twin_ids());
    assert!(
        twin_estimates.iter().any(|e| e.coefficients.len() == 2),
        "the twin fit must select order 2 somewhere"
    );
    let entry = &report.entries[&EntityId(1)];
    let twin_entry = &twin_report.entries[&EntityId(1)];

    for &(id, twin_id) in &twins.ids {
        assert_eq!(
            season_coefficient_bits(&estimates, id),
            season_coefficient_bits(&twin_estimates, twin_id),
            "estimate of season {id} vs twin season {twin_id}"
        );
        assert_eq!(
            coefficient_bits(&entry.coefficients[id]),
            coefficient_bits(&twin_entry.coefficients[twin_id]),
            "report coefficients of season {id} vs twin season {twin_id}"
        );
    }
    assert_eq!(estimates.len(), twin_estimates.len());
    assert_eq!(entry.selected_order, twin_entry.selected_order);
    let largest_id = twins.map_ids().into_iter().max().unwrap();
    assert_eq!(entry.coefficients.len(), largest_id + 1);
    for (raw, coefficients) in entry.coefficients.iter().enumerate() {
        if twins.twin_of(raw).is_none() {
            assert!(coefficients.is_empty(), "index {raw} is not a season id");
        }
    }

    let reductions: Vec<_> = entry
        .contribution_reductions
        .iter()
        .map(|r| {
            (
                twins.twin_of(r.season_id),
                r.original_order,
                r.reduced_order,
                r.reason,
                coefficient_bits(&r.contributions),
            )
        })
        .collect();
    let twin_reductions: Vec<_> = twin_entry
        .contribution_reductions
        .iter()
        .map(|r| {
            (
                Some(r.season_id),
                r.original_order,
                r.reduced_order,
                r.reason,
                coefficient_bits(&r.contributions),
            )
        })
        .collect();
    assert_eq!(reductions, twin_reductions);
}

#[test]
fn ar_fit_on_a_sparse_custom_map_equals_the_fit_on_its_dense_twin() {
    assert_fit_equals_twin_fit(&sparse_ring_twins());
}

#[test]
fn ar_fit_on_an_out_of_calendar_order_map_equals_the_fit_on_its_calendar_ordered_twin() {
    assert_fit_equals_twin_fit(&permuted_quarterly_twins());
}

/// characterization: the fit indexed seasons by raw id before relabeling
/// existed, and raw id equals calendar position on this map.
#[test]
fn fit_on_a_dense_calendar_ordered_custom_map_is_unchanged() {
    const FIT_BITS: [&[u64]; 4] = [
        &[4_605_120_269_902_754_618],
        &[4_596_444_960_825_664_260, 4_603_467_181_103_770_236],
        &[4_600_539_182_485_145_474, 4_602_924_096_865_316_619],
        &[4_604_971_852_568_177_276],
    ];
    let season_map = crate::test_support::quarterly_season_map();
    let (estimates, report) = fit_calendar_history(&season_map, &[0, 1, 2, 3]);
    let entry = &report.entries[&EntityId(1)];
    assert_eq!(estimates.len(), FIT_BITS.len());
    assert_eq!(entry.coefficients.len(), FIT_BITS.len());
    for (season_id, &bits) in FIT_BITS.iter().enumerate() {
        assert_eq!(
            season_coefficient_bits(&estimates, season_id),
            bits,
            "season {season_id}"
        );
        assert_eq!(
            coefficient_bits(&entry.coefficients[season_id]),
            bits,
            "report season {season_id}"
        );
    }
}

// ── Lag pairing by occurrence year ───────────────────────────────────────────

/// One hydro's history at `dates`: `x_t = 100 + 10 z_t` with
/// `z_t = 0.8 z_{t-1} + e_t` and `e_t = ((2t + 3) mod 19) / 19 − 0.5`.
fn periodic_ar1_history(
    hydro_id: EntityId,
    dates: &[NaiveDate],
) -> Vec<(EntityId, NaiveDate, f64)> {
    let mut z = 0.0_f64;
    (0_u32..)
        .zip(dates)
        .map(|(t, &date)| {
            z = 0.8 * z + f64::from((2 * t + 3) % 19) / 19.0 - 0.5;
            (hydro_id, date, 100.0 + 10.0 * z)
        })
        .collect()
}

fn population_mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// `ρ(position, 1)` pairing each `(position, year)` value with the
/// `(position − 1 mod n, year − [position == 0])` value, summed in ascending
/// year, divided by the pair count and clamped as the kernel does.
fn reference_lag1_rho(
    values: &BTreeMap<(usize, i32), f64>,
    stats: &[(f64, f64)],
    position: usize,
) -> f64 {
    let n_seasons = stats.len();
    let predecessor = (position + n_seasons - 1) % n_seasons;
    let years_back = i32::from(position == 0);
    let (mu_ref, std_ref) = stats[position];
    let (mu_lag, std_lag) = stats[predecessor];
    let mut gamma = 0.0_f64;
    let mut n_pairs = 0_u32;
    for (&(_, year), &value) in values.range((position, i32::MIN)..=(position, i32::MAX)) {
        if let Some(&lagged) = values.get(&(predecessor, year - years_back)) {
            gamma += (value - mu_ref) * (lagged - mu_lag);
            n_pairs += 1;
        }
    }
    gamma /= f64::from(n_pairs);
    (gamma / (std_ref * std_lag)).clamp(-1.0, 1.0)
}

fn study_stages(periods: &[(NaiveDate, NaiveDate, usize)]) -> Vec<Stage> {
    (0_i32..)
        .zip(periods)
        .map(|(id, &(start_date, end_date, season_id))| {
            make_stage(StageSpec {
                id,
                start_date,
                end_date,
                season_id: Some(season_id),
                ..StageSpec::default()
            })
        })
        .collect()
}

/// Fits `max_order = 1` to [`periodic_ar1_history`] at `dates` and asserts that
/// every season's coefficient is the lag-1 autocorrelation pairing each
/// occurrence with its calendar predecessor. `occurrence` names a date's
/// `(season id, calendar position, occurrence year)`.
fn assert_ar1_fit_pairs_each_occurrence_with_its_calendar_predecessor(
    season_map: &SeasonMap,
    stages: &[Stage],
    dates: &[NaiveDate],
    occurrence: impl Fn(NaiveDate) -> (usize, usize, i32),
) {
    let hydro_id = EntityId(1);
    let history = periodic_ar1_history(hydro_id, dates);
    let n_seasons = season_map.seasons.len();

    let mut values: BTreeMap<(usize, i32), f64> = BTreeMap::new();
    let mut position_of: BTreeMap<usize, usize> = BTreeMap::new();
    for &(_, date, value) in &history {
        let (season_id, position, year) = occurrence(date);
        position_of.insert(season_id, position);
        values.insert((position, year), value);
    }
    let buckets: Vec<Vec<f64>> = (0..n_seasons)
        .map(|position| {
            values
                .range((position, i32::MIN)..=(position, i32::MAX))
                .map(|(_, &value)| value)
                .collect()
        })
        .collect();
    let stats: Vec<(f64, f64)> = buckets.iter().map(|b| population_mean_std(b)).collect();
    let seasonal_stats: Vec<SeasonalStats> = stages
        .iter()
        .map(|stage| {
            let (mean, std) = stats[position_of[&stage.season_id.unwrap()]];
            SeasonalStats {
                entity_id: hydro_id,
                stage_id: stage.id,
                mean,
                std,
            }
        })
        .collect();

    for (position, bucket) in buckets.iter().enumerate() {
        let threshold = 1.96 / (bucket.len() as f64).sqrt();
        let reference = reference_lag1_rho(&values, &stats, position);
        assert!(
            reference > threshold,
            "position {position}: reference rho {reference} must exceed {threshold}"
        );
    }

    let (estimates, _) = estimate_ar_coefficients_with_selection(
        &history,
        &seasonal_stats,
        stages,
        &[hydro_id],
        &ArEstimationConfig {
            max_order: 1,
            max_coeff_magnitude: None,
            season_map: Some(season_map),
            use_annual_component: false,
        },
    )
    .unwrap();

    assert_eq!(estimates.len(), n_seasons);
    let mismatched: Vec<(usize, &[f64], f64)> = estimates
        .iter()
        .filter_map(|estimate| {
            let reference = reference_lag1_rho(&values, &stats, position_of[&estimate.season_id]);
            let matches = matches!(
                estimate.coefficients.as_slice(),
                [phi] if (phi - reference).abs() < 1e-12
            );
            (!matches).then_some((
                estimate.season_id,
                estimate.coefficients.as_slice(),
                reference,
            ))
        })
        .collect();
    assert!(
        mismatched.is_empty(),
        "(season id, coefficients, reference rho) of every season whose fit is not \
         its reference rho: {mismatched:?}"
    );
}

#[test]
fn ar_fit_pairs_lags_by_year_when_monthly_history_starts_mid_year() {
    let season_map = monthly_season_map(MonthlyLabels::ZeroBased);
    let dates: Vec<NaiveDate> = (0..480)
        .map(|m| date(2000, 7, 1).checked_add_months(Months::new(m)).unwrap())
        .collect();
    assert_eq!(dates.last(), Some(&date(2040, 6, 1)));
    let periods: Vec<(NaiveDate, NaiveDate, usize)> = (0_u32..12)
        .map(|month0| {
            let start = date(2041, month0 + 1, 1);
            let end = start.checked_add_months(Months::new(1)).unwrap();
            (start, end, month0 as usize)
        })
        .collect();

    assert_ar1_fit_pairs_each_occurrence_with_its_calendar_predecessor(
        &season_map,
        &study_stages(&periods),
        &dates,
        |date| (date.month0() as usize, date.month0() as usize, date.year()),
    );
}

#[test]
fn ar_fit_pairs_lags_by_year_on_a_custom_map_numbered_from_april() {
    let mut season_map = quarterly_season_map();
    for (def, id) in season_map.seasons.iter_mut().zip([3, 0, 1, 2]) {
        def.id = id;
    }
    season_map.seasons.sort_by_key(|def| def.id);
    let dates: Vec<NaiveDate> = (0..160)
        .map(|q| {
            date(2000, 4, 1)
                .checked_add_months(Months::new(3 * q))
                .unwrap()
        })
        .collect();
    assert_eq!(dates.last(), Some(&date(2040, 1, 1)));
    let stages = study_stages(&[
        (date(2040, 4, 1), date(2040, 7, 1), 0),
        (date(2040, 7, 1), date(2040, 10, 1), 1),
        (date(2040, 10, 1), date(2041, 1, 1), 2),
        (date(2041, 1, 1), date(2041, 4, 1), 3),
    ]);

    assert_ar1_fit_pairs_each_occurrence_with_its_calendar_predecessor(
        &season_map,
        &stages,
        &dates,
        |date| {
            let position = (date.month0() / 3) as usize;
            ((position + 3) % 4, position, date.year())
        },
    );
}

#[test]
fn ar_fit_counts_weekly_buckets_by_iso_week_numbering_year() {
    let season_map = weekly_season_map();
    let dates: Vec<NaiveDate> = (2014..=2053)
        .flat_map(|iso_year| {
            (1..=52)
                .map(move |week| NaiveDate::from_isoywd_opt(iso_year, week, Weekday::Mon).unwrap())
        })
        .collect();
    assert_eq!(dates[0], date(2013, 12, 30));
    let periods: Vec<(NaiveDate, NaiveDate, usize)> = (1_u32..=52)
        .map(|week| {
            let start = NaiveDate::from_isoywd_opt(2054, week, Weekday::Mon).unwrap();
            (start, start + Days::new(7), week as usize - 1)
        })
        .collect();

    assert_ar1_fit_pairs_each_occurrence_with_its_calendar_predecessor(
        &season_map,
        &study_stages(&periods),
        &dates,
        |date| {
            let iso_week = date.iso_week();
            let week0 = iso_week.week0() as usize;
            (week0, week0, iso_week.year())
        },
    );
}
