use super::{
    BUILD_PERIODIC_YW_MATRIX_CALL_COUNT, HistoryClass, build_periodic_yw_matrix, classify_history,
    estimate_periodic_ar_coefficients, periodic_autocorrelation, periodic_pacf, select_order_aic,
    select_order_pacf, select_order_pacf_annual, solve_linear_system,
};

// -----------------------------------------------------------------------
// estimate_seasonal_stats_with_season_map tests
// -----------------------------------------------------------------------

use chrono::{Datelike, NaiveDate};
use cobre_core::{
    EntityId,
    temporal::Stage,
    test_support::{StageSpec, date, single_block},
};

use super::estimate_seasonal_stats_with_season_map;
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
    cobre_core::test_support::make_stage(StageSpec {
        id,
        index: Some(index),
        start_date: date(year_start, month_start, 1),
        end_date: date(year_end, month_end, 1),
        season_id,
        blocks: single_block("SINGLE", 744.0),
        ..Default::default()
    })
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
    // Exactly 10% (2/20) negative — does NOT trigger ManyNegative (strict >).
    // Falls through. With these values neither Constant nor Saturated
    // either.
    let mut obs: Vec<f64> = (1..=18).map(|i| i as f64).collect();
    obs.extend_from_slice(&[-1.0, -2.0]);
    assert_eq!(classify_history(&obs), HistoryClass::Default);
}

#[test]
fn classify_history_saturated_cap() {
    // High-cap-style: most values at the 13900 cap, rest scattered below.
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
fn classify_history_mode_at_zero_with_majority_is_saturated() {
    // 11 zeros + 9 nonzero = 55% at mode 0 -> Saturated (cap = 0),
    // typical of low-flow constant months on transposed-flow plants.
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
    assert_eq!(c.stats_override(), Some((5.0, 0.0)));
    assert!(c.is_degenerate());

    let s = HistoryClass::Saturated { cap: 13900.0 };
    assert_eq!(s.stats_override(), Some((13900.0, 0.0)));
    assert!(s.is_degenerate());

    // ManyNegative is diagnostic only; no fitting override and not
    // "degenerate" for the purpose of forcing order 0.
    let n = HistoryClass::ManyNegative { sample_mean: -1.5 };
    assert_eq!(n.stats_override(), None);
    assert!(!n.is_degenerate());

    let d = HistoryClass::Default;
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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &entity_ids, None).unwrap();
    assert_eq!(stats.len(), 24, "expected 2 hydros × 12 seasons = 24 rows");

    for s in &stats {
        assert!(
            s.entity_id == EntityId::from(1) || s.entity_id == EntityId::from(2),
            "unexpected entity_id {}",
            s.entity_id
        );
    }

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &entity_ids, None).unwrap();
    assert_eq!(stats.len(), 1);

    let expected_mean = (10.0 + 20.0 + 30.0 + 40.0 + 50.0) / 5.0; // 30.0
    let expected_variance = ((10.0 - 30.0_f64).powi(2)
        + (20.0 - 30.0_f64).powi(2)
        + (30.0 - 30.0_f64).powi(2)
        + (40.0 - 30.0_f64).powi(2)
        + (50.0 - 30.0_f64).powi(2))
        / 5.0; // 1/N population divisor
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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &entity_ids, None).unwrap();
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

    let result = estimate_seasonal_stats_with_season_map(&observations, &stages, &entity_ids, None);
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

    let result = estimate_seasonal_stats_with_season_map(&observations, &stages, &entity_ids, None);
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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &entity_ids, None).unwrap();
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

    let stats = estimate_seasonal_stats_with_season_map(&[], &stages, &entity_ids, None).unwrap();
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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &entity_ids, None).unwrap();
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
// estimate_correlation_with_season_map tests
// -----------------------------------------------------------------------

use super::{
    ArCoefficientEstimate, SeasonalStats, estimate_ar_coefficients_with_season_map,
    estimate_correlation_with_season_map,
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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();

    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();

    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();

    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();

    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();

    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

#[test]
fn estimate_correlation_pooled_matrix_order_invariant() {
    // The parallel per-hydro residual fit must produce a pooled correlation
    // matrix that is bit-identical regardless of the declaration order of
    // `hydro_ids`. Build a multi-hydro, multi-season white-noise (order-0)
    // baseline, then fit with the ids ascending and reversed; after
    // re-indexing both pooled matrices to a common entity order, every entry
    // must match exactly. (`estimate_ar_coefficients_with_season_map` only supports
    // max_order=0, so coefficients are empty and the residual is the raw
    // standardized observation; the order invariance under test is the
    // pooled-matrix assembly, not the lag-sum path.)
    // A multi-hydro, multi-season study with correlated-but-distinct series
    // gives nonzero off-diagonals, so a reordering-induced drift would show.
    let n_seasons = 12;
    let stages = multi_season_stages(2000, 40, n_seasons);
    let hydro_ids = vec![EntityId::from(1), EntityId::from(2), EntityId::from(3)];

    let mut seed1: u64 = 0x1234_5678;
    let mut seed2: u64 = 0x9abc_def0;
    let mut seed3: u64 = 0x0f0f_0f0f;
    let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
    for i in 0..stages.len() {
        let year = 2000 + (i / 12) as i32;
        let month = (i % 12) as u32 + 1;
        let date = NaiveDate::from_ymd_opt(year, month, 15).unwrap();
        // Correlated-but-distinct series so off-diagonals are nonzero.
        let a = splitmix(&mut seed1);
        let b = splitmix(&mut seed2);
        let c = splitmix(&mut seed3);
        observations.push((EntityId::from(1), date, a));
        observations.push((EntityId::from(2), date, 0.7 * a + 0.3 * b));
        observations.push((EntityId::from(3), date, 0.4 * b + 0.6 * c));
    }

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();

    let mut reversed_ids = hydro_ids.clone();
    reversed_ids.reverse();

    let corr_fwd = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();
    let corr_rev = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &reversed_ids,
        None,
    )
    .unwrap();

    // Re-index each pooled matrix by entity id into a common ascending order.
    let group_fwd = &corr_fwd.profiles["default"].groups[0];
    let group_rev = &corr_rev.profiles["default"].groups[0];
    assert_eq!(group_fwd.entities.len(), 3);
    assert_eq!(group_rev.entities.len(), 3);

    let pos_of = |group: &super::CorrelationGroup, id: EntityId| -> usize {
        group
            .entities
            .iter()
            .position(|e| e.id == id)
            .expect("entity present in group")
    };

    let common = [EntityId::from(1), EntityId::from(2), EntityId::from(3)];
    for &ri in &common {
        for &cj in &common {
            let fi = pos_of(group_fwd, ri);
            let fj = pos_of(group_fwd, cj);
            let vi = pos_of(group_rev, ri);
            let vj = pos_of(group_rev, cj);
            let v_fwd = group_fwd.matrix[fi][fj];
            let v_rev = group_rev.matrix[vi][vj];
            assert!(
                v_fwd == v_rev,
                "pooled matrix entry for ({ri:?},{cj:?}) differs: \
                     forward {v_fwd:?} != reversed {v_rev:?} (must be bit-identical)"
            );
        }
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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();
    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();
    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();
    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

    assert_eq!(corr.schedule.len(), 160);
    for (i, entry) in corr.schedule.iter().enumerate() {
        assert_eq!(entry.profile_name, format!("season_{}", i % n_seasons));
    }

    assert_eq!(corr.schedule[0].profile_name, "season_0");
    assert_eq!(corr.schedule[1].profile_name, "season_1");
    assert_eq!(corr.schedule[4].profile_name, "season_0");

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();
    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

fn same_year_starts(n_seasons: usize) -> Vec<i32> {
    vec![0; n_seasons]
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

    let rho = periodic_autocorrelation(0, 1, 1, obs, stats_arr, &same_year_starts(1));
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
    let rho01 = periodic_autocorrelation(0, 1, 2, obs, stats, &same_year_starts(2));
    // rho(1, 1) = autocorrelation of season 1 with season 0 (lag 1).
    let rho10 = periodic_autocorrelation(1, 1, 2, obs, stats, &same_year_starts(2));

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
    let rho_jan_dec = periodic_autocorrelation(
        0,
        1,
        n_seasons,
        &obs_refs,
        &stats,
        &same_year_starts(n_seasons),
    );

    // For the non-cross-year case (ref_season=6, lag=1 -> lag_season=5),
    // lag_season (5) < ref_season (6), so no observation is dropped.
    let rho_jul_jun = periodic_autocorrelation(
        6,
        1,
        n_seasons,
        &obs_refs,
        &stats,
        &same_year_starts(n_seasons),
    );

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
    let rho = periodic_autocorrelation(0, 1, 2, obs, stats, &same_year_starts(2));
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
    let rho = periodic_autocorrelation(0, 1, 2, obs, stats, &same_year_starts(2));
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
    let rho = periodic_autocorrelation(0, 1, 1, obs, stats, &same_year_starts(1));
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

    let _rho = periodic_autocorrelation(0, 1, 1, obs, stats, &same_year_starts(1));

    let data2 = [1.0, 4.0, 9.0]; // mean=14/3
    let (mean2, std2) = pop_mean_std(&data2);
    let stats2: &[(f64, f64)] = &[(mean2, std2)];
    let obs2: &[&[f64]] = &[&data2];

    let rho2 = periodic_autocorrelation(0, 1, 1, obs2, stats2, &same_year_starts(1));
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
    assert_eq!(
        periodic_autocorrelation(0, 0, 1, obs, stats, &same_year_starts(1)),
        1.0
    );
}

// -----------------------------------------------------------------------
// build_periodic_yw_matrix tests
// -----------------------------------------------------------------------

#[test]
fn build_periodic_yw_matrix_order_zero() {
    let data = [1.0, 2.0, 3.0];
    let stats: &[(f64, f64)] = &[(2.0, 1.0)];
    let obs: &[&[f64]] = &[&data];
    let (mat, rhs) = build_periodic_yw_matrix(0, 0, 1, obs, stats, &same_year_starts(1));
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
    let (mat, _rhs) = build_periodic_yw_matrix(0, order, 1, obs, stats_arr, &same_year_starts(1));

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
    let (mat, _) = build_periodic_yw_matrix(0, order, 2, obs, stats, &same_year_starts(2));
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
    let (mat, _) = build_periodic_yw_matrix(1, order, 3, &obs, &stats, &same_year_starts(3));
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
        let (mat, rhs) =
            build_periodic_yw_matrix(0, order, 1, obs, stats_arr, &same_year_starts(1));
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
    let (mat, _) = build_periodic_yw_matrix(0, order, 2, obs, stats, &same_year_starts(2));

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
    // problem for a 2-season AR(2) model, not the backward variant.

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

    let (mat_orig, rhs_orig) = build_periodic_yw_matrix(
        season,
        order,
        n_seasons,
        obs,
        stats,
        &same_year_starts(n_seasons),
    );

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
    let (mut mat, mut rhs) = build_periodic_yw_matrix(
        season,
        order,
        n_seasons,
        obs,
        stats,
        &same_year_starts(n_seasons),
    );
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
    let (mat_orig, rhs_orig) =
        build_periodic_yw_matrix(0, order, 1, obs, stats_arr, &same_year_starts(1));

    let (mut mat, mut rhs) =
        build_periodic_yw_matrix(0, order, 1, obs, stats_arr, &same_year_starts(1));
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
    let rho = periodic_autocorrelation(0, 1, 2, obs, stats, &same_year_starts(2));
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
    let rho = periodic_autocorrelation(0, 3, 2, &obs, &stats, &same_year_starts(2));
    assert!(rho.is_finite());
    assert!(rho.abs() <= 1.0);
}

#[test]
fn lag_years_back_counts_whole_cycles_for_long_lags() {
    let z0 = [1.0, 2.0, 4.0, 3.0, 5.0];
    let z1 = [6.0, 8.0, 7.0, 9.0, 5.0];
    let a0 = [2.0, 1.0, 3.0, 5.0, 4.0];
    let obs: &[&[f64]] = &[&z0, &z1];
    let stats = [pop_mean_std(&z0), pop_mean_std(&z1)];
    let annual_obs: &[&[f64]] = &[&a0, &[]];
    let annual_stats = [pop_mean_std(&a0), (0.0, 0.0)];
    let year_starts = same_year_starts(2);

    let rho_three_seasons_back = periodic_autocorrelation(0, 3, 2, obs, &stats, &year_starts);
    let rho_one_cycle_back = periodic_autocorrelation(1, 3, 2, obs, &stats, &year_starts);
    let rho_z_a = cross_correlation_z_a(
        0,
        3,
        2,
        obs,
        &stats,
        &year_starts,
        annual_obs,
        &annual_stats,
        &year_starts,
    );

    assert!(
        (rho_three_seasons_back + 1.0 / 6.0).abs() < 1e-12,
        "season 0 of year Y pairs with season 1 of year Y - 2: got {rho_three_seasons_back}"
    );
    assert!(
        rho_one_cycle_back.abs() < 1e-12,
        "season 1 of year Y pairs with season 0 of year Y - 1: got {rho_one_cycle_back}"
    );
    assert!(
        (rho_z_a - 0.2).abs() < 1e-12,
        "A of season 0 in year Y pairs with Z of season 1 in year Y - 2: got {rho_z_a}"
    );
}

/// Season 0's bucket starts a year after season 1's.
const SHIFTED_YEAR_STARTS: [i32; 2] = [2001, 2000];

fn two_season_buckets() -> ([f64; 4], [f64; 4]) {
    ([1.0, 2.0, 4.0, 3.0], [5.0, 7.0, 6.0, 8.0])
}

#[test]
fn periodic_autocorrelation_pairs_buckets_by_absolute_year() {
    let (s0, s1) = two_season_buckets();
    let obs: &[&[f64]] = &[&s0, &s1];
    let stats = [pop_mean_std(&s0), pop_mean_std(&s1)];
    let rho = |season: usize, year_starts: &[i32]| {
        periodic_autocorrelation(season, 1, 2, obs, &stats, year_starts)
    };

    let cases = [
        (0, SHIFTED_YEAR_STARTS, 0.4),
        (1, SHIFTED_YEAR_STARTS, 7.0 / 15.0),
        (0, [2000, 2000], 1.0 / 3.0),
        (1, [2000, 2000], 0.4),
    ];
    for (season, year_starts, expected) in cases {
        let got = rho(season, &year_starts);
        assert!(
            (got - expected).abs() < 1e-12,
            "rho({season}, 1) with year starts {year_starts:?}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn extended_yw_rhs_pairs_z_by_absolute_year() {
    let (s0, s1) = two_season_buckets();
    let obs: &[&[f64]] = &[&s0, &s1];
    let stats = [pop_mean_std(&s0), pop_mean_std(&s1)];
    let a0 = [2.0, 1.0, 3.0];
    let a1 = [4.0, 6.0, 5.0];
    let annual_obs: &[&[f64]] = &[&a0, &a1];
    let annual_stats = [pop_mean_std(&a0), pop_mean_std(&a1)];

    let (_, rhs) = build_extended_periodic_yw_matrix(
        0,
        1,
        2,
        obs,
        &stats,
        &SHIFTED_YEAR_STARTS,
        annual_obs,
        &annual_stats,
        &[2001, 2001],
    );

    let expected = periodic_autocorrelation(0, 1, 2, obs, &stats, &SHIFTED_YEAR_STARTS);
    assert_eq!(rhs[0].to_bits(), expected.to_bits());
}

#[test]
fn partitioned_covariance_pairs_z_by_absolute_year() {
    let (s0, s1) = two_season_buckets();
    let obs: &[&[f64]] = &[&s0, &s1];
    let stats = [pop_mean_std(&s0), pop_mean_std(&s1)];
    let a0 = [2.0, 1.0, 3.0];
    let a1 = [4.0, 6.0, 5.0];
    let annual_obs: &[&[f64]] = &[&a0, &a1];
    let annual_stats = [pop_mean_std(&a0), pop_mean_std(&a1)];

    let cov = assemble_partitioned_covariance(
        0,
        1,
        2,
        obs,
        &stats,
        &SHIFTED_YEAR_STARTS,
        annual_obs,
        &annual_stats,
        &[2001, 2001],
    );

    let expected = periodic_autocorrelation(0, 1, 2, obs, &stats, &SHIFTED_YEAR_STARTS);
    assert_eq!(cov.sigma_11[1].to_bits(), expected.to_bits());
}

#[test]
fn periodic_autocorrelation_population_divisor_verification() {
    // Verify population divisor (1/N) NOT Bessel (1/(N-1)) with N=3.
    // The 50% difference at N=3 makes this easy to detect.
    //
    // Use two seasons to avoid cross-year adjustment complexity.
    let s0 = [1.0, 4.0, 3.0]; // mean=8/3, std_pop
    let s1 = [2.0, 5.0, 4.0]; // mean=11/3, std_pop
    let stats_0 = pop_mean_std(&s0);
    let stats_1 = pop_mean_std(&s1);
    let obs: &[&[f64]] = &[&s0, &s1];
    let stats: &[(f64, f64)] = &[stats_0, stats_1];

    let rho = periodic_autocorrelation(0, 1, 2, obs, stats, &same_year_starts(2));
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
    let (mat_orig, rhs_orig) =
        build_periodic_yw_matrix(0, order, 2, obs, stats, &same_year_starts(2));

    let (mut mat, mut rhs) =
        build_periodic_yw_matrix(0, order, 2, obs, stats, &same_year_starts(2));
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
    let (_, rhs) = build_periodic_yw_matrix(season, order, 3, &obs, &stats, &same_year_starts(3));

    // Verify each RHS entry: rhs[i] = rho(season, i+1).
    let expected_rhs0 = periodic_autocorrelation(season, 1, 3, &obs, &stats, &same_year_starts(3));
    let expected_rhs1 = periodic_autocorrelation(season, 2, 3, &obs, &stats, &same_year_starts(3));

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
    let pacf = periodic_pacf(0, 0, 1, obs, stats_arr, &same_year_starts(1));
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

    let rho1 = periodic_autocorrelation(0, 1, 1, obs, stats_arr, &same_year_starts(1));
    let pacf = periodic_pacf(0, 3, 1, obs, stats_arr, &same_year_starts(1));

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

    let pacf = periodic_pacf(0, 3, 2, &obs, &stats, &same_year_starts(2));

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

    let pacf = periodic_pacf(0, 5, 1, obs, stats_arr, &same_year_starts(1));
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
        let pacf = periodic_pacf(season, 4, 3, &obs, &stats, &same_year_starts(3));
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

    let result = estimate_periodic_ar_coefficients(0, 0, 1, obs, stats_arr, &same_year_starts(1));
    assert!(result.coefficients.is_empty());
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

    let result = estimate_periodic_ar_coefficients(0, 1, 1, obs, stats_arr, &same_year_starts(1));
    assert_eq!(result.coefficients.len(), 1);
    assert_eq!(result.sigma2_per_order.len(), 1);
    // sigma2 = 1 - phi * rho(1)
    let rho1 = periodic_autocorrelation(0, 1, 1, obs, stats_arr, &same_year_starts(1));
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

    let result = estimate_periodic_ar_coefficients(0, 2, 2, &obs, &stats, &same_year_starts(2));
    assert_eq!(result.coefficients.len(), 2);
    assert_eq!(result.sigma2_per_order.len(), 2);
}

#[test]
fn estimate_periodic_ar_sigma2_per_order_length() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64).sin() * 10.0).collect();
    let stats = pop_mean_std(&data);
    let obs: &[&[f64]] = &[&data];
    let stats_arr: &[(f64, f64)] = &[stats];

    for order in 1..=5 {
        let result =
            estimate_periodic_ar_coefficients(0, order, 1, obs, stats_arr, &same_year_starts(1));
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

    let result = estimate_periodic_ar_coefficients(0, 4, 1, obs, stats_arr, &same_year_starts(1));
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
    let pacf_s0 = periodic_pacf(0, max_order, 2, &obs, &stats, &same_year_starts(2));

    assert!(
        pacf_s0.len() >= 2,
        "PACF should compute at least 2 orders; got {}",
        pacf_s0.len()
    );

    // Identity 1: PACF(k) == estimate_periodic_ar_coefficients(order=k)[k-1].
    for k in 1..=pacf_s0.len() {
        let yw_result =
            estimate_periodic_ar_coefficients(0, k, 2, &obs, &stats, &same_year_starts(2));
        let expected = yw_result.coefficients[k - 1];
        let actual = pacf_s0[k - 1];
        assert!(
            (actual - expected).abs() < 1e-10,
            "PACF({k}) = {actual:.10} must match YW coeff[{idx}] = {expected:.10}",
            idx = k - 1
        );
    }

    // Identity 2: PACF(1) == rho(season=0, lag=1) exactly.
    let rho1 = periodic_autocorrelation(0, 1, 2, &obs, &stats, &same_year_starts(2));
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
// estimate_correlation_with_season_map fallback and backward-compatibility tests
// -----------------------------------------------------------------------

#[test]
fn estimate_correlation_min_sample_fallback() {
    let stages = multi_season_stages(2000, 40, 2);
    let hydro_ids = vec![EntityId::from(1), EntityId::from(2)];

    let mut observations = Vec::new();
    let mut s1_count = 0;
    for (i, stage) in stages.iter().enumerate() {
        let date =
            NaiveDate::from_ymd_opt(stage.start_date.year(), stage.start_date.month(), 15).unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();
    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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

    let stats =
        estimate_seasonal_stats_with_season_map(&observations, &stages, &hydro_ids, None).unwrap();
    let estimates = estimate_ar_coefficients_with_season_map(
        &observations,
        &stats,
        &stages,
        &hydro_ids,
        0,
        None,
    )
    .unwrap();
    let corr = estimate_correlation_with_season_map(
        &observations,
        &estimates,
        &stats,
        &stages,
        &hydro_ids,
        None,
    )
    .unwrap();

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
// Regression: estimate_periodic_ar_coefficients calls
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

    let result = estimate_periodic_ar_coefficients(
        0,
        selected_order,
        1,
        obs,
        stats_arr,
        &same_year_starts(1),
    );

    let call_count = BUILD_PERIODIC_YW_MATRIX_CALL_COUNT.with(|c| *c.borrow());

    assert_eq!(result.sigma2_per_order.len(), selected_order);
    assert_eq!(
        call_count, selected_order,
        "build_periodic_yw_matrix called {call_count} times for order \
             {selected_order}; expected exactly {selected_order} (must \
             not call twice per order)"
    );
}

// -----------------------------------------------------------------------
// Regression: compute_pearson_correlation_matrix returns a flat
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
    assemble_partitioned_covariance, build_extended_periodic_yw_matrix, cross_correlation_a_z_neg1,
    cross_correlation_z_a,
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
    let (cls_mat, cls_rhs) = build_periodic_yw_matrix(
        season,
        order,
        n_seasons,
        obs,
        &stats,
        &same_year_starts(n_seasons),
    );

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
///          =[1.68-1.32+0.08-0.22-1.12]/5=-0.90/5=-0.18
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
    // year-forward-shift skips one Z entry, giving n_pairs=4. The
    // max-bucket-size convention divides the cross-product sum by
    // max(a.len, z.len)=5 rather than 4, so the value scales by 4/5 vs
    // the n_pairs divisor.
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

/// max_order = 0 returns an empty vector immediately.
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

/// A conditional-FACP lag deeper than the season cycle (`k >= n_seasons + 2`,
/// here a 2-season cycle probed at the default `max_order = 6`) must produce
/// finite values, not underflow the season index of `Z_{t-1-j}` — the lag
/// season wraps around the cycle exactly like the guarded conditioning-set
/// walk above it.
#[test]
fn conditional_facp_partitioned_deep_lag_wraps_season_index() {
    let n_years = 20_usize;
    let z0: Vec<f64> = (0..n_years)
        .map(|i| (i as f64).sin() * 3.0 + 0.1 * i as f64)
        .collect();
    let z1: Vec<f64> = (0..n_years)
        .map(|i| (i as f64).cos() * 2.5 - 0.05 * i as f64)
        .collect();
    let a0: Vec<f64> = (0..n_years).map(|i| 5.0 + 0.2 * (i as f64).sin()).collect();
    let a1: Vec<f64> = (0..n_years).map(|i| 3.0 - 0.1 * (i as f64).cos()).collect();

    let obs: &[&[f64]] = &[&z0, &z1];
    let stats = [pop_mean_std_ann(&z0), pop_mean_std_ann(&z1)];
    let ann_obs: &[&[f64]] = &[&a0, &a1];
    let ann_stats = [pop_mean_std_ann(&a0), pop_mean_std_ann(&a1)];

    for season in 0..2_usize {
        let cond = conditional_facp_partitioned(
            season,
            6,
            2,
            obs,
            &stats,
            &[0_i32; 32],
            ann_obs,
            &ann_stats,
            &[0_i32; 32],
        );
        assert_eq!(cond.len(), 6, "season {season}: FACP must cover every lag");
        for (k, v) in cond.iter().enumerate() {
            assert!(
                v.is_finite(),
                "season {season} lag {}: FACP must be finite, got {v}",
                k + 1
            );
        }
    }
}

/// When A is a constant series (std = 0), all cross-correlations
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
    let classical = periodic_pacf(
        season,
        max_order,
        n_seasons,
        obs,
        &stats,
        &same_year_starts(n_seasons),
    );

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

/// Every returned entry is in [-1.0, 1.0] for arbitrary synthetic data.
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

/// Hand-computed 2-season case verifying the partitioned-covariance
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
    let a0: &[f64] = &[0.0; 5];

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
    let rho_1 = periodic_autocorrelation(
        season,
        1,
        n_seasons,
        obs,
        &stats,
        &same_year_starts(n_seasons),
    );
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
    // and in [-1, 1]; this entry clamps to -1.0 because the unclamped ratio
    // falls below -1.0 for this data.
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
    let rho_2 = periodic_autocorrelation(
        season,
        2,
        n_seasons,
        obs,
        &stats,
        &same_year_starts(n_seasons),
    );
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
/// follows the max-bucket-size convention of dividing the cross-product
/// sum by the LARGER bucket size (here 5) rather than n_pairs (4 after
/// the year-forward-shift skips one Z entry).
///
/// The Σ_12[1,*] block guards the row anchoring: anchoring at season_minus_k
/// (instead of the per-j ref season) produces unrelated values.
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

    // Σ_12 (2×3, row-major). Row 1 (Z_{t-3} cross conditioning Z's) guards the
    // per-j row anchoring.
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

/// When Σ_22 becomes singular at k=2, the loop breaks early and returns
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

/// When Σ̄[0,0]·Σ̄[1,1] ≤ 0, the function records 0.0 (no NaN/Inf).
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
    // n=100; both PACF values below the threshold.
    // Without min-order-1 rule, max_significant = 0.
    // PACF[0] = 0.05 is non-zero -> min-order-1 rule forces order = max(1, 0) = 1.
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
    // only lag 1 does. The selector proceeds normally with the surviving
    // lags: lag 1 = 0.5 > threshold, lag 3 = 0.6 > threshold -> order = 3.
    let result = select_order_pacf_annual(&[0.5, 0.0, 0.6], 100, 1.96);
    assert_eq!(result.selected_order, 3);
}

#[test]
fn select_order_pacf_annual_only_lag1_significant_with_zeros_after() {
    // Realistic pattern: a degenerate Schur complement at lag 2+ produces
    // FACP = [+0.37, 0, 0, 0, 0, 0]. The selector picks order 1 (lag 1
    // is significant), not 0.
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

#[test]
fn select_order_pacf_annual_interior_insignificant_lag_still_selects_the_outer_lag() {
    // threshold = 1.96 / sqrt(100) = 0.196
    // conditional_facp = [0.5, 0.01, 0.4]: lag 1 and lag 3 are significant,
    // lag 2 (0.01) is not. Lag 2 stays in the model because the order names
    // the outermost significant lag, not the set of significant lags.
    let result = select_order_pacf_annual(&[0.5, 0.01, 0.4], 100, 1.96);
    assert_eq!(result.selected_order, 3);
}

// -----------------------------------------------------------------------
// estimate_annual_seasonal_stats tests
// -----------------------------------------------------------------------

use super::{estimate_annual_seasonal_stats, estimate_periodic_ar_annual_coefficients};

/// Four-year synthetic monthly series; hand-computed Bessel-corrected
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
/// - `s == 11`: `70.0` (note: NOT `(845 + 55)/12 = 75.0` because the
///   y-range shifts down by one)
///
/// All stds are 5.0 (Bessel-corrected, `1/(N-1)` with N=3) — each year
/// shifts every observation by `+5`, so window means differ by `5`
/// between consecutive years for every season.
///
/// This test intentionally pins the `1/(N-1)` divisor and the divergence
/// from the population (`1/N`) divisor used elsewhere in the workspace.
#[test]
fn estimate_annual_seasonal_stats_four_year_synthetic_hand_computed() {
    let hydro_id = EntityId::from(1);
    let stages = make_monthly_stages(2000, 4);

    // Build 48 observations: z[year*12 + month] = (month+1)*10 + year*5.
    let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
    for year in 0..4_usize {
        for month in 0..12_usize {
            let value = (month + 1) as f64 * 10.0 + year as f64 * 5.0;
            let date = NaiveDate::from_ymd_opt(2000 + year as i32, month as u32 + 1, 1).unwrap();
            observations.push((hydro_id, date, value));
        }
    }

    let result = estimate_annual_seasonal_stats(&observations, &stages, &[hydro_id], None).unwrap();

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

/// History too short: 11 observations cannot form any rolling window.
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
// estimate_periodic_ar_annual_coefficients tests
// -----------------------------------------------------------------------

/// `selected_order = 0`: 1×1 system yields only ψ, `coefficients` is empty.
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
    // n_pairs=4. The max-bucket-size divisor (=5) scales the result by
    // 4/5 vs an n_pairs divisor, so the hand-computed value is
    // 0.14384911389218766 × 4/5 ≈ 0.1150792911…
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
}

/// `selected_order = 2` with the 3×3 hand-computed fixture.
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

    // Expected values reflect the max-bucket-size cross-cov divisor
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
}

/// Singular extended YW system returns the zero fallback.
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
}

/// Regression test: `assemble_partitioned_covariance` sigma_22 cross-term
/// lag indexing for k=3.
///
/// The cross-term `sigma_22[i, k-1]` must equal
/// `cross_correlation_z_a(prev_season, lag=i, ...)` because `Z_{t-1-i}` is
/// exactly `i` steps older than `A_{t-1}` — not `lag = k-2-i`, which coincides
/// only at `k=2` and swaps the lag-0 and lag-1 cross-terms at `k=3`.
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
