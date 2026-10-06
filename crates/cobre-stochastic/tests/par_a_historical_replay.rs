#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

//! Regression tests: PAR(p)-A annual + historical replay must reproduce the
//! historical observation exactly when the stage-0 lag state equals the
//! derived lag seed (the rolling-chain seed).
//!
//! Two standing contracts the tests pin:
//! - When a model carries `annual: Some(_)`, `PrecomputedPar` widens the `psi`
//!   stride to 12 and spreads ψ̂ across the extra positions as `psi_hat / 12`;
//!   any PAR primitive that iterates only `psi[0..order]` drops the annual
//!   contribution at standardisation time and diverges the replay.
//! - `standardize_historical_windows` inverts η against the derived lag seed,
//!   not the window's own pre-study lags; inverting against the window lags
//!   offsets the replay algebraically whenever the two differ.

use chrono::{Months, NaiveDate};
use cobre_core::{
    EntityId,
    scenario::{AnnualComponent, InflowHistoryRow, InflowModel},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageLagTransition,
        StageRiskConfig, StageStateConfig,
    },
};
use cobre_stochastic::{
    DerivedSeed, HistoricalScenarioLibrary, check_historical_structure, evaluate_par_batch,
    par::precompute::PrecomputedPar, standardize_historical_windows,
};

fn monthly_stage(index: usize, season_id: usize) -> Stage {
    let month = u32::try_from(season_id).expect("season_id fits in u32") + 1;
    Stage {
        index,
        id: i32::try_from(index).expect("index fits in i32"),
        start_date: NaiveDate::from_ymd_opt(2024, month, 1).expect("valid date"),
        end_date: NaiveDate::from_ymd_opt(2024, month, 28).expect("valid date"),
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

fn row(hydro_id: EntityId, year: i32, month0: u32, value: f64) -> InflowHistoryRow {
    let start_date = NaiveDate::from_ymd_opt(year, month0 + 1, 1).expect("valid date");
    InflowHistoryRow {
        hydro_id,
        start_date,
        end_date: start_date
            .checked_add_months(Months::new(1))
            .expect("valid date"),
        value_m3s: value,
    }
}

fn uniform_monthly_transition() -> StageLagTransition {
    StageLagTransition {
        accumulate_weight: 1.0,
        spillover_weight: 0.0,
        finalize_period: true,
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    }
}

/// Shifts `lag_state` (h-major: `lag_state[h * safe_max_order + l]`) by one step,
/// inserting `realised` at lag 0; assumes `n_hydros == 1`.
fn advance_lag_state_uniform(lag_state: &mut [f64], realised: f64, max_order: usize) {
    let safe = max_order.max(1);
    for l in (1..safe).rev() {
        lag_state[l] = lag_state[l - 1];
    }
    lag_state[0] = realised;
}

// ---------------------------------------------------------------------------
// T1: AR(1)+PAR-A, derived lag seed == window pre-study lags
// ---------------------------------------------------------------------------

/// AR(1) + PAR(p)-A on a single hydro, the derived lag seed == the window's
/// own pre-study lags: every stage's observation must reproduce exactly.
#[test]
fn par_a_historical_replay_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    // The 12 pre-study stages (-12..0) exist so the lag-stage stats lookup
    // resolves instead of falling through to season-only stats.
    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0 + f64::from(sid).abs() * 0.5,
            std_m3s: 20.0,
            ar_coefficients: vec![0.4],
            residual_std_ratio: 0.85,
            annual: Some(annual.clone()),
        })
        .collect();
    all_models.extend((0_i32..12).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 100.0 + f64::from(sid) * 5.0,
        std_m3s: 25.0 + f64::from(sid) * 1.0,
        ar_coefficients: vec![0.4],
        residual_std_ratio: 0.85,
        annual: Some(annual.clone()),
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");

    assert_eq!(
        par.max_order(),
        12,
        "PAR-A must widen max_order to 12; test does not exercise the fix otherwise"
    );

    // Observations are arbitrary but distinct so a roundtrip miss is loud.
    let window_year: i32 = 1990;
    let pre_window_year: i32 = window_year - 1;
    let history_values: Vec<f64> = (0..24).map(|i| 80.0 + 7.5 * f64::from(i)).collect();
    let mut history: Vec<InflowHistoryRow> = Vec::with_capacity(24);
    for m in 0..12_u32 {
        history.push(row(hydro, pre_window_year, m, history_values[m as usize]));
    }
    for m in 0..12_u32 {
        history.push(row(hydro, window_year, m, history_values[12 + m as usize]));
    }

    // The derived seed is in lag order: lag-1 first (Dec 1989 =
    // history_values[11]), down to lag-12 (Jan 1989 = history_values[0]).
    let max_order = par.max_order();
    let past_values_m3s: Vec<f64> = (0..max_order).map(|lag| history_values[11 - lag]).collect();
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(1, 12, 1, par.max_order(), vec![window_year]);
    let hydro_ids = [hydro];
    let structure = check_historical_structure(&library, &hydro_ids, &stages).unwrap();
    standardize_historical_windows(
        &structure,
        &mut library,
        &history,
        &par,
        &[window_year],
        None,
        DerivedSeed {
            lag_values: &past_values_m3s,
            l_state: max_order,
            accum: &[],
            weight: &[],
        },
        &transitions,
        0,
    );

    let n_hydros = 1;
    let safe_max_order = max_order.max(1);
    let mut lag_state = vec![0.0_f64; n_hydros * safe_max_order];
    for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
        lag_state[lag] = v;
    }

    // evaluate_par_batch expects the lag_matrix layout `[lag * n_hydros + hydro]`,
    // which coincides with this h-major lag_state only because n_hydros == 1.
    let mut reconstructed = vec![0.0_f64];
    for t in 0..12 {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let target = history_values[12 + t];
        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T1 stage {t}: reconstructed {got:.12} != target {target:.12} \
             (diff {:.3e}) — PAR-A annual contribution likely dropped \
             from η at standardisation time",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T2: AR(1)+PAR-A, derived lag seed deliberately differs from window pre-study lags
// ---------------------------------------------------------------------------

/// The derived lag seed differs from the window's antecedent lags: η is
/// inverted against the derived seed, so a forward pass from that seed
/// reproduces the target exactly while one from the window lags would diverge.
#[test]
fn t2_derived_seed_differs_from_window_lags_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.4],
            residual_std_ratio: 0.85,
            annual: Some(annual.clone()),
        })
        .collect();
    all_models.extend((0_i32..12).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 100.0 + f64::from(sid) * 5.0,
        std_m3s: 25.0 + f64::from(sid) * 1.0,
        ar_coefficients: vec![0.4],
        residual_std_ratio: 0.85,
        annual: Some(annual.clone()),
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    let max_order = par.max_order();
    assert_eq!(max_order, 12);

    let window_year: i32 = 1990;
    let pre_window_year: i32 = window_year - 1;
    let window_lag_values: Vec<f64> = (0..12).map(|i| 80.0 + 3.0 * f64::from(i)).collect();
    let study_values: Vec<f64> = (0..12).map(|i| 150.0 + 5.0 * f64::from(i)).collect();

    let mut history: Vec<InflowHistoryRow> = Vec::with_capacity(24);
    for m in 0..12_u32 {
        history.push(row(
            hydro,
            pre_window_year,
            m,
            window_lag_values[m as usize],
        ));
    }
    for m in 0..12_u32 {
        history.push(row(hydro, window_year, m, study_values[m as usize]));
    }

    let past_values_m3s: Vec<f64> = vec![200.0; max_order];
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(1, 12, 1, max_order, vec![window_year]);
    let hydro_ids = [hydro];
    let structure = check_historical_structure(&library, &hydro_ids, &stages).unwrap();
    standardize_historical_windows(
        &structure,
        &mut library,
        &history,
        &par,
        &[window_year],
        None,
        DerivedSeed {
            lag_values: &past_values_m3s,
            l_state: max_order,
            accum: &[],
            weight: &[],
        },
        &transitions,
        0,
    );

    let safe_max_order = max_order.max(1);
    let mut lag_state = vec![0.0_f64; safe_max_order];
    for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
        lag_state[lag] = v;
    }

    let mut reconstructed = vec![0.0_f64];
    for (t, &target) in study_values.iter().enumerate() {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T2 stage {t}: replay from derived seed: reconstructed {got:.12} != \
             target {target:.12} (diff {:.3e})",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T3: AR(0), derived lag seed == window lags (no-AR regression baseline)
// ---------------------------------------------------------------------------

/// AR(0)+PAR-A: with no AR coefficient, `η = (obs - det_base) / sigma`, so a
/// replay from a derived seed matching the window lags reproduces each observation.
#[test]
fn t3_ar0_par_a_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: Some(annual.clone()),
        })
        .collect();
    all_models.extend((0_i32..12).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 100.0 + f64::from(sid) * 5.0,
        std_m3s: 25.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: Some(annual.clone()),
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    let max_order = par.max_order();
    assert_eq!(max_order, 12, "PAR-A must widen max_order to 12");

    let window_year: i32 = 1990;
    let pre_window_year: i32 = window_year - 1;
    let pre_values: Vec<f64> = (0..12).map(|i| 70.0 + 4.0 * f64::from(i)).collect();
    let study_values: Vec<f64> = (0..12).map(|i| 130.0 + 6.0 * f64::from(i)).collect();

    let mut history: Vec<InflowHistoryRow> = Vec::with_capacity(24);
    for m in 0..12_u32 {
        history.push(row(hydro, pre_window_year, m, pre_values[m as usize]));
    }
    for m in 0..12_u32 {
        history.push(row(hydro, window_year, m, study_values[m as usize]));
    }

    // Lag order: lag-1 first (Dec 1989), down to lag-12 (Jan 1989).
    let past_values_m3s: Vec<f64> = (0..max_order).map(|lag| pre_values[11 - lag]).collect();
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(1, 12, 1, max_order, vec![window_year]);
    let hydro_ids = [hydro];
    let structure = check_historical_structure(&library, &hydro_ids, &stages).unwrap();
    standardize_historical_windows(
        &structure,
        &mut library,
        &history,
        &par,
        &[window_year],
        None,
        DerivedSeed {
            lag_values: &past_values_m3s,
            l_state: max_order,
            accum: &[],
            weight: &[],
        },
        &transitions,
        0,
    );

    let safe_max_order = max_order.max(1);
    let mut lag_state = vec![0.0_f64; safe_max_order];
    for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
        lag_state[lag] = v;
    }

    let mut reconstructed = vec![0.0_f64];
    for (t, &target) in study_values.iter().enumerate() {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T3 stage {t}: AR(0)+PAR-A reconstructed {got:.12} != target {target:.12} \
             (diff {:.3e})",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T4: AR(1), derived lag seed has fewer entries than max_order (absent = 0)
// ---------------------------------------------------------------------------

/// A derived lag seed shorter than `max_order`: missing lag slots default
/// to `0.0`, and a replay seeded from the same truncated state still reproduces
/// every study observation.
#[test]
fn t4_derived_seed_shorter_than_max_order_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..4).map(|i| monthly_stage(i, i)).collect();

    let mut all_models: Vec<InflowModel> = (-2_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 100.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.3, 0.15],
            residual_std_ratio: 0.9,
            annual: None,
        })
        .collect();
    all_models.extend((0_i32..4).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 110.0 + f64::from(sid) * 3.0,
        std_m3s: 20.0,
        ar_coefficients: vec![0.3, 0.15],
        residual_std_ratio: 0.9,
        annual: None,
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    assert_eq!(par.max_order(), 2);

    let window_year: i32 = 2000;
    // 4-stage study from season 0 ⇒ pre-study seasons 2 and 3 are Nov, Dec 1999.
    let pre_2 = 55.0_f64; // lag-2: Nov 1999
    let pre_1 = 66.0_f64; // lag-1: Dec 1999
    let study_values = [130.0_f64, 145.0, 120.0, 110.0];

    let mut history: Vec<InflowHistoryRow> = Vec::new();
    // Populate all of 1999 so the lags resolve.
    for m in 0..12_u32 {
        let v = if m == 10 {
            pre_2
        } else if m == 11 {
            pre_1
        } else {
            100.0
        };
        history.push(row(hydro, 1999, m, v));
    }
    for (t, &v) in study_values.iter().enumerate() {
        history.push(row(
            hydro,
            window_year,
            u32::try_from(t).expect("t fits u32"),
            v,
        ));
    }

    let derived_lag_values = [66.0]; // lag-1 only; lag-2 defaults to 0.0
    let transitions: Vec<StageLagTransition> =
        (0..4).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(1, 4, 1, par.max_order(), vec![window_year]);
    let hydro_ids = [hydro];
    let structure = check_historical_structure(&library, &hydro_ids, &stages).unwrap();
    standardize_historical_windows(
        &structure,
        &mut library,
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
        0,
    );

    let max_order = par.max_order();
    let safe = max_order.max(1);
    let mut lag_state = vec![0.0_f64; safe];
    lag_state[0] = 66.0; // lag-1; lag_state[1] stays 0.0 (lag-2 absent)

    let mut reconstructed = vec![0.0_f64];
    for (t, &target) in study_values.iter().enumerate() {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T4 stage {t}: truncated derived seed: reconstructed {got:.12} != \
             target {target:.12} (diff {:.3e})",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T5: Two windows, derived lag seed shared; both windows replay exactly
// ---------------------------------------------------------------------------

#[test]
fn t5_two_windows_shared_derived_seed_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.4],
            residual_std_ratio: 0.85,
            annual: Some(annual.clone()),
        })
        .collect();
    all_models.extend((0_i32..12).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 100.0 + f64::from(sid) * 5.0,
        std_m3s: 25.0 + f64::from(sid) * 1.0,
        ar_coefficients: vec![0.4],
        residual_std_ratio: 0.85,
        annual: Some(annual.clone()),
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    let max_order = par.max_order();
    assert_eq!(max_order, 12);

    // Years must not overlap (incl. pre-windows) so the obs_table has no
    // collisions: 1990 (pre 1989) and 1992 (pre 1991).
    let window_years = [1990_i32, 1992];
    let history_for = |w_year: i32, offset: f64| {
        let pre_year = w_year - 1;
        let mut h: Vec<InflowHistoryRow> = Vec::with_capacity(24);
        for m in 0..12_u32 {
            h.push(row(hydro, pre_year, m, 80.0 + 7.5 * f64::from(m) + offset));
        }
        for m in 0..12_u32 {
            h.push(row(hydro, w_year, m, 160.0 + 3.0 * f64::from(m) + offset));
        }
        h
    };
    let mut full_history = history_for(1990, 0.0);
    full_history.extend(history_for(1992, 50.0));

    // Shared across windows and different from both windows' antecedent lags,
    // so the replay re-roots against the derived seed rather than the window
    // lags.
    let past_values_m3s = vec![150.0_f64; max_order];
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(2, 12, 1, max_order, window_years.to_vec());
    let hydro_ids = [hydro];
    let structure = check_historical_structure(&library, &hydro_ids, &stages).unwrap();
    standardize_historical_windows(
        &structure,
        &mut library,
        &full_history,
        &par,
        &window_years,
        None,
        DerivedSeed {
            lag_values: &past_values_m3s,
            l_state: max_order,
            accum: &[],
            weight: &[],
        },
        &transitions,
        0,
    );

    let safe = max_order.max(1);
    let mut reconstructed = vec![0.0_f64];
    for (w, &w_year) in window_years.iter().enumerate() {
        let offset = if w == 0 { 0.0 } else { 50.0 };
        let mut lag_state = vec![0.0_f64; safe];
        for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
            lag_state[lag] = v;
        }

        for t in 0..12_usize {
            let eta = library.eta_slice(w, t);
            evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

            #[allow(clippy::cast_precision_loss)] // t in 0..12, exact in f64
            let target = 160.0 + 3.0 * (t as f64) + offset;
            let got = reconstructed[0];
            assert!(
                (got - target).abs() < 1e-9,
                "T5 window={w} (year={w_year}) stage {t}: reconstructed {got:.12} != \
                 target {target:.12} (diff {:.3e})",
                (got - target).abs(),
            );

            advance_lag_state_uniform(&mut lag_state, got, max_order);
        }
    }
}
