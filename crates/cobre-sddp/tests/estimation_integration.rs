//! Integration tests for the estimation pipeline.
//!
//! Exercises [`cobre_sddp::estimation::estimate_from_history`] end-to-end with a
//! real temporary case directory, a synthetic `inflow_history.parquet`, and
//! minimal supporting files. Covers acceptance criteria C2 (fixed-order) and
//! C5 (AIC order selection) from ticket-036.
//!
//! ## Design constraints
//!
//! - Only public `cobre_sddp::` and `cobre_core::` APIs are used.
//! - Each test is fully self-contained with its own `TempDir`.
//! - Parquet files are written using the `arrow` + `parquet` crates (dev-deps).
//! - Stages span the full history period so every observation date falls within
//!   a stage's `[start_date, end_date)` range, which is required by
//!   `estimate_seasonal_stats`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Date32Array, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::NaiveDate;
use cobre_core::{
    Bus, DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_io::Config;
use cobre_sddp::estimation::estimate_from_history;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of seasons (months) in one annual cycle.
const N_SEASONS: usize = 12;

/// Number of simulated history years — enough for stable PAR(2) estimation.
const N_YEARS: usize = 15;

/// First year of the history period.
const START_YEAR: i32 = 2000;

/// Hydro plant ID used throughout these tests.
const HYDRO_ID: i32 = 1;

/// Bus ID for the hydro plant.
const BUS_ID: i32 = 10;

// ── Parquet helpers ───────────────────────────────────────────────────────────

fn inflow_history_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]))
}

/// Convert a `NaiveDate` to a Date32 integer (days since Unix epoch 1970-01-01).
fn date_to_date32(date: NaiveDate) -> i32 {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    i32::try_from((date - epoch).num_days()).expect("date in Date32 range")
}

/// Write a Parquet file containing synthetic inflow history.
///
/// Generates `N_YEARS * N_SEASONS` observations for one hydro plant, one
/// per calendar month, starting from January of `START_YEAR`. Each observation
/// is dated the 15th of the month so it falls comfortably inside the stage's
/// `[1st, 1st-of-next-month)` window. Values use a seasonal sine-wave pattern
/// to provide non-trivial autocorrelation structure for PAR(p) fitting.
fn write_inflow_history(path: &Path) {
    let schema = inflow_history_schema();

    let mut hydro_ids: Vec<i32> = Vec::new();
    let mut dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for year in 0..N_YEARS {
        for month in 0..N_SEASONS {
            let cal_year = START_YEAR + year as i32;
            let cal_month = (month as u32) + 1;
            // Use the 15th so the date is well inside [1st, 1st-of-next-month).
            let date = NaiveDate::from_ymd_opt(cal_year, cal_month, 15).unwrap();

            // Seasonal pattern: mean ~500, amplitude ~200, plus deterministic
            // perturbation so values differ across years.
            let phase = std::f64::consts::TAU * (month as f64) / (N_SEASONS as f64);
            let noise_seed = (year * N_SEASONS + month) as f64;
            let noise = (noise_seed * std::f64::consts::PI).sin() * 30.0;
            let value = 500.0 + 200.0 * phase.sin() + noise;

            hydro_ids.push(HYDRO_ID);
            dates.push(date_to_date32(date));
            values.push(value.max(1.0));
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Date32Array::from(dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

// ── Case directory helpers ────────────────────────────────────────────────────

/// Minimal `config.json` string with the requested estimation settings.
fn config_json(order_selection: &str, max_order: u32) -> String {
    format!(
        r#"{{
            "training": {{ "seed": 42 }},
            "simulation": {{ "enabled": false, "num_scenarios": 0, "io_channel_capacity": 16 }},
            "modeling": {{}},
            "policy": {{}},
            "exports": {{}},
            "output": {{}},
            "estimation": {{
                "order_selection": "{order_selection}",
                "max_order": {max_order},
                "min_observations_per_season": 2
            }}
        }}"#
    )
}

/// Create the minimal directory skeleton required by `validate_structure`.
///
/// Required stubs (content `{}` — only existence matters for the manifest):
/// - `config.json` (written with the given estimation settings)
/// - `penalties.json`, `stages.json`, `initial_conditions.json`
/// - `system/buses.json`, `system/lines.json`, `system/hydros.json`,
///   `system/thermals.json`
fn create_minimal_case_skeleton(case_dir: &Path, order_selection: &str, max_order: u32) {
    std::fs::create_dir_all(case_dir.join("system")).unwrap();
    std::fs::create_dir_all(case_dir.join("scenarios")).unwrap();

    std::fs::write(
        case_dir.join("config.json"),
        config_json(order_selection, max_order),
    )
    .unwrap();

    for name in &[
        "penalties.json",
        "stages.json",
        "initial_conditions.json",
        "system/buses.json",
        "system/lines.json",
        "system/hydros.json",
        "system/thermals.json",
    ] {
        std::fs::write(case_dir.join(name), b"{}").unwrap();
    }
}

// ── System builder helpers ────────────────────────────────────────────────────

/// Build a `System` with one hydro plant and `N_YEARS × N_SEASONS` monthly
/// study stages spanning the full history period [`START_YEAR`, `START_YEAR +
/// N_YEARS`).
///
/// Each stage covers one calendar month and carries `season_id = month_index`
/// (0 = January, …, 11 = December), so every observation in `inflow_history.parquet`
/// falls within a stage's `[start_date, end_date)` window. This mirrors what
/// `load_case` would produce when `stages.json` has per-stage `season_id` fields.
fn build_system_with_one_hydro() -> cobre_core::System {
    let bus = Bus {
        id: EntityId::from(BUS_ID),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId::from(HYDRO_ID),
        name: "H1".to_string(),
        bus_id: EntityId::from(BUS_ID),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 5000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 0.9,
        },
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        min_generation_mw: 0.0,
        max_generation_mw: 900.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 1000.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        },
    };

    // Generate N_YEARS * N_SEASONS monthly stages, each covering [1st, 1st-of-next-month).
    // season_id cycles 0..11 (January=0, December=11) regardless of the year.
    let mut stages: Vec<Stage> = Vec::with_capacity(N_YEARS * N_SEASONS);
    let mut stage_index: usize = 0;

    for year in 0..N_YEARS {
        let cal_year = START_YEAR + year as i32;
        for month in 0..N_SEASONS {
            let cal_month = (month as u32) + 1;

            let start_date = NaiveDate::from_ymd_opt(cal_year, cal_month, 1).unwrap();
            // End date wraps to January of the next year for December.
            let (end_year, end_month) = if cal_month == 12 {
                (cal_year + 1, 1u32)
            } else {
                (cal_year, cal_month + 1)
            };
            let end_date = NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap();

            stages.push(Stage {
                index: stage_index,
                id: stage_index as i32,
                start_date,
                end_date,
                season_id: Some(month), // season repeats every 12 months
                blocks: vec![Block {
                    index: 0,
                    name: format!("M{month:02}"),
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
            });
            stage_index += 1;
        }
    }

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .build()
        .expect("valid system")
}

/// Parse the `Config` from the `config.json` written in the case directory.
fn parse_config(case_dir: &Path) -> Config {
    let content = std::fs::read_to_string(case_dir.join("config.json")).unwrap();
    serde_json::from_str(&content).expect("valid config JSON")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// C2 — Fixed-order estimation: `estimate_from_history` with
/// `estimation.order_selection = "fixed"` and `estimation.max_order = 2`
/// produces one `InflowModel` per (hydro, stage) pair with `ar_order() == 2`
/// and finite, positive `mean_m3s` and `std_m3s`.
///
/// Setup:
/// - 1 hydro plant (ID 1), 12 seasons, 15 years × 12 months = 180 stages.
/// - `inflow_history.parquet`: 180 monthly observations (one per stage),
///   dated the 15th of each month so they fall within each stage's window.
/// - No `inflow_seasonal_stats.parquet` or `inflow_ar_coefficients.parquet`
///   → estimation path is triggered.
#[test]
fn test_estimate_from_history_fixed_order() {
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "fixed", 2);
    write_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"));

    let system = build_system_with_one_hydro();
    let config = parse_config(case_dir);

    let (updated, _report) = estimate_from_history(system, case_dir, &config)
        .expect("estimation should succeed with 15 years of monthly data");

    // One InflowModel per (hydro_id, stage_id) pair, but estimation groups by
    // season: 1 hydro × 12 seasons = 12 models.
    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        12,
        "expected 12 inflow models (1 hydro × 12 seasons), got {}",
        models.len()
    );

    for m in models {
        assert_eq!(m.hydro_id, EntityId::from(HYDRO_ID));

        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s should be non-negative and finite, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );

        // Fixed order = 2: every model must have exactly 2 AR coefficients.
        assert_eq!(
            m.ar_order(),
            2,
            "fixed-order=2 should produce ar_order()==2, got {} for stage {}",
            m.ar_order(),
            m.stage_id
        );
    }
}

/// C5 — AIC order selection: `estimate_from_history` with
/// `estimation.order_selection = "aic"` and `estimation.max_order = 3`
/// produces `InflowModel`s whose `ar_order()` is in the range `[0, max_order]`.
/// All models still have finite, positive `mean_m3s` and `std_m3s`.
///
/// The test does NOT assert that AIC always selects a lower order than
/// `max_order` because whether it does depends on the synthetic data's
/// autocorrelation structure. The invariant is that AIC order ≤ `max_order`.
#[test]
fn test_estimate_from_history_aic_order() {
    const MAX_ORDER: u32 = 3;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "aic", MAX_ORDER);
    write_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"));

    let system = build_system_with_one_hydro();
    let config = parse_config(case_dir);

    let (updated, _report) = estimate_from_history(system, case_dir, &config)
        .expect("AIC estimation should succeed with 15 years of monthly data");

    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        12,
        "expected 12 inflow models (1 hydro × 12 seasons), got {}",
        models.len()
    );

    for m in models {
        assert_eq!(m.hydro_id, EntityId::from(HYDRO_ID));

        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s should be non-negative and finite, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );

        // AIC-selected order must be in [0, max_order].
        assert!(
            m.ar_order() <= MAX_ORDER as usize,
            "AIC order {} exceeds max_order {} for stage {}",
            m.ar_order(),
            MAX_ORDER,
            m.stage_id
        );
    }
}

// ── PAR(1) round-trip tests ───────────────────────────────────────────────────

/// Number of observations per (hydro, season) for the round-trip tests.
const N_OBS_PER_SEASON: usize = 200;

/// Number of seasons used in the round-trip tests.
const N_SEASONS_RT: usize = 2;

/// True AR(1) coefficient for the round-trip tests.
const TRUE_PHI: f64 = 0.6;

/// True mean for the round-trip tests (m³/s).
const TRUE_MEAN: f64 = 300.0;

/// True standard deviation for the round-trip tests (m³/s).
const TRUE_STD: f64 = 60.0;

/// Statistical tolerance for AR(1) coefficient estimates with N=200 observations.
/// AR(1) coefficient SE ≈ 1/sqrt(N) ≈ 0.07; two standard errors ≈ 0.14 < 0.15.
const PHI_TOLERANCE: f64 = 0.15;

/// Relative tolerance for mean and std estimates (10% of true value).
const STAT_RELATIVE_TOLERANCE: f64 = 0.10;

/// Generate an interleaved PAR(1) time series for two seasons with identical
/// mean, std, and phi for both seasons.
///
/// The PAR(1) model used is the cross-season form matching `estimate_ar_coefficients`:
///
/// `(x_t,s - mu) / sigma = phi * (x_{t-1,s-1} - mu) / sigma + sqrt(1 - phi^2) * eps_t`
///
/// equivalently:
///
/// `x_t,s = mu + phi * (x_{t-1} - mu) + sigma * sqrt(1 - phi^2) * eps_t`
///
/// where `x_{t-1}` is the IMMEDIATELY preceding observation (from the previous season).
///
/// Returns a vector of `2 * n_per_season` values alternating between season 0 and
/// season 1: `[s0_0, s1_0, s0_1, s1_1, ..., s0_{n-1}, s1_{n-1}]`.
/// This interleaved ordering matches the chronological date ordering used in the
/// lag-lookup inside `estimate_ar_coefficients`.
fn generate_par1_interleaved(
    mean: f64,
    std: f64,
    phi: f64,
    n_per_season: usize,
    seed: u64,
) -> Vec<f64> {
    use rand::{SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1.0_f64).expect("valid normal distribution");

    let residual_std = std * (1.0 - phi * phi).sqrt();
    let total = 2 * n_per_season;
    let mut values = Vec::with_capacity(total);

    // Initialise with the stationary mean.
    let mut prev = mean;

    for _ in 0..total {
        let eps = normal.sample(&mut rng);
        // PAR(1): x_t = mu + phi * (x_{t-1} - mu) + sigma * sqrt(1-phi^2) * eps
        let x = mean + phi * (prev - mean) + residual_std * eps;
        let x_clamped = x.max(1.0);
        values.push(x_clamped);
        prev = x_clamped;
    }

    values
}

/// Write `inflow_history.parquet` for `n_hydros` hydros with 2 seasons.
///
/// Observations are placed in January (season 0) and February (season 1) of
/// successive years, dated the 15th so they fall within their stage windows.
///
/// The chronological ordering is: Jan2000, Feb2000, Jan2001, Feb2001, ...
/// which ensures the lag-1 lookup in `estimate_ar_coefficients` (which looks
/// `lag` positions back in the sorted-by-date observation vector) correctly
/// finds the cross-season predecessor needed for PAR(1) estimation.
fn write_par1_inflow_history(path: &Path, n_hydros: usize) {
    let schema = inflow_history_schema();

    let mut hydro_ids: Vec<i32> = Vec::new();
    let mut dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for h in 0..n_hydros {
        let hydro_id = (h + 1) as i32;

        // Generate 2 * N_OBS_PER_SEASON interleaved values for this hydro.
        // Use a different seed per hydro to make hydros statistically independent.
        let series =
            generate_par1_interleaved(TRUE_MEAN, TRUE_STD, TRUE_PHI, N_OBS_PER_SEASON, h as u64);

        // Assign dates: even-indexed observations → January, odd → February.
        // Jan2000, Feb2000, Jan2001, Feb2001, ...
        for (i, v) in series.into_iter().enumerate() {
            let year = 2000 + (i / 2) as i32;
            let month = if i % 2 == 0 { 1u32 } else { 2u32 };
            let date = chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap();
            hydro_ids.push(hydro_id);
            dates.push(date_to_date32(date));
            values.push(v);
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Date32Array::from(dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Build a `System` with `n_hydros` hydro plants and stages covering
/// `N_OBS_PER_SEASON` years, each year containing 2 monthly stages:
/// - Stage at January (`season_id` = 0)
/// - Stage at February (`season_id` = 1)
///
/// This structure ensures every observation in the PAR(1) history (dated the
/// 15th of January or February) falls within a stage's `[1st, 1st-of-next)` window.
fn build_system_for_par1(n_hydros: usize) -> cobre_core::System {
    use cobre_core::{
        Bus, DeficitSegment, EntityId, SystemBuilder,
        entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    let bus = Bus {
        id: EntityId::from(BUS_ID),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let hydros: Vec<Hydro> = (0..n_hydros)
        .map(|h| Hydro {
            id: EntityId::from((h + 1) as i32),
            name: format!("H{}", h + 1),
            bus_id: EntityId::from(BUS_ID),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 5000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 1000.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        })
        .collect();

    // Build N_OBS_PER_SEASON years × N_SEASONS_RT months per year of stages.
    // Year y has: January stage (season 0) then February stage (season 1).
    let mut stages: Vec<Stage> = Vec::with_capacity(N_OBS_PER_SEASON * N_SEASONS_RT);
    let mut stage_index: usize = 0;
    for year in 0..N_OBS_PER_SEASON {
        let cal_year = 2000 + year as i32;

        // January stage: season_id = 0
        stages.push(Stage {
            index: stage_index,
            id: stage_index as i32,
            start_date: chrono::NaiveDate::from_ymd_opt(cal_year, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(cal_year, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "JAN".to_string(),
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
        stage_index += 1;

        // February stage: season_id = 1
        stages.push(Stage {
            index: stage_index,
            id: stage_index as i32,
            start_date: chrono::NaiveDate::from_ymd_opt(cal_year, 2, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(cal_year, 3, 1).unwrap(),
            season_id: Some(1),
            blocks: vec![Block {
                index: 0,
                name: "FEB".to_string(),
                duration_hours: 672.0,
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
        stage_index += 1;
    }

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .build()
        .expect("valid system for PAR(1) round-trip")
}

/// C-037-4: PAR(1) round-trip accuracy with 1 hydro and 2 seasons.
///
/// Generates synthetic PAR(1) observations with `phi = 0.6` for 1 hydro and
/// 2 seasons (200 observations per season), runs the full estimation pipeline,
/// and verifies that the estimated AR(1) coefficient is within 0.15 of the
/// true value for each (hydro, season) pair.
///
/// Also verifies that `mean_m3s` and `std_m3s` are within 10% of the true
/// values for N=200.
#[test]
fn test_estimation_round_trip_par1() {
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "fixed", 1);
    write_par1_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"), 1);

    let system = build_system_for_par1(1);
    let config = parse_config(case_dir);

    let (updated, _report) = estimate_from_history(system, case_dir, &config)
        .expect("PAR(1) estimation with N=200 per season should succeed");

    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        N_SEASONS_RT,
        "expected {} inflow models (1 hydro × {} seasons), got {}",
        N_SEASONS_RT,
        N_SEASONS_RT,
        models.len()
    );

    for m in models {
        assert_eq!(
            m.hydro_id,
            cobre_core::EntityId::from(1),
            "hydro_id should be 1"
        );

        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s should be non-negative and finite, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );

        // AR(1) coefficient accuracy: within PHI_TOLERANCE of true value.
        assert_eq!(
            m.ar_order(),
            1,
            "fixed-order=1 should give ar_order()==1 for stage {}",
            m.stage_id
        );
        let phi_est = m.ar_coefficients[0];
        assert!(
            (phi_est - TRUE_PHI).abs() < PHI_TOLERANCE,
            "estimated phi ({phi_est:.4}) is not within {PHI_TOLERANCE} of true phi ({TRUE_PHI}) \
             for stage {}",
            m.stage_id
        );

        // Mean accuracy: within 10% of true value.
        let mean_err = (m.mean_m3s - TRUE_MEAN).abs() / TRUE_MEAN;
        assert!(
            mean_err < STAT_RELATIVE_TOLERANCE,
            "mean_m3s ({:.2}) deviates more than {:.0}% from true mean ({TRUE_MEAN}) \
             for stage {}",
            m.mean_m3s,
            STAT_RELATIVE_TOLERANCE * 100.0,
            m.stage_id
        );

        // Std accuracy: within 10% of true value.
        let std_err = (m.std_m3s - TRUE_STD).abs() / TRUE_STD;
        assert!(
            std_err < STAT_RELATIVE_TOLERANCE,
            "std_m3s ({:.2}) deviates more than {:.0}% from true std ({TRUE_STD}) \
             for stage {}",
            m.std_m3s,
            STAT_RELATIVE_TOLERANCE * 100.0,
            m.stage_id
        );
    }
}

/// C-037-5: PAR(1) round-trip with 2 hydros and 2 seasons.
///
/// Same as `test_estimation_round_trip_par1` but with 2 hydros. Verifies that
/// `system.inflow_models().len() == n_hydros * n_seasons` and that all
/// `n_hydros * n_seasons` models have `mean_m3s > 0.0` and `std_m3s > 0.0`.
/// Also verifies AR(1) coefficient accuracy for each pair.
#[test]
fn test_estimation_round_trip_two_hydros() {
    const N_HYDROS: usize = 2;
    let expected_models = N_HYDROS * N_SEASONS_RT;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "fixed", 1);
    write_par1_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"), N_HYDROS);

    let system = build_system_for_par1(N_HYDROS);
    let config = parse_config(case_dir);

    let (updated, _report) = estimate_from_history(system, case_dir, &config)
        .expect("PAR(1) estimation with 2 hydros should succeed");

    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        expected_models,
        "expected {} inflow models ({N_HYDROS} hydros × {N_SEASONS_RT} seasons), got {}",
        expected_models,
        models.len()
    );

    for m in models {
        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for hydro {} stage {}",
            m.mean_m3s,
            m.hydro_id.0,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s > 0.0,
            "std_m3s should be positive and finite, got {} for hydro {} stage {}",
            m.std_m3s,
            m.hydro_id.0,
            m.stage_id
        );

        // AR(1) coefficient accuracy.
        assert_eq!(
            m.ar_order(),
            1,
            "fixed-order=1 should give ar_order()==1 for hydro {} stage {}",
            m.hydro_id.0,
            m.stage_id
        );
        let phi_est = m.ar_coefficients[0];
        assert!(
            (phi_est - TRUE_PHI).abs() < PHI_TOLERANCE,
            "estimated phi ({phi_est:.4}) is not within {PHI_TOLERANCE} of true phi ({TRUE_PHI}) \
             for hydro {} stage {}",
            m.hydro_id.0,
            m.stage_id
        );
    }
}
