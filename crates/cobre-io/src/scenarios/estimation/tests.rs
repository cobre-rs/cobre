#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::useless_vec
)]

use super::*;
use cobre_core::scenario::{CorrelationModel, InflowModel};
use cobre_core::{EntityId, Hydro, HydroPenalties, SeasonMap, Stage, SystemBuilder};
use cobre_stochastic::PrecomputedPar;

fn minimal_system_with_inflow_models(models: Vec<InflowModel>) -> System {
    SystemBuilder::new()
        .inflow_models(models)
        .build()
        .expect("valid system")
}

#[test]
fn test_with_scenario_models_replaces_fields() {
    use cobre_core::{
        Bus, DeficitSegment,
        scenario::{CorrelationModel, InflowModel},
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let old_model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 10.0,
        std_m3s: 1.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .inflow_models(vec![old_model.clone(), {
            let mut m = old_model.clone();
            m.stage_id = 1;
            m
        }])
        .build()
        .expect("valid system");

    assert_eq!(system.inflow_models().len(), 2);
    assert_eq!(system.n_buses(), 1);

    let new_models: Vec<InflowModel> = (0..4)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i,
            mean_m3s: 50.0,
            std_m3s: 5.0,
            ar_coefficients: vec![0.4],
            residual_std_ratio: 0.9,
            annual: None,
        })
        .collect();
    let new_corr = CorrelationModel::default();

    let updated = system
        .with_scenario_models(new_models.clone(), new_corr.clone())
        .expect("new_models is sorted by (hydro_id, stage_id)");

    assert_eq!(updated.inflow_models().len(), 4, "expected 4 inflow models");
    assert_eq!(
        *updated.correlation(),
        new_corr,
        "correlation should equal new_corr"
    );

    assert_eq!(updated.n_buses(), 1, "buses must be preserved");
    assert!(
        updated.hydros().is_empty(),
        "hydros must be preserved (empty)"
    );
    assert!(
        updated.stages().is_empty(),
        "stages must be preserved (empty)"
    );
}

#[test]
fn test_with_scenario_models_clears_when_empty() {
    let model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 10.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let system = minimal_system_with_inflow_models(vec![model]);
    assert_eq!(system.inflow_models().len(), 1);

    let updated = system
        .with_scenario_models(vec![], CorrelationModel::default())
        .expect("empty inflow_models table is trivially sorted");
    assert!(updated.inflow_models().is_empty());
}

#[test]
fn test_estimate_explicit_stats_returns_unchanged() {
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);

    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();
    std::fs::write(scenarios.join("inflow_history.parquet"), b"").unwrap();
    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"").unwrap();
    std::fs::write(scenarios.join("inflow_ar_coefficients.parquet"), b"").unwrap();

    let model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 10.0,
        ar_coefficients: vec![0.5],
        residual_std_ratio: 0.87,
        annual: None,
    };
    let system = minimal_system_with_inflow_models(vec![model]);
    let original_len = system.inflow_models().len();

    let config = default_config();
    let (result, report, _path) = estimate_from_history(system, case_dir, &config).unwrap();

    assert_eq!(
        result.inflow_models().len(),
        original_len,
        "explicit stats: system must be unchanged"
    );
    assert!(
        report.is_none(),
        "explicit stats path must return None report"
    );
}

#[test]
fn test_estimate_no_history_returns_unchanged() {
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);

    let model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 10.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let system = minimal_system_with_inflow_models(vec![model]);
    let original_len = system.inflow_models().len();

    let config = default_config();
    let (result, report, _path) = estimate_from_history(system, case_dir, &config).unwrap();

    assert_eq!(
        result.inflow_models().len(),
        original_len,
        "no history: system must be unchanged"
    );
    assert!(report.is_none(), "no history path must return None report");
}

#[test]
fn test_estimation_path_resolve_all_8_combinations() {
    use crate::{FileManifest, InputFile};

    let make = |history: bool, stats: bool, ar: bool| {
        let mut manifest = FileManifest::default();
        if history {
            manifest.set_present(InputFile::ScenariosInflowHistoryParquet);
        }
        if stats {
            manifest.set_present(InputFile::ScenariosInflowSeasonalStatsParquet);
        }
        if ar {
            manifest.set_present(InputFile::ScenariosInflowArCoefficientsParquet);
        }
        manifest
    };

    assert_eq!(
        EstimationPath::resolve(&make(false, false, false)),
        EstimationPath::Deterministic,
    );
    assert_eq!(
        EstimationPath::resolve(&make(false, false, true)),
        EstimationPath::Deterministic,
    );
    assert_eq!(
        EstimationPath::resolve(&make(false, true, false)),
        EstimationPath::UserStatsWhiteNoise,
    );
    assert_eq!(
        EstimationPath::resolve(&make(false, true, true)),
        EstimationPath::UserProvidedNoHistory,
    );
    assert_eq!(
        EstimationPath::resolve(&make(true, false, false)),
        EstimationPath::FullEstimation,
    );
    assert_eq!(
        EstimationPath::resolve(&make(true, false, true)),
        EstimationPath::UserArHistoryStats,
    );
    assert_eq!(
        EstimationPath::resolve(&make(true, true, false)),
        EstimationPath::PartialEstimation,
    );
    assert_eq!(
        EstimationPath::resolve(&make(true, true, true)),
        EstimationPath::UserProvidedAll,
    );
}

#[test]
fn test_estimation_path_as_str_round_trip() {
    let variants = [
        EstimationPath::Deterministic,
        EstimationPath::UserStatsWhiteNoise,
        EstimationPath::UserProvidedNoHistory,
        EstimationPath::FullEstimation,
        EstimationPath::UserArHistoryStats,
        EstimationPath::PartialEstimation,
        EstimationPath::UserProvidedAll,
    ];

    let strings: Vec<&str> = variants.iter().map(|v| v.as_str()).collect();

    for s in &strings {
        assert!(!s.is_empty(), "as_str() returned empty string");
    }

    let unique: std::collections::HashSet<&&str> = strings.iter().collect();
    assert_eq!(
        unique.len(),
        variants.len(),
        "as_str() must return unique strings for each variant"
    );
}

#[test]
fn test_user_stats_to_rows_maps_all_models() {
    let models = vec![
        InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(1),
            stage_id: 1,
            mean_m3s: 120.0,
            std_m3s: 12.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 5.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];
    let system = minimal_system_with_inflow_models(models.clone());
    let rows = user_stats_to_rows(&system);

    assert_eq!(rows.len(), 3, "must produce one row per InflowModel");

    for (model, row) in models.iter().zip(rows.iter()) {
        assert_eq!(row.hydro_id, model.hydro_id, "hydro_id must be preserved");
        assert_eq!(row.stage_id, model.stage_id, "stage_id must be preserved");
        assert_eq!(
            row.mean_m3s.to_bits(),
            model.mean_m3s.to_bits(),
            "mean_m3s must be bitwise identical"
        );
        assert_eq!(
            row.std_m3s.to_bits(),
            model.std_m3s.to_bits(),
            "std_m3s must be bitwise identical"
        );
    }
}

#[test]
fn test_user_stats_to_rows_empty_system() {
    let system = minimal_system_with_inflow_models(vec![]);
    let rows = user_stats_to_rows(&system);
    assert!(rows.is_empty(), "empty system must produce empty rows");
}

/// Observation dates must fall within `make_two_season_stage` stages (map to seasons 0/1).
fn write_unit_test_inflow_history(path: &std::path::Path, hydro_id: i32, n_years: usize) {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let date_to_days = |d: NaiveDate| -> i32 {
        i32::try_from((d - epoch).num_days()).expect("date in Date32 range")
    };

    let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 99);

    let mut ids: Vec<i32> = Vec::with_capacity(n_years * 2);
    let mut start_dates: Vec<i32> = Vec::with_capacity(n_years * 2);
    let mut end_dates: Vec<i32> = Vec::with_capacity(n_years * 2);
    let mut values: Vec<f64> = Vec::with_capacity(n_years * 2);

    for y in 0..n_years {
        let year = (1970 + y) as i32;
        let start_s0 = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
        ids.push(hydro_id);
        start_dates.push(date_to_days(start_s0));
        end_dates.push(date_to_days(start_s0.succ_opt().unwrap()));
        // +300 shifts the ~0-mean series positive; inflows must be physically plausible.
        values.push(obs_s0[y] + 300.0);

        let start_s1 = NaiveDate::from_ymd_opt(year, 7, 15).unwrap();
        ids.push(hydro_id);
        start_dates.push(date_to_days(start_s1));
        end_dates.push(date_to_days(start_s1.succ_opt().unwrap()));
        values.push(obs_s1[y] + 300.0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Precondition for `PartialEstimation` path: user stats loaded, no AR coefficients.
#[allow(clippy::cast_possible_wrap)]
fn build_system_with_user_stats(n_years: usize) -> System {
    use cobre_core::scenario::InflowModel;
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

    let hydro_id = EntityId(1);
    let bus = Bus {
        id: EntityId(10),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    let inflow_models: Vec<InflowModel> = stages
        .iter()
        .map(|s| InflowModel {
            hydro_id,
            stage_id: s.id,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let hydro = make_hydro(hydro_id, EntityId(10));

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .build()
        .expect("valid system with user stats")
}

fn setup_partial_estimation_case(case_dir: &std::path::Path, n_years: usize) {
    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    write_unit_test_inflow_history(
        &scenarios.join("inflow_history.parquet"),
        1, // hydro_id
        n_years,
    );

    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
        .expect("write sentinel");
}

#[test]
fn test_partial_estimation_preserves_user_stats() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_partial_estimation_case(case_dir, N_YEARS);
    let system = build_system_with_user_stats(N_YEARS);
    let config = default_config();

    let (updated, report, path) =
        estimate_from_history(system, case_dir, &config).expect("partial estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::PartialEstimation,
        "expected PartialEstimation path"
    );
    assert!(
        report.is_some(),
        "PartialEstimation must return Some(report)"
    );

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "partial estimation must produce at least one inflow model"
    );

    for m in models {
        assert_eq!(
            m.mean_m3s.to_bits(),
            100.0_f64.to_bits(),
            "mean_m3s must be bitwise identical to user value 100.0 for stage {}",
            m.stage_id
        );
        assert_eq!(
            m.std_m3s.to_bits(),
            10.0_f64.to_bits(),
            "std_m3s must be bitwise identical to user value 10.0 for stage {}",
            m.stage_id
        );
        assert!(
            !m.ar_coefficients.is_empty(),
            "ar_coefficients must be non-empty for stage {} (estimated from history)",
            m.stage_id
        );
    }
}

/// `residual_std_ratio` must come from the periodic-ACF closure, not raw YW.
#[test]
fn test_partial_estimation_populates_closure_derived_ratio() {
    use cobre_stochastic::par::derive_residual_std_ratios;
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_partial_estimation_case(case_dir, N_YEARS);
    let system = build_system_with_user_stats(N_YEARS);
    let config = default_config();

    let (updated, _report, path) =
        estimate_from_history(system, case_dir, &config).expect("partial estimation must succeed");
    assert_eq!(
        path,
        EstimationPath::PartialEstimation,
        "expected PartialEstimation path"
    );

    let models = updated.inflow_models();
    let season0 = models
        .iter()
        .find(|m| m.stage_id == 0)
        .expect("stage 0 (season 0) model present");
    let season1 = models
        .iter()
        .find(|m| m.stage_id == 1)
        .expect("stage 1 (season 1) model present");

    let psi_by_season = vec![
        season0.ar_coefficients.clone(),
        season1.ar_coefficients.clone(),
    ];
    let orders = vec![season0.ar_order(), season1.ar_order()];
    let expected = derive_residual_std_ratios(&psi_by_season, &orders, 2)
        .expect("closure solves for the internally-fitted coefficients");

    for (m, expected_r) in [(season0, expected[0]), (season1, expected[1])] {
        assert!(
            (m.residual_std_ratio - expected_r).abs() < 1e-12,
            "stage {}: residual_std_ratio must equal the closure-derived value \
             {expected_r}, got {} — the raw YW-fitted value must no longer reach \
             the returned System",
            m.stage_id,
            m.residual_std_ratio
        );
    }
}

#[test]
fn test_partial_estimation_returns_report() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_partial_estimation_case(case_dir, N_YEARS);
    let system = build_system_with_user_stats(N_YEARS);
    let config = default_config();

    let (_updated, report, _path) =
        estimate_from_history(system, case_dir, &config).expect("partial estimation must succeed");

    let report = report.expect("PartialEstimation must return Some(EstimationReport)");

    assert_eq!(
        report.method, "PACF",
        "estimation method must be PACF, got '{}'",
        report.method
    );
    assert_eq!(
        report.entries.len(),
        1,
        "report must contain exactly 1 entry (one hydro), got {}",
        report.entries.len()
    );
    assert!(
        report.entries.contains_key(&EntityId(1)),
        "report must contain an entry for hydro_id=1"
    );
}

/// Partial-year system (seasons 8-11) with `SeasonMap` and user stats (exercises prestudy synthesis).
fn build_partial_year_system_with_user_stats() -> System {
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::InflowModel;
    use cobre_core::{Bus, DeficitSegment, EntityId, HorizonGraph, PolicyGraphType, SystemBuilder};

    let hydro_id = EntityId(1);
    let bus = Bus {
        id: EntityId(10),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let stages = partial_year_stages(8, 4, 2000);

    let inflow_models: Vec<InflowModel> = stages
        .iter()
        .map(|s| InflowModel {
            hydro_id,
            stage_id: s.id,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: hydro_id,
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 5000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 900.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 1000.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };
    hydro.declare_mirror_unit_group(EntityId(10));

    let policy_graph = HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions: vec![],
        nodes: Vec::new(),
        season_map: Some(monthly_season_map()),
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .policy_graph(policy_graph)
        .inflow_models(inflow_models)
        .build()
        .expect("valid partial-year system with user stats")
}

/// Prestudy models (negative stage_id) must be sorted canonically with user models.
#[test]
fn test_partial_estimation_partial_year_study_orders_canonically() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();
    write_full_month_history_one_hydro(
        &scenarios.join("inflow_history.parquet"),
        EntityId(1),
        N_YEARS,
    );
    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
        .expect("write sentinel");

    let system = build_partial_year_system_with_user_stats();
    let config = default_config();

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("partial estimation over a genuine partial-year study must succeed");

    assert_eq!(
        path,
        EstimationPath::PartialEstimation,
        "expected PartialEstimation path"
    );
    assert!(
        report.is_some(),
        "PartialEstimation must return Some(report)"
    );

    let models = updated.inflow_models();
    let stage_ids: Vec<i32> = models.iter().map(|m| m.stage_id).collect();
    assert!(
        stage_ids.iter().any(|&id| id < 0),
        "expected pre-study (negative stage_id) models from the synthesized lag \
         window, got {stage_ids:?}"
    );

    let mut sorted = models.to_vec();
    sorted.sort_by_key(|m| (m.hydro_id.0, m.stage_id));
    assert_eq!(
        models,
        sorted.as_slice(),
        "inflow_models must come back canonically ordered by (hydro_id, stage_id) \
         even though prestudy (negative id) rows are computed after user \
         (positive id) rows"
    );
}

fn default_config() -> Config {
    use crate::config::{EstimationConfig, OrderSelectionMethod};
    let mut cfg: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
    cfg.estimation = EstimationConfig {
        max_order: 2,
        order_selection: OrderSelectionMethod::Pacf,
        min_observations_per_season: 2,
        max_coefficient_magnitude: None,
    };
    cfg
}

const MINIMAL_CONFIG_JSON: &str = r#"{
        "training": { "tree_seed": 42 },
        "simulation": { "enabled": false, "io_channel_capacity": 16 },
        "modeling": {},
        "policy": {},
        "exports": {}
    }"#;

fn create_required_files(case_dir: &std::path::Path) {
    // validate_structure only checks existence; content doesn't matter here.
    let _ = std::fs::create_dir_all(case_dir.join("system"));
    let _ = std::fs::create_dir_all(case_dir.join("scenarios"));
    let write = |name: &str| {
        let _ = std::fs::write(case_dir.join(name), b"{}");
    };
    write("config.json");
    write("penalties.json");
    write("stages.json");
    write("initial_conditions.json");
    write("system/buses.json");
    write("system/lines.json");
    write("system/hydros.json");
    write("system/thermals.json");
}

#[test]
fn test_estimation_report_structure() {
    use cobre_stochastic::par::fitting::{ContributionReduction, build_estimation_report};

    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_seasons = 3_usize;

    let mut estimates = Vec::new();
    for &hydro_id in &[h1, h2] {
        for season_id in 0..n_seasons {
            estimates.push(ArCoefficientEstimate {
                hydro_id,
                season_id,
                coefficients: vec![0.5, 0.3],
                annual: None,
            });
        }
    }

    let contribution_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();
    let report = build_estimation_report(&estimates, n_seasons, &contribution_reductions, "PACF");

    assert_eq!(report.entries.len(), 2, "expected 2 hydro entries");

    for &hydro_id in &[h1, h2] {
        let entry = report.entries.get(&hydro_id).expect("entry must exist");
        assert_eq!(entry.selected_order, 2, "selected_order must be 2");
        assert_eq!(
            entry.coefficients.len(),
            n_seasons,
            "one coefficient vec per season"
        );
    }
}

#[test]
fn test_estimation_report_empty_for_pacf() {
    let observations: Vec<(EntityId, chrono::NaiveDate, f64)> = vec![];
    let seasonal_stats: Vec<SeasonalStats> = vec![];
    let stages: Vec<Stage> = vec![];
    let hydro_ids: Vec<EntityId> = vec![];
    let max_order = 2;

    let (_, report) = estimate_ar_coefficients_with_selection(
        &observations,
        &seasonal_stats,
        &stages,
        &hydro_ids,
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: None,
            season_map: None,
            use_annual_component: false,
        },
    )
    .unwrap();

    assert!(
        report.entries.is_empty(),
        "empty observations must produce empty EstimationReport"
    );
}

fn make_expansion_stage(index: usize, id: i32, season_id: Option<usize>) -> Stage {
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
            branching_factor: 10,
            noise_method: NoiseMethod::Saa,
        },
    }
}

#[test]
fn seasonal_stats_to_rows_includes_prestudy_stages() {
    let stages = vec![
        make_expansion_stage(0, -2, Some(1)),
        make_expansion_stage(1, -1, Some(2)),
        make_expansion_stage(2, 0, Some(0)),
        make_expansion_stage(3, 1, Some(1)),
        make_expansion_stage(4, 2, Some(2)),
    ];

    let h1 = EntityId(1);
    let stats = vec![
        SeasonalStats {
            entity_id: h1,
            stage_id: 0,
            mean: 100.0,
            std: 20.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 1,
            mean: 110.0,
            std: 22.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 2,
            mean: 120.0,
            std: 24.0,
        },
    ];

    let rows = seasonal_stats_to_rows(&stats, &stages);

    assert_eq!(rows.len(), 5, "expected 5 rows (3 study + 2 pre-study)");

    let prestudy_rows: Vec<_> = rows.iter().filter(|r| r.stage_id < 0).collect();
    assert_eq!(
        prestudy_rows.len(),
        2,
        "expected 2 pre-study rows, got {}",
        prestudy_rows.len()
    );

    // stage_id = -2 has season 1 -> (mean=110, std=22).
    let neg2 = rows.iter().find(|r| r.stage_id == -2).expect("row for -2");
    assert!((neg2.mean_m3s - 110.0).abs() < f64::EPSILON);
    assert!((neg2.std_m3s - 22.0).abs() < f64::EPSILON);

    // stage_id = -1 has season 2 -> (mean=120, std=24).
    let neg1 = rows.iter().find(|r| r.stage_id == -1).expect("row for -1");
    assert!((neg1.mean_m3s - 120.0).abs() < f64::EPSILON);
    assert!((neg1.std_m3s - 24.0).abs() < f64::EPSILON);

    for w in rows.windows(2) {
        assert!(
            (w[0].hydro_id.0, w[0].stage_id) <= (w[1].hydro_id.0, w[1].stage_id),
            "rows not sorted"
        );
    }
}

#[test]
fn ar_estimates_to_rows_includes_prestudy_stages() {
    let stages = vec![
        make_expansion_stage(0, -2, Some(1)),
        make_expansion_stage(1, -1, Some(2)),
        make_expansion_stage(2, 0, Some(0)),
        make_expansion_stage(3, 1, Some(1)),
        make_expansion_stage(4, 2, Some(2)),
    ];

    let h1 = EntityId(1);
    let ar_estimates = vec![
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 0,
            coefficients: vec![0.3],
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 1,
            coefficients: vec![0.4],
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 2,
            coefficients: vec![0.5],
            annual: None,
        },
    ];

    let rows = ar_estimates_to_rows(&ar_estimates, &stages);

    assert_eq!(rows.len(), 5, "expected 5 rows");

    let prestudy_rows: Vec<_> = rows.iter().filter(|r| r.stage_id < 0).collect();
    assert_eq!(prestudy_rows.len(), 2);

    // stage_id = -2 is season 1, coefficient = 0.4.
    let neg2 = rows.iter().find(|r| r.stage_id == -2).expect("row for -2");
    assert!((neg2.coefficient - 0.4).abs() < f64::EPSILON);

    // stage_id = -1 is season 2, coefficient = 0.5.
    let neg1 = rows.iter().find(|r| r.stage_id == -1).expect("row for -1");
    assert!((neg1.coefficient - 0.5).abs() < f64::EPSILON);
}

#[test]
fn full_estimation_produces_prestudy_inflow_models() {
    use crate::scenarios::assemble_inflow_models;

    let stages = vec![
        make_expansion_stage(0, -2, Some(1)),
        make_expansion_stage(1, -1, Some(2)),
        make_expansion_stage(2, 0, Some(0)),
        make_expansion_stage(3, 1, Some(1)),
        make_expansion_stage(4, 2, Some(2)),
    ];

    let h1 = EntityId(1);

    let stats = vec![
        SeasonalStats {
            entity_id: h1,
            stage_id: 0,
            mean: 100.0,
            std: 20.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 1,
            mean: 110.0,
            std: 22.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 2,
            mean: 120.0,
            std: 24.0,
        },
    ];
    let stats_rows = seasonal_stats_to_rows(&stats, &stages);

    let ar_ests = vec![
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 0,
            coefficients: vec![0.3],
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 1,
            coefficients: vec![0.4],
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 2,
            coefficients: vec![0.5],
            annual: None,
        },
    ];
    let coeff_rows = ar_estimates_to_rows(&ar_ests, &stages);

    let inflow_models =
        assemble_inflow_models(stats_rows, coeff_rows, vec![]).expect("assembly should succeed");

    assert!(
        inflow_models.iter().any(|m| m.stage_id < 0),
        "expected pre-study InflowModel entries (negative stage_id)"
    );

    let prestudy_neg2 = inflow_models
        .iter()
        .find(|m| m.stage_id == -2)
        .expect("InflowModel for stage -2");
    assert!((prestudy_neg2.mean_m3s - 110.0).abs() < f64::EPSILON);
    assert!((prestudy_neg2.std_m3s - 22.0).abs() < f64::EPSILON);
}

/// Simulate a 2-season PAR(2) process with a deterministic seeded LCG.
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

fn make_two_season_stage(
    index: usize,
    id: i32,
    season_id: usize,
    year: i32,
    first_half: bool,
) -> Stage {
    use chrono::NaiveDate;
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

/// Groups AR coefficients by season (2 hydros × 2 seasons = 4 estimates).
#[test]
#[allow(clippy::cast_sign_loss)]
fn test_ar_rows_to_estimates_groups_by_season() {
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let make_stage = |id: i32, season_id: usize| Stage {
        index: id as usize,
        id,
        start_date: chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
        end_date: chrono::NaiveDate::from_ymd_opt(1970, 7, 1).unwrap(),
        season_id: Some(season_id),
        blocks: vec![Block {
            index: 0,
            name: "T".to_string(),
            duration_hours: 1.0,
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
    };

    let stages = vec![make_stage(0, 0), make_stage(1, 0), make_stage(2, 1)];

    // AR(1) rows, sorted by (hydro_id, stage_id, lag).
    let rows = vec![
        InflowArCoefficientRow {
            hydro_id: EntityId(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.50,
        },
        InflowArCoefficientRow {
            hydro_id: EntityId(1),
            stage_id: 1,
            lag: 1,
            coefficient: 0.50,
        },
        InflowArCoefficientRow {
            hydro_id: EntityId(1),
            stage_id: 2,
            lag: 1,
            coefficient: 0.60,
        },
        InflowArCoefficientRow {
            hydro_id: EntityId(2),
            stage_id: 0,
            lag: 1,
            coefficient: 0.40,
        },
        InflowArCoefficientRow {
            hydro_id: EntityId(2),
            stage_id: 1,
            lag: 1,
            coefficient: 0.40,
        },
        InflowArCoefficientRow {
            hydro_id: EntityId(2),
            stage_id: 2,
            lag: 1,
            coefficient: 0.35,
        },
    ];

    let estimates = ar_rows_to_estimates(&rows, &stages);

    assert_eq!(
        estimates.len(),
        4,
        "expected 4 estimates (2 hydros * 2 seasons), got {}",
        estimates.len()
    );

    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(1) && e.season_id == 0)
        .expect("hydro 1, season 0 estimate must exist");
    assert_eq!(e.coefficients.len(), 1, "AR(1) must have 1 coefficient");
    assert!(
        (e.coefficients[0] - 0.50).abs() < f64::EPSILON,
        "coeff must be 0.50, got {}",
        e.coefficients[0]
    );
    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(1) && e.season_id == 1)
        .expect("hydro 1, season 1 estimate must exist");
    assert_eq!(e.coefficients.len(), 1);
    assert!((e.coefficients[0] - 0.60).abs() < f64::EPSILON);

    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(2) && e.season_id == 0)
        .expect("hydro 2, season 0 estimate must exist");
    assert_eq!(e.coefficients.len(), 1);
    assert!((e.coefficients[0] - 0.40).abs() < f64::EPSILON);

    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(2) && e.season_id == 1)
        .expect("hydro 2, season 1 estimate must exist");
    assert_eq!(e.coefficients.len(), 1);
    assert!((e.coefficients[0] - 0.35).abs() < f64::EPSILON);
}

/// Writes one AR(1) lag-1 row per stage.
fn write_unit_test_ar_coefficients(
    path: &std::path::Path,
    hydro_id: i32,
    stages: &[Stage],
    coefficient: f64,
    residual_std_ratio: f64,
) {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("lag", DataType::Int32, false),
        Field::new("coefficient", DataType::Float64, false),
        Field::new("residual_std_ratio", DataType::Float64, false),
    ]));

    let n = stages.len();
    let hydro_ids: Vec<i32> = vec![hydro_id; n];
    let stage_ids: Vec<i32> = stages.iter().map(|s| s.id).collect();
    let lags: Vec<i32> = vec![1; n];
    let coefficients: Vec<f64> = vec![coefficient; n];
    let ratios: Vec<f64> = vec![residual_std_ratio; n];

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(lags)),
            Arc::new(Float64Array::from(coefficients)),
            Arc::new(Float64Array::from(ratios)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// System with empty inflow_models (UserArHistoryStats precondition).
#[allow(clippy::cast_possible_wrap)]
fn build_system_empty_models(n_years: usize) -> System {
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

    let hydro_id = EntityId(1);
    let bus = Bus {
        id: EntityId(10),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    let hydro = make_hydro(hydro_id, EntityId(10));

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .build()
        .expect("valid system with empty inflow models")
}

/// Sets up UserArHistoryStats: history + AR coefficients, no seasonal stats.
#[allow(clippy::cast_possible_wrap)]
fn setup_user_ar_case(
    case_dir: &std::path::Path,
    n_years: usize,
    ar_coefficient: f64,
    residual_std_ratio: f64,
) {
    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    write_unit_test_inflow_history(&scenarios.join("inflow_history.parquet"), 1, n_years);

    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    write_unit_test_ar_coefficients(
        &scenarios.join("inflow_ar_coefficients.parquet"),
        1,
        &stages,
        ar_coefficient,
        residual_std_ratio,
    );
}

/// User AR coefficients preserved bitwise; residual_std_ratio superseded by closure.
#[test]
fn test_user_ar_estimation_preserves_ar_coefficients() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    const KNOWN_COEFF: f64 = 0.72;
    const KNOWN_RATIO: f64 = 0.69;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_user_ar_case(case_dir, N_YEARS, KNOWN_COEFF, KNOWN_RATIO);
    let system = build_system_empty_models(N_YEARS);
    let config = default_config();

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::UserArHistoryStats,
        "expected UserArHistoryStats path"
    );
    assert!(
        report.is_some(),
        "UserArHistoryStats must return Some(report)"
    );

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );

    let expected_ratio = (1.0 - KNOWN_COEFF * KNOWN_COEFF).sqrt();
    assert!(
        (expected_ratio - KNOWN_RATIO).abs() > 1e-3,
        "fixture sanity: KNOWN_RATIO must differ from the closure value to prove \
         the user-column value is superseded, got expected={expected_ratio} \
         known={KNOWN_RATIO}"
    );

    for m in models {
        assert_eq!(
            m.ar_coefficients.len(),
            1,
            "every model must have AR(1) coefficients (lag 1 only), stage {}",
            m.stage_id
        );
        assert_eq!(
            m.ar_coefficients[0].to_bits(),
            KNOWN_COEFF.to_bits(),
            "ar_coefficients[0] must be bitwise identical to {KNOWN_COEFF} for stage {}",
            m.stage_id
        );
        assert!(
            (m.residual_std_ratio - expected_ratio).abs() < 1e-12,
            "residual_std_ratio must equal the closure-derived value {expected_ratio} \
             (order-1 decouples exactly), got {} for stage {} — the raw user-column \
             value {KNOWN_RATIO} must no longer reach the returned System",
            m.residual_std_ratio,
            m.stage_id
        );
    }
}

#[test]
fn test_user_ar_estimation_estimates_stats_from_history() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_user_ar_case(case_dir, N_YEARS, 0.55, 0.83);
    let system = build_system_empty_models(N_YEARS);
    let config = default_config();

    let (updated, _report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );

    for m in models {
        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s must be finite and positive, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s must be finite and non-negative, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );
    }
}

#[test]
fn test_user_ar_estimation_returns_user_provided_report() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_user_ar_case(case_dir, N_YEARS, 0.55, 0.83);
    let system = build_system_empty_models(N_YEARS);
    let config = default_config();

    let (_updated, report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    let report = report.expect("UserArHistoryStats must return Some(EstimationReport)");

    assert_eq!(
        report.method, "user_provided",
        "report method must be 'user_provided', got '{}'",
        report.method
    );
    assert!(
        report.entries.is_empty(),
        "report entries must be empty (no AR was estimated), got {} entries",
        report.entries.len()
    );
}

fn make_hydro(hydro_id: EntityId, bus_id: EntityId) -> Hydro {
    use cobre_core::entities::hydro::HydroGenerationModel;
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: hydro_id,
        name: format!("H{}", hydro_id.0),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 5000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 900.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 1000.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };
    hydro.declare_mirror_unit_group(bus_id);
    hydro
}

/// Two-hydro system with selective stats (only stats_hydro_ids have inflow_models).
#[allow(clippy::cast_possible_wrap)]
fn build_two_hydro_system_selective_stats(
    n_years: usize,
    all_hydro_ids: &[EntityId],
    stats_hydro_ids: &[EntityId],
) -> System {
    use cobre_core::scenario::InflowModel;
    use cobre_core::{Bus, DeficitSegment, SystemBuilder};

    let bus_id = EntityId(10);
    let bus = Bus {
        id: bus_id,
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    let inflow_models: Vec<InflowModel> = stats_hydro_ids
        .iter()
        .flat_map(|&hid| {
            stages.iter().map(move |s| InflowModel {
                hydro_id: hid,
                stage_id: s.id,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let hydros: Vec<_> = all_hydro_ids
        .iter()
        .map(|&hid| make_hydro(hid, bus_id))
        .collect();

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .build()
        .expect("valid two-hydro system")
}

fn write_history_for_hydros(path: &std::path::Path, hydro_ids: &[i32], n_years: usize) {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let date_to_days = |d: NaiveDate| -> i32 {
        i32::try_from((d - epoch).num_days()).expect("date in Date32 range")
    };

    let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 99);

    let mut ids: Vec<i32> = Vec::new();
    let mut start_dates: Vec<i32> = Vec::new();
    let mut end_dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for &hid in hydro_ids {
        for y in 0..n_years {
            let year = (1970 + y) as i32;
            let start_s0 = NaiveDate::from_ymd_opt(year, 1, 15).unwrap();
            ids.push(hid);
            start_dates.push(date_to_days(start_s0));
            end_dates.push(date_to_days(start_s0.succ_opt().unwrap()));
            values.push(obs_s0[y] + 300.0);

            let start_s1 = NaiveDate::from_ymd_opt(year, 7, 15).unwrap();
            ids.push(hid);
            start_dates.push(date_to_days(start_s1));
            end_dates.push(date_to_days(start_s1.succ_opt().unwrap()));
            values.push(obs_s1[y] + 300.0);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Direction A: hydro 2 has history (so AR is estimated) but no user stats —
/// the coverage error must name the uncovered hydro 2.
#[test]
fn test_partial_estimation_direction_a_missing_stats() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1, 2], N_YEARS);

    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
        .expect("write sentinel");

    let system = build_two_hydro_system_selective_stats(
        N_YEARS,
        &[EntityId(1), EntityId(2)],
        &[EntityId(1)], // stats only for hydro 1
    );

    let config = default_config();
    let result = estimate_from_history(system, case_dir, &config);

    assert!(
        result.is_err(),
        "Direction A must return Err when hydro 2 has AR estimates but no user stats"
    );

    let err = result.unwrap_err();
    let description = err.to_string();
    assert!(
        description.contains('2'),
        "error description must contain the uncovered hydro ID '2', got: {description}"
    );
}

/// Direction B: hydro 2 has user stats but no history — it falls back to white
/// noise (empty `ar_coefficients`, listed in `white_noise_fallbacks`).
#[test]
fn test_partial_estimation_direction_b_white_noise_fallback() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1], N_YEARS);

    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
        .expect("write sentinel");

    let system = build_two_hydro_system_selective_stats(
        N_YEARS,
        &[EntityId(1), EntityId(2)],
        &[EntityId(1), EntityId(2)], // stats for both
    );

    let config = default_config();
    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("Direction B must succeed (not an error)");

    assert_eq!(
        path,
        EstimationPath::PartialEstimation,
        "expected PartialEstimation path"
    );

    let report = report.expect("PartialEstimation must return Some(EstimationReport)");

    assert_eq!(
        report.white_noise_fallbacks,
        vec![EntityId(2)],
        "white_noise_fallbacks must be [EntityId(2)], got {:?}",
        report.white_noise_fallbacks
    );

    let hydro2_models: Vec<_> = updated
        .inflow_models()
        .iter()
        .filter(|m| m.hydro_id == EntityId(2))
        .collect();
    assert!(
        !hydro2_models.is_empty(),
        "returned system must have inflow models for hydro 2"
    );
    for m in &hydro2_models {
        assert!(
            m.ar_coefficients.is_empty(),
            "hydro 2 must have empty ar_coefficients (white-noise fallback), stage {}",
            m.stage_id
        );
    }
}

#[test]
fn test_partial_estimation_exact_coverage_no_fallback() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_partial_estimation_case(case_dir, N_YEARS);
    let system = build_system_with_user_stats(N_YEARS);
    let config = default_config();

    let (_updated, report, _path) =
        estimate_from_history(system, case_dir, &config).expect("exact coverage must succeed");

    let report = report.expect("PartialEstimation must return Some(EstimationReport)");

    assert!(
        report.white_noise_fallbacks.is_empty(),
        "white_noise_fallbacks must be empty for exact coverage, got {:?}",
        report.white_noise_fallbacks
    );
}

#[test]
fn test_full_estimation_report_has_empty_fallbacks() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1], N_YEARS);

    let system = build_two_hydro_system_selective_stats(
        N_YEARS,
        &[EntityId(1)],
        &[], // no user stats
    );

    let config = default_config();
    let (_, report, path) =
        estimate_from_history(system, case_dir, &config).expect("FullEstimation must succeed");

    assert_eq!(
        path,
        EstimationPath::FullEstimation,
        "expected FullEstimation path"
    );

    let report = report.expect("FullEstimation must return Some(EstimationReport)");

    assert!(
        report.white_noise_fallbacks.is_empty(),
        "FullEstimation must never populate white_noise_fallbacks, got {:?}",
        report.white_noise_fallbacks
    );
}

/// Collects std ratio divergence warnings for given user/estimated stds.
fn collect_std_ratio_warnings(
    hydro_id: EntityId,
    user_stds: &[f64],
    est_stds: &[f64],
) -> Vec<StdRatioDivergence> {
    use cobre_core::scenario::InflowModel;

    assert_eq!(
        user_stds.len(),
        est_stds.len(),
        "user_stds and est_stds must have equal length"
    );
    let n = user_stds.len();

    let stages: Vec<Stage> = (0..n)
        .map(|i| {
            let year = 1970_i32;
            let first_half = i % 2 == 0;
            make_two_season_stage(i, i as i32, i, year, first_half)
        })
        .collect();

    let user_models: Vec<InflowModel> = (0..n)
        .map(|i| InflowModel {
            hydro_id,
            stage_id: i as i32,
            mean_m3s: 100.0,
            std_m3s: user_stds[i],
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let system = SystemBuilder::new()
        .inflow_models(user_models)
        .stages(stages.clone())
        .build()
        .expect("valid system");

    let fitting_stats: Vec<SeasonalStats> = (0..n)
        .map(|i| SeasonalStats {
            entity_id: hydro_id,
            stage_id: i as i32,
            mean: 100.0,
            std: est_stds[i],
        })
        .collect();

    check_std_ratio_divergence(&system, &fitting_stats, &stages)
}

#[test]
fn test_std_ratio_divergence_fires_when_ratios_diverge() {
    let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0], &[100.0, 100.0]);
    assert!(
        !warnings.is_empty(),
        "expected at least one StdRatioDivergence when ratio diverges by 5x"
    );
    let pair_0_1 = warnings.iter().find(|w| w.season_a == 0 && w.season_b == 1);
    assert!(
        pair_0_1.is_some(),
        "expected a warning for season pair 0→1, got {warnings:?}"
    );
    let w = pair_0_1.unwrap();
    assert_eq!(
        w.hydro_id,
        EntityId(1),
        "warning must record the correct hydro_id"
    );
    assert!(
        (w.divergence - 5.0).abs() < 1e-10,
        "divergence for pair 0→1 must be 5.0, got {}",
        w.divergence
    );
}

#[test]
fn test_std_ratio_divergence_not_fires_when_similar() {
    let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0], &[90.0, 18.0]);
    assert!(
        warnings.is_empty(),
        "expected no StdRatioDivergence when ratios are similar, got {warnings:?}"
    );
}

/// Near-zero std edge case must not panic.
#[test]
fn test_std_ratio_divergence_skips_near_zero_std() {
    let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 0.0], &[90.0, 18.0]);
    let _ = warnings; // the assertion is that no panic occurs; the result may be empty or one entry.
}

/// Wrap-around pair (last season → first) is checked.
#[test]
fn test_std_ratio_divergence_wraps_last_to_first() {
    let warnings =
        collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0, 50.0], &[100.0, 20.0, 10.0]);
    assert!(
        warnings.len() >= 2,
        "expected at least 2 StdRatioDivergence entries (including wrap), got {}",
        warnings.len()
    );
    let has_wrap = warnings.iter().any(|w| w.season_a == 2 && w.season_b == 0);
    assert!(
        has_wrap,
        "expected a warning for the wrap-around pair season 2 → season 0"
    );
}

use chrono::NaiveDate;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
};

/// 12-season monthly stages over `n_years` from year 2000; season_id cycles 0..12.
fn make_monthly_stages_for_annual(n_years: usize) -> Vec<Stage> {
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

/// Two-hydro 12-season monthly system (empty inflow_models for FullEstimation path).
#[allow(clippy::cast_possible_wrap)]
fn build_two_hydro_monthly_system(n_years: usize) -> System {
    use cobre_core::{Bus, DeficitSegment, SystemBuilder};
    let bus_id = EntityId(10);
    let bus = Bus {
        id: bus_id,
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let stages = make_monthly_stages_for_annual(n_years);
    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![make_hydro(h1, bus_id), make_hydro(h2, bus_id)])
        .stages(stages)
        .build()
        .expect("valid two-hydro monthly system")
}

fn write_monthly_inflow_history_two_hydros(path: &std::path::Path, n_years: usize) {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let date_to_days = |d: NaiveDate| -> i32 { i32::try_from((d - epoch).num_days()).unwrap() };

    let obs_h1 = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    let obs_h2 = synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5);

    let mut ids: Vec<i32> = Vec::new();
    let mut start_dates: Vec<i32> = Vec::new();
    let mut end_dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for &(eid, date, value) in obs_h1.iter().chain(obs_h2.iter()) {
        ids.push(eid.0);
        start_dates.push(date_to_days(date));
        end_dates.push(date_to_days(date.succ_opt().unwrap()));
        values.push(value);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

#[test]
fn estimate_from_history_pacf_annual_populates_annual_field() {
    use crate::config::{EstimationConfig, OrderSelectionMethod};
    use tempfile::TempDir;

    const N_YEARS: usize = 5;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();
    write_monthly_inflow_history_two_hydros(&scenarios.join("inflow_history.parquet"), N_YEARS);

    let system = build_two_hydro_monthly_system(N_YEARS);

    let mut config: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
    config.estimation = EstimationConfig {
        max_order: 2,
        order_selection: OrderSelectionMethod::PacfAnnual,
        min_observations_per_season: 2,
        max_coefficient_magnitude: None,
    };

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("full estimation with PacfAnnual must succeed");

    assert_eq!(
        path,
        EstimationPath::FullEstimation,
        "expected FullEstimation path"
    );
    assert!(report.is_some(), "FullEstimation must return Some(report)");

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );
    assert!(
        models.iter().any(|m| m.annual.is_some()),
        "PacfAnnual path must set annual=Some on at least one model"
    );
}

#[test]
fn estimate_from_history_pacf_classical_keeps_annual_none() {
    use crate::config::{EstimationConfig, OrderSelectionMethod};
    use tempfile::TempDir;

    const N_YEARS: usize = 5;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();
    write_monthly_inflow_history_two_hydros(&scenarios.join("inflow_history.parquet"), N_YEARS);

    let system = build_two_hydro_monthly_system(N_YEARS);

    let mut config: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
    config.estimation = EstimationConfig {
        max_order: 2,
        order_selection: OrderSelectionMethod::Pacf,
        min_observations_per_season: 2,
        max_coefficient_magnitude: None,
    };

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("full estimation with Pacf must succeed");

    assert_eq!(
        path,
        EstimationPath::FullEstimation,
        "expected FullEstimation path"
    );
    assert!(report.is_some(), "FullEstimation must return Some(report)");

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );
    assert!(
        models.iter().all(|m| m.annual.is_none()),
        "classical Pacf path must keep annual=None for every model"
    );
}

#[test]
fn estimate_ar_coefficients_with_selection_classical_path_unchanged() {
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_years = 30;
    let stages = make_monthly_stages_for_annual(n_years);

    let mut obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    obs.extend(synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5));

    let seasonal_stats = {
        use cobre_stochastic::par::fitting::estimate_seasonal_stats_with_season_map;
        estimate_seasonal_stats_with_season_map(&obs, &stages, &[h1, h2], None).unwrap()
    };

    let (estimates, report) = estimate_ar_coefficients_with_selection(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1, h2],
        &ArEstimationConfig {
            max_order: 3,
            max_coeff_magnitude: None,
            season_map: None,
            use_annual_component: false,
        },
    )
    .expect("classical path must succeed");

    assert_eq!(
        report.method, "PACF",
        "classical path must produce method=PACF, got {}",
        report.method
    );
    for est in &estimates {
        assert!(
            est.annual.is_none(),
            "classical path: hydro={} season={} must have annual=None",
            est.hydro_id.0,
            est.season_id
        );
    }
}

fn monthly_season_map() -> SeasonMap {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition};
    let seasons = (0..12usize)
        .map(|m| SeasonDefinition {
            id: m,
            label: format!("M{m}"),
            month_start: u32::try_from(m + 1).unwrap(),
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons,
    }
}

fn partial_year_stages(first_season: usize, n: usize, start_year: i32) -> Vec<Stage> {
    (0..n)
        .map(|k| {
            let season = first_season + k;
            let m = u32::try_from(season + 1).unwrap();
            let (ey, em) = if m == 12 {
                (start_year + 1, 1u32)
            } else {
                (start_year, m + 1)
            };
            Stage {
                index: k,
                id: i32::try_from(k).unwrap(),
                start_date: NaiveDate::from_ymd_opt(start_year, m, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(ey, em, 1).unwrap(),
                season_id: Some(season),
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
        })
        .collect()
}

/// Partial-year study (seasons 8-11) with full-cycle history must synthesize prestudy lag models.
#[test]
fn partial_year_par2_synthesizes_prestudy_lag_models() {
    let h1 = EntityId(1);
    let n_years = 30;
    let max_order = 2usize;
    let season_map = monthly_season_map();

    let study_stages = partial_year_stages(8, 4, 2030);

    // Full-cycle history (all 12 months) so the out-of-study lag seasons 6/7
    // have observations to fit.
    let obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);

    let prestudy = synthesize_prestudy_stages(&study_stages, max_order, Some(&season_map));
    assert_eq!(
        prestudy.len(),
        2,
        "expected 2 synthetic pre-study stages, got {}",
        prestudy.len()
    );
    let mut pre_ids: Vec<i32> = prestudy.iter().map(|s| s.id).collect();
    pre_ids.sort_unstable();
    assert_eq!(pre_ids, vec![-2, -1], "pre-study ids must be -1, -2");
    let season_of = |id: i32| prestudy.iter().find(|s| s.id == id).unwrap().season_id;
    assert_eq!(
        season_of(-1),
        Some(7),
        "stage -1 must map to season 7 (Aug)"
    );
    assert_eq!(
        season_of(-2),
        Some(6),
        "stage -2 must map to season 6 (Jul)"
    );

    let stages: Vec<Stage> = study_stages
        .iter()
        .cloned()
        .chain(prestudy.iter().cloned())
        .collect();

    let seasonal_stats = {
        use cobre_stochastic::par::fitting::estimate_seasonal_stats_with_season_map;
        estimate_seasonal_stats_with_season_map(&obs, &stages, &[h1], Some(&season_map))
            .expect("seasonal stats must fit without panic")
    };

    let (ar_estimates, _report) = estimate_ar_coefficients_with_selection(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1],
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: None,
            season_map: Some(&season_map),
            use_annual_component: false,
        },
    )
    .expect("AR(2) fit must succeed");

    let stats_rows = seasonal_stats_to_rows(&seasonal_stats, &stages);
    let coeff_rows = ar_estimates_to_rows(&ar_estimates, &stages);
    let annual_rows = ar_estimates_to_annual_rows(&ar_estimates, &stages);
    let inflow_models = assemble_inflow_models(stats_rows, coeff_rows, annual_rows)
        .expect("assembly must succeed with pre-study rows present");

    let neg_models: Vec<&InflowModel> = inflow_models.iter().filter(|m| m.stage_id < 0).collect();
    assert!(
        neg_models.iter().any(|m| m.stage_id == -1) && neg_models.iter().any(|m| m.stage_id == -2),
        "expected InflowModel entries at stage_id -1 and -2, got {:?}",
        neg_models.iter().map(|m| m.stage_id).collect::<Vec<_>>()
    );

    let par = PrecomputedPar::build(&inflow_models, &study_stages, &[h1], Some(&season_map))
        .expect("PrecomputedPar build must succeed");

    // A non-zero psi for the first study stage's pre-study lags proves the lag
    // stats came from history, not the (0,0) season fallback. PACF may select any
    // order ≤ max_order, so assert at least one lag is non-zero, not a fixed stride.
    let par_order = par.max_order();
    assert!(
        par_order >= 1 && par_order <= max_order,
        "selected order {par_order} must be in 1..={max_order}"
    );
    let psi0 = par.psi_slice(0, 0);
    assert_eq!(
        psi0.len(),
        par_order,
        "psi stride must equal selected order"
    );
    assert!(
        psi0.iter().any(|&p| p.abs() > 1e-9),
        "first study stage psi must be non-zero for pre-study lags, got {psi0:?}"
    );
}

/// Synthesized prestudy stage ids resolve via back-walk; declared prestudy via season_id.
#[test]
fn resolve_model_stage_seasons_recovers_synthesized_prestudy_gap_seasons() {
    let season_map = monthly_season_map();
    // Declared study stages: ids 0..3, seasons 8..11 (Sep-Dec).
    let mut stages = partial_year_stages(8, 4, 2030);
    // A declared pre-study stage (season 3) — resolved directly, not via synthesis.
    let mut declared_prestudy = partial_year_stages(3, 1, 2029);
    declared_prestudy[0].id = -5;
    stages.push(declared_prestudy.into_iter().next().unwrap());

    let model_stage_ids = vec![-1, -2, -5, 100];

    let (resolved, n_seasons) =
        resolve_model_stage_seasons(&stages, model_stage_ids.into_iter(), &season_map);

    assert_eq!(n_seasons, 12);
    assert_eq!(
        resolved.get(&-1),
        Some(&7),
        "synthesized stage -1 (1 lag before Sep) must resolve to season 7 (Aug)"
    );
    assert_eq!(
        resolved.get(&-2),
        Some(&6),
        "synthesized stage -2 (2 lags before Sep) must resolve to season 6 (Jul)"
    );
    assert_eq!(
        resolved.get(&-5),
        Some(&3),
        "declared pre-study stage -5 must resolve via its own season_id"
    );
    assert_eq!(
        resolved.get(&100),
        None,
        "an id not below the first study stage id is underivable and stays unmapped"
    );
}

fn dated_stage(id: i32, start_date: NaiveDate, end_date: NaiveDate, season: usize) -> Stage {
    cobre_core::test_support::make_stage(cobre_core::test_support::StageSpec {
        id,
        start_date,
        end_date,
        season_id: Some(season),
        ..Default::default()
    })
}

#[test]
fn lag_seasons_agree_across_the_calendar_walk_estimation_and_precompute_on_a_sparse_custom_map() {
    use cobre_core::test_support::{date, f64_bits_eq};
    use cobre_stochastic::season_cast::StitchedSeasonMap;
    use cobre_stochastic::test_support::{
        InflowModelSpec, make_inflow_model, sparse_ring_season_map,
    };

    let ring = sparse_ring_season_map();
    let stages = vec![
        dated_stage(0, date(2026, 1, 1), date(2026, 2, 1), 0),
        dated_stage(1, date(2026, 2, 1), date(2026, 3, 1), 1),
        dated_stage(2, date(2026, 3, 1), date(2026, 4, 1), 2),
        dated_stage(3, date(2026, 4, 1), date(2026, 7, 1), 12),
        dated_stage(4, date(2026, 7, 1), date(2026, 10, 1), 13),
    ];
    let lag_ids = [-1, -2, -3];

    let stitched = StitchedSeasonMap::build(&stages, &ring, 3);
    assert_eq!(
        lag_ids.map(|id| stitched.season_of(id)),
        [Some(13), Some(12), Some(2)]
    );

    let (resolved, _) = resolve_model_stage_seasons(&stages, lag_ids.into_iter(), &ring);
    assert_eq!(
        lag_ids.map(|id| resolved.get(&id).copied()),
        [Some(4), Some(3), Some(2)],
        "dense ordinals of raw seasons 13, 12, 2"
    );

    let models = vec![
        make_inflow_model(InflowModelSpec {
            hydro_id: 1,
            stage_id: 0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.5],
            ..Default::default()
        }),
        make_inflow_model(InflowModelSpec {
            hydro_id: 1,
            stage_id: 4,
            std_m3s: 40.0,
            ..Default::default()
        }),
    ];
    let par = PrecomputedPar::build(&models, &stages, &[EntityId(1)], Some(&ring))
        .expect("PrecomputedPar build must succeed");
    let psi = par.psi_slice(0, 0)[0];
    assert!(
        f64_bits_eq(psi, 0.25),
        "stage-0 lag-1 psi must be phi * s0 / s13 = 0.25, got {psi}"
    );
}

#[test]
fn pre_study_lags_fold_iso_week_53_on_a_weekly_map() {
    use cobre_core::test_support::date;
    use cobre_stochastic::test_support::weekly_season_map;

    let weekly = weekly_season_map();
    let stages = vec![
        dated_stage(0, date(2021, 1, 4), date(2021, 1, 11), 0),
        dated_stage(1, date(2021, 1, 11), date(2021, 1, 18), 1),
    ];
    let lag_ids = [-1, -2, -3];

    let (resolved, _) = resolve_model_stage_seasons(&stages, lag_ids.into_iter(), &weekly);
    assert_eq!(
        lag_ids.map(|id| resolved.get(&id).copied()),
        [Some(51), Some(51), Some(50)]
    );

    let prestudy = synthesize_prestudy_stages(&stages, 3, Some(&weekly));
    let synthesized: Vec<(i32, Option<usize>)> =
        prestudy.iter().map(|s| (s.id, s.season_id)).collect();
    assert_eq!(
        synthesized,
        vec![(-1, Some(51)), (-2, Some(51)), (-3, Some(50))]
    );
}

#[test]
fn every_synthesized_lag_stage_carries_statistics_when_two_lags_share_a_season() {
    use cobre_core::test_support::date;

    let prestudy = vec![
        dated_stage(-1, date(2020, 12, 4), date(2021, 1, 4), 51),
        dated_stage(-2, date(2020, 11, 4), date(2020, 12, 4), 51),
    ];
    let fitting_stats = vec![
        SeasonalStats {
            entity_id: EntityId(1),
            stage_id: -2,
            mean: 80.0,
            std: 12.0,
        },
        SeasonalStats {
            entity_id: EntityId(1),
            stage_id: 0,
            mean: 90.0,
            std: 15.0,
        },
    ];

    let rows = prestudy_seasonal_rows(&fitting_stats, &prestudy);

    let row = |stage_id| InflowSeasonalStatsRow {
        hydro_id: EntityId(1),
        stage_id,
        mean_m3s: 80.0,
        std_m3s: 12.0,
    };
    assert_eq!(rows, vec![row(-2), row(-1)]);
}

fn full_month_row(hydro_id: EntityId, year: i32, month: u32, value: f64) -> InflowHistoryRow {
    let start_date = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let end_date = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap()
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap()
    };
    InflowHistoryRow {
        hydro_id,
        start_date,
        end_date,
        value_m3s: value,
    }
}

#[test]
fn test_estimation_full_coverage_one_sample_per_occurrence() {
    let season_map = monthly_season_map();
    let template = partial_year_stages(0, 1, 2000).remove(0);
    let hydro_id = EntityId(1);

    let history: Vec<InflowHistoryRow> = (1..=12u32)
        .map(|month| full_month_row(hydro_id, 2000, month, 100.0 + f64::from(month)))
        .collect();

    let (observations, skipped_partial) =
        resolve_coverage_gated_observations(&history, Some(&season_map), Some(&template));

    assert_eq!(
        observations.len(),
        12,
        "12 full-coverage monthly occurrences must produce 12 samples, got {}",
        observations.len()
    );
    assert!(
        skipped_partial.is_empty(),
        "full-coverage-only data must not report any skipped-partial occurrence, got {skipped_partial:?}"
    );
}

/// Partial-coverage occurrence is skipped and counted in skipped_partial.
#[test]
fn test_estimation_partial_occurrence_skipped_and_counted() {
    let season_map = monthly_season_map();
    let template = partial_year_stages(0, 1, 2000).remove(0);
    let hydro_id = EntityId(1);

    // History is built in ascending start-date order per hydro, matching
    // `parse_inflow_history`'s output contract that `resolve_coverage_gated_observations`
    // relies on. April (30 days) has only its first 15 days present.
    let mut history: Vec<InflowHistoryRow> = Vec::new();
    for month in 1..=12u32 {
        if month == 4 {
            history.push(InflowHistoryRow {
                hydro_id,
                start_date: NaiveDate::from_ymd_opt(2000, 4, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2000, 4, 16).unwrap(),
                value_m3s: 999.0,
            });
        } else {
            history.push(full_month_row(
                hydro_id,
                2000,
                month,
                100.0 + f64::from(month),
            ));
        }
    }

    let (observations, skipped_partial) =
        resolve_coverage_gated_observations(&history, Some(&season_map), Some(&template));

    assert_eq!(
        observations.len(),
        11,
        "11 full occurrences must produce 11 samples (April's partial window excluded), got {}",
        observations.len()
    );
    assert_eq!(
        skipped_partial.get(&hydro_id).copied(),
        Some(1),
        "April's 15/30-day window must be counted as exactly 1 skipped partial \
         occurrence for hydro {hydro_id}, got {skipped_partial:?}"
    );
}

/// Partial occurrence must not affect fitted sigma.
#[test]
fn test_estimation_partial_does_not_affect_sigma() {
    let season_map = monthly_season_map();
    let template = partial_year_stages(0, 1, 2000).remove(0);
    let hydro_id = EntityId(1);

    let history = vec![
        full_month_row(hydro_id, 2000, 1, 100.0),
        full_month_row(hydro_id, 2001, 1, 200.0),
        InflowHistoryRow {
            hydro_id,
            start_date: NaiveDate::from_ymd_opt(2002, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2002, 1, 16).unwrap(),
            value_m3s: 1.0e6,
        },
    ];

    let (observations, skipped_partial) =
        resolve_coverage_gated_observations(&history, Some(&season_map), Some(&template));

    assert_eq!(skipped_partial.get(&hydro_id).copied(), Some(1));

    let seasonal_stats = estimate_seasonal_stats_with_season_map(
        &observations,
        std::slice::from_ref(&template),
        &[hydro_id],
        Some(&season_map),
    )
    .expect("seasonal stats must fit from the two full-coverage samples");

    assert_eq!(
        seasonal_stats.len(),
        1,
        "exactly one (hydro, season) group must be fitted"
    );
    let stats = &seasonal_stats[0];
    assert_eq!(stats.mean, 150.0, "mean must come only from [100.0, 200.0]");
    assert_eq!(
        stats.std, 50.0,
        "population std of [100.0, 200.0] is exactly 50.0 -- the excluded \
         partial occurrence's 1.0e6 value must not shift it"
    );
}

fn build_monthly_system_with_conditioning(
    n_years: usize,
    recent_observations: Vec<cobre_core::RecentObservation>,
) -> System {
    use cobre_core::{
        Bus, DeficitSegment, HorizonGraph, InitialConditions, PolicyGraphType, SystemBuilder,
    };

    let bus_id = EntityId(10);
    let bus = Bus {
        id: bus_id,
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };
    let h1 = EntityId(1);
    let stages = make_monthly_stages_for_annual(n_years);
    let policy_graph = HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions: vec![],
        nodes: Vec::new(),
        season_map: Some(monthly_season_map()),
    };
    let initial_conditions = InitialConditions {
        recent_observations,
        ..Default::default()
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![make_hydro(h1, bus_id)])
        .stages(stages)
        .policy_graph(policy_graph)
        .initial_conditions(initial_conditions)
        .build()
        .expect("valid monthly system with conditioning")
}

fn write_full_month_history_one_hydro(path: &std::path::Path, hydro_id: EntityId, n_years: usize) {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let date_to_days = |d: NaiveDate| -> i32 { i32::try_from((d - epoch).num_days()).unwrap() };

    let mut ids: Vec<i32> = Vec::new();
    let mut start_dates: Vec<i32> = Vec::new();
    let mut end_dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for year in 0..n_years {
        for month in 1..=12u32 {
            let y = 2000 + i32::try_from(year).unwrap();
            let start = NaiveDate::from_ymd_opt(y, month, 1).unwrap();
            let end = if month == 12 {
                NaiveDate::from_ymd_opt(y + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(y, month + 1, 1).unwrap()
            };
            let value = 100.0 + f64::from(month) * 5.0 + f64::from(u32::try_from(year).unwrap());
            ids.push(hydro_id.0);
            start_dates.push(date_to_days(start));
            end_dates.push(date_to_days(end));
            values.push(value);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Owner gate: recent_observations must not affect fitted statistics (grep-verified: estimation.rs never reads it).
#[test]
fn test_conditioning_window_does_not_change_fitted_statistics() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let h1 = EntityId(1);

    let dir_a = TempDir::new().unwrap();
    let case_dir_a = dir_a.path();
    create_required_files(case_dir_a);
    let scenarios_a = case_dir_a.join("scenarios");
    std::fs::create_dir_all(&scenarios_a).unwrap();
    write_full_month_history_one_hydro(&scenarios_a.join("inflow_history.parquet"), h1, N_YEARS);

    let dir_b = TempDir::new().unwrap();
    let case_dir_b = dir_b.path();
    create_required_files(case_dir_b);
    let scenarios_b = case_dir_b.join("scenarios");
    std::fs::create_dir_all(&scenarios_b).unwrap();
    write_full_month_history_one_hydro(&scenarios_b.join("inflow_history.parquet"), h1, N_YEARS);

    let system_without_conditioning = build_monthly_system_with_conditioning(N_YEARS, vec![]);

    // Wildly different from the record series: if it leaked into estimation,
    // it would obviously shift the fitted mean/std for whichever season it
    // resolves to.
    let conditioning = vec![cobre_core::RecentObservation {
        hydro_id: h1,
        start_date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2000, 2, 1).unwrap(),
        value_m3s: 1.0e6,
    }];
    let system_with_conditioning = build_monthly_system_with_conditioning(N_YEARS, conditioning);

    let config = default_config();

    let (updated_without, _report_a, path_a) =
        estimate_from_history(system_without_conditioning, case_dir_a, &config)
            .expect("estimation without conditioning must succeed");
    let (updated_with, _report_b, path_b) =
        estimate_from_history(system_with_conditioning, case_dir_b, &config)
            .expect("estimation with conditioning must succeed");

    assert_eq!(path_a, EstimationPath::FullEstimation);
    assert_eq!(path_b, EstimationPath::FullEstimation);

    let mut models_without = updated_without.inflow_models().to_vec();
    let mut models_with = updated_with.inflow_models().to_vec();
    models_without.sort_by_key(|m| (m.hydro_id.0, m.stage_id));
    models_with.sort_by_key(|m| (m.hydro_id.0, m.stage_id));

    assert_eq!(
        models_without.len(),
        models_with.len(),
        "conditioning must not change the number of fitted models"
    );
    for (a, b) in models_without.iter().zip(models_with.iter()) {
        assert_eq!(a.hydro_id, b.hydro_id);
        assert_eq!(a.stage_id, b.stage_id);
        assert_eq!(
            a.mean_m3s, b.mean_m3s,
            "fitted mean must be bit-identical regardless of a recent_observations \
             conditioning window"
        );
        assert_eq!(
            a.std_m3s, b.std_m3s,
            "fitted std must be bit-identical regardless of a recent_observations \
             conditioning window"
        );
        assert_eq!(
            a.ar_coefficients, b.ar_coefficients,
            "fitted AR coefficients must be bit-identical regardless of a \
             recent_observations conditioning window"
        );
    }
}

/// Cursor-based window selection must match filter-based reference across straddling occurrences.
#[test]
fn cursor_subslice_matches_filter_collect_across_straddling_occurrences() {
    let season_map = monthly_season_map();
    let template = partial_year_stages(0, 1, 2000).remove(0);
    let hydro_id = EntityId(1);

    let history = vec![
        InflowHistoryRow {
            hydro_id,
            start_date: NaiveDate::from_ymd_opt(2000, 1, 10).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2000, 2, 10).unwrap(),
            value_m3s: 10.0,
        },
        InflowHistoryRow {
            hydro_id,
            start_date: NaiveDate::from_ymd_opt(2000, 2, 10).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2000, 3, 10).unwrap(),
            value_m3s: 20.0,
        },
        InflowHistoryRow {
            hydro_id,
            start_date: NaiveDate::from_ymd_opt(2000, 3, 10).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2000, 4, 10).unwrap(),
            value_m3s: 30.0,
        },
    ];

    let (observations, skipped_partial) =
        resolve_coverage_gated_observations(&history, Some(&season_map), Some(&template));

    let windows: Vec<RealizedWindow> = history
        .iter()
        .map(|row| RealizedWindow {
            start_date: row.start_date,
            end_date: row.end_date,
            value_m3s: row.value_m3s,
        })
        .collect();
    let occurrences = discover_hydro_occurrences(&season_map, &template, &windows);
    assert!(
        occurrences.len() >= 4,
        "fixture must straddle at least 4 monthly occurrences, got {}",
        occurrences.len()
    );

    let mut reference_observations = Vec::new();
    let mut reference_skip_count = 0usize;
    let mut max_overlap = 0usize;
    for occurrence in &occurrences {
        let overlapping: Vec<RealizedWindow> = windows
            .iter()
            .filter(|w| w.start_date < occurrence.end && w.end_date > occurrence.start)
            .map(|w| RealizedWindow {
                start_date: w.start_date,
                end_date: w.end_date,
                value_m3s: w.value_m3s,
            })
            .collect();
        max_overlap = max_overlap.max(overlapping.len());

        let projection = cast(&overlapping, occurrence);
        if projection.coverage == 1.0 {
            reference_observations.push((hydro_id, occurrence.start, projection.value));
        } else if projection.coverage > 0.0 {
            reference_skip_count += 1;
        }
    }
    reference_observations.sort_by_key(|(id, date, _)| (id.0, *date));

    assert!(
        max_overlap >= 2,
        "fixture must exercise a straddling occurrence overlapped by \
         multiple windows, got a max of {max_overlap}"
    );
    assert_eq!(
        observations, reference_observations,
        "resolve_coverage_gated_observations's cursor-selected windows must produce \
         the same observations as an independent per-occurrence filter"
    );
    assert_eq!(
        skipped_partial.get(&hydro_id).copied().unwrap_or(0),
        reference_skip_count,
        "skipped-partial count must match the independent filter-based reference"
    );
}
