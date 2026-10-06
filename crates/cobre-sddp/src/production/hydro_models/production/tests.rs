#![allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

use std::collections::HashMap;

use chrono::NaiveDate;
use cobre_core::{
    Bus, EfficiencyModel, EntityId, HydraulicLossesModel, InflowHistoryRow, StudyPos,
    SystemBuilder, TailraceModel,
    entities::hydro::{HydroGenerationModel, HydroPenalties},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_io::extensions::{
    FittingWindow, FphaColumnLayout, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig,
    SeasonConfig, SelectionMode, StageRange, TailraceCurveRow, parse_fpha_hyperplanes,
};
use cobre_io::output::write_fpha_hyperplanes;

use super::*;

// ── Test helpers ──────────────────────────────────────────────────────────

fn make_stage(id: i32) -> Stage {
    Stage {
        index: usize::try_from(id.max(0)).unwrap_or(0),
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
        season_id: Some(0),
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
            branching_factor: 50,
            noise_method: NoiseMethod::Saa,
        },
    }
}

fn zero_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: 0.0,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
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
    }
}

fn make_hydro(id: i32, model: HydroGenerationModel) -> Hydro {
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId::from(id),
        name: format!("Hydro{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 100.0,
        max_storage_hm3: 2000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: model,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 500.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 1000.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_penalties(),
    };
    hydro.declare_mirror_unit_group(EntityId::from(10));
    hydro
}

fn valid_row(hydro_id: i32, stage_id: Option<i32>, plane_id: i32) -> FphaHyperplaneRow {
    FphaHyperplaneRow {
        hydro_id: EntityId::from(hydro_id),
        stage_id,
        plane_id,
        gamma_0: 1000.0,
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    }
}

fn precomputed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
    ProductionModelConfig {
        hydro_id: EntityId::from(hydro_id),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "fpha".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "precomputed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    }
}

fn computed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
    ProductionModelConfig {
        hydro_id: EntityId::from(hydro_id),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "fpha".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "computed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    }
}

// ── resolve_production_models unit tests (in-memory, no disk I/O) ─────────

#[test]
fn all_constant_no_config_returns_default_constant_provenance() {
    let hydro0 = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let hydro1 = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let src0 = determine_source(&hydro0, None).expect("should succeed");
    let src1 = determine_source(&hydro1, None).expect("should succeed");
    assert_eq!(src0, ProductionModelSource::DefaultConstant);
    assert_eq!(src1, ProductionModelSource::DefaultConstant);
}

#[test]
fn linearized_head_entity_resolves_to_constant_productivity() {
    let hydro = make_hydro(0, HydroGenerationModel::LinearizedHead);

    let src = determine_source(&hydro, None).expect("should succeed");
    assert_eq!(src, ProductionModelSource::DefaultConstant);
}

#[test]
fn fpha_entity_without_config_entry_returns_validation_error() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let err = determine_source(&hydro, None).expect_err("should fail");
    assert!(
        matches!(err, SddpError::Validation(ref msg) if
                msg.contains("fpha") || msg.contains("no entry") || msg.contains("hydro_production_models")),
        "expected Validation error mentioning missing config entry, got {err:?}"
    );
}

#[test]
fn computed_source_returns_computed_from_geometry() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let config = computed_fpha_config(0);

    let source = determine_source(&hydro, Some(&config)).expect("should succeed");
    assert_eq!(
        source,
        ProductionModelSource::ComputedFromGeometry,
        "expected ComputedFromGeometry, got {source:?}"
    );
}

fn make_computed_hydro(id: i32) -> Hydro {
    let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
    hydro.tailrace = Some(TailraceModel::Polynomial {
        coefficients: vec![300.0],
    });
    hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.02 });
    hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
    hydro
}

fn make_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
    vec![
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 100.0,
            height_m: 400.0,
            area_km2: 10.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 2000.0,
            height_m: 450.0,
            area_km2: 50.0,
        },
    ]
}

#[test]
fn computed_source_missing_tailrace_returns_validation_error() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let rows = make_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
    geometry_map.insert(EntityId::from(0), row_refs);

    let err = validate_computed_prerequisites(&hydro, &geometry_map)
        .expect_err("should fail when tailrace is None");
    let msg = err.to_string();
    assert!(
        msg.contains("tailrace"),
        "error must mention 'tailrace', got: {msg}"
    );
    assert!(
        msg.contains(&hydro.name),
        "error must include hydro name '{}', got: {msg}",
        hydro.name
    );
}

#[test]
fn computed_source_missing_geometry_returns_validation_error() {
    let hydro = make_computed_hydro(0);
    let empty_geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();

    let err = validate_computed_prerequisites(&hydro, &empty_geometry_map)
        .expect_err("should fail when geometry rows are absent");
    let msg = err.to_string();
    assert!(
        msg.contains("geometry"),
        "error must mention 'geometry', got: {msg}"
    );
    assert!(
        msg.contains(&hydro.name),
        "error must include hydro name '{}', got: {msg}",
        hydro.name
    );
}

#[test]
fn find_fpha_config_for_stage_returns_config_in_range() {
    let config = computed_fpha_config(0);
    let stage = make_stage(5);

    let result = find_fpha_config_for_stage(&config, &stage);
    assert!(
        result.is_some(),
        "expected Some(FphaColumnLayout) for stage 5, got None"
    );
    assert_eq!(
        result.expect("just checked is_some").source,
        "computed",
        "expected source 'computed'"
    );
}

#[test]
fn find_fpha_config_for_stage_returns_none_outside_range() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 5,
                end_stage_id: Some(10),
                model: "fpha".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "computed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };

    let stage = make_stage(0);
    let result = find_fpha_config_for_stage(&config, &stage);
    assert!(
        result.is_none(),
        "expected None for stage 0 (outside range [5,10]), got {result:?}"
    );
}

#[test]
fn gamma_0_is_scaled_by_kappa() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: None,
        plane_id: 0,
        gamma_0: 1000.0,
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 0.95,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };

    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        vec![&row as &FphaHyperplaneRow],
    );

    let model = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect("should build FPHA model");

    match model {
        ResolvedProductionModel::Fpha { planes, .. } => {
            assert_eq!(planes.len(), 1);
            let expected = 1000.0 * 0.95;
            assert!(
                (planes[0].intercept - expected).abs() < 1e-10,
                "intercept must be gamma_0 * kappa = {expected}, got {}",
                planes[0].intercept
            );
        }
        other => panic!("expected Fpha variant, got {other:?}"),
    }
}

#[test]
fn validation_rejects_gamma_v_negative() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.gamma_v = -0.1;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("gamma_v"),
        "error must mention gamma_v, got: {msg}"
    );
}

#[test]
fn validation_accepts_gamma_v_zero() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.gamma_v = 0.0;

    validate_hyperplane_row(&hydro, &stage, &row)
        .expect("gamma_v = 0.0 must be valid for constant-head plants");
}

#[test]
fn validation_rejects_gamma_s_positive() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.gamma_s = 0.01;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("gamma_s"),
        "error must mention gamma_s, got: {msg}"
    );
}

#[test]
fn validation_rejects_gamma_q_negative() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    for gamma_q in [-0.1, -1e-12] {
        let mut row = valid_row(0, None, 0);
        row.gamma_q = gamma_q;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("gamma_q must be >= 0"),
            "gamma_q = {gamma_q}: error must say gamma_q must be >= 0, got: {msg}"
        );
    }
}

#[test]
fn validation_accepts_gamma_q_zero() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    for (gamma_v, gamma_q) in [(-0.0, -0.0), (0.0, 0.0), (0.002, 0.0)] {
        let mut row = valid_row(0, None, 0);
        row.gamma_v = gamma_v;
        row.gamma_q = gamma_q;

        validate_hyperplane_row(&hydro, &stage, &row).unwrap_or_else(|e| {
            panic!("gamma_v = {gamma_v}, gamma_q = {gamma_q} must be valid, got: {e}")
        });
    }
}

#[test]
fn validation_rejects_kappa_zero() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.kappa = 0.0;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("kappa"),
        "error must mention kappa, got: {msg}"
    );
}

#[test]
fn validation_rejects_kappa_above_one() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.kappa = 1.5;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("kappa"),
        "error must mention kappa, got: {msg}"
    );
}

#[test]
fn stage_specific_hyperplanes_override_all_stage() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let global_row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: None,
        plane_id: 0,
        gamma_0: 500.0,
        gamma_v: 0.001,
        gamma_q: 0.80,
        gamma_s: -0.005,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };
    let stage_row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: Some(0),
        plane_id: 0,
        gamma_0: 900.0,
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };

    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        vec![&global_row as &FphaHyperplaneRow],
    );
    map.insert(
        (EntityId::from(0), Some(0i32)),
        vec![&stage_row as &FphaHyperplaneRow],
    );

    let model = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect("should succeed");

    match model {
        ResolvedProductionModel::Fpha { planes, .. } => {
            assert!(
                (planes[0].intercept - 900.0).abs() < 1e-10,
                "stage-specific intercept 900 should override global 500, got {}",
                planes[0].intercept
            );
        }
        other => panic!("expected Fpha variant, got {other:?}"),
    }
}

#[test]
fn all_stage_hyperplanes_used_when_no_stage_specific_rows() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(5);

    let global_row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: None,
        plane_id: 0,
        gamma_0: 700.0,
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };

    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        vec![&global_row as &FphaHyperplaneRow],
    );

    let model = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect("should succeed using global rows");

    match model {
        ResolvedProductionModel::Fpha { planes, .. } => {
            assert!(
                (planes[0].intercept - 700.0).abs() < 1e-10,
                "expected global intercept 700, got {}",
                planes[0].intercept
            );
        }
        other => panic!("expected Fpha, got {other:?}"),
    }
}

#[test]
fn zero_hyperplanes_for_stage_returns_validation_error() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        Vec::<&FphaHyperplaneRow>::new(),
    );

    let err = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect_err("should fail with zero hyperplanes");

    assert!(
        matches!(err, SddpError::Validation(_)),
        "expected Validation error, got {err:?}"
    );
}

#[test]
fn precomputed_stage_needs_a_flow_dependent_plane() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let rising = valid_row(0, None, 0);
    let cap = FphaHyperplaneRow {
        plane_id: 1,
        gamma_0: 999.668,
        gamma_v: -0.0,
        gamma_q: -0.0,
        gamma_s: -0.0045,
        ..valid_row(0, None, 1)
    };
    let build = |rows: Vec<&FphaHyperplaneRow>| {
        let map = HashMap::from([((EntityId::from(0), None::<i32>), rows)]);
        build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
    };

    for rows in [vec![&rising, &cap], vec![&cap, &rising]] {
        let Ok(ResolvedProductionModel::Fpha { planes }) = build(rows) else {
            panic!("a flat plane next to a rising plane must build");
        };
        assert_eq!(planes.len(), 2);
        let built_cap = planes
            .iter()
            .find(|p| p.gamma_q == 0.0)
            .expect("the cap plane is kept");
        assert_eq!(built_cap.gamma_q.to_bits(), (-0.0_f64).to_bits());
    }

    let err = build(vec![&cap]).expect_err("a stage with only flat planes must be refused");
    assert!(
        matches!(&err, SddpError::Validation(msg)
            if msg.contains("no hyperplane has gamma_q > 0") && msg.contains("stage 0")),
        "got: {err}"
    );
}

#[test]
fn find_model_for_stage_returns_correct_model_name_in_range() {
    let config = precomputed_fpha_config(0);
    let stage = make_stage(3);
    let result = find_model_for_stage(&config, &stage);
    assert_eq!(result.as_ref().map(|(name, _)| name.as_str()), Some("fpha"));
}

#[test]
fn find_model_for_stage_returns_none_when_before_range_start() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 5,
                end_stage_id: Some(10),
                model: "fpha".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    let stage = make_stage(3);
    let result = find_model_for_stage(&config, &stage);
    assert!(
        result.is_none(),
        "stage 3 is before range [5, 10], expected None"
    );
}

#[test]
fn find_model_for_stage_open_ended_range_covers_all_stages() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    for stage_id in [0, 5, 11, 100] {
        let stage = make_stage(stage_id);
        let result = find_model_for_stage(&config, &stage);
        assert_eq!(
            result.as_ref().map(|(name, _)| name.as_str()),
            Some("constant_productivity"),
            "open-ended range must cover stage {stage_id}"
        );
    }
}

// ── Productivity override tests ─────────────────────────────────────────

#[test]
fn resolve_stage_model_uses_productivity_override() {
    let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(0);
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.55),
            }],
        },
    };
    let empty_map = std::collections::HashMap::new();
    let model = super::resolve_stage_model(
        &hydro,
        &stage,
        Some(&config),
        ProductionModelSource::DefaultConstant,
        &empty_map,
        None,
        None,
    )
    .expect("should succeed");
    assert!(
        matches!(model, ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.55).abs() < f64::EPSILON),
        "expected ConstantProductivity 0.55 (override), got {model:?}"
    );
}

/// JSON config entry present but `productivity_mw_per_m3s` is `None`:
/// `resolve_stage_model` returns a sentinel `ConstantProductivity { 0.0 }` that
/// `build_energy_conversion_set` later overwrites from the parquet override.
/// A debug_assert (debug only) catches configs that escape load-time
/// `cobre_io::validation::productivity_resolution`; release returns the sentinel.
#[test]
fn test_resolve_stage_model_returns_sentinel_when_json_lacks_productivity() {
    let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(0);
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    let empty_map = std::collections::HashMap::new();

    #[cfg(debug_assertions)]
    {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::resolve_stage_model(
                &hydro,
                &stage,
                Some(&config),
                ProductionModelSource::DefaultConstant,
                &empty_map,
                None,
                None,
            )
        }));
        let panic_payload =
            result.expect_err("debug build must panic via debug_assert! when productivity is None");
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
            })
            .unwrap_or_default();
        assert!(
            msg.contains("Hydro0") && msg.contains("validation::productivity_resolution"),
            "panic message must name the hydro and the validator; got: {msg}"
        );
    }

    #[cfg(not(debug_assertions))]
    {
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
            None,
        )
        .expect("release build returns sentinel");
        assert!(
            matches!(
                model,
                ResolvedProductionModel::ConstantProductivity { productivity }
                if productivity == 0.0
            ),
            "release build must return ConstantProductivity {{ productivity: 0.0 }}; got {model:?}"
        );
    }
}

#[test]
fn test_resolve_stage_model_uses_parquet_override_when_json_omits_productivity() {
    let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(0);
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    let empty_map = std::collections::HashMap::new();
    let override_table =
        build_hydro_energy_productivity_override(&[cobre_io::HydroEnergyProductivityRow {
            hydro_id: EntityId::from(0),
            stage_id: Some(0),
            equivalent_productivity_mw_per_m3s: Some(0.42),
            reference_outflow_m3s: None,
            specific_productivity_mw_per_m3s_per_m: None,
        }])
        .expect("override builds");

    let model = super::resolve_stage_model(
        &hydro,
        &stage,
        Some(&config),
        ProductionModelSource::DefaultConstant,
        &empty_map,
        None,
        Some(&override_table),
    )
    .expect("resolve succeeds");
    assert!(
        matches!(
            model,
            ResolvedProductionModel::ConstantProductivity { productivity }
            if (productivity - 0.42).abs() < 1e-12
        ),
        "override value must reach the resolved model; got {model:?}"
    );
}

/// No JSON config entry at all for a non-FPHA hydro: `resolve_stage_model`
/// returns the same sentinel. A debug_assert (debug only) catches missing
/// entries that escape load-time `cobre_io::validation::productivity_resolution`;
/// release trusts the invariant and returns the sentinel.
#[test]
fn test_resolve_stage_model_returns_sentinel_when_no_config_entry() {
    let hydro = make_hydro(7, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(3);
    let empty_map = std::collections::HashMap::new();

    #[cfg(debug_assertions)]
    {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::resolve_stage_model(
                &hydro,
                &stage,
                None,
                ProductionModelSource::DefaultConstant,
                &empty_map,
                None,
                None,
            )
        }));
        let panic_payload = result
            .expect_err("debug build must panic via debug_assert! when no config entry exists");
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
            })
            .unwrap_or_default();
        assert!(
            msg.contains("Hydro7") && msg.contains("validation::productivity_resolution"),
            "panic message must name the hydro and the validator; got: {msg}"
        );
    }

    #[cfg(not(debug_assertions))]
    {
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            None,
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
            None,
        )
        .expect("release build returns sentinel");
        assert!(
            matches!(
                model,
                ResolvedProductionModel::ConstantProductivity { productivity }
                if productivity == 0.0
            ),
            "release build must return ConstantProductivity {{ productivity: 0.0 }}; got {model:?}"
        );
    }
}

#[test]
fn find_model_for_stage_returns_override_in_tuple() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.75),
            }],
        },
    };
    let stage = make_stage(0);
    let result = find_model_for_stage(&config, &stage);
    assert_eq!(
        result,
        Some(("constant_productivity".to_string(), Some(0.75)))
    );
}

#[test]
fn find_model_for_stage_seasonal_with_override() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "constant_productivity".to_string(),
            seasons: vec![SeasonConfig {
                season_id: 1,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.60),
            }],
        },
    };
    let mut stage_match = make_stage(0);
    stage_match.season_id = Some(1);
    let result = find_model_for_stage(&config, &stage_match);
    assert_eq!(
        result,
        Some(("constant_productivity".to_string(), Some(0.60)))
    );

    let mut stage_default = make_stage(0);
    stage_default.season_id = Some(99);
    let result = find_model_for_stage(&config, &stage_default);
    assert_eq!(result, Some(("constant_productivity".to_string(), None)));
}

// ── Canonical resolver (resolve_stage / selection_entries) ────────────────

#[test]
fn resolve_stage_seasonal_season_miss_uses_default_model() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "constant_productivity".to_string(),
            seasons: vec![SeasonConfig {
                season_id: 1,
                model: "fpha".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "precomputed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    let mut stage = make_stage(0);
    stage.season_id = Some(99);

    let resolution = resolve_stage(&config, &stage);

    assert_eq!(resolution.model, Some("constant_productivity"));
    assert!(resolution.fpha_config.is_none());
    assert!(resolution.reference_volume.is_none());
    assert!(resolution.productivity.is_none());
}

#[test]
fn resolve_stage_seasonal_matching_season_passes_through() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "constant_productivity".to_string(),
            seasons: vec![SeasonConfig {
                season_id: 2,
                model: "linearized_head".to_string(),
                fpha_config: None,
                reference_volume: Some(ReferenceVolume::AbsoluteHm3(1200.0)),
                productivity_mw_per_m3s: Some(0.77),
            }],
        },
    };
    let mut stage = make_stage(3);
    stage.season_id = Some(2);

    let resolution = resolve_stage(&config, &stage);

    assert_eq!(resolution.model, Some("linearized_head"));
    assert!(resolution.fpha_config.is_none());
    assert_eq!(
        resolution.reference_volume,
        Some(&ReferenceVolume::AbsoluteHm3(1200.0))
    );
    assert_eq!(resolution.productivity, Some(0.77));
}

/// `selection_entries` emits a synthetic `default_model` entry, so
/// `default_model == "fpha"` with no `fpha_config` (precomputed by parquet)
/// classifies as precomputed FPHA even when every listed season is non-FPHA.
#[test]
fn selection_entries_classifies_default_fpha_as_precomputed() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "fpha".to_string(),
            seasons: vec![SeasonConfig {
                season_id: 1,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.5),
            }],
        },
    };

    let uses_precomputed = selection_entries(&config).any(|entry| {
        entry.fpha_config.is_some_and(|f| f.source == "precomputed")
            || (entry.model == "fpha" && entry.fpha_config.is_none())
    });

    assert!(
        uses_precomputed,
        "default_model == \"fpha\" with no fpha_config must classify as using precomputed FPHA"
    );
}

/// A `Seasonal` config with `default_model == "fpha"` and every listed season
/// non-FPHA classifies as precomputed FPHA on both `determine_source` and
/// `config_uses_precomputed_fpha`: `default_model` participates in source
/// classification, not only in per-stage resolution. Guards the
/// validate-then-blow-up path: the wrong-but-compiling alternative has
/// `determine_source` ignore `default_model` and classify `DefaultConstant`,
/// so the `fpha_hyperplanes.parquet` load gate (keyed on
/// `config_uses_precomputed_fpha`) skips loading it — yet `find_model_for_stage`
/// already resolves `"fpha"` on the season miss, so `resolve_stage_model`
/// reaches `build_fpha_model` against an empty hyperplane map and errors "no
/// hyperplane rows" for a config with no real coverage gap.
#[test]
fn seasonal_default_fpha_classifies_as_precomputed_source() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "fpha".to_string(),
            seasons: vec![SeasonConfig {
                season_id: 1,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.5),
            }],
        },
    };

    let source = determine_source(&hydro, Some(&config)).expect("should succeed");
    assert_eq!(source, ProductionModelSource::PrecomputedHyperplanes);
    assert!(config_uses_precomputed_fpha(&config));
}

#[test]
fn resolve_stage_stage_ranges_covering_and_uncovered() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: Some(10),
                model: "constant_productivity".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "precomputed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: Some(ReferenceVolume::Percentile(0.5)),
                productivity_mw_per_m3s: Some(0.42),
            }],
        },
    };

    let covered = make_stage(5);
    let resolution = resolve_stage(&config, &covered);
    assert_eq!(resolution.model, Some("constant_productivity"));
    assert_eq!(
        resolution.fpha_config.map(|f| f.source.as_str()),
        Some("precomputed")
    );
    assert_eq!(
        resolution.reference_volume,
        Some(&ReferenceVolume::Percentile(0.5))
    );
    assert_eq!(resolution.productivity, Some(0.42));

    let uncovered = make_stage(20);
    let resolution = resolve_stage(&config, &uncovered);
    assert!(resolution.model.is_none());
    assert!(resolution.fpha_config.is_none());
    assert!(resolution.reference_volume.is_none());
    assert!(resolution.productivity.is_none());
}

#[test]
fn precomputed_config_returns_precomputed_source() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let config = precomputed_fpha_config(0);
    let src = determine_source(&hydro, Some(&config)).expect("should succeed");
    assert_eq!(src, ProductionModelSource::PrecomputedHyperplanes);
}

// ── Computed-source integration tests ─────────────────────────────────────

fn make_sobradinho_computed_hydro(id: i32) -> Hydro {
    let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
    hydro.name = format!("Sobradinho{id}");
    hydro.min_storage_hm3 = 100.0;
    hydro.max_storage_hm3 = 20_000.0;
    hydro.max_turbined_m3s = 500.0;
    hydro.tailrace = Some(TailraceModel::Polynomial {
        coefficients: vec![0.0, 0.001_f64],
    });
    hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 2.0 });
    hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
    hydro
}

fn make_sobradinho_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
    vec![
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 100.0,
            height_m: 386.5,
            area_km2: 500.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 5_000.0,
            height_m: 392.0,
            area_km2: 800.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 12_000.0,
            height_m: 396.5,
            area_km2: 1_100.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 20_000.0,
            height_m: 400.0,
            area_km2: 1_400.0,
        },
    ]
}

#[test]
fn computed_source_end_to_end_produces_valid_fpha_planes() {
    let hydro = make_sobradinho_computed_hydro(0);
    let config = computed_fpha_config(0);
    let stage = make_stage(0);

    let geo_rows = make_sobradinho_geometry_rows(0);
    let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), geo_refs);

    let layout =
        find_fpha_config_for_stage(&config, &stage).expect("computed config covers stage 0");
    let fit_result = super::fit_planes_for_hydro(
        &hydro,
        layout,
        &geometry_map,
        0.0,
        super::TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        0,
        false,
    )
    .expect("fit_planes_for_hydro must succeed for valid Sobradinho-style input");
    let planes = &fit_result.planes;

    assert!(
        !planes.is_empty(),
        "expected at least one plane, got {}",
        planes.len()
    );

    // Coefficient signs must satisfy physical constraints: non-negative volume
    // and turbine gradients, non-positive lateral secant. The
    // installed-capacity ceiling contributes a flat `generation ≤ capacity` plane with
    // both gradients exactly zero, so the bound is `>= 0`, not strictly `> 0`.
    for (idx, plane) in planes.iter().enumerate() {
        assert!(
            plane.gamma_v >= 0.0,
            "plane {idx}: gamma_v={} must be >= 0",
            plane.gamma_v
        );
        assert!(
            plane.gamma_q >= 0.0,
            "plane {idx}: gamma_q={} must be >= 0",
            plane.gamma_q
        );
        assert!(
            plane.gamma_s <= 0.0,
            "plane {idx}: gamma_s={} must be <= 0",
            plane.gamma_s
        );
    }

    let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    let model = super::resolve_stage_model(
        &hydro,
        &stage,
        Some(&config),
        ProductionModelSource::ComputedFromGeometry,
        &empty_hyperplane_map,
        Some(planes),
        None,
    )
    .expect("resolve_stage_model must succeed for ComputedFromGeometry with cached planes");

    match model {
        ResolvedProductionModel::Fpha {
            planes: out_planes, ..
        } => {
            assert_eq!(
                out_planes.len(),
                planes.len(),
                "stage model must have the same plane count as the fitted planes"
            );
        }
        other => panic!("expected Fpha variant, got {other:?}"),
    }
}

/// Computed-FPHA export rows round-trip through `parse_fpha_hyperplanes`: the
/// computed path writes α-scaled coefficients with the IO `kappa` column pinned
/// to 1.0, so the precomputed reader's `intercept = gamma_0 * kappa`
/// reconstruction reproduces the fitted planes without re-scaling the
/// already-corrected coefficients.
#[test]
fn computed_export_rows_round_trip_through_parquet() {
    let hydro = make_sobradinho_computed_hydro(0);
    let config = computed_fpha_config(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    let stages = [make_stage(0)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-stage fit must succeed");

    let fitted_planes = &per_stage[0];
    assert!(!export_rows.is_empty(), "export rows must be non-empty");
    assert_eq!(
        export_rows.len(),
        fitted_planes.len(),
        "one export row per fitted plane"
    );
    for row in &export_rows {
        assert_eq!(row.kappa, 1.0, "computed rows pin kappa = 1.0");
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fpha_hyperplanes.parquet");
    write_fpha_hyperplanes(&path, &export_rows).expect("write_fpha_hyperplanes must succeed");

    let read_rows = parse_fpha_hyperplanes(&path).expect("re-read must succeed");
    assert_eq!(
        read_rows.len(),
        fitted_planes.len(),
        "re-read row count must match fitted plane count"
    );

    for (row, fitted) in read_rows.iter().zip(fitted_planes) {
        let reconstructed = FphaPlane {
            intercept: row.gamma_0 * row.kappa,
            gamma_v: row.gamma_v,
            gamma_q: row.gamma_q,
            gamma_s: row.gamma_s,
        };
        assert!(
            (reconstructed.intercept - fitted.intercept).abs() < 1e-9,
            "intercept mismatch: {} vs {}",
            reconstructed.intercept,
            fitted.intercept
        );
        assert!((reconstructed.gamma_v - fitted.gamma_v).abs() < 1e-9);
        assert!((reconstructed.gamma_q - fitted.gamma_q).abs() < 1e-9);
        assert!((reconstructed.gamma_s - fitted.gamma_s).abs() < 1e-9);
    }
}

#[test]
fn mixed_precomputed_and_computed_sources_resolve_correctly() {
    let hydro0 = make_hydro(0, HydroGenerationModel::Fpha);
    let config0 = precomputed_fpha_config(0);

    let precomp_row_a = valid_row(0, None, 0);
    let precomp_row_b = valid_row(0, None, 1);
    let precomp_row_c = valid_row(0, None, 2);
    let mut hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    hyperplane_map.insert(
        (EntityId::from(0), None),
        vec![&precomp_row_a, &precomp_row_b, &precomp_row_c],
    );

    let hydro1 = make_sobradinho_computed_hydro(1);
    let config1 = computed_fpha_config(1);

    let geo_rows = make_sobradinho_geometry_rows(1);
    let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(1), geo_refs);

    let stage = make_stage(0);

    let src0 = determine_source(&hydro0, Some(&config0)).expect("hydro0 source");
    let src1 = determine_source(&hydro1, Some(&config1)).expect("hydro1 source");
    assert_eq!(
        src0,
        ProductionModelSource::PrecomputedHyperplanes,
        "hydro 0 must be PrecomputedHyperplanes"
    );
    assert_eq!(
        src1,
        ProductionModelSource::ComputedFromGeometry,
        "hydro 1 must be ComputedFromGeometry"
    );

    let layout1 =
        find_fpha_config_for_stage(&config1, &stage).expect("computed config covers stage 0");
    let computed_fit = super::fit_planes_for_hydro(
        &hydro1,
        layout1,
        &geometry_map,
        0.0,
        super::TailraceSource::Entity(hydro1.tailrace.clone()),
        None,
        0,
        false,
    )
    .expect("fit_planes_for_hydro must succeed for hydro 1");

    let model0 = super::resolve_stage_model(
        &hydro0,
        &stage,
        Some(&config0),
        src0,
        &hyperplane_map,
        None,
        None,
    )
    .expect("resolve_stage_model must succeed for hydro 0 (precomputed)");

    let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    let model1 = super::resolve_stage_model(
        &hydro1,
        &stage,
        Some(&config1),
        src1,
        &empty_hyperplane_map,
        Some(&computed_fit.planes),
        None,
    )
    .expect("resolve_stage_model must succeed for hydro 1 (computed)");

    assert!(
        matches!(model0, ResolvedProductionModel::Fpha { .. }),
        "hydro 0 must resolve to Fpha, got {model0:?}"
    );
    assert!(
        matches!(model1, ResolvedProductionModel::Fpha { .. }),
        "hydro 1 must resolve to Fpha, got {model1:?}"
    );

    assert_eq!(
        src0,
        ProductionModelSource::PrecomputedHyperplanes,
        "provenance[0] must be PrecomputedHyperplanes"
    );
    assert_eq!(
        src1,
        ProductionModelSource::ComputedFromGeometry,
        "provenance[1] must be ComputedFromGeometry"
    );
}

#[test]
fn computed_source_all_stages_produce_identical_planes() {
    let hydro = make_sobradinho_computed_hydro(0);
    let config = computed_fpha_config(0);

    let geo_rows = make_sobradinho_geometry_rows(0);
    let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), geo_refs);

    let stages = [make_stage(0), make_stage(1), make_stage(2)];

    let layout =
        find_fpha_config_for_stage(&config, &stages[0]).expect("computed config covers stage 0");
    let cached_fit = super::fit_planes_for_hydro(
        &hydro,
        layout,
        &geometry_map,
        0.0,
        super::TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        0,
        false,
    )
    .expect("fit_planes_for_hydro must succeed");

    let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();

    let stage_planes: Vec<Vec<FphaPlane>> = stages
        .iter()
        .map(|stage| {
            let model = super::resolve_stage_model(
                &hydro,
                stage,
                Some(&config),
                ProductionModelSource::ComputedFromGeometry,
                &empty_hyperplane_map,
                Some(&cached_fit.planes),
                None,
            )
            .expect("resolve_stage_model must succeed");
            match model {
                ResolvedProductionModel::Fpha { planes, .. } => planes,
                other => panic!("expected Fpha, got {other:?}"),
            }
        })
        .collect();

    assert_eq!(
        stage_planes.len(),
        3,
        "must have plane vectors for 3 stages"
    );

    let expected_count = stage_planes[0].len();
    for (s, planes) in stage_planes.iter().enumerate() {
        assert_eq!(
            planes.len(),
            expected_count,
            "stage {s}: plane count must be {expected_count}, got {}",
            planes.len()
        );
    }

    for (s, planes) in stage_planes.iter().enumerate().skip(1) {
        for (p, plane) in planes.iter().enumerate() {
            assert_eq!(
                *plane, stage_planes[0][p],
                "stage {s} plane {p}: must be identical to stage 0 plane {p}"
            );
        }
    }
}

#[test]
fn computed_source_missing_efficiency_returns_validation_error() {
    let mut hydro = make_computed_hydro(0);
    hydro.name = "TestHydro".to_string();
    hydro.efficiency = None;

    let rows = make_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
    geometry_map.insert(EntityId::from(0), row_refs);

    let err = validate_computed_prerequisites(&hydro, &geometry_map)
        .expect_err("should fail when efficiency is None");
    let msg = err.to_string();
    assert!(
        msg.contains("efficiency"),
        "error must mention 'efficiency', got: {msg}"
    );
    assert!(
        msg.contains("TestHydro"),
        "error must include hydro name 'TestHydro', got: {msg}"
    );
}

#[test]
fn computed_source_missing_losses_returns_validation_error() {
    let mut hydro = make_computed_hydro(0);
    hydro.hydraulic_losses = None;

    let rows = make_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
    geometry_map.insert(EntityId::from(0), row_refs);

    let err = validate_computed_prerequisites(&hydro, &geometry_map)
        .expect_err("should fail when hydraulic_losses is None");
    let msg = err.to_string();
    assert!(
        msg.contains("hydraulic_losses"),
        "error must mention 'hydraulic_losses', got: {msg}"
    );
    assert!(
        msg.contains(&hydro.name),
        "error must include hydro name '{}', got: {msg}",
        hydro.name
    );
}

// ── Stage-aware per-entry fitting tests ───────────────────────────────────

fn sobradinho_geometry_map(rows: &[HydroGeometryRow]) -> HashMap<EntityId, Vec<&HydroGeometryRow>> {
    let mut map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    map.insert(rows[0].hydro_id, rows.iter().collect());
    map
}

/// An empty tailrace-families map — every hydro falls back to its entity
/// `TailraceModel` (the inert fallback).
fn empty_families_map() -> HashMap<EntityId, TailraceFamilies> {
    HashMap::new()
}

/// A minimal single-hydro `System` + flat reference-fraction resolver for
/// `fit_computed_planes_per_stage`. The hydro has no `downstream_id`, so the
/// downstream level resolves to `None` (unused with an empty families map).
fn entity_fit_context(hydro: &Hydro) -> (System, HydroReferenceVolumeFractions) {
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro.clone()])
        .build()
        .expect("single-hydro system builds");
    let fractions = flat_reference_fractions(0.65, system.hydros());
    (system, fractions)
}

/// FPHA layout with an explicit absolute volume window. Distinct windows
/// drive distinct fitting bounds and therefore distinct plane sets.
fn computed_layout_with_window(min: Option<f64>, max: Option<f64>) -> FphaColumnLayout {
    FphaColumnLayout {
        source: "computed".to_string(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: Some(FittingWindow {
            volume_min_hm3: min,
            volume_max_hm3: max,
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
    }
}

#[test]
fn per_range_fits_differ_when_windows_differ() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![
                StageRange {
                    start_stage_id: 0,
                    end_stage_id: Some(1),
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(8_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
                StageRange {
                    start_stage_id: 2,
                    end_stage_id: None,
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(20_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
            ],
        },
    };

    let stages = [make_stage(0), make_stage(1), make_stage(2), make_stage(3)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-stage fit must succeed");

    assert_eq!(per_stage[0], per_stage[1], "range A stages must match");
    assert_eq!(per_stage[2], per_stage[3], "range B stages must match");
    assert_ne!(
        per_stage[0], per_stage[2],
        "range A and range B plane sets must differ when windows differ"
    );
}

#[test]
fn per_season_fits_differ_when_season_configs_differ() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "fpha".to_string(),
            seasons: vec![
                SeasonConfig {
                    season_id: 0,
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(8_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
                SeasonConfig {
                    season_id: 1,
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(20_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
            ],
        },
    };

    let mut stage_s0 = make_stage(0);
    stage_s0.season_id = Some(0);
    let mut stage_s1 = make_stage(1);
    stage_s1.season_id = Some(1);
    let stages = [stage_s0, stage_s1];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-season fit must succeed");

    assert_ne!(
        per_stage[0], per_stage[1],
        "season 0 and season 1 plane sets must differ when configs differ"
    );
}

#[test]
fn single_config_dedup_yields_identical_planes_across_stages() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    let config = computed_fpha_config(0);

    let stages = [make_stage(0), make_stage(1), make_stage(2)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("single-config fit must succeed");

    assert_eq!(per_stage.len(), 3, "one plane set per study stage");
    for s in 1..3 {
        assert_eq!(
            per_stage[0], per_stage[s],
            "single-config hydro: stage {s} planes must be identical to stage 0 (dedup)"
        );
    }

    // Each stage emits one export-row block, so the row count is
    // n_stages * planes_per_stage.
    assert_eq!(
        export_rows.len(),
        per_stage[0].len() * 3,
        "export rows must cover every (stage, plane)"
    );
}

#[test]
fn export_rows_carry_stage_id_and_canonical_order() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);
    let config = computed_fpha_config(0);

    let stages = [make_stage(0), make_stage(1), make_stage(2)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("fit must succeed");

    for row in &export_rows {
        assert!(
            row.stage_id.is_some(),
            "computed-FPHA export row must carry stage_id = Some(...), got None"
        );
    }

    let keys: Vec<(i32, Option<i32>, i32)> = export_rows
        .iter()
        .map(|r| (r.hydro_id.0, r.stage_id, r.plane_id))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(
        keys, sorted,
        "export rows must be ordered by (hydro_id, stage_id, plane_id)"
    );

    let mut seen_stages: Vec<i32> = export_rows.iter().filter_map(|r| r.stage_id).collect();
    seen_stages.sort_unstable();
    seen_stages.dedup();
    assert_eq!(
        seen_stages,
        vec![0, 1, 2],
        "every study stage must appear in the export rows"
    );
}

#[test]
fn coverage_gap_returns_validation_error() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: Some(1),
                model: "fpha".to_string(),
                fpha_config: Some(computed_layout_with_window(None, None)),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };

    let stages = [make_stage(0), make_stage(1), make_stage(2)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let err = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect_err("a coverage gap must error, not silently drop the stage");
    assert!(
        matches!(err, SddpError::Validation(ref msg) if msg.contains("stage 2")),
        "expected Validation naming the uncovered stage, got {err:?}"
    );
}

// ── long-term mean inflow for the lateral secant ────────────────────

fn inflow_row(hydro_id: i32, day: u32, value_m3s: f64) -> InflowHistoryRow {
    let start_date = NaiveDate::from_ymd_opt(2000, 1, day).unwrap();
    InflowHistoryRow {
        hydro_id: EntityId::from(hydro_id),
        start_date,
        end_date: start_date.succ_opt().unwrap(),
        value_m3s,
    }
}

#[test]
fn long_term_mean_inflow_is_per_hydro_canonical_mean() {
    let rows = vec![
        inflow_row(1, 1, 100.0),
        inflow_row(2, 1, 9_999.0), // different hydro — must be excluded
        inflow_row(1, 2, 200.0),
        inflow_row(1, 3, 300.0),
    ];
    let system = SystemBuilder::new()
        .inflow_history(rows)
        .build()
        .expect("valid system");

    // mean(100, 200, 300) = 200. This feeds S_max = 2·mean at the fit site
    // (resolve_s_max's contract, exercised in the fpha_fitting::secant tests).
    let long_term_mean_inflow_m3s =
        super::long_term_mean_inflow_reference(&system, EntityId::from(1));
    assert_eq!(
        long_term_mean_inflow_m3s, 200.0,
        "long-term mean inflow must be the per-hydro mean of its own series"
    );
}

#[test]
fn long_term_mean_inflow_empty_history_is_zero() {
    let system = SystemBuilder::new().build().expect("empty system is valid");
    let long_term_mean_inflow_m3s =
        super::long_term_mean_inflow_reference(&system, EntityId::from(1));
    assert_eq!(
        long_term_mean_inflow_m3s, 0.0,
        "no inflow history must yield long-term mean inflow = 0"
    );
}

#[test]
fn long_term_mean_inflow_is_order_independent() {
    let ascending = vec![
        inflow_row(1, 1, 10.0),
        inflow_row(1, 2, 20.0),
        inflow_row(1, 3, 60.0),
    ];
    let descending = vec![
        inflow_row(1, 3, 60.0),
        inflow_row(1, 2, 20.0),
        inflow_row(1, 1, 10.0),
    ];
    let sys_asc = SystemBuilder::new()
        .inflow_history(ascending)
        .build()
        .expect("valid");
    let sys_desc = SystemBuilder::new()
        .inflow_history(descending)
        .build()
        .expect("valid");

    let mlt_asc = super::long_term_mean_inflow_reference(&sys_asc, EntityId::from(1));
    let mlt_desc = super::long_term_mean_inflow_reference(&sys_desc, EntityId::from(1));
    assert_eq!(
        mlt_asc, mlt_desc,
        "long-term mean inflow must be order-independent (declaration-order invariance)"
    );
}

/// The one-pass batch table's per-hydro mean must be bit-identical to
/// [`super::long_term_mean_inflow_reference`]'s direct per-hydro filtered scan,
/// for every hydro, including one with no history at all.
#[test]
fn build_long_term_mean_inflow_table_matches_per_hydro_reference() {
    let rows = vec![
        inflow_row(1, 1, 100.0),
        inflow_row(2, 1, 9_999.0),
        inflow_row(1, 2, 200.0),
        inflow_row(3, 1, 5.0),
        inflow_row(1, 3, 300.0),
        inflow_row(2, 2, 1.0),
    ];
    let system = SystemBuilder::new()
        .inflow_history(rows)
        .build()
        .expect("valid system");

    let table = super::build_long_term_mean_inflow_table(&system);
    for hydro_id in [1, 2, 3, 4] {
        let hydro_id = EntityId::from(hydro_id);
        let expected = super::long_term_mean_inflow_reference(&system, hydro_id);
        let actual = table.get(&hydro_id).copied().unwrap_or(0.0);
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "hydro {hydro_id:?}: batch table mean must be bit-identical to the per-hydro scan"
        );
    }
}

// ── resolve_downstream_level tests ───────────────────────────────────────

fn make_bus() -> Bus {
    Bus {
        id: EntityId::from(10),
        name: "Bus10".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: Vec::new(),
        excess_cost: 0.0,
    }
}

/// Build a `HydroReferenceVolumeFractions` whose `get(hydro, stage)` returns
/// the absolute hm³ for `default_fraction` resolved against each plant's band —
/// the resolved-hm³ semantics the downstream-level consumer reads.
fn flat_reference_fractions(
    default_fraction: f64,
    hydros: &[Hydro],
) -> HydroReferenceVolumeFractions {
    let resolved: Vec<(EntityId, StudyPos, f64)> = hydros
        .iter()
        .flat_map(|h| {
            let v = h.min_storage_hm3 + default_fraction * (h.max_storage_hm3 - h.min_storage_hm3);
            (0..8).map(move |s| (h.id, StudyPos(s), v))
        })
        .collect();
    build_hydro_reference_volumes_resolved(&resolved, 0.0)
}

fn geometry_map_for(rows: &[HydroGeometryRow]) -> HashMap<EntityId, Vec<&HydroGeometryRow>> {
    let mut map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    for r in rows {
        map.entry(r.hydro_id).or_default().push(r);
    }
    map
}

#[test]
fn resolve_downstream_level_no_downstream_is_none() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro.clone()])
        .build()
        .expect("system");
    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let fractions = flat_reference_fractions(0.5, system.hydros());

    let level = resolve_downstream_level(
        &hydro,
        StudyPos(stage.index),
        &system,
        &geometry_map,
        &fractions,
    );
    assert!(level.is_none(), "no downstream_id must resolve to None");
}

#[test]
fn resolve_downstream_level_matches_independent_forebay_height() {
    let mut upstream = make_hydro(0, HydroGenerationModel::Fpha);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream.clone()])
        .build()
        .expect("system");

    let geo_rows = make_geometry_rows(1);
    let geometry_map = geometry_map_for(&geo_rows);
    let fraction = 0.5;
    let fractions = flat_reference_fractions(fraction, system.hydros());

    let level = resolve_downstream_level(
        &upstream,
        StudyPos(stage.index),
        &system,
        &geometry_map,
        &fractions,
    )
    .expect("downstream level must resolve");

    // Independent reference: v_ref = v_min + fraction·(v_max − v_min) on the
    // downstream plant, evaluated through a freshly-built ForebayTable.
    let v_ref = downstream.min_storage_hm3
        + fraction * (downstream.max_storage_hm3 - downstream.min_storage_hm3);
    let table = ForebayTable::new(&geo_rows, &downstream.name).expect("forebay");
    let expected = table.height(v_ref);

    assert!(
        (level - expected).abs() < 1e-9,
        "resolved level {level} must match independent ForebayTable::height {expected}"
    );
}

#[test]
fn resolve_downstream_level_missing_geometry_is_none() {
    let mut upstream = make_hydro(0, HydroGenerationModel::Fpha);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream])
        .build()
        .expect("system");

    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let fractions = flat_reference_fractions(0.5, system.hydros());

    let level = resolve_downstream_level(
        &upstream,
        StudyPos(stage.index),
        &system,
        &geometry_map,
        &fractions,
    );
    assert!(
        level.is_none(),
        "missing downstream geometry must resolve to None"
    );
}

// ── Tailrace-source resolution + dedup-key tests ──────────────────────────

/// A hydro absent from the families map falls back to `TailraceSource::Entity`
/// (its entity `TailraceModel`) — a per-plant fallback: a table for other plants
/// is irrelevant.
#[test]
fn resolve_tailrace_source_absent_from_map_is_entity() {
    let hydro = make_sobradinho_computed_hydro(0);
    let stage = make_stage(0);
    let (system, fractions) = entity_fit_context(&hydro);
    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    // Families map populated for a DIFFERENT plant id only.
    let other_rows = vec![TailraceCurveRow {
        hydro_id: EntityId::from(99),
        family_id: 1,
        downstream_reference_level_m: None,
        segment_id: 1,
        outflow_min_m3s: 0.0,
        outflow_max_m3s: 1000.0,
        coefficient_0: 5.0,
        coefficient_1: 0.0,
        coefficient_2: 0.0,
        coefficient_3: 0.0,
        coefficient_4: 0.0,
    }];
    let families_map =
        super::build_tailrace_families_map(&other_rows).expect("families map builds");

    let source = super::resolve_tailrace_source(
        &hydro,
        StudyPos(stage.index),
        &families_map,
        &geometry_map,
        &fractions,
        &system,
    );
    match source {
        TailraceSource::Entity(model) => {
            assert_eq!(
                model, hydro.tailrace,
                "fallback must carry the entity TailraceModel unchanged"
            );
        }
        TailraceSource::Families { .. } => {
            panic!("a plant absent from the families map must resolve to Entity")
        }
    }
}

/// The dedup cache key includes `downstream_level_m`, so two entries of one
/// hydro resolving DISTINCT downstream levels (via a multi-family table) are not
/// collapsed to one fit — their fitted planes differ. (Dropping the level from
/// the key would wrongly collapse them.)
#[test]
fn dedup_key_separates_distinct_downstream_levels() {
    // Upstream computed-FPHA plant 0 discharges into downstream plant 1.
    let mut upstream = make_sobradinho_computed_hydro(0);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    // Downstream geometry spanning heights 380 → 420 over volume 100 → 2000,
    // so fraction 0.0 → 380 m and fraction 1.0 → 420 m.
    let down_geo = [
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 100.0,
            height_m: 380.0,
            area_km2: 10.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 2000.0,
            height_m: 420.0,
            area_km2: 50.0,
        },
    ];
    let up_geo = make_sobradinho_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), up_geo.iter().collect());
    geometry_map.insert(EntityId::from(1), down_geo.iter().collect());

    // Two-family tailrace for the upstream plant keyed at 380 and 420 m with
    // distinct heights (constant 2.0 vs 40.0 m), so the resolved level picks a
    // materially different tailrace.
    let tailrace_rows = vec![
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 1,
            downstream_reference_level_m: Some(380.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 2.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 2,
            downstream_reference_level_m: Some(420.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 40.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
    ];
    let families_map =
        super::build_tailrace_families_map(&tailrace_rows).expect("families map builds");

    let mut stage0 = make_stage(0);
    stage0.season_id = Some(0);
    let mut stage1 = make_stage(1);
    stage1.season_id = Some(1);
    // Downstream plant 1 band is [100, 2000]; stage 0 resolves fraction 0.0001
    // (→ low level), stage 1 fraction 1.0 (→ high level), keyed by stage index.
    let down_v_min = downstream.min_storage_hm3;
    let down_span = downstream.max_storage_hm3 - downstream.min_storage_hm3;
    let fractions = build_hydro_reference_volumes_resolved(
        &[
            (
                EntityId::from(1),
                StudyPos(0),
                down_v_min + 0.0001 * down_span,
            ),
            (EntityId::from(1), StudyPos(1), down_v_min + 1.0 * down_span),
        ],
        0.0,
    );

    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream])
        .build()
        .expect("two-hydro system builds");

    // One open-ended range → one config across both stages, so ONLY the
    // downstream level varies between the two stages.
    let config = computed_fpha_config(0);
    let stages = [&stage0, &stage1];
    let stage_refs: Vec<&Stage> = stages.to_vec();

    let mut export_rows = Vec::new();
    let per_stage = super::fit_computed_planes_per_stage(
        &upstream,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-stage fit succeeds");

    let lvl0 = resolve_downstream_level(
        &upstream,
        StudyPos(stage0.index),
        &system,
        &geometry_map,
        &fractions,
    )
    .expect("stage 0 level");
    let lvl1 = resolve_downstream_level(
        &upstream,
        StudyPos(stage1.index),
        &system,
        &geometry_map,
        &fractions,
    )
    .expect("stage 1 level");
    assert_ne!(
        lvl0.to_bits(),
        lvl1.to_bits(),
        "the two stages must resolve different downstream levels"
    );

    assert_ne!(
        per_stage[0], per_stage[1],
        "distinct downstream levels must yield distinct fits (dedup key includes the level)"
    );
}

/// Capture-all: `fit_computed_planes_per_stage` records one
/// `FphaDeviationDiagnostic` per DISTINCT fit regardless of the warn threshold —
/// even a well-fit (under-threshold) surface is captured. A regression
/// re-introducing a warn-worthy push filter would drop the under-threshold
/// entries and fail the count.
#[test]
fn fit_computed_planes_per_stage_records_every_distinct_fit() {
    let mut upstream = make_sobradinho_computed_hydro(0);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let down_geo = [
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 100.0,
            height_m: 380.0,
            area_km2: 10.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 2000.0,
            height_m: 420.0,
            area_km2: 50.0,
        },
    ];
    let up_geo = make_sobradinho_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), up_geo.iter().collect());
    geometry_map.insert(EntityId::from(1), down_geo.iter().collect());

    let tailrace_rows = vec![
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 1,
            downstream_reference_level_m: Some(380.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 2.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 2,
            downstream_reference_level_m: Some(420.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 40.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
    ];
    let families_map =
        super::build_tailrace_families_map(&tailrace_rows).expect("families map builds");

    let mut stage0 = make_stage(0);
    stage0.season_id = Some(0);
    let mut stage1 = make_stage(1);
    stage1.season_id = Some(1);
    let down_v_min = downstream.min_storage_hm3;
    let down_span = downstream.max_storage_hm3 - downstream.min_storage_hm3;
    let fractions = build_hydro_reference_volumes_resolved(
        &[
            (
                EntityId::from(1),
                StudyPos(0),
                down_v_min + 0.0001 * down_span,
            ),
            (EntityId::from(1), StudyPos(1), down_v_min + 1.0 * down_span),
        ],
        0.0,
    );

    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream])
        .build()
        .expect("two-hydro system builds");

    let config = computed_fpha_config(0);
    let stage_refs: Vec<&Stage> = vec![&stage0, &stage1];

    let mut export_rows = Vec::new();
    let mut diagnostics = Vec::new();
    let per_stage = super::fit_computed_planes_per_stage(
        &upstream,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut diagnostics,
        &mut Vec::new(),
    )
    .expect("per-stage fit succeeds");

    assert_ne!(
        per_stage[0], per_stage[1],
        "the two stages must resolve distinct fits for this test to exercise two captures"
    );
    assert_eq!(
        diagnostics.len(),
        2,
        "one diagnostic per distinct fit must be captured (capture-all), not only warn-worthy ones"
    );
    // Each fit is tagged with the first stage it covers (0 and 1).
    assert_eq!(diagnostics[0].stage_id, 0);
    assert_eq!(diagnostics[1].stage_id, 1);
    assert!(
        diagnostics
            .iter()
            .any(|d| !d.deviation.exceeds_warn_threshold()),
        "a well-fit (under-threshold) fit must still be captured"
    );
}

/// Every distinct computed-FPHA fit reaches `fpha_fit_deviations` (capture-all)
/// in canonical `(hydro, stage)` order, with `hydro_id` from the outer zip —
/// even an under-threshold fit that raises no warn. The warn predicate itself
/// is exercised in `deviation.rs`.
#[test]
fn resolver_carries_every_fit_deviation_in_canonical_order() {
    let hydro = make_sobradinho_computed_hydro(0);
    let geo_rows = make_sobradinho_geometry_rows(0);
    let artifacts = cobre_io::CaseArtifacts {
        production_models: vec![computed_fpha_config(0)],
        hydro_geometry: geo_rows,
        ..Default::default()
    };

    let stages = vec![make_stage(0), make_stage(1)];
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(stages)
        .build()
        .expect("computed-FPHA system builds");

    let (_, _, _, _, _, fpha_fit_deviations, _) =
        super::resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve must succeed for a computed-FPHA case");

    // One open-ended range → a single distinct fit shared across both stages,
    // tagged with the first covered stage (id 0); the carry-up is per distinct
    // fit, not per covered stage.
    assert_eq!(
        fpha_fit_deviations.len(),
        1,
        "a single distinct fit must yield exactly one carried deviation entry"
    );
    let entry = fpha_fit_deviations[0];
    assert_eq!(entry.hydro_id, EntityId::from(0), "hydro_id from outer zip");
    assert_eq!(entry.stage_id, 0, "tagged with the first covered stage");
    assert!(
        entry.mean_abs_mw >= 0.0 && entry.max_abs_mw >= entry.mean_abs_mw,
        "deviation magnitudes must be well-formed"
    );
}

#[test]
fn export_rows_are_declaration_order_invariant_with_tailrace() {
    fn build_artifacts(
        tailrace_rows: Vec<TailraceCurveRow>,
        geo_rows: Vec<HydroGeometryRow>,
    ) -> cobre_io::CaseArtifacts {
        cobre_io::CaseArtifacts {
            production_models: vec![computed_fpha_config(0)],
            hydro_geometry: geo_rows,
            tailrace_curves: tailrace_rows,
            ..Default::default()
        }
    }

    let hydro = make_sobradinho_computed_hydro(0);
    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro.clone()])
        .stages(vec![stage])
        .build()
        .expect("single-hydro system builds");

    // Geometry rows are re-sorted by `build_geometry_map` inside the resolve
    // path, so a reversed geometry input must yield bit-identical fitted
    // planes — the declaration-order invariance the families path must uphold.
    let geo_fwd = make_sobradinho_geometry_rows(0);
    let mut geo_rev = geo_fwd.clone();
    geo_rev.reverse();

    // Flat single-segment tailrace in canonical order (as the loader produces
    // it); the shuffled dimension under test is the geometry input.
    let segment = TailraceCurveRow {
        hydro_id: EntityId::from(0),
        family_id: 1,
        downstream_reference_level_m: None,
        segment_id: 1,
        outflow_min_m3s: 0.0,
        outflow_max_m3s: 100_000.0,
        coefficient_0: 2.0,
        coefficient_1: 0.0,
        coefficient_2: 0.0,
        coefficient_3: 0.0,
        coefficient_4: 0.0,
    };

    let art_fwd = build_artifacts(vec![segment.clone()], geo_fwd);
    let art_rev = build_artifacts(vec![segment], geo_rev);

    let (_, _, _, rows_fwd, _, _, _) =
        resolve_production_models_from_artifacts(&system, &art_fwd, false)
            .expect("forward resolve");
    let (_, _, _, rows_rev, _, _, _) =
        resolve_production_models_from_artifacts(&system, &art_rev, false)
            .expect("reversed resolve");

    assert_eq!(
        rows_fwd.len(),
        rows_rev.len(),
        "export-row counts must match across input orderings"
    );
    for (a, b) in rows_fwd.iter().zip(&rows_rev) {
        assert_eq!(a.hydro_id, b.hydro_id);
        assert_eq!(a.stage_id, b.stage_id);
        assert_eq!(a.plane_id, b.plane_id);
        assert_eq!(
            a.gamma_0.to_bits(),
            b.gamma_0.to_bits(),
            "gamma_0 must be bit-identical across input orderings"
        );
        assert_eq!(a.gamma_v.to_bits(), b.gamma_v.to_bits());
        assert_eq!(a.gamma_q.to_bits(), b.gamma_q.to_bits());
        assert_eq!(a.gamma_s.to_bits(), b.gamma_s.to_bits());
    }
}

// ── reference_volume resolution ─────────────────────────────────────────

fn config_with_reference_volume(rv: Option<ReferenceVolume>) -> ProductionModelConfig {
    ProductionModelConfig {
        hydro_id: EntityId::from(1),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: rv,
                productivity_mw_per_m3s: Some(0.5),
            }],
        },
    }
}

#[test]
fn find_reference_volume_for_stage_returns_covering_entry_value() {
    let config = config_with_reference_volume(Some(ReferenceVolume::AbsoluteHm3(800.0)));
    let stage0 = make_stage(0);
    assert_eq!(
        find_reference_volume_for_stage(&config, &stage0),
        Some(&ReferenceVolume::AbsoluteHm3(800.0))
    );
}

#[test]
fn find_reference_volume_for_stage_returns_none_when_unset() {
    let config = config_with_reference_volume(None);
    let stage0 = make_stage(0);
    assert_eq!(find_reference_volume_for_stage(&config, &stage0), None);
}

#[test]
fn find_reference_volume_for_stage_returns_none_outside_coverage() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(1),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 5,
                end_stage_id: Some(10),
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: Some(ReferenceVolume::AbsoluteHm3(800.0)),
                productivity_mw_per_m3s: Some(0.5),
            }],
        },
    };
    let stage0 = make_stage(0);
    assert_eq!(find_reference_volume_for_stage(&config, &stage0), None);
}

#[test]
fn resolve_reference_volume_hm3_absolute_passthrough() {
    let rv = ReferenceVolume::AbsoluteHm3(800.0);
    let resolved = resolve_reference_volume_hm3(Some(&rv), 100.0, 2000.0);
    assert_eq!(resolved, 800.0);
}

#[test]
fn resolve_reference_volume_hm3_percentile_against_band() {
    let rv = ReferenceVolume::Percentile(0.5);
    let resolved = resolve_reference_volume_hm3(Some(&rv), 100.0, 200.0);
    assert!((resolved - 150.0).abs() <= f64::EPSILON);
}

#[test]
fn resolve_reference_volume_hm3_none_reproduces_065_fraction() {
    let v_min = 100.0_f64;
    let v_max = 200.0_f64;
    let resolved = resolve_reference_volume_hm3(None, v_min, v_max);
    // Bit-identical to the 0.65-fraction formula — the byte-identity guarantee
    // the consumer relies on.
    let expected = v_min + 0.65 * (v_max - v_min);
    assert_eq!(resolved.to_bits(), expected.to_bits());
}

#[test]
fn resolve_reference_volume_hm3_degenerate_band_is_not_nan() {
    // A `v_max == v_min` band must collapse to `v_min` via multiplication,
    // never produce a `0/0` NaN from a division-based percentile conversion.
    let rv = ReferenceVolume::Percentile(0.5);
    let resolved = resolve_reference_volume_hm3(Some(&rv), 50.0, 50.0);
    assert!(!resolved.is_nan());
    assert_eq!(resolved, 50.0);
}

// ── Resolver-wiring tests (the JSON-sourced reference volume) ─────────────

fn ref_vol_system(v_min: f64, v_max: f64) -> System {
    let mut hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    hydro.min_storage_hm3 = v_min;
    hydro.max_storage_hm3 = v_max;
    SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(vec![make_stage(0), make_stage(1)])
        .build()
        .expect("single-hydro two-stage system builds")
}

fn ref_vol_artifacts(rv: Option<ReferenceVolume>) -> cobre_io::CaseArtifacts {
    cobre_io::CaseArtifacts {
        production_models: vec![ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    reference_volume: rv,
                    productivity_mw_per_m3s: Some(1.0),
                }],
            },
        }],
        ..Default::default()
    }
}

/// With no `reference_volume` declared, the resolver returns the 0.65-fraction
/// hm³ bit-for-bit — the byte-identity a baseline regression depends on.
#[test]
fn resolver_default_path_returns_065_fraction_hm3_bit_for_bit() {
    let v_min = 100.0_f64;
    let v_max = 200.0_f64;
    let system = ref_vol_system(v_min, v_max);
    let artifacts = ref_vol_artifacts(None);

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    let expected = v_min + 0.65 * (v_max - v_min);
    for stage_idx in [0_usize, 1] {
        assert_eq!(
            resolver
                .get(EntityId::from(0), StudyPos(stage_idx))
                .to_bits(),
            expected.to_bits(),
            "stage {stage_idx}: undeclared reference volume must be the 0.65-fraction hm³"
        );
    }
}

#[test]
fn resolver_declared_absolute_flows_through_get() {
    let system = ref_vol_system(100.0, 2000.0);
    let artifacts = ref_vol_artifacts(Some(ReferenceVolume::AbsoluteHm3(800.0)));

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    assert_eq!(resolver.get(EntityId::from(0), StudyPos(0)), 800.0);
    assert_eq!(resolver.get(EntityId::from(0), StudyPos(1)), 800.0);
}

/// A declared `percentile: 0.5` on band `[100, 200]` resolves to `150.0`.
#[test]
fn resolver_declared_percentile_resolves_against_band() {
    let system = ref_vol_system(100.0, 200.0);
    let artifacts = ref_vol_artifacts(Some(ReferenceVolume::Percentile(0.5)));

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    assert_eq!(resolver.get(EntityId::from(0), StudyPos(0)), 150.0);
}

/// A study horizon may begin in any season and wrap the cycle. With a SEASONAL
/// `reference_volume` config, the resolver is keyed by 0-based study position —
/// never by `season_id` or `stage.index` — so each position resolves THAT
/// stage's own season value (via `find_reference_volume_for_stage`). A
/// season-keyed or index-keyed lookup would mis-assign for a horizon not
/// starting at season 0.
#[test]
fn seasonal_reference_volume_supports_nonzero_start_season() {
    let (v_min, v_max) = (100.0_f64, 200.0_f64);

    // Horizon begins in season 2 and wraps to season 1: study position 0 →
    // season 2, position 1 → season 1. Neither is season 0.
    let mut hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    hydro.min_storage_hm3 = v_min;
    hydro.max_storage_hm3 = v_max;
    let mut stage0 = make_stage(0);
    stage0.season_id = Some(2);
    let mut stage1 = make_stage(1);
    stage1.season_id = Some(1);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(vec![stage0, stage1])
        .build()
        .expect("two-stage non-zero-start-season system builds");

    let artifacts = cobre_io::CaseArtifacts {
        production_models: vec![ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::Seasonal {
                default_model: "constant_productivity".to_string(),
                seasons: vec![
                    SeasonConfig {
                        season_id: 1,
                        model: "constant_productivity".to_string(),
                        fpha_config: None,
                        reference_volume: Some(ReferenceVolume::Percentile(0.8)),
                        productivity_mw_per_m3s: Some(1.0),
                    },
                    SeasonConfig {
                        season_id: 2,
                        model: "constant_productivity".to_string(),
                        fpha_config: None,
                        reference_volume: Some(ReferenceVolume::Percentile(0.2)),
                        productivity_mw_per_m3s: Some(1.0),
                    },
                ],
            },
        }],
        ..Default::default()
    };

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    // Position 0 carries season 2's value (0.2 → 120), position 1 season 1's (0.8 → 180).
    assert_eq!(
        resolver.get(EntityId::from(0), StudyPos(0)),
        v_min + 0.2 * (v_max - v_min),
        "study position 0 (season 2) must resolve percentile 0.2"
    );
    assert_eq!(
        resolver.get(EntityId::from(0), StudyPos(1)),
        v_min + 0.8 * (v_max - v_min),
        "study position 1 (season 1) must resolve percentile 0.8"
    );
}

fn resolve_single_computed_hydro(hydro: Hydro) -> ResolveProductionResult {
    let artifacts = cobre_io::CaseArtifacts {
        production_models: vec![computed_fpha_config(0)],
        hydro_geometry: make_sobradinho_geometry_rows(0),
        ..Default::default()
    };
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(vec![make_stage(0), make_stage(1)])
        .build()
        .expect("computed-FPHA system builds");
    super::resolve_production_models_from_artifacts(&system, &artifacts, false)
        .expect("resolve must succeed for a computed-FPHA case")
}

#[test]
fn computed_fpha_without_turbine_capacity_resolves_to_zero_productivity() {
    let mut hydro = make_sobradinho_computed_hydro(0);
    hydro.max_turbined_m3s = 0.0;

    let (set, _, provenance, export_rows, _, fpha_fit_deviations, _) =
        resolve_single_computed_hydro(hydro);

    for stage_idx in 0..2 {
        assert!(
            matches!(
                set.model(0, stage_idx),
                ResolvedProductionModel::ConstantProductivity { productivity } if *productivity == 0.0
            ),
            "stage {stage_idx}: got {:?}",
            set.model(0, stage_idx)
        );
    }
    assert_eq!(
        provenance,
        vec![(EntityId::from(0), ProductionModelSource::NoTurbineCapacity)]
    );
    assert!(export_rows.is_empty() && fpha_fit_deviations.is_empty());
}

#[test]
fn computed_fpha_without_generation_capacity_still_fits_planes() {
    let mut hydro = make_sobradinho_computed_hydro(0);
    hydro.max_generation_mw = 0.0;

    let (set, _, provenance, _, _, _, _) = resolve_single_computed_hydro(hydro);

    assert!(
        matches!(
            set.model(0, 0),
            ResolvedProductionModel::Fpha { planes, .. } if !planes.is_empty()
        ),
        "got {:?}",
        set.model(0, 0)
    );
    assert_eq!(
        provenance,
        vec![(
            EntityId::from(0),
            ProductionModelSource::ComputedFromGeometry
        )]
    );
}

#[test]
fn computed_fpha_without_turbine_capacity_still_requires_its_prerequisites() {
    let mut hydro = make_sobradinho_computed_hydro(0);
    hydro.max_turbined_m3s = 0.0;
    hydro.tailrace = None;
    let artifacts = cobre_io::CaseArtifacts {
        production_models: vec![computed_fpha_config(0)],
        hydro_geometry: make_sobradinho_geometry_rows(0),
        ..Default::default()
    };
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(vec![make_stage(0)])
        .build()
        .expect("computed-FPHA system builds");

    let err = super::resolve_production_models_from_artifacts(&system, &artifacts, false)
        .expect_err("a missing tailrace must still be rejected");
    assert!(err.to_string().contains("tailrace"), "got: {err}");
}

#[test]
fn precomputed_round_trip_keeps_a_zero_capacity_plant_at_zero_productivity() {
    let mut uncapped = make_sobradinho_computed_hydro(0);
    uncapped.max_generation_mw = 10_000.0;
    let mut capacity_less = make_sobradinho_computed_hydro(1);
    capacity_less.max_turbined_m3s = 0.0;
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![uncapped, capacity_less])
        .stages(vec![make_stage(0), make_stage(1)])
        .build()
        .expect("two-hydro computed-FPHA system builds");
    let computed_artifacts = cobre_io::CaseArtifacts {
        production_models: vec![computed_fpha_config(0), computed_fpha_config(1)],
        hydro_geometry: [
            make_sobradinho_geometry_rows(0),
            make_sobradinho_geometry_rows(1),
        ]
        .concat(),
        ..Default::default()
    };
    let (computed_set, _, _, export_rows, _, _, _) =
        super::resolve_production_models_from_artifacts(&system, &computed_artifacts, false)
            .expect("computed resolve must succeed");

    let precomputed_artifacts = cobre_io::CaseArtifacts {
        production_models: vec![precomputed_fpha_config(0), precomputed_fpha_config(1)],
        fpha_hyperplanes: export_rows,
        ..Default::default()
    };
    let (set, _, provenance, _, _, _, _) =
        super::resolve_production_models_from_artifacts(&system, &precomputed_artifacts, false)
            .expect("the exported hyperplanes must resolve as precomputed");

    let plane_count = |set: &ProductionModelSet, stage_idx: usize| match set.model(0, stage_idx) {
        ResolvedProductionModel::Fpha { planes } => planes.len(),
        other => panic!("hydro 0 stage {stage_idx}: expected Fpha, got {other:?}"),
    };
    for stage_idx in 0..2 {
        assert_eq!(
            plane_count(&set, stage_idx),
            plane_count(&computed_set, stage_idx),
            "stage {stage_idx}: precomputed plane count must match the computed fit"
        );
        assert!(
            matches!(
                set.model(1, stage_idx),
                ResolvedProductionModel::ConstantProductivity { productivity } if *productivity == 0.0
            ),
            "hydro 1 stage {stage_idx}: got {:?}",
            set.model(1, stage_idx)
        );
    }
    assert_eq!(
        provenance,
        vec![
            (
                EntityId::from(0),
                ProductionModelSource::PrecomputedHyperplanes
            ),
            (EntityId::from(1), ProductionModelSource::NoTurbineCapacity),
        ]
    );
}

fn resolve_single_precomputed_hydro(
    hydro: Hydro,
    fpha_hyperplanes: Vec<FphaHyperplaneRow>,
) -> Result<ResolveProductionResult, SddpError> {
    let artifacts = cobre_io::CaseArtifacts {
        production_models: vec![precomputed_fpha_config(0)],
        fpha_hyperplanes,
        ..Default::default()
    };
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(vec![make_stage(0)])
        .build()
        .expect("precomputed-FPHA system builds");
    super::resolve_production_models_from_artifacts(&system, &artifacts, false)
}

#[test]
fn precomputed_fpha_without_planes_is_exempt_only_for_zero_capacity() {
    let with_capacity = make_hydro(0, HydroGenerationModel::Fpha);
    assert!(with_capacity.has_turbine_capacity());
    let err = resolve_single_precomputed_hydro(with_capacity, Vec::new())
        .expect_err("a plant with turbine capacity still needs its hyperplane rows");
    assert!(
        matches!(&err, SddpError::Validation(msg) if msg.contains("no hyperplane rows")),
        "got: {err}"
    );

    let mut capacity_less = make_hydro(0, HydroGenerationModel::Fpha);
    capacity_less.max_turbined_m3s = 0.0;
    let (set, _, provenance, _, _, _, _) = resolve_single_precomputed_hydro(
        capacity_less,
        vec![valid_row(0, None, 0), valid_row(0, None, 1)],
    )
    .expect("a capacity-less plant with rows keeps its precomputed planes");
    assert!(
        matches!(
            set.model(0, 0),
            ResolvedProductionModel::Fpha { planes } if planes.len() == 2
        ),
        "got {:?}",
        set.model(0, 0)
    );
    assert_eq!(
        provenance,
        vec![(
            EntityId::from(0),
            ProductionModelSource::PrecomputedHyperplanes
        )]
    );
}

#[test]
fn precomputed_round_trip_reproduces_the_capped_computed_fit() {
    fn planes(set: &ProductionModelSet, stage_idx: usize) -> &[FphaPlane] {
        match set.model(0, stage_idx) {
            ResolvedProductionModel::Fpha { planes } => planes,
            other => panic!("stage {stage_idx}: expected Fpha, got {other:?}"),
        }
    }

    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![make_sobradinho_computed_hydro(0)])
        .stages(vec![make_stage(0), make_stage(1)])
        .build()
        .expect("computed-FPHA system builds");
    let computed_artifacts = cobre_io::CaseArtifacts {
        production_models: vec![computed_fpha_config(0)],
        hydro_geometry: make_sobradinho_geometry_rows(0),
        ..Default::default()
    };
    let (computed_set, _, _, export_rows, _, _, _) =
        super::resolve_production_models_from_artifacts(&system, &computed_artifacts, false)
            .expect("computed resolve must succeed");
    for stage_id in 0..2 {
        assert!(
            export_rows.iter().any(|row| row.stage_id == Some(stage_id)
                && row.gamma_q == 0.0
                && row.gamma_v == 0.0),
            "stage {stage_id}: the binding ceiling must export a flat cap plane"
        );
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fpha_hyperplanes.parquet");
    write_fpha_hyperplanes(&path, &export_rows).expect("write_fpha_hyperplanes must succeed");
    let precomputed_artifacts = cobre_io::CaseArtifacts {
        production_models: vec![precomputed_fpha_config(0)],
        fpha_hyperplanes: parse_fpha_hyperplanes(&path).expect("re-read must succeed"),
        ..Default::default()
    };
    let (set, _, provenance, _, _, _, _) =
        super::resolve_production_models_from_artifacts(&system, &precomputed_artifacts, false)
            .expect("the exported hyperplanes must resolve as precomputed");

    assert_eq!(
        provenance,
        vec![(
            EntityId::from(0),
            ProductionModelSource::PrecomputedHyperplanes
        )]
    );
    for stage_idx in 0..2 {
        let (got, want) = (planes(&set, stage_idx), planes(&computed_set, stage_idx));
        assert_eq!(got.len(), want.len(), "stage {stage_idx}: plane count");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            let bits =
                |p: &FphaPlane| [p.intercept, p.gamma_v, p.gamma_q, p.gamma_s].map(f64::to_bits);
            assert_eq!(bits(g), bits(w), "stage {stage_idx} plane {k}");
        }
    }
}
