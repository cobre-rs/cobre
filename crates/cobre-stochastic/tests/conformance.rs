//! End-to-end pipeline conformance tests for `cobre-stochastic`.
//!
//! Exercises the full pipeline from `System` input through to `sample_forward`
//! output, using a shared fixture matching the sampling-scheme-testing.md spec
//! (SS1): 3 study stages, 2 hydros, branching factor 5, identity correlation,
//! AR(1) models with known coefficients, and `base_seed` 42.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_core::{
    Bus, DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_stochastic::{build_stochastic_context, sample_forward};

fn make_bus(id: i32) -> Bus {
    Bus {
        id: EntityId(id),
        name: format!("Bus{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    }
}

fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
    Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    }
}

fn make_hydro(id: i32) -> Hydro {
    Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        bus_id: EntityId(0),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 1.0,
        },
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
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
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        },
    }
}

fn make_inflow_model(
    hydro_id: i32,
    stage_id: i32,
    mean_m3s: f64,
    std_m3s: f64,
    ar_coefficients: Vec<f64>,
    residual_std_ratio: f64,
) -> InflowModel {
    InflowModel {
        hydro_id: EntityId(hydro_id),
        stage_id,
        mean_m3s,
        std_m3s,
        ar_coefficients,
        residual_std_ratio,
    }
}

fn identity_correlation(entity_ids: &[i32]) -> CorrelationModel {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn shared_fixture() -> cobre_core::System {
    let hydros = vec![make_hydro(1), make_hydro(2)];

    // Pre-study stage (id=-1) provides lag-1 inflow statistics only.
    // Study stages (ids 0, 1, 2) form the optimization horizon.
    let stages = vec![
        make_stage(0, -1, 5), // pre-study — excluded from opening tree
        make_stage(1, 0, 5),
        make_stage(2, 1, 5),
        make_stage(3, 2, 5),
    ];

    // Pre-study stage inflow models (lag-1 statistics for coefficient conversion).
    // Same mean and std as the study stages because both hydros are stationary.
    let inflow_models = vec![
        // Hydro 1 — pre-study and three study stages
        make_inflow_model(1, -1, 100.0, 30.0, vec![], 1.0),
        make_inflow_model(1, 0, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(1, 1, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(1, 2, 100.0, 30.0, vec![0.3], 0.954),
        // Hydro 2 — pre-study and three study stages
        make_inflow_model(2, -1, 200.0, 40.0, vec![], 1.0),
        make_inflow_model(2, 0, 200.0, 40.0, vec![0.4], 0.917),
        make_inflow_model(2, 1, 200.0, 40.0, vec![0.4], 0.917),
        make_inflow_model(2, 2, 200.0, 40.0, vec![0.4], 0.917),
    ];

    SystemBuilder::new()
        .buses(vec![make_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(identity_correlation(&[1, 2]))
        .build()
        .expect("shared_fixture: system build must succeed")
}

#[test]
fn pipeline_builds_with_correct_dimensions() {
    let system = shared_fixture();
    let ctx = build_stochastic_context(&system, 42, &[], &[], None)
        .expect("build_stochastic_context must succeed for the shared fixture");

    assert_eq!(ctx.dim(), 2, "expected dim=2 (two hydros)");
    assert_eq!(ctx.n_stages(), 3, "expected n_stages=3 (study stages only)");
    assert_eq!(ctx.base_seed(), 42, "expected base_seed=42");
}

/// PAR coefficient cache contains the hand-computed reference values.
///
/// For stage 0 (study stage id=0), the lag-1 stage has id=-1 with the same
/// mean and std (stationary series with a same-std pre-study stage).
///
/// Hand-computed expected values follow the formula
/// `psi = psi_star * s_m / s_lag` and
/// `base = mu - psi * mu_lag`.
///
/// Hydro 1 (hydro index 0, sorted by `EntityId`):
///   psi = 0.3 * 30.0 / 30.0 = 0.3, base = 100.0 - 0.3*100.0 = 70.0,
///   sigma = 30.0 * 0.954 = 28.62
///
/// Hydro 2 (hydro index 1, sorted by `EntityId`):
///   psi = 0.4 * 40.0 / 40.0 = 0.4, base = 200.0 - 0.4*200.0 = 120.0,
///   sigma = 40.0 * 0.917 = 36.68
#[test]
fn par_lp_coefficients_match_hand_computed() {
    let system = shared_fixture();
    let ctx = build_stochastic_context(&system, 42, &[], &[], None)
        .expect("build_stochastic_context must succeed for the shared fixture");

    let par = ctx.par();
    let tol = 1e-10;

    // --- Hydro 1 (h_idx = 0, EntityId(1)) at stage 0 ---
    let expected_base_h1 = 70.0_f64; // 100.0 - (0.3 * 30.0 / 30.0) * 100.0
    assert!(
        (par.deterministic_base(0, 0) - expected_base_h1).abs() < tol,
        "hydro 1 stage 0 deterministic_base: expected {expected_base_h1}, got {}",
        par.deterministic_base(0, 0)
    );

    let expected_sigma_h1 = 30.0 * 0.954; // 28.62
    assert!(
        (par.sigma(0, 0) - expected_sigma_h1).abs() < tol,
        "hydro 1 stage 0 sigma: expected {expected_sigma_h1}, got {}",
        par.sigma(0, 0)
    );

    let expected_psi_h1 = 0.3_f64; // 0.3 * 30.0 / 30.0
    let psi_h1 = par.psi_slice(0, 0);
    assert!(!psi_h1.is_empty(), "psi_slice for AR(1) must not be empty");
    assert!(
        (psi_h1[0] - expected_psi_h1).abs() < tol,
        "hydro 1 stage 0 psi[0]: expected {expected_psi_h1}, got {}",
        psi_h1[0]
    );

    // --- Hydro 2 (h_idx = 1, EntityId(2)) at stage 0 ---
    let expected_base_h2 = 120.0_f64; // 200.0 - (0.4 * 40.0 / 40.0) * 200.0
    assert!(
        (par.deterministic_base(0, 1) - expected_base_h2).abs() < tol,
        "hydro 2 stage 0 deterministic_base: expected {expected_base_h2}, got {}",
        par.deterministic_base(0, 1)
    );

    let expected_sigma_h2 = 40.0 * 0.917; // 36.68
    assert!(
        (par.sigma(0, 1) - expected_sigma_h2).abs() < tol,
        "hydro 2 stage 0 sigma: expected {expected_sigma_h2}, got {}",
        par.sigma(0, 1)
    );

    let expected_psi_h2 = 0.4_f64; // 0.4 * 40.0 / 40.0
    let psi_h2 = par.psi_slice(0, 1);
    assert!(!psi_h2.is_empty(), "psi_slice for AR(1) must not be empty");
    assert!(
        (psi_h2[0] - expected_psi_h2).abs() < tol,
        "hydro 2 stage 0 psi[0]: expected {expected_psi_h2}, got {}",
        psi_h2[0]
    );
}

/// Opening tree has the expected structural dimensions.
///
/// - 3 stages (pre-study stage excluded)
/// - 5 openings per stage (uniform `branching_factor` 5)
/// - dim 2 (two hydros)
/// - All values are finite
#[test]
fn opening_tree_structure_correct() {
    let system = shared_fixture();
    let ctx = build_stochastic_context(&system, 42, &[], &[], None)
        .expect("build_stochastic_context must succeed for the shared fixture");

    let tree = ctx.opening_tree();

    assert_eq!(tree.n_stages(), 3, "expected 3 study stages");
    assert_eq!(tree.n_openings(0), 5, "stage 0 must have 5 openings");
    assert_eq!(tree.n_openings(1), 5, "stage 1 must have 5 openings");
    assert_eq!(tree.n_openings(2), 5, "stage 2 must have 5 openings");
    assert_eq!(tree.dim(), 2, "dim must equal number of hydros");

    for stage in 0..tree.n_stages() {
        for opening in 0..tree.n_openings(stage) {
            for &v in tree.opening(stage, opening) {
                assert!(
                    v.is_finite(),
                    "non-finite value at stage={stage} opening={opening}"
                );
            }
        }
    }
}

/// `sample_forward` returns valid `(index, slice)` pairs across multiple calls.
#[test]
fn sample_forward_returns_valid_output() {
    let system = shared_fixture();
    let ctx = build_stochastic_context(&system, 42, &[], &[], None)
        .expect("build_stochastic_context must succeed for the shared fixture");

    let view = ctx.tree_view();
    let base_seed = ctx.base_seed();

    for iteration in 0_u32..3 {
        for scenario in 0_u32..5 {
            for (stage_idx, stage_domain_id) in [(0usize, 0u32), (1, 1), (2, 2)] {
                let (idx, slice) = sample_forward(
                    &view,
                    base_seed,
                    iteration,
                    scenario,
                    stage_domain_id,
                    stage_idx,
                );

                assert!(
                    idx < 5,
                    "index {idx} out of bounds (n_openings=5) for \
                     iteration={iteration} scenario={scenario} stage_idx={stage_idx}"
                );
                assert_eq!(
                    slice.len(),
                    2,
                    "slice length must equal dim=2 for \
                     iteration={iteration} scenario={scenario} stage_idx={stage_idx}"
                );
            }
        }
    }
}

/// With identity correlation and 500 openings, the marginal noise distribution
/// approximates N(0,1) (statistical bounds: `|mean| < 0.15` and `|std - 1| < 0.15`).
#[test]
#[allow(clippy::cast_precision_loss)]
fn opening_tree_marginal_statistics() {
    let n_openings = 500_usize;

    let hydros = vec![make_hydro(1), make_hydro(2)];
    let stages = vec![
        make_stage(0, -1, n_openings),
        make_stage(1, 0, n_openings),
        make_stage(2, 1, n_openings),
        make_stage(3, 2, n_openings),
    ];
    let inflow_models = vec![
        make_inflow_model(1, -1, 100.0, 30.0, vec![], 1.0),
        make_inflow_model(1, 0, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(1, 1, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(1, 2, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(2, -1, 200.0, 40.0, vec![], 1.0),
        make_inflow_model(2, 0, 200.0, 40.0, vec![0.4], 0.917),
        make_inflow_model(2, 1, 200.0, 40.0, vec![0.4], 0.917),
        make_inflow_model(2, 2, 200.0, 40.0, vec![0.4], 0.917),
    ];

    let system = SystemBuilder::new()
        .buses(vec![make_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(identity_correlation(&[1, 2]))
        .build()
        .expect("system build must succeed for marginal statistics test");

    let ctx = build_stochastic_context(&system, 42, &[], &[], None)
        .expect("build_stochastic_context must succeed for marginal statistics test");

    let tree = ctx.opening_tree();

    for stage in 0..tree.n_stages() {
        for dim_idx in 0..tree.dim() {
            let values: Vec<f64> = (0..tree.n_openings(stage))
                .map(|o| tree.opening(stage, o)[dim_idx])
                .collect();

            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let std = variance.sqrt();

            assert!(
                mean.abs() < 0.15,
                "stage={stage} dim={dim_idx}: mean {mean:.4} too far from 0 \
                 (expected N(0,1), |mean| < 0.15)"
            );
            assert!(
                (std - 1.0).abs() < 0.15,
                "stage={stage} dim={dim_idx}: std {std:.4} too far from 1 \
                 (expected N(0,1), |std - 1| < 0.15)"
            );
        }
    }
}
