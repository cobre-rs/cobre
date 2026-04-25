//! Integration tests for the [`ForwardSampler`] abstraction.
//!
//! Covers dispatch correctness, `InSample` copy equivalence, `OutOfSample`
//! determinism, `OutOfSample` correlation correctness, factory error paths,
//! and resume invariance.
//!
//! This is the quality-gate test suite for the [`ForwardSampler`] abstraction.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_core::{
    Bus, DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_stochastic::{
    StochasticError,
    context::{ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context},
    sampling::insample::sample_forward,
    sampling::{ForwardSamplerConfig, SampleRequest, build_forward_sampler},
    tree::generate::ClassDimensions,
};

fn make_sampler_config<'a>(
    scheme: SamplingScheme,
    ctx: &'a StochasticContext,
    stages: &'a [Stage],
) -> ForwardSamplerConfig<'a> {
    let dim = ctx.dim();
    ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(scheme),
            load: Some(scheme),
            ncs: Some(scheme),
        },
        ctx,
        stages,
        dims: ClassDimensions {
            n_hydros: dim,
            n_load_buses: 0,
            n_ncs: 0,
        },
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
    }
}

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
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    }
}

fn make_stage(index: usize, id: i32, bf: usize, method: NoiseMethod) -> Stage {
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
            branching_factor: bf,
            noise_method: method,
        },
    }
}

fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
    InflowModel {
        hydro_id: EntityId(hydro_id),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
    }
}

fn identity_correlation(ids: &[i32]) -> CorrelationModel {
    let n = ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: ids
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
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn correlated_correlation(ids: &[i32], rho: f64) -> CorrelationModel {
    let n = ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: ids
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
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn build_test_system(methods: &[NoiseMethod], correlation: CorrelationModel) -> cobre_core::System {
    assert_eq!(methods.len(), 3, "must supply exactly 3 per-stage methods");
    let hydros = vec![make_hydro(1), make_hydro(2)];
    let stages = vec![
        make_stage(0, 0, 5, methods[0]),
        make_stage(1, 1, 5, methods[1]),
        make_stage(2, 2, 5, methods[2]),
    ];
    let inflow_models = vec![
        make_inflow_model(1, 0),
        make_inflow_model(1, 1),
        make_inflow_model(1, 2),
        make_inflow_model(2, 0),
        make_inflow_model(2, 1),
        make_inflow_model(2, 2),
    ];
    SystemBuilder::new()
        .buses(vec![make_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap()
}

fn build_test_ctx(system: &cobre_core::System, forward_seed: Option<u64>) -> StochasticContext {
    build_stochastic_context(
        system,
        42,
        forward_seed,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap()
}

fn stages_from_system(system: &cobre_core::System) -> Vec<Stage> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect()
}

#[test]
fn insample_dispatch_returns_tree_slice_of_correct_dim() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);
    let sampler =
        build_forward_sampler(make_sampler_config(SamplingScheme::InSample, &ctx, &stages))
            .unwrap();
    let dim = ctx.dim();

    let mut noise_buf = vec![0.0f64; dim];
    let mut perm_scratch = vec![0usize; 5];

    let result = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    let slice = result.as_slice();
    assert_eq!(
        slice.len(),
        dim,
        "noise slice length {} != dim {}",
        slice.len(),
        dim
    );
}

#[test]
fn insample_copy_equivalence_matches_direct_call() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);
    let sampler =
        build_forward_sampler(make_sampler_config(SamplingScheme::InSample, &ctx, &stages))
            .unwrap();
    let dim = ctx.dim();

    let mut noise_buf = vec![0.0f64; dim];
    let mut perm_scratch = vec![0usize; 5];

    let result = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    let tree_view = ctx.tree_view();
    let (_direct_idx, direct_slice) = sample_forward(&tree_view, ctx.base_seed(), 0, 0, 0, 0);

    assert_eq!(
        result.as_slice(),
        direct_slice,
        "ForwardSampler::InSample and direct sample_forward must return bitwise-identical slices"
    );
}

#[test]
fn out_of_sample_dispatch_returns_fresh_noise_of_correct_dim() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut noise_buf = vec![0.0f64; dim];
    let mut perm_scratch = vec![0usize; 5];

    let result = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    let slice = result.as_slice();
    assert_eq!(
        slice.len(),
        dim,
        "fresh noise slice length {} != dim {}",
        slice.len(),
        dim
    );
}

#[test]
fn out_of_sample_is_deterministic() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut buf_a = vec![0.0f64; dim];
    let mut buf_b = vec![0.0f64; dim];
    let mut perm_a = vec![0usize; 5];
    let mut perm_b = vec![0usize; 5];

    let a = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_a,
            perm_scratch: &mut perm_a,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    let b = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_b,
            perm_scratch: &mut perm_b,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    assert_eq!(
        a.as_slice(),
        b.as_slice(),
        "identical OutOfSample calls must produce bitwise-identical noise"
    );
}

#[test]
fn out_of_sample_scenario_changes_noise() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut buf_0 = vec![0.0f64; dim];
    let mut buf_1 = vec![0.0f64; dim];
    let mut perm_0 = vec![0usize; 5];
    let mut perm_1 = vec![0usize; 5];

    let result_0 = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_0,
            perm_scratch: &mut perm_0,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    let result_1 = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 1,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_1,
            perm_scratch: &mut perm_1,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    let any_differ = result_0
        .as_slice()
        .iter()
        .zip(result_1.as_slice())
        .any(|(a, b)| a != b);

    assert!(
        any_differ,
        "noise for scenario=0 and scenario=1 must differ in at least one element"
    );
}

#[test]
fn out_of_sample_noise_is_finite() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    let total_scenarios: u32 = 100;

    let mut noise_buf = vec![0.0f64; dim];
    let mut perm_scratch = vec![0usize; total_scenarios as usize];

    for scenario in 0..total_scenarios {
        let result = sampler
            .sample(SampleRequest {
                iteration: 0,
                scenario,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut noise_buf,
                perm_scratch: &mut perm_scratch,
                total_scenarios,
                noise_group_id: 0,
            })
            .unwrap();

        for (i, &v) in result.as_slice().iter().enumerate() {
            assert!(
                v.is_finite(),
                "scenario={scenario} element[{i}] is not finite: {v}"
            );
        }
    }
}

#[test]
fn out_of_sample_correlation_matches_target() {
    let rho = 0.8_f64;
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        correlated_correlation(&[1, 2], rho),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    assert_eq!(dim, 2, "expected dim=2 for 2 hydros");

    let n_scenarios: u32 = 2000;
    let mut noise_buf = vec![0.0f64; dim];
    let mut perm_scratch = vec![0usize; n_scenarios as usize];

    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n_scenarios as usize);

    for scenario in 0..n_scenarios {
        let result = sampler
            .sample(SampleRequest {
                iteration: 0,
                scenario,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut noise_buf,
                perm_scratch: &mut perm_scratch,
                total_scenarios: n_scenarios,
                noise_group_id: 0,
            })
            .unwrap();

        let s = result.as_slice();
        pairs.push((s[0], s[1]));
    }

    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;

    let cov_xy = pairs
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / (n - 1.0);
    let var_x = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>() / (n - 1.0);
    let var_y = pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / (n - 1.0);

    let sample_corr = cov_xy / (var_x.sqrt() * var_y.sqrt());

    assert!(
        (sample_corr - rho).abs() < 0.15,
        "sample correlation {sample_corr:.4} not within 0.15 of target {rho}"
    );
}

#[test]
fn out_of_sample_per_stage_method_mixing() {
    let system = build_test_system(
        &[NoiseMethod::Lhs, NoiseMethod::Saa, NoiseMethod::QmcHalton],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    let total_scenarios: u32 = 10;

    let mut noise_buf = vec![0.0f64; dim];
    // LHS requires perm_scratch of length total_scenarios.
    let mut perm_scratch = vec![0usize; total_scenarios as usize];

    for stage_idx in 0..3_usize {
        let stage_id = stage_idx as u32;
        for scenario in 0..total_scenarios {
            let result = sampler
                .sample(SampleRequest {
                    iteration: 0,
                    scenario,
                    stage: stage_id,
                    stage_idx,
                    noise_buf: &mut noise_buf,
                    perm_scratch: &mut perm_scratch,
                    total_scenarios,
                    noise_group_id: 0,
                })
                .unwrap();

            let slice = result.as_slice();
            assert_eq!(
                slice.len(),
                dim,
                "stage_idx={stage_idx} scenario={scenario}: expected dim={dim}, got {}",
                slice.len()
            );
            for (i, &v) in slice.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "stage_idx={stage_idx} scenario={scenario} element[{i}] is not finite: {v}"
                );
            }
        }
    }
}

#[test]
fn factory_rejects_out_of_sample_without_seed() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None); // forward_seed = None
    let stages = stages_from_system(&system);

    let result = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ));

    match result {
        Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
            assert!(
                scheme.contains("out_of_sample"),
                "expected scheme to contain 'out_of_sample', got: {scheme}"
            );
        }
        other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
    }
}

#[test]
fn factory_rejects_historical() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);

    let result = build_forward_sampler(make_sampler_config(
        SamplingScheme::Historical,
        &ctx,
        &stages,
    ));

    match result {
        Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
            assert!(
                scheme.contains("historical"),
                "expected scheme to contain 'historical', got: {scheme}"
            );
        }
        other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
    }
}

#[test]
fn factory_rejects_external() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);

    let result =
        build_forward_sampler(make_sampler_config(SamplingScheme::External, &ctx, &stages));

    match result {
        Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
            assert!(
                scheme.contains("external"),
                "expected scheme to contain 'external', got: {scheme}"
            );
        }
        other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
    }
}

#[test]
fn out_of_sample_resume_invariance() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut buf_first = vec![0.0f64; dim];
    let mut buf_resume = vec![0.0f64; dim];
    let mut perm_first = vec![0usize; 5];
    let mut perm_resume = vec![0usize; 5];

    // First call — simulates a running solver at iteration=5, scenario=3.
    let first = sampler
        .sample(SampleRequest {
            iteration: 5,
            scenario: 3,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_first,
            perm_scratch: &mut perm_first,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();
    let first_values: Vec<f64> = first.as_slice().to_vec();

    // Second call — simulates a resumed solver: same arguments, no prior state.
    let resumed = sampler
        .sample(SampleRequest {
            iteration: 5,
            scenario: 3,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_resume,
            perm_scratch: &mut perm_resume,
            total_scenarios: 5,
            noise_group_id: 0,
        })
        .unwrap();

    assert_eq!(
        first_values.as_slice(),
        resumed.as_slice(),
        "OutOfSample noise must be identical for a resumed call with the same (iteration, scenario, stage)"
    );
}
