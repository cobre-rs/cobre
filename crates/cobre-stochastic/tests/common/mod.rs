//! Shared fixtures for the `cobre-stochastic` integration-test suite:
//! correlation-model builders, entity-order dimensions, and preset
//! bus/hydro/inflow-model/stage constructors built on top of
//! `cobre_core::test_support` and `cobre_stochastic::test_support`.

#![allow(dead_code, unused_imports)]
// Each `tests/*.rs` binary compiles this module separately and uses only the
// subset of items it needs, so an item or re-export unused by one binary is
// not dead code.

use std::collections::BTreeMap;

pub use cobre_core::test_support::{StageSpec, make_stage, norm_cdf, single_block};
use cobre_core::{
    Bus, DeficitSegment, EntityId, Hydro, InflowModel, NoiseMethod, SamplingScheme,
    ScenarioSourceConfig, Stage, SystemBuilder,
    scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
    test_support::{BusSpec, HydroSpec, make_bus, make_hydro},
};
pub use cobre_stochastic::test_support::{InflowModelSpec, make_inflow_model};
use cobre_stochastic::{
    ClassDimensions, ClassSchemes, ForwardNoiseTables, ForwardSampler, ForwardSamplerConfig,
    OpeningTreeInputs, StochasticContext, build_stochastic_context,
    correlation::resolve::DecomposedCorrelation,
};

/// The shared [`CorrelationModel`]: one `"default"` profile with a single group `"g1"` of
/// inflow entities whose off-diagonal correlation is `rho`.
#[must_use]
pub fn correlated_correlation_model(entity_ids: &[i32], rho: f64) -> CorrelationModel {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
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
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn inflow_entity_order_and_dims(entity_ids: &[i32]) -> (Vec<EntityId>, ClassDimensions) {
    (
        entity_ids.iter().map(|&id| EntityId(id)).collect(),
        ClassDimensions {
            n_hydros: entity_ids.len(),
            n_load_buses: 0,
            n_ncs: 0,
        },
    )
}

pub fn identity_correlation_model(entity_ids: &[i32]) -> CorrelationModel {
    correlated_correlation_model(entity_ids, 0.0)
}

pub fn identity_correlation(entity_ids: &[i32]) -> DecomposedCorrelation {
    let (entity_order, dims) = inflow_entity_order_and_dims(entity_ids);
    DecomposedCorrelation::build(&identity_correlation_model(entity_ids), &entity_order, dims)
        .unwrap()
}

pub fn correlated_correlation(entity_ids: &[i32], rho: f64) -> DecomposedCorrelation {
    let (entity_order, dims) = inflow_entity_order_and_dims(entity_ids);
    DecomposedCorrelation::build(
        &correlated_correlation_model(entity_ids, rho),
        &entity_order,
        dims,
    )
    .unwrap()
}

/// The shared [`Bus`]: one uncapped deficit segment at `1000.0`/`MWh`, zero excess cost.
#[must_use]
pub fn deficit_bus(id: i32) -> Bus {
    make_bus(BusSpec {
        id,
        name: format!("Bus{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        ..Default::default()
    })
}

/// The shared [`Hydro`]: `100.0` hm³ storage, `100.0` m³/s turbined capacity, `100.0` `MW`.
#[must_use]
pub fn sized_hydro(id: i32) -> Hydro {
    make_hydro(HydroSpec {
        id,
        name: format!("H{id}"),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 100.0,
        max_generation_mw: 100.0,
        ..Default::default()
    })
}

/// The shared [`InflowModel`]: mean `100.0` m³/s, std `30.0` m³/s, no AR lags or annual term.
#[must_use]
pub fn default_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
    make_inflow_model(InflowModelSpec {
        hydro_id,
        stage_id,
        ..Default::default()
    })
}

/// The shared SAA [`Stage`]: one `"SINGLE"` 744-hour block, season `0`.
#[must_use]
pub fn saa_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
    method_stage(index, id, branching_factor, NoiseMethod::Saa)
}

/// The forward-sampler stage preset: one `"SINGLE"` 744-hour block, season 0,
/// with the per-stage noise method supplied by the caller.
#[must_use]
pub fn method_stage(
    index: usize,
    id: i32,
    branching_factor: usize,
    noise_method: NoiseMethod,
) -> Stage {
    make_stage(StageSpec {
        id,
        index: Some(index),
        season_id: Some(0),
        blocks: single_block("SINGLE", 744.0),
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method,
        },
        ..Default::default()
    })
}

/// Build a [`ForwardSamplerConfig`] over `stages` and `ctx`, with `scheme`
/// set on every noise class.
#[must_use]
pub fn make_sampler_config<'a>(
    scheme: SamplingScheme,
    ctx: &'a StochasticContext,
    stages: &'a [Stage],
) -> ForwardSamplerConfig<'a> {
    ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(scheme),
            load: Some(scheme),
            ncs: Some(scheme),
        },
        ctx,
        forward_seed: ctx.forward_seed(),
        stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
    }
}

/// The two-hydro, three-stage forward-sampler fixture system for `methods`
/// (one per stage) and `correlation`.
#[must_use]
pub fn build_test_system(
    methods: &[NoiseMethod],
    correlation: CorrelationModel,
) -> cobre_core::System {
    assert_eq!(methods.len(), 3, "must supply exactly 3 per-stage methods");
    let hydros = vec![sized_hydro(1), sized_hydro(2)];
    let stages = vec![
        method_stage(0, 0, 5, methods[0]),
        method_stage(1, 1, 5, methods[1]),
        method_stage(2, 2, 5, methods[2]),
    ];
    let inflow_models = vec![
        default_inflow_model(1, 0),
        default_inflow_model(1, 1),
        default_inflow_model(1, 2),
        default_inflow_model(2, 0),
        default_inflow_model(2, 1),
        default_inflow_model(2, 2),
    ];
    SystemBuilder::new()
        .buses(vec![deficit_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap()
}

/// The in-sample [`StochasticContext`] for `system`, base seed `42`.
#[must_use]
pub fn build_test_ctx(system: &cobre_core::System, forward_seed: Option<u64>) -> StochasticContext {
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

/// The study stages of `system` (excludes the pre-study `id < 0` stage).
#[must_use]
pub fn stages_from_system(system: &cobre_core::System) -> Vec<Stage> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect()
}

/// Rebuild `sampler`'s noise tables for one `(iteration, total, groups)`
/// triple — the `SampleRequest.tables` every `sample()` call in
/// `forward_sampler.rs` and `forward_sampler_golden.rs` needs.
#[must_use]
pub fn tables_for(
    sampler: &ForwardSampler<'_>,
    iteration: u32,
    total: u32,
    groups: &[u32],
) -> ForwardNoiseTables {
    let mut tables = ForwardNoiseTables::default();
    sampler
        .rebuild_noise_tables(iteration, total, groups, &mut tables)
        .expect("test fixtures never exceed the Sobol dimension cap");
    tables
}
