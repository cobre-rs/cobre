//! Pins the out-of-sample forward draw to zero heap allocation, once its
//! per-iteration noise tables are built and its buffers have warmed up, for
//! the Sobol, Halton and Latin-hypercube point methods and for a correlation
//! group wider than `apply_groups_for_class`'s stack-allocated fast path.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use cobre_core::{
    DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    },
    temporal::{Block, NoiseMethod, ScenarioSourceConfig, Stage},
};
use cobre_stochastic::{
    ForwardNoiseTables, ForwardSamplerConfig, SampleRequest, build_forward_sampler,
    context::{ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context},
};

mod common;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method is a verbatim forward to `System`, which upholds
// `GlobalAlloc`'s contract on its own; this wrapper only adds a counter
// increment around the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` are forwarded unchanged to `System::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr`/`layout`/`new_size` are forwarded unchanged to
        // `System::realloc`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

fn reset_alloc_count() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

fn make_bus_spec(id: i32) -> BusSpec {
    BusSpec {
        name: format!("Bus{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        ..Default::default()
    }
}

fn make_hydro_spec(id: i32) -> HydroSpec {
    HydroSpec {
        name: format!("H{id}"),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 100.0,
        max_generation_mw: 100.0,
        penalties: HydroPenalties {
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
        },
        ..Default::default()
    }
}

fn make_stage_spec(bf: usize, method: NoiseMethod) -> StageSpec {
    StageSpec {
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "SINGLE".to_string(),
            duration_hours: 744.0,
        }],
        scenario_config: ScenarioSourceConfig {
            branching_factor: bf,
            noise_method: method,
        },
        ..Default::default()
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
        annual: None,
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

fn build_test_system(
    methods: [NoiseMethod; 3],
    correlation: CorrelationModel,
) -> cobre_core::System {
    let hydros: Vec<Hydro> = (1..=70)
        .map(|id| make_hydro(EntityId(id), make_hydro_spec(id)))
        .collect();
    let stages = vec![
        make_stage(0, make_stage_spec(5, methods[0])),
        make_stage(1, make_stage_spec(5, methods[1])),
        make_stage(2, make_stage_spec(5, methods[2])),
    ];
    let inflow_models: Vec<InflowModel> = (1..=70)
        .flat_map(|hydro_id| (0..3).map(move |stage_id| make_inflow_model(hydro_id, stage_id)))
        .collect();
    SystemBuilder::new()
        .buses(vec![make_bus(EntityId(0), make_bus_spec(0))])
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

fn make_sampler_config<'a>(
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

#[test]
fn out_of_sample_forward_draw_allocates_nothing() {
    let entity_ids: Vec<i32> = (1..=70).collect();
    let system = build_test_system(
        [
            NoiseMethod::QmcSobol,
            NoiseMethod::QmcHalton,
            NoiseMethod::Lhs,
        ],
        correlated_correlation(&entity_ids, 0.4),
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
    assert_eq!(
        dim, 70,
        "fixture must supply exactly 70 correlated entities to exercise \
         apply_groups_for_class's scratch (non-stack) path"
    );

    let total_scenarios: u32 = 8;
    let mut tables = ForwardNoiseTables::default();
    sampler
        .rebuild_noise_tables(0, total_scenarios, &[], &mut tables)
        .expect("70-dim Sobol stays within the Sobol dimension cap");

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];

    let mut sample_stage = |stage_idx: usize, scenario: u32| {
        let stage_u32 = u32::try_from(stage_idx).unwrap();
        sampler
            .sample(SampleRequest {
                iteration: 0,
                scenario,
                stage: stage_u32,
                stage_idx,
                noise_buf: &mut noise_buf,
                corr_scratch: &mut corr_scratch,
                total_scenarios,
                noise_group_id: stage_u32,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
    };

    // Warm up every stage once so any first-draw buffer growth happens before
    // the counter resets; a warm-up allocation is one-time, not per draw.
    for stage_idx in 0..3_usize {
        sample_stage(stage_idx, 0);
    }

    reset_alloc_count();

    for stage_idx in 0..3_usize {
        for scenario in 0..total_scenarios {
            sample_stage(stage_idx, scenario);
        }
    }

    let observed = alloc_count();
    assert_eq!(
        observed, 0,
        "out-of-sample forward draw allocated {observed} times across the measured draws"
    );
}
