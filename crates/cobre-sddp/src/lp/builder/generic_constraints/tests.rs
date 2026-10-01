#![expect(
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::identity_op,
    clippy::erasing_op,
    reason = "test docs name LP symbols that are not code identifiers, the equivalence check takes each varied input explicitly, and offsets spell the full base-plus-stride formula even when a term is zero"
)]

use std::collections::HashMap;
use std::ops::Range;

use super::super::layout::variable_ref_is_block_independent;
use super::{contract_family_slot, resolve_variable_ref};
use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, FphaPlane, ProductionModelSet, ResolvedProductionModel,
};
use crate::lp::builder::{StageLayout, TemplateBuildCtx};
use crate::lp::indexer::{
    AnticipatedPlants, Boundary, EntityPositions, HydroCell, HydroCellIndex, HydroSys,
};
use crate::test_support::ctx_fixture::CtxFixture;
use crate::test_support::{
    geometry_hydro, geometry_hydro_with_groups, make_unit_group, minimal_hydros,
};
use crate::time_value::{PostStudyResolved, TimeValue};
use cobre_core::entities::{HydroGenerationModel, HydroPenalties};
use cobre_core::{
    AnticipatedConfig, Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, CascadeTopology,
    ContractBlockBounds, ContractType, DeficitSegment, EnergyContract, EntityId, Hydro,
    HydroBlockBounds, HydroStageBounds, Line, LineBlockBounds, NoiseMethod, PumpingBlockBounds,
    PumpingStation, ResolvedBounds, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
    Thermal, ThermalBlockBounds, ThermalStageBounds, VariableRef,
};

// ── Test helpers ──────────────────────────────────────────────────────────

/// A single-stage `Stage` fixture: `n_blks` parallel-mode blocks. Every
/// resolver test in this file uses this [`BlockMode::Parallel`] shape except
/// the chronological storage-boundary tests, which flip `ResolverFixture`'s
/// `stage.block_mode` after construction (see `chronological_default_fixture`)
/// to give the interior-boundary family a genuinely non-empty column range.
fn fixture_stage(n_blks: usize) -> Stage {
    Stage {
        index: 0,
        id: 0,
        start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: (0..n_blks)
            .map(|index| Block {
                index,
                name: format!("BLK{index}"),
                duration_hours: 744.0,
            })
            .collect(),
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

/// Owns every borrow target for a resolver test's `TemplateBuildCtx`/
/// `StageLayout`, through a [`CtxFixture`] plus the `stage` value
/// `StageLayout::new` also needs. `max_par_order` is fixed at `0`: no
/// resolver test in this file needs a nonzero PAR lag order.
struct ResolverFixture {
    base: CtxFixture,
    stage: Stage,
}

impl ResolverFixture {
    /// `hydros`/`thermals`/`lines`/`buses`/`pumping_stations`/`contracts` must
    /// already be in id-ascending (canonical) order — `ctx()` derives every
    /// position map by enumeration, mirroring `System`'s own canonical sort.
    /// `anticipated_lead_stages` is anticipated-local (length must equal the
    /// number of `thermals` carrying an `anticipated_config`).
    fn new(
        hydros: Vec<Hydro>,
        thermals: Vec<Thermal>,
        lines: Vec<Line>,
        buses: Vec<Bus>,
        n_blks: usize,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        anticipated_lead_stages: Vec<usize>,
        pumping_stations: Vec<PumpingStation>,
        contracts: Vec<EnergyContract>,
    ) -> Self {
        let hydro_cell_index = HydroCellIndex::build(&hydros);
        let cascade = CascadeTopology::build(&hydros);
        let anticipated_plants = AnticipatedPlants::build(&thermals);
        let n_anticipated = anticipated_lead_stages.len();
        debug_assert_eq!(anticipated_plants.len(), n_anticipated);
        let n_stages = anticipated_lead_stages.iter().copied().max().unwrap_or(0) + 2;
        let stage = fixture_stage(n_blks);
        // Delivery axis wide enough to cover stage 0 + the widest declared
        // lead — otherwise a genuinely-reachable delivery stage indexes past
        // CtxFixture::default's 1-long bounds/time_value axis.
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let time_value = TimeValue::from_parts(
            Vec::new(),
            vec![1.0; n_stages],
            vec![744.0; n_stages],
            (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
            PostStudyResolved::default(),
        );
        Self {
            base: CtxFixture {
                hydros,
                thermals,
                lines,
                buses,
                cascade,
                hydro_cell_index,
                bounds,
                production_models,
                evaporation_models,
                pumping_stations,
                contracts,
                anticipated_lead_stages,
                anticipated_plants,
                time_value,
                ..CtxFixture::default()
            },
            stage,
        }
    }
}

/// A `Thermal` carrying only the `id`/`anticipated_config` the resolver and
/// `StageLayout::new` read; every other field is an inert value.
fn make_thermal(id: i32, anticipated_config: Option<AnticipatedConfig>) -> Thermal {
    Thermal {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 0.0,
        min_generation_mw: 0.0,
        max_generation_mw: 0.0,
        anticipated_config,
    }
}

/// A `Line` carrying only the `id`; every other field is an inert value.
fn make_line(id: i32) -> Line {
    Line {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        source_bus_id: EntityId(0),
        target_bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        direct_capacity_mw: 0.0,
        reverse_capacity_mw: 0.0,
        losses_percent: 0.0,
        exchange_cost: 0.0,
    }
}

/// A `Bus` carrying the `id` and `max_deficit_segments` deficit segments —
/// `StageLayout::new` derives `max_deficit_segments` as
/// `ctx.buses.iter().map(|b| b.deficit_segments.len()).max()`, so every bus
/// must carry the caller's count for that derivation to reproduce it.
fn make_bus(id: i32, max_deficit_segments: usize) -> Bus {
    Bus {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![
            DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 0.0,
            };
            max_deficit_segments
        ],
        excess_cost: 0.0,
    }
}

/// A single-stage [`ProductionModelSet`] where every one of `n_hydros` hydros
/// is `ConstantProductivity { productivity }`.
fn constant_productivity_models(n_hydros: usize, productivity: f64) -> ProductionModelSet {
    ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::ConstantProductivity { productivity }]; n_hydros],
        &minimal_hydros(n_hydros),
        1,
    )
}

/// N=4 hydros (2 FPHA at positions 0, 2), L=0, T=2 thermals, Ln=1 line, B=2
/// buses (S=2 max deficit segments each), K=3 blocks, parallel mode — the
/// shape most resolver tests in this file share.
///
/// Column layout (identical to a real `StageLayout::new` build of these
/// dims, since that is exactly how this fixture is built):
///   storage:   [0, 4)         = 0..4
///   lags:      [4, 4*(1+0))   = 4..4   (L=0, empty)
///   z_inflow:  [4*(1+0), 4*(2+0)) = 4..8
///   storage_in:[4*(2+0), 4*(3+0)) = 8..12
///   theta = N*(3+L) = 4*(3+0) = 12
///   decision_start = 13
///   turbine:    [13, 13+4*3) = 13..25   (4 hydros * 3 blocks)
///   spillage:   [25, 25+4*3) = 25..37
///   diversion:  [37, 37+4*3) = 37..49  (4 hydros * 3 blocks)
///   thermal:    [49, 49+2*3) = 49..55  (2 thermals * 3 blocks)
///   line_fwd:   [55, 55+1*3) = 55..58  (1 line * 3 blocks)
///   line_rev:   [58, 58+1*3) = 58..61
///   deficit:    [61, 61+2*2*3) = 61..73 (2 buses * 2 segs * 3 blocks)
///   excess:     [73, 73+2*3) = 73..79  (2 buses * 3 blocks)
///   generation: [79, 79+2*3) = 79..85  (2 FPHA hydros * 3 blocks)
///   evap: none
///   withdrawal_slack_neg: [85, 89)  withdrawal_slack_pos: [89, 93) (4 hydros)
///
/// Later families (slacks, pumping, contracts) are read through the layout
/// accessors, not hand-derived.
fn default_fixture() -> ResolverFixture {
    let hydros = vec![
        make_hydro(10, None),
        make_hydro(20, None),
        make_hydro(30, None),
        make_hydro(40, None),
    ];
    let thermals = vec![make_thermal(5, None), make_thermal(6, None)];
    let lines = vec![make_line(50)];
    let buses = vec![make_bus(100, 2), make_bus(200, 2)];
    let pumping_stations = vec![
        make_pumping_station(10, 2.5, EntityId(10), EntityId(20)),
        make_pumping_station(20, 0.75, EntityId(30), EntityId(40)),
    ];
    let contracts = vec![
        make_contract(10, ContractType::Import),
        make_contract(30, ContractType::Import),
        make_contract(20, ContractType::Export),
    ];
    ResolverFixture::new(
        hydros,
        thermals,
        lines,
        buses,
        3,
        make_production_models(),
        EvaporationModelSet::new(vec![EvaporationModel::None; 4]),
        vec![],
        pumping_stations,
        contracts,
    )
}

/// N=2 hydros (hydro 10 evaporating, hydro 20 not), B=1 bus, `n_blks` blocks,
/// parallel mode.
fn evaporation_fixture(n_blks: usize) -> ResolverFixture {
    let hydros = vec![make_hydro(10, None), make_hydro(20, None)];
    let evaporation_models = EvaporationModelSet::new(vec![
        EvaporationModel::Linearized {
            coefficients: Vec::new(),
            reference_volumes_hm3: Vec::new(),
        },
        EvaporationModel::None,
    ]);
    ResolverFixture::new(
        hydros,
        vec![],
        vec![],
        vec![make_bus(100, 1)],
        n_blks,
        constant_productivity_models(2, 1.0),
        evaporation_models,
        vec![],
        vec![],
        vec![],
    )
}

/// N=0 hydros, T=2 thermals (thermal 5 regular, thermal 6 anticipated with
/// `lead_stages = 2`), B=1 bus, K=2 blocks.
fn anticipated_fixture() -> ResolverFixture {
    let thermals = vec![
        make_thermal(5, None),
        make_thermal(6, Some(AnticipatedConfig::LeadStages(2))),
    ];
    ResolverFixture::new(
        vec![],
        thermals,
        vec![],
        vec![make_bus(100, 1)],
        2,
        ProductionModelSet::new(vec![], &[], 1),
        EvaporationModelSet::new(vec![]),
        vec![2],
        vec![],
        vec![],
    )
}

/// Minimal `Hydro` carrying only the `id`/`downstream_id` that
/// [`CascadeTopology::build`] reads; every other field is an inert default.
fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
    let zero_penalties = HydroPenalties {
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
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: downstream_id.map(EntityId),
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 1.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 1.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_penalties,
    };
    hydro.declare_mirror_unit_group(EntityId(0));
    hydro
}

fn make_production_models() -> ProductionModelSet {
    let fpha_plane = FphaPlane {
        intercept: 0.0,
        gamma_v: 0.1,
        gamma_q: 0.5,
        gamma_s: 0.0,
    };
    let fpha_model = || ResolvedProductionModel::Fpha {
        planes: vec![fpha_plane],
    };
    let models: Vec<Vec<ResolvedProductionModel>> = vec![
        vec![fpha_model(), fpha_model()],
        vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
            ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
        ],
        vec![fpha_model(), fpha_model()],
        vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
        ],
    ];
    ProductionModelSet::new(models, &minimal_hydros(4), 2)
}

/// Resolve `var_ref` at `block_idx` (stage 0) against `ctx`/`layout` — the same
/// pair `resolve_variable_ref` reads in production. A test that needs a cascade,
/// pumping/contract set, or production-model set different from the fixture's own
/// overrides the relevant `ctx` field with struct-update syntax before calling.
fn call(
    var_ref: VariableRef,
    block_idx: usize,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_variable_ref(&var_ref, block_idx, 0, ctx, layout)
}

/// An energy contract carrying only the `id`/`bus_id`/`contract_type` the
/// resolver and load-balance fill read; every other field is an inert value.
fn make_contract(id: i32, contract_type: ContractType) -> EnergyContract {
    EnergyContract {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        contract_type,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh: 0.0,
        min_mw: 0.0,
        max_mw: 1.0,
    }
}

/// A pumping station carrying a `consumption_mw_per_m3s` rate and its
/// source/destination hydro references (naming fixture hydros); every other
/// field is an inert value the resolver does not read.
fn make_pumping_station(
    id: i32,
    consumption_mw_per_m3s: f64,
    source_hydro_id: EntityId,
    destination_hydro_id: EntityId,
) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        source_hydro_id,
        destination_hydro_id,
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s,
        min_flow_m3s: 0.0,
        max_flow_m3s: 1.0,
    }
}

// ── ThermalGeneration tests ───────────────────────────────────────────────

/// `ThermalGeneration` column arithmetic across the `block_id`/position axes
/// the per-arm coverage requires: one `block_id = None`, one `block_id = Some`,
/// and one `position != 0`. All resolve through `resolve_thermal_generation`
/// with `layout.geometry.thermal.start = 49`, `n_blks = 3`.
#[test]
fn thermal_generation_column_arithmetic() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    // (case_name, thermal_id, block_id, block_idx, expected_col)
    let cases: [(&str, EntityId, Option<usize>, usize, usize); 3] = [
        ("none_block_1", EntityId(5), None, 1, 49 + 0 * 3 + 1),
        ("some_block_2", EntityId(5), Some(2), 2, 49 + 0 * 3 + 2),
        ("second_thermal", EntityId(6), None, 0, 49 + 1 * 3 + 0),
    ];

    for (case_name, thermal_id, block_id, block_idx, expected_col) in cases {
        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id,
                block_id,
            },
            block_idx,
            &ctx,
            &layout,
        );
        assert_eq!(
            result,
            vec![(expected_col, 1.0)],
            "thermal_generation case `{case_name}`",
        );
    }
}

// ── HydroStorage tests ────────────────────────────────────────────────────

/// storage.start = 0, hydro_pos[EntityId(10)] = 0 → column 0, regardless of block_idx.
#[test]
fn hydro_storage_stage_level_ignores_block() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    for block_idx in [0, 1, 2] {
        let result = call(
            VariableRef::HydroStorage {
                hydro_id: EntityId(10),
            },
            block_idx,
            &ctx,
            &layout,
        );
        assert_eq!(result, vec![(0, 1.0)], "block_idx={block_idx}");
    }

    let result2 = call(
        VariableRef::HydroStorage {
            hydro_id: EntityId(30),
        },
        0,
        &ctx,
        &layout,
    );
    // storage.start = 0, pos = 2 → column 2
    assert_eq!(result2, vec![(2, 1.0)]);
}

// ── HydroOutflow tests ────────────────────────────────────────────────────

/// hydro_pos[EntityId(40)] = 3, block_idx=0, turbine.start = 13, spillage.start = 25,
/// n_blks = 3 → [(13 + 3*3, 1.0), (25 + 3*3, 1.0)] = [(22, 1.0), (34, 1.0)].
#[test]
fn hydro_outflow_expands_to_turbine_and_spillage() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroOutflow {
            hydro_id: EntityId(40),
            block_id: None,
        },
        0, // block_idx
        &ctx,
        &layout,
    );

    let turbine_col = 13 + 3 * 3 + 0; // 22
    let spillage_col = 25 + 3 * 3 + 0; // 34
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (turbine_col, 1.0));
    assert_eq!(result[1], (spillage_col, 1.0));
}

#[test]
fn hydro_outflow_block_id_some_uses_explicit_block() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroOutflow {
            hydro_id: EntityId(10),
            block_id: Some(1),
        },
        0, // block_idx is irrelevant when block_id = Some
        &ctx,
        &layout,
    );

    // hydro pos=0, turbine.start=13, spillage.start=25, block=1, n_blks=3
    assert_eq!(result, vec![(13 + 0 * 3 + 1, 1.0), (25 + 0 * 3 + 1, 1.0)]);
}

// ── HydroGeneration tests ─────────────────────────────────────────────────

/// hydro_pos[EntityId(20)] = 1 → constant productivity 2.5; turbine.start = 13,
/// n_blks = 3, block_idx = 0 → [(13 + 1*3, 2.5)] = [(16, 2.5)].
#[test]
fn hydro_generation_constant_productivity_maps_to_turbine() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroGeneration {
            hydro_id: EntityId(20),
            block_id: None,
            bus_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(13 + 1 * 3 + 0, 2.5)]);
}

/// hydro_pos[EntityId(10)] = 0 → FPHA (local FPHA index 0); generation.start = 79,
/// n_blks = 3, block_idx = 0 → [(79, 1.0)].
#[test]
fn hydro_generation_fpha_maps_to_generation_column() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroGeneration {
            hydro_id: EntityId(10),
            block_id: None,
            bus_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(79 + 0 * 3 + 0, 1.0)]);
}

/// hydro_pos[EntityId(30)] = 2 → FPHA (local FPHA index 1); generation.start = 79,
/// n_blks = 3, block_idx = 2 → [(79 + 1*3 + 2, 1.0)] = [(84, 1.0)].
#[test]
fn hydro_generation_fpha_second_hydro_block_2() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroGeneration {
            hydro_id: EntityId(30),
            block_id: None,
            bus_id: None,
        },
        2,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(79 + 1 * 3 + 2, 1.0)]);
}

// ── Bus-selector tests ────────────────────────────────────────────────────

/// A padding plant (1 cell, global index 0) plus a two-bus split plant (cells
/// 1 and 2) whose group ids (78, 77) differ from their `unit_groups`
/// positions — the fixture shape both bus-selector resolver tests need to
/// discriminate a cell-bus lookup from a first-cell shortcut.
fn bus_selector_hydros(generation_model: HydroGenerationModel) -> Vec<Hydro> {
    let padding = geometry_hydro(0);
    let split = geometry_hydro_with_groups(
        1,
        vec![
            make_unit_group(EntityId(78), EntityId(10), 0.0, 100.0, 0.0, 50.0),
            make_unit_group(EntityId(77), EntityId(20), 0.0, 200.0, 0.0, 80.0),
        ],
        generation_model,
    );
    vec![padding, split]
}

/// Shared fixture builder for the two `ConstantProductivity` bus-selector
/// tests below ([`resolve_turbine_bus_selector_picks_one_cell`],
/// [`resolve_generation_bus_selector_on_constant_productivity_picks_one_cell`]):
/// 3 cells (1 padding + 2 split), n_blks=5.
fn turbine_bus_selector_fixture() -> ResolverFixture {
    let hydros = bus_selector_hydros(HydroGenerationModel::ConstantProductivity);
    let n = hydros.len();
    ResolverFixture::new(
        hydros,
        vec![],
        vec![],
        vec![],
        5,
        constant_productivity_models(n, 1.0),
        EvaporationModelSet::new(vec![EvaporationModel::None; n]),
        vec![],
        vec![],
        vec![],
    )
}

/// `HydroTurbined{bus_id: Some(b)}` resolves to exactly the requested cell's
/// own turbine column — never the plant's other cell, never both — and
/// `bus_id: None` stays the unchanged one-pair-per-cell sum. The split
/// plant's `Hydro::bus_id` (999) equals neither group's bus (10, 20):
/// resolving against it (instead of `HydroCellIndex::bus_of`) misses and
/// returns empty for `bus_id: Some(20)`; resolving via `first_cell_of`
/// instead of `cell_of_bus` returns the FIRST cell's column instead.
#[test]
fn resolve_turbine_bus_selector_picks_one_cell() {
    let mut fx = turbine_bus_selector_fixture();
    let ctx = fx.base.ctx();
    assert_eq!(ctx.hydro_cell_index.n_cells(), 3);
    assert_eq!(ctx.hydro_cell_index.cells_of(HydroSys::new(1)), 1..3);
    assert_eq!(ctx.hydro_cell_index.bus_of(HydroCell::new(1)), EntityId(10));
    assert_eq!(
        ctx.hydro_cell_index.bus_of(HydroCell::new(2)),
        EntityId(20),
        "cell 2 (ascending bus order) is the SECOND cell under test"
    );
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let n_blks = layout.clock.n_blks();
    let turbine_start = layout.geometry.turbine.start;

    let picked = call(
        VariableRef::HydroTurbined {
            hydro_id: ctx.hydros[1].id,
            block_id: Some(3),
            bus_id: Some(EntityId(20)),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(picked, vec![(turbine_start + 2 * n_blks + 3, 1.0)]);

    let summed = call(
        VariableRef::HydroTurbined {
            hydro_id: ctx.hydros[1].id,
            block_id: Some(3),
            bus_id: None,
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        summed,
        vec![
            (turbine_start + 1 * n_blks + 3, 1.0),
            (turbine_start + 2 * n_blks + 3, 1.0),
        ]
    );
}

/// `HydroGeneration{bus_id: Some(b)}` on a `ConstantProductivity` plant
/// resolves to exactly the requested cell's own turbine column scaled by the
/// plant's `ρ` — never the plant's first cell, and never the sum over cells.
/// Same fixture as [`resolve_turbine_bus_selector_picks_one_cell`] (the split
/// plant's `Hydro::bus_id` (999) equals neither group's bus (10, 20)), but
/// resolved through `resolve_hydro_generation`'s `ConstantProductivity` arm —
/// the combination neither that test (`HydroTurbined`) nor
/// [`resolve_generation_bus_selector_maps_to_the_cells_fpha_column`]
/// (`HydroGeneration` × FPHA) reaches. `productivity != 1.0` distinguishes the
/// scaled result from the raw turbine coefficient.
#[test]
fn resolve_generation_bus_selector_on_constant_productivity_picks_one_cell() {
    let mut fx = turbine_bus_selector_fixture();
    let mut ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let n_blks = layout.clock.n_blks();
    let turbine_start = layout.geometry.turbine.start;

    let productivity = 2.5;
    let prod = ProductionModelSet::new(
        vec![
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            vec![ResolvedProductionModel::ConstantProductivity { productivity }],
        ],
        ctx.hydros,
        1,
    );
    ctx.production_models = &prod;

    let picked = call(
        VariableRef::HydroGeneration {
            hydro_id: ctx.hydros[1].id,
            block_id: Some(3),
            bus_id: Some(EntityId(20)),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(picked, vec![(turbine_start + 2 * n_blks + 3, productivity)]);

    let summed = call(
        VariableRef::HydroGeneration {
            hydro_id: ctx.hydros[1].id,
            block_id: Some(3),
            bus_id: None,
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        summed,
        vec![
            (turbine_start + 1 * n_blks + 3, productivity),
            (turbine_start + 2 * n_blks + 3, productivity),
        ]
    );
}

/// Production models for the FPHA bus-selector fixture: hydro 0
/// (`ConstantProductivity`, padding), hydro 1 (`Fpha`, one zero-coefficient
/// plane).
fn fpha_bus_selector_production_models() -> ProductionModelSet {
    ProductionModelSet::new(
        vec![
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            vec![ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 0.0,
                    gamma_v: 0.0,
                    gamma_q: 0.0,
                    gamma_s: 0.0,
                }],
            }],
        ],
        &minimal_hydros(2),
        1,
    )
}

/// `HydroGeneration{bus_id: Some(b)}` on an FPHA plant maps the selected cell
/// to its OFFSET within the plant's own cell range
/// (`fpha_cell_local_start[fpha_local] + (cell.get() - cells_of(h).start)`),
/// never the cell's absolute index — the leading non-FPHA padding plant makes
/// the two diverge (`cells_of(h).start == 1` while `fpha_cell_local_start[0]
/// == 0`, since the FPHA-local prefix sums only over FPHA plants).
/// `bus_id: None` stays the unchanged one-pair-per-cell resolution.
#[test]
fn resolve_generation_bus_selector_maps_to_the_cells_fpha_column() {
    let hydros = bus_selector_hydros(HydroGenerationModel::Fpha);
    let mut fx = ResolverFixture::new(
        hydros,
        vec![],
        vec![],
        vec![],
        4,
        fpha_bus_selector_production_models(),
        EvaporationModelSet::new(vec![EvaporationModel::None; 2]),
        vec![],
        vec![],
        vec![],
    );
    let ctx = fx.base.ctx();
    assert_eq!(ctx.hydro_cell_index.cells_of(HydroSys::new(1)), 1..3);
    assert_eq!(ctx.hydro_cell_index.bus_of(HydroCell::new(2)), EntityId(20));

    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let n_blks = layout.clock.n_blks();
    let generation_start = layout.geometry.generation.start;

    let picked = call(
        VariableRef::HydroGeneration {
            hydro_id: ctx.hydros[1].id,
            block_id: Some(3),
            bus_id: Some(EntityId(20)),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(picked, vec![(generation_start + 1 * n_blks + 3, 1.0)]);

    let summed = call(
        VariableRef::HydroGeneration {
            hydro_id: ctx.hydros[1].id,
            block_id: Some(3),
            bus_id: None,
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        summed,
        vec![
            (generation_start + 0 * n_blks + 3, 1.0),
            (generation_start + 1 * n_blks + 3, 1.0),
        ]
    );
}

// ── HydroEvaporation tests ────────────────────────────────────────────────

/// Dedicated evap-hydro geom (evap hydro at pos 0):
/// N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA.
/// theta = 2*(3+0) = 6
/// turbine:    [7, 9)
/// spillage:   [9, 11)
/// diversion: [11, 13)
/// deficit:   [13, 14)
/// excess:    [14, 15)
/// evap cols: [15, 18)  → evaporation_flow=15, f_evap_plus=16, f_evap_minus=17
#[test]
fn hydro_evaporation_maps_to_evaporation_flow_col() {
    let mut fx = evaporation_fixture(1);
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroEvaporation {
            hydro_id: EntityId(10),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(15, 1.0)]);
}

#[test]
fn hydro_evaporation_no_evap_model_returns_empty() {
    let mut fx = evaporation_fixture(1);
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    // Hydro 20 (pos=1) has no evaporation in evap_hydro_indices=[0]
    let result = call(
        VariableRef::HydroEvaporation {
            hydro_id: EntityId(20),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert!(result.is_empty());
}

/// At `K = 3` on a parallel stage, `HydroEvaporation{None}` and every named block
/// resolve to the SAME single stage-level slot (one entry, NOT a sum over blocks);
/// an out-of-range block still resolves to empty.
#[test]
fn hydro_evaporation_parallel_every_block_resolves_stage_slot() {
    let mut fx = evaporation_fixture(3);
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    assert_eq!(
        layout.geometry.evap_indices.len(),
        1,
        "a parallel stage reserves exactly one evaporation slot per evaporating hydro"
    );

    let resolve = |block_id: Option<usize>| {
        call(
            VariableRef::HydroEvaporation {
                hydro_id: EntityId(10),
                block_id,
            },
            0,
            &ctx,
            &layout,
        )
    };

    let none = resolve(None);
    assert_eq!(
        none.len(),
        1,
        "None resolves to one column, not a K-block sum"
    );
    assert_eq!(none, resolve(Some(0)), "None resolves to block 0");
    assert_eq!(
        resolve(Some(0)),
        resolve(Some(1)),
        "every block on a parallel stage names the same stage-level slot"
    );
    assert_eq!(
        resolve(Some(1)),
        resolve(Some(2)),
        "every block on a parallel stage names the same stage-level slot"
    );
    assert!(
        resolve(Some(3)).is_empty(),
        "out-of-range block resolves to empty"
    );
}

// ── Pumping tests ─────────────────────────────────────────────────────────
//
// `default_fixture` declares two real pumping stations, id 10 (p_idx 0,
// consumption 2.5 MW/(m³/s), pumping hydro 10 into hydro 20) and id 20
// (p_idx 1, consumption 0.75, pumping hydro 30 into hydro 40) — real fixture
// hydro references, not the inert `EntityId(0)` placeholder. Block-major
// column = col_pumping_start + p_idx * n_blks + blk, read from the real
// layout, never a hand-picked base.

/// The pumping-flow column family's full extent for this stage
/// (`col_pumping_start .. col_pumping_start + n_pumping * n_blks`) — the range
/// every resolved `PumpingFlow`/`PumpingPower` column must fall inside.
fn pumping_col_range(layout: &StageLayout<'_>) -> Range<usize> {
    layout.geometry.pumping_flow.clone()
}

/// `PumpingFlow{station, Some(blk)}` → the block-major flow column × 1.0.
///
/// Station id 20 at p_idx 1, blk 2: col = col_pumping_start + 1*n_blks + 2.
#[test]
fn pumping_flow_resolves_to_flow_column_with_unit_coeff() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let col_pumping_start = layout.geometry.pumping_flow.start;
    let n_blks = layout.clock.n_blks();

    let result = call(
        VariableRef::PumpingFlow {
            station_id: EntityId(20),
            block_id: Some(2),
        },
        0, // block_idx — overridden by block_id = Some(2)
        &ctx,
        &layout,
    );

    let expected_col = col_pumping_start + 1 * n_blks + 2;
    assert_eq!(result, vec![(expected_col, 1.0)]);
    assert!(pumping_col_range(&layout).contains(&expected_col));
}

/// `PumpingPower{station, Some(blk)}` → the SAME flow column × consumption.
///
/// Station id 10 at p_idx 0, blk 1: col = col_pumping_start + 0*n_blks + 1,
/// coeff = 2.5. The column is identical to `PumpingFlow` for the same
/// (station, blk) — the power term aliases the flow column, it is not a
/// separate column.
#[test]
fn pumping_power_resolves_to_flow_column_with_consumption_coeff() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let col_pumping_start = layout.geometry.pumping_flow.start;
    let n_blks = layout.clock.n_blks();

    let blk = 1;
    let power = call(
        VariableRef::PumpingPower {
            station_id: EntityId(10),
            block_id: Some(blk),
        },
        0,
        &ctx,
        &layout,
    );
    let flow = call(
        VariableRef::PumpingFlow {
            station_id: EntityId(10),
            block_id: Some(blk),
        },
        0,
        &ctx,
        &layout,
    );

    let expected_col = col_pumping_start + 0 * n_blks + blk;
    assert_eq!(power, vec![(expected_col, 2.5)]);
    // Same column as flow — PumpingPower must alias, not allocate a new column.
    assert_eq!(power[0].0, flow[0].0);
    assert!(pumping_col_range(&layout).contains(&expected_col));
}

/// `PumpingFlow{station, None}` with `block_idx = k` resolves the single
/// column for block `k` (`eff_blk = block_id.unwrap_or(block_idx)`), so the
/// caller's per-block loop yields one `(col, 1.0)` entry per block in order.
#[test]
fn pumping_flow_none_resolves_per_block() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let col_pumping_start = layout.geometry.pumping_flow.start;
    let n_blks = layout.clock.n_blks();
    let range = pumping_col_range(&layout);

    let per_block: Vec<(usize, f64)> = (0..n_blks)
        .map(|blk| {
            let r = call(
                VariableRef::PumpingFlow {
                    station_id: EntityId(10),
                    block_id: None,
                },
                blk, // block_idx supplies the effective block
                &ctx,
                &layout,
            );
            assert_eq!(r.len(), 1);
            assert!(range.contains(&r[0].0));
            r[0]
        })
        .collect();

    assert_eq!(
        per_block,
        vec![
            (col_pumping_start + 0, 1.0),
            (col_pumping_start + 1, 1.0),
            (col_pumping_start + 2, 1.0),
        ]
    );
}

/// `PumpingPower{station, None}` resolves to the per-block column × consumption.
///
/// Station id 20 at p_idx 1, consumption 0.75.
#[test]
fn pumping_power_none_resolves_per_block_with_consumption() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let col_pumping_start = layout.geometry.pumping_flow.start;
    let n_blks = layout.clock.n_blks();
    let range = pumping_col_range(&layout);

    let per_block: Vec<(usize, f64)> = (0..n_blks)
        .map(|blk| {
            let r = call(
                VariableRef::PumpingPower {
                    station_id: EntityId(20),
                    block_id: None,
                },
                blk,
                &ctx,
                &layout,
            );
            assert_eq!(r.len(), 1);
            assert!(range.contains(&r[0].0));
            r[0]
        })
        .collect();

    assert_eq!(
        per_block,
        vec![
            (col_pumping_start + 1 * n_blks + 0, 0.75),
            (col_pumping_start + 1 * n_blks + 1, 0.75),
            (col_pumping_start + 1 * n_blks + 2, 0.75),
        ]
    );
}

#[test]
fn pumping_unknown_station_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    for var_ref in [
        VariableRef::PumpingFlow {
            station_id: EntityId(999),
            block_id: None,
        },
        VariableRef::PumpingPower {
            station_id: EntityId(999),
            block_id: Some(0),
        },
    ] {
        let result = call(var_ref, 0, &ctx, &layout);
        assert!(
            result.is_empty(),
            "unknown station must return empty vec, got: {result:?} for {var_ref:?}"
        );
    }
}

/// `n_pumping == 0` (no stations) resolves to `vec![]` — the empty `positions`
/// lookup misses before `col_pumping_start` is ever used. Deliberately overrides
/// the fixture's own declared two stations with none.
#[test]
fn pumping_no_stations_returns_empty() {
    let mut fx = default_fixture();
    let mut ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let no_stations: Vec<PumpingStation> = Vec::new();
    ctx.pumping_stations = &no_stations;
    let positions_no_stations = EntityPositions::from_slices(
        ctx.hydros.iter().map(|h| h.id),
        ctx.thermals.iter().map(|t| t.id),
        ctx.lines.iter().map(|l| l.id),
        ctx.buses.iter().map(|b| b.id),
        [],
        ctx.contracts.iter().map(|c| c.id),
    );
    ctx.positions = &positions_no_stations;

    for var_ref in [
        VariableRef::PumpingFlow {
            station_id: EntityId(10),
            block_id: Some(0),
        },
        VariableRef::PumpingPower {
            station_id: EntityId(10),
            block_id: None,
        },
    ] {
        let result = call(var_ref, 0, &ctx, &layout);
        assert!(
            result.is_empty(),
            "n_pumping == 0 must return empty vec, got: {result:?} for {var_ref:?}"
        );
    }
}

// ── Contract resolution tests ─────────────────────────────────────────────

/// `contract_family_slot` counts only same-direction contracts before `c_sys`.
/// Slice order: import(10), export(20), import(30), export(40) → import slots
/// 0,1 and export slots 0,1.
#[test]
fn contract_family_slot_counts_per_direction() {
    let contracts = vec![
        make_contract(10, ContractType::Import),
        make_contract(20, ContractType::Export),
        make_contract(30, ContractType::Import),
        make_contract(40, ContractType::Export),
    ];
    assert_eq!(
        contract_family_slot(&contracts, 0),
        (ContractType::Import, 0)
    );
    assert_eq!(
        contract_family_slot(&contracts, 1),
        (ContractType::Export, 0)
    );
    assert_eq!(
        contract_family_slot(&contracts, 2),
        (ContractType::Import, 1)
    );
    assert_eq!(
        contract_family_slot(&contracts, 3),
        (ContractType::Export, 1)
    );
}

/// Two imports + one export, on the default fixture's own REAL declared
/// contracts (`n_contract_import = 2`, `n_contract_export = 1`) and their real
/// `layout.geometry.contract_import`/`contract_export` bases — the second
/// import (id 30, per-family slot 1) at block 0 is
/// `layout.geometry.contract_col(Import, 1, 0) = import_start + n_blks`.
#[test]
fn contract_import_resolves_to_column_with_unit_coefficient() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::ContractImport {
            contract_id: EntityId(30),
            block_id: Some(0),
        },
        0,
        &ctx,
        &layout,
    );

    let import_start = layout.geometry.contract_import.start;
    let expected_col = import_start + 1 * layout.clock.n_blks() + 0;
    assert_eq!(result, vec![(expected_col, 1.0)]);
    assert!(layout.geometry.contract_import.contains(&expected_col));
}

/// The variable's own coefficient is `+1.0`; the injection/withdrawal sign is
/// owned by the load-balance fill, not here. Same fixture's real contracts as
/// the import test: export id 20 is per-family slot 0,
/// `grid.flat(export_start, 0, 2)`.
#[test]
fn contract_export_resolves_to_column_with_unit_coefficient() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::ContractExport {
            contract_id: EntityId(20),
            block_id: Some(2),
        },
        0,
        &ctx,
        &layout,
    );

    let export_start = layout.geometry.contract_export.start;
    let expected_col = export_start + 0 * layout.clock.n_blks() + 2;
    assert_eq!(result, vec![(expected_col, 1.0)]);
    assert!(layout.geometry.contract_export.contains(&expected_col));
}

/// An unknown contract id misses `contract_pos` and resolves to empty — the
/// defense-in-depth fallback past referential validation.
#[test]
fn contract_unknown_id_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::ContractImport {
            contract_id: EntityId(99),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert!(result.is_empty());
}

// ── Stub entity tests ─────────────────────────────────────────────────────

#[test]
fn non_controllable_generation_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::NonControllableGeneration {
            source_id: EntityId(7),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert!(result.is_empty());
}

/// `HydroWithdrawal` resolves to an empty vec: withdrawal carries no LP
/// decision column (a schedule fixed by bounds, not a decision variable), so
/// a generic-constraint term referencing it contributes no `(column,
/// coefficient)` pair — the deliberate stub contract documented above the
/// `resolve_variable_ref` stub arm. `EntityId(999)` is in no `hydro_pos`
/// entry, confirming the empty return is unconditional, not a missing-id
/// fall-through.
#[test]
fn hydro_withdrawal_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroWithdrawal {
            hydro_id: EntityId(999),
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(
        result,
        Vec::<(usize, f64)>::new(),
        "HydroWithdrawal must resolve to no column (no-LP-column stub contract)"
    );
}

/// `NonControllableCurtailment` resolves to an empty vec: non-controllable
/// sources carry no decision column, so a generic-constraint term referencing
/// curtailment contributes no `(column, coefficient)` pair — the same
/// deliberate stub contract as `NonControllableGeneration`. `EntityId(999)` is
/// in no position map, confirming the empty return is unconditional, not a
/// missing-id fall-through.
#[test]
fn non_controllable_curtailment_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::NonControllableCurtailment {
            source_id: EntityId(999),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(
        result,
        Vec::<(usize, f64)>::new(),
        "NonControllableCurtailment must resolve to no column (no-LP-column stub contract)"
    );
}

// ── Missing entity ID test ─────────────────────────────────────────────────

#[test]
fn missing_entity_id_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::ThermalGeneration {
            thermal_id: EntityId(999),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert!(result.is_empty());
}

// ── BusDeficit tests ──────────────────────────────────────────────────────

/// bus_pos[EntityId(100)] = 0, deficit.start = 61, S = 2, n_blks = 3, block_idx = 0 →
/// [(61 + 0*2*3 + 0*3, 1.0), (61 + 0*2*3 + 1*3, 1.0)] = [(61, 1.0), (64, 1.0)].
#[test]
fn bus_deficit_returns_one_entry_per_segment() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::BusDeficit {
            bus_id: EntityId(100),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (61, 1.0));
    assert_eq!(result[1], (64, 1.0));
}

/// bus_pos[EntityId(200)] = 1, deficit.start = 61, S = 2, n_blks = 3, blk = 1:
/// seg0 = 61 + 1*2*3 + 0*3 + 1 = 68; seg1 = 61 + 1*2*3 + 1*3 + 1 = 71.
#[test]
fn bus_deficit_second_bus_block_1() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::BusDeficit {
            bus_id: EntityId(200),
            block_id: None,
        },
        1,
        &ctx,
        &layout,
    );

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (68, 1.0));
    assert_eq!(result[1], (71, 1.0));
}

// ── BusExcess tests ───────────────────────────────────────────────────────

/// bus_pos[EntityId(100)] = 0, excess.start = 73, n_blks = 3, block = 2 →
/// [(73 + 0*3 + 2, 1.0)] = [(75, 1.0)].
#[test]
fn bus_excess_maps_to_excess_column() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::BusExcess {
            bus_id: EntityId(100),
            block_id: None,
        },
        2,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(73 + 0 * 3 + 2, 1.0)]);
}

// ── LineDirect / LineReverse tests ────────────────────────────────────────

/// line_pos[EntityId(50)] = 0, line_fwd.start = 55, n_blks = 3, block = 1 →
/// [(55 + 0*3 + 1, 1.0)] = [(56, 1.0)].
#[test]
fn line_direct_maps_to_fwd_column() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::LineDirect {
            line_id: EntityId(50),
            block_id: None,
        },
        1,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(55 + 0 * 3 + 1, 1.0)]);
}

/// line_pos[EntityId(50)] = 0, line_rev.start = 58, n_blks = 3, block = 0 → [(58, 1.0)].
#[test]
fn line_reverse_maps_to_rev_column() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::LineReverse {
            line_id: EntityId(50),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(58, 1.0)]);
}

// ── LineExchange tests ──────────────────────────────────────────────────────

/// LineExchange maps to both line_fwd and line_rev columns with opposite signs.
///
/// line_pos[EntityId(50)] = 0, line_fwd.start = 55, line_rev.start = 58,
/// n_blks = 3, block = 1
/// Expected: [(55 + 0*3 + 1, 1.0), (58 + 0*3 + 1, -1.0)] = [(56, 1.0), (59, -1.0)]
#[test]
fn line_exchange_maps_to_fwd_and_rev_columns() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::LineExchange {
            line_id: EntityId(50),
            block_id: None,
        },
        1,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(56, 1.0), (59, -1.0)]);
}

/// LineExchange with explicit block_id overrides current block_idx.
///
/// block_idx = 2 but block_id = Some(0)
/// Expected: [(55, 1.0), (58, -1.0)]
#[test]
fn line_exchange_with_explicit_block() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::LineExchange {
            line_id: EntityId(50),
            block_id: Some(0),
        },
        2,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(55, 1.0), (58, -1.0)]);
}

#[test]
fn line_exchange_unknown_id_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::LineExchange {
            line_id: EntityId(999),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert!(result.is_empty());
}

// ── AnticipatedDecision tests ─────────────────────────────────────────────
//
// `anticipated_fixture` builds N=0 hydros, T=2 thermals (pos 0 = regular, pos
// 1 = anticipated), Ln=0, B=1 bus, K=2 blocks, n_anticipated=1, lead_stages=2
// — the same dims [`anticipated_decision_maps_to_correct_column`]'s doc
// derives `anticipated_decision.start = 9` from.

/// `AnticipatedDecision` for an anticipated thermal maps to
/// `anticipated_decision.start + local_idx`: EntityId(6) at sys_pos=1
/// (`anticipated_plants`'s only entry), anticipated_decision.start = 9, local_idx = 0
/// → column 9.
#[test]
fn anticipated_decision_maps_to_correct_column() {
    let mut fx = anticipated_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    assert_eq!(
        layout.geometry.anticipated_decision.start, 9,
        "anticipated_decision.start should be 9, got {}",
        layout.geometry.anticipated_decision.start
    );

    let result = call(
        VariableRef::AnticipatedDecision {
            thermal_id: EntityId(6), // sys_pos=1, local anticipated idx=0
        },
        0, // block_idx is ignored for stage-level variable
        &ctx,
        &layout,
    );

    assert_eq!(
        result,
        vec![(9, 1.0)],
        "AnticipatedDecision(6) should resolve to column 9 (anticipated_decision.start + 0)"
    );
}

/// `AnticipatedDecision` is stage-level — the returned column is the same
/// regardless of `block_idx`.
#[test]
fn anticipated_decision_ignores_block_idx() {
    let mut fx = anticipated_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    for block_idx in [0, 1] {
        let result = call(
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(6),
            },
            block_idx,
            &ctx,
            &layout,
        );
        assert_eq!(
            result,
            vec![(9, 1.0)],
            "AnticipatedDecision must be stage-level (block_idx={block_idx} should not change column)"
        );
    }
}

/// A regular (non-anticipated) thermal returns empty (defense-in-depth):
/// EntityId(5) at sys_pos=0 is NOT in `anticipated_plants`.
#[test]
fn anticipated_decision_non_anticipated_thermal_returns_empty() {
    let mut fx = anticipated_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5), // sys_pos=0, NOT anticipated
        },
        0,
        &ctx,
        &layout,
    );

    assert!(
        result.is_empty(),
        "AnticipatedDecision for non-anticipated thermal must return empty vec, got: {result:?}"
    );
}

#[test]
fn anticipated_decision_unknown_entity_returns_empty() {
    let mut fx = anticipated_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::AnticipatedDecision {
            thermal_id: EntityId(999), // unknown
        },
        0,
        &ctx,
        &layout,
    );

    assert!(
        result.is_empty(),
        "AnticipatedDecision for unknown entity must return empty vec, got: {result:?}"
    );
}

// ── Single-column resolver family/range tests ─────────────────────────────

/// Each single-column family's resolver (`resolve_hydro_spillage`,
/// `resolve_hydro_diversion`, `resolve_thermal_generation`,
/// `resolve_line_direct`, `resolve_line_reverse`, `resolve_bus_excess`) lands
/// inside its matching `layout.geometry.<family>` range — the contract
/// `resolve_block_column` upholds for every one of its six callers.
#[test]
fn single_column_resolvers_land_inside_their_equipment_range() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let cases: [(VariableRef, &Range<usize>); 6] = [
        (
            VariableRef::HydroSpillage {
                hydro_id: EntityId(20),
                block_id: Some(1),
            },
            &layout.geometry.spillage,
        ),
        (
            VariableRef::HydroDiversion {
                hydro_id: EntityId(20),
                block_id: Some(1),
            },
            &layout.geometry.diversion,
        ),
        (
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: Some(1),
            },
            &layout.geometry.thermal,
        ),
        (
            VariableRef::LineDirect {
                line_id: EntityId(50),
                block_id: Some(1),
            },
            &layout.geometry.line_fwd,
        ),
        (
            VariableRef::LineReverse {
                line_id: EntityId(50),
                block_id: Some(1),
            },
            &layout.geometry.line_rev,
        ),
        (
            VariableRef::BusExcess {
                bus_id: EntityId(100),
                block_id: Some(1),
            },
            &layout.geometry.excess,
        ),
    ];

    for (var_ref, range) in cases {
        let result = call(var_ref, 0, &ctx, &layout);
        assert_eq!(result.len(), 1, "{var_ref:?}");
        assert!(
            range.contains(&result[0].0),
            "{var_ref:?} resolved outside its family range"
        );
    }
}

// ── HydroTurbined / HydroSpillage tests ───────────────────────────────────

#[test]
fn hydro_turbined_maps_to_turbine_column() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    // hydro pos=1 (EntityId 20), turbine.start=13, n_blks=3, block=2
    let result = call(
        VariableRef::HydroTurbined {
            hydro_id: EntityId(20),
            block_id: None,
            bus_id: None,
        },
        2,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(13 + 1 * 3 + 2, 1.0)]);
}

#[test]
fn hydro_spillage_maps_to_spillage_column() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    // hydro pos=3 (EntityId 40), spillage.start=25, n_blks=3, block=1
    let result = call(
        VariableRef::HydroSpillage {
            hydro_id: EntityId(40),
            block_id: None,
        },
        1,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(25 + 3 * 3 + 1, 1.0)]);
}

/// `layout.geometry.diversion.start = 37`. For hydro pos=1 (EntityId 20),
/// n_blks=3, block=2 the flat block-major address is `37 + 1*3 + 2 = 42` with
/// the unit coefficient.
#[test]
fn diversion_maps_to_diversion_column() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroDiversion {
            hydro_id: EntityId(20),
            block_id: None,
        },
        2,
        &ctx,
        &layout,
    );

    assert_eq!(result, vec![(37 + 1 * 3 + 2, 1.0)]);
}

// ── HydroInflow tests ──────────────────────────────────────────────────────

/// Cascade for the total-inflow tests: EntityId(10) and EntityId(20) both
/// flow into EntityId(40), so `upstream(40) = [10, 20]` (ID-sorted). The
/// three hydros map to system positions 0, 1, 3 via `make_hydro_pos`.
fn make_inflow_cascade() -> CascadeTopology {
    CascadeTopology::build(&[
        make_hydro(10, Some(40)),
        make_hydro(20, Some(40)),
        make_hydro(30, None),
        make_hydro(40, None),
    ])
}

/// A two-upstream hydro (no diversion-into) resolves at block `k` to the
/// incremental `z_inflow` column plus, in canonical upstream order, each
/// upstream plant's turbine + spillage column, all coefficients `+1.0`.
/// Target EntityId(40) at pos_h=3; upstream [10, 20] at pos 0, 1.
/// z_inflow.start=4 → (7, 1.0); turbine.start=13, spillage.start=25, n_blks=3, k=2.
#[test]
fn hydro_inflow_two_upstream_canonical_order() {
    let mut fx = default_fixture();
    let mut ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let cascade = make_inflow_cascade();
    ctx.cascade = &cascade;

    let blk = 2;
    let result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: Some(blk),
        },
        0, // block_idx — overridden by block_id = Some(blk)
        &ctx,
        &layout,
    );

    let z_col = 4 + 3; // z_inflow.start + pos_h
    let turb = 13; // turbine.start
    let spill = 25; // spillage.start
    let nb = 3; // n_blks
    assert_eq!(
        result,
        vec![
            (z_col, 1.0),
            (turb + 0 * nb + blk, 1.0),  // upstream 10 turbine
            (spill + 0 * nb + blk, 1.0), // upstream 10 spillage
            (turb + 1 * nb + blk, 1.0),  // upstream 20 turbine
            (spill + 1 * nb + blk, 1.0), // upstream 20 spillage
        ]
    );
}

/// `block_id = None` with `block_idx = k` matches `block_id = Some(k)`
/// (the resolver uses `eff_blk = block_id.unwrap_or(block_idx)`).
#[test]
fn hydro_inflow_none_matches_some_block_idx() {
    let mut fx = default_fixture();
    let mut ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let cascade = make_inflow_cascade();
    ctx.cascade = &cascade;

    let blk = 2;
    let none_result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: None,
        },
        blk, // block_idx supplies the effective block
        &ctx,
        &layout,
    );
    let some_result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: Some(blk),
        },
        0,
        &ctx,
        &layout,
    );

    assert_eq!(none_result, some_result);
}

/// A hydro that also has a plant diverting into it gets the diversion column
/// appended right after its own `z_inflow` column, before the upstream terms
/// (both are the hydro's own local inflow rate). `diversion_upstream[40] =
/// [2]` (system index 2), diversion.start=37, n_blks=3, k=1 →
/// (37 + 2*3 + 1, 1.0) = (44, 1.0).
#[test]
fn hydro_inflow_diversion_into_appends_diversion_column() {
    let mut fx = default_fixture();
    let mut ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let cascade = make_inflow_cascade();
    let div: HashMap<EntityId, Vec<usize>> = [(EntityId(40), vec![2])].into_iter().collect();
    ctx.cascade = &cascade;
    ctx.diversion_upstream = &div;

    let blk = 1;
    let result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: Some(blk),
        },
        0,
        &ctx,
        &layout,
    );

    let z_col = 4 + 3;
    let turb = 13; // turbine.start
    let spill = 25; // spillage.start
    let div_start = 37; // diversion.start
    let nb = 3; // n_blks
    assert_eq!(
        result,
        vec![
            (z_col, 1.0),
            (div_start + 2 * nb + blk, 1.0), // diversion-into, system index 2
            (turb + 0 * nb + blk, 1.0),
            (spill + 0 * nb + blk, 1.0),
            (turb + 1 * nb + blk, 1.0),
            (spill + 1 * nb + blk, 1.0),
        ]
    );
}

/// A headwater hydro (no upstream, no diversion-into) resolves to exactly the
/// incremental `z_inflow` column. EntityId(30) at pos=2 is a headwater in
/// `make_inflow_cascade`; z_inflow.start=4 → (6, 1.0).
#[test]
fn hydro_inflow_headwater_resolves_to_z_inflow_only() {
    let mut fx = default_fixture();
    let mut ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let cascade = make_inflow_cascade();
    ctx.cascade = &cascade;

    for block_idx in [0, 1, 2] {
        let result = call(
            VariableRef::HydroInflow {
                hydro_id: EntityId(30),
                block_id: None,
            },
            block_idx,
            &ctx,
            &layout,
        );
        assert_eq!(result, vec![(6, 1.0)], "block_idx={block_idx}");
    }
}

/// `hydro_count == 0` (empty `z_inflow`) resolves to `vec![]`:
/// `anticipated_fixture` has no hydros, so `z_inflow` is empty and
/// `z_inflow.start` is meaningless; the resolver must short-circuit to `[]`.
#[test]
fn hydro_inflow_empty_when_no_hydros() {
    let mut fx = anticipated_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    assert!(
        ctx.state.z_inflow.is_empty(),
        "z_inflow must be empty with hydro_count == 0"
    );

    let result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(0),
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert!(
        result.is_empty(),
        "HydroInflow with hydro_count == 0 must return empty vec, got: {result:?}"
    );
}

#[test]
fn hydro_inflow_unknown_id_returns_empty() {
    let mut fx = default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(999), // unknown
            block_id: None,
        },
        0,
        &ctx,
        &layout,
    );

    assert!(
        result.is_empty(),
        "HydroInflow for unknown id must return empty vec, got: {result:?}"
    );
}

// ── Per-block storage boundary tests ──────────────────────────────────────
//
// A K=3 chronological build of `default_fixture`'s own hydro/thermal/line/bus
// set: `default_fixture`'s `Stage` is always parallel, whose interior stride
// is zero, so the interior-boundary family needs a genuinely chronological
// stage to reserve real columns — never a copied/overridden grid.

/// `default_fixture` with its `Stage` flipped to [`BlockMode::Chronological`]
/// after construction, so `StageLayout::new` reserves a real, non-empty
/// interior-boundary column family (`n_interior = n_blks - 1`) for the
/// per-block storage boundary tests below. `ctx.state` (storage/storage_in) is
/// block-mode-independent and unaffected.
fn chronological_default_fixture() -> ResolverFixture {
    let mut fx = default_fixture();
    fx.stage.block_mode = BlockMode::Chronological;
    fx
}

/// `VariableRef::HydroStorageInitial`/`HydroStorageFinal` resolve to the S⁰/
/// interior/Sᴷ boundary columns, each computed from the layout's own owners
/// (`ctx.state.storage_in`/`storage`, `layout.geometry.storage_internal_start`)
/// independently of `StorageBoundaryGrid::col` itself, so a regression in its
/// match arms fails this test, not just an identity of the owner with itself.
///
/// Seam B of the geometry cross-check guard — pairs with
/// `stage_geometry_block_storage_col_matches_layout` (Seam A, in
/// `super::super::template::tests`), each independently anchored
/// against its own hand-computed oracle rather than compared to each other.
#[test]
fn hydro_storage_boundary_resolves_each_boundary() {
    let mut fx = chronological_default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let storage_internal_start = layout.geometry.storage_internal_start;

    // Hydro EntityId(10) at pos 0; K = 3.
    let initial_0 = call(
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(10),
            block_id: Some(0),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        initial_0,
        vec![(ctx.state.storage_in.start + 0, 1.0)],
        "S⁰ = storage_in.start + 0"
    );

    let initial_1 = call(
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(10),
            block_id: Some(1),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        initial_1,
        vec![(storage_internal_start + 0, 1.0)],
        "S¹ = storage_internal_start + 0 (interior boundary)"
    );

    let final_2 = call(
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(10),
            block_id: Some(2),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        final_2,
        vec![(ctx.state.storage.start + 0, 1.0)],
        "S³ = Sᴷ = storage.start + 0 (K=3, last block)"
    );
}

/// `HydroStorageFinal{K-1}` resolves to the SAME column as `HydroStorage` (Sᴷ).
#[test]
fn hydro_storage_final_last_block_equals_hydro_storage() {
    let mut fx = chronological_default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let final_last = call(
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(10),
            block_id: Some(2),
        },
        0,
        &ctx,
        &layout,
    );
    let storage = call(
        VariableRef::HydroStorage {
            hydro_id: EntityId(10),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(final_last, storage);
    assert_eq!(final_last, vec![(ctx.state.storage.start + 0, 1.0)]);
}

/// A block's final boundary is the next block's initial: `Final{0}` and
/// `Initial{1}` both resolve to the interior column `S¹`.
#[test]
fn hydro_storage_final_shares_interior_column_with_next_initial() {
    let mut fx = chronological_default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);
    let storage_internal_start = layout.geometry.storage_internal_start;

    let final_0 = call(
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(10),
            block_id: Some(0),
        },
        0,
        &ctx,
        &layout,
    );
    let initial_1 = call(
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(10),
            block_id: Some(1),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(final_0, initial_1);
    assert_eq!(final_0, vec![(storage_internal_start + 0, 1.0)]);
}

/// `block_id = None` resolves to the fixed stage endpoint — `S⁰` for initial,
/// `Sᴷ` for final — independent of the caller's `block_idx`.
#[test]
fn hydro_storage_boundary_none_resolves_stage_endpoint() {
    let mut fx = chronological_default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    for blk in 0..3 {
        let initial = call(
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(10),
                block_id: None,
            },
            blk,
            &ctx,
            &layout,
        );
        assert_eq!(
            initial,
            vec![(
                layout.block_storage_col(HydroSys::new(0), Boundary::Incoming),
                1.0
            )]
        );

        let final_ = call(
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(10),
                block_id: None,
            },
            blk,
            &ctx,
            &layout,
        );
        assert_eq!(
            final_,
            vec![(
                layout.block_storage_col(HydroSys::new(0), Boundary::Outgoing),
                1.0
            )]
        );
    }
}

#[test]
fn hydro_storage_boundary_unknown_id_returns_empty() {
    let mut fx = chronological_default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    for var_ref in [
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(999),
            block_id: Some(1),
        },
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(999),
            block_id: None,
        },
    ] {
        let result = call(var_ref, 0, &ctx, &layout);
        assert!(result.is_empty(), "unknown id must resolve to empty vec");
    }
}

/// `HydroUsefulVolumeInitial`/`HydroUsefulVolumeFinal` resolve to the SAME
/// column as `HydroStorageInitial`/`HydroStorageFinal` (the `-V_lo` shift is a
/// bound-fold concern, not a resolution-time offset), and are block-independent
/// like their storage counterparts.
#[test]
fn hydro_useful_volume_boundary_matches_storage_boundary() {
    let mut fx = chronological_default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    for block_id in [None, Some(0), Some(1), Some(2)] {
        let useful_initial = call(
            VariableRef::HydroUsefulVolumeInitial {
                hydro_id: EntityId(10),
                block_id,
            },
            0,
            &ctx,
            &layout,
        );
        let storage_initial = call(
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(10),
                block_id,
            },
            0,
            &ctx,
            &layout,
        );
        assert_eq!(useful_initial, storage_initial);

        let useful_final = call(
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(10),
                block_id,
            },
            0,
            &ctx,
            &layout,
        );
        let storage_final = call(
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(10),
                block_id,
            },
            0,
            &ctx,
            &layout,
        );
        assert_eq!(useful_final, storage_final);

        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroUsefulVolumeInitial {
                hydro_id: EntityId(10),
                block_id,
            }
        ));
        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(10),
                block_id,
            }
        ));
    }
}

/// `resolve_hydro_storage_boundary` resolves both useful-volume boundary
/// variants at coefficient exactly `1.0` — the unit coefficient
/// `useful_volume_bound_shift` relies on when it folds `V_lo` without a
/// `* multiplier` factor.
#[test]
fn hydro_useful_volume_boundary_multiplier_is_exactly_one() {
    let mut fx = chronological_default_fixture();
    let ctx = fx.base.ctx();
    let layout = StageLayout::new(&ctx, &fx.stage, 0);

    let useful_initial = call(
        VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId(10),
            block_id: Some(0),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        useful_initial,
        vec![(ctx.state.storage_in.start + 0, 1.0)],
        "S⁰ = storage_in.start + 0"
    );

    let useful_final = call(
        VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(10),
            block_id: Some(2),
        },
        0,
        &ctx,
        &layout,
    );
    assert_eq!(
        useful_final,
        vec![(ctx.state.storage.start + 0, 1.0)],
        "S³ = Sᴷ = storage.start + 0 (K=3, last block)"
    );
}
