//! Crate-internal test-support builders: role-(a) [`StateSpace`], role-(b)
//! [`StageGeometry`], and [`StudyDimensions`] fixtures shared across the crate's
//! unit tests and `tests/` integration suites.
//!
//! [`geometry`] drives the production `StageLayout::new`/`StageLayout::geometry`
//! constructors from explicit equipment dimensions, so a test exercises the exact
//! construction path a study of those dimensions would, without a full
//! `StudySetup`. They construct crate-internal types, so they live in `src/` under
//! `#[cfg(any(test, feature = "test-support"))]` — reachable by plain `cargo test`
//! and by downstream integration tests via the `test-support` feature.

#![deny(clippy::allow_attributes, clippy::allow_attributes_without_reason)]

use std::collections::BTreeMap;
use std::path::Path;

use chrono::NaiveDate;
use cobre_comm::LocalBackend;
use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel, SamplingScheme};
use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, StageLagTransition, Transition};
use cobre_core::{
    AnticipatedConfig, Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties,
    ContractBlockBounds, DeficitSegment, EntityId, HorizonGraph, Hydro, HydroBlockBounds,
    HydroGenerationModel, HydroPenalties, HydroStageBounds, HydroStorage, HydroUnitGroup,
    InitialConditions, Line, LineBlockBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod,
    PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties,
    ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, System, SystemBuilder, Thermal,
    ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::StageIdResolver;
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, InflowNonNegativityMethod,
    ModelingConfig, ParallelismConfig, PolicyConfig, RawClassConfigEntry, RawSamplingScheme,
    RawScenarioSourceConfig, RowSelectionConfig, SelectionMethod,
    SimulationConfig as IoSimulationConfig, SimulationSelection, StoppingMode, StoppingRuleConfig,
    TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_io::{
    EntitySlot, GraphManifest, ManifestEdge, ManifestNode, PolicyCutRecord, ProducerBlock,
    StageCutsPayload, decode_slot_date, encode_slot_date, write_policy_checkpoint,
};
use cobre_stochastic::par::precompute::PrecomputedPar;
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

use crate::BoundaryStateRequirements;
use crate::StudySetup;
#[cfg(test)]
use crate::bucket_topology::{TransitBucketTopology, build_transit_bucket_topology};
use crate::context::{StageContext, TrainingContext};
use crate::cut::pool::CutPool;
use crate::error::SddpError;
use crate::horizon_mode::HorizonMode;
use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, FphaPlane, PrepareHydroModelsResult, ProductionModelSet,
    ResolvedProductionModel,
};
use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};
use crate::lower_bound::{LbEvalScratch, LbEvalScratchBundle, evaluate_lower_bound};
use crate::lp::builder::{
    FactGroups, PatchBuffer, StageGeometry, StageLayout, StageTemplates, StateBox,
    encode_stage_templates_facts, encode_time_value_facts,
};
use crate::lp::indexer::{
    AnticipatedPlants, BlockRowFamily, CutStateProjection, EntityPositions, HydroCellIndex,
    HydroSys, StateDim, StateSpace, StudyDimensions,
};
use crate::noise::{DownstreamAccumState, LagAccumState};
use crate::policy::policy_load::{
    FullFcf, POLICY_COBRE_VERSION, PolicyLoadProof, PolicyStageManifest, validate_policy_load,
};
use crate::risk_measure::BackwardOutcome;
use crate::setup::node_graph::{
    NodeGraph, NodeId, NodePos, OpeningSource, StageIdx, build_node_graph,
    enumerated_node_visit_counts, enumerated_scenario_count,
};
#[cfg(test)]
use crate::setup::{ResolvedStateLayout, resolve_state_layout};
use crate::solve::stage_solve::{StageInputs, assemble_outgoing_state, run_stage_solve};
use crate::solver_stats::SolverStatsDelta;
#[cfg(test)]
use crate::time_value::DeliveryCalendar;
use crate::time_value::{PostStudyResolved, TimeValue};
use crate::training::backward::{
    extract_state_duals_only, fill_external_opening_noise, write_opening_outcome,
};
use crate::training::stage_solve_prep::{
    InflowNoise, StageSolvePrep, StageSolvePrepParams, StateSource,
};
use crate::trajectory::TrajectoryRecord;
use crate::workspace::{CapturedBasis, ScratchBuffers, SolverWorkspace, WorkspaceSizing};
use cobre_core::scenario::{ExternalLoadRow, ExternalScenarioRow};
use cobre_solver::{
    ActiveSolver, Basis, BasisStatus, LpSolution, RowBatch, SolutionView, SolverError,
    SolverInterface, SolverStatistics, StageTemplate,
};

pub mod decks;
pub mod template_structure;

pub(crate) mod ctx_fixture;
use ctx_fixture::CtxFixture;

/// Equipment dimensions for the [`geometry`] / [`study_dims_for`] test builders.
///
/// `Default` sets `max_deficit_segments == 1` (a non-degenerate deficit stride);
/// every other count is `0`.
#[derive(Debug, Clone)]
pub struct GeometryDims {
    /// Number of hydro plants.
    pub hydro_count: usize,
    /// Maximum PAR model order across all hydros.
    pub max_par_order: usize,
    /// Number of thermal units.
    pub n_thermals: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of buses.
    pub n_buses: usize,
    /// Number of demand blocks in the stage.
    pub n_blks: usize,
    /// Whether to include inflow penalty slack columns.
    pub has_inflow_penalty: bool,
    /// Maximum number of deficit segments across all buses.
    pub max_deficit_segments: usize,
    /// Number of anticipated thermals.
    pub n_anticipated: usize,
    /// Per-plant lead stage, uniform across every anticipated thermal.
    pub lead_stages: usize,
    /// The anticipated-plant set.
    pub anticipated_plants: AnticipatedPlants,
}

impl Default for GeometryDims {
    fn default() -> Self {
        Self {
            hydro_count: 0,
            max_par_order: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            n_blks: 0,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 0,
            lead_stages: 0,
            anticipated_plants: AnticipatedPlants::default(),
        }
    }
}

/// Overwrite `out` with a basis satisfying `col_basic + row_basic ==
/// row_status.len()` — the consistency invariant every real solver's `get_basis`
/// returns and [`enforce_basic_count_invariant`] requires of a stored basis.
///
/// A mock `get_basis` that leaves `out` untouched keeps the all-`Lower` zero-fill
/// from `Basis::new`, i.e. `total_basic == 0`. No solver produces that, and
/// offering it back as a warm start is a basic-count deficit, which
/// [`enforce_basic_count_invariant`] rejects as a shape mismatch.
///
/// [`enforce_basic_count_invariant`]: crate::basis_reconstruct::enforce_basic_count_invariant
pub fn fill_consistent_basis(out: &mut Basis) {
    let num_row = out.row_status.len();
    out.col_status.fill(BasisStatus::Lower);
    out.row_status.fill(BasisStatus::Lower);
    let basic_cols = num_row.min(out.col_status.len());
    out.col_status[..basic_cols].fill(BasisStatus::Basic);
    out.row_status[..num_row - basic_cols].fill(BasisStatus::Basic);
}

/// A fully-permissive `(-inf, inf)` box per stage, for fixtures driving a solve
/// through the seam without exercising the clamp.
#[must_use]
pub fn permissive_state_boxes(n_state: usize, n_stages: usize) -> Vec<StateBox> {
    vec![
        StateBox {
            lower: vec![f64::NEG_INFINITY; n_state],
            upper: vec![f64::INFINITY; n_state],
        };
        n_stages
    ]
}

/// Build [`GeometryDims`] with the scalar entity counts set and no anticipated
/// thermals.
#[must_use]
pub fn eq(
    hydro_count: usize,
    max_par_order: usize,
    n_thermals: usize,
    n_lines: usize,
    n_buses: usize,
    n_blks: usize,
    has_inflow_penalty: bool,
) -> GeometryDims {
    GeometryDims {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        ..Default::default()
    }
}

/// Build [`GeometryDims`] with explicit anticipated-thermal fields.
///
/// The anticipated-plant set defaults to positions `0..n_anticipated`.
#[must_use]
pub fn eq_with_anticipated(
    hydro_count: usize,
    max_par_order: usize,
    n_thermals: usize,
    n_lines: usize,
    n_buses: usize,
    n_blks: usize,
    has_inflow_penalty: bool,
    n_anticipated: usize,
    lead_stages: usize,
) -> GeometryDims {
    GeometryDims {
        n_anticipated,
        lead_stages,
        anticipated_plants: anticipated_plants_at(&(0..n_anticipated).collect::<Vec<usize>>()),
        ..eq(
            hydro_count,
            max_par_order,
            n_thermals,
            n_lines,
            n_buses,
            n_blks,
            has_inflow_penalty,
        )
    }
}

/// Build an [`AnticipatedPlants`] whose only members are `positions`, each with
/// a one-stage lead — the fixture builder every positionless test uses in place
/// of owning a `System`. `positions` must be strictly ascending.
#[must_use]
pub fn anticipated_plants_at(positions: &[usize]) -> AnticipatedPlants {
    debug_assert!(
        positions.windows(2).all(|w| w[0] < w[1]),
        "positions must be strictly ascending"
    );
    let n = positions.last().map_or(0, |&p| p + 1);
    let thermals: Vec<Thermal> = (0..n)
        .map(|idx| Thermal {
            id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
            name: String::new(),
            operational_start_date: ymd(2024, 1, 1),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 0.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: positions
                .contains(&idx)
                .then_some(AnticipatedConfig::LeadStages(1)),
        })
        .collect();
    AnticipatedPlants::build(&thermals)
}

/// Resolve `lead_stages` (anticipated-local order, one constant per-plant
/// lead each) over a synthetic `n_stages`-long delivery axis with no
/// post-study continuation (`n_delivery == n_stages`) — the fixture
/// substitute for a real study's calendar-derived resolution, built through
/// [`AnticipatedResolution::resolve`] the same way
/// [`crate::setup::resolve_anticipated_commitments_core`] builds its axis.
#[must_use]
pub fn constant_lead_resolution(lead_stages: &[usize], n_stages: usize) -> AnticipatedResolution {
    let leads: Vec<LeadTime> = lead_stages
        .iter()
        .map(|&l| LeadTime::Stages(u32::try_from(l).unwrap_or(u32::MAX)))
        .collect();
    let study_stage_hours = vec![720.0; n_stages];
    AnticipatedResolution::resolve(
        &leads,
        DeliveryAxis {
            study_stage_hours: &study_stage_hours,
            post_study_stage_hours: &[],
        },
    )
}

#[cfg(test)]
mod constant_lead_resolution_tests {
    use super::{StateDim, StateSpace, constant_lead_resolution};

    /// Every plant reaches every ring slot through its own depth-0
    /// (next-stage) term alone as `stage_idx` sweeps `0..n_stages`, so a
    /// margin of `max(lead) + 2` saturates the commitment-hold mask to the
    /// whole region.
    #[test]
    fn constant_lead_resolution_marks_every_ring_slot_live() {
        for lead_stages in [vec![1_usize], vec![3], vec![1, 3], vec![2, 2]] {
            let k_max = lead_stages.iter().copied().max().unwrap_or(0);
            let n_stages = k_max + 2;
            let n_anticipated = lead_stages.len();
            let resolution = constant_lead_resolution(&lead_stages, n_stages);
            let state = StateSpace::new(0, 0, Vec::new(), lead_stages, resolution, &[]);

            let start = state.commit_out.start;
            let expected: Vec<StateDim> = (start..start + state.n_anticipated * state.k_max)
                .map(StateDim::new)
                .collect();
            assert_eq!(
                state.nonzero_state_indices, expected,
                "n_anticipated={n_anticipated} k_max={k_max}: mask must saturate to the whole region"
            );
        }
    }
}

/// Resolve the bucket topology and role-(a) state layout for `system`/`par_lp`
/// through the same setup entry points production uses
/// ([`build_transit_bucket_topology`], [`crate::setup::resolve_state_layout`]),
/// for a builder-module test that needs production's own resolution rather
/// than a hand-built [`TemplateBuildCtx`](crate::lp::builder::TemplateBuildCtx).
///
/// # Panics
///
/// If `resolve_state_layout` rejects `system` (a `LeadTime` fan-out) — a test
/// fixture is expected to be resolvable.
#[expect(
    clippy::expect_used,
    reason = "a test fixture that resolve_state_layout rejects is a fixture bug, not a runtime error to propagate"
)]
#[cfg(test)]
#[must_use]
pub(crate) fn resolved_layout_for(
    system: &System,
    par_lp: &PrecomputedPar,
) -> (TransitBucketTopology, ResolvedStateLayout) {
    let calendar = DeliveryCalendar::from_system(system);
    let topology = build_transit_bucket_topology(system, &calendar, false);
    let layout = resolve_state_layout(system, &calendar, par_lp, &topology, None)
        .expect("resolved_layout_for: valid test fixture");
    (topology, layout)
}

/// A [`StateSpace`] over `system`'s hydros and `topology`'s bucket order, with
/// no PAR lags and no anticipated plants — the seed tests' fixture for
/// [`crate::setup::build_initial_transit_bucket_state`], built the way
/// [`crate::test_support::ctx_fixture::CtxFixture::build_state`]'s
/// `max_par_order == 0` branch is.
#[cfg(test)]
#[must_use]
pub(crate) fn bucket_seed_state(system: &System, topology: &TransitBucketTopology) -> StateSpace {
    let hydros = system.hydros();
    let effective_lag_counts = vec![0; hydros.len()];
    StateSpace::build(
        hydros,
        0,
        &effective_lag_counts,
        topology,
        Vec::new(),
        AnticipatedResolution::default(),
    )
}

/// Setup steps a builder-module test needs without a direct
/// `lp::builder` → `setup` import edge.
#[cfg(test)]
pub(crate) use crate::setup::lp_build_inputs::build_filling_v_target;
#[cfg(test)]
pub(crate) use crate::setup::{
    build_study_dimensions, resolve_anticipated_commitments, resolve_lp_build_inputs,
};

/// All-zero [`HydroPenalties`] for [`geometry_hydro`] — no fixture-side penalty
/// cost reaches the column/objective arithmetic `StageLayout::new` computes.
fn geometry_zero_penalties() -> HydroPenalties {
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
        inflow_nonnegativity_cost: 0.0,
    }
}

/// Build a [`HydroUnitGroup`] with the given `id`, `bus_id`, and four bounds —
/// the shared multi-bus fixture helper (`unit_groups` cannot otherwise be
/// populated from outside `cobre-io`'s `pub(super)` validation-module helper).
#[must_use]
pub fn make_unit_group(
    id: EntityId,
    bus_id: EntityId,
    min_generation_mw: f64,
    max_generation_mw: f64,
    min_turbined_m3s: f64,
    max_turbined_m3s: f64,
) -> HydroUnitGroup {
    HydroUnitGroup {
        id,
        name: format!("G{id}"),
        bus_id,
        min_generation_mw,
        max_generation_mw,
        min_turbined_m3s,
        max_turbined_m3s,
    }
}

/// Fixture hydro at system position `idx`: always `Operating` (`filling`,
/// `entry_stage_id`, `exit_stage_id` all `None`), so `StageLayout::new`'s FPHA/
/// evaporation membership filters never drop a caller-requested index regardless
/// of `stage.id`. Declares its mirror unit group before return — `geometry` never
/// mutates the collected `Vec`, so the return is the finalization boundary — and
/// stays `pub(crate)` for `lp::indexer::hydro_cell`'s identity test, which needs these
/// exact hydros.
pub(crate) fn geometry_hydro(idx: usize) -> Hydro {
    let id = EntityId(i32::try_from(idx).unwrap_or(i32::MAX));
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id,
        name: String::new(),
        operational_start_date: NaiveDate::default(),
        downstream_id: None,
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
        penalties: geometry_zero_penalties(),
    };
    hydro.declare_mirror_unit_group(id);
    hydro
}

/// `geometry_hydro` with caller-chosen `unit_groups` and `generation_model`,
/// for multi-bus [`HydroCellIndex`] fixtures — every other field matches
/// `geometry_hydro` exactly, so a single-group caller gets the identical hydro.
#[must_use]
pub fn geometry_hydro_with_groups(
    idx: usize,
    unit_groups: Vec<HydroUnitGroup>,
    generation_model: HydroGenerationModel,
) -> Hydro {
    let mut hydro = geometry_hydro(idx);
    hydro.unit_groups = unit_groups;
    hydro.generation_model = generation_model;
    hydro.declare_mirror_unit_group(hydro.id);
    // Both calls earn their place: the declare covers a caller passing an empty
    // vec, and the sort supplies the id-ascending `unit_groups` that
    // `HydroCellIndex::build` documents as its precondition for keeping
    // `cell_group_pos` ascending within a cell.
    hydro.sort_unit_groups();
    hydro
}

/// `n` hydros for a model-set constructor test that exercises pure grid or
/// accessor semantics and needs nothing beyond a hydro slice's shape — ids
/// `0..n`, [`geometry_hydro`]'s defaults otherwise.
#[must_use]
pub fn minimal_hydros(n: usize) -> Vec<Hydro> {
    (0..n).map(geometry_hydro).collect()
}

/// Identity [`HydroCellIndex`] for `n_hydros` single-bus hydros
/// (`cells_of(h) == h..h+1` for every `h`) — the single shared builder for every
/// fixture in the crate that needs a `HydroCellIndex` but is not itself testing
/// the multi-bus partition. Safe to over-size: a caller passing more than its
/// fixture's actual hydro count still gets a correct `cells_of` for every hydro
/// it actually queries.
#[must_use]
pub fn identity_hydro_cell_index(n_hydros: usize) -> HydroCellIndex {
    HydroCellIndex::build(&minimal_hydros(n_hydros))
}

/// Fixture bus at system position `idx` carrying exactly `max_deficit_segments`
/// deficit segments — `StageLayout::new` derives its own `max_deficit_segments` as
/// `ctx.buses.iter().map(|b| b.deficit_segments.len()).max()`, so every bus must
/// carry the caller's count for that derivation to reproduce it.
fn geometry_bus(idx: usize, max_deficit_segments: usize) -> Bus {
    Bus {
        id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
        name: String::new(),
        operational_start_date: NaiveDate::default(),
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

/// Fixture thermal at system position `idx`, inert past its `id`: `geometry`
/// needs only the count `ctx.thermals.len()` reserves, never a bound or cost.
fn geometry_thermal(idx: usize) -> Thermal {
    Thermal {
        id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
        name: String::new(),
        operational_start_date: NaiveDate::default(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 0.0,
        min_generation_mw: 0.0,
        max_generation_mw: 0.0,
        anticipated_config: None,
    }
}

/// Fixture line at system position `idx`, inert past its `id`: `geometry`
/// needs only the count `ctx.lines.len()` reserves, never a capacity.
fn geometry_line(idx: usize) -> Line {
    Line {
        id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
        name: String::new(),
        operational_start_date: NaiveDate::default(),
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

/// Single-stage [`ProductionModelSet`]: `Fpha` with `fpha_planes[local]` planes at
/// each `fpha_hydro_indices[local]`, `ConstantProductivity` elsewhere — the exact
/// classification `StageLayout::new`'s FPHA-membership filter reconstructs from
/// `(hydro, stage)`.
fn geometry_production_models(
    hydro_count: usize,
    fpha_hydro_indices: &[usize],
    fpha_planes: &[usize],
) -> ProductionModelSet {
    let mut plane_count: Vec<Option<usize>> = vec![None; hydro_count];
    for (&h, &planes) in fpha_hydro_indices.iter().zip(fpha_planes) {
        if let Some(slot) = plane_count.get_mut(h) {
            *slot = Some(planes);
        }
    }
    let models = plane_count
        .into_iter()
        .map(|planes| {
            vec![planes.map_or(
                ResolvedProductionModel::ConstantProductivity { productivity: 0.0 },
                |n| ResolvedProductionModel::Fpha {
                    planes: vec![
                        FphaPlane {
                            intercept: 0.0,
                            gamma_v: 0.0,
                            gamma_q: 0.0,
                            gamma_s: 0.0,
                        };
                        n
                    ],
                },
            )]
        })
        .collect();
    ProductionModelSet::new(models, &minimal_hydros(hydro_count), 1)
}

/// Single-hydro [`EvaporationModelSet`]: `Linearized` (membership only — no field
/// beyond variant identity reaches `StageLayout::new`) at each `evap_hydro_indices`
/// position, `None` elsewhere.
fn geometry_evaporation_models(
    hydro_count: usize,
    evap_hydro_indices: &[usize],
) -> EvaporationModelSet {
    let mut is_evap = vec![false; hydro_count];
    for &h in evap_hydro_indices {
        if let Some(slot) = is_evap.get_mut(h) {
            *slot = true;
        }
    }
    let models = is_evap
        .into_iter()
        .map(|evap| {
            if evap {
                EvaporationModel::Linearized {
                    coefficients: Vec::new(),
                    reference_volumes_hm3: Vec::new(),
                }
            } else {
                EvaporationModel::None
            }
        })
        .collect();
    EvaporationModelSet::new(models)
}

/// Single-stage [`Stage`] fixture: `n_blks` uniform blocks, [`BlockMode::Parallel`].
fn geometry_stage(n_blks: usize) -> Stage {
    Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::default(),
        end_date: NaiveDate::default(),
        season_id: Some(0),
        blocks: (0..n_blks)
            .map(|i| Block {
                index: i,
                name: String::new(),
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

/// Build the role-(b) [`StageGeometry`] for a single stage from explicit
/// equipment dimensions, FPHA plane counts, and evaporation hydro indices.
///
/// `fpha_hydro_indices` / `fpha_planes` are parallel (equal length). Builds the
/// production `TemplateBuildCtx`/[`StateSpace`]/[`Stage`] the dimensions
/// describe and delegates to `StageLayout::new` — the single owner of the
/// offset arithmetic.
#[must_use]
#[expect(
    clippy::needless_pass_by_value,
    reason = "fpha_hydro_indices/evap_hydro_indices stay owned Vec<usize> — the signature is a stability contract its call sites depend on — even though the body only borrows them (StageLayout::new re-derives the authoritative membership from ctx.hydros/production_models/evaporation_models, not from the caller's raw list)"
)]
pub fn geometry(
    dims: &GeometryDims,
    fpha_hydro_indices: Vec<usize>,
    fpha_planes: &[usize],
    evap_hydro_indices: Vec<usize>,
) -> StageGeometry {
    let hydros = minimal_hydros(dims.hydro_count);
    let hydro_cell_index = HydroCellIndex::build(&hydros);
    let thermals: Vec<Thermal> = (0..dims.n_thermals).map(geometry_thermal).collect();
    let lines: Vec<Line> = (0..dims.n_lines).map(geometry_line).collect();
    let buses: Vec<Bus> = (0..dims.n_buses)
        .map(|idx| geometry_bus(idx, dims.max_deficit_segments))
        .collect();
    let production_models =
        geometry_production_models(dims.hydro_count, &fpha_hydro_indices, fpha_planes);
    let evaporation_models = geometry_evaporation_models(dims.hydro_count, &evap_hydro_indices);

    let anticipated_lead_stages = vec![dims.lead_stages; dims.n_anticipated];

    // Delivery axis wide enough to cover stage 0 + the widest declared lead,
    // matching state_layout_with_transit_buckets's own margin (mirrors
    // AntFixtures::bounds_with_n_stages's widening in entries.rs) — otherwise
    // a genuinely-reachable delivery stage indexes past CtxFixture::default's
    // 1-long bounds/time_value axis.
    let delivery_axis_len = anticipated_lead_stages.iter().copied().max().unwrap_or(0) + 2;
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: delivery_axis_len,
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
        vec![1.0; delivery_axis_len],
        vec![744.0; delivery_axis_len],
        (0..i32::try_from(delivery_axis_len).unwrap_or(0)).collect(),
        PostStudyResolved::default(),
    );

    let par_lp = geometry_par(dims.max_par_order, &hydros);
    let mut fixture = CtxFixture {
        hydros,
        thermals,
        lines,
        buses,
        hydro_cell_index,
        par_lp,
        production_models,
        evaporation_models,
        anticipated_lead_stages,
        anticipated_plants: dims.anticipated_plants.clone(),
        has_penalty: dims.has_inflow_penalty,
        bounds,
        time_value,
        ..CtxFixture::default()
    };
    let mut ctx = fixture.ctx();
    // Geometry allocation never resolves an EntityId through a position map —
    // restore the empty-positions default over the derived (non-empty
    // hydro/bus) one.
    let empty_positions = EntityPositions::from_slices([], [], [], [], [], []);
    ctx.positions = &empty_positions;

    let stage = geometry_stage(dims.n_blks);

    StageLayout::new(&ctx, &stage, 0).geometry
}

/// A PAR(`order`) model for every hydro in `hydros` (finite placeholder
/// coefficients), so a [`CtxFixture`] built over them derives a state with
/// `order` dense lags per hydro.
#[expect(
    clippy::expect_used,
    reason = "geometry_stage carries a season_id, the one input PrecomputedPar::build rejects an AR model for lacking"
)]
fn geometry_par(order: usize, hydros: &[Hydro]) -> PrecomputedPar {
    let hydro_ids: Vec<EntityId> = hydros.iter().map(|hydro| hydro.id).collect();
    let models: Vec<InflowModel> = hydro_ids
        .iter()
        .map(|&hydro_id| InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: 1.0,
            std_m3s: 1.0,
            ar_coefficients: vec![0.1; order],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    PrecomputedPar::build(&models, &[geometry_stage(1)], &hydro_ids, None)
        .expect("geometry_par: placeholder PAR models are valid")
}

/// Build a [`StageGeometry`] carrying only a load-balance row family
/// (`n_buses * n_blks` rows starting at `load_start`), for fixtures that need a
/// non-empty `geometry_per_stage` entry to exercise the load patch.
#[must_use]
pub fn geometry_with_load_balance(
    load_start: usize,
    n_buses: usize,
    n_blks: usize,
) -> StageGeometry {
    StageGeometry {
        load_balance: BlockRowFamily::per_block(load_start..load_start + n_buses * n_blks),
        n_blks,
        ..equipment_free_geometry(&[n_blks]).remove(0)
    }
}

/// One [`StageGeometry`] per entry, the production empty-stage layout
/// (`geometry(&GeometryDims { n_blks, ..zeros }, ..)`) — every column/row
/// family empty, addressing no equipment.
#[must_use]
pub fn equipment_free_geometry(block_counts: &[usize]) -> Vec<StageGeometry> {
    block_counts
        .iter()
        .map(|&n_blks| {
            geometry(
                &GeometryDims {
                    n_blks,
                    ..GeometryDims::default()
                },
                vec![],
                &[],
                vec![],
            )
        })
        .collect()
}

/// Build the [`StageGeometry`] for N=1 hydro, 1 bus, 1 block — the production
/// layout `geometry` builds for those dims. Shared by `pipeline/tests.rs`,
/// `simulation_integration.rs`, and `simulation_pipeline_integration.rs`, each
/// pairing it with a stage template whose per-block hydro/load extraction
/// addresses real columns instead of an empty family.
#[must_use]
pub fn hydro_only_bus_geometry() -> StageGeometry {
    geometry(
        &GeometryDims {
            hydro_count: 1,
            n_buses: 1,
            n_blks: 1,
            ..GeometryDims::default()
        },
        vec![],
        &[],
        vec![],
    )
}

/// Stage template matching [`hydro_only_bus_geometry`]'s N=1 hydro, 1 bus,
/// 1-block layout, so per-block hydro/load extraction addresses real columns
/// and rows instead of an empty family. A mock solver never reads the
/// coefficients, so every column past the state region (`storage_out`(0),
/// `z_inflow`(1), `storage_in`(2), `theta`(3)) is free (zero cost, zero NZ).
#[must_use]
pub fn hydro_only_bus_template() -> StageTemplate {
    let num_cols = 15;
    let num_rows = 7;
    let mut col_lower = vec![0.0; num_cols];
    col_lower[1] = f64::NEG_INFINITY;
    let mut objective = vec![0.0; num_cols];
    objective[3] = 1.0;
    StageTemplate {
        num_cols,
        num_rows,
        num_nz: 0,
        col_starts: vec![0_i32; num_cols + 1],
        row_indices: Vec::new(),
        values: Vec::new(),
        col_lower,
        col_upper: vec![f64::INFINITY; num_cols],
        objective,
        row_lower: vec![0.0; num_rows],
        row_upper: vec![0.0; num_rows],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Fixed [`LpSolution`] for [`hydro_only_bus_template`]'s layout; theta at col 3.
#[must_use]
pub fn hydro_only_bus_solution(objective: f64, theta_val: f64) -> LpSolution {
    let num_cols = 15;
    let mut primal = vec![0.0_f64; num_cols];
    primal[3] = theta_val;
    LpSolution {
        objective,
        primal,
        dual: vec![0.0_f64; 7],
        reduced_costs: vec![0.0_f64; num_cols],
        iterations: 0,
        solve_time_seconds: 0.0,
    }
}

/// Test-only [`StageContext`] builder. Slice fields default to `&[]`; a
/// setter exists only for a field some literal in the crate sets away from
/// that default.
pub struct StageContextFixture<'a> {
    templates: &'a [StageTemplate],
    state_boxes: &'a [StateBox],
    geometry_per_stage: &'a [StageGeometry],
    cost_scale_factor: f64,
    load_bus_indices: &'a [usize],
    ncs_stochastic_dense_col: &'a [usize],
    ncs_stochastic_windows: &'a [(Option<i32>, Option<i32>)],
    ncs_max_gen: &'a [f64],
    ncs_allow_curtailment: &'a [bool],
    discount_factors: &'a [f64],
    cumulative_discount_factors: &'a [f64],
    study_stage_ids: &'a [i32],
    anticipated_windows: &'a [(Option<i32>, Option<i32>)],
    stage_lag_transitions: &'a [StageLagTransition],
    noise_group_ids: &'a [u32],
}

impl<'a> StageContextFixture<'a> {
    /// Borrows `templates`/`state_boxes`/`geometry_per_stage` as given.
    ///
    /// # Panics
    /// Panics if `geometry_per_stage.len() != templates.len()` — every stage
    /// must have one geometry.
    #[must_use]
    pub fn new(
        templates: &'a [StageTemplate],
        state_boxes: &'a [StateBox],
        geometry_per_stage: &'a [StageGeometry],
    ) -> Self {
        assert_eq!(
            geometry_per_stage.len(),
            templates.len(),
            "every stage must have one geometry"
        );
        Self {
            templates,
            state_boxes,
            geometry_per_stage,
            cost_scale_factor: 1_000_000.0,
            load_bus_indices: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            study_stage_ids: &[],
            anticipated_windows: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
        }
    }

    /// [`Self::new`], reading `templates`, `geometry_per_stage`,
    /// `load_bus_indices` and `cost_scale_factor` from `stage_templates` — the
    /// same fields [`StudySetup::stage_ctx`] reads from it.
    ///
    /// # Panics
    /// See [`Self::new`].
    #[must_use]
    pub fn from_stage_templates(
        stage_templates: &'a StageTemplates,
        state_boxes: &'a [StateBox],
    ) -> Self {
        let mut fixture = Self::new(
            &stage_templates.templates,
            state_boxes,
            &stage_templates.geometry_per_stage,
        );
        fixture.load_bus_indices = &stage_templates.load_bus_indices;
        fixture.cost_scale_factor = stage_templates.cost_scale_factor;
        fixture
    }

    /// Sets [`StageContext::load_bus_indices`].
    #[must_use]
    pub fn load_bus_indices(mut self, v: &'a [usize]) -> Self {
        self.load_bus_indices = v;
        self
    }

    /// Sets [`StageContext::ncs_stochastic_dense_col`].
    #[must_use]
    pub fn ncs_stochastic_dense_col(mut self, v: &'a [usize]) -> Self {
        self.ncs_stochastic_dense_col = v;
        self
    }

    /// Sets [`StageContext::ncs_stochastic_windows`].
    #[must_use]
    pub fn ncs_stochastic_windows(mut self, v: &'a [(Option<i32>, Option<i32>)]) -> Self {
        self.ncs_stochastic_windows = v;
        self
    }

    /// Sets [`StageContext::ncs_max_gen`].
    #[must_use]
    pub fn ncs_max_gen(mut self, v: &'a [f64]) -> Self {
        self.ncs_max_gen = v;
        self
    }

    /// Sets [`StageContext::ncs_allow_curtailment`].
    #[must_use]
    pub fn ncs_allow_curtailment(mut self, v: &'a [bool]) -> Self {
        self.ncs_allow_curtailment = v;
        self
    }

    /// Sets [`StageContext::discount_factors`].
    #[must_use]
    pub fn discount_factors(mut self, v: &'a [f64]) -> Self {
        self.discount_factors = v;
        self
    }

    /// Sets [`StageContext::cumulative_discount_factors`].
    #[must_use]
    pub fn cumulative_discount_factors(mut self, v: &'a [f64]) -> Self {
        self.cumulative_discount_factors = v;
        self
    }

    /// Sets [`StageContext::study_stage_ids`].
    #[must_use]
    pub fn study_stage_ids(mut self, v: &'a [i32]) -> Self {
        self.study_stage_ids = v;
        self
    }

    /// Sets [`StageContext::anticipated_windows`].
    #[must_use]
    pub fn anticipated_windows(mut self, v: &'a [(Option<i32>, Option<i32>)]) -> Self {
        self.anticipated_windows = v;
        self
    }

    /// Sets [`StageContext::stage_lag_transitions`].
    #[must_use]
    pub fn stage_lag_transitions(mut self, v: &'a [StageLagTransition]) -> Self {
        self.stage_lag_transitions = v;
        self
    }

    /// Sets [`StageContext::noise_group_ids`].
    #[must_use]
    pub fn noise_group_ids(mut self, v: &'a [u32]) -> Self {
        self.noise_group_ids = v;
        self
    }

    /// Lends a [`StageContext`] borrowing this fixture's fields.
    #[must_use]
    pub fn ctx(&self) -> StageContext<'_> {
        StageContext {
            templates: self.templates,
            state_boxes: self.state_boxes,
            geometry_per_stage: self.geometry_per_stage,
            cost_scale_factor: self.cost_scale_factor,
            load_bus_indices: self.load_bus_indices,
            ncs_stochastic_dense_col: self.ncs_stochastic_dense_col,
            ncs_stochastic_windows: self.ncs_stochastic_windows,
            anticipated_windows: self.anticipated_windows,
            study_stage_ids: self.study_stage_ids,
            ncs_max_gen: self.ncs_max_gen,
            ncs_allow_curtailment: self.ncs_allow_curtailment,
            discount_factors: self.discount_factors,
            cumulative_discount_factors: self.cumulative_discount_factors,
            stage_lag_transitions: self.stage_lag_transitions,
            noise_group_ids: self.noise_group_ids,
        }
    }
}

/// Owns the pieces a [`TrainingContext`] borrows, so a test can lend one
/// without threading a `stochastic`/`node_graph`/`study_dims` triple through
/// every call site.
pub struct TrainingContextFixture {
    state: StateSpace,
    stochastic: StochasticContext,
    node_graph: NodeGraph,
    study_dims: StudyDimensions,
    cut_state_layouts: Vec<CutStateProjection>,
    horizon: HorizonMode,
    initial_state: Vec<f64>,
}

impl TrainingContextFixture {
    /// Single-stage, hydro-free stochastic context and chain node graph;
    /// `state` supplies every state-defining dimension.
    #[must_use]
    pub fn new(state: StateSpace) -> Self {
        let stochastic = hydro_free_stochastic_context(1, 1);
        let node_graph = chain_node_graph(&stochastic);
        let cut_state_layouts = all_enabled_cut_state_layouts(&state, 1);
        Self {
            stochastic,
            node_graph,
            study_dims: study_dims(),
            cut_state_layouts,
            horizon: HorizonMode::Finite { num_stages: 1 },
            initial_state: Vec::new(),
            state,
        }
    }

    /// Sets [`StudyDimensions::downstream_par_order`] on the lent context.
    #[must_use]
    pub fn downstream_par_order(mut self, v: usize) -> Self {
        self.study_dims.downstream_par_order = v;
        self
    }

    /// Sets the lent context's horizon to `HorizonMode::Finite { num_stages }`.
    #[must_use]
    pub fn num_stages(mut self, num_stages: usize) -> Self {
        self.horizon = HorizonMode::Finite { num_stages };
        self
    }

    /// Lends a [`TrainingContext`] borrowing this fixture's fields.
    #[must_use]
    pub fn training_ctx(&self) -> TrainingContext<'_> {
        TrainingContext {
            horizon: &self.horizon,
            state: &self.state,
            cut_state_layouts: &self.cut_state_layouts,
            study_dims: &self.study_dims,
            inflow_method: &self.study_dims.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            node_graph: &self.node_graph,
        }
    }
}

/// Build a finalized storage+lag [`StateSpace`] (no anticipated thermals) with the
/// full `max_par_order` lag stride for every hydro — the dense coverage
/// `crate::setup::resolve_state_layout` finalizes with no per-hydro AR truncation.
#[must_use]
pub fn state_layout(hydro_count: usize, max_par_order: usize) -> StateSpace {
    state_layout_full(hydro_count, max_par_order, Vec::new())
}

/// Build a finalized [`StateSpace`] from explicit state-vector dimensions,
/// including anticipated thermals. Lag coverage is dense (full `max_par_order`).
#[must_use]
pub fn state_layout_full(
    hydro_count: usize,
    max_par_order: usize,
    anticipated_lead_stages: Vec<usize>,
) -> StateSpace {
    state_layout_with_transit_buckets(
        hydro_count,
        max_par_order,
        Vec::new(),
        anticipated_lead_stages,
    )
}

/// Build a finalized [`StateSpace`] with a declared travel-time bucket block
/// (`transit_buckets_out`/`transit_buckets_in`), optionally combined with anticipated
/// thermals. `effective_lag_count` is dense (full `max_par_order` for every
/// hydro), matching [`state_layout_full`]. Attaches a [`constant_lead_resolution`]
/// over a margin wide enough (`max(lead) + 2`) to saturate the commitment-hold
/// mask to the whole region; [`state_layout_with_transit_buckets_and_resolution`]
/// is the sibling for a caller that needs a specific reachability shape.
#[must_use]
pub fn state_layout_with_transit_buckets(
    hydro_count: usize,
    max_par_order: usize,
    transit_bucket_column_order: Vec<(HydroSys, usize)>,
    anticipated_lead_stages: Vec<usize>,
) -> StateSpace {
    let n_stages = anticipated_lead_stages.iter().copied().max().unwrap_or(0) + 2;
    let resolution = constant_lead_resolution(&anticipated_lead_stages, n_stages);
    state_layout_with_transit_buckets_and_resolution(
        hydro_count,
        max_par_order,
        transit_bucket_column_order,
        anticipated_lead_stages,
        resolution,
    )
}

/// Like [`state_layout_with_transit_buckets`] but with a caller-supplied
/// [`AnticipatedResolution`] instead of the saturating [`constant_lead_resolution`]
/// default.
#[must_use]
pub fn state_layout_with_transit_buckets_and_resolution(
    hydro_count: usize,
    max_par_order: usize,
    transit_bucket_column_order: Vec<(HydroSys, usize)>,
    anticipated_lead_stages: Vec<usize>,
    anticipated_resolution: AnticipatedResolution,
) -> StateSpace {
    let effective_lag_count = vec![max_par_order; hydro_count];
    StateSpace::new(
        hydro_count,
        max_par_order,
        transit_bucket_column_order,
        anticipated_lead_stages,
        anticipated_resolution,
        &effective_lag_count,
    )
}

/// Per-hydro inflow `extract_hydros`/`extract_hydro_bus_generation` read from
/// `StageExtractionSpec::inflow_m3s_per_hydro`: the lag-0 incoming column when
/// `state` carries PAR lags, `0.0` otherwise.
#[cfg(test)]
#[must_use]
pub(crate) fn inflow_m3s_per_hydro_from_primal(
    state: &StateSpace,
    primal: &[f64],
    n_hydros: usize,
) -> Vec<f64> {
    (0..n_hydros)
        .map(|h| {
            if state.max_par_order > 0 {
                primal[state.lag_incoming_col(0, HydroSys::new(h)).get()]
            } else {
                0.0
            }
        })
        .collect()
}

/// Bucket-only [`StageTemplate`]: `num_cols` free columns, zero rows.
#[must_use]
pub fn transit_bucket_only_template(num_cols: usize, n_state: usize) -> StageTemplate {
    StageTemplate {
        num_cols,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0_i32; num_cols + 1],
        row_indices: Vec::new(),
        values: Vec::new(),
        col_lower: vec![f64::NEG_INFINITY; num_cols],
        col_upper: vec![f64::INFINITY; num_cols],
        objective: vec![0.0; num_cols],
        row_lower: Vec::new(),
        row_upper: Vec::new(),
        n_state,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build the all-enabled per-pool [`CutStateProjection`] vector (one per stage): every
/// pool projects the full global state (`n_state() == global.n_state` for all `t`),
/// keeping the extracted subgradient bit-identical to the unprojected global loop.
#[must_use]
pub fn all_enabled_cut_state_layouts(
    global: &StateSpace,
    n_stages: usize,
) -> Vec<CutStateProjection> {
    (0..n_stages)
        .map(|_| cut_state_projection(global))
        .collect()
}

/// Build a single all-enabled [`CutStateProjection`] projecting the full global state.
#[must_use]
pub fn cut_state_projection(global: &StateSpace) -> CutStateProjection {
    CutStateProjection::new(
        global,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    )
}

/// Build an all-default [`StudyDimensions`] (every count `0`, every flag
/// `false`, empty anticipated list).
#[must_use]
pub fn study_dims() -> StudyDimensions {
    StudyDimensions::default()
}

/// Build the [`StudyDimensions`] matching the [`GeometryDims`] a test built its
/// stage geometry from.
#[must_use]
pub fn study_dims_for(dims: &GeometryDims) -> StudyDimensions {
    StudyDimensions {
        max_deficit_segments: dims.max_deficit_segments,
        inflow_method: if dims.has_inflow_penalty {
            crate::InflowNonNegativityMethod::Penalty
        } else {
            crate::InflowNonNegativityMethod::None
        },
        anticipated_plants: dims.anticipated_plants.clone(),
        downstream_par_order: 0,
    }
}

/// Trivial matching [`PolicyLoadProof`] typed to [`FullFcf`] for tests
/// exercising FCF reconstruction rather than cross-study compatibility: identical
/// `state_dimension`/`num_stages` on both sides with an empty manifest (the
/// "identity could not be verified" warning path). [`validate_policy_load`] is
/// the only constructor of [`PolicyLoadProof`], so tests route through it here
/// rather than a forged literal.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[expect(
    clippy::expect_used,
    reason = "matching state_dimension/num_stages with an empty manifest on both sides cannot hit validate_policy_load's error paths (state_dimension and num_stages equality hold trivially; an empty manifest short-circuits identity comparison with a warning, never an error)"
)]
#[must_use]
pub fn trivial_full_fcf_proof(state_dimension: u32, num_stages: u32) -> PolicyLoadProof<FullFcf> {
    let graph = cobre_io::GraphManifest::default();
    let manifest = PolicyStageManifest {
        state_dimension,
        num_stages,
        n_pools: num_stages,
        slots: &[],
        graph: &graph,
    };
    validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &manifest, &manifest)
        .expect("trivial matching manifest cannot fail validate_policy_load")
}

/// Assemble the [`CheckpointManifest`] for a checkpoint fixture: the three
/// invariant fields (`format_version` = [`FORMAT_VERSION`], `cobre_version`
/// matching the production writer, a fixed `created_at` no consumer reads) are
/// filled here — the sole owner of the manifest literal for `cobre-sddp`
/// tests. `season_manifest` defaults absent; a caller exercising the
/// boundary-load season/PAR-identity gate
/// (`policy::policy_load::check_season_compatibility`) overrides it via
/// struct-update syntax on the returned value.
///
/// [`CheckpointManifest`]: cobre_io::CheckpointManifest
/// [`FORMAT_VERSION`]: cobre_io::FORMAT_VERSION
#[must_use]
pub fn checkpoint_metadata(
    num_stages: u32,
    graph_manifest: cobre_io::GraphManifest,
    producer: cobre_io::ProducerBlock,
) -> cobre_io::CheckpointManifest {
    cobre_io::CheckpointManifest {
        format_version: cobre_io::FORMAT_VERSION,
        cobre_version: POLICY_COBRE_VERSION.to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        num_stages,
        graph_manifest,
        producer,
        season_manifest: cobre_io::SeasonManifest::default(),
    }
}

/// `YYYY-MM-DD` as a [`NaiveDate`], for a fixture's literal calendar date.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[expect(
    clippy::expect_used,
    reason = "every caller passes a literal, calendar-valid date; from_ymd_opt only returns None for an out-of-range one"
)]
#[must_use]
pub fn ymd(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).expect("valid calendar date")
}

/// `base` plus `pool` months — a multi-pool checkpoint fixture's per-pool
/// `priced_state_date`, distinct and ascending across pools. `base` is the
/// caller's own epoch, never a shared constant: a boundary-date-selection
/// fixture is sensitive to which pool's date is nearest the boundary, so
/// unifying the epoch across fixtures would silently move which pool a
/// date-driven selector resolves.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[expect(
    clippy::expect_used,
    reason = "every caller passes a small pool count; checked_add_months only overflows past NaiveDate's year range"
)]
#[must_use]
pub fn fixture_priced_date(base: NaiveDate, pool: u32) -> NaiveDate {
    base.checked_add_months(chrono::Months::new(pool))
        .expect("in-range fixture date")
}

/// The following month's day-01 `YYYYMMDD` anchor of `month_anchor` (itself a
/// day-01 anchor), routed through the production date codec
/// ([`decode_slot_date`]/[`encode_slot_date`]) rather than hand-rolled
/// packed-integer arithmetic.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[expect(
    clippy::expect_used,
    reason = "every caller passes a day-01 YYYYMMDD anchor in range; decoding then re-encoding one only fails on a malformed or out-of-range stamp"
)]
#[must_use]
pub fn next_month_anchor(month_anchor: i32) -> i32 {
    encode_slot_date(
        decode_slot_date(month_anchor)
            .expect("day-01 YYYYMMDD anchor")
            .checked_add_months(chrono::Months::new(1))
            .expect("in-range anchor"),
    )
}

/// A minimal, all-zeroed [`ProducerBlock`] for artifact-writing test fixtures.
/// Callers override the fields their own fixture cares about via struct-update
/// syntax.
#[must_use]
pub fn producer_block() -> ProducerBlock {
    ProducerBlock {
        completed_iterations: 0,
        final_lower_bound: 0.0,
        best_upper_bound: None,
        max_iterations: 0,
        forward_passes: 0,
        warm_start_cuts: 0,
        warm_start_counts: vec![],
        rng_seed: 0,
        total_visited_states: 0,
        training_block_mode: "parallel".to_string(),
        training_block_mode_per_stage: vec![],
        cost_scale_factor: None,
    }
}

/// A 1:1 chain [`GraphManifest`] over `n_stages` nodes (node id == stage id
/// == pool id) — the shape a chain-degenerate study writes.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[expect(
    clippy::expect_used,
    reason = "every caller passes a small stage count; the u32->i32 casts only fail past i32::MAX stages"
)]
#[must_use]
pub fn chain_graph_manifest(n_stages: u32) -> GraphManifest {
    let nodes = (0..n_stages)
        .map(|t| ManifestNode {
            id: i32::try_from(t).expect("small stage count"),
            stage_id: i32::try_from(t).expect("small stage count"),
            pool_id: t,
        })
        .collect();
    let edges = (0..n_stages.saturating_sub(1))
        .map(|t| ManifestEdge {
            source_id: i32::try_from(t).expect("small stage count"),
            target_id: i32::try_from(t + 1).expect("small stage count"),
            probability: 1.0,
        })
        .collect();
    GraphManifest {
        n_pools: n_stages,
        nodes,
        edges,
    }
}

/// Write a synthetic single-cut boundary checkpoint carrying `intercept` and
/// the explicit per-slot `coefficients`, priced at `priced_state_date`. No
/// entity manifest (`&[]`): the loader's identity check short-circuits with a
/// warning, so this controls only the state dimension, never entity-identity
/// matching.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[expect(
    clippy::expect_used,
    reason = "write_policy_checkpoint only fails on a write-path IO error, never on this fixture's own well-formed payload"
)]
pub fn write_synthetic_boundary(
    dir: &Path,
    state_dimension: u32,
    intercept: f64,
    coefficients: &[f64],
    priced_state_date: NaiveDate,
) {
    let cuts = vec![PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept,
        coefficients,
        is_active: true,
    }];
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension,
        capacity: 1,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: &[],
        cost_scale_factor: 1_000_000.0,
        node_id: 100,
        graph_stage_id: -1,
        priced_state_date: encode_slot_date(priced_state_date),
    };
    let metadata = checkpoint_metadata(
        1,
        GraphManifest {
            n_pools: 1,
            nodes: vec![ManifestNode {
                id: 100,
                stage_id: 0,
                pool_id: 0,
            }],
            edges: vec![],
        },
        ProducerBlock {
            cost_scale_factor: Some(1.0),
            ..producer_block()
        },
    );
    write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
}

/// A single active `HydroStorage` slot.
#[must_use]
pub fn storage_slot(id: i32) -> EntitySlot {
    EntitySlot::storage(id, true)
}

/// A single active, sentinel-dated `HydroInflowLag` slot; `lag_depth` is
/// 1-based.
#[must_use]
pub fn inflow_lag_slot(id: i32, lag_depth: u32) -> EntitySlot {
    EntitySlot::inflow_lag(id, lag_depth, true)
}

/// Like [`inflow_lag_slot`] but carrying `reference_date` instead of the
/// sentinel.
#[must_use]
pub fn inflow_lag_slot_at(id: i32, lag_depth: u32, reference_date: i32) -> EntitySlot {
    EntitySlot::inflow_lag(id, lag_depth, true).with_reference_date(reference_date)
}

/// A single active, sentinel-dated `HydroTransitBucket` slot; `id` is the
/// DOWNSTREAM hydro, `lag` the maturity subindex.
#[must_use]
pub fn transit_bucket_slot(id: i32, lag: u32) -> EntitySlot {
    EntitySlot::transit_bucket(id, lag, true)
}

/// Like [`transit_bucket_slot`] but carrying a real `[start, end)` arrival
/// interval instead of the sentinel.
#[must_use]
pub fn transit_bucket_slot_over(id: i32, lag: u32, start: i32, end: i32) -> EntitySlot {
    transit_bucket_slot(id, lag).with_interval(start, end)
}

/// A single active, sentinel-dated `AnticipatedThermalState` slot.
#[must_use]
pub fn anticipated_slot(thermal_id: i32, ring_slot: u32) -> EntitySlot {
    EntitySlot::anticipated(thermal_id, ring_slot, true)
}

/// Like [`anticipated_slot`] but carrying its own calendar-month
/// `[month_anchor, next_month_anchor(month_anchor))` delivery interval.
#[must_use]
pub fn anticipated_slot_at(thermal_id: i32, ring_slot: u32, month_anchor: i32) -> EntitySlot {
    EntitySlot::anticipated(thermal_id, ring_slot, true)
        .with_interval(month_anchor, next_month_anchor(month_anchor))
}

/// Like [`anticipated_slot`] but carrying an explicit `[start, end)`
/// interval instead of one derived from [`next_month_anchor`].
#[must_use]
pub fn anticipated_slot_over(thermal_id: i32, ring_slot: u32, start: i32, end: i32) -> EntitySlot {
    EntitySlot::anticipated(thermal_id, ring_slot, true).with_interval(start, end)
}

/// Patch one stage-LP solve exactly as the production backward pass's
/// `patch_opening_bounds` does (`training/backward/lp_setup.rs`): delegates
/// verbatim to `StageSolvePrep::run` with the backward-opening variation
/// point (`InflowNoise::Transform`) — no probe-side reimplementation of the
/// patch pipeline (the z-inflow column's RHS, NCS availability).
pub fn patch_backward_opening_for_probe<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    stage: StageIdx,
    pinned_state: &[f64],
    raw_noise: &[f64],
) {
    let prep_params = StageSolvePrepParams {
        state_source: StateSource(pinned_state),
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        stage,
        &prep_params,
    );
}

/// Variant of [`patch_backward_opening_for_probe`] for a deliberate
/// counterfactual pin that lies outside its producer's admissible box on
/// purpose (an LP-sensitivity probe measuring the response to a value the
/// seam would never itself produce): same pipeline, but skips the pin-time
/// box-membership assert the production path always runs.
pub fn patch_backward_opening_for_counterfactual_probe<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    stage: StageIdx,
    pinned_state: &[f64],
    raw_noise: &[f64],
) {
    let prep_params = StageSolvePrepParams {
        state_source: StateSource(pinned_state),
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run_ignoring_producer_box(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        stage,
        &prep_params,
    );
}

/// Run one stage-LP solve exactly as production's shared `run_stage_solve`
/// entry point does (`solve::stage_solve`, `pub(crate)` — the module
/// forward/backward/simulation all route through, no external consumer by
/// design): basis reconstruction by cut-pool slot identity when
/// `stored_basis` is `Some`, else an implicit warm start from whatever the
/// solver instance currently retains. Reachable here so a probe can drive the
/// identical hot path on a scratch workspace without duplicating its
/// reconstruction/invariant-enforcement logic.
///
/// # Errors
///
/// Propagates the same [`SddpError`] variants as production's stage solve
/// (`Infeasible`, `Solver`, or a basis-shape mismatch).
pub fn solve_stage_for_probe<'ws, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    stage_context: &StageContext<'_>,
    pool: &CutPool,
    stored_basis: Option<&CapturedBasis>,
    stage_index: StageIdx,
    scenario_index: usize,
    node_id: NodeId,
) -> Result<SolutionView<'ws>, SddpError> {
    let inputs = StageInputs {
        stage_context,
        pool,
        stored_basis,
        stage_index,
        scenario_index,
        iteration: None,
        node_id,
    };
    run_stage_solve(ws, &inputs)
}

/// The three observations [`write_backward_opening_outcome_for_probe`] returns.
pub struct CanonicalCutProbe {
    /// The cut `write_opening_outcome` wrote for opening 0 (coefficients +
    /// intercept + objective), read back — never recomputed.
    pub outcome: BackwardOutcome,
    /// The producer-stage outgoing state after `assemble_outgoing_state`'s
    /// clamp: the value the read-back seam canonicalizes the raw input to.
    pub canonical_x_hat: Vec<f64>,
    /// The state the successor LP was actually pinned at, recovered from the
    /// solved primal of the pinned incoming-state columns
    /// (`state_to_lp_incoming_column`, `lb == ub`) and unscaled by `col_scale`.
    /// Its data path — set-bounds, solve, read primal — is independent of the
    /// `x_hat` handed to `write_opening_outcome`, so the returned cut and pin
    /// can be cross-checked rather than tautologically re-derived.
    pub pinned_x_hat: Vec<f64>,
}

/// Canonicalize a RAW producer-stage state exactly as the forward/simulation
/// read-back seam does (`raw_producer_state`, possibly outside the producer's
/// admissible box; [`assemble_outgoing_state`] performs the clamp), then
/// delegate to [`write_backward_opening_outcome_at_canonical_state_for_probe`]
/// with the resulting canonical `x_hat`.
///
/// # Panics
///
/// Panics if `stage.0 == 0` (the backward never solves stage 0, so no producer
/// box exists) or if `raw_producer_state.len() != StateSpace::n_state`.
///
/// # Errors
///
/// Propagates [`SddpError`] from the stage solve.
pub fn write_backward_opening_outcome_for_probe<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    cut_pool: &CutPool,
    cut_state: &CutStateProjection,
    stage: StageIdx,
    node_id: NodeId,
    raw_producer_state: &[f64],
    raw_noise: &[f64],
) -> Result<CanonicalCutProbe, SddpError> {
    assert!(
        stage.0 > 0,
        "write_backward_opening_outcome_for_probe: stage must be >= 1"
    );
    let layout = training_ctx.state;
    assert_eq!(
        raw_producer_state.len(),
        layout.n_state,
        "raw_producer_state must have one entry per state dimension"
    );
    let producer_stage = StageIdx(stage.0 - 1);

    let mut unscaled_primal =
        vec![0.0_f64; ctx.template(producer_stage).num_cols.max(layout.n_state)];
    unscaled_primal[..layout.n_state].copy_from_slice(raw_producer_state);

    let mut canonical_state: Vec<f64> = Vec::new();
    let mut lag_accumulator = vec![0.0_f64; layout.hydro_count.max(1)];
    let mut lag_weight_accum = vec![0.0_f64; layout.hydro_count.max(1)];
    let incoming_lags = vec![0.0_f64; layout.hydro_count * layout.max_par_order];
    let ds_par_order = training_ctx.study_dims.downstream_par_order;
    let mut ds_accumulator = vec![
        0.0_f64;
        if ds_par_order > 0 {
            layout.hydro_count
        } else {
            0
        }
    ];
    let mut ds_weight_accum = 0.0_f64;
    let mut ds_completed_lags = vec![0.0_f64; ds_par_order * layout.hydro_count];
    let mut ds_n_completed = 0_usize;
    assemble_outgoing_state(
        &mut canonical_state,
        &unscaled_primal,
        &incoming_lags,
        layout,
        ctx.state_box(producer_stage),
        ctx.stage_lag(producer_stage),
        &mut LagAccumState {
            accumulator: &mut lag_accumulator,
            weight_accum: &mut lag_weight_accum,
        },
        &mut DownstreamAccumState {
            accumulator: &mut ds_accumulator,
            weight_accum: &mut ds_weight_accum,
            completed_lags: &mut ds_completed_lags,
            n_completed: &mut ds_n_completed,
            par_order: ds_par_order,
        },
    );
    let canonical_x_hat = canonical_state[..layout.n_state].to_vec();

    write_backward_opening_outcome_at_canonical_state_for_probe(
        ws,
        ctx,
        training_ctx,
        cut_pool,
        cut_state,
        stage,
        node_id,
        &canonical_x_hat,
        raw_noise,
    )
}

/// Drive the real read-back canonicalization and the real backward
/// cut-intercept write for one opening at an already-canonical state `x̂`.
///
/// That canonical value is the single `x_hat` threaded into BOTH the pin
/// ([`patch_backward_opening_for_probe`] → `StageSolvePrep::run` →
/// `set_col_bounds`) and the intercept ([`write_opening_outcome`]), mirroring
/// the backward opening loop. The state the LP was pinned at is recovered
/// separately from the solved primal — an independent data path — so
/// [`CanonicalCutProbe::pinned_x_hat`] cross-checks the pin against the cut
/// instead of re-deriving the intercept's own formula.
///
/// # Panics
///
/// Panics if `canonical_x_hat.len() != StateSpace::n_state`.
///
/// # Errors
///
/// Propagates [`SddpError`] from the stage solve.
pub fn write_backward_opening_outcome_at_canonical_state_for_probe<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    cut_pool: &CutPool,
    cut_state: &CutStateProjection,
    stage: StageIdx,
    node_id: NodeId,
    canonical_x_hat: &[f64],
    raw_noise: &[f64],
) -> Result<CanonicalCutProbe, SddpError> {
    let layout = training_ctx.state;
    assert_eq!(
        canonical_x_hat.len(),
        layout.n_state,
        "canonical_x_hat must have one entry per state dimension"
    );

    let template = ctx.template(stage);
    ws.solver.reset_solver_state();
    ws.solver.load_model(template);
    patch_backward_opening_for_probe(ws, ctx, training_ctx, stage, canonical_x_hat, raw_noise);

    let mut stats_before = SolverStatistics::default();
    ws.solver.statistics_into(&mut stats_before);
    let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
    let view = solve_stage_for_probe(ws, ctx, cut_pool, None, stage, 0, node_id)?;

    let col_scale = &template.col_scale;
    let objective = extract_state_duals_only(&view, cut_state, col_scale, &mut state_duals);
    let pinned_x_hat: Vec<f64> = (0..layout.n_state)
        .map(|j| {
            let col = layout.state_to_lp_incoming_column(StateDim::new(j)).get();
            view.primal[col] * col_scale.get(col).copied().unwrap_or(1.0)
        })
        .collect();
    ws.backward_accum.state_duals_buf = state_duals;

    let mut stats_after = SolverStatistics::default();
    ws.solver.statistics_into(&mut stats_after);

    let n_slots = cut_state.n_slots();
    if ws.backward_accum.outcomes.is_empty() {
        ws.backward_accum.outcomes.push(BackwardOutcome {
            intercept: 0.0,
            coefficients: vec![0.0; n_slots],
            objective_value: 0.0,
        });
    }
    ws.backward_accum.outcomes[0]
        .coefficients
        .resize(n_slots, 0.0);
    if ws.backward_accum.per_opening_stats.is_empty() {
        ws.backward_accum
            .per_opening_stats
            .push(SolverStatsDelta::default());
    }

    write_opening_outcome(
        ws,
        cut_state,
        0,
        objective,
        canonical_x_hat,
        &stats_before,
        &stats_after,
    );

    Ok(CanonicalCutProbe {
        outcome: ws.backward_accum.outcomes[0].clone(),
        canonical_x_hat: canonical_x_hat.to_vec(),
        pinned_x_hat,
    })
}

/// Trial-point states in the flat shape the passes index, `records[m * n_stages + stage]`.
///
/// Each scenario's state is replicated across every stage, so the backward pass's
/// per-stage repack of the gather buffers reproduces exactly `states` at whichever
/// stage it runs — letting a test pin cut arithmetic against a known trial point.
#[must_use]
pub fn trial_state_records(states: &[Vec<f64>], n_stages: usize) -> Vec<TrajectoryRecord> {
    states
        .iter()
        .flat_map(|state| {
            (0..n_stages).map(move |_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: state.clone(),
            })
        })
        .collect()
}

/// The byte-exact chain [`NodeGraph`] for `stochastic`: one node per
/// stage, pools 1:1, each node's `Ω` view spanning exactly that stage's
/// [`StochasticContext::opening_tree`] openings. Delegates to
/// [`build_node_graph`]'s own chain path (an empty `nodes[]`) instead of
/// re-deriving the shape, so a fixture's declared branching factor and its
/// node graph can never drift apart.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[expect(
    clippy::expect_used,
    reason = "build_node_graph only returns Err for a transition/node naming an undeclared id; HorizonGraph::default() carries no nodes/transitions at all, so that error path is unreachable here"
)]
#[must_use]
pub fn chain_node_graph(stochastic: &StochasticContext) -> NodeGraph {
    let n_stages = stochastic.n_stages();
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "n_stages is a small fixture stage count, far below i32::MAX"
    )]
    let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
    let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    build_node_graph(&HorizonGraph::default(), n_stages, &resolver, stochastic)
        .expect("chain_node_graph: build_node_graph never errors for an empty nodes[] graph")
}

/// Test-support view of the crate-internal per-node prefix count `π(n)`
/// ([`enumerated_node_visit_counts`]) — the value oracle's expander-coverage
/// self-check compares the expander's one-copy-per-node construction against it.
///
/// # Errors
///
/// Propagates the overflow [`SddpError::Validation`] the underlying counter
/// returns on a `u64` path-product overflow.
pub fn node_prefix_counts(graph: &NodeGraph) -> Result<Vec<u64>, SddpError> {
    enumerated_node_visit_counts(graph).map(|counts| counts.into_iter().collect())
}

/// Test-support view of the crate-internal enumerated scenario count
/// ([`enumerated_scenario_count`]) — the root→leaf path count the self-check
/// matches against the graph's leaf count.
///
/// # Errors
///
/// Propagates the overflow [`SddpError::Validation`] the underlying counter
/// returns on a `u64` path-product overflow.
pub fn node_scenario_count(graph: &NodeGraph) -> Result<u64, SddpError> {
    enumerated_scenario_count(graph)
}

// ── DECOMP K-fan fixture ─────────────────────────────────────────────────

/// Fixed training seed for [`k_fan_setup`] — every caller trains the identical,
/// reproducible sampled walk; not a caller-configurable knob.
const K_FAN_SEED: u64 = 42;
/// [`K_FAN_SEED`], typed for [`Config`]'s `training.tree_seed` field.
const K_FAN_TREE_SEED: i64 = 42;

/// Study-stage id of the K-fan's root (a single node, the sole stage-0 alive node).
const K_FAN_ROOT_STAGE_ID: i32 = 0;
/// Study-stage id of the fan level: `k` distinct nodes, each cut-generating (each
/// owns exactly one leaf successor).
const K_FAN_BRANCH_STAGE_ID: i32 = 1;
/// Study-stage id of the leaf level: `k` distinct nodes sharing ONE pool
/// (`build_node_graph`'s leaf-sharing rule) — never cut-generating.
const K_FAN_LEAF_STAGE_ID: i32 = 2;

/// The all-in-sample [`ClassSchemes`]: every fixture whose stochastic context
/// draws every noise class from the in-sample library shares this literal.
fn in_sample_class_schemes() -> ClassSchemes {
    ClassSchemes {
        inflow: Some(SamplingScheme::InSample),
        load: Some(SamplingScheme::InSample),
        ncs: Some(SamplingScheme::InSample),
    }
}

/// A hydro-free [`StochasticContext`] over `n_stages` single-block stages,
/// each with the given `branching_factor` — one deficit-fallback bus, no hydros.
///
/// # Panics
///
/// Never in practice: the system and stochastic literals built here are
/// fixed and internally consistent.
#[expect(
    clippy::expect_used,
    reason = "the system and stochastic literals built here are fixed and internally consistent, so SystemBuilder::build/build_stochastic_context never return their error paths"
)]
#[must_use]
pub fn hydro_free_stochastic_context(
    n_stages: usize,
    branching_factor: usize,
) -> StochasticContext {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: ymd(2024, 1, 1),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "idx is a small fixture stage index, far below i32::MAX"
    )]
    let make_stage = |idx: usize| Stage {
        index: idx,
        id: idx as i32,
        start_date: ymd(2024, 1, 1),
        end_date: ymd(2024, 2, 1),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    };
    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles: BTreeMap::new(),
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .stages(stages)
        .correlation(correlation)
        .build()
        .expect("hydro_free_stochastic_context: valid study");
    build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("hydro_free_stochastic_context: build_stochastic_context must succeed")
}

/// Reverse `nodes`/`transitions` in place when `reversed`: `build_node_graph`
/// sorts nodes by id and out-edges by target, so this must recover the
/// identical canonical graph — the shared input every declaration-order-
/// invariance fixture in this module builds from.
fn reverse_declaration_order_if(
    reversed: bool,
    nodes: &mut [PolicyNode],
    transitions: &mut [Transition],
) {
    if reversed {
        nodes.reverse();
        transitions.reverse();
    }
}

fn finite_horizon_graph(nodes: Vec<PolicyNode>, transitions: Vec<Transition>) -> HorizonGraph {
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions,
        nodes,
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    }
}

/// The declared `nodes[]`/`transitions[]` K-fan: root (id `0`) branches into fan
/// nodes `1..=k` under strictly non-uniform weights `i / Σj` (never a uniform
/// `1/k` split — a uniform split would make every reduction order sum identical
/// terms, defeating the canonical-order gate's power), each fan node `i` then
/// deterministically reaching its own leaf `k+i`. `num_nodes = 2k+1`,
/// `n_pools = k+2` (root + k fan pools + one shared leaf pool) —
/// `num_nodes > n_pools` for every `k >= 2`, so a canonical-node-position-as-pool-id
/// conflation bug would misroute or overflow a pool here, where it cannot on a
/// chain (`node_index == pool_id` there hides the bug).
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "k is a small fixture fan width, far below i32::MAX and below f64's exact-integer range"
)]
fn k_fan_policy_graph(k: usize, reversed: bool) -> HorizonGraph {
    debug_assert!(k >= 2, "k_fan_policy_graph: k must be >= 2 (DECOMP shape)");
    let mut nodes = Vec::with_capacity(1 + 2 * k);
    let mut transitions = Vec::with_capacity(2 * k);
    nodes.push(PolicyNode {
        id: 0,
        stage_id: K_FAN_ROOT_STAGE_ID,
        scenario_id: None,
        label: None,
    });
    let total_weight: f64 = (1..=k).map(|i| i as f64).sum();
    for i in 1..=k {
        let fan_id = i as i32;
        let leaf_id = fan_id + k as i32;
        nodes.push(PolicyNode {
            id: fan_id,
            stage_id: K_FAN_BRANCH_STAGE_ID,
            scenario_id: None,
            label: None,
        });
        nodes.push(PolicyNode {
            id: leaf_id,
            stage_id: K_FAN_LEAF_STAGE_ID,
            scenario_id: None,
            label: None,
        });
        transitions.push(Transition {
            source_id: 0,
            target_id: fan_id,
            probability: (i as f64) / total_weight,
            annual_discount_rate_override: None,
        });
        transitions.push(Transition {
            source_id: fan_id,
            target_id: leaf_id,
            probability: 1.0,
            annual_discount_rate_override: None,
        });
    }
    reverse_declaration_order_if(reversed, &mut nodes, &mut transitions);
    finite_horizon_graph(nodes, transitions)
}

/// The default `state_config` every [`fan_or_chain_system_ext`] caller gets when
/// it passes `None` for `stage_state_configs`: storage-only, matching the
/// original K-fan/chain fixtures byte-for-byte.
const K_FAN_DEFAULT_STATE_CONFIG: StageStateConfig = StageStateConfig {
    storage: true,
    inflow_lags: false,
};

/// One study stage at `(index, id)` with a single 744h block, `state_config` as
/// given, `branching_factor: 1` — every scale/routing signal in the K-fan comes
/// from the declared node branching, never from within-node opening variance.
fn k_fan_stage(index: usize, id: i32, state_config: StageStateConfig) -> Stage {
    Stage {
        index,
        id,
        start_date: ymd(2024, 1, 1),
        end_date: ymd(2024, 2, 1),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config,
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }
}

/// A single-hydro, single-bus [`System`] over the 3-stage K-fan calendar: hydro
/// storage/inflow/turbine dynamics against a bus deficit fallback, so every
/// visited node solves a genuine (non-degenerate) LP with real dual activity.
pub(crate) fn k_fan_system(k: usize, reversed: bool) -> System {
    fan_or_chain_system(3, k_fan_policy_graph(k, reversed))
}

/// Shared single-hydro/single-bus study over `n_stages` stages (each a 744h
/// block, `branching_factor: 1`), driven by an explicit `policy_graph` — the
/// declared K-fan (`n_stages == 3`) or, with an empty graph, the chain
/// degeneracy the enumerated engine's single-path (count-1) 2-rank stub needs.
/// Every stage gets [`K_FAN_DEFAULT_STATE_CONFIG`] — see [`fan_or_chain_system_ext`]
/// for a caller that varies it.
fn fan_or_chain_system(n_stages: usize, policy_graph: HorizonGraph) -> System {
    fan_or_chain_system_ext(
        n_stages,
        policy_graph,
        0.0,
        0.0,
        StorageSpec::wide(),
        Vec::new(),
        Vec::new(),
        None,
        &[],
    )
}

/// Reservoir sizing for [`fan_or_chain_system_ext`]. [`StorageSpec::wide`] keeps
/// the chain/fan fixtures' historic `200`/`100` hm³ envelope (storage never binds);
/// a scarce cap makes each leaf's own inflow the binding swing factor instead of
/// inherited root storage — the water-binding value-red's precondition.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StorageSpec {
    /// Reservoir capacity (hm³) — the hydro's and the resolved bounds' `max_storage_hm3`.
    pub(crate) max_storage_hm3: f64,
    /// Initial reservoir volume (hm³) — the study's stage-0 incoming storage.
    pub(crate) initial_storage_hm3: f64,
}

impl StorageSpec {
    /// The historic non-binding envelope every pre-water-binding fixture uses.
    pub(crate) fn wide() -> Self {
        Self {
            max_storage_hm3: 200.0,
            initial_storage_hm3: 100.0,
        }
    }
}

/// [`fan_or_chain_system`] with explicit inflow/load standard deviations, reservoir
/// sizing, external inflow + load scenario rows, and per-stage `state_config`
/// overrides — the seam the External-openings fixtures and the non-uniform
/// cut-state-projection branching fixture both build on. A positive `load_std`
/// makes the bus a stochastic load class (required for its external-load library
/// to carry a column); `0.0` leaves the chain's deterministic load unchanged.
/// `stage_state_configs`, when `Some`, supplies one [`StageStateConfig`] per stage
/// (length `n_stages`); `None` keeps every stage at [`K_FAN_DEFAULT_STATE_CONFIG`],
/// byte-identical to every caller predating this parameter.
/// `inflow_ar_coefficients` populates every stage's `InflowModel::ar_coefficients`
/// (empty leaves the PAR-free `vec![]`, byte-identical to every caller predating it);
/// a non-empty slice with a near-zero `inflow_std` builds a deterministic PAR series,
/// mirroring `d16_par1_lag_shift`.
#[expect(
    clippy::too_many_lines,
    reason = "one linear entity/bounds/penalties assembly shared by every fan/chain fixture in this file; splitting the newest knob into a struct would cost a one-off type for a single call site, while every existing caller already reads as a flat parameter list at its call site"
)]
#[expect(
    clippy::expect_used,
    reason = "the fixed study built here is valid and internally consistent by construction"
)]
fn fan_or_chain_system_ext(
    n_stages: usize,
    policy_graph: HorizonGraph,
    inflow_std: f64,
    load_std: f64,
    storage: StorageSpec,
    external_rows: Vec<ExternalScenarioRow>,
    external_load_rows: Vec<ExternalLoadRow>,
    stage_state_configs: Option<&[StageStateConfig]>,
    inflow_ar_coefficients: &[f64],
) -> System {
    let bus_id = EntityId(1);
    let hydro_id = EntityId(2);

    let bus = Bus {
        id: bus_id,
        name: "B".to_string(),
        operational_start_date: ymd(2024, 1, 1),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: hydro_id,
        name: "H".to_string(),
        operational_start_date: ymd(2024, 1, 1),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: storage.max_storage_hm3,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: Some(0.5),
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };
    hydro.declare_mirror_unit_group(bus_id);

    debug_assert!(
        stage_state_configs.is_none_or(|cfgs| cfgs.len() == n_stages),
        "fan_or_chain_system_ext: stage_state_configs, when Some, must supply one \
         StageStateConfig per stage"
    );
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "n_stages is a small fixture stage count, far below i32::MAX"
    )]
    let stages: Vec<_> = (0..n_stages)
        .map(|i| {
            let config = stage_state_configs.map_or(K_FAN_DEFAULT_STATE_CONFIG, |cfgs| cfgs[i]);
            let mut stage = k_fan_stage(i, i as i32, config);
            // A PAR series (nonempty ar_coefficients) requires a season_id for the
            // lag-stage statistics lookup (`fill_stage_arrays`); the pre-study lag
            // stats then resolve through the seasonal fallback. PAR-free callers
            // leave season_id None, unchanged.
            if !inflow_ar_coefficients.is_empty() {
                stage.season_id = Some(0);
            }
            stage
        })
        .collect();

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "n_stages is a small fixture stage count, far below i32::MAX"
    )]
    let inflow_models: Vec<_> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id,
            stage_id: i as i32,
            mean_m3s: 60.0,
            std_m3s: inflow_std,
            ar_coefficients: inflow_ar_coefficients.to_vec(),
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "n_stages is a small fixture stage count, far below i32::MAX"
    )]
    let load_models: Vec<_> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id,
            stage_id: i as i32,
            mean_mw: 80.0,
            std_mw: load_std,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
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
                max_storage_hm3: storage.max_storage_hm3,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id,
            value_hm3: storage.initial_storage_hm3,
        }],
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .external_scenarios(external_rows)
        .external_load_scenarios(external_load_rows)
        .build()
        .expect("fan_or_chain_system: valid study")
}

/// The `sampled`-mode training [`Config`] for [`k_fan_setup`]: `forward_passes`
/// trajectories per iteration, `max_iterations` fixed via an iteration-limit
/// stopping rule, no `enumerated` selection anywhere.
pub(crate) fn k_fan_config(forward_passes: u32, max_iterations: u32) -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::None,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(K_FAN_TREE_SEED),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: max_iterations,
            }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// The DECOMP K-fan [`StudySetup`], bundled with the fixture parameters
/// callers need to derive their own expected values — never a magic literal,
/// always re-derived from these fields or from `setup.inputs.node_graph` directly.
#[derive(Debug)]
pub struct KFanFixture {
    /// The built study: a single-hydro, single-bus system over the declared
    /// K-fan graph, trained `sampled` with a fixed seed.
    pub setup: StudySetup,
    /// Fan-out width: `setup.inputs.node_graph` has `k` nodes at
    /// [`K_FAN_BRANCH_STAGE_ID`] and `k` leaves at [`K_FAN_LEAF_STAGE_ID`].
    pub k: usize,
    /// `forward_passes` this fixture was configured with (mirrors
    /// `setup`'s own resolved value; exposed so callers never re-guess it).
    pub forward_passes: u32,
    /// `node_graph::enumerated_scenario_count` for this fixture's graph — the
    /// per-path enumeration the sampled scale assertion must stay strictly
    /// below.
    pub enumerated_scenario_count: u64,
}

/// Build the DECOMP K-fan [`StudySetup`]: a root (stage [`K_FAN_ROOT_STAGE_ID`])
/// fanning into `k` distinct nodes (stage [`K_FAN_BRANCH_STAGE_ID`]) under
/// non-uniform transition weights, each with its own deterministic leaf (stage
/// [`K_FAN_LEAF_STAGE_ID`]) — `num_nodes (2k+1) > n_pools (k+2)` via leaf
/// sharing, and a genuine `k`-node reverse-topological backward level at
/// [`K_FAN_BRANCH_STAGE_ID`] (`backward_cut_levels`). Configured `sampled`
/// (never `enumerated`) with a fixed seed, `forward_passes` trajectories per
/// iteration, `max_iterations` iterations.
///
/// # Panics
///
/// Never in practice — see [`k_fan_fixture`].
#[must_use]
pub fn k_fan_setup(k: usize, forward_passes: u32, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, false, k_fan_config(forward_passes, max_iterations))
}

/// [`k_fan_setup`] with the node/transition declaration order reversed — the
/// SAMPLED-mode declaration-order-invariance fixture (the enumerated engine
/// already has this in [`k_fan_setup_enumerated_reversed`]; sampled had no
/// reversed builder until now, even though [`k_fan_fixture`] always supported
/// it). `build_node_graph`'s canonical sort must recover the identical graph,
/// so training over this fixture is bit-for-bit identical to [`k_fan_setup`].
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[must_use]
pub fn k_fan_setup_reversed(k: usize, forward_passes: u32, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, true, k_fan_config(forward_passes, max_iterations))
}

/// The DECOMP K-fan [`StudySetup`] configured `enumerated`: the forward
/// distribution is the exhaustive all-paths engine, so `forward_passes` is
/// resolved by the node graph to `enumerated_scenario_count` (= `k`), not a
/// caller value. Otherwise identical to [`k_fan_setup`] (fixed seed, iteration
/// limit `max_iterations`).
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[must_use]
pub fn k_fan_setup_enumerated(k: usize, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, false, k_fan_config_enumerated(max_iterations))
}

/// [`k_fan_setup_enumerated`] with the node/transition declaration order
/// reversed. `build_node_graph`'s canonical sort must recover the identical
/// graph, so an enumerated run over this fixture is bit-for-bit identical to
/// the canonical one — the declaration-order-invariance gate.
#[must_use]
pub fn k_fan_setup_enumerated_reversed(k: usize, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, true, k_fan_config_enumerated(max_iterations))
}

/// The DECOMP K-fan [`StudySetup`] configured `enumerated` under the fixture's
/// expectation measure ([`StageRiskConfig::Expectation`] on every stage) with a
/// `Gap { tolerance }` stopping rule ahead of the mandatory iteration-limit
/// fallback — the gap-attainability fixture. The `Gap` carries only the absolute
/// canonical-R$ `tolerance` arm (no `relative_tolerance`), and no `BoundStalling`
/// is ever auto-added, so an unattainable `tolerance` falls through to the
/// mandatory `IterationLimit`. `cost_scale_factor == None` keeps the default factor;
/// `Some(f)` prescales the LP by `f` — `Some(1.0)` runs it unscaled, the two
/// settings the canonical-units pinning regression compares. Otherwise identical
/// to [`k_fan_setup_enumerated`] (fixed seed, iteration limit `max_iterations`).
#[must_use]
pub fn k_fan_setup_gap(
    k: usize,
    tolerance: f64,
    max_iterations: u32,
    cost_scale_factor: Option<f64>,
) -> KFanFixture {
    k_fan_fixture(
        k,
        false,
        k_fan_config_gap(tolerance, max_iterations, cost_scale_factor),
    )
}

/// A single-path (count-1) `enumerated` chain study: a 2-stage,
/// `branching_factor: 1` chain (empty policy graph) whose enumerated path count
/// resolves to `1`. The faithful fixture for the enumerated engine's 2-rank
/// stub — `RankOf2` only faithfully simulates rank 0 of 2 when `forward_passes
/// == 1` (rank 1 then genuinely does nothing), matching the backward by-node
/// gates.
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[must_use]
pub fn single_path_enumerated_setup(max_iterations: u32) -> StudySetup {
    let system = fan_or_chain_system(2, HorizonGraph::default());
    let config = k_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("single_path_enumerated_setup: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("single_path_enumerated_setup: StudySetup::new must succeed")
}

/// Shared build for [`k_fan_setup`]/[`k_fan_setup_enumerated`]: builds the
/// stochastic context and study for `config` and derives the fixture's exposed
/// counts from the resolved study (never a caller literal).
///
/// # Panics
///
/// Never in practice: every literal in `k_fan_system`/`k_fan_config` is a
/// hand-checked, internally-consistent fixture; `StudySetup::new` only errors
/// on a malformed system or config, neither of which any caller here
/// produces, and `enumerated_scenario_count` only errors on a `u64`
/// path-product overflow, unreachable at this fixture's scale.
#[expect(
    clippy::expect_used,
    reason = "every literal in k_fan_system/k_fan_config is a hand-checked, internally-consistent fixture; StudySetup::new only errors on a malformed system or config, and enumerated_scenario_count only errors on a u64 path-product overflow, neither reachable at this fixture's scale"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "config is taken by value so callers pass an owned builder result inline; the body only borrows it for StudySetup::new"
)]
#[must_use]
fn k_fan_fixture(k: usize, reversed: bool, config: Config) -> KFanFixture {
    let system = k_fan_system(k, reversed);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("k_fan_fixture: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);

    let setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("k_fan_fixture: StudySetup::new must succeed");
    let enumerated = enumerated_scenario_count(&setup.inputs.node_graph)
        .expect("k_fan_fixture: enumerated_scenario_count must not overflow at this scale");

    KFanFixture {
        forward_passes: setup.loop_params.forward_passes,
        setup,
        k,
        enumerated_scenario_count: enumerated,
    }
}

/// The `enumerated`-mode training [`Config`] for [`k_fan_setup_enumerated`]:
/// `selection = enumerated` (no explicit `forward_passes` — the graph derives
/// it), fixed seed, iteration-limit stopping rule.
fn k_fan_config_enumerated(max_iterations: u32) -> Config {
    let mut config = k_fan_config(1, max_iterations);
    config.training.selection = Some(TrainingSelection::Enumerated {});
    config
}

/// [`k_fan_config_enumerated`] whose stopping-rule set is a `Gap` rule (the
/// absolute canonical-R$ `tolerance` arm only, no `relative_tolerance`) declared
/// FIRST, then the mandatory `IterationLimit`. The `Gap` leads so a closed-gap
/// iteration reports the `gap` reason rather than `iteration_limit` — the
/// termination reason is the first triggered rule in declaration order. When
/// `cost_scale_factor` is `Some`, it overrides `modeling.cost_scale_factor`
/// (left `None`, i.e. the default factor, otherwise).
fn k_fan_config_gap(tolerance: f64, max_iterations: u32, cost_scale_factor: Option<f64>) -> Config {
    let mut config = k_fan_config_enumerated(max_iterations);
    config.training.stopping_rules = Some(vec![
        StoppingRuleConfig::Gap {
            tolerance: Some(tolerance),
            relative_tolerance: None,
        },
        StoppingRuleConfig::IterationLimit {
            limit: max_iterations,
        },
    ]);
    config.modeling.cost_scale_factor = cost_scale_factor;
    config
}

/// [`k_fan_config_enumerated`] whose forward scheme selects external for the
/// inflow and load classes (NCS stays in-sample), with the seed the non-in-sample
/// schemes require. The forward scheme is governed by the config's scenario
/// source, so an all-external fixture MUST declare it here; the `ClassSchemes`
/// passed to `build_stochastic_context` governs only the opening-tree/library
/// provenance.
fn external_fan_config_enumerated(max_iterations: u32) -> Config {
    let mut config = k_fan_config_enumerated(max_iterations);
    config.training.scenario_source = Some(RawScenarioSourceConfig {
        seed: Some(K_FAN_TREE_SEED),
        inflow: Some(RawClassConfigEntry {
            scheme: RawSamplingScheme::External,
        }),
        load: Some(RawClassConfigEntry {
            scheme: RawSamplingScheme::External,
        }),
        ..RawScenarioSourceConfig::default()
    });
    config
}

/// Attempt to build the K-fan study with `simulation.selection = enumerated`,
/// returning the setup result unpanicked — a `Result`, not a fixture, so
/// callers assert on it directly: an admitted single-predecessor tree resolves
/// `Ok`, with `n_scenarios` set to the derived leaf-path count `K`.
///
/// # Errors
///
/// Returns whatever `StudySetup::new` returns for the enumerated-simulation
/// config.
///
/// # Panics
///
/// Never in practice: `build_stochastic_context` is infallible for this
/// hand-checked fixture (only the `StudySetup::new` result is returned).
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context is infallible for this hand-checked fixture; only the StudySetup::new result is propagated via ?"
)]
pub fn try_k_fan_simulation_enumerated(k: usize) -> Result<StudySetup, SddpError> {
    let system = k_fan_system(k, false);
    let mut config = k_fan_config(1, 1);
    config.simulation.enabled = true;
    config.simulation.selection = Some(SimulationSelection::Enumerated {});
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("try_k_fan_simulation_enumerated: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
}

// ── Branching value oracle: per-node patched-template capture ─────────────────

/// Recording [`SolverInterface`] that captures a fully-patched stage template:
/// `load_model` clones the base template, `set_row_bounds`/`set_col_bounds`
/// overwrite the clone's bounds at the patched indices. `solve` is never reached —
/// the capture drives only [`StageSolvePrep::run`]'s bound-patch side, which never
/// calls `add_rows` or `solve`.
struct TemplateCaptureSolver {
    template: Option<StageTemplate>,
}

impl SolverInterface for TemplateCaptureSolver {
    type Profile = ();

    fn apply_profile(&mut self, _profile: &()) {}

    fn load_model(&mut self, template: &StageTemplate) {
        self.template = Some(template.clone());
    }

    fn add_rows(&mut self, _rows: &RowBatch) {}

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before set_row_bounds, so template is always Some"
    )]
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        let t = self
            .template
            .as_mut()
            .expect("capture: load_model precedes set_row_bounds");
        for (k, &i) in indices.iter().enumerate() {
            t.row_lower[i] = lower[k];
            t.row_upper[i] = upper[k];
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before set_col_bounds, so template is always Some"
    )]
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        let t = self
            .template
            .as_mut()
            .expect("capture: load_model precedes set_col_bounds");
        for (k, &i) in indices.iter().enumerate() {
            t.col_lower[i] = lower[k];
            t.col_upper[i] = upper[k];
        }
    }

    fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        Err(SolverError::Unsupported(
            "TemplateCaptureSolver never solves",
        ))
    }

    fn get_basis(&mut self, _out: &mut Basis) {}

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&SolverStatistics::default());
    }

    fn name(&self) -> &'static str {
        "TemplateCapture"
    }

    fn solver_name_version(&self) -> String {
        "TemplateCapture 0.0.0".to_string()
    }
}

/// Recording [`SolverInterface`] wrapping a real solver: every method forwards
/// to `inner`, while `load_model`/`set_row_bounds`/`set_col_bounds` also mirror
/// the bound-patch onto `current` (the bookkeeping [`TemplateCaptureSolver`]
/// performs standalone), and `solve` snapshots `current` into `recorded`
/// before forwarding — so `recorded` ends up holding the exact LP a caller
/// (e.g. [`evaluate_lower_bound`]) solves, one entry per `solve` call.
struct BoundRecordingSolver<S: SolverInterface> {
    inner: S,
    current: Option<StageTemplate>,
    recorded: Vec<StageTemplate>,
}

impl<S: SolverInterface> SolverInterface for BoundRecordingSolver<S> {
    type Profile = S::Profile;

    fn apply_profile(&mut self, profile: &Self::Profile) {
        self.inner.apply_profile(profile);
    }

    fn load_model(&mut self, template: &StageTemplate) {
        self.current = Some(template.clone());
        self.inner.load_model(template);
    }

    fn add_rows(&mut self, rows: &RowBatch) {
        self.inner.add_rows(rows);
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before set_row_bounds, so current is always Some"
    )]
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        let t = self
            .current
            .as_mut()
            .expect("BoundRecordingSolver: load_model precedes set_row_bounds");
        for (k, &i) in indices.iter().enumerate() {
            t.row_lower[i] = lower[k];
            t.row_upper[i] = upper[k];
        }
        self.inner.set_row_bounds(indices, lower, upper);
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before set_col_bounds, so current is always Some"
    )]
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        let t = self
            .current
            .as_mut()
            .expect("BoundRecordingSolver: load_model precedes set_col_bounds");
        for (k, &i) in indices.iter().enumerate() {
            t.col_lower[i] = lower[k];
            t.col_upper[i] = upper[k];
        }
        self.inner.set_col_bounds(indices, lower, upper);
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before solve, so current is always Some"
    )]
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        let current = self
            .current
            .clone()
            .expect("BoundRecordingSolver: load_model precedes solve");
        self.recorded.push(current);
        self.inner.solve(basis)
    }

    fn get_basis(&mut self, out: &mut Basis) {
        self.inner.get_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        self.inner.statistics()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        self.inner.statistics_into(out);
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn solver_name_version(&self) -> String {
        self.inner.solver_name_version()
    }

    fn record_reconstruction_stats(&mut self) {
        self.inner.record_reconstruction_stats();
    }

    fn reset_solver_state(&mut self) {
        self.inner.reset_solver_state();
    }
}

/// `setup`'s stage-invariant [`StateSpace`] — `StudySetup::stage_data.state` is
/// `pub(crate)`; this is the test-support reach-through.
#[must_use]
pub fn state_space(setup: &StudySetup) -> &StateSpace {
    &setup.inputs.stage_data.state
}

/// The `[hydro | load-bus | NCS]` standardized noise draw for `node_pos`: an
/// `External` node reads its own scenario column from the standardized external
/// inflow library; a `Generated` node draws zeros — the oracle fixtures carry
/// `std == 0` on every generated stage, so `transform_inflow_noise` recovers the
/// mean regardless.
fn oracle_raw_noise(setup: &StudySetup, node_pos: NodePos) -> Vec<f64> {
    let stage = setup.inputs.node_graph.nodes[node_pos].stage;
    let n_hydros = setup.inputs.stage_data.state.hydro_count;
    let mut raw = vec![0.0_f64; setup.inputs.stochastic.dim()];
    let openings = setup.inputs.node_graph.nodes[node_pos].openings;
    if openings.source == OpeningSource::External
        && let Some(lib) = setup
            .inputs
            .scenario_libraries
            .training
            .external_inflow
            .as_ref()
    {
        let eta = lib.eta_slice(stage.0, openings.offset);
        let take = eta.len().min(n_hydros);
        raw[..take].copy_from_slice(&eta[..take]);
    }
    raw
}

/// Capture `node_pos`'s engine-exact, fully-patched single-stage [`StageTemplate`]
/// — the base template for the node's stage with the incoming-state pin and the
/// realized noise applied exactly as the training solve does (via the shared
/// [`StageSolvePrep::run`] pipeline). The incoming-state columns land pinned to
/// `SolveInputs::initial`; the extensive-form composer frees and couples them for
/// non-root nodes.
///
/// # Panics
///
/// Panics if `node_pos` (or its resolved stage) is out of range, or if the
/// template is absent after [`StageSolvePrep::run`].
#[must_use]
pub fn capture_patched_node_template(setup: &StudySetup, node_pos: NodePos) -> StageTemplate {
    let raw_noise = oracle_raw_noise(setup, node_pos);
    capture_patched_node_template_with_raw_noise(
        setup,
        node_pos,
        &raw_noise,
        &setup.inputs.initial.state,
    )
}

/// [`capture_patched_node_template`] with a caller-chosen standardized inflow
/// draw (one entry per hydro) in place of the node's own; load-bus and NCS draws
/// are zero, so those resolve to their means.
///
/// # Panics
///
/// Panics if `inflow_eta` is not `hydro_count` long, if `node_pos` (or its
/// resolved stage) is out of range, or if the template is absent after
/// [`StageSolvePrep::run`].
#[must_use]
pub fn capture_patched_node_template_with_inflow_noise(
    setup: &StudySetup,
    node_pos: NodePos,
    inflow_eta: &[f64],
) -> StageTemplate {
    let hydro = setup.inputs.stochastic.class_dimensions().hydro_range();
    assert_eq!(
        inflow_eta.len(),
        hydro.len(),
        "inflow_eta must hold one standardized draw per hydro"
    );
    let mut raw_noise = vec![0.0_f64; setup.inputs.stochastic.dim()];
    raw_noise[hydro].copy_from_slice(inflow_eta);
    capture_patched_node_template_with_raw_noise(
        setup,
        node_pos,
        &raw_noise,
        &setup.inputs.initial.state,
    )
}

/// [`capture_patched_node_template`] at a caller-chosen raw noise vector and
/// incoming state, in place of the node's own oracle draw and
/// `SolveInputs::initial`.
///
/// # Panics
///
/// Panics if `raw_noise.len() != setup.inputs.stochastic.dim()`, if
/// `incoming_state.len() != setup.inputs.stage_data.state.n_state`, if `node_pos`
/// (or its resolved stage) is out of range, or if the template is absent
/// after [`StageSolvePrep::run`].
#[must_use]
pub fn capture_patched_node_template_at(
    setup: &StudySetup,
    node_pos: NodePos,
    raw_noise: &[f64],
    incoming_state: &[f64],
) -> StageTemplate {
    assert_eq!(
        raw_noise.len(),
        setup.inputs.stochastic.dim(),
        "raw_noise must be the `[hydro | load-bus | NCS]` raw-noise length"
    );
    assert_eq!(
        incoming_state.len(),
        setup.inputs.stage_data.state.n_state,
        "incoming_state must hold one entry per state dimension"
    );
    capture_patched_node_template_with_raw_noise(setup, node_pos, raw_noise, incoming_state)
}

/// Opening `opening`'s raw `[hydro | load-bus | NCS]` noise vector at
/// `node_pos`: a `Generated` node's own draw from
/// [`StochasticContext::opening_tree`]; an `External` node's sole opening
/// (`0`) assembled through [`fill_external_opening_noise`], the same routine
/// the backward pass and the lower bound read.
///
/// # Panics
///
/// Panics if `opening >= node_pos`'s opening count, if (for an `External`
/// node) `opening != 0`, or if [`fill_external_opening_noise`] fails.
#[expect(
    clippy::expect_used,
    reason = "fill_external_opening_noise fails only on a malformed setup this fixture never produces"
)]
#[must_use]
pub fn node_opening_noise(setup: &StudySetup, node_pos: NodePos, opening: usize) -> Vec<f64> {
    let stage = setup.inputs.node_graph.nodes[node_pos].stage;
    let openings = setup.inputs.node_graph.nodes[node_pos].openings;
    assert!(
        opening < openings.len,
        "opening must be < node_pos's opening count"
    );
    match openings.source {
        OpeningSource::Generated => setup
            .inputs
            .stochastic
            .opening_tree()
            .opening(stage.0, openings.offset + opening)
            .to_vec(),
        OpeningSource::External => {
            assert_eq!(opening, 0, "an External node has exactly one opening");
            let training_ctx = setup.training_ctx();
            let mut buf = Vec::new();
            fill_external_opening_noise(
                &training_ctx,
                stage,
                openings.offset,
                setup.inputs.node_graph.node_ids[node_pos],
                &mut buf,
            )
            .expect("node_opening_noise: fill_external_opening_noise must succeed");
            buf
        }
    }
}

#[expect(
    clippy::expect_used,
    reason = "TemplateCaptureSolver::template is set before StageSolvePrep::run and only load_model/set_row_bounds/set_col_bounds touch it, so it is always Some after run"
)]
fn capture_patched_node_template_with_raw_noise(
    setup: &StudySetup,
    node_pos: NodePos,
    raw_noise: &[f64],
    incoming_state: &[f64],
) -> StageTemplate {
    let stage = setup.inputs.node_graph.nodes[node_pos].stage;
    let base = setup.inputs.stage_data.stage_templates.templates[stage.0].clone();
    let mut solver = TemplateCaptureSolver {
        template: Some(base),
    };

    let space = &setup.inputs.stage_data.state;
    let ctx = setup.stage_ctx();
    let mut patch_buf = PatchBuffer::new(space, ctx.load_bus_indices, ctx.geometry_per_stage);
    let training_ctx = setup.training_ctx();
    let mut scratch = ScratchBuffers::new(&training_ctx, &ctx, WorkspaceSizing::default());
    let params = StageSolvePrepParams {
        state_source: StateSource(incoming_state),
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut solver,
        &mut patch_buf,
        &mut scratch,
        &ctx,
        &training_ctx,
        stage,
        &params,
    );

    solver
        .template
        .take()
        .expect("capture_patched_node_template: template present after run")
}

/// The study's incoming initial state vector (`n_state` entries) — the value the
/// extensive-form root's incoming-state columns are pinned to.
#[must_use]
pub fn oracle_initial_state(setup: &StudySetup) -> Vec<f64> {
    setup.inputs.initial.state.clone()
}

/// `stage`'s admissible box (per outgoing state dimension) as plain `(lower,
/// upper)` vectors, for integration tests, which cannot reach the crate-private
/// `StageTemplates::state_boxes`.
#[must_use]
pub fn stage_state_box_bounds(setup: &StudySetup, stage: usize) -> (Vec<f64>, Vec<f64>) {
    let state_box = &setup.inputs.stage_data.stage_templates.state_boxes()[stage];
    (state_box.lower.clone(), state_box.upper.clone())
}

/// The no-cut root lower bound: [`evaluate_lower_bound`] over `setup`'s stage-0
/// LP with whatever cuts `setup.fcf` currently holds (empty on a freshly built
/// `StudySetup`), with the lower-bound scratch sized from the training
/// session's own arguments.
///
/// # Errors
///
/// Returns whatever [`evaluate_lower_bound`] returns.
pub fn no_cut_root_lower_bound<S: SolverInterface>(
    setup: &StudySetup,
    solver: &mut S,
) -> Result<f64, SddpError> {
    let training_ctx = setup.training_ctx();
    let state = training_ctx.state;
    let stage_ctx = setup.stage_ctx();

    let mut patch_buf = crate::lower_bound::lower_bound_patch_buffer(state, &stage_ctx);
    let mut lb_cut_batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    let mut noise_scratch =
        ScratchBuffers::new(&training_ctx, &stage_ctx, WorkspaceSizing::default());
    let mut lb_scratch = LbEvalScratch::new();
    let mut bundle = LbEvalScratchBundle::from_scratch_fields(
        &mut patch_buf,
        &mut lb_cut_batch,
        None,
        &mut noise_scratch,
        &mut lb_scratch,
    );

    evaluate_lower_bound(
        solver,
        &setup.fcf,
        &stage_ctx,
        &training_ctx,
        &setup.inputs.cut_management.risk_measures[0],
        &mut bundle,
        &LocalBackend,
    )
}

/// The lower bound's fully-patched root-opening templates, one entry per root
/// opening in opening order — the exact LP [`no_cut_root_lower_bound`] solves
/// for each, recorded via [`BoundRecordingSolver`] so the patch buffer stays
/// sized however [`no_cut_root_lower_bound`] sizes it.
///
/// # Errors
///
/// Returns whatever [`no_cut_root_lower_bound`] returns.
pub fn lower_bound_root_templates<S: SolverInterface>(
    setup: &StudySetup,
    solver: S,
) -> Result<Vec<StageTemplate>, SddpError> {
    let mut recorder = BoundRecordingSolver {
        inner: solver,
        current: None,
        recorded: Vec::new(),
    };
    no_cut_root_lower_bound(setup, &mut recorder)?;
    Ok(recorder.recorded)
}

/// `(downstream_completed_lags.len(), lag_accumulator.len())` of a built
/// workspace's scratch — both `pub(crate)`, unreachable from a `tests/`
/// integration crate without this accessor.
#[must_use]
pub fn workspace_downstream_lag_shape<S: SolverInterface>(
    ws: &SolverWorkspace<S>,
) -> (usize, usize) {
    (
        ws.scratch.downstream_completed_lags.len(),
        ws.scratch.lag_accumulator.len(),
    )
}

// ── Branching value oracle: fixtures ─────────────────────────────────────────

/// Number of stages in the terminal-Generated fan control (root + one leaf level).
const TERMINAL_FAN_STAGES: usize = 2;

/// A 3-stage single-path (chain) generated study, trained `enumerated`
/// (deterministic, one path): the degenerate control where every node has exactly
/// one successor, so the backward's first-successor read is the only successor and
/// today's engine is correct. Converges to the true 3-stage optimum.
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[must_use]
pub fn oracle_chain_setup(max_iterations: u32) -> StudySetup {
    let system = fan_or_chain_system(3, HorizonGraph::default());
    let config = k_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("oracle_chain_setup: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("oracle_chain_setup: StudySetup::new must succeed")
}

/// Declared graph for the terminal-Generated fan control: root (id `0`,
/// stage `0`) fans into `k` **leaves** (ids `1..=k`, stage `1`) under non-uniform
/// weights `i / Σj`, every node `Generated` (no `scenario_id`). The leaves are
/// terminal, so [`build_node_graph`] assigns them ONE shared leaf pool: the fan's
/// successors are interchangeable and today's engine prices the root correctly.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "k is a small fixture fan width, far below i32::MAX and below f64's exact-integer range"
)]
fn terminal_generated_fan_policy_graph(k: usize) -> HorizonGraph {
    debug_assert!(
        k >= 2,
        "terminal_generated_fan_policy_graph: k must be >= 2"
    );
    let mut nodes = Vec::with_capacity(1 + k);
    let mut transitions = Vec::with_capacity(k);
    nodes.push(PolicyNode {
        id: 0,
        stage_id: 0,
        scenario_id: None,
        label: None,
    });
    let total_weight: f64 = (1..=k).map(|i| i as f64).sum();
    for i in 1..=k {
        nodes.push(PolicyNode {
            id: i as i32,
            stage_id: 1,
            scenario_id: None,
            label: None,
        });
        transitions.push(Transition {
            source_id: 0,
            target_id: i as i32,
            probability: (i as f64) / total_weight,
            annual_discount_rate_override: None,
        });
    }
    finite_horizon_graph(nodes, transitions)
}

/// The terminal-Generated fan control [`StudySetup`]: a root fanning into `k`
/// interchangeable Generated leaves that share one pool, trained `enumerated`.
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[must_use]
pub fn terminal_generated_fan_setup(k: usize, max_iterations: u32) -> StudySetup {
    let system = fan_or_chain_system(TERMINAL_FAN_STAGES, terminal_generated_fan_policy_graph(k));
    let config = k_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("terminal_generated_fan_setup: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("terminal_generated_fan_setup: StudySetup::new must succeed")
}

/// [`k_fan_config`] with an active Dynamic cut-selection strategy (`start_iteration
/// = 1`, so every training iteration takes the DCS lazy path). The DCS backward
/// falls back to the by-scenario driver whose successor-pool resolution the DCS-arm
/// oracle case exercises on a `pool_id != stage` node.
fn k_fan_config_dcs(forward_passes: u32, max_iterations: u32) -> Config {
    let mut config = k_fan_config(forward_passes, max_iterations);
    config.training.cut_selection = RowSelectionConfig {
        row_activity_tolerance: None,
        max_active_per_stage: None,
        selection: Some(SelectionMethod::Dynamic {
            start_iteration: 1,
            seed_window: 5,
            candidate_recency: None,
            max_added_per_round: 10,
            violation_tolerance: 1e-10,
        }),
    };
    config
}

/// The DECOMP K-fan [`StudySetup`] with Dynamic cut-selection active from
/// iteration 1 — the DCS-arm oracle fixture. The K-fan's fan pools carry
/// `pool_id != stage`, so the DCS backward's successor-pool resolution is
/// exercised on a genuinely branching graph.
#[must_use]
pub fn dcs_k_fan_setup(k: usize, forward_passes: u32, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, false, k_fan_config_dcs(forward_passes, max_iterations))
}

/// Per-child External inflow realizations for [`external_distinct_fan_setup`]:
/// materially different m³/s so a child-0 collapse (every leaf priced against
/// column 0) yields a different value than the correct fan-out.
const EXTERNAL_FAN_INFLOWS: [f64; 3] = [20.0, 90.0, 160.0];

/// Deterministic external LOAD (MW) every column of the all-external fan carries:
/// equal to the load-model mean so the standardized eta is zero and load is
/// identical across leaves — only inflow distinguishes the fan's successors.
const EXTERNAL_FAN_LOAD_MW: f64 = 80.0;

/// Positive load std that makes the fan's bus a stochastic load class, so its
/// external-load library carries a column per external inflow column.
const EXTERNAL_FAN_LOAD_STD: f64 = 20.0;

/// A 2-stage all-external distinct fan: root (stage 0, External column 0) fans
/// into `k` leaves (stage 1), each pinning its own raw scenario column
/// ([`OpeningSource::External`], distinct `openings.offset`) whose inflow differs
/// materially from the others. Inflow and load both draw external; load is
/// deterministic ([`EXTERNAL_FAN_LOAD_MW`] on every column) so only inflow
/// distinguishes the successors, and NCS stays the empty degenerate in-sample
/// class. The last-stage branching shape; the fan's successors are NOT
/// interchangeable (distinct inflow columns), so today's backward — which prices
/// every leaf against column 0 — is silently wrong.
///
/// # Panics
///
/// Never in practice — every literal is a hand-checked, internally-consistent
/// fixture.
#[must_use]
pub fn external_distinct_fan_setup(k: usize, max_iterations: u32) -> StudySetup {
    build_external_distinct_fan_setup(k, max_iterations, None)
}

/// [`external_distinct_fan_setup`] with an active inflow-lag slot
/// (a boundary-inferred depth of `Some(1)`) — the terminal-fusion regression
/// fixture whose leaf pool and cut-generating parent pool project DIFFERENT
/// dimensions, unlike every other fan/chain fixture in this module.
/// `build_cut_state_layouts` (`setup/mod.rs`) starts every pool at
/// `FULL_STATE_CONFIG` and only overwrites a NON-leaf pool with its
/// successor's declared `state_config`; every stage here keeps the module
/// default storage-only `state_config` ([`K_FAN_DEFAULT_STATE_CONFIG`]), so the
/// root's pool (sized from the leaf's declared config) projects storage only
/// while the leaves' own terminal pool (no successor to resize it) keeps the
/// unconditional full storage+lag projection — divergent dimensions with no
/// declared per-stage `state_config` override at all
/// ([`K_FAN_DEFAULT_STATE_CONFIG`]). Every other fan/chain fixture leaves
/// `inflow_lag_depth` at its `None` default, so the lag block is empty and
/// the full-vs-storage-only distinction is dimensionally moot — this is the
/// one fixture that gives it a nonzero width.
///
/// # Panics
///
/// Never in practice — as [`external_distinct_fan_setup`].
#[must_use]
pub fn external_distinct_fan_setup_heterogeneous_cut_state(
    k: usize,
    max_iterations: u32,
) -> StudySetup {
    build_external_distinct_fan_setup(k, max_iterations, Some(1))
}

#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "k and its derived scenario/node ids are small fixture counts, far below i32::MAX and below f64's exact-integer range"
)]
fn build_external_distinct_fan_setup(
    k: usize,
    max_iterations: u32,
    inflow_lag_depth: Option<u32>,
) -> StudySetup {
    assert!(
        k >= 2 && k <= EXTERNAL_FAN_INFLOWS.len(),
        "external fan k in 2..=3"
    );

    let policy_graph = {
        let mut nodes = Vec::with_capacity(1 + k);
        let mut transitions = Vec::with_capacity(k);
        nodes.push(PolicyNode {
            id: 0,
            stage_id: 0,
            scenario_id: Some(0),
            label: None,
        });
        let total: f64 = (1..=k).map(|i| i as f64).sum();
        for i in 1..=k {
            nodes.push(PolicyNode {
                id: i as i32,
                stage_id: 1,
                scenario_id: Some((i - 1) as i32),
                label: None,
            });
            transitions.push(Transition {
                source_id: 0,
                target_id: i as i32,
                probability: (i as f64) / total,
                annual_discount_rate_override: None,
            });
        }
        finite_horizon_graph(nodes, transitions)
    };

    let hydro_id = EntityId(2);
    let bus_id = EntityId(1);
    // Distinct external inflow columns: stage 0 has one column (the root); stage 1
    // has `k` columns, each carrying a materially different inflow.
    let mut rows = vec![ExternalScenarioRow {
        stage_id: 0,
        scenario_id: 0,
        hydro_id,
        value_m3s: 60.0,
    }];
    // One external load column per inflow column on the same (stage, scenario) grid,
    // so every node pins a deterministic load.
    let mut load_rows = vec![ExternalLoadRow {
        stage_id: 0,
        scenario_id: 0,
        bus_id,
        value_mw: EXTERNAL_FAN_LOAD_MW,
    }];
    for (col, &value_m3s) in EXTERNAL_FAN_INFLOWS.iter().take(k).enumerate() {
        rows.push(ExternalScenarioRow {
            stage_id: 1,
            scenario_id: col as i32,
            hydro_id,
            value_m3s,
        });
        load_rows.push(ExternalLoadRow {
            stage_id: 1,
            scenario_id: col as i32,
            bus_id,
            value_mw: EXTERNAL_FAN_LOAD_MW,
        });
    }
    let system = fan_or_chain_system_ext(
        2,
        policy_graph,
        20.0,
        EXTERNAL_FAN_LOAD_STD,
        StorageSpec::wide(),
        rows,
        load_rows,
        None,
        &[],
    );

    let config = external_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::External),
            load: Some(SamplingScheme::External),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_external_distinct_fan_setup: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new_with_boundary_requirements(
        &system,
        &config,
        stochastic,
        hydro_models,
        inflow_lag_depth.map_or_else(
            BoundaryStateRequirements::none,
            BoundaryStateRequirements::present,
        ),
        Vec::new(),
    )
    .expect("build_external_distinct_fan_setup: StudySetup::new must succeed")
}

/// A 2-stage all-external fan whose SINGLE stage-0 root is itself an external node
/// pinning a **non-zero** column: stage 0 declares two external columns and the
/// root's `scenario_id` selects column 1, fanning into `k` external-distinct
/// stage-1 leaves. The non-zero root column makes the root exercise the sampling
/// offset (the root — not only the leaves — draws an offset ≥ 1 against the
/// branching-factor-1 generated tree), and drives the lower bound evaluating an
/// external stage-0 node. Inflow and load draw external; load is deterministic
/// ([`EXTERNAL_FAN_LOAD_MW`]); NCS stays the empty degenerate in-sample class.
///
/// # Panics
///
/// Never in practice — every literal is a hand-checked, internally-consistent
/// fixture.
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "k and its derived scenario/node ids are small fixture counts, far below i32::MAX and below f64's exact-integer range"
)]
#[must_use]
pub fn external_root_fan_setup(k: usize, max_iterations: u32) -> StudySetup {
    // Root pins the non-zero stage-0 column; column 0 is declared but unused so the
    // root's offset is genuinely ≥ 1.
    const ROOT_COLUMN: i32 = 1;
    const STAGE0_COLUMNS: usize = 2;

    assert!(
        k >= 2 && k <= EXTERNAL_FAN_INFLOWS.len(),
        "external fan k in 2..=3"
    );

    let policy_graph = {
        let mut nodes = Vec::with_capacity(1 + k);
        let mut transitions = Vec::with_capacity(k);
        nodes.push(PolicyNode {
            id: 0,
            stage_id: 0,
            scenario_id: Some(ROOT_COLUMN),
            label: None,
        });
        let total: f64 = (1..=k).map(|i| i as f64).sum();
        for i in 1..=k {
            nodes.push(PolicyNode {
                id: i as i32,
                stage_id: 1,
                scenario_id: Some((i - 1) as i32),
                label: None,
            });
            transitions.push(Transition {
                source_id: 0,
                target_id: i as i32,
                probability: (i as f64) / total,
                annual_discount_rate_override: None,
            });
        }
        finite_horizon_graph(nodes, transitions)
    };

    let hydro_id = EntityId(2);
    let bus_id = EntityId(1);
    // Stage 0 declares two external columns; the root pins column 1. Stage 1 has `k`
    // columns, each a materially different inflow.
    let mut rows = Vec::with_capacity(STAGE0_COLUMNS + k);
    let mut load_rows = Vec::with_capacity(STAGE0_COLUMNS + k);
    for col in 0..STAGE0_COLUMNS {
        rows.push(ExternalScenarioRow {
            stage_id: 0,
            scenario_id: col as i32,
            hydro_id,
            value_m3s: 60.0,
        });
        load_rows.push(ExternalLoadRow {
            stage_id: 0,
            scenario_id: col as i32,
            bus_id,
            value_mw: EXTERNAL_FAN_LOAD_MW,
        });
    }
    for (col, &value_m3s) in EXTERNAL_FAN_INFLOWS.iter().take(k).enumerate() {
        rows.push(ExternalScenarioRow {
            stage_id: 1,
            scenario_id: col as i32,
            hydro_id,
            value_m3s,
        });
        load_rows.push(ExternalLoadRow {
            stage_id: 1,
            scenario_id: col as i32,
            bus_id,
            value_mw: EXTERNAL_FAN_LOAD_MW,
        });
    }
    let system = fan_or_chain_system_ext(
        2,
        policy_graph,
        20.0,
        EXTERNAL_FAN_LOAD_STD,
        StorageSpec::wide(),
        rows,
        load_rows,
        None,
        &[],
    );

    let config = external_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::External),
            load: Some(SamplingScheme::External),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("external_root_fan_setup: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("external_root_fan_setup: StudySetup::new must succeed")
}

/// Injected constant productivity (MW per m³/s) for the water-binding fan's hydro —
/// large enough that turbined flow generates real MW against the load, so a leaf's
/// own scarce inflow binds. `default_from_system`'s `0.0` placeholder generates
/// zero MW and is exactly why the collapse is invisible on the other fixtures.
const WATER_BINDING_PRODUCTIVITY: f64 = 0.95;

/// Reservoir cap and initial volume (hm³) for the water-binding fan — scarce, so a
/// leaf cannot cover its load from inherited root storage and its own inflow is the
/// binding swing factor.
const WATER_BINDING_STORAGE: StorageSpec = StorageSpec {
    max_storage_hm3: 10.0,
    initial_storage_hm3: 10.0,
};

/// Per-child External inflow realizations (m³/s) for the water-binding fan: a
/// low/mid/high spread whose low leaf (column 0) cannot cover the load, so the
/// child-0 collapse — pricing every leaf against column 0 — overstates future cost.
const WATER_BINDING_INFLOWS: [f64; 3] = [10.0, 60.0, 110.0];

/// [`external_fan_config_enumerated`] declaring only inflow external: the load is a
/// deterministic (`std = 0`) in-sample class the oracle reproduces at its mean, so
/// training and the extensive-form expander price the same demand. Declaring load
/// external instead would make the oracle (which applies external inflow only) and
/// training disagree on demand and contaminate the value gap.
fn water_binding_fan_config_enumerated(max_iterations: u32) -> Config {
    let mut config = k_fan_config_enumerated(max_iterations);
    config.training.scenario_source = Some(RawScenarioSourceConfig {
        seed: Some(K_FAN_TREE_SEED),
        inflow: Some(RawClassConfigEntry {
            scheme: RawSamplingScheme::External,
        }),
        ..RawScenarioSourceConfig::default()
    });
    config
}

/// The water-binding sibling of [`external_distinct_fan_setup`]: a 2-stage
/// all-external-**inflow** distinct fan whose hydro genuinely generates
/// ([`WATER_BINDING_PRODUCTIVITY`]) against a scarce reservoir
/// ([`WATER_BINDING_STORAGE`]), so each leaf's own inflow ([`WATER_BINDING_INFLOWS`])
/// binds. On this fixture the backward child-0 collapse — pricing every leaf against
/// column 0 (the low-inflow, most-expensive leaf) — overstates future cost and
/// drives `final_lb` above the extensive-form optimum (an invalid lower bound). The
/// demand is a deterministic 80 MW in-sample load (`std = 0`, a zero-count class
/// exempt from the external-node in-sample guard), which the oracle reproduces at
/// its mean.
///
/// # Panics
///
/// Never in practice — every literal is a hand-checked, internally-consistent fixture.
/// The water-binding fan trained with its nodes/transitions declared in canonical
/// (id-ascending) order — the value-red / determinism baseline.
#[must_use]
pub fn water_binding_external_fan_setup(k: usize, max_iterations: u32) -> StudySetup {
    build_water_binding_external_fan(k, max_iterations, false)
}

/// The same water-binding fan with its nodes/transitions declared in REVERSED order.
/// `build_node_graph` canonicalizes by id, so this must produce a bit-identical
/// trained policy — the declaration-order-invariance probe for a genuine
/// non-interchangeable fan.
#[must_use]
pub fn water_binding_external_fan_setup_reversed(k: usize, max_iterations: u32) -> StudySetup {
    build_water_binding_external_fan(k, max_iterations, true)
}

#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "k and its derived scenario/node ids are small fixture counts, far below i32::MAX and below f64's exact-integer range"
)]
fn build_water_binding_external_fan(k: usize, max_iterations: u32, reversed: bool) -> StudySetup {
    assert!(
        k >= 2 && k <= WATER_BINDING_INFLOWS.len(),
        "water-binding fan k in 2..=3"
    );

    let policy_graph = {
        let mut nodes = Vec::with_capacity(1 + k);
        let mut transitions = Vec::with_capacity(k);
        nodes.push(PolicyNode {
            id: 0,
            stage_id: 0,
            scenario_id: Some(0),
            label: None,
        });
        let total: f64 = (1..=k).map(|i| i as f64).sum();
        for i in 1..=k {
            nodes.push(PolicyNode {
                id: i as i32,
                stage_id: 1,
                scenario_id: Some((i - 1) as i32),
                label: None,
            });
            transitions.push(Transition {
                source_id: 0,
                target_id: i as i32,
                probability: (i as f64) / total,
                annual_discount_rate_override: None,
            });
        }
        reverse_declaration_order_if(reversed, &mut nodes, &mut transitions);
        finite_horizon_graph(nodes, transitions)
    };

    let hydro_id = EntityId(2);
    // Stage 0: the root's single column. Stage 1: k distinct inflow columns.
    let mut rows = vec![ExternalScenarioRow {
        stage_id: 0,
        scenario_id: 0,
        hydro_id,
        value_m3s: 60.0,
    }];
    for (col, &value_m3s) in WATER_BINDING_INFLOWS.iter().take(k).enumerate() {
        rows.push(ExternalScenarioRow {
            stage_id: 1,
            scenario_id: col as i32,
            hydro_id,
            value_m3s,
        });
    }
    // Load is a deterministic in-sample class (std = 0): no external-load library,
    // so the oracle's inflow-only noise reproduces the load at its mean.
    let system = fan_or_chain_system_ext(
        2,
        policy_graph,
        20.0,
        0.0,
        WATER_BINDING_STORAGE,
        rows,
        Vec::new(),
        None,
        &[],
    );

    let config = water_binding_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::External),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("water_binding_external_fan_setup: build_stochastic_context must succeed");
    let mut hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    hydro_models.production = ProductionModelSet::new(
        vec![vec![
            ResolvedProductionModel::ConstantProductivity {
                productivity: WATER_BINDING_PRODUCTIVITY,
            };
            2
        ]],
        system.hydros(),
        2,
    );
    StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("water_binding_external_fan_setup: StudySetup::new must succeed")
}

// ── Non-uniform-projection branching fixture (3-stage binary tree) ──────────

/// Left/right transition weights for [`branching_tree_policy_graph`] — never a
/// uniform `0.5`/`0.5` split (mirrors [`k_fan_policy_graph`]'s `i / Σj` rule): a
/// uniform split makes every canonical-order reduction sum identical terms,
/// defeating the weighted-aggregation gates' power.
const BRANCHING_TREE_LEFT_WEIGHT: f64 = 0.35;
const BRANCHING_TREE_RIGHT_WEIGHT: f64 = 0.65;

/// Declared graph for the non-uniform branching fixture: a genuine 3-stage BINARY
/// TREE branching at TWO node-graph levels (root → 2 fan nodes, each fan node → 2
/// leaves) — the shape the collapse-era predicate rejected. Distinct
/// from [`k_fan_policy_graph`], whose fan nodes each deterministically reach ONE
/// leaf (branching at ONE level only). Node ids: `0` (root), `1`/`2` (fan,
/// [`K_FAN_BRANCH_STAGE_ID`]), `3..=6` (leaves, [`K_FAN_LEAF_STAGE_ID`], `1`'s
/// children first). All `Generated` (no `scenario_id`).
fn branching_tree_policy_graph(reversed: bool) -> HorizonGraph {
    let mut nodes = Vec::with_capacity(7);
    let mut transitions = Vec::with_capacity(6);
    nodes.push(PolicyNode {
        id: 0,
        stage_id: K_FAN_ROOT_STAGE_ID,
        scenario_id: None,
        label: None,
    });
    let mut next_leaf_id: i32 = 3;
    for (fan_id, fan_weight) in [
        (1_i32, BRANCHING_TREE_LEFT_WEIGHT),
        (2, BRANCHING_TREE_RIGHT_WEIGHT),
    ] {
        nodes.push(PolicyNode {
            id: fan_id,
            stage_id: K_FAN_BRANCH_STAGE_ID,
            scenario_id: None,
            label: None,
        });
        transitions.push(Transition {
            source_id: 0,
            target_id: fan_id,
            probability: fan_weight,
            annual_discount_rate_override: None,
        });
        for leaf_weight in [BRANCHING_TREE_LEFT_WEIGHT, BRANCHING_TREE_RIGHT_WEIGHT] {
            let leaf_id = next_leaf_id;
            next_leaf_id += 1;
            nodes.push(PolicyNode {
                id: leaf_id,
                stage_id: K_FAN_LEAF_STAGE_ID,
                scenario_id: None,
                label: None,
            });
            transitions.push(Transition {
                source_id: fan_id,
                target_id: leaf_id,
                probability: leaf_weight,
                annual_discount_rate_override: None,
            });
        }
    }
    reverse_declaration_order_if(reversed, &mut nodes, &mut transitions);
    finite_horizon_graph(nodes, transitions)
}

/// Per-study-stage `state_config` for the branching-tree fixture — the non-uniform
/// cut-state-projection axis (`d43-storage-only-cut`'s technique, lifted from
/// chain to branching). `build_cut_state_layouts` sizes a non-leaf node's pool
/// from its SUCCESSOR's stage config: stage 1 (fan level) sizes the ROOT's
/// pool, stage 2 (leaf level) sizes each FAN NODE's pool. Declaring stage 1
/// `inflow_lags: true` and stage 2 `inflow_lags: false` makes the root's pool
/// project the full storage+lag state (2 dims: 1 hydro × 1 declared lag slot,
/// widened via the depth passed to `new_with_boundary_requirements` in
/// [`build_non_uniform_branching_setup`]'s config — not a fitted PAR order,
/// which the K-fan/chain fixtures' zero-std inflow cannot normalize) while each
/// fan node's pool projects storage only (1 dim) — and the trailing shared leaf
/// pool is unconditionally full-dimension regardless of stage 2's declared
/// config (`build_cut_state_layouts`'s no-successor rule), so the per-pool
/// shape shrinks at the fan level and regrows at the leaf, mirroring
/// `d43-storage-only-cut`'s shrink-then-regrow shape one level deeper. Stage 0's
/// own config is inert for pool sizing (the root has no predecessor pool to
/// size) — declared `true` for consistency with stage 1, not because it matters.
fn non_uniform_branching_stage_configs() -> [StageStateConfig; 3] {
    [
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
        StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
    ]
}

/// The branching-tree fixture's [`System`]: [`branching_tree_policy_graph`] over the shared
/// single-hydro/single-bus [`fan_or_chain_system_ext`] boilerplate, with
/// [`non_uniform_branching_stage_configs`]. Zero inflow/load std and a wide
/// (non-binding) reservoir — Generated, `branching_factor: 1`, the
/// oracle-compatible (`|Ω| = 1` per node) shape [`extensive_form_optimum`]
/// requires, matching [`k_fan_system`]/[`oracle_chain_setup`]. The lag SLOT
/// itself (`max_par_order`) comes from the depth passed to `new_with_boundary_requirements` in
/// [`build_non_uniform_branching_setup`]'s config, not from an `ar_coefficients`
/// declared here — the AR-normalization path this crate's inflow estimator uses
/// rejects a zero-std series (`ar_order >= 1` needs nonzero variance to
/// normalize its coefficients), which the oracle-compatible zero-std design
/// above rules out.
fn branching_tree_system(reversed: bool) -> System {
    fan_or_chain_system_ext(
        3,
        branching_tree_policy_graph(reversed),
        0.0,
        0.0,
        StorageSpec::wide(),
        Vec::new(),
        Vec::new(),
        Some(&non_uniform_branching_stage_configs()),
        &[],
    )
}

/// The non-uniform-cut-state-projection branching [`StudySetup`]: a 3-stage binary
/// tree branching at two levels. Trained `sampled` (never `enumerated`) with a fixed seed,
/// `forward_passes` trajectories per iteration, `max_iterations` iterations —
/// the same config shape as [`k_fan_setup`].
///
/// # Panics
///
/// Never in practice — every literal is a hand-checked, internally-consistent
/// fixture (as [`k_fan_setup`]).
#[must_use]
pub fn non_uniform_branching_setup(forward_passes: u32, max_iterations: u32) -> StudySetup {
    build_non_uniform_branching_setup(forward_passes, max_iterations, false)
}

/// [`non_uniform_branching_setup`] with the node/transition declaration order
/// reversed — the branching-tree fixture's own declaration-order-invariance probe.
/// `build_node_graph`'s canonical sort must recover the identical graph, so
/// training over this fixture is bit-for-bit identical to
/// [`non_uniform_branching_setup`].
///
/// # Panics
///
/// Never in practice — as [`non_uniform_branching_setup`].
#[must_use]
pub fn non_uniform_branching_setup_reversed(
    forward_passes: u32,
    max_iterations: u32,
) -> StudySetup {
    build_non_uniform_branching_setup(forward_passes, max_iterations, true)
}

#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
fn build_non_uniform_branching_setup(
    forward_passes: u32,
    max_iterations: u32,
    reversed: bool,
) -> StudySetup {
    let system = branching_tree_system(reversed);
    let config = k_fan_config(forward_passes, max_iterations);
    // Declares one lag slot (`max_par_order = 1`) independent of any fitted AR
    // order, so `non_uniform_branching_stage_configs`'s `inflow_lags` toggle has
    // a non-degenerate lag block to project even though every stage's inflow std
    // is zero (the oracle-compatible design `branching_tree_system` documents).
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("non_uniform_branching_setup: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new_with_boundary_requirements(
        &system,
        &config,
        stochastic,
        hydro_models,
        BoundaryStateRequirements::present(1),
        Vec::new(),
    )
    .expect("non_uniform_branching_setup: StudySetup::new must succeed")
}

/// The `enumerated`-mode 3-stage binary tree oracle fixture: root branches into 2
/// nodes at [`K_FAN_BRANCH_STAGE_ID`], each branching into 2 leaves at
/// [`K_FAN_LEAF_STAGE_ID`], reusing [`branching_tree_policy_graph`]'s non-uniform
/// `0.35`/`0.65` weights unchanged — the shape a shape-based enumerated-admission
/// clause would have rejected. Otherwise identical to [`oracle_chain_setup`]/
/// [`k_fan_setup_enumerated`]: the plain [`K_FAN_DEFAULT_STATE_CONFIG`]
/// single-hydro system, `enumerated` selection, fixed seed, iteration-limit
/// stopping rule.
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[must_use]
pub fn branching_tree_setup_enumerated(max_iterations: u32) -> StudySetup {
    let system = fan_or_chain_system(3, branching_tree_policy_graph(false));
    let config = k_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("branching_tree_setup_enumerated: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("branching_tree_setup_enumerated: StudySetup::new must succeed")
}

/// Per-pool cut-state dimension (`CutStateProjection::n_slots`), pool-id-indexed
/// — the branching-tree fixture's power self-check reads this to confirm the projection
/// genuinely varies across pools (`build_cut_state_layouts` sizes each non-leaf
/// pool from its successor's `state_config`). `SolveInputs::cut_state_layouts`
/// itself is `pub(crate)`, unreachable from an integration test without this
/// accessor.
#[must_use]
pub fn pool_cut_state_dimensions(setup: &StudySetup) -> Vec<usize> {
    setup
        .inputs
        .cut_state_layouts
        .iter()
        .map(CutStateProjection::n_slots)
        .collect()
}

// ── Extensive-form oracle (shared by `branching_value_oracle.rs` and any other
//    integration test that needs the graph's true first-stage value) ─────────

/// Marginal visit probability of every node in `setup.inputs.node_graph`: `P(root) = 1`,
/// propagated `P(child) += P(node)·prob(node→child)` in ascending-stage order. On
/// a tree this is the product of edge probabilities on the unique root→node path
/// — the weight each node copy's stage cost carries in
/// [`extensive_form_optimum`]'s objective.
#[must_use]
pub fn node_visit_probabilities(setup: &StudySetup) -> Vec<f64> {
    let g = &setup.inputs.node_graph;
    let n = g.nodes.len();
    let mut has_pred = vec![false; n];
    for succs in &g.successors {
        for s in succs {
            has_pred[s.child.0] = true;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| g.nodes[NodePos(i)].stage);

    let mut prob = vec![0.0_f64; n];
    for &i in &order {
        if !has_pred[i] {
            prob[i] = 1.0;
        }
        for s in &g.successors[NodePos(i)] {
            prob[s.child.0] += prob[i] * s.probability;
        }
    }
    prob
}

/// The extensive-form LP optimum — `setup.inputs.node_graph`'s true first-stage value,
/// computed independently of the node-native training algorithm: one column
/// block per node (its engine-exact patched template via
/// [`capture_patched_node_template`]), objective scaled by
/// [`node_visit_probabilities`], state coupled parent-outgoing to child-incoming
/// by one equality row per state dimension, root incoming state left pinned to
/// the initial state. Solved once. The blessed extensive-form oracle for any
/// finite acyclic branching graph small enough to expand —
/// validates VALUE only, not cuts, duals, or unvisited states.
///
/// # Panics
///
/// Panics if the extensive-form LP fails to build a solver or fails to solve —
/// unreachable for a well-formed, feasible `setup`.
#[expect(
    clippy::expect_used,
    reason = "the extensive-form LP fails to build a solver or to solve only for a malformed, infeasible setup, unreachable for this oracle's inputs"
)]
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "node/row/column counts here are small fixture sizes, far below i32::MAX and always non-negative"
)]
#[must_use]
pub fn extensive_form_optimum(setup: &StudySetup) -> f64 {
    let g = &setup.inputs.node_graph;
    let n = g.nodes.len();
    let state = setup.stage_state();
    let n_state = state.n_state;
    let prob = node_visit_probabilities(setup);

    let mut has_pred = vec![false; n];
    for succs in &g.successors {
        for s in succs {
            has_pred[s.child.0] = true;
        }
    }

    let templates: Vec<StageTemplate> = (0..n)
        .map(|i| capture_patched_node_template(setup, NodePos(i)))
        .collect();

    // Column / row bases of each node's block, block-diagonal.
    let mut cbase = vec![0_usize; n + 1];
    let mut rbase = vec![0_usize; n + 1];
    for i in 0..n {
        cbase[i + 1] = cbase[i] + templates[i].num_cols;
        rbase[i + 1] = rbase[i] + templates[i].num_rows;
    }
    let total_cols = cbase[n];
    let total_rows = rbase[n];

    let mut col_starts = Vec::with_capacity(total_cols + 1);
    col_starts.push(0_i32);
    let mut row_indices: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut col_lower = Vec::with_capacity(total_cols);
    let mut col_upper = Vec::with_capacity(total_cols);
    let mut objective = Vec::with_capacity(total_cols);
    let mut row_lower = Vec::with_capacity(total_rows);
    let mut row_upper = Vec::with_capacity(total_rows);

    for i in 0..n {
        let t = &templates[i];
        for c in 0..t.num_cols {
            let start = t.col_starts[c] as usize;
            let end = t.col_starts[c + 1] as usize;
            for e in start..end {
                row_indices.push(t.row_indices[e] + rbase[i] as i32);
                values.push(t.values[e]);
            }
            col_starts.push(row_indices.len() as i32);
        }
        objective.extend(t.objective.iter().map(|&o| o * prob[i]));
        col_lower.extend_from_slice(&t.col_lower);
        col_upper.extend_from_slice(&t.col_upper);
        row_lower.extend_from_slice(&t.row_lower);
        row_upper.extend_from_slice(&t.row_upper);
    }

    // Free every non-root node's incoming-state columns: their value is fixed by
    // the coupling rows below, not by the engine's per-node pin (which the capture
    // left set to the initial state).
    for c in 0..n {
        if !has_pred[c] {
            continue;
        }
        for s in 0..n_state {
            let local = state.state_to_lp_incoming_column(StateDim::new(s)).get();
            let g_col = cbase[c] + local;
            col_lower[g_col] = f64::NEG_INFINITY;
            col_upper[g_col] = f64::INFINITY;
        }
    }

    let mono = StageTemplate {
        num_cols: total_cols,
        num_rows: total_rows,
        num_nz: values.len(),
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    // Coupling equality rows: cs_p·out(p,s) − cs_c·in(c,s) = 0, i.e. the parent's
    // outgoing physical state equals the child's incoming physical state (col_scale
    // maps scaled columns back to physical units).
    let mut row_starts = vec![0_i32];
    let mut col_ix: Vec<i32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut clower: Vec<f64> = Vec::new();
    let mut cupper: Vec<f64> = Vec::new();
    for p in 0..n {
        for succ in &g.successors[NodePos(p)] {
            let c = succ.child.0;
            for s in 0..n_state {
                let out_local = state.lp_column_for_state(StateDim::new(s)).get();
                let in_local = state.state_to_lp_incoming_column(StateDim::new(s)).get();
                let cs_p = templates[p]
                    .col_scale
                    .get(out_local)
                    .copied()
                    .unwrap_or(1.0);
                let cs_c = templates[c].col_scale.get(in_local).copied().unwrap_or(1.0);
                col_ix.push((cbase[p] + out_local) as i32);
                vals.push(cs_p);
                col_ix.push((cbase[c] + in_local) as i32);
                vals.push(-cs_c);
                row_starts.push(col_ix.len() as i32);
                clower.push(0.0);
                cupper.push(0.0);
            }
        }
    }
    let coupling = RowBatch {
        num_rows: clower.len(),
        row_starts,
        col_indices: col_ix,
        values: vals,
        row_lower: clower,
        row_upper: cupper,
    };

    let mut solver =
        ActiveSolver::new().expect("extensive_form_optimum: ActiveSolver::new must succeed");
    solver.load_model(&mono);
    if coupling.num_rows > 0 {
        solver.add_rows(&coupling);
    }
    let solution = solver
        .solve(None)
        .expect("extensive_form_optimum: extensive-form LP must solve");
    // The template objective carries the cost-scale divisor; rescale to the physical
    // cost units the engine reports `final_lb` in.
    solution.objective * setup.inputs.stage_data.stage_templates.cost_scale_factor
}

// ── Dual-folding trunk+fan fixture ────────────────────────────────────────────

/// PAR(1) coefficient `ψ` for the dual-folding trunk — nonzero so the incoming
/// inflow-lag column carries a genuine subgradient (`∂Q/∂lag ≠ 0`); a zero `ψ`
/// makes the lag inert and the fold vacuous.
const DUAL_FOLDING_PSI: f64 = 0.5;
/// Near-zero inflow std for the dual-folding trunk (mirrors
/// `d16_par1_lag_shift`'s `1e-10`): the series is deterministic yet non-degenerate,
/// so the PAR-normalization path — which rejects an exactly-zero-variance AR
/// series — still accepts it.
const DUAL_FOLDING_INFLOW_STD: f64 = 1e-10;
/// Terminal-fan width — `≥ 2` non-uniform successors so the fan is genuine (the
/// power precondition asserts this at run time).
const DUAL_FOLDING_FAN_K: usize = 3;
/// Constant hydro productivity (MW per m³/s) for the dual-folding hydro —
/// nonzero so turbined inflow generates real MW against the load, giving both the
/// reservoir and the inflow lag a nonzero marginal value (`default_from_system`'s
/// `0.0` placeholder would zero every cut coefficient and make the fold vacuous).
const DUAL_FOLDING_PRODUCTIVITY: f64 = 0.5;
/// Scarce reservoir (hm³) for the dual-folding hydro: a small cap and a matching
/// initial volume make stored water bind, so the trunk storage cut coefficient is
/// nonzero — the "trunk cut coefficient" comparison has power.
const DUAL_FOLDING_STORAGE: StorageSpec = StorageSpec {
    max_storage_hm3: 30.0,
    initial_storage_hm3: 30.0,
};

/// Which inflow-lag representation a dual-folding build carries on its
/// deterministic trunk. Both variants share one [`System`] and one
/// the boundary-inferred inflow-lag depth; they differ ONLY in the per-stage
/// `StageStateConfig.inflow_lags` toggle, so the LP columns (and the PAR
/// dynamics) are identical and only the trunk cut PROJECTION changes.
#[derive(Debug, Clone, Copy)]
pub enum LagFold {
    /// Bound-fixed lag: trunk cuts project storage only. The incoming lag column
    /// is still pinned, but its reduced cost is not projected into the cut — the
    /// lag is priced through the (storage-only) boundary future cost.
    Folded,
    /// Lag-state-enabled: trunk cuts carry the inflow lag as an explicit cut-state
    /// dimension alongside storage.
    Unfolded,
}

/// Per-stage `state_config` for a dual-folding build: `inflow_lags` follows
/// `fold` on every stage. A node's cut pool is sized by its SUCCESSOR's stage
/// config (`build_cut_state_layouts`), so toggling every stage uniformly toggles
/// both trunk pools (the root's, sized by stage 1; the mid's, sized by stage 2).
fn dual_folding_stage_configs(fold: LagFold) -> [StageStateConfig; 3] {
    let inflow_lags = matches!(fold, LagFold::Unfolded);
    [StageStateConfig {
        storage: true,
        inflow_lags,
    }; 3]
}

/// The deterministic-trunk-then-terminal-fan policy graph: root (id `0`, stage
/// [`K_FAN_ROOT_STAGE_ID`]) → mid (id `1`, stage [`K_FAN_BRANCH_STAGE_ID`]) → `k`
/// terminal leaves (ids `2..=k+1`, stage [`K_FAN_LEAF_STAGE_ID`]) under
/// non-uniform weights `i / Σj` (never uniform `1/k`). The two trunk nodes (root,
/// mid) each own their own cut pool; the `k` leaves share one terminal pool.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "k is a small fixture fan width, far below i32::MAX and below f64's exact-integer range"
)]
fn dual_folding_policy_graph(k: usize) -> HorizonGraph {
    debug_assert!(k >= 2, "dual_folding_policy_graph: k must be >= 2");
    let mut nodes = Vec::with_capacity(2 + k);
    let mut transitions = Vec::with_capacity(1 + k);
    nodes.push(PolicyNode {
        id: 0,
        stage_id: K_FAN_ROOT_STAGE_ID,
        scenario_id: None,
        label: None,
    });
    nodes.push(PolicyNode {
        id: 1,
        stage_id: K_FAN_BRANCH_STAGE_ID,
        scenario_id: None,
        label: None,
    });
    transitions.push(Transition {
        source_id: 0,
        target_id: 1,
        probability: 1.0,
        annual_discount_rate_override: None,
    });
    let total_weight: f64 = (1..=k).map(|i| i as f64).sum();
    for i in 1..=k {
        let leaf_id = 1 + i as i32;
        nodes.push(PolicyNode {
            id: leaf_id,
            stage_id: K_FAN_LEAF_STAGE_ID,
            scenario_id: None,
            label: None,
        });
        transitions.push(Transition {
            source_id: 1,
            target_id: leaf_id,
            probability: (i as f64) / total_weight,
            annual_discount_rate_override: None,
        });
    }
    finite_horizon_graph(nodes, transitions)
}

/// The dual-folding [`System`]: [`dual_folding_policy_graph`] over the shared
/// single-hydro/single-bus [`fan_or_chain_system_ext`] boilerplate, with a
/// deterministic PAR(1) trunk ([`DUAL_FOLDING_PSI`], [`DUAL_FOLDING_INFLOW_STD`])
/// and the `fold`-selected per-stage `state_config`.
fn dual_folding_system(fold: LagFold) -> System {
    fan_or_chain_system_ext(
        3,
        dual_folding_policy_graph(DUAL_FOLDING_FAN_K),
        DUAL_FOLDING_INFLOW_STD,
        0.0,
        DUAL_FOLDING_STORAGE,
        Vec::new(),
        Vec::new(),
        Some(&dual_folding_stage_configs(fold)),
        &[DUAL_FOLDING_PSI],
    )
}

/// Build the dual-folding [`StudySetup`] for `fold`: a deterministic PAR(1) trunk
/// (root → mid) followed by a terminal fan of [`DUAL_FOLDING_FAN_K`] leaves,
/// trained `sampled` with the shared fixed seed. Both `fold` variants set the
/// same boundary-inferred depth `Some(1)`, so they share one lag-state
/// dimension and one LP column layout; only [`dual_folding_stage_configs`]'s
/// `inflow_lags` toggle differs (see [`LagFold`]).
///
/// # Panics
///
/// Never in practice — every literal is a hand-checked, internally-consistent
/// fixture (as [`k_fan_setup`]); `build_stochastic_context`/`StudySetup::new`
/// only error on a malformed study.
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context/StudySetup::new never error on this fixture's fixed, internally consistent inputs"
)]
#[must_use]
pub fn dual_folding_setup(fold: LagFold, forward_passes: u32, max_iterations: u32) -> StudySetup {
    let system = dual_folding_system(fold);
    let config = k_fan_config(forward_passes, max_iterations);
    // One explicit lag slot, independent of the fitted AR order, so both `fold`
    // variants share one lag-state dimension and differ only in the `inflow_lags`
    // cut toggle.
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("dual_folding_setup: build_stochastic_context must succeed");
    let mut hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    // Override the 0.0 productivity placeholder so turbined inflow generates real
    // MW; otherwise storage and lag carry no marginal value and every cut
    // coefficient is zero (the fold would fold nothing).
    hydro_models.production = ProductionModelSet::new(
        vec![vec![
            ResolvedProductionModel::ConstantProductivity {
                productivity: DUAL_FOLDING_PRODUCTIVITY,
            };
            3
        ]],
        system.hydros(),
        3,
    );
    StudySetup::new_with_boundary_requirements(
        &system,
        &config,
        stochastic,
        hydro_models,
        BoundaryStateRequirements::present(1),
        Vec::new(),
    )
    .expect("dual_folding_setup: StudySetup::new must succeed")
}

// ── Deterministic-trunk + terminal-fan fixture (node-native enumerated backward) ──

/// Injected constant productivity (MW per m³/s) for the trunk+fan hydro —
/// nonzero so turbined inflow generates real MW against the load, giving the
/// trunk's carried storage genuine marginal value (`default_from_system`'s
/// `0.0` placeholder would zero every cut coefficient and make the bound
/// trivially deficit-cost-constant, as on [`k_fan_setup_enumerated`]/
/// [`branching_tree_setup_enumerated`]).
const TRUNK_FAN_PRODUCTIVITY: f64 = 0.5;

/// Scarce reservoir (hm³) for the trunk+fan hydro: a small cap and a matching
/// initial volume make stored water bind across the trunk, so the backward
/// cut coefficients are genuinely nonzero (mirrors [`DUAL_FOLDING_STORAGE`]).
const TRUNK_FAN_STORAGE: StorageSpec = StorageSpec {
    max_storage_hm3: 30.0,
    initial_storage_hm3: 30.0,
};

/// Deterministic-trunk + terminal-fan declared graph: `t_trunk` single-successor
/// trunk nodes (ids `0..t_trunk`, stage ids `0..t_trunk`, each deterministically
/// reaching the next), followed by a terminal fan of `k` distinct leaves (ids
/// `t_trunk..t_trunk+k`, stage `t_trunk`) off the LAST trunk node, under
/// strictly non-uniform weights `i / Σj` (mirrors
/// [`k_fan_policy_graph`]/[`dual_folding_policy_graph`]: a uniform split would
/// make every canonical-order reduction sum identical terms, defeating the
/// canonical-order gate's power).
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "t_trunk and k are small fixture counts, far below i32::MAX and below f64's exact-integer range"
)]
fn trunk_fan_policy_graph(t_trunk: usize, k: usize) -> HorizonGraph {
    debug_assert!(
        t_trunk >= 2,
        "trunk_fan_policy_graph: t_trunk must be >= 2 (a genuine chain-then-fan)"
    );
    debug_assert!(
        k >= 2,
        "trunk_fan_policy_graph: k must be >= 2 (DECOMP shape)"
    );
    let mut nodes = Vec::with_capacity(t_trunk + k);
    let mut transitions = Vec::with_capacity(t_trunk - 1 + k);
    for i in 0..t_trunk {
        nodes.push(PolicyNode {
            id: i as i32,
            stage_id: i as i32,
            scenario_id: None,
            label: None,
        });
        if i > 0 {
            transitions.push(Transition {
                source_id: (i - 1) as i32,
                target_id: i as i32,
                probability: 1.0,
                annual_discount_rate_override: None,
            });
        }
    }
    let last_trunk_id = (t_trunk - 1) as i32;
    let total_weight: f64 = (1..=k).map(|i| i as f64).sum();
    for i in 1..=k {
        let leaf_id = last_trunk_id + i as i32;
        nodes.push(PolicyNode {
            id: leaf_id,
            stage_id: t_trunk as i32,
            scenario_id: None,
            label: None,
        });
        transitions.push(Transition {
            source_id: last_trunk_id,
            target_id: leaf_id,
            probability: (i as f64) / total_weight,
            annual_discount_rate_override: None,
        });
    }
    finite_horizon_graph(nodes, transitions)
}

/// The trunk+fan [`System`]: [`trunk_fan_policy_graph`] over the shared
/// single-hydro/single-bus [`fan_or_chain_system_ext`] boilerplate, with
/// [`TRUNK_FAN_STORAGE`] and a deterministic (zero-std) inflow/load — every
/// scale signal comes from the declared trunk depth and fan width, never from
/// within-node opening variance.
fn trunk_fan_system(t_trunk: usize, k: usize) -> System {
    fan_or_chain_system_ext(
        t_trunk + 1,
        trunk_fan_policy_graph(t_trunk, k),
        0.0,
        0.0,
        TRUNK_FAN_STORAGE,
        Vec::new(),
        Vec::new(),
        None,
        &[],
    )
}

/// Fixture bundle for the deterministic-trunk + terminal-fan [`StudySetup`],
/// carrying the parameters callers need to derive their own expected
/// solves/cut counts — never a magic literal, always re-derived from these
/// fields or from `setup.inputs.node_graph` directly.
#[derive(Debug)]
pub struct TrunkFanFixture {
    /// The built study: a single-hydro, single-bus system over the declared
    /// trunk+fan graph.
    pub setup: StudySetup,
    /// Trunk depth: `t_trunk` deterministic, single-successor trunk nodes
    /// (stages `0..t_trunk`), the last of which owns the terminal fan.
    pub t_trunk: usize,
    /// Terminal fan width: `k` distinct leaves at stage `t_trunk`, reached
    /// from the last trunk node under non-uniform weights.
    pub k: usize,
    /// Total non-leaf (cut-generating) node count in `setup.inputs.node_graph` —
    /// every trunk node, derived from the graph's own successor lists, never
    /// assumed equal to `t_trunk` by construction alone.
    pub n_nonleaf_nodes: usize,
}

/// Shared build for [`trunk_fan_setup_enumerated`]/[`trunk_fan_setup`]: builds
/// the stochastic context and study for `config`, injects
/// [`TRUNK_FAN_PRODUCTIVITY`] over `default_from_system`'s `0.0` placeholder,
/// and derives `n_nonleaf_nodes` from the resolved graph.
#[expect(
    clippy::expect_used,
    reason = "build_stochastic_context never errors on this fixture's fixed, internally consistent inputs"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "config is taken by value so callers pass an owned builder result inline; the body only borrows it for StudySetup::new"
)]
#[must_use]
fn trunk_fan_fixture(t_trunk: usize, k: usize, config: Config) -> TrunkFanFixture {
    let system = trunk_fan_system(t_trunk, k);
    let n_stages = t_trunk + 1;
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        in_sample_class_schemes(),
    )
    .expect("trunk_fan_fixture: build_stochastic_context must succeed");
    let mut hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    hydro_models.production = ProductionModelSet::new(
        vec![vec![
            ResolvedProductionModel::ConstantProductivity {
                productivity: TRUNK_FAN_PRODUCTIVITY,
            };
            n_stages
        ]],
        system.hydros(),
        n_stages,
    );
    let setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("trunk_fan_fixture: StudySetup::new must succeed");

    let n_nonleaf_nodes = (0..setup.inputs.node_graph.nodes.len())
        .map(NodePos)
        .filter(|&pos| !setup.inputs.node_graph.successors[pos].is_empty())
        .count();

    TrunkFanFixture {
        setup,
        t_trunk,
        k,
        n_nonleaf_nodes,
    }
}

/// The deterministic-trunk + terminal-fan [`StudySetup`] configured
/// `enumerated`: the node-native enumerated backward's headline linear-work
/// regression fixture (`backward_solves/iter == k + (t_trunk - 1)`, never
/// `k²`). Fixed seed, iteration-limit stopping rule, mirroring
/// [`k_fan_setup_enumerated`]/[`branching_tree_setup_enumerated`].
///
/// # Panics
///
/// Never in practice — every literal is a hand-checked, internally-consistent
/// fixture (as [`k_fan_setup`]); `StudySetup::new` only errors on a malformed
/// system or config, neither of which this builder can produce.
#[must_use]
pub fn trunk_fan_setup_enumerated(
    t_trunk: usize,
    k: usize,
    max_iterations: u32,
) -> TrunkFanFixture {
    trunk_fan_fixture(t_trunk, k, k_fan_config_enumerated(max_iterations))
}

/// [`trunk_fan_setup_enumerated`]'s `sampled` sibling: the SAME graph trained
/// with `forward_passes` trajectories per iteration, never `enumerated`.
/// `Traversal::Enumerated` always dispatches to the node-native backward
/// regardless of `backward_scheduler` (`by_scenario`/`by_node` is a
/// `Traversal::Sampled`-only axis), so the by-scenario/by-node backward-
/// scheduler comparison needs a fixture that genuinely exercises it — this is
/// that fixture.
///
/// # Panics
///
/// Never in practice — as [`trunk_fan_setup_enumerated`].
#[must_use]
pub fn trunk_fan_setup(
    t_trunk: usize,
    k: usize,
    forward_passes: u32,
    max_iterations: u32,
) -> TrunkFanFixture {
    trunk_fan_fixture(t_trunk, k, k_fan_config(forward_passes, max_iterations))
}

/// Canonical byte encoding of `setup`'s stage-LP builder facts, keyed by fact
/// group name, for the plan safety net's per-deck snapshot.
#[must_use]
pub fn template_fact_groups(setup: &StudySetup) -> BTreeMap<&'static str, Vec<u8>> {
    let mut groups = FactGroups::new();
    encode_stage_templates_facts(
        &setup.inputs.stage_data.stage_templates,
        &setup.inputs.stage_data.state,
        &mut groups,
    );
    encode_time_value_facts(&setup.inputs.stage_data.time_value, &mut groups);
    groups
}

/// Asserts `a` and `b` are the same LP; every `f64` compares by `to_bits()`, so
/// `0.0` and `-0.0` differ.
///
/// # Panics
///
/// Panics with `"{label}: <field>"` at the first field that differs.
pub fn assert_templates_byte_identical(a: &StageTemplate, b: &StageTemplate, label: &str) {
    let StageTemplate {
        num_cols,
        num_rows,
        num_nz,
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        n_state,
        col_scale,
        row_scale,
    } = a;
    assert_eq!(*num_cols, b.num_cols, "{label}: num_cols");
    assert_eq!(*num_rows, b.num_rows, "{label}: num_rows");
    assert_eq!(*num_nz, b.num_nz, "{label}: num_nz");
    assert_eq!(*n_state, b.n_state, "{label}: n_state");
    assert_eq!(*col_starts, b.col_starts, "{label}: col_starts");
    assert_eq!(*row_indices, b.row_indices, "{label}: row_indices");
    let bits = |xs: &[f64]| xs.iter().map(|v| v.to_bits()).collect::<Vec<u64>>();
    assert_eq!(bits(values), bits(&b.values), "{label}: values");
    assert_eq!(bits(col_lower), bits(&b.col_lower), "{label}: col_lower");
    assert_eq!(bits(col_upper), bits(&b.col_upper), "{label}: col_upper");
    assert_eq!(bits(objective), bits(&b.objective), "{label}: objective");
    assert_eq!(bits(row_lower), bits(&b.row_lower), "{label}: row_lower");
    assert_eq!(bits(row_upper), bits(&b.row_upper), "{label}: row_upper");
    assert_eq!(bits(col_scale), bits(&b.col_scale), "{label}: col_scale");
    assert_eq!(bits(row_scale), bits(&b.row_scale), "{label}: row_scale");
}

/// [`assert_templates_byte_identical`] over two multi-stage builds: asserts
/// `a`/`b` have the same stage count, then compares each stage pair.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`, or at the first stage/field that differs
/// (`"{label}: stage {n}: <field>"`).
pub fn assert_all_templates_byte_identical(a: &[StageTemplate], b: &[StageTemplate], label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: stage count must match ({} vs {})",
        a.len(),
        b.len()
    );
    for (stage, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_templates_byte_identical(x, y, &format!("{label}: stage {stage}"));
    }
}

#[cfg(test)]
mod byte_identity_tests {
    use super::assert_templates_byte_identical;
    use cobre_solver::StageTemplate;

    #[test]
    #[should_panic(expected = "probe: col_scale")]
    fn signed_zero_in_a_scale_factor_is_not_byte_identical() {
        let a = StageTemplate {
            num_cols: 0,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![],
            col_upper: vec![],
            objective: vec![],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            col_scale: vec![0.0],
            row_scale: vec![],
        };
        let b = StageTemplate {
            col_scale: vec![-0.0],
            ..a.clone()
        };
        assert_templates_byte_identical(&a, &b, "probe");
    }
}

#[cfg(test)]
mod trunk_fan_tests {
    use super::{
        ActiveSolver, NodePos, ResolvedProductionModel, TRUNK_FAN_PRODUCTIVITY, TrunkFanFixture,
        trunk_fan_setup_enumerated,
    };
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};

    /// Single-rank `Communicator` stub, mirroring `tests/common/mod.rs`'s
    /// `StubComm` (unreachable here — this module lives in `src/`, not
    /// `tests/`): broadcasts/reductions copy data locally; other collectives
    /// are no-ops.
    struct StubComm;

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].clone_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv.clone_from_slice(send);
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            1
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    /// Power self-check: `n_nonleaf_nodes` equals `t_trunk` (every trunk
    /// node is non-leaf; the last one owns the terminal fan) and the terminal
    /// fan has exactly `k` leaves — asserted against the fixture's own
    /// resolved graph, never a hard-coded literal — plus a training smoke
    /// check: the fixture trains without error. The exhaustive solves/cuts/
    /// bound/reproducibility gates live in `tests/node_native_backward_gate.rs`
    /// (slow-tests gated); this stays small and unconditional.
    #[test]
    fn trunk_fan_fixture_has_declared_shape_and_trains() {
        let TrunkFanFixture {
            mut setup,
            t_trunk,
            k,
            n_nonleaf_nodes,
        } = trunk_fan_setup_enumerated(3, 3, 3);

        assert_eq!(
            n_nonleaf_nodes, t_trunk,
            "every trunk node must be non-leaf, got {n_nonleaf_nodes} for t_trunk={t_trunk}"
        );
        let leaf_count = (0..setup.inputs.node_graph.nodes.len())
            .map(NodePos)
            .filter(|&pos| setup.inputs.node_graph.successors[pos].is_empty())
            .count();
        assert_eq!(
            leaf_count, k,
            "the terminal fan must have exactly k leaves, got {leaf_count} for k={k}"
        );
        for stage in 0..=t_trunk {
            match setup.hydro_models.production.model(0, stage) {
                ResolvedProductionModel::ConstantProductivity { productivity } => {
                    assert!(
                        (productivity - TRUNK_FAN_PRODUCTIVITY).abs() < 1e-12,
                        "stage {stage} productivity must be {TRUNK_FAN_PRODUCTIVITY}, got {productivity}"
                    );
                }
                fpha @ ResolvedProductionModel::Fpha { .. } => {
                    panic!("stage {stage} must be ConstantProductivity, got {fpha:?}")
                }
            }
        }

        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
            .expect("training must return Ok");
        assert!(
            outcome.error.is_none(),
            "trunk+fan fixture must train without error: {:?}",
            outcome.error
        );
    }
}

#[cfg(test)]
mod stage_context_fixture_tests {
    use super::{
        StageContextFixture, equipment_free_geometry, geometry_with_load_balance,
        permissive_state_boxes, state_layout, transit_bucket_only_template,
    };
    use crate::setup::node_graph::StageIdx;

    #[test]
    fn stage_context_fixture_derives_counts_from_owners() {
        let state = state_layout(2, 0);
        let templates = vec![transit_bucket_only_template(1, state.n_state); 2];
        let state_boxes = permissive_state_boxes(state.n_state, 2);
        let geometry_per_stage = vec![geometry_with_load_balance(0, 1, 3); 2];
        let load_bus_indices = vec![0_usize];
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
            .load_bus_indices(&load_bus_indices);
        let ctx = fixture.ctx();
        assert_eq!(ctx.load_bus_indices.len(), 1);
        assert_eq!(ctx.block_count(StageIdx(0)), 3);
        assert_eq!(ctx.block_count(StageIdx(1)), 3);
    }

    #[test]
    #[should_panic(expected = "every stage must have one geometry")]
    fn stage_context_fixture_rejects_a_geometry_per_stage_length_mismatch() {
        let state = state_layout(1, 0);
        let templates = vec![transit_bucket_only_template(1, state.n_state); 2];
        let state_boxes = permissive_state_boxes(state.n_state, 2);
        let geometry_per_stage = equipment_free_geometry(&[0]);
        let _ = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage);
    }
}
