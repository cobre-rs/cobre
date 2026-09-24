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

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use chrono::NaiveDate;
use cobre_core::scenario::{InflowModel, LoadModel, SamplingScheme};
use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, Transition};
use cobre_core::{
    Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, CascadeTopology,
    ContractBlockBounds, DeficitSegment, EntityId, HorizonGraph, Hydro, HydroBlockBounds,
    HydroGenerationModel, HydroPenalties, HydroStageBounds, HydroStorage, HydroUnitGroup,
    InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod,
    PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, System,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
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
use crate::context::{StageContext, TrainingContext};
use crate::cut::pool::CutPool;
use crate::error::SddpError;
use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, FphaPlane, PrepareHydroModelsResult, ProductionModelSet,
    ResolvedProductionModel,
};
use crate::lead_time::AnticipatedResolution;
#[cfg(test)]
use crate::lp::builder::StateBox;
use crate::lp::builder::{
    FactGroups, PatchBuffer, ResolvedTables, StageGeometry, StageLayout, TemplateBuildCtx,
    encode_stage_templates_facts,
};
use crate::lp::indexer::{
    CutStateProjection, HydroCellIndex, StateDim, StateSpace, StudyDimensions, ThermalSys,
};
use crate::noise::{DownstreamAccumState, LagAccumState};
use crate::policy::policy_load::{
    FullFcf, PolicyLoadProof, PolicyStageManifest, validate_policy_load,
};
use crate::resolved_parameters::ResolvedParameters;
use crate::risk_measure::BackwardOutcome;
use crate::setup::PostStudyResolved;
use crate::setup::node_graph::{
    NodeGraph, NodeId, NodePos, OpeningSource, StageIdx, build_node_graph,
    enumerated_node_visit_counts, enumerated_scenario_count,
};
use crate::solve::stage_solve::{StageInputs, assemble_outgoing_state, run_stage_solve};
use crate::solver_stats::SolverStatsDelta;
use crate::training::backward::{extract_state_duals_only, write_opening_outcome};
use crate::training::stage_solve_prep::{
    InflowNoise, LoadNoise, StageSolvePrep, StageSolvePrepParams, StateSource,
};
use crate::trajectory::TrajectoryRecord;
use crate::workspace::{CapturedBasis, ScratchBuffers, SolverWorkspace, WorkspaceSizing};
use cobre_core::scenario::{ExternalLoadRow, ExternalScenarioRow};
use cobre_solver::{
    ActiveSolver, Basis, BasisStatus, RowBatch, SolutionView, SolverError, SolverInterface,
    SolverStatistics, StageTemplate,
};

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
    /// Maximum `lead_stages` across the anticipated thermals.
    pub k_max: usize,
    /// Mapping from anticipated-local position to global thermal index.
    pub anticipated_thermal_indices: Vec<usize>,
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
            k_max: 0,
            anticipated_thermal_indices: Vec::new(),
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
#[cfg(test)]
pub(crate) fn permissive_state_boxes(n_state: usize, n_stages: usize) -> Vec<StateBox> {
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
/// The anticipated identity list defaults to `0..n_anticipated`.
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
    k_max: usize,
) -> GeometryDims {
    GeometryDims {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        n_anticipated,
        k_max,
        anticipated_thermal_indices: (0..n_anticipated).collect(),
        ..Default::default()
    }
}

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

/// Identity [`HydroCellIndex`] for `n_hydros` single-bus hydros
/// (`cells_of(h) == h..h+1` for every `h`) — the single shared builder for every
/// fixture in the crate that needs a `HydroCellIndex` but is not itself testing
/// the multi-bus partition. Safe to over-size: a caller passing more than its
/// fixture's actual hydro count still gets a correct `cells_of` for every hydro
/// it actually queries.
#[must_use]
pub fn identity_hydro_cell_index(n_hydros: usize) -> HydroCellIndex {
    HydroCellIndex::build(&(0..n_hydros).map(geometry_hydro).collect::<Vec<_>>())
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
    ProductionModelSet::new(models, hydro_count, 1)
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
/// describe and delegates to `StageLayout::new`/`StageLayout::geometry` — the
/// single owner of the offset arithmetic.
#[must_use]
// Rationale: `fpha_hydro_indices`/`evap_hydro_indices` stay owned `Vec<usize>` —
// the signature is a stability contract its call sites depend on — even
// though the body only borrows them (`StageLayout::new` re-derives the
// authoritative membership from `ctx.hydros`/`production_models`/
// `evaporation_models`, not from the caller's raw list).
#[allow(clippy::needless_pass_by_value)]
// Rationale: clippy::similar_names flags `state` next to `stage`; both names
// are established (the `StageLayout`/`StageData` field is `state`, the
// per-stage input is `stage`), so renaming either would obscure intent rather
// than clarify it — mirrors `build_single_stage_template`.
#[allow(clippy::similar_names)]
pub fn geometry(
    dims: &GeometryDims,
    fpha_hydro_indices: Vec<usize>,
    fpha_planes: &[usize],
    evap_hydro_indices: Vec<usize>,
) -> StageGeometry {
    let hydros: Vec<Hydro> = (0..dims.hydro_count).map(geometry_hydro).collect();
    let hydro_cell_index = HydroCellIndex::build(&hydros);
    let buses: Vec<Bus> = (0..dims.n_buses)
        .map(|idx| geometry_bus(idx, dims.max_deficit_segments))
        .collect();
    let production_models =
        geometry_production_models(dims.hydro_count, &fpha_hydro_indices, fpha_planes);
    let evaporation_models = geometry_evaporation_models(dims.hydro_count, &evap_hydro_indices);

    let bounds = ResolvedBounds::empty();
    let penalties = ResolvedPenalties::empty();
    let resolved_generic_bounds = ResolvedGenericConstraintBounds::empty();
    let resolved_load_factors = ResolvedLoadFactors::empty();
    let resolved_ncs_bounds = ResolvedNcsBounds::empty();
    let resolved_ncs_factors = ResolvedNcsFactors::empty();
    let resolved_parameters = ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
        cost_scale_factor: 1_000_000.0,
    };
    let cascade = CascadeTopology::build(&[]);
    let par_lp = PrecomputedPar::default();
    let anticipated_lead_stages = vec![dims.k_max; dims.n_anticipated];

    let ctx = TemplateBuildCtx {
        hydros: &hydros,
        thermals: &[],
        lines: &[],
        buses: &buses,
        load_models: &[],
        cascade: &cascade,
        hydro_cell_index: &hydro_cell_index,
        resolved: ResolvedTables {
            bounds: &bounds,
            penalties: &penalties,
            resolved_generic_bounds: &resolved_generic_bounds,
            resolved_load_factors: &resolved_load_factors,
            resolved_ncs_bounds: &resolved_ncs_bounds,
            resolved_ncs_factors: &resolved_ncs_factors,
            resolved_parameters: &resolved_parameters,
        },
        hydro_pos: BTreeMap::new(),
        thermal_pos: BTreeMap::new(),
        line_pos: BTreeMap::new(),
        bus_pos: BTreeMap::new(),
        par_lp: &par_lp,
        production_models: &production_models,
        evaporation_models: &evaporation_models,
        generic_constraints: &[],
        non_controllable_sources: &[],
        pumping_stations: &[],
        pumping_pos: BTreeMap::new(),
        n_pumping: 0,
        contracts: &[],
        contract_pos: BTreeMap::new(),
        n_contract_import: 0,
        n_contract_export: 0,
        diversion_upstream: HashMap::new(),
        n_hydros: dims.hydro_count,
        n_thermals: dims.n_thermals,
        n_lines: dims.n_lines,
        n_buses: dims.n_buses,
        max_par_order: dims.max_par_order,
        n_anticipated: dims.n_anticipated,
        k_max: dims.k_max,
        anticipated_lead_stages: anticipated_lead_stages.clone(),
        anticipated_thermal_indices: dims
            .anticipated_thermal_indices
            .iter()
            .map(|&t| ThermalSys::new(t))
            .collect(),
        anticipated_windows: vec![(None, None); dims.n_anticipated],
        anticipated_resolution: AnticipatedResolution::default(),
        study_stage_ids: Vec::new(),
        delivery_stage_ids: Vec::new(),
        has_penalty: dims.has_inflow_penalty,
        delivery_cumulative_discount_factors: vec![1.0],
        delivery_total_hours: vec![744.0],
        filling_v_target: BTreeMap::new(),
        arc_stage_weights: HashMap::new(),
        arc_spread_chrono: HashMap::new(),
        arc_arrival_density: HashMap::new(),
        per_stage_mask: Vec::new(),
        post_study_resolved: PostStudyResolved::default(),
    };

    let state = state_layout_full(
        dims.hydro_count,
        dims.max_par_order,
        dims.n_anticipated,
        dims.k_max,
        anticipated_lead_stages,
    );
    let stage = geometry_stage(dims.n_blks);

    StageLayout::new(&ctx, &state, &stage, 0).geometry(BlockMode::Parallel)
}

/// Build the empty-equipment role-(b) [`StageGeometry`] (every range `0..0`).
///
/// The `_hydro_count` / `_max_par_order` arguments are ignored — accepted only for
/// call-site symmetry with [`geometry`].
#[must_use]
pub fn geom(_hydro_count: usize, _max_par_order: usize) -> StageGeometry {
    StageGeometry::default()
}

/// Build a finalized storage+lag [`StateSpace`] (no anticipated thermals) with the
/// full `max_par_order` lag stride for every hydro — the dense coverage
/// `crate::setup::resolve_state_layout` finalizes with no per-hydro AR truncation.
#[must_use]
pub fn state_layout(hydro_count: usize, max_par_order: usize) -> StateSpace {
    state_layout_full(hydro_count, max_par_order, 0, 0, Vec::new())
}

/// Build a finalized [`StateSpace`] from explicit state-vector dimensions,
/// including anticipated thermals. Lag coverage is dense (full `max_par_order`).
///
/// `anticipated_lead_stages` must have length `n_anticipated` and its max (when
/// non-empty) must equal `k_max`.
#[must_use]
pub fn state_layout_full(
    hydro_count: usize,
    max_par_order: usize,
    n_anticipated: usize,
    k_max: usize,
    anticipated_lead_stages: Vec<usize>,
) -> StateSpace {
    state_layout_with_transit_buckets(
        hydro_count,
        max_par_order,
        0,
        Vec::new(),
        n_anticipated,
        k_max,
        anticipated_lead_stages,
    )
}

/// Build a finalized [`StateSpace`] with a declared travel-time bucket block
/// (`transit_buckets_out`/`transit_buckets_in`), optionally combined with anticipated
/// thermals. `effective_lag_count` is dense (full `max_par_order` for every
/// hydro), matching [`state_layout_full`].
#[must_use]
pub fn state_layout_with_transit_buckets(
    hydro_count: usize,
    max_par_order: usize,
    n_buckets: usize,
    transit_bucket_column_order: Vec<(usize, usize)>,
    n_anticipated: usize,
    k_max: usize,
    anticipated_lead_stages: Vec<usize>,
) -> StateSpace {
    let effective_lag_count = vec![max_par_order; hydro_count];
    StateSpace::new(
        hydro_count,
        max_par_order,
        n_buckets,
        transit_bucket_column_order,
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_count,
    )
}

/// Bucket-only [`StageTemplate`]: `num_cols` free columns, zero rows. `n_hydro = 0`
/// so noise transformation never runs.
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
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
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
/// stage geometry from. `has_ncs` is always `false`: these fixtures never model NCS
/// (production sets it from `!ncs_col_starts.is_empty()`).
#[must_use]
pub fn study_dims_for(dims: &GeometryDims) -> StudyDimensions {
    StudyDimensions {
        n_thermals: dims.n_thermals,
        n_lines: dims.n_lines,
        n_buses: dims.n_buses,
        max_deficit_segments: dims.max_deficit_segments,
        has_ncs: false,
        has_inflow_penalty: dims.has_inflow_penalty,
        has_withdrawal: dims.hydro_count > 0,
        has_operational_violations: dims.hydro_count != 0,
        anticipated_thermal_indices: dims.anticipated_thermal_indices.clone(),
        n_pumping: 0,
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
#[allow(clippy::expect_used)]
// Rationale: matching state_dimension/num_stages with an empty manifest on
// both sides cannot hit validate_policy_load's error paths (state_dimension
// and num_stages equality hold trivially; an empty manifest short-circuits
// identity comparison with a warning, never an error).
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
    validate_policy_load::<FullFcf>(&manifest, &manifest)
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
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
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
#[allow(clippy::expect_used)]
// Rationale: every caller passes a literal, calendar-valid date;
// `from_ymd_opt` only returns `None` for an out-of-range one.
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
#[allow(clippy::expect_used)]
// Rationale: every caller passes a small pool count; `checked_add_months`
// only overflows past `NaiveDate`'s year range.
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
#[allow(clippy::expect_used)]
// Rationale: every caller passes a day-01 YYYYMMDD anchor in range; decoding
// then re-encoding one only fails on a malformed or out-of-range stamp.
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
#[allow(clippy::expect_used)]
// Rationale: every caller passes a small stage count; the `u32`→`i32` casts
// only fail past `i32::MAX` stages.
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
#[allow(clippy::expect_used)]
// Rationale: `write_policy_checkpoint` only fails on a write-path IO error,
// never on this fixture's own well-formed payload.
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
/// point (`LoadNoise::Present`, `InflowNoise::Transform`) — no probe-side
/// reimplementation of the patch pipeline (lag-folded water-balance RHS, NCS
/// availability).
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
        load_noise: LoadNoise::Present,
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
        load_noise: LoadNoise::Present,
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

/// Drive the real read-back canonicalization and the real backward
/// cut-intercept write for one opening.
///
/// The caller supplies a RAW producer-stage state (`raw_producer_state`,
/// possibly outside the producer's admissible box); production does the
/// clamping — [`assemble_outgoing_state`] canonicalizes it exactly as the
/// forward/simulation read-back seam does. That canonical value is the single
/// `x_hat` threaded into BOTH the pin ([`patch_backward_opening_for_probe`] →
/// `StageSolvePrep::run` → `set_col_bounds`) and the intercept
/// ([`write_opening_outcome`]), mirroring the backward opening loop. The state
/// the LP was pinned at is recovered separately from the solved primal — an
/// independent data path — so [`CanonicalCutProbe::pinned_x_hat`] cross-checks
/// the pin against the cut instead of re-deriving the intercept's own formula.
///
/// # Panics
///
/// Panics if `stage.0 == 0` (the backward never solves stage 0, so no producer
/// box exists) or if `raw_producer_state.len() != StateSpace::n_state`.
///
/// # Errors
///
/// Propagates [`SddpError`] from the stage solve.
// Rationale (too_many_arguments): a test-only probe that threads the same
// borrows the production backward opening loop passes individually (workspace,
// contexts, pool, cut-state, stage/node ids, trial state, noise); grouping them
// into a struct would diverge from the call shape it mirrors.
#[allow(clippy::too_many_arguments)]
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
    let ds_par_order = ctx.downstream_par_order;
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

    let template = ctx.template(stage);
    ws.solver.reset_solver_state();
    ws.solver.load_model(template);
    patch_backward_opening_for_probe(ws, ctx, training_ctx, stage, &canonical_x_hat, raw_noise);

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
        &canonical_x_hat,
        &stats_before,
        &stats_after,
    );

    Ok(CanonicalCutProbe {
        outcome: ws.backward_accum.outcomes[0].clone(),
        canonical_x_hat,
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
#[allow(clippy::expect_used)]
// Rationale: build_node_graph only returns Err for a transition/node naming
// an undeclared id; HorizonGraph::default() carries no nodes/transitions at
// all, so that error path is unreachable here.
#[must_use]
pub fn chain_node_graph(stochastic: &StochasticContext) -> NodeGraph {
    let n_stages = stochastic.n_stages();
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
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

/// Test-support view of the crate-internal chronological-noise predicate
/// ([`crate::setup::chronological_multi_block_inflow_noise`]): the flagged
/// `(stage.id, hydro.id.0)`, if any.
#[must_use]
pub fn chronological_multi_block_inflow_noise(
    system: &System,
    par: &PrecomputedPar,
) -> Option<(i32, i32)> {
    crate::setup::chronological_multi_block_inflow_noise(system, par)
        .map(|(stage, hydro)| (stage.id, hydro.id.0))
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

/// The declared `nodes[]`/`transitions[]` K-fan: root (id `0`) branches into fan
/// nodes `1..=k` under strictly non-uniform weights `i / Σj` (never a uniform
/// `1/k` split — a uniform split would make every reduction order sum identical
/// terms, defeating the canonical-order gate's power), each fan node `i` then
/// deterministically reaching its own leaf `k+i`. `num_nodes = 2k+1`,
/// `n_pools = k+2` (root + k fan pools + one shared leaf pool) —
/// `num_nodes > n_pools` for every `k >= 2`, so a canonical-node-position-as-pool-id
/// conflation bug would misroute or overflow a pool here, where it cannot on a
/// chain (`node_index == pool_id` there hides the bug).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
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
    // A reversed declaration order must resolve to the identical canonical node
    // graph (build_node_graph sorts nodes by id, out-edges by target) — the input
    // for the enumerated engine's declaration-order-invariance gate.
    if reversed {
        nodes.reverse();
        transitions.reverse();
    }
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions,
        nodes,
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    }
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
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[allow(clippy::expect_used)]
// Rationale: the literal calendar dates below are valid by construction
// (checked at write time); `from_ymd_opt` only returns `None` for an
// out-of-range calendar date.
fn k_fan_stage(index: usize, id: i32, state_config: StageStateConfig) -> Stage {
    Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
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
// RATIONALE: one linear fixture that builds a self-consistent multi-stage
// System; splitting it scatters the study across single-use helpers and hides
// the whole-study shape a test reads at a glance.
#[allow(clippy::too_many_lines, clippy::expect_used)]
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
// Rationale: one linear entity/bounds/penalties assembly shared by every fan/chain
// fixture in this file; splitting the newest knob into a struct would cost a
// one-off type for a single call site, while every existing caller already reads
// as a flat parameter list at its call site.
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::expect_used
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
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
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
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
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
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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
/// always re-derived from these fields or from `setup.node_graph` directly.
#[derive(Debug)]
pub struct KFanFixture {
    /// The built study: a single-hydro, single-bus system over the declared
    /// K-fan graph, trained `sampled` with a fixed seed.
    pub setup: StudySetup,
    /// Fan-out width: `setup.node_graph` has `k` nodes at
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
#[allow(clippy::expect_used)]
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
// `config` is taken by value so callers pass an owned builder result inline; the
// body only borrows it for `StudySetup::new`.
#[allow(clippy::expect_used, clippy::needless_pass_by_value)]
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
    let enumerated = enumerated_scenario_count(&setup.node_graph)
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
#[allow(clippy::expect_used)]
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

    #[allow(clippy::expect_used)]
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

    #[allow(clippy::expect_used)]
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

/// The `[hydro | load-bus | NCS]` raw-noise vector length:
/// `hydro_count + n_load_buses + n_stochastic_ncs`.
#[must_use]
pub fn raw_noise_len(setup: &StudySetup) -> usize {
    setup.stage_data.state.hydro_count
        + setup.stage_data.stage_templates.n_load_buses
        + setup.stochastic.n_stochastic_ncs()
}

/// The `[hydro | load-bus | NCS]` standardized noise draw for `node_pos`: an
/// `External` node reads its own scenario column from the standardized external
/// inflow library; a `Generated` node draws zeros — the oracle fixtures carry
/// `std == 0` on every generated stage, so `transform_inflow_noise` recovers the
/// mean regardless.
fn oracle_raw_noise(setup: &StudySetup, node_pos: NodePos) -> Vec<f64> {
    let stage = setup.node_graph.nodes[node_pos].stage;
    let n_hydros = setup.stage_data.state.hydro_count;
    let mut raw = vec![0.0_f64; raw_noise_len(setup)];
    let openings = setup.node_graph.nodes[node_pos].openings;
    if openings.source == OpeningSource::External
        && let Some(lib) = setup.scenario_libraries.training.external_inflow.as_ref()
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
/// [`StudySetup::initial_state`]; the extensive-form composer frees and couples them for
/// non-root nodes.
///
/// # Panics
///
/// Panics if `node_pos` (or its resolved stage) is out of range, or if the
/// template is absent after [`StageSolvePrep::run`].
#[must_use]
pub fn capture_patched_node_template(setup: &StudySetup, node_pos: NodePos) -> StageTemplate {
    let raw_noise = oracle_raw_noise(setup, node_pos);
    capture_patched_node_template_with_raw_noise(setup, node_pos, &raw_noise, &setup.initial_state)
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
    let n_hydros = setup.stage_data.state.hydro_count;
    assert_eq!(
        inflow_eta.len(),
        n_hydros,
        "inflow_eta must hold one standardized draw per hydro"
    );
    let mut raw_noise = vec![0.0_f64; raw_noise_len(setup)];
    raw_noise[..n_hydros].copy_from_slice(inflow_eta);
    capture_patched_node_template_with_raw_noise(setup, node_pos, &raw_noise, &setup.initial_state)
}

/// [`capture_patched_node_template`] at a caller-chosen raw noise vector and
/// incoming state, in place of the node's own oracle draw and
/// [`StudySetup::initial_state`].
///
/// # Panics
///
/// Panics if `raw_noise.len() != raw_noise_len(setup)`, if
/// `incoming_state.len() != setup.stage_data.state.n_state`, if `node_pos`
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
        raw_noise_len(setup),
        "raw_noise must be the `[hydro | load-bus | NCS]` raw-noise length"
    );
    assert_eq!(
        incoming_state.len(),
        setup.stage_data.state.n_state,
        "incoming_state must hold one entry per state dimension"
    );
    capture_patched_node_template_with_raw_noise(setup, node_pos, raw_noise, incoming_state)
}

/// Opening `opening`'s raw `[hydro | load-bus | NCS]` noise vector at
/// `node_pos`: a `Generated` node's own draw from
/// [`StochasticContext::opening_tree`]; an `External` node's sole opening
/// (`0`), which equals [`oracle_raw_noise`].
///
/// # Panics
///
/// Panics if `opening >= node_pos`'s opening count, or (for an `External`
/// node) if `opening != 0`.
#[must_use]
pub fn node_opening_noise(setup: &StudySetup, node_pos: NodePos, opening: usize) -> Vec<f64> {
    let stage = setup.node_graph.nodes[node_pos].stage;
    let openings = setup.node_graph.nodes[node_pos].openings;
    assert!(
        opening < openings.len,
        "opening must be < node_pos's opening count"
    );
    match openings.source {
        OpeningSource::Generated => setup
            .stochastic
            .opening_tree()
            .opening(stage.0, openings.offset + opening)
            .to_vec(),
        OpeningSource::External => {
            assert_eq!(opening, 0, "an External node has exactly one opening");
            oracle_raw_noise(setup, node_pos)
        }
    }
}

#[allow(clippy::expect_used)]
fn capture_patched_node_template_with_raw_noise(
    setup: &StudySetup,
    node_pos: NodePos,
    raw_noise: &[f64],
    incoming_state: &[f64],
) -> StageTemplate {
    let stage = setup.node_graph.nodes[node_pos].stage;
    let base = setup.stage_data.stage_templates.templates[stage.0].clone();
    let mut solver = TemplateCaptureSolver {
        template: Some(base),
    };

    let space = &setup.stage_data.state;
    let n_load_buses = setup.stage_data.stage_templates.n_load_buses;
    let max_blocks = setup.loop_params.max_blocks;
    let mut patch_buf = PatchBuffer::new(
        space.hydro_count,
        space.max_par_order,
        n_load_buses,
        max_blocks,
        space.n_buckets,
        space.n_anticipated,
        space.k_max,
    );
    let mut scratch = ScratchBuffers::new(WorkspaceSizing {
        hydro_count: space.hydro_count,
        max_par_order: space.max_par_order,
        n_load_buses,
        max_blocks,
        n_buckets: space.n_buckets,
        downstream_par_order: setup.downstream_par_order,
        max_openings: 0,
        initial_pool_capacity: 0,
        n_state: space.n_state,
        max_local_fwd: 0,
        noise_dim: 0,
        n_anticipated: space.n_anticipated,
        k_max: space.k_max,
    });

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let params = StageSolvePrepParams {
        state_source: StateSource(incoming_state),
        load_noise: LoadNoise::Present,
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
    setup.initial_state.clone()
}

/// `stage`'s admissible box (per outgoing state dimension) as plain `(lower,
/// upper)` vectors. `StateBox`/`StageTemplates::state_boxes` are `pub(crate)`, so
/// this returns their data rather than naming either type in a `pub` signature.
#[must_use]
pub fn stage_state_box_bounds(setup: &StudySetup, stage: usize) -> (Vec<f64>, Vec<f64>) {
    let state_box = &setup.stage_data.stage_templates.state_boxes[stage];
    (state_box.lower.clone(), state_box.upper.clone())
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
#[allow(clippy::expect_used)]
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
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
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions,
        nodes,
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    }
}

/// The terminal-Generated fan control [`StudySetup`]: a root fanning into `k`
/// interchangeable Generated leaves that share one pool, trained `enumerated`.
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[allow(clippy::expect_used)]
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
// RATIONALE: one linear fixture that builds a self-consistent external-distinct
// fan study; splitting it scatters the study across single-use helpers and hides
// the whole-study shape a test reads at a glance.
#[allow(
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
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

// Rationale (too_many_lines): one linear pass assembling a StudySetup fixture —
// policy graph, per-node external inflow openings, and config — from literal deck
// data; splitting it would scatter shared locals across helpers for no
// test-readability gain.
#[allow(
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
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

    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
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
        HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions,
            nodes,
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        }
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
// RATIONALE: one linear fixture that builds a self-consistent external-root
// fan study; splitting it scatters the study across single-use helpers and hides
// the whole-study shape a test reads at a glance.
#[allow(
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
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

    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
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
        HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions,
            nodes,
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        }
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

#[allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
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
        // A reversed declaration order must resolve to the identical canonical node
        // graph (build_node_graph sorts by id, out-edges by target).
        if reversed {
            nodes.reverse();
            transitions.reverse();
        }
        HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions,
            nodes,
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        }
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
        1,
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
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
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
    // A reversed declaration order must resolve to the identical canonical node
    // graph (build_node_graph sorts nodes by id, out-edges by target) — the input
    // for the declaration-order-invariance gate.
    if reversed {
        nodes.reverse();
        transitions.reverse();
    }
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions,
        nodes,
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    }
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

#[allow(clippy::expect_used)]
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
#[allow(clippy::expect_used)]
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
/// pool from its successor's `state_config`). `StageData::cut_state_layouts`
/// itself is `pub(crate)`, unreachable from an integration test without this
/// accessor.
#[must_use]
pub fn pool_cut_state_dimensions(setup: &StudySetup) -> Vec<usize> {
    setup
        .stage_data
        .cut_state_layouts
        .iter()
        .map(CutStateProjection::n_slots)
        .collect()
}

// ── Extensive-form oracle (shared by `branching_value_oracle.rs` and any other
//    integration test that needs the graph's true first-stage value) ─────────

/// Marginal visit probability of every node in `setup.node_graph`: `P(root) = 1`,
/// propagated `P(child) += P(node)·prob(node→child)` in ascending-stage order. On
/// a tree this is the product of edge probabilities on the unique root→node path
/// — the weight each node copy's stage cost carries in
/// [`extensive_form_optimum`]'s objective.
#[must_use]
pub fn node_visit_probabilities(setup: &StudySetup) -> Vec<f64> {
    let g = &setup.node_graph;
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

/// The extensive-form LP optimum — `setup.node_graph`'s true first-stage value,
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
#[allow(
    clippy::expect_used,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn extensive_form_optimum(setup: &StudySetup) -> f64 {
    let g = &setup.node_graph;
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
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
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
    solution.objective * setup.stage_data.stage_templates.cost_scale_factor
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
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
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions,
        nodes,
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    }
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
#[allow(clippy::expect_used)]
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
        1,
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
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
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions,
        nodes,
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    }
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
/// fields or from `setup.node_graph` directly.
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
    /// Total non-leaf (cut-generating) node count in `setup.node_graph` —
    /// every trunk node, derived from the graph's own successor lists, never
    /// assumed equal to `t_trunk` by construction alone.
    pub n_nonleaf_nodes: usize,
}

/// Shared build for [`trunk_fan_setup_enumerated`]/[`trunk_fan_setup`]: builds
/// the stochastic context and study for `config`, injects
/// [`TRUNK_FAN_PRODUCTIVITY`] over `default_from_system`'s `0.0` placeholder,
/// and derives `n_nonleaf_nodes` from the resolved graph.
// `config` is taken by value so callers pass an owned builder result inline;
// the body only borrows it for `StudySetup::new`.
#[allow(clippy::expect_used, clippy::needless_pass_by_value)]
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
        1,
        n_stages,
    );
    let setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("trunk_fan_fixture: StudySetup::new must succeed");

    let n_nonleaf_nodes = (0..setup.node_graph.nodes.len())
        .map(NodePos)
        .filter(|&pos| !setup.node_graph.successors[pos].is_empty())
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
    encode_stage_templates_facts(&setup.stage_data.stage_templates, &mut groups);
    groups
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
        let leaf_count = (0..setup.node_graph.nodes.len())
            .map(NodePos)
            .filter(|&pos| setup.node_graph.successors[pos].is_empty())
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
