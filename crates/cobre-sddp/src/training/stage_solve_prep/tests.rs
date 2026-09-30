//! [`StageSolvePrep::run`] must reproduce the open-coded forward block
//! (`training/forward/stage_solve.rs`) call-for-call for the same fixture.

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
use cobre_core::entities::non_controllable::NonControllableSource;
use cobre_core::scenario::{
    CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    LoadModel, NcsModel, SamplingScheme,
};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
use cobre_solver::{
    Basis, RowBatch, SolutionView, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::StochasticContext;
use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

use super::{InflowNoise, StageSolvePrep, StageSolvePrepParams, StateSource};
use crate::{
    context::TrainingContext,
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    lp::builder::{PatchBuffer, StageGeometry, StateBox},
    lp::indexer::{BlockGrid, BlockRowFamily, StudyDimensions},
    noise::{
        build_dense_ncs_col_indices, gather_dense_ncs_bounds, transform_inflow_noise,
        transform_load_noise, transform_ncs_noise,
    },
    setup::node_graph::StageIdx,
    test_support::{
        StageContextFixture, all_enabled_cut_state_layouts, equipment_free_geometry,
        geometry_with_load_balance, state_layout, study_dims,
    },
    workspace::{NoisePreallocation, ScratchBuffers, WorkspaceSizing},
};

/// Single-hydro, single-stage [`StochasticContext`] with a real PAR(0) inflow
/// model — exercises the `InflowNoise::Transform` path exactly as the forward
/// site does, rather than vacuously skipping it. `load` additionally declares
/// a stochastic load-bus noise model on the same bus, when present.
fn make_stochastic_context(load: Option<LoadModel>) -> StochasticContext {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
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
    hydro.declare_mirror_unit_group(EntityId(0));

    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
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
    };

    let inflow = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };

    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(vec![stage])
        .inflow_models(vec![inflow])
        .correlation(correlation);
    if let Some(load_model) = load {
        builder = builder.load_models(vec![load_model]);
    }
    let system = builder.build().unwrap();

    build_stochastic_context(
        &system,
        42,
        None,
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

/// N=1, L=0 template: columns `[storage(0), storage_in(1), theta(2)]`, one
/// water-balance row (`storage_in` has the row-0 nonzero), unscaled.
fn minimal_forward_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0],
        row_upper: vec![0.0],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Single hydro, single stochastic load bus, 2 rows (load balance + z-inflow).
fn single_hydro_load_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0, 0.0],
        row_upper: vec![0.0, 0.0],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Every dimension unbounded — the pin-time box-membership assert is vacuous.
fn unbounded_state_box(n_state: usize) -> StateBox {
    StateBox {
        lower: vec![f64::NEG_INFINITY; n_state],
        upper: vec![f64::INFINITY; n_state],
    }
}

fn minimal_sizing() -> WorkspaceSizing {
    WorkspaceSizing {
        max_openings: 1,
        initial_pool_capacity: 1,
        max_local_fwd: 1,
        noise: NoisePreallocation::StochasticDim,
    }
}

/// Records every `set_col_bounds`/`set_row_bounds` call verbatim; `solve` is
/// never exercised by the solve-preparation pipeline.
#[derive(Default)]
struct RecordingSolver {
    col_bounds_calls: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)>,
    row_bounds_calls: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)>,
}

impl SolverInterface for RecordingSolver {
    type Profile = cobre_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &Self::Profile) {}

    fn solver_name_version(&self) -> String {
        "RecordingSolver 0.0.0".to_string()
    }

    fn load_model(&mut self, _template: &StageTemplate) {}

    fn add_rows(&mut self, _rows: &RowBatch) {}

    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.row_bounds_calls
            .push((indices.to_vec(), lower.to_vec(), upper.to_vec()));
    }

    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.col_bounds_calls
            .push((indices.to_vec(), lower.to_vec(), upper.to_vec()));
    }

    fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        unreachable!("solve() is not exercised by the solve-preparation pipeline")
    }

    fn get_basis(&mut self, out: &mut Basis) {
        crate::test_support::fill_consistent_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&SolverStatistics::default());
    }

    fn name(&self) -> &'static str {
        "Recording"
    }
}

#[test]
fn run_matches_open_coded_forward_block_for_minimal_fixture() {
    let state = state_layout(1, 0);
    let stochastic = make_stochastic_context(None);
    let template = minimal_forward_template();
    let templates = vec![template.clone()];
    let state_boxes = vec![unbounded_state_box(state.n_state)];
    let geometry = equipment_free_geometry(&[1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let study_dims = study_dims();
    let training_ctx = TrainingContext {
        node_graph: &crate::test_support::chain_node_graph(&stochastic),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &all_enabled_cut_state_layouts(&state, 1),
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
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
    };

    let current_state = vec![42.0_f64];
    let raw_noise = vec![0.3_f64];
    let sizing = minimal_sizing();

    // ---- reference: the literal open-coded forward block for this fixture
    // (no load buses, no anticipated thermals, no NCS in scope) ----
    let mut reference_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    let mut reference_solver = RecordingSolver::default();
    let mut reference_patch_buf = PatchBuffer::new(&state, &[], &[]);

    transform_inflow_noise(
        &raw_noise,
        StageIdx(0),
        &current_state,
        &training_ctx,
        &mut reference_scratch,
    );
    reference_patch_buf.fill_col_state_patches(&state, &current_state, &template.col_scale, None);
    reference_patch_buf.fill_z_inflow_patches(
        &state,
        &reference_scratch.z_inflow_rhs_buf,
        &template.row_scale,
    );
    let cp = reference_patch_buf.state_col_patch_count();
    reference_solver.set_col_bounds(
        &reference_patch_buf.col_indices[..cp],
        &reference_patch_buf.col_lower[..cp],
        &reference_patch_buf.col_upper[..cp],
    );
    let pc = reference_patch_buf.forward_patch_count();
    reference_solver.set_row_bounds(
        &reference_patch_buf.indices[..pc],
        &reference_patch_buf.lower[..pc],
        &reference_patch_buf.upper[..pc],
    );

    // ---- owner: StageSolvePrep::run configured the way forward would ----
    let mut owner_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    let mut owner_solver = RecordingSolver::default();
    let mut owner_patch_buf = PatchBuffer::new(&state, &[], &[]);
    let params = StageSolvePrepParams {
        state_source: StateSource(&current_state),
        inflow_noise: InflowNoise::Transform,
        raw_noise: &raw_noise,
    };
    StageSolvePrep::run(
        &mut owner_solver,
        &mut owner_patch_buf,
        &mut owner_scratch,
        &ctx,
        &training_ctx,
        StageIdx(0),
        &params,
    );

    assert_eq!(
        owner_solver.col_bounds_calls, reference_solver.col_bounds_calls,
        "set_col_bounds calls must match the open-coded forward block one-for-one"
    );
    assert_eq!(
        owner_solver.row_bounds_calls, reference_solver.row_bounds_calls,
        "set_row_bounds calls must match the open-coded forward block one-for-one"
    );
    assert_eq!(
        owner_solver.col_bounds_calls.len(),
        1,
        "the minimal fixture pins exactly one state column (storage)"
    );
    assert_eq!(
        owner_solver.row_bounds_calls.len(),
        1,
        "the minimal fixture patches exactly one z-inflow row"
    );
}

/// One bus, one NCS entity, availability factor `mean=0.5, std=0.1` — the D15
/// NCS fixture shape (`lb_evaluate_stage_0_patches_ncs_bounds_per_opening` in
/// `training/lower_bound.rs`), adapted to drive [`StageSolvePrep::run`].
fn make_ncs_stochastic_context() -> (StochasticContext, Stage) {
    let ncs_entity_id = EntityId(10);
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let ncs_source = NonControllableSource {
        id: ncs_entity_id,
        name: "W1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 100.0,
        allow_curtailment: true,
        curtailment_cost: 0.0,
    };
    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };
    let ncs_model = NcsModel {
        ncs_id: ncs_entity_id,
        stage_id: 0,
        mean: 0.5,
        std: 0.1,
    };
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "ncs_group".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "ncs".to_string(),
                    id: ncs_entity_id,
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .non_controllable_sources(vec![ncs_source])
        .stages(vec![stage.clone()])
        .ncs_models(vec![ncs_model])
        .correlation(correlation)
        .build()
        .unwrap();
    let stoch = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: None,
            load: None,
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap();
    assert_eq!(stoch.n_stochastic_ncs(), 1);
    (stoch, stage)
}

/// [`StageSolvePrep::run`]'s internal NCS-patch wiring must reach the solver
/// with the same `set_col_bounds` call the pre-collapse inline pattern produces
/// (`transform_ncs_noise` → `build_dense_ncs_col_indices` → `gather_dense_ncs_bounds`
/// → `set_col_bounds`, as forward/backward/lower-bound each still write it inline).
// Rationale: the inline System/StochasticContext fixture and the reference vs.
// owner comparison are one coherent scenario; splitting them into helpers would
// scatter the setup the assertions depend on and obscure the test.
#[allow(clippy::too_many_lines)]
#[test]
fn run_wires_ncs_patch_matching_pre_collapse_inline_pattern() {
    let (stoch, ncs_stage) = make_ncs_stochastic_context();
    let state = state_layout(0, 0);
    let templates = vec![StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0_i32, 0],
        row_indices: vec![],
        values: vec![],
        col_lower: vec![0.0],
        col_upper: vec![100.0],
        objective: vec![0.0],
        row_lower: vec![],
        row_upper: vec![],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }];
    let ncs_max_gen = vec![100.0_f64];
    let ncs_allow_curtailment = vec![true];
    let ncs_stochastic_dense_col = vec![0_usize];
    let ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)> = vec![(None, None)];
    let state_boxes = vec![unbounded_state_box(state.n_state)];
    let geometry = vec![StageGeometry {
        ncs_generation: 0..1,
        n_blks: 1,
        ..crate::test_support::equipment_free_geometry(&[1]).remove(0)
    }];
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry)
        .ncs_stochastic_dense_col(&ncs_stochastic_dense_col)
        .ncs_stochastic_windows(&ncs_stochastic_windows)
        .ncs_max_gen(&ncs_max_gen)
        .ncs_allow_curtailment(&ncs_allow_curtailment);
    let ctx = fixture.ctx();
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let study_dims = StudyDimensions::default();
    let stages = vec![ncs_stage];
    let training_ctx = TrainingContext {
        node_graph: &crate::test_support::chain_node_graph(&stoch),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &all_enabled_cut_state_layouts(&state, 1),
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stoch,
        initial_state: &[],
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };

    let raw_noise = vec![0.37_f64];
    let sizing = minimal_sizing();
    let params = StageSolvePrepParams {
        state_source: StateSource(&[]),
        inflow_noise: InflowNoise::PreBuilt,
        raw_noise: &raw_noise,
    };

    // ---- reference: transform_ncs_noise -> build indices -> gather -> set,
    // called directly (the pre-collapse inline pattern), independent of
    // StageSolvePrep::run's own wiring ----
    let mut reference_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    let mut reference_solver = RecordingSolver::default();
    transform_ncs_noise(
        &raw_noise,
        &stoch,
        StageIdx(0),
        1,
        &ncs_max_gen,
        &ncs_allow_curtailment,
        &mut reference_scratch.ncs_col_lower_buf,
        &mut reference_scratch.ncs_col_upper_buf,
    );
    build_dense_ncs_col_indices(
        &ncs_stochastic_dense_col,
        &geometry[0],
        &mut reference_scratch.ncs_col_indices_buf,
    );
    gather_dense_ncs_bounds(
        &ncs_stochastic_windows,
        0,
        1,
        &reference_scratch.ncs_col_lower_buf,
        &reference_scratch.ncs_col_upper_buf,
        &mut reference_scratch.ncs_col_lower_active_buf,
        &mut reference_scratch.ncs_col_upper_active_buf,
    );
    reference_solver.set_col_bounds(
        &reference_scratch.ncs_col_indices_buf,
        &reference_scratch.ncs_col_lower_active_buf,
        &reference_scratch.ncs_col_upper_active_buf,
    );

    // ---- owner: StageSolvePrep::run's internal NCS-patch wiring ----
    let mut owner_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    let mut owner_solver = RecordingSolver::default();
    let mut owner_patch_buf = PatchBuffer::new(&state, &[], &[]);
    StageSolvePrep::run(
        &mut owner_solver,
        &mut owner_patch_buf,
        &mut owner_scratch,
        &ctx,
        &training_ctx,
        StageIdx(0),
        &params,
    );

    assert_eq!(
        owner_solver.col_bounds_calls.last(),
        reference_solver.col_bounds_calls.last(),
        "StageSolvePrep::run's NCS patch must match the pre-collapse inline pattern"
    );
    let (indices, lower, upper) = owner_solver
        .col_bounds_calls
        .last()
        .expect("the NCS patch must issue at least one set_col_bounds call");
    assert_eq!(indices, &[0_usize]);
    assert_eq!(lower, &[0.0]);
    let expected_upper = 100.0_f64 * (0.5 + 0.1 * 0.37_f64).clamp(0.0, 1.0);
    assert!((upper[0] - expected_upper).abs() < 1e-9);
}

/// Under `InflowNoise::PreBuilt` — the lower bound's own parameterization —
/// `StageSolvePrep::run` must skip `transform_inflow_noise` and read
/// `scratch.z_inflow_rhs_buf` verbatim, even with a real PAR(0) inflow model
/// `Transform` would otherwise patch. The load patch has no variation point:
/// it must still reach the load-balance row here, matching a direct
/// `transform_load_noise` + `fill_load_patches` reference call.
#[test]
fn run_reads_prebuilt_inflow_rhs_verbatim_under_prebuilt() {
    let state = state_layout(1, 0);
    let stochastic = make_stochastic_context(Some(LoadModel {
        bus_id: EntityId(0),
        stage_id: 0,
        mean_mw: 300.0,
        std_mw: 50.0,
    }));
    let templates = vec![single_hydro_load_template()];
    let state_boxes = vec![unbounded_state_box(state.n_state)];
    let geometry_per_stage = vec![geometry_with_load_balance(1, 1, 1)];
    let load_bus_indices = [0_usize];
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
        .load_bus_indices(&load_bus_indices);
    let ctx = fixture.ctx();
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let study_dims = study_dims();
    let training_ctx = TrainingContext {
        node_graph: &crate::test_support::chain_node_graph(&stochastic),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &all_enabled_cut_state_layouts(&state, 1),
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
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
    };

    let current_state = vec![42.0_f64];
    // [hydro eta (unread under PreBuilt) | load eta].
    let raw_noise = vec![0.3_f64, 0.4_f64];
    let sizing = minimal_sizing();
    let mut scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    // Sentinel pre-fill: PreBuilt must leave this untouched, since
    // transform_inflow_noise clears its target buffer before refilling it.
    scratch.z_inflow_rhs_buf = vec![222.0];

    let mut solver = RecordingSolver::default();
    let mut patch_buf = PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage);
    let params = StageSolvePrepParams {
        state_source: StateSource(&current_state),
        inflow_noise: InflowNoise::PreBuilt,
        raw_noise: &raw_noise,
    };
    StageSolvePrep::run(
        &mut solver,
        &mut patch_buf,
        &mut scratch,
        &ctx,
        &training_ctx,
        StageIdx(0),
        &params,
    );

    assert_eq!(
        scratch.z_inflow_rhs_buf,
        vec![222.0],
        "InflowNoise::PreBuilt must not recompute z_inflow_rhs_buf"
    );
    assert_eq!(
        solver.col_bounds_calls.len(),
        1,
        "the state pin still runs under PreBuilt"
    );

    // Reference: transform_load_noise -> fill_load_patches, called directly —
    // the load patch is unconditional, so it must match this exactly even
    // under PreBuilt.
    let mut reference_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    transform_load_noise(
        &raw_noise,
        &stochastic,
        StageIdx(0),
        1,
        &mut reference_scratch.load_rhs_buf,
    );
    let mut reference_patch_buf = PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage);
    let grid = BlockGrid::new(1, training_ctx.study_dims.max_deficit_segments);
    let load_rows = BlockRowFamily::per_block(1..2);
    reference_patch_buf.fill_load_patches(
        load_rows,
        grid,
        &reference_scratch.load_rhs_buf,
        &[0],
        &[],
    );

    let (indices, lower, upper) = solver
        .row_bounds_calls
        .last()
        .expect("the load and z-inflow patches must issue one set_row_bounds call");
    assert_eq!(solver.row_bounds_calls.len(), 1);
    assert_eq!(indices, &[1_usize, 0_usize]);
    assert_eq!(
        (lower[0], upper[0]),
        (reference_patch_buf.lower[0], reference_patch_buf.upper[0]),
        "the load-balance row must match the direct transform_load_noise + fill_load_patches reference"
    );
    assert_eq!(
        (lower[1], upper[1]),
        (222.0, 222.0),
        "the z-inflow row must read the pre-built sentinel verbatim, not a PAR-transformed value"
    );
}
