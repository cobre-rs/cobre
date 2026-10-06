//! Lower bound evaluation for iterative LP-based optimization algorithms.
//!
//! [`evaluate_lower_bound`] computes the risk-adjusted lower bound by solving
//! the stage-0 LP for every opening in the scenario tree and aggregating the
//! per-opening objectives through the stage-0 risk measure, then broadcasts the
//! scalar from rank 0.
//!
//! Must be called **after** the backward pass and cut sync so the FCF holds the
//! latest cuts when the LPs are solved.

use cobre_comm::Communicator;
use cobre_solver::{RowBatch, SolverError, SolverInterface};
use cobre_stochastic::{evaluate_par_batch, solve_par_noise_batch};

use crate::cut::CutRowMap;
use crate::cut::row::append_new_cuts_to_lp;
use crate::{
    backward::fill_external_opening_noise,
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    cut::row::build_cut_row_batch_into,
    error::SddpError,
    inflow_method::InflowNonNegativityMethod,
    lp::builder::PatchBuffer,
    lp::indexer::StateSpace,
    noise::{compute_effective_eta, has_par_model},
    rank_reconcile::reconcile_error_flag,
    risk_measure::{RiskMeasure, RiskMeasureScratch},
    setup::{
        NodeGraph, NodeSuccessor, OpeningSource,
        node_graph::{NodePos, StageIdx, assemble_outcome_weights},
    },
    training::stage_solve_prep::{InflowNoise, StageSolvePrep, StageSolvePrepParams, StateSource},
    workspace::ScratchBuffers,
};

/// Rank-0 risk-measure aggregation scratch, reused across iterations.
#[derive(Default)]
pub struct LbEvalScratch {
    /// Per-opening objectives.
    pub objectives_buf: Vec<f64>,
    /// Root outcome-set product weights `P(root→n)·q_{n,ω}`.
    pub weights_buf: Vec<f64>,
    /// `CVaR` weight scratch for the root aggregation.
    pub risk_scratch: RiskMeasureScratch,
}

impl LbEvalScratch {
    /// Empty buffers; no allocation until the first `evaluate_lower_bound` call.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Mutable scratch bundle for [`evaluate_lower_bound`]; build via
/// [`LbEvalScratchBundle::from_scratch_fields`].
pub struct LbEvalScratchBundle<'a> {
    /// Reusable LP row-bound patch buffer.
    pub patch_buf: &'a mut PatchBuffer,
    /// Stage-0 cut row batch for the lower-bound LP.
    pub lb_cut_batch: &'a mut RowBatch,
    /// Cut row map for append-only lower-bound LP management.
    pub lb_cut_row_map: Option<&'a mut CutRowMap>,
    /// Noise/NCS-transform scratch, shared in shape with every other solve site.
    pub noise_scratch: &'a mut ScratchBuffers,
    /// Rank-0 risk-measure aggregation scratch.
    pub lb_scratch: &'a mut LbEvalScratch,
}

impl<'a> LbEvalScratchBundle<'a> {
    /// Take the disjoint `IterationScratch` fields separately — so the borrow
    /// checker can verify non-aliasing — then bundle them; the same disjoint-borrow
    /// factory pattern as `BackwardPassInputs::from_session_fields`.
    pub fn from_scratch_fields(
        patch_buf: &'a mut PatchBuffer,
        lb_cut_batch: &'a mut RowBatch,
        lb_cut_row_map: Option<&'a mut CutRowMap>,
        noise_scratch: &'a mut ScratchBuffers,
        lb_scratch: &'a mut LbEvalScratch,
    ) -> Self {
        Self {
            patch_buf,
            lb_cut_batch,
            lb_cut_row_map,
            noise_scratch,
            lb_scratch,
        }
    }
}

/// The lower bound patches the root stage's own load rows, so the load region
/// is sized for `stage_ctx`'s root-stage (`StageIdx(0)`) block count, never the
/// global max. Bucket and anticipated capacity MUST match `n_buckets` /
/// `n_anticipated * k_max` — undersizing panics in `fill_col_state_patches`.
pub(crate) fn lower_bound_patch_buffer(
    state: &StateSpace,
    stage_ctx: &StageContext<'_>,
) -> PatchBuffer {
    PatchBuffer::new(
        state,
        stage_ctx.load_bus_indices,
        stage_ctx.geometry_per_stage.get(..1).unwrap_or_default(),
    )
}

/// Rank-0 setup: run the append-only LP load. Only called on rank 0.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if stage 0 carries no alive node.
fn lb_init_rank0<S: SolverInterface>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    lb_cut_batch: &mut RowBatch,
    lb_cut_row_map: Option<&mut CutRowMap>,
) -> Result<(), SddpError> {
    let state_layout = training_ctx.state;
    // Root pool resolved by STAGE (the sole stage-0 node), never by array
    // position: build_declared_node_graph sorts nodes by ascending id, so
    // nodes[0] is the smallest-id node, not necessarily the root — reading
    // nodes[0].pool_id would inject a sibling's cuts onto the root's stage-0 LP.
    // On a chain the root IS nodes[0] (pool 0).
    let root = training_ctx
        .node_graph
        .frontier_node(StageIdx(0))
        .ok_or_else(|| {
            SddpError::Validation("lower bound: stage 0 carries no alive node".to_string())
        })?;
    let pool_id = training_ctx.node_graph.nodes[root].pool_id;
    let cut_state = &training_ctx.cut_state_layouts[pool_id];
    let template = ctx.template(StageIdx(0));

    // Append-only: cuts are never removed, keeping the lower bound monotone across
    // iterations. The CutRowMap-less branch (tests) rebuilds the model each call.
    if let Some(row_map) = lb_cut_row_map {
        if row_map.total_cut_rows() == 0 {
            solver.load_model(template);
        }
        append_new_cuts_to_lp(
            solver,
            fcf,
            pool_id,
            state_layout,
            cut_state,
            &template.col_scale,
            row_map,
            lb_cut_batch,
        );
    } else {
        build_cut_row_batch_into(
            lb_cut_batch,
            fcf,
            pool_id,
            state_layout,
            cut_state,
            &template.col_scale,
        );
        solver.load_model(template);
        if lb_cut_batch.num_rows > 0 {
            solver.add_rows(lb_cut_batch);
        }
    }
    Ok(())
}

/// Truncation precompute then per-opening stage-0 LP solve via [`StageSolvePrep::run`],
/// writing objectives into `objectives_buf`.
///
/// # Errors
///
/// Returns [`SddpError::Infeasible`] if any opening LP is infeasible, or
/// [`SddpError::Solver`] for other LP solve failures.
fn lb_evaluate_stage_0<S: SolverInterface>(
    solver: &mut S,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    patch_buf: &mut PatchBuffer,
    scratch: &mut ScratchBuffers,
    objectives_buf: &mut Vec<f64>,
) -> Result<(), SddpError> {
    let n_hydros = training_ctx.state.hydro_count;
    let initial_state = training_ctx.initial_state;
    let opening_tree = training_ctx.stochastic.opening_tree();
    // Enumerate the ROOT NODE's own Ω, not the stage-0 generated tree: an External
    // root pins a single scenario column its external library carries, which the
    // generated tree does not know. Reading the generated tree there evaluates the
    // wrong stage-0 realization and mis-prices the bound.
    let root_pos = find_root_position(training_ctx.node_graph)?;
    let root_openings = training_ctx.node_graph.nodes[root_pos].openings;
    let root_node_id = training_ctx.node_graph.node_ids[root_pos];
    let n_openings = root_openings.len;

    let needs_truncation = matches!(
        training_ctx.inflow_method,
        InflowNonNegativityMethod::Truncation | InflowNonNegativityMethod::TruncationWithPenalty
    );
    let par_lp = training_ctx.stochastic.par();
    let has_par = has_par_model(training_ctx.stochastic);
    let truncation_par = (needs_truncation && has_par).then_some(par_lp);

    scratch.par_inflow_buf.clear();
    scratch.par_inflow_buf.resize(n_hydros, 0.0);

    if let Some(par_lp) = truncation_par {
        let max_order = training_ctx.state.max_par_order;
        scratch.lag_matrix_buf.clear();
        scratch
            .lag_matrix_buf
            .extend_from_slice(&initial_state[training_ctx.state.inflow_lags.clone()]);
        debug_assert_eq!(scratch.lag_matrix_buf.len(), max_order * n_hydros);

        scratch.eta_floor_buf.clear();
        scratch.eta_floor_buf.resize(n_hydros, f64::NEG_INFINITY);
        scratch.zero_targets_buf.clear();
        scratch.zero_targets_buf.resize(n_hydros, 0.0);
        solve_par_noise_batch(
            par_lp,
            0,
            &scratch.lag_matrix_buf,
            &scratch.zero_targets_buf,
            &mut scratch.eta_floor_buf,
        );
    }

    // An External root reads its pinned column (assembled once, `len == 1`); a
    // Generated root reads the generated tree opening-by-opening. `raw_noise_buf`
    // holds the External column across the loop without a per-iteration allocation.
    let mut ext_buf = std::mem::take(&mut scratch.raw_noise_buf);
    if root_openings.source == OpeningSource::External {
        fill_external_opening_noise(
            training_ctx,
            StageIdx(0),
            root_openings.offset,
            root_node_id,
            &mut ext_buf,
        )?;
    }

    let hydro = training_ctx.stochastic.class_dimensions().hydro_range();

    objectives_buf.clear();

    for opening_idx in 0..n_openings {
        let raw_noise: &[f64] = if root_openings.source == OpeningSource::External {
            &ext_buf
        } else {
            opening_tree.opening(0, root_openings.offset + opening_idx)
        };

        if let Some(par_lp) = truncation_par {
            evaluate_par_batch(
                par_lp,
                0,
                &scratch.lag_matrix_buf,
                &raw_noise[hydro.clone()],
                &mut scratch.par_inflow_buf,
            );
        }

        compute_effective_eta(
            &raw_noise[hydro.clone()],
            *training_ctx.inflow_method,
            &scratch.par_inflow_buf,
            &scratch.eta_floor_buf,
            &mut scratch.effective_eta_buf,
        );

        scratch.z_inflow_rhs_buf.clear();
        for h in 0..n_hydros {
            let eta_eff = scratch.effective_eta_buf[h];
            let z_rhs = if has_par {
                par_lp.deterministic_base(0, h) + par_lp.sigma(0, h) * eta_eff
            } else {
                0.0
            };
            scratch.z_inflow_rhs_buf.push(z_rhs);
        }

        let prep_params = StageSolvePrepParams {
            state_source: StateSource(initial_state),
            inflow_noise: InflowNoise::PreBuilt,
            raw_noise,
        };
        StageSolvePrep::run(
            solver,
            patch_buf,
            scratch,
            ctx,
            training_ctx,
            StageIdx(0),
            &prep_params,
        );

        let view = solver.solve(None).map_err(|e| match e {
            SolverError::Infeasible => SddpError::Infeasible {
                stage: 0,
                iteration: 0,
                scenario: opening_idx,
            },
            other => SddpError::Solver(other),
        })?;
        objectives_buf.push(view.objective);
    }

    scratch.raw_noise_buf = ext_buf;
    Ok(())
}

/// Find the node graph's implicit root: the unique node at stage 0 (root
/// implicit for a single-node stage 0). A stage 0 covered by anything
/// but exactly one node has no way to resolve `P(root→n)` across siblings
/// without the reserved root source, not yet built.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming stage 0 when it holds zero or more
/// than one node.
fn find_root_position(node_graph: &NodeGraph) -> Result<NodePos, SddpError> {
    let mut stage_0 = node_graph
        .nodes
        .iter_indexed()
        .filter(|(_, n)| n.stage == StageIdx(0))
        .map(|(pos, _)| pos);
    let root_pos = stage_0.next().ok_or_else(|| {
        SddpError::Validation("lower bound: stage 0 has no node in the node graph".to_string())
    })?;
    if stage_0.next().is_some() {
        return Err(SddpError::Validation(
            "lower bound: stage 0 holds more than one node; evaluating the root \
             requires a reserved root source for P(root→n) across siblings, not \
             yet built"
                .to_string(),
        ));
    }
    Ok(root_pos)
}

/// Assemble the root's successor outcome set `O(root) = {(n, ω) : n ∈ root⁺,
/// ω ∈ Ω_n}` into `out`. Root is implicit for a single-node stage 0
/// (`P(root→n) = 1`): the single child's own `q` — the pinned `1.0/(n as f64)`
/// bit pattern — is what this reduces to, replicated over its openings, so a
/// chain's lower bound stays bit-identical to today's.
///
/// # Errors
///
/// Returns whatever [`find_root_position`] returns.
fn assemble_root_outcome_weights(
    node_graph: &NodeGraph,
    out: &mut Vec<f64>,
) -> Result<(), SddpError> {
    let root_pos = find_root_position(node_graph)?;
    let root_successors = [NodeSuccessor {
        child: root_pos,
        probability: 1.0,
    }];
    assemble_outcome_weights(node_graph, &root_successors, out);
    Ok(())
}

/// Apply `risk_measure` to `objectives` under `weights` (the root's product
/// weights, canonical order), scale by `cost_scale_factor`, then broadcast the
/// scalar from rank 0.
///
/// # Errors
///
/// Returns [`SddpError::Communication`] if the broadcast fails.
fn lb_aggregate_and_broadcast<C: Communicator>(
    objectives: &[f64],
    weights: &[f64],
    risk_measure: &RiskMeasure,
    risk_scratch: &mut RiskMeasureScratch,
    cost_scale_factor: f64,
    comm: &C,
) -> Result<f64, SddpError> {
    // Hard length check, not a debug assert: the objectives (root Ω evaluations)
    // and the weights (root outcome set) must be the same set in the same canonical
    // order. A mismatch is a real configuration error (e.g. an External root paired
    // against the generated stage-0 tree), not a programmer slip — reject it rather
    // than silently pairing a truncated vector.
    if objectives.len() != weights.len() {
        return Err(SddpError::Validation(format!(
            "lower bound: root evaluated {} openings but the root outcome set has {} weights; \
             the root's Ω and its weights must be the same set in canonical order",
            objectives.len(),
            weights.len()
        )));
    }
    let mut lb =
        risk_measure.evaluate_risk_into(objectives, weights, risk_scratch) * cost_scale_factor;
    comm.broadcast(std::slice::from_mut(&mut lb), 0)?;
    Ok(lb)
}

/// Evaluate the global lower bound for the current FCF approximation.
///
/// # Errors
///
/// - [`SddpError::Infeasible`] — a stage-0 opening LP is infeasible (a modelling
///   error; stage 0 should always be feasible via the penalty/recourse structure).
/// - [`SddpError::Solver`] — LP solve failed for another reason.
/// - [`SddpError::Communication`] — the cross-rank reconcile or the broadcast to
///   non-root ranks failed (a peer rank-0 failure surfaces here on the healthy ranks).
///
/// # Panics
///
/// Panics if `training_ctx.stochastic.opening_tree().n_openings(0) == 0` on rank
/// 0 — stage 0 must have at least one opening (a caller contract).
pub fn evaluate_lower_bound<S: SolverInterface, C: Communicator>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    risk_measure: &RiskMeasure,
    scratch: &mut LbEvalScratchBundle<'_>,
    comm: &C,
) -> Result<f64, SddpError> {
    let mut lb = 0.0_f64;
    let mut reconcile_scratch = [0_i32; 1];

    if comm.rank() == 0 {
        assert!(
            training_ctx.stochastic.opening_tree().n_openings(0) > 0,
            "evaluate_lower_bound: stage 0 must have at least one opening"
        );

        let rank0_result = lb_init_rank0(
            solver,
            fcf,
            ctx,
            training_ctx,
            scratch.lb_cut_batch,
            scratch.lb_cut_row_map.as_deref_mut(),
        )
        .and_then(|()| {
            assemble_root_outcome_weights(
                training_ctx.node_graph,
                &mut scratch.lb_scratch.weights_buf,
            )
        })
        .and_then(|()| {
            lb_evaluate_stage_0(
                solver,
                ctx,
                training_ctx,
                scratch.patch_buf,
                scratch.noise_scratch,
                &mut scratch.lb_scratch.objectives_buf,
            )
        });
        // Reconcile rank 0's root-weight assembly and stage-0 solve across ranks
        // BEFORE the broadcast pair (mirrors the forward-phase precedent): a
        // rank-0 failure makes every rank return Err here rather than let a
        // non-root rank block at its matching broadcast while rank 0 skipped
        // `lb_aggregate_and_broadcast`.
        reconcile_error_flag(rank0_result, comm, &mut reconcile_scratch)?;

        return lb_aggregate_and_broadcast(
            &scratch.lb_scratch.objectives_buf,
            &scratch.lb_scratch.weights_buf,
            risk_measure,
            &mut scratch.lb_scratch.risk_scratch,
            ctx.cost_scale_factor,
            comm,
        );
    }

    reconcile_error_flag(Ok(()), comm, &mut reconcile_scratch)?;
    comm.broadcast(std::slice::from_mut(&mut lb), 0)?;
    Ok(lb)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::{
        LbEvalScratch, LbEvalScratchBundle, assemble_outcome_weights,
        assemble_root_outcome_weights, evaluate_lower_bound, find_root_position,
        lb_aggregate_and_broadcast, lb_evaluate_stage_0, lb_init_rank0,
    };
    use crate::{
        PrepareHydroModelsResult, ResolvedParameters, StageTemplates,
        build_stage_templates_resolving_layout,
        context::TrainingContext,
        cut::FutureCostFunction,
        error::SddpError,
        horizon_mode::HorizonMode,
        inflow_method::InflowNonNegativityMethod,
        lp::builder::{PatchBuffer, StageGeometry, StateBox},
        lp::indexer::{BlockIdx, CutStateProjection, HydroSys, StateSpace, StudyDimensions},
        risk_measure::{RiskMeasure, RiskMeasureScratch},
        setup::node_graph::StageIdx,
        setup::{
            NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        },
        test_support::{
            self, StageContextFixture, equipment_free_geometry, permissive_state_boxes,
        },
        workspace::{ScratchBuffers, WorkspaceSizing},
    };
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::SystemBuilder;
    use cobre_core::scenario::SamplingScheme;
    use cobre_solver::ActiveSolver;
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::tree::generate::OpeningTreeGenerationInputs;
    use cobre_stochastic::{
        ClassSchemes, OpeningTree, OpeningTreeInputs, PrecomputedNormal, StochasticContext,
        build_stochastic_context, generate_opening_tree,
    };

    fn empty_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    /// Return the owned locals needed to build an [`LbEvalScratchBundle`] for tests.
    fn make_lb_locals() -> (RowBatch, LbEvalScratch) {
        (empty_row_batch(), LbEvalScratch::new())
    }

    /// Minimal stage template for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout: [storage (0), `z_inflow` (1), theta (2)]
    /// Row layout: [z-inflow definition (0)]
    fn minimal_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1], // col 1 (z_inflow) has NZ at row 0
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0], // minimise theta
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Build an `OpeningTree` with `n_openings` openings at stage 0.
    ///
    /// Uses `generate_opening_tree` with a single-entity identity-correlation
    /// model. Because `MockSolver` ignores the noise values returned by
    /// `opening_tree.opening(...)`, the tree only needs to have the right shape
    /// (correct stage count and branching factor at stage 0).
    ///
    /// `dim = 1` throughout (single hydro, the N=1, L=0 state layout).
    fn simple_opening_tree(n_openings: usize) -> OpeningTree {
        use chrono::NaiveDate;
        use cobre_core::{
            EntityId,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
        use std::collections::BTreeMap;

        // Single study stage with the requested branching factor.
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let entity_id = EntityId(1);
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: entity_id,
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let entity_order = vec![entity_id];
        let decomposed = DecomposedCorrelation::build(
            &corr_model,
            &entity_order,
            cobre_stochastic::ClassDimensions {
                n_hydros: 1,
                n_load_buses: 0,
                n_ncs: 0,
            },
        )
        .unwrap();

        generate_opening_tree(
            42,
            &[stage],
            &decomposed,
            &entity_order,
            cobre_stochastic::ClassDimensions {
                n_hydros: 1,
                n_load_buses: 0,
                n_ncs: 0,
            },
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap()
    }

    /// Wrap a pre-built stage-0 [`OpeningTree`] into a degenerate
    /// [`StochasticContext`] carrying `n_hydros` bare hydro entities and no
    /// load/NCS/inflow models (`n_stochastic_ncs() == 0`,
    /// `par().n_stages() == 0`), so a test can pass a real `&StochasticContext`
    /// in place of the retired `stochastic: None` field with a hydro count that
    /// agrees with the LP-side `StateSpace::hydro_count` the caller's fixture
    /// derives independently. `user_tree` bypasses generation entirely, so the
    /// injected tree's shape is preserved verbatim.
    #[allow(clippy::cast_possible_wrap)]
    fn wrap_opening_tree(n_hydros: usize, tree: OpeningTree) -> StochasticContext {
        use cobre_core::test_support::{BusSpec, HydroSpec, make_bus, make_hydro};

        let system = SystemBuilder::new()
            .buses(vec![make_bus(BusSpec {
                id: 0,
                ..Default::default()
            })])
            .hydros(
                (0..n_hydros)
                    .map(|i| {
                        make_hydro(HydroSpec {
                            id: i as i32 + 1,
                            bus_id: 0,
                            ..Default::default()
                        })
                    })
                    .collect(),
            )
            .build()
            .expect("hydro-only system with no inflow models must be valid");
        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(tree),
                ..Default::default()
            },
            ClassSchemes {
                inflow: None,
                load: None,
                ncs: None,
            },
        )
        .expect("degenerate stochastic context must build")
    }

    // ── Mock communicator ────────────────────────────────────────────────────

    /// Single-rank stub communicator. broadcast is a no-op (identity operation).
    struct LocalComm;

    impl Communicator for LocalComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("LocalComm allgatherv not used in lower_bound tests")
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("LocalComm allreduce not used in lower_bound tests")
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            // Single rank: no-op. The value is already in buf from the rank-0 computation.
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

    /// Communicator that fails on `broadcast` with `CommError::CollectiveFailed`.
    struct FailingBcastComm;

    impl Communicator for FailingBcastComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!()
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!()
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Err(CommError::CollectiveFailed {
                operation: "broadcast",
                mpi_error_code: -1,
                message: "test-induced broadcast failure".to_string(),
            })
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

    /// Two-rank stub whose `allreduce` reduces the i32 reconcile flag with `Max`
    /// against a fixed `peer_flag`, and whose `broadcast` is unreachable — so a
    /// test that reaches the reconcile's `Err` exit proves no rank blocks at the
    /// LB broadcast.
    struct LbReconcileStub {
        rank: usize,
        peer_flag: i32,
    }

    impl Communicator for LbReconcileStub {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("evaluate_lower_bound reconcile does not call allgatherv")
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            use std::any::Any;
            assert_eq!(op, ReduceOp::Max, "reconcile must reduce with Max");
            for (s, r) in send.iter().zip(recv.iter_mut()) {
                let flag = *(s as &dyn Any)
                    .downcast_ref::<i32>()
                    .expect("reconcile stub only reduces i32 flags");
                let reduced = flag.max(self.peer_flag);
                *r = *(&reduced as &dyn Any)
                    .downcast_ref::<T>()
                    .expect("reconcile stub only reduces i32 flags");
            }
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("a reconciled failure must return Err before the LB broadcast")
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            self.rank
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    // ── Mock solver ──────────────────────────────────────────────────────────

    /// Mock solver that records `set_col_bounds` calls and returns configurable
    /// objective values in sequence.
    ///
    /// Each call to `solve()` returns the next value from `objectives`. If
    /// `infeasible_on_call` is set and the call index matches, returns
    /// `SolverError::Infeasible` instead.
    struct MockSolver {
        objectives: Vec<f64>,
        call_count: usize,
        infeasible_on_call: Option<usize>,
        /// Number of times `set_col_bounds` was called.
        set_col_bounds_calls: usize,
    }

    impl MockSolver {
        fn with_objectives(objectives: Vec<f64>) -> Self {
            Self {
                objectives,
                call_count: 0,
                infeasible_on_call: None,
                set_col_bounds_calls: 0,
            }
        }

        fn infeasible_on_first() -> Self {
            Self {
                objectives: vec![0.0],
                call_count: 0,
                infeasible_on_call: Some(0),
                set_col_bounds_calls: 0,
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _cuts: &RowBatch) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {
            self.set_col_bounds_calls += 1;
        }

        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_on_call == Some(call) {
                return Err(SolverError::Infeasible);
            }
            let obj = self.objectives[call % self.objectives.len()];
            // evaluate_lower_bound reads only `view.objective`, so the slices stay empty.
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[],
                dual: &[],
                reduced_costs: &[],
                iterations: 0,
                solve_time_seconds: 0.0,
            })
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
            "Mock"
        }
    }

    // ── Shared test setup ────────────────────────────────────────────────────

    fn make_fcf(n_stages: usize, n_state: usize) -> FutureCostFunction {
        FutureCostFunction::new(n_stages, n_state, 2, 100, &vec![0; n_stages])
    }

    /// Owned backing data for the "simple" LB unit tests' `StageContext`/
    /// `TrainingContext` pair, mirroring how `StudySetup` owns `StageData` and
    /// lends `stage_ctx()`/`training_ctx()`. Every simple test shares this shape
    /// (single 1-block stage, no load buses/NCS/anticipated thermals); only the
    /// template, hydro count, opening tree, inflow method, and initial state
    /// differ per test.
    struct SimpleLbFixture {
        templates: Vec<StageTemplate>,
        state_boxes: Vec<StateBox>,
        geometry_per_stage: Vec<StageGeometry>,
        state: StateSpace,
        cut_state_layouts: Vec<CutStateProjection>,
        study_dims: StudyDimensions,
        horizon: HorizonMode,
        stochastic: StochasticContext,
        initial_state: Vec<f64>,
        node_graph: NodeGraph,
    }

    impl SimpleLbFixture {
        fn new(
            template: StageTemplate,
            hydro_count: usize,
            opening_tree: OpeningTree,
            inflow_method: InflowNonNegativityMethod,
            initial_state: Vec<f64>,
        ) -> Self {
            let state = test_support::state_layout(hydro_count, 0);
            let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, 2);
            let stochastic = wrap_opening_tree(hydro_count, opening_tree);
            let node_graph = test_support::chain_node_graph(&stochastic);
            Self {
                state_boxes: permissive_state_boxes(state.n_state, 1),
                templates: vec![template],
                geometry_per_stage: equipment_free_geometry(&[1]),
                cut_state_layouts,
                study_dims: StudyDimensions {
                    inflow_method,
                    ..test_support::study_dims()
                },
                horizon: HorizonMode::Finite { num_stages: 2 },
                stochastic,
                initial_state,
                state,
                node_graph,
            }
        }

        fn ctx(&self) -> StageContextFixture<'_> {
            StageContextFixture::new(&self.templates, &self.state_boxes, &self.geometry_per_stage)
        }

        fn training_ctx(&self) -> TrainingContext<'_> {
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

    // ── Unit tests ───────────────────────────────────────────────────────────

    /// AC1: 1 opening, Expectation — LB equals the single LP objective.
    #[test]
    fn one_opening_expectation_lb_equals_single_objective() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(1),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let (mut row_batch, mut lb_scratch) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch,
            None,
            &mut noise_scratch,
            &mut lb_scratch,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle,
            &comm,
        )
        .unwrap();

        assert!(
            (lb - 100_000_000.0).abs() < 1e-7,
            "single opening expectation LB must equal objective 100.0 * COST_SCALE_FACTOR = 100_000_000.0, got {lb}"
        );
    }

    /// AC2: 3 openings, Expectation — LB equals mean of objectives.
    #[test]
    fn three_openings_expectation_lb_equals_mean() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(3),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![60.0, 80.0, 100.0]);

        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut noise_scratch,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_lb,
            &comm,
        )
        .unwrap();

        // E[60, 80, 100] with uniform probs = (60+80+100)/3 = 80.0; * COST_SCALE_FACTOR = 80_000_000.0
        assert!(
            (lb - 80_000_000.0).abs() < 1e-7,
            "three openings expectation LB must equal 80_000_000.0, got {lb}"
        );
    }

    /// AC3: 2 openings, CVaR(alpha=0.5, lambda=1.0) — pure `CVaR` selects worst.
    #[test]
    fn two_openings_pure_cvar_alpha_half_lb_equals_worst() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(2),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        // CVaR(alpha=0.5, lambda=1.0): pure CVaR; upper bound per scenario =
        // p / alpha = 0.5 / 0.5 = 1.0. With 2 equal-probability scenarios the
        // greedy allocation places all mass on the worst scenario.
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut noise_scratch,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_lb,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=0.5, lambda=1.0) with 2 uniform-probability openings
        // concentrates all weight on the worst (150.0); * COST_SCALE_FACTOR = 150_000_000.0.
        assert!(
            (lb - 150_000_000.0).abs() < 1e-7,
            "pure CVaR(0.5, 1.0) with 2 openings must equal 150_000_000.0, got {lb}"
        );
    }

    /// AC4 (extra): 2 openings, CVaR(alpha=1.0, lambda=1.0) = Expectation.
    #[test]
    fn two_openings_cvar_alpha_one_equals_expectation() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(2),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        };
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut noise_scratch,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_lb,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=1) = Expectation = (50+150)/2 = 100.0; * COST_SCALE_FACTOR = 100_000_000.0
        assert!(
            (lb - 100_000_000.0).abs() < 1e-7,
            "CVaR(alpha=1, lambda=1) must equal expectation 100_000_000.0, got {lb}"
        );
    }

    /// The lower-bound aggregation rejects a root-Ω/weights length mismatch with a
    /// hard `SddpError::Validation` — never a panic, never a silent truncation. The
    /// aligned `lb_evaluate_stage_0` + `assemble_root_outcome_weights` path never
    /// produces one (both size to the root node's own Ω), so this defensive check —
    /// the guard against an External root evaluated against a differently-sized
    /// weight vector — is verified here directly.
    #[test]
    fn lb_aggregate_rejects_root_omega_weight_length_mismatch() {
        let comm = LocalComm;
        let rm = RiskMeasure::Expectation;
        let objectives = vec![10.0, 20.0, 30.0];
        let weights = vec![1.0];
        let mut risk_scratch = RiskMeasureScratch::new();
        let result =
            lb_aggregate_and_broadcast(&objectives, &weights, &rm, &mut risk_scratch, 1.0, &comm);
        assert!(
            matches!(result, Err(SddpError::Validation(_))),
            "a root-Ω/weights length mismatch must be a Validation error, got {result:?}"
        );
    }

    #[test]
    fn lower_bound_aggregation_applies_the_probability_floor() {
        let comm = LocalComm;
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.5,
        };
        let objectives = vec![10.0, 20.0, 30.0, 40.0];
        let weights = vec![0.25; 4];
        let mut risk_scratch = RiskMeasureScratch::new();
        let lb =
            lb_aggregate_and_broadcast(&objectives, &weights, &rm, &mut risk_scratch, 1.0, &comm)
                .unwrap();
        assert!(
            (lb - 30.0).abs() < 1e-12,
            "the floored root aggregation must be 30.0, got {lb}"
        );
    }

    /// AC5: solver returns Infeasible for the first opening — must propagate as `SddpError::Infeasible`.
    #[test]
    fn infeasible_solve_maps_to_sddp_infeasible() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(1),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::infeasible_on_first();

        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut noise_scratch,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_result,
            &comm,
        );

        assert!(
            matches!(result, Err(SddpError::Infeasible { stage: 0, .. })),
            "infeasible solver must produce SddpError::Infeasible at stage 0, got {result:?}"
        );
    }

    /// AC6: broadcast failure maps to `SddpError::Communication`.
    #[test]
    fn broadcast_failure_maps_to_communication_error() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(1),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = FailingBcastComm;
        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut noise_scratch,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_result,
            &comm,
        );

        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "broadcast failure must produce SddpError::Communication, got {result:?}"
        );
    }

    /// Deadlock-freedom: a rank-0 stage-0 solve failure makes BOTH ranks return
    /// `Err` before the broadcast pair rather than hang (rank 0 skips
    /// `lb_aggregate_and_broadcast` while a non-root rank blocks at its matching
    /// broadcast). `LbReconcileStub::broadcast` is unreachable, so reaching it
    /// would panic instead of hanging as the real broadcast would.
    #[test]
    fn evaluate_lower_bound_rank0_failure_errors_all_ranks_before_broadcast() {
        // Rank 0: stage-0 solve is infeasible; the reconcile hands rank 0 its own
        // error before lb_aggregate_and_broadcast's broadcast.
        {
            let fixture = SimpleLbFixture::new(
                minimal_template(),
                1,
                simple_opening_tree(1),
                InflowNonNegativityMethod::None,
                vec![0.0_f64],
            );
            let fcf = make_fcf(2, fixture.state.n_state);
            let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
            let rm = RiskMeasure::Expectation;
            let comm = LbReconcileStub {
                rank: 0,
                peer_flag: 0,
            };
            let mut solver = MockSolver::infeasible_on_first();

            let (mut row_batch, mut lb_scratch) = make_lb_locals();
            let mut noise_scratch = ScratchBuffers::new(
                &fixture.training_ctx(),
                &fixture.ctx().ctx(),
                WorkspaceSizing::default(),
            );
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut noise_scratch,
                &mut lb_scratch,
            );
            let result = evaluate_lower_bound(
                &mut solver,
                &fcf,
                &fixture.ctx().ctx(),
                &fixture.training_ctx(),
                &rm,
                &mut bundle,
                &comm,
            );
            assert!(
                matches!(result, Err(SddpError::Infeasible { stage: 0, .. })),
                "rank 0 must return its own Infeasible before the broadcast, got {result:?}"
            );
        }

        // Rank 1 (non-root): its peer (rank 0) failed; the reconcile returns a
        // peer-failure Communication error before rank 1 blocks at its broadcast.
        {
            let fixture = SimpleLbFixture::new(
                minimal_template(),
                1,
                simple_opening_tree(1),
                InflowNonNegativityMethod::None,
                vec![0.0_f64],
            );
            let fcf = make_fcf(2, fixture.state.n_state);
            let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
            let rm = RiskMeasure::Expectation;
            let comm = LbReconcileStub {
                rank: 1,
                peer_flag: 1,
            };
            let mut solver = MockSolver::with_objectives(vec![100.0]);

            let (mut row_batch, mut lb_scratch) = make_lb_locals();
            let mut noise_scratch = ScratchBuffers::new(
                &fixture.training_ctx(),
                &fixture.ctx().ctx(),
                WorkspaceSizing::default(),
            );
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut noise_scratch,
                &mut lb_scratch,
            );
            let result = evaluate_lower_bound(
                &mut solver,
                &fcf,
                &fixture.ctx().ctx(),
                &fixture.training_ctx(),
                &rm,
                &mut bundle,
                &comm,
            );
            assert!(
                matches!(result, Err(SddpError::Communication(_))),
                "non-root rank must return a peer-failure Communication error before its broadcast, got {result:?}"
            );
        }
    }

    // ── Integration tests ────────────────────────────────────────────────────

    /// Integration: full round-trip with `LocalComm` and 2 openings.
    ///
    /// Verifies that the function correctly integrates with `build_cut_row_batch`
    /// (`cut_batch` with 0 cuts still produces the right result), `fill_z_inflow_patches`,
    /// and `RiskMeasure::Expectation`.
    #[test]
    fn integration_two_openings_local_backend_expectation() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(2),
            InflowNonNegativityMethod::None,
            vec![50.0_f64], // non-zero initial state
        );
        // Start with 0 cuts (empty FCF).
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![200.0, 300.0]);

        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut noise_scratch,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_lb,
            &comm,
        )
        .unwrap();

        // E[200, 300] = 250.0; * COST_SCALE_FACTOR = 250_000_000.0
        assert!(
            (lb - 250_000_000.0).abs() < 1e-7,
            "integration round-trip must produce 250_000_000.0, got {lb}"
        );
    }

    /// Integration: monotonicity — adding cuts can only increase the LB.
    ///
    /// This test calls `evaluate_lower_bound` twice: first with 0 cuts, then
    /// with objectives set higher (simulating tighter cuts). The second LB
    /// must be >= the first.
    #[test]
    fn integration_monotonicity_more_cuts_yields_higher_or_equal_lb() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(2),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        // First call: solver returns [50, 100] → LB = 75.
        let mut solver1 = MockSolver::with_objectives(vec![50.0, 100.0]);
        let (mut row_batch_lb1, mut lb_scratch_lb1) = make_lb_locals();
        let mut noise_scratch1 = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_lb1 = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb1,
            None,
            &mut noise_scratch1,
            &mut lb_scratch_lb1,
        );
        let lb1 = evaluate_lower_bound(
            &mut solver1,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_lb1,
            &comm,
        )
        .unwrap();

        // Second call: solver returns [80, 120] → LB = 100 (tighter cuts raise obj).
        let mut solver2 = MockSolver::with_objectives(vec![80.0, 120.0]);
        let (mut row_batch_lb2, mut lb_scratch_lb2) = make_lb_locals();
        let mut noise_scratch2 = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_lb2 = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb2,
            None,
            &mut noise_scratch2,
            &mut lb_scratch_lb2,
        );
        let lb2 = evaluate_lower_bound(
            &mut solver2,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_lb2,
            &comm,
        )
        .unwrap();

        assert!(
            lb2 >= lb1,
            "second LB ({lb2}) must be >= first LB ({lb1}) when cuts are tighter"
        );
    }

    // ── Inflow truncation tests ─────────────────────────────────────────────

    /// `None` method passes raw noise through unchanged (regression test).
    ///
    /// With the degenerate wrapped stochastic context, the truncation path is a
    /// no-op since `has_par_model` reports no PAR model. This validates that
    /// the `compute_effective_eta` control flow works correctly when no PAR
    /// model applies.
    #[test]
    fn test_lb_none_method_unchanged() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(2),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![60.0, 80.0]);

        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut noise_scratch,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_lb,
            &comm,
        )
        .unwrap();

        // E[60, 80] = 70.0; * COST_SCALE_FACTOR = 70_000_000.0
        assert!(
            (lb - 70_000_000.0).abs() < 1e-7,
            "None method must produce correct LB, got {lb}"
        );
    }

    /// `Truncation` method does not cause a crash or infeasibility.
    ///
    /// With the degenerate wrapped stochastic context, the truncation path is a
    /// no-op since `has_par_model` reports no PAR model, but this validates that
    /// the control flow (`needs_truncation` = true, `truncation_par` = `None`)
    /// does not panic.
    #[test]
    fn test_lb_truncation_no_crash() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(1),
            InflowNonNegativityMethod::Truncation,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut noise_scratch,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_result,
            &comm,
        );

        assert!(
            result.is_ok(),
            "Truncation method must not panic or fail, got {result:?}"
        );
    }

    /// `TruncationWithPenalty` method does not cause a crash or infeasibility.
    #[test]
    fn test_lb_truncation_with_penalty_no_crash() {
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(1),
            InflowNonNegativityMethod::TruncationWithPenalty,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut noise_scratch,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &fixture.ctx().ctx(),
            &fixture.training_ctx(),
            &rm,
            &mut bundle_result,
            &comm,
        );

        assert!(
            result.is_ok(),
            "TruncationWithPenalty method must not panic or fail, got {result:?}"
        );
    }

    // ── NCS column-bound patching regression test ────────────────────────────

    /// Correctness guard for the D15 NCS column-bound patch contract: NCS column
    /// bounds must be patched *per opening*, not once before the loop or not at
    /// all, so the solver receives one `set_col_bounds` call per opening.
    // `clippy::too_many_lines`: the inline `System`/`StochasticContext` fixture and
    // its per-opening assertions are one coherent scenario; splitting them into
    // helpers would scatter the setup the assertions depend on and obscure the test.
    // `clippy::similar_names`: the role-(a) `state` handle reads next to `stage`-
    // named locals; both are established names.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    #[test]
    fn lb_evaluate_stage_0_patches_ncs_bounds_per_opening() {
        use cobre_core::{
            Bus, DeficitSegment, EntityId, SystemBuilder,
            entities::non_controllable::NonControllableSource,
            scenario::{
                CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, NcsModel,
            },
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };
        use std::collections::BTreeMap;

        let n_openings = 3_usize;
        let n_ncs = 1_usize;
        let block_count = 1_usize;
        let ncs_entity_id = EntityId(10);

        // Build a minimal System with one bus and one NCS entity.
        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        let ncs_source = NonControllableSource {
            id: ncs_entity_id,
            name: "W1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        // NCS model: mean=0.5, std=0.1 availability factor.
        let ncs_model = NcsModel {
            ncs_id: ncs_entity_id,
            stage_id: 0,
            mean: 0.5,
            std: 0.1,
        };

        // Correlation: single NCS entity, identity correlation.
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

        assert_eq!(
            stoch.n_stochastic_ncs(),
            n_ncs,
            "StochasticContext must report {n_ncs} stochastic NCS entity"
        );

        // Build a template with 1 NCS generation column (col index 0).
        // The NCS generation column range is 0..block_count (= 0..1).
        let template = StageTemplate {
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
        };
        let templates = vec![template];

        let state = test_support::state_layout(0, 0);
        let ncs_max_gen = vec![100.0_f64; n_ncs];
        let ncs_allow_curtailment = vec![true; n_ncs];
        // Dense: every stochastic slot maps to its own NCS column (slot order ==
        // system order here), and every slot is windowless (active at every stage),
        // so none is commissioning-dormant at stage 0.
        let ncs_stochastic_dense_col: Vec<usize> = (0..n_ncs).collect();
        let ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)> = vec![(None, None); n_ncs];
        let state_boxes = permissive_state_boxes(state.n_state, 1);

        let geometry = vec![StageGeometry {
            ncs_generation: 0..n_ncs * block_count,
            n_blks: block_count,
            ..test_support::equipment_free_geometry(&[block_count]).remove(0)
        }];
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry)
            .ncs_stochastic_dense_col(&ncs_stochastic_dense_col)
            .ncs_stochastic_windows(&ncs_stochastic_windows)
            .ncs_max_gen(&ncs_max_gen)
            .ncs_allow_curtailment(&ncs_allow_curtailment);
        let ctx = fixture.ctx();

        let horizon = HorizonMode::Finite { num_stages: 1 };
        let study_dims = StudyDimensions::default();
        let stages = vec![stage];
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, 1);
        let initial_state: Vec<f64> = Vec::new();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stoch),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &cut_state_layouts,
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stoch,
            initial_state: &initial_state,
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

        let mut patch_buf = PatchBuffer::new(&state, ctx.load_bus_indices, ctx.geometry_per_stage);
        let mut scratch = ScratchBuffers::new(&training_ctx, &ctx, WorkspaceSizing::default());
        let mut objectives_buf = Vec::new();
        let actual_n_openings = stoch.opening_tree().n_openings(0);
        let mut solver =
            MockSolver::with_objectives(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);

        lb_evaluate_stage_0(
            &mut solver,
            &ctx,
            &training_ctx,
            &mut patch_buf,
            &mut scratch,
            &mut objectives_buf,
        )
        .unwrap();

        assert_eq!(
            solver.set_col_bounds_calls,
            2 * actual_n_openings,
            "set_col_bounds must be called twice per opening ({actual_n_openings} openings), \
             got {} calls — NCS bounds are not being patched per opening",
            solver.set_col_bounds_calls
        );
        assert!(
            actual_n_openings > 0,
            "opening tree must have at least one opening at stage 0"
        );
    }

    // ── Scratch reuse regression test ────────────────────────────────────────

    /// Verify that `ScratchBuffers` noise buffers are reused across consecutive
    /// [`evaluate_lower_bound`] calls.
    ///
    /// Calls `evaluate_lower_bound` twice on the same `noise_scratch` and
    /// verifies that `z_inflow_rhs_buf.capacity()` does not decrease on the
    /// second call (i.e., no reallocation occurred). This guards against
    /// regressions that would re-introduce per-iteration heap allocation on
    /// the lower-bound hot path.
    #[test]
    fn lb_eval_scratch_reuses_buffers_across_calls() {
        // Use n_hydros = 1 so that z_inflow_rhs_buf gets populated (capacity
        // grows to 1 after the first call).
        let fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(1),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let fcf = make_fcf(2, fixture.state.n_state);
        let mut patch_buf = PatchBuffer::new(&fixture.state, &[], &[]);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut row_batch = empty_row_batch();
        let mut lb_scratch = LbEvalScratch::new();
        let mut noise_scratch = ScratchBuffers::new(
            &fixture.training_ctx(),
            &fixture.ctx().ctx(),
            WorkspaceSizing::default(),
        );

        let mut solver1 = MockSolver::with_objectives(vec![10.0]);
        {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut noise_scratch,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver1,
                &fcf,
                &fixture.ctx().ctx(),
                &fixture.training_ctx(),
                &rm,
                &mut bundle,
                &comm,
            )
            .unwrap();
        }

        let cap_after_first = noise_scratch.z_inflow_rhs_buf.capacity();
        assert!(
            cap_after_first > 0,
            "z_inflow_rhs_buf must have nonzero capacity after first call (n_hydros = 1)"
        );

        let mut solver2 = MockSolver::with_objectives(vec![20.0]);
        {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut noise_scratch,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver2,
                &fcf,
                &fixture.ctx().ctx(),
                &fixture.training_ctx(),
                &rm,
                &mut bundle,
                &comm,
            )
            .unwrap();
        }

        let cap_after_second = noise_scratch.z_inflow_rhs_buf.capacity();
        assert_eq!(
            cap_after_second, cap_after_first,
            "z_inflow_rhs_buf capacity must be stable across calls (first={cap_after_first}, second={cap_after_second}); \
             a decrease indicates reallocation on the lower-bound hot path"
        );
    }

    // ── Filling phase-gating inheritance (template-driven, no per-opening patch) ─
    //
    // Filling gating is stage-deterministic (a function of `stage.id` + `FillingConfig`),
    // so it lives entirely in the per-stage `StageTemplate` the lower bound already
    // loads — it inherits filling structure by construction, with no per-opening
    // patch (unlike NCS, whose per-opening stochastic draw forces a re-patch).
    // Hand-wiring a filling patch here would duplicate template structure into the
    // hot path; the source-text guard below fails that edit.

    /// Build the per-stage templates for a study whose hydros exercise filling at
    /// stage 0, via the SAME `build_stage_templates` (`geometry_per_stage`) path
    /// the training loop uses.
    ///
    /// Two filling hydros, both white-noise (`max_par_order = 0`), independent (no
    /// cascade), on a single bus:
    /// - `H_A` (id 3): `start_stage_id = 0`, `entry_stage_id = 1`. At stage 0 it is
    ///   in the terminal `Filling` stage (`entry − 1 == 0`), so stage 0 carries the
    ///   `filling_target`/`σ_fill` row+column family. Its own z-inflow coupling is
    ///   NOT routed away (Filling keeps PAR noise).
    /// - `H_B` (id 4): `start_stage_id = 2`, `entry_stage_id = 4`. At stage 0
    ///   (`id 0 < start 2`) it is `PreFilling`, so its z-inflow column carries no
    ///   entry on any water row (a sink: neither hydro declares a downstream).
    ///
    /// Returns the built [`StageTemplates`] plus the system indices of the two
    /// hydros (id-sorted, so `H_A`→0, `H_B`→1).
    // Rationale: this is a verbose `System`-builder fixture (bounds, penalties,
    // inflow models, two hydros over five stages); the nested default-bounds /
    // default-penalties `fn` helpers keep the per-field defaults local to the
    // fixture. Splitting it or hoisting the helpers out would scatter the fixture
    // without making it clearer — the same allow-set the sibling `minimal_system`
    // fixtures in `setup` carry.
    #[allow(
        clippy::too_many_lines,
        clippy::items_after_statements,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn filling_study_templates() -> (StageTemplates, usize, usize) {
        use chrono::NaiveDate;
        use cobre_core::scenario::InflowModel;
        use cobre_core::{
            Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties,
            ContractBlockBounds, DeficitSegment, EntityId, FillingConfig, Hydro, HydroBlockBounds,
            HydroGenerationModel, HydroPenalties, HydroStageBounds, LineBlockBounds,
            LineStagePenalties, NcsStagePenalties, NoiseMethod, PenaltiesCountsSpec,
            PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties,
            ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, SystemBuilder,
            ThermalBlockBounds, ThermalStageBounds,
        };
        use cobre_stochastic::par::precompute::PrecomputedPar;

        let n_hydros = 2_usize;
        // Five study stages (ids 0..=4) span every filling phase of both hydros.
        let n_stages = 5_usize;

        fn zero_hydro_penalties() -> HydroPenalties {
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

        // A filling hydro with the given start/entry stage ids. ConstantProductivity
        // keeps the LP simple (no FPHA rows); the soft-floor/target slacks come from
        // the filling family, not the generation model.
        let filling_hydro = |id: i32, start_stage_id: i32, entry: i32| {
            let mut hydro = Hydro {
                unit_groups: Vec::new(),
                id: EntityId(id),
                name: format!("H{id}"),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                downstream_id: None,
                travel_time_hours: None,
                entry_stage_id: Some(entry),
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: Some(FillingConfig {
                    start_stage_id,
                    filling_min_rate_m3s: 0.0,
                }),
                penalties: zero_hydro_penalties(),
            };
            hydro.declare_mirror_unit_group(EntityId(1));
            hydro
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydros = vec![filling_hydro(3, 0, 1), filling_hydro(4, 2, 4)];

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
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
            })
            .collect();

        // White-noise inflow models (non-zero std so the Operating/Filling
        // z-inflow coupling is real where the PreFilling routing is the contrast).
        let inflow_models: Vec<InflowModel> = [EntityId(3), EntityId(4)]
            .into_iter()
            .flat_map(|hid| {
                (0..n_stages).map(move |s| InflowModel {
                    hydro_id: hid,
                    stage_id: s as i32,
                    mean_m3s: 80.0,
                    std_m3s: 20.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                })
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            }
        }

        fn default_hydro_penalties() -> HydroPenalties {
            HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
                filling_target_violation_cost: 100.0,
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

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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
                n_hydros,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("filling_study_templates: valid system");

        // Every fixture stage has id >= 0, so the full slices match the study-stage
        // filter `build_stage_templates` applies internally (it filters id >= 0 too).
        let par_lp = PrecomputedPar::build(
            system.inflow_models(),
            system.stages(),
            &[EntityId(3), EntityId(4)],
            None,
        )
        .expect("white-noise PrecomputedPar build");
        let normal_lp = PrecomputedNormal::default();
        let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
        let resolved_params = ResolvedParameters {
            per_param: vec![],
            id_to_slot: vec![],
            cost_scale_factor: 1_000_000.0,
        };

        let templates = build_stage_templates_resolving_layout(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &normal_lp,
            &hydro_models.production,
            &hydro_models.evaporation,
            &resolved_params,
        )
        .expect("build_stage_templates: filling system");

        // Id-sorted: H_A (id 3) → 0, H_B (id 4) → 1.
        (templates, 0, 1)
    }

    /// Build a stage-0 [`OpeningTree`] for the two filling hydros (ids 3, 4) with
    /// `n_openings` openings, matching the inflow noise dimension the
    /// [`filling_study_templates`] system carries (`dim = 2`).
    ///
    /// The per-opening noise vector must have at least `n_hydros` entries because
    /// `lb_evaluate_stage_0` slices the noise vector's hydro segment; a 1-hydro
    /// tree (the sibling `simple_opening_tree`) would under-size it. Identity correlation
    /// between the two inflow entities keeps the tree shape trivial.
    fn filling_opening_tree(n_openings: usize) -> OpeningTree {
        use chrono::NaiveDate;
        use cobre_core::{
            EntityId,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
        use std::collections::BTreeMap;

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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let entity_order = vec![EntityId(3), EntityId(4)];
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g_inflow".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(3),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(4),
                        },
                    ],
                    matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let decomposed = DecomposedCorrelation::build(
            &corr_model,
            &entity_order,
            cobre_stochastic::ClassDimensions {
                n_hydros: 2,
                n_load_buses: 0,
                n_ncs: 0,
            },
        )
        .unwrap();

        generate_opening_tree(
            42,
            &[stage],
            &decomposed,
            &entity_order,
            cobre_stochastic::ClassDimensions {
                n_hydros: 2,
                n_load_buses: 0,
                n_ncs: 0,
            },
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap()
    }

    /// AC1: the lower-bound `StageTemplate` (= forward/backward `templates[0]`)
    /// carries the filling row families present in the production template — the
    /// per-stage `filling_target`/`σ_fill` family at EVERY Filling stage (the
    /// per-stage widening, not only the terminal `entry − 1` stage) and the renamed
    /// `filled_min_storage_floor`/`σ^{v-}` operating-floor family — and the
    /// `PreFilling` hydro's own z-inflow column carries no water-row entry.
    ///
    /// The lower bound's `StageContext.templates[0]` is bound to
    /// `stage_ctx.templates[0]` — the SAME object the forward/backward passes
    /// load. So inspecting the `build_stage_templates` output IS inspecting what
    /// the lower bound consumes: there is no separate lower-bound template build
    /// to diverge.
    #[test]
    fn lower_bound_template_matches_forward_for_filling_stage() {
        let (templates, h_a, h_b) = filling_study_templates();
        let water_balance = templates.geometry_per_stage[0].water_balance;
        let n_hydros = water_balance.range().len()
            / water_balance.rows_per_entity(templates.geometry_per_stage[0].n_blks);
        assert_eq!(n_hydros, 2, "fixture has two filling hydros");

        // Stage 0 is H_A's terminal Filling stage: the per-stage geometry the
        // template was frozen with must carry the filling_target (σ_fill) row family
        // plus the σ_fill slack column.
        let geom0 = &templates.geometry_per_stage[0];
        assert!(
            !geom0.filling_target.is_empty(),
            "stage 0 (H_A entry − 1) must carry a filling_target/σ_fill row, got {:?}",
            geom0.filling_target
        );
        assert!(
            !geom0.filling_target_col.is_empty(),
            "stage 0 must carry the σ_fill slack column, got {:?}",
            geom0.filling_target_col
        );

        // The filling families must lie INSIDE the structural region of the same
        // `templates[0]` the lower bound loads — i.e. they are real rows/columns of
        // the LP `evaluate_lower_bound` solves, not phantom geometry. (A row at
        // `>= num_rows` would alias a cut row.)
        let tpl0 = &templates.templates[0];
        assert!(
            geom0.filling_target.end <= tpl0.num_rows,
            "filling rows must lie within templates[0].num_rows ({}): target {:?}",
            tpl0.num_rows,
            geom0.filling_target
        );
        assert!(
            geom0.filling_target_col.end <= tpl0.num_cols,
            "σ_fill column must lie within templates[0].num_cols ({}): {:?}",
            tpl0.num_cols,
            geom0.filling_target_col
        );

        // Per-stage widening: the `filling_target`/`σ_fill` family fires at EVERY
        // Filling stage (`start ≤ id < entry`), not only the terminal `entry − 1`
        // one. H_A (start 0, entry 1) is Filling at id 0 only; H_B (start 2, entry 4)
        // is Filling at ids 2 AND 3. A pre-widening build that emitted the family only
        // at the terminal stage would leave stages 2/3 empty — the regression pinned.
        let filling_stage_ids = [0_usize, 2, 3];
        for &fs in &filling_stage_ids {
            let geom = &templates.geometry_per_stage[fs];
            assert!(
                !geom.filling_target.is_empty(),
                "stage {fs} is a Filling stage and must carry a filling_target/σ_fill \
                 row family (per-stage widening), got {:?}",
                geom.filling_target
            );
            assert!(
                !geom.filling_target_col.is_empty(),
                "stage {fs} is a Filling stage and must carry the σ_fill slack column \
                 (per-stage widening), got {:?}",
                geom.filling_target_col
            );
            assert_eq!(
                geom.filling_target_hydro_indices.len(),
                geom.filling_target_col.len(),
                "stage {fs}: the sparse filling_target hydro-index list must be parallel \
                 to the σ_fill column range"
            );
        }
        // Contrast: the non-filling stages carry NO filling_target family, so the
        // assertion above is a real per-stage signal, not a tautology that would also
        // pass for an always-on family. Stage 1: H_A Operating (id 1 ≥ entry 1),
        // H_B PreFilling (id 1 < start 2). Stage 4: H_B Operating (id 4 ≥ entry 4).
        for &ns in &[1_usize, 4] {
            let geom = &templates.geometry_per_stage[ns];
            assert!(
                geom.filling_target.is_empty(),
                "stage {ns} is NOT a Filling stage for either hydro, so it must carry \
                 no filling_target/σ_fill row, got {:?}",
                geom.filling_target
            );
        }

        // A later Operating stage of H_B (id 4 == entry) must carry the
        // filled_min_storage_floor/σ^{v-} family — proving the third filling row family is
        // template-driven too (it just does not fire at stage 0 for these hydros).
        let geom4 = &templates.geometry_per_stage[4];
        assert!(
            !geom4.filled_min_storage_floor.is_empty(),
            "stage 4 (H_B Operating) must carry a filled_min_storage_floor/σ^{{v-}} row, got {:?}",
            geom4.filled_min_storage_floor
        );

        // PreFilling z-inflow routing: H_B is PreFilling at stage 0 (id 0 <
        // start 2), so its own z-inflow column carries no entry on any water
        // row — the frozen-storage-identity freeze (the PreFilling row-pinning
        // contract) the lower bound inherits via the loaded template. Neither
        // hydro declares a downstream, so H_B's coupling routes nowhere (a
        // sink), never onto H_A's row. H_A is Filling at stage 0 (not
        // PreFilling), so its OWN z-inflow column carries `-zeta` on its own
        // water row — the contrast that makes the routing check non-vacuous.
        let state = test_support::state_layout(n_hydros, 0);
        let raw_at = |col: usize, row: usize| -> f64 {
            let start = usize::try_from(tpl0.col_starts[col]).unwrap();
            let end = usize::try_from(tpl0.col_starts[col + 1]).unwrap();
            tpl0.row_indices[start..end]
                .iter()
                .zip(&tpl0.values[start..end])
                .filter(|&(&r, _)| usize::try_from(r).unwrap() == row)
                .map(|(_, &v)| v)
                .sum()
        };
        let z_col_a = state.z_inflow.start + h_a;
        let z_col_b = state.z_inflow.start + h_b;
        let water_row_a = geom0.water_balance_row(HydroSys::new(h_a), BlockIdx::new(0));
        assert_eq!(
            raw_at(z_col_a, water_row_a),
            -templates.block_hours_per_stage[0].iter().sum::<f64>()
                * crate::block_clock::M3S_TO_HM3,
            "Filling hydro H_A's own z-inflow column must carry -zeta on its own water row"
        );
        for row in geom0.water_balance.range() {
            assert_eq!(
                raw_at(z_col_b, row),
                0.0,
                "PreFilling hydro H_B's z-inflow column must carry no entry on water row {row}"
            );
        }
    }

    /// AC2: the lower-bound evaluation code references NO filling-specific symbol —
    /// filling structure arrives ONLY via the loaded template, never via a
    /// hand-written per-opening patch.
    ///
    /// Modeled on the `builder_never_references_dual_extraction` guard: a
    /// future "simplification" that hand-wires a filling patch into `lb_init_rank0`
    /// / `lb_evaluate_stage_0` (mirroring the NCS per-opening patch, which filling
    /// does NOT need because it is stage-deterministic) would re-introduce the
    /// filling symbols here and fail. The needles are assembled from char fragments
    /// so this guard's own source text does not contain the literals (else the test
    /// would flag itself), exactly as that guard's source-text scan does — the
    /// filling families are template-driven, not gated by hand-written code here.
    #[test]
    fn lower_bound_never_references_filling_gating() {
        // Filling-gating symbols assembled from fragments so they are absent from
        // this file's own bytes (the source text scanned below is this very file).
        // The needle set tracks the live filling row families: the per-stage
        // `σ_fill`/`filling_target` family and the renamed `filled_min_storage_floor`
        // (`σ^{v-}`) operating-floor family. There is no `filling_retention` needle:
        // the retention family was removed (the Filling phase keeps PAR noise via
        // its own z-inflow coupling, not a retention row), so referencing it would
        // itself be a stale symbol — the rename/removal is mirrored here so the
        // guard cannot rot back to the abandoned family name.
        let needles: [String; 4] = [
            ["filling", "_phase"].concat(),
            ["Phase", "::"].concat(),
            ["sigma", "_fill"].concat(),
            ["filled", "_min_storage_floor"].concat(),
        ];

        // The full lower-bound module source. Only the PRODUCTION region (above the
        // `#[cfg(test)] mod tests` line) is the code under test; the test module
        // legitimately names filling symbols (this guard, the fixture, the AC1/AC3
        // assertions), so scanning the whole file would flag the tests themselves.
        // Split on the test module attribute and scan only the production prefix.
        let src = include_str!("lower_bound.rs");
        let prod_src = src
            .split("#[cfg(test)]")
            .next()
            .expect("module has a production region before the test module");

        let mut offenders: Vec<&str> = Vec::new();
        for needle in &needles {
            if prod_src.contains(needle.as_str()) {
                offenders.push(needle.as_str());
            }
        }
        assert!(
            offenders.is_empty(),
            "lower-bound production code must reference NO filling-gating symbol (filling \
             structure arrives only via the loaded template, never a \
             hand-written per-opening patch); offending symbols: {offenders:?}"
        );
    }

    /// AC3: `evaluate_lower_bound` returns a finite, valid bound for a filling-hydro
    /// system whose stage-0 LP carries the filling slack columns (`σ_fill`/`σ^{v-}`).
    ///
    /// Runs against the REAL solver (`ActiveSolver`) loading the production
    /// `templates[0]`, so the solve genuinely sees the filling-structured LP — a
    /// `MockSolver` would ignore the template and prove nothing about the LP shape.
    /// The assertion is structural/qualitative (finite, valid minorant; filling
    /// slack columns present); no numeric bound is pinned, because the HiGHS/CLP
    /// floating-point result is not reproducible across backends/hosts.
    #[test]
    fn evaluate_lower_bound_uses_filling_structured_template() {
        let (templates, _h_a, _h_b) = filling_study_templates();

        // Precondition: stage 0 carries the σ_fill slack column block, so the LP the
        // lower bound solves is the filling-structured LP (not an un-gated one).
        let geom0 = &templates.geometry_per_stage[0];
        assert!(
            !geom0.filling_target_col.is_empty(),
            "stage 0 must carry the σ_fill slack column for this test to be meaningful"
        );
        let tpl0 = &templates.templates[0];
        assert!(
            geom0.filling_target_col.end <= tpl0.num_cols,
            "σ_fill slack column must be a real column of templates[0]"
        );

        // The per-opening z-inflow patch reads the loaded template's own z-inflow
        // coupling — H_B's PreFilling row carries none — exactly as the
        // forward/backward passes do, so the bound sees the same stage-0
        // constraints.
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let comm = LocalComm;

        // 2 storage states (one per hydro), white noise ⇒ max_par_order = 0.
        let state = test_support::state_layout(2, 0);
        let fcf = make_fcf(templates.templates.len(), state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(&state, &[], &[]);
        let opening_tree = filling_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let stochastic = wrap_opening_tree(state.hydro_count, opening_tree);
        let state_boxes = permissive_state_boxes(state.n_state, templates.templates.len());

        let fixture = StageContextFixture::new(
            &templates.templates,
            &state_boxes,
            &templates.geometry_per_stage,
        );
        let ctx = fixture.ctx();
        let horizon = HorizonMode::Finite {
            num_stages: templates.templates.len(),
        };
        let study_dims = test_support::study_dims();
        let cut_state_layouts =
            test_support::all_enabled_cut_state_layouts(&state, templates.templates.len());
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &cut_state_layouts,
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
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
        let (mut row_batch, mut lb_scratch) = make_lb_locals();
        let mut noise_scratch =
            ScratchBuffers::new(&training_ctx, &ctx, WorkspaceSizing::default());
        let mut bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch,
            None,
            &mut noise_scratch,
            &mut lb_scratch,
        );

        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &ctx,
            &training_ctx,
            &rm,
            &mut bundle,
            &comm,
        );

        let lb = lb.expect("filling-structured stage-0 LP must be feasible (penalty recourse)");
        assert!(
            lb.is_finite(),
            "lower bound over the filling-structured LP must be finite, got {lb}"
        );
    }

    /// Regression: the LB-eval consumer path (`evaluate_lower_bound` →
    /// `lb_evaluate_stage_0`) inherits the `PatchBuffer` single-owner fix —
    /// every travel-time bucket incoming column is pinned to `initial_state`, a
    /// value constant across the opening loop (not re-derived per opening), so a
    /// single-opening stage-0 evaluation already exercises the per-stage-visit
    /// pinning contract.
    #[test]
    fn evaluate_lower_bound_pins_transit_bucket_incoming_columns() {
        let state = test_support::state_layout_with_transit_buckets(
            0,
            0,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![],
        );
        assert_eq!(state.n_state, 2);

        let template = test_support::transit_bucket_only_template(state.theta + 1, state.n_state);
        let templates = vec![template];
        let fcf = make_fcf(1, state.n_state);
        let initial_state = vec![7.0_f64, 11.0];
        let mut patch_buf = PatchBuffer::new(&state, &[], &[]);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![0.0]);
        let stochastic = wrap_opening_tree(state.hydro_count, opening_tree);
        let state_boxes = permissive_state_boxes(state.n_state, 1);

        let geometry = equipment_free_geometry(&[0]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, 1);
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &cut_state_layouts,
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
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
        let (mut row_batch, mut lb_scratch) = make_lb_locals();
        let mut noise_scratch =
            ScratchBuffers::new(&training_ctx, &ctx, WorkspaceSizing::default());
        let mut bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch,
            None,
            &mut noise_scratch,
            &mut lb_scratch,
        );

        evaluate_lower_bound(
            &mut solver,
            &fcf,
            &ctx,
            &training_ctx,
            &rm,
            &mut bundle,
            &comm,
        )
        .unwrap();

        let cp = bundle.patch_buf.state_col_patch_count();
        assert_eq!(
            cp, 2,
            "state_col_patch_count must equal n_buckets when N=0, A=0"
        );
        for (i, &expected) in initial_state.iter().enumerate() {
            let col = state.transit_buckets_in.start + i;
            let pos = bundle.patch_buf.col_indices[..cp]
                .iter()
                .position(|&c| c == col)
                .unwrap_or_else(|| panic!("bucket incoming column {col} must be pinned"));
            assert_eq!(bundle.patch_buf.col_lower[pos], expected);
            assert_eq!(bundle.patch_buf.col_upper[pos], expected);
        }
    }

    // ── Root outcome-set assembly ────────────────────────────────────────────

    /// Hand-built 3-node graph: root (position 0, stage 0) with two children —
    /// child A (position 1, 2 openings, `P(root→A) = 0.25`) and child B
    /// (position 2, 4 openings, `P(root→B) = 0.75`) — every factor a dyadic
    /// fraction, so every product weight below is exactly representable.
    fn k_fan_root_node_graph() -> NodeGraph {
        let child_a = NodeRuntime {
            stage: StageIdx(1),
            pool_id: 0,
            openings: NodeOpenings {
                source: OpeningSource::Generated,
                offset: 0,
                len: 2,
                q: 0.5,
            },
        };
        let child_b = NodeRuntime {
            stage: StageIdx(1),
            pool_id: 1,
            openings: NodeOpenings {
                source: OpeningSource::Generated,
                offset: 0,
                len: 4,
                q: 0.25,
            },
        };
        let root = NodeRuntime {
            stage: StageIdx(0),
            pool_id: 2,
            openings: NodeOpenings {
                source: OpeningSource::Generated,
                offset: 0,
                len: 1,
                q: 1.0,
            },
        };
        NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1), NodeId(2)].into(),
            nodes: vec![root, child_a, child_b].into(),
            successors: vec![
                vec![
                    NodeSuccessor {
                        child: NodePos(1),
                        probability: 0.25,
                    },
                    NodeSuccessor {
                        child: NodePos(2),
                        probability: 0.75,
                    },
                ],
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 3,
            pool_stage: vec![StageIdx(1), StageIdx(1), StageIdx(0)],
        }
    }

    /// AC1: a multi-node root successor set with distinct edge (`0.25`/`0.75`)
    /// and within-node (`0.5`/`0.25`) weights assembles to the flat
    /// canonical-order product-weight vector — child A's openings (ascending
    /// child node id) before child B's, un-re-normalized.
    #[test]
    fn assemble_outcome_weights_k_fan_canonical_order_and_product_weights() {
        let ng = k_fan_root_node_graph();
        let mut out = Vec::new();
        assemble_outcome_weights(&ng, &ng.successors[NodePos(0)], &mut out);

        assert_eq!(
            out,
            vec![0.125, 0.125, 0.1875, 0.1875, 0.1875, 0.1875],
            "O(root) must be canonical-order (child A's 2 openings, then child \
             B's 4), each entry the exact product P(root→n)·q_(n,ω)"
        );
    }

    /// AC1: the analytical expectation regression — the K-fan's assembled
    /// weights, applied to a hand-supplied objective vector via
    /// `evaluate_risk`, equal `Σ_(n,ω) P(root→n)·q_(n,ω)·objective ·
    /// cost_scale_factor` exactly (every input is dyadic).
    #[test]
    fn root_outcome_weights_expectation_matches_analytical_sum() {
        let ng = k_fan_root_node_graph();
        let mut weights = Vec::new();
        assemble_outcome_weights(&ng, &ng.successors[NodePos(0)], &mut weights);

        let objectives = [8.0, 16.0, 40.0, 80.0, 160.0, 320.0];
        let cost_scale_factor = 1_000_000.0;

        let bound =
            RiskMeasure::Expectation.evaluate_risk(&objectives, &weights) * cost_scale_factor;

        // 0.125*8 + 0.125*16 + 0.1875*(40+80+160+320) = 3.0 + 112.5 = 115.5.
        assert!(
            (bound - 115_500_000.0).abs() < 1e-9,
            "expectation over the K-fan root outcome set must equal the \
             analytical sum 115.5 * cost_scale_factor, got {bound}"
        );
    }

    /// AC2: the `CVaR` regression on the same K-fan fixture — `evaluate_risk`'s
    /// own weighted-`CVaR` value (a hand-computed greedy allocation), no second
    /// `CVaR` α introduced in this module.
    #[test]
    fn root_outcome_weights_cvar_matches_evaluate_risk() {
        let ng = k_fan_root_node_graph();
        let mut weights = Vec::new();
        assemble_outcome_weights(&ng, &ng.successors[NodePos(0)], &mut weights);

        let objectives = [8.0, 16.0, 40.0, 80.0, 160.0, 320.0];
        let cost_scale_factor = 1_000_000.0;
        let rm = RiskMeasure::CVaR {
            alpha: 0.25,
            lambda: 1.0,
        };

        let bound = rm.evaluate_risk(&objectives, &weights) * cost_scale_factor;

        // Pure CVaR(alpha=0.25): per-outcome upper bound = weight / alpha = 4 *
        // weight. Greedy worst-first allocation exhausts remaining=1.0 on the
        // worst outcome (320.0, weight 0.1875): alloc = min(0.75, 1.0) = 0.75,
        // remaining = 0.25; next-worst (160.0, weight 0.1875): alloc =
        // min(0.75, 0.25) = 0.25, remaining = 0. CVaR = 320*0.75 + 160*0.25 = 280.
        let expected = 280.0 * cost_scale_factor;
        assert!(
            (bound - expected).abs() < 1e-9,
            "CVaR(0.25, 1.0) over the K-fan root outcome set must equal the \
             hand-computed greedy allocation {expected}, got {bound}"
        );
    }

    /// Chain byte-parity precondition (pinned, not hoped): on a chain, root's
    /// single successor at `P = 1` carrying `K` openings assembles to exactly
    /// the `1.0/(K as f64)` bit pattern, replicated `K` times.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn assemble_root_outcome_weights_chain_pins_uniform_bit_pattern() {
        let stochastic = wrap_opening_tree(1, simple_opening_tree(5));
        let node_graph = test_support::chain_node_graph(&stochastic);

        let mut weights = Vec::new();
        assemble_root_outcome_weights(&node_graph, &mut weights).unwrap();

        let expected_q = 1.0_f64 / 5.0_f64;
        assert_eq!(weights.len(), 5);
        for &w in &weights {
            assert_eq!(
                w.to_bits(),
                expected_q.to_bits(),
                "chain root weights must be the exact 1.0/(n as f64) bit \
                 pattern, not a normalized accumulation"
            );
        }
    }

    /// Error handling: a multi-node stage 0 has no reserved root source for
    /// `P(root→n)` across siblings — `SddpError::Validation` naming stage 0
    /// (defensive; structural validation of `nodes[]` happens upstream, not
    /// here).
    #[test]
    fn assemble_root_outcome_weights_multi_node_stage_0_is_validation_error() {
        let node_a = NodeRuntime {
            stage: StageIdx(0),
            pool_id: 0,
            openings: NodeOpenings {
                source: OpeningSource::Generated,
                offset: 0,
                len: 2,
                q: 0.5,
            },
        };
        let node_b = NodeRuntime {
            stage: StageIdx(0),
            pool_id: 1,
            openings: NodeOpenings {
                source: OpeningSource::Generated,
                offset: 0,
                len: 2,
                q: 0.5,
            },
        };
        let ng = NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1)].into(),
            nodes: vec![node_a, node_b].into(),
            successors: vec![Vec::new(), Vec::new()].into(),
            n_pools: 2,
            pool_stage: vec![StageIdx(0), StageIdx(0)],
        };

        let mut weights = Vec::new();
        let result = assemble_root_outcome_weights(&ng, &mut weights);

        match result {
            Err(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("stage 0"),
                    "validation error must name stage 0, got: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation naming stage 0, got {other:?}"),
        }
    }

    /// Root-pool regression pin: `lb_init_rank0` must load the ROOT's own pool cuts
    /// onto the stage-0 LB LP, resolving the root by STAGE, never by array position.
    ///
    /// The declared graph's stage-0 root is NOT the smallest-id node: canonical
    /// ascending-id order puts leaves (ids 1, 2) at positions 0, 1 and the root
    /// (id 3, stage 0) at position 2, so `nodes[0]` is a leaf and
    /// `nodes[0].pool_id` is the shared leaf pool, distinct from the root's pool.
    /// With the root pool carrying a different active-cut count than the leaf
    /// pool, the built LB cut batch's row count reveals which pool was resolved:
    /// the pre-fix `nodes[0].pool_id` read yields the leaf pool's count and
    /// fails this assertion; the stage-resolved root yields the root pool's.
    #[test]
    fn lb_init_rank0_resolves_root_pool_by_stage_not_array_position() {
        let mut fixture = SimpleLbFixture::new(
            minimal_template(),
            1,
            simple_opening_tree(1),
            InflowNonNegativityMethod::None,
            vec![0.0_f64],
        );
        let n_state = fixture.state.n_state;

        let leaf = |pool_id| NodeRuntime {
            stage: StageIdx(1),
            pool_id,
            openings: NodeOpenings {
                source: OpeningSource::Generated,
                offset: 0,
                len: 1,
                q: 1.0,
            },
        };
        let root = NodeRuntime {
            stage: StageIdx(0),
            pool_id: 0,
            openings: NodeOpenings {
                source: OpeningSource::Generated,
                offset: 0,
                len: 1,
                q: 1.0,
            },
        };
        fixture.node_graph = NodeGraph {
            node_ids: vec![NodeId(1), NodeId(2), NodeId(3)].into(),
            nodes: vec![leaf(1), leaf(1), root].into(),
            successors: vec![
                Vec::new(),
                Vec::new(),
                vec![
                    NodeSuccessor {
                        child: NodePos(0),
                        probability: 0.5,
                    },
                    NodeSuccessor {
                        child: NodePos(1),
                        probability: 0.5,
                    },
                ],
            ]
            .into(),
            n_pools: 2,
            pool_stage: vec![StageIdx(0), StageIdx(1)],
        };

        // Power precondition: the buggy nodes[0] read and the stage-resolved root
        // genuinely disagree on which pool is the root's.
        let root_pos = find_root_position(&fixture.node_graph).unwrap();
        let root_pool = fixture.node_graph.nodes[root_pos].pool_id;
        let leaf_pool = fixture.node_graph.nodes[NodePos(0)].pool_id;
        assert_eq!(
            root_pos,
            NodePos(2),
            "root is at canonical position 2 (id 3), not 0"
        );
        assert_ne!(
            root_pool, leaf_pool,
            "the fixture must distinguish the root pool from nodes[0].pool_id"
        );

        // Root pool: 3 active cuts; leaf pool at nodes[0]: 1. Distinct counts so
        // the resolved pool is observable through the built batch's row count.
        let mut fcf = make_fcf(2, n_state);
        let coeff = vec![0.0_f64; n_state];
        fcf.add_cut(NodeId(0), root_pool, 1, 0, 0.0, &coeff);
        fcf.add_cut(NodeId(0), root_pool, 1, 1, 0.0, &coeff);
        fcf.add_cut(NodeId(0), root_pool, 2, 0, 0.0, &coeff);
        fcf.add_cut(NodeId(0), leaf_pool, 1, 0, 0.0, &coeff);
        assert_eq!(fcf.pools[root_pool].active_count(), 3);
        assert_eq!(fcf.pools[leaf_pool].active_count(), 1);

        let mut solver = MockSolver::with_objectives(vec![0.0]);
        let mut lb_cut_batch = empty_row_batch();
        let sc_fixture = fixture.ctx();
        let ctx = sc_fixture.ctx();
        let training_ctx = fixture.training_ctx();
        lb_init_rank0(
            &mut solver,
            &fcf,
            &ctx,
            &training_ctx,
            &mut lb_cut_batch,
            None,
        )
        .expect("fixture's stage 0 carries a single alive node");

        assert_eq!(
            lb_cut_batch.num_rows,
            fcf.pools[root_pool].active_count(),
            "lb_init_rank0 must build the stage-0 LB LP from the ROOT's own pool \
             cuts (resolved by stage), not nodes[0]'s leaf pool"
        );
    }

    /// Inspection guard: production lower-bound code carries no
    /// `uniform_prob_buf` field or equivalent uniform fill — the root's
    /// assembled product weights are the only probability source
    /// `evaluate_risk` ever sees. Needles are assembled from fragments so this
    /// guard's own source text does not self-match (the scan below excludes
    /// the test module regardless).
    #[test]
    fn lower_bound_never_references_uniform_probability_fill() {
        let needles: [String; 2] = [
            ["uniform", "_prob_buf"].concat(),
            ["1.0_f64", " / objectives.len()"].concat(),
        ];
        let src = include_str!("lower_bound.rs");
        let prod_src = src
            .split("#[cfg(test)]")
            .next()
            .expect("module has a production region before the test module");

        let mut offenders: Vec<&str> = Vec::new();
        for needle in &needles {
            if prod_src.contains(needle.as_str()) {
                offenders.push(needle.as_str());
            }
        }
        assert!(
            offenders.is_empty(),
            "lower-bound production code must contain no uniform-probability \
             fill; offending fragments: {offenders:?}"
        );
    }
}
