//! Simulation forward pass for policy evaluation.
//!
//! [`simulate`] evaluates the trained policy on scenarios. Scenarios distributed
//! across ranks via two-level distribution; within-rank parallelism via Rayon.
//! Seed domain separated from training. No per-scenario allocations on hot path.

use std::collections::HashMap;
use std::sync::mpsc::{Sender, SyncSender};

use chrono::NaiveDate;
use cobre_comm::Communicator;
use cobre_core::commissioning::commissioning_active;
use cobre_core::{ContractType, EntityId, HydroPastDefluence, TrainingEvent};
use cobre_solver::ActiveProfile;
use cobre_solver::{SolverInterface, StageTemplate};
use cobre_stochastic::{ClassSampleRequest, ForwardNoiseTables, ForwardSampler, SampleRequest};

use crate::energy_conversion::EnergyConversionSet;
use crate::error::SddpError::Infeasible;
use crate::error::SddpError::Solver;
use crate::lp::builder::GenericConstraintRowEntry;
use crate::lp::builder::StageGeometry;
use crate::lp::indexer::{AnticipatedPlants, BlockIdx, BlockRowFamily, StudyDimensions};
use crate::noise::DownstreamAccumState;
use crate::noise::LagAccumState;
use crate::stage_solve::StageInputs;
use crate::stage_solve::assemble_outgoing_state;
use crate::stage_solve::fill_unscaled;
use crate::stage_solve::fill_unscaled_dual;
use crate::stage_solve::run_stage_solve;
use crate::{
    FutureCostFunction, SddpError,
    context::{StageContext, TrainingContext},
    dcs::{DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    lp::indexer::HydroCellIndex,
    setup::node_graph::{NodeId, NodePos, StageIdx, Traversal, advance_sampled_node},
    simulation::{
        config::SimulationConfig,
        error::SimulationError,
        extraction::EntityCounts,
        extraction::{
            HydroReverseLookup, SolutionView, StageExtractionSpec, TransitSeedArc,
            accumulate_category_costs, build_transit_seed, extract_anticipated_lanes,
            extract_stage_result_with_lookups,
        },
        types::{ScenarioCategoryCosts, SimulationScenarioResult, SimulationStageResult},
    },
    solver_stats::SolverStatsDelta,
    training::stage_solve_prep::{InflowNoise, StageSolvePrep, StageSolvePrepParams, StateSource},
    workspace::{CapturedBasis, SolverWorkspace},
};

/// Reserved sentinel iteration for every simulation seed derivation call
/// (noise draws and, on a declared graph, the transition draw).
/// `training::session::iteration_range` is 1-based
/// (`start_iteration + 1 ..= max_iterations`), so `0` never collides with a
/// real training iteration — domain separation from the training forward
/// pass follows from this alone, structurally, with no scenario-index offset
/// required for that purpose.
pub(crate) const SIMULATION_ITERATION: u32 = 0;

/// Per-worker scenario cost accumulation: `(scenario_id, total_cost, category_costs)`.
pub(crate) type WorkerCosts = Vec<(u32, f64, ScenarioCategoryCosts)>;

/// Per-worker solver statistics: `(scenario_id, opening, delta)`.
///
/// `opening` is always `-1` for simulation (no opening loop); the sentinel maps
/// to a NULL `Int32` in the parquet schema.
pub(crate) type WorkerStats = Vec<(u32, i32, SolverStatsDelta)>;

/// Result of a simulation run, containing per-scenario costs and solver statistics.
#[derive(Debug)]
pub struct SimulationRunResult {
    /// Per-scenario `(scenario_id, total_cost, category_costs)`, sorted by `scenario_id`.
    pub costs: Vec<(u32, f64, ScenarioCategoryCosts)>,
    /// Per-scenario `(scenario_id, opening, delta)`, sorted by `scenario_id`.
    pub solver_stats: Vec<(u32, i32, SolverStatsDelta)>,
    /// The run's resolved census leaf-path weights (canonical path order) when
    /// the traversal was `Enumerated`, else `None` — the exact
    /// [`crate::simulation::SimulationWeighting::Census`] input the caller feeds
    /// to [`crate::simulation::aggregate_simulation`], carried out here so the
    /// aggregation reuses the one `EnumeratedPlan` this run already built rather
    /// than re-resolving the `Traversal` a second time.
    pub census_weights: Option<Vec<f64>>,
}

/// Output-related inputs bundled from the caller for [`simulate`].
///
/// Groups the output-channel, unit-conversion arrays, and optional event sender
/// that would otherwise push the argument count of `simulate` beyond seven.
pub struct SimulationOutputSpec<'a> {
    /// Bounded channel used to stream completed scenario results to the caller.
    pub result_tx: &'a SyncSender<SimulationScenarioResult>,

    /// Per-stage block hours used to compute hourly energy from block dispatch.
    pub block_hours_per_stage: &'a [Vec<f64>],

    /// Entity counts for result extraction (hydros, thermals, lines, etc.).
    pub entity_counts: &'a EntityCounts,

    /// Per-stage active generic-constraint row metadata for extraction.
    pub generic_constraint_row_entries: &'a [Vec<GenericConstraintRowEntry>],

    /// Study-scope hydro-cell partition, threaded into every stage's
    /// `StageExtractionSpec`.
    pub hydro_cell_index: &'a HydroCellIndex,

    /// Per-station pumping power-consumption rate \[MW/(m³/s)\], ID-sorted
    /// parallel to `entity_counts.pumping_station_ids` and indexed by the SYSTEM
    /// station index — which, under the dense layout, IS the column-block position.
    pub pumping_consumption_mw_per_m3s: &'a [f64],

    /// Per-stage RESOLVED contract price \[$/`MWh`\]: one inner slice per study
    /// stage, flat with the per-stage stride `n_blks` — index `c * n_blks + blk`,
    /// `c` ID-sorted parallel to `entity_counts.contract_ids`. The resolved,
    /// possibly block-overridden `contract_bounds_at_block(c, t, blk).price_per_mwh`
    /// — never the `col_scale`-scaled LP objective.
    pub contract_prices_per_stage: &'a [Vec<f64>],

    /// Per-contract `(ContractType, per-family slot)`, ID-sorted parallel to
    /// `entity_counts.contract_ids`. Stage-invariant.
    pub contract_slots: &'a [(ContractType, usize)],

    /// Map from target hydro ID to source hydro indices that divert to it; empty
    /// when no hydros have diversion.
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,

    /// Per-stage per-hydro productivity (per-stage override, or base if none);
    /// `0.0` for FPHA hydros.
    pub hydro_productivities_per_stage: &'a [Vec<f64>],

    /// Pre-computed energy-conversion scalars for every `(hydro, stage)` pair.
    pub energy_conversion: &'a EnergyConversionSet,

    /// Minimum storage volume `V_min` per hydro plant (hm³), ID-sorted.
    pub hydro_min_storage_hm3: &'a [f64],

    /// Optional event sender for streaming progress events to the CLI/UI.
    pub event_sender: Option<Sender<TrainingEvent>>,

    /// Extended delivery-stage anchors: the `YYYYMM01` anchor of each delivery
    /// target stage `m`, indexed by `m` (study stages then the synthetic
    /// post-study continuation). The `anticipated_lanes` extractor dates a
    /// post-study-targeted decision by `[m]`, matching the policy manifest's
    /// `delivery_anchor_at` walk over the same extended calendar.
    pub extended_delivery_anchors: &'a [i32],

    /// Declared travel-time arcs for the rolling-seed emitter
    /// ([`crate::setup::StudySetup::transit_seed_arcs`]). Empty when the study
    /// declares no travel-time arc.
    pub transit_seed_arcs: &'a [TransitSeedArc],

    /// This run's own `past_defluences`, for the rolling-seed emitter's
    /// pre-study input-tail stitch.
    pub past_defluences: &'a [HydroPastDefluence],

    /// `(start_date, end_date)` per in-study stage, parallel to
    /// [`SimulationStageResult::stage_id`]'s positional index. Feeds the
    /// rolling-seed emitter's per-stage windows.
    pub study_stage_dates: &'a [(NaiveDate, NaiveDate)],
}

/// Per-scenario context bundled for `process_scenario_stages`.
///
/// Groups the scenario identifiers, scratch buffers, and noise sampler so the
/// processor does not exceed the clippy `too_many_arguments` budget.
pub(crate) struct ScenarioIds<'a> {
    /// Local scenario ID (0-based index within this rank's assigned slice).
    pub(crate) scenario_id: u32,
    /// This scenario's "path" identity passed to `ForwardSampler::sample` and
    /// the transition draw — the scenario's own native id, mirroring how a
    /// training forward pass's global scenario index plays the same role.
    pub(crate) global_scenario: u32,
    /// Total simulation scenario count, passed to `SampleRequest::total_scenarios`.
    pub(crate) total_scenarios: u32,
    /// Caller-owned buffer for raw noise output (reused across stages).
    pub(crate) raw_noise_buf: &'a mut [f64],
    /// Caller-owned gather/correlate scratch for wide correlation groups (reused across stages).
    pub(crate) corr_scratch: &'a mut [f64],
    /// Noise sampler used to draw per-stage stochastic values.
    pub(crate) sampler: &'a ForwardSampler<'a>,
    /// Per-run scenario-invariant tables backing `sampler`'s `OutOfSample` draws.
    pub(crate) noise_tables: &'a ForwardNoiseTables,
    /// The stage-0 root's canonical `NodeGraph` position — this scenario's
    /// sampled walk starts here, mirroring the training forward pass.
    pub(crate) root_node: NodePos,
}

/// Rebuild the stage `row_lower` slice into `scratch_buf` in original (unscaled)
/// units. An empty `row_scale` means no prescaling, so template rows are copied
/// without per-element division.
fn build_row_lower_unscaled<'a>(
    template_row_lower: &[f64],
    row_scale: &[f64],
    load_rhs_buf: &[f64],
    scratch_buf: &'a mut Vec<f64>,
    load_rows: BlockRowFamily,
    n_blks: usize,
    load_bus_indices: &[usize],
) -> &'a [f64] {
    scratch_buf.clear();
    scratch_buf.reserve(template_row_lower.len());

    if row_scale.is_empty() {
        scratch_buf.extend_from_slice(template_row_lower);
    } else {
        for (i, &val) in template_row_lower.iter().enumerate() {
            let scale = if i < row_scale.len() && row_scale[i] != 0.0 {
                row_scale[i]
            } else {
                1.0
            };
            scratch_buf.push(val / scale);
        }
    }

    // load_rhs_buf is already in unscaled MW.
    if !load_bus_indices.is_empty() && !load_rhs_buf.is_empty() {
        let mut rhs_idx = 0;
        for &bus_pos in load_bus_indices {
            for blk in 0..n_blks {
                scratch_buf[load_rows.row(bus_pos, BlockIdx::new(blk), n_blks)] =
                    load_rhs_buf[rhs_idx];
                rhs_idx += 1;
            }
        }
    }

    &scratch_buf[..template_row_lower.len()]
}

/// Stage identifiers bundled for `solve_simulation_stage`.
pub(crate) struct SimStageIds {
    /// Stage index (0-based) — seeds and array indexing key off this.
    pub(crate) t: StageIdx,
    /// Declared study `stage_id` (domain id) stamped into result records and
    /// error messages; resolved by position from the ordered study stage ids,
    /// never the positional index `t`.
    pub(crate) stage_id_u32: u32,
    /// Scenario ID for error messages.
    pub(crate) scenario_id: u32,
    /// Canonical `NodeGraph` position this scenario's sampled walk visits at
    /// stage `t` — the pool/node-id resolution site, never `t` itself once a
    /// stage carries more than one alive node.
    pub(crate) node: NodePos,
    /// Declared id of `node` (`node_graph.node_ids[node]`), resolved once at
    /// construction and reused by the solve context and result extraction.
    pub(crate) node_id: NodeId,
}

/// Load-path inputs bundled for `solve_simulation_stage`.
///
/// The LP is loaded via `load_model(frozen_template)` — the frozen template
/// already embeds all active cut rows as structural rows. No `add_rows` call
/// is needed.
pub(crate) struct SimStageLoadSpec<'a> {
    /// Frozen template for this stage; always populated after the startup re-freeze.
    pub(crate) frozen_template: &'a StageTemplate,
    /// Warm-start basis captured during training at the visited node, if any.
    pub(crate) warm_basis: Option<&'a CapturedBasis>,
}

/// Per-stage batched form of [`SimStageLoadSpec`] consumed by
/// `process_scenario_stages`; indexed by node via [`Self::stage`].
pub(crate) struct SimScenarioLoadSpec<'a> {
    pub(crate) frozen_templates: &'a [StageTemplate],
    /// Warm-start basis cache, one entry per canonical `NodeGraph` position
    /// (`build_basis_cache_from_checkpoint`'s and the training session's own
    /// `BasisStore`'s shape) — never per stage: on a branching graph several
    /// nodes share a stage, and a stage-keyed lookup would silently warm-start
    /// from whichever node's basis happened to land at that stage index,
    /// going cold for every other node sharing it. On a chain, node position
    /// and stage coincide, so this is byte-identical to the pre-rekey lookup.
    pub(crate) node_bases: &'a [Option<CapturedBasis>],
}

impl<'a> SimScenarioLoadSpec<'a> {
    /// `pool_id` indexes the per-pool frozen overlay; `node` indexes the
    /// node-keyed warm-start basis cache — both the visited node's own.
    #[inline]
    pub(crate) fn stage(&self, node: NodePos, pool_id: usize) -> SimStageLoadSpec<'a> {
        SimStageLoadSpec {
            frozen_template: &self.frozen_templates[pool_id],
            warm_basis: self.node_bases.get(node.0).and_then(Option::as_ref),
        }
    }
}

/// Pre-built reverse-lookup tables for simulation extraction.
///
/// Built once per worker via [`SimLookups::build`] and passed by reference into
/// every per-stage extraction call to eliminate per-`(scenario, stage)`
/// allocations on the hot path.
///
/// Thermal (anticipated-plant) membership is study-invariant (one owner);
/// FPHA/evaporation membership is per-`(hydro, stage)`, so a single global
/// hydro lookup would misclassify any stage whose membership differs from
/// stage 0's.
pub(crate) struct SimLookups {
    /// The study's anticipated-plant set (study-invariant).
    pub(crate) anticipated_plants: AnticipatedPlants,
    /// Per-stage hydro FPHA/evaporation lookups, indexed by stage.
    pub(crate) hydro_per_stage: Vec<HydroReverseLookup>,
}

impl SimLookups {
    /// Build the per-stage hydro lookups and clone the study's anticipated-plant
    /// set, from study dimensions, the per-stage geometry table, and entity
    /// counts.
    pub(crate) fn build(
        study_dims: &StudyDimensions,
        geometry_per_stage: &[StageGeometry],
        hydro_cell_index: &HydroCellIndex,
        n_hydros: usize,
    ) -> Self {
        Self {
            anticipated_plants: study_dims.anticipated_plants.clone(),
            hydro_per_stage: HydroReverseLookup::build_per_stage(
                geometry_per_stage,
                hydro_cell_index,
                n_hydros,
            ),
        }
    }
}

/// Map a stage-solve [`SddpError`] to a
/// [`SimulationError`], carrying the scenario/stage ids. Shared by the frozen
/// `run_stage_solve` path and the DCS `lazy_solve_preloaded` path so both report
/// failures identically.
fn map_sim_solver_error(e: SddpError, ids: &SimStageIds) -> SimulationError {
    match e {
        Infeasible {
            stage, scenario, ..
        } => {
            #[allow(clippy::cast_possible_truncation)]
            let scenario_id = scenario as u32;
            #[allow(clippy::cast_possible_truncation)]
            let stage_id = stage as u32;
            SimulationError::LpInfeasible {
                scenario_id,
                stage_id,
                solver_message: "LP infeasible".to_string(),
            }
        }
        Solver(other) => SimulationError::SolverError {
            scenario_id: ids.scenario_id,
            stage_id: ids.stage_id_u32,
            solver_message: other.to_string(),
        },
        other => SimulationError::SolverError {
            scenario_id: ids.scenario_id,
            stage_id: ids.stage_id_u32,
            solver_message: format!("{other}"),
        },
    }
}

/// Solve one stage for one simulation scenario, updating workspace in-place.
///
/// Returns `(immediate_cost, SimulationStageResult)`. The warm-start basis is a
/// read-only, per-stage training artifact, so determinism is preserved. Under
/// dynamic cut-selection the stage is solved lazily against the cut pool from the
/// cut-free base template (the frozen cut rows are unused); the realized primal is
/// identical at the optimum by exactness.
// RATIONALE: the sequential per-stage steps cannot split without fragmenting the
// per-stage invariant tracking.
#[allow(clippy::too_many_lines)]
pub(crate) fn solve_simulation_stage<S: SolverInterface>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    load_spec: &SimStageLoadSpec<'_>,
    output: &SimulationOutputSpec<'_>,
    ids: &SimStageIds,
    lookups: &SimLookups,
    raw_noise: &[f64],
) -> Result<(f64, SimulationStageResult), SimulationError> {
    let TrainingContext {
        state,
        study_dims,
        stochastic,
        dcs,
        ..
    } = training_ctx;
    let dcs = *dcs;
    let t = ids.t;
    // DCS loads the cut-free base so its fresh CutRowMap owns the resident cut
    // subset; loading the frozen template would double-append the embedded cut rows.
    if dcs.is_some() {
        ws.solver.load_model(ctx.template(t));
    } else {
        ws.solver.load_model(load_spec.frozen_template);
    }
    let prep_params = StageSolvePrepParams {
        state_source: StateSource(&ws.current_state),
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        t,
        &prep_params,
    );
    // stage_id (the commissioning key the dormancy predicate compares NCS windows
    // against), NOT the stage index `t`: they differ when negative-id placeholder
    // stages are filtered out.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let stage_id = training_ctx
        .stages
        .get(t.0)
        .map_or(t.0 as i32, |stage| stage.id);
    let n_stochastic_ncs = stochastic.n_stochastic_ncs();

    // mem::take (capacity retained) so these can be filled from `view` slices tied
    // to `ws` while `&mut ws` is live; restored at function end for buffer reuse.
    let mut unscaled_primal = std::mem::take(&mut ws.scratch.unscaled_primal);
    let mut unscaled_dual = std::mem::take(&mut ws.scratch.unscaled_dual);

    let col_scale = &ctx.template(t).col_scale;
    let row_scale = &ctx.template(t).row_scale;

    let pool_id = training_ctx.node_graph.nodes[ids.node].pool_id;

    let view_objective: f64 = if let Some(params) = dcs {
        // Simulation has no iteration counter; seed with `current_iteration = 0`.
        build_initial_resident_set(
            &fcf.pools[pool_id],
            0,
            params.k2,
            &mut ws.backward_accum.dcs_initial_resident,
        );
        let dcs_ctx = DcsSolveContext {
            stage_index: t,
            scenario_index: ids.scenario_id as usize,
            iteration: None, // disables the k1 window → every cut a candidate
            // Simulation solves one LP per (stage, scenario): always fresh.
            continue_carry: false,
            node_id: ids.node_id,
        };
        // Disjoint borrows of `ws`: `solver`, `dcs_initial_resident` (shared), and
        // `dcs_solve` (mut) are distinct fields.
        lazy_solve_preloaded(
            &mut ws.solver,
            ctx.template(t),
            &fcf.pools[pool_id],
            state,
            &training_ctx.cut_state_layouts[pool_id],
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )
        .map_err(|e| map_sim_solver_error(e, ids))?;
        let view = ws.backward_accum.dcs_solve.result_view();
        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        // INVARIANT: on the DCS path `view.dual` is LONGER than the structural
        // template (the lazy loop appends resident cut rows), carrying cut-row
        // duals at indices `>= template_num_rows`. Harmless: the reader only reads
        // structural-row indices. Do NOT truncate or add a `dual.len() ==
        // template_num_rows` check — that holds on the frozen path but NOT here, and
        // would drop the structural duals the reader needs.
        fill_unscaled_dual(&mut unscaled_dual, view.dual, row_scale);
        objective
    } else {
        let inputs = StageInputs {
            stage_context: ctx,
            pool: &fcf.pools[pool_id],
            stored_basis: load_spec.warm_basis,
            stage_index: t,
            scenario_index: ids.scenario_id as usize,
            iteration: None, // simulation has no iteration counter
            node_id: ids.node_id,
        };

        let view = run_stage_solve(ws, &inputs).map_err(|e| map_sim_solver_error(e, ids))?;

        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        fill_unscaled_dual(&mut unscaled_dual, view.dual, row_scale);
        objective
    };

    ws.scratch.unscaled_primal = unscaled_primal;
    ws.scratch.unscaled_dual = unscaled_dual;

    let include_terminal_theta =
        training_ctx.horizon.is_terminal(t.next().0) && fcf.pools[pool_id].has_warm_start_cuts();

    let (immediate_cost, result) = extract_sim_stage_result(
        &mut ws.scratch.inflow_m3s_buf,
        &mut ws.scratch.row_lower_buf,
        &ws.scratch.load_rhs_buf,
        &ws.scratch.ncs_col_upper_buf,
        &mut ws.scratch.ncs_col_upper_extract_buf,
        &ws.scratch.unscaled_primal,
        &ws.scratch.unscaled_dual,
        view_objective,
        include_terminal_theta,
        ctx,
        output,
        training_ctx,
        ids,
        stage_id,
        n_stochastic_ncs,
        lookups,
    );
    // Snapshot incoming lags before state is overwritten below.
    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[state.inflow_lags.clone()]);

    let stage_lag = ctx.stage_lag(t);
    let downstream_par_order = study_dims.downstream_par_order;
    // Pass unscaled_primal as a separate borrow so the borrow checker sees it is
    // disjoint from the &mut ws.scratch.lag_* fields passed alongside it.
    let unscaled_primal_ref: &[f64] = &ws.scratch.unscaled_primal;
    assemble_outgoing_state(
        &mut ws.current_state,
        unscaled_primal_ref,
        &ws.scratch.lag_matrix_buf,
        state,
        ctx.state_box(t),
        stage_lag,
        &mut LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );

    Ok((immediate_cost, result))
}

/// `t`'s load-balance row family and block count, or the empty family with `0`
/// blocks when the stage has no stochastic load buses.
fn resolve_load_rows(ctx: &StageContext<'_>, t: StageIdx) -> (BlockRowFamily, usize) {
    if ctx.load_bus_indices.is_empty() {
        (BlockRowFamily::default(), 0)
    } else {
        (ctx.geometry_per_stage[t.0].load_balance, ctx.block_count(t))
    }
}

/// Extract the cost and result record from a solved simulation stage LP.
///
/// RATIONALE (`too_many_arguments`): takes individual scratch field borrows rather
/// than `&mut ScratchBuffers` so `unscaled_primal`/`unscaled_dual` can be passed as
/// `&[f64]` while `inflow_m3s_buf`/`row_lower_buf` are `&mut`; a single
/// `&mut ScratchBuffers` would block that disjoint split.
// Rationale (too_many_lines): a single linear per-stage assembly culminating in one
// `StageExtractionSpec` literal; splitting it would only relocate the ~25 borrowed
// inputs into a parameter list, scattering the assembly the literal reads.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn extract_sim_stage_result(
    inflow_m3s_buf: &mut Vec<f64>,
    row_lower_buf: &mut Vec<f64>,
    load_rhs_buf: &[f64],
    ncs_col_upper_buf: &[f64],
    ncs_col_upper_extract_buf: &mut Vec<f64>,
    unscaled_primal: &[f64],
    unscaled_dual: &[f64],
    view_objective: f64,
    include_terminal_theta: bool,
    ctx: &StageContext<'_>,
    output: &SimulationOutputSpec<'_>,
    training_ctx: &TrainingContext<'_>,
    ids: &SimStageIds,
    stage_id: i32,
    n_stochastic_ncs: usize,
    lookups: &SimLookups,
) -> (f64, SimulationStageResult) {
    let t = ids.t;
    // Terminal boundary θ prices the post-horizon value-to-go: KEEP it in the
    // reported per-scenario cost (matching the training UB/LB); the interior
    // subtraction drops it. sddp.md "Terminal boundary FCF in the reported total
    // cost". Byte-neutral when terminal θ is [0,0].
    let immediate_cost = if include_terminal_theta {
        view_objective * ctx.cost_scale_factor
    } else {
        let theta_obj_coeff = ctx
            .templates
            .get(t.0)
            .and_then(|tmpl| tmpl.objective.get(training_ctx.state.theta).copied())
            .unwrap_or(1.0);
        let theta_contribution = unscaled_primal[training_ctx.state.theta] * theta_obj_coeff;
        (view_objective - theta_contribution) * ctx.cost_scale_factor
    };
    // Realized inflow Z_t from the z_h primal: total natural inflow (PAR lag
    // included), gross of withdrawal.
    inflow_m3s_buf.clear();
    inflow_m3s_buf.extend_from_slice(&unscaled_primal[training_ctx.state.z_inflow.clone()]);
    debug_assert_eq!(inflow_m3s_buf.len(), training_ctx.state.hydro_count);
    let blk_hrs = output.block_hours_per_stage[t.0].as_slice();
    let (load_rows, load_n_blks) = resolve_load_rows(ctx, t);
    let row_lower_ref = build_row_lower_unscaled(
        &ctx.template(t).row_lower,
        &ctx.template(t).row_scale,
        load_rhs_buf,
        row_lower_buf,
        load_rows,
        load_n_blks,
        ctx.load_bus_indices,
    );
    // NCS upper bounds for extraction, in dense system-column order
    // (`ncs_sys * stage_n_blks + blk`).
    let stage_n_blks = ctx.block_count(t);
    let geometry = &ctx.geometry_per_stage[t.0];
    // Start from the template `col_upper`, then overwrite each non-dormant
    // stochastic column with the per-scenario realized availability. A dormant slot
    // is skipped so its template `0` survives — copying its stochastic cap would
    // report a nonzero available for a column the LP pinned to `0`.
    let ncs_col_upper: &[f64] = if geometry.ncs_generation.is_empty() {
        &[]
    } else {
        let ncs_cols = geometry.ncs_generation.clone();
        ncs_col_upper_extract_buf.clear();
        ncs_col_upper_extract_buf.extend_from_slice(&ctx.template(t).col_upper[ncs_cols]);
        if n_stochastic_ncs > 0 && !ncs_col_upper_buf.is_empty() {
            let dense_col = ctx.ncs_stochastic_dense_col;
            let windows = ctx.ncs_stochastic_windows;
            for (slot, &col) in dense_col.iter().enumerate() {
                let (entry, exit) = windows[slot];
                if !commissioning_active(entry, exit, stage_id) {
                    continue;
                }
                let dst = col * stage_n_blks;
                let src = slot * stage_n_blks;
                debug_assert!(
                    dst + stage_n_blks <= ncs_col_upper_extract_buf.len(),
                    "ncs extract dst out of range: dense_col < ncs_n by construction",
                );
                debug_assert!(
                    src + stage_n_blks <= ncs_col_upper_buf.len(),
                    "ncs extract src out of range: slot < n_stochastic_ncs by construction",
                );
                ncs_col_upper_extract_buf[dst..dst + stage_n_blks]
                    .copy_from_slice(&ncs_col_upper_buf[src..src + stage_n_blks]);
            }
        }
        ncs_col_upper_extract_buf.as_slice()
    };
    let hydro_lookup = &lookups.hydro_per_stage[t.0];
    let view = SolutionView {
        primal: unscaled_primal,
        dual: unscaled_dual,
        objective: view_objective,
        objective_coeffs: &ctx.template(t).objective,
        row_lower: row_lower_ref,
    };
    let spec = StageExtractionSpec {
        state: training_ctx.state,
        study_dims: training_ctx.study_dims,
        geometry,
        hydro_cell_index: output.hydro_cell_index,
        entity_counts: output.entity_counts,
        inflow_m3s_per_hydro: inflow_m3s_buf,
        block_hours: blk_hrs,
        generic_constraint_entries: &output.generic_constraint_row_entries[t.0],
        ncs_col_upper,
        pumping_consumption_mw_per_m3s: output.pumping_consumption_mw_per_m3s,
        contract_prices: &output.contract_prices_per_stage[t.0],
        contract_slots: output.contract_slots,
        diversion_upstream: output.diversion_upstream,
        hydro_productivities: &output.hydro_productivities_per_stage[t.0],
        col_scale: &ctx.template(t).col_scale,
        row_scale: &ctx.template(t).row_scale,
        cumulative_discount_factor: ctx.cumulative_discount_factor(t),
        cost_scale_factor: ctx.cost_scale_factor,
        energy_conversion: output.energy_conversion,
        hydro_min_storage_hm3: output.hydro_min_storage_hm3,
        stage_index: t.0,
        horizon: training_ctx.horizon,
        anticipated_windows: ctx.anticipated_windows,
        study_stage_ids: ctx.study_stage_ids,
    };
    let mut result = extract_stage_result_with_lookups(
        &view,
        &spec,
        ids.stage_id_u32,
        ids.node_id,
        hydro_lookup,
        &lookups.anticipated_plants,
    );
    result.anticipated_lanes = extract_anticipated_lanes(
        &view,
        &spec,
        output.extended_delivery_anchors,
        ids.stage_id_u32,
    );
    (immediate_cost, result)
}

/// Reset workspace state to the initial conditions for a new scenario.
pub(crate) fn reset_scenario_state<S: SolverInterface>(
    ws: &mut SolverWorkspace<S>,
    sampler: &ForwardSampler<'_>,
    global_scenario: u32,
    total_scenarios: u32,
    inflow_lags_start: usize,
    training_ctx: &TrainingContext<'_>,
    root_node: NodePos,
) {
    // Reset solver simplex state at the scenario boundary so a scenario's result
    // cannot depend on which scenarios ran before it (determinism across thread/
    // rank counts). No-op for HiGHS; recreates the model for CLP, whose
    // `Clp_loadProblem` leaves the rim/pricing state stale.
    ws.solver.reset_solver_state();

    let TrainingContext {
        initial_state,
        lag_accum_seed,
        lag_weight_seed,
        node_graph,
        ..
    } = training_ctx;
    let (node_opening_offset, node_opening_len) = node_graph.node_opening_range(root_node);
    ws.current_state.clear();
    ws.current_state.extend_from_slice(initial_state);
    sampler.apply_initial_state(
        &ClassSampleRequest {
            iteration: SIMULATION_ITERATION,
            scenario: global_scenario,
            stage: 0,
            stage_idx: 0,
            total_scenarios,
            noise_group_id: 0,
            node_opening_offset,
            node_opening_len,
            pinned_scenario: node_graph.node_pinned_scenario(root_node),
        },
        &mut ws.current_state,
        inflow_lags_start,
    );
    // Seed (or zero) the lag accumulator so it does not carry state across
    // scenarios; a non-empty seed pre-fills the partial period with pre-study data.
    if lag_accum_seed.is_empty() {
        ws.scratch.lag_accumulator.fill(0.0);
        ws.scratch.lag_weight_accum.fill(0.0);
    } else {
        ws.scratch.lag_accumulator[..lag_accum_seed.len()].copy_from_slice(lag_accum_seed);
        ws.scratch.lag_weight_accum[..lag_weight_seed.len()].copy_from_slice(lag_weight_seed);
    }
    ws.scratch.downstream_accumulator.fill(0.0);
    ws.scratch.downstream_weight_accum = 0.0;
    ws.scratch.downstream_completed_lags.fill(0.0);
    ws.scratch.downstream_n_completed = 0;
}

/// Advance `node` to the one this scenario visits at `t + 1` (chain-parity
/// contract stated once at
/// [`advance_sampled_node`]).
fn advance_simulation_node(
    training_ctx: &TrainingContext<'_>,
    node: NodePos,
    stage_id_u32: u32,
    global_scenario: u32,
) -> NodePos {
    advance_sampled_node(
        training_ctx.node_graph,
        node,
        SIMULATION_ITERATION,
        global_scenario,
        stage_id_u32,
    )
}

pub(crate) fn process_scenario_stages<S: SolverInterface>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    load_spec: &SimScenarioLoadSpec<'_>,
    output: &SimulationOutputSpec<'_>,
    ids: &mut ScenarioIds<'_>,
    lookups: &SimLookups,
) -> Result<(f64, Vec<SimulationStageResult>), SimulationError> {
    let TrainingContext {
        horizon,
        state,
        node_graph,
        ..
    } = training_ctx;
    let num_stages = horizon.num_stages();
    reset_scenario_state(
        ws,
        ids.sampler,
        ids.global_scenario,
        ids.total_scenarios,
        state.inflow_lags.start,
        training_ctx,
        ids.root_node,
    );
    let mut total_cost = 0.0_f64;
    let mut stage_results = Vec::with_capacity(num_stages);
    let mut node = ids.root_node;

    #[allow(clippy::needless_range_loop)] // t indexes load_spec, ctx arrays, and SimStageIds
    for t in (0..num_stages).map(StageIdx) {
        // Seeds key off the positional stage index `t` (unchanged — a re-key here
        // would perturb the noise/transition draws); the output stage_id is the
        // declared domain id, resolved by position from the ordered study ids.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let stage_seed = t.0 as u32;
        // Domain id by position; falls back to the positional index only if a
        // caller supplies a short study_stage_ids (never in production, where it
        // has one entry per study stage).
        let output_stage_id = ctx
            .study_stage_id(t)
            .unwrap_or_else(|| i32::try_from(t.0).unwrap_or(i32::MAX));
        let (node_opening_offset, node_opening_len) = node_graph.node_opening_range(node);
        let noise = ids.sampler.sample(SampleRequest {
            iteration: SIMULATION_ITERATION,
            scenario: ids.global_scenario,
            stage: stage_seed,
            stage_idx: t.0,
            noise_buf: ids.raw_noise_buf,
            corr_scratch: ids.corr_scratch,
            total_scenarios: ids.total_scenarios,
            noise_group_id: ctx.noise_group_id_at(t),
            node_opening_offset,
            node_opening_len,
            pinned_scenario: node_graph.node_pinned_scenario(node),
            tables: ids.noise_tables,
        })?;
        let raw_noise = noise.as_slice();

        let (cost, result) = solve_simulation_stage(
            ws,
            ctx,
            fcf,
            training_ctx,
            &load_spec.stage(node, node_graph.nodes[node].pool_id),
            output,
            &SimStageIds {
                t,
                #[allow(clippy::cast_sign_loss)]
                stage_id_u32: output_stage_id as u32,
                scenario_id: ids.scenario_id,
                node,
                node_id: node_graph.node_ids[node],
            },
            lookups,
            raw_noise,
        )?;
        let cum_d = ctx.cumulative_discount_factor(t);
        total_cost += cum_d * cost;
        stage_results.push(result);

        if t.next().0 < num_stages {
            node = advance_simulation_node(training_ctx, node, stage_seed, ids.global_scenario);
        }
    }
    Ok((total_cost, stage_results))
}

/// Emit an in-progress simulation event if a sender is available.
pub(crate) fn emit_sim_progress(
    sender: Option<&Sender<TrainingEvent>>,
    scenario_cost: f64,
    solve_time_ms: f64,
    lp_solves: u64,
    completed: u32,
    total: u32,
    elapsed_ms: u64,
) {
    if let Some(s) = sender {
        let _ = s.send(TrainingEvent::SimulationProgress {
            scenarios_complete: completed,
            scenarios_total: total,
            elapsed_ms,
            scenario_cost,
            solve_time_ms,
            lp_solves,
        });
    }
}

/// Accumulate per-category costs from all stage results, send the scenario
/// result through the channel, and return a compact `(scenario_id, total_cost,
/// category_costs)` tuple for MPI aggregation.
pub(crate) fn dispatch_scenario_result(
    output: &SimulationOutputSpec<'_>,
    scenario_id: u32,
    total_cost: f64,
    stage_results: Vec<SimulationStageResult>,
) -> Result<(u32, f64, ScenarioCategoryCosts), SimulationError> {
    let mut category_costs = ScenarioCategoryCosts {
        resource_cost: 0.0,
        recourse_cost: 0.0,
        violation_cost: 0.0,
        regularization_cost: 0.0,
        imputed_cost: 0.0,
    };
    for sr in &stage_results {
        for c in &sr.costs {
            accumulate_category_costs(c, &mut category_costs);
        }
    }
    let compact_category = category_costs.clone();
    let transit_seed = build_transit_seed(
        &stage_results,
        output.study_stage_dates,
        output.transit_seed_arcs,
        output.past_defluences,
        output.block_hours_per_stage,
    );
    output
        .result_tx
        .send(SimulationScenarioResult {
            scenario_id,
            total_cost,
            per_category_costs: category_costs,
            stages: stage_results,
            transit_seed,
        })
        .map_err(|_| SimulationError::ChannelClosed)?;
    Ok((scenario_id, total_cost, compact_category))
}

/// Evaluate the trained SDDP policy on a set of scenarios.
///
/// Thin shim that delegates to `SimulationState::run`.
/// All scheduling, freeze logic, and rayon parallelism live in
/// `crate::simulation::state`.
///
/// # Errors
///
/// Returns `Err(SimulationError::LpInfeasible { .. })` when a stage LP has no
/// feasible solution, `Err(SimulationError::SolverError { .. })` for other
/// terminal LP solver failures, and `Err(SimulationError::ChannelClosed)` when
/// the channel receiver has been dropped.
// RATIONALE: splitting would only relocate the parameter list, not reduce it —
// SimulationInputs already bundles what can be bundled.
#[allow(clippy::too_many_arguments)]
pub fn simulate<S, C: Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    config: &SimulationConfig,
    output: SimulationOutputSpec<'_>,
    frozen_templates: Option<&[StageTemplate]>,
    node_bases: &[Option<CapturedBasis>],
    comm: &C,
    traversal: &Traversal,
) -> Result<SimulationRunResult, SimulationError>
where
    S: SolverInterface<Profile = ActiveProfile> + Send,
{
    use crate::simulation::state::{SimulationInputs, SimulationState};
    let mut state = SimulationState::new(training_ctx.horizon.num_stages());
    state.set_profile(config.profile);
    state.run(&mut SimulationInputs {
        workspaces,
        ctx,
        fcf,
        training_ctx,
        config,
        output,
        frozen_templates,
        node_bases,
        comm,
        traversal,
    })
}

#[cfg(test)]
mod tests;
