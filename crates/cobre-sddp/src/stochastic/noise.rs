//! Shared noise transformation functions for the LP patching hot path.
//!
//! Single home for the noise→RHS transforms so a fix to one applies to every
//! call site (forward, backward, lower-bound).

use cobre_core::commissioning::commissioning_active;
use cobre_core::temporal::StageLagTransition;
use cobre_solver::SolverInterface;
use cobre_stochastic::par::lag_kernel::{LagMajor, advance_lag_chain};
use cobre_stochastic::{StochasticContext, evaluate_par_batch, solve_par_noise_batch};

use crate::indexer::{BlockIdx, NcsSys, StateSpace};
use crate::lp::builder::StageGeometry;
use crate::{
    InflowNonNegativityMethod, context::TrainingContext, setup::node_graph::StageIdx,
    workspace::ScratchBuffers,
};

/// Compute effective (possibly clamped) eta for each hydro.
///
/// Under truncation, when any PAR(p) inflow would be negative each negative
/// hydro's eta is raised to `eta_floor` (the value producing zero inflow);
/// other methods pass raw eta through.
pub(crate) fn compute_effective_eta(
    hydro_noise: &[f64],
    inflow_method: InflowNonNegativityMethod,
    par_inflows: &[f64],
    eta_floor: &[f64],
    effective_eta: &mut Vec<f64>,
) {
    effective_eta.clear();
    let n_hydros = hydro_noise.len();

    match inflow_method {
        InflowNonNegativityMethod::Truncation
        | InflowNonNegativityMethod::TruncationWithPenalty => {
            let has_negative = par_inflows.iter().take(n_hydros).any(|&a| a < 0.0);
            for h in 0..n_hydros {
                let eta = hydro_noise[h];
                let clamped = if has_negative && par_inflows[h] < 0.0 {
                    eta.max(eta_floor[h])
                } else {
                    eta
                };
                effective_eta.push(clamped);
            }
        }
        InflowNonNegativityMethod::None | InflowNonNegativityMethod::Penalty => {
            effective_eta.extend_from_slice(hydro_noise);
        }
    }
}

/// Returns `true` when `stochastic` carries a PAR model — the shared guard
/// for the water-balance patch. Its shape is validated once at setup
/// (`validate_par_shape`); every reader here checks presence only.
#[inline]
pub(crate) fn has_par_model(stochastic: &StochasticContext) -> bool {
    stochastic.par().n_stages() > 0
}

/// Transform raw inflow noise `η` into the pure z-inflow anchor rate
/// (`z_inflow_rhs_buf`), applying [`compute_effective_eta`] clamping under
/// truncation.
// Rationale: clippy::similar_names flags the role-(a) `state` handle (bound from
// `training_ctx.state`) next to the `stage` index; both are established names, so
// renaming either to satisfy the heuristic would obscure intent.
#[allow(clippy::similar_names)]
pub(crate) fn transform_inflow_noise(
    raw_noise: &[f64],
    stage: StageIdx,
    current_state: &[f64],
    training_ctx: &TrainingContext<'_>,
    scratch: &mut ScratchBuffers,
) {
    let inflow_method = training_ctx.inflow_method;
    let stochastic = training_ctx.stochastic;
    let dims = stochastic.class_dimensions();
    let n_hydros = dims.n_hydros;
    let hydro_noise = &raw_noise[dims.hydro_range()];
    let state = training_ctx.state;

    scratch.z_inflow_rhs_buf.clear();

    let par_lp = stochastic.par();
    let has_par = has_par_model(stochastic);

    match inflow_method {
        InflowNonNegativityMethod::Truncation
        | InflowNonNegativityMethod::TruncationWithPenalty => {
            let max_order = state.max_par_order;
            scratch.lag_matrix_buf.clear();
            scratch
                .lag_matrix_buf
                .extend_from_slice(&current_state[state.inflow_lags.clone()]);
            debug_assert_eq!(scratch.lag_matrix_buf.len(), max_order * n_hydros);

            scratch.par_inflow_buf.clear();
            scratch.par_inflow_buf.resize(n_hydros, 0.0);
            evaluate_par_batch(
                par_lp,
                stage.0,
                &scratch.lag_matrix_buf,
                hydro_noise,
                &mut scratch.par_inflow_buf,
            );

            let has_negative = scratch.par_inflow_buf.iter().any(|&a| a < 0.0);
            if has_negative {
                scratch.eta_floor_buf.clear();
                scratch.eta_floor_buf.resize(n_hydros, f64::NEG_INFINITY);
                let zero_targets = &scratch.zero_targets_buf[..n_hydros];
                solve_par_noise_batch(
                    par_lp,
                    stage.0,
                    &scratch.lag_matrix_buf,
                    zero_targets,
                    &mut scratch.eta_floor_buf,
                );
            }
        }
        InflowNonNegativityMethod::None | InflowNonNegativityMethod::Penalty => {}
    }

    compute_effective_eta(
        hydro_noise,
        *inflow_method,
        &scratch.par_inflow_buf,
        &scratch.eta_floor_buf,
        &mut scratch.effective_eta_buf,
    );

    for (h, &eta_eff) in scratch.effective_eta_buf.iter().enumerate() {
        // Z-inflow RHS in m3/s: no zeta, no withdrawal.
        if has_par {
            let base = par_lp.deterministic_base(stage.0, h);
            let sigma = par_lp.sigma(stage.0, h);
            scratch.z_inflow_rhs_buf.push(base + sigma * eta_eff);
        } else {
            scratch.z_inflow_rhs_buf.push(0.0);
        }
    }
}

/// Shift the lag portion of the outgoing state vector using realized inflow,
/// newest lag set to the realized inflow from the LP primal.
#[cfg(test)]
pub(crate) fn shift_lag_state(
    state: &mut [f64],
    incoming_lags: &[f64],
    unscaled_primal: &[f64],
    layout: &StateSpace,
) {
    let n_h = layout.hydro_count;
    let l_max = layout.max_par_order;
    if l_max == 0 || n_h == 0 {
        return;
    }
    let lag_start = layout.inflow_lags.start;
    for h in 0..n_h {
        let z_t_h = unscaled_primal[layout.z_inflow.start + h];
        // Read from incoming_lags (lag-major: lag * n_h + h) to avoid aliasing state.
        for lag in (1..l_max).rev() {
            state[lag_start + lag * n_h + h] = incoming_lags[(lag - 1) * n_h + h];
        }
        state[lag_start + h] = z_t_h;
    }
}

// LagAccumState/DownstreamAccumState alias the kernel's own accumulator
// structs (identical fields) so stage_solve.rs/pipeline.rs and their tests
// keep constructing them under these names.
pub(crate) use cobre_stochastic::par::lag_kernel::DownstreamLagAccum as DownstreamAccumState;
pub(crate) use cobre_stochastic::par::lag_kernel::PrimaryLagAccum as LagAccumState;

/// Accumulate this stage's inflow and, when a lag period finalizes, shift the
/// lag state — supporting multi-resolution studies where stages are shorter than
/// the lag granularity (e.g. weekly stages feeding a monthly lag slot).
///
/// For the monthly identity case (`accumulate_weight=1.0, spillover_weight=0.0,
/// finalize_period=true`) this produces bit-for-bit identical results to
/// the test-only `shift_lag_state`.
///
/// Thin adapter: resolves the LP-`StateSpace` offsets into plain slices, then
/// delegates the accumulate/finalize/shift/downstream-ring algorithm to
/// [`advance_lag_chain`].
///
/// # Panics (debug only)
///
/// Panics in debug builds if `lag.accumulator.len() < layout.hydro_count`.
///
/// # Downstream accumulation (multi-resolution studies)
///
/// When `ds.accumulator` is non-empty, a coarser-resolution ring buffer is
/// maintained in parallel; see [`DownstreamAccumState`] for the empty-slice
/// (uniform-resolution) contract.
///
/// # Anticipated-thermal state
///
/// Does NOT touch the anticipated ring: it transitions in-LP via
/// `commit_out`'s definition rows and rides the same
/// plain-copy-outgoing path as storage and travel-time buckets.
pub(crate) fn accumulate_and_shift_lag_state(
    state: &mut [f64],
    incoming_lags: &[f64],
    unscaled_primal: &[f64],
    layout: &StateSpace,
    stage_lag: &StageLagTransition,
    lag: &mut LagAccumState<'_>,
    ds: &mut DownstreamAccumState<'_>,
) {
    let lag_start = layout.inflow_lags.start;
    let n_h = layout.hydro_count;
    let l_max = layout.max_par_order;

    // LagMajor::index treats offset 0 as this call's lag block, not the state
    // vector's absolute start — slicing from `lag_start` keeps every kernel
    // write on the right hydro/lag; slicing from 0 would silently misdirect
    // every write by `lag_start`.
    advance_lag_chain(
        LagMajor {
            entity_count: n_h,
            max_order: l_max,
        },
        &mut state[lag_start..],
        incoming_lags,
        &unscaled_primal[layout.z_inflow.clone()],
        stage_lag,
        lag,
        ds,
    );
}

/// Trajectory-carried scratch accumulators captured at a node's outgoing edge
/// and restored as its children's incoming state — the water-travel/derived-lag
/// counterpart of the LP-column state that rides the record's `state` field.
/// Empty/zero on a study with no PAR lags or downstream travel time (the K-fan
/// path), where every field below is length-0 or `0.0`.
#[derive(Clone, Default)]
pub(crate) struct AccumSnapshot {
    lag_accumulator: Vec<f64>,
    lag_weight_accum: Vec<f64>,
    downstream_accumulator: Vec<f64>,
    downstream_weight_accum: f64,
    downstream_completed_lags: Vec<f64>,
    downstream_n_completed: usize,
}

fn copy_vec_from_slice(dst: &mut Vec<f64>, src: &[f64]) {
    dst.clear();
    dst.extend_from_slice(src);
}

impl AccumSnapshot {
    pub(crate) fn capture_from(&mut self, ws_scratch: &ScratchBuffers) {
        copy_vec_from_slice(&mut self.lag_accumulator, &ws_scratch.lag_accumulator);
        copy_vec_from_slice(&mut self.lag_weight_accum, &ws_scratch.lag_weight_accum);
        copy_vec_from_slice(
            &mut self.downstream_accumulator,
            &ws_scratch.downstream_accumulator,
        );
        self.downstream_weight_accum = ws_scratch.downstream_weight_accum;
        copy_vec_from_slice(
            &mut self.downstream_completed_lags,
            &ws_scratch.downstream_completed_lags,
        );
        self.downstream_n_completed = ws_scratch.downstream_n_completed;
    }

    pub(crate) fn copy_into(&self, dst: &mut AccumSnapshot) {
        copy_vec_from_slice(&mut dst.lag_accumulator, &self.lag_accumulator);
        copy_vec_from_slice(&mut dst.lag_weight_accum, &self.lag_weight_accum);
        copy_vec_from_slice(
            &mut dst.downstream_accumulator,
            &self.downstream_accumulator,
        );
        dst.downstream_weight_accum = self.downstream_weight_accum;
        copy_vec_from_slice(
            &mut dst.downstream_completed_lags,
            &self.downstream_completed_lags,
        );
        dst.downstream_n_completed = self.downstream_n_completed;
    }

    pub(crate) fn restore_into(&self, ws_scratch: &mut ScratchBuffers) {
        copy_vec_from_slice(&mut ws_scratch.lag_accumulator, &self.lag_accumulator);
        copy_vec_from_slice(&mut ws_scratch.lag_weight_accum, &self.lag_weight_accum);
        copy_vec_from_slice(
            &mut ws_scratch.downstream_accumulator,
            &self.downstream_accumulator,
        );
        ws_scratch.downstream_weight_accum = self.downstream_weight_accum;
        copy_vec_from_slice(
            &mut ws_scratch.downstream_completed_lags,
            &self.downstream_completed_lags,
        );
        ws_scratch.downstream_n_completed = self.downstream_n_completed;
    }
}

/// Transform raw load noise `η` into patched load-balance RHS values, one per
/// load bus and block, clamped at zero so load demand is never negative.
pub(crate) fn transform_load_noise(
    raw_noise: &[f64],
    stochastic: &StochasticContext,
    stage: StageIdx,
    block_count: usize,
    load_rhs_buf: &mut Vec<f64>,
) {
    let dims = stochastic.class_dimensions();
    load_rhs_buf.clear();
    if dims.n_load_buses == 0 {
        return;
    }
    let load_lp = stochastic.normal();
    for (lb_idx, &eta) in raw_noise[dims.load_bus_range()].iter().enumerate() {
        let mean = load_lp.mean(stage.0, lb_idx);
        let std = load_lp.std(stage.0, lb_idx);
        let realization = (mean + std * eta).max(0.0);
        for blk in 0..block_count {
            let factor = load_lp.block_factor(stage.0, lb_idx, blk);
            load_rhs_buf.push(realization * factor);
        }
    }
}

/// Transform raw NCS noise into per-block column lower/upper bounds.
///
/// Availability `α = clamp(mean + std · η, 0, 1)` is a **dimensionless factor**;
/// the realized cap is `max_gen · α · block_factor`. The parquet `(mean, std)`
/// are stored as factors, not MW (the authoritative statement of the NCS
/// stochastic availability-factor contract).
///
/// With `allow_curtailment == false` the lower bound equals the upper bound, so
/// the source must run at exactly the realized availability (aggregate
/// generation pre-netted from load); with `true` the lower bound is zero and the
/// LP may curtail.
///
/// # Panics
///
/// Panics in debug builds when `ncs_allow_curtailment.len() !=
/// ncs_max_gen.len()` or either slice is shorter than
/// `stochastic.n_stochastic_ncs()`.
pub(crate) fn transform_ncs_noise(
    raw_noise: &[f64],
    stochastic: &StochasticContext,
    stage: StageIdx,
    block_count: usize,
    ncs_max_gen: &[f64],
    ncs_allow_curtailment: &[bool],
    ncs_col_lower_buf: &mut Vec<f64>,
    ncs_col_upper_buf: &mut Vec<f64>,
) {
    let dims = stochastic.class_dimensions();
    ncs_col_upper_buf.clear();
    ncs_col_lower_buf.clear();
    if dims.n_ncs == 0 {
        return;
    }
    debug_assert_eq!(
        ncs_allow_curtailment.len(),
        ncs_max_gen.len(),
        "ncs_allow_curtailment and ncs_max_gen must have matching length",
    );
    let ncs_lp = stochastic.ncs_normal();
    for (ncs_idx, &eta) in raw_noise[dims.ncs_range()].iter().enumerate() {
        let mean = ncs_lp.mean(stage.0, ncs_idx);
        let std = ncs_lp.std(stage.0, ncs_idx);
        let max_gen = ncs_max_gen[ncs_idx];
        let availability_ratio = (mean + std * eta).clamp(0.0, 1.0);
        let realization = max_gen * availability_ratio;
        let allow_curtailment = ncs_allow_curtailment[ncs_idx];
        for blk in 0..block_count {
            let factor = ncs_lp.block_factor(stage.0, ncs_idx, blk);
            let upper = realization * factor;
            ncs_col_upper_buf.push(upper);
            ncs_col_lower_buf.push(if allow_curtailment { 0.0 } else { upper });
        }
    }
}

/// Rebuild the NCS column-index buffer for one stage's stochastic NCS columns.
///
/// Each column index strides by `dense_col[slot]` — the slot's NCS **system
/// index** (dense column position), not the raw slot index: the dense column
/// block is system-indexed, so striding by slot misaddresses the column whenever
/// only a subset of NCS are stochastic or their orders diverge.
///
/// Callers rebuild lazily on a stage transition, when the stage's own
/// `geometry.ncs_generation` range changes.
pub(crate) fn build_dense_ncs_col_indices(
    dense_col: &[usize],
    geometry: &StageGeometry,
    indices_out: &mut Vec<usize>,
) {
    indices_out.clear();
    for &col in dense_col {
        for blk in 0..geometry.n_blks {
            indices_out.push(geometry.ncs_generation_col(NcsSys::new(col), BlockIdx::new(blk)));
        }
    }
}

/// Gather the stochastic slots' NCS column bounds, forcing dormant slots to `[0, 0]`.
///
/// Copies each `transform_ncs_noise` block (one `block_count`-wide block per slot)
/// into `lower_out` / `upper_out`, EXCEPT a commissioning-dormant slot (whose
/// `windows[slot]` excludes `stage_id` per `commissioning_active`), whose block is
/// forced to `[0, 0]`. The forbidden alternative — copying the stochastic cap for a
/// dormant slot — would let a not-yet-commissioned source dispatch. The output runs
/// parallel to [`build_dense_ncs_col_indices`] (equal length `windows.len() *
/// block_count`) and reproduces the source buffers exactly when no slot is dormant.
/// Refilled every opening (the bounds change each opening, unlike the index buffer).
///
/// # Panics
///
/// Panics in debug builds when `lower_src` or `upper_src` is shorter than
/// `windows.len() * block_count`.
pub(crate) fn gather_dense_ncs_bounds(
    windows: &[(Option<i32>, Option<i32>)],
    stage_id: i32,
    block_count: usize,
    lower_src: &[f64],
    upper_src: &[f64],
    lower_out: &mut Vec<f64>,
    upper_out: &mut Vec<f64>,
) {
    lower_out.clear();
    upper_out.clear();
    debug_assert!(
        lower_src.len() >= windows.len() * block_count,
        "lower_src too short for windows: every slot strides by block_count",
    );
    debug_assert!(
        upper_src.len() >= windows.len() * block_count,
        "upper_src too short for windows: every slot strides by block_count",
    );
    for (slot, &(entry, exit)) in windows.iter().enumerate() {
        if commissioning_active(entry, exit, stage_id) {
            let base = slot * block_count;
            lower_out.extend_from_slice(&lower_src[base..base + block_count]);
            upper_out.extend_from_slice(&upper_src[base..base + block_count]);
        } else {
            for _ in 0..block_count {
                lower_out.push(0.0);
                upper_out.push(0.0);
            }
        }
    }
}

/// Patch NCS availability bounds onto this stage's dense NCS columns for one
/// solve — the single owner every solve site reaches the gather-and-set half
/// of the NCS patch through.
///
/// `scratch.ncs_col_lower_buf`/`ncs_col_upper_buf` must already hold this
/// solve's bounds (full stochastic-slot order) via a preceding
/// [`transform_ncs_noise`] call. `geometry` is this stage's own equipment
/// geometry, never a single global stage-0 geometry — per-stage block counts
/// make stage NCS bases diverge. [`gather_dense_ncs_bounds`] forces `[0, 0]`
/// for a slot dormant at this stage — the "patch NCS identically" contract
/// shared by every solve site (D15: a divergence understates the bound).
pub(crate) fn apply_ncs_col_bounds<S: SolverInterface>(
    solver: &mut S,
    scratch: &mut ScratchBuffers,
    geometry: &StageGeometry,
    dense_col: &[usize],
    windows: &[(Option<i32>, Option<i32>)],
    stage_id: i32,
) {
    let expected_len = dense_col.len() * geometry.n_blks;
    // Rebuild on the geometry's NCS start changing, not length alone: two stages
    // can share a length yet address different columns, so keying on length
    // would set bounds on the previous stage's columns.
    if scratch.last_ncs_col_start != geometry.ncs_generation.start
        || scratch.ncs_col_indices_buf.len() != expected_len
    {
        build_dense_ncs_col_indices(dense_col, geometry, &mut scratch.ncs_col_indices_buf);
        scratch.last_ncs_col_start = geometry.ncs_generation.start;
    }
    gather_dense_ncs_bounds(
        windows,
        stage_id,
        geometry.n_blks,
        &scratch.ncs_col_lower_buf,
        &scratch.ncs_col_upper_buf,
        &mut scratch.ncs_col_lower_active_buf,
        &mut scratch.ncs_col_upper_active_buf,
    );
    solver.set_col_bounds(
        &scratch.ncs_col_indices_buf,
        &scratch.ncs_col_lower_active_buf,
        &scratch.ncs_col_upper_active_buf,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::erasing_op,
    clippy::identity_op,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::entities::non_controllable::NonControllableSource;
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        LoadModel, NcsModel, SamplingScheme,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_solver::{
        Basis, RowBatch, SolutionView, SolverError, SolverInterface, SolverStatistics,
        StageTemplate,
    };
    use cobre_stochastic::StochasticContext;
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
    use std::collections::BTreeMap;

    use crate::{
        context::TrainingContext,
        horizon_mode::HorizonMode,
        indexer::StateSpace,
        inflow_method::InflowNonNegativityMethod,
        lp::builder::StageGeometry,
        noise::{
            apply_ncs_col_bounds, build_dense_ncs_col_indices, compute_effective_eta,
            gather_dense_ncs_bounds, shift_lag_state, transform_inflow_noise, transform_load_noise,
            transform_ncs_noise,
        },
        setup::node_graph::StageIdx,
        test_support,
        workspace::ScratchBuffers,
    };

    /// Records every `set_col_bounds` call verbatim; no other method is
    /// exercised by the NCS column-bound patch.
    #[derive(Default)]
    struct RecordingSolver {
        col_bounds_calls: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)>,
    }

    impl SolverInterface for RecordingSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &Self::Profile) {}

        fn solver_name_version(&self) -> String {
            "RecordingSolver 0.0.0".to_string()
        }

        fn load_model(&mut self, _template: &StageTemplate) {}

        fn add_rows(&mut self, _rows: &RowBatch) {}

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
            self.col_bounds_calls
                .push((indices.to_vec(), lower.to_vec(), upper.to_vec()));
        }

        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            unreachable!("solve() is not exercised by the NCS column-bound patch")
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

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a `ScratchBuffers` with the given pre-filled `zero_targets_buf`.
    fn make_scratch(n_hydros: usize) -> ScratchBuffers {
        ScratchBuffers {
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: vec![0.0_f64; n_hydros],
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::new(),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::with_capacity(n_hydros),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: Vec::new(),
            lag_weight_accum: Vec::new(),
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            corr_scratch: Vec::new(),
            current_node_buf: Vec::new(),
        }
    }

    /// One-hydro, one-stage `StochasticContext` with AR(0) (white noise).
    ///
    /// PAR(0): inflow = `std_m3s` * eta (no autoregressive term).
    /// With `mean_m3s = 0.0` and `std_m3s = 1.0`, inflow = eta.
    #[allow(clippy::too_many_lines)]
    fn make_one_hydro_stochastic(n_stages: usize) -> StochasticContext {
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
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

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx as i32,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

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

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();

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

    /// One-hydro, one-load-bus, n-stage `StochasticContext`.
    ///
    /// Load bus has `mean_mw` and `std_mw`, one block per stage.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_with_load(n_stages: usize, mean_mw: f64, std_mw: f64) -> StochasticContext {
        let bus0 = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let bus1 = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(10),
            name: "H10".to_string(),
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
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

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(10),
                stage_id: idx as i32,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|idx| LoadModel {
                bus_id: EntityId(1),
                stage_id: idx as i32,
                mean_mw,
                std_mw,
            })
            .collect();

        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles: BTreeMap::new(),
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(correlation)
            .build()
            .unwrap();

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

    // ── transform_inflow_noise: None method ──────────────────────────────────

    /// None method: raw eta applied directly without clamping.
    #[test]
    fn test_transform_inflow_noise_none_method() {
        let stochastic = make_one_hydro_stochastic(1);
        // State layout: 1 hydro, 0 PAR lags → n_state = 1
        let layout = test_support::state_layout(1, 0);
        let state = test_support::state_layout(1, 0);
        let current_state = vec![0.0; layout.n_state];

        // sigma = 1.0, base = 0.0 (the fixture's AR(0) white-noise model), eta = -3.0
        // expected z_inflow_rhs: 0.0 + 1.0 * (-3.0) = -3.0
        let raw_noise = vec![-3.0_f64];
        let inflow_method = InflowNonNegativityMethod::None;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let study_dims = test_support::study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
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
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            StageIdx(0),
            &current_state,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.z_inflow_rhs_buf.len(), 1);
        assert!((scratch.z_inflow_rhs_buf[0] - (-3.0)).abs() < 1e-12);
    }

    // ── transform_inflow_noise: Truncation ───────────────────────────────────

    /// Truncation: when the PAR inflow would be negative, eta is clamped.
    ///
    /// AR(0) model: inflow = sigma * eta.  With sigma=1.0 and lag=0:
    /// inflow = 1.0 * eta.  For eta = -5.0, inflow = -5.0 < 0 → clamp.
    #[test]
    fn test_transform_inflow_noise_truncation_clamps() {
        let stochastic = make_one_hydro_stochastic(1);
        // 1 hydro, 0 PAR lags
        let layout = test_support::state_layout(1, 0);
        let state = test_support::state_layout(1, 0);
        let current_state = vec![0.0; layout.n_state];

        // Very negative eta guarantees negative inflow (AR(0) with sigma=1).
        let raw_noise = vec![-5.0_f64];
        let inflow_method = InflowNonNegativityMethod::Truncation;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let study_dims = test_support::study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
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
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            StageIdx(0),
            &current_state,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.z_inflow_rhs_buf.len(), 1);
        // z_inflow_rhs = base + sigma * clamped_eta = the realized inflow (m3/s).
        // After clamping, the realized inflow must be >= 0.
        assert!(
            scratch.z_inflow_rhs_buf[0] >= -1e-10,
            "truncation must yield non-negative realized inflow, got {}",
            scratch.z_inflow_rhs_buf[0]
        );
    }

    /// Truncation passthrough: positive-inflow eta passes through unchanged.
    #[test]
    fn test_transform_inflow_noise_truncation_passthrough() {
        let stochastic = make_one_hydro_stochastic(1);
        let layout = test_support::state_layout(1, 0);
        let state = test_support::state_layout(1, 0);
        let current_state = vec![0.0; layout.n_state];

        // eta = 3.0 → inflow = 1.0 * 3.0 = 3.0 > 0 → no clamping.
        let raw_noise = vec![3.0_f64];
        let inflow_method = InflowNonNegativityMethod::Truncation;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let study_dims = test_support::study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
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
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            StageIdx(0),
            &current_state,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.z_inflow_rhs_buf.len(), 1);
        // Expected z_inflow_rhs: 0.0 + 1.0 * 3.0 = 3.0 (no clamping).
        assert!(
            (scratch.z_inflow_rhs_buf[0] - 3.0).abs() < 1e-12,
            "expected 3.0, got {}",
            scratch.z_inflow_rhs_buf[0]
        );
    }

    // ── transform_load_noise ──────────────────────────────────────────────────

    /// Basic load noise: verify RHS computation matches expected values.
    ///
    /// 1 hydro + 1 load bus.  Load bus is at noise index 1.
    /// eta = 0.0 → realization = (mean + std * 0.0).max(0.0) = mean.
    #[test]
    fn test_transform_load_noise_basic() {
        let mean_mw = 5.0_f64;
        let std_mw = 1.0_f64;
        let stochastic = make_stochastic_with_load(1, mean_mw, std_mw);

        // n_hydros=1 (hydro noise at index 0), load bus noise at index 1.
        // eta_load = 0.0 → realization = 5.0; block_factor = 1.0 → rhs = 5.0.
        let raw_noise = vec![0.0_f64, 0.0_f64]; // [hydro_eta, load_eta]
        let mut load_rhs_buf = Vec::new();

        transform_load_noise(&raw_noise, &stochastic, StageIdx(0), 1, &mut load_rhs_buf);

        assert_eq!(load_rhs_buf.len(), 1);
        // The block_factor for a single Parallel block is the block duration
        // divided by total stage hours; with one block it equals 1.0.
        // Expected: 5.0 * 1.0 = 5.0.
        assert!(
            (load_rhs_buf[0] - 5.0).abs() < 1e-10,
            "expected 5.0, got {}",
            load_rhs_buf[0]
        );
    }

    /// Negative realizations are clamped to zero.
    ///
    /// Very negative eta drives `mean + std * eta` below zero; must be clamped.
    #[test]
    fn test_transform_load_noise_clamped_non_negative() {
        let mean_mw = 2.0_f64;
        let std_mw = 1.0_f64;
        let stochastic = make_stochastic_with_load(1, mean_mw, std_mw);

        // eta_load = -10.0 → realization = (2.0 - 10.0).max(0.0) = 0.0.
        let raw_noise = vec![0.0_f64, -10.0_f64];
        let mut load_rhs_buf = Vec::new();

        transform_load_noise(&raw_noise, &stochastic, StageIdx(0), 1, &mut load_rhs_buf);

        assert_eq!(load_rhs_buf.len(), 1);
        assert!(
            load_rhs_buf[0].abs() < 1e-12,
            "expected 0.0, got {}",
            load_rhs_buf[0]
        );
    }

    // ── shift_lag_state tests ────────────────────────────────────────────────

    #[test]
    fn shift_lag_state_par0_is_noop() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(2, 0);
        let mut state = vec![100.0, 200.0]; // storage only, no lags
        let incoming_lags: Vec<f64> = vec![];
        let primal = vec![0.0; 10];
        shift_lag_state(&mut state, &incoming_lags, &primal, &layout);
        assert_eq!(
            state,
            vec![100.0, 200.0],
            "state must be unchanged for PAR(0)"
        );
    }

    #[test]
    fn shift_lag_state_par1_single_hydro() {
        // N=1, L=1: state = [v_out, lag0], inflow_lags.start = 1
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);
        let mut state = vec![500.0, 99.0]; // v_out, stale lag
        let incoming_lags = vec![42.0]; // lag0 (lag-major: lag * n_h + h = 0*1+0 = 0)
        // z_inflow.start = N*(1+L) = 1*(1+1) = 2
        let mut primal = vec![0.0; 10];
        primal[layout.z_inflow.start] = 77.0; // Z_t for hydro 0
        shift_lag_state(&mut state, &incoming_lags, &primal, &layout);
        assert_eq!(state[1], 77.0, "lag[0] must be Z_t = 77.0");
    }

    #[test]
    fn shift_lag_state_par3_single_hydro() {
        // N=1, L=3: state = [v_out, lag0, lag1, lag2]
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 3);
        let mut state = vec![500.0, 0.0, 0.0, 0.0];
        // incoming_lags in lag-major: [lag0, lag1, lag2] = [10.0, 20.0, 30.0]
        let incoming_lags = vec![10.0, 20.0, 30.0];
        let mut primal = vec![0.0; 20];
        primal[layout.z_inflow.start] = 55.0;
        shift_lag_state(&mut state, &incoming_lags, &primal, &layout);
        // After shift: lag[0]=Z_t=55, lag[1]=incoming[0]=10, lag[2]=incoming[1]=20
        assert_eq!(state[1], 55.0, "lag[0] must be Z_t");
        assert_eq!(state[2], 10.0, "lag[1] must be incoming lag[0]");
        assert_eq!(state[3], 20.0, "lag[2] must be incoming lag[1]");
    }

    #[test]
    fn shift_lag_state_par1_two_hydros() {
        // N=2, L=1: state = [v0, v1, lag0_h0, lag0_h1]
        // inflow_lags.start = 2, lag-major: lag0 * 2 + 0 = 0, lag0 * 2 + 1 = 1
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(2, 1);
        let mut state = vec![100.0, 200.0, 0.0, 0.0];
        let incoming_lags = vec![10.0, 20.0]; // lag0_h0=10, lag0_h1=20
        let mut primal = vec![0.0; 20];
        primal[layout.z_inflow.start] = 33.0; // Z_t for hydro 0
        primal[layout.z_inflow.start + 1] = 44.0; // Z_t for hydro 1
        shift_lag_state(&mut state, &incoming_lags, &primal, &layout);
        assert_eq!(state[2], 33.0, "lag[0] for h0 must be Z_t_h0");
        assert_eq!(state[3], 44.0, "lag[0] for h1 must be Z_t_h1");
    }

    #[test]
    fn shift_lag_state_preserves_storage() {
        // Verify storage portion [0..N] is unchanged after shift.
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(2, 2);
        let mut state = vec![100.0, 200.0, 0.0, 0.0, 0.0, 0.0];
        let incoming_lags = vec![1.0, 2.0, 3.0, 4.0];
        let mut primal = vec![0.0; 20];
        primal[layout.z_inflow.start] = 50.0;
        primal[layout.z_inflow.start + 1] = 60.0;
        shift_lag_state(&mut state, &incoming_lags, &primal, &layout);
        assert_eq!(state[0], 100.0, "storage[0] must be preserved");
        assert_eq!(state[1], 200.0, "storage[1] must be preserved");
    }

    // ── compute_effective_eta tests ─────────────────────────────────────────

    #[test]
    fn test_compute_effective_eta_none_passes_through() {
        let raw_noise = [0.5, -1.0];
        let par_inflows = []; // unused for None
        let eta_floor = []; // unused for None
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            InflowNonNegativityMethod::None,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![0.5, -1.0]);
    }

    #[test]
    fn test_compute_effective_eta_penalty_passes_through() {
        let raw_noise = [0.5, -1.0];
        let par_inflows = [];
        let eta_floor = [];
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            InflowNonNegativityMethod::Penalty,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![0.5, -1.0]);
    }

    #[test]
    fn test_compute_effective_eta_truncation_clamps_negative() {
        // 2 hydros: par_inflows[0] < 0 -> clamp eta[0]; par_inflows[1] > 0 -> pass through.
        let raw_noise = [-2.0, 1.0];
        let par_inflows = [-5.0, 3.0];
        let eta_floor = [-1.0, -0.5]; // floor for hydro 0 is -1.0
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            InflowNonNegativityMethod::Truncation,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        // hydro 0: eta=-2.0, floor=-1.0 -> max(-2, -1) = -1.0
        // hydro 1: par_inflows[1]=3.0 >= 0 -> no clamp -> eta=1.0
        assert_eq!(effective, vec![-1.0, 1.0]);
    }

    #[test]
    fn test_compute_effective_eta_truncation_passes_positive() {
        // All PAR inflows positive -> no clamping at all.
        let raw_noise = [-2.0, 1.0];
        let par_inflows = [3.0, 5.0];
        let eta_floor = [-1.0, -0.5]; // floors are irrelevant when no negative inflow
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            InflowNonNegativityMethod::Truncation,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![-2.0, 1.0]);
    }

    #[test]
    fn test_compute_effective_eta_truncation_with_penalty_clamps() {
        // TruncationWithPenalty behaves the same as Truncation for clamping.
        let raw_noise = [-2.0, 1.0];
        let par_inflows = [-5.0, 3.0];
        let eta_floor = [-1.0, -0.5];
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            InflowNonNegativityMethod::TruncationWithPenalty,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![-1.0, 1.0]);
    }

    // ── accumulate_and_shift_lag_state tests ─────────────────────────────────

    use cobre_core::temporal::StageLagTransition;

    use crate::noise::{DownstreamAccumState, LagAccumState, accumulate_and_shift_lag_state};
    // Convenience helper: build a no-op DownstreamAccumState for tests that
    // exercise only primary accumulation (uniform-resolution path).
    fn noop_ds<'a>(
        accumulator: &'a mut Vec<f64>,
        weight_accum: &'a mut f64,
        completed_lags: &'a mut Vec<f64>,
        n_completed: &'a mut usize,
    ) -> DownstreamAccumState<'a> {
        DownstreamAccumState {
            accumulator: accumulator.as_mut_slice(),
            weight_accum,
            completed_lags: completed_lags.as_mut_slice(),
            n_completed,
            par_order: 0,
        }
    }

    fn identity_stage_lag() -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        }
    }

    fn rebuild_stage_lag() -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: true,
        }
    }

    /// Monthly identity: `accumulate_weight=1.0`, `spillover_weight=0.0`, `finalize_period=true`.
    ///
    /// With a single finalization stage the result must be bit-for-bit
    /// identical to `shift_lag_state`.
    #[test]
    fn test_accumulate_monthly_identity() {
        // N=1 hydro, L=1 lag order.
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);

        // Reference: shift_lag_state result.
        let mut state_ref = vec![500.0, 99.0];
        let incoming_lags = vec![42.0];
        let mut primal = vec![0.0; 10];
        primal[layout.z_inflow.start] = 77.0;
        shift_lag_state(&mut state_ref, &incoming_lags, &primal, &layout);

        // Accumulate: single stage with identity weights.
        let mut state_acc = vec![500.0, 99.0];
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];
        let stage_lag = identity_stage_lag();
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state_acc,
            &incoming_lags,
            &primal,
            &layout,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );

        assert_eq!(
            state_acc, state_ref,
            "monthly identity must produce identical result to shift_lag_state"
        );
        // Accumulator must be zeroed (clean for next period).
        assert_eq!(lag_accumulator[0], 0.0);
        assert_eq!(lag_weight_accum[0], 0.0);
    }

    /// Four weekly stages each contributing weight=0.25, finalize only on stage 3.
    ///
    /// After processing all four stages the lag[0] must equal the weighted
    /// average: (500 + 480 + 520 + 510) / 4 = 502.5.
    #[test]
    fn test_accumulate_four_weeks_then_finalize() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);
        let mut state = vec![500.0, 0.0]; // storage, lag0
        let incoming_lags = vec![0.0]; // lag-major: lag0 for hydro 0
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];

        let z_inflows = [500.0_f64, 480.0, 520.0, 510.0];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;

        for (week, &z) in z_inflows.iter().enumerate() {
            let finalize = week == 3;
            let stage_lag = StageLagTransition {
                accumulate_weight: 0.25,
                spillover_weight: 0.0,
                finalize_period: finalize,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };
            let mut primal = vec![0.0; 10];
            primal[layout.z_inflow.start] = z;
            accumulate_and_shift_lag_state(
                &mut state,
                &incoming_lags,
                &primal,
                &layout,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_accumulator,
                    weight_accum: &mut lag_weight_accum,
                },
                &mut noop_ds(
                    &mut ds_accum,
                    &mut ds_weight,
                    &mut ds_completed,
                    &mut ds_n_completed,
                ),
            );
        }

        // lag[0] is at inflow_lags.start = 1 (state index).
        let expected = (500.0 + 480.0 + 520.0 + 510.0) / 4.0;
        assert!(
            (state[layout.inflow_lags.start] - expected).abs() < 1e-12,
            "lag[0] must equal weighted average {expected}, got {}",
            state[layout.inflow_lags.start]
        );
        // Accumulator reset after finalization.
        assert_eq!(lag_accumulator[0], 0.0);
        assert_eq!(lag_weight_accum[0], 0.0);
    }

    /// Spillover seeds the next lag period with raw `z_inflow` * `spillover_weight`.
    #[test]
    fn test_accumulate_spillover_seeds_next_period() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);
        let mut state = vec![0.0, 0.0];
        let incoming_lags = vec![0.0];
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];
        let mut primal = vec![0.0; 10];
        primal[layout.z_inflow.start] = 200.0;

        let stage_lag = StageLagTransition {
            accumulate_weight: 0.968, // 1.0 - 0.032 = days in period / days in month
            spillover_weight: 0.032,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        };
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state,
            &incoming_lags,
            &primal,
            &layout,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );

        // After finalization, accumulator seeded with raw z_inflow * spillover_weight.
        let expected_seed = 200.0 * 0.032;
        assert!(
            (lag_accumulator[0] - expected_seed).abs() < 1e-12,
            "accumulator must be seeded with z_inflow * spillover_weight = {expected_seed}, got {}",
            lag_accumulator[0]
        );
        assert!(
            (lag_weight_accum[0] - 0.032).abs() < 1e-12,
            "lag_weight_accum must equal spillover_weight = 0.032, got {}",
            lag_weight_accum[0]
        );
    }

    /// `max_par_order == 0`: function must return immediately, nothing modified.
    #[test]
    fn test_accumulate_noop_for_par0() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0); // no lag order
        let layout = test_support::state_layout(2, 0);
        let mut state = vec![100.0, 200.0];
        let incoming_lags: Vec<f64> = vec![];
        let primal = vec![0.0; 10];
        let mut lag_accumulator: Vec<f64> = vec![]; // empty — should never be accessed
        let mut lag_weight_accum: Vec<f64> = vec![]; // empty — should never be accessed
        let stage_lag = identity_stage_lag();
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state,
            &incoming_lags,
            &primal,
            &layout,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );
        assert_eq!(
            state,
            vec![100.0, 200.0],
            "state must be unchanged for PAR(0)"
        );
        assert_eq!(
            lag_weight_accum,
            Vec::<f64>::new(),
            "weight must be unchanged for PAR(0)"
        );
    }

    /// Storage region of state (indices 0..N) must not be touched by the shift.
    #[test]
    fn test_accumulate_preserves_storage() {
        // N=2 hydros, L=2 lag order: state = [v0, v1, lag0_h0, lag0_h1, lag1_h0, lag1_h1]
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(2, 2);
        let mut state = vec![100.0, 200.0, 0.0, 0.0, 0.0, 0.0];
        let incoming_lags = vec![1.0, 2.0, 3.0, 4.0]; // lag-major: lag0 h0,h1; lag1 h0,h1
        let mut primal = vec![0.0; 20];
        primal[layout.z_inflow.start] = 50.0;
        primal[layout.z_inflow.start + 1] = 60.0;
        let mut lag_accumulator = vec![0.0_f64; 2];
        let mut lag_weight_accum = vec![0.0_f64; 2];
        let stage_lag = identity_stage_lag();
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state,
            &incoming_lags,
            &primal,
            &layout,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );
        assert_eq!(state[0], 100.0, "storage[0] must be preserved");
        assert_eq!(state[1], 200.0, "storage[1] must be preserved");
    }

    // ── downstream accumulation tests ────────────────────────────────────────

    /// Build a `StageLagTransition` for a standard monthly stage that also
    /// accumulates into the downstream (quarterly) ring buffer.
    ///
    /// weight = 1/3 per month (3 months per quarter, no spillover).
    fn monthly_with_downstream(
        finalize_primary: bool,
        downstream_finalize: bool,
    ) -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0 / 3.0,
            spillover_weight: 0.0,
            finalize_period: finalize_primary,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize,
            rebuild_from_downstream: false,
        }
    }

    /// Drive one stage through `accumulate_and_shift_lag_state` with full
    /// downstream buffers.
    #[allow(clippy::too_many_arguments)]
    fn run_stage(
        state: &mut [f64],
        incoming_lags: &[f64],
        z_inflow: f64,
        layout: &StateSpace,
        stage_lag: &StageLagTransition,
        lag: &mut LagAccumState<'_>,
        ds: &mut DownstreamAccumState<'_>,
    ) {
        let mut primal = vec![0.0; layout.z_inflow.start + layout.hydro_count + 4];
        primal[layout.z_inflow.start] = z_inflow;
        accumulate_and_shift_lag_state(state, incoming_lags, &primal, layout, stage_lag, lag, ds);
    }

    /// Drive one stage with two hydros.
    fn run_stage_2h(
        state: &mut [f64],
        incoming_lags: &[f64],
        z_inflows: [f64; 2],
        layout: &StateSpace,
        stage_lag: &StageLagTransition,
        lag: &mut LagAccumState<'_>,
        ds: &mut DownstreamAccumState<'_>,
    ) {
        let n = layout.z_inflow.start + layout.hydro_count + 4;
        let mut primal = vec![0.0; n];
        primal[layout.z_inflow.start] = z_inflows[0];
        primal[layout.z_inflow.start + 1] = z_inflows[1];
        accumulate_and_shift_lag_state(state, incoming_lags, &primal, layout, stage_lag, lag, ds);
    }

    /// Test 1: PAR(1) downstream accumulation with a 3-stage quarterly window.
    ///
    /// 3 monthly stages (each weight=1/3, no primary finalize, no primary
    /// spillover, `downstream_finalize` on last month) populate the downstream
    /// ring buffer.  After all 3 stages, `downstream_completed_lags[0]` must
    /// equal `(90.0 + 100.0 + 110.0) / 3.0 = 100.0`.  Then calling with
    /// `rebuild_from_downstream = true` on the first quarterly stage overwrites
    /// `state[lag_start]` with `100.0`.
    #[test]
    fn test_downstream_par1_accumulation_and_rebuild() {
        // N=1 hydro, L=1 lag (primary monthly PAR(1) order).
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);
        let lag_start = layout.inflow_lags.start;

        // Primary state: storage=500, lag0=old_value_to_be_replaced.
        let mut state = vec![500.0, 42.0];
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        // downstream: par_order=1, ring buf capacity n_h * 1 = 1.
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 1];
        let mut ds_n = 0_usize;

        let z_vals = [90.0_f64, 100.0, 110.0];
        for (i, &z) in z_vals.iter().enumerate() {
            let ds_finalize = i == 2; // last month of the quarter
            let stage_lag = monthly_with_downstream(false, ds_finalize);
            run_stage(
                &mut state,
                &incoming_lags,
                z,
                &layout,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 1,
                },
            );
        }

        // After 3 monthly stages the ring buffer should hold the quarterly average.
        let expected_avg = (90.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (ds_completed[0] - expected_avg).abs() < 1e-12,
            "ring buf slot 0 should be {expected_avg}, got {}",
            ds_completed[0]
        );
        assert_eq!(ds_n, 1, "n_completed must be 1 after one quarter");

        // Now simulate the transition stage (first quarterly stage).
        // rebuild_from_downstream=true; primary accumulation is quarterly.
        let rebuild_lag = rebuild_stage_lag();
        run_stage(
            &mut state,
            &incoming_lags,
            999.0, // z_inflow irrelevant — rebuild returns before primary finalize
            &layout,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // lag[0] must be rebuilt to the quarterly average.
        assert!(
            (state[lag_start] - expected_avg).abs() < 1e-12,
            "state[lag_start] must be rebuilt to {expected_avg}, got {}",
            state[lag_start]
        );
        // Storage must be untouched.
        assert_eq!(state[0], 500.0, "storage must be untouched during rebuild");
        // Downstream state must be fully reset.
        assert_eq!(ds_n, 0, "n_completed must reset to 0 after rebuild");
        assert_eq!(
            ds_completed[0], 0.0,
            "completed_lags must be zeroed after rebuild"
        );
        assert_eq!(ds_w, 0.0, "downstream weight must reset after rebuild");
    }

    /// Test 2: PAR(2) downstream accumulation with two consecutive quarters.
    ///
    /// 6 monthly stages (Q3: stages 0-2, Q4: stages 3-5) with `downstream_par_order=2`.
    /// After Q3, `completed_lags[slot=0] == avg(60,70,80) == 70.0`, `n_completed==1`.
    /// After Q4, `completed_lags[slot=1] == avg(90,100,110) == 100.0`, `n_completed==2`.
    /// At rebuild: `state[lag_start] == 100.0` (newest Q4), `state[lag_start+1] == 70.0` (Q3).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_downstream_par2_two_quarters() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0); // L=2 lag order
        let layout = test_support::state_layout(1, 2);
        let lag_start = layout.inflow_lags.start;

        let mut state = vec![0.0; 1 + 2]; // storage + lag0 + lag1
        let incoming_lags = vec![0.0, 0.0]; // lag-major: lag0 h0, lag1 h0
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        // par_order=2: ring buf capacity n_h * 2 = 2.
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 2];
        let mut ds_n = 0_usize;

        // Q3: z_inflows 60, 70, 80 — no primary finalize, downstream finalize on month 3.
        let q3_vals = [60.0_f64, 70.0, 80.0];
        for (i, &z) in q3_vals.iter().enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            run_stage(
                &mut state,
                &incoming_lags,
                z,
                &layout,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 2,
                },
            );
        }
        let q3_avg = (60.0 + 70.0 + 80.0) / 3.0;
        assert!(
            (ds_completed[0] - q3_avg).abs() < 1e-12,
            "slot 0 should be Q3 avg {q3_avg}, got {}",
            ds_completed[0]
        );
        assert_eq!(ds_n, 1);

        // Q4: z_inflows 90, 100, 110 — downstream finalize on month 3.
        let q4_vals = [90.0_f64, 100.0, 110.0];
        for (i, &z) in q4_vals.iter().enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            run_stage(
                &mut state,
                &incoming_lags,
                z,
                &layout,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 2,
                },
            );
        }
        let q4_avg = (90.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (ds_completed[1] - q4_avg).abs() < 1e-12,
            "slot 1 should be Q4 avg {q4_avg}, got {}",
            ds_completed[1]
        );
        assert_eq!(ds_n, 2);

        // Rebuild stage: lag[0] <- newest (Q4), lag[1] <- second-newest (Q3).
        let rebuild_lag = rebuild_stage_lag();
        run_stage(
            &mut state,
            &incoming_lags,
            999.0,
            &layout,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 2,
            },
        );

        // lag[0] = newest = Q4 avg; lag[1] = Q3 avg.
        assert!(
            (state[lag_start] - q4_avg).abs() < 1e-12,
            "lag[0] should be newest Q4 avg {q4_avg}, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - q3_avg).abs() < 1e-12,
            "lag[1] should be Q3 avg {q3_avg}, got {}",
            state[lag_start + 1]
        );
    }

    /// Test 3: Uniform monthly study — empty downstream buffers, zero overhead.
    ///
    /// Calls `accumulate_and_shift_lag_state` with `downstream_accumulator = &mut []`.
    /// The function must produce exactly the same result,
    /// with no downstream fields accessed.
    #[test]
    fn test_no_downstream_for_uniform_monthly() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);
        let mut state_ds = vec![500.0, 0.0]; // with empty downstream
        let mut state_ref = vec![500.0, 0.0]; // with noop downstream
        let incoming_lags = vec![0.0];
        let z_inflows = [100.0_f64, 110.0, 120.0];

        let mut lag_acc_ref = vec![0.0_f64; 1];
        let mut lag_w_ref = vec![0.0_f64; 1];
        let mut lag_acc_ds = vec![0.0_f64; 1];
        let mut lag_w_ds = vec![0.0_f64; 1];

        for (i, &z) in z_inflows.iter().enumerate() {
            let finalize = i == 2;
            let stage_lag = StageLagTransition {
                accumulate_weight: 1.0 / 3.0,
                spillover_weight: 0.0,
                finalize_period: finalize,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };

            // Reference: empty downstream (noop path).
            let mut ds_accum_ref: Vec<f64> = vec![];
            let mut ds_weight_ref = 0.0_f64;
            let mut ds_completed_ref: Vec<f64> = vec![];
            let mut ds_n_completed_ref = 0_usize;
            run_stage(
                &mut state_ref,
                &incoming_lags,
                z,
                &layout,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc_ref,
                    weight_accum: &mut lag_w_ref,
                },
                &mut noop_ds(
                    &mut ds_accum_ref,
                    &mut ds_weight_ref,
                    &mut ds_completed_ref,
                    &mut ds_n_completed_ref,
                ),
            );

            // Test: inline empty downstream (par_order=0).
            let mut ds_accum_ds: Vec<f64> = vec![];
            let mut ds_weight_ds = 0.0_f64;
            let mut ds_completed_ds: Vec<f64> = vec![];
            let mut ds_n_completed_ds = 0_usize;
            run_stage(
                &mut state_ds,
                &incoming_lags,
                z,
                &layout,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc_ds,
                    weight_accum: &mut lag_w_ds,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_accum_ds,
                    weight_accum: &mut ds_weight_ds,
                    completed_lags: &mut ds_completed_ds,
                    n_completed: &mut ds_n_completed_ds,
                    par_order: 0,
                },
            );
        }

        assert_eq!(
            state_ds, state_ref,
            "uniform monthly study must be identical with or without downstream buffers"
        );
    }

    /// Test 4: `rebuild_from_downstream` resets all downstream state.
    ///
    /// After rebuild, `n_completed == 0`, `completed_lags` all zero, and
    /// `downstream_weight_accum == 0.0`.
    #[test]
    fn test_rebuild_resets_downstream_state() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);
        let mut state = vec![0.0, 0.0];
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.5_f64; // non-zero before rebuild
        let mut ds_completed = vec![77.0_f64; 1]; // non-zero before rebuild
        let mut ds_n = 1_usize; // pretend one quarter was completed

        let rebuild_lag = rebuild_stage_lag();

        run_stage(
            &mut state,
            &incoming_lags,
            0.0,
            &layout,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        assert_eq!(ds_n, 0, "n_completed must reset to 0 after rebuild");
        assert_eq!(
            ds_completed[0], 0.0,
            "completed_lags must be zeroed after rebuild"
        );
        assert_eq!(
            ds_w, 0.0,
            "downstream weight_accum must reset after rebuild"
        );
        assert_eq!(
            ds_acc[0], 0.0,
            "downstream accumulator must be zeroed after rebuild"
        );
    }

    /// Test 5: Downstream spillover seeds the next quarterly accumulation.
    ///
    /// A monthly stage with `downstream_spillover_weight = 0.1` and
    /// `downstream_finalize = true` should: (a) finalize the current quarter,
    /// (b) seed the next quarter's accumulator with `z_inflow * 0.1`.
    #[test]
    fn test_downstream_spillover_seeds_next_quarter() {
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(1, 1);
        let mut state = vec![0.0, 0.0];
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_completed = vec![0.0_f64; 1];
        let mut ds_n = 0_usize;

        // Single monthly stage that finalizes the quarter with spillover.
        // Pre-load the accumulator to simulate prior months already accumulated.
        ds_acc[0] = 200.0; // months 1+2 already accumulated
        let mut ds_w = 2.0 / 3.0_f64; // two months of weight 1/3 each

        let spillover_weight = 0.1;
        let stage_lag = StageLagTransition {
            accumulate_weight: 1.0 / 3.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: spillover_weight,
            downstream_finalize: true,
            rebuild_from_downstream: false,
        };

        let z = 120.0_f64;
        run_stage(
            &mut state,
            &incoming_lags,
            z,
            &layout,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // Quarter should be finalized and the ring buffer filled.
        assert_eq!(ds_n, 1, "one quarter must be finalized");
        // Downstream accumulator must be seeded with z * spillover_weight.
        let expected_seed = z * spillover_weight;
        assert!(
            (ds_acc[0] - expected_seed).abs() < 1e-12,
            "accumulator should be seeded with {expected_seed}, got {}",
            ds_acc[0]
        );
        assert!(
            (ds_w - spillover_weight).abs() < 1e-12,
            "weight_accum should be {spillover_weight}, got {ds_w}"
        );
    }

    /// Test 6: Multi-hydro downstream — 2 hydros, PAR(1).
    ///
    /// Each hydro has its own `z_inflow` values.  After 3 monthly stages,
    /// `downstream_completed_lags[0]` (hydro 0) and `[1]` (hydro 1) must
    /// each equal the independently computed quarterly average for that hydro.
    #[test]
    fn test_downstream_multi_hydro() {
        // N=2 hydros, L=1 lag order.
        let _indexer = test_support::equipment_free_geometry(&[0]).remove(0);
        let layout = test_support::state_layout(2, 1);
        let lag_start = layout.inflow_lags.start;

        let mut state = vec![0.0; 2 + 2]; // 2 storage + 2 lag entries (lag0 h0, lag0 h1)
        let incoming_lags = vec![0.0, 0.0]; // lag-major: lag0 h0, lag0 h1
        let mut lag_acc = vec![0.0_f64; 2];
        let mut lag_w = vec![0.0_f64; 2];
        // ring buf capacity: n_h * par_order = 2 * 1 = 2
        let mut ds_acc = vec![0.0_f64; 2];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 2];
        let mut ds_n = 0_usize;

        // 3 monthly stages: hydro 0 inflows [10, 20, 30], hydro 1 inflows [40, 50, 60].
        let h0_vals = [10.0_f64, 20.0, 30.0];
        let h1_vals = [40.0_f64, 50.0, 60.0];

        for (i, (&z0, &z1)) in h0_vals.iter().zip(h1_vals.iter()).enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            run_stage_2h(
                &mut state,
                &incoming_lags,
                [z0, z1],
                &layout,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 1,
                },
            );
        }

        let expected_h0 = (10.0 + 20.0 + 30.0) / 3.0;
        let expected_h1 = (40.0 + 50.0 + 60.0) / 3.0;
        assert!(
            (ds_completed[0] - expected_h0).abs() < 1e-12,
            "hydro 0 quarterly avg should be {expected_h0}, got {}",
            ds_completed[0]
        );
        assert!(
            (ds_completed[1] - expected_h1).abs() < 1e-12,
            "hydro 1 quarterly avg should be {expected_h1}, got {}",
            ds_completed[1]
        );
        assert_eq!(ds_n, 1);

        // Rebuild: both hydros rebuilt independently.
        let rebuild_lag = rebuild_stage_lag();
        run_stage_2h(
            &mut state,
            &incoming_lags,
            [999.0, 888.0],
            &layout,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // lag[0] for hydro 0 and hydro 1 must be rebuilt independently.
        assert!(
            (state[lag_start] - expected_h0).abs() < 1e-12,
            "rebuilt lag[0] hydro 0 should be {expected_h0}, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - expected_h1).abs() < 1e-12,
            "rebuilt lag[0] hydro 1 should be {expected_h1}, got {}",
            state[lag_start + 1]
        );
    }

    // ── dense NCS column/bound mapping ───────────────────────────────────────

    /// Dormant stage: slot 0's window excludes the stage, slots 1,2 windowless.
    /// Under the dense layout every slot keeps a column, so the index buffer
    /// strides all three by their dense column position (here slot == dense
    /// column). The gather computes dormancy inline from the windows and the
    /// stage id, copying the active slots' bounds verbatim and forcing `[0, 0]`
    /// for the dormant slot — the buffers are full length `n_stochastic_ncs * n_blks`.
    #[test]
    fn dense_ncs_dormant_slot_is_zeroed() {
        let n_blks = 2_usize;
        let ncs_start = 100_usize;
        let n_ncs = 3_usize;
        let dense_col = vec![0_usize, 1, 2];
        let stage_id = 0_i32;
        // slot 0 enters at stage 1 (dormant at stage 0); slots 1,2 windowless.
        let windows = vec![(Some(1_i32), None), (None, None), (None, None)];
        let geometry = StageGeometry {
            ncs_generation: ncs_start..ncs_start + n_ncs * n_blks,
            n_blks,
            ..test_support::equipment_free_geometry(&[n_blks]).remove(0)
        };

        let mut indices = Vec::new();
        build_dense_ncs_col_indices(&dense_col, &geometry, &mut indices);
        // Every slot contributes a block: 100,101 | 102,103 | 104,105.
        assert_eq!(indices, vec![100, 101, 102, 103, 104, 105]);
        assert_eq!(indices.len(), dense_col.len() * n_blks);

        let lower_src = vec![0.0, 0.0, 10.0, 11.0, 20.0, 21.0];
        let upper_src = vec![1.0, 2.0, 13.0, 14.0, 23.0, 24.0];
        let mut lower_out = Vec::new();
        let mut upper_out = Vec::new();
        gather_dense_ncs_bounds(
            &windows,
            stage_id,
            n_blks,
            &lower_src,
            &upper_src,
            &mut lower_out,
            &mut upper_out,
        );
        // Slot 0 forced to [0,0]; slots 1,2 copied verbatim.
        assert_eq!(lower_out, vec![0.0, 0.0, 10.0, 11.0, 20.0, 21.0]);
        assert_eq!(upper_out, vec![0.0, 0.0, 13.0, 14.0, 23.0, 24.0]);
        assert_eq!(upper_out.len(), indices.len(), "bounds parallel to indices");
    }

    /// No-window case: every slot windowless (active at every stage) and slot
    /// order == dense column order — the gathered bounds and built indices are
    /// bit-identical to a verbatim full `n_stochastic_ncs` block, the
    /// hash-neutrality contract for every existing case.
    #[test]
    fn dense_ncs_no_dormancy_is_slot_order_identical() {
        let n_blks = 2_usize;
        let ncs_start = 0_usize;
        let n_ncs = 3_usize;
        let dense_col = vec![0_usize, 1, 2];
        let windows = vec![(None, None), (None, None), (None, None)];
        let geometry = StageGeometry {
            ncs_generation: ncs_start..ncs_start + n_ncs * n_blks,
            n_blks,
            ..test_support::equipment_free_geometry(&[n_blks]).remove(0)
        };

        let mut indices = Vec::new();
        build_dense_ncs_col_indices(&dense_col, &geometry, &mut indices);
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);

        let lower_src = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let upper_src = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut lower_out = Vec::new();
        let mut upper_out = Vec::new();
        gather_dense_ncs_bounds(
            &windows,
            7,
            n_blks,
            &lower_src,
            &upper_src,
            &mut lower_out,
            &mut upper_out,
        );
        // Verbatim copy of the full slot-order buffers.
        assert_eq!(lower_out, lower_src);
        assert_eq!(upper_out, upper_src);
    }

    /// Two stages with **equal** length but **different** `ncs_col_start` must
    /// rebuild the index buffer on the second stage. This pins the invariant: the
    /// patch-site guard keys the lazy rebuild on `(last_ncs_col_start, len)`, not
    /// on `len` alone. A length-only guard (the forbidden alternative) would not
    /// fire here — both stages have length `2 * 2 = 4` — and would leave the
    /// previous stage's indices in place, writing `set_col_bounds` onto the wrong
    /// LP columns.
    #[test]
    fn index_buffer_rebuilds_on_ncs_col_start_change_at_equal_length() {
        let n_blks = 2_usize;
        // Same dense column set / same length at both stages: 2 slots × 2 blocks.
        let dense_col = vec![0_usize, 1];
        let expected_len = dense_col.len() * n_blks;
        assert_eq!(expected_len, 4);

        let geometry_at = |ncs_start: usize| StageGeometry {
            ncs_generation: ncs_start..ncs_start + dense_col.len() * n_blks,
            n_blks,
            ..test_support::equipment_free_geometry(&[n_blks]).remove(0)
        };

        // Reproduce the patch-site guard verbatim: rebuild iff the stored start
        // differs OR the buffer length differs.
        let mut indices_buf: Vec<usize> = Vec::new();
        let mut last_ncs_col_start = usize::MAX;
        let rebuild = |geometry: &StageGeometry, buf: &mut Vec<usize>, last: &mut usize| {
            if *last != geometry.ncs_generation.start || buf.len() != expected_len {
                build_dense_ncs_col_indices(&dense_col, geometry, buf);
                *last = geometry.ncs_generation.start;
                true
            } else {
                false
            }
        };

        // Stage A at start 100: first call always rebuilds (last == usize::MAX).
        assert!(rebuild(
            &geometry_at(100),
            &mut indices_buf,
            &mut last_ncs_col_start
        ));
        assert_eq!(indices_buf, vec![100, 101, 102, 103]);
        assert_eq!(last_ncs_col_start, 100);

        // Stage B at start 200, SAME length (4): the start-tracking guard fires and
        // the buffer tracks the new base. A length-only guard would have skipped
        // this rebuild and left [100,101,102,103] — the latent bug.
        assert!(rebuild(
            &geometry_at(200),
            &mut indices_buf,
            &mut last_ncs_col_start
        ));
        assert_eq!(indices_buf, vec![200, 201, 202, 203]);
        assert_eq!(last_ncs_col_start, 200);

        // Re-entering stage B (same start, same length): no rebuild.
        assert!(!rebuild(
            &geometry_at(200),
            &mut indices_buf,
            &mut last_ncs_col_start
        ));
        assert_eq!(indices_buf, vec![200, 201, 202, 203]);
    }

    // ── apply_ncs_col_bounds: collapsed-function equivalence ────────────────

    /// One bus, one NCS entity, availability factor `mean=0.5, std=0.1` — the
    /// D15 NCS fixture shape (`lb_evaluate_stage_0_patches_ncs_bounds_per_opening`
    /// in `training/lower_bound.rs`).
    // Rationale: the inline System/StochasticContext fixture and the reference
    // vs. owner comparison are one coherent scenario; splitting them into
    // helpers would scatter the setup the assertions depend on and obscure the
    // test.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn apply_ncs_col_bounds_matches_pre_collapse_gather_and_set() {
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
            .stages(vec![stage])
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

        let raw_noise = vec![0.37_f64];
        let ncs_max_gen = vec![100.0_f64];
        let ncs_allow_curtailment = vec![true];
        let dense_col = vec![0_usize];
        let windows: Vec<(Option<i32>, Option<i32>)> = vec![(None, None)];
        let ncs_start = 5_usize;
        let stage_id = 0_i32;
        let n_blks = 1_usize;
        let geometry = StageGeometry {
            ncs_generation: ncs_start..ncs_start + n_blks,
            n_blks,
            ..test_support::equipment_free_geometry(&[n_blks]).remove(0)
        };

        // ---- reference: transform, then the pre-collapse gather+set called
        // directly (independent of `apply_ncs_col_bounds`) ----
        let mut reference_scratch = make_scratch(0);
        transform_ncs_noise(
            &raw_noise,
            &stoch,
            StageIdx(0),
            n_blks,
            &ncs_max_gen,
            &ncs_allow_curtailment,
            &mut reference_scratch.ncs_col_lower_buf,
            &mut reference_scratch.ncs_col_upper_buf,
        );
        build_dense_ncs_col_indices(
            &dense_col,
            &geometry,
            &mut reference_scratch.ncs_col_indices_buf,
        );
        gather_dense_ncs_bounds(
            &windows,
            stage_id,
            n_blks,
            &reference_scratch.ncs_col_lower_buf,
            &reference_scratch.ncs_col_upper_buf,
            &mut reference_scratch.ncs_col_lower_active_buf,
            &mut reference_scratch.ncs_col_upper_active_buf,
        );
        let mut reference_solver = RecordingSolver::default();
        reference_solver.set_col_bounds(
            &reference_scratch.ncs_col_indices_buf,
            &reference_scratch.ncs_col_lower_active_buf,
            &reference_scratch.ncs_col_upper_active_buf,
        );

        // ---- owner: transform (unchanged), then the collapsed
        // `apply_ncs_col_bounds` ----
        let mut owner_scratch = make_scratch(0);
        transform_ncs_noise(
            &raw_noise,
            &stoch,
            StageIdx(0),
            n_blks,
            &ncs_max_gen,
            &ncs_allow_curtailment,
            &mut owner_scratch.ncs_col_lower_buf,
            &mut owner_scratch.ncs_col_upper_buf,
        );
        let mut owner_solver = RecordingSolver::default();
        apply_ncs_col_bounds(
            &mut owner_solver,
            &mut owner_scratch,
            &geometry,
            &dense_col,
            &windows,
            stage_id,
        );

        assert_eq!(
            owner_solver.col_bounds_calls, reference_solver.col_bounds_calls,
            "apply_ncs_col_bounds must match the pre-collapse gather+set output"
        );
        assert_eq!(owner_solver.col_bounds_calls.len(), 1);
        let (indices, lower, upper) = &owner_solver.col_bounds_calls[0];
        assert_eq!(indices, &[ncs_start]);
        // A_r = max_gen * clamp(mean + std * eta, 0, 1); allow_curtailment == true
        // pins the lower bound to 0 (dispatch is free to curtail down to it).
        let expected_upper = 100.0_f64 * (0.5 + 0.1 * 0.37_f64).clamp(0.0, 1.0);
        assert_eq!(lower, &[0.0]);
        assert!((upper[0] - expected_upper).abs() < 1e-9);
    }
}
