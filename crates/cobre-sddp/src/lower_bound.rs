//! Lower bound evaluation for iterative LP-based optimization algorithms.
//!
//! [`evaluate_lower_bound`] computes the risk-adjusted lower bound by solving
//! the stage-0 LP for every opening in the scenario tree and aggregating the
//! per-opening objectives through the stage-0 risk measure. The result is
//! broadcast from rank 0 to all other ranks so that every rank holds the same
//! global lower bound value.
//!
//! ## Algorithm
//!
//! 1. Rank 0 iterates over all `n_openings` openings at stage 0.
//! 2. For each opening the LP is rebuilt: `load_model` → `add_rows(cut_batch)`
//!    → `fill_forward_patches` → `set_row_bounds` → `solve`.
//! 3. The per-opening objectives are aggregated by the risk measure using
//!    uniform probabilities `1 / n_openings`.
//! 4. The scalar lower bound is broadcast from rank 0 to all ranks.
//!
//! ## Correctness notes
//!
//! - The cut batch is built **once** before the opening loop because it does
//!   not change per opening.
//! - `fill_forward_patches` is used (not `fill_state_patches`) because each
//!   opening carries different noise values that must be patched into the LP.
//! - The function must be called **after** the backward pass and cut sync so
//!   that the FCF holds the latest cuts when the LPs are solved.
//! - Stage 0 should never be infeasible (the penalty system guarantees recourse),
//!   so `SddpError::Infeasible` from this function indicates a modelling error.

use std::ops::Range;

use cobre_comm::Communicator;
use cobre_solver::{RowBatch, SolverError, SolverInterface};
use cobre_stochastic::{OpeningTree, StochasticContext, evaluate_par_batch, solve_par_noise_batch};

use crate::{
    FutureCostFunction, InflowNonNegativityMethod, PatchBuffer, RiskMeasure, SddpError,
    StageIndexer,
    forward::build_cut_row_batch_into,
    lp_builder::COST_SCALE_FACTOR,
    noise::{NcsNoiseOffsets, compute_effective_eta, transform_ncs_noise},
};
use cobre_solver::StageTemplate;

/// Stage-0 inputs for lower bound evaluation, bundled to reduce parameter count.
///
/// Passed to [`evaluate_lower_bound`] in place of four separate parameters.
pub struct LbEvalSpec<'a> {
    /// Structural LP template for stage 0.
    pub template: &'a StageTemplate,
    /// AR-dynamics base row index for stage 0.
    pub base_row: usize,
    /// Pre-computed `ζ*σ` noise scale per hydro at stage 0.
    pub noise_scale: &'a [f64],
    /// Number of hydro plants with inflow noise.
    pub n_hydros: usize,
    /// Opening tree for stage-0 noise realizations.
    pub opening_tree: &'a OpeningTree,
    /// Risk measure for stage-0 objective aggregation.
    pub risk_measure: &'a RiskMeasure,
    /// Stochastic context for NCS availability patching.
    ///
    /// When `Some`, stochastic NCS column bounds are patched per opening using
    /// `transform_ncs_noise`. When `None`, NCS patching is skipped (used in
    /// tests or when no stochastic NCS entities are present).
    pub stochastic: Option<&'a StochasticContext>,
    /// Number of buses with stochastic load noise (needed as offset into the
    /// raw noise vector to locate the NCS noise dimensions).
    pub n_load_buses: usize,
    /// Maximum generation (MW) per stochastic NCS entity, sorted by entity ID.
    /// Length equals the number of stochastic NCS entities. Empty when none exist.
    pub ncs_max_gen: &'a [f64],
    /// Number of blocks at stage 0.
    pub block_count: usize,
    /// Column range for NCS generation variables in the stage-0 LP.
    /// Empty when no NCS entities are present; NCS patching is skipped.
    pub ncs_generation: Range<usize>,
    /// Inflow non-negativity treatment method.
    ///
    /// When `Truncation` or `TruncationWithPenalty`, the opening loop clamps
    /// negative PAR(p) inflows to zero before patching the LP. When `None` or
    /// `Penalty`, raw noise is used directly.
    pub inflow_method: &'a InflowNonNegativityMethod,
}

/// Per-evaluation scratch buffers for [`evaluate_lower_bound`] on rank 0.
///
/// Allocated once and stored on `IterationScratch`;
/// reused across training iterations to eliminate per-iteration heap allocation.
/// The first call to `evaluate_lower_bound` still allocates (grows Vec capacity);
/// subsequent iterations reuse the existing capacity.
///
/// All fields are plain `f64` / `usize` working buffers — no LP-specific state.
// All fields are scratch buffers; the shared `_buf` postfix is intentional.
#[allow(clippy::struct_field_names)]
pub struct LbEvalScratch {
    /// Per-opening noise realization (one entry per hydro).
    pub noise_buf: Vec<f64>,
    /// Z-inflow RHS values per hydro for PAR(p) rows.
    pub z_inflow_rhs_buf: Vec<f64>,
    /// NCS column upper bounds, written by `transform_ncs_noise` per opening.
    pub ncs_col_upper_buf: Vec<f64>,
    /// NCS column indices (constant across openings for a given stage).
    pub ncs_col_indices_buf: Vec<usize>,
    /// NCS column lower bounds (constant zeros, parallel to `ncs_col_indices_buf`).
    pub ncs_col_lower_buf: Vec<f64>,
    /// PAR lag matrix (constant across openings for a given call).
    pub lag_matrix_buf: Vec<f64>,
    /// Per-hydro eta floor computed from lags (constant across openings).
    pub eta_floor_buf: Vec<f64>,
    /// Per-hydro PAR inflow evaluated per opening.
    pub par_inflow_buf: Vec<f64>,
    /// Per-hydro effective eta after clamping (recomputed per opening).
    pub effective_eta_buf: Vec<f64>,
    /// Per-hydro zero-target vector for truncation precompute.
    pub zero_targets_buf: Vec<f64>,
}

impl LbEvalScratch {
    /// Construct a new scratch with all buffers empty.
    ///
    /// No heap allocation occurs until `evaluate_lower_bound` populates the
    /// buffers on the first call.
    #[must_use]
    pub fn new() -> Self {
        Self {
            noise_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            zero_targets_buf: Vec::new(),
        }
    }
}

impl Default for LbEvalScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundle of mutable scratch references passed to [`evaluate_lower_bound`].
///
/// Groups `patch_buf`, `lb_cut_batch`, `lb_cut_row_map`, and `lb_scratch` so
/// that the public signature of `evaluate_lower_bound` stays within the
/// clippy `too-many-arguments-threshold = 9`.  Construct via
/// [`LbEvalScratchBundle::from_scratch_fields`] when calling from a
/// `TrainingSession` (disjoint-borrow factory pattern).
pub struct LbEvalScratchBundle<'a> {
    /// Reusable patch buffer for LP row-bound patching.
    pub patch_buf: &'a mut PatchBuffer,
    /// Cut row batch for the lower-bound LP (stage 0).
    pub lb_cut_batch: &'a mut cobre_solver::RowBatch,
    /// Optional cut row map for append-only lower-bound LP management.
    pub lb_cut_row_map: Option<&'a mut crate::cut::CutRowMap>,
    /// Per-evaluation scratch buffers (reused across training iterations).
    pub lb_scratch: &'a mut LbEvalScratch,
}

impl<'a> LbEvalScratchBundle<'a> {
    /// Construct from disjoint fields of `IterationScratch`.
    ///
    /// Analogous to `BackwardPassInputs::from_session_fields`: the caller takes
    /// the fields it needs separately so that the borrow checker can verify
    /// non-aliasing, then passes them here.
    ///
    /// ```text
    /// let bundle = LbEvalScratchBundle::from_scratch_fields(
    ///     &mut self.scratch.patch_buf,
    ///     &mut self.scratch.lb_cut_batch,
    ///     Some(&mut self.scratch.lb_cut_row_map),
    ///     &mut self.scratch.lb_scratch,
    /// );
    /// ```
    pub fn from_scratch_fields(
        patch_buf: &'a mut PatchBuffer,
        lb_cut_batch: &'a mut cobre_solver::RowBatch,
        lb_cut_row_map: Option<&'a mut crate::cut::CutRowMap>,
        lb_scratch: &'a mut LbEvalScratch,
    ) -> Self {
        Self {
            patch_buf,
            lb_cut_batch,
            lb_cut_row_map,
            lb_scratch,
        }
    }
}

/// Phase 1 — rank-0 buffer pre-population and append-only LP management.
///
/// Pre-populates the constant NCS column-bound index/lower buffers (same across
/// all openings at a given stage) in `scratch` and performs the append-only LP
/// load. The caller-supplied `scratch` is populated in-place so that its
/// capacity is reused across training iterations.
///
/// Only called on rank 0.
fn lb_init_rank0<S: SolverInterface>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    spec: &LbEvalSpec<'_>,
    indexer: &StageIndexer,
    lb_cut_batch: &mut RowBatch,
    lb_cut_row_map: Option<&mut crate::cut::CutRowMap>,
    scratch: &mut LbEvalScratch,
) {
    // Clear the append-only buffers before repopulating (capacity is kept).
    scratch.ncs_col_upper_buf.clear();
    scratch.ncs_col_indices_buf.clear();
    scratch.ncs_col_lower_buf.clear();

    // Pre-populate index/lower buffers for NCS column bound patching.
    // These are constant across openings (same stage, same block count),
    // so we build them once before the opening loop.
    if let Some(stoch) = spec.stochastic {
        let n_stochastic_ncs = stoch.n_stochastic_ncs();
        if n_stochastic_ncs > 0 && !spec.ncs_generation.is_empty() {
            for ncs_idx in 0..n_stochastic_ncs {
                for blk in 0..spec.block_count {
                    scratch
                        .ncs_col_indices_buf
                        .push(spec.ncs_generation.start + ncs_idx * spec.block_count + blk);
                    scratch.ncs_col_lower_buf.push(0.0);
                }
            }
        }
    }

    // Resize par_inflow_buf to the current n_hydros (no-op when capacity
    // is already sufficient, which is the common case from iteration 2 on).
    scratch.par_inflow_buf.resize(spec.n_hydros, 0.0);

    // Append-only LP management for the lower bound solver.
    //
    // When a CutRowMap is provided, the solver persists across iterations:
    //   - First call (row_map empty): load_model + append all active cuts.
    //   - Subsequent calls: append only new cuts (row_map tracks existing).
    // Cuts are never removed from the LB LP — this keeps the lower bound
    // monotonically non-decreasing across iterations.
    //
    // When no CutRowMap is provided (test contexts with no persistent
    // state), fall back to full rebuild each call.
    if let Some(row_map) = lb_cut_row_map {
        if row_map.total_cut_rows() == 0 {
            // First call: full load.
            solver.load_model(spec.template);
        }
        // Append only cuts not yet present in the LP.
        crate::forward::append_new_cuts_to_lp(
            solver,
            fcf,
            0,
            indexer,
            &spec.template.col_scale,
            row_map,
            lb_cut_batch,
        );
    } else {
        // Test-only path: full rebuild every call.
        build_cut_row_batch_into(lb_cut_batch, fcf, 0, indexer, &spec.template.col_scale);
        solver.load_model(spec.template);
        if lb_cut_batch.num_rows > 0 {
            solver.add_rows(lb_cut_batch);
        }
    }
}

/// Phase 2 — truncation precompute and per-opening LP evaluation.
///
/// Precomputes the PAR lag matrix and eta floor (constant across openings), then
/// iterates over all stage-0 openings. For each opening: evaluates PAR inflows,
/// computes effective eta, patches row bounds, patches NCS column bounds
/// (correctness-critical per-opening step), solves, and records the objective.
///
/// Returns the vector of per-opening objectives.
///
/// # Errors
///
/// Returns [`SddpError::Infeasible`] if any opening LP is infeasible, or
/// [`SddpError::Solver`] for other LP solve failures.
// The per-opening loop body (noise build + NCS patch + solve) accounts for the
// length; it cannot be meaningfully split without fragmenting correctness-critical
// sequential steps (especially the NCS column-bound patch inside the opening loop).
fn lb_evaluate_stage_0<S: SolverInterface>(
    solver: &mut S,
    spec: &LbEvalSpec<'_>,
    patch_buf: &mut PatchBuffer,
    initial_state: &[f64],
    indexer: &StageIndexer,
    scratch: &mut LbEvalScratch,
) -> Result<Vec<f64>, SddpError> {
    let n_openings = spec.opening_tree.n_openings(0);
    let n_hydros = spec.n_hydros;
    let base_row = spec.base_row;

    // Truncation precomputation: lag matrix and eta floor are constant
    // across openings (same initial_state, same stage 0).
    let needs_truncation = matches!(
        spec.inflow_method,
        InflowNonNegativityMethod::Truncation
            | InflowNonNegativityMethod::TruncationWithPenalty { .. }
    );

    // Resolve the PAR LP once; used for both truncation and z-inflow RHS.
    let par_lp_opt = spec.stochastic.map(StochasticContext::par);
    let truncation_par = if needs_truncation {
        par_lp_opt.filter(|p| p.n_stages() > 0 && p.n_hydros() == n_hydros)
    } else {
        None
    };

    if let Some(par_lp) = truncation_par {
        let max_order = indexer.max_par_order;
        let lag_len = max_order * n_hydros;
        scratch.lag_matrix_buf.resize(lag_len, 0.0);
        for h in 0..n_hydros {
            for l in 0..max_order {
                scratch.lag_matrix_buf[l * n_hydros + h] =
                    initial_state[indexer.inflow_lags.start + l * n_hydros + h];
            }
        }

        // eta_floor is constant across openings: depends only on lags
        // (initial_state) and stage (0), not on the opening noise.
        scratch.eta_floor_buf.resize(n_hydros, f64::NEG_INFINITY);
        // zero_targets is constant (all zeros); reuse the scratch buffer to
        // avoid a per-call allocation.
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

    let mut objectives = Vec::with_capacity(n_openings);

    for opening_idx in 0..n_openings {
        let raw_noise = spec.opening_tree.opening(0, opening_idx);
        scratch.noise_buf.clear();
        scratch.z_inflow_rhs_buf.clear();

        // Per-opening: evaluate PAR inflows (only when truncation active).
        if let Some(par_lp) = truncation_par {
            evaluate_par_batch(
                par_lp,
                0,
                &scratch.lag_matrix_buf,
                raw_noise,
                &mut scratch.par_inflow_buf,
            );
        }

        // Compute effective eta (clamped or raw).
        compute_effective_eta(
            raw_noise,
            n_hydros,
            spec.inflow_method,
            &scratch.par_inflow_buf,
            &scratch.eta_floor_buf,
            &mut scratch.effective_eta_buf,
        );

        // Build noise_buf and z_inflow_rhs_buf from effective eta.
        for (h, &eta_eff) in scratch.effective_eta_buf.iter().enumerate() {
            scratch
                .noise_buf
                .push(spec.template.row_lower[base_row + h] + spec.noise_scale[h] * eta_eff);
            if let Some(stoch) = spec.stochastic {
                let par_lp = stoch.par();
                if par_lp.n_stages() > 0 && par_lp.n_hydros() == n_hydros {
                    let base = par_lp.deterministic_base(0, h);
                    let sigma = par_lp.sigma(0, h);
                    scratch.z_inflow_rhs_buf.push(base + sigma * eta_eff);
                } else {
                    scratch.z_inflow_rhs_buf.push(0.0);
                }
            } else {
                scratch.z_inflow_rhs_buf.push(0.0);
            }
        }

        patch_buf.fill_forward_patches(
            indexer,
            initial_state,
            &scratch.noise_buf,
            base_row,
            &spec.template.row_scale,
        );
        patch_buf.fill_z_inflow_patches(
            indexer.z_inflow_row_start,
            &scratch.z_inflow_rhs_buf,
            &spec.template.row_scale,
        );
        let n_patches = patch_buf.forward_patch_count();
        solver.set_row_bounds(
            &patch_buf.indices[..n_patches],
            &patch_buf.lower[..n_patches],
            &patch_buf.upper[..n_patches],
        );

        // Patch NCS column upper bounds with per-opening stochastic availability.
        // CORRECTNESS: this patch MUST be inside the per-opening loop; each opening
        // has a different noise realization that changes the available NCS generation.
        // Moving this outside the loop would be a bug (see MEMORY.md D15 note).
        if let Some(stoch) = spec.stochastic {
            let n_stochastic_ncs = stoch.n_stochastic_ncs();
            if n_stochastic_ncs > 0 && !spec.ncs_generation.is_empty() {
                transform_ncs_noise(
                    raw_noise,
                    &NcsNoiseOffsets {
                        n_hydros,
                        n_load_buses: spec.n_load_buses,
                    },
                    stoch,
                    0,
                    spec.block_count,
                    spec.ncs_max_gen,
                    &mut scratch.ncs_col_upper_buf,
                );
                // ncs_col_indices_buf and ncs_col_lower_buf were pre-populated
                // in lb_init_rank0 — no rebuild needed here.
                solver.set_col_bounds(
                    &scratch.ncs_col_indices_buf,
                    &scratch.ncs_col_lower_buf,
                    &scratch.ncs_col_upper_buf,
                );
            }
        }

        let view = solver.solve(None).map_err(|e| match e {
            SolverError::Infeasible => SddpError::Infeasible {
                stage: 0,
                iteration: 0,
                scenario: opening_idx,
            },
            other => SddpError::Solver(other),
        })?;
        objectives.push(view.objective);
    }

    Ok(objectives)
}

/// Phase 3 — risk-measure aggregation and MPI broadcast.
///
/// Applies `risk_measure` to the per-opening objectives with uniform
/// probabilities, scales by [`COST_SCALE_FACTOR`], then broadcasts the scalar
/// lower bound from rank 0 to all other ranks.
///
/// # Errors
///
/// Returns [`SddpError::Communication`] if the broadcast fails.
fn lb_aggregate_and_broadcast<C: Communicator>(
    objectives: &[f64],
    risk_measure: &RiskMeasure,
    comm: &C,
) -> Result<f64, SddpError> {
    #[allow(clippy::cast_precision_loss)]
    let uniform_prob = 1.0_f64 / objectives.len() as f64;
    let mut lb = risk_measure.evaluate_risk(objectives, &vec![uniform_prob; objectives.len()])
        * COST_SCALE_FACTOR;
    comm.broadcast(std::slice::from_mut(&mut lb), 0)
        .map_err(SddpError::from)?;
    Ok(lb)
}

/// Evaluate the global lower bound for the current FCF approximation.
///
/// On rank 0 the function iterates over all stage-0 openings, solves the LP
/// for each, and applies the risk measure to produce a risk-adjusted scalar
/// lower bound. The scalar is then broadcast to all ranks.
///
/// ## Arguments
///
/// - `solver` — Mutable LP solver instance. Only rank 0 calls `solve`; other
///   ranks skip the opening loop entirely.
/// - `fcf` — Future Cost Function with all accumulated cuts.
/// - `initial_state` — Known initial state vector `x_0` (length `indexer.n_state`).
/// - `indexer` — LP layout map for stage 0.
/// - `scratch` — Bundled mutable scratch references (patch buffer, cut batch,
///   cut row map, and per-evaluation buffers). See [`LbEvalScratchBundle`].
/// - `spec` — Stage-0 data bundle: template, base row, noise scale, hydro count,
///   opening tree, and risk measure. See [`LbEvalSpec`].
/// - `comm` — Communicator for rank/size queries and broadcast.
///
/// ## Errors
///
/// - [`SddpError::Infeasible`] — Stage-0 LP has no feasible solution for an
///   opening. This indicates a modelling error; stage 0 should always be
///   feasible due to the penalty/recourse structure.
/// - [`SddpError::Solver`] — LP solve failed for a reason other than
///   infeasibility (numerical difficulty, time limit, etc.).
/// - [`SddpError::Communication`] — The broadcast to non-root ranks failed.
///
/// ## Panics
///
/// Panics if `spec.opening_tree.n_openings(0) == 0` on rank 0. Stage 0 must
/// have at least one opening; this is a caller contract violation.
pub fn evaluate_lower_bound<S: SolverInterface, C: Communicator>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    initial_state: &[f64],
    indexer: &StageIndexer,
    scratch: &mut LbEvalScratchBundle<'_>,
    spec: &LbEvalSpec<'_>,
    comm: &C,
) -> Result<f64, SddpError> {
    let mut lb = 0.0_f64;

    if comm.rank() == 0 {
        assert!(
            spec.opening_tree.n_openings(0) > 0,
            "evaluate_lower_bound: stage 0 must have at least one opening"
        );

        // Phase 1: populate scratch buffers and perform append-only LP load.
        lb_init_rank0(
            solver,
            fcf,
            spec,
            indexer,
            scratch.lb_cut_batch,
            scratch.lb_cut_row_map.as_deref_mut(),
            scratch.lb_scratch,
        );

        // Phase 2: truncation precompute + per-opening loop.
        let objectives = lb_evaluate_stage_0(
            solver,
            spec,
            scratch.patch_buf,
            initial_state,
            indexer,
            scratch.lb_scratch,
        )?;

        // Phase 3: risk-measure aggregation + broadcast.
        lb = lb_aggregate_and_broadcast(&objectives, spec.risk_measure, comm)?;
        return Ok(lb);
    }

    comm.broadcast(std::slice::from_mut(&mut lb), 0)
        .map_err(SddpError::from)?;
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
        LbEvalScratch, LbEvalScratchBundle, LbEvalSpec, evaluate_lower_bound, lb_evaluate_stage_0,
    };
    use crate::{
        FutureCostFunction, InflowNonNegativityMethod, PatchBuffer, RiskMeasure, SddpError,
        StageIndexer,
    };
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::OpeningTree;

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
    ///
    /// Call pattern:
    /// ```text
    /// let (mut row_batch, mut lb_scratch) = make_lb_locals();
    /// let mut bundle = LbEvalScratchBundle::from_scratch_fields(
    ///     &mut patch_buf, &mut row_batch, None, &mut lb_scratch,
    /// );
    /// evaluate_lower_bound(..., &mut bundle, ...);
    /// ```
    fn make_lb_locals() -> (RowBatch, LbEvalScratch) {
        (empty_row_batch(), LbEvalScratch::new())
    }

    /// Minimal stage template for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout: [storage (0), `storage_in` (1), theta (2)]
    /// Row layout: [`storage_fixing` (0)]
    fn minimal_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1], // col 1 (storage_in) has NZ at row 0
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0], // minimise theta
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
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
    /// `dim = 1` throughout (single hydro, `StageIndexer::new(1, 0)`).
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
        let decomposed = DecomposedCorrelation::build(&corr_model).unwrap();
        let entity_order = vec![entity_id];

        cobre_stochastic::tree::generate::generate_opening_tree(
            42,
            &[stage],
            1, // dim = 1 hydro
            &decomposed,
            &entity_order,
            cobre_stochastic::ClassDimensions {
                n_hydros: 1,
                n_load_buses: 0,
                n_ncs: 0,
            },
            &cobre_stochastic::tree::generate::OpeningTreeGenerationInputs::default(),
        )
        .unwrap()
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
            // Return a minimal view; evaluate_lower_bound only reads `view.objective`.
            // Use static empty slices for primal/dual/reduced_costs.
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[],
                dual: &[],
                reduced_costs: &[],
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    // ── Shared test setup ────────────────────────────────────────────────────

    fn make_fcf(n_stages: usize, n_state: usize) -> FutureCostFunction {
        // max_cuts=100, n_transfer=0
        FutureCostFunction::new(n_stages, n_state, 2, 100, &vec![0; n_stages])
    }

    // ── Unit tests ───────────────────────────────────────────────────────────

    /// AC1: 1 opening, Expectation — LB equals the single LP objective.
    #[test]
    fn one_opening_expectation_lb_equals_single_objective() {
        let indexer = StageIndexer::new(1, 0); // n_state=1, hydro_count=1
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch, mut lb_scratch) = make_lb_locals();
        let mut bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch,
            None,
            &mut lb_scratch,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle,
            &spec,
            &comm,
        )
        .unwrap();

        assert!(
            (lb - 100_000.0).abs() < 1e-7,
            "single opening expectation LB must equal objective 100.0 * COST_SCALE_FACTOR = 100_000.0, got {lb}"
        );
    }

    /// AC2: 3 openings, Expectation — LB equals mean of objectives.
    #[test]
    fn three_openings_expectation_lb_equals_mean() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(3);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        // Solver returns 60, 80, 100 for the three openings.
        let mut solver = MockSolver::with_objectives(vec![60.0, 80.0, 100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // E[60, 80, 100] with uniform probs = (60+80+100)/3 = 80.0; * COST_SCALE_FACTOR = 80_000.0
        assert!(
            (lb - 80_000.0).abs() < 1e-7,
            "three openings expectation LB must equal 80_000.0, got {lb}"
        );
    }

    /// AC3: 2 openings, CVaR(alpha=0.5, lambda=1.0) — pure `CVaR` selects worst.
    #[test]
    fn two_openings_pure_cvar_alpha_half_lb_equals_worst() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(2);
        // CVaR(alpha=0.5, lambda=1.0): pure CVaR; upper bound per scenario =
        // p / alpha = 0.5 / 0.5 = 1.0. With 2 equal-probability scenarios the
        // greedy allocation places all mass on the worst scenario.
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let comm = LocalComm;

        // Solver returns 50, 150.
        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=0.5, lambda=1.0) with 2 uniform-probability openings
        // concentrates all weight on the worst (150.0); * COST_SCALE_FACTOR = 150_000.0.
        assert!(
            (lb - 150_000.0).abs() < 1e-7,
            "pure CVaR(0.5, 1.0) with 2 openings must equal 150_000.0, got {lb}"
        );
    }

    /// AC4 (extra): 2 openings, CVaR(alpha=1.0, lambda=1.0) = Expectation.
    #[test]
    fn two_openings_cvar_alpha_one_equals_expectation() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        };
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=1) = Expectation = (50+150)/2 = 100.0; * COST_SCALE_FACTOR = 100_000.0
        assert!(
            (lb - 100_000.0).abs() < 1e-7,
            "CVaR(alpha=1, lambda=1) must equal expectation 100_000.0, got {lb}"
        );
    }

    /// AC5: solver returns Infeasible for the first opening — must propagate as `SddpError::Infeasible`.
    #[test]
    fn infeasible_solve_maps_to_sddp_infeasible() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::infeasible_on_first();

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_result,
            &spec,
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
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = FailingBcastComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_result,
            &spec,
            &comm,
        );

        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "broadcast failure must produce SddpError::Communication, got {result:?}"
        );
    }

    // ── Integration tests ────────────────────────────────────────────────────

    /// Integration: full round-trip with `LocalComm` and 2 openings.
    ///
    /// Verifies that the function correctly integrates with `build_cut_row_batch`
    /// (`cut_batch` with 0 cuts still produces the right result), `fill_forward_patches`,
    /// and `RiskMeasure::Expectation`.
    #[test]
    fn integration_two_openings_local_backend_expectation() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        // Start with 0 cuts (empty FCF).
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![50.0_f64]; // non-zero initial state
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![200.0, 300.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // E[200, 300] = 250.0; * COST_SCALE_FACTOR = 250_000.0
        assert!(
            (lb - 250_000.0).abs() < 1e-7,
            "integration round-trip must produce 250_000.0, got {lb}"
        );
    }

    /// Integration: monotonicity — adding cuts can only increase the LB.
    ///
    /// This test calls `evaluate_lower_bound` twice: first with 0 cuts, then
    /// with objectives set higher (simulating tighter cuts). The second LB
    /// must be >= the first.
    #[test]
    fn integration_monotonicity_more_cuts_yields_higher_or_equal_lb() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };

        // First call: solver returns [50, 100] → LB = 75.
        let mut solver1 = MockSolver::with_objectives(vec![50.0, 100.0]);
        let (mut row_batch_lb1, mut lb_scratch_lb1) = make_lb_locals();
        let mut bundle_lb1 = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb1,
            None,
            &mut lb_scratch_lb1,
        );
        let lb1 = evaluate_lower_bound(
            &mut solver1,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_lb1,
            &spec,
            &comm,
        )
        .unwrap();

        // Second call: solver returns [80, 120] → LB = 100 (tighter cuts raise obj).
        let mut solver2 = MockSolver::with_objectives(vec![80.0, 120.0]);
        let (mut row_batch_lb2, mut lb_scratch_lb2) = make_lb_locals();
        let mut bundle_lb2 = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb2,
            None,
            &mut lb_scratch_lb2,
        );
        let lb2 = evaluate_lower_bound(
            &mut solver2,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_lb2,
            &spec,
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
    /// With `stochastic: None`, the truncation path is a no-op since
    /// `has_par == false`. This validates that the `compute_effective_eta`
    /// control flow works correctly when no PAR model is present.
    #[test]
    fn test_lb_none_method_unchanged() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![60.0, 80.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // E[60, 80] = 70.0; * COST_SCALE_FACTOR = 70_000.0
        assert!(
            (lb - 70_000.0).abs() < 1e-7,
            "None method must produce correct LB, got {lb}"
        );
    }

    /// `Truncation` method does not cause a crash or infeasibility.
    ///
    /// With `stochastic: None`, the truncation path is a no-op since
    /// `has_par == false`, but this validates that the control flow
    /// (`needs_truncation` = true, `truncation_par` = `None`) does not panic.
    #[test]
    fn test_lb_truncation_no_crash() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::Truncation,
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_result,
            &spec,
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
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::TruncationWithPenalty { cost: 100.0 },
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &indexer,
            &mut bundle_result,
            &spec,
            &comm,
        );

        assert!(
            result.is_ok(),
            "TruncationWithPenalty method must not panic or fail, got {result:?}"
        );
    }

    // ── NCS column-bound patching regression test ────────────────────────────

    /// Regression: `lb_evaluate_stage_0` calls `set_col_bounds` once per
    /// opening when stochastic NCS entities are present.
    ///
    /// This is the correctness guard for the MEMORY.md D15 bug: NCS column
    /// bounds must be patched *per opening*, not once before the loop or not
    /// at all. With `n_openings` openings and stochastic NCS present, the
    /// solver must receive exactly `n_openings` `set_col_bounds` calls.
    ///
    /// A real `StochasticContext` is built via `build_stochastic_context` with
    /// a minimal `System` containing one `NonControllableSource` and one
    /// `NcsModel`. The `LbRank0State` is pre-populated to mirror the output of
    /// `lb_init_rank0`. `lb_evaluate_stage_0` is called directly so that we can
    /// inspect the `MockSolver`'s `set_col_bounds_calls` counter afterwards.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn lb_evaluate_stage_0_patches_ncs_bounds_per_opening() {
        use cobre_core::{
            Bus, DeficitSegment, EntityId, SystemBuilder,
            entities::non_controllable::NonControllableSource,
            scenario::{
                CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
                NcsModel, SamplingScheme,
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
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        let ncs_source = NonControllableSource {
            id: ncs_entity_id,
            name: "W1".to_string(),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 100.0,
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

        assert_eq!(
            stoch.n_stochastic_ncs(),
            n_ncs,
            "StochasticContext must report {n_ncs} stochastic NCS entity"
        );

        let opening_tree = stoch.opening_tree();

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
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };

        let indexer = StageIndexer::new(0, 0);
        let ncs_max_gen = vec![100.0_f64; n_ncs];

        let spec = LbEvalSpec {
            template: &template,
            base_row: 0,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree,
            risk_measure: &RiskMeasure::Expectation,
            stochastic: Some(&stoch),
            n_load_buses: 0,
            ncs_max_gen: &ncs_max_gen,
            block_count,
            ncs_generation: 0..block_count,
            inflow_method: &InflowNonNegativityMethod::None,
        };

        // Pre-populate LbEvalScratch as lb_init_rank0 would.
        let mut lb_scratch = LbEvalScratch::new();
        for ncs_idx in 0..n_ncs {
            for blk in 0..block_count {
                lb_scratch
                    .ncs_col_indices_buf
                    .push(spec.ncs_generation.start + ncs_idx * block_count + blk);
                lb_scratch.ncs_col_lower_buf.push(0.0);
            }
        }

        let mut patch_buf = PatchBuffer::new(0, 0, 0, 0);
        let initial_state: Vec<f64> = Vec::new();
        let actual_n_openings = opening_tree.n_openings(0);
        let mut solver =
            MockSolver::with_objectives(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);

        lb_evaluate_stage_0(
            &mut solver,
            &spec,
            &mut patch_buf,
            &initial_state,
            &indexer,
            &mut lb_scratch,
        )
        .unwrap();

        // set_col_bounds must have been called exactly once per opening.
        assert_eq!(
            solver.set_col_bounds_calls, actual_n_openings,
            "set_col_bounds must be called once per opening ({actual_n_openings} openings), \
             got {} calls — NCS bounds are not being patched per opening",
            solver.set_col_bounds_calls
        );
        // Sanity: the opening count must be positive.
        assert!(
            actual_n_openings > 0,
            "opening tree must have at least one opening at stage 0"
        );
    }

    // ── Scratch reuse regression test ────────────────────────────────────────

    /// Verify that `LbEvalScratch` buffers are reused across consecutive calls.
    ///
    /// Calls `evaluate_lower_bound` twice on the same scratch and verifies that
    /// `noise_buf.capacity()` does not decrease on the second call (i.e., no
    /// reallocation occurred). This guards against regressions that would re-
    /// introduce per-iteration heap allocation on the lower-bound hot path.
    #[test]
    fn lb_eval_scratch_reuses_buffers_across_calls() {
        // Use n_hydros = 1 so that noise_buf gets populated (capacity grows to 1
        // after the first call). The template must have at least 1 row to avoid
        // index-out-of-bounds in fill_forward_patches when n_hydros = 1.
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let spec = LbEvalSpec {
            template: &template,
            base_row: 0,
            noise_scale: &[1.0],
            n_hydros: 1,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            block_count: 1,
            ncs_generation: 0..0,
            inflow_method: &InflowNonNegativityMethod::None,
        };

        let mut row_batch = empty_row_batch();
        let mut lb_scratch = LbEvalScratch::new();

        // First call — allocates scratch buffers for the first time.
        let mut solver1 = MockSolver::with_objectives(vec![10.0]);
        {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver1,
                &fcf,
                &initial_state,
                &indexer,
                &mut bundle,
                &spec,
                &comm,
            )
            .unwrap();
        }

        // Capture capacity after the first call.
        let cap_after_first = lb_scratch.noise_buf.capacity();
        assert!(
            cap_after_first > 0,
            "noise_buf must have nonzero capacity after first call (n_hydros = 1)"
        );

        // Second call — must reuse the existing capacity (no reallocation).
        let mut solver2 = MockSolver::with_objectives(vec![20.0]);
        {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver2,
                &fcf,
                &initial_state,
                &indexer,
                &mut bundle,
                &spec,
                &comm,
            )
            .unwrap();
        }

        let cap_after_second = lb_scratch.noise_buf.capacity();
        assert_eq!(
            cap_after_second, cap_after_first,
            "noise_buf capacity must be stable across calls (first={cap_after_first}, second={cap_after_second}); \
             a decrease indicates reallocation on the lower-bound hot path"
        );
    }
}
