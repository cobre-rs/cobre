//! Probe substrate and lag-fold probes over the `cobre_rodada` case.
//!
//! Kept entry points:
//!
//! - `classify_stage_rows_reconciles_on_a_hand_built_geometry` — fast,
//!   non-ignored unit test pinning the row-family classifier against the
//!   real `StageLayout::new` allocation order.
//! - `lag_fold_chain_rule_identity` (`#[ignore]`) — verifies the pinned
//!   inflow-lag columns' coupling channels and the chain-rule
//!   reconstruction of Benders cut coefficients from row duals + pool
//!   coefficients (~10 min, trains one checkpoint).
//! - `lag_fold_static_ab` (`#[ignore]`) — original-vs-cut-row-folded twin
//!   performance A/B over full backward chains (~1 h, trains three
//!   checkpoints).
//!
//! ```text
//! cargo nextest run --release -p cobre-sddp --features test-support --test basis_trajectory_probe \
//!     --run-ignored ignored-only -E 'test(lag_fold_chain_rule_identity)'
//! ```
//!
//! `case_dir` resolves the converted `cobre_rodada` case from `$HOME` — it
//! lives outside this repository (a sibling checkout), so no repo-relative
//! path can name it; the probes panic with a clear message if the checkout
//! is absent rather than silently skipping.
//!
//! `StageGeometry::fpha` exposes the FPHA row range these probes read
//! (per-hydro plane counts are non-uniform, so no exposed count multiplies
//! out to it); `training_ctx()`/`solve_stage_for_probe` are the test-support
//! entry points supplying the rest of the reach.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::single_match_else
)]

use cobre_io::config::TrainingSelection;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};

use cobre_solver::{ActiveSolver, SolverInterface, StageTemplate};

use cobre_sddp::context::StageContext;
use cobre_sddp::cut::pool::CutPool;
use cobre_sddp::forward::{ForwardPassBatch, run_forward_pass};
use cobre_sddp::indexer::{StateDim, StateSpace};
use cobre_sddp::setup::NodeId;
use cobre_sddp::setup::NodePos;
use cobre_sddp::setup::StageIdx;
use cobre_sddp::test_support::{
    StageContextFixture, patch_backward_opening_for_probe, solve_stage_for_probe,
};
use cobre_sddp::workspace::{BasisStore, CapturedBasis, SolverWorkspace};
use cobre_sddp::{PrepareHydroModelsResult, StudySetup, TrajectoryRecord};

mod common;
use common::StubComm;

// ---------------------------------------------------------------------------
// Row-family classification
// ---------------------------------------------------------------------------

/// One of the row families `StageLayout::new`'s row cursor allocates, in its
/// fixed allocation order — NOT `fill_stage_rows`'s call order, which only
/// writes bounds into already-allocated positions and is order-independent —
/// plus a residual `GenericConstraint` region and the appended-cuts region.
/// The real order (`lp/builder/layout.rs`, the `row.alloc(...)` sequence in
/// `StageLayout::new`): `z_inflow`, `water_balance`, `transit_bucket_definition`,
/// `load_balance`, `fpha`, `evaporation`, the four operational-violation families,
/// `filling_target`, `filled_min_storage_floor`, the three anticipated-thermal
/// families, then generic constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum RowFamily {
    ZInflow = 0,
    WaterBalance = 1,
    TransitBucketDefinition = 2,
    LoadBalance = 3,
    Fpha = 4,
    Evaporation = 5,
    MinOutflowViolation = 6,
    MaxOutflowViolation = 7,
    MinTurbineViolation = 8,
    MinGenerationViolation = 9,
    FillingTarget = 10,
    FilledMinStorageFloor = 11,
    GenericConstraint = 12,
    CutRow = 13,
}

impl RowFamily {
    fn as_str(self) -> &'static str {
        match self {
            RowFamily::ZInflow => "z_inflow",
            RowFamily::WaterBalance => "water_balance",
            RowFamily::TransitBucketDefinition => "transit_bucket_definition",
            RowFamily::LoadBalance => "load_balance",
            RowFamily::Fpha => "fpha",
            RowFamily::Evaporation => "evaporation",
            RowFamily::MinOutflowViolation => "min_outflow_violation",
            RowFamily::MaxOutflowViolation => "max_outflow_violation",
            RowFamily::MinTurbineViolation => "min_turbine_violation",
            RowFamily::MinGenerationViolation => "min_generation_violation",
            RowFamily::FillingTarget => "filling_target",
            RowFamily::FilledMinStorageFloor => "filled_min_storage_floor",
            RowFamily::GenericConstraint => "generic_constraint",
            RowFamily::CutRow => "cut_row",
        }
    }
}

/// Assign every LP row of `template` (loaded at `stage`) to a [`RowFamily`],
/// given `base_row_count` — the template's own pre-cut row count
/// (`StageContext::templates[stage].num_rows`), the authoritative boundary
/// between structural rows and appended cut rows.
///
/// Reconstructs each unexposed family's position by chaining off the ranges
/// `StageGeometry` does expose (`water_balance`, `filling_target`,
/// `filled_min_storage_floor`, `load_balance`, `fpha`) and
/// `StateSpace::z_inflow_rows()`, plus counts derivable from
/// `StageContext`/`StageGeometry`
/// (`evap_indices.len()`, `n_hydros * n_blks` per operational-violation
/// family) — never a hand-copied row-fill formula.
/// Three families this walk cannot place directly are handled by
/// construction: `transit_bucket_definition`'s size falls out of the
/// `water_balance.end() .. load_balance.start()` gap regardless of its value; the
/// three anticipated-thermal families are asserted empty (panics naming the
/// gap) rather than silently misclassified — `cobre_rodada` has neither
/// declared travel-time arcs nor anticipated thermals, confirmed by the
/// `filling_target.start` reconciliation this function asserts; and
/// `generic_constraint` is the un-subdivided residual between the anticipated
/// region and `base_row_count` (this classifier does not further decompose
/// active generic-constraint rows by declared constraint).
fn classify_stage_rows(
    ctx: &StageContext<'_>,
    state_space: &StateSpace,
    stage: usize,
    template: &StageTemplate,
    base_row_count: usize,
) -> Vec<RowFamily> {
    let geom = &ctx.geometry_per_stage[stage];
    let n_hydros = state_space.hydro_count;
    let n_blks = geom.n_blks;
    let total_rows = template.num_rows;

    let mut fam = vec![RowFamily::CutRow; total_rows];
    let mark = |fam: &mut [RowFamily], range: Range<usize>, family: RowFamily| {
        fam[range].fill(family);
    };

    let z_inflow_range = state_space.z_inflow_rows();
    mark(&mut fam, z_inflow_range.clone(), RowFamily::ZInflow);
    assert_eq!(
        z_inflow_range.start, 0,
        "classify_stage_rows: z_inflow does not start at row 0 at stage {stage} — state \
         pinning uses column bounds, so no row family should precede it"
    );
    assert_eq!(
        geom.water_balance.start(),
        z_inflow_range.end,
        "classify_stage_rows: water_balance does not immediately follow z_inflow at stage {stage}"
    );

    mark(
        &mut fam,
        geom.water_balance.range(),
        RowFamily::WaterBalance,
    );
    mark(
        &mut fam,
        geom.water_balance.end()..geom.load_balance.start(),
        RowFamily::TransitBucketDefinition,
    );
    mark(&mut fam, geom.load_balance.range(), RowFamily::LoadBalance);
    assert_eq!(
        geom.fpha.start,
        geom.load_balance.end(),
        "classify_stage_rows: fpha does not immediately follow load_balance at stage {stage}"
    );
    mark(&mut fam, geom.fpha.clone(), RowFamily::Fpha);

    let evap_len = geom.evap_indices.len();
    let evap_range = geom.fpha.end..geom.fpha.end + evap_len;
    mark(&mut fam, evap_range.clone(), RowFamily::Evaporation);

    let mut next = evap_range.end;
    let opviol_len = n_hydros * n_blks;
    for family in [
        RowFamily::MinOutflowViolation,
        RowFamily::MaxOutflowViolation,
        RowFamily::MinTurbineViolation,
        RowFamily::MinGenerationViolation,
    ] {
        mark(&mut fam, next..next + opviol_len, family);
        next += opviol_len;
    }

    assert_eq!(
        next, geom.filling_target.start,
        "classify_stage_rows: computed row offset {next} before filling_target does not match \
         StageGeometry::filling_target.start {} at stage {stage} — family-order or size \
         assumption is wrong",
        geom.filling_target.start
    );
    mark(
        &mut fam,
        geom.filling_target.clone(),
        RowFamily::FillingTarget,
    );
    assert_eq!(
        geom.filled_min_storage_floor.start, geom.filling_target.end,
        "classify_stage_rows: filled_min_storage_floor does not immediately follow \
         filling_target at stage {stage}"
    );
    mark(
        &mut fam,
        geom.filled_min_storage_floor.clone(),
        RowFamily::FilledMinStorageFloor,
    );

    assert_eq!(
        state_space.n_anticipated, 0,
        "classify_stage_rows: n_anticipated={} but this classifier does not yet derive the \
         sparse anticipated row families (anticipated_fishing / anticipated_state_out_def / \
         anticipated_slot_definition) — extend it before probing a case with anticipated \
         thermals",
        state_space.n_anticipated
    );
    assert!(
        geom.filled_min_storage_floor.end <= base_row_count,
        "classify_stage_rows: filled_min_storage_floor.end {} exceeds base_row_count {} at \
         stage {stage}",
        geom.filled_min_storage_floor.end,
        base_row_count
    );
    mark(
        &mut fam,
        geom.filled_min_storage_floor.end..base_row_count,
        RowFamily::GenericConstraint,
    );

    fam
}

/// Structural equality-floor / FPHA / operational-violation census for one
/// stage, logged so family-composition drift across runs is visible.
struct RowCensus {
    water_balance: usize,
    z_inflow: usize,
    evaporation: usize,
    load_balance: usize,
    fpha: usize,
    operational_violation: usize,
    cut_rows: usize,
}

fn census(fam: &[RowFamily]) -> RowCensus {
    let count = |f: RowFamily| fam.iter().filter(|&&x| x == f).count();
    RowCensus {
        water_balance: count(RowFamily::WaterBalance),
        z_inflow: count(RowFamily::ZInflow),
        evaporation: count(RowFamily::Evaporation),
        load_balance: count(RowFamily::LoadBalance),
        fpha: count(RowFamily::Fpha),
        operational_violation: count(RowFamily::MinOutflowViolation)
            + count(RowFamily::MaxOutflowViolation)
            + count(RowFamily::MinTurbineViolation)
            + count(RowFamily::MinGenerationViolation),
        cut_rows: count(RowFamily::CutRow),
    }
}

#[test]
fn classify_stage_rows_reconciles_on_a_hand_built_geometry() {
    use cobre_sddp::test_support::{eq, geometry, state_layout};

    let dims = eq(
        /* hydro_count */ 3, /* max_par_order */ 0, /* n_thermals */ 0,
        /* n_lines */ 0, /* n_buses */ 2, /* n_blks */ 2,
        /* has_inflow_penalty */ false,
    );
    let fpha_hydro_indices = vec![0_usize, 1];
    let fpha_planes = [2_usize, 3];
    let evap_hydro_indices = vec![2_usize];
    let geom = geometry(&dims, fpha_hydro_indices, &fpha_planes, evap_hydro_indices);
    let n_hydros = dims.hydro_count;
    let n_blks = dims.n_blks;
    let state = state_layout(n_hydros, dims.max_par_order);

    let total_fpha_rows = 2 * n_blks + 3 * n_blks;
    let evap_rows = geom.evap_indices.len(); // one evap hydro
    let opviol_rows = 4 * n_hydros * n_blks;
    // z_inflow (n_hydros) + water_balance (n_hydros, BlockMode::Parallel) +
    // load_balance + fpha + evaporation + 4 operational-violation families;
    // no filling/anticipated/generic-constraint rows in this fixture.
    let base_row_count = n_hydros
        + n_hydros
        + geom.load_balance.range().len()
        + total_fpha_rows
        + evap_rows
        + opviol_rows;
    assert_eq!(
        state.z_inflow_rows().start,
        0,
        "fixture arithmetic sanity: z_inflow must be the first row family"
    );

    let total_rows = base_row_count + 5; // + 5 synthetic cut rows
    let template = StageTemplate {
        num_cols: 1,
        num_rows: total_rows,
        num_nz: 0,
        col_starts: vec![0, 0],
        row_indices: vec![],
        values: vec![],
        col_lower: vec![0.0],
        col_upper: vec![0.0],
        objective: vec![0.0],
        row_lower: vec![0.0; total_rows],
        row_upper: vec![0.0; total_rows],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let fixture = StageContextFixture::new(
        std::slice::from_ref(&template),
        &[],
        std::slice::from_ref(&geom),
    )
    .study_stage_ids(&[0])
    .discount_factors(&[1.0])
    .cumulative_discount_factors(&[1.0]);
    let ctx = fixture.ctx();

    let fam = classify_stage_rows(&ctx, &state, 0, &template, base_row_count);
    let c = census(&fam);
    assert_eq!(c.water_balance, n_hydros);
    assert_eq!(c.z_inflow, n_hydros);
    assert_eq!(c.evaporation, evap_rows);
    assert_eq!(c.load_balance, 2 * n_blks);
    assert_eq!(c.fpha, total_fpha_rows);
    assert_eq!(c.operational_violation, opviol_rows);
    assert_eq!(c.cut_rows, 5);
}

// ---------------------------------------------------------------------------
// Probe configuration
// ---------------------------------------------------------------------------

const SEED: u64 = 42;

fn case_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set to resolve the cobre_rodada case");
    let dir = PathBuf::from(home).join("git/cobre-bridge/example/cobre_rodada");
    assert!(
        dir.join("config.json").exists(),
        "cobre_rodada case not found at {}; these probes target the converted case checked \
         out as a sibling of this repository",
        dir.display()
    );
    dir
}

// ---------------------------------------------------------------------------
// Case loading (fresh per checkpoint)
// ---------------------------------------------------------------------------

/// Load a fresh `StudySetup` for `case_dir`, overriding `forward_passes` and
/// the iteration-limit stopping rule so each checkpoint is an independent,
/// from-scratch training run. With `cut_selection_off`, also overrides
/// `training.cut_selection` off: resident cuts then equal populated cuts,
/// reaching production-comparable density by deeper checkpoints.
fn fresh_setup(
    case_dir: &Path,
    forward_passes: u32,
    iteration_limit: u32,
    cut_selection_off: bool,
) -> StudySetup {
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config.json must parse");
    config.training.selection = Some(TrainingSelection::Sampled { forward_passes });
    config.training.stopping_rules =
        Some(vec![cobre_io::config::StoppingRuleConfig::IterationLimit {
            limit: iteration_limit,
        }]);
    if cut_selection_off {
        config.training.cut_selection = cobre_io::config::RowSelectionConfig::default();
    }

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed for cobre_rodada");
    let source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must resolve");
    let prepared =
        cobre_sddp::setup::prepare_stochastic(system, case_dir, &config, SEED, &source, None)
            .expect("prepare_stochastic must succeed for cobre_rodada");
    let hydro_models: PrepareHydroModelsResult =
        cobre_sddp::hydro_models::prepare_hydro_models(&prepared.system, case_dir, false)
            .expect("prepare_hydro_models must succeed for cobre_rodada");

    StudySetup::new(
        &prepared.system,
        &config,
        prepared.stochastic,
        hydro_models,
        Vec::new(),
    )
    .expect("StudySetup::new must build for cobre_rodada")
}

struct CheckpointRun {
    setup: StudySetup,
    frozen_templates: Vec<StageTemplate>,
}

/// Train `setup` to `iteration_limit`, returning the trained setup (its
/// `fcf` now holds every cut through that iteration) and the final frozen
/// templates.
fn train_checkpoint(
    case_dir: &Path,
    forward_passes: u32,
    iteration_limit: u32,
    threads: usize,
    cut_selection_off: bool,
) -> CheckpointRun {
    let mut setup = fresh_setup(case_dir, forward_passes, iteration_limit, cut_selection_off);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, threads, ActiveSolver::new, None, None)
        .expect("train() must not return Err on cobre_rodada");
    assert!(
        outcome.error.is_none(),
        "training error at checkpoint {iteration_limit}: {:?}",
        outcome.error
    );

    let frozen_templates = outcome
        .result
        .frozen_templates
        .expect("frozen_templates is always Some on a completed run");

    CheckpointRun {
        setup,
        frozen_templates,
    }
}

// ---------------------------------------------------------------------------
// Extra deterministic forward pass: trial states + forward bases
// ---------------------------------------------------------------------------

struct ForwardProbe {
    basis_store: BasisStore,
    records: Vec<TrajectoryRecord>,
}

/// Run one more deterministic forward pass against the checkpoint's trained
/// FCF, producing the trial states and forward-captured bases iteration
/// `checkpoint_iteration + 1`'s real backward pass would consume. Single-rank:
/// `local_forward_passes == total_forward_passes`, `fwd_offset == 0`; `threads`
/// only fans work across this rank's workspaces (`run_forward_pass`'s own
/// parallelism), it does not change which noise draws or which seed produce
/// the trajectory.
fn run_extra_forward_pass(
    setup: &StudySetup,
    frozen_templates: &[StageTemplate],
    forward_passes: u32,
    checkpoint_iteration: u64,
    threads: usize,
) -> ForwardProbe {
    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, threads, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let num_stages = setup.num_stages();
    let local_forward_passes = forward_passes as usize;

    let mut basis_store = BasisStore::new(local_forward_passes, num_stages);
    let mut records: Vec<TrajectoryRecord> = (0..local_forward_passes * num_stages)
        .map(|_| TrajectoryRecord {
            primal: Vec::new(),
            dual: Vec::new(),
            stage_cost: 0.0,
            node_id: NodeId(0),
            state: Vec::new(),
        })
        .collect();

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let batch = ForwardPassBatch {
        local_forward_passes,
        total_forward_passes: local_forward_passes,
        iteration: checkpoint_iteration + 1,
        fwd_offset: 0,
        event_sender: None,
    };

    let _forward_result = run_forward_pass(
        &mut pool.workspaces,
        &mut basis_store,
        &ctx,
        frozen_templates,
        &setup.fcf,
        &training_ctx,
        &batch,
        &mut records,
    )
    .expect("the extra deterministic forward pass must not error on cobre_rodada");

    ForwardProbe {
        basis_store,
        records,
    }
}

// ---------------------------------------------------------------------------
// Lag-fold chain-rule identity probe
// ---------------------------------------------------------------------------

/// Per-stage summary of the lag-fold identity checks (see
/// [`lag_fold_chain_rule_identity`]).
struct LagFoldStats {
    stage: usize,
    solves: usize,
    resident_cuts: usize,
    /// Max relative error of `rc == objective − Aᵀy` over every incoming
    /// state column's own CSC support, across all solves.
    max_rel_err_full_support: f64,
    /// Max relative error between frozen-template cut-row entries and
    /// `−coeff × col_scale` recomputed from the live pool.
    max_rel_err_pool_vs_template: f64,
    /// Max relative error of the chain-rule reconstruction (base-row dual
    /// terms + pool-coefficient GEMV) against the `rc / col_scale`
    /// cut-coefficient fingerprint, across all solves.
    max_rel_err_chain_rule: f64,
    /// Lag dims whose incoming column doubles as an outgoing render target
    /// (the shift alias); the remainder is the aged-out oldest lag per hydro.
    aliased_lag_dims: usize,
    unaliased_lag_dims: usize,
    /// The single non-state column cut rows touch (the theta column).
    theta_col: Option<usize>,
    /// Base-row families supporting the incoming storage columns (context
    /// for the fold design; storage stays in the LP).
    storage_support_families: BTreeMap<&'static str, usize>,
    /// Base-row entry counts per family over the incoming lag columns — the
    /// fold's per-node RHS patch sites and the chain rule's dual terms.
    lag_support_families: BTreeMap<&'static str, usize>,
    /// Base-row entry counts per family over the z-inflow columns — shows
    /// whether the water balance consumes `z_inflow` or bypasses it.
    z_col_support_families: BTreeMap<&'static str, usize>,
}

/// Replay every trial's opening chain at `stage` (same flow as
/// [`replay_stage`], no recording) and verify the identities a pinned-lag
/// fold relies on:
///
/// - **support**: an incoming inflow-lag column's base-template entries lie
///   only in its z-inflow and water-balance rows (measured: the PAR lag
///   combination is written into BOTH, at most one entry of each per lag
///   column) — any further family is a new coupling channel the fold design
///   has not accounted for (hard failure naming it);
/// - **full-support rc**: `reduced_cost == objective − Aᵀ·row_duals` over the
///   frozen template's CSC, per incoming state column (conventions check);
/// - **pool-vs-template**: every cut-row entry equals `−coeff × col_scale`
///   recomputed from `CutPool::active_cuts` at
///   `StateSpace::lp_column_for_state` (slot order = row order), and cut
///   rows touch no column outside the state render targets plus theta;
/// - **chain rule**: `rc/col_scale` on a lag column is reproduced from pool
///   coefficients and row duals alone — the z-row dual term plus the
///   shift-aliased entry of the cut-dual GEMV `g = Σ_c y_c · coeffs_c`.
fn lag_fold_check_stage(
    setup: &StudySetup,
    frozen_templates: &[StageTemplate],
    stage: usize,
    forward_probe: &ForwardProbe,
) -> LagFoldStats {
    assert!(stage >= 1, "the backward pass never solves stage 0");

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws: &mut SolverWorkspace<ActiveSolver> = &mut pool.workspaces[0];

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let state_space = setup.stage_state();
    let template = &frozen_templates[stage];
    let pool_ref: &CutPool = &setup.fcf.pools[stage];
    let opening_tree = training_ctx.stochastic.tree_view();
    let n_openings = opening_tree.n_openings(stage);
    let solve_order = opening_tree.solve_order_data(stage);

    let num_stages = setup.num_stages();
    let n_trials = forward_probe.records.len() / num_stages;
    let base_row_count = ctx.templates[stage].num_rows;
    let family_ids = classify_stage_rows(&ctx, state_space, stage, template, base_row_count);

    let n_state = state_space.n_state;
    let n = state_space.hydro_count;
    let p = state_space.max_par_order;
    assert_eq!(
        n_state,
        n * (1 + p),
        "stage {stage}: state space is not storage+lags only (buckets/anticipated present) — \
         extend this probe's region arithmetic before trusting it"
    );

    let in_cols: Vec<usize> = (0..n_state)
        .map(|j| {
            state_space
                .state_to_lp_incoming_column(StateDim::new(j))
                .get()
        })
        .collect();
    let out_cols: Vec<usize> = (0..n_state)
        .map(|j| state_space.lp_column_for_state(StateDim::new(j)).get())
        .collect();
    let mut out_col_to_dim: BTreeMap<usize, usize> = BTreeMap::new();
    for (jj, &c) in out_cols.iter().enumerate() {
        assert!(
            out_col_to_dim.insert(c, jj).is_none(),
            "stage {stage}: two state dims render onto LP column {c}"
        );
    }

    let col_scale_at = |c: usize| -> f64 { template.col_scale.get(c).copied().unwrap_or(1.0) };
    let csc_col = |c: usize| {
        let s = template.col_starts[c] as usize;
        let e = template.col_starts[c + 1] as usize;
        (s..e).map(|k| (template.row_indices[k] as usize, template.values[k]))
    };
    let rel_err = |a: f64, b: f64| -> f64 { (a - b).abs() / b.abs().max(1.0) };

    // -- static structural scan (template-only, solve-independent) --

    let mut unexpected_lag_support: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut lag_support_families: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut storage_support_families: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut z_col_support_families: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut aliased_lag_dims = 0usize;
    let mut unaliased_lag_dims = 0usize;
    for j in 0..n_state {
        let cin = in_cols[j];
        let mut z_rows = 0usize;
        let mut wb_rows = 0usize;
        for (r, _v) in csc_col(cin) {
            if r >= base_row_count {
                continue;
            }
            let fam = family_ids[r].as_str();
            if j < n {
                *storage_support_families.entry(fam).or_insert(0) += 1;
            } else {
                *lag_support_families.entry(fam).or_insert(0) += 1;
                match family_ids[r] {
                    RowFamily::ZInflow => z_rows += 1,
                    RowFamily::WaterBalance => wb_rows += 1,
                    _ => {
                        *unexpected_lag_support.entry(fam).or_insert(0) += 1;
                    }
                }
            }
        }
        if j >= n {
            let lag = (j - n) / n;
            let h = (j - n) % n;
            assert!(
                z_rows <= 1 && wb_rows <= 1,
                "stage {stage}: incoming lag (h={h}, lag={lag}) has {z_rows} z-inflow and \
                 {wb_rows} water-balance entries — expected at most one of each (its own \
                 hydro's rows); a cross-hydro PAR coupling changes the fold's patch shape"
            );
            match out_col_to_dim.get(&cin) {
                Some(&alias) => {
                    aliased_lag_dims += 1;
                    assert_eq!(
                        alias,
                        n + (lag + 1) * n + h,
                        "stage {stage}: incoming lag (h={h}, lag={lag}) aliases state dim \
                         {alias}, not the expected (h={h}, lag={})",
                        lag + 1
                    );
                }
                None => {
                    unaliased_lag_dims += 1;
                    assert_eq!(
                        lag,
                        p - 1,
                        "stage {stage}: incoming lag (h={h}, lag={lag}) has no outgoing \
                         alias but is not the oldest lag"
                    );
                }
            }
        }
    }
    for h in 0..n {
        let zc = out_cols[n + h];
        for (r, _v) in csc_col(zc) {
            if r < base_row_count {
                *z_col_support_families
                    .entry(family_ids[r].as_str())
                    .or_insert(0) += 1;
            }
        }
    }
    assert!(
        unexpected_lag_support.is_empty(),
        "stage {stage}: lag columns are supported by base rows outside \
         z_inflow/water_balance — a coupling channel the fold design has not accounted \
         for: {unexpected_lag_support:?}"
    );

    // -- pool-vs-template consistency (template-only) --

    let active: Vec<(usize, f64, &[f64])> = pool_ref.active_cuts().collect();
    let resident_cuts = active.len();
    assert_eq!(
        base_row_count + resident_cuts,
        template.num_rows,
        "stage {stage}: base_row_count + active cut count must equal frozen template rows"
    );

    let mut cut_entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for c in 0..template.num_cols {
        for (r, v) in csc_col(c) {
            if r >= base_row_count {
                cut_entries.insert((r, c), v);
            }
        }
    }
    let mut max_rel_err_pool_vs_template = 0.0f64;
    for (idx, (_slot, _intercept, coeffs)) in active.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            n_state,
            "stage {stage}: cut-state projection is not identity — map projected dims \
             before trusting the alias arithmetic"
        );
        let row = base_row_count + idx;
        for jj in 0..n_state {
            let col = out_cols[jj];
            let expected = -coeffs[jj] * col_scale_at(col);
            let actual = cut_entries.get(&(row, col)).copied().unwrap_or(0.0);
            max_rel_err_pool_vs_template =
                max_rel_err_pool_vs_template.max(rel_err(actual, expected));
        }
    }
    let extra_cols: BTreeSet<usize> = cut_entries
        .keys()
        .filter(|(_r, c)| !out_col_to_dim.contains_key(c))
        .map(|&(_r, c)| c)
        .collect();
    assert!(
        extra_cols.len() <= 1,
        "stage {stage}: cut rows touch more than one non-state column: {extra_cols:?}"
    );
    let theta_col = extra_cols.iter().next().copied();

    // -- per-solve numerical identities --

    let mut max_rel_err_full_support = 0.0f64;
    let mut max_rel_err_chain_rule = 0.0f64;
    let mut solves = 0usize;
    let mut gemv = vec![0.0f64; n_state];

    for m in 0..n_trials {
        let x_hat: &[f64] = &forward_probe.records[m * num_stages + (stage - 1)].state;
        let forward_basis: &CapturedBasis = forward_probe
            .basis_store
            .get(m, NodePos(stage))
            .unwrap_or_else(|| panic!("no forward-captured basis at (trial {m}, stage {stage})"));

        ws.solver.reset_solver_state();
        ws.solver.load_model(template);

        for omega_position in 0..n_openings {
            let omega = solve_order[omega_position] as usize;
            let raw_noise = opening_tree.opening(stage, omega);
            patch_backward_opening_for_probe(
                ws,
                &ctx,
                &training_ctx,
                StageIdx(stage),
                x_hat,
                raw_noise,
            );

            let stored_basis = (omega_position == 0).then_some(forward_basis);
            let view = solve_stage_for_probe(
                ws,
                &ctx,
                pool_ref,
                stored_basis,
                StageIdx(stage),
                m,
                NodeId(stage as i32),
            )
            .expect("stage solve must not error on cobre_rodada");
            let rc = view.reduced_costs;
            let y = view.dual;
            solves += 1;

            for j in 0..n_state {
                let cin = in_cols[j];
                let mut manual = template.objective[cin];
                for (r, v) in csc_col(cin) {
                    manual -= v * y[r];
                }
                max_rel_err_full_support = max_rel_err_full_support.max(rel_err(manual, rc[cin]));
            }

            // The fold's own kernel: g = Σ_c y_c · coeffs_c over resident cuts,
            // then per lag dim the z-row dual term plus the aliased g entry.
            gemv.fill(0.0);
            for (idx, (_slot, _intercept, coeffs)) in active.iter().enumerate() {
                let yc = y[base_row_count + idx];
                if yc != 0.0 {
                    for (g, &b) in gemv.iter_mut().zip(coeffs.iter()) {
                        *g += yc * b;
                    }
                }
            }
            for j in n..n_state {
                let cin = in_cols[j];
                let cs = col_scale_at(cin);
                let mut reconstructed = 0.0f64;
                for (r, v) in csc_col(cin) {
                    if r < base_row_count {
                        reconstructed -= (v / cs) * y[r];
                    }
                }
                if let Some(&alias) = out_col_to_dim.get(&cin) {
                    reconstructed += gemv[alias];
                }
                max_rel_err_chain_rule =
                    max_rel_err_chain_rule.max(rel_err(reconstructed, rc[cin] / cs));
            }
        }
    }

    LagFoldStats {
        stage,
        solves,
        resident_cuts,
        max_rel_err_full_support,
        max_rel_err_pool_vs_template,
        max_rel_err_chain_rule,
        aliased_lag_dims,
        unaliased_lag_dims,
        theta_col,
        storage_support_families,
        lag_support_families,
        z_col_support_families,
    }
}

/// Lag-fold feasibility probe: trains one checkpoint of the `cobre_rodada`
/// case (cut selection off, so resident-cut mass is production-shaped), then
/// verifies on every replayed backward solve that the pinned inflow-lag
/// columns couple to the LP through exactly three channels (their z-inflow
/// row, their water-balance row, and the resident cut rows) and that their
/// reduced costs — the Benders cut coefficients — are reproduced bit-tightly
/// from row duals and pool coefficients alone. See [`lag_fold_check_stage`]
/// for the four identities.
#[test]
#[ignore = "slow: trains a real checkpoint against the cobre_rodada case; \
            run with --run-ignored --release"]
fn lag_fold_chain_rule_identity() {
    let case = case_dir();
    let forward_passes: u32 = 8;
    let checkpoint: u32 = 10;
    let threads: usize = 8;

    let ckpt = train_checkpoint(&case, forward_passes, checkpoint, threads, true);
    let fp = run_extra_forward_pass(
        &ckpt.setup,
        &ckpt.frozen_templates,
        forward_passes,
        u64::from(checkpoint),
        threads,
    );

    for &stage in &[5usize, 32, 60] {
        let stats = lag_fold_check_stage(&ckpt.setup, &ckpt.frozen_templates, stage, &fp);
        println!(
            "stage {}: solves={} resident_cuts={} | rc-vs-Aᵀy max rel {:.3e} | \
             pool-vs-template max rel {:.3e} | chain-rule max rel {:.3e} | \
             aliased/unaliased lag dims {}/{} | theta_col={:?} | \
             lag support {:?} | z-col support {:?} | storage support {:?}",
            stats.stage,
            stats.solves,
            stats.resident_cuts,
            stats.max_rel_err_full_support,
            stats.max_rel_err_pool_vs_template,
            stats.max_rel_err_chain_rule,
            stats.aliased_lag_dims,
            stats.unaliased_lag_dims,
            stats.theta_col,
            stats.lag_support_families,
            stats.z_col_support_families,
            stats.storage_support_families,
        );
        assert!(
            stats.max_rel_err_full_support <= 1e-6,
            "stage {stage}: rc != objective − Aᵀy beyond tolerance \
             (max rel {:.3e}) — the solver reports transformed duals/rcs",
            stats.max_rel_err_full_support
        );
        assert!(
            stats.max_rel_err_pool_vs_template <= 1e-12,
            "stage {stage}: frozen-template cut entries drift from pool coefficients \
             (max rel {:.3e})",
            stats.max_rel_err_pool_vs_template
        );
        assert!(
            stats.max_rel_err_chain_rule <= 1e-6,
            "stage {stage}: chain-rule reconstruction diverges from the recorded \
             cut-coefficient fingerprint (max rel {:.3e})",
            stats.max_rel_err_chain_rule
        );
    }
}

// ---------------------------------------------------------------------------
// Lag-fold static A/B: original vs cut-row-folded twin
// ---------------------------------------------------------------------------

/// Build this trial's folded twin: remove every pinned-lag-column entry from
/// the cut rows and absorb it into the cut-row bounds at the trial's pinned
/// (scaled) lag values. Base matrix, columns, pins, and row/column counts
/// are untouched, so the production patch path and the forward-basis warm
/// start drive the twin verbatim; only cut-row density changes (the fold's
/// dominant claimed effect — the measured speedup is therefore a lower
/// bound on a full fold, which would also drop the lag columns and their
/// two base-row entries each).
fn fold_cut_rows_for_trial(
    template: &StageTemplate,
    base_row_count: usize,
    lag_cols: &BTreeMap<usize, f64>,
) -> StageTemplate {
    let mut folded = template.clone();

    let mut row_delta = vec![0.0f64; template.num_rows];
    let mut new_col_starts: Vec<i32> = Vec::with_capacity(template.num_cols + 1);
    let mut new_row_indices: Vec<i32> = Vec::with_capacity(template.num_nz);
    let mut new_values: Vec<f64> = Vec::with_capacity(template.num_nz);
    new_col_starts.push(0);
    for c in 0..template.num_cols {
        let s = template.col_starts[c] as usize;
        let e = template.col_starts[c + 1] as usize;
        let pinned = lag_cols.get(&c);
        for k in s..e {
            let r = template.row_indices[k] as usize;
            match pinned {
                Some(&x_scaled) if r >= base_row_count => {
                    row_delta[r] += template.values[k] * x_scaled;
                }
                _ => {
                    new_row_indices.push(template.row_indices[k]);
                    new_values.push(template.values[k]);
                }
            }
        }
        new_col_starts.push(new_row_indices.len() as i32);
    }
    for r in base_row_count..template.num_rows {
        if folded.row_lower[r].is_finite() {
            folded.row_lower[r] -= row_delta[r];
        }
        if folded.row_upper[r].is_finite() {
            folded.row_upper[r] -= row_delta[r];
        }
    }
    folded.num_nz = new_values.len();
    folded.col_starts = new_col_starts;
    folded.row_indices = new_row_indices;
    folded.values = new_values;
    folded
}

/// One A/B row: chain totals for one (checkpoint, stage), both arms.
struct LagFoldAbStats {
    checkpoint: u32,
    stage: usize,
    resident_cuts: usize,
    nnz_orig: usize,
    nnz_folded: usize,
    solves_per_arm: usize,
    wall_orig_ms: f64,
    wall_folded_ms: f64,
    hop_wall_orig_ms: f64,
    hop_wall_folded_ms: f64,
    pivots_orig: u64,
    pivots_folded: u64,
    max_obj_rel_diff: f64,
}

/// Replay every trial's opening chain twice — original frozen template vs
/// its per-trial folded twin (`fold_cut_rows_for_trial`) — through the
/// identical patch/warm-start/solve flow, recording per-solve wall and
/// pivots for both arms and asserting per-opening objective equality (the
/// fold is an exact substitution of the pinned lag values; a bound bug or a
/// wrong pin reconstruction surfaces here, not as a silent timing artifact).
fn run_lag_fold_ab_stage(
    setup: &StudySetup,
    frozen_templates: &[StageTemplate],
    checkpoint: u32,
    stage: usize,
    forward_probe: &ForwardProbe,
) -> LagFoldAbStats {
    assert!(stage >= 1, "the backward pass never solves stage 0");

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws: &mut SolverWorkspace<ActiveSolver> = &mut pool.workspaces[0];

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let state_space = setup.stage_state();
    let template = &frozen_templates[stage];
    let pool_ref: &CutPool = &setup.fcf.pools[stage];
    let opening_tree = training_ctx.stochastic.tree_view();
    let n_openings = opening_tree.n_openings(stage);
    let solve_order = opening_tree.solve_order_data(stage);

    let num_stages = setup.num_stages();
    let n_trials = forward_probe.records.len() / num_stages;
    let base_row_count = ctx.templates[stage].num_rows;
    let resident_cuts = template.num_rows - base_row_count;

    let n_state = state_space.n_state;
    let n = state_space.hydro_count;

    let mut stats = LagFoldAbStats {
        checkpoint,
        stage,
        resident_cuts,
        nnz_orig: template.num_nz,
        nnz_folded: 0,
        solves_per_arm: 0,
        wall_orig_ms: 0.0,
        wall_folded_ms: 0.0,
        hop_wall_orig_ms: 0.0,
        hop_wall_folded_ms: 0.0,
        pivots_orig: 0,
        pivots_folded: 0,
        max_obj_rel_diff: 0.0,
    };

    let mut objectives_orig = vec![0.0f64; n_openings];

    for m in 0..n_trials {
        let x_hat: &[f64] = &forward_probe.records[m * num_stages + (stage - 1)].state;
        let forward_basis: &CapturedBasis = forward_probe
            .basis_store
            .get(m, NodePos(stage))
            .unwrap_or_else(|| panic!("no forward-captured basis at (trial {m}, stage {stage})"));

        // Pinned scaled lag values: the pin writes `v_orig / col_scale` (the
        // Benders subgradient contract's inverse direction).
        let lag_cols: BTreeMap<usize, f64> = (n..n_state)
            .map(|j| {
                let c = state_space
                    .state_to_lp_incoming_column(StateDim::new(j))
                    .get();
                let cs = template.col_scale.get(c).copied().unwrap_or(1.0);
                (c, x_hat[j] / cs)
            })
            .collect();
        let folded = fold_cut_rows_for_trial(template, base_row_count, &lag_cols);
        stats.nnz_folded = folded.num_nz;

        for (arm_folded, arm_template) in [(false, template), (true, &folded)] {
            ws.solver.reset_solver_state();
            ws.solver.load_model(arm_template);

            for omega_position in 0..n_openings {
                let omega = solve_order[omega_position] as usize;
                let raw_noise = opening_tree.opening(stage, omega);
                patch_backward_opening_for_probe(
                    ws,
                    &ctx,
                    &training_ctx,
                    StageIdx(stage),
                    x_hat,
                    raw_noise,
                );

                let stored_basis = (omega_position == 0).then_some(forward_basis);
                let view = solve_stage_for_probe(
                    ws,
                    &ctx,
                    pool_ref,
                    stored_basis,
                    StageIdx(stage),
                    m,
                    NodeId(stage as i32),
                )
                .expect("stage solve must not error on cobre_rodada");
                let wall_ms = view.solve_time_seconds * 1_000.0;
                let pivots = view.iterations;
                let objective = view.objective;

                if arm_folded {
                    let rel = (objective - objectives_orig[omega_position]).abs()
                        / objectives_orig[omega_position].abs().max(1.0);
                    stats.max_obj_rel_diff = stats.max_obj_rel_diff.max(rel);
                    assert!(
                        rel <= 1e-7,
                        "checkpoint {checkpoint} stage {stage} trial {m} chain_pos \
                         {omega_position}: folded objective {objective} diverges from \
                         original {} (rel {rel:.3e}) — the fold is not an exact \
                         substitution (bound arithmetic or pin reconstruction bug)",
                        objectives_orig[omega_position]
                    );
                    stats.wall_folded_ms += wall_ms;
                    stats.pivots_folded += pivots;
                    if omega_position > 0 {
                        stats.hop_wall_folded_ms += wall_ms;
                    }
                } else {
                    objectives_orig[omega_position] = objective;
                    stats.wall_orig_ms += wall_ms;
                    stats.pivots_orig += pivots;
                    if omega_position > 0 {
                        stats.hop_wall_orig_ms += wall_ms;
                    }
                    stats.solves_per_arm += 1;
                }
            }
        }
    }

    stats
}

/// Lag-fold static A/B (the fold's go/no-go performance readout): trains
/// checkpoints {3, 10, 30} of the `cobre_rodada` case with cut selection off
/// (resident cuts ~24/80/240 — the density curve up to production-shaped
/// mass), then replays every trial's backward chain at stages {5, 32, 60}
/// on the original frozen template and on its per-trial cut-row-folded twin,
/// asserting per-opening objective equality and reporting wall/pivot ratios
/// per (checkpoint, stage).
#[test]
#[ignore = "slow (~1h): trains three checkpoints against the cobre_rodada case; \
            run with --run-ignored --release"]
fn lag_fold_static_ab() {
    let case = case_dir();
    let forward_passes: u32 = 8;
    let threads: usize = 8;

    println!(
        "ckpt stage resident nnz_orig nnz_folded solves/arm | wall_orig_ms wall_folded_ms \
         speedup | hop_speedup | pivots_orig pivots_folded pivot_ratio | wall/pivot ratio | \
         max_obj_rel"
    );
    for &checkpoint in &[3u32, 10, 30] {
        let ckpt = train_checkpoint(&case, forward_passes, checkpoint, threads, true);
        let fp = run_extra_forward_pass(
            &ckpt.setup,
            &ckpt.frozen_templates,
            forward_passes,
            u64::from(checkpoint),
            threads,
        );
        for &stage in &[5usize, 32, 60] {
            let s =
                run_lag_fold_ab_stage(&ckpt.setup, &ckpt.frozen_templates, checkpoint, stage, &fp);
            let speedup = s.wall_orig_ms / s.wall_folded_ms.max(1e-9);
            let hop_speedup = s.hop_wall_orig_ms / s.hop_wall_folded_ms.max(1e-9);
            let pivot_ratio = s.pivots_orig as f64 / (s.pivots_folded as f64).max(1.0);
            let wpp_ratio = (s.wall_orig_ms / (s.pivots_orig as f64).max(1.0))
                / (s.wall_folded_ms / (s.pivots_folded as f64).max(1.0)).max(1e-12);
            println!(
                "{} {} {} {} {} {} | {:.1} {:.1} {:.3}x | {:.3}x | {} {} {:.3} | {:.3}x | {:.2e}",
                s.checkpoint,
                s.stage,
                s.resident_cuts,
                s.nnz_orig,
                s.nnz_folded,
                s.solves_per_arm,
                s.wall_orig_ms,
                s.wall_folded_ms,
                speedup,
                hop_speedup,
                s.pivots_orig,
                s.pivots_folded,
                pivot_ratio,
                wpp_ratio,
                s.max_obj_rel_diff,
            );
        }
    }
}
