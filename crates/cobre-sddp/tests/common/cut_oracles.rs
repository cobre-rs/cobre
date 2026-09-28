//! Cut-validity, mask-soundness, and tightness oracles for stored Benders cuts, run over an in-code or committed study.

#![allow(clippy::expect_used, clippy::panic, clippy::cast_precision_loss)]

use cobre_io::Config;
use cobre_io::config::{RowSelectionConfig, StoppingRuleConfig};
use cobre_sddp::indexer::CutSlot;
use cobre_sddp::setup::{NodePos, StageIdx};
use cobre_sddp::test_support::{
    stage_state_box_bounds, write_backward_opening_outcome_at_canonical_state_for_probe,
};
use cobre_sddp::workspace::StageContext;
use cobre_sddp::{SddpError, StudySetup};
use cobre_solver::{ActiveSolver, StageTemplate};
use cobre_stochastic::OpeningTreeView;

use super::StubComm;

const ORACLE_ITERATIONS: u32 = 3;
const ORACLE_RANDOM_POINTS: usize = 5;
const ORACLE_SEED: u64 = 0x5EED_C0BE_5EED_C0BE;
const ORACLE_ABS_TOL: f64 = 1e-6;
const ORACLE_REL_TOL: f64 = 1e-7;
const MASK_ABS_TOL: f64 = 1e-9;
const MASK_REL_TOL: f64 = 1e-9;
const ORACLE_MIN_FEASIBLE_POINTS: usize = 3;

/// Sets the training loop to run exactly [`ORACLE_ITERATIONS`] iterations with
/// cut selection disabled; every other config field is left as the study's own.
pub fn apply_oracle_config(config: &mut Config) {
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: ORACLE_ITERATIONS,
    }]);
    config.training.cut_selection = RowSelectionConfig::default();
}

/// Counters and violation lists from one [`run_cut_oracles`] run.
pub struct CutOracleReport {
    /// Producer-stage pools examined (those with at least one active cut).
    pub pools_checked: usize,
    /// Total active cuts across every checked pool.
    pub cuts_checked: usize,
    /// Total feasible sample points examined across every checked pool.
    pub points_checked: usize,
    /// Box-point probe solves that came back infeasible, skipped rather than
    /// counted as a violation.
    pub infeasible_sample_points: usize,
    /// Total off-mask cut-state slots examined across every checked pool.
    pub off_mask_slots_checked: usize,
    /// The minimum, over every checked pool, of its own feasible-point count.
    pub min_feasible_points: usize,
    /// Cut-validity violations.
    pub validity: Vec<String>,
    /// Tightness violations at a final-iteration trial point.
    pub tightness: Vec<String>,
    /// Mask-soundness violations.
    pub mask: Vec<String>,
}

impl Default for CutOracleReport {
    fn default() -> Self {
        Self {
            pools_checked: 0,
            cuts_checked: 0,
            points_checked: 0,
            infeasible_sample_points: 0,
            off_mask_slots_checked: 0,
            min_feasible_points: usize::MAX,
            validity: Vec::new(),
            tightness: Vec::new(),
            mask: Vec::new(),
        }
    }
}

impl CutOracleReport {
    /// Prints the report's counters, then panics naming every violation, or on
    /// vacuity (no pool checked, no cut checked, or a checked pool with fewer
    /// than [`ORACLE_MIN_FEASIBLE_POINTS`] feasible sample points).
    pub fn assert_sound(&self, label: &str) {
        eprintln!(
            "{label}: pools_checked={} cuts_checked={} points_checked={} \
             infeasible_sample_points={} off_mask_slots_checked={} min_feasible_points={}",
            self.pools_checked,
            self.cuts_checked,
            self.points_checked,
            self.infeasible_sample_points,
            self.off_mask_slots_checked,
            self.min_feasible_points,
        );
        assert!(
            self.pools_checked > 0,
            "{label}: vacuity guard: no pool was checked"
        );
        assert!(
            self.cuts_checked > 0,
            "{label}: vacuity guard: no active cut was checked"
        );
        assert!(
            self.min_feasible_points >= ORACLE_MIN_FEASIBLE_POINTS,
            "{label}: vacuity guard: a checked pool had only {} feasible sample points (need >= {ORACLE_MIN_FEASIBLE_POINTS})",
            self.min_feasible_points
        );

        let mut violations: Vec<String> = Vec::new();
        violations.extend(self.validity.iter().map(|v| format!("[validity] {v}")));
        violations.extend(self.tightness.iter().map(|v| format!("[tightness] {v}")));
        violations.extend(self.mask.iter().map(|v| format!("[mask] {v}")));
        assert!(
            violations.is_empty(),
            "{label}: cut oracle violations:\n{}",
            violations.join("\n")
        );
    }
}

/// Per-run borrows [`run_cut_oracles`] threads into [`check_pool`]/[`evaluate_point`].
struct OracleCtx<'a> {
    setup: &'a StudySetup,
    tree_view: OpeningTreeView<'a>,
    /// One per pool (`pool_id == stage` on a chain), each its base template
    /// plus that pool's own active cuts as rows — unlike the setup's own
    /// `StageContext::templates`, which carries no cuts and would leave a
    /// successor's theta unconstrained if loaded directly.
    frozen_templates: &'a [StageTemplate],
}

/// The three observations [`evaluate_point`] gathers from one sample point's
/// probe solves, aggregated across every opening of the successor stage.
struct PointEval {
    canonical_x: Vec<f64>,
    q_hat: f64,
    per_opening_coeffs: Vec<Vec<f64>>,
}

/// Mirrors `production::fpha_fitting::rng::SplitMix64`'s algorithm and constants
/// byte-for-byte, hand-rolled here rather than reused: that generator is
/// `pub(crate)` inside a private submodule with no re-export reaching this
/// integration-test crate, and `box_random`'s sampled points must not move with
/// a `rand` upgrade the way a `rand::rngs::StdRng`-seeded draw could.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

fn box_corner(lo: &[f64], hi: &[f64], fallback: &[f64], pick: fn(f64, f64) -> f64) -> Vec<f64> {
    (0..lo.len())
        .map(|j| {
            if lo[j].is_finite() && hi[j].is_finite() {
                pick(lo[j], hi[j])
            } else {
                fallback[j]
            }
        })
        .collect()
}

fn box_random(lo: &[f64], hi: &[f64], fallback: &[f64], rng: &mut SplitMix64) -> Vec<f64> {
    (0..lo.len())
        .map(|j| {
            if lo[j].is_finite() && hi[j].is_finite() {
                lo[j] + rng.next_unit() * (hi[j] - lo[j])
            } else {
                fallback[j]
            }
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

fn q_tol(a: f64, b: f64) -> f64 {
    ORACLE_ABS_TOL + ORACLE_REL_TOL * a.abs().max(b.abs())
}

fn check_mask(
    label: &str,
    t: usize,
    idx: usize,
    off_mask: &[usize],
    coeffs: &[f64],
    violations: &mut Vec<String>,
) {
    let scale = max_abs(coeffs);
    for &j in off_mask {
        if coeffs[j].abs() > MASK_ABS_TOL + MASK_REL_TOL * scale {
            violations.push(format!(
                "{label}: pool {t} point {idx}: off-mask slot {j} has |coefficient|={} (scale {scale})",
                coeffs[j]
            ));
        }
    }
}

/// Probe every opening of stage `s = t + 1` at sample point `x` (stage `t`'s
/// outgoing state), aggregating the equal-weighted successor value `Q̂(x̃)` and
/// collecting each opening's raw (unaggregated) cut coefficients for the mask
/// child-side check.
fn evaluate_point(
    label: &str,
    occ: &OracleCtx<'_>,
    ws: &mut cobre_sddp::workspace::SolverWorkspace<ActiveSolver>,
    t: usize,
    s: usize,
    x: &[f64],
) -> Result<PointEval, SddpError> {
    let stage_ctx = StageContext {
        templates: occ.frozen_templates,
        ..occ.setup.stage_ctx()
    };
    let training_ctx = occ.setup.training_ctx();
    let successor_pool = &occ.setup.fcf.pools[s];
    let cut_state = &training_ctx.cut_state_layouts[t];
    let node_id = occ.setup.inputs.node_graph.node_ids[NodePos(s)];
    let n_openings = occ.tree_view.n_openings(s);

    let mut canonical_x: Option<Vec<f64>> = None;
    let mut q_hat = 0.0_f64;
    let weight = 1.0 / n_openings as f64;
    let mut per_opening_coeffs = Vec::with_capacity(n_openings);

    for o in 0..n_openings {
        let raw_noise = occ.tree_view.opening(s, o);
        let probe = write_backward_opening_outcome_at_canonical_state_for_probe(
            ws,
            &stage_ctx,
            &training_ctx,
            successor_pool,
            cut_state,
            StageIdx(s),
            node_id,
            x,
            raw_noise,
        )?;
        assert_eq!(
            probe.outcome.coefficients.len(),
            occ.setup.fcf.pools[t].state_dimension,
            "{label}: pool {t} coefficient length does not match cut_state_layouts[{t}]'s pairing"
        );
        if canonical_x.is_none() {
            canonical_x = Some(probe.canonical_x_hat.clone());
        }
        q_hat += weight * probe.outcome.objective_value;
        per_opening_coeffs.push(probe.outcome.coefficients);
    }

    Ok(PointEval {
        canonical_x: canonical_x.expect("evaluate_point: n_openings must be >= 1"),
        q_hat,
        per_opening_coeffs,
    })
}

// Rationale (too_many_arguments): bundles the per-run oracle context, this
// pool's own trial/box points, the solver workspace, and the accumulating
// report; a second wrapper struct would just relocate the same values.
#[allow(clippy::too_many_arguments)]
fn check_pool(
    label: &str,
    t: usize,
    s: usize,
    occ: &OracleCtx<'_>,
    trial_points: &[Vec<f64>],
    box_points: &[Vec<f64>],
    ws: &mut cobre_sddp::workspace::SolverWorkspace<ActiveSolver>,
    report: &mut CutOracleReport,
) {
    let pool_t = &occ.setup.fcf.pools[t];
    report.pools_checked += 1;
    report.cuts_checked += pool_t.active_count();

    let training_ctx = occ.setup.training_ctx();
    let cut_state_t = &training_ctx.cut_state_layouts[t];
    let nonzero = &occ.setup.stage_state().nonzero_state_indices;
    let off_mask: Vec<usize> = (0..cut_state_t.n_slots())
        .filter(|&j| {
            nonzero
                .binary_search(&cut_state_t.global_state_index(CutSlot::new(j)))
                .is_err()
        })
        .collect();
    report.off_mask_slots_checked += off_mask.len();

    let n_trial = trial_points.len();
    let mut feasible_count = 0usize;

    for (idx, x) in trial_points.iter().chain(box_points).enumerate() {
        let is_trial = idx < n_trial;
        match evaluate_point(label, occ, ws, t, s, x) {
            Ok(eval) => {
                feasible_count += 1;
                report.points_checked += 1;

                for coeffs in &eval.per_opening_coeffs {
                    check_mask(label, t, idx, &off_mask, coeffs, &mut report.mask);
                }

                let mut proj = vec![0.0; cut_state_t.n_slots()];
                for (j, slot) in proj.iter_mut().enumerate() {
                    *slot = eval.canonical_x[cut_state_t.global_state_index(CutSlot::new(j)).get()];
                }

                for (_, intercept, beta) in pool_t.active_cuts() {
                    let lhs = intercept + dot(beta, &proj);
                    let bound = q_tol(eval.q_hat, lhs);
                    if lhs > eval.q_hat + bound {
                        report.validity.push(format!(
                            "{label}: pool {t} point {idx}: cut value {lhs} exceeds \
                             Q_hat={} + tol={bound}",
                            eval.q_hat
                        ));
                    }
                }

                if is_trial {
                    let evaluated = pool_t.evaluate_at_state(&proj);
                    let bound = q_tol(evaluated, eval.q_hat);
                    if (evaluated - eval.q_hat).abs() > bound {
                        report.tightness.push(format!(
                            "{label}: pool {t} trial point {idx}: evaluate_at_state={evaluated} \
                             Q_hat={} tol={bound}",
                            eval.q_hat
                        ));
                    }
                }
            }
            Err(SddpError::Infeasible { .. }) => {
                if is_trial {
                    report
                        .validity
                        .push(format!("{label}: pool {t} trial point {idx} is infeasible"));
                } else {
                    report.infeasible_sample_points += 1;
                }
            }
            Err(other) => {
                panic!("{label}: pool {t} point {idx} probe failed unexpectedly: {other}");
            }
        }
    }

    for (_, _, beta) in pool_t.active_cuts() {
        let scale = max_abs(beta);
        for &j in &off_mask {
            if beta[j].abs() > MASK_ABS_TOL + MASK_REL_TOL * scale {
                report.mask.push(format!(
                    "{label}: pool {t} stored cut off-mask slot {j}: |beta|={} exceeds tol \
                     (scale {scale})",
                    beta[j]
                ));
            }
        }
    }

    report.min_feasible_points = report.min_feasible_points.min(feasible_count);
}

/// Trains `setup` with `HiGHS` on a single rank/thread under
/// [`apply_oracle_config`]'s settings, then gates every stored cut on the
/// validity, tightness, and mask-soundness oracles over each producer stage
/// `t` whose pool has at least one active cut.
///
/// # Panics
///
/// Panics if `setup`'s node graph is not a chain, if training errors or
/// deactivates a cut, or if a probe solve fails for a reason other than
/// infeasibility.
pub fn run_cut_oracles(label: &str, mut setup: StudySetup) -> CutOracleReport {
    let n_stages = setup.inputs.stage_data.stage_templates.templates.len();
    assert_eq!(
        setup.inputs.node_graph.nodes.len(),
        n_stages,
        "{label}: run_cut_oracles requires a chain graph (one node per stage)"
    );
    for pos in 0..n_stages {
        assert_eq!(
            setup.inputs.node_graph.nodes[NodePos(pos)].stage.0,
            pos,
            "{label}: node {pos} is not chain-ordered onto stage {pos}"
        );
    }

    setup.set_export_states(true);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("run_cut_oracles: ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("run_cut_oracles: training must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: training must not error: {:?}",
        outcome.error
    );
    assert_eq!(
        setup.fcf.total_active_cuts(),
        setup.fcf.total_generated_cuts(),
        "{label}: a cut was deactivated during training — the oracle assumes cuts only grow"
    );

    let archive = outcome
        .result
        .visited_archive
        .expect("run_cut_oracles: set_export_states(true) must produce a visited-states archive");
    let frozen_templates = outcome
        .result
        .frozen_templates
        .expect("run_cut_oracles: training must populate frozen_templates");
    let forward_passes = setup.fcf.forward_passes as usize;

    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("run_cut_oracles: create_workspace_pool must succeed");
    let ws = &mut workspace_pool.workspaces[0];

    let occ = OracleCtx {
        setup: &setup,
        tree_view: setup.training_ctx().stochastic.tree_view(),
        frozen_templates: &frozen_templates,
    };

    let mut report = CutOracleReport::default();

    for t in 0..n_stages.saturating_sub(1) {
        if setup.fcf.pools[t].active_count() == 0 {
            continue;
        }
        let s = t + 1;
        let (lo, hi) = stage_state_box_bounds(&setup, t);
        let n_state = lo.len();

        let node_pos = NodePos(t);
        assert_eq!(
            archive.packing_stride(),
            n_state,
            "{label}: archive packing stride must equal the state-box length"
        );
        let count = archive.count(node_pos);
        assert!(
            count >= forward_passes,
            "{label}: node {t} archived {count} states, fewer than forward_passes ({forward_passes})"
        );
        let flat = archive.states_for_node(node_pos);
        let start = (count - forward_passes) * n_state;
        let trial_points: Vec<Vec<f64>> = (0..forward_passes)
            .map(|m| {
                let base = start + m * n_state;
                flat[base..base + n_state].to_vec()
            })
            .collect();

        let mut box_points = vec![
            box_corner(&lo, &hi, &trial_points[0], f64::midpoint),
            box_corner(&lo, &hi, &trial_points[0], |l, _h| l),
            box_corner(&lo, &hi, &trial_points[0], |_l, h| h),
        ];
        let mut rng = SplitMix64::new(ORACLE_SEED ^ (t as u64));
        for _ in 0..ORACLE_RANDOM_POINTS {
            box_points.push(box_random(&lo, &hi, &trial_points[0], &mut rng));
        }

        check_pool(
            label,
            t,
            s,
            &occ,
            &trial_points,
            &box_points,
            ws,
            &mut report,
        );
    }

    report
}
