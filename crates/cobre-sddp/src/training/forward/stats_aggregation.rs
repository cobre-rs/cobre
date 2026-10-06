//! Forward-pass upper-bound statistics aggregation.
//!
//! Owns `sync_forward`: the cross-rank `allgatherv` plus the per-source
//! assembly of the upper bound. Under a sampled forward it is the canonical-order
//! `WelfordAccumulator` summation (sample mean + 95% CI); under an enumerated
//! forward it is the exact `Σ wᵢ·cᵢ` compensated reduction
//! ([`weighted_cost_reduction`]). Both iterate the gathered costs in one fixed
//! global order — the determinism contract that makes the result
//! rank-count-invariant.

use std::time::Instant;

use cobre_comm::{Communicator, per_rank_counts, prefix_displs};
use cobre_core::WelfordAccumulator;

use super::{ForwardResult, SyncResult};
use crate::error::SddpError;
use crate::risk_measure::{RiskMeasure, RiskMeasureScratch};
use crate::setup::node_graph::NestedUbTopology;
// Rationale: imported solely so the `[run_forward_pass]` intra-doc link in
// `sync_forward`'s rustdoc resolves; the function lives in the parent `mod.rs`.
#[cfg(any(test, feature = "test-support"))]
#[allow(unused_imports)]
use super::run_forward_pass;

/// Which upper-bound estimator [`sync_forward`] assembles from the gathered
/// forward-pass costs.
#[derive(Debug)]
pub enum ForwardBound<'a> {
    /// Sampled forward: Welford sample mean, standard deviation, and 95% CI
    /// half-width.
    Statistical,
    /// Enumerated forward under a risk-neutral (effective `Expectation`) measure:
    /// the exact probability-weighted `Σ wᵢ·cᵢ` over `path_weights` (canonical
    /// order); standard deviation and CI half-width are `0` — a deduplicated
    /// enumeration carries no sampling distribution.
    Exact {
        /// Per-path probability weights, one per gathered cost, in canonical order.
        path_weights: &'a [f64],
    },
    /// Enumerated forward under a uniform effective `CVaR`: the exact NESTED
    /// risk-adjusted bound ([`nested_ub_recursion`]) over the enumerated tree,
    /// which the end-of-horizon `Σ wᵢ·cᵢ` cannot represent. Std/CI are `0`.
    NestedRisk {
        /// This rank's per-path per-stage raw immediate costs, `num_stages` per
        /// path in canonical path order.
        path_stage_costs: &'a [f64],
        /// The enumerated tree's precomputed reduction structure.
        topology: &'a NestedUbTopology,
        /// Per-stage cumulative discount factors.
        cumulative_discounts: &'a [f64],
        /// The uniform effective measure applied at every node.
        risk_measure: RiskMeasure,
        /// Stages per path (the `path_stage_costs` stride).
        num_stages: usize,
        /// Persistent gather and recursion scratch, reused across iterations.
        scratch: &'a mut NestedUbScratch,
    },
}

/// Gather layout, gathered costs and recursion buffers of
/// [`ForwardBound::NestedRisk`], grown on first use and reused after.
#[derive(Debug, Default)]
pub struct NestedUbScratch {
    layout_for: Option<(usize, usize, usize)>,
    stage_counts: Vec<usize>,
    stage_displs: Vec<usize>,
    gathered: Vec<f64>,
    recursion: NestedUbRecursionScratch,
}

impl NestedUbScratch {
    /// Size the per-path-per-stage gather, recomputing the rank layout only when
    /// the `(total_forward_passes, num_ranks, num_stages)` shape changes.
    fn prepare_gather(&mut self, total_forward_passes: usize, num_ranks: usize, num_stages: usize) {
        let layout = (total_forward_passes, num_ranks, num_stages);
        if self.layout_for != Some(layout) {
            self.stage_counts = per_rank_counts(total_forward_passes, num_ranks);
            for count in &mut self.stage_counts {
                *count *= num_stages;
            }
            self.stage_displs = prefix_displs(&self.stage_counts);
            self.layout_for = Some(layout);
        }
        self.gathered.clear();
        self.gathered.resize(self.stage_counts.iter().sum(), 0.0);
    }
}

/// Per-node working buffers of [`nested_ub_recursion`].
#[derive(Debug, Default)]
pub(crate) struct NestedUbRecursionScratch {
    node_cost: Vec<f64>,
    value: Vec<f64>,
    child_vals: Vec<f64>,
    child_probs: Vec<f64>,
    risk: RiskMeasureScratch,
}

/// Compensated (Neumaier) `Σ wᵢ·cᵢ` over paired cost/weight slices in slice-index
/// order.
///
/// The index order is fixed and decoupled from solve/gather order, so the
/// reduction is bit-identical across rank and thread counts; Neumaier
/// compensation (not a naive running sum) holds accuracy when the weighted terms
/// span wide magnitudes.
pub(crate) fn weighted_cost_reduction(costs: &[f64], weights: &[f64]) -> f64 {
    debug_assert_eq!(
        costs.len(),
        weights.len(),
        "weighted_cost_reduction: costs ({}) and weights ({}) must align 1:1",
        costs.len(),
        weights.len(),
    );
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for (&cost, &weight) in costs.iter().zip(weights.iter()) {
        let term = weight * cost;
        let t = sum + term;
        if sum.abs() >= term.abs() {
            compensation += (sum - t) + term;
        } else {
            compensation += (term - t) + sum;
        }
        sum = t;
    }
    sum + compensation
}

/// Aggregate one rank's forward-pass statistics across all MPI ranks.
///
/// `allgatherv`s `local.scenario_costs` into a canonical-order global buffer,
/// then assembles the upper bound per `bound`: [`ForwardBound::Statistical`]
/// computes the Welford sample mean/std/CI, [`ForwardBound::Exact`] the
/// probability-weighted `Σ wᵢ·cᵢ`. Both reduce in the fixed global order, so the
/// result is bit-identical regardless of rank count. In single-rank mode
/// `LocalBackend.allgatherv` is an identity copy, needing no special case.
///
/// The lower bound is **not** computed here; it is evaluated after the backward
/// pass adds new cuts to the FCF.
///
/// # Errors
///
/// Returns `Err(SddpError::Communication(_))` if the `allgatherv` call fails.
/// No partial results are produced on error.
pub fn sync_forward<C: Communicator>(
    local: &ForwardResult,
    comm: &C,
    total_forward_passes: usize,
    bound: ForwardBound<'_>,
) -> Result<SyncResult, SddpError> {
    let start = Instant::now();

    let num_ranks = comm.size();
    let my_rank = comm.rank();

    // NestedRisk gathers per-path-per-stage costs and recurses over the enumerated
    // tree; it needs neither the path-total gather nor the reduction below.
    if let ForwardBound::NestedRisk {
        path_stage_costs,
        topology,
        cumulative_discounts,
        risk_measure,
        num_stages,
        scratch,
    } = bound
    {
        scratch.prepare_gather(total_forward_passes, num_ranks, num_stages);
        debug_assert_eq!(
            path_stage_costs.len(),
            scratch.stage_counts[my_rank],
            "rank {my_rank}: path_stage_costs length {} != expected count {}",
            path_stage_costs.len(),
            scratch.stage_counts[my_rank],
        );
        comm.allgatherv(
            path_stage_costs,
            &mut scratch.gathered,
            &scratch.stage_counts,
            &scratch.stage_displs,
        )?;
        let ub = nested_ub_recursion(
            topology,
            &scratch.gathered,
            num_stages,
            cumulative_discounts,
            risk_measure,
            &mut scratch.recursion,
        );
        #[allow(clippy::cast_possible_truncation)]
        let sync_time_ms = start.elapsed().as_millis() as u64;
        return Ok(SyncResult {
            global_ub_mean: ub,
            global_ub_std: 0.0_f64,
            ci_95_half_width: 0.0_f64,
            sync_time_ms,
        });
    }

    // Per-rank path counts derived arithmetically from the total, so no
    // preliminary communication round is needed.
    let path_counts = per_rank_counts(total_forward_passes, num_ranks);
    let displs = prefix_displs(&path_counts);
    let global_n = path_counts.iter().sum::<usize>();
    debug_assert_eq!(
        global_n, total_forward_passes,
        "counts sum {global_n} != total_forward_passes {total_forward_passes}",
    );
    let mut global_costs = vec![0.0_f64; global_n];
    debug_assert_eq!(
        local.scenario_costs.len(),
        path_counts[my_rank],
        "rank {my_rank}: scenario_costs length {} != expected count {}",
        local.scenario_costs.len(),
        path_counts[my_rank],
    );
    comm.allgatherv(
        &local.scenario_costs,
        &mut global_costs,
        &path_counts,
        &displs,
    )?;

    let (mean, std_dev, ci_95) = match bound {
        ForwardBound::Statistical => {
            // Single canonical-order pass: every rank iterates global_costs in the
            // same order, so the statistics are bit-identical regardless of rank
            // count. Welford's online algorithm, not the two-pass naive formula,
            // avoids catastrophic cancellation when sum_sq ≈ n * mean^2; the full
            // gathered array is in hand, so no MPI Welford merge is needed.
            let mut welford = WelfordAccumulator::new();
            for &c in &global_costs {
                welford.update(c);
            }
            let mean = welford.mean();
            if global_n > 1 {
                (
                    mean,
                    welford.sample_std_dev(),
                    welford.sample_ci_95_half_width(),
                )
            } else {
                (mean, 0.0_f64, 0.0_f64)
            }
        }
        ForwardBound::Exact { path_weights } => {
            // Risk-neutral exact bound; std/CI are 0 because a deduplicated
            // enumeration has no sampling distribution — routing it through the
            // Welford accumulator as S samples would be a category error. Under an
            // effective CVaR the session selects `ForwardBound::NestedRisk` instead.
            let ub_exact = weighted_cost_reduction(&global_costs, path_weights);
            (ub_exact, 0.0_f64, 0.0_f64)
        }
        ForwardBound::NestedRisk { .. } => {
            unreachable!("NestedRisk is handled by the early return above")
        }
    };

    #[allow(clippy::cast_possible_truncation)]
    let sync_time_ms = start.elapsed().as_millis() as u64;

    Ok(SyncResult {
        global_ub_mean: mean,
        global_ub_std: std_dev,
        ci_95_half_width: ci_95,
        sync_time_ms,
    })
}

/// Nested backward risk recursion over the enumerated scenario tree, on the
/// gathered `global_path_stage_costs` (path-major, `num_stages` per path), using
/// the plan's precomputed [`NestedUbTopology`].
///
/// `Ṽ(n) = cum_d[stage(n)]·c(n) + ρ_children(Ṽ(child))`, `ρ` = `risk_measure`
/// over each node's children weighted by their conditional probabilities. The
/// root value is the exact nested risk-adjusted bound. Applying the measure once
/// to whole-path totals (the end-of-horizon form) is the wrong-but-compiling
/// alternative: for a nested measure it under-states the bound and can fall below
/// the nested lower bound. Reduces to the risk-neutral `Σ wᵢ·cᵢ` under
/// `Expectation` (nesting is linear there).
pub(crate) fn nested_ub_recursion(
    topology: &NestedUbTopology,
    global_path_stage_costs: &[f64],
    num_stages: usize,
    cumulative_discounts: &[f64],
    risk_measure: RiskMeasure,
    scratch: &mut NestedUbRecursionScratch,
) -> f64 {
    let NestedUbRecursionScratch {
        node_cost,
        value,
        child_vals,
        child_probs,
        risk,
    } = scratch;
    let n_nodes = topology.node_stage.len();
    // Per-node cumulative-discounted immediate cost from this iteration's gathered
    // costs, read through each node's representative path (idempotent across paths
    // sharing the node). The tree structure itself is precomputed on the plan.
    node_cost.clear();
    node_cost.resize(n_nodes, 0.0);
    for node in 0..n_nodes {
        let t = topology.node_stage[node];
        let cum_d = cumulative_discounts.get(t).copied().unwrap_or(1.0);
        node_cost[node] =
            cum_d * global_path_stage_costs[topology.node_path[node] * num_stages + t];
    }

    value.clear();
    value.resize(n_nodes, 0.0);
    for &node in &topology.valuation_order {
        let kids = &topology.children[node.0];
        let v_future = if kids.is_empty() {
            0.0
        } else {
            child_vals.clear();
            child_probs.clear();
            let p_node = topology.node_prob[node.0];
            for &c in kids {
                child_vals.push(value[c.0]);
                child_probs.push(topology.node_prob[c.0] / p_node);
            }
            risk_measure.evaluate_risk_into(child_vals, child_probs, risk)
        };
        value[node.0] = node_cost[node.0] + v_future;
    }

    // A single root under the one initial state is the norm; a defensive
    // multi-root graph reduces over the roots by their marginals.
    if let [only] = topology.roots.as_slice() {
        value[only.0]
    } else {
        child_vals.clear();
        child_probs.clear();
        for r in &topology.roots {
            child_vals.push(value[r.0]);
            child_probs.push(topology.node_prob[r.0]);
        }
        risk_measure.evaluate_risk_into(child_vals, child_probs, risk)
    }
}

#[cfg(test)]
mod tests {
    use cobre_comm::LocalBackend;

    use super::{ForwardBound, NestedUbScratch, sync_forward};
    use crate::risk_measure::RiskMeasure;
    use crate::setup::node_graph::{NestedUbTopology, NodePos, TypedVec};
    use crate::training::forward::ForwardResult;

    fn four_leaf_fan() -> NestedUbTopology {
        let parent: TypedVec<NodePos, Option<NodePos>> = vec![
            None,
            Some(NodePos(0)),
            Some(NodePos(0)),
            Some(NodePos(0)),
            Some(NodePos(0)),
        ]
        .into();
        let leaf = [NodePos(1), NodePos(2), NodePos(3), NodePos(4)];
        NestedUbTopology::new(&parent, &leaf, &[0.25; 4])
    }

    fn nested_sync(topology: &NestedUbTopology, scratch: &mut NestedUbScratch) -> f64 {
        let local = ForwardResult {
            scenario_costs: vec![10.0, 20.0, 30.0, 40.0],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            stage_stats: Vec::new(),
        };
        let path_stage_costs = [0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0];
        sync_forward(
            &local,
            &LocalBackend,
            4,
            ForwardBound::NestedRisk {
                path_stage_costs: &path_stage_costs,
                topology,
                cumulative_discounts: &[1.0, 1.0],
                risk_measure: RiskMeasure::CVaR {
                    alpha: 0.5,
                    lambda: 0.5,
                },
                num_stages: 2,
                scratch,
            },
        )
        .unwrap()
        .global_ub_mean
    }

    fn buffer_identities(scratch: &NestedUbScratch) -> Vec<(usize, usize)> {
        fn id<T>(v: &Vec<T>) -> (usize, usize) {
            (v.as_ptr() as usize, v.capacity())
        }
        let r = &scratch.recursion;
        vec![
            id(&scratch.stage_counts),
            id(&scratch.stage_displs),
            id(&scratch.gathered),
            id(&r.node_cost),
            id(&r.value),
            id(&r.child_vals),
            id(&r.child_probs),
            id(&r.risk.upper_bounds),
            id(&r.risk.order),
            id(&r.risk.mu),
        ]
    }

    #[test]
    fn nested_risk_sync_reuses_its_scratch_without_reallocating() {
        let topology = four_leaf_fan();
        let mut scratch = NestedUbScratch::default();

        let first = nested_sync(&topology, &mut scratch);
        let after_first = buffer_identities(&scratch);
        let second = nested_sync(&topology, &mut scratch);
        let after_second = buffer_identities(&scratch);
        let fresh = nested_sync(&topology, &mut NestedUbScratch::default());

        assert!(
            (first - 30.0).abs() < 1e-12,
            "the nested bound must be 30.0, got {first}"
        );
        assert_eq!(
            second.to_bits(),
            first.to_bits(),
            "a reused scratch must reproduce the bound bit for bit"
        );
        assert_eq!(
            fresh.to_bits(),
            first.to_bits(),
            "a fresh scratch must reproduce the bound bit for bit"
        );
        assert_eq!(
            after_second, after_first,
            "the second call must reuse every scratch buffer as (pointer, capacity)"
        );
    }

    #[test]
    fn nested_risk_gather_layout_recomputes_only_on_a_new_shape() {
        let mut scratch = NestedUbScratch::default();

        scratch.prepare_gather(5, 3, 2);
        assert_eq!(scratch.stage_counts, [4, 4, 2]);
        assert_eq!(scratch.stage_displs, [0, 4, 8]);
        assert_eq!(scratch.gathered.len(), 10);

        scratch.prepare_gather(5, 2, 2);
        assert_eq!(scratch.stage_counts, [6, 4]);
        assert_eq!(scratch.stage_displs, [0, 6]);
        assert_eq!(scratch.gathered.len(), 10);

        let counts_ptr = scratch.stage_counts.as_ptr();
        scratch.prepare_gather(5, 2, 2);
        assert_eq!(
            scratch.stage_counts.as_ptr(),
            counts_ptr,
            "an unchanged shape must keep the layout without recomputing it"
        );
    }
}
