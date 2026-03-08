//! MPI aggregation and `SimulationSummary` computation.
//!
//! After all MPI ranks complete their assigned simulation scenarios,
//! [`aggregate_simulation`] gathers per-scenario cost data across ranks and
//! computes the final [`SimulationSummary`].
//!
//! ## Aggregation protocol (SS4.4)
//!
//! The aggregation proceeds in four steps:
//!
//! 1. **Local min/max → global min/max** via two `allreduce` calls
//!    (`ReduceOp::Min`, `ReduceOp::Max`).
//!
//! 2. **Total costs gathered** via two `allgatherv` calls:
//!    - First `allgatherv` exchanges per-rank `u64` scenario counts so that
//!      every rank knows the displacement layout for the data gather.
//!    - Second `allgatherv` gathers the flat `f64` total-cost vector.
//!      Mean, standard deviation (Bessel-corrected), and `CVaR` are then
//!      computed locally on every rank from the gathered array.
//!
//! 3. **Per-category costs gathered** via an additional `allgatherv` on a
//!    stride-5 `f64` buffer packed from the five `ScenarioCategoryCosts` fields.
//!    Per-category mean, max, and frequency are computed from the gathered data.
//!
//! 4. **`SimulationSummary` assembled** with `stage_stats = None` and
//!    operational statistics set to `0.0` (deferred — requires per-stage
//!    deficit/spillage tracking not yet in the cost buffer).
//!
//! ## `CVaR` computation
//!
//! `CVaR` at confidence level α is the mean of the worst `ceil((1-α)*S)`
//! scenario costs, where `S` is the total number of scenarios. Costs are
//! sorted in descending order; the first `tail_size` elements form the tail.
//! When `S == 1`, `cvar` equals the single cost (tail size clamped to 1).
//!
//! ## Standard deviation
//!
//! Uses Bessel correction (`n-1` denominator). When `n <= 1`, returns `0.0`
//! to avoid division by zero, matching the pattern in `sync_forward`.
//!
//! ## `allgatherv` vs `gatherv`
//!
//! The `Communicator` trait does not expose `gatherv`. `allgatherv` is used
//! instead: all ranks receive all data, which is slightly more bandwidth but
//! functionally equivalent and avoids a subsequent broadcast step.

use cobre_comm::{Communicator, ReduceOp};

use crate::simulation::{
    config::SimulationConfig,
    error::SimulationError,
    types::{CategoryCostStats, ScenarioCategoryCosts, SimulationSummary},
};

/// Hardcoded `CVaR` confidence level for the minimal viable solver.
const CVAR_ALPHA: f64 = 0.95;

/// Number of per-category cost fields in [`ScenarioCategoryCosts`].
const N_CATEGORIES: usize = 5;

/// Category names in field declaration order (matching `ScenarioCategoryCosts`).
const CATEGORY_NAMES: [&str; N_CATEGORIES] = [
    "resource",
    "recourse",
    "violation",
    "regularization",
    "imputed",
];

/// Aggregate per-scenario cost data across all MPI ranks and compute the
/// final [`SimulationSummary`].
///
/// Uses `allgatherv` to collect all scenario costs on all ranks (the
/// `Communicator` trait has no `gatherv`), then computes global statistics
/// locally on each rank from the gathered data. The returned
/// `SimulationSummary` is identical on all ranks.
///
/// # Arguments
///
/// - `local_costs` — per-scenario `(scenario_id, total_cost, category_costs)`
///   from this rank's simulation forward pass.
/// - `config` — provides `n_scenarios` (total across all ranks).
/// - `comm` — communicator for collective operations.
///
/// # Returns
///
/// `Ok(SimulationSummary)` with:
/// - `min_cost`, `max_cost` — from `allreduce` (identical on all ranks).
/// - `mean_cost`, `std_cost`, `cvar`, `category_stats` — computed from the
///   `allgatherv` result (identical on all ranks).
/// - `n_scenarios` — actual count gathered across all ranks (equals `config.n_scenarios`).
/// - `cvar_alpha` — hardcoded to `0.95`.
/// - `deficit_frequency`, `total_deficit_mwh`, `total_spillage_mwh` — `0.0`
///   (deferred: requires per-stage deficit/spillage accumulation in the
///   simulation forward pass).
/// - `stage_stats` — `None` (deferred: requires per-stage aggregation).
///
/// # Errors
///
/// Returns `Err(SimulationError::IoError { message })` if any collective
/// operation (`allreduce`, `allgatherv`) fails. The `message` includes the
/// operation name and the underlying `CommError` display string.
///
/// # Examples
///
/// ```rust
/// use cobre_comm::LocalBackend;
/// use cobre_sddp::simulation::aggregation::aggregate_simulation;
/// use cobre_sddp::{SimulationConfig, ScenarioCategoryCosts};
///
/// let zero_cats = ScenarioCategoryCosts {
///     resource_cost: 0.0,
///     recourse_cost: 0.0,
///     violation_cost: 0.0,
///     regularization_cost: 0.0,
///     imputed_cost: 0.0,
/// };
/// let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = vec![
///     (0, 100.0, zero_cats),
/// ];
/// let config = SimulationConfig { n_scenarios: 1, io_channel_capacity: 1 };
/// let comm = LocalBackend;
///
/// let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
/// assert_eq!(summary.n_scenarios, 1);
/// assert_eq!(summary.mean_cost, 100.0);
/// assert_eq!(summary.cvar, 100.0);
/// assert_eq!(summary.std_cost, 0.0);
/// ```
pub fn aggregate_simulation<C: Communicator>(
    local_costs: &[(u32, f64, ScenarioCategoryCosts)],
    config: &SimulationConfig,
    comm: &C,
) -> Result<SimulationSummary, SimulationError> {
    let num_ranks = comm.size();
    let n_local = local_costs.len();

    // Global min/max via allreduce.
    let (local_min, local_max) = compute_local_min_max(local_costs);
    let mut global_min_buf = [0.0_f64];
    let mut global_max_buf = [0.0_f64];
    comm.allreduce(&[local_min], &mut global_min_buf, ReduceOp::Min)
        .map_err(|e| SimulationError::IoError {
            message: format!("allreduce(Min) failed: {e}"),
        })?;
    comm.allreduce(&[local_max], &mut global_max_buf, ReduceOp::Max)
        .map_err(|e| SimulationError::IoError {
            message: format!("allreduce(Max) failed: {e}"),
        })?;
    let global_min = global_min_buf[0];
    let global_max = global_max_buf[0];

    // Gather per-rank scenario counts to compute displacements.
    #[allow(clippy::cast_possible_truncation)]
    let counts_send = [n_local as u64];
    let mut counts_recv = vec![0u64; num_ranks];
    let counts_counts = vec![1usize; num_ranks];
    let counts_displs: Vec<usize> = (0..num_ranks).collect();
    comm.allgatherv(
        &counts_send,
        &mut counts_recv,
        &counts_counts,
        &counts_displs,
    )
    .map_err(|e| SimulationError::IoError {
        message: format!("allgatherv(counts) failed: {e}"),
    })?;

    // Gather total costs across ranks.
    let (cost_displs, total_gathered) = compute_displs_and_total(&counts_recv);
    let cost_send: Vec<f64> = local_costs.iter().map(|(_, c, _)| *c).collect();
    let mut cost_recv = vec![0.0_f64; total_gathered];
    let cost_counts: Vec<usize> = counts_recv
        .iter()
        .map(|&c| usize::try_from(c).unwrap_or(usize::MAX))
        .collect();
    comm.allgatherv(&cost_send, &mut cost_recv, &cost_counts, &cost_displs)
        .map_err(|e| SimulationError::IoError {
            message: format!("allgatherv(costs) failed: {e}"),
        })?;

    debug_assert_eq!(
        total_gathered, config.n_scenarios as usize,
        "gathered scenario count must match configured n_scenarios"
    );

    // Compute aggregate statistics from gathered costs.
    let n = total_gathered;
    let (mean_cost, std_cost) = compute_mean_std(&cost_recv);
    let cvar = compute_cvar(&cost_recv, CVAR_ALPHA);

    // Gather per-category costs (stride N_CATEGORIES).
    let cat_send = pack_category_costs(local_costs);
    let cat_send_count = n_local * N_CATEGORIES;
    let cat_counts: Vec<usize> = counts_recv
        .iter()
        .map(|&c| usize::try_from(c).unwrap_or(usize::MAX) * N_CATEGORIES)
        .collect();
    let cat_displs: Vec<usize> = cost_displs.iter().map(|&d| d * N_CATEGORIES).collect();
    let cat_total = total_gathered * N_CATEGORIES;

    debug_assert_eq!(
        cat_send.len(),
        cat_send_count,
        "packed category buffer length mismatch"
    );

    let mut cat_recv = vec![0.0_f64; cat_total];
    comm.allgatherv(&cat_send, &mut cat_recv, &cat_counts, &cat_displs)
        .map_err(|e| SimulationError::IoError {
            message: format!("allgatherv(categories) failed: {e}"),
        })?;

    // Compute per-category statistics.
    let category_stats = compute_category_stats(&cat_recv, n);

    Ok(SimulationSummary {
        mean_cost,
        std_cost,
        min_cost: global_min,
        max_cost: global_max,
        cvar,
        cvar_alpha: CVAR_ALPHA,
        category_stats,
        deficit_frequency: 0.0,
        total_deficit_mwh: 0.0,
        total_spillage_mwh: 0.0,
        #[allow(clippy::cast_possible_truncation)]
        n_scenarios: total_gathered as u32,
        stage_stats: None,
    })
}

// ── Private helpers ────────────────────────────────────────────────────────────

/// Compute local min and max total costs.
///
/// Returns `(f64::INFINITY, f64::NEG_INFINITY)` when `local_costs` is empty,
/// which after `allreduce(Min)` / `allreduce(Max)` across ranks yields the
/// correct global values from any rank that has scenarios.
fn compute_local_min_max(local_costs: &[(u32, f64, ScenarioCategoryCosts)]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for (_, cost, _) in local_costs {
        if *cost < min {
            min = *cost;
        }
        if *cost > max {
            max = *cost;
        }
    }
    (min, max)
}

/// Compute prefix-sum displacements from per-rank element counts.
///
/// Returns `(displs, total)` where `displs[r]` is the offset in the receive
/// buffer at which rank `r`'s data starts, and `total` is the total number
/// of elements across all ranks.
fn compute_displs_and_total(counts_recv: &[u64]) -> (Vec<usize>, usize) {
    let mut displs = Vec::with_capacity(counts_recv.len());
    let mut offset = 0usize;
    for &c in counts_recv {
        displs.push(offset);
        offset += usize::try_from(c).unwrap_or(usize::MAX);
    }
    (displs, offset)
}

/// Compute mean and Bessel-corrected standard deviation from a slice of costs.
///
/// Returns `(0.0, 0.0)` for empty input and `(mean, 0.0)` when `n <= 1`
/// (no variance with a single observation).
fn compute_mean_std(costs: &[f64]) -> (f64, f64) {
    let n = costs.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    let sum: f64 = costs.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let mean = sum / n as f64;

    if n <= 1 {
        return (mean, 0.0);
    }

    let sum_sq_diff: f64 = costs.iter().map(|&c| (c - mean) * (c - mean)).sum();
    #[allow(clippy::cast_precision_loss)]
    let variance = sum_sq_diff / (n as f64 - 1.0);
    let std_dev = variance.max(0.0).sqrt();

    (mean, std_dev)
}

/// Compute `CVaR` at confidence level `alpha` from a cost slice.
///
/// Sorts costs in descending order and returns the mean of the worst
/// `max(1, ceil((1 - alpha) * n))` scenarios. When `costs` is empty,
/// returns `0.0`.
///
/// The tail size is computed as `n - floor(alpha * n)` to avoid the
/// floating-point imprecision of `ceil((1 - alpha) * n)` (for example,
/// `1.0 - 0.95 = 0.050000000000000044` in IEEE 754 double precision, which
/// would cause `ceil(0.050...044 * 100) = 6` instead of `5`).
fn compute_cvar(costs: &[f64], alpha: f64) -> f64 {
    let n = costs.len();
    if n == 0 {
        return 0.0;
    }

    // Sort descending (worst first) into a scratch buffer.
    let mut sorted = costs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Use n - floor(alpha * n) to avoid floating-point imprecision in
    // ceil((1 - alpha) * n). Both formulas are mathematically equivalent
    // but this one avoids the double subtraction error.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    let scenarios_in_tail = {
        let alpha_n = (alpha * n as f64).floor() as usize;
        n - alpha_n
    };
    let tail_size = scenarios_in_tail.max(1).min(n);

    let tail_sum: f64 = sorted[..tail_size].iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let cvar = tail_sum / tail_size as f64;
    cvar
}

/// Pack per-category costs into a flat f64 buffer (stride [`N_CATEGORIES`]).
///
/// Layout per scenario (5 consecutive elements):
/// `[resource_cost, recourse_cost, violation_cost, regularization_cost, imputed_cost]`
fn pack_category_costs(local_costs: &[(u32, f64, ScenarioCategoryCosts)]) -> Vec<f64> {
    let mut buf = Vec::with_capacity(local_costs.len() * N_CATEGORIES);
    for (_, _, cats) in local_costs {
        buf.push(cats.resource_cost);
        buf.push(cats.recourse_cost);
        buf.push(cats.violation_cost);
        buf.push(cats.regularization_cost);
        buf.push(cats.imputed_cost);
    }
    buf
}

/// Compute per-category statistics from a gathered flat buffer.
///
/// `cat_buf` has stride [`N_CATEGORIES`]: element `s * N_CATEGORIES + k` is
/// category `k` for scenario `s`. Returns one [`CategoryCostStats`] per
/// category in `CATEGORY_NAMES` order. Returns all-zero stats when `n == 0`.
fn compute_category_stats(cat_buf: &[f64], n: usize) -> Vec<CategoryCostStats> {
    let mut stats = Vec::with_capacity(N_CATEGORIES);

    for k in 0..N_CATEGORIES {
        let (mean, max, frequency) = if n == 0 {
            (0.0, 0.0, 0.0)
        } else {
            let mut sum = 0.0_f64;
            let mut local_max = f64::NEG_INFINITY;
            let mut nonzero_count = 0usize;

            for s in 0..n {
                let val = cat_buf[s * N_CATEGORIES + k];
                sum += val;
                if val > local_max {
                    local_max = val;
                }
                if val != 0.0 {
                    nonzero_count += 1;
                }
            }

            #[allow(clippy::cast_precision_loss)]
            let mean = sum / n as f64;
            #[allow(clippy::cast_precision_loss)]
            let frequency = nonzero_count as f64 / n as f64;
            (mean, local_max, frequency)
        };

        stats.push(CategoryCostStats {
            category: CATEGORY_NAMES[k].to_string(),
            mean,
            max,
            frequency,
        });
    }

    stats
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_lossless
    )]

    use cobre_comm::LocalBackend;

    use super::{
        compute_cvar, compute_local_min_max, compute_mean_std, pack_category_costs, CVAR_ALPHA,
        N_CATEGORIES,
    };
    use crate::{
        simulation::aggregation::aggregate_simulation, ScenarioCategoryCosts, SimulationConfig,
    };

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn zero_cats() -> ScenarioCategoryCosts {
        ScenarioCategoryCosts {
            resource_cost: 0.0,
            recourse_cost: 0.0,
            violation_cost: 0.0,
            regularization_cost: 0.0,
            imputed_cost: 0.0,
        }
    }

    fn make_config(n: u32) -> SimulationConfig {
        SimulationConfig {
            n_scenarios: n,
            io_channel_capacity: 1,
        }
    }

    // ── Unit tests: compute_local_min_max ──────────────────────────────────────

    #[test]
    fn local_min_max_basic() {
        let cats = zero_cats();
        let costs = vec![(0u32, 100.0, cats)];
        let (min, max) = compute_local_min_max(&costs);
        assert_eq!(min, 100.0);
        assert_eq!(max, 100.0);
    }

    #[test]
    fn local_min_max_multiple() {
        let costs: Vec<(u32, f64, ScenarioCategoryCosts)> = vec![
            (0, 300.0, zero_cats()),
            (1, 100.0, zero_cats()),
            (2, 200.0, zero_cats()),
        ];
        let (min, max) = compute_local_min_max(&costs);
        assert_eq!(min, 100.0);
        assert_eq!(max, 300.0);
    }

    #[test]
    fn local_min_max_empty_returns_infinities() {
        let (min, max) = compute_local_min_max(&[]);
        assert!(min.is_infinite() && min.is_sign_positive());
        assert!(max.is_infinite() && max.is_sign_negative());
    }

    // ── Unit tests: compute_mean_std ───────────────────────────────────────────

    #[test]
    fn mean_std_five_costs() {
        // costs = [100, 200, 300, 400, 500], mean = 300
        // Bessel-corrected variance = 100000 / 4 = 25000, std = sqrt(25000)
        let costs = [100.0_f64, 200.0, 300.0, 400.0, 500.0];
        let (mean, std) = compute_mean_std(&costs);
        assert_eq!(mean, 300.0);
        let expected_std = 25000.0_f64.sqrt();
        assert!(
            (std - expected_std).abs() < 1e-9,
            "std={std} expected={expected_std}"
        );
    }

    #[test]
    fn mean_std_single_scenario_yields_zero_std() {
        let costs = [42.0_f64];
        let (mean, std) = compute_mean_std(&costs);
        assert_eq!(mean, 42.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn mean_std_empty_yields_zeros() {
        let (mean, std) = compute_mean_std(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    // ── Unit tests: compute_cvar ───────────────────────────────────────────────

    #[test]
    fn cvar_five_scenarios_alpha_095() {
        // costs = [100, 200, 300, 400, 500], alpha = 0.95
        // tail_size = n - floor(0.95 * 5) = 5 - 4 = 1
        // cvar = mean of top 1 = 500.0
        let costs = [100.0_f64, 200.0, 300.0, 400.0, 500.0];
        let cvar = compute_cvar(&costs, 0.95);
        assert_eq!(cvar, 500.0);
    }

    #[test]
    fn cvar_single_scenario_equals_cost() {
        let costs = [42.0_f64];
        let cvar = compute_cvar(&costs, CVAR_ALPHA);
        assert_eq!(cvar, 42.0);
    }

    #[test]
    fn cvar_empty_returns_zero() {
        let cvar = compute_cvar(&[], 0.95);
        assert_eq!(cvar, 0.0);
    }

    #[test]
    fn cvar_100_scenarios_alpha_095() {
        // costs 1.0..=100.0, alpha=0.95
        // tail_size = 100 - floor(0.95 * 100) = 100 - 95 = 5
        // top 5: 100, 99, 98, 97, 96 → mean = 490/5 = 98.0
        let costs: Vec<f64> = (1..=100).map(f64::from).collect();
        let cvar = compute_cvar(&costs, 0.95);
        assert_eq!(cvar, 98.0);
    }

    // ── Unit tests: pack_category_costs ───────────────────────────────────────

    #[test]
    fn pack_category_costs_layout() {
        let cats = ScenarioCategoryCosts {
            resource_cost: 1.0,
            recourse_cost: 2.0,
            violation_cost: 3.0,
            regularization_cost: 4.0,
            imputed_cost: 5.0,
        };
        let local_costs = vec![(0u32, 15.0, cats)];
        let packed = pack_category_costs(&local_costs);
        assert_eq!(packed.len(), N_CATEGORIES);
        assert_eq!(packed[0], 1.0, "resource_cost at index 0");
        assert_eq!(packed[1], 2.0, "recourse_cost at index 1");
        assert_eq!(packed[2], 3.0, "violation_cost at index 2");
        assert_eq!(packed[3], 4.0, "regularization_cost at index 3");
        assert_eq!(packed[4], 5.0, "imputed_cost at index 4");
    }

    #[test]
    fn pack_category_costs_empty() {
        let packed = pack_category_costs(&[]);
        assert!(packed.is_empty());
    }

    // ── Integration tests: aggregate_simulation with LocalBackend ─────────────

    #[test]
    fn aggregate_basic_three_scenarios_mean_min_max() {
        // AC: local_costs [(0,100), (1,200), (2,150)], n_scenarios=3
        // mean=150, min=100, max=200, n_scenarios=3
        let local_costs = vec![
            (0u32, 100.0, zero_cats()),
            (1u32, 200.0, zero_cats()),
            (2u32, 150.0, zero_cats()),
        ];
        let config = make_config(3);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        assert_eq!(summary.mean_cost, 150.0);
        assert_eq!(summary.min_cost, 100.0);
        assert_eq!(summary.max_cost, 200.0);
        assert_eq!(summary.n_scenarios, 3);
        assert_eq!(summary.cvar_alpha, 0.95);
    }

    #[test]
    fn aggregate_cvar_five_scenarios() {
        // AC: 5 scenarios [100,200,300,400,500], alpha=0.95
        // tail_size = 5 - floor(4.75) = 5 - 4 = 1, cvar = 500.0
        let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = (0u32..5)
            .map(|i| (i, f64::from(i + 1) * 100.0, zero_cats()))
            .collect();
        let config = make_config(5);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        assert_eq!(summary.cvar, 500.0);
    }

    #[test]
    fn aggregate_single_scenario_std_zero_cvar_equals_cost() {
        // AC: n=1, std=0.0, cvar=total_cost
        let local_costs = vec![(0u32, 999.0, zero_cats())];
        let config = make_config(1);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        assert_eq!(summary.std_cost, 0.0);
        assert_eq!(summary.cvar, 999.0);
        assert_eq!(summary.mean_cost, 999.0);
        assert_eq!(summary.min_cost, 999.0);
        assert_eq!(summary.max_cost, 999.0);
    }

    #[test]
    fn aggregate_category_stats_frequency() {
        // AC: resource_cost non-zero in 3 of 5 scenarios
        // → category_stats[0].frequency = 0.6, category = "resource"
        let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = (0i32..5)
            .map(|i| {
                let cats = if i < 3 {
                    ScenarioCategoryCosts {
                        resource_cost: 100.0,
                        recourse_cost: 0.0,
                        violation_cost: 0.0,
                        regularization_cost: 0.0,
                        imputed_cost: 0.0,
                    }
                } else {
                    zero_cats()
                };
                let total = if i < 3 { 100.0 } else { 0.0 };
                #[allow(clippy::cast_sign_loss)]
                (i as u32, total, cats)
            })
            .collect();
        let config = make_config(5);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        let resource_stats = &summary.category_stats[0];
        assert_eq!(resource_stats.category, "resource");
        assert!(
            (resource_stats.frequency - 0.6).abs() < 1e-12,
            "expected frequency=0.6, got {}",
            resource_stats.frequency
        );
    }

    #[test]
    fn aggregate_category_stats_mean_max() {
        // resource_cost: 100, 200, 300 → mean=200, max=300
        let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = vec![
            (
                0,
                100.0,
                ScenarioCategoryCosts {
                    resource_cost: 100.0,
                    recourse_cost: 0.0,
                    violation_cost: 0.0,
                    regularization_cost: 0.0,
                    imputed_cost: 0.0,
                },
            ),
            (
                1,
                200.0,
                ScenarioCategoryCosts {
                    resource_cost: 200.0,
                    recourse_cost: 0.0,
                    violation_cost: 0.0,
                    regularization_cost: 0.0,
                    imputed_cost: 0.0,
                },
            ),
            (
                2,
                300.0,
                ScenarioCategoryCosts {
                    resource_cost: 300.0,
                    recourse_cost: 0.0,
                    violation_cost: 0.0,
                    regularization_cost: 0.0,
                    imputed_cost: 0.0,
                },
            ),
        ];
        let config = make_config(3);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        let resource = &summary.category_stats[0];
        assert_eq!(resource.mean, 200.0);
        assert_eq!(resource.max, 300.0);
        assert_eq!(resource.frequency, 1.0);
    }

    #[test]
    fn aggregate_category_names_in_order() {
        // Verify category names match spec order.
        let local_costs = vec![(0u32, 0.0, zero_cats())];
        let config = make_config(1);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        assert_eq!(summary.category_stats.len(), N_CATEGORIES);
        assert_eq!(summary.category_stats[0].category, "resource");
        assert_eq!(summary.category_stats[1].category, "recourse");
        assert_eq!(summary.category_stats[2].category, "violation");
        assert_eq!(summary.category_stats[3].category, "regularization");
        assert_eq!(summary.category_stats[4].category, "imputed");
    }

    #[test]
    fn aggregate_operational_stats_are_zero_placeholders() {
        let local_costs = vec![(0u32, 50.0, zero_cats())];
        let config = make_config(1);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        assert_eq!(summary.deficit_frequency, 0.0);
        assert_eq!(summary.total_deficit_mwh, 0.0);
        assert_eq!(summary.total_spillage_mwh, 0.0);
    }

    #[test]
    fn aggregate_stage_stats_is_none() {
        let local_costs = vec![(0u32, 50.0, zero_cats())];
        let config = make_config(1);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        assert!(summary.stage_stats.is_none());
    }

    #[test]
    fn aggregate_cvar_100_scenarios() {
        // Costs 1.0..=100.0, alpha=0.95
        // tail_size = 100 - floor(95) = 5, cvar = (96+97+98+99+100)/5 = 98.0
        let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = (1u32..=100)
            .map(|i| (i - 1, f64::from(i), zero_cats()))
            .collect();
        let config = make_config(100);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        assert_eq!(summary.cvar, 98.0);
    }

    #[test]
    fn aggregate_std_five_costs_bessel_corrected() {
        // costs [100,200,300,400,500], std = sqrt(25000)
        let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = (0u32..5)
            .map(|i| (i, f64::from(i + 1) * 100.0, zero_cats()))
            .collect();
        let config = make_config(5);
        let comm = LocalBackend;

        let summary = aggregate_simulation(&local_costs, &config, &comm).unwrap();
        let expected_std = 25000.0_f64.sqrt();
        assert!(
            (summary.std_cost - expected_std).abs() < 1e-9,
            "expected std={expected_std}, got {}",
            summary.std_cost
        );
    }
}
