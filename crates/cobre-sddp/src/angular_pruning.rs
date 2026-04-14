//! Angular diversity pruning configuration and runtime parameters.
//!
//! This module provides the runtime counterpart to the IO-layer
//! [`cobre_io::config::AngularPruningConfig`]. Angular pruning is stage 2 of
//! the cut selection pipeline: it uses cosine similarity clustering as a
//! computational accelerator for pointwise dominance verification.
//!
//! **This is NOT standalone pruning.** A cut that appears in the same cosine
//! cluster as another is only a *candidate* for pointwise dominance
//! verification, not automatically removed. This design preserves Assumption
//! (H2) from Guigues 2017 and finite convergence of the SDDP algorithm.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_io::config::AngularPruningConfig;
//! use cobre_sddp::angular_pruning::{AngularPruningParams, parse_angular_pruning_config};
//!
//! // Disabled (default): parse returns Ok(None)
//! let disabled = AngularPruningConfig::default();
//! assert!(parse_angular_pruning_config(&disabled, None).unwrap().is_none());
//!
//! // Enabled with defaults
//! let enabled = AngularPruningConfig { enabled: Some(true), ..Default::default() };
//! let params = parse_angular_pruning_config(&enabled, Some(3)).unwrap().unwrap();
//! assert!((params.cosine_threshold - 0.999).abs() < f64::EPSILON);
//! assert_eq!(params.check_frequency, 3);
//! ```

use crate::cut::pool::CutPool;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of a single angular-accelerated dominance pass over one cut pool.
///
/// Returned by [`select_angular_dominated`]. The caller uses `deactivate` to
/// mark cuts as inactive in the pool; the other fields are for observability
/// and diagnostics.
#[derive(Debug, Clone, Default)]
pub struct AngularPruningResult {
    /// Slot indices of cuts that should be deactivated, in ascending order.
    ///
    /// A cut appears here only if it was dominated at **every** visited trial
    /// point by some other cut in its angular cluster. This preserves
    /// Assumption (H2) from Guigues 2017.
    pub deactivate: Vec<u32>,

    /// Total number of clusters formed during Phase 1 (including singletons
    /// and the zero-norm cluster if non-empty).
    pub clusters_formed: usize,

    /// Total number of pairwise within-cluster dominance checks performed
    /// during Phase 2.
    pub dominance_checks: usize,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Dominance comparison tolerance. A cut is dominated at a trial point if
/// another cut's value exceeds it by more than `-DOMINANCE_EPSILON`.
/// Matches the tolerance used in `select_dominated` in `cut_selection`.
const DOMINANCE_EPSILON: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Main public function
// ---------------------------------------------------------------------------

/// Identify dominated cuts using angular clustering as a computational
/// accelerator for pointwise dominance verification.
///
/// ## Algorithm
///
/// **Phase 1 — Angular clustering**: Eligible cuts (active, not from
/// `current_iteration`) are clustered by cosine similarity of their
/// normalized coefficient vectors. The greedy single-linkage algorithm
/// assigns each unassigned cut to the first cluster whose representative
/// satisfies `cos(n_i, n_j) > cosine_threshold`. Zero-norm cuts
/// (`‖π‖ < 1e-12`) form a dedicated cluster. Singleton clusters are
/// skipped in Phase 2.
///
/// **Phase 2 — Within-cluster dominance**: For each cluster with ≥ 2 cuts,
/// every pair is tested for pointwise dominance at all `visited_states`.
/// Cut `i` is dominated if, for every trial point `x`, some other cut `j`
/// in the same cluster satisfies `value_j(x) ≥ value_i(x) − ε` where
/// `ε = 1e-8`. Dominated cuts are added to `deactivate`.
///
/// A cut is **never** deactivated unless it is dominated at **all** trial
/// points — this is the (H2)-preserving guarantee.
///
/// ## Arguments
///
/// * `pool` — Immutable reference to one stage's cut pool.
/// * `visited_states` — Flat buffer of trial points, length
///   `n_trials * pool.state_dimension`. Same format as used by
///   `select_dominated` in `cut_selection`.
/// * `cosine_threshold` — Similarity threshold already validated in
///   `(0.0, 1.0]`.
/// * `current_iteration` — Cuts added during this iteration are excluded.
///
/// ## Returns
///
/// [`AngularPruningResult`] with `deactivate` indices sorted ascending.
/// Returns an empty result when `pool.active_count() < 2` or
/// `visited_states.is_empty()`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn select_angular_dominated(
    pool: &CutPool,
    visited_states: &[f64],
    cosine_threshold: f64,
    current_iteration: u64,
) -> AngularPruningResult {
    let n_state = pool.state_dimension;

    debug_assert!(
        visited_states.len() % n_state.max(1) == 0,
        "visited_states length {} is not a multiple of state_dimension {}",
        visited_states.len(),
        n_state
    );

    // Short-circuit: need ≥ 2 active cuts and at least one trial point.
    if pool.active_count() < 2 || visited_states.is_empty() {
        return AngularPruningResult::default();
    }

    let (clusters, zero_norm_cluster) =
        cluster_by_angular_similarity(pool, cosine_threshold, current_iteration);

    let clusters_formed = clusters.len() + usize::from(!zero_norm_cluster.is_empty());

    let mut deactivate: Vec<u32> = Vec::new();
    let mut dominance_checks: usize = 0;

    for cluster in &clusters {
        if cluster.len() < 2 {
            continue;
        }
        dominance_checks += cluster.len() * (cluster.len() - 1);
        let dominated = dominated_within_cluster(pool, cluster, visited_states, DOMINANCE_EPSILON);
        deactivate.extend(dominated.iter().map(|&idx| idx as u32));
    }

    if zero_norm_cluster.len() >= 2 {
        dominance_checks += zero_norm_cluster.len() * (zero_norm_cluster.len() - 1);
        let dominated = dominated_zero_norm_cluster(
            pool,
            &zero_norm_cluster,
            visited_states,
            DOMINANCE_EPSILON,
        );
        deactivate.extend(dominated.iter().map(|&idx| idx as u32));
    }

    deactivate.sort_unstable();

    AngularPruningResult {
        deactivate,
        clusters_formed,
        dominance_checks,
    }
}

// ---------------------------------------------------------------------------
// Phase 1: angular clustering
// ---------------------------------------------------------------------------

/// Cluster eligible cuts by cosine similarity using greedy single-linkage.
///
/// Returns `(clusters, zero_norm_cluster)`.
///
/// Each element of `clusters` is a `Vec<usize>` of slot indices whose
/// normalized coefficient vectors are mutually within `cosine_threshold` of
/// each other (single-linkage: a cut joins a cluster if its cosine with
/// the cluster's *first* member exceeds the threshold). Cuts with
/// `‖π‖ < 1e-12` are collected separately in `zero_norm_cluster`.
///
/// Eligible slots: `active[k] == true` AND
/// `metadata[k].iteration_generated < current_iteration`.
fn cluster_by_angular_similarity(
    pool: &CutPool,
    cosine_threshold: f64,
    current_iteration: u64,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let populated = pool.populated_count;
    let n_state = pool.state_dimension;

    let eligible: Vec<usize> = (0..populated)
        .filter(|&k| pool.active[k] && pool.metadata[k].iteration_generated < current_iteration)
        .collect();

    let unit_normals: Vec<Option<Vec<f64>>> = eligible
        .iter()
        .map(|&k| {
            let start = k * n_state;
            let coeffs = &pool.coefficients[start..start + n_state];
            let norm_sq: f64 = coeffs.iter().map(|&c| c * c).sum();
            let norm = norm_sq.sqrt();
            if norm < 1e-12 {
                None
            } else {
                let inv = 1.0 / norm;
                Some(coeffs.iter().map(|&c| c * inv).collect())
            }
        })
        .collect();

    let zero_norm_cluster: Vec<usize> = eligible
        .iter()
        .zip(unit_normals.iter())
        .filter_map(|(&slot, n)| if n.is_none() { Some(slot) } else { None })
        .collect();

    // Greedy single-linkage clustering for non-zero-norm cuts.
    // `cluster_rep[i]` holds the unit normal of the representative (first
    // member) of cluster i.
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut cluster_reps: Vec<Vec<f64>> = Vec::new();

    for (&slot, normal_opt) in eligible.iter().zip(unit_normals.iter()) {
        let Some(normal) = normal_opt else {
            continue;
        };

        let mut assigned = false;
        for (c_idx, rep) in cluster_reps.iter().enumerate() {
            let cosine: f64 = normal.iter().zip(rep.iter()).map(|(a, b)| a * b).sum();
            if cosine > cosine_threshold {
                clusters[c_idx].push(slot);
                assigned = true;
                break;
            }
        }

        if !assigned {
            cluster_reps.push(normal.clone());
            clusters.push(vec![slot]);
        }
    }

    (clusters, zero_norm_cluster)
}

// ---------------------------------------------------------------------------
// Phase 2: within-cluster dominance
// ---------------------------------------------------------------------------

/// Check pointwise dominance for a cluster of cuts with non-zero norms.
///
/// Returns the slot indices of cuts that are dominated at **every** trial
/// point in `visited_states` by some other cut in the same cluster.
///
/// For each cut `i` in `cluster`, it is dominated if there exists another
/// cut `j` in `cluster` such that for **all** trial points `x`:
/// `value_j(x) ≥ value_i(x) − epsilon`.
///
/// Value formula: `intercept[k] + dot(coefficients[k * n_state ..], x)`.
fn dominated_within_cluster(
    pool: &CutPool,
    cluster: &[usize],
    visited_states: &[f64],
    epsilon: f64,
) -> Vec<usize> {
    let n_state = pool.state_dimension;
    let n_cuts = cluster.len();

    let mut is_candidate = vec![true; n_cuts];
    let mut n_candidates = n_cuts;
    let mut values = vec![0.0_f64; n_cuts];

    for x_hat in visited_states.chunks_exact(n_state) {
        for (i, &slot) in cluster.iter().enumerate() {
            let start = slot * n_state;
            values[i] = pool.intercepts[slot]
                + pool.coefficients[start..start + n_state]
                    .iter()
                    .zip(x_hat.iter())
                    .map(|(c, x)| c * x)
                    .sum::<f64>();
        }

        // Find the cluster-local maximum value at this trial point.
        // Cuts achieving the maximum survive (are not dominated).
        let max_val = values[..n_cuts]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let cutoff = max_val - epsilon;
        for i in 0..n_cuts {
            if is_candidate[i] && values[i] >= cutoff {
                is_candidate[i] = false;
                n_candidates -= 1;
            }
        }

        if n_candidates == 0 {
            break;
        }
    }

    cluster
        .iter()
        .enumerate()
        .filter_map(|(i, &slot)| if is_candidate[i] { Some(slot) } else { None })
        .collect()
}

/// Check pointwise dominance for the zero-norm cluster.
///
/// Zero-norm cuts have `value(x) = intercept` (independent of `x`).
/// Cut `i` is dominated if some other cut `j` in the cluster has
/// `intercept[j] >= intercept[i] - epsilon`.
fn dominated_zero_norm_cluster(
    pool: &CutPool,
    cluster: &[usize],
    _visited_states: &[f64],
    epsilon: f64,
) -> Vec<usize> {
    // Find the maximum intercept in the cluster. Cuts tied at the max survive.
    let max_intercept = cluster
        .iter()
        .map(|&s| pool.intercepts[s])
        .fold(f64::NEG_INFINITY, f64::max);
    let cutoff = max_intercept - epsilon;
    cluster
        .iter()
        .filter(|&&slot_i| pool.intercepts[slot_i] < cutoff)
        .copied()
        .collect()
}

// ---------------------------------------------------------------------------
// Config and params
// ---------------------------------------------------------------------------

/// Resolved runtime parameters for angular diversity pruning.
///
/// Constructed from [`cobre_io::config::AngularPruningConfig`] by
/// [`parse_angular_pruning_config`] after validation. Both fields are
/// always valid when this struct exists: `cosine_threshold` is in
/// `(0.0, 1.0]` and `check_frequency` is `> 0`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct AngularPruningParams {
    /// Cosine similarity threshold for clustering candidates.
    ///
    /// Cuts with pairwise cosine similarity `>= cosine_threshold` belong to
    /// the same cluster and become candidates for pointwise dominance
    /// verification. Always in `(0.0, 1.0]`.
    pub cosine_threshold: f64,

    /// Iterations between angular pruning runs. Always `> 0`.
    pub check_frequency: u64,
}

impl AngularPruningParams {
    /// Determine whether angular pruning should run at the given iteration.
    ///
    /// Returns `true` if `iteration > 0` and `iteration` is a multiple of
    /// `check_frequency`. Never runs at iteration 0 (no cuts exist yet).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::angular_pruning::AngularPruningParams;
    ///
    /// let params = AngularPruningParams { cosine_threshold: 0.999, check_frequency: 5 };
    /// assert!(!params.should_run(0));
    /// assert!(!params.should_run(3));
    /// assert!(params.should_run(5));
    /// assert!(params.should_run(10));
    /// ```
    #[must_use]
    pub fn should_run(&self, iteration: u64) -> bool {
        iteration > 0 && iteration % self.check_frequency == 0
    }
}

/// Parse a [`cobre_io::config::AngularPruningConfig`] into optional runtime
/// parameters.
///
/// Returns `Ok(None)` when angular pruning is disabled (the `enabled` field is
/// `None` or `false`). Returns `Ok(Some(AngularPruningParams { .. }))` when
/// enabled and valid. Returns `Err(String)` with a descriptive message when
/// enabled but configuration is invalid.
///
/// # Default resolution
///
/// When `enabled = true` and a field is absent:
/// - `cosine_threshold` defaults to `0.999`.
/// - `check_frequency` defaults to `parent_check_frequency.unwrap_or(5)`.
///
/// # Errors
///
/// - `cosine_threshold <= 0.0` or `> 1.0`:
///   `"angular_pruning.cosine_threshold must be in (0.0, 1.0], got {value}"`.
/// - `check_frequency == 0`:
///   `"angular_pruning.check_frequency must be > 0"`.
pub fn parse_angular_pruning_config(
    config: &cobre_io::config::AngularPruningConfig,
    parent_check_frequency: Option<u32>,
) -> Result<Option<AngularPruningParams>, String> {
    let enabled = config.enabled.unwrap_or(false);
    if !enabled {
        return Ok(None);
    }

    // Validate and resolve cosine_threshold.
    let cosine_threshold = config.cosine_threshold.unwrap_or(0.999);
    if cosine_threshold <= 0.0 || cosine_threshold > 1.0 {
        return Err(format!(
            "angular_pruning.cosine_threshold must be in (0.0, 1.0], got {cosine_threshold}"
        ));
    }

    // Validate and resolve check_frequency.
    let check_frequency_u32 = config
        .check_frequency
        .unwrap_or_else(|| parent_check_frequency.unwrap_or(5));
    if check_frequency_u32 == 0 {
        return Err("angular_pruning.check_frequency must be > 0".to_string());
    }

    Ok(Some(AngularPruningParams {
        cosine_threshold,
        check_frequency: u64::from(check_frequency_u32),
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{AngularPruningParams, parse_angular_pruning_config};
    use cobre_io::config::AngularPruningConfig;

    // ── Disabled paths ───────────────────────────────────────────────────────

    #[test]
    fn parse_disabled_returns_none() {
        let config = AngularPruningConfig {
            enabled: None,
            cosine_threshold: None,
            check_frequency: None,
        };
        let result = parse_angular_pruning_config(&config, None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_enabled_false_returns_none() {
        let config = AngularPruningConfig {
            enabled: Some(false),
            cosine_threshold: None,
            check_frequency: None,
        };
        let result = parse_angular_pruning_config(&config, None).unwrap();
        assert!(result.is_none());
    }

    // ── Default resolution ───────────────────────────────────────────────────

    #[test]
    fn parse_enabled_default_threshold() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: None,
            check_frequency: None,
        };
        let params = parse_angular_pruning_config(&config, None)
            .unwrap()
            .unwrap();
        assert!(
            (params.cosine_threshold - 0.999).abs() < f64::EPSILON,
            "expected default threshold 0.999, got {}",
            params.cosine_threshold
        );
    }

    #[test]
    fn parse_enabled_explicit_threshold() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: Some(0.99),
            check_frequency: None,
        };
        let params = parse_angular_pruning_config(&config, None)
            .unwrap()
            .unwrap();
        assert!(
            (params.cosine_threshold - 0.99).abs() < f64::EPSILON,
            "expected 0.99, got {}",
            params.cosine_threshold
        );
    }

    #[test]
    fn parse_enabled_inherits_parent_frequency() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: None,
            check_frequency: None,
        };
        let params = parse_angular_pruning_config(&config, Some(3))
            .unwrap()
            .unwrap();
        assert_eq!(params.check_frequency, 3);
    }

    #[test]
    fn parse_enabled_explicit_frequency_overrides_parent() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: None,
            check_frequency: Some(2),
        };
        let params = parse_angular_pruning_config(&config, Some(3))
            .unwrap()
            .unwrap();
        assert_eq!(params.check_frequency, 2);
    }

    #[test]
    fn parse_enabled_no_parent_frequency_defaults_to_five() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: None,
            check_frequency: None,
        };
        let params = parse_angular_pruning_config(&config, None)
            .unwrap()
            .unwrap();
        assert_eq!(params.check_frequency, 5);
    }

    // ── Validation errors ────────────────────────────────────────────────────

    #[test]
    fn parse_invalid_threshold_zero() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: Some(0.0),
            check_frequency: None,
        };
        let err = parse_angular_pruning_config(&config, None).unwrap_err();
        assert!(
            err.contains("cosine_threshold must be in (0.0, 1.0]"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_invalid_threshold_negative() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: Some(-0.5),
            check_frequency: None,
        };
        let err = parse_angular_pruning_config(&config, None).unwrap_err();
        assert!(
            err.contains("cosine_threshold must be in (0.0, 1.0]"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_invalid_threshold_above_one() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: Some(1.1),
            check_frequency: None,
        };
        let err = parse_angular_pruning_config(&config, None).unwrap_err();
        assert!(
            err.contains("cosine_threshold must be in (0.0, 1.0]"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_invalid_frequency_zero() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: None,
            check_frequency: Some(0),
        };
        let err = parse_angular_pruning_config(&config, None).unwrap_err();
        assert!(
            err.contains("check_frequency must be > 0"),
            "unexpected error: {err}"
        );
    }

    // ── should_run ──────────────────────────────────────────────────────────

    #[test]
    fn should_run_false_at_zero() {
        let params = AngularPruningParams {
            cosine_threshold: 0.999,
            check_frequency: 5,
        };
        assert!(!params.should_run(0));
    }

    #[test]
    fn should_run_true_at_multiples() {
        let params = AngularPruningParams {
            cosine_threshold: 0.999,
            check_frequency: 5,
        };
        assert!(params.should_run(5));
        assert!(params.should_run(10));
        assert!(params.should_run(15));
    }

    #[test]
    fn should_run_false_between_multiples() {
        let params = AngularPruningParams {
            cosine_threshold: 0.999,
            check_frequency: 5,
        };
        assert!(!params.should_run(1));
        assert!(!params.should_run(2));
        assert!(!params.should_run(3));
        assert!(!params.should_run(4));
        assert!(!params.should_run(6));
    }

    // ── Boundary: cosine_threshold = 1.0 is valid ────────────────────────────

    #[test]
    fn parse_threshold_exactly_one_is_valid() {
        let config = AngularPruningConfig {
            enabled: Some(true),
            cosine_threshold: Some(1.0),
            check_frequency: None,
        };
        let params = parse_angular_pruning_config(&config, None)
            .unwrap()
            .unwrap();
        assert!(
            (params.cosine_threshold - 1.0).abs() < f64::EPSILON,
            "threshold 1.0 should be accepted"
        );
    }

    // ── select_angular_dominated tests ───────────────────────────────────────

    use super::select_angular_dominated;
    use crate::cut::pool::CutPool;

    /// Build a minimal CutPool with `n_state`-dimensional cuts.
    /// Each call to `add_cut_simple` uses iteration=1, sequential fp_idx.
    fn make_pool(n_state: usize) -> CutPool {
        // capacity=32, forward_passes=32 so slot = 0 + 1*32 + fp_idx = 32+fp_idx
        CutPool::new(64, n_state, 32, 0)
    }

    fn add(pool: &mut CutPool, fp_idx: u32, intercept: f64, coefficients: &[f64]) {
        pool.add_cut(1, fp_idx, intercept, coefficients);
    }

    // ── H2 preservation: two crossing cuts, neither deactivated ─────────────

    /// Two near-parallel cuts where cut A is tightest at x1 and cut B is
    /// tightest at x2.  Neither should be deactivated.
    #[test]
    fn h2_preservation_crossing_cuts_both_survive() {
        let mut pool = make_pool(1);
        // A: value(x) = 0 + 2*x    (tightest at x=1: value=2)
        // B: value(x) = 1 + 0.5*x  (tightest at x=0: value=1 > A's 0)
        // At x=0: A=0, B=1  → B tighter
        // At x=1: A=2, B=1.5 → A tighter
        // Neither is dominated at all trial points.
        add(&mut pool, 0, 0.0, &[2.0]);
        add(&mut pool, 1, 1.0, &[0.5]);

        // Cosine between [2] and [0.5] normalized: both are unit (1-d), both
        // positive → cosine = 1.0, threshold 0.999 → same cluster.
        let visited = vec![0.0_f64, 1.0_f64];
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);
        assert!(
            result.deactivate.is_empty(),
            "neither cut should be deactivated; got {:?}",
            result.deactivate
        );
    }

    // ── Dominated cut is deactivated ─────────────────────────────────────────

    /// Cut B is dominated by cut A at all trial points → B is deactivated.
    #[test]
    fn dominated_cut_deactivated() {
        let mut pool = make_pool(1);
        // A: value(x) = 5 + 1*x
        // B: value(x) = 2 + 1*x   (always below A by exactly 3)
        add(&mut pool, 0, 5.0, &[1.0]);
        add(&mut pool, 1, 2.0, &[1.0]);

        // Cosine = 1.0 → same cluster.
        let visited = vec![0.0_f64, 10.0_f64, 100.0_f64];
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);

        // Slot for fp_idx=0: 1*32+0=32, fp_idx=1: 1*32+1=33
        assert_eq!(
            result.deactivate,
            vec![33],
            "B (slot 33) should be deactivated"
        );
    }

    // ── Equal-value cuts: both survive (H2 preservation) ────────────────────

    /// Two cuts with identical values at all trial points must NOT both be
    /// deactivated — at least one must survive to preserve H2.
    #[test]
    fn equal_valued_cuts_one_survives() {
        let mut pool = make_pool(1);
        // A and B have identical coefficients and intercepts.
        add(&mut pool, 0, 5.0, &[1.0]);
        add(&mut pool, 1, 5.0, &[1.0]);

        let visited = vec![0.0_f64, 10.0_f64];
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);
        // Neither should be deactivated — they are tied, not dominated.
        assert!(
            result.deactivate.is_empty(),
            "equal-valued cuts must not be deactivated: got {:?}",
            result.deactivate
        );
    }

    /// Two zero-norm cuts with equal intercepts: both survive.
    #[test]
    fn equal_intercept_zero_norm_both_survive() {
        let mut pool = make_pool(1);
        add(&mut pool, 0, 5.0, &[0.0]);
        add(&mut pool, 1, 5.0, &[0.0]);

        let visited = vec![0.0_f64];
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);
        assert!(
            result.deactivate.is_empty(),
            "equal-intercept zero-norm cuts must not be deactivated"
        );
    }

    // ── Orthogonal cuts: different clusters, no deactivation ─────────────────

    /// Two orthogonal cuts have cosine similarity 0 → different clusters.
    #[test]
    fn orthogonal_cuts_no_deactivation() {
        let mut pool = make_pool(2);
        // A: coefficients [1, 0]  (points along x-axis)
        // B: coefficients [0, 1]  (points along y-axis)
        // Cosine = 0.0 < threshold 0.999 → different clusters (singletons).
        add(&mut pool, 0, 0.0, &[1.0, 0.0]);
        add(&mut pool, 1, 0.0, &[0.0, 1.0]);

        let visited = vec![1.0_f64, 0.0, 0.0, 1.0]; // two 2-d points
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);
        assert!(
            result.deactivate.is_empty(),
            "orthogonal cuts must not be deactivated; got {:?}",
            result.deactivate
        );
    }

    // ── Zero-norm dominated ──────────────────────────────────────────────────

    /// Two zero-norm cuts: intercept 5.0 and 10.0 → the 5.0 cut is dominated.
    #[test]
    fn zero_norm_dominated() {
        let mut pool = make_pool(2);
        // Both cuts have zero-norm coefficient vectors.
        add(&mut pool, 0, 5.0, &[0.0, 0.0]);
        add(&mut pool, 1, 10.0, &[0.0, 0.0]);

        let visited = vec![1.0_f64, 2.0, 3.0, 4.0]; // two trial points (irrelevant for zero-norm)
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);

        // Slot for fp_idx=0 is 32 (intercept 5.0 → dominated).
        assert_eq!(
            result.deactivate,
            vec![32],
            "cut with intercept 5.0 (slot 32) should be deactivated"
        );
    }

    // ── Current iteration excluded ────────────────────────────────────────────

    /// Cuts from the current iteration must never appear in deactivate.
    #[test]
    fn current_iteration_excluded() {
        let mut pool = make_pool(1);
        // Add two cuts in iteration=1 (dominated relationship holds).
        add(&mut pool, 0, 5.0, &[1.0]);
        add(&mut pool, 1, 2.0, &[1.0]);

        // Pass current_iteration=1 → both cuts were added in iteration 1 → excluded.
        let visited = vec![0.0_f64, 1.0];
        let result = select_angular_dominated(&pool, &visited, 0.999, 1);
        assert!(
            result.deactivate.is_empty(),
            "cuts from current iteration must not be deactivated; got {:?}",
            result.deactivate
        );
    }

    // ── Empty visited states returns empty ────────────────────────────────────

    #[test]
    fn empty_visited_states_returns_empty() {
        let mut pool = make_pool(1);
        add(&mut pool, 0, 5.0, &[1.0]);
        add(&mut pool, 1, 2.0, &[1.0]);

        let result = select_angular_dominated(&pool, &[], 0.999, 2);
        assert!(
            result.deactivate.is_empty(),
            "empty visited states must yield empty result"
        );
    }

    // ── Single active cut returns empty ──────────────────────────────────────

    #[test]
    fn single_active_cut_returns_empty() {
        let mut pool = make_pool(1);
        add(&mut pool, 0, 5.0, &[1.0]);

        let visited = vec![0.0_f64, 1.0, 2.0];
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);
        assert!(
            result.deactivate.is_empty(),
            "single cut cannot be dominated"
        );
    }

    // ── Three-cut cluster: only the dominated one removed ────────────────────

    /// Cuts A, B, C all near-parallel (1-d, same direction).
    /// A: intercept=10, B: intercept=7, C: intercept=3.
    /// A dominates both B and C, B dominates C.
    /// Only C is dominated at all trial points (by both A and B).
    /// B is NOT dominated (A dominates B, but there must be no point where B
    /// beats everything — wait, actually A always beats B, so B IS dominated).
    /// Corrected: A dominates B (always above by 3), A dominates C (always
    /// above by 7), B dominates C (always above by 4).
    /// So C is dominated by both A and B. B is dominated by A. A is not
    /// dominated.  Result: deactivate = [B, C].
    #[test]
    fn three_cut_cluster_partial_dominance() {
        let mut pool = make_pool(1);
        // A: value(x) = 10 + x  (slot 32)
        // B: value(x) =  7 + x  (slot 33)
        // C: value(x) =  3 + x  (slot 34)
        add(&mut pool, 0, 10.0, &[1.0]);
        add(&mut pool, 1, 7.0, &[1.0]);
        add(&mut pool, 2, 3.0, &[1.0]);

        let visited = vec![0.0_f64, 5.0, 10.0];
        let result = select_angular_dominated(&pool, &visited, 0.999, 2);

        // C (slot 34) dominated by A and by B.
        // B (slot 33) dominated by A.
        // A (slot 32) not dominated.
        let mut got = result.deactivate.clone();
        got.sort_unstable();
        assert_eq!(got, vec![33, 34], "B and C should be deactivated");
    }

    // ── Determinism ──────────────────────────────────────────────────────────

    /// Identical inputs produce bit-for-bit identical results across two calls.
    #[test]
    fn determinism() {
        let mut pool = make_pool(2);
        add(&mut pool, 0, 10.0, &[1.0, 0.0]);
        add(&mut pool, 1, 7.0, &[1.0, 0.0]);
        add(&mut pool, 2, 0.0, &[0.0, 1.0]); // orthogonal, different cluster

        let visited = vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 3.0];
        let r1 = select_angular_dominated(&pool, &visited, 0.999, 2);
        let r2 = select_angular_dominated(&pool, &visited, 0.999, 2);

        assert_eq!(
            r1.deactivate, r2.deactivate,
            "results must be deterministic"
        );
        assert_eq!(r1.clusters_formed, r2.clusters_formed);
        assert_eq!(r1.dominance_checks, r2.dominance_checks);
    }

    // ── High-dimensional correctness ─────────────────────────────────────────

    /// 10-dimensional state vectors: dominated cut correctly identified.
    #[test]
    fn high_dimensional_correctness() {
        let mut pool = make_pool(10);
        // A: all coefficients 1.0, intercept 100.0
        // B: all coefficients 1.0, intercept  50.0   (always below A by 50)
        let coeffs = vec![1.0_f64; 10];
        add(&mut pool, 0, 100.0, &coeffs);
        add(&mut pool, 1, 50.0, &coeffs);

        // Three 10-d trial points.
        let visited: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ];

        let result = select_angular_dominated(&pool, &visited, 0.999, 2);
        assert_eq!(
            result.deactivate,
            vec![33], // slot 33 = B (intercept 50.0)
            "B should be deactivated in 10-d case"
        );
    }
}
