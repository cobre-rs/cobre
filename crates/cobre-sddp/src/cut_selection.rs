//! Cut selection strategy for controlling cut pool growth during SDDP training.
//!
//! This module defines [`CutSelectionStrategy`] (three variants: Level1, LML1,
//! Dominated), [`CutMetadata`] (per-cut tracking data), and [`DeactivationSet`]
//! (the output of a selection scan for one stage).
//!
//! ## Design
//!
//! Both methods (`should_run`, `select`) are pure and infallible. Configuration
//! parameters are validated at load time, so runtime panics from zero
//! `check_frequency` are impossible.
//!
//! The `Dominated` variant identifies cuts that are dominated at every visited
//! forward-pass state: if a cut is always below the best active cut minus a
//! tolerance, it contributes nothing and is deactivated.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::cut::CutPool;
//! use cobre_sddp::cut_selection::{
//!     CutMetadata, CutSelectionStrategy, DeactivationSet,
//! };
//!
//! let strategy = CutSelectionStrategy::Level1 {
//!     threshold: 0,
//!     check_frequency: 5,
//! };
//!
//! // Should run at multiples of check_frequency (excluding 0).
//! assert!(!strategy.should_run(0));
//! assert!(!strategy.should_run(3));
//! assert!(strategy.should_run(5));
//! assert!(strategy.should_run(10));
//!
//! // Cuts with zero active_count are deactivated.
//! let mut pool = CutPool::new(2, 1, 1, 0);
//! pool.add_cut(0, 0, 1.0, &[1.0]);
//! pool.add_cut(1, 0, 2.0, &[2.0]);
//! pool.metadata[0].active_count = 0;
//! pool.metadata[1].active_count = 3;
//! let deact = strategy.select(&pool, &[], 10);
//! assert_eq!(deact.indices, vec![0]);
//! ```

// ---------------------------------------------------------------------------
// CutMetadata
// ---------------------------------------------------------------------------

/// Per-cut tracking metadata for cut selection strategies.
///
/// Stored alongside cut coefficients and intercepts in the pre-allocated cut
/// pool. All fields are initialized to zero / default values when the cut
/// slot is first populated. Updated inline during the backward pass (see
/// `crates/cobre-sddp/src/backward.rs` around line 994).
#[derive(Debug, Clone)]
pub struct CutMetadata {
    /// Iteration at which this cut was generated (1-based).
    ///
    /// Used to prevent deactivation of cuts generated in the current
    /// iteration.
    pub iteration_generated: u64,

    /// Forward pass index that generated this cut.
    ///
    /// Combined with `iteration_generated`, uniquely identifies the
    /// deterministic slot for this cut.
    pub forward_pass_index: u32,

    /// Cumulative number of times this cut was binding at an LP solution.
    ///
    /// Used by [`CutSelectionStrategy::Level1`]: deactivate if
    /// `active_count <= threshold`.
    /// Initialized to 0; incremented inline by the backward pass
    /// (`backward.rs:994`).
    pub active_count: u64,

    /// Most recent iteration at which this cut was binding.
    ///
    /// Used by [`CutSelectionStrategy::Lml1`]: deactivate if
    /// `current_iteration - last_active_iter > memory_window`.
    /// Initialized to `iteration_generated`; updated inline by the backward
    /// pass (`backward.rs:995`).
    pub last_active_iter: u64,

    /// Sliding-window binding-activity bitmap.
    ///
    /// Bit 0 = current iteration; bit `i` = iteration `current_iter - i`.
    /// Updated to bit-0-set when the cut was binding (dual >
    /// `cut_activity_tolerance`) during any backward solve of the current
    /// iteration; shifted left by 1 at end-of-iteration so the next
    /// iteration's bit 0 records fresh activity.
    ///
    /// Populated by the MPI `allreduce(BitwiseOr)` in the backward pass
    /// (so any rank observing the cut binding sets bit 0 globally). Consumed
    /// by the activity-guided basis classifier in `reconstruct_basis`.
    ///
    /// **Transient seed**: `add_cut` sets
    /// [`crate::basis_reconstruct::SEED_BIT`] (bit 31, outside
    /// `RECENT_WINDOW_BITS`) so the classifier fires LOWER on a freshly
    /// generated cut during the same iteration's remaining backward stages —
    /// the cut is tight at the x̂ it was derived from by construction. The
    /// end-of-iteration logic clears `SEED_BIT` *before* the `<<= 1` shift so
    /// the seed does **not** persist into the next iteration's basis
    /// reconstruction. From iteration i+1 onward, only genuine binding
    /// observations drive classification decisions.
    pub active_window: u32,
}

// ---------------------------------------------------------------------------
// DeactivationSet
// ---------------------------------------------------------------------------

/// Set of cut indices to deactivate at a single stage.
///
/// Returned by [`CutSelectionStrategy::select`]. The caller applies each
/// index to the activity bitmap by clearing the corresponding bit and
/// decrementing the active count. Indices are zero-based slot positions in
/// the pre-allocated cut pool.
///
/// The set may be empty if no cuts meet the deactivation criteria.
#[derive(Debug, Clone)]
pub struct DeactivationSet {
    /// Stage index (0-based) that this deactivation set belongs to.
    pub stage_index: u32,

    /// Cut slot indices to deactivate.
    pub indices: Vec<u32>,
}

// ---------------------------------------------------------------------------
// CutSelectionStrategy
// ---------------------------------------------------------------------------

/// Cut selection strategy for controlling cut pool growth during SDDP training.
///
/// One strategy is active for the entire training run (global setting, one
/// variant per run). All stages use the same strategy. Selection runs
/// periodically via [`should_run`] to amortize the cost of scanning the pool.
///
/// [`should_run`]: CutSelectionStrategy::should_run
#[derive(Debug, Clone)]
pub enum CutSelectionStrategy {
    /// Level-1 selection: retain any cut that has ever been binding.
    ///
    /// A cut is deactivated only if its cumulative `active_count` is at or
    /// below `threshold`. With `threshold = 0` (recommended), a cut is
    /// deactivated if and only if it has never been binding at any visited
    /// state. This is the least aggressive strategy and preserves the
    /// convergence guarantee.
    Level1 {
        /// Activity count threshold. A cut is deactivated when
        /// `active_count <= threshold`. Typical value: 0.
        threshold: u64,

        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,
    },

    /// Limited Memory Level-1 (LML1): retain cuts active within a recent window.
    ///
    /// Each cut is timestamped with the most recent iteration at which it was
    /// binding. Cuts whose timestamp is older than `memory_window` iterations
    /// are deactivated. More aggressive than Level1 because cuts that were
    /// active early but are now dominated by newer cuts will eventually be
    /// removed.
    Lml1 {
        /// Number of iterations to retain inactive cuts before deactivation.
        /// A cut is deactivated when `current_iteration - last_active_iter >
        /// memory_window`.
        memory_window: u64,

        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,
    },

    /// Dominated cut detection: remove cuts dominated at all visited states.
    ///
    /// A cut is dominated if at every visited forward pass state, some other
    /// active cut achieves a higher (or equal within threshold) value.
    /// Computationally expensive: O(|active cuts| x |visited states|) per
    /// stage per check.
    Dominated {
        /// Activity threshold epsilon. Ignored by the stub implementation.
        threshold: f64,

        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,
    },
}

impl CutSelectionStrategy {
    /// Determine whether cut selection should run at the given iteration.
    ///
    /// Returns `true` if `iteration > 0` and `iteration` is a multiple of
    /// the variant's `check_frequency`. Never runs at iteration 0 (no cuts
    /// exist yet).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::cut_selection::CutSelectionStrategy;
    ///
    /// let s = CutSelectionStrategy::Level1 { threshold: 0, check_frequency: 5 };
    /// assert!(!s.should_run(0));
    /// assert!(!s.should_run(3));
    /// assert!(s.should_run(5));
    /// assert!(s.should_run(10));
    /// ```
    #[must_use]
    pub fn should_run(&self, iteration: u64) -> bool {
        let freq = match self {
            Self::Level1 {
                check_frequency, ..
            }
            | Self::Lml1 {
                check_frequency, ..
            }
            | Self::Dominated {
                check_frequency, ..
            } => *check_frequency,
        };
        iteration > 0 && iteration % freq == 0
    }

    /// Scan the cut pool metadata for a single stage and identify cuts to
    /// deactivate.
    ///
    /// Returns a [`DeactivationSet`] whose `indices` are the zero-based slot
    /// positions of cuts that should be deactivated. The caller is responsible
    /// for applying the deactivation to the activity bitmap. This method does
    /// not modify the metadata — it is a pure query.
    ///
    /// `stage_index` identifies which stage this selection runs for (used to
    /// populate [`DeactivationSet::stage_index`]).
    ///
    /// Only slots where `active[i]` is `true` are considered. Slots that are
    /// already inactive (e.g. unpopulated slots below the high-water mark or
    /// previously deactivated cuts) are unconditionally skipped.
    ///
    /// # Variant behavior
    ///
    /// - **Level1**: deactivates cuts with `active_count <= threshold`.
    /// - **Lml1**: deactivates cuts with
    ///   `current_iteration - last_active_iter > memory_window`.
    /// - **Dominated**: deactivates cuts dominated at every visited state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::cut::{CutPool};
    /// use cobre_sddp::cut_selection::{CutMetadata, CutSelectionStrategy};
    ///
    /// let strategy = CutSelectionStrategy::Level1 { threshold: 0, check_frequency: 5 };
    /// let mut pool = CutPool::new(2, 1, 1, 0);
    /// pool.add_cut(0, 0, 1.0, &[1.0]);
    /// pool.add_cut(1, 0, 2.0, &[2.0]);
    /// pool.metadata[0].active_count = 0;
    /// pool.metadata[1].active_count = 2;
    /// let deact = strategy.select(&pool, &[], 10);
    /// assert_eq!(deact.indices, vec![0]);
    /// ```
    #[must_use]
    pub fn select(
        &self,
        pool: &crate::cut::CutPool,
        visited_states: &[f64],
        current_iteration: u64,
    ) -> DeactivationSet {
        self.select_for_stage(pool, visited_states, current_iteration, 0)
    }

    /// Scan the cut pool for a specific stage and identify cuts to deactivate.
    ///
    /// Accepts the full [`CutPool`](crate::cut::CutPool) reference so that the
    /// `Dominated` variant can access coefficients and intercepts. `Level1`
    /// and `Lml1` read only `pool.metadata` and `pool.active`.
    ///
    /// `visited_states` is a flat `&[f64]` of visited forward-pass state
    /// vectors (row-major, one state per `pool.state_dimension` elements).
    /// Pass `&[]` when using `Level1` or `Lml1`.
    ///
    /// [`select`]: CutSelectionStrategy::select
    #[must_use]
    pub fn select_for_stage(
        &self,
        pool: &crate::cut::CutPool,
        visited_states: &[f64],
        current_iteration: u64,
        stage_index: u32,
    ) -> DeactivationSet {
        let populated = pool.populated_count;
        let metadata = &pool.metadata[..populated];
        let active = &pool.active[..populated];

        // Cut pool capacity is bounded by a u32 field in the pool header, so
        // enumerate indices always fit in u32. The cast is safe by structural
        // invariant established at pool construction time.
        #[allow(clippy::cast_possible_truncation)]
        let indices = match self {
            Self::Level1 { threshold, .. } => metadata
                .iter()
                .enumerate()
                .filter(|(i, m)| {
                    active[*i]
                        && m.active_count <= *threshold
                        && m.iteration_generated < current_iteration
                })
                .map(|(i, _)| i as u32)
                .collect(),

            Self::Lml1 { memory_window, .. } => metadata
                .iter()
                .enumerate()
                .filter(|(i, m)| {
                    active[*i]
                        && m.iteration_generated < current_iteration
                        && current_iteration.saturating_sub(m.last_active_iter) > *memory_window
                })
                .map(|(i, _)| i as u32)
                .collect(),

            Self::Dominated { threshold, .. } => {
                select_dominated(pool, visited_states, *threshold, current_iteration)
            }
        };

        DeactivationSet {
            stage_index,
            indices,
        }
    }
}

// ---------------------------------------------------------------------------
// Dominated selection algorithm
// ---------------------------------------------------------------------------

/// Identify cuts that are dominated at every visited forward-pass state.
///
/// A cut `k` is *dominated* if, for every state `x_hat` in `visited_states`,
/// there exists another active cut whose value at `x_hat` exceeds (or matches
/// within `threshold`) the value of cut `k`.  Dominated cuts contribute
/// nothing to the policy and can safely be deactivated.
///
/// Returns the slot indices of dominated cuts as `Vec<u32>`.
#[allow(clippy::cast_possible_truncation, clippy::needless_range_loop)]
fn select_dominated(
    pool: &crate::cut::CutPool,
    visited_states: &[f64],
    threshold: f64,
    current_iteration: u64,
) -> Vec<u32> {
    let populated = pool.populated_count;
    let n_state = pool.state_dimension;

    // No states means no evidence of domination.
    if visited_states.is_empty() || n_state == 0 {
        return vec![];
    }

    // Need at least 2 active cuts for one to dominate another.
    if pool.active_count() < 2 {
        return vec![];
    }

    // is_candidate[k] = true means cut k is still a candidate for
    // deactivation (it has been dominated at every state seen so far).
    // Initialize to active && not from the current iteration.
    let mut is_candidate: Vec<bool> = pool.active[..populated]
        .iter()
        .zip(pool.metadata[..populated].iter())
        .map(|(&a, m)| a && m.iteration_generated < current_iteration)
        .collect();

    let mut n_candidates: usize = is_candidate.iter().filter(|&&c| c).count();
    if n_candidates == 0 {
        return vec![];
    }

    // Scratch buffer for cut values at the current state.
    let mut scratch = vec![0.0_f64; populated];

    for x_hat in visited_states.chunks_exact(n_state) {
        // Step 1: compute values for all active cuts, find max.
        let mut max_val = f64::NEG_INFINITY;
        for k in 0..populated {
            if pool.active[k] {
                let coeff_start = k * n_state;
                let val = pool.intercepts[k]
                    + pool.coefficients[coeff_start..coeff_start + n_state]
                        .iter()
                        .zip(x_hat)
                        .map(|(c, x)| c * x)
                        .sum::<f64>();
                scratch[k] = val;
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // Step 2: any candidate that achieves (max - threshold) is NOT
        // dominated at this state -- remove it from candidates.
        let cutoff = max_val - threshold;
        for k in 0..populated {
            if is_candidate[k] && scratch[k] >= cutoff {
                is_candidate[k] = false;
                n_candidates -= 1;
            }
        }

        // Step 3: early exit when no candidates remain.
        if n_candidates == 0 {
            break;
        }
    }

    // Remaining candidates are dominated at ALL visited states.
    (0..populated)
        .filter(|&k| is_candidate[k])
        .map(|k| k as u32)
        .collect()
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Parse a [`cobre_io::config::CutSelectionConfig`] into an optional
/// [`CutSelectionStrategy`].
///
/// Returns `None` when disabled (default). Returns `Err` when explicitly
/// enabled with invalid configuration (unknown method, `enabled = true` with no
/// method, or `check_frequency = 0`). Defaults: `threshold = 0`,
/// `check_frequency = 5`.
///
/// # Errors
///
/// Returns `Err(String)` when `enabled = true` but no `method` is specified,
/// when the `method` string is not a recognised variant, or when
/// `check_frequency = 0`.
pub fn parse_cut_selection_config(
    config: &cobre_io::config::CutSelectionConfig,
) -> Result<Option<CutSelectionStrategy>, String> {
    let enabled = config.enabled.unwrap_or(false);
    if !enabled {
        return Ok(None);
    }

    let method = config
        .method
        .as_deref()
        .ok_or_else(|| "cut_selection.enabled is true but method is not specified".to_string())?;

    let threshold = config.threshold.unwrap_or(0);
    let check_frequency = config.check_frequency.unwrap_or(5);

    if check_frequency == 0 {
        return Err("cut_selection.check_frequency must be > 0".to_string());
    }

    match method {
        "level1" => Ok(Some(CutSelectionStrategy::Level1 {
            threshold: u64::from(threshold),
            check_frequency: u64::from(check_frequency),
        })),
        "lml1" => {
            let window = config.memory_window.unwrap_or(threshold);
            Ok(Some(CutSelectionStrategy::Lml1 {
                memory_window: u64::from(window),
                check_frequency: u64::from(check_frequency),
            }))
        }
        "domination" => {
            let epsilon = config
                .domination_epsilon
                .unwrap_or_else(|| f64::from(threshold));
            Ok(Some(CutSelectionStrategy::Dominated {
                threshold: epsilon,
                check_frequency: u64::from(check_frequency),
            }))
        }
        other => Err(format!("unknown cut_selection.method: \"{other}\"")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::parse_cut_selection_config;
    use super::{CutMetadata, CutSelectionStrategy, DeactivationSet};
    use crate::cut::CutPool;
    use cobre_io::config::CutSelectionConfig;

    fn make_meta(active_count: u64, last_active_iter: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: 1,
            forward_pass_index: 0,
            active_count,
            last_active_iter,
            active_window: 0,
        }
    }

    /// Build a `CutPool` pre-populated with the given metadata and active flags.
    #[allow(clippy::cast_possible_truncation)]
    fn make_pool(metadata: &[CutMetadata], active: &[bool]) -> CutPool {
        let n = metadata.len();
        let mut pool = CutPool::new(n, 1, 1, 0);
        // Populate dummy cuts so populated_count advances.
        for i in 0..n {
            pool.add_cut(0, i as u32, 0.0, &[0.0]);
        }
        pool.metadata[..n].clone_from_slice(metadata);
        pool.active[..n].clone_from_slice(active);
        pool.cached_active_count = active.iter().filter(|&&a| a).count();
        pool
    }

    #[test]
    fn should_run_false_at_zero() {
        let s = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        assert!(!s.should_run(0));
    }

    #[test]
    fn should_run_false_between_multiples() {
        let s = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        assert!(!s.should_run(3));
        assert!(!s.should_run(7));
    }

    #[test]
    fn should_run_true_at_multiples() {
        let s = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        assert!(s.should_run(5));
        assert!(s.should_run(10));
        assert!(s.should_run(15));
    }

    #[test]
    fn should_run_lml1_respects_check_frequency() {
        let s = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        assert!(!s.should_run(0));
        assert!(!s.should_run(3));
        assert!(s.should_run(5));
        assert!(s.should_run(10));
    }

    #[test]
    fn should_run_dominated_respects_check_frequency() {
        let s = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 10,
        };
        assert!(!s.should_run(5));
        assert!(s.should_run(10));
    }

    #[test]
    fn level1_deactivates_zero_activity_cuts() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let pool = make_pool(&[make_meta(0, 1), make_meta(1, 5)], &[true, true]);
        let deact = strategy.select(&pool, &[], 10);
        assert_eq!(
            deact.indices,
            vec![0],
            "only the inactive cut is deactivated"
        );
    }

    #[test]
    fn level1_retains_positive_activity_cuts() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let pool = make_pool(&[make_meta(3, 1), make_meta(7, 5)], &[true, true]);
        let deact = strategy.select(&pool, &[], 10);
        assert!(
            deact.indices.is_empty(),
            "no cuts should be deactivated when all have activity"
        );
    }

    #[test]
    fn level1_threshold_1_deactivates_cuts_with_count_at_most_1() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 1,
            check_frequency: 5,
        };
        let pool = make_pool(
            &[make_meta(0, 1), make_meta(1, 5), make_meta(2, 8)],
            &[true, true, true],
        );
        let deact = strategy.select(&pool, &[], 10);
        assert_eq!(deact.indices, vec![0, 1]);
    }

    #[test]
    fn level1_empty_metadata_returns_empty_set() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let pool = CutPool::new(0, 1, 1, 0);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.indices.is_empty());
    }

    #[test]
    fn lml1_deactivates_cuts_outside_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let pool = make_pool(&[make_meta(0, 5)], &[true]);
        let deact = strategy.select(&pool, &[], 20);
        assert_eq!(deact.indices, vec![0]);
    }

    #[test]
    fn lml1_retains_cuts_within_memory_window() {
        // memory_window=10, iteration=20. Cut with last_active_iter=12:
        // 20 - 12 = 8, not > 10 → retained.
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let pool = make_pool(&[make_meta(0, 12)], &[true]);
        let deact = strategy.select(&pool, &[], 20);
        assert!(deact.indices.is_empty());
    }

    #[test]
    fn lml1_retains_cuts_exactly_at_boundary() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let pool = make_pool(&[make_meta(0, 10)], &[true]);
        let deact = strategy.select(&pool, &[], 20);
        assert!(
            deact.indices.is_empty(),
            "boundary case: exactly at window edge, retained"
        );
    }

    #[test]
    fn lml1_mixed_cuts_deactivates_correct_indices() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let pool = make_pool(
            &[make_meta(0, 5), make_meta(0, 12), make_meta(0, 1)],
            &[true, true, true],
        );
        let deact = strategy.select(&pool, &[], 20);
        assert_eq!(deact.indices, vec![0, 2]);
    }

    // Dominated select (stub): always returns empty set

    #[test]
    fn dominated_select_always_returns_empty_set() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.001,
            check_frequency: 10,
        };
        let pool = make_pool(&[make_meta(0, 1), make_meta(0, 1)], &[true, true]);
        let deact = strategy.select(&pool, &[], 20);
        assert!(deact.indices.is_empty());
    }

    #[test]
    fn ac_level1_threshold_0_deactivates_zero_activity_cut() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let pool = make_pool(
            &[CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 1,
                active_window: 0,
            }],
            &[true],
        );
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.indices.contains(&0));
    }

    #[test]
    fn ac_lml1_deactivates_cut_outside_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let pool = make_pool(
            &[CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 5,
                active_window: 0,
            }],
            &[true],
        );
        let deact = strategy.select(&pool, &[], 20);
        assert!(deact.indices.contains(&0));
    }

    #[test]
    fn select_for_stage_sets_stage_index() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let pool = make_pool(&[make_meta(0, 1)], &[true]);
        let deact = strategy.select_for_stage(&pool, &[], 10, 7);
        assert_eq!(deact.stage_index, 7);
    }

    #[test]
    fn select_sets_stage_index_to_zero() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let pool = CutPool::new(0, 1, 1, 0);
        let deact = strategy.select(&pool, &[], 10);
        assert_eq!(deact.stage_index, 0);
    }

    #[test]
    fn deactivation_set_derives_debug_and_clone() {
        let deact = DeactivationSet {
            stage_index: 2,
            indices: vec![0, 3, 7],
        };
        let cloned = deact.clone();
        assert_eq!(cloned.stage_index, 2);
        assert_eq!(cloned.indices, vec![0, 3, 7]);
        assert!(!format!("{deact:?}").is_empty());
    }

    #[test]
    fn cut_metadata_derives_debug_and_clone() {
        let meta = make_meta(5, 10);
        let cloned = meta.clone();
        assert_eq!(cloned.active_count, 5);
        assert!(!format!("{meta:?}").is_empty());
    }

    // -----------------------------------------------------------------------
    // parse_cut_selection_config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_disabled_default() {
        let cfg = CutSelectionConfig::default();
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "default config must produce None (disabled)"
        );
    }

    #[test]
    fn test_parse_level1() {
        let cfg = CutSelectionConfig {
            enabled: Some(true),
            method: Some("level1".to_string()),
            threshold: Some(0),
            check_frequency: Some(5),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        let strategy = result
            .unwrap()
            .expect("must produce Some for enabled level1");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Level1 {
                    threshold: 0,
                    check_frequency: 5,
                }
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_lml1() {
        let cfg = CutSelectionConfig {
            enabled: Some(true),
            method: Some("lml1".to_string()),
            threshold: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        let strategy = result.unwrap().expect("must produce Some for enabled lml1");
        // threshold defaults to 0 (mapped to memory_window), check_frequency defaults to 5.
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Lml1 {
                    memory_window: 0,
                    check_frequency: 5,
                }
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_domination() {
        let cfg = CutSelectionConfig {
            enabled: Some(true),
            method: Some("domination".to_string()),
            threshold: Some(0),
            check_frequency: Some(10),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        let strategy = result
            .unwrap()
            .expect("must produce Some for enabled domination");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dominated {
                    threshold,
                    check_frequency: 10,
                } if threshold == 0.0
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_unknown_method() {
        let cfg = CutSelectionConfig {
            enabled: Some(true),
            method: Some("bogus".to_string()),
            threshold: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("bogus"),
            "error message must contain the unrecognized method name, got: {msg}"
        );
    }

    #[test]
    fn test_parse_enabled_without_method() {
        let cfg = CutSelectionConfig {
            enabled: Some(true),
            method: None,
            threshold: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_enabled_false_with_method_returns_none() {
        let cfg = CutSelectionConfig {
            enabled: Some(false),
            method: Some("level1".to_string()),
            threshold: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
        };
        let result = parse_cut_selection_config(&cfg).unwrap();
        assert!(
            result.is_none(),
            "enabled=false must return None even when method is set"
        );
    }

    #[test]
    fn test_parse_zero_check_frequency() {
        let cfg = CutSelectionConfig {
            enabled: Some(true),
            method: Some("level1".to_string()),
            threshold: None,
            check_frequency: Some(0),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("check_frequency"),
            "error message must mention check_frequency, got: {msg}"
        );
    }

    /// Re-deactivation of already-inactive cut does not corrupt `cached_active_count`.
    ///
    /// When Level1 `select_for_stage` returns an index that is already
    /// deactivated, `CutPool::deactivate` must skip the decrement. This test
    /// verifies the safety invariant via the cut pool directly.
    #[test]
    fn select_skips_already_inactive_slots() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[1.0]); // slot 0, active
        pool.add_cut(1, 0, 2.0, &[2.0]); // slot 1, active
        pool.add_cut(2, 0, 3.0, &[3.0]); // slot 2, active
        assert_eq!(pool.active_count(), 3);

        // Mark slots 1 and 2 as having binding activity so Level1 retains them.
        pool.metadata[1].active_count = 5;
        pool.metadata[2].active_count = 3;

        // Deactivate slot 0 manually.
        pool.deactivate(&[0]);
        assert_eq!(pool.active_count(), 2);

        // Level1 selection: slot 0 is already inactive and must be skipped.
        // Slots 1 and 2 have active_count > 0 and are retained.
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 1,
        };
        let deact = strategy.select_for_stage(&pool, &[], 5, 0);
        assert!(
            deact.indices.is_empty(),
            "no cuts should be selected: slot 0 is already inactive, \
             slots 1 and 2 have activity"
        );
    }

    /// Lml1 `memory_window` boundary: cuts at exactly the window edge are
    /// retained, cuts beyond are deactivated.
    ///
    /// With `memory_window: 3, current_iteration: 10`:
    /// - `last_active_iter=1`: 10-1=9 > 3 → deactivated
    /// - `last_active_iter=5`: 10-5=5 > 3 → deactivated
    /// - `last_active_iter=7`: 10-7=3 not > 3 → retained (boundary)
    /// - `last_active_iter=8`: 10-8=2 not > 3 → retained
    /// - `last_active_iter=10`: 10-10=0 not > 3 → retained
    #[test]
    fn lml1_memory_window_boundary_behavior() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 3,
            check_frequency: 1,
        };
        let pool = make_pool(
            &[
                make_meta(0, 1),  // last_active_iter = 1
                make_meta(0, 5),  // last_active_iter = 5
                make_meta(0, 7),  // last_active_iter = 7 (boundary)
                make_meta(0, 8),  // last_active_iter = 8
                make_meta(0, 10), // last_active_iter = 10
            ],
            &[true; 5],
        );
        let deact = strategy.select_for_stage(&pool, &[], 10, 0);

        // Slots 0 and 1 deactivated, slots 2-4 retained.
        assert_eq!(
            deact.indices,
            vec![0, 1],
            "only cuts with last_active_iter 1 and 5 should be deactivated"
        );
    }

    /// Cuts generated in the current iteration must never be deactivated by
    /// Level1, even if their `active_count` is 0 (they haven't been tested
    /// yet). This prevents the pathology where every new cut is immediately
    /// killed, causing the lower bound to stagnate.
    #[test]
    fn level1_spares_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 1,
        };
        let pool = make_pool(
            &[
                CutMetadata {
                    iteration_generated: 10, // same as current_iteration
                    forward_pass_index: 0,
                    active_count: 0,
                    last_active_iter: 10,
                    active_window: 0,
                },
                CutMetadata {
                    iteration_generated: 5, // older, zero activity
                    forward_pass_index: 0,
                    active_count: 0,
                    last_active_iter: 5,
                    active_window: 0,
                },
            ],
            &[true, true],
        );
        let deact = strategy.select(&pool, &[], 10);
        assert_eq!(
            deact.indices,
            vec![1],
            "only the older cut (iter 5) should be deactivated; \
             the current-iteration cut (iter 10) must be spared"
        );
    }

    /// Lml1 also spares cuts from the current iteration via the
    /// `iteration_generated` guard (defense-in-depth; `last_active_iter`
    /// initialization already protects, but the guard is explicit).
    #[test]
    fn lml1_spares_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 0,
            check_frequency: 1,
        };
        let pool = make_pool(
            &[CutMetadata {
                iteration_generated: 10,
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 10,
                active_window: 0,
            }],
            &[true],
        );
        let deact = strategy.select(&pool, &[], 10);
        assert!(
            deact.indices.is_empty(),
            "current-iteration cut must not be deactivated even with memory_window=0"
        );
    }

    // -----------------------------------------------------------------------
    // Dominated algorithm tests (SS1.3 conformance + aggressiveness ordering)
    // -----------------------------------------------------------------------

    /// Build a `CutPool` with known coefficients, intercepts, and metadata
    /// for testing the dominated selection algorithm.
    #[allow(clippy::cast_possible_truncation)]
    fn make_dominated_pool(
        intercepts: &[f64],
        coefficients: &[Vec<f64>],
        active: &[bool],
        metadata: &[CutMetadata],
    ) -> CutPool {
        let n = intercepts.len();
        let state_dim = coefficients[0].len();
        let mut pool = CutPool::new(n, state_dim, 1, 0);
        for i in 0..n {
            // Use add_cut to advance populated_count correctly.
            pool.add_cut(0, i as u32, intercepts[i], &coefficients[i]);
            pool.metadata[i] = metadata[i].clone();
            pool.active[i] = active[i];
        }
        pool.cached_active_count = active.iter().filter(|&&a| a).count();
        pool
    }

    fn default_meta_at(iter: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: iter,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: iter,
            active_window: 0,
        }
    }

    fn default_meta_vec(n: usize, iter: u64) -> Vec<CutMetadata> {
        (0..n).map(|_| default_meta_at(iter)).collect()
    }

    /// SS1.3 test 1: 5 cuts, 3 states (1D). Cuts 0,3,4 dominated at all states.
    #[test]
    fn dominated_select_deactivate_dominated() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 0.0, 3.0, 0.5, 0.0],
            &[
                vec![0.0],  // cut 0: constant 1
                vec![2.0],  // cut 1: 2x
                vec![-1.0], // cut 2: 3 - x
                vec![0.0],  // cut 3: constant 0.5
                vec![0.5],  // cut 4: 0.5x
            ],
            &[true; 5],
            &default_meta_vec(5, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert_eq!(deact.indices, vec![0, 3, 4]);
    }

    /// SS1.3 test 2: cut dominated at 2/3 states but tied at 1 -> retained.
    #[test]
    fn dominated_select_partial_domination_retained() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: intercept=2, slope=0 (constant 2)
        // Cut 1: intercept=0, slope=2 (2x)
        // At x=0: values=[2, 0] -> max=2, cut 0 achieves max -> not dominated
        // At x=1: values=[2, 2] -> max=2, cut 0 achieves max -> not dominated
        // At x=3: values=[2, 6] -> max=6, cut 0 below -> dominated at this state
        // Net: cut 0 is NOT dominated (achieves max at x=0 and x=1)
        let pool = make_dominated_pool(
            &[2.0, 0.0],
            &[vec![0.0], vec![2.0]],
            &[true, true],
            &default_meta_vec(2, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert!(
            deact.indices.is_empty(),
            "cut 0 achieves max at x=0 and x=1, must not be deactivated"
        );
    }

    /// SS1.3 test 3: 3 cuts, each achieves max at >= 1 state.
    /// But cut 2 never achieves max alone (always below another).
    #[test]
    fn dominated_select_none_dominated_when_all_achieve_max() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: 5 - 2x (max at x=0: 5)
        // Cut 1: 0 + 3x (max at x=3: 9)
        // Cut 2: 2 + 0x (constant 2, never achieves max)
        // At x=0: [5, 0, 2] -> max=5, cut 0 achieves max
        // At x=1: [3, 3, 2] -> max=3, cuts 0,1 achieve max
        // At x=3: [−1, 9, 2] -> max=9, cut 1 achieves max
        // Dominated: cut 2 (never achieves max)
        let pool = make_dominated_pool(
            &[5.0, 0.0, 2.0],
            &[vec![-2.0], vec![3.0], vec![0.0]],
            &[true; 3],
            &default_meta_vec(3, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert_eq!(
            deact.indices,
            vec![2],
            "only cut 2 (constant 2) should be dominated"
        );
    }

    /// SS1.3 test 4: empty `visited_states` returns empty set.
    #[test]
    fn dominated_select_empty_states() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 2.0],
            &[vec![0.0], vec![0.0]],
            &[true, true],
            &default_meta_vec(2, 1),
        );
        let deact = strategy.select(&pool, &[], 10);
        assert!(
            deact.indices.is_empty(),
            "empty visited_states must produce empty deactivation set"
        );
    }

    /// SS1.3 test 5: single active cut returns empty set.
    #[test]
    fn dominated_select_single_active_cut() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 2.0, 3.0],
            &[vec![0.0], vec![0.0], vec![0.0]],
            &[true, false, false],
            &default_meta_vec(3, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0];
        let deact = strategy.select(&pool, &states, 10);
        assert!(
            deact.indices.is_empty(),
            "single active cut cannot be dominated"
        );
    }

    /// SS1.3 test 6: cut from current iteration excluded from deactivation.
    #[test]
    fn dominated_select_current_iteration_excluded() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: constant 1 (from current iteration 10 -- protected)
        // Cut 1: constant 5 (from iteration 1 -- dominates cut 0)
        let pool = make_dominated_pool(
            &[1.0, 5.0],
            &[vec![0.0], vec![0.0]],
            &[true, true],
            &[default_meta_at(10), default_meta_at(1)],
        );
        let states: Vec<f64> = vec![0.0, 1.0];
        let deact = strategy.select(&pool, &states, 10);
        assert!(
            deact.indices.is_empty(),
            "cut from current iteration must not be deactivated even if dominated"
        );
    }

    /// Aggressiveness ordering: |Level1| <= |LML1| <= |Dominated|.
    ///
    /// Build a fixture where:
    /// - Level1 (threshold=0) deactivates cuts with 0 activity count
    /// - LML1 (window=3) deactivates those + cuts with old `last_active_iter`
    /// - Dominated deactivates geometrically dominated cuts
    #[test]
    fn aggressiveness_ordering_level1_leq_lml1_leq_dominated() {
        // 5 cuts (1D):
        // Cut 0: intercept=0, slope=0 (constant 0), activity=0, last_active=1
        // Cut 1: intercept=0, slope=0.1, activity=0, last_active=2
        // Cut 2: intercept=1, slope=0, activity=3, last_active=3
        // Cut 3: intercept=0, slope=2, activity=5, last_active=10
        // Cut 4: intercept=5, slope=-1, activity=5, last_active=10
        let meta = [
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 1,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 1,
                active_count: 0,
                last_active_iter: 2,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 2,
                active_count: 3,
                last_active_iter: 3,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 3,
                active_count: 5,
                last_active_iter: 10,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 4,
                active_count: 5,
                last_active_iter: 10,
                active_window: 0,
            },
        ];
        let pool = make_dominated_pool(
            &[0.0, 0.0, 1.0, 0.0, 5.0],
            &[vec![0.0], vec![0.1], vec![0.0], vec![2.0], vec![-1.0]],
            &[true; 5],
            &meta,
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0, 5.0];

        // Level1 threshold=0: deactivates cuts with active_count=0
        let l1 = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 1,
        };
        let deact_l1 = l1.select(&pool, &[], 11);

        // LML1 window=3: deactivates cuts with last_active_iter < 11-3=8
        let lml1 = CutSelectionStrategy::Lml1 {
            memory_window: 3,
            check_frequency: 1,
        };
        let deact_lml1 = lml1.select(&pool, &[], 11);

        // Dominated threshold=0
        let dom = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let deact_dom = dom.select(&pool, &states, 11);

        assert!(
            deact_l1.indices.len() <= deact_lml1.indices.len(),
            "Level1 ({}) should deactivate <= LML1 ({})",
            deact_l1.indices.len(),
            deact_lml1.indices.len()
        );
        assert!(
            deact_lml1.indices.len() <= deact_dom.indices.len(),
            "LML1 ({}) should deactivate <= Dominated ({})",
            deact_lml1.indices.len(),
            deact_dom.indices.len()
        );
    }
}
