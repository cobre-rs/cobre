//! Cut selection strategy for controlling cut pool growth during SDDP training.
//!
//! This module defines [`CutSelectionStrategy`] (three variants: Level1, LML1,
//! Dominated), [`CutMetadata`] (per-cut tracking data), and [`DeactivationSet`]
//! (the output of a selection scan for one stage).
//!
//! ## Design
//!
//! All three methods (`should_run`, `select`, `update_activity`) are pure and
//! infallible. Configuration parameters are validated at load time, so runtime
//! panics from zero `check_frequency` are impossible.
//!
//! The `Dominated` variant's `select` is a stub returning an empty
//! [`DeactivationSet`]; full implementation requires access to forward pass
//! visited states and is deferred to Epic 04.
//!
//! ## Usage
//!
//! ```rust
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
//! let metadata = vec![
//!     CutMetadata { iteration_generated: 1, forward_pass_index: 0,
//!                   active_count: 0, last_active_iter: 1, domination_count: 0 },
//!     CutMetadata { iteration_generated: 1, forward_pass_index: 1,
//!                   active_count: 3, last_active_iter: 5, domination_count: 0 },
//! ];
//! let deact = strategy.select(&metadata, 10);
//! assert_eq!(deact.indices, vec![0]);
//! ```

// ---------------------------------------------------------------------------
// CutMetadata
// ---------------------------------------------------------------------------

/// Per-cut tracking metadata for cut selection strategies.
///
/// Stored alongside cut coefficients and intercepts in the pre-allocated cut
/// pool. All fields are initialized to zero / default values when the cut
/// slot is first populated. Updated during the backward pass via
/// [`CutSelectionStrategy::update_activity`].
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
    /// Initialized to 0; incremented by `update_activity` for Level1.
    pub active_count: u64,

    /// Most recent iteration at which this cut was binding.
    ///
    /// Used by [`CutSelectionStrategy::Lml1`]: deactivate if
    /// `current_iteration - last_active_iter > memory_window`.
    /// Initialized to `iteration_generated`; updated by `update_activity` for
    /// LML1.
    pub last_active_iter: u64,

    /// Number of visited states at which this cut is dominated by other cuts.
    ///
    /// Used by [`CutSelectionStrategy::Dominated`]: deactivate if dominated
    /// at ALL visited states. Reset to 0 when the cut is binding at any
    /// state.
    pub domination_count: u64,
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
    /// Computationally expensive: O(|active cuts| × |visited states|) per
    /// stage per check.
    ///
    /// **Note:** `select` for this variant is a stub returning an empty
    /// [`DeactivationSet`]. Full implementation requires visited states from
    /// the forward pass and is deferred to Epic 04.
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
    /// # Variant behavior
    ///
    /// - **Level1**: deactivates cuts with `active_count <= threshold`.
    /// - **Lml1**: deactivates cuts with
    ///   `current_iteration - last_active_iter > memory_window`.
    /// - **Dominated**: stub — always returns an empty set (Epic 04).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::cut_selection::{CutMetadata, CutSelectionStrategy};
    ///
    /// let strategy = CutSelectionStrategy::Level1 { threshold: 0, check_frequency: 5 };
    /// let metadata = vec![
    ///     CutMetadata { iteration_generated: 1, forward_pass_index: 0,
    ///                   active_count: 0, last_active_iter: 1, domination_count: 0 },
    ///     CutMetadata { iteration_generated: 1, forward_pass_index: 1,
    ///                   active_count: 2, last_active_iter: 5, domination_count: 0 },
    /// ];
    /// let deact = strategy.select(&metadata, 10);
    /// assert_eq!(deact.indices, vec![0]);
    /// ```
    #[must_use]
    pub fn select(&self, metadata: &[CutMetadata], current_iteration: u64) -> DeactivationSet {
        self.select_for_stage(metadata, current_iteration, 0)
    }

    /// Scan the cut pool metadata for a specific stage and identify cuts to
    /// deactivate.
    ///
    /// Identical to [`select`] but also sets [`DeactivationSet::stage_index`].
    ///
    /// [`select`]: CutSelectionStrategy::select
    #[must_use]
    pub fn select_for_stage(
        &self,
        metadata: &[CutMetadata],
        current_iteration: u64,
        stage_index: u32,
    ) -> DeactivationSet {
        // Cut pool capacity is bounded by a u32 field in the pool header, so
        // enumerate indices always fit in u32. The cast is safe by structural
        // invariant established at pool construction time.
        #[allow(clippy::cast_possible_truncation)]
        let indices = match self {
            Self::Level1 { threshold, .. } => metadata
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    m.active_count <= *threshold && m.iteration_generated < current_iteration
                })
                .map(|(i, _)| i as u32)
                .collect(),

            Self::Lml1 { memory_window, .. } => metadata
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    m.iteration_generated < current_iteration
                        && current_iteration.saturating_sub(m.last_active_iter) > *memory_window
                })
                .map(|(i, _)| i as u32)
                .collect(),

            // Stub: Dominated selection requires visited forward pass states.
            // Full implementation deferred to Epic 04.
            Self::Dominated { .. } => vec![],
        };

        DeactivationSet {
            stage_index,
            indices,
        }
    }

    /// Update tracking metadata for a cut that was binding at an LP solution.
    ///
    /// Called during the backward pass for every cut whose dual multiplier
    /// exceeds the solver tolerance (`is_binding == true`). When
    /// `is_binding == false`, the metadata is not modified.
    ///
    /// The update performed depends on the active strategy:
    ///
    /// | Strategy  | Update                                          |
    /// |-----------|--------------------------------------------------|
    /// | Level1    | Increments `metadata.active_count` by 1         |
    /// | Lml1      | Sets `metadata.last_active_iter = current_iteration` |
    /// | Dominated | Resets `metadata.domination_count` to 0          |
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::cut_selection::{CutMetadata, CutSelectionStrategy};
    ///
    /// let strategy = CutSelectionStrategy::Level1 { threshold: 0, check_frequency: 5 };
    /// let mut meta = CutMetadata {
    ///     iteration_generated: 1, forward_pass_index: 0,
    ///     active_count: 0, last_active_iter: 1, domination_count: 0,
    /// };
    /// strategy.update_activity(&mut meta, true, 5);
    /// assert_eq!(meta.active_count, 1);
    /// ```
    pub fn update_activity(
        &self,
        metadata: &mut CutMetadata,
        is_binding: bool,
        current_iteration: u64,
    ) {
        if !is_binding {
            return;
        }

        match self {
            Self::Level1 { .. } => {
                metadata.active_count += 1;
            }
            Self::Lml1 { .. } => {
                metadata.last_active_iter = current_iteration;
            }
            Self::Dominated { .. } => {
                metadata.domination_count = 0;
            }
        }
    }
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
        "lml1" => Ok(Some(CutSelectionStrategy::Lml1 {
            memory_window: u64::from(threshold),
            check_frequency: u64::from(check_frequency),
        })),
        "domination" => Ok(Some(CutSelectionStrategy::Dominated {
            threshold: f64::from(threshold),
            check_frequency: u64::from(check_frequency),
        })),
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
    use cobre_io::config::CutSelectionConfig;

    fn make_meta(active_count: u64, last_active_iter: u64, domination_count: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: 1,
            forward_pass_index: 0,
            active_count,
            last_active_iter,
            domination_count,
        }
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
        let metadata = vec![make_meta(0, 1, 0), make_meta(1, 5, 0)];
        let deact = strategy.select(&metadata, 10);
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
        let metadata = vec![make_meta(3, 1, 0), make_meta(7, 5, 0)];
        let deact = strategy.select(&metadata, 10);
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
        let metadata = vec![make_meta(0, 1, 0), make_meta(1, 5, 0), make_meta(2, 8, 0)];
        let deact = strategy.select(&metadata, 10);
        assert_eq!(deact.indices, vec![0, 1]);
    }

    #[test]
    fn level1_empty_metadata_returns_empty_set() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let deact = strategy.select(&[], 10);
        assert!(deact.indices.is_empty());
    }

    #[test]
    fn lml1_deactivates_cuts_outside_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let metadata = vec![make_meta(0, 5, 0)]; // last_active_iter = 5
        let deact = strategy.select(&metadata, 20);
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
        let metadata = vec![make_meta(0, 12, 0)];
        let deact = strategy.select(&metadata, 20);
        assert!(deact.indices.is_empty());
    }

    #[test]
    fn lml1_retains_cuts_exactly_at_boundary() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let metadata = vec![make_meta(0, 10, 0)];
        let deact = strategy.select(&metadata, 20);
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
        let metadata = vec![make_meta(0, 5, 0), make_meta(0, 12, 0), make_meta(0, 1, 0)];
        let deact = strategy.select(&metadata, 20);
        assert_eq!(deact.indices, vec![0, 2]);
    }

    // Dominated select (stub): always returns empty set

    #[test]
    fn dominated_select_always_returns_empty_set() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.001,
            check_frequency: 10,
        };
        let metadata = vec![make_meta(0, 1, 5), make_meta(0, 1, 10)];
        let deact = strategy.select(&metadata, 20);
        assert!(deact.indices.is_empty());
    }

    #[test]
    fn level1_update_activity_increments_active_count_when_binding() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let mut meta = make_meta(0, 1, 0);
        strategy.update_activity(&mut meta, true, 5);
        assert_eq!(meta.active_count, 1);
        strategy.update_activity(&mut meta, true, 6);
        assert_eq!(meta.active_count, 2);
    }

    #[test]
    fn level1_update_activity_does_nothing_when_not_binding() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let mut meta = make_meta(3, 1, 0);
        strategy.update_activity(&mut meta, false, 5);
        assert_eq!(meta.active_count, 3, "must not modify when not binding");
    }

    #[test]
    fn lml1_update_activity_sets_last_active_iter_when_binding() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let mut meta = make_meta(0, 1, 0);
        strategy.update_activity(&mut meta, true, 15);
        assert_eq!(meta.last_active_iter, 15);
    }

    #[test]
    fn lml1_update_activity_does_nothing_when_not_binding() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let mut meta = make_meta(0, 7, 0);
        strategy.update_activity(&mut meta, false, 15);
        assert_eq!(meta.last_active_iter, 7, "must not modify when not binding");
    }

    #[test]
    fn dominated_update_activity_resets_domination_count_when_binding() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.001,
            check_frequency: 10,
        };
        let mut meta = make_meta(0, 1, 42);
        strategy.update_activity(&mut meta, true, 10);
        assert_eq!(
            meta.domination_count, 0,
            "domination_count must be reset when cut is binding"
        );
    }

    #[test]
    fn dominated_update_activity_does_nothing_when_not_binding() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.001,
            check_frequency: 10,
        };
        let mut meta = make_meta(0, 1, 42);
        strategy.update_activity(&mut meta, false, 10);
        assert_eq!(
            meta.domination_count, 42,
            "must not modify when not binding"
        );
    }

    #[test]
    fn ac_level1_threshold_0_deactivates_zero_activity_cut() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let metadata = vec![CutMetadata {
            iteration_generated: 1,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: 1,
            domination_count: 0,
        }];
        let deact = strategy.select(&metadata, 10);
        assert!(deact.indices.contains(&0));
    }

    #[test]
    fn ac_lml1_deactivates_cut_outside_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            memory_window: 10,
            check_frequency: 5,
        };
        let metadata = vec![CutMetadata {
            iteration_generated: 1,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: 5,
            domination_count: 0,
        }];
        let deact = strategy.select(&metadata, 20);
        assert!(deact.indices.contains(&0));
    }

    #[test]
    fn select_for_stage_sets_stage_index() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let metadata = vec![make_meta(0, 1, 0)];
        let deact = strategy.select_for_stage(&metadata, 10, 7);
        assert_eq!(deact.stage_index, 7);
    }

    #[test]
    fn select_sets_stage_index_to_zero() {
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 5,
        };
        let metadata: Vec<CutMetadata> = vec![];
        let deact = strategy.select(&metadata, 10);
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
        let meta = make_meta(5, 10, 2);
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
    fn redeactivation_of_inactive_cut_is_noop() {
        use crate::cut::pool::CutPool;

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

        // Level1 selection: slot 0 still has active_count == 0 in metadata,
        // so it appears in the deactivation set again. Slots 1 and 2 have
        // active_count > 0 and are retained.
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 1,
        };
        let deact = strategy.select_for_stage(&pool.metadata[..pool.populated_count], 5, 0);
        // Slot 0 is in the deactivation set (active_count == 0).
        assert!(
            deact.indices.contains(&0),
            "slot 0 should be selected for deactivation"
        );
        assert_eq!(
            deact.indices.len(),
            1,
            "only slot 0 should be selected (slots 1,2 have activity)"
        );

        // Re-apply deactivation — must not double-decrement.
        pool.deactivate(&deact.indices);
        assert_eq!(
            pool.active_count(),
            2,
            "active_count must not change when re-deactivating an already-inactive cut"
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
        let metadata = vec![
            make_meta(0, 1, 0),  // last_active_iter = 1
            make_meta(0, 5, 0),  // last_active_iter = 5
            make_meta(0, 7, 0),  // last_active_iter = 7 (boundary)
            make_meta(0, 8, 0),  // last_active_iter = 8
            make_meta(0, 10, 0), // last_active_iter = 10
        ];

        let deact = strategy.select_for_stage(&metadata, 10, 0);

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
        let metadata = vec![
            CutMetadata {
                iteration_generated: 10, // same as current_iteration
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 10,
                domination_count: 0,
            },
            CutMetadata {
                iteration_generated: 5, // older, zero activity
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 5,
                domination_count: 0,
            },
        ];
        let deact = strategy.select(&metadata, 10);
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
        let metadata = vec![CutMetadata {
            iteration_generated: 10,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: 10,
            domination_count: 0,
        }];
        let deact = strategy.select(&metadata, 10);
        assert!(
            deact.indices.is_empty(),
            "current-iteration cut must not be deactivated even with memory_window=0"
        );
    }
}
