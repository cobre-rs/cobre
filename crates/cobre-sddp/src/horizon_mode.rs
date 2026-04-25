//! Horizon mode abstraction for SDDP stage traversal.
//!
//! [`HorizonMode`] is a flat enum that controls how the training loop traverses
//! stages and determines terminal conditions. Only `Finite` horizon is
//! implemented; `Cyclic` horizon is deferred to a future release.
//!
//! Discount factors are computed from the [`PolicyGraph`](cobre_core::PolicyGraph)
//! at setup time and stored in [`StageTemplates`](crate::StageTemplates).
//!
//! ## Stage indexing convention
//!
//! Stages use **1-based** indexing throughout this module, matching the
//! horizon mode spec convention. Stage 1 is the first stage; stage `T` is
//! the last stage for `HorizonMode::Finite { num_stages: T }`.
//!
//! ## Finite horizon
//!
//! The finite horizon forms a linear chain `1 → 2 → ··· → T`. Stage `T` is
//! the terminal stage and has no successors. The terminal value is implicitly
//! zero: `V_{T+1} = 0`. Each stage has a unique cut pool.
//!
//! ## Examples
//!
//! ```rust
//! use cobre_sddp::horizon_mode::HorizonMode;
//!
//! let horizon = HorizonMode::Finite { num_stages: 5 };
//! assert_eq!(horizon.successors(3), vec![4]);
//! assert!(horizon.is_terminal(5));
//! assert!(!horizon.is_terminal(4));
//! assert!(horizon.validate().is_ok());
//! assert_eq!(horizon.num_stages(), 5);
//! ```

use crate::SddpError;

/// Horizon mode controlling stage traversal and terminal conditions.
///
/// A single `HorizonMode` value governs the structural topology of the entire
/// training run. The enum is matched at each forward and backward pass stage to
/// select the correct traversal behaviour (enum dispatch for closed variant sets, avoiding `Box<dyn>`).
///
/// ## Variants
///
/// - [`HorizonMode::Finite`]: linear chain `1 → 2 → ··· → T` with terminal
///   value `V_{T+1} = 0`. The only variant currently implemented.
///
/// ## Stage indexing
///
/// All methods use **1-based** stage indices. Stage 1 is the first; stage
/// `num_stages` is the terminal stage for finite horizon.
///
/// ## Examples
///
/// ```rust
/// use cobre_sddp::horizon_mode::HorizonMode;
///
/// let h = HorizonMode::Finite { num_stages: 12 };
/// assert_eq!(h.successors(1), vec![2]);
/// assert!(h.successors(12).is_empty());
/// assert!(h.is_terminal(12));
/// assert!(h.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub enum HorizonMode {
    /// Finite (acyclic) horizon with linear chain topology and zero terminal value.
    Finite {
        /// Total number of stages `T` in the finite chain.
        ///
        /// Must be at least 2 (a single-stage problem has no predecessor to
        /// generate cuts for, making SDDP degenerate). Validated by
        /// [`HorizonMode::validate`].
        num_stages: usize,
    },
}

impl HorizonMode {
    /// Return successor stage indices reachable from `stage`.
    ///
    /// For `Finite` horizon:
    /// - Non-terminal stages return `vec![stage + 1]`.
    /// - The terminal stage (`stage == num_stages`) returns an empty `Vec`.
    ///
    /// Stage indices are **1-based**. The returned vec is always sorted
    /// (deterministic order across MPI ranks).
    ///
    /// This method is infallible — all stage IDs are validated at
    /// configuration load time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::horizon_mode::HorizonMode;
    ///
    /// let h = HorizonMode::Finite { num_stages: 5 };
    /// assert_eq!(h.successors(3), vec![4]);
    /// assert!(h.successors(5).is_empty());
    /// ```
    #[must_use]
    pub fn successors(&self, stage: usize) -> Vec<usize> {
        match self {
            HorizonMode::Finite { num_stages } => {
                if stage >= *num_stages {
                    vec![]
                } else {
                    vec![stage + 1]
                }
            }
        }
    }

    /// Return whether `stage` has no successors.
    ///
    /// For `Finite` horizon, exactly the last stage (`stage == num_stages`)
    /// is terminal.
    ///
    /// Consistent with [`HorizonMode::successors`]: `is_terminal(s)` is
    /// `true` if and only if `successors(s).is_empty()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::horizon_mode::HorizonMode;
    ///
    /// let h = HorizonMode::Finite { num_stages: 5 };
    /// assert!(h.is_terminal(5));
    /// assert!(!h.is_terminal(4));
    /// assert!(!h.is_terminal(1));
    /// ```
    #[must_use]
    pub fn is_terminal(&self, stage: usize) -> bool {
        match self {
            HorizonMode::Finite { num_stages } => stage >= *num_stages,
        }
    }

    /// Validate the horizon mode configuration.
    ///
    /// Called once during initialization. Returns `Ok(())` when the
    /// configuration is valid; returns `Err(SddpError::Validation(_))`
    /// describing the violated rule when invalid.
    ///
    /// ## Validation rules
    ///
    /// | Rule | Condition | Error |
    /// |------|-----------|-------|
    /// | H1   | `num_stages >= 2` | A single-stage finite problem is degenerate |
    ///
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] when `num_stages < 2`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::horizon_mode::HorizonMode;
    ///
    /// assert!(HorizonMode::Finite { num_stages: 5 }.validate().is_ok());
    /// assert!(HorizonMode::Finite { num_stages: 1 }.validate().is_err());
    /// assert!(HorizonMode::Finite { num_stages: 0 }.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), SddpError> {
        match self {
            HorizonMode::Finite { num_stages } => {
                if *num_stages < 2 {
                    return Err(SddpError::Validation(format!(
                        "HorizonMode::Finite requires at least 2 stages, got {num_stages}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// Return the total number of stages in the horizon.
    ///
    /// For `Finite` horizon, this is the `num_stages` field.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::horizon_mode::HorizonMode;
    ///
    /// let h = HorizonMode::Finite { num_stages: 12 };
    /// assert_eq!(h.num_stages(), 12);
    /// ```
    #[must_use]
    pub fn num_stages(&self) -> usize {
        match self {
            HorizonMode::Finite { num_stages } => *num_stages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HorizonMode;
    use crate::SddpError;

    // ── successors ────────────────────────────────────────────────────────────

    #[test]
    fn successors_mid_stage_returns_next() {
        // Acceptance criterion: successors(3) returns [4] for num_stages=5
        let h = HorizonMode::Finite { num_stages: 5 };
        assert_eq!(h.successors(3), vec![4]);
    }

    #[test]
    fn successors_first_stage_returns_second() {
        let h = HorizonMode::Finite { num_stages: 5 };
        assert_eq!(h.successors(1), vec![2]);
    }

    #[test]
    fn successors_terminal_stage_returns_empty() {
        let h = HorizonMode::Finite { num_stages: 5 };
        assert_eq!(h.successors(5), Vec::<usize>::new());
    }

    #[test]
    fn successors_beyond_terminal_returns_empty() {
        // Stages beyond num_stages are also treated as terminal.
        let h = HorizonMode::Finite { num_stages: 5 };
        assert_eq!(h.successors(6), Vec::<usize>::new());
    }

    #[test]
    fn successors_all_non_terminal_stages() {
        let h = HorizonMode::Finite { num_stages: 4 };
        for stage in 1..=3 {
            let succ = h.successors(stage);
            assert_eq!(succ, vec![stage + 1], "stage {stage}");
        }
    }

    #[test]
    fn successors_consistent_with_is_terminal() {
        // Postcondition: is_terminal(s) iff successors(s).is_empty()
        let h = HorizonMode::Finite { num_stages: 6 };
        for stage in 1..=7 {
            assert_eq!(
                h.is_terminal(stage),
                h.successors(stage).is_empty(),
                "stage {stage}: is_terminal and successors must be consistent"
            );
        }
    }

    // ── is_terminal ───────────────────────────────────────────────────────────

    #[test]
    fn is_terminal_last_stage_is_true() {
        // Acceptance criterion: is_terminal(5) returns true for num_stages=5
        let h = HorizonMode::Finite { num_stages: 5 };
        assert!(h.is_terminal(5));
    }

    #[test]
    fn is_terminal_preceding_stage_is_false() {
        let h = HorizonMode::Finite { num_stages: 5 };
        assert!(!h.is_terminal(4));
    }

    #[test]
    fn is_terminal_first_stage_is_false() {
        let h = HorizonMode::Finite { num_stages: 5 };
        assert!(!h.is_terminal(1));
    }

    #[test]
    fn is_terminal_single_stage_is_terminal() {
        // Even for num_stages=1 (invalid config), stage 1 should be terminal.
        let h = HorizonMode::Finite { num_stages: 1 };
        assert!(h.is_terminal(1));
    }

    // ── validate ─────────────────────────────────────────────────────────────

    #[test]
    fn validate_accepts_two_or_more_stages() {
        for n in [2, 3, 10, 60, 120] {
            let h = HorizonMode::Finite { num_stages: n };
            assert!(h.validate().is_ok(), "num_stages={n} should be valid");
        }
    }

    #[test]
    fn validate_rejects_one_stage() {
        // Acceptance criterion: num_stages=1 returns Err(SddpError::Validation(_))
        let h = HorizonMode::Finite { num_stages: 1 };
        let result = h.validate();
        assert!(
            matches!(result, Err(SddpError::Validation(_))),
            "expected Err(Validation), got {result:?}"
        );
    }

    #[test]
    fn validate_rejects_zero_stages() {
        let h = HorizonMode::Finite { num_stages: 0 };
        let result = h.validate();
        assert!(
            matches!(result, Err(SddpError::Validation(_))),
            "expected Err(Validation), got {result:?}"
        );
    }

    #[test]
    fn validate_error_message_contains_stage_count() {
        let h = HorizonMode::Finite { num_stages: 1 };
        let err = h.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains('1'),
            "error message should contain the invalid stage count: {msg}"
        );
    }

    // ── num_stages ────────────────────────────────────────────────────────────

    #[test]
    fn num_stages_returns_field_value() {
        let h = HorizonMode::Finite { num_stages: 12 };
        assert_eq!(h.num_stages(), 12);
    }

    #[test]
    fn num_stages_single() {
        let h = HorizonMode::Finite { num_stages: 1 };
        assert_eq!(h.num_stages(), 1);
    }

    // ── Derive traits ─────────────────────────────────────────────────────────

    #[test]
    fn debug_output_contains_variant_name() {
        let h = HorizonMode::Finite { num_stages: 5 };
        let debug_str = format!("{h:?}");
        assert!(debug_str.contains("Finite"));
        assert!(debug_str.contains("num_stages"));
    }

    #[test]
    fn clone_produces_equal_num_stages() {
        let h = HorizonMode::Finite { num_stages: 8 };
        let cloned = h.clone();
        assert_eq!(cloned.num_stages(), h.num_stages());
    }
}
