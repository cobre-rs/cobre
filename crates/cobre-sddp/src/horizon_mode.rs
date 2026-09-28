//! Horizon mode abstraction for SDDP stage traversal.
//!
//! [`HorizonMode`] controls how the training loop traverses stages and
//! determines terminal conditions. Discount factors are computed from the
//! [`HorizonGraph`](cobre_core::HorizonGraph) at setup time and stored in
//! `StageTemplates`.

use crate::SddpError;

/// Horizon mode controlling stage traversal and terminal conditions.
///
/// A single value governs the topology of the entire training run, matched at
/// each forward/backward stage.
///
/// All methods use **1-based** stage indices: stage 1 is the first; stage
/// `num_stages` is the terminal stage. Only [`HorizonMode::Finite`] is
/// currently implemented.
///
/// ## Examples
///
/// ```rust
/// use cobre_sddp::horizon_mode::HorizonMode;
///
/// let h = HorizonMode::Finite { num_stages: 12 };
/// assert!(h.is_terminal(12));
/// assert!(!h.is_terminal(11));
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
    /// Return whether `stage` has no successors (for `Finite`, the last stage).
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
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] when `num_stages < 2` (a single-stage
    /// finite problem is degenerate).
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
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::horizon_mode::HorizonMode;
    ///
    /// let h = HorizonMode::Finite { num_stages: 12 };
    /// assert_eq!(h.num_stages(), 12);
    /// ```
    #[must_use]
    #[inline]
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

    // ── is_terminal ───────────────────────────────────────────────────────────

    #[test]
    fn is_terminal_last_stage_is_true() {
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
