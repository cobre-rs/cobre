//! Inflow non-negativity treatment method for SDDP subproblems.
//!
//! [`InflowNonNegativityMethod`] is a flat enum representing the strategies
//! for handling negative PAR(p) inflow realisations in the LP subproblems.
//! It is dispatched via `match` when constructing LP templates and extracting
//! simulation results. Enum dispatch is used because the variant set is closed
//! (enum dispatch for closed variant sets).

/// Inflow non-negativity treatment method.
///
/// Determines whether slack columns are added to the LP and what objective
/// coefficient they carry.  The variant must be the same across all stages
/// (set once at solver initialisation from the loaded case config).
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::InflowNonNegativityMethod;
///
/// let penalty = InflowNonNegativityMethod::Penalty { cost: 1000.0 };
/// assert!(penalty.has_slack_columns());
///
/// let none = InflowNonNegativityMethod::None;
/// assert!(!none.has_slack_columns());
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InflowNonNegativityMethod {
    /// No inflow non-negativity enforcement.
    ///
    /// The LP does not include slack columns.  Negative inflow noise may cause
    /// LP infeasibility if the PAR(p) noise realisation is sufficiently negative.
    None,

    /// Truncation-based enforcement: clamps negative PAR(p) inflows to zero
    /// by modifying the noise vector before LP patching.
    ///
    /// Does not add slack columns to the LP.  Instead, the PAR(p) model is
    /// evaluated outside the LP to obtain the full inflow value; if the result
    /// is negative, the noise component is adjusted so that the inflow is zero.
    /// This prevents LP infeasibility without perturbing the objective function.
    Truncation,

    /// Penalty-based enforcement with objective cost `penalty_cost` per m³/s
    /// per stage-hour.
    ///
    /// Appends `N` slack columns (`sigma_inf_h >= 0`) to the LP.  Each slack
    /// enters the water balance row for hydro `h` with coefficient
    /// `tau_total * M3S_TO_HM3`, where `tau_total` is the total stage duration
    /// in hours.  The objective coefficient is `penalty_cost * tau_total`.
    Penalty {
        /// Penalty coefficient `c^{inf}` applied to each slack unit.
        cost: f64,
    },

    /// Combined truncation and penalty enforcement.
    ///
    /// The PAR(p) noise is clamped (identical to [`Truncation`]) so that the
    /// mean + noise inflow is never negative. In addition, penalty slack
    /// columns are added (identical to [`Penalty`]) so the solver can "undo"
    /// part of the clamping if cost-effective. This matches `SPTcpp`'s
    /// `truncamento_penalizacao` mode.
    TruncationWithPenalty {
        /// Legacy global cost (deprecated in favor of penalty cascade).
        cost: f64,
    },
}

impl InflowNonNegativityMethod {
    /// Returns `true` when slack columns are appended to the LP.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::InflowNonNegativityMethod;
    ///
    /// assert!(!InflowNonNegativityMethod::None.has_slack_columns());
    /// assert!(InflowNonNegativityMethod::Penalty { cost: 100.0 }.has_slack_columns());
    /// ```
    #[must_use]
    pub fn has_slack_columns(&self) -> bool {
        matches!(
            self,
            InflowNonNegativityMethod::Penalty { .. }
                | InflowNonNegativityMethod::TruncationWithPenalty { .. }
        )
    }

    /// Returns the penalty cost when slack columns are active, or `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::InflowNonNegativityMethod;
    ///
    /// assert_eq!(InflowNonNegativityMethod::Penalty { cost: 500.0 }.penalty_cost(), Some(500.0));
    /// assert_eq!(InflowNonNegativityMethod::None.penalty_cost(), None);
    /// ```
    #[must_use]
    pub fn penalty_cost(&self) -> Option<f64> {
        match self {
            InflowNonNegativityMethod::Penalty { cost }
            | InflowNonNegativityMethod::TruncationWithPenalty { cost } => Some(*cost),
            InflowNonNegativityMethod::Truncation | InflowNonNegativityMethod::None => None,
        }
    }
}

impl From<&cobre_io::config::InflowNonNegativityConfig> for InflowNonNegativityMethod {
    /// Convert from the cobre-io config type.
    ///
    /// Recognised method strings are `"none"`, `"truncation"`, `"penalty"`,
    /// and `"truncation_with_penalty"`.
    /// Any other value is treated as `None`.  Method string validation is the
    /// responsibility of the cobre-io loading pipeline (five-layer validation),
    /// so unrecognised values indicate a programming error that should have been
    /// caught upstream.
    fn from(cfg: &cobre_io::config::InflowNonNegativityConfig) -> Self {
        match cfg.method.as_str() {
            "truncation" => InflowNonNegativityMethod::Truncation,
            "penalty" => InflowNonNegativityMethod::Penalty {
                cost: cfg.penalty_cost,
            },
            "truncation_with_penalty" => InflowNonNegativityMethod::TruncationWithPenalty {
                cost: cfg.penalty_cost,
            },
            _ => InflowNonNegativityMethod::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InflowNonNegativityMethod;
    use cobre_io::config::InflowNonNegativityConfig;

    // ── has_slack_columns ────────────────────────────────────────────────────

    #[test]
    fn none_has_no_slack_columns() {
        assert!(!InflowNonNegativityMethod::None.has_slack_columns());
    }

    #[test]
    fn truncation_has_no_slack_columns() {
        assert!(!InflowNonNegativityMethod::Truncation.has_slack_columns());
    }

    #[test]
    fn penalty_has_slack_columns() {
        assert!(InflowNonNegativityMethod::Penalty { cost: 100.0 }.has_slack_columns());
    }

    // ── penalty_cost ─────────────────────────────────────────────────────────

    #[test]
    fn penalty_cost_for_penalty_variant() {
        assert_eq!(
            InflowNonNegativityMethod::Penalty { cost: 500.0 }.penalty_cost(),
            Some(500.0)
        );
    }

    #[test]
    fn penalty_cost_none_for_none_variant() {
        assert_eq!(InflowNonNegativityMethod::None.penalty_cost(), None);
    }

    #[test]
    fn truncation_penalty_cost_is_none() {
        assert_eq!(InflowNonNegativityMethod::Truncation.penalty_cost(), None);
    }

    // ── conversion from config ───────────────────────────────────────────────

    #[test]
    fn test_inflow_method_conversion_none() {
        let cfg = InflowNonNegativityConfig {
            method: "none".to_string(),
            penalty_cost: 0.0,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::None
        );
    }

    #[test]
    fn test_inflow_method_conversion_penalty() {
        let cfg = InflowNonNegativityConfig {
            method: "penalty".to_string(),
            penalty_cost: 500.0,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::Penalty { cost: 500.0 }
        );
    }

    #[test]
    fn test_inflow_method_conversion_truncation() {
        let cfg = InflowNonNegativityConfig {
            method: "truncation".to_string(),
            penalty_cost: 0.0,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::Truncation
        );
    }

    #[test]
    fn test_truncation_ignores_penalty_cost() {
        let cfg = InflowNonNegativityConfig {
            method: "truncation".to_string(),
            penalty_cost: 999.0,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::Truncation
        );
    }

    #[test]
    fn test_inflow_method_conversion_unknown_falls_back_to_none() {
        let cfg = InflowNonNegativityConfig {
            method: "unknown_method".to_string(),
            penalty_cost: 100.0,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::None
        );
    }

    #[test]
    fn test_penalty_config_propagation() {
        let costs = [0.0, 100.0, 500.0, 1000.0, f64::MAX];
        for &expected_cost in &costs {
            let cfg = InflowNonNegativityConfig {
                method: "penalty".to_string(),
                penalty_cost: expected_cost,
            };
            let method = InflowNonNegativityMethod::from(&cfg);
            assert_eq!(method.penalty_cost(), Some(expected_cost));
        }
    }

    // ── TruncationWithPenalty ───────────────────────────────────────────────

    #[test]
    fn truncation_with_penalty_has_slack_columns() {
        assert!(
            InflowNonNegativityMethod::TruncationWithPenalty { cost: 100.0 }.has_slack_columns()
        );
    }

    #[test]
    fn truncation_with_penalty_cost() {
        assert_eq!(
            InflowNonNegativityMethod::TruncationWithPenalty { cost: 500.0 }.penalty_cost(),
            Some(500.0)
        );
    }

    #[test]
    fn test_inflow_method_conversion_truncation_with_penalty() {
        let cfg = InflowNonNegativityConfig {
            method: "truncation_with_penalty".to_string(),
            penalty_cost: 750.0,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::TruncationWithPenalty { cost: 750.0 }
        );
    }
}
