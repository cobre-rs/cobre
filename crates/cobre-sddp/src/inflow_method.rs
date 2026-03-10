//! Inflow non-negativity treatment method for SDDP subproblems.
//!
//! [`InflowNonNegativityMethod`] is a flat enum representing the strategies
//! for handling negative PAR(p) inflow realisations in the LP subproblems.
//! It is dispatched via `match` when constructing LP templates and extracting
//! simulation results. Enum dispatch is used because the variant set is closed
//! (see `docs/adr/002-enum-dispatch.md`).
//!
//! ## Variants
//!
//! | Variant            | Slack columns |
//! | ------------------ | ------------- |
//! | `None`             | No            |
//! | `Penalty { cost }` | Yes           |
//!
//! The `None` variant leaves the LP unchanged: negative inflow can cause
//! infeasibility when scenario noise is sufficiently negative.
//!
//! The `Penalty` variant appends `N` slack columns (`sigma_inf_h >= 0`) to the
//! LP, one per hydro plant.  Each slack enters the water balance row for
//! hydro `h` with a positive coefficient, acting as virtual inflow that
//! prevents infeasibility.  The objective coefficient is
//! `penalty_cost * total_stage_hours`.
//!
//! ## Deferred: truncation methods
//!
//! The truncation and truncation-with-penalty methods described in the
//! literature (see `docs/deferred-truncation-design.md`) require external
//! AR model evaluation before LP patching — the full inflow value must be
//! computed outside the LP to determine whether truncation is needed.
//! This is deferred to a post-v0.1.0 release.
//!
//! ## Conversion
//!
//! Convert from the cobre-io config type at the CLI boundary using the
//! provided `From<&cobre_io::config::InflowNonNegativityConfig>` implementation.
//! This avoids passing raw strings through the algorithm code.
//!
//! # Examples
//!
//! ```rust
//! use cobre_sddp::InflowNonNegativityMethod;
//!
//! let method = InflowNonNegativityMethod::Penalty { cost: 500.0 };
//! assert!(method.has_slack_columns());
//! assert_eq!(method.penalty_cost(), Some(500.0));
//!
//! let none = InflowNonNegativityMethod::None;
//! assert!(!none.has_slack_columns());
//! assert_eq!(none.penalty_cost(), None);
//! ```

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
        matches!(self, InflowNonNegativityMethod::Penalty { .. })
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
            InflowNonNegativityMethod::Penalty { cost } => Some(*cost),
            InflowNonNegativityMethod::None => None,
        }
    }
}

impl From<&cobre_io::config::InflowNonNegativityConfig> for InflowNonNegativityMethod {
    /// Convert from the cobre-io config type.
    ///
    /// Recognised method strings are `"penalty"` and `"none"`.  Any other value
    /// is treated as `None`.  Method string validation is the responsibility of
    /// the cobre-io loading pipeline (five-layer validation), so unrecognised
    /// values indicate a programming error that should have been caught upstream.
    fn from(cfg: &cobre_io::config::InflowNonNegativityConfig) -> Self {
        match cfg.method.as_str() {
            "penalty" => InflowNonNegativityMethod::Penalty {
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
}
