//! Post-training simulation configuration types for `config.json → simulation`.

use serde::{Deserialize, Serialize};

use super::scenario_source::RawScenarioSourceConfig;
use super::training::PhaseSolverProfileConfig;

/// Default scenario count when `simulation.selection` is absent. Sole owner of
/// the value; the count resolver reads it.
pub(crate) const DEFAULT_NUM_SCENARIOS: u32 = 2000;

/// Post-training simulation settings (`config.json → simulation`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SimulationConfig {
    /// Enable post-training simulation.
    pub enabled: bool,

    /// Bounded channel capacity between simulation threads and the I/O writer thread.
    pub io_channel_capacity: u32,

    /// Scenario source configuration for the post-training simulation forward pass.
    /// When absent, falls back to the training scenario source.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,

    /// Simulation solver profile. Absent leaves the phase's built-in
    /// tuned profile.
    #[serde(default)]
    pub solver: Option<PhaseSolverProfileConfig>,

    /// Phase-level scenario selection. Absent resolves to the default sampled
    /// count.
    #[serde(default)]
    pub selection: Option<SimulationSelection>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            io_channel_capacity: 64,
            scenario_source: None,
            solver: None,
            selection: None,
        }
    }
}

/// Post-training scenario selection and its method-specific parameters
/// (`config.json → simulation.selection`).
///
/// The `method` key selects how scenarios are chosen: `sampled` draws
/// `num_scenarios` trajectories; `enumerated` walks the scenario set
/// exhaustively. Each method accepts only its own parameters, so pairing a
/// count with `enumerated` is a parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SimulationSelection {
    /// Sampled scenarios: draw `num_scenarios` trajectories.
    Sampled {
        /// Number of simulation trajectories to draw.
        num_scenarios: u32,
    },
    /// Exhaustive enumeration of the scenario set.
    // A braced variant, not a unit one: serde enforces `deny_unknown_fields`
    // only for braced variants of an internally tagged enum, and this variant
    // must reject a stray `num_scenarios`.
    Enumerated {},
}

/// Effective simulation scenario-count resolution
/// ([`Config::resolve_num_scenarios`](super::Config::resolve_num_scenarios)):
/// either a concrete sampled count or a signal that the count is derived from
/// the policy graph downstream, since config load holds no graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumScenariosResolution {
    /// `num_scenarios` sampled simulation trajectories.
    Sampled(u32),
    /// Exhaustive enumeration; the count is derived from the policy graph.
    Enumerated,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{SimulationConfig, SimulationSelection};
    use crate::config::training::{PriceStrategy, ScaleStrategy};

    /// A `sampled` phase selection round-trips into the `Sampled` variant.
    #[test]
    fn sampled_selection_round_trips() {
        let json = r#"{"enabled": true, "selection": {"method": "sampled", "num_scenarios": 500}}"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.selection,
            Some(SimulationSelection::Sampled { num_scenarios: 500 })
        );
    }

    /// The removed flat `num_scenarios` alias is an unknown-field deserialize
    /// error under `deny_unknown_fields`; the count lives solely in the
    /// `selection.sampled` arm.
    #[test]
    fn flat_num_scenarios_alias_is_deserialize_error() {
        let alias = r#"{"enabled": true, "num_scenarios": 500}"#;
        assert!(
            serde_json::from_str::<SimulationConfig>(alias).is_err(),
            "flat num_scenarios must be rejected as an unknown field"
        );

        let arm = r#"{"enabled": true, "selection": {"method": "sampled", "num_scenarios": 500}}"#;
        let cfg: SimulationConfig = serde_json::from_str(arm).unwrap();
        assert_eq!(
            cfg.selection,
            Some(SimulationSelection::Sampled { num_scenarios: 500 })
        );
    }

    /// A count under `enumerated` is unrepresentable — `deny_unknown_fields` on
    /// the braced variant rejects it at parse time.
    #[test]
    fn enumerated_selection_with_count_is_deserialize_error() {
        let json =
            r#"{"enabled": true, "selection": {"method": "enumerated", "num_scenarios": 500}}"#;
        let result = serde_json::from_str::<SimulationConfig>(json);
        assert!(
            result.is_err(),
            "a count under enumerated must be rejected as unrepresentable"
        );
    }

    #[test]
    fn simulation_solver_profile_block_round_trips() {
        let json = r#"{
            "enabled": true,
            "selection": { "method": "sampled", "num_scenarios": 500 },
            "solver": {
                "scale": "off",
                "price": "row"
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.scale, Some(ScaleStrategy::Off));
        assert_eq!(solver.price, Some(PriceStrategy::Row));
    }

    #[test]
    fn simulation_solver_profile_absent_is_none() {
        assert!(SimulationConfig::default().solver.is_none());
    }

    #[test]
    fn simulation_solver_profile_steepest_edge_fallback_threshold_round_trips() {
        let json = r#"{
            "enabled": true,
            "selection": { "method": "sampled", "num_scenarios": 500 },
            "solver": {
                "steepest_edge_devex_fallback_threshold": 12.5
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.steepest_edge_devex_fallback_threshold, Some(12.5));
    }
}
