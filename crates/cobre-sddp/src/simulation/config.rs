//! Simulation configuration type for the SDDP policy evaluation phase.
//!
//! [`SimulationConfig`] bundles all parameters that control the simulation
//! pipeline: number of scenarios to evaluate and the bounded channel capacity
//! that throttles the background I/O thread.

/// Parameters controlling the SDDP simulation pipeline.
///
/// Construct this struct directly — all fields are public and there is no
/// builder or `Default` implementation. Every field must be set explicitly
/// to prevent silent misconfiguration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::SimulationConfig;
///
/// let config = SimulationConfig {
///     n_scenarios: 500,
///     io_channel_capacity: 32,
///     basis_activity_window: 5,
/// };
/// assert_eq!(config.n_scenarios, 500);
/// assert_eq!(config.io_channel_capacity, 32);
/// ```
#[derive(Debug)]
pub struct SimulationConfig {
    /// Total number of scenarios to simulate across all MPI ranks.
    ///
    /// Scenarios are distributed statically across ranks using the same
    /// two-level distribution strategy as training (see
    /// simulation-architecture.md SS3.1). Must be at least 1.
    pub n_scenarios: u32,

    /// Bounded channel capacity for the background I/O thread.
    ///
    /// Controls the maximum number of [`SimulationScenarioResult`] instances
    /// that can be buffered in the channel between simulation threads and the
    /// background I/O thread. When the channel is full, simulation threads
    /// block until the I/O thread consumes a result, providing backpressure.
    ///
    /// Larger values increase memory usage but allow the I/O thread more
    /// headroom to absorb bursts. Default in practice is 64.
    ///
    /// [`SimulationScenarioResult`]: crate::SimulationScenarioResult
    pub io_channel_capacity: usize,

    /// Activity-window size for the basis-reconstruction classifier (1..=31).
    ///
    /// Must match the value used during training. Validated at study setup time
    /// via [`crate::StudyParams::from_config`].
    pub basis_activity_window: u32,
}

#[cfg(test)]
mod tests {
    use super::SimulationConfig;

    #[test]
    fn simulation_config_construction() {
        let config = SimulationConfig {
            n_scenarios: 2000,
            io_channel_capacity: 64,
            basis_activity_window: 5,
        };
        assert_eq!(config.n_scenarios, 2000);
        assert_eq!(config.io_channel_capacity, 64);
    }

    #[test]
    fn simulation_config_arbitrary_values() {
        let config = SimulationConfig {
            n_scenarios: 1,
            io_channel_capacity: 1,
            basis_activity_window: 5,
        };
        assert_eq!(config.n_scenarios, 1);
        assert_eq!(config.io_channel_capacity, 1);
    }

    #[test]
    fn simulation_config_debug_non_empty() {
        let config = SimulationConfig {
            n_scenarios: 100,
            io_channel_capacity: 16,
            basis_activity_window: 5,
        };
        let debug = format!("{config:?}");
        assert!(!debug.is_empty());
        assert!(
            debug.contains("n_scenarios"),
            "debug must contain field name: {debug}"
        );
        assert!(
            debug.contains("io_channel_capacity"),
            "debug must contain field name: {debug}"
        );
    }
}
