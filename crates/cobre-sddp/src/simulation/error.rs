//! Error type for the SDDP simulation execution phase.
//!
//! [`SimulationError`] is returned by the `fn simulate()` entry point; a
//! `From<SimulationError> for SddpError` conversion unifies it with `SddpError`.

use crate::SddpError;

use cobre_stochastic::StochasticError;

/// Errors that can occur during simulation execution.
#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    /// LP infeasibility at a simulation stage. Signals a bug: recourse slack
    /// variables should always make the LP feasible, so this is an LP-construction
    /// defect or a degenerate system, not normal operation.
    #[error("LP infeasible at scenario {scenario_id}, stage {stage_id}: {solver_message}")]
    LpInfeasible {
        /// 0-based scenario identifier.
        scenario_id: u32,
        /// 0-based stage identifier.
        stage_id: u32,
        /// Solver-provided diagnostic message.
        solver_message: String,
    },

    /// LP solver returned an unexpected status (e.g., numerical difficulties,
    /// unbounded). Includes solver-specific diagnostics.
    #[error("solver error at scenario {scenario_id}, stage {stage_id}: {solver_message}")]
    SolverError {
        /// 0-based scenario identifier.
        scenario_id: u32,
        /// 0-based stage identifier.
        stage_id: u32,
        /// Solver-provided diagnostic message.
        solver_message: String,
    },

    /// I/O failure during output writing; the simulation cannot continue because
    /// results would be lost.
    #[error("I/O error during simulation output: {message}")]
    IoError {
        /// Description of the I/O failure.
        message: String,
    },

    /// Channel send failure — the receiving end (I/O thread) has dropped
    /// unexpectedly. Indicates a panic or crash in the output writer.
    #[error("simulation output channel closed unexpectedly")]
    ChannelClosed,

    /// Forward sampler construction or sampling failure.
    #[error("stochastic error: {0}")]
    Stochastic(#[from] StochasticError),

    /// Invalid configuration passed to `simulate`, e.g. frozen-template slice length
    /// does not match `num_stages`.
    #[error("invalid simulation configuration: {0}")]
    InvalidConfiguration(String),
}

impl From<SimulationError> for SddpError {
    fn from(err: SimulationError) -> Self {
        Self::Simulation(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::SimulationError;
    use crate::SddpError;

    fn assert_send_sync_static<E: std::error::Error + Send + Sync + 'static>() {}

    #[test]
    fn simulation_error_is_send_sync_static() {
        assert_send_sync_static::<SimulationError>();
    }

    #[test]
    fn simulation_error_lp_infeasible_display() {
        let err = SimulationError::LpInfeasible {
            scenario_id: 5,
            stage_id: 3,
            solver_message: "primal infeasible".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains('5'), "{msg}");
        assert!(msg.contains('3'), "{msg}");
        assert!(msg.contains("primal infeasible"), "{msg}");
    }

    #[test]
    fn simulation_error_solver_error_display() {
        let err = SimulationError::SolverError {
            scenario_id: 10,
            stage_id: 7,
            solver_message: "numerical difficulties".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("10"), "{msg}");
        assert!(msg.contains('7'), "{msg}");
        assert!(msg.contains("numerical difficulties"), "{msg}");
    }

    #[test]
    fn simulation_error_io_error_display() {
        let err = SimulationError::IoError {
            message: "disk full".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("disk full"), "{msg}");
    }

    #[test]
    fn simulation_error_channel_closed_display() {
        let err = SimulationError::ChannelClosed;
        let msg = err.to_string();
        assert!(!msg.is_empty(), "ChannelClosed display must not be empty");
        assert!(
            msg.contains("channel") || msg.contains("closed"),
            "ChannelClosed display should mention channel or closed: {msg}"
        );
    }

    #[test]
    fn simulation_error_satisfies_std_error_trait() {
        let variants: Vec<SimulationError> = vec![
            SimulationError::LpInfeasible {
                scenario_id: 0,
                stage_id: 0,
                solver_message: "infeasible".to_string(),
            },
            SimulationError::SolverError {
                scenario_id: 0,
                stage_id: 0,
                solver_message: "error".to_string(),
            },
            SimulationError::IoError {
                message: "disk full".to_string(),
            },
            SimulationError::ChannelClosed,
        ];
        for err in &variants {
            let _: &dyn std::error::Error = err;
        }
    }

    #[test]
    fn from_simulation_error_to_sddp_error() {
        let variants: Vec<SimulationError> = vec![
            SimulationError::LpInfeasible {
                scenario_id: 1,
                stage_id: 2,
                solver_message: "primal infeasible".to_string(),
            },
            SimulationError::SolverError {
                scenario_id: 3,
                stage_id: 4,
                solver_message: "numerical issue".to_string(),
            },
            SimulationError::IoError {
                message: "write failed".to_string(),
            },
            SimulationError::ChannelClosed,
        ];

        for variant in variants {
            let original_msg = variant.to_string();
            let sddp_err: SddpError = variant.into();
            assert!(
                matches!(sddp_err, SddpError::Simulation(_)),
                "expected SddpError::Simulation, got {sddp_err:?}"
            );
            let sddp_msg = sddp_err.to_string();
            assert!(
                sddp_msg.contains(original_msg.as_str()) || sddp_msg.contains("simulation"),
                "SddpError display '{sddp_msg}' should contain original message or 'simulation'"
            );
        }
    }

    #[test]
    fn sddp_error_simulation_variant_display() {
        let err = SddpError::Simulation("test".to_string());
        let msg = err.to_string();
        assert!(
            msg.contains("simulation"),
            "display must contain 'simulation': {msg}"
        );
        assert!(msg.contains("test"), "display must contain 'test': {msg}");
    }
}
