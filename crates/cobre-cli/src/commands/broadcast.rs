//! Postcard-serializable types for MPI broadcast.
//!
//! Wraps SDDP configuration, stopping rules, cut selection, and opening tree
//! data for broadcast from rank 0 to all ranks.

use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{
    CutSelectionStrategy, InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet,
    StudyParams, DEFAULT_MAX_ITERATIONS,
};

use crate::error::CliError;

/// Postcard-serializable stopping rule.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastStoppingRule {
    IterationLimit { limit: u64 },
    TimeLimit { seconds: f64 },
    BoundStalling { iterations: u64, tolerance: f64 },
}

/// Postcard-serializable stopping mode.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastStoppingMode {
    Any,
    All,
}

/// Postcard-serializable warm-start basis mode.
///
/// Mirrors [`cobre_io::config::WarmStartBasisMode`] for postcard broadcast.
/// The I/O type uses `#[serde(rename_all = "snake_case")]` which is not
/// compatible with postcard's non-self-describing format; this plain enum
/// round-trips cleanly.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastWarmStartBasisMode {
    AlienOnly,
    NonAlienFirst,
}

impl From<cobre_io::config::WarmStartBasisMode> for BroadcastWarmStartBasisMode {
    fn from(mode: cobre_io::config::WarmStartBasisMode) -> Self {
        match mode {
            cobre_io::config::WarmStartBasisMode::AlienOnly => Self::AlienOnly,
            cobre_io::config::WarmStartBasisMode::NonAlienFirst => Self::NonAlienFirst,
        }
    }
}

impl From<BroadcastWarmStartBasisMode> for cobre_solver::highs::WarmStartBasisMode {
    fn from(mode: BroadcastWarmStartBasisMode) -> Self {
        match mode {
            BroadcastWarmStartBasisMode::AlienOnly => Self::AlienOnly,
            BroadcastWarmStartBasisMode::NonAlienFirst => Self::NonAlienFirst,
        }
    }
}

/// Postcard-serializable cut selection strategy.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastCutSelection {
    Disabled,
    Level1 {
        threshold: u64,
        check_frequency: u64,
    },
    Lml1 {
        memory_window: u64,
        check_frequency: u64,
    },
    Dominated {
        threshold: f64,
        check_frequency: u64,
    },
}

impl BroadcastCutSelection {
    pub(crate) fn from_strategy(strategy: Option<&CutSelectionStrategy>) -> Self {
        match strategy {
            None => Self::Disabled,
            Some(CutSelectionStrategy::Level1 {
                threshold,
                check_frequency,
            }) => Self::Level1 {
                threshold: *threshold,
                check_frequency: *check_frequency,
            },
            Some(CutSelectionStrategy::Lml1 {
                memory_window,
                check_frequency,
            }) => Self::Lml1 {
                memory_window: *memory_window,
                check_frequency: *check_frequency,
            },
            Some(CutSelectionStrategy::Dominated {
                threshold,
                check_frequency,
            }) => Self::Dominated {
                threshold: *threshold,
                check_frequency: *check_frequency,
            },
        }
    }

    pub(crate) fn into_strategy(self) -> Option<CutSelectionStrategy> {
        match self {
            Self::Disabled => None,
            Self::Level1 {
                threshold,
                check_frequency,
            } => Some(CutSelectionStrategy::Level1 {
                threshold,
                check_frequency,
            }),
            Self::Lml1 {
                memory_window,
                check_frequency,
            } => Some(CutSelectionStrategy::Lml1 {
                memory_window,
                check_frequency,
            }),
            Self::Dominated {
                threshold,
                check_frequency,
            } => Some(CutSelectionStrategy::Dominated {
                threshold,
                check_frequency,
            }),
        }
    }
}

/// Configuration snapshot broadcast from rank 0 to all ranks.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct BroadcastConfig {
    pub(crate) seed: u64,
    pub(crate) forward_passes: u32,
    pub(crate) stopping_rules: Vec<BroadcastStoppingRule>,
    pub(crate) stopping_mode: BroadcastStoppingMode,
    pub(crate) n_scenarios: u32,
    pub(crate) io_channel_capacity: u32,
    pub(crate) policy_path: String,
    pub(crate) inflow_method: InflowNonNegativityMethod,
    pub(crate) cut_selection: BroadcastCutSelection,
    pub(crate) cut_activity_tolerance: f64,
    /// Whether the training phase is enabled. When `false`, all ranks skip
    /// training and proceed directly to simulation (or exit).
    pub(crate) training_enabled: bool,
    /// Policy initialization mode.
    pub(crate) policy_mode: cobre_io::PolicyMode,
    /// Whether the visited-states archive should be allocated for export.
    pub(crate) export_states: bool,
    /// Maximum number of active cuts per stage (hard cap on LP size).
    ///
    /// `None` means no cap is enforced. Derived from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub(crate) budget: Option<u32>,
    /// Scenario source for the training forward pass, broadcast so non-root
    /// ranks can build the stochastic context with matching sampling schemes.
    pub(crate) training_source: ScenarioSource,
    /// Scenario source for the post-training simulation forward pass.
    pub(crate) simulation_source: ScenarioSource,
    /// Which `HiGHS` basis-setter to call on each warm-start.
    ///
    /// Broadcast so all ranks construct their solver workspace with the same
    /// mode. Defaults to `NonAlienFirst` (post-ticket-010).
    pub(crate) warm_start_basis_mode: BroadcastWarmStartBasisMode,
}

impl BroadcastConfig {
    pub(crate) fn from_config(config: &cobre_io::Config) -> Result<Self, CliError> {
        let params = StudyParams::from_config(config).map_err(CliError::from)?;
        // Use a sentinel path; the scenario source helpers only use the path for
        // historical-years look-up and error messages, which are not exercised here.
        let sentinel_path = std::path::Path::new("config.json");
        let training_source = config
            .training_scenario_source(sentinel_path)
            .map_err(CliError::from)?;
        let simulation_source = config
            .simulation_scenario_source(sentinel_path)
            .map_err(CliError::from)?;

        let stopping_rules = params
            .stopping_rule_set
            .rules
            .iter()
            .map(|r| match r {
                StoppingRule::IterationLimit { limit } => {
                    BroadcastStoppingRule::IterationLimit { limit: *limit }
                }
                StoppingRule::TimeLimit { seconds } => {
                    BroadcastStoppingRule::TimeLimit { seconds: *seconds }
                }
                StoppingRule::BoundStalling {
                    iterations,
                    tolerance,
                } => BroadcastStoppingRule::BoundStalling {
                    iterations: *iterations,
                    tolerance: *tolerance,
                },
                // SimulationBased and GracefulShutdown are evaluated on rank 0
                // only and are not broadcastable; fold into iteration limit for
                // non-root ranks. Warn so the user knows the rule was substituted.
                StoppingRule::SimulationBased { .. } | StoppingRule::GracefulShutdown => {
                    eprintln!(
                        "warning: stopping rule not broadcastable, \
                         substituting IterationLimit({DEFAULT_MAX_ITERATIONS})"
                    );
                    BroadcastStoppingRule::IterationLimit {
                        limit: DEFAULT_MAX_ITERATIONS,
                    }
                }
            })
            .collect();

        let stopping_mode = match params.stopping_rule_set.mode {
            StoppingMode::All => BroadcastStoppingMode::All,
            StoppingMode::Any => BroadcastStoppingMode::Any,
        };

        let cut_selection = BroadcastCutSelection::from_strategy(params.cut_selection.as_ref());

        Ok(Self {
            seed: params.seed,
            forward_passes: params.forward_passes,
            stopping_rules,
            stopping_mode,
            n_scenarios: params.n_scenarios,
            io_channel_capacity: u32::try_from(params.io_channel_capacity).unwrap_or(64),
            policy_path: params.policy_path,
            inflow_method: params.inflow_method,
            cut_selection,
            cut_activity_tolerance: params.cut_activity_tolerance,
            training_enabled: config.training.enabled,
            policy_mode: config.policy.mode,
            export_states: config.exports.states,
            budget: params.budget,
            training_source,
            simulation_source,
            warm_start_basis_mode: config.training.solver.warm_start_basis_mode.into(),
        })
    }
}

/// Postcard-serializable wrapper for [`OpeningTree`] broadcast.
///
/// Reconstructs the tree via [`OpeningTree::from_parts`] on all ranks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BroadcastOpeningTree {
    pub(crate) data: Vec<f64>,
    pub(crate) openings_per_stage: Vec<usize>,
    pub(crate) dim: usize,
}

pub(crate) fn stopping_rules_from_broadcast(cfg: &BroadcastConfig) -> StoppingRuleSet {
    let rules = cfg
        .stopping_rules
        .iter()
        .map(|r| match r {
            BroadcastStoppingRule::IterationLimit { limit } => {
                StoppingRule::IterationLimit { limit: *limit }
            }
            BroadcastStoppingRule::TimeLimit { seconds } => {
                StoppingRule::TimeLimit { seconds: *seconds }
            }
            BroadcastStoppingRule::BoundStalling {
                iterations,
                tolerance,
            } => StoppingRule::BoundStalling {
                iterations: *iterations,
                tolerance: *tolerance,
            },
        })
        .collect();

    let mode = match cfg.stopping_mode {
        BroadcastStoppingMode::All => StoppingMode::All,
        BroadcastStoppingMode::Any => StoppingMode::Any,
    };

    StoppingRuleSet { rules, mode }
}

/// Broadcast a serializable value from rank 0 to all ranks.
///
/// Rank 0 serializes and broadcasts length + bytes; non-root ranks deserialize.
/// Length 0 signals rank 0 failure, allowing all ranks to participate.
///
/// # Errors
///
/// Returns [`CliError::Internal`] on serialization, broadcast, or deserialization failure.
pub(crate) fn broadcast_value<T, C>(value: Option<T>, comm: &C) -> Result<T, CliError>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
    C: cobre_comm::Communicator,
{
    let is_root = comm.rank() == 0;

    let serialized: Vec<u8> = if is_root {
        match value {
            Some(ref v) => postcard::to_allocvec(v).map_err(|e| CliError::Internal {
                message: format!("serialization error: {e}"),
            })?,
            None => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let raw_len = serialized.len();
    #[allow(clippy::cast_possible_truncation)]
    let mut len_buf = [raw_len as u64];
    comm.broadcast(&mut len_buf, 0)
        .map_err(|e| CliError::Internal {
            message: format!("broadcast error (length): {e}"),
        })?;

    let len = usize::try_from(len_buf[0]).map_err(|e| CliError::Internal {
        message: format!("broadcast error (length conversion): {e}"),
    })?;
    if len == 0 {
        return Err(CliError::Internal {
            message: "rank 0 signaled broadcast failure (length 0)".to_string(),
        });
    }

    let mut bytes = if is_root { serialized } else { vec![0u8; len] };
    comm.broadcast(&mut bytes, 0)
        .map_err(|e| CliError::Internal {
            message: format!("broadcast error (data): {e}"),
        })?;

    if is_root {
        value.ok_or_else(|| CliError::Internal {
            message: "broadcast_value: root value disappeared after serialization".to_string(),
        })
    } else {
        postcard::from_bytes(&bytes).map_err(|e| CliError::Internal {
            message: format!("deserialization error: {e}"),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::{broadcast_value, BroadcastOpeningTree};

    /// A minimal serializable struct for testing the broadcast helper.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Simple {
        x: f64,
        label: String,
    }

    /// `broadcast_value` with `LocalBackend` (single rank) round-trips a struct.
    ///
    /// With `LocalBackend`, broadcast is a no-op and the root-path code path is
    /// exercised: the function returns the original `Some(value)` unchanged after
    /// verifying that serialization succeeds (len > 0).
    #[test]
    fn broadcast_value_local_round_trips_simple() {
        let comm = cobre_comm::LocalBackend;
        let original = Simple {
            x: std::f64::consts::PI,
            label: "test".to_string(),
        };
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    /// `broadcast_value` with `LocalBackend` round-trips a `Vec<f64>`.
    ///
    /// Verifies that the helper handles collection types that postcard can serialize.
    #[test]
    fn broadcast_value_local_round_trips_vec() {
        let comm = cobre_comm::LocalBackend;
        let original: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    /// `broadcast_value` with `LocalBackend` round-trips a nested struct matching
    /// the shape of `cobre_io::config::TrainingConfig`.
    ///
    /// Uses a locally defined struct to avoid a test dependency on cobre-io internals.
    #[test]
    fn broadcast_value_local_round_trips_config_like() {
        #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
        struct ConfigLike {
            forward_passes: u32,
            seed: Option<i64>,
        }

        let comm = cobre_comm::LocalBackend;
        let original = ConfigLike {
            forward_passes: 4,
            seed: Some(42),
        };
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    /// `broadcast_value` returns `CliError::Internal` when `None` is passed on root.
    ///
    /// Root rank must always supply `Some(value)`. Passing `None` on the only rank
    /// (`LocalBackend`, rank 0 == root) triggers the internal error path, returning
    /// [`crate::error::CliError::Internal`] rather than panicking.
    #[test]
    fn broadcast_value_returns_err_when_root_passes_none() {
        let comm = cobre_comm::LocalBackend;
        let result: Result<Simple, _> = broadcast_value(None, &comm);
        assert!(result.is_err(), "expected Err when root passes None");
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::error::CliError::Internal { .. }),
            "expected CliError::Internal, got: {err:?}"
        );
    }

    /// `broadcast_value` with `LocalBackend` round-trips a `u64` value.
    ///
    /// Verifies the broadcast helper serializes and deserializes primitive
    /// integer types correctly. Gated behind the `mpi` feature because this
    /// test exercises the same code path invoked by MPI-enabled runs (the
    /// `LocalBackend` substitutes for the real MPI communicator in tests).
    #[cfg(feature = "mpi")]
    #[test]
    fn broadcast_value_round_trips_u64() {
        let comm = cobre_comm::LocalBackend;
        let value: u64 = 42;
        let result = broadcast_value(Some(value), &comm).unwrap();
        assert_eq!(result, 42u64);
    }

    // ------------------------------------------------------------------
    // BroadcastOpeningTree tests
    // ------------------------------------------------------------------

    /// `BroadcastOpeningTree` round-trips through postcard serialization.
    ///
    /// Verifies that the wrapper type is fully postcard-compatible and that
    /// no field is lost during the serialize → deserialize round-trip.
    #[test]
    fn broadcast_opening_tree_round_trips_via_postcard() {
        let original = BroadcastOpeningTree {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            openings_per_stage: vec![2, 1],
            dim: 3,
        };
        let bytes = postcard::to_allocvec(&original).unwrap();
        let decoded: BroadcastOpeningTree = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.data, original.data, "data must survive round-trip");
        assert_eq!(
            decoded.openings_per_stage, original.openings_per_stage,
            "openings_per_stage must survive round-trip"
        );
        assert_eq!(decoded.dim, original.dim, "dim must survive round-trip");
    }

    // ------------------------------------------------------------------
    // BroadcastConfig tests
    // ------------------------------------------------------------------

    /// `BroadcastConfig::from_config()` propagates `config.training.enabled`
    /// to `training_enabled`. Verifies that the flag defaults to `true` and
    /// changes to `false` when explicitly set.
    #[test]
    fn broadcast_config_propagates_training_enabled() {
        use super::BroadcastConfig;

        // Config with training.enabled defaulting to true (not specified in JSON).
        let enabled_json = r#"{ "training": {} }"#;
        let enabled_config: cobre_io::Config = serde_json::from_str(enabled_json).unwrap();
        let bcast = BroadcastConfig::from_config(&enabled_config).unwrap();
        assert!(
            bcast.training_enabled,
            "training_enabled should default to true"
        );

        // Config with training.enabled explicitly set to false.
        let disabled_json = r#"{ "training": { "enabled": false } }"#;
        let disabled_config: cobre_io::Config = serde_json::from_str(disabled_json).unwrap();
        let bcast = BroadcastConfig::from_config(&disabled_config).unwrap();
        assert!(
            !bcast.training_enabled,
            "training_enabled should be false when config.training.enabled is false"
        );
    }

    /// `BroadcastOpeningTree` wrapped in `Option` round-trips via `broadcast_value`
    /// with `LocalBackend`. Covers both the `None` and `Some` cases.
    ///
    /// `Some(None)` represents "no user-supplied tree" and `Some(Some(...))` represents
    /// a valid user tree. Both must survive the broadcast without data loss.
    #[test]
    fn broadcast_optional_opening_tree_local_round_trips() {
        use cobre_stochastic::context::OpeningTree;

        let comm = cobre_comm::LocalBackend;

        // Case 1: no user tree — broadcast Some(None)
        let no_tree: Option<BroadcastOpeningTree> = None;
        let result = broadcast_value(Some(no_tree), &comm).unwrap();
        assert!(result.is_none(), "Some(None) must round-trip to None");

        // Case 2: user tree present — broadcast Some(Some(...))
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let ops = vec![2];
        let dim = 2usize;
        let source_tree = OpeningTree::from_parts(data.clone(), ops.clone(), dim);
        let bcast = Some(BroadcastOpeningTree {
            data: source_tree.data().to_vec(),
            openings_per_stage: source_tree.openings_per_stage_slice().to_vec(),
            dim: source_tree.dim(),
        });
        let result = broadcast_value(Some(bcast), &comm).unwrap();
        let bt = result.unwrap();
        let reconstructed = OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim);
        assert_eq!(
            reconstructed.data(),
            source_tree.data(),
            "reconstructed tree data must match source"
        );
        assert_eq!(
            reconstructed.dim(),
            source_tree.dim(),
            "reconstructed tree dim must match source"
        );
        assert_eq!(
            reconstructed.openings_per_stage_slice(),
            source_tree.openings_per_stage_slice(),
            "reconstructed tree openings_per_stage must match source"
        );
    }
}
