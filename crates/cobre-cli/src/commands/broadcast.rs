//! Postcard-serializable types for MPI broadcast from rank 0 to all ranks.

use std::num::NonZeroUsize;

use cobre_comm::Communicator;
use cobre_core::scenario::ScenarioSource;
use cobre_io::Config;
use cobre_io::PolicyMode;
use cobre_io::config::{BackwardScheduler, PhaseSolverProfileConfig};
use cobre_sddp::{
    BoundaryStateRequirements, CutSelectionStrategy, DEFAULT_MAX_ITERATIONS,
    InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet, StudyParams,
    setup::{
        NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        SimulationEnumeratedRequest, StageIdx, TypedVec,
    },
};

use crate::error::CliError;

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Postcard-serializable stopping rule. Mirrors [`StoppingRule`], which
/// derives neither `Serialize` nor `Deserialize`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastStoppingRule {
    IterationLimit {
        limit: u64,
    },
    TimeLimit {
        seconds: f64,
    },
    BoundStalling {
        iterations: u64,
        tolerance: f64,
    },
    Gap {
        /// Absolute gap tolerance, canonical R$.
        tolerance: Option<f64>,
        /// Relative gap tolerance (fraction).
        relative_tolerance: Option<f64>,
    },
}

/// Postcard-serializable stopping mode.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastStoppingMode {
    Any,
    All,
}

/// Postcard-serializable backward scheduler. Mirrors
/// [`BackwardScheduler`], whose internally-tagged serde representation
/// postcard (non-self-describing) refuses to deserialize (`WontImplement`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastBackwardScheduler {
    ByScenario,
    ByNode { block_size: Option<NonZeroUsize> },
}

impl From<BackwardScheduler> for BroadcastBackwardScheduler {
    fn from(value: BackwardScheduler) -> Self {
        match value {
            BackwardScheduler::ByScenario {} => Self::ByScenario,
            BackwardScheduler::ByNode { block_size } => Self::ByNode { block_size },
        }
    }
}

impl From<BroadcastBackwardScheduler> for BackwardScheduler {
    fn from(value: BroadcastBackwardScheduler) -> Self {
        match value {
            BroadcastBackwardScheduler::ByScenario => Self::ByScenario {},
            BroadcastBackwardScheduler::ByNode { block_size } => Self::ByNode { block_size },
        }
    }
}

/// Configuration snapshot broadcast from rank 0 to all ranks.
// Rationale (struct_excessive_bools): each bool is an independent config
// flag `StudyParams::from_config` resolves from a DIFFERENT `Config` section
// (training selection, training enable, exports); grouping them into an enum or
// a state machine would invent a joint state no config section actually declares.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct BroadcastConfig {
    pub(crate) seed: u64,
    pub(crate) forward_passes: u32,
    /// `true` when `training.selection = enumerated` is declared; every rank
    /// re-resolves `forward_passes` from the (identically-broadcast-derived)
    /// node graph once it exists.
    pub(crate) training_enumerated: bool,
    pub(crate) stopping_rules: Vec<BroadcastStoppingRule>,
    pub(crate) stopping_mode: BroadcastStoppingMode,
    pub(crate) n_scenarios: u32,
    /// `simulation.selection`'s resolution; every rank re-resolves
    /// `n_scenarios` from the node graph once it exists when this is
    /// [`SimulationEnumeratedRequest::Enumerated`].
    pub(crate) simulation_enumerated: SimulationEnumeratedRequest,
    pub(crate) io_channel_capacity: u32,
    pub(crate) policy_path: String,
    pub(crate) inflow_method: InflowNonNegativityMethod,
    pub(crate) cut_selection: Option<CutSelectionStrategy>,
    pub(crate) cut_activity_tolerance: f64,
    /// When `false`, all ranks skip training and proceed to simulation (or exit).
    pub(crate) training_enabled: bool,
    pub(crate) policy_mode: PolicyMode,
    /// Whether the visited-states archive is allocated for export.
    pub(crate) export_states: bool,
    /// Hard cap on active rows per stage; `None` means no cap. Sourced from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub(crate) budget: Option<u32>,
    /// Scenario source for the training forward pass, broadcast so non-root
    /// ranks build the stochastic context with matching sampling schemes.
    pub(crate) training_source: ScenarioSource,
    pub(crate) simulation_source: ScenarioSource,
    /// Backward-pass solver profile override (`training.solver.backward`),
    /// resolved identically on every rank by
    /// `cobre_sddp::solve::solver_phase::Phase::resolve_profile`.
    pub(crate) training_solver_backward: Option<PhaseSolverProfileConfig>,
    /// Forward-pass solver profile override (`training.solver.forward`).
    pub(crate) training_solver_forward: Option<PhaseSolverProfileConfig>,
    /// Simulation solver profile override (`simulation.solver`).
    pub(crate) simulation_solver: Option<PhaseSolverProfileConfig>,
    /// Backward-pass scheduler (`training.parallelism.backward_scheduler`),
    /// carrying the opening-block size when the `by_node` method is
    /// selected.
    pub(crate) backward_scheduler: BroadcastBackwardScheduler,
    /// Resolved objective cost-scale factor (`modeling.cost_scale_factor`),
    /// resolved identically on every rank by [`StudyParams::from_config`].
    pub(crate) cost_scale_factor: f64,
    /// Boundary-derived state requirements, resolved on rank 0
    /// (`resolve_boundary_state_requirements`) and set after `from_config`, then
    /// broadcast so every rank rebuilds the identical `StudyParams`: the
    /// inflow-lag depth widens `L_state` in `resolve_state_layout` and the
    /// present flag gates the water-bucket terminal mask before
    /// `inject_boundary_cuts` runs.
    pub(crate) boundary: BoundaryStateRequirements,
}

impl BroadcastConfig {
    pub(crate) fn from_config(config: &Config) -> Result<Self, CliError> {
        let params = StudyParams::from_config(config).map_err(CliError::from)?;
        // Sentinel path: the scenario-source helpers use it only for historical-years
        // look-up and error messages, neither exercised here.
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
                StoppingRule::Gap {
                    tolerance,
                    relative_tolerance,
                } => BroadcastStoppingRule::Gap {
                    tolerance: *tolerance,
                    relative_tolerance: *relative_tolerance,
                },
                // GracefulShutdown evaluates on rank 0 only and is not
                // broadcastable; non-root ranks fall back to an iteration limit.
                StoppingRule::GracefulShutdown => {
                    tracing::warn!(
                        "stopping rule not broadcastable, \
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

        Ok(Self {
            seed: params.seed,
            forward_passes: params.forward_passes,
            training_enumerated: params.training_enumerated,
            stopping_rules,
            stopping_mode,
            n_scenarios: params.n_scenarios,
            simulation_enumerated: params.simulation_enumerated,
            io_channel_capacity: u32::try_from(params.io_channel_capacity).unwrap_or(64),
            policy_path: params.policy_path,
            inflow_method: params.inflow_method,
            cut_selection: params.cut_selection.clone(),
            cut_activity_tolerance: params.cut_activity_tolerance,
            training_enabled: config.training.enabled,
            policy_mode: config.policy.mode,
            export_states: config.exports.states,
            budget: params.budget,
            training_source,
            simulation_source,
            training_solver_backward: params.training_solver_backward,
            training_solver_forward: params.training_solver_forward,
            simulation_solver: params.simulation_solver,
            backward_scheduler: params.backward_scheduler.into(),
            cost_scale_factor: params.cost_scale_factor,
            boundary: params.boundary,
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

/// Postcard-serializable wrapper for [`NodeGraph`] broadcast — plain
/// struct-of-`Vec`s (no tagged enum on the wire), mirroring
/// [`BroadcastOpeningTree`]'s shape. `NodeOpenings::source` becomes the
/// `is_external` flag; successor lists flatten to a CSR triple
/// (`successor_offsets`/`successor_child`/`successor_probability`).
///
/// Not currently wired into the live MPI broadcast: [`NodeGraph`] is a pure,
/// deterministic function of already-broadcast inputs (`System::policy_graph`
/// — carried whole by the existing `System` broadcast — and the standardized
/// scenario libraries, themselves rebuilt identically on every rank inside
/// `StudySetup::from_broadcast_params`), so every rank constructs a
/// bitwise-identical graph without a wire hop — the same guarantee
/// `cut_state_layouts`/`stage_templates`/`scenario_libraries` already rely on.
/// This type exists so a future caller can transport the graph explicitly
/// (e.g. to cross-check the deterministic-construction guarantee, the way
/// [`BroadcastOpeningTree`] transports a user-supplied tree) without
/// inventing a second wire shape.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BroadcastNodeGraph {
    pub(crate) node_ids: Vec<i32>,
    pub(crate) stage: Vec<usize>,
    pub(crate) pool_id: Vec<usize>,
    pub(crate) is_external: Vec<bool>,
    pub(crate) offset: Vec<usize>,
    pub(crate) len: Vec<usize>,
    pub(crate) q: Vec<f64>,
    pub(crate) n_pools: usize,
    /// CSR row-pointer over `successor_child`/`successor_probability`,
    /// length `node_ids.len() + 1`; node `i`'s successors are
    /// `successor_offsets[i]..successor_offsets[i + 1]`.
    pub(crate) successor_offsets: Vec<usize>,
    pub(crate) successor_child: Vec<usize>,
    pub(crate) successor_probability: Vec<f64>,
}

impl From<&NodeGraph> for BroadcastNodeGraph {
    fn from(ng: &NodeGraph) -> Self {
        let n = ng.nodes.len();
        let mut stage = Vec::with_capacity(n);
        let mut pool_id = Vec::with_capacity(n);
        let mut is_external = Vec::with_capacity(n);
        let mut offset = Vec::with_capacity(n);
        let mut len = Vec::with_capacity(n);
        let mut q = Vec::with_capacity(n);
        for node in &ng.nodes {
            stage.push(node.stage.0);
            pool_id.push(node.pool_id);
            is_external.push(node.openings.source == OpeningSource::External);
            offset.push(node.openings.offset);
            len.push(node.openings.len);
            q.push(node.openings.q);
        }

        let mut successor_offsets = Vec::with_capacity(n + 1);
        let mut successor_child = Vec::new();
        let mut successor_probability = Vec::new();
        successor_offsets.push(0);
        for succs in &ng.successors {
            for s in succs {
                successor_child.push(s.child.0);
                successor_probability.push(s.probability);
            }
            successor_offsets.push(successor_child.len());
        }

        Self {
            node_ids: ng.node_ids.iter().map(|id| id.0).collect(),
            stage,
            pool_id,
            is_external,
            offset,
            len,
            q,
            n_pools: ng.n_pools,
            successor_offsets,
            successor_child,
            successor_probability,
        }
    }
}

impl From<BroadcastNodeGraph> for NodeGraph {
    fn from(b: BroadcastNodeGraph) -> Self {
        let n = b.node_ids.len();
        let nodes: TypedVec<NodePos, NodeRuntime> = (0..n)
            .map(|i| NodeRuntime {
                stage: StageIdx(b.stage[i]),
                pool_id: b.pool_id[i],
                openings: NodeOpenings {
                    source: if b.is_external[i] {
                        OpeningSource::External
                    } else {
                        OpeningSource::Generated
                    },
                    offset: b.offset[i],
                    len: b.len[i],
                    q: b.q[i],
                },
            })
            .collect();
        // Derived, not transported: pool -> stage is a pure function of the
        // nodes, so re-deriving here keeps the wire format minimal and drift-free.
        let mut pool_stage = vec![StageIdx(0); b.n_pools];
        for node in &nodes {
            pool_stage[node.pool_id] = node.stage;
        }
        let successors = (0..n)
            .map(|i| {
                let start = b.successor_offsets[i];
                let end = b.successor_offsets[i + 1];
                (start..end)
                    .map(|j| NodeSuccessor {
                        child: NodePos(b.successor_child[j]),
                        probability: b.successor_probability[j],
                    })
                    .collect()
            })
            .collect();
        NodeGraph {
            node_ids: b.node_ids.into_iter().map(NodeId).collect(),
            nodes,
            successors,
            n_pools: b.n_pools,
            pool_stage,
        }
    }
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
            BroadcastStoppingRule::Gap {
                tolerance,
                relative_tolerance,
            } => StoppingRule::Gap {
                tolerance: *tolerance,
                relative_tolerance: *relative_tolerance,
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
/// A broadcast length of 0 signals rank 0 failure, letting all ranks return an error
/// in lockstep rather than deadlocking.
///
/// # Errors
///
/// Returns [`CliError::Internal`] on serialization, broadcast, or deserialization failure.
pub(crate) fn broadcast_value<T, C>(value: Option<T>, comm: &C) -> Result<T, CliError>
where
    T: Serialize + DeserializeOwned,
    C: Communicator,
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
    use super::{BroadcastNodeGraph, BroadcastOpeningTree, BroadcastStoppingRule, broadcast_value};
    use crate::error::CliError;
    use cobre_sddp::setup::{
        NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        StageIdx,
    };

    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Simple {
        x: f64,
        label: String,
    }

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

    #[test]
    fn broadcast_value_local_round_trips_vec() {
        let comm = cobre_comm::LocalBackend;
        let original: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    /// The reconciled boundary cuts ride `broadcast_value` to non-root ranks
    /// (`apply_training_policy`), so `OwnedPolicyCutRecord` must survive the
    /// postcard wire hop — losing its `Serialize` derive would silently strand
    /// every non-root rank with an empty terminal pool.
    #[test]
    fn broadcast_value_round_trips_owned_policy_cut_records() {
        let comm = cobre_comm::LocalBackend;
        let original = vec![cobre_io::OwnedPolicyCutRecord {
            cut_id: 7,
            slot_index: 3,
            iteration: 2,
            forward_pass_index: 0,
            intercept: 1.5,
            coefficients: vec![0.25, -0.5, 1.0],
            is_active: true,
        }];
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].cut_id, original[0].cut_id);
        assert_eq!(result[0].slot_index, original[0].slot_index);
        assert_eq!(result[0].intercept, original[0].intercept);
        assert_eq!(result[0].coefficients, original[0].coefficients);
        assert_eq!(result[0].is_active, original[0].is_active);
    }

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

    #[test]
    fn broadcast_value_returns_err_when_root_passes_none() {
        let comm = cobre_comm::LocalBackend;
        let result: Result<Simple, _> = broadcast_value(None, &comm);
        assert!(result.is_err(), "expected Err when root passes None");
        let err = result.unwrap_err();
        assert!(
            matches!(err, CliError::Internal { .. }),
            "expected CliError::Internal, got: {err:?}"
        );
    }

    /// Gated on `mpi`: exercises via `LocalBackend` the same path MPI runs invoke.
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
    // BroadcastNodeGraph tests
    // ------------------------------------------------------------------

    /// A 3-node fixture (one root fanning into a `Generated` and an
    /// `External` child) exercising both `OpeningSource` variants and a
    /// non-trivial successor CSR.
    fn sample_node_graph() -> NodeGraph {
        NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1), NodeId(2)].into(),
            nodes: vec![
                NodeRuntime {
                    stage: StageIdx(0),
                    pool_id: 0,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 3,
                        q: 1.0 / 3.0,
                    },
                },
                NodeRuntime {
                    stage: StageIdx(1),
                    pool_id: 1,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 4,
                        q: 0.25,
                    },
                },
                NodeRuntime {
                    stage: StageIdx(1),
                    pool_id: 2,
                    openings: NodeOpenings {
                        source: OpeningSource::External,
                        offset: 7,
                        len: 1,
                        q: 1.0,
                    },
                },
            ]
            .into(),
            successors: vec![
                vec![
                    NodeSuccessor {
                        child: NodePos(1),
                        probability: 0.5,
                    },
                    NodeSuccessor {
                        child: NodePos(2),
                        probability: 0.5,
                    },
                ],
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 3,
            pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1)],
        }
    }

    #[test]
    fn broadcast_node_graph_round_trips_via_postcard() {
        let original = sample_node_graph();
        let bcast = BroadcastNodeGraph::from(&original);
        let bytes = postcard::to_allocvec(&bcast).unwrap();
        let decoded_bcast: BroadcastNodeGraph = postcard::from_bytes(&bytes).unwrap();
        let decoded: NodeGraph = decoded_bcast.into();

        assert_eq!(decoded.node_ids, original.node_ids);
        assert_eq!(decoded.n_pools, original.n_pools);
        assert_eq!(
            decoded.pool_stage, original.pool_stage,
            "pool_stage must be re-derived identically after the wire hop"
        );
        assert_eq!(decoded.nodes.len(), original.nodes.len());
        for (d, o) in decoded.nodes.iter().zip(original.nodes.iter()) {
            assert_eq!(d.stage, o.stage);
            assert_eq!(d.pool_id, o.pool_id);
            assert_eq!(d.openings.source, o.openings.source);
            assert_eq!(d.openings.offset, o.openings.offset);
            assert_eq!(d.openings.len, o.openings.len);
            assert_eq!(
                d.openings.q.to_bits(),
                o.openings.q.to_bits(),
                "q must survive the wire hop bit-exact"
            );
        }
        assert_eq!(decoded.successors.len(), original.successors.len());
        for (d, o) in decoded.successors.iter().zip(original.successors.iter()) {
            assert_eq!(d.len(), o.len());
            for (de, oe) in d.iter().zip(o.iter()) {
                assert_eq!(de.child, oe.child);
                assert_eq!(de.probability.to_bits(), oe.probability.to_bits());
            }
        }
    }

    /// Guardrail mirroring `broadcast_config_wire_excludes_deleted_fields`:
    /// the wire type is a plain struct-of-`Vec`s, never a tagged enum — a
    /// serde-internally-tagged `OpeningSource` on the wire would inject a
    /// `"Generated"`/`"External"` string tag into the postcard bytes, which
    /// this test would catch.
    #[test]
    fn broadcast_node_graph_wire_carries_no_variant_tag_strings() {
        let bcast = BroadcastNodeGraph::from(&sample_node_graph());
        let bytes = postcard::to_allocvec(&bcast).unwrap();
        let as_string = String::from_utf8_lossy(&bytes);
        for tag in ["Generated", "External"] {
            assert!(
                !as_string.contains(tag),
                "BroadcastNodeGraph postcard bytes must not contain the enum variant tag '{tag}' \
                 — is_external must stay a plain bool field, not a tagged enum on the wire"
            );
        }
    }

    #[test]
    fn broadcast_optional_node_graph_local_round_trips() {
        let comm = cobre_comm::LocalBackend;
        let original = sample_node_graph();
        let bcast = Some(BroadcastNodeGraph::from(&original));
        let result = broadcast_value(Some(bcast), &comm).unwrap();
        let decoded: NodeGraph = result.unwrap().into();
        assert_eq!(decoded.node_ids, original.node_ids);
        assert_eq!(decoded.n_pools, original.n_pools);
    }

    // ------------------------------------------------------------------
    // BroadcastStoppingRule tests
    // ------------------------------------------------------------------

    #[test]
    fn broadcast_stopping_rule_gap_round_trips_via_postcard() {
        let original = BroadcastStoppingRule::Gap {
            tolerance: Some(1000.0),
            relative_tolerance: Some(0.01),
        };
        let bytes = postcard::to_allocvec(&original).unwrap();
        let decoded: BroadcastStoppingRule = postcard::from_bytes(&bytes).unwrap();
        assert!(matches!(
            decoded,
            BroadcastStoppingRule::Gap {
                tolerance: Some(t),
                relative_tolerance: Some(rt)
            } if (t - 1000.0).abs() < f64::EPSILON && (rt - 0.01).abs() < f64::EPSILON
        ));
    }

    // ------------------------------------------------------------------
    // BroadcastConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn broadcast_config_propagates_training_enabled() {
        use super::BroadcastConfig;

        // training.enabled omitted from JSON → defaults to true.
        let enabled_json = r#"{ "training": {} }"#;
        let enabled_config: cobre_io::Config = serde_json::from_str(enabled_json).unwrap();
        let bcast = BroadcastConfig::from_config(&enabled_config).unwrap();
        assert!(
            bcast.training_enabled,
            "training_enabled should default to true"
        );

        let disabled_json = r#"{ "training": { "enabled": false } }"#;
        let disabled_config: cobre_io::Config = serde_json::from_str(disabled_json).unwrap();
        let bcast = BroadcastConfig::from_config(&disabled_config).unwrap();
        assert!(
            !bcast.training_enabled,
            "training_enabled should be false when config.training.enabled is false"
        );
    }

    /// The resolved boundary requirements — patched onto `BroadcastConfig` by
    /// rank 0 after `from_config` (`from_config` itself leaves the `none()`
    /// placeholder) — survive the postcard wire hop to non-root ranks as one
    /// value carrying both the present flag and the inflow-lag depth.
    #[test]
    fn broadcast_config_carries_boundary_requirements() {
        use super::BroadcastConfig;
        use cobre_sddp::BoundaryStateRequirements;

        let json = r#"{
            "training": {
                "selection": { "method": "sampled", "forward_passes": 4 },
                "stopping_rules": [{ "type": "iteration_limit", "limit": 10 }]
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let mut original = BroadcastConfig::from_config(&config).unwrap();
        assert!(
            !original.boundary.is_present(),
            "from_config leaves the none() placeholder; the resolver owns presence"
        );
        original.boundary = BoundaryStateRequirements::present(12);

        let bytes = postcard::to_allocvec(&original).expect("postcard serialization must succeed");
        let decoded: BroadcastConfig =
            postcard::from_bytes(&bytes).expect("postcard deserialization must succeed");
        assert!(decoded.boundary.is_present());
        assert_eq!(decoded.boundary.inflow_lag_depth(), Some(12));
    }

    #[test]
    fn broadcast_config_roundtrips_via_postcard() {
        use cobre_core::scenario::{SamplingScheme, ScenarioSource};

        use super::BroadcastConfig;

        let json = r#"{
            "training": {
                "selection": { "method": "sampled", "forward_passes": 4 },
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 10 }
                ]
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let original = BroadcastConfig::from_config(&config).unwrap();

        let bytes = postcard::to_allocvec(&original)
            .expect("postcard serialization of BroadcastConfig must succeed");
        let decoded: BroadcastConfig = postcard::from_bytes(&bytes)
            .expect("postcard deserialization of BroadcastConfig must succeed");

        assert_eq!(decoded.seed, original.seed);
        assert_eq!(decoded.seed, 42u64);
        assert_eq!(decoded.forward_passes, original.forward_passes);
        assert_eq!(decoded.forward_passes, 4u32);
        assert_eq!(decoded.n_scenarios, original.n_scenarios);
        assert_eq!(decoded.n_scenarios, 0u32);
        assert_eq!(decoded.training_source, original.training_source);
        let default_source = ScenarioSource::default();
        assert_eq!(
            decoded.training_source.inflow_scheme,
            default_source.inflow_scheme
        );
        assert_eq!(
            decoded.training_source.inflow_scheme,
            SamplingScheme::InSample
        );
        assert_eq!(decoded.simulation_source, original.simulation_source);
        assert_eq!(
            decoded.simulation_source.inflow_scheme,
            SamplingScheme::InSample
        );
        assert!(
            decoded.training_solver_backward.is_none(),
            "absent training.solver.backward must round-trip to None"
        );
        assert!(
            decoded.training_solver_forward.is_none(),
            "absent training.solver.forward must round-trip to None"
        );
        assert!(
            decoded.simulation_solver.is_none(),
            "absent simulation.solver must round-trip to None"
        );
    }

    /// Postcard round-trip for the scheduler field at its non-default value —
    /// `broadcast_config_roundtrips_via_postcard` above only exercises the
    /// `ByScenario` default.
    #[test]
    fn broadcast_config_roundtrips_via_postcard_with_by_node_scheduler() {
        use std::num::NonZeroUsize;

        use super::{BroadcastBackwardScheduler, BroadcastConfig};

        let json = r#"{
            "training": {
                "parallelism": {
                    "backward_scheduler": { "method": "by_node", "block_size": 4 }
                }
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let original = BroadcastConfig::from_config(&config).unwrap();

        let bytes = postcard::to_allocvec(&original)
            .expect("postcard serialization of BroadcastConfig must succeed");
        let decoded: BroadcastConfig = postcard::from_bytes(&bytes)
            .expect("postcard deserialization of BroadcastConfig must succeed");

        assert_eq!(decoded.backward_scheduler, original.backward_scheduler);
        assert_eq!(
            decoded.backward_scheduler,
            BroadcastBackwardScheduler::ByNode {
                block_size: NonZeroUsize::new(4)
            }
        );
    }

    /// Postcard round-trip for a populated `training.solver.backward` /
    /// `.forward` / `simulation.solver` block: every field, not just presence,
    /// must survive the wire hop identically.
    #[test]
    fn broadcast_config_solver_profile_roundtrips_via_postcard() {
        use super::BroadcastConfig;

        let json = r#"{
            "training": {
                "selection": { "method": "sampled", "forward_passes": 4 },
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 10 }
                ],
                "solver": {
                    "backward": {
                        "price": "row_hyper_sparse"
                    },
                    "forward": {
                        "dual_edge_weight": "dantzig"
                    }
                }
            },
            "simulation": {
                "solver": { "scale": "solver_scaling" }
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let original = BroadcastConfig::from_config(&config).unwrap();

        let bytes = postcard::to_allocvec(&original)
            .expect("postcard serialization of BroadcastConfig must succeed");
        let decoded: BroadcastConfig = postcard::from_bytes(&bytes)
            .expect("postcard deserialization of BroadcastConfig must succeed");

        let original_backward = original
            .training_solver_backward
            .as_ref()
            .expect("backward profile must be Some");
        let decoded_backward = decoded
            .training_solver_backward
            .as_ref()
            .expect("backward profile must survive the round trip");
        assert_eq!(decoded_backward.price, original_backward.price);
        assert_eq!(
            decoded_backward.price,
            Some(cobre_io::config::PriceStrategy::RowHyperSparse)
        );
        assert!(decoded_backward.dual_edge_weight.is_none());

        let original_forward = original
            .training_solver_forward
            .as_ref()
            .expect("forward profile must be Some");
        let decoded_forward = decoded
            .training_solver_forward
            .as_ref()
            .expect("forward profile must survive the round trip");
        assert_eq!(
            decoded_forward.dual_edge_weight,
            original_forward.dual_edge_weight
        );
        assert_eq!(
            decoded_forward.dual_edge_weight,
            Some(cobre_io::config::DualEdgeWeight::Dantzig)
        );

        let original_sim = original
            .simulation_solver
            .as_ref()
            .expect("simulation profile must be Some");
        let decoded_sim = decoded
            .simulation_solver
            .as_ref()
            .expect("simulation profile must survive the round trip");
        assert_eq!(decoded_sim.scale, original_sim.scale);
        assert_eq!(
            decoded_sim.scale,
            Some(cobre_io::config::ScaleStrategy::SolverScaling)
        );
    }

    /// Cross-rank identity: `Phase::resolve_profile` is a pure function of the
    /// config, so resolving against rank 0's own value and against the
    /// postcard-decoded value (what a non-root rank sees after the broadcast)
    /// must produce bitwise-identical `ActiveProfile`s for every phase.
    #[test]
    fn resolve_profile_is_identical_before_and_after_broadcast_roundtrip() {
        use cobre_sddp::Phase;

        use super::BroadcastConfig;

        let json = r#"{
            "training": {
                "selection": { "method": "sampled", "forward_passes": 4 },
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 10 }
                ],
                "solver": {
                    "backward": { "dual_edge_weight": "steepest_edge" },
                    "forward": { "price": "row_hyper_sparse" }
                }
            },
            "simulation": {
                "solver": { "dual_edge_weight": "steepest_edge" }
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let rank0 = BroadcastConfig::from_config(&config).unwrap();

        let bytes = postcard::to_allocvec(&rank0).expect("postcard serialization must succeed");
        let non_root: BroadcastConfig =
            postcard::from_bytes(&bytes).expect("postcard deserialization must succeed");

        let rank0_backward =
            Phase::Backward.resolve_profile(rank0.training_solver_backward.as_ref());
        let non_root_backward =
            Phase::Backward.resolve_profile(non_root.training_solver_backward.as_ref());
        assert_eq!(
            rank0_backward, non_root_backward,
            "backward profile must resolve identically on every rank"
        );

        let rank0_forward = Phase::Forward.resolve_profile(rank0.training_solver_forward.as_ref());
        let non_root_forward =
            Phase::Forward.resolve_profile(non_root.training_solver_forward.as_ref());
        assert_eq!(
            rank0_forward, non_root_forward,
            "forward profile must resolve identically on every rank"
        );

        let rank0_sim = Phase::Simulation.resolve_profile(rank0.simulation_solver.as_ref());
        let non_root_sim = Phase::Simulation.resolve_profile(non_root.simulation_solver.as_ref());
        assert_eq!(
            rank0_sim, non_root_sim,
            "simulation profile must resolve identically on every rank"
        );
    }

    /// Guardrail: catches a future serializer switching to named fields and
    /// re-emitting stale field names into the postcard wire bytes.
    #[test]
    fn broadcast_config_wire_excludes_deleted_fields() {
        use super::BroadcastConfig;

        let json = r#"{
            "training": {
                "selection": { "method": "sampled", "forward_passes": 4 },
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 10 }
                ]
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let bcast = BroadcastConfig::from_config(&config).unwrap();
        let bytes = postcard::to_allocvec(&bcast).expect("postcard serialization must succeed");

        let as_string = String::from_utf8_lossy(&bytes);
        for stale in [
            "warm_start_basis_mode",
            "canonical_state_strategy",
            "basis_padding",
        ] {
            assert!(
                !as_string.contains(stale),
                "BroadcastConfig postcard bytes must not contain '{stale}'"
            );
        }
    }

    #[test]
    fn broadcast_optional_opening_tree_local_round_trips() {
        use cobre_stochastic::context::OpeningTree;

        let comm = cobre_comm::LocalBackend;

        // Some(None) = no user-supplied tree.
        let no_tree: Option<BroadcastOpeningTree> = None;
        let result = broadcast_value(Some(no_tree), &comm).unwrap();
        assert!(result.is_none(), "Some(None) must round-trip to None");

        // Some(Some(..)) = a user-supplied tree.
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
