//! Policy loading and compatibility validation.
//!
//! [`validate_policy_load`] is the single entry point for compatibility
//! validation; every load path (full-FCF warm-start/resume/simulation-only and
//! boundary-cut injection) routes through it, which first refuses a policy
//! written by another Cobre version, and returns a [`PolicyLoadProof`]
//! kind-typed to [`FullFcf`] or [`BoundaryInjection`] — the only way to obtain
//! one, so [`FutureCostFunction::new_with_warm_start`],
//! [`FutureCostFunction::from_deserialized`], and [`inject_boundary_cuts`]
//! cannot compile against unvalidated data.
//!
//! [`FutureCostFunction`]: crate::FutureCostFunction
//! [`FutureCostFunction::new_with_warm_start`]: crate::FutureCostFunction::new_with_warm_start
//! [`FutureCostFunction::from_deserialized`]: crate::FutureCostFunction::from_deserialized

use chrono::NaiveDate;
use cobre_core::AnticipatedCommitmentHistory;
use cobre_io::Config;
use cobre_io::EntitySlot;
use cobre_io::GraphManifest;
use cobre_io::OwnedPolicyBasisRecord;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::PolicyCheckpoint;
use cobre_io::SEASON_CYCLE_CODE_ABSENT;
use cobre_io::SEASON_CYCLE_CODE_CUSTOM;
use cobre_io::SEASON_CYCLE_CODE_MONTHLY;
use cobre_io::SEASON_CYCLE_CODE_WEEKLY;
use cobre_io::STAGE_CUTS_NODE_ID_SENTINEL;
use cobre_io::STAGE_CUTS_PRICED_STATE_DATE_SENTINEL;
use cobre_io::SeasonManifest;
use cobre_io::StageCutsReadResult;
use cobre_io::decode_slot_date;
use cobre_io::encode_slot_date;
use cobre_io::read_policy_checkpoint;
use cobre_solver::{Basis, BasisStatus};

use crate::SddpError;
use crate::cut::pool::CutPool;
use crate::policy::orchestration::StudySeasonManifest;
use crate::policy::reconcile::{
    BoundaryReconciliationReport, SlotKey, build_boundary_fold, build_identity_index, build_rebind,
    build_reconciliation_report, build_source_interval_index, rebind_cut,
};
use crate::setup::{BoundaryStateRequirements, NodeId, NodePos, StudySetup, TypedVec};
use crate::workspace::CapturedBasis;
use cobre_io::StateFamily;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::Path;

/// The version this build stamps into every checkpoint it writes, and the
/// only one [`validate_policy_load`] accepts.
pub const POLICY_COBRE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The constant every unmarked policy checkpoint (no `cost_scale_factor`
/// provenance) was unconditionally scaled at. A pool or checkpoint whose
/// `cost_scale_factor` reads `None` is interpreted under this constant.
///
/// Reserved seam: this repo's own front ends never reach the `None`/Legacy
/// branch — both the boundary path ([`load_boundary_cuts`]) and the [`FullFcf`]
/// path ([`checkpoint_terminal_cost_scale_factor`]) hard-reject a `None`
/// `cost_scale_factor` before rescale, so [`rescale_cut_records_for_load`]'s
/// Legacy branch is reachable only by a direct library caller.
pub const LEGACY_COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Rescale one stage's cut records from their at-rest
/// representation into the LOADING study's internal scaled cost space.
///
/// - **Marked** (`source_cost_scale_factor: Some(s)`): the checkpoint holds
///   canonical currency units — export multiplied every value by the writing
///   study's own `s` ([`crate::policy_export::scale_cut_records_for_export`]).
///   Every value here is divided by `loading_cost_scale_factor`, UNCONDITIONALLY
///   — even when `s` equals `loading_cost_scale_factor` — since the file
///   already carries one export-side rounding; a second division is the
///   accepted same-factor ULP drift, never special-cased
///   away.
/// - **Legacy** (`None`): the checkpoint holds the writing study's OWN internal
///   scaled values under [`LEGACY_COST_SCALE_FACTOR`] — legacy files carry
///   no export-side multiply. When `loading_cost_scale_factor ==
///   LEGACY_COST_SCALE_FACTOR` (the overwhelmingly common case: every existing
///   policy directory read at the still-default factor) this is an exact,
///   bit-identical no-op — a correctness requirement, not an optimization: a
///   legacy checkpoint at the default factor must load bit-identically, never
///   re-baselined. Otherwise every value is
///   multiplied by `LEGACY_COST_SCALE_FACTOR / loading_cost_scale_factor`.
pub(crate) fn rescale_cut_records_for_load(
    records: &mut [OwnedPolicyCutRecord],
    source_cost_scale_factor: Option<f64>,
    loading_cost_scale_factor: f64,
) {
    if source_cost_scale_factor.is_some() {
        for cut in records {
            cut.intercept /= loading_cost_scale_factor;
            for c in &mut cut.coefficients {
                *c /= loading_cost_scale_factor;
            }
        }
        return;
    }

    if loading_cost_scale_factor == LEGACY_COST_SCALE_FACTOR {
        return;
    }
    let ratio = LEGACY_COST_SCALE_FACTOR / loading_cost_scale_factor;
    for cut in records {
        cut.intercept *= ratio;
        for c in &mut cut.coefficients {
            *c *= ratio;
        }
    }
}

/// [`rescale_cut_records_for_load`] applied to every stage of a full policy
/// checkpoint — the [`FullFcf`] load path.
pub fn rescale_checkpoint_cuts_for_load(
    stage_cuts: &mut [StageCutsReadResult],
    source_cost_scale_factor: Option<f64>,
    loading_cost_scale_factor: f64,
) {
    for stage in stage_cuts {
        rescale_cut_records_for_load(
            &mut stage.cuts,
            source_cost_scale_factor,
            loading_cost_scale_factor,
        );
    }
}

/// The terminal pool's (`stage_cuts.last()`) `cost_scale_factor` — the
/// [`FullFcf`] load path's mirror of [`load_boundary_cuts`]'s own per-pool
/// read, resolving the source cost scale to feed
/// [`rescale_checkpoint_cuts_for_load`].
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `checkpoint.stage_cuts` is empty or
/// the terminal pool's `cost_scale_factor` reads `None`.
pub fn checkpoint_terminal_cost_scale_factor(
    checkpoint: &PolicyCheckpoint,
) -> Result<f64, SddpError> {
    checkpoint
        .stage_cuts
        .last()
        .and_then(|stage| stage.cost_scale_factor)
        .ok_or_else(|| {
            SddpError::Validation(
                "policy checkpoint predates self-describing cuts (its resolved \
                 cuts/<pool>.bin carries no cost_scale_factor); re-export it with a current Cobre"
                    .to_string(),
            )
        })
}

/// Per-side state layout fed to [`validate_policy_load`]: one manifest for the
/// loaded policy (`source`) and one for the study being trained or simulated
/// (`current`). The caller builds both — one from checkpoint metadata and its
/// entity manifest, the other from the live [`StudySetup`].
#[derive(Debug, Clone, Copy)]
pub struct PolicyStageManifest<'a> {
    /// Length of the state vector (one entry per reservoir/lag/bucket dimension).
    pub state_dimension: u32,
    /// Number of stages in the study.
    pub num_stages: u32,
    /// Number of storage pools (the pool-set size) — the pool-count analogue of
    /// `num_stages`, checked only under [`FullFcf`].
    pub n_pools: u32,
    /// Per-slot entity identity, in state-vector order.
    pub slots: &'a [EntitySlot],
    /// Graph manifest (node list, edges, node → pool map). Checked for identity
    /// only under [`FullFcf`]; a [`BoundaryInjection`] load ignores it.
    pub graph: &'a GraphManifest,
}

mod sealed {
    pub trait Sealed {}
}

/// Selects [`validate_policy_load`]'s check matrix. Sealed so [`FullFcf`] and
/// [`BoundaryInjection`] are the only implementors, making a
/// [`PolicyLoadProof<K>`] a proof of validation under exactly one real kind —
/// never a third, uncatalogued one.
pub trait PolicyLoadKind: sealed::Sealed {
    /// Whether `state_dimension` equality is hard-rejected for this kind.
    const CHECK_STATE_DIMENSION: bool;
    /// Whether `num_stages` equality is hard-rejected for this kind.
    const CHECK_NUM_STAGES: bool;
    /// Whether the pool count and graph-manifest identity are hard-rejected for
    /// this kind.
    const CHECK_N_POOLS: bool;
    /// Whether per-slot identity is checked here as an EXACT positional match
    /// ([`compare_manifest_slot_identity`]); `false` means the kind reconciles
    /// it by its own mechanism instead.
    const CHECK_SLOT_IDENTITY_EXACT: bool;
}

/// Full future-cost-function load (warm-start, resume, simulation-only):
/// `state_dimension`, `num_stages`, the pool count, the graph manifest, and
/// per-slot identity must match `current` exactly.
#[derive(Debug, Clone, Copy)]
pub struct FullFcf;

impl sealed::Sealed for FullFcf {}
impl PolicyLoadKind for FullFcf {
    const CHECK_STATE_DIMENSION: bool = true;
    const CHECK_NUM_STAGES: bool = true;
    const CHECK_N_POOLS: bool = true;
    const CHECK_SLOT_IDENTITY_EXACT: bool = true;
}

/// Single-stage boundary-cut injection into the terminal pool:
/// `state_dimension`, `num_stages`, the pool count, and the graph manifest are
/// unchecked (a monthly source may feed a weekly+monthly current study on a
/// different graph at a different state dimension); per-slot identity is
/// RECONCILED, not exact-matched — [`load_boundary_cuts`] wires
/// [`crate::policy::reconcile::build_rebind`]/`rebind_cut` in after this
/// validation succeeds.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryInjection;

impl sealed::Sealed for BoundaryInjection {}
impl PolicyLoadKind for BoundaryInjection {
    const CHECK_STATE_DIMENSION: bool = false;
    const CHECK_NUM_STAGES: bool = false;
    const CHECK_N_POOLS: bool = false;
    const CHECK_SLOT_IDENTITY_EXACT: bool = false;
}

/// Unforgeable, kind-typed evidence that [`validate_policy_load`] accepted a
/// `source`/`current` pair for load kind `K`. The private marker field means a
/// struct literal cannot be written outside this module, so a consumer
/// requiring `&PolicyLoadProof<K>` cannot compile against unvalidated data —
/// and a proof typed to the wrong `K` cannot substitute, since `K` is a
/// distinct type per kind.
#[derive(Debug)]
pub struct PolicyLoadProof<K: PolicyLoadKind> {
    /// Human-readable warning messages, in emission order.
    pub warnings: Vec<String>,
    _kind: PhantomData<K>,
}

/// Validate that `source` was written by this build's [`POLICY_COBRE_VERSION`]
/// — checked first, for every `K` — and that `source`'s state layout is
/// compatible with `current`'s, per `K`'s check matrix ([`FullFcf`],
/// [`BoundaryInjection`]). `col_scale`/scaling is never a compatibility
/// dimension: a state variable's identity and physical unit are independent of
/// how the LP happens to scale its column. This is the single entry point for
/// policy-load validation — its success is the only way to construct a
/// [`PolicyLoadProof<K>`], so every load path routes through it.
///
/// # Errors
///
/// Returns [`SddpError::PolicyVersionMismatch`] if `source_cobre_version` does
/// not exactly match [`POLICY_COBRE_VERSION`]. Otherwise returns
/// [`SddpError::Validation`] under [`FullFcf`] on a `state_dimension`
/// mismatch, a `num_stages` mismatch, or a per-slot identity mismatch (see
/// [`compare_manifest_slot_identity`]).
pub fn validate_policy_load<K: PolicyLoadKind>(
    source_cobre_version: &str,
    source: &PolicyStageManifest<'_>,
    current: &PolicyStageManifest<'_>,
) -> Result<PolicyLoadProof<K>, SddpError> {
    if source_cobre_version != POLICY_COBRE_VERSION {
        return Err(SddpError::PolicyVersionMismatch {
            policy_version: source_cobre_version.to_string(),
        });
    }

    if K::CHECK_STATE_DIMENSION && source.state_dimension != current.state_dimension {
        return Err(SddpError::Validation(format!(
            "policy state_dimension mismatch: policy has {}, current system has {} (a lag-state \
             depth mismatch is a common cause)",
            source.state_dimension, current.state_dimension
        )));
    }

    if K::CHECK_NUM_STAGES && source.num_stages != current.num_stages {
        return Err(SddpError::Validation(format!(
            "policy num_stages mismatch: policy has {}, current system has {}",
            source.num_stages, current.num_stages
        )));
    }

    if K::CHECK_N_POOLS && source.n_pools != current.n_pools {
        return Err(SddpError::Validation(format!(
            "policy n_pools mismatch: policy has {}, current system has {}",
            source.n_pools, current.n_pools
        )));
    }

    let mut warnings = Vec::new();
    if K::CHECK_SLOT_IDENTITY_EXACT {
        compare_manifest_slot_identity(source.slots, current.slots, &mut |msg| {
            warnings.push(msg.to_string());
        })?;
    }

    if K::CHECK_N_POOLS {
        compare_graph_manifest_identity(source.graph, current.graph)?;
    }

    Ok(PolicyLoadProof {
        warnings,
        _kind: PhantomData,
    })
}

/// Compare two graph manifests for structural identity: pool-set size, per-node
/// `(id, stage_id, pool_id)`, and per-edge `(source_id, target_id)`.
///
/// A [`FullFcf`] resume/warm-start continues the SAME node topology, so a
/// divergence means the loaded value function attaches to a different graph and
/// is REJECTED. An empty manifest on either side (a graph-less artifact — e.g.
/// one authored from raw records) cannot be verified: silently skip, leaving the
/// `state_dimension`/`num_stages`/`n_pools` checks standing.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] on a pool-count, node, or edge divergence.
pub fn compare_graph_manifest_identity(
    source: &GraphManifest,
    current: &GraphManifest,
) -> Result<(), SddpError> {
    if source.nodes.is_empty() || current.nodes.is_empty() {
        return Ok(());
    }

    if source.n_pools != current.n_pools {
        return Err(SddpError::Validation(format!(
            "graph manifest n_pools mismatch: source has {}, current study has {}",
            source.n_pools, current.n_pools
        )));
    }

    if source.nodes.len() != current.nodes.len() {
        return Err(SddpError::Validation(format!(
            "graph manifest node-count mismatch: source has {} nodes, current study has {}",
            source.nodes.len(),
            current.nodes.len()
        )));
    }

    for (i, (src, cur)) in source.nodes.iter().zip(&current.nodes).enumerate() {
        if (src.id, src.stage_id, src.pool_id) != (cur.id, cur.stage_id, cur.pool_id) {
            return Err(SddpError::Validation(format!(
                "graph manifest node {i} mismatch: source (id={}, stage_id={}, pool_id={}) != \
                 current (id={}, stage_id={}, pool_id={})",
                src.id, src.stage_id, src.pool_id, cur.id, cur.stage_id, cur.pool_id
            )));
        }
    }

    if source.edges.len() != current.edges.len() {
        return Err(SddpError::Validation(format!(
            "graph manifest edge-count mismatch: source has {} edges, current study has {}",
            source.edges.len(),
            current.edges.len()
        )));
    }

    for (i, (src, cur)) in source.edges.iter().zip(&current.edges).enumerate() {
        if (src.source_id, src.target_id) != (cur.source_id, cur.target_id) {
            return Err(SddpError::Validation(format!(
                "graph manifest edge {i} mismatch: source ({} -> {}) != current ({} -> {})",
                src.source_id, src.target_id, cur.source_id, cur.target_id
            )));
        }
    }

    Ok(())
}

/// Build a basis cache from deserialized checkpoint basis records: one entry
/// per canonical node position, `None` where no record matches.
///
/// `node_dims[pos]` — built once per load from `setup`'s current LP
/// templates — is `(template columns, template rows)` for node `pos`;
/// [`build_basis_cache_for_nodes`] refuses a record whose column count
/// differs, or whose row count falls outside `[template rows, template rows +
/// its own recorded cut rows]`, before decoding it.
///
/// # Errors
///
/// Returns [`SddpError::StoredBasisDimensionMismatch`] when a stored record's
/// column or row count does not fit its node's current LP dimensions.
pub fn build_basis_cache_from_checkpoint(
    stage_bases: &[OwnedPolicyBasisRecord],
    stage_cuts: &[StageCutsReadResult],
    setup: &StudySetup,
) -> Result<Vec<Option<CapturedBasis>>, SddpError> {
    let node_dims: Vec<(usize, usize)> = setup
        .inputs
        .node_graph
        .nodes
        .iter()
        .map(|n| {
            let t = &setup.inputs.stage_data.stage_templates.templates[n.stage.0];
            (t.num_cols, t.num_rows)
        })
        .collect();
    build_basis_cache_for_nodes(
        stage_bases,
        stage_cuts,
        &setup.inputs.node_graph.node_ids,
        &setup.inputs.node_graph.node_pool_ids(),
        &node_dims,
    )
}

/// Each basis record is keyed by its own node ordinal (its `stage_id`), so
/// leaves sharing a pool land in distinct node slots — no `>= num_stages` drop,
/// no cross-node collision. `u8` status codes decode via
/// `from_discriminant_code`, the mirror of `convert_basis_cache`'s export-side
/// `to_discriminant_code`; a pre-existing checkpoint (bytes `0..=4`) decodes
/// identically, since that range means the same in the canonical and `HiGHS`
/// code spaces.
///
/// # Cut-slot reconstruction
///
/// `base_row_count` is taken from `node_dims[pos]` — the STUDY's own template row
/// count — never from the record's `row_status.len() - num_cut_rows`: the root
/// node is captured in the last forward pass, before that iteration's backward
/// pass appends cuts to its pool, so its recorded `num_cut_rows` (the pool's size
/// at export) legitimately exceeds the cut rows the basis was captured with. The
/// trailing `k = row_status.len() - base_row_count` rows are matched to pool slot
/// identity only when a node's OWN pool's [`StageCutsReadResult`]
/// (`sc.stage_id == node_pools[node]`, never the record whose pool id happens to
/// equal the node ordinal) reports its active `slot_index`es as exactly the
/// unbroken prefix `0..populated` — proof that no cut was deactivated since
/// capture, so the basis's k oldest cut rows are exactly that pool's first k
/// slots. `reconstruct_basis` then preserves stored cut-row statuses across
/// cut-set churn.
///
/// # Graceful fallback
///
/// When the pool's active slots are not the unbroken `0..populated` prefix, when
/// they are fewer than `k`, or when no cut record matches the node's pool at all,
/// slot identity cannot be proven: `row_status` is truncated to `base_row_count`
/// and `cut_row_slots` stays empty, so every current cut row reconstructs BASIC.
/// This changes only the warm-start solve path, never the optimum.
///
/// `node_ids` / `node_pools` are the CURRENT study's, since a resume/warm-start
/// continues the SAME node topology — never a value recovered from the
/// checkpoint itself, whose wire carries no node id.
///
/// # Errors
///
/// Returns [`SddpError::StoredBasisDimensionMismatch`] when a stored record's
/// column count differs from `node_dims[pos].0`, or its row count falls
/// outside `[node_dims[pos].1, node_dims[pos].1 + its own num_cut_rows]` —
/// checked against the RECORDED `num_cut_rows`, never the derived
/// `base_row_count` below.
fn build_basis_cache_for_nodes(
    stage_bases: &[OwnedPolicyBasisRecord],
    stage_cuts: &[StageCutsReadResult],
    node_ids: &TypedVec<NodePos, NodeId>,
    node_pools: &TypedVec<NodePos, usize>,
    node_dims: &[(usize, usize)],
) -> Result<Vec<Option<CapturedBasis>>, SddpError> {
    let n_nodes = node_ids.len();
    let mut cache: Vec<Option<CapturedBasis>> = vec![None; n_nodes];
    for record in stage_bases {
        // Wire boundary: the checkpoint's `stage_id` field is a legacy name for
        // what the node-native engine writes/reads as a node position — convert
        // to `NodePos` immediately, never carry the raw wire int past this line.
        let node = NodePos(record.stage_id as usize);
        if node.0 >= n_nodes {
            continue;
        }

        let (expected_cols, expected_template_rows) = node_dims[node.0];
        let found_cols = record.column_status.len();
        let found_rows = record.row_status.len();
        let found_cut_rows = record.num_cut_rows as usize;
        if found_cols != expected_cols
            || found_rows < expected_template_rows
            || found_rows - expected_template_rows > found_cut_rows
        {
            return Err(SddpError::StoredBasisDimensionMismatch {
                node_id: node_ids[node].0,
                expected_cols,
                found_cols,
                expected_template_rows,
                found_rows,
                found_cut_rows,
            });
        }

        let col_status: Vec<BasisStatus> = record
            .column_status
            .iter()
            .map(|&c| BasisStatus::from_discriminant_code(c))
            .collect();
        let mut row_status: Vec<BasisStatus> = record
            .row_status
            .iter()
            .map(|&r| BasisStatus::from_discriminant_code(r))
            .collect();

        let pool = node_pools[node];
        let base_row_count = expected_template_rows;
        let k = row_status.len() - base_row_count;
        let cut_row_slots = stage_cuts
            .iter()
            .find(|sc| sc.stage_id as usize == pool)
            .and_then(|sc| {
                let populated = sc.cuts.len();
                let active: Vec<u32> = sc
                    .cuts
                    .iter()
                    .filter(|c| c.is_active)
                    .map(|c| c.slot_index)
                    .collect();
                let is_unbroken_prefix = active.len() == populated
                    && active.iter().enumerate().all(|(i, &s)| s as usize == i);
                (is_unbroken_prefix && k <= active.len()).then(|| active[..k].to_vec())
            });
        let cut_row_slots = cut_row_slots.unwrap_or_else(|| {
            row_status.truncate(base_row_count);
            Vec::new()
        });
        debug_assert_eq!(
            cut_row_slots.len(),
            row_status.len() - base_row_count,
            "build_basis_cache_for_nodes: cut_row_slots length must equal the trailing \
             cut-row count for the CapturedBasis invariant",
        );

        cache[node.0] = Some(CapturedBasis {
            basis: Basis {
                col_status,
                row_status,
            },
            base_row_count,
            cut_row_slots,
            state_at_capture: Vec::new(),
            node_id: node_ids[node],
        });
    }
    Ok(cache)
}

/// Positional identity of one state-vector slot; `was_active` is excluded —
/// adding it would reject a cut whose entity merely changed activity across
/// studies.
fn slot_identity(slot: &EntitySlot) -> (u8, i32, u32) {
    (slot.entity_type, slot.entity_id, slot.subindex)
}

/// Whether `source`/`current` can be identity-verified at all: an empty
/// manifest on either side (a pre-manifest checkpoint) cannot be. Answers the
/// emptiness question only — each caller decides how to report a `false`
/// result, leaving `state_dimension` as the sole compatibility guard.
fn manifest_identity_verifiable(source: &[EntitySlot], current: &[EntitySlot]) -> bool {
    !source.is_empty() && !current.is_empty()
}

/// Compare two entity manifests slot-for-slot by `slot_identity`.
///
/// A per-slot [`slot_identity`] mismatch means a cut coefficient would attach
/// to the wrong state variable and is REJECTED; a `source`-dormant slot now
/// active only warns. An unverifiable manifest
/// ([`manifest_identity_verifiable`]) warns and returns `Ok`, leaving the
/// caller's `state_dimension` check standing.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `source` and `current` differ in length
/// or in any slot's `(entity_type, entity_id, subindex)`.
pub fn compare_manifest_slot_identity(
    source: &[EntitySlot],
    current: &[EntitySlot],
    on_warning: &mut dyn FnMut(&str),
) -> Result<(), SddpError> {
    if !manifest_identity_verifiable(source, current) {
        on_warning(&format!(
            "entity manifest absent (source slots: {}, current slots: {}); slot identity \
             could not be verified, relying on state_dimension alone",
            source.len(),
            current.len(),
        ));
        return Ok(());
    }

    if source.len() != current.len() {
        return Err(SddpError::Validation(format!(
            "entity manifest length mismatch: source has {} slots, current study has {}",
            source.len(),
            current.len()
        )));
    }

    for (i, (src, cur)) in source.iter().zip(current).enumerate() {
        if slot_identity(src) != slot_identity(cur) {
            return Err(SddpError::Validation(format!(
                "entity-identity mismatch at slot {i}: \
                 source (entity_type={}, entity_id={}, subindex={}) != \
                 current (entity_type={}, entity_id={}, subindex={}); \
                 the cut coefficient at this slot would attach to the wrong state variable",
                src.entity_type,
                src.entity_id,
                src.subindex,
                cur.entity_type,
                cur.entity_id,
                cur.subindex
            )));
        }
        if !src.was_active && cur.was_active {
            on_warning(&format!(
                "slot {i} (entity_type={}, entity_id={}, subindex={}) was dormant in the source \
                 policy but is active in the current study; loading its cut",
                cur.entity_type, cur.entity_id, cur.subindex
            ));
        }
    }

    Ok(())
}

/// The deepest inflow-lag slot a manifest carries a cut coefficient on — the
/// 1-based `HydroInflowLag` subindex (`policy_export::build_stage_entity_manifest`
/// emits `lag + 1`), `0` when the manifest carries no lag slot.
fn boundary_cut_lag_depth(manifest: &[EntitySlot]) -> u32 {
    manifest
        .iter()
        .filter(|slot| slot.entity_type == StateFamily::HydroInflowLag.code())
        .map(|slot| slot.subindex)
        .max()
        .unwrap_or(0)
}

/// The deepest inflow-lag slot any cut pool in the boundary policy at
/// `boundary_path` references — the number of inflow-lag state slots the current
/// study must reserve so the loaded cuts project without truncation. Taken as the
/// max over every pool: the inflow-lag block is a property of the policy's one
/// state space, so the deepest slot is pool-invariant for a coherent policy and
/// the max never under-reserves. `0` when the policy carries no inflow-lag slot.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if the checkpoint cannot be read or parsed.
pub fn boundary_policy_required_lag_depth(boundary_path: &Path) -> Result<u32, SddpError> {
    let checkpoint = read_policy_checkpoint(boundary_path).map_err(|e| {
        SddpError::Validation(format!(
            "failed to read boundary policy checkpoint at {}: {e}",
            boundary_path.display()
        ))
    })?;
    Ok(checkpoint
        .stage_cuts
        .iter()
        .map(|sr| boundary_cut_lag_depth(&sr.entity_manifest))
        .max()
        .unwrap_or(0))
}

/// Resolve the boundary-derived state requirements for a study: the single owner
/// of the boundary → state-space channel. `config.policy.boundary` absent yields
/// [`BoundaryStateRequirements::none`] (the lag block sizes from the PAR model
/// alone); otherwise the requirements are inferred from the source checkpoint's
/// cuts (the inflow-lag depth via [`boundary_policy_required_lag_depth`]). Every
/// entry point resolves this once and threads the value onto the
/// config-projection carriers, so [`resolve_state_layout`](crate::setup::resolve_state_layout)
/// and the boundary-load reject see the identical requirements on every rank.
///
/// # Errors
///
/// Propagates [`boundary_policy_required_lag_depth`]'s checkpoint read failure.
pub fn resolve_boundary_state_requirements(
    case_dir: &Path,
    config: &Config,
) -> Result<BoundaryStateRequirements, SddpError> {
    match config.policy.boundary.as_ref() {
        None => Ok(BoundaryStateRequirements::none()),
        Some(bp) => Ok(BoundaryStateRequirements::present(
            boundary_policy_required_lag_depth(&bp.checkpoint_path(case_dir))?,
        )),
    }
}

/// A resolved boundary pool whose `cuts/<pool>.bin` predates the
/// self-describing per-pool facts (`cost_scale_factor` reads `None`): no
/// silent default, no `metadata.json` fallback — the checkpoint must be
/// re-exported.
fn boundary_predates_self_describing_cuts(boundary_path: &Path) -> SddpError {
    SddpError::Validation(format!(
        "boundary policy checkpoint at {} predates self-describing cuts (its resolved \
         cuts/<pool>.bin carries no cost_scale_factor); re-export it with a current Cobre",
        boundary_path.display()
    ))
}

/// The parameter carrier for [`load_boundary_cuts`]. Fields are private and
/// reached through the `with_*` builders below, each of which leaves every
/// other field unchanged, so the parameter set can grow without touching call
/// sites that do not supply the new field.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryLoadRequest<'a> {
    boundary_path: &'a Path,
    boundary_date: NaiveDate,
    current_state_dimension: u32,
    current_manifest: &'a [EntitySlot],
    loading_cost_scale_factor: f64,
    fixed_windows: &'a [AnticipatedCommitmentHistory],
    inflow_lag_depth: Option<u32>,
    study_seasons: Option<&'a StudySeasonManifest>,
    strict: bool,
}

impl<'a> BoundaryLoadRequest<'a> {
    /// Builds a request from the values every [`load_boundary_cuts`] caller
    /// must supply; the rest default until a `with_*` call overrides them.
    #[must_use]
    pub fn new(
        boundary_path: &'a Path,
        boundary_date: NaiveDate,
        current_state_dimension: u32,
        current_manifest: &'a [EntitySlot],
        loading_cost_scale_factor: f64,
    ) -> Self {
        Self {
            boundary_path,
            boundary_date,
            current_state_dimension,
            current_manifest,
            loading_cost_scale_factor,
            fixed_windows: &[],
            inflow_lag_depth: None,
            study_seasons: None,
            strict: false,
        }
    }

    /// Returns `self` with `fixed_windows` replaced.
    #[must_use]
    pub fn with_fixed_windows(self, fixed_windows: &'a [AnticipatedCommitmentHistory]) -> Self {
        Self {
            fixed_windows,
            ..self
        }
    }

    /// Returns `self` with `inflow_lag_depth` replaced.
    #[must_use]
    pub fn with_inflow_lag_depth(self, inflow_lag_depth: Option<u32>) -> Self {
        Self {
            inflow_lag_depth,
            ..self
        }
    }

    /// Returns `self` with `study_seasons` set. Build it with
    /// [`crate::policy::orchestration::build_season_manifest`] — the same
    /// function the checkpoint writer calls — so the study and source sides
    /// are never constructed by diverging code paths.
    #[must_use]
    pub fn with_study_seasons(self, study_seasons: &'a StudySeasonManifest) -> Self {
        Self {
            study_seasons: Some(study_seasons),
            ..self
        }
    }

    /// Returns `self` with `strict` replaced.
    #[must_use]
    pub fn with_strict(self, strict: bool) -> Self {
        Self { strict, ..self }
    }
}

/// A checkpoint whose every pool carries
/// [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`] predates recorded priced dates:
/// date selection has nothing to compare against, so the checkpoint must be
/// re-exported.
fn boundary_checkpoint_undated(boundary_path: &Path) -> SddpError {
    SddpError::Validation(format!(
        "boundary policy checkpoint at {} carries no priced_state_date on any pool (a pool \
         written before priced dates were recorded); re-export it with a current Cobre",
        boundary_path.display()
    ))
}

/// Selects the unique pool in `checkpoint` whose own `priced_state_date`
/// equals `boundary_date` — an index-based pool lookup could silently pick a
/// different, wrong future-cost function whose reconciliation tally is
/// identical, so only a calendar equality check catches the off-by-one.
/// Encodes the study side ONCE via
/// [`encode_slot_date`] and compares integers; never decodes a pool's stamp
/// to compare `NaiveDate` values, so a malformed stamp can never compare
/// equal to a real date. [`load_boundary_cuts`]'s `# Errors` catalogues the
/// rejects this enforces.
fn select_boundary_pool<'a>(
    checkpoint: &'a PolicyCheckpoint,
    boundary_path: &Path,
    boundary_date: NaiveDate,
) -> Result<&'a StageCutsReadResult, SddpError> {
    if checkpoint
        .stage_cuts
        .iter()
        .all(|sr| sr.priced_state_date == STAGE_CUTS_PRICED_STATE_DATE_SENTINEL)
    {
        return Err(boundary_checkpoint_undated(boundary_path));
    }

    let target = encode_slot_date(boundary_date);
    // Positional walk: `read_policy_checkpoint` sorts `stage_cuts` by pool id,
    // which is what orders the tie and available-dates lists below.
    let matched: Vec<&StageCutsReadResult> = checkpoint
        .stage_cuts
        .iter()
        .filter(|sr| sr.priced_state_date == target)
        .collect();

    match matched.as_slice() {
        [] => {
            let available = checkpoint
                .stage_cuts
                .iter()
                .map(|sr| match decode_slot_date(sr.priced_state_date) {
                    Some(date) => format!("(pool {}, priced {date})", sr.stage_id),
                    None => format!("(pool {}, priced {})", sr.stage_id, sr.priced_state_date),
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(SddpError::Validation(format!(
                "boundary policy at {}: no pool is priced at the study's boundary date \
                 {boundary_date} (available: {available})",
                boundary_path.display()
            )))
        }
        [single] => Ok(*single),
        multiple => {
            let pool_ids: Vec<u32> = multiple.iter().map(|sr| sr.stage_id).collect();
            Err(SddpError::Validation(format!(
                "boundary policy at {}: more than one pool is priced at the study's boundary \
                 date {boundary_date} (pools {pool_ids:?}); boundary injection requires a \
                 unique priced source",
                boundary_path.display()
            )))
        }
    }
}

/// Word label for a [`SeasonManifest::cycle_code`] discriminant, for
/// [`check_season_compatibility`]'s reject messages.
fn season_cycle_label(code: u8) -> &'static str {
    match code {
        SEASON_CYCLE_CODE_MONTHLY => "monthly",
        SEASON_CYCLE_CODE_WEEKLY => "weekly",
        SEASON_CYCLE_CODE_CUSTOM => "custom",
        SEASON_CYCLE_CODE_ABSENT => "absent",
        _ => "unknown",
    }
}

/// Rejects a boundary load whose season cycle or per-hydro PAR order
/// disagrees with `study`'s. `resolve_inflow_lag`'s join maps a lag depth to
/// a season context, and the coefficient priced under it, so the two sides
/// must share the same season definitions and PAR orders before any lag
/// coefficient moves. Checked most-diagnostic-first: source absence, cycle,
/// season count, then a per-hydro walk.
///
/// Walks `study.hydro_orders` positionally — both sides are canonical
/// ascending `hydro_id`
/// ([`orchestration::build_season_manifest`](crate::policy::orchestration::build_season_manifest)) —
/// and looks each hydro up in `source.hydro_orders` by binary search, never a
/// `HashMap`. A hydro present only in `source` is a superset drop and is
/// never examined here.
fn check_season_compatibility(
    boundary_path: &Path,
    study: &StudySeasonManifest,
    source: &SeasonManifest,
) -> Result<(), SddpError> {
    if source.cycle_code == SEASON_CYCLE_CODE_ABSENT {
        return Err(SddpError::Validation(format!(
            "boundary policy checkpoint at {} predates the season descriptor (its manifest \
             carries no season cycle or PAR orders); re-export it with a current Cobre",
            boundary_path.display()
        )));
    }

    if study.cycle_code != source.cycle_code {
        return Err(SddpError::Validation(format!(
            "boundary policy at {}: season cycle mismatch (study is {}, source is {})",
            boundary_path.display(),
            season_cycle_label(study.cycle_code),
            season_cycle_label(source.cycle_code)
        )));
    }

    if study.n_seasons != source.n_seasons {
        return Err(SddpError::Validation(format!(
            "boundary policy at {}: season count mismatch (study has {} seasons, source has {})",
            boundary_path.display(),
            study.n_seasons,
            source.n_seasons
        )));
    }

    for study_hydro in &study.hydro_orders {
        let Ok(source_idx) = source
            .hydro_orders
            .binary_search_by_key(&study_hydro.hydro_id, |h| h.hydro_id)
        else {
            return Err(SddpError::Validation(format!(
                "boundary policy at {}: hydro {} has a modeled inflow season/PAR-order entry \
                 in the current study but none in the boundary source (the boundary was fitted \
                 on a different set of inflow processes)",
                boundary_path.display(),
                study_hydro.hydro_id
            )));
        };
        let source_hydro = &source.hydro_orders[source_idx];
        if study_hydro.orders.len() != source_hydro.orders.len() {
            return Err(SddpError::Validation(format!(
                "boundary policy at {}: hydro {} carries {} PAR orders in the current study \
                 but {} in the boundary source",
                boundary_path.display(),
                study_hydro.hydro_id,
                study_hydro.orders.len(),
                source_hydro.orders.len()
            )));
        }

        if let Some((season, study_order, source_order)) = study_hydro
            .orders
            .iter()
            .zip(&source_hydro.orders)
            .enumerate()
            .find_map(|(season, (study, source))| {
                study
                    .filter(|order| order != source)
                    .map(|order| (season, order, *source))
            })
        {
            return Err(SddpError::Validation(format!(
                "boundary policy at {}: hydro {} PAR order mismatch at season {season} (study \
                 has order {study_order}, source has order {source_order})",
                boundary_path.display(),
                study_hydro.hydro_id
            )));
        }
    }

    Ok(())
}

/// Rejects a boundary load whose current terminal manifest declares a
/// `HydroStorage` or `HydroInflowLag` slot with no same-identity source
/// counterpart — the state's must-correspond core (see `reconcile`'s module
/// doc). Checked up front, over the whole `current` manifest at once, rather
/// than the first miss [`build_rebind`] would otherwise report: storage
/// first, since a missing reservoir is the louder "different deck" signal and
/// a deck missing a plant is usually missing its lag block too. A source
/// slot with no `current` counterpart is a superset drop, not examined here.
///
/// `identity_index` is [`super::reconcile::build_identity_index`]'s output over the
/// source manifest, built once by the caller and shared with
/// [`build_rebind`] so the source is hashed once, not per consumer; only
/// probed, never iterated. The missing lists come from a positional walk of
/// `current`, so their order matches `current`'s own declaration order.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming every `current` storage slot
/// (respectively every `current` inflow-lag slot) with no source
/// counterpart, one message per family.
fn check_topology_subset(
    current: &[EntitySlot],
    identity_index: &HashMap<SlotKey, usize>,
) -> Result<(), SddpError> {
    let mut missing_storage: Vec<i32> = Vec::new();
    let mut missing_inflow_lag: Vec<(i32, u32)> = Vec::new();
    for slot in current {
        match slot.family() {
            Some(StateFamily::HydroStorage) => {
                if !identity_index.contains_key(&slot_identity(slot)) {
                    missing_storage.push(slot.entity_id);
                }
            }
            Some(StateFamily::HydroInflowLag) => {
                if !identity_index.contains_key(&slot_identity(slot)) {
                    missing_inflow_lag.push((slot.entity_id, slot.subindex));
                }
            }
            _ => {}
        }
    }

    if !missing_storage.is_empty() {
        let names = missing_storage
            .iter()
            .map(|id| format!("hydro {id}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(SddpError::Validation(format!(
            "boundary policy does not price {names}; it was trained on a different set of plants"
        )));
    }

    if !missing_inflow_lag.is_empty() {
        let names = missing_inflow_lag
            .iter()
            .map(|(id, depth)| format!("hydro {id} at lag depth {depth}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(SddpError::Validation(format!(
            "boundary policy has no inflow-lag coefficient for {names}: the boundary is \
             lag-depth-incompatible with the current study"
        )));
    }

    Ok(())
}

/// Load boundary cuts from the pool of a source Cobre policy checkpoint that
/// prices the state at the study's boundary date.
///
/// Pool selection ([`select_boundary_pool`]) and the source cost scale are
/// read entirely from the resolved pool's own `cuts/<pool>.bin`, never from
/// `metadata.json`'s `graph_manifest`/`producer.cost_scale_factor`. A resolved
/// pool whose `cost_scale_factor` reads `None`, or which is shared by more
/// than one node (`node_id == STAGE_CUTS_NODE_ID_SENTINEL`), REJECTS: there is
/// no silent default and no `metadata.json` fallback, and a boundary source
/// must be a single-node terminal pool.
///
/// The source pool's manifest is compared against the current TERMINAL-stage
/// manifest (`current_manifest`, built via
/// [`StudySetup::build_terminal_entity_manifest`](crate::StudySetup::build_terminal_entity_manifest));
/// `num_stages` may differ. Per-slot identity is RECONCILED, never
/// exact-matched, via [`crate::policy::reconcile::build_rebind`]/`rebind_cut`.
/// An empty manifest on either side skips reconciliation, relying on
/// `state_dimension` alone. The path never warns: every outcome is either an
/// `Err` reject or `Ok` cuts plus a
/// [`ValidatedBoundaryCuts::report`] tally.
///
/// The three gates ahead of [`validate_policy_load`] are ordered so each one's
/// specific message wins over the generic `state_dimension` reject:
/// `effective_inflow_lag_depth` (a cut reaching deeper than the layout
/// reserves is a coupling regression, since the layout is itself sized from
/// this boundary via [`resolve_boundary_state_requirements`]), then
/// [`check_season_compatibility`], then [`check_topology_subset`]. The last
/// shares the manifest-verifiability gate with the intercept fold and the
/// rebind, so an unverifiable manifest defers to `state_dimension` throughout.
///
/// `check_season_compatibility` is the one study-global fact read from
/// `metadata.json` rather than a resolved pool's own `cuts/<pool>.bin`: unlike
/// `cost_scale_factor` and the graph, a season cycle is genuinely study-global
/// (one per study, not one per pool), has no per-pool counterpart to go stale
/// against, and duplicating it onto every pool would itself be the drift
/// hazard.
///
/// `fixed_windows` are the current study's fixed post-horizon anticipated
/// commitments (built via
/// [`StudySetup::build_terminal_fixed_post_horizon_windows`](crate::StudySetup::build_terminal_fixed_post_horizon_windows)).
/// [`crate::policy::reconcile::build_boundary_fold`] folds their displacement
/// value into each boundary cut's intercept on the RAW records, BEFORE
/// `rescale_cut_records_for_load`, so the folded term rides both rescale
/// transforms with the rest of the intercept; an empty fold leaves every
/// intercept bit-identical.
///
/// A superset boundary — a source pricing state this study does not model —
/// is recorded in the returned report either way; `strict` changes only
/// whether the load itself proceeds, never a coefficient, a tally, or another
/// reject.
///
/// # Errors
///
/// Returns [`SddpError::PolicyVersionMismatch`] if the resolved checkpoint's
/// recorded `cobre_version` does not exactly match [`POLICY_COBRE_VERSION`].
/// Otherwise returns [`SddpError::Validation`] if:
/// - The checkpoint cannot be read
/// - Every pool in the checkpoint carries
///   [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`] (a pre-dated checkpoint);
///   re-export with a current Cobre
/// - More than one pool is priced at the boundary date (a branching source's
///   terminal date tie); names the boundary date and every matching pool id
/// - No pool is priced at the boundary date; names the boundary date and
///   every pool's own `(pool id, priced date)`
/// - The resolved pool's `.bin` predates self-describing cuts
///   (`cost_scale_factor` reads `None`); re-export with a current Cobre
/// - The resolved pool is shared by more than one node (`node_id` reads the
///   sentinel); a boundary source must be a single-node terminal pool
/// - A cut references inflow-lag state deeper than `effective_inflow_lag_depth`
///   (a layout-sizing coupling regression)
/// - `study_seasons` is present and not absent, and the source's
///   `metadata.season_manifest` is absent, disagrees on season cycle or
///   season count, is missing a hydro the study models, or disagrees on a
///   modeled hydro's per-season PAR order (see [`check_season_compatibility`])
/// - A current-manifest storage or inflow-lag slot has no source counterpart
///   under identity (when the manifest is verifiable); names every such
///   storage hydro, or every such `(hydro, lag depth)` pair, in one message
///   per family (`check_topology_subset`)
/// - The source pool's state dimension does not match `current_state_dimension`
///   (the unverifiable-manifest fallback, once the topology gate above is
///   skipped)
/// - A live anticipated source slot's `interval_start`/`interval_end` fails to
///   decode, or decodes to a non-positive span; names the slot's identity and
///   the raw values (propagated from `build_source_interval_index`, built once
///   and shared by the intercept fold and the rebind/report path)
/// - A live forward-family target slot's `interval_start`/`interval_end` fails
///   to decode; names the slot's identity and the undecodable raw value (see
///   `reconcile::resolve_by_interval_overlap`)
/// - `strict` is set and the load's reconciliation report carries a nonzero
///   `dropped_source` tally (a superset boundary); names the boundary path,
///   the total dropped count and every dropping family
///
/// A live forward-family target slot whose interval ends at or before
/// `boundary_date` is NOT an error: it is an in-study delivery the boundary
/// does not price, resolved to `Zero` (see
/// `reconcile::resolve_by_interval_overlap`).
pub fn load_boundary_cuts(
    request: &BoundaryLoadRequest<'_>,
) -> Result<ValidatedBoundaryCuts, SddpError> {
    let BoundaryLoadRequest {
        boundary_path,
        boundary_date,
        current_state_dimension,
        current_manifest,
        loading_cost_scale_factor,
        fixed_windows,
        inflow_lag_depth: effective_inflow_lag_depth,
        study_seasons,
        strict,
    } = *request;

    let checkpoint = read_policy_checkpoint(boundary_path).map_err(|e| {
        SddpError::Validation(format!(
            "failed to read boundary policy checkpoint at {}: {e}",
            boundary_path.display()
        ))
    })?;

    let stage_result = select_boundary_pool(&checkpoint, boundary_path, boundary_date)?;

    let Some(source_cost_scale_factor) = stage_result.cost_scale_factor else {
        return Err(boundary_predates_self_describing_cuts(boundary_path));
    };

    if stage_result.node_id == STAGE_CUTS_NODE_ID_SENTINEL {
        return Err(SddpError::Validation(format!(
            "boundary policy at {}: a boundary source must be a single-node terminal pool; the \
             resolved pool is shared by multiple nodes",
            boundary_path.display()
        )));
    }

    if let Some(reserved) = effective_inflow_lag_depth {
        let depth = boundary_cut_lag_depth(&stage_result.entity_manifest);
        if depth > reserved {
            return Err(SddpError::Validation(format!(
                "internal: boundary policy pool {} references inflow-lag state to depth \
                 {depth}, but the resolved state layout reserves only {reserved} inflow-lag \
                 slots — resolve_boundary_state_requirements must infer the depth from the \
                 boundary policy before the layout is built; reaching here means that coupling \
                 broke and cut coefficients would truncate",
                stage_result.stage_id
            )));
        }
    }

    // A study declaring no season map has no PAR context to compare and, having
    // no inflow models, no lag slots to protect.
    if let Some(study_seasons) = study_seasons
        && study_seasons.cycle_code != SEASON_CYCLE_CODE_ABSENT
    {
        check_season_compatibility(
            boundary_path,
            study_seasons,
            &checkpoint.metadata.season_manifest,
        )?;
    }

    // Hoisted above `validate_policy_load` so the topology gate below, the
    // intercept fold and the rebind all share one emptiness decision.
    let verifiable = manifest_identity_verifiable(&stage_result.entity_manifest, current_manifest);
    // Shared with `build_rebind` below via `identity_index`, so the two
    // consumers of `stage_result.entity_manifest`'s identity hash it once.
    let identity_index = verifiable.then(|| build_identity_index(&stage_result.entity_manifest));
    if let Some(identity_index) = &identity_index {
        check_topology_subset(current_manifest, identity_index)?;
    }

    // `BoundaryInjection` checks neither `num_stages`, `n_pools`, nor the graph
    // manifest, so those fields carry placeholders.
    let empty_graph = GraphManifest::default();
    let source = PolicyStageManifest {
        state_dimension: stage_result.state_dimension,
        num_stages: 0,
        n_pools: 0,
        slots: &stage_result.entity_manifest,
        graph: &empty_graph,
    };
    let current = PolicyStageManifest {
        state_dimension: current_state_dimension,
        num_stages: 0,
        n_pools: 0,
        slots: current_manifest,
        graph: &empty_graph,
    };
    let proof = validate_policy_load::<BoundaryInjection>(
        &checkpoint.metadata.cobre_version,
        &source,
        &current,
    )?;
    debug_assert!(
        proof.warnings.is_empty(),
        "BoundaryInjection sets every CHECK_* flag false, so validate_policy_load never \
         populates proof.warnings on this path: {:?}",
        proof.warnings
    );

    let mut records = stage_result.cuts.clone();

    // Guard only the unverifiable (empty-manifest) branch: a verifiable manifest
    // defers to per-slot reconciliation, which tolerates a differing source/current
    // dimension — hoisting this to an unconditional check would reject that load.
    if !verifiable && stage_result.state_dimension != current_state_dimension {
        return Err(SddpError::Validation(format!(
            "boundary policy state_dimension mismatch: policy has {}, current system has {} (a \
             lag-state depth mismatch is a common cause); an absent entity manifest cannot be \
             reconciled per-slot, so a differing state dimension cannot be resolved",
            stage_result.state_dimension, current_state_dimension
        )));
    }

    // Built here (the fold's first use) and reused below for the rebind/report
    // pair, so `stage_result.entity_manifest`'s interval index is hashed once.
    let source_index = if verifiable {
        let source_index = build_source_interval_index(&stage_result.entity_manifest)?;
        // The fold reads SOURCE coefficients by source_pos, so it MUST run before
        // rebind_cut replaces record.coefficients with the target-aligned vector.
        let fold = build_boundary_fold(&stage_result.entity_manifest, fixed_windows, &source_index);
        if !fold.is_empty() {
            for record in &mut records {
                debug_assert_eq!(
                    stage_result.entity_manifest.len(),
                    record.coefficients.len(),
                    "boundary cut coefficients are source-manifest-aligned"
                );
                let delta: f64 = fold
                    .iter()
                    .map(|&(pos, f)| record.coefficients[pos] * f)
                    .sum();
                record.intercept += delta;
            }
        }
        Some(source_index)
    } else {
        None
    };

    rescale_cut_records_for_load(
        &mut records,
        Some(source_cost_scale_factor),
        loading_cost_scale_factor,
    );

    let report = match (&identity_index, &source_index) {
        (Some(identity_index), Some(source_index)) => {
            let rebind = build_rebind(
                &stage_result.entity_manifest,
                current_manifest,
                boundary_date,
                source_index,
                identity_index,
            )?;
            for record in &mut records {
                record.coefficients = rebind_cut(record, &rebind);
            }
            build_reconciliation_report(
                &stage_result.entity_manifest,
                current_manifest,
                &rebind,
                source_index,
            )
        }
        _ => BoundaryReconciliationReport::default(),
    };

    if strict && let Some(summary) = report.superset_summary() {
        return Err(SddpError::Validation(format!(
            "boundary policy at {} is a superset: {summary}; see the reconciliation report \
             or set policy.boundary.strict = false",
            boundary_path.display()
        )));
    }

    Ok(ValidatedBoundaryCuts { records, report })
}

/// Boundary cut records validated by [`load_boundary_cuts`] on the reading
/// rank, or reconstructed by [`Self::from_broadcast_records`] on a peer rank
/// that received them over MPI from a rank that ran it. The private fields
/// mean only these two named constructors can produce one, so
/// [`inject_boundary_cuts`] cannot compile against a bare, unvalidated
/// `Vec<OwnedPolicyCutRecord>`. Derefs to `[OwnedPolicyCutRecord]` for read
/// access; carries the load's [`BoundaryReconciliationReport`], read via
/// [`Self::report`].
#[derive(Debug, Clone)]
pub struct ValidatedBoundaryCuts {
    records: Vec<OwnedPolicyCutRecord>,
    report: BoundaryReconciliationReport,
}

impl ValidatedBoundaryCuts {
    /// The reconciliation report [`load_boundary_cuts`] built for this load.
    #[must_use]
    pub fn report(&self) -> &BoundaryReconciliationReport {
        &self.report
    }

    /// Reconstruct a validated set from records a peer rank produced with
    /// [`load_boundary_cuts`] and broadcast over MPI. The single-disk-reader rank
    /// reads and reconciles the source once; every rank must then inject the
    /// identical terminal pool via [`inject_boundary_cuts`], or a non-root rank's
    /// terminal pool stays empty and its forward/backward/simulation terminal
    /// solves drop the post-horizon value-to-go — a rank-count-dependent wrong
    /// bound. The `records` are already validated upstream, so no re-check runs;
    /// the report is not carried across the wire (only the reading rank prints it).
    #[must_use]
    pub fn from_broadcast_records(records: Vec<OwnedPolicyCutRecord>) -> Self {
        Self {
            records,
            report: BoundaryReconciliationReport::default(),
        }
    }
}

impl Deref for ValidatedBoundaryCuts {
    type Target = [OwnedPolicyCutRecord];

    fn deref(&self) -> &Self::Target {
        &self.records
    }
}

/// Inject boundary cuts into the terminal stage of the study's FCF.
///
/// Replaces the terminal stage's [`CutPool`] with one fixed to exactly
/// `boundary_cuts.len()` slots — a leaf never receives a new cut, so no
/// growable training capacity is reserved. The resulting nonzero
/// `warm_start_count` is what makes the forward pass treat the terminal
/// stage as boundary-loaded ([`CutPool::has_warm_start_cuts`]) and skip
/// theta zeroing.
///
/// `boundary_cuts` must come from [`load_boundary_cuts`] or
/// [`ValidatedBoundaryCuts::from_broadcast_records`] — its private fields
/// mean a bare `Vec<OwnedPolicyCutRecord>`/slice cannot substitute, so an
/// unvalidated boundary load cannot compile.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if the study has no terminal pool.
///
/// ```compile_fail
/// use cobre_sddp::{StudySetup, inject_boundary_cuts};
///
/// fn call_with_bare_records(
///     setup: &mut StudySetup,
///     records: &[cobre_io::OwnedPolicyCutRecord],
/// ) {
///     inject_boundary_cuts(setup, records); // bare records, not ValidatedBoundaryCuts
/// }
/// ```
pub fn inject_boundary_cuts(
    setup: &mut StudySetup,
    boundary_cuts: &ValidatedBoundaryCuts,
) -> Result<(), SddpError> {
    let terminal_idx = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .ok_or_else(|| {
            SddpError::Validation(
                "boundary cuts need a terminal pool, but the study has no terminal stage"
                    .to_string(),
            )
        })?;
    let fcf = &mut setup.fcf;
    let state_dimension = fcf.state_dimension;
    let forward_passes = fcf.forward_passes;
    fcf.pools[terminal_idx] =
        CutPool::new_with_warm_start(state_dimension, forward_passes, 0, boundary_cuts);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{AnticipatedCommitmentHistory, EntityId};
    use cobre_io::{
        EntitySlot, GraphManifest, HydroSeasonOrders, ProducerBlock, SEASON_CYCLE_CODE_MONTHLY,
        SEASON_CYCLE_CODE_WEEKLY, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, SeasonManifest,
        StageCutsPayload, decode_slot_date, encode_slot_date, read_policy_checkpoint,
    };

    use super::{
        BoundaryInjection, BoundaryLoadRequest, BoundaryReconciliationReport,
        BoundaryStateRequirements, CutPool, FullFcf, NodeId, NodePos, POLICY_COBRE_VERSION,
        PolicyStageManifest, TypedVec, ValidatedBoundaryCuts, boundary_policy_required_lag_depth,
        check_season_compatibility, compare_manifest_slot_identity, inject_boundary_cuts,
        load_boundary_cuts, validate_policy_load,
    };
    use crate::SddpError;
    use crate::policy::orchestration::{StudyHydroSeasonOrders, StudySeasonManifest};
    use crate::policy::reconcile::overlap_hours;
    use crate::test_support;

    // ── helpers ───────────────────────────────────────────────────────────────

    use crate::test_support::{
        anticipated_slot_at, chain_graph_manifest, inflow_lag_slot, next_month_anchor,
        storage_slot, transit_bucket_slot, transit_bucket_slot_over, ymd,
    };

    /// Pool `pool`'s fixture `priced_state_date`: `2030-01-01` plus `pool`
    /// months, so every pool in a multi-pool checkpoint written by this
    /// module's fixtures carries a distinct, non-sentinel date.
    fn fixture_priced_date(pool: u32) -> NaiveDate {
        test_support::fixture_priced_date(ymd(2030, 1, 1), pool)
    }

    /// Discard warnings: a `&mut dyn FnMut(&str)` for tests asserting only the
    /// `Result`.
    fn ignore_warnings() -> impl FnMut(&str) {
        |_| {}
    }

    /// `loading_cost_scale_factor` for tests that assert raw coefficient/
    /// intercept VALUES and don't care about cost-scale semantics: every
    /// resolved pool is marked (`cost_scale_factor: Some(_)`), and the marked
    /// rescale branch always divides by `loading_cost_scale_factor`
    /// (never a no-op, even at a matching source factor), so `1.0` is the one
    /// value that keeps a written raw value equal to its loaded value.
    const NEUTRAL_LOADING_FACTOR: f64 = 1.0;

    /// A minimal producer block for artifact-writing test helpers.
    fn producer_block() -> ProducerBlock {
        ProducerBlock {
            completed_iterations: 10,
            max_iterations: 50,
            forward_passes: 1,
            ..test_support::producer_block()
        }
    }

    /// A shared empty graph manifest for `validate_policy_load` unit tests that
    /// exercise only the `state_dimension`/`num_stages`/slot-identity checks —
    /// an empty graph makes the graph-identity check a silent no-op.
    static EMPTY_GRAPH: std::sync::LazyLock<GraphManifest> =
        std::sync::LazyLock::new(GraphManifest::default);

    /// Build a [`PolicyStageManifest`] with `n_pools == num_stages` and the shared
    /// [`EMPTY_GRAPH`] (graph identity skipped) — for the pure validate tests.
    fn psm(state_dimension: u32, num_stages: u32, slots: &[EntitySlot]) -> PolicyStageManifest<'_> {
        PolicyStageManifest {
            state_dimension,
            num_stages,
            n_pools: num_stages,
            slots,
            graph: &EMPTY_GRAPH,
        }
    }

    /// Write a minimal policy checkpoint to `dir` with `n_stages` stages each
    /// having `state_dimension` state variables and the supplied cut intercepts,
    /// with no entity manifest (the pre-manifest checkpoint shape).
    ///
    /// Each stage gets `cuts.len()` cuts with coefficients all set to 1.0.
    fn write_minimal_checkpoint(
        dir: &std::path::Path,
        n_stages: u32,
        state_dimension: u32,
        cut_intercepts: &[f64],
    ) {
        write_checkpoint_with_manifest(dir, n_stages, state_dimension, cut_intercepts, &[]);
    }

    /// Like [`write_minimal_checkpoint`] but attaches `manifest` to every stage's
    /// cuts payload (an empty `manifest` reproduces the pre-manifest shape).
    fn write_checkpoint_with_manifest(
        dir: &std::path::Path,
        n_stages: u32,
        state_dimension: u32,
        cut_intercepts: &[f64],
        manifest: &[EntitySlot],
    ) {
        let metadata = test_support::checkpoint_metadata(
            n_stages,
            chain_graph_manifest(n_stages),
            producer_block(),
        );
        write_checkpoint_with_manifest_metadata(
            dir,
            n_stages,
            state_dimension,
            cut_intercepts,
            manifest,
            &metadata,
        );
    }

    /// Like [`write_checkpoint_with_manifest`] but stamps `season_manifest` on
    /// the checkpoint metadata instead of the absent default — for the
    /// boundary-load season/PAR-identity gate tests.
    fn write_checkpoint_with_manifest_and_seasons(
        dir: &std::path::Path,
        n_stages: u32,
        state_dimension: u32,
        cut_intercepts: &[f64],
        manifest: &[EntitySlot],
        season_manifest: cobre_io::SeasonManifest,
    ) {
        let metadata = cobre_io::CheckpointManifest {
            season_manifest,
            ..test_support::checkpoint_metadata(
                n_stages,
                chain_graph_manifest(n_stages),
                producer_block(),
            )
        };
        write_checkpoint_with_manifest_metadata(
            dir,
            n_stages,
            state_dimension,
            cut_intercepts,
            manifest,
            &metadata,
        );
    }

    /// Shared body for [`write_checkpoint_with_manifest`] and
    /// [`write_checkpoint_with_manifest_and_seasons`], under the
    /// caller-supplied `metadata`.
    fn write_checkpoint_with_manifest_metadata(
        dir: &std::path::Path,
        n_stages: u32,
        state_dimension: u32,
        cut_intercepts: &[f64],
        manifest: &[EntitySlot],
        metadata: &cobre_io::CheckpointManifest,
    ) {
        let state_dim = state_dimension as usize;
        let coefficients = vec![1.0_f64; state_dim];
        let n_cuts = cut_intercepts.len();

        let cut_records: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = (0..n_stages)
            .map(|_| {
                cut_intercepts
                    .iter()
                    .enumerate()
                    .map(|(i, &intercept)| cobre_io::PolicyCutRecord {
                        cut_id: i as u64,
                        slot_index: i as u32,
                        iteration: i as u32,
                        forward_pass_index: 0,
                        intercept,
                        coefficients: &coefficients,
                        is_active: true,
                    })
                    .collect()
            })
            .collect();

        let active_indices: Vec<Vec<u32>> = (0..n_stages)
            .map(|_| (0..n_cuts as u32).collect())
            .collect();

        let payloads: Vec<StageCutsPayload<'_>> = (0..n_stages as usize)
            .map(|s| StageCutsPayload {
                stage_id: s as u32,
                state_dimension,
                capacity: n_cuts as u32,
                warm_start_count: 0,
                cuts: &cut_records[s],
                active_cut_indices: &active_indices[s],
                populated_count: n_cuts as u32,
                entity_manifest: manifest,
                cost_scale_factor: 1_000_000.0,
                node_id: s as i32,
                graph_stage_id: -1,
                priced_state_date: encode_slot_date(fixture_priced_date(s as u32)),
            })
            .collect();

        cobre_io::write_policy_checkpoint(dir, &payloads, &[], metadata, &[]).unwrap();
    }

    /// Every pool [`write_checkpoint_with_manifest`] writes carries a
    /// non-sentinel `priced_state_date`, and no two pools share one.
    #[test]
    fn write_checkpoint_with_manifest_pools_carry_distinct_priced_dates() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &[]);

        let checkpoint = read_policy_checkpoint(tmp.path()).unwrap();
        let dates: Vec<i32> = checkpoint
            .stage_cuts
            .iter()
            .map(|sr| sr.priced_state_date)
            .collect();
        assert!(
            dates
                .iter()
                .all(|&d| d != STAGE_CUTS_PRICED_STATE_DATE_SENTINEL)
        );
        let mut sorted = dates.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), dates.len());
    }

    /// Write a single-stage checkpoint whose one cut has the given at-rest
    /// `intercept`/`coefficients` (written byte-for-byte, no transform applied
    /// here), for [`load_boundary_cuts`] round-trip tests across differing
    /// loading factors. The pool's own `.bin`-level `cost_scale_factor` is
    /// always marked (`Some`) — `metadata.producer.cost_scale_factor` is not
    /// read by the boundary path and is left at [`producer_block`]'s default.
    fn write_checkpoint_with_scale(
        dir: &std::path::Path,
        stage_id: u32,
        intercept: f64,
        coefficients: &[f64],
    ) {
        let state_dimension = coefficients.len() as u32;
        let cut = cobre_io::PolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active: true,
        };
        let cuts = vec![cut];
        let payload = StageCutsPayload {
            stage_id,
            state_dimension,
            capacity: 1,
            warm_start_count: 0,
            cuts: &cuts,
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: &[],
            cost_scale_factor: 1_000_000.0,
            node_id: i32::try_from(stage_id).unwrap_or(-1),
            graph_stage_id: -1,
            priced_state_date: encode_slot_date(fixture_priced_date(stage_id)),
        };
        let metadata = test_support::checkpoint_metadata(
            stage_id + 1,
            chain_graph_manifest(stage_id + 1),
            producer_block(),
        );
        cobre_io::write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).unwrap();
    }

    /// Behavioral: [`load_boundary_cuts`] on a MARKED checkpoint (canonical
    /// currency units at rest) loaded at a series of differing
    /// `loading_cost_scale_factor` values recovers `at_rest / loading_factor`
    /// for every value, matching [`rescale_cut_records_for_load`]'s contract at
    /// the file-I/O boundary — not just as a pure-function unit test.
    #[test]
    fn load_boundary_cuts_across_differing_loading_factors() {
        let at_rest_intercept = 1_234_000.0;
        let at_rest_coefficients = [10_000.0, -25_000.0];

        for loading_factor in [500_000.0, 1_000_000.0, 2_500_000.0, 1e10] {
            let tmp = tempfile::tempdir().unwrap();
            write_checkpoint_with_scale(tmp.path(), 0, at_rest_intercept, &at_rest_coefficients);

            let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                2,
                &[],
                loading_factor,
            ))
            .unwrap();

            assert_eq!(cuts.len(), 1);
            let expected_intercept = at_rest_intercept / loading_factor;
            assert!(
                (cuts[0].intercept - expected_intercept).abs()
                    < expected_intercept.abs().max(1.0) * 1e-9,
                "loading_factor={loading_factor}: intercept {} != expected {expected_intercept}",
                cuts[0].intercept
            );
            for (c, &at_rest) in cuts[0].coefficients.iter().zip(&at_rest_coefficients) {
                let expected = at_rest / loading_factor;
                assert!(
                    (c - expected).abs() < expected.abs().max(1.0) * 1e-9,
                    "loading_factor={loading_factor}: coefficient {c} != expected {expected}"
                );
            }
        }
    }

    // ── rescale_cut_records_for_load unit tests ──────────────────────────────

    use super::{LEGACY_COST_SCALE_FACTOR, rescale_cut_records_for_load};
    use crate::policy_export::scale_cut_records_for_export;
    use cobre_io::OwnedPolicyCutRecord;

    fn owned_cut(intercept: f64, coefficients: Vec<f64>) -> OwnedPolicyCutRecord {
        OwnedPolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active: true,
        }
    }

    /// A legacy checkpoint (`source_cost_scale_factor: None`) loaded at the
    /// still-default [`LEGACY_COST_SCALE_FACTOR`] is a bit-exact no-op — the
    /// requirement that a legacy policy at the default factor never
    /// re-baselines.
    #[test]
    fn legacy_no_marker_at_default_factor_is_bit_exact_noop() {
        let mut records = vec![owned_cut(42.5, vec![1.0, -2.5, 3.75])];
        let original = records.clone();

        rescale_cut_records_for_load(&mut records, None, LEGACY_COST_SCALE_FACTOR);

        assert_eq!(
            records[0].intercept.to_bits(),
            original[0].intercept.to_bits()
        );
        for (a, b) in records[0]
            .coefficients
            .iter()
            .zip(&original[0].coefficients)
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "legacy default-factor load must be bit-exact"
            );
        }
    }

    /// A legacy checkpoint loaded at a NON-default factor is interpreted as
    /// scaled-at-[`LEGACY_COST_SCALE_FACTOR`] and rescaled by
    /// `LEGACY_COST_SCALE_FACTOR / loading_cost_scale_factor`.
    #[test]
    fn legacy_no_marker_at_non_default_factor_rescales_by_ratio() {
        let mut records = vec![owned_cut(10.0, vec![2.0, 4.0])];
        let loading_factor = 2_000_000.0;

        rescale_cut_records_for_load(&mut records, None, loading_factor);

        let ratio = LEGACY_COST_SCALE_FACTOR / loading_factor;
        assert!((records[0].intercept - 10.0 * ratio).abs() < 1e-9);
        assert!((records[0].coefficients[0] - 2.0 * ratio).abs() < 1e-9);
        assert!((records[0].coefficients[1] - 4.0 * ratio).abs() < 1e-9);
    }

    /// A marked checkpoint (`Some(s)`) is ALWAYS divided by
    /// `loading_cost_scale_factor` — even when `s` equals the loading factor —
    /// never special-cased to a no-op. The `source_cost_scale_factor` VALUE is
    /// irrelevant once the file holds canonical currency units; only its
    /// presence (marked vs. legacy) selects the code path.
    #[test]
    fn marked_checkpoint_always_divides_regardless_of_source_value() {
        let loading_factor = 1_000_000.0;
        let mut with_matching_source = vec![owned_cut(100.0, vec![50.0])];
        let mut with_different_source = vec![owned_cut(100.0, vec![50.0])];

        rescale_cut_records_for_load(
            &mut with_matching_source,
            Some(loading_factor),
            loading_factor,
        );
        rescale_cut_records_for_load(&mut with_different_source, Some(42.0), loading_factor);

        assert_eq!(
            with_matching_source[0].intercept.to_bits(),
            with_different_source[0].intercept.to_bits(),
            "the source factor's VALUE must not affect the loaded result"
        );
        assert!((with_matching_source[0].intercept - 100.0 / loading_factor).abs() < 1e-12);
    }

    /// Export/load transform property: export (multiply by `S`) then load at
    /// the SAME factor (divide by `S`) recovers the original value within 1
    /// ULP per value — the accepted same-factor round-trip drift (1e6 is not a
    /// power of two, so two roundings do not cancel exactly).
    #[test]
    fn export_then_load_same_factor_round_trips_within_one_ulp() {
        let cost_scale_factor = 1_000_000.0;
        let originals = [vec![1.0_f64, -3.5, 1e-6, 123_456.789]];
        let intercepts = [7.25_f64];

        let internal_records: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = vec![
            originals
                .iter()
                .zip(&intercepts)
                .map(|(coeffs, &intercept)| cobre_io::PolicyCutRecord {
                    cut_id: 0,
                    slot_index: 0,
                    iteration: 0,
                    forward_pass_index: 0,
                    intercept,
                    coefficients: coeffs,
                    is_active: true,
                })
                .collect(),
        ];

        let exported = scale_cut_records_for_export(&internal_records, cost_scale_factor);
        let mut round_tripped = exported[0].clone();
        rescale_cut_records_for_load(
            &mut round_tripped,
            Some(cost_scale_factor),
            cost_scale_factor,
        );

        let original_intercept = intercepts[0];
        let ulp_intercept = (round_tripped[0].intercept - original_intercept).abs();
        assert!(
            ulp_intercept <= original_intercept.abs() * f64::EPSILON * 4.0,
            "intercept round-trip drift {ulp_intercept} exceeds a few ULP of {original_intercept}"
        );
        for (rt, orig) in round_tripped[0].coefficients.iter().zip(&originals[0]) {
            let drift = (rt - orig).abs();
            let tol = (orig.abs().max(1.0)) * f64::EPSILON * 4.0;
            assert!(
                drift <= tol,
                "coefficient round-trip drift {drift} exceeds tolerance {tol} for original {orig}"
            );
        }
    }

    /// Export/load transform property: cross-factor linearity — exporting at
    /// `S_train` then loading at `S_prime` recovers `original * (S_train /
    /// S_prime)` (the net two-rounding transform), for `S_prime != S_train`.
    #[test]
    fn export_then_load_cross_factor_is_linear() {
        let s_train = 1_000_000.0;
        let s_prime = 4_000_000.0;
        let original = [vec![2.0_f64, -0.5]];
        let intercept = 9.0_f64;

        let internal_records: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> =
            vec![vec![cobre_io::PolicyCutRecord {
                cut_id: 0,
                slot_index: 0,
                iteration: 0,
                forward_pass_index: 0,
                intercept,
                coefficients: &original[0],
                is_active: true,
            }]];

        let exported = scale_cut_records_for_export(&internal_records, s_train);
        let mut loaded = exported[0].clone();
        rescale_cut_records_for_load(&mut loaded, Some(s_train), s_prime);

        let ratio = s_train / s_prime;
        assert!((loaded[0].intercept - intercept * ratio).abs() < 1e-9);
        for (l, o) in loaded[0].coefficients.iter().zip(&original[0]) {
            assert!((l - o * ratio).abs() < 1e-9);
        }
    }

    // ── checkpoint_terminal_cost_scale_factor unit tests ─────────────────────

    use super::checkpoint_terminal_cost_scale_factor;

    /// [`checkpoint_terminal_cost_scale_factor`] reads the terminal pool's own
    /// `cost_scale_factor` from the on-disk `.bin`, never
    /// `metadata.producer.cost_scale_factor` — `write_checkpoint_with_scale`
    /// marks only the pool, leaving `producer_block`'s own factor at `None`, so
    /// a correct read proves the source.
    #[test]
    fn full_fcf_source_cost_scale_reads_terminal_pool_from_bin() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_scale(tmp.path(), 0, 1.0, &[1.0]);
        let checkpoint = cobre_io::read_policy_checkpoint(tmp.path()).unwrap();

        let resolved = checkpoint_terminal_cost_scale_factor(&checkpoint).unwrap();

        assert_eq!(resolved, 1_000_000.0);
    }

    /// A checkpoint whose terminal pool's `cost_scale_factor` reads `None` (a
    /// pre-self-describing `.bin`) rejects — the [`FullFcf`] mirror of
    /// [`load_boundary_cuts`]'s own clean-break reject.
    #[test]
    fn full_fcf_source_cost_scale_rejects_pre_self_describing() {
        let checkpoint = cobre_io::PolicyCheckpoint {
            metadata: test_support::checkpoint_metadata(
                1,
                GraphManifest::default(),
                producer_block(),
            ),
            stage_cuts: vec![pool_cuts(0, &[])],
            stage_bases: Vec::new(),
            stage_states: Vec::new(),
        };

        let msg = checkpoint_terminal_cost_scale_factor(&checkpoint)
            .unwrap_err()
            .to_string();

        assert!(msg.contains("predates self-describing cuts"), "{msg}");
        assert!(msg.contains("re-export it with a current Cobre"), "{msg}");
    }

    // ── load_boundary_cuts tests ──────────────────────────────────────────────

    /// Given a valid checkpoint with 12 stages and `state_dimension=10`, when
    /// `load_boundary_cuts` is called for stage 2 with matching dimension,
    /// then it returns `Ok` with the cuts from that stage.
    #[test]
    fn load_boundary_cuts_valid_stage() {
        let tmp = tempfile::tempdir().unwrap();
        let intercepts = vec![10.0, 20.0, 30.0];
        write_minimal_checkpoint(tmp.path(), 12, 10, &intercepts);

        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(2),
            10,
            &[],
            NEUTRAL_LOADING_FACTOR,
        ))
        .unwrap();

        assert_eq!(cuts.len(), 3, "should return all 3 cuts from stage 2");
        let returned_intercepts: Vec<f64> = cuts.iter().map(|c| c.intercept).collect();
        assert_eq!(
            returned_intercepts, intercepts,
            "intercepts should match written values"
        );
        for cut in cuts.iter() {
            assert_eq!(
                cut.coefficients.len(),
                10,
                "each cut should have state_dimension=10 coefficients"
            );
        }
    }

    /// Given a checkpoint whose pools are dated at `fixture_priced_date(0..5)`,
    /// when `load_boundary_cuts` is called with a boundary date matching none
    /// of them, then it returns `Err(SddpError::Validation)` naming that date.
    #[test]
    fn load_boundary_cuts_missing_date_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);
        let unmatched_date = fixture_priced_date(99);

        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            unmatched_date,
            10,
            &[],
            1_000_000.0,
        ));

        assert!(
            result.is_err(),
            "should fail for an unmatched boundary date"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains(&unmatched_date.to_string()),
            "error should name the unmatched boundary date: {msg}"
        );
        assert!(
            msg.contains("no pool is priced"),
            "error should describe the selection failure: {msg}"
        );
    }

    /// Given a checkpoint with `state_dimension=10`, when `load_boundary_cuts` is
    /// called with `current_state_dimension=5`, then it returns
    /// `Err(SddpError::Validation)` with a message containing `"state_dimension"`.
    #[test]
    fn load_boundary_cuts_state_dimension_mismatch_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);

        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            5,
            &[],
            1_000_000.0,
        ));

        assert!(result.is_err(), "should fail for dimension mismatch");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "error should mention 'state_dimension': {msg}"
        );
    }

    /// Given a non-existent path, when `load_boundary_cuts` is called, then it
    /// returns `Err(SddpError::Validation)` with a message describing the failure.
    #[test]
    fn load_boundary_cuts_nonexistent_path_returns_error() {
        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            std::path::Path::new("/nonexistent/path/to/policy"),
            fixture_priced_date(0),
            10,
            &[],
            1_000_000.0,
        ));

        assert!(result.is_err(), "should fail for non-existent path");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed to read boundary policy checkpoint"),
            "error should describe the IO failure: {msg}"
        );
    }

    /// Build a 2-slot storage manifest with the given hydro ids, both active.
    fn storage_manifest(id0: i32, id1: i32) -> Vec<EntitySlot> {
        vec![storage_slot(id0), storage_slot(id1)]
    }

    /// A boundary cut at depth 12 reaching `load_boundary_cuts` with a reserved
    /// depth of only 6 is a layout-sizing coupling regression (the effective depth
    /// should already have been inferred to fit the policy). The defensive guard
    /// rejects before the manifest checks, naming both depths and the inference
    /// entry point — never a user-facing "raise the config" instruction, since the
    /// depth is auto-inferred from the boundary policy.
    #[test]
    fn load_boundary_cuts_lag_depth_exceeds_reserved_rejects_as_coupling_regression() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, 1_000_000.0)
                .with_inflow_lag_depth(Some(6)),
        );

        assert!(
            result.is_err(),
            "a boundary cut deeper than the reserved inflow-lag depth must reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("12"),
            "must name the boundary-cut depth 12: {msg}"
        );
        assert!(msg.contains('6'), "must name the reserved depth 6: {msg}");
        assert!(
            msg.contains("resolve_boundary_state_requirements"),
            "must name the inference entry point: {msg}"
        );
        assert!(
            msg.contains("internal"),
            "must frame the shortfall as an internal coupling regression: {msg}"
        );
    }

    /// A boundary cut at depth 12 loaded against `inflow_lag_depth = 12` clears the
    /// depth gate; a slot-for-slot matching manifest then loads.
    #[test]
    fn load_boundary_cuts_lag_depth_within_declared_loads() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        let cuts = load_boundary_cuts(
            &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, 1_000_000.0)
                .with_inflow_lag_depth(Some(12)),
        )
        .unwrap();

        assert_eq!(
            cuts.len(),
            2,
            "a boundary within the declared depth must load"
        );
    }

    /// A [`SeasonManifest`] literal for the season/PAR-identity gate tests'
    /// SOURCE side (the checkpoint's own dense, zero-filled descriptor).
    fn seasons(
        cycle_code: u8,
        n_seasons: u32,
        hydro_orders: Vec<HydroSeasonOrders>,
    ) -> SeasonManifest {
        SeasonManifest {
            cycle_code,
            n_seasons,
            hydro_orders,
        }
    }

    /// A [`StudySeasonManifest`] literal for the season/PAR-identity gate
    /// tests' STUDY side, whose per-season entries can be absent.
    fn study_seasons(
        cycle_code: u8,
        n_seasons: u32,
        hydro_orders: Vec<StudyHydroSeasonOrders>,
    ) -> StudySeasonManifest {
        StudySeasonManifest {
            cycle_code,
            n_seasons,
            hydro_orders,
        }
    }

    /// Given a study descriptor declaring `Monthly`/12 seasons and a source
    /// checkpoint whose manifest declares `Weekly`/52, `load_boundary_cuts`
    /// rejects naming both cycles by word — and the cycle check fires before
    /// the season-count check, so the message never mentions "season count".
    #[test]
    fn boundary_load_rejects_differing_season_cycle() {
        let tmp = tempfile::tempdir().unwrap();
        let source_seasons = seasons(SEASON_CYCLE_CODE_WEEKLY, 52, vec![]);
        write_checkpoint_with_manifest_and_seasons(tmp.path(), 1, 1, &[10.0], &[], source_seasons);

        let study = study_seasons(SEASON_CYCLE_CODE_MONTHLY, 12, vec![]);
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                1,
                &[],
                NEUTRAL_LOADING_FACTOR,
            )
            .with_study_seasons(&study),
        );

        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("monthly"), "must name the study cycle: {msg}");
        assert!(msg.contains("weekly"), "must name the source cycle: {msg}");
        assert!(
            !msg.contains("season count"),
            "the cycle check must fire before the season-count check: {msg}"
        );
    }

    /// Given descriptors that agree on cycle but disagree on season count,
    /// `load_boundary_cuts` rejects naming both counts.
    #[test]
    fn boundary_load_rejects_differing_season_count() {
        let tmp = tempfile::tempdir().unwrap();
        let source_seasons = seasons(SEASON_CYCLE_CODE_MONTHLY, 4, vec![]);
        write_checkpoint_with_manifest_and_seasons(tmp.path(), 1, 1, &[10.0], &[], source_seasons);

        let study = study_seasons(SEASON_CYCLE_CODE_MONTHLY, 12, vec![]);
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                1,
                &[],
                NEUTRAL_LOADING_FACTOR,
            )
            .with_study_seasons(&study),
        );

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("study has 12 seasons"),
            "must name the study count: {msg}"
        );
        assert!(
            msg.contains("source has 4)"),
            "must name the source count: {msg}"
        );
    }

    /// Given a study descriptor modeling hydro 7 and a source whose
    /// `hydro_orders` omits it entirely, `load_boundary_cuts` rejects naming
    /// hydro 7.
    #[test]
    fn boundary_load_rejects_source_missing_a_modeled_hydro_par_order() {
        let tmp = tempfile::tempdir().unwrap();
        let source_seasons = seasons(SEASON_CYCLE_CODE_MONTHLY, 3, vec![]);
        write_checkpoint_with_manifest_and_seasons(tmp.path(), 1, 1, &[10.0], &[], source_seasons);

        let study = study_seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![StudyHydroSeasonOrders {
                hydro_id: 7,
                orders: vec![Some(1), Some(2), Some(3)],
            }],
        );
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                1,
                &[],
                NEUTRAL_LOADING_FACTOR,
            )
            .with_study_seasons(&study),
        );

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("hydro 7"),
            "must name the missing hydro: {msg}"
        );
    }

    /// Given hydro 7 declaring `orders` `[2, 2, 3]` on the study side, a
    /// hand-built source whose `orders` vector is shorter rejects on the
    /// length, so a truncated descriptor can never pass the positional
    /// comparison by silently comparing fewer seasons. Called directly with
    /// a hand-built [`SeasonManifest`], not through [`load_boundary_cuts`]'s
    /// checkpoint round trip: `cobre_io`'s decode-time shape validation now
    /// rejects a truncated `orders` vector before this gate ever runs, so a
    /// wire-sourced manifest can no longer reach this branch — this test
    /// pins the gate's own defense for a manifest constructed directly.
    #[test]
    fn check_season_compatibility_rejects_truncated_source_par_orders_for_a_hydro() {
        let study = study_seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![StudyHydroSeasonOrders {
                hydro_id: 7,
                orders: vec![Some(2), Some(2), Some(3)],
            }],
        );
        let source = seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![HydroSeasonOrders {
                hydro_id: 7,
                orders: vec![2, 2],
            }],
        );

        let err = check_season_compatibility(std::path::Path::new("boundary.bin"), &study, &source)
            .expect_err("a hand-built source with a truncated orders vector must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("hydro 7"), "must name the hydro: {msg}");
        assert!(
            msg.contains("3 PAR orders") && msg.contains("but 2"),
            "must name both order counts: {msg}"
        );
    }

    /// `[2, 4, 3]` on the source side, `load_boundary_cuts` rejects naming
    /// hydro 7, the first differing season ordinal, and both orders there.
    #[test]
    fn boundary_load_rejects_differing_par_order_naming_hydro_and_season() {
        let tmp = tempfile::tempdir().unwrap();
        let source_seasons = seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![HydroSeasonOrders {
                hydro_id: 7,
                orders: vec![2, 4, 3],
            }],
        );
        write_checkpoint_with_manifest_and_seasons(tmp.path(), 1, 1, &[10.0], &[], source_seasons);

        let study = study_seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![StudyHydroSeasonOrders {
                hydro_id: 7,
                orders: vec![Some(2), Some(2), Some(3)],
            }],
        );
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                1,
                &[],
                NEUTRAL_LOADING_FACTOR,
            )
            .with_study_seasons(&study),
        );

        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("hydro 7"), "must name the hydro: {msg}");
        assert!(
            msg.contains("season 1"),
            "must name the first differing season ordinal: {msg}"
        );
        assert!(
            msg.contains("order 2"),
            "must name the study's order at that season: {msg}"
        );
        assert!(
            msg.contains("order 4"),
            "must name the source's order at that season: {msg}"
        );
    }

    /// Given a study descriptor with hydro 7's season 0 fitted and seasons 1
    /// and 2 unreferenced (`orders` `[Some(2), None, None]`), against a
    /// source whose `orders` are `[2, 4, 3]`, `load_boundary_cuts` succeeds:
    /// the differing orders at the two unreferenced seasons are never
    /// compared — a season no inflow model reaches carries no opinion.
    #[test]
    fn boundary_load_accepts_a_source_par_order_at_a_season_the_study_never_references() {
        let tmp = tempfile::tempdir().unwrap();
        let source_seasons = seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![HydroSeasonOrders {
                hydro_id: 7,
                orders: vec![2, 4, 3],
            }],
        );
        write_checkpoint_with_manifest_and_seasons(tmp.path(), 1, 1, &[10.0], &[], source_seasons);

        let study = study_seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![StudyHydroSeasonOrders {
                hydro_id: 7,
                orders: vec![Some(2), None, None],
            }],
        );
        let cuts = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                1,
                &[],
                NEUTRAL_LOADING_FACTOR,
            )
            .with_study_seasons(&study),
        )
        .unwrap();

        assert_eq!(
            cuts.len(),
            1,
            "the load must succeed when the only differing seasons are unreferenced"
        );
    }

    /// Given a study descriptor with hydro 7's seasons 0 and 1 fitted and
    /// season 2 unreferenced (`orders` `[Some(2), Some(2), None]`), against a
    /// source whose `orders` are `[2, 4, 3]`, `load_boundary_cuts` still
    /// rejects at season 1 — the genuine mismatch at a referenced season is
    /// not swallowed by the unreferenced gap at season 2.
    #[test]
    fn boundary_load_rejects_differing_par_order_at_a_referenced_season_despite_unreferenced_gaps()
    {
        let tmp = tempfile::tempdir().unwrap();
        let source_seasons = seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![HydroSeasonOrders {
                hydro_id: 7,
                orders: vec![2, 4, 3],
            }],
        );
        write_checkpoint_with_manifest_and_seasons(tmp.path(), 1, 1, &[10.0], &[], source_seasons);

        let study = study_seasons(
            SEASON_CYCLE_CODE_MONTHLY,
            3,
            vec![StudyHydroSeasonOrders {
                hydro_id: 7,
                orders: vec![Some(2), Some(2), None],
            }],
        );
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                1,
                &[],
                NEUTRAL_LOADING_FACTOR,
            )
            .with_study_seasons(&study),
        );

        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("hydro 7"), "must name the hydro: {msg}");
        assert!(
            msg.contains("season 1"),
            "must name the referenced differing season ordinal: {msg}"
        );
        assert!(
            msg.contains("order 2"),
            "must name the study's order at that season: {msg}"
        );
        assert!(
            msg.contains("order 4"),
            "must name the source's order at that season: {msg}"
        );
    }

    /// Given a source whose `season_manifest` is `SeasonManifest::default()`
    /// (a pre-`id:19` checkpoint) and a present study descriptor,
    /// `load_boundary_cuts` rejects advising re-export.
    #[test]
    fn boundary_load_rejects_absent_source_season_descriptor_with_reexport_hint() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 1, 1, &[10.0], &[]);

        let study = study_seasons(SEASON_CYCLE_CODE_MONTHLY, 12, vec![]);
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                1,
                &[],
                NEUTRAL_LOADING_FACTOR,
            )
            .with_study_seasons(&study),
        );

        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("re-export"), "must advise re-export: {msg}");
    }

    /// Given a study whose descriptor is absent (no `with_study_seasons`
    /// call), `load_boundary_cuts` proceeds even against a source descriptor
    /// that would reject a present study — the gate never runs.
    #[test]
    fn boundary_load_without_a_study_season_descriptor_skips_the_gate() {
        let tmp = tempfile::tempdir().unwrap();
        let source_seasons = seasons(
            SEASON_CYCLE_CODE_WEEKLY,
            52,
            vec![HydroSeasonOrders {
                hydro_id: 7,
                orders: vec![9; 52],
            }],
        );
        write_checkpoint_with_manifest_and_seasons(tmp.path(), 1, 1, &[10.0], &[], source_seasons);

        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            1,
            &[],
            NEUTRAL_LOADING_FACTOR,
        ))
        .unwrap();

        assert_eq!(cuts.len(), 1, "the load proceeds without the season gate");
    }

    /// `boundary_policy_required_lag_depth` reads the deepest `HydroInflowLag`
    /// subindex from the checkpoint's own manifest (`12` here), and `0` when the
    /// policy carries no inflow-lag slot.
    #[test]
    fn boundary_policy_required_lag_depth_reads_deepest_lag_slot() {
        let deep = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(
            deep.path(),
            3,
            2,
            &[10.0, 20.0],
            &[storage_slot(1), inflow_lag_slot(1, 12)],
        );
        assert_eq!(boundary_policy_required_lag_depth(deep.path()).unwrap(), 12);

        let flat = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(flat.path(), 1, 1, &[10.0], &[storage_slot(1)]);
        assert_eq!(boundary_policy_required_lag_depth(flat.path()).unwrap(), 0);
    }

    /// The boundary requirements fold the depth inferred from the policy's cuts:
    /// `none()` (no boundary) reserves nothing; a loaded boundary's required depth
    /// (via `boundary_policy_required_lag_depth`) becomes the requirements' depth.
    #[test]
    fn boundary_state_requirements_fold_the_inferred_depth() {
        assert_eq!(BoundaryStateRequirements::none().inflow_lag_depth(), None);

        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(
            tmp.path(),
            1,
            2,
            &[10.0, 20.0],
            &[storage_slot(1), inflow_lag_slot(1, 12)],
        );
        let depth = boundary_policy_required_lag_depth(tmp.path()).unwrap();
        assert_eq!(
            depth, 12,
            "the depth is inferred from the policy's required lag depth"
        );
        assert_eq!(
            BoundaryStateRequirements::present(depth).inflow_lag_depth(),
            Some(12),
        );
    }

    /// Given a checkpoint whose source-stage manifest matches the current study's
    /// terminal manifest slot-for-slot, `load_boundary_cuts` returns `Ok` with the
    /// source cuts and emits no warning.
    #[test]
    fn load_boundary_cuts_matching_manifest_loads_without_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = storage_manifest(1, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = storage_manifest(1, 2);
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            1_000_000.0,
        ))
        .unwrap();

        assert_eq!(cuts.len(), 2, "matching manifest must load all cuts");
    }

    /// Given a current terminal storage slot for hydro `9`, absent from a
    /// boundary source that prices hydros `7` and `2` only,
    /// `load_boundary_cuts` reconciles by identity and rejects naming the
    /// unpriced hydro `9`.
    #[test]
    fn load_boundary_cuts_entity_id_mismatch_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = storage_manifest(7, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = storage_manifest(9, 2);
        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            1_000_000.0,
        ));

        assert!(result.is_err(), "an unpriced hydro must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains('9'),
            "error must name the unpriced current hydro 9: {msg}"
        );
    }

    /// Given a current terminal storage slot for hydro `2`, but the boundary
    /// source prices hydro `2` only as an inflow-lag slot (no storage
    /// counterpart), `load_boundary_cuts` reconciles by identity and rejects
    /// naming the unpriced storage hydro `2` — the differently-typed slot the
    /// source happens to carry at the same raw position is irrelevant under
    /// identity matching.
    #[test]
    fn load_boundary_cuts_storage_slot_absent_from_differently_typed_source_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let mut boundary = storage_manifest(1, 2);
        boundary[1].entity_type = 1; // HydroInflowLag
        boundary[1].subindex = 1;
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = storage_manifest(1, 2);
        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            1_000_000.0,
        ));

        assert!(
            result.is_err(),
            "a storage slot with no identity counterpart must reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains('2'),
            "error must name the unpriced storage hydro 2: {msg}"
        );
    }

    /// Given a current terminal manifest with storage slots for hydros 0, 3
    /// and 7, and a source manifest that prices hydro 3 only, when
    /// `load_boundary_cuts` runs, then it rejects with a single message
    /// naming both unpriced hydros and not the one the source does price.
    #[test]
    fn boundary_load_rejects_every_unpriced_hydro_in_one_message() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = vec![storage_slot(3)];
        write_checkpoint_with_manifest(tmp.path(), 1, 1, &[10.0], &boundary);

        let current = vec![storage_slot(0), storage_slot(3), storage_slot(7)];
        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            3,
            &current,
            1_000_000.0,
        ));

        assert!(result.is_err(), "two unpriced hydros must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("hydro 0"), "must name hydro 0: {msg}");
        assert!(msg.contains("hydro 7"), "must name hydro 7: {msg}");
        assert!(
            msg.contains("different set of plants"),
            "must use the storage-family wording: {msg}"
        );
        assert!(
            !msg.contains("hydro 3"),
            "must not name hydro 3, which the source does price: {msg}"
        );
    }

    /// Given a current manifest whose hydro 2 needs inflow-lag depths 1 and
    /// 2, with storage matching on both sides, and a source carrying only
    /// depth 1, when `load_boundary_cuts` runs, then it rejects naming
    /// hydro 2 and lag depth 2.
    #[test]
    fn boundary_load_rejects_missing_inflow_lag_depth_naming_hydro_and_depth() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = vec![storage_slot(2), inflow_lag_slot(2, 1)];
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[10.0], &boundary);

        let current = vec![
            storage_slot(2),
            inflow_lag_slot(2, 1),
            inflow_lag_slot(2, 2),
        ];
        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            3,
            &current,
            1_000_000.0,
        ));

        assert!(result.is_err(), "a missing lag depth must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("hydro 2"), "must name hydro 2: {msg}");
        assert!(msg.contains("lag depth 2"), "must name lag depth 2: {msg}");
    }

    /// Given a source manifest that is a strict superset of the current
    /// manifest (an extra hydro the source prices but the study does not
    /// model), when `load_boundary_cuts` runs, then the load succeeds and
    /// the extra source slot is reported as a dropped coupling.
    #[test]
    fn boundary_load_superset_source_passes_the_topology_gate() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = vec![storage_slot(1), storage_slot(2)];
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[10.0], &boundary);

        let current = vec![storage_slot(1)];
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            1,
            &current,
            1_000_000.0,
        ))
        .unwrap();

        assert_eq!(
            cuts.len(),
            1,
            "a superset source must still load the current cuts"
        );
        assert!(
            cuts.report().tally_totals().3 > 0,
            "the extra source hydro must be reported as a dropped coupling"
        );
    }

    /// Given the same superset fixture as
    /// [`boundary_load_superset_source_passes_the_topology_gate`] and a
    /// default (non-strict) request, when `load_boundary_cuts` runs, then it
    /// returns `Ok`, emits no warning, and the dropped source slot is still
    /// visible in the reconciliation report.
    #[test]
    fn load_boundary_cuts_superset_source_loads_silently_and_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = vec![storage_slot(1), storage_slot(2)];
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[10.0], &boundary);

        let current = vec![storage_slot(1)];
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            1,
            &current,
            1_000_000.0,
        ))
        .unwrap();

        assert_eq!(cuts.len(), 1, "a superset source must load, never reject");
        assert!(
            cuts.report().tally_totals().3 > 0,
            "the dropped source slot must still be reported"
        );
    }

    /// Given the same fixture with `.with_strict(true)`, when
    /// `load_boundary_cuts` runs, then it rejects naming the boundary path,
    /// the total dropped count and every dropping family, and no cuts are
    /// returned.
    #[test]
    fn load_boundary_cuts_strict_superset_source_rejects_naming_families() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = vec![storage_slot(1), storage_slot(2)];
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[10.0], &boundary);

        let current = vec![storage_slot(1)];
        let result = load_boundary_cuts(
            &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 1, &current, 1_000_000.0)
                .with_strict(true),
        );

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains(&format!(
                "boundary policy at {} is a superset: 1 source slot(s) price entities this \
                 study does not model (storage: 1); see the reconciliation report or set \
                 policy.boundary.strict = false",
                tmp.path().display()
            )),
            "must name the path, the total and the dropping family, and state the remedy: {msg}"
        );
    }

    /// Given a boundary checkpoint with an empty manifest and a matching
    /// `current_state_dimension`, `load_boundary_cuts` returns `Ok` (no hard fail on
    /// absence), emits no warning, and reports the load as dimension-only.
    #[test]
    fn load_boundary_cuts_absent_manifest_loads_reporting_dimension_only() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &[]);

        let current = storage_manifest(1, 2);
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            1_000_000.0,
        ))
        .unwrap();

        assert_eq!(cuts.len(), 2, "absent manifest must still load cuts");
        let report = cuts.report();
        assert!(!report.reconciled);
        assert!(
            report.summary_line().contains("dimension-only"),
            "must state a dimension-only load: {}",
            report.summary_line()
        );
    }

    /// Given a boundary checkpoint with an empty entity manifest and a
    /// current manifest whose storage slots the (absent) source manifest can
    /// neither confirm nor deny, when `load_boundary_cuts` runs with a
    /// matching `state_dimension`, then the topology gate is skipped — no
    /// per-hydro reject, no warning — and the `state_dimension` fallback
    /// lets the load proceed.
    #[test]
    fn boundary_load_absent_manifest_skips_the_topology_gate() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[10.0, 20.0], &[]);

        let current = vec![storage_slot(1), storage_slot(2)];
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            1_000_000.0,
        ))
        .unwrap();

        assert_eq!(
            cuts.len(),
            2,
            "an unverifiable manifest must fall back to state_dimension, not reject"
        );
    }

    /// Given a boundary slot whose identity matches the current study but whose
    /// `was_active` is `false` while the current study treats it as active,
    /// `load_boundary_cuts` returns `Ok` (cut loaded) and surfaces no warning —
    /// a dormant-to-active transition is expected for a boundary load, not an
    /// advisory.
    #[test]
    fn load_boundary_cuts_was_active_divergence_loads_without_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let mut boundary = storage_manifest(1, 2);
        boundary[1].was_active = false; // dormant at the boundary stage
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = storage_manifest(1, 2);
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            1_000_000.0,
        ))
        .unwrap();

        assert_eq!(cuts.len(), 2, "was_active divergence must still load cuts");
    }

    /// A manifest carrying a `HydroTransitBucket` slot (`entity_type 3`, the
    /// downstream hydro id, the maturity lag as `subindex`) whose SOURCE and
    /// CURRENT arrival intervals are IDENTICAL round-trips: written to a
    /// checkpoint and reloaded against a slot-for-slot matching current
    /// manifest, `load_boundary_cuts` returns `Ok` with no warning and the
    /// bucket's coefficient blends at unit weight, bit-identical to the
    /// source — the exact-match case of the arrival-interval join.
    #[test]
    fn load_boundary_cuts_matching_transit_bucket_arrival_interval_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let start = encode_slot_date(fixture_priced_date(0));
        let end = encode_slot_date(fixture_priced_date(1));
        let manifest = vec![storage_slot(1), transit_bucket_slot_over(2, 1, start, end)];
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = vec![storage_slot(1), transit_bucket_slot_over(2, 1, start, end)];
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            1.0,
        ))
        .unwrap();

        assert_eq!(
            cuts.len(),
            2,
            "matching bucket arrival interval must load all cuts"
        );
        for cut in cuts.iter() {
            assert_eq!(
                cut.coefficients[1].to_bits(),
                1.0_f64.to_bits(),
                "an identical arrival interval blends at unit weight, bit-identical to source"
            );
        }
    }

    /// A policy exported WITHOUT travel-time buckets loaded by bucket-aware
    /// code whose terminal manifest has a `HydroTransitBucket` (type 3) slot
    /// at the same `state_dimension` succeeds: a target transit bucket with no
    /// source match defaults to `0.0` (distinct from storage/lag's
    /// reject-on-miss; a matching source transit slot is instead blended). The
    /// storage slot still loads its identity-matched coefficient.
    #[test]
    fn load_boundary_cuts_missing_transit_bucket_slot_identity_defaults_to_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = storage_manifest(1, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            NEUTRAL_LOADING_FACTOR,
        ))
        .expect(
            "a transit-bucket slot with no source counterpart must default to zero, not reject",
        );

        assert_eq!(cuts.len(), 2, "both cuts must still load");
        for cut in cuts.iter() {
            assert_eq!(
                cut.coefficients,
                vec![1.0, 0.0],
                "the storage slot copies its matched coefficient (the fixture's uniform 1.0); the \
                 transit slot defaults to 0.0"
            );
        }
    }

    /// A policy exported WITHOUT travel-time buckets has a smaller `state_dimension`
    /// (2) than the bucket-aware current study (3). That differing dimension does
    /// not reject under a verifiable manifest: per-slot reconciliation copies
    /// the two identity-matched storage coefficients through and resolves the
    /// target-only transit bucket to `0.0`.
    #[test]
    fn load_boundary_cuts_no_transit_bucket_export_reconciles_transit_bucket_to_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &storage_manifest(1, 2));

        let current = vec![storage_slot(1), storage_slot(2), transit_bucket_slot(2, 1)];
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            3,
            &current,
            NEUTRAL_LOADING_FACTOR,
        ))
        .expect("a differing-dimension load with a reconcilable manifest must load, not reject");

        assert_eq!(cuts.len(), 2, "both cuts must load");
        for cut in cuts.iter() {
            assert_eq!(
                cut.coefficients,
                vec![1.0, 1.0, 0.0],
                "storage slots copy their matched coefficients (the fixture's uniform 1.0); the \
                 target-only transit bucket reconciles to 0.0"
            );
        }
    }

    // ── load_boundary_cuts date selection ─────────────────────────────────────

    /// Write a checkpoint with one pool per `(pool_id, priced_state_date,
    /// intercepts)` entry, each pool's own `priced_state_date` self-describing
    /// fact set directly to the given RAW encoded value (never through
    /// `fixture_priced_date`'s per-pool default) — for the date-tie and
    /// undated-checkpoint selection tests, which need two pools sharing one
    /// priced date or every pool sharing the sentinel.
    fn write_pools_with_priced_state_dates(dir: &std::path::Path, pools: &[(u32, i32, &[f64])]) {
        let coefficients = [1.0_f64];
        let cuts_per_pool: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = pools
            .iter()
            .map(|(_, _, intercepts)| {
                intercepts
                    .iter()
                    .enumerate()
                    .map(|(i, &intercept)| cobre_io::PolicyCutRecord {
                        cut_id: i as u64,
                        slot_index: i as u32,
                        iteration: 0,
                        forward_pass_index: 0,
                        intercept,
                        coefficients: &coefficients,
                        is_active: true,
                    })
                    .collect()
            })
            .collect();
        let active_per_pool: Vec<Vec<u32>> = pools
            .iter()
            .map(|(_, _, intercepts)| (0..intercepts.len() as u32).collect())
            .collect();
        let payloads: Vec<StageCutsPayload<'_>> = pools
            .iter()
            .zip(&cuts_per_pool)
            .zip(&active_per_pool)
            .map(
                |(((pool_id, priced_state_date, intercepts), cuts), active)| StageCutsPayload {
                    stage_id: *pool_id,
                    state_dimension: 1,
                    capacity: intercepts.len() as u32,
                    warm_start_count: 0,
                    cuts,
                    active_cut_indices: active,
                    populated_count: intercepts.len() as u32,
                    entity_manifest: &[],
                    cost_scale_factor: 1_000_000.0,
                    node_id: i32::try_from(*pool_id).unwrap_or(-1),
                    graph_stage_id: -1,
                    priced_state_date: *priced_state_date,
                },
            )
            .collect();
        let metadata = test_support::checkpoint_metadata(
            pools.len() as u32,
            GraphManifest::default(),
            producer_block(),
        );
        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    /// Given a two-pool checkpoint whose pools are stamped `fixture_priced_date(0)`
    /// (`2030-01-01`) and `fixture_priced_date(1)` (`2030-02-01`), when
    /// `load_boundary_cuts` runs with a boundary date equal to pool 1's stamp,
    /// then it loads pool 1's cuts.
    #[test]
    fn load_boundary_cuts_selects_the_pool_priced_at_the_boundary_date() {
        let tmp = tempfile::tempdir().unwrap();
        let intercepts = vec![10.0, 20.0];
        write_minimal_checkpoint(tmp.path(), 2, 2, &intercepts);

        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(1),
            2,
            &[],
            NEUTRAL_LOADING_FACTOR,
        ))
        .unwrap();

        assert_eq!(cuts.len(), 2, "must load pool 1's two cuts");
        let loaded_intercepts: Vec<f64> = cuts.iter().map(|c| c.intercept).collect();
        assert_eq!(loaded_intercepts, intercepts);
    }

    /// Given the same two-pool checkpoint, when `load_boundary_cuts` runs with
    /// a boundary date matching neither pool, then it returns
    /// `Err(SddpError::Validation)` naming the boundary date and every pool's
    /// own priced date.
    #[test]
    fn load_boundary_cuts_unmatched_boundary_date_rejects_naming_available_dates() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 2, 2, &[10.0, 20.0]);
        let unmatched = fixture_priced_date(2);

        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            unmatched,
            2,
            &[],
            NEUTRAL_LOADING_FACTOR,
        ));

        assert!(result.is_err(), "an unmatched boundary date must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains(&unmatched.to_string()),
            "must name the boundary date: {msg}"
        );
        assert!(
            msg.contains(&fixture_priced_date(0).to_string()),
            "must name pool 0's priced date: {msg}"
        );
        assert!(
            msg.contains(&fixture_priced_date(1).to_string()),
            "must name pool 1's priced date: {msg}"
        );
    }

    /// Given a checkpoint whose two pools (2 and 5) are both stamped the same
    /// date — a branching source's terminal date tie — when
    /// `load_boundary_cuts` runs at that date, then it rejects naming both
    /// pool ids in ascending order.
    #[test]
    fn load_boundary_cuts_multi_pool_date_tie_rejects_naming_both_pools() {
        let tmp = tempfile::tempdir().unwrap();
        let tie_date = encode_slot_date(fixture_priced_date(0));
        write_pools_with_priced_state_dates(
            tmp.path(),
            &[(2, tie_date, &[10.0]), (5, tie_date, &[20.0])],
        );

        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            1,
            &[],
            NEUTRAL_LOADING_FACTOR,
        ));

        assert!(result.is_err(), "a date tie between two pools must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("[2, 5]"),
            "must name both tied pool ids in ascending order: {msg}"
        );
    }

    /// Given a checkpoint whose every pool carries
    /// [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`] — a pre-dated checkpoint —
    /// when `load_boundary_cuts` runs, then it rejects with a message
    /// containing "re-export" and the checkpoint path.
    #[test]
    fn load_boundary_cuts_undated_pools_reject_with_reexport_hint() {
        let tmp = tempfile::tempdir().unwrap();
        write_pools_with_priced_state_dates(
            tmp.path(),
            &[
                (0, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, &[10.0]),
                (1, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, &[20.0]),
            ],
        );

        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            1,
            &[],
            NEUTRAL_LOADING_FACTOR,
        ));

        assert!(
            result.is_err(),
            "an every-pool-undated checkpoint must reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("re-export"), "must advise re-export: {msg}");
        assert!(
            msg.contains(&tmp.path().display().to_string()),
            "must name the checkpoint path: {msg}"
        );
    }

    // ── compare_manifest_slot_identity tests ──────────────────────────────────

    /// Two same-length manifests differing only at slot 0's `entity_id` (7 vs 9)
    /// are rejected, naming slot `0` and both ids.
    #[test]
    fn compare_manifest_slot_identity_same_dim_different_id_rejects() {
        let source = storage_manifest(7, 2);
        let current = storage_manifest(9, 2);

        let result = compare_manifest_slot_identity(&source, &current, &mut ignore_warnings());

        assert!(result.is_err(), "different entity_id at slot 0 must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
        assert!(msg.contains("entity_id=7"), "error must name id 7: {msg}");
        assert!(msg.contains("entity_id=9"), "error must name id 9: {msg}");
    }

    /// An empty `source` manifest cannot be verified: warn once and return `Ok`.
    #[test]
    fn compare_manifest_slot_identity_empty_source_warns_and_oks() {
        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();

        let result = compare_manifest_slot_identity(&[], &current, &mut |m| {
            warnings.push(m.to_string());
        });

        assert!(result.is_ok(), "empty manifest must not hard-fail");
        assert_eq!(
            warnings.len(),
            1,
            "absence must surface exactly one warning"
        );
        assert!(
            warnings[0].contains("manifest absent"),
            "warning must flag the absent manifest: {}",
            warnings[0]
        );
    }

    /// Identical manifests pass with no warning.
    #[test]
    fn compare_manifest_slot_identity_identical_oks_without_warning() {
        let source = storage_manifest(1, 2);
        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();

        let result = compare_manifest_slot_identity(&source, &current, &mut |m| {
            warnings.push(m.to_string());
        });

        assert!(result.is_ok(), "identical manifests must pass");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot match must emit no warning: {warnings:?}"
        );
    }

    /// A `source`-dormant slot whose current counterpart is active warns but
    /// loads (`Ok`).
    #[test]
    fn compare_manifest_slot_identity_was_active_divergence_warns_and_oks() {
        let mut source = storage_manifest(1, 2);
        source[1].was_active = false;
        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();

        let result = compare_manifest_slot_identity(&source, &current, &mut |m| {
            warnings.push(m.to_string());
        });

        assert!(result.is_ok(), "was_active divergence must not hard-fail");
        assert_eq!(warnings.len(), 1, "divergence must surface one warning");
        assert!(
            warnings[0].contains("dormant") && warnings[0].contains("slot 1"),
            "warning must flag slot 1's dormancy divergence: {}",
            warnings[0]
        );
    }

    /// The full-FCF terminal-manifest shape: a checkpoint terminal manifest
    /// `[storage(7), storage(2)]` vs a current terminal manifest
    /// `[storage(9), storage(2)]` at equal `state_dimension` is rejected with a
    /// `Validation` error naming slot `0` — the same guard
    /// `load_and_validate_checkpoint` applies after the dims/`num_stages` check.
    #[test]
    fn compare_manifest_full_fcf_terminal_entity_swap_rejects() {
        let checkpoint_terminal = storage_manifest(7, 2);
        let current_terminal = storage_manifest(9, 2);

        let result = compare_manifest_slot_identity(
            &checkpoint_terminal,
            &current_terminal,
            &mut ignore_warnings(),
        );

        assert!(
            matches!(result, Err(SddpError::Validation(_))),
            "same-dimension terminal entity swap must be a Validation error"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
        assert!(
            msg.contains("entity_id=7") && msg.contains("entity_id=9"),
            "error must name both diverging ids: {msg}"
        );
    }

    // ── validate_policy_load tests ────────────────────────────────────────────

    /// `POLICY_COBRE_VERSION` as the source version passes the version gate
    /// under both `FullFcf` and `BoundaryInjection`.
    #[test]
    fn policy_version_accepted_for_every_kind() {
        let slots = storage_manifest(1, 2);
        let source = psm(2, 12, &slots);
        let current = psm(2, 12, &slots);

        assert!(validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current).is_ok());
        assert!(
            validate_policy_load::<BoundaryInjection>(POLICY_COBRE_VERSION, &source, &current)
                .is_ok()
        );
    }

    /// A source version other than `POLICY_COBRE_VERSION` is refused under
    /// both kinds, naming the recorded version.
    #[test]
    fn policy_version_refused_for_every_kind() {
        assert_ne!(POLICY_COBRE_VERSION, "0.0.1");
        let slots = storage_manifest(1, 2);
        let source = psm(2, 12, &slots);
        let current = psm(2, 12, &slots);

        let full_fcf_result = validate_policy_load::<FullFcf>("0.0.1", &source, &current);
        assert!(
            matches!(
                full_fcf_result,
                Err(SddpError::PolicyVersionMismatch { ref policy_version }) if policy_version == "0.0.1"
            ),
            "a different-version source must be refused under FullFcf: {full_fcf_result:?}"
        );

        let boundary_result = validate_policy_load::<BoundaryInjection>("0.0.1", &source, &current);
        assert!(
            matches!(
                boundary_result,
                Err(SddpError::PolicyVersionMismatch { ref policy_version }) if policy_version == "0.0.1"
            ),
            "a different-version source must be refused under BoundaryInjection: {boundary_result:?}"
        );
    }

    /// A version differing only by a suffix, or an empty string, is refused —
    /// the check is plain string equality, never a prefix or semver match.
    #[test]
    fn policy_version_refused_when_only_a_suffix_differs() {
        let slots = storage_manifest(1, 2);
        let source = psm(2, 12, &slots);
        let current = psm(2, 12, &slots);
        let suffixed = format!("{POLICY_COBRE_VERSION}-rc.1");

        for version in [suffixed.as_str(), ""] {
            let result = validate_policy_load::<FullFcf>(version, &source, &current);
            assert!(
                matches!(result, Err(SddpError::PolicyVersionMismatch { .. })),
                "{version:?} must be refused: {result:?}"
            );
        }
    }

    /// The version gate runs before the `state_dimension`/`num_stages`/slot
    /// check matrix: a mismatched version AND a mismatched `state_dimension`
    /// returns the version variant, not `Validation`.
    #[test]
    fn policy_version_checked_before_the_layout() {
        let slots = storage_manifest(1, 2);
        let source = psm(10, 12, &slots);
        let current = psm(8, 12, &slots);

        let result = validate_policy_load::<FullFcf>("0.0.1", &source, &current);

        assert!(
            matches!(result, Err(SddpError::PolicyVersionMismatch { .. })),
            "the version gate must win over the state_dimension gate: {result:?}"
        );
    }

    /// A boundary source recording a different version is refused by
    /// `load_boundary_cuts`, before per-slot reconciliation runs.
    #[test]
    fn policy_version_refused_at_boundary_load() {
        let tmp = tempfile::tempdir().unwrap();
        let metadata = cobre_io::CheckpointManifest {
            cobre_version: "0.0.1".to_string(),
            ..test_support::checkpoint_metadata(1, chain_graph_manifest(1), producer_block())
        };
        write_checkpoint_with_manifest_metadata(tmp.path(), 1, 2, &[10.0, 20.0], &[], &metadata);

        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &[],
            NEUTRAL_LOADING_FACTOR,
        ));

        assert!(
            matches!(
                result,
                Err(SddpError::PolicyVersionMismatch { ref policy_version }) if policy_version == "0.0.1"
            ),
            "a different-version boundary source must be refused before reconciliation: {result:?}"
        );
    }

    /// Identical `state_dimension`, `num_stages`, and slot-for-slot matching
    /// manifests pass `FullFcf` with no warnings.
    #[test]
    fn validate_policy_load_full_fcf_identical_oks_without_warning() {
        let slots = storage_manifest(1, 2);
        let source = psm(2, 12, &slots);
        let current = psm(2, 12, &slots);

        let report =
            validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current).unwrap();

        assert!(
            report.warnings.is_empty(),
            "identical manifests must emit no warning: {:?}",
            report.warnings
        );
    }

    /// A `state_dimension` mismatch is a hard reject on `FullFcf`, and its message
    /// names lag depth as a probable cause so an `inflow_lag_depth`-driven mismatch
    /// is legible.
    #[test]
    fn validate_policy_load_full_fcf_state_dimension_mismatch_rejects() {
        let slots = storage_manifest(1, 2);
        let source = psm(10, 12, &slots);
        let current = psm(8, 12, &slots);

        let result = validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current);

        assert!(result.is_err(), "state_dimension mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("state_dimension"), "{msg}");
        assert!(msg.contains("10"), "should include source value: {msg}");
        assert!(msg.contains('8'), "should include current value: {msg}");
        assert!(
            msg.contains("lag-state depth"),
            "message must name lag depth as a probable cause: {msg}"
        );
    }

    /// A differing `state_dimension` (14 vs 17) passes `BoundaryInjection`:
    /// `validate_policy_load` does not hard-reject it there, deferring to the
    /// per-slot reconciliation in `load_boundary_cuts`.
    #[test]
    fn validate_policy_load_boundary_injection_allows_differing_state_dimension() {
        let slots = storage_manifest(1, 2);
        let source = psm(14, 12, &slots);
        let current = psm(17, 12, &slots);

        let result =
            validate_policy_load::<BoundaryInjection>(POLICY_COBRE_VERSION, &source, &current);

        assert!(
            result.is_ok(),
            "BoundaryInjection defers a differing state_dimension to reconcile: {result:?}"
        );
    }

    /// The same differing `state_dimension` (14 vs 17) still hard-rejects under
    /// `FullFcf`: `CHECK_STATE_DIMENSION` stays `true` there.
    #[test]
    fn validate_policy_load_full_fcf_still_rejects_differing_state_dimension() {
        let slots = storage_manifest(1, 2);
        let source = psm(14, 12, &slots);
        let current = psm(17, 12, &slots);

        let result = validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current);

        assert!(
            result.is_err(),
            "FullFcf still rejects a differing state_dimension"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension mismatch"),
            "message must name the state_dimension mismatch: {msg}"
        );
    }

    /// A `num_stages` mismatch is a hard reject on `FullFcf` but the identical
    /// inputs pass `BoundaryInjection` (unchecked there).
    #[test]
    fn validate_policy_load_num_stages_mismatch_rejects_full_fcf_oks_boundary() {
        let slots = storage_manifest(1, 2);
        let source = psm(10, 12, &slots);
        let current = psm(10, 24, &slots);

        let full_fcf_result =
            validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current);
        assert!(
            full_fcf_result.is_err(),
            "num_stages mismatch must reject FullFcf"
        );
        let msg = full_fcf_result.unwrap_err().to_string();
        assert!(msg.contains("num_stages"), "{msg}");
        assert!(msg.contains("12"), "should include source value: {msg}");
        assert!(msg.contains("24"), "should include current value: {msg}");

        let boundary_result =
            validate_policy_load::<BoundaryInjection>(POLICY_COBRE_VERSION, &source, &current);
        assert!(
            boundary_result.is_ok(),
            "num_stages is unchecked under BoundaryInjection: {boundary_result:?}"
        );
    }

    /// Both `state_dimension` (10 vs 8) and `num_stages` (12 vs 24) mismatch under
    /// `FullFcf`; the `state_dimension` guard fires first, so the error names
    /// `state_dimension`, not `num_stages`.
    #[test]
    fn validate_policy_load_full_fcf_both_dimensions_mismatched_rejects() {
        let slots = storage_manifest(1, 2);
        let source = psm(10, 12, &slots);
        let current = psm(8, 24, &slots);

        let result = validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current);

        assert!(result.is_err(), "both-dimension mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "should report state_dimension mismatch first: {msg}"
        );
    }

    /// A per-slot identity mismatch is a hard reject under `FullFcf`, naming the
    /// mismatched slot and both diverging `entity_id`s.
    #[test]
    fn validate_policy_load_full_fcf_slot_mismatch_rejects() {
        let source_slots = storage_manifest(7, 2);
        let current_slots = storage_manifest(9, 2);
        let source = psm(2, 12, &source_slots);
        let current = psm(2, 12, &current_slots);

        let result = validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current);

        assert!(result.is_err(), "slot identity mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
        assert!(
            msg.contains("entity_id=7"),
            "error must name the source id 7: {msg}"
        );
        assert!(
            msg.contains("entity_id=9"),
            "error must name the current id 9: {msg}"
        );
    }

    /// `validate_policy_load` itself no longer checks per-slot identity under
    /// `BoundaryInjection` (unlike `FullFcf`): a same-`state_dimension`
    /// identity mismatch it would have hard-rejected now passes here, because
    /// slot identity is reconciled separately, by `reconcile::build_rebind`,
    /// in `load_boundary_cuts` — confined to that load path, not this
    /// lower-level manifest check.
    #[test]
    fn validate_policy_load_boundary_injection_does_not_check_slot_identity() {
        let source_slots = storage_manifest(7, 2);
        let current_slots = storage_manifest(9, 2);
        let source = psm(2, 12, &source_slots);
        let current = psm(2, 6, &current_slots);

        let result =
            validate_policy_load::<BoundaryInjection>(POLICY_COBRE_VERSION, &source, &current);

        assert!(
            result.is_ok(),
            "BoundaryInjection defers slot identity to reconcile::build_rebind: {result:?}"
        );
    }

    /// The ring-format checkpoint consequence, pinned as a pair. A lane-era
    /// checkpoint's state vector carried a retired trailing-lane block on top
    /// of the anticipated ring, so its `state_dimension` is WIDER (ring
    /// `A·k_max` plus the lane's `W`) than a current-version layout. Loaded as
    /// `FullFcf` (resume/warm-start) it rejects at the unconditional
    /// `state_dimension` gate — the intended clean break. The SAME slot layout
    /// boundary-injected at equal width but with its slots repositioned (a
    /// positional match would fail: the storage and ring slots no longer line
    /// up) reconciles by identity (storage) and by date (each ring slot fans
    /// onto the current ring by calendar overlap of its own carried interval,
    /// which reaches past the boundary date) and loads, so boundary injection
    /// across the version boundary is unaffected.
    #[test]
    fn lane_era_wider_checkpoint_rejects_full_fcf_but_boundary_injection_reconciles() {
        let first_month = encode_slot_date(fixture_priced_date(0));
        let second_month = encode_slot_date(fixture_priced_date(1));
        let third_month = encode_slot_date(fixture_priced_date(2));
        let lane_era = vec![
            storage_slot(1),
            anticipated_slot_at(9, 0, first_month),
            anticipated_slot_at(9, 1, second_month),
        ];

        let current_narrow = vec![storage_slot(1), anticipated_slot_at(9, 0, first_month)];
        let full_fcf = validate_policy_load::<FullFcf>(
            POLICY_COBRE_VERSION,
            &psm(lane_era.len() as u32, 12, &lane_era),
            &psm(current_narrow.len() as u32, 12, &current_narrow),
        );
        assert!(
            matches!(full_fcf, Err(SddpError::Validation(_))),
            "a lane-era (wider) checkpoint must reject under FullFcf: {full_fcf:?}"
        );
        let msg = full_fcf.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "the FullFcf reject must name the state_dimension gate: {msg}"
        );

        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(
            tmp.path(),
            1,
            lane_era.len() as u32,
            &[10.0, 20.0],
            &lane_era,
        );

        let current_reordered = vec![
            anticipated_slot_at(9, 0, second_month).with_interval(second_month, third_month),
            storage_slot(1),
            anticipated_slot_at(9, 1, first_month).with_interval(first_month, second_month),
        ];

        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            current_reordered.len() as u32,
            &current_reordered,
            1_000_000.0,
        ))
        .expect("the same lane-era source must load under BoundaryInjection by identity + date");

        assert_eq!(cuts.len(), 2, "both boundary cuts must load");
        assert!(
            cuts.report().reconciled,
            "the load must run the identity + date reconcile path, not the absent-manifest skip"
        );
        for cut in cuts.iter() {
            assert_eq!(
                cut.coefficients.len(),
                current_reordered.len(),
                "each reconciled cut spans the current state vector"
            );
            let storage = cut.coefficients[1];
            assert!(storage != 0.0, "the storage lane copies by identity");
            assert_eq!(
                cut.coefficients[0], storage,
                "a ring slot fully covered by one source month blends at unit weight"
            );
            assert_eq!(
                cut.coefficients[2], storage,
                "a ring slot fully covered by one source month blends at unit weight"
            );
        }
    }

    /// An empty manifest on either side cannot be verified by slot identity:
    /// `validate_policy_load` falls back to the `state_dimension` check alone,
    /// returning `Ok` with one warning.
    #[test]
    fn validate_policy_load_full_fcf_empty_manifest_oks_with_warning() {
        let current_slots = storage_manifest(1, 2);
        let source = psm(2, 12, &[]);
        let current = psm(2, 12, &current_slots);

        let report =
            validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current).unwrap();

        assert_eq!(report.warnings.len(), 1, "absence must surface one warning");
        assert!(
            report.warnings[0].contains("manifest absent"),
            "warning must flag the absent manifest: {}",
            report.warnings[0]
        );
    }

    /// An empty source manifest cannot be reconciled per-slot, so a differing
    /// `state_dimension` there falls back to the `state_dimension` guard: the
    /// boundary load returns `Err(Validation)` rather than carrying source-length
    /// coefficients into the cut-pool copy, which would panic on the mismatch.
    #[test]
    fn load_boundary_cuts_empty_manifest_differing_state_dimension_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[10.0, 20.0], &[]);

        let result = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            3,
            &[],
            1_000_000.0,
        ));

        assert!(
            matches!(result, Err(SddpError::Validation(_))),
            "an unreconcilable empty-manifest differing-dimension load must reject: {result:?}"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension mismatch"),
            "the reject must name the state_dimension mismatch: {msg}"
        );
    }

    // ── basis-status export+load round-trip (§E6) ─────────────────────────────

    /// Every one of the seven `BasisStatus` variants survives the full
    /// export→disk→load path losslessly — including the CLP-only `Superbasic`
    /// and `Fixed`, which the old `to_highs_code` path folded onto `Nonbasic`/
    /// `Lower`. Encode via `convert_basis_cache`, round-trip through the codec,
    /// decode via `build_basis_cache_for_nodes`.
    #[test]
    fn basis_status_round_trips_through_export_and_load_for_every_variant() {
        use cobre_io::{PolicyBasisRecord, deserialize_stage_basis, serialize_stage_basis};
        use cobre_solver::{Basis, BasisStatus};

        use super::build_basis_cache_for_nodes;
        use crate::TrainingResult;
        use crate::policy_export::convert_basis_cache;
        use crate::workspace::CapturedBasis;

        let col_status = vec![
            BasisStatus::Lower,
            BasisStatus::Basic,
            BasisStatus::Upper,
            BasisStatus::Zero,
            BasisStatus::Nonbasic,
            BasisStatus::Superbasic,
            BasisStatus::Fixed,
        ];
        // Reversed so the column and row vectors are checked independently.
        let mut row_status = col_status.clone();
        row_status.reverse();

        let captured = CapturedBasis {
            basis: Basis {
                col_status: col_status.clone(),
                row_status: row_status.clone(),
            },
            base_row_count: row_status.len(),
            cut_row_slots: Vec::new(),
            state_at_capture: Vec::new(),
            node_id: NodeId(0),
        };
        let training_result = TrainingResult::new(
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            "test".to_string(),
            0,
            vec![Some(captured)],
            Vec::new(),
            None,
            None,
        );

        let (col_u8, row_u8) = convert_basis_cache(&training_result);

        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &col_u8[0],
            row_status: &row_u8[0],
            num_cut_rows: 0,
        };
        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("codec round-trip must succeed");

        let cache = build_basis_cache_for_nodes(
            std::slice::from_ref(&owned),
            &[],
            &vec![NodeId(0)].into(),
            &vec![0].into(),
            &[(7, 7)],
        )
        .expect("a same-shape record must load");
        let recovered = cache[0].as_ref().expect("stage 0 basis must be present");

        assert_eq!(
            recovered.basis.col_status, col_status,
            "every column variant must round-trip losslessly"
        );
        assert_eq!(
            recovered.basis.row_status, row_status,
            "every row variant must round-trip losslessly, including Superbasic/Fixed"
        );
    }

    /// Pre-existing-file compatibility: a checkpoint written by the pre-canonical
    /// writer stored `HiGHS` codes (bytes `0..=4`) directly in `column_status`/
    /// `row_status`. Those same bytes must load to exactly the statuses the old
    /// `HiGHS`-space decode produced, since `0..=4` means the same in the canonical
    /// and `HiGHS` code spaces.
    #[test]
    fn pre_existing_checkpoint_bytes_load_to_highs_space_statuses() {
        use cobre_io::{PolicyBasisRecord, deserialize_stage_basis, serialize_stage_basis};
        use cobre_solver::BasisStatus;

        use super::build_basis_cache_for_nodes;

        // HiGHS codes 0..=4, exactly what the pre-canonical writer stored on disk.
        let col_bytes: [u8; 5] = [0, 1, 2, 3, 4];
        let row_bytes: [u8; 5] = [4, 3, 2, 1, 0];

        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &col_bytes,
            row_status: &row_bytes,
            num_cut_rows: 0,
        };
        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("codec round-trip must succeed");

        let cache = build_basis_cache_for_nodes(
            std::slice::from_ref(&owned),
            &[],
            &vec![NodeId(0)].into(),
            &vec![0].into(),
            &[(5, 5)],
        )
        .expect("a same-shape record must load");
        let recovered = cache[0].as_ref().expect("stage 0 basis must be present");

        let expected_col: Vec<BasisStatus> = col_bytes
            .iter()
            .map(|&c| BasisStatus::from_highs_code(i32::from(c)))
            .collect();
        let expected_row: Vec<BasisStatus> = row_bytes
            .iter()
            .map(|&c| BasisStatus::from_highs_code(i32::from(c)))
            .collect();
        assert_eq!(
            recovered.basis.col_status, expected_col,
            "old-file bytes must load to the old HiGHS-space column statuses"
        );
        assert_eq!(
            recovered.basis.row_status, expected_row,
            "old-file bytes must load to the old HiGHS-space row statuses"
        );
    }

    // ── branching-graph basis-cache keying ────────────────────────────────────

    use cobre_io::{OwnedPolicyBasisRecord, StageCutsReadResult};

    /// An active cut record at LP slot `slot`.
    fn active_cut(slot: u32) -> OwnedPolicyCutRecord {
        OwnedPolicyCutRecord {
            cut_id: u64::from(slot),
            slot_index: slot,
            iteration: 0,
            forward_pass_index: 0,
            intercept: 0.0,
            coefficients: vec![0.0],
            is_active: true,
        }
    }

    /// A pool-keyed cut collection (`stage_id` is the pool id) holding the given
    /// active slots.
    fn pool_cuts(pool: u32, slots: &[u32]) -> StageCutsReadResult {
        StageCutsReadResult {
            stage_id: pool,
            state_dimension: 1,
            capacity: 8,
            warm_start_count: 0,
            populated_count: slots.len() as u32,
            cuts: slots.iter().copied().map(active_cut).collect(),
            entity_manifest: Vec::new(),
            cost_scale_factor: None,
            node_id: -1,
            graph_stage_id: -1,
            priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        }
    }

    /// A node-keyed basis record (`stage_id` is the node ordinal) with
    /// `num_cut` trailing cut rows over `3` template rows.
    fn node_basis(node: u32, num_cut: usize) -> OwnedPolicyBasisRecord {
        OwnedPolicyBasisRecord {
            stage_id: node,
            iteration: 0,
            column_status: vec![1_u8, 1_u8],
            row_status: vec![1_u8; 3 + num_cut],
            num_cut_rows: num_cut as u32,
        }
    }

    /// A branching (K-fan) checkpoint where `n_nodes (7) > num_stages`: nodes
    /// 0/1/2 are interior (pools 0/1/2), leaves 3..=6 share pool 3. The cache is
    /// sized by `n_nodes`, every node lands in its own slot (no `>= num_stages`
    /// drop, no cross-node collision), each carries its own `node_id`, and the
    /// cut-slot reconstruction resolves through the node's OWN pool
    /// (`node_pools[node]`) — never the pool whose id equals the node ordinal,
    /// which for leaves 4/5/6 names no pool at all.
    #[test]
    fn build_basis_cache_from_checkpoint_keys_branching_graph_by_node() {
        use super::build_basis_cache_for_nodes;

        let node_ids: TypedVec<NodePos, NodeId> = vec![10, 11, 12, 13, 14, 15, 16]
            .into_iter()
            .map(NodeId)
            .collect();
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2, 3, 3, 3, 3].into();
        let node_dims = [(2, 3); 7];
        // Ascending per-pool active-slot counts (1/2/3/4): a node wrongly
        // resolved to a pool with FEWER slots than its own cut-row count falls
        // back to an empty cut_row_slots rather than coincidentally matching
        // (a swap to a pool with equal-or-more slots shares the same [0..k)
        // prefix and is not distinguishable by value alone).
        let stage_cuts = vec![
            pool_cuts(0, &[0]),
            pool_cuts(1, &[0, 1]),
            pool_cuts(2, &[0, 1, 2]),
            pool_cuts(3, &[0, 1, 2, 3]),
        ];
        let stage_bases = vec![
            node_basis(0, 1),
            node_basis(1, 2),
            node_basis(2, 3),
            node_basis(3, 4),
            node_basis(4, 4),
            node_basis(5, 4),
            node_basis(6, 4),
        ];

        let cache = build_basis_cache_for_nodes(
            &stage_bases,
            &stage_cuts,
            &node_ids,
            &node_pools,
            &node_dims,
        )
        .expect("every node's record fits its (2, 3) template");

        assert_eq!(cache.len(), 7, "cache is sized by n_nodes, not num_stages");
        for (node, slot) in cache.iter().enumerate() {
            let cb = slot
                .as_ref()
                .unwrap_or_else(|| panic!("node {node} must not be dropped"));
            assert_eq!(
                cb.node_id,
                node_ids[NodePos(node)],
                "node {node} must carry its own node_id (no cross-node collision)"
            );
        }

        // Interior node 2 (pool 2) recovers pool 2's own active slots — distinct
        // in length from pool 0's and pool 1's, so a wrong-pool lookup would fail.
        assert_eq!(
            cache[2].as_ref().unwrap().cut_row_slots,
            vec![0_u32, 1_u32, 2_u32]
        );
        // Every leaf (nodes 3..=6) recovers the SHARED pool 3's active slots,
        // keyed by node_pools[node] == 3 — a node-ordinal key would drop 4/5/6.
        for (node, slot) in cache.iter().enumerate().skip(3) {
            assert_eq!(
                slot.as_ref().unwrap().cut_row_slots,
                vec![0_u32, 1_u32, 2_u32, 3_u32],
                "leaf node {node} must recover shared pool 3's active slots"
            );
        }
    }

    /// Chain degeneracy: `node_pools` is the identity (`node_pools[t] == t`), so
    /// node-keyed sizing and pool-keyed cut matching reduce exactly to the
    /// pre-branching per-stage behavior — one basis per stage, keyed by ordinal.
    #[test]
    fn build_basis_cache_from_checkpoint_chain_is_identity_keyed() {
        use super::build_basis_cache_for_nodes;

        let node_ids: TypedVec<NodePos, NodeId> = vec![0, 1, 2].into_iter().map(NodeId).collect();
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2].into();
        let node_dims = [(2, 3); 3];
        // Ascending per-pool active-slot counts (1/2/3): a node wrongly resolved
        // to a pool with FEWER slots than its own cut-row count falls back to an
        // empty cut_row_slots rather than coincidentally matching (a swap to a
        // pool with equal-or-more slots shares the same [0..k) prefix and is not
        // distinguishable by value alone).
        let stage_cuts = vec![
            pool_cuts(0, &[0]),
            pool_cuts(1, &[0, 1]),
            pool_cuts(2, &[0, 1, 2]),
        ];
        let stage_bases = vec![node_basis(0, 1), node_basis(1, 2), node_basis(2, 3)];

        let cache = build_basis_cache_for_nodes(
            &stage_bases,
            &stage_cuts,
            &node_ids,
            &node_pools,
            &node_dims,
        )
        .expect("every node's record fits its (2, 3) template");

        assert_eq!(cache.len(), 3);
        assert_eq!(cache[0].as_ref().unwrap().cut_row_slots, vec![0_u32]);
        assert_eq!(cache[1].as_ref().unwrap().cut_row_slots, vec![0_u32, 1_u32]);
        assert_eq!(
            cache[2].as_ref().unwrap().cut_row_slots,
            vec![0_u32, 1_u32, 2_u32]
        );
    }

    // ── stored-basis dimension check ──────────────────────────────────────────

    /// A node-0 basis record with `cols` columns, `rows` total row entries, and
    /// `num_cut` recorded cut rows.
    fn dim_basis_record(cols: usize, rows: usize, num_cut: usize) -> OwnedPolicyBasisRecord {
        OwnedPolicyBasisRecord {
            stage_id: 0,
            iteration: 0,
            column_status: vec![1_u8; cols],
            row_status: vec![1_u8; rows],
            num_cut_rows: num_cut as u32,
        }
    }

    /// A single node, keyed by its own id and pool 0.
    fn single_node_ids_and_pools() -> (TypedVec<NodePos, NodeId>, TypedVec<NodePos, usize>) {
        (vec![NodeId(0)].into(), vec![0].into())
    }

    #[test]
    fn stored_basis_with_extra_column_is_refused() {
        use super::build_basis_cache_for_nodes;

        let (node_ids, node_pools) = single_node_ids_and_pools();
        let stage_bases = vec![dim_basis_record(5, 3, 0)];

        let err = build_basis_cache_for_nodes(&stage_bases, &[], &node_ids, &node_pools, &[(4, 3)])
            .expect_err("a wider column count must be refused");

        assert!(
            matches!(
                err,
                SddpError::StoredBasisDimensionMismatch {
                    expected_cols: 4,
                    found_cols: 5,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn stored_basis_with_fewer_rows_than_the_template_is_refused() {
        use super::build_basis_cache_for_nodes;

        let (node_ids, node_pools) = single_node_ids_and_pools();
        let stage_bases = vec![dim_basis_record(4, 2, 0)];

        let err = build_basis_cache_for_nodes(&stage_bases, &[], &node_ids, &node_pools, &[(4, 3)])
            .expect_err("fewer rows than the template must be refused");

        assert!(
            matches!(
                err,
                SddpError::StoredBasisDimensionMismatch {
                    expected_template_rows: 3,
                    found_rows: 2,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn stored_basis_with_more_rows_than_recorded_cuts_allow_is_refused() {
        use super::build_basis_cache_for_nodes;

        let (node_ids, node_pools) = single_node_ids_and_pools();
        let stage_bases = vec![dim_basis_record(4, 6, 2)];

        let err = build_basis_cache_for_nodes(&stage_bases, &[], &node_ids, &node_pools, &[(4, 3)])
            .expect_err("more rows than the recorded cut count allows must be refused");

        assert!(
            matches!(
                err,
                SddpError::StoredBasisDimensionMismatch {
                    expected_template_rows: 3,
                    found_rows: 6,
                    found_cut_rows: 2,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// The root-record shape: captured before the last backward pass appends
    /// its cuts, so `row_status` carries fewer trailing rows than the
    /// `num_cut_rows` recorded at export time — a shape the dimension check admits.
    #[test]
    fn stored_basis_captured_before_the_last_cuts_loads() {
        use super::build_basis_cache_for_nodes;

        let (node_ids, node_pools) = single_node_ids_and_pools();
        let stage_bases = vec![dim_basis_record(4, 4, 3)];

        let cache =
            build_basis_cache_for_nodes(&stage_bases, &[], &node_ids, &node_pools, &[(4, 3)])
                .expect("a record within the bound must load");

        assert!(cache[0].is_some(), "the node's basis must be present");
    }

    #[test]
    fn stored_basis_with_every_recorded_cut_row_loads() {
        use super::build_basis_cache_for_nodes;

        let (node_ids, node_pools) = single_node_ids_and_pools();
        let stage_bases = vec![dim_basis_record(4, 5, 2)];

        let cache =
            build_basis_cache_for_nodes(&stage_bases, &[], &node_ids, &node_pools, &[(4, 3)])
                .expect("a record within the bound must load");

        assert!(cache[0].is_some(), "the node's basis must be present");
    }

    /// The root-record shape (consult 1): captured before the last backward pass
    /// appends its cuts, so the pool's active slots outrun the basis's own cut
    /// rows. `base_row_count` must come from the template (`node_dims`), not
    /// `row_status.len() - num_cut_rows`, and the unbroken `0..populated` prefix
    /// still lets the single captured cut row resolve to slot 0.
    #[test]
    fn stored_basis_captured_before_the_last_cuts_keeps_the_template_rows() {
        use super::build_basis_cache_for_nodes;

        let (node_ids, node_pools) = single_node_ids_and_pools();
        let stage_bases = vec![dim_basis_record(4, 4, 3)];
        let stage_cuts = vec![pool_cuts(0, &[0, 1, 2])];

        let cache = build_basis_cache_for_nodes(
            &stage_bases,
            &stage_cuts,
            &node_ids,
            &node_pools,
            &[(4, 3)],
        )
        .expect("a record within the bound must load");

        let cb = cache[0].as_ref().expect("the node's basis must be present");
        assert_eq!(
            cb.base_row_count, 3,
            "base_row_count must come from the template, not row_status.len() - num_cut_rows"
        );
        assert_eq!(cb.cut_row_slots, vec![0_u32]);
    }

    /// When a cut between the captured row and the pool's populated tail was
    /// deactivated, the active slots are no longer the unbroken `0..populated`
    /// prefix, so slot identity cannot be proven: `row_status` is truncated to
    /// the template rows and `cut_row_slots` stays empty.
    #[test]
    fn stored_basis_after_cut_deactivation_drops_cut_statuses() {
        use super::build_basis_cache_for_nodes;

        let (node_ids, node_pools) = single_node_ids_and_pools();
        let stage_bases = vec![dim_basis_record(4, 4, 3)];
        let mut pool = pool_cuts(0, &[0, 1, 2]);
        pool.cuts[1].is_active = false;
        let stage_cuts = vec![pool];

        let cache = build_basis_cache_for_nodes(
            &stage_bases,
            &stage_cuts,
            &node_ids,
            &node_pools,
            &[(4, 3)],
        )
        .expect("a record within the bound must load");

        let cb = cache[0].as_ref().expect("the node's basis must be present");
        assert_eq!(
            cb.basis.row_status.len(),
            3,
            "row_status must be truncated to the template row count once a cut was deactivated"
        );
        assert!(
            cb.cut_row_slots.is_empty(),
            "cut_row_slots must be empty when slot identity cannot be proven"
        );
    }

    // ── inject_boundary_cuts tests ──────────────────────────────────────────────

    #[test]
    fn inject_boundary_cuts_produces_fixed_capacity_terminal_pool() {
        let mut setup = test_support::oracle_chain_setup(10);
        let terminal_idx = setup
            .inputs
            .node_graph
            .terminal_pool(setup.num_stages())
            .unwrap();
        let state_dimension = setup.fcf.state_dimension;
        let records = vec![
            owned_cut(5.0, vec![1.0; state_dimension]),
            owned_cut(6.0, vec![2.0; state_dimension]),
        ];
        let boundary_cuts = ValidatedBoundaryCuts {
            records: records.clone(),
            report: BoundaryReconciliationReport::default(),
        };

        inject_boundary_cuts(&mut setup, &boundary_cuts).unwrap();

        let pool = &setup.fcf.pools[terminal_idx];
        assert_eq!(pool.warm_start_count as usize, records.len());
        assert_eq!(
            pool.capacity,
            records.len(),
            "terminal pool must carry no growable training slack after injection"
        );
    }

    /// `active_cuts()` on the fixed terminal pool must be identical to what a
    /// growable construction (nonzero `max_iterations`) would yield for the
    /// same records — the fixed capacity changes only the unused tail, never
    /// the loaded region's slot/intercept/coefficient sequence.
    #[test]
    fn inject_boundary_cuts_active_cuts_matches_growable_construction() {
        let mut setup = test_support::oracle_chain_setup(10);
        let terminal_idx = setup
            .inputs
            .node_graph
            .terminal_pool(setup.num_stages())
            .unwrap();
        let state_dimension = setup.fcf.state_dimension;
        let forward_passes = setup.fcf.forward_passes;
        let records = vec![
            owned_cut(5.0, vec![1.0; state_dimension]),
            owned_cut(6.0, vec![2.0; state_dimension]),
            owned_cut(7.0, vec![3.0; state_dimension]),
        ];
        let boundary_cuts = ValidatedBoundaryCuts {
            records: records.clone(),
            report: BoundaryReconciliationReport::default(),
        };

        inject_boundary_cuts(&mut setup, &boundary_cuts).unwrap();

        let fixed_active: Vec<(usize, f64, Vec<f64>)> = setup.fcf.pools[terminal_idx]
            .active_cuts()
            .map(|(slot, intercept, coeffs)| (slot, intercept, coeffs.to_vec()))
            .collect();

        let growable = CutPool::new_with_warm_start(state_dimension, forward_passes, 5, &records);
        let growable_active: Vec<(usize, f64, Vec<f64>)> = growable
            .active_cuts()
            .map(|(slot, intercept, coeffs)| (slot, intercept, coeffs.to_vec()))
            .collect();

        assert_eq!(
            fixed_active, growable_active,
            "active_cuts() must be identical whether or not growable training slack is reserved"
        );
        for (i, (slot, intercept, coeffs)) in fixed_active.iter().enumerate() {
            assert_eq!(*slot, i);
            assert_eq!(*intercept, records[i].intercept);
            assert_eq!(coeffs, &records[i].coefficients);
        }
    }

    /// The non-root-rank boundary path — [`ValidatedBoundaryCuts::from_broadcast_records`]
    /// on the records the reading rank broadcast — injects a terminal pool
    /// bit-identical to the reading rank's own directly-loaded set. Gating
    /// injection on the rank-0-only config instead leaves a non-root rank's
    /// terminal pool empty, so its forward/backward/simulation terminal solves
    /// drop the post-horizon value-to-go (a rank-count-dependent wrong bound).
    #[test]
    fn from_broadcast_records_injects_pool_identical_to_direct_load() {
        let state_dimension = test_support::oracle_chain_setup(10).fcf.state_dimension;
        let records = vec![
            owned_cut(5.0, vec![1.0; state_dimension]),
            owned_cut(6.0, vec![2.0; state_dimension]),
            owned_cut(7.0, vec![3.0; state_dimension]),
        ];
        // Reading rank (rank 0): the set `load_boundary_cuts` returns.
        let direct = ValidatedBoundaryCuts {
            records: records.clone(),
            report: BoundaryReconciliationReport::default(),
        };
        // Non-root rank: reconstructed from the broadcast record vec.
        let broadcast = ValidatedBoundaryCuts::from_broadcast_records(records.clone());

        let mut setup_direct = test_support::oracle_chain_setup(10);
        let mut setup_bcast = test_support::oracle_chain_setup(10);
        inject_boundary_cuts(&mut setup_direct, &direct).unwrap();
        inject_boundary_cuts(&mut setup_bcast, &broadcast).unwrap();

        let terminal_idx = setup_direct
            .inputs
            .node_graph
            .terminal_pool(setup_direct.num_stages())
            .unwrap();
        let pool_direct = &setup_direct.fcf.pools[terminal_idx];
        let pool_bcast = &setup_bcast.fcf.pools[terminal_idx];

        assert_eq!(
            pool_bcast.warm_start_count as usize,
            records.len(),
            "the broadcast path must populate the terminal pool, never leave it empty"
        );
        assert_eq!(pool_bcast.warm_start_count, pool_direct.warm_start_count);
        assert_eq!(pool_bcast.capacity, pool_direct.capacity);

        let active = |pool: &CutPool| -> Vec<(usize, f64, Vec<f64>)> {
            pool.active_cuts()
                .map(|(slot, intercept, coeffs)| (slot, intercept, coeffs.to_vec()))
                .collect()
        };
        assert_eq!(
            active(pool_bcast),
            active(pool_direct),
            "broadcast-reconstructed terminal pool must be bit-identical to the directly-loaded one"
        );
    }

    // ── intercept-fold wiring tests ───────────────────────────────────────────

    /// One fixed post-horizon window for `thermal_id` spanning `[start, end)`.
    fn fixed_window(
        thermal_id: i32,
        start: NaiveDate,
        end: NaiveDate,
        value_mw: f64,
    ) -> AnticipatedCommitmentHistory {
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(thermal_id),
            start_date: start,
            end_date: end,
            value_mw,
        }
    }

    /// Write a single-stage MARKED checkpoint (`cost_scale_factor: Some(s)`)
    /// whose one cut carries the given at-rest `intercept`/`coefficients`
    /// (byte-for-byte, no transform here) and whose stage payload attaches
    /// `manifest` — the fold wiring tests need a verifiable manifest, a marked
    /// scale, and custom coefficients simultaneously, which neither
    /// `write_checkpoint_with_scale` (empty manifest) nor
    /// `write_checkpoint_with_manifest` (unmarked, uniform coefficients) gives.
    fn write_marked_checkpoint_with_manifest(
        dir: &std::path::Path,
        intercept: f64,
        coefficients: &[f64],
        manifest: &[EntitySlot],
        cost_scale_factor: f64,
    ) {
        let state_dimension = coefficients.len() as u32;
        let cut = cobre_io::PolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active: true,
        };
        let cuts = vec![cut];
        let payload = StageCutsPayload {
            stage_id: 0,
            state_dimension,
            capacity: 1,
            warm_start_count: 0,
            cuts: &cuts,
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: manifest,
            cost_scale_factor: 1_000_000.0,
            node_id: 0,
            graph_stage_id: -1,
            priced_state_date: encode_slot_date(fixture_priced_date(0)),
        };
        let metadata = test_support::checkpoint_metadata(
            1,
            chain_graph_manifest(1),
            ProducerBlock {
                cost_scale_factor: Some(cost_scale_factor),
                ..producer_block()
            },
        );
        cobre_io::write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).unwrap();
    }

    /// A marked source whose cut carries a known anticipated coefficient at
    /// `source_pos = 1`, reconciled by a single fixed window fully covering the
    /// source's March 2026 month (`overlap / H_M == 1.0`, fold vector
    /// `[(1, value_mw)]`), folds `raw_coeff[1]·value_mw` into the intercept on
    /// the RAW record and then rides the marked `÷s` transform: the loaded
    /// intercept is `(raw_intercept + raw_coeff[1]·value_mw) / s`. The current
    /// manifest's anticipated slot reconciles to `Zero` (dated, no interval), so
    /// the fold necessarily read the SOURCE coefficient before `rebind_cut`
    /// zeroed it.
    #[test]
    fn load_boundary_cuts_marked_fold_moves_intercept_by_expected_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let raw_intercept = 1000.0;
        let raw_coefficients = [7.0, 3.0];
        let manifest = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_301)];
        let s = 1_000_000.0;
        write_marked_checkpoint_with_manifest(
            tmp.path(),
            raw_intercept,
            &raw_coefficients,
            &manifest,
            s,
        );

        let current = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_301)];
        let value_mw = 50.0;
        let windows = vec![fixed_window(9, ymd(2026, 3, 1), ymd(2026, 4, 1), value_mw)];

        let cuts = load_boundary_cuts(
            &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, s)
                .with_fixed_windows(&windows),
        )
        .unwrap();

        let expected = (raw_intercept + raw_coefficients[1] * value_mw) / s;
        assert!(
            (cuts[0].intercept - expected).abs() <= expected.abs().max(1.0) * 1e-9,
            "folded intercept {} != expected {expected} (raw + raw_coeff[1]·factor, then ÷s)",
            cuts[0].intercept
        );
    }

    /// An empty `fixed_windows` leaves the intercept bit-identical to the
    /// marked-rescale-only load (`raw / s`): the empty-fold guard skips the
    /// intercept mutation entirely — no `+= 0.0`.
    #[test]
    fn load_boundary_cuts_empty_fixed_windows_intercept_bit_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let raw_intercept = 1000.0;
        let raw_coefficients = [7.0, 3.0];
        let manifest = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_301)];
        let s = 1_000_000.0;
        write_marked_checkpoint_with_manifest(
            tmp.path(),
            raw_intercept,
            &raw_coefficients,
            &manifest,
            s,
        );

        let current = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_301)];
        let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            2,
            &current,
            s,
        ))
        .unwrap();

        assert_eq!(
            cuts[0].intercept.to_bits(),
            (raw_intercept / s).to_bits(),
            "an empty fold must leave the intercept bit-identical to the marked-rescale-only load"
        );
    }

    /// An empty (unverifiable) source manifest with a non-empty `fixed_windows`
    /// applies no fold — the fold shares the rebind's `verifiable` gate — and
    /// emits no warning. The loaded intercept is the plain marked rescale
    /// (`raw / loading_factor`) with no fold term added, isolating the
    /// fold-skip from the mandatory rescale.
    #[test]
    fn load_boundary_cuts_unverifiable_manifest_skips_fold_without_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let raw_intercept = 42.0;
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[raw_intercept], &[]);

        let current = storage_manifest(1, 2);
        let windows = vec![fixed_window(9, ymd(2026, 3, 1), ymd(2026, 4, 1), 50.0)];
        let loading_cost_scale_factor = 1_000_000.0;
        let cuts = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                tmp.path(),
                fixture_priced_date(0),
                2,
                &current,
                loading_cost_scale_factor,
            )
            .with_fixed_windows(&windows),
        )
        .unwrap();

        assert_eq!(
            cuts[0].intercept.to_bits(),
            (raw_intercept / loading_cost_scale_factor).to_bits(),
            "an unverifiable manifest must skip the fold, leaving the intercept equal to the \
             plain marked rescale with no fold term added"
        );
    }

    // ── intercept-fold analytics (hand-computed deltas, both frames) ──────────

    /// The `overlap(window, m) / H_m` weight a window contributes at a source
    /// month anchor — decoded independently via `decode_slot_date`, never a
    /// decimal literal, so a weight that drifted together with production
    /// would not be masked.
    fn fold_weight(window: (NaiveDate, NaiveDate), source_anchor: i32) -> f64 {
        let month_start = decode_slot_date(source_anchor).unwrap();
        let month_end = decode_slot_date(next_month_anchor(source_anchor)).unwrap();
        let h_m = f64::from(u32::try_from((month_end - month_start).num_days()).unwrap()) * 24.0;
        overlap_hours(window, (month_start, month_end)) / h_m
    }

    /// MARKED frame: a source cut with anticipated coefficient `c` at a dated
    /// month, loaded against a whole-month fixed window (weight `1.0`), moves the
    /// intercept to `(raw_intercept + c·1·v) / s` — the folded term rides the
    /// marked `÷s` transform.
    #[test]
    fn boundary_fold_marked_frame_moves_intercept_by_hand_computed_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let anchor = 20_260_401; // April 2026
        let manifest = vec![storage_slot(1), anticipated_slot_at(9, 0, anchor)];
        let c = 3.0;
        let raw_intercept = 200.0;
        let s = 2_000_000.0;
        write_marked_checkpoint_with_manifest(tmp.path(), raw_intercept, &[1.0, c], &manifest, s);

        let (ws, we) = (ymd(2026, 4, 1), ymd(2026, 5, 1)); // whole month → weight 1.0
        let v = 50.0;
        let current = manifest.clone();
        let cuts = load_boundary_cuts(
            &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, s)
                .with_fixed_windows(&[fixed_window(9, ws, we, v)]),
        )
        .unwrap();

        let weight = fold_weight((ws, we), anchor);
        let expected = (raw_intercept + c * weight * v) / s;
        assert!(
            (cuts[0].intercept - expected).abs() < expected.abs().max(1.0) * 1e-9,
            "marked frame: intercept {} != (raw + c·{weight}·v) / s = {expected}",
            cuts[0].intercept
        );
    }

    /// A fixed window overlapping no source month contributes zero: the loaded
    /// intercept is `to_bits()`-identical to the same load with
    /// `fixed_windows = &[]`. Complements the builder-level no-term pin — this
    /// pins that the LOAD path moves no intercept.
    #[test]
    fn boundary_fold_no_overlap_window_leaves_intercept_bit_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let anchor = 20_260_401; // April 2026
        let manifest = vec![storage_slot(1), anticipated_slot_at(9, 0, anchor)];
        let s = 2_000_000.0;
        write_marked_checkpoint_with_manifest(tmp.path(), 1000.0, &[1.0, 3.0], &manifest, s);

        let current = manifest.clone();
        let load = |windows: &[AnticipatedCommitmentHistory]| {
            load_boundary_cuts(
                &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, s)
                    .with_fixed_windows(windows),
            )
            .unwrap()
        };

        let baseline = load(&[]);
        // June: disjoint from the April source month — the builder runs but emits no term.
        let disjoint = fixed_window(9, ymd(2026, 6, 1), ymd(2026, 6, 8), 50.0);
        let with_window = load(&[disjoint]);

        assert_eq!(
            with_window[0].intercept.to_bits(),
            baseline[0].intercept.to_bits(),
            "a window overlapping no source month must leave the intercept bit-identical"
        );
    }

    /// Empty-fold byte-neutrality: an empty `fixed_windows` slice, and
    /// (separately) a single all-zero-value window whose builder output is empty,
    /// both leave the loaded intercept `to_bits()`-identical to the
    /// no-fixed-window baseline.
    #[test]
    fn boundary_fold_empty_windows_intercept_bit_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let anchor = 20_260_401; // April 2026
        let manifest = vec![storage_slot(1), anticipated_slot_at(9, 0, anchor)];
        let s = 2_000_000.0;
        write_marked_checkpoint_with_manifest(tmp.path(), 1000.0, &[1.0, 3.0], &manifest, s);

        let current = manifest.clone();
        let load = |windows: &[AnticipatedCommitmentHistory]| {
            load_boundary_cuts(
                &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, s)
                    .with_fixed_windows(windows),
            )
            .unwrap()
        };

        // No-fold reference: with no fixed window the intercept is just the at-rest
        // 1000.0 carried through the marked rescale (/s), with no fold term added.
        let baseline = load(&[]);
        assert_eq!(
            baseline[0].intercept.to_bits(),
            (1000.0_f64 / s).to_bits(),
            "an empty window slice adds no fold term: intercept is the rescaled raw value"
        );
        // value_mw == 0.0 → build_boundary_fold skips it → empty fold.
        let all_zero = load(&[fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), 0.0)]);
        assert_eq!(
            all_zero[0].intercept.to_bits(),
            baseline[0].intercept.to_bits(),
            "an all-zero-value window must be byte-neutral"
        );
    }

    /// Multi-month accumulation: one fixed window overlapping TWO dated source
    /// months for thermal 9 — each carrying a genuinely different coefficient and
    /// month length — moves the intercept by the summed contribution
    /// `Σ_M c_M·(overlap(w,M)/H_M)·v`, pinning `build_boundary_fold`'s
    /// per-source-position accumulation. The distinct `c_M` catch an aliased
    /// source position returning the wrong month's value.
    #[test]
    fn boundary_fold_multi_month_window_sums_contributions() {
        let tmp = tempfile::tempdir().unwrap();
        let march = 20_260_301; // 31 days
        let april = 20_260_401; // 30 days
        let manifest = vec![
            storage_slot(1),
            anticipated_slot_at(9, 0, march),
            anticipated_slot_at(9, 1, april),
        ];
        let c_march = 3.0;
        let c_april = 7.0;
        let raw_intercept = 100.0;
        let s = 2_000_000.0;
        write_marked_checkpoint_with_manifest(
            tmp.path(),
            raw_intercept,
            &[1.0, c_march, c_april],
            &manifest,
            s,
        );

        let (ws, we) = (ymd(2026, 3, 25), ymd(2026, 4, 8)); // straddles March → April
        let v = 50.0;
        let current = manifest.clone();
        let cuts = load_boundary_cuts(
            &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 3, &current, s)
                .with_fixed_windows(&[fixed_window(9, ws, we, v)]),
        )
        .unwrap();

        let delta =
            c_march * fold_weight((ws, we), march) * v + c_april * fold_weight((ws, we), april) * v;
        let expected = (raw_intercept + delta) / s;
        assert!(
            (cuts[0].intercept - expected).abs() < expected.abs().max(1.0) * 1e-9,
            "multi-month: intercept {} != (raw + Σ c_M·w_M·v) / s = {expected}",
            cuts[0].intercept
        );
    }
}
